"""classify._classify — the ORGANIC status-machine transitions (no manual trigger).

test_trigger/test_confirm cover the forced-scan pedal and the confirm latch; this pins the
auto-settle lifecycle itself, which the audit flagged as untested: SETTLING dwell→capture, the
covered/unstable branches, the RESTING move-debounce + under-motion HOLD, and the MOVED re-entry.
classify.py was edited 2026-07-20 (ttl_seconds/hidden_since removal) with nothing guarding these.

The change threshold is thr = k·ref_area (k=0.05); ref_area is the footprint area (or obs.area). With
area 100 → thr 5.0: xor_prev/residual/missing < 5 reads as "no change", > 5 as "changed".
"""
from types import SimpleNamespace

import numpy as np

from lib.classify import Classify
from lib.scene import Detection, Scene, Status


def _cfg(**kw):
    return SimpleNamespace(get=lambda k, d=None: kw.get(k, d))


def _obs(*, xor_prev=0.0, area=100.0, residual=0.0, missing=0.0, under_motion=False):
    return SimpleNamespace(xor_prev=xor_prev, area=area, residual=residual,
                           missing=missing, under_motion=under_motion, mask=None)


def _drive(cls, scene, ts, obs=None):
    """One classify tick at ``ts`` (optionally refreshing the object's observation first)."""
    if obs is not None:
        with scene.writing("observe"):
            for o in scene.objects():
                scene.set(o.id, "observation", obs)
    with scene.writing("classify"):
        cls(SimpleNamespace(ts=ts), scene)


def _settling(scene, obs):
    """Mint an object → SETTLING with an observation, at a fresh tick (classify owns it, not observe)."""
    scene.begin_tick(0, ts=0.0)
    with scene.writing("observe"):
        oid = scene.mint(Detection((50.0, 50.0), (45, 45, 10, 10), 100.0, None))
    scene.begin_tick(1, ts=0.0)
    with scene.writing("observe"):
        scene.set(oid, "observation", obs)
        scene.set_status(oid, Status.SETTLING)
    scene.begin_tick(2, ts=0.0)
    return oid


def _resting(scene, obs, *, status=Status.RESTING):
    """Mint an object WITH a baked footprint, left in ``status`` (RESTING/MOVED) with an observation.
    All setup is under the ``observe`` writer (as mint is) then a fresh tick opens — begin_tick resets
    per-field writer tracking, so classify may then own status/off_since (the real ownership)."""
    fp = np.zeros((100, 100), np.uint8)
    fp[45:55, 45:55] = 255                                   # a 10×10 footprint, area 100
    scene.begin_tick(0, ts=0.0)
    with scene.writing("observe"):
        oid = scene.mint(Detection((50.0, 50.0), (45, 45, 10, 10), 100.0, None))
        scene.set(oid, "footprint", fp)
        scene.set(oid, "footprint_area", int(np.count_nonzero(fp)))
        scene.set_status(oid, status)
    scene.begin_tick(1, ts=0.0)
    with scene.writing("observe"):
        scene.set(oid, "observation", obs)
    scene.begin_tick(2, ts=0.0)
    return oid


# ---- SETTLING: the auto-settle dwell + the two exits ---- #

def test_settling_stable_captures_after_the_dwell(monkeypatch):
    s = Scene()
    cls = Classify(_cfg(settle_seconds=0.5, k=0.05))
    oid = _settling(s, _obs(xor_prev=0.0))                   # stable + motion-free
    got = []
    monkeypatch.setattr(cls, "_capture", lambda ctx, sc, obj, obs: got.append((obj.id, ctx.ts)))
    _drive(cls, s, 1.0)
    assert got == [] and s.get(oid).settle_since == 1.0      # first stable tick → the N-sec clock starts
    _drive(cls, s, 1.4)
    assert got == []                                         # 0.4 s < 0.5 → still dwelling
    _drive(cls, s, 1.5)
    assert got == [(oid, 1.5)]                               # 0.5 s elapsed → CAPTURE


def test_settling_unstable_goes_moved():
    s = Scene()
    cls = Classify(_cfg(settle_seconds=0.5, k=0.05))
    oid = _settling(s, _obs(xor_prev=1000.0))                # xor_prev ≫ thr → still moving
    _drive(cls, s, 1.0)
    assert s.get(oid).status is Status.MOVED


def test_settling_gone_mask_goes_hidden():
    s = Scene()
    cls = Classify(_cfg(settle_seconds=0.5, k=0.05))
    oid = _settling(s, _obs(area=0.0))                       # mask vanished (gone) → covered
    _drive(cls, s, 1.0)
    assert s.get(oid).status is Status.HIDDEN


# ---- RESTING: the move-debounce entry, the under-motion HOLD, the hide ---- #

def test_resting_off_footprint_must_persist_the_debounce():
    s = Scene()
    cls = Classify(_cfg(settle_seconds=0.5, k=0.05, move_debounce_seconds=0.1))
    oid = _resting(s, _obs(residual=50.0))                   # residual ≫ thr 5 → off the footprint
    _drive(cls, s, 1.0)
    assert s.get(oid).status is Status.RESTING and s.get(oid).off_since == 1.0   # debounce clock starts
    _drive(cls, s, 1.1)
    assert s.get(oid).status is Status.MOVED                 # off-footprint persisted ≥ 0.1 s → MOVED


def test_resting_one_frame_off_spike_does_not_flicker_moved():
    s = Scene()
    cls = Classify(_cfg(settle_seconds=0.5, k=0.05, move_debounce_seconds=0.1))
    oid = _resting(s, _obs(residual=50.0))
    _drive(cls, s, 1.0)                                      # off for one frame → off_since armed
    assert s.get(oid).status is Status.RESTING
    _drive(cls, s, 1.05, obs=_obs(residual=0.0))             # back ON the footprint within the window
    assert s.get(oid).status is Status.RESTING and s.get(oid).off_since is None   # spike absorbed


def test_resting_under_motion_holds_pinned():
    s = Scene()
    cls = Classify(_cfg(settle_seconds=0.5, k=0.05, move_debounce_seconds=0.1, hold_max_seconds=3.0))
    oid = _resting(s, _obs(residual=50.0, under_motion=True))   # off-footprint BUT a mass is over it
    for ts in (1.0, 1.2, 1.5, 2.0):
        _drive(cls, s, ts)
    assert s.get(oid).status is Status.RESTING               # a moving mass pins it — no MOVED flicker


def test_resting_under_motion_hold_is_bounded_by_hold_max():
    s = Scene()
    cls = Classify(_cfg(settle_seconds=0.5, k=0.05, move_debounce_seconds=0.1, hold_max_seconds=0.5))
    oid = _resting(s, _obs(residual=50.0, under_motion=True))
    _drive(cls, s, 1.0)                                      # under_motion_since armed at 1.0
    assert s.get(oid).status is Status.RESTING
    _drive(cls, s, 1.6)                                      # 0.6 s > hold_max 0.5 → hold releases,
    _drive(cls, s, 1.8)                                      #   off_footprint debounce then fires
    assert s.get(oid).status is Status.MOVED


def test_resting_footprint_missing_goes_hidden():
    s = Scene()
    cls = Classify(_cfg(settle_seconds=0.5, k=0.05))
    oid = _resting(s, _obs(missing=50.0, residual=0.0))      # footprint empty, nothing off it → covered
    _drive(cls, s, 1.0)
    assert s.get(oid).status is Status.HIDDEN


# ---- MOVED: back-on-footprint vs relocated ---- #

def test_moved_back_on_original_footprint_resumes_resting():
    s = Scene()
    cls = Classify(_cfg(settle_seconds=0.5, k=0.05))
    oid = _resting(s, _obs(residual=0.0, missing=0.0), status=Status.MOVED)   # on its footprint again
    _drive(cls, s, 1.0)
    assert s.get(oid).status is Status.RESTING               # never really relocated → resume rest


def test_moved_stable_elsewhere_goes_settling():
    s = Scene()
    cls = Classify(_cfg(settle_seconds=0.5, k=0.05))
    # relocated: off the old footprint (residual high) AND missing there, but the mask is stable now
    oid = _resting(s, _obs(xor_prev=0.0, residual=50.0, missing=50.0), status=Status.MOVED)
    _drive(cls, s, 1.0)
    assert s.get(oid).status is Status.SETTLING              # settled at a new spot → re-settle via SETTLING


# ---- commit_mode = global: the whole-table inactivity trigger ---- #

def _drive_g(cls, scene, ts, frac, obs=None):
    """One tick carrying the WHOLE-FRAME moving-pixel fraction observe publishes as ctx.motion_frac."""
    if obs is not None:
        with scene.writing("observe"):
            for o in scene.objects():
                scene.set(o.id, "observation", obs)
    with scene.writing("classify"):
        cls(SimpleNamespace(ts=ts, motion_frac=frac), scene)


def _global(**kw):
    base = dict(settle_seconds=0.5, k=0.05, commit_mode="global",
                global_quiet_seconds=1.0, global_quiet_frac=0.002)
    return Classify(_cfg(**{**base, **kw}))


def test_global_mode_never_commits_on_an_objects_own_dwell(monkeypatch):
    """The point of the mode: nothing commits between global events, however long a single object
    has been still. Its settle_since clock keeps running so switching back to partial commits it."""
    s = Scene()
    cls = _global()
    oid = _settling(s, _obs(xor_prev=0.0))
    got = []
    monkeypatch.setattr(cls, "_capture", lambda ctx, sc, obj, obs: got.append(ctx.ts))
    for ts in (1.0, 1.6, 2.2, 3.0):                      # way past settle_seconds — but the TABLE is busy
        _drive_g(cls, s, ts, frac=0.05)
    assert got == [], "an individual dwell must not commit in global mode"
    assert s.get(oid).settle_since == 1.0                # the per-object clock still runs


def test_global_mode_commits_every_object_once_the_table_is_idle(monkeypatch):
    s = Scene()
    cls = _global()
    _settling(s, _obs(xor_prev=0.0))
    got = []
    monkeypatch.setattr(cls, "_capture", lambda ctx, sc, obj, obs: got.append(round(ctx.ts, 2)))
    _drive_g(cls, s, 1.0, frac=0.05)                     # busy
    _drive_g(cls, s, 1.5, frac=0.0)                      # the table goes idle HERE
    _drive_g(cls, s, 2.0, frac=0.0)                      # 0.5 s idle — not yet
    assert got == []
    _drive_g(cls, s, 2.6, frac=0.0)                      # 1.1 s idle — FIRE
    assert got == [2.6]
    _drive_g(cls, s, 3.2, frac=0.0)                      # still idle …
    _drive_g(cls, s, 9.0, frac=0.0)                      # … and much later
    assert got == [2.6], "a table left alone must commit ONCE, not every quiet period"


def test_global_commit_re_arms_only_after_the_table_moves_again(monkeypatch):
    s = Scene()
    cls = _global()
    _settling(s, _obs(xor_prev=0.0))
    got = []
    monkeypatch.setattr(cls, "_capture", lambda ctx, sc, obj, obs: got.append(round(ctx.ts, 2)))
    _drive_g(cls, s, 1.0, frac=0.0)
    _drive_g(cls, s, 2.1, frac=0.0)                      # first commit
    assert got == [2.1]
    _drive_g(cls, s, 3.0, frac=0.05)                     # the operator moves something → re-arm
    _drive_g(cls, s, 3.5, frac=0.0)                      # idle again from here
    _drive_g(cls, s, 4.0, frac=0.0)                      # 0.5 s — not yet
    assert got == [2.1]
    _drive_g(cls, s, 4.6, frac=0.0)                      # 1.1 s — the SECOND arrangement commits
    assert got == [2.1, 4.6]


def test_partial_is_the_default_and_ignores_table_stillness(monkeypatch):
    """Regression guard on the default: partial mode commits on the object's own dwell and is
    indifferent to the global signal, so shipping this knob changes nothing unless it is set."""
    s = Scene()
    cls = Classify(_cfg(settle_seconds=0.5, k=0.05))     # NO commit_mode → default
    assert cls._commit_mode == "partial"
    _settling(s, _obs(xor_prev=0.0))
    got = []
    monkeypatch.setattr(cls, "_capture", lambda ctx, sc, obj, obs: got.append(round(ctx.ts, 2)))
    _drive_g(cls, s, 1.0, frac=0.9)                      # a BUSY table — irrelevant in partial
    _drive_g(cls, s, 1.6, frac=0.9)                      # dwell elapsed → commit anyway
    assert got == [1.6]
