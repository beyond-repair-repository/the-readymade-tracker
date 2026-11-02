"""``/rmt/trigger`` — the manual scan (a pedal / OSC client): read the table as it stands NOW. On the
next tick every SETTLING object is captured WITHOUT waiting out its ``settle_seconds`` dwell (and
confirmed so the outputs emit at once), and every quiet RESTING object is RE-BAKED from the live mask,
flushing the sub-threshold drift the change-gate deliberately holds.

The operator asserts "the arrangement is set — read it" instead of letting each object's timer decide.
It bypasses the DWELL only, never the stability test: an object in motion, covered, or under a moving
hand is not touched (that would bake a smeared/occluded footprint). With ``commit_mode = manual``,
this verb is the ONLY way the registry commits — the explicit shutter."""

from types import SimpleNamespace

import numpy as np

from lib.classify import Classify
from lib.control import ControlRouter, ScanTrigger
from lib.scene import Detection, Scene, Status
from plugins.control.oscin.oscin import translate


def _cfg(**kw):
    return SimpleNamespace(get=lambda k, d=None: kw.get(k, d))


def _obs(*, xor_prev=0.0, area=100.0, residual=0.0, missing=0.0, under_motion=False):
    return SimpleNamespace(xor_prev=xor_prev, area=area, residual=residual,
                           missing=missing, under_motion=under_motion, mask=None)


def _settling_object(scene, *, ts=0.0, obs=None):
    """Mint an object and leave it SETTLING with an observation, then open a FRESH tick — so it is no
    longer minted-this-tick and classify (not observe) owns the fields it writes."""
    scene.begin_tick(0, ts=ts)
    with scene.writing("observe"):
        oid = scene.mint(Detection((50.0, 50.0), (45, 45, 10, 10), 100.0, None))
    scene.begin_tick(1, ts=ts)                       # a new tick → no longer minted_this_tick
    with scene.writing("observe"):
        scene.set(oid, "observation", obs if obs is not None else _obs())
        scene.set_status(oid, Status.SETTLING)
    scene.begin_tick(2, ts=ts)                       # fresh tick → classify may own status/settle_since
    return oid


# --------------------------------------------------------------------------- #
# the ScanTrigger primitive
# --------------------------------------------------------------------------- #

def test_scan_trigger_is_a_one_shot_flag():
    t = ScanTrigger()
    assert t.take() is False              # nothing requested yet
    t.fire()
    assert t.take() is True               # consumed once…
    assert t.take() is False              # …and only once
    t.fire()
    t.fire()
    assert t.take() is True               # two stomps before a tick = ONE scan
    assert t.take() is False


# --------------------------------------------------------------------------- #
# the wire: OSC address → command → router → trigger
# --------------------------------------------------------------------------- #

def test_oscin_translates_the_trigger_address():
    assert translate("/rmt/trigger", []) == ("trigger", ())
    assert translate("/rmt/trigger", [], prefix="/rmt") == ("trigger", ())
    assert translate("/other/trigger", []) is None          # wrong prefix → ignored


def test_router_trigger_fires_the_scan():
    t = ScanTrigger()
    ControlRouter(scan_trigger=t)._trigger(())
    assert t.take() is True


def test_router_trigger_without_a_scan_trigger_is_a_no_op():
    ControlRouter()._trigger(())                            # warns, never raises


def test_trigger_is_a_registered_handler():
    assert "trigger" in ControlRouter._HANDLERS


# --------------------------------------------------------------------------- #
# the behaviour: bypass the dwell, but never the stability test
# --------------------------------------------------------------------------- #

def test_trigger_captures_a_settled_object_without_its_dwell(monkeypatch):
    s = Scene()
    t = ScanTrigger()
    cls = Classify(_cfg(settle_seconds=999.0, k=0.05), scan_trigger=t)   # auto-settle can never fire
    oid = _settling_object(s, obs=_obs(xor_prev=0.0))                    # stable + at rest
    got = []
    monkeypatch.setattr(cls, "_capture", lambda ctx, scene, obj, obs: got.append(obj.id))

    with s.writing("classify"):
        cls(SimpleNamespace(ts=1.0), s)
    assert got == []                       # dwell 999s not elapsed → no capture

    t.fire()
    with s.writing("classify"):
        cls(SimpleNamespace(ts=1.1), s)
    assert got == [oid]                    # manual scan → captured NOW, dwell skipped


def test_trigger_does_not_capture_an_object_still_in_motion(monkeypatch):
    s = Scene()
    t = ScanTrigger()
    cls = Classify(_cfg(settle_seconds=999.0, k=0.05), scan_trigger=t)
    # xor_prev way over k*area → NOT stable (still moving)
    oid = _settling_object(s, obs=_obs(xor_prev=1000.0))
    got = []
    monkeypatch.setattr(cls, "_capture", lambda ctx, scene, obj, obs: got.append(obj.id))

    t.fire()
    with s.writing("classify"):
        cls(SimpleNamespace(ts=1.0), s)
    assert got == []                       # a moving object is never captured, trigger or not
    assert s.get(oid).status is Status.MOVED


def test_trigger_is_consumed_so_the_next_tick_is_normal(monkeypatch):
    s = Scene()
    t = ScanTrigger()
    cls = Classify(_cfg(settle_seconds=999.0, k=0.05), scan_trigger=t)
    _settling_object(s, obs=_obs(xor_prev=0.0))
    got = []
    monkeypatch.setattr(cls, "_capture", lambda ctx, scene, obj, obs: got.append(obj.id))

    t.fire()
    with s.writing("classify"):
        cls(SimpleNamespace(ts=1.0), s)
    with s.writing("classify"):
        cls(SimpleNamespace(ts=1.1), s)    # trigger already consumed → back to the dwell
    assert len(got) == 1


def test_no_scan_trigger_wired_keeps_the_normal_dwell(monkeypatch):
    s = Scene()
    cls = Classify(_cfg(settle_seconds=999.0, k=0.05))    # no scan_trigger at all
    _settling_object(s, obs=_obs(xor_prev=0.0))
    got = []
    monkeypatch.setattr(cls, "_capture", lambda ctx, scene, obj, obs: got.append(obj.id))
    with s.writing("classify"):
        cls(SimpleNamespace(ts=1.0), s)
    assert got == []                       # unchanged behaviour when the pedal isn't wired


# --------------------------------------------------------------------------- #
# a manual scan emits at once (bypasses the confirm grace)
# --------------------------------------------------------------------------- #

def test_trigger_confirms_immediately_so_outputs_emit():
    s = Scene()
    t = ScanTrigger()
    cls = Classify(_cfg(confirm_seconds=10.0), scan_trigger=t)      # long grace: would normally withhold
    s.begin_tick(0, ts=0.0)
    with s.writing("observe"):
        oid = s.mint(Detection((50.0, 50.0), (45, 45, 10, 10), 100.0, None))
    assert s.get(oid).confirmed is False

    with s.writing("classify"):
        cls._latch_confirmed(SimpleNamespace(ts=0.1), s, s.get(oid), False)
    assert s.get(oid).confirmed is False                 # age 0.1s < 10s grace → still withheld

    with s.writing("classify"):
        cls._latch_confirmed(SimpleNamespace(ts=0.1), s, s.get(oid), True)
    assert s.get(oid).confirmed is True                  # manual scan → confirmed at once


# --------------------------------------------------------------------------- #
# the refresh: a quiet RESTING object is re-baked from the live mask
# --------------------------------------------------------------------------- #

def _real_obs(mask, *, xor_prev=0.0, residual=0.0, missing=0.0, under_motion=False):
    """An observation carrying a REAL mask (for the real ``_capture``), centroid included."""
    ys, xs = np.nonzero(mask)
    centroid = (float(xs.mean()), float(ys.mean())) if len(xs) else (0.0, 0.0)
    return SimpleNamespace(xor_prev=xor_prev, area=float(len(xs)), residual=residual,
                           missing=missing, under_motion=under_motion,
                           mask=mask, centroid=centroid)


def _rested_object(scene, cls, trigger):
    """Mint, settle, and capture an object via a first trigger → RESTING with a 100 px² footprint."""
    m1 = np.zeros((40, 40), np.uint8)
    m1[10:20, 10:20] = 255                                   # area 100 → thr = k*100 = 5 px
    oid = _settling_object(scene, obs=_real_obs(m1))
    trigger.fire()
    with scene.writing("classify"):
        cls(SimpleNamespace(ts=1.0), scene)
    assert scene.get(oid).status is Status.RESTING
    assert scene.get(oid).footprint_area == 100
    return oid, m1


def test_trigger_rebakes_a_quiet_resting_object():
    s = Scene()
    t = ScanTrigger()
    cls = Classify(_cfg(settle_seconds=999.0, k=0.05), scan_trigger=t)
    oid, m1 = _rested_object(s, cls, t)

    # sub-k drift: 3 extra px against thr 5 — the change-gate holds this forever by design
    m2 = m1.copy()
    m2[10:13, 20] = 255                                      # area 103, residual 3 < thr
    s.begin_tick(3, ts=2.0)
    with s.writing("observe"):
        s.set(oid, "observation", _real_obs(m2, residual=3.0))
    s.begin_tick(4, ts=2.1)

    with s.writing("classify"):
        cls(SimpleNamespace(ts=2.1), s)                      # no trigger → the drift stays held
    assert s.get(oid).footprint_area == 100

    t.fire()
    with s.writing("classify"):
        cls(SimpleNamespace(ts=2.2), s)                      # stomp → re-baked from the live mask
    assert s.get(oid).footprint_area == 103
    assert s.get(oid).status is Status.RESTING               # a re-bake is not a transition


def test_trigger_leaves_a_resting_object_under_motion_alone():
    s = Scene()
    t = ScanTrigger()
    cls = Classify(_cfg(settle_seconds=999.0, k=0.05), scan_trigger=t)
    oid, m1 = _rested_object(s, cls, t)

    m2 = m1.copy()
    m2[10:13, 20] = 255
    s.begin_tick(3, ts=2.0)
    with s.writing("observe"):                               # a hand over the spot: motion on the footprint
        s.set(oid, "observation", _real_obs(m2, residual=3.0, under_motion=True))
    s.begin_tick(4, ts=2.1)

    t.fire()
    with s.writing("classify"):
        cls(SimpleNamespace(ts=2.1), s)
    assert s.get(oid).footprint_area == 100                  # never bake under a moving hand
    assert s.get(oid).status is Status.RESTING


# --------------------------------------------------------------------------- #
# commit_mode = manual: the trigger is the ONLY shutter
# --------------------------------------------------------------------------- #

def test_commit_mode_manual_commits_only_on_trigger(monkeypatch):
    s = Scene()
    t = ScanTrigger()
    cls = Classify(_cfg(settle_seconds=0.0, commit_mode="manual", k=0.05), scan_trigger=t)
    oid = _settling_object(s, obs=_obs(xor_prev=0.0))
    got = []
    monkeypatch.setattr(cls, "_capture", lambda ctx, scene, obj, obs: got.append(obj.id))

    with s.writing("classify"):
        cls(SimpleNamespace(ts=1.0, motion_frac=0.0), s)     # starts the (irrelevant) dwell
    with s.writing("classify"):
        cls(SimpleNamespace(ts=99.0, motion_frac=0.0), s)    # dwell long past AND table dead quiet
    assert got == []                                         # …still held: manual never auto-commits
    assert s.get(oid).status is Status.SETTLING

    t.fire()
    with s.writing("classify"):
        cls(SimpleNamespace(ts=99.1, motion_frac=0.0), s)
    assert got == [oid]                                      # the pedal is the shutter
