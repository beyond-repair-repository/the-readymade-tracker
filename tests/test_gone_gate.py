"""identity._retire_maybe — the contour-level GONE gate on a HIDDEN captured object.

The owner's rule: a HIDDEN object is evicted (pooled) only when it is GONE — its rest FOOTPRINT has
BARED — not while it is merely OCCLUDED by a hovering hand. GONE is a contour-level test: the object's
rasterized footprint silhouette ∩ the live foreground (obj_fg), NOT a coarse bounding box.

The memory clock is ONE window — pool_ttl_seconds — anchored at the moment the spot BARES (not at when
the object went HIDDEN) and FROZEN while occluded, spanning the drawn stay-then-snap coast AND the
gone-pool. So occlusion time never ages an object out: we only start forgetting once we SEE the place
clear. A pooled entry is stamped with the baring time, so its expiry runs from baring too.
"""
from types import SimpleNamespace

import numpy as np

from lib.identity import Identity
from lib.scene import Status


def _fp(size=20, side=10):
    """A footprint mask: a filled `side`×`side` square (uint8 0/255) in a `size`×`size` frame."""
    m = np.zeros((size, size), np.uint8)
    m[5:5 + side, 5:5 + side] = 255
    return m


def _obj(fp):
    return SimpleNamespace(
        id=7, status=Status.HIDDEN, footprint=fp, footprint_area=int(np.count_nonzero(fp)),
        centroid=(10.0, 10.0), footprint_centroid=(10.0, 10.0),
        embeds=None, area=100.0, canon_area=100.0, pose_ref=None)


def _scene():
    retired = []
    return SimpleNamespace(retire=lambda i: retired.append(i), _retired=retired)


def _idn(ttl=1.0):
    idn = Identity(None)
    idn._ttl = ttl               # the one memory window — short so ctx.ts steps cross it
    return idn


def _bare(size=20):
    return np.zeros((size, size), np.uint8)


def test_covered_footprint_holds_the_object():
    """Footprint fully covered (a hovering hand) → NOT gone: no retire even long past the window."""
    fp = _fp()
    obj, scene = _obj(fp), _scene()
    ctx = SimpleNamespace(ts=10.0, obj_fg=fp.copy())      # fg overlaps the whole footprint → cov 1.0
    _idn()._retire_maybe(ctx, scene, obj)
    assert scene._retired == []                           # occluded → kept, id preserved


def test_bared_past_the_window_is_pooled():
    """Bared, and still bared a full window later → GONE: pooled + retired (stay-then-snap unchanged)."""
    obj, scene = _obj(_fp()), _scene()
    idn = _idn(ttl=1.0)
    idn._retire_maybe(SimpleNamespace(ts=9.0, obj_fg=_bare()), scene, obj)    # first bared tick → clock starts
    assert scene._retired == []                                              # just bared — not yet gone
    idn._retire_maybe(SimpleNamespace(ts=10.0, obj_fg=_bare()), scene, obj)   # 1.0 s bared ≥ ttl
    assert scene._retired == [7]
    assert 7 in idn._pool                                 # carried to the gone-pool for re-acquire


def test_bared_within_the_window_still_coasts():
    """Bared but still inside the window → drawn + claimable, not yet pooled."""
    obj, scene = _obj(_fp()), _scene()
    ctx = SimpleNamespace(ts=10.0, obj_fg=_bare())
    _idn(ttl=5.0)._retire_maybe(ctx, scene, obj)          # first bared tick → clock starts, 0 elapsed
    assert scene._retired == []


def test_occlusion_does_not_age_the_memory_clock():
    """THE OWNER'S RULE. A hand hovering over a HIDDEN object must NOT age it out: occluded far longer
    than the window, then a single bared tick → the clock only just started; it still coasts. We learn
    the spot is empty ONLY when the hand clears it."""
    fp = _fp()
    obj, scene = _obj(fp), _scene()
    idn = _idn(ttl=1.0)
    for ts in (1.0, 2.0, 5.0, 9.0):                       # covered the whole time (a hovering hand)
        idn._retire_maybe(SimpleNamespace(ts=ts, obj_fg=fp.copy()), scene, obj)
    assert scene._retired == []                           # occlusion never confirmed absence
    # the hand lifts at t=10 revealing an EMPTY spot: the memory clock starts NOW, not 9 s ago
    idn._retire_maybe(SimpleNamespace(ts=10.0, obj_fg=_bare()), scene, obj)
    assert scene._retired == []                           # just bared — a full window must still elapse
    idn._retire_maybe(SimpleNamespace(ts=11.0, obj_fg=_bare()), scene, obj)
    assert scene._retired == [7]                          # 1.0 s AFTER baring → gone (not 9 s after hiding)


def test_re_occlusion_restarts_the_memory_clock():
    """Bared briefly, then covered again (the hand sweeps back) → the anchor resets; the next baring
    starts a fresh window. A hand that keeps re-covering the spot never forgets the id."""
    fp = _fp()
    obj, scene = _obj(fp), _scene()
    idn = _idn(ttl=1.0)
    idn._retire_maybe(SimpleNamespace(ts=1.0, obj_fg=_bare()), scene, obj)       # bares at 1.0
    idn._retire_maybe(SimpleNamespace(ts=1.5, obj_fg=fp.copy()), scene, obj)     # re-covered → anchor cleared
    idn._retire_maybe(SimpleNamespace(ts=1.9, obj_fg=_bare()), scene, obj)       # bares AGAIN at 1.9
    idn._retire_maybe(SimpleNamespace(ts=2.0, obj_fg=_bare()), scene, obj)       # only 0.1 s since re-baring
    assert scene._retired == []                                                 # clock restarted at 1.9
    idn._retire_maybe(SimpleNamespace(ts=3.0, obj_fg=_bare()), scene, obj)       # 1.1 s since re-baring ≥ ttl
    assert scene._retired == [7]


def test_pool_entry_is_stamped_at_baring_not_at_pooling():
    """The one window runs from BARING: a pooled entry's ts is the first-bared time, so the pool expiry
    (_ttl) is measured from baring — unbroken across coast → pool."""
    obj, scene = _obj(_fp()), _scene()
    idn = _idn(ttl=1.0)
    idn._retire_maybe(SimpleNamespace(ts=9.0, obj_fg=_bare()), scene, obj)    # bares at 9.0
    idn._retire_maybe(SimpleNamespace(ts=10.0, obj_fg=_bare()), scene, obj)   # pools at 10.0
    assert idn._pool[7]["ts"] == 9.0                                          # stamped at baring, not 10.0


def test_obj_fg_absent_holds_cannot_confirm_absence():
    """No obj_fg on ctx → the GONE test can't run, so we cannot confirm the spot is empty → HOLD
    (present, memory frozen). We only forget once we SEE the place clear."""
    obj, scene = _obj(_fp()), _scene()
    ctx = SimpleNamespace(ts=10.0, hidden_present=set())      # no obj_fg
    _idn(ttl=1.0)._retire_maybe(ctx, scene, obj)
    assert scene._retired == [] and obj.id in ctx.hidden_present


def test_footprint_cov_is_contour_level_not_bbox():
    """Coverage is footprint-silhouette ∩ fg over the silhouette area — a bbox corner does NOT count."""
    fp = _fp(size=20, side=10)                            # square footprint at rows/cols 5..15, area 100
    idn = _idn()
    # foreground only in the bbox CORNER outside the silhouette would read 0 against a non-square shape;
    # here the silhouette IS the square, so covering half the square gives ~0.5.
    half = np.zeros((20, 20), np.uint8)
    half[5:10, 5:15] = 255                                # top half of the square → 50 of 100 px
    assert abs(idn._footprint_cov(SimpleNamespace(footprint=fp, footprint_area=100), half) - 0.5) < 1e-6
    assert idn._footprint_cov(SimpleNamespace(footprint=fp, footprint_area=100),
                              np.zeros((20, 20), np.uint8)) == 0.0


# ---- EARLY GONE: bared + motionless spot + idle table → pool at once, long before the window ---- #

def _quiet_ctx(ts, size=20):
    """Bared spot, no motion anywhere, table idle — the early-GONE qualifying frame."""
    return SimpleNamespace(ts=ts, obj_fg=np.zeros((size, size), np.uint8),
                           motion=np.zeros((size, size), np.uint8), motion_frac=0.0)


def test_early_gone_fires_after_away_quiet_seconds():
    """Bared + quiet + idle table held ≥ away_quiet_seconds → pooled NOW, long before the window."""
    obj, scene = _obj(_fp()), _scene()
    idn = _idn(ttl=100.0)                                 # window far away — only early GONE can retire
    idn._retire_maybe(_quiet_ctx(1.0), scene, obj)        # clock starts
    assert scene._retired == []
    idn._retire_maybe(_quiet_ctx(1.6), scene, obj)        # 0.6 s ≥ away_quiet 0.5
    assert scene._retired == [7]
    assert 7 in idn._pool                                 # still re-ID-able for the rest of the window


def test_early_gone_blocked_while_the_table_is_busy():
    """A hand moving ANYWHERE (mid-carry) → the table-quiet guard holds the husk claimable."""
    obj, scene = _obj(_fp()), _scene()
    idn = _idn(ttl=100.0)
    for ts in (1.0, 2.0, 5.0, 9.0):
        ctx = _quiet_ctx(ts)
        ctx.motion_frac = 0.05                            # ~5% of pixels moving = a working hand
        idn._retire_maybe(ctx, scene, obj)
    assert scene._retired == []


def test_early_gone_blocked_by_motion_at_the_spot():
    """The table reads idle but something stirs ON the footprint → hold (a grab in progress)."""
    fp = _fp()
    obj, scene = _obj(fp), _scene()
    idn = _idn(ttl=100.0)
    for ts in (1.0, 2.0, 5.0):
        ctx = _quiet_ctx(ts)
        ctx.motion = fp.copy()                            # the whole footprint is in the motion mask
        idn._retire_maybe(ctx, scene, obj)
    assert scene._retired == []


def test_early_gone_clock_resets_on_activity():
    """A busy tick in the middle restarts the quiet clock — the window must be CONTINUOUS."""
    obj, scene = _obj(_fp()), _scene()
    idn = _idn(ttl=100.0)
    idn._retire_maybe(_quiet_ctx(1.0), scene, obj)        # quiet begins
    busy = _quiet_ctx(1.3)
    busy.motion_frac = 0.05
    idn._retire_maybe(busy, scene, obj)                   # interruption → reset
    idn._retire_maybe(_quiet_ctx(1.7), scene, obj)        # quiet again (new clock)
    assert scene._retired == []                           # 1.7 vs 1.0 would be 0.7 s — must NOT count
    idn._retire_maybe(_quiet_ctx(2.3), scene, obj)        # 0.6 s after the RESTART
    assert scene._retired == [7]


def test_early_gone_disabled_by_zero_leaves_only_the_window():
    """away_quiet_seconds = 0 → the early exit is off; the plain window (from baring) still decides."""
    obj, scene = _obj(_fp()), _scene()
    idn = _idn(ttl=1.0)
    idn._away_quiet = 0.0
    idn._retire_maybe(_quiet_ctx(0.5), scene, obj)        # perfect early-GONE conditions, rule off → clock starts
    assert scene._retired == []
    idn._retire_maybe(_quiet_ctx(1.5), scene, obj)        # 1.0 s after baring at 0.5 → the window elapses
    assert scene._retired == [7]


def test_identity_publishes_the_presence_verdict():
    """_retire_maybe marks a HIDDEN object PRESENT (ctx.hidden_present) while its spot is occluded OR
    ACTIVE (a hand at it); only a BARED-and-QUIET spot is dropped — a candidate for GONE."""
    fp = _fp()
    obj, scene = _obj(fp), _scene()
    idn = _idn(ttl=100.0)
    occluded = SimpleNamespace(ts=1.0, obj_fg=fp.copy(), hidden_present=set())        # spot occluded
    idn._retire_maybe(occluded, scene, obj)
    assert obj.id in occluded.hidden_present and scene._retired == []
    active = SimpleNamespace(ts=1.0, obj_fg=np.zeros((20, 20), np.uint8),             # bared, but a hand moves
                             motion=np.ones((20, 20), np.uint8), motion_frac=0.2, hidden_present=set())
    idn._retire_maybe(active, scene, obj)
    assert obj.id in active.hidden_present                                            # bared + ACTIVE → present
    quiet = _quiet_ctx(1.0); quiet.hidden_present = set()                             # bared AND quiet
    idn._retire_maybe(quiet, scene, obj)
    assert quiet.hidden_present == set()                                              # only this is a GONE candidate
