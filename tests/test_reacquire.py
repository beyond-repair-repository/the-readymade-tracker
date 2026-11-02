"""identity._reacquire — gone-pool re-adoption, driven through the REAL Scene.

Pins the two id-lifecycle rules the 2026-07-20 audit found broken:

  * ONCE PER SETTLE CYCLE: a still-settling blob adopts AT MOST ONE pool entry. The guard must be
    intersected with POST-rewrite ids (rewrite_id mutates obj.id in place), else it empties the same
    tick and the blob hops through successive pool entries, corrupting ids and draining the pool.
  * STATE DIES WITH THE ID: _pool_and_retire / the ghost-retire branch must drop _bared/_away/_fp_pos
    at retire time — retire defers eviction, so the end-of-tick "departed ids" prune never sees the id
    leave, and a leftover _bared anchor from the FIRST lift would instantly expire the SECOND lift's
    memory window after a re-acquire of the same id.
"""
from types import SimpleNamespace

import numpy as np

from lib.identity import Identity
from lib.scene import Detection, Scene, Status


def _pool_entry(x, y, ts=0.0, area=100.0):
    return {"c": (float(x), float(y)), "ts": ts, "embeds": None,
            "area": area, "canon_area": area, "canon": None}


def _tick(scene, idn, ts, mint_at=None):
    """One pipeline-shaped tick: optional mint (observe), then identity."""
    scene.begin_tick(int(ts * 30), ts=ts)
    oid = None
    if mint_at is not None:
        x, y = mint_at
        with scene.writing("observe"):
            oid = scene.mint(Detection((float(x), float(y)), (int(x) - 5, int(y) - 5, 10, 10), 100.0, None))
    with scene.writing("identity"):
        idn(SimpleNamespace(ts=ts), scene)
    scene.end_tick()
    return oid


def test_settling_blob_adopts_at_most_one_pool_entry():
    """The once-per-settle guard holds across ticks: after adopting the nearest pool id, the SAME
    still-settling object must NOT hop onto the second pool entry on the next tick."""
    s, idn = Scene(), Identity(None)
    idn._pool[101] = _pool_entry(50, 50)                  # nearest — should win
    idn._pool[102] = _pool_entry(60, 60)                  # in reach too — must SURVIVE
    _tick(s, idn, 0.0, mint_at=(52, 52))                  # mint + adopt in one tick
    ids = {o.id for o in s.objects()}
    assert ids == {101}                                   # the mint was rewritten to the nearest pool id
    assert 101 in idn._reacquired                         # guard armed with the POST-rewrite id
    _tick(s, idn, 0.1)                                    # object still SETTLING/footprint-None
    _tick(s, idn, 0.2)
    assert {o.id for o in s.objects()} == {101}           # did NOT hop to 102
    assert 102 in idn._pool                               # second entry intact for its real owner


def test_guard_frees_after_capture():
    """Once the object captures (RESTING), the guard drops it — a LATER settle cycle may re-adopt."""
    s, idn = Scene(), Identity(None)
    idn._pool[101] = _pool_entry(50, 50)
    _tick(s, idn, 0.0, mint_at=(52, 52))
    assert 101 in idn._reacquired
    s.begin_tick(9, ts=0.3)                               # the settle completes: give it a footprint
    with s.writing("classify"):
        fp = np.zeros((20, 20), np.uint8); fp[5:15, 5:15] = 255
        s.set(101, "footprint", fp)
        s.set_status(101, Status.RESTING)
    with s.writing("identity"):
        idn(SimpleNamespace(ts=0.3), s)
    s.end_tick()
    assert 101 not in idn._reacquired                     # no longer settling → freed for the future


def _fp(size=20, side=10):
    m = np.zeros((size, size), np.uint8)
    m[5:5 + side, 5:5 + side] = 255
    return m


def test_pool_and_retire_drops_per_id_state():
    """Pooling an id forgets its _bared/_away/_fp_pos — the second lift gets a FULL memory window."""
    idn = Identity(None)
    obj = SimpleNamespace(id=7, footprint=_fp(), footprint_area=100,
                          centroid=(10.0, 10.0), footprint_centroid=(10.0, 10.0),
                          embeds=None, area=100.0, canon_area=100.0, pose_ref=None)
    idn._bared[7], idn._away[7], idn._fp_pos[7] = 1.0, 1.2, (10.0, 10.0)
    retired = []
    scene = SimpleNamespace(retire=lambda i: retired.append(i))
    idn._pool_and_retire(SimpleNamespace(ts=2.0), scene, obj, bared_ts=1.0)
    assert retired == [7] and idn._pool[7]["ts"] == 1.0
    assert 7 not in idn._bared and 7 not in idn._away and 7 not in idn._fp_pos


def test_ghost_retire_drops_per_id_state():
    """The ghost branch (HIDDEN, never captured) also forgets the trackers."""
    idn = Identity(None)
    obj = SimpleNamespace(id=7, status=Status.HIDDEN, footprint=None)
    idn._bared[7], idn._away[7], idn._fp_pos[7] = 1.0, 1.2, (10.0, 10.0)
    retired = []
    scene = SimpleNamespace(retire=lambda i: retired.append(i))
    idn._retire_maybe(SimpleNamespace(ts=2.0), scene, obj)
    assert retired == [7]
    assert 7 not in idn._bared and 7 not in idn._away and 7 not in idn._fp_pos


def test_geometric_reacquire_applies_the_shape_gate():
    """A compact square set-down must NOT steal a pooled WIRE's id (fold-change elongation gate) —
    the same gate observe's relocation pass applies on the husk half of the ONE set-down reach."""
    import cv2
    from lib.observe import Observation, Observe
    wire = np.array([[[0, 0]], [[100, 0]], [[100, 4]], [[0, 4]]], np.int32)      # elong 25
    s, idn = Scene(), Identity(None, observe=Observe(None))                      # tol default 1.0 (2x)
    idn._pool[101] = dict(_pool_entry(50, 50), canon=(wire, (50.0, 50.0)))
    mask = np.zeros((100, 100), np.uint8)
    cv2.rectangle(mask, (40, 40), (70, 70), 255, -1)                             # a 30x30 square blob
    s.begin_tick(0, ts=0.0)
    with s.writing("observe"):
        oid = s.mint(Detection((55.0, 55.0), (40, 40, 30, 30), 900.0, None))
        s.set(oid, "observation", Observation(residual=0, missing=0, xor_prev=0,
                                              centroid=(55.0, 55.0), area=900.0,
                                              bbox=(40, 40, 30, 30), mask=mask))
    idn._pool[101]["area"] = 900.0                       # size gate passes — shape must decide
    idn._pool[101]["canon_area"] = 900.0
    with s.writing("identity"):
        idn(SimpleNamespace(ts=0.0), s)
    s.end_tick()
    assert {o.id for o in s.objects()} == {oid}          # refused: square ≠ wire, id NOT adopted
    assert 101 in idn._pool                              # the wire's identity survives for its owner

    # …while a same-shape set-down (square canon) IS adopted through the identical path.
    square = np.array([[[0, 0]], [[30, 0]], [[30, 30]], [[0, 30]]], np.int32)
    idn._pool[101]["canon"] = (square, (50.0, 50.0))
    s.begin_tick(1, ts=0.1)
    with s.writing("identity"):
        idn(SimpleNamespace(ts=0.1), s)
    s.end_tick()
    assert {o.id for o in s.objects()} == {101}


# --------------------------------------------------------------------------- #
# the APPEARANCE path — a FAR set-down the geometric gate can't reach, adopted
# by classical-descriptor cosine. Untested until now (every case above runs with
# no embedder, so emb is always None and only the geometric fallback fires).
# --------------------------------------------------------------------------- #

def _embedded_idn(query):
    """An Identity with the appearance path LIVE, returning a fixed query embedding per settle."""
    idn = Identity(None)
    idn._embedder = object()                              # truthy → the cosine path runs
    idn._embed = lambda ctx, obj: np.asarray(query, np.float32)   # deterministic query vector
    return idn


def _vec(*xs):
    return np.asarray(xs, np.float32)


def test_appearance_reacquire_adopts_a_far_matching_print():
    """A set-down FAR out of geometric reach (pool at 999,999; blob at 52,52) is re-acquired purely on
    the descriptor: the best-cosine pooled print above the floor, uniquely, adopts its id."""
    s, idn = Scene(), _embedded_idn([1.0, 0.0])
    idn._pool[101] = dict(_pool_entry(999, 999), embeds=(_vec(1.0, 0.0),))    # cosine 1.0 — the match
    idn._pool[102] = dict(_pool_entry(998, 998), embeds=(_vec(0.0, 1.0),))    # cosine 0.0 — orthogonal
    _tick(s, idn, 0.0, mint_at=(52, 52))
    assert {o.id for o in s.objects()} == {101}           # adopted by appearance despite the distance
    assert 101 not in idn._pool and 102 in idn._pool      # the match consumed, the non-match survives


def test_appearance_twin_guard_refuses_look_alikes():
    """Two near-identical pooled prints (cosine within uniqueness_margin) → the twin guard refuses BOTH;
    a fresh mint is safer than a coin-flip between two look-alikes."""
    s, idn = Scene(), _embedded_idn([1.0, 0.0])
    idn._pool[101] = dict(_pool_entry(999, 999), embeds=(_vec(1.0, 0.0),))         # cosine 1.000
    idn._pool[102] = dict(_pool_entry(998, 998), embeds=(_vec(0.999, 0.0447),))    # cosine ≈ 0.999
    oid = _tick(s, idn, 0.0, mint_at=(52, 52))
    assert {o.id for o in s.objects()} == {oid}           # neither adopted → the object keeps its fresh id
    assert 101 in idn._pool and 102 in idn._pool          # both identities survive for their real owners


def test_appearance_size_gate_rejects_a_wrong_size_return():
    """A perfect cosine still can't adopt across a gross size mismatch — the size gate closes the door a
    paused arm (right texture, wrong area) could otherwise use."""
    s, idn = Scene(), _embedded_idn([1.0, 0.0])
    idn._pool[101] = dict(_pool_entry(999, 999), embeds=(_vec(1.0, 0.0),), area=10000.0, canon_area=10000.0)
    oid = _tick(s, idn, 0.0, mint_at=(52, 52))            # the mint's area is 100 vs the pool's 10000
    assert {o.id for o in s.objects()} == {oid}           # cosine 1.0 but 100:10000 fails size_tol → no adopt
    assert 101 in idn._pool


def test_reacquire_restores_the_pose_ref():
    """Adoption restores the pooled canonical pose reference, so absolute rotation (theta_cumulative)
    survives the relocation instead of re-basing to 0."""
    s, idn = Scene(), _embedded_idn([1.0, 0.0])
    canon = ("canon-contour-sentinel", (50.0, 50.0))
    idn._pool[101] = dict(_pool_entry(52, 52), embeds=(_vec(1.0, 0.0),), canon=canon)
    _tick(s, idn, 0.0, mint_at=(52, 52))
    assert s.get(101) is not None and s.get(101).pose_ref == canon
