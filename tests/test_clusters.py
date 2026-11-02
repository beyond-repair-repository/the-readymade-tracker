"""clusters stage — proximity grouping, persistent ids, and a cluster-centric TABLE (id / members /
centroid / bbox / min-circle / convex-hull) published via scene.set_clusters. Facts, not layout:
a client joins `members` to /pose for any ring/step it wants. Flows through oscout as /rmt/cluster/*."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

import readymade_tracker
from lib.clusters import Clusters
from lib.config import build_config
from lib.scene import Detection, ObjectSnap, Scene, Status


class _Cfg:
    def __init__(self, **kw):
        self._d = kw
    def get(self, k, d=None):
        return self._d.get(k, d)


def _scene(positions):
    s = Scene()
    s.begin_tick(0, ts=0.0)
    ids = []
    with s.writing("setup"):
        for (x, y) in positions:
            oid = s.mint(Detection((float(x), float(y)), (int(x) - 5, int(y) - 5, 10, 10), 100.0, None))
            s.set_status(oid, Status.RESTING)       # established objects are SETTLED (mint defaults SETTLING)
            ids.append(oid)                         #   → so they can JOIN a cluster under the settle-held rule
    with s.writing("pose"):
        for oid, (x, y) in zip(ids, positions):
            s.set(oid, "pose", (float(x), float(y), 0.0))
    with s.writing("classify"):
        for oid in ids:
            s.set(oid, "confirmed", True)
    return s, ids


def _run(s, stage=None, radius=0.12, min_objects=2, w=400, h=400, ts=0.0, present=None):
    ctx = SimpleNamespace(bgr=np.zeros((h, w, 3), np.uint8), ts=ts)
    if present is not None:
        ctx.hidden_present = present                   # identity's per-tick verdict (still-present HIDDEN ids)
    st = stage or Clusters(_Cfg(radius=radius, min_objects=min_objects))
    with s.writing("clusters"):
        st(ctx, s)
    return st


def _table(s):
    """This tick's cluster table: {cluster_id: ClusterSnap}."""
    return {c.id: c for c in s.snapshot().clusters}


def _cluster_of(s, oid):
    """The ClusterSnap whose members include ``oid`` (disjoint → at most one), or None."""
    for c in s.snapshot().clusters:
        if oid in c.members:
            return c
    return None


# --------------------------------------------------------------------------- #
# grouping
# --------------------------------------------------------------------------- #

def test_three_close_objects_form_one_cluster():
    s, ids = _scene([(100, 100), (110, 105), (105, 115)])
    _run(s, radius=0.12)                                # thresh = 0.12·400 = 48 px → all linked
    assert len(s.snapshot().clusters) == 1
    assert set(s.snapshot().clusters[0].members) == set(ids)


def test_two_separated_groups_are_distinct_clusters():
    s, ids = _scene([(100, 100), (110, 105), (300, 300), (305, 310)])
    _run(s, radius=0.05)                                # thresh = 20 px: within-group only
    a, b = _cluster_of(s, ids[0]), _cluster_of(s, ids[2])
    assert a.id != b.id                                 # two distinct clusters
    assert set(a.members) == {ids[0], ids[1]} and set(b.members) == {ids[2], ids[3]}


def test_pair_below_min_objects_not_clustered():
    s, ids = _scene([(100, 100), (110, 105)])
    _run(s, radius=0.12, min_objects=3)                # need 3, have 2 → no cluster
    assert s.snapshot().clusters == ()


def test_gone_hidden_member_leaves_the_ring():
    # LEAVE when GONE: a HIDDEN member whose spot is bared AND quiet is NOT in identity's presence set,
    # so it leaves at once. Here the survivor drops below min_objects → the cluster dissolves.
    s, ids = _scene([(100, 100), (110, 105)])          # two RESTING near objects → a cluster of 2
    stage = Clusters(_Cfg(radius=0.12, min_objects=2))
    _run(s, stage=stage)
    assert len(_cluster_of(s, ids[0]).members) == 2    # clustered while both rest
    s.begin_tick(1, ts=1.0)
    with s.writing("classify"):
        s.set_status(ids[0], Status.HIDDEN)            # lifted, spot bared
    _run(s, stage=stage, ts=1.0, present=set())        # identity says: NOT present (bared + quiet → GONE)
    assert s.snapshot().clusters == ()


def test_present_hidden_member_is_held():
    # HELD while present: a HIDDEN member that is still PRESENT (occluded, or its spot is active — a
    # hovering hand) stays in the cluster, same id; it must NOT dissolve or churn under the hand.
    s, ids = _scene([(100, 100), (110, 105)])
    stage = Clusters(_Cfg(radius=0.12, min_objects=2))
    _run(s, stage=stage)
    cid0 = _cluster_of(s, ids[0]).id
    assert len(_cluster_of(s, ids[0]).members) == 2
    s.begin_tick(1, ts=1.0)
    with s.writing("classify"):
        s.set_status(ids[0], Status.HIDDEN)            # hand hovers over it
    _run(s, stage=stage, ts=1.0, present={ids[0]})     # identity says: still present
    assert set(_cluster_of(s, ids[0]).members) == set(ids)
    assert _cluster_of(s, ids[0]).id == cid0           # same id — no churn


def test_settling_object_does_not_join_until_rested():
    # JOIN by settling: a blob still in motion (SETTLING, never a member) does NOT join a nearby cluster —
    # it becomes a member only once RESTING (settled). This is what the old persist_seconds faked.
    s, ids = _scene([(100, 100), (110, 105)])          # two RESTING → a cluster of 2
    stage = Clusters(_Cfg(radius=0.12, min_objects=2))
    _run(s, stage=stage)
    assert len(_cluster_of(s, ids[0]).members) == 2
    s.begin_tick(1, ts=1.0)                            # a third object appears nearby, still SETTLING (mint default)
    with s.writing("setup"):
        c = s.mint(Detection((105.0, 115.0), (100, 110, 10, 10), 100.0, None))
        s.set(c, "confirmed", True)                    # confirmed but NOT settled — the settle rule decides
    with s.writing("pose"):
        s.set(c, "pose", (105.0, 115.0, 0.0))
    _run(s, stage=stage, ts=1.0)
    assert len(_cluster_of(s, ids[0]).members) == 2    # still just the two — the settling blob doesn't flash in
    assert _cluster_of(s, c) is None
    s.begin_tick(2, ts=2.0)
    with s.writing("classify"):
        s.set_status(c, Status.RESTING)                # SETTLING → RESTING (legal) → now it joins
    _run(s, stage=stage, ts=2.0)
    assert len(_cluster_of(s, c).members) == 3


def test_held_member_survives_a_resettle():
    # HELD while present: a settled member briefly perturbed (RESTING → MOVED — a hand brushes it, it never
    # left) is HELD because it was a member last frame → the cluster neither drops it nor churns its id.
    s, ids = _scene([(100, 100), (110, 105)])          # two RESTING → cluster of 2
    stage = Clusters(_Cfg(radius=0.12, min_objects=2))
    _run(s, stage=stage)
    cid0 = _cluster_of(s, ids[0]).id
    s.begin_tick(1, ts=1.0)
    with s.writing("classify"):
        s.set_status(ids[0], Status.MOVED)             # brushed in place — in motion, but still there, held
    _run(s, stage=stage, ts=1.0)
    assert len(_cluster_of(s, ids[0]).members) == 2    # cluster intact …
    assert _cluster_of(s, ids[0]).id == cid0           # … same id, no churn


def _compound(gid, cx, cy, area=800.0, r=20):
    """A hand-built compound ObjectSnap, as the grouper publishes on the fused emit view."""
    contour = np.stack([cx + r * np.cos(np.linspace(0, 2 * np.pi, 60, endpoint=False)),
                        cy + r * np.sin(np.linspace(0, 2 * np.pi, 60, endpoint=False))],
                       1).reshape(-1, 1, 2).astype(np.float32)
    return ObjectSnap(id=gid, status=Status.RESTING, ignored=False, centroid=(float(cx), float(cy)),
                      bbox=(int(cx) - r, int(cy) - r, 2 * r, 2 * r), area=area, contour=contour,
                      pose=(float(cx), float(cy), 0.0),
                      footprint_contour=contour, footprint_centroid=(float(cx), float(cy)))


def test_compounds_cluster_as_single_voices():
    # grouper active → clusters form over the FUSED EMIT VIEW: two compounds make ONE cluster of 2
    # entities, whose `members` are the VOICE ids (the gids — both on the /pose wire), never the
    # absorbed member pieces.
    s, ids = _scene([(100, 100), (110, 105), (150, 100), (155, 110)])
    s.set_fused(frozenset(ids), (_compound(101, 105, 102), _compound(102, 152, 105)))
    _run(s, radius=0.2)
    assert len(s.snapshot().clusters) == 1
    assert set(s.snapshot().clusters[0].members) == {101, 102}   # the voices, not the absorbed objects


def test_claimed_members_do_not_cluster_directly():
    # objects absorbed into a compound are skipped: one lone compound (< min 2 entities) forms no
    # cluster even though its two member OBJECTS sit within linking range.
    s, ids = _scene([(100, 100), (110, 105)])
    s.set_fused(frozenset(ids), (_compound(101, 105, 102),))
    _run(s, radius=0.12)
    assert s.snapshot().clusters == ()


def test_retiring_object_excluded_from_clustering():
    # an object RETIRED this tick (genuinely leaving) drops out of the group
    s, ids = _scene([(100, 100), (110, 105)])
    s.begin_tick(1, ts=1.0)
    with s.writing("identity"):
        s.retire(ids[0])
    _run(s, radius=0.12)                               # only 1 present → below min → no cluster
    assert s.snapshot().clusters == ()


def test_ignored_object_excluded_from_clustering():
    # an edge/border-residue blob (a hand or arm reaching in) is flagged `ignored` — excluded from
    # output, so it must not shape cluster geometry or membership either.
    s, ids = _scene([(100, 100), (110, 105), (105, 115)])   # three near objects → would be one cluster of 3
    with s.writing("observe"):
        s.set(ids[2], "ignored", True)                      # the third is edge residue (a hand)
    _run(s, radius=0.12)
    assert len(s.snapshot().clusters) == 1
    assert set(s.snapshot().clusters[0].members) == {ids[0], ids[1]}   # only the two real objects …
    assert _cluster_of(s, ids[2]) is None                             # … the ignored blob gets no seat


# --------------------------------------------------------------------------- #
# cluster geometry (facts, not layout)
# --------------------------------------------------------------------------- #

def test_cluster_geometry_centroid_bbox_circle_hull():
    # the cluster carries membership + geometry FACTS only (no ring/step): centroid, axis-aligned bbox,
    # min-enclosing circle, convex hull — all RAW px over the member bodies (each output normalizes).
    s, ids = _scene([(100, 100), (200, 100), (150, 180)])   # a triangle; bodies are 10px boxes
    _run(s, radius=0.5)
    cl = _cluster_of(s, ids[0])
    assert cl is not None and set(cl.members) == set(ids)
    cx, cy = cl.centroid
    assert cx == pytest.approx(150.0, abs=1.0)              # mean x of 100,200,150
    assert cy == pytest.approx(126.7, abs=1.0)              # mean y of 100,100,180
    bx, by, bw, bh = cl.bbox                                # boundingRect over the three 10px bodies
    assert (bx, by) == pytest.approx((95.0, 95.0), abs=1.0)
    assert (bw, bh) == pytest.approx((110.0, 90.0), abs=1.0)
    ccx, ccy, cr = cl.circle
    for x, y in [(100, 100), (200, 100), (150, 180)]:       # every member centre inside the min-circle
        assert math.hypot(x - ccx, y - ccy) <= cr + 2.0
    assert cl.hull is not None and len(cl.hull) >= 3        # a real polygon


# --------------------------------------------------------------------------- #
# id persistence
# --------------------------------------------------------------------------- #

def test_cluster_dissolves_when_object_leaves():
    # the scene-level table drops a cluster whose survivors fall below min_objects — no stale entry.
    s, ids = _scene([(100, 100), (110, 105)])          # two near → a cluster of 2
    stage = _run(s)
    assert len(s.snapshot().clusters) == 1
    s.begin_tick(1, ts=1.0)
    with s.writing("pose"):
        s.set(ids[1], "pose", (390.0, 390.0, 0.0))     # move one far → the group breaks
    _run(s, stage=stage)
    assert s.snapshot().clusters == ()                 # gone, not stale


def test_cluster_id_persists_across_frames():
    s, ids = _scene([(100, 100), (110, 105), (105, 115)])
    stage = _run(s)                                    # tick 0
    id0 = _cluster_of(s, ids[0]).id
    s.begin_tick(1, ts=1.0)
    with s.writing("pose"):
        s.set(ids[0], "pose", (103.0, 101.0, 0.0))     # jiggle
    _run(s, stage=stage)                               # same stage → membership overlap keeps the id
    assert _cluster_of(s, ids[0]).id == id0


# --------------------------------------------------------------------------- #
# wiring + end-to-end through oscout
# --------------------------------------------------------------------------- #

def test_clusters_enabled_by_section_presence():
    assert build_config({"clusters": {"radius": "0.1"}}, schemas=readymade_tracker.SCHEMAS).is_enabled("clusters")
    assert not build_config({"input": {"source": "video"}}, schemas=readymade_tracker.SCHEMAS).is_enabled("clusters")


def test_no_clusters_flag_disables_a_present_section():
    ov = readymade_tracker._cli_overrides(readymade_tracker._parse_args(["--no-clusters"]))
    assert ov["clusters"] == {"enabled": "false"}
    assert not build_config({"clusters": {"radius": "0.1"}},
                            schemas=readymade_tracker.SCHEMAS, overrides=ov).is_enabled("clusters")


def test_cluster_flows_through_oscout():
    from lib.osc import decode_packet
    from plugins.output.oscout.oscout import OscOut

    s, ids = _scene([(100, 100), (110, 105), (105, 115)])
    _run(s)
    out = OscOut()
    out.setup(_Cfg(send="cluster"), send_fn=lambda d: None)
    msgs = {a: args for a, args in decode_packet(out.build_packet(s.snapshot()))}
    assert {"/rmt/cluster", "/rmt/cluster/members", "/rmt/cluster/circle",
            "/rmt/cluster/bbox", "/rmt/cluster/alv"} <= set(msgs)
    cid = msgs["/rmt/cluster"][0]
    assert msgs["/rmt/cluster"][1] == 3                         # n = 3 members
    assert set(msgs["/rmt/cluster/members"][1:]) == set(ids)   # the member object ids
    assert msgs["/rmt/cluster/members"][0] == cid              # keyed by the cluster id
    assert msgs["/rmt/cluster/alv"] == [cid]


def test_unconfirmed_object_is_not_a_voice():
    # every output withholds an unconfirmed object (confirm_seconds), so clusters must too — else the
    # emitted cluster counts an INVISIBLE member a client can't find in the /pose table.
    s, ids = _scene([(100, 100), (110, 105)])          # two confirmed RESTING → a cluster of 2
    stage = Clusters(_Cfg(radius=0.12, min_objects=2))
    _run(s, stage=stage)
    s.begin_tick(1, ts=1.0)                            # a third settles nearby but is NOT yet confirmed
    with s.writing("setup"):
        c = s.mint(Detection((105.0, 115.0), (100, 110, 10, 10), 100.0, None))
        s.set_status(c, Status.RESTING)
    with s.writing("pose"):
        s.set(c, "pose", (105.0, 115.0, 0.0))
    _run(s, stage=stage, ts=1.0)
    assert len(_cluster_of(s, ids[0]).members) == 2    # the invisible voice did not inflate the cluster
    assert _cluster_of(s, c) is None
    s.begin_tick(2, ts=2.0)
    with s.writing("classify"):
        s.set(c, "confirmed", True)                    # confirm latches → NOW it appears everywhere at once
    _run(s, stage=stage, ts=2.0)
    assert len(_cluster_of(s, c).members) == 3


def test_live_disable_clears_the_table_and_reenable_reclusters():
    # `enabled` is a LIVE param (mirrors the grouper): off → the very next tick publishes an EMPTY
    # table (wire + overlay show nothing), persistence dropped; on → clusters re-form (ids re-mint).
    s, ids = _scene([(100, 100), (110, 105)])          # two confirmed RESTING → a cluster of 2
    stage = Clusters(_Cfg(radius=0.12, min_objects=2))
    _run(s, stage=stage)
    assert _cluster_of(s, ids[0]) is not None
    spec = next(p for p in Clusters.PARAMS if p.key == "enabled")
    spec.apply(stage, False)                           # the ParamStore's live-set path
    s.begin_tick(1, ts=1.0)
    _run(s, stage=stage, ts=1.0)
    assert s.snapshot().clusters == ()                 # off → nothing published this tick
    spec.apply(stage, True)
    s.begin_tick(2, ts=2.0)
    _run(s, stage=stage, ts=2.0)
    assert _cluster_of(s, ids[0]) is not None          # back on → the ring re-forms live


def test_disabled_at_build_goes_live_via_the_manifest_kwarg():
    # build() passes the manifest state so a `[clusters] enabled = false` section still constructs
    # the stage (the documented live off→on without a restart).
    s, _ = _scene([(100, 100), (110, 105)])
    stage = Clusters(_Cfg(radius=0.12, min_objects=2), enabled=False)
    _run(s, stage=stage)
    assert s.snapshot().clusters == ()
