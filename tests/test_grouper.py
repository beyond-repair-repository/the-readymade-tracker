"""grouper Phase 1 — the pure EFD merge primitive: blend contours into one representative 'averaged'
shape via a weighted complex-Fourier sum. Phase 2 — the Grouper STAGE: cluster-to-budget + compound
super-objects on the scene ``groups`` channel."""

from types import SimpleNamespace

import cv2
import numpy as np

from lib.grouper import Grouper, merge_efd
from lib.scene import Detection, Scene


def _circle(cx, cy, r, n=120):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([cx + r * np.cos(t), cy + r * np.sin(t)], 1).reshape(-1, 1, 2).astype(np.int32)


def _square(cx, cy, s):
    h, e = s / 2, []
    for i in range(30): e.append((cx - h + i * s / 30, cy - h))
    for i in range(30): e.append((cx + h, cy - h + i * s / 30))
    for i in range(30): e.append((cx + h - i * s / 30, cy + h))
    for i in range(30): e.append((cx - h, cy + h - i * s / 30))
    return np.array(e, np.int32).reshape(-1, 1, 2)


def _centroid(c):
    m = cv2.moments(c)
    return (m["m10"] / m["m00"], m["m01"] / m["m00"])


def _roundness(c):                                   # std(radius)/mean — ~0 for a circle, ~0.11 for a square
    p = c.reshape(-1, 2).astype(float)
    g = p.mean(0)
    d = np.hypot(*(p - g).T)
    return d.std() / d.mean()


def test_merge_is_cv2_contour_shape():
    m = merge_efd([_circle(100, 100, 40)], samples=200)
    assert m.shape == (200, 1, 2) and m.dtype == np.float32   # SUB-PIXEL wire (int only at draw sites)


def test_two_circles_blend_at_the_midpoint_and_stay_round():
    m = merge_efd([_circle(100, 100, 40), _circle(300, 100, 40)])
    cx, cy = _centroid(m)
    assert abs(cx - 200) <= 3 and abs(cy - 100) <= 3    # weight-blended centroid = midpoint
    assert _roundness(m) < 0.05                          # two circles blend to a circle


def test_weight_collapses_to_one_member():
    m = merge_efd([_circle(100, 100, 40), _circle(300, 100, 40)], weights=[1, 0])
    cx, _ = _centroid(m)
    assert abs(cx - 100) <= 3                            # all weight on the first → its centroid


def test_circle_plus_square_is_an_averaged_shape():
    c, s = _circle(100, 100, 40), _square(100, 100, 80)
    m = merge_efd([c, s])
    a_c, a_s, a_m = cv2.contourArea(c), cv2.contourArea(s), cv2.contourArea(m)
    assert min(a_c, a_s) - 250 <= a_m <= max(a_c, a_s) + 250   # blended area sits between the members
    assert _roundness(m) > 0.03                               # a real merged shape, not a pure circle


def test_rotated_members_blend_without_collapse():
    # two squares at different orientations must NOT cancel to a tiny shape (the canonicalization guard).
    s0 = _square(100, 100, 80)
    p = _square(0, 0, 80).reshape(-1, 2).astype(float)
    R = np.array([[np.cos(0.6), -np.sin(0.6)], [np.sin(0.6), np.cos(0.6)]])
    s1 = ((p @ R.T) + [100, 100]).astype(np.int32).reshape(-1, 1, 2)
    m = merge_efd([s0, s1])
    assert cv2.contourArea(m) > 0.6 * cv2.contourArea(s0)   # area preserved, not a collapsed mush


def test_degenerate_inputs_return_none():
    assert merge_efd([]) is None
    assert merge_efd([np.array([[[5, 5]], [[6, 6]]], np.int32)]) is None   # < 3 points


# --------------------------------------------------------------------------- #
# Phase 2 — the Grouper STAGE: cluster-to-budget + compound super-objects
# --------------------------------------------------------------------------- #

class _Cfg:
    def __init__(self, **kw):
        self._d = kw
    def get(self, k, d=None):
        return self._d.get(k, d)


def _scene(objs):
    """objs = list of (x, y, area, contour|None[, confirmed[, ignored]]). Mints each as a confirmed,
    pose-pinned object — the exact shape the Grouper reads at output time."""
    s = Scene()
    s.begin_tick(0, ts=0.0)
    ids = []
    with s.writing("setup"):
        for o in objs:
            x, y, area, contour = o[0], o[1], o[2], o[3]
            ids.append(s.mint(Detection((float(x), float(y)),
                                        (int(x) - 5, int(y) - 5, 10, 10), float(area), contour)))
    with s.writing("pose"):
        for oid, o in zip(ids, objs):
            s.set(oid, "pose", (float(o[0]), float(o[1]), 0.0))
    with s.writing("classify"):
        for oid, o in zip(ids, objs):
            s.set(oid, "confirmed", bool(o[4]) if len(o) > 4 else True)
    with s.writing("observe"):
        for oid, o in zip(ids, objs):
            if len(o) > 5 and o[5]:
                s.set(oid, "ignored", True)
    return s, ids


def _run(s, max_objects=8, stage=None):
    st = stage or Grouper(_Cfg(max_objects=max_objects))
    with s.writing("grouper"):
        st(SimpleNamespace(ts=0.0), s)
    return st, s.snapshot().groups


def test_under_budget_publishes_no_groups():
    s, _ = _scene([(100, 100, 100, None), (300, 300, 100, None)])
    _, groups = _run(s, max_objects=8)
    assert groups == ()                                     # 2 objects, budget 8 → nothing to fuse


def test_disabled_never_fuses_even_over_budget():
    objs = [(x, 100, 100, None) for x in range(100, 700, 100)]     # 6 objects, budget 2 → WOULD fuse
    s, _ = _scene(objs)
    st = Grouper(_Cfg(max_objects=2, enabled=False))
    with s.writing("grouper"):
        st(SimpleNamespace(ts=0.0), s)
    assert s.snapshot().groups == ()                       # off → outputs emit raw objects


def test_over_budget_fuses_to_the_budget_and_keeps_every_object():
    objs = [(x, 100, 100, None) for x in range(100, 700, 100)]     # 6 objects in a row
    s, ids = _scene(objs)
    _, groups = _run(s, max_objects=3)
    assert len(groups) == 3                                 # 6 → exactly the budget
    members = [m for g in groups for m in g.members]
    assert sorted(members) == sorted(ids)                  # no object dropped, none double-counted


def test_nearest_objects_fuse_first():
    # two tight triples far apart; budget 2 → each triple becomes ONE group (closest-pair-first order).
    a = [(100, 100, 100, None), (112, 100, 100, None), (106, 112, 100, None)]
    b = [(500, 500, 100, None), (512, 500, 100, None), (506, 512, 100, None)]
    s, ids = _scene(a + b)
    _, groups = _run(s, max_objects=2)
    assert len(groups) == 2
    sets = sorted((frozenset(g.members) for g in groups), key=min)
    assert sets[0] == frozenset(ids[:3]) and sets[1] == frozenset(ids[3:])


def test_group_ids_are_stable_across_frames():
    objs = [(x, 100, 100, None) for x in range(100, 600, 100)]     # 5 objects
    s1, _ = _scene(objs)
    st, g1 = _run(s1, max_objects=2)
    s2, _ = _scene(objs)                                    # same ids (minted 1..N in the same order)
    _, g2 = _run(s2, stage=st)                             # reuse the SAME stage → membership overlap
    assert {g.id for g in g1} == {g.id for g in g2}         # compound ids persist, not re-minted


def test_compound_contour_blends_member_shapes():
    c1, c2 = _circle(100, 100, 30), _circle(180, 100, 30)
    s, _ = _scene([(100, 100, cv2.contourArea(c1), c1), (180, 100, cv2.contourArea(c2), c2)])
    _, groups = _run(s, max_objects=1)
    assert len(groups) == 1
    g = groups[0]
    assert g.contour is not None and g.contour.shape[1:] == (1, 2)   # a real merged cv2 contour
    cx, cy = g.centroid
    assert abs(cx - 140) <= 6 and abs(cy - 100) <= 6                 # area-weighted midpoint of the two
    assert g.area == cv2.contourArea(c1) + cv2.contourArea(c2)       # summed loudness
    assert _roundness(g.contour) < 0.08                             # two circles blend to a circle


def test_compound_is_cached_across_unchanged_frames():
    # an unchanged group must reuse last frame's GroupSnap verbatim (the expensive merge_efd is skipped) —
    # asserted by object identity: the second frame returns the SAME GroupSnap instance.
    c1, c2, c3 = _circle(100, 100, 30), _circle(180, 100, 30), _circle(260, 100, 30)
    s, _ = _scene([(100, 100, cv2.contourArea(c1), c1),
                   (180, 100, cv2.contourArea(c2), c2),
                   (260, 100, cv2.contourArea(c3), c3)])
    st = Grouper(_Cfg(max_objects=1))
    with s.writing("grouper"):
        st(SimpleNamespace(ts=0.0), s)
    g1 = s.snapshot().groups[0]
    with s.writing("grouper"):
        st(SimpleNamespace(ts=0.0), s)                     # identical scene → cache hit
    g2 = s.snapshot().groups[0]
    assert g2 is g1                                        # same instance → merge_efd was NOT recomputed


def test_unconfirmed_and_ignored_objects_are_excluded():
    # only CONFIRMED, non-ignored objects count toward the budget (what the outputs actually emit).
    objs = [(100, 100, 100, None, True, False), (200, 100, 100, None, True, False),
            (300, 100, 100, None, False, False),   # unconfirmed
            (400, 100, 100, None, True, True)]     # ignored
    s, ids = _scene(objs)
    _, groups = _run(s, max_objects=1)
    members = {m for g in groups for m in g.members}
    assert members == {ids[0], ids[1]}             # the two eligible ones fuse; the other two are absent


# --------------------------------------------------------------------------- #
# `enabled = false` constructs the stage OFF — and it can go live (off -> on)
# --------------------------------------------------------------------------- #

def test_disabled_section_constructs_off_and_goes_live():
    """The config loader pops `enabled` out of the section, so bind_params alone would start a
    disabled section's stage as enabled=True (the ParamSpec default). build() passes the manifest
    state; a live /rmt/set (the ParamSpec's apply) then flips it on WITHOUT a restart — the on/off
    the conf comment always promised."""
    import readymade_tracker
    from lib.config import build_config

    cfg = build_config({"grouper": {"enabled": "false", "max_objects": "2"}},
                       schemas=readymade_tracker.CONF_SCHEMAS)
    assert not cfg.is_enabled("grouper")
    g = Grouper(cfg.get("grouper"), enabled=cfg.is_enabled("grouper"))
    assert g._enabled is False                        # starts OFF, matching the manifest
    assert g._budget == 2                             # the rest of the section still binds

    spec = {s.key: s for s in Grouper.PARAMS}["enabled"]
    spec.apply(g, True)                               # exactly what ParamStore._write runs on /rmt/set
    assert g._enabled is True                         # off -> on, live


def test_enabled_kwarg_none_keeps_the_bound_value():
    g = Grouper(None)                                 # no manifest info -> the ParamSpec default (True)
    assert g._enabled is True


def test_hidden_objects_never_fuse_or_count():
    # HIDDEN = gone-soon: it is not a voice, must not consume the budget, and must never be fused.
    # 3 visible + 2 hidden with budget 3 → NOT over budget (3 ≤ 3) → no groups at all.
    from lib.scene import Status
    s, ids = _scene([(100, 100, 100.0, _circle(100, 100, 20)),
                     (300, 100, 100.0, _circle(300, 100, 20)),
                     (100, 300, 100.0, _circle(100, 300, 20)),
                     (300, 300, 100.0, _circle(300, 300, 20)),
                     (200, 200, 100.0, _circle(200, 200, 20))])
    s.begin_tick(1, ts=1.0)                            # status is a new-tick write (single-writer)
    with s.writing("classify"):
        s.set_status(ids[3], Status.HIDDEN)
        s.set_status(ids[4], Status.HIDDEN)
    g = Grouper(SimpleNamespace(get=lambda k, d=None: {"max_objects": 3}.get(k, d)))
    with s.writing("grouper"):
        g(SimpleNamespace(ts=1.0), s)
    assert s.snapshot().groups == ()                   # 3 visible ≤ budget 3 → no fusing

    # a fresh scene with only ONE hidden (4 visible > 3) → groups form, but the hidden id
    # appears in NO membership.
    s2, ids2 = _scene([(100, 100, 100.0, _circle(100, 100, 20)),
                       (300, 100, 100.0, _circle(300, 100, 20)),
                       (100, 300, 100.0, _circle(100, 300, 20)),
                       (300, 300, 100.0, _circle(300, 300, 20)),
                       (200, 200, 100.0, _circle(200, 200, 20))])
    s2.begin_tick(1, ts=1.0)
    with s2.writing("classify"):
        s2.set_status(ids2[4], Status.HIDDEN)
    g2 = Grouper(SimpleNamespace(get=lambda k, d=None: {"max_objects": 3}.get(k, d)))
    with s2.writing("grouper"):
        g2(SimpleNamespace(ts=1.0), s2)
    groups = s2.snapshot().groups
    assert groups                                      # 4 visible > 3 → fused to budget
    fused = {m for grp in groups for m in grp.members}
    assert ids2[4] not in fused and fused <= set(ids2)


def test_covered_hidden_keeps_its_voice_in_the_grouper():
    # a HIDDEN object still COVERED by a hand is bodily present: it keeps its voice — it counts
    # toward the budget and can be fused (identity publishes ctx.hidden_present per tick).
    from lib.scene import Status
    s, ids = _scene([(100, 100, 100.0, _circle(100, 100, 20)),
                     (300, 100, 100.0, _circle(300, 100, 20)),
                     (100, 300, 100.0, _circle(100, 300, 20)),
                     (300, 300, 100.0, _circle(300, 300, 20))])
    s.begin_tick(1, ts=1.0)
    with s.writing("classify"):
        s.set_status(ids[3], Status.HIDDEN)
    g = Grouper(SimpleNamespace(get=lambda k, d=None: {"max_objects": 3}.get(k, d)))
    with s.writing("grouper"):
        g(SimpleNamespace(ts=1.0, hidden_present={ids[3]}), s)   # covered → still a voice
    groups = s.snapshot().groups
    assert groups                                      # 4 present voices > 3 → fused
    assert ids[3] in {m for grp in groups for m in grp.members}


def test_blend_budget_follows_the_shared_efd_knobs():
    # no grouper-private EFD knobs: the compound blend budget = the [efd] linear bin budget
    # round(1 + detail*(bins-1)), LIVE — turning the detail knob re-blends the compounds.
    c1, c2 = _circle(100, 100, 30), _circle(180, 100, 30)
    efd = SimpleNamespace(bins=64, detail=1.0)
    s, _ = _scene([(100, 100, cv2.contourArea(c1), c1), (180, 100, cv2.contourArea(c2), c2)])
    st = Grouper(_Cfg(max_objects=1), efd=efd)
    assert st._blend_k() == 64                          # detail 1.0 → the full bin budget
    _run(s, stage=st)
    efd.detail = 0.0                                    # live knob turn → budget 1 (pure ellipse)
    assert st._blend_k() == 1
    s2, _ = _scene([(100, 100, cv2.contourArea(c1), c1), (180, 100, cv2.contourArea(c2), c2)])
    _, groups = _run(s2, stage=st)                      # cache signature includes the budget → re-blend
    assert len(groups) == 1 and groups[0].contour is not None
    assert Grouper(_Cfg(max_objects=1))._blend_k() == Grouper._DEFAULT_K   # unwired fallback


def test_contour_samples_track_the_detail_knob():
    # the wire outline (/rmt/grp/contour) is reconstructed at ~8 points per kept harmonic,
    # clamped 32..512 — the client's contour resolution follows the SAME detail knob as the blend.
    c1, c2 = _circle(100, 100, 30), _circle(180, 100, 30)
    efd = SimpleNamespace(bins=64, detail=1.0)
    st = Grouper(_Cfg(max_objects=1), efd=efd)
    assert st._contour_samples(st._blend_k()) == 512    # K=64 → 8·64 = 512 (also the wire-safe cap)
    efd.detail = 0.5
    assert st._contour_samples(st._blend_k()) == 256    # K=32 → 256
    efd.detail = 0.0
    assert st._contour_samples(st._blend_k()) == 32     # K=1 → the 32-point ellipse floor
    s, _ = _scene([(100, 100, cv2.contourArea(c1), c1), (180, 100, cv2.contourArea(c2), c2)])
    _, groups = _run(s, stage=st)                       # detail 0.0 live → the built contour matches
    assert len(groups) == 1 and len(groups[0].contour) == 32


def test_gids_share_the_scene_id_sequence_and_never_collide():
    """Compound ids draw from scene.reserve_id() — the SAME never-recycled sequence objects mint
    from — so a gid can never alias a live-or-future object id (the fused emit view puts both
    populations on the same wires)."""
    objs = [(x, 100, 100, _square(x, 100, 20)) for x in range(100, 700, 100)]  # 6 objects, budget 2
    s, ids = _scene(objs)
    _, groups = _run(s, max_objects=2)
    gids = {g.id for g in groups}
    assert gids and not (gids & set(ids))                   # disjoint from every existing object id
    assert min(gids) > max(ids)                             # drawn from the shared monotonic sequence
    with s.writing("setup"):
        newid = s.mint(Detection((50.0, 50.0), (45, 45, 10, 10), 100.0, None))
    assert newid not in gids                                # and a later mint can never collide back


# --------------------------------------------------------------------------- #
# Phase 3 — the FUSED EMIT VIEW: compounds replace their members on the wires
# --------------------------------------------------------------------------- #

def _fused_scene(max_objects=2):
    """4 contoured objects: three near (fuse) + one far (singleton passthrough)."""
    objs = [(100, 100, 400, _square(100, 100, 20)),
            (140, 100, 400, _square(140, 100, 20)),
            (120, 140, 400, _square(120, 140, 20)),
            (600, 400, 400, _square(600, 400, 20))]
    s, ids = _scene(objs)
    _run(s, max_objects=max_objects)
    return s, ids


def test_emit_view_replaces_claimed_members_with_the_compound():
    s, ids = _fused_scene()
    snap = s.snapshot()
    assert snap.emit is not None
    emit_ids = [o.id for o in snap.emit]
    multi = [g for g in snap.groups if len(g.members) > 1]
    assert multi                                            # over budget → at least one real fusion
    for g in multi:
        assert g.id in emit_ids                             # the compound stands in ...
        for m in g.members:
            assert m not in emit_ids                        # ... and its members are gone from the wire
    assert snap.emit_objects() == snap.emit
    assert {o.id for o in snap.objects} == set(ids)         # the REAL view keeps every member (webgui/MOT)


def test_emit_view_passes_singletons_through_as_the_same_snap():
    s, _ = _fused_scene(max_objects=3)                      # 4 objects, one merge -> two singletons
    snap = s.snapshot()
    singles = {g.members[0] for g in snap.groups if len(g.members) == 1}
    assert singles                                          # the far object stayed alone
    by_id = {o.id: o for o in snap.objects}
    emit_by_id = {o.id: o for o in snap.emit}
    for oid in singles:
        assert emit_by_id[oid] is by_id[oid]                # the untouched real snap, not a copy


def test_no_fusion_means_emit_is_none():
    s, _ = _scene([(100, 100, 100, _square(100, 100, 10)), (300, 300, 100, _square(300, 300, 10))])
    _run(s, max_objects=8)                                  # under budget
    snap = s.snapshot()
    assert snap.emit is None
    assert snap.emit_objects() is snap.objects              # the fallback IS the real view


def test_compound_contour_carries_the_summed_mass():
    s, _ = _fused_scene()
    snap = s.snapshot()
    for g in (g for g in snap.groups if len(g.members) > 1):
        comp = next(o for o in snap.emit if o.id == g.id)
        blend_area = cv2.contourArea(comp.footprint_contour.astype(np.float32))
        assert abs(blend_area - g.area) / g.area < 0.02     # mass-scaled: blend area == summed member area
        assert comp.area == g.area


def test_compound_objectsnap_contract_for_the_wires():
    s, _ = _fused_scene()
    snap = s.snapshot()
    comp = next(o for o in snap.emit if o.id in {g.id for g in snap.groups if len(g.members) > 1})
    from lib.scene import Status
    assert comp.status is Status.RESTING and comp.confirmed and not comp.ignored
    assert comp.pose == (comp.centroid[0], comp.centroid[1], 0.0)   # world-placed contour → identity placement
    assert comp.footprint_contour is comp.contour                   # one blend, both fields
    assert comp.footprint_centroid == comp.centroid
    assert not comp.footprint_contour.flags.writeable               # readonly across the thread boundary


def _describe(s, ctx=None):
    """Run the Features stage over the scene (compound pass included), pipeline-style."""
    from lib.features import Features
    st = Features(_Cfg(), efd=None)
    ctx = ctx or SimpleNamespace(ts=0.0, gray=np.zeros((480, 640), np.uint8))
    with s.writing("features"):
        st(ctx, s)
    return st


def test_features_describe_the_compound_from_its_mass_scaled_contour():
    from lib.features import size_fraction
    s, _ = _fused_scene()
    st = _describe(s)
    snap = s.snapshot()
    fa = 480.0 * 640.0
    for g in (g for g in snap.groups if len(g.members) > 1):
        comp = next(o for o in snap.emit if o.id == g.id)
        feats = comp.annotations["features"]
        assert "size" in feats and "circularity" in feats     # the normal person-facing set, no special case
        want = size_fraction(g.area, fa)                      # the contour carries the mass -> size falls out
        assert abs(feats["size"][0] - want) / want < 0.15
        assert comp.wire_contour is not None and not comp.wire_contour.flags.writeable
        assert comp.efd_full is not None                      # the features view follows the wire: the
        #                                                       inspector dims the compound's rejected bins


def test_compound_features_are_cached_on_blend_identity():
    objs = [(100, 100, 400, _square(100, 100, 20)),
            (140, 100, 400, _square(140, 100, 20)),
            (120, 140, 400, _square(120, 140, 20)),
            (600, 400, 400, _square(600, 400, 20))]
    s, _ = _scene(objs)
    grouper, _ = _run(s, max_objects=2)                       # ONE stage instance, like the pipeline
    st = _describe(s)
    first = {o.id: o for o in s.snapshot().emit}
    # a second identical tick THROUGH THE SAME STAGES: grouper reuses the cached pair -> features
    # must republish the SAME annotated snaps (no recompute) -> identity-keyed caches stay warm
    s.begin_tick(1, ts=1.0)
    _run(s, stage=grouper)
    with s.writing("features"):
        st(SimpleNamespace(ts=1.0, gray=np.zeros((480, 640), np.uint8)), s)
    second = {o.id: o for o in s.snapshot().emit}
    comps = [o for o in second.values() if o.annotations]
    assert comps                                              # the fusion survived the tick
    for snap_obj in comps:
        assert snap_obj is first[snap_obj.id]


def _rect(cx, cy, w, h):
    e = []
    for i in range(30): e.append((cx - w / 2 + i * w / 30, cy - h / 2))
    for i in range(30): e.append((cx + w / 2, cy - h / 2 + i * h / 30))
    for i in range(30): e.append((cx + w / 2 - i * w / 30, cy + h / 2))
    for i in range(30): e.append((cx - w / 2, cy + h / 2 - i * h / 30))
    return np.array(e, np.int32).reshape(-1, 1, 2)


def test_compound_contour_follows_the_union_axis():
    """merge_efd returns the blend CANONICAL — major axis along +x — so before the fix EVERY
    anisotropic compound drew WIDE/horizontal regardless of the physical layout (the crumbs-clip
    screenshot: a vertical mass wearing a horizontal ghost). The stage must rotate the blend to
    the members' union principal axis: a TALL member dominating the blend + a crumb below it is a
    VERTICAL union and must draw TALL. (Isotropic members stay isotropic — the blend is a shape
    AVERAGE, not a union silhouette — so the rotation is a no-op there, by design.)"""
    import math

    def _axis_and_aspect(contour):
        pts = np.asarray(contour, float).reshape(-1, 2)
        pts = pts - pts.mean(0)
        mu20, mu02 = (pts[:, 0] ** 2).mean(), (pts[:, 1] ** 2).mean()
        mu11 = (pts[:, 0] * pts[:, 1]).mean()
        ang = math.degrees(0.5 * math.atan2(2 * mu11, mu20 - mu02))
        w = pts[:, 0].max() - pts[:, 0].min()
        h = pts[:, 1].max() - pts[:, 1].min()
        return ang, w, h

    # a TALL sheet + a small crumb below -> the union axis is vertical -> the compound draws TALL
    s, _ = _scene([(300, 240, 1200, _rect(300, 240, 20, 60)),
                   (300, 330, 400, _square(300, 330, 20))])
    _run(s, max_objects=1)
    snap = s.snapshot()
    comp = next(o for o in snap.emit if o.id in {g.id for g in snap.groups if len(g.members) > 1})
    ang, w, h = _axis_and_aspect(comp.contour)
    assert h > w, f"vertical layout must blend TALL, got {w:.0f}x{h:.0f}"
    assert abs(abs(ang) - 90.0) < 15.0, f"axis should be near vertical, got {ang:.1f}deg"

    # the same pair laid HORIZONTALLY -> WIDE
    s2, _ = _scene([(240, 300, 1200, _rect(240, 300, 60, 20)),
                    (330, 300, 400, _square(330, 300, 20))])
    _run(s2, max_objects=1)
    snap2 = s2.snapshot()
    comp2 = next(o for o in snap2.emit if o.id in {g.id for g in snap2.groups if len(g.members) > 1})
    ang2, w2, h2 = _axis_and_aspect(comp2.contour)
    assert w2 > h2, f"horizontal layout must blend WIDE, got {w2:.0f}x{h2:.0f}"
    assert abs(ang2) < 15.0 or abs(abs(ang2) - 180.0) < 15.0, f"axis should be near horizontal, got {ang2:.1f}deg"
