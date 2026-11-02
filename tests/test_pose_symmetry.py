"""pose — the SYMMETRY LOCK. A rotationally symmetric shape (circle / ring / disc) has no
measurable orientation: the correlation matches at every angle, so a fitted θ is noise. Such
objects are detected once from the canonical contour (``efd.rotation_information``) and their θ
is held at a constant 0, with the settle fit skipped (it never charges the per-frame budget)."""

import math
from types import SimpleNamespace

import numpy as np

from lib.efd import rotation_by_correlation, rotation_information
from lib.pose import Pose, asymmetry_axis_up, principal_axis_up
from lib.scene import Status


def _circle(r=60.0, n=256):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([r * np.cos(t) + 100, r * np.sin(t) + 100], axis=1)


def _blob(n=256):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    rad = 60 + 14 * np.sin(3 * t) + 8 * np.cos(5 * t)
    return np.stack([rad * np.cos(t) + 100, rad * np.sin(t) + 100], axis=1)


def _rotated(contour, deg):
    ctr = contour.mean(axis=0)
    a = math.radians(deg)
    rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    return (contour - ctr) @ rot.T + ctr


class _Cfg:
    def __init__(self, **kw):
        self._d = kw
    def get(self, k, d=None):
        return self._d.get(k, d)


class _Scene:
    """Records set() calls and mirrors them onto the object so later reads see them."""
    def __init__(self, objs):
        self._objs = {o.id: o for o in objs}
        self.sets = []
    def recent_rewrites(self):
        return ()
    def objects(self):
        return list(self._objs.values())
    def set(self, oid, field, value):
        self.sets.append((oid, field, value))
        setattr(self._objs[oid], field, value)


def _obj(oid, contour):
    ctr = contour.mean(axis=0)
    return SimpleNamespace(
        id=oid, status=Status.RESTING, footprint=object(), footprint_contour=contour,
        footprint_centroid=(float(ctr[0]), float(ctr[1])), pose_ref=None, pose=None,
        theta_cumulative=0.0)


def _settle(pose, scene, obj):
    """Drive one first-settle / re-settle of ``obj`` through the real _place_at_settle path."""
    pose._last[obj.id] = Status.MOVED
    pose(SimpleNamespace(), scene)


def test_rotation_information_separates_circle_from_blob():
    assert rotation_information(_circle()) < 0.005          # no orientation signal
    assert rotation_information(_blob()) > 0.02             # plenty


def test_circle_theta_locked_at_zero():
    p = Pose(_Cfg())
    o = _obj(1, _circle())
    scene = _Scene([o])
    _settle(p, scene, o)                                    # first settle → canon + lock verdict
    assert p._locked[1] is True
    o.footprint_contour = _rotated(_circle(), 40.0)         # a "rotated" circle re-settles
    _settle(p, scene, o)
    assert o.theta_cumulative == 0.0                        # θ stays a constant 0
    assert o.pose_conf == 1.0


def test_blob_rotation_still_measured():
    p = Pose(_Cfg())
    o = _obj(1, _blob())
    scene = _Scene([o])
    _settle(p, scene, o)
    assert p._locked[1] is False
    asym = asymmetry_axis_up(_blob())                       # the shape anchor, asymmetry-first
    axis = math.degrees(asym if asym is not None else principal_axis_up(_blob()))
    o.footprint_contour = _rotated(_blob(), 25.0)
    ctr = o.footprint_contour.mean(axis=0)
    o.footprint_centroid = (float(ctr[0]), float(ctr[1]))
    _settle(p, scene, o)
    err = (math.degrees(o.theta_cumulative) - (axis + 25.0) + 180.0) % 360.0 - 180.0
    assert abs(err) < 3.0                                   # the real fit runs: anchor + ~25° of rotation


def test_symmetry_lock_zero_disables():
    p = Pose(_Cfg(symmetry_lock=0.0))
    o = _obj(1, _circle())
    scene = _Scene([o])
    _settle(p, scene, o)
    assert p._locked[1] is False                            # 0 = never lock, always fit


def _ellipse(a=120.0, b=50.0, n=256):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([a * np.cos(t) + 200, b * np.sin(t) + 200], axis=1)


def test_two_fold_tie_resolves_toward_previous_theta():
    # a perfect ellipse correlates EXACTLY equally at θ and θ+180° — without hysteresis numeric
    # noise picks the winner (the bar's random 180° flip). prev_theta must select the continuity
    # branch in BOTH directions.
    ell = _ellipse()
    th0, s0 = rotation_by_correlation(ell, ell, prev_theta=0.0)
    assert min(th0, 2 * math.pi - th0) < math.radians(3)
    th180, s180 = rotation_by_correlation(ell, ell, prev_theta=math.pi)
    assert abs(th180 - math.pi) < math.radians(3)
    assert s0 > 0.95 and s180 > 0.95


def test_bar_resettle_holds_theta_not_a_flip():
    # a 2-fold bar-like object re-settling UNMOVED must keep θ ≈ its anchor — never read a 180° flip.
    # (It is well above the symmetry lock: elongation carries real orientation signal.)
    p = Pose(_Cfg())
    o = _obj(1, _ellipse())
    scene = _Scene([o])
    _settle(p, scene, o)
    assert p._locked[1] is False
    axis = principal_axis_up(_ellipse())                    # horizontal bar → π/2 (arrow along the bar)
    assert abs(o.theta_cumulative - axis) < math.radians(1)  # first settle reports the shape anchor
    o.footprint_contour = _ellipse()
    _settle(p, scene, o)
    d = abs(o.theta_cumulative - axis) % (2 * math.pi)      # unmoved re-settle: still the anchor, no flip
    assert min(d, 2 * math.pi - d) < math.radians(3)


def test_first_settle_theta_is_the_shape_axis():
    # the reported θ is SHAPE-ANCHORED: 0 = principal axis screen-up. A vertical bar settles at ~0,
    # a horizontal one at ~π/2 (the arrow lies along the visible elongation either way); a circle
    # (locked) and a near-isotropic shape anchor at exactly 0.
    for contour, want in [(_ellipse(50.0, 120.0), 0.0), (_ellipse(120.0, 50.0), math.pi / 2), (_circle(), 0.0)]:
        p = Pose(_Cfg())
        o = _obj(1, contour)
        scene = _Scene([o])
        _settle(p, scene, o)
        d = abs(o.theta_cumulative - want) % (2 * math.pi)
        assert min(d, 2 * math.pi - d) < math.radians(1)


def test_locked_resettle_does_not_charge_the_fit_budget():
    # one locked circle + one blob re-settling in the SAME frame with budget 1: the circle costs
    # nothing, so the blob's expensive fit must still run this frame.
    p = Pose(_Cfg(max_fits_per_frame=1))
    circ, blob = _obj(1, _circle()), _obj(2, _blob())
    scene = _Scene([circ, blob])
    _settle(p, scene, circ)                                 # first settles (decide the lock)
    _settle(p, scene, blob)
    circ.footprint_contour = _circle()
    blob.footprint_contour = _rotated(_blob(), 10.0)
    for o in (circ, blob):
        p._last[o.id] = Status.MOVED
    scene.sets.clear()
    p(SimpleNamespace(), scene)                             # ONE frame, both re-settle
    placed = {oid for oid, field, _ in scene.sets if field == "pose_conf"}
    assert placed == {1, 2}


# --------------------------------------------------------------------------- #
# [pose] measure_rotation — rotation off, placement intact
# --------------------------------------------------------------------------- #

def test_measure_rotation_off_still_places_but_reports_zero_theta():
    """OFF must cost only the ROTATION. Placement, ids and the emitted outline are untouched; θ is 0.
    That is correct rather than lossy: the footprint is re-captured at the object's CURRENT
    orientation each settle and the outline is placed with pose[2] (always 0), so the turn is already
    baked into the shape every consumer draws. A held θ would double-count it."""
    shape = _blob()
    pose = Pose(_Cfg(measure_rotation=False))
    obj = _obj(1, shape)
    scene = _Scene([obj])
    _settle(pose, scene, obj)
    assert obj.pose is not None and obj.pose[:2] == obj.footprint_centroid   # PLACED
    assert obj.pose[2] == 0.0                                               # committed θ always 0
    assert obj.theta_cumulative == 0.0                                      # nothing reported

    # …and a re-settle of a physically ROTATED object still reports 0 (no fit ran at all)
    obj.footprint_contour = _rotated(shape, 40)
    ctr = obj.footprint_contour.mean(axis=0)
    obj.footprint_centroid = (float(ctr[0]), float(ctr[1]))
    _settle(pose, scene, obj)
    assert obj.theta_cumulative == 0.0


def test_measure_rotation_off_spends_no_fit_budget():
    """A skipped fit must not consume max_fits_per_frame, or several objects re-settling together
    would defer each other for nothing — the budget exists to bound REAL fit cost."""
    pose = Pose(_Cfg(measure_rotation=False, max_fits_per_frame=1))
    objs = [_obj(i, _blob()) for i in (1, 2, 3)]
    scene = _Scene(objs)
    for o in objs:                                   # every object on its settle edge THIS frame
        pose._canon[o.id] = (o.footprint_contour, o.footprint_centroid)   # → all are RE-settles
        pose._last[o.id] = Status.MOVED
    pose(SimpleNamespace(), scene)
    assert all(o.pose is not None for o in objs), "a skipped fit wrongly consumed the frame budget"


def test_measure_rotation_can_be_turned_back_on_live():
    """The knob is LIVE. Turning it on must measure against the ORIGINAL canonical shape — the off
    path still seeds the reference, so θ stays absolute instead of re-anchoring to wherever the
    object happened to lie while measurement was off."""
    shape = _blob()
    pose = Pose(_Cfg(measure_rotation=False))
    obj = _obj(1, shape)
    scene = _Scene([obj])
    _settle(pose, scene, obj)                        # first settle with rotation OFF
    assert obj.id in pose._canon, "the off path must still seed the canonical reference"
    pose._measure_rot = True                         # operator flips it on (a live /rmt/set)
    obj.footprint_contour = _rotated(shape, 30)      # the object is turned 30°
    ctr = obj.footprint_contour.mean(axis=0)
    obj.footprint_centroid = (float(ctr[0]), float(ctr[1]))
    _settle(pose, scene, obj)
    # θ is SHAPE-ANCHORED: the reported value is the shape's own axis anchor PLUS the rotation
    # since capture (same convention as test_blob_rotation_still_measured).
    asym = asymmetry_axis_up(shape)                  # the anchor is asymmetry-first (0.32.130)
    axis = math.degrees(asym if asym is not None else principal_axis_up(shape))
    err = (math.degrees(obj.theta_cumulative) - (axis + 30.0) + 180.0) % 360.0 - 180.0
    assert abs(err) < 3.0, \
        f"measured {math.degrees(obj.theta_cumulative):.1f}°, expected anchor {axis:.1f}° + 30°"


# ── the ASYMMETRY anchor (0.32.130): a unique bump — a cup's handle — defines zero, so
#    look-alike objects agree and the reported θ points where the handle points. Shapes
#    with no unique bump (an ellipse: both ends tie) keep the principal-axis anchor. ──

def _cup(handle_deg=0.0, n=256):
    """A mildly elliptical rim with one narrow handle bump at the given MATH angle — a pure
    disc+bump falls under the symmetry lock (the bump alone carries ~0.003 rotation info);
    real cup rims are elliptical enough to pass, so the synthetic one is too. The ellipse
    pulls the tip-arc average ~3° off the bump — a constant, equivariant bias (the anchor
    is a per-object constant), hence the 5° tolerance."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    h = math.radians(handle_deg)
    dang = np.angle(np.exp(1j * (t - h)))
    rad = 60 + 8 * np.cos(2 * (t - h - 0.3)) + 14 * np.exp(-(dang / 0.35) ** 2)
    return np.stack([rad * np.cos(t) + 100, rad * np.sin(t) + 100], axis=1)


def _upcw(math_ang):
    """A math-angle direction → the tracker's 0-is-up clockwise-positive angle."""
    return math.atan2(math.cos(math_ang), -math.sin(math_ang))


def test_asymmetry_anchor_points_at_the_handle():
    for hdeg in (0.0, 90.0, 180.0, -45.0):
        a = asymmetry_axis_up(_cup(hdeg))
        assert a is not None
        err = (math.degrees(a - _upcw(math.radians(hdeg))) + 180.0) % 360.0 - 180.0
        assert abs(err) < 5.0


def test_asymmetry_anchor_agrees_between_lookalikes():
    # two identical cups first met at different orientations: each anchor tracks ITS handle,
    # so the anchors differ by exactly the placement difference — zero MEANS the same thing.
    a1, a2 = asymmetry_axis_up(_cup(30.0)), asymmetry_axis_up(_cup(150.0))
    d = (math.degrees(a2 - a1) + 180.0) % 360.0 - 180.0
    assert abs(abs(d) - 120.0) < 5.0


def test_ellipse_has_no_unique_bump():
    assert asymmetry_axis_up(_ellipse()) is None            # both ends tie → principal-axis fallback


def test_cup_first_settle_reports_the_handle_direction():
    p = Pose(_Cfg())
    o = _obj(1, _cup(0.0))                                  # handle at math-angle 0 = screen-RIGHT
    scene = _Scene([o])
    _settle(p, scene, o)
    assert p._locked[1] is False                            # the handle carries rotation signal
    err = (math.degrees(o.theta_cumulative) - 90.0 + 180.0) % 360.0 - 180.0
    assert abs(err) < 5.0                                   # right = +90° in the up-cw convention
