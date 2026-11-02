"""lib/geometry — the ONE placement transform every outline consumer shares (gui, tuio2, oscout,
webgui) and the approx-poly simplifier. Until now this module had no direct test: placement was
only ever exercised THROUGH the output plugins, so its arithmetic was asserted as coordinate
RANGES, never as exact values, and the 0.31.13 int32 -> float32 change rode on those loose bounds.
These tests pin the transform itself: exact rotated coordinates, sub-pixel survival, rigidity."""

import math

import numpy as np

from lib.geometry import place_footprint, simplify_contour


def _square(cx, cy, s, dtype=np.int32):
    """The four corners of a square centred on (cx, cy), cv2 (4,1,2) — corner order matters to the
    exact-rotation test, so it is fixed: TL, TR, BR, BL in a y-DOWN frame."""
    h = s / 2.0
    return np.array([[[cx - h, cy - h]], [[cx + h, cy - h]],
                     [[cx + h, cy + h]], [[cx - h, cy + h]]], dtype)


def _pairwise(c):
    p = np.asarray(c, float).reshape(-1, 2)
    return np.hypot(*(p[:, None, :] - p[None, :, :]).T)      # every point-to-point distance


def test_place_footprint_lands_the_rest_centroid_on_the_pose():
    # theta = 0: pure translation. The rest centroid (10,10) must land EXACTLY on the pose (100,50),
    # and every vertex keeps its offset from it — this is what "the outline tracks the pose" means.
    fp = _square(10, 10, 10)
    placed = place_footprint(fp, (10.0, 10.0), (100.0, 50.0, 0.0)).reshape(-1, 2)
    assert np.allclose(placed.mean(axis=0), [100.0, 50.0])
    assert np.allclose(placed, [[95, 45], [105, 45], [105, 55], [95, 55]])


def test_place_footprint_rotates_about_the_footprint_centroid_not_the_origin():
    # theta = +90 deg about the REST centroid, then translate to the pose. Rotating about the origin
    # instead would fling the shape across the frame, so the exact corners are the assertion:
    # (dx,dy) -> (-dy, dx) in this y-down frame, + the pose. TL(-5,-5) -> (+5,-5) -> (105,45).
    fp = _square(10, 10, 10)
    placed = place_footprint(fp, (10.0, 10.0), (100.0, 50.0, math.pi / 2)).reshape(-1, 2)
    assert np.allclose(placed, [[105, 45], [105, 55], [95, 55], [95, 45]], atol=1e-4)


def test_place_footprint_keeps_sub_pixel_precision():
    # THE 0.31.13 GUARD. The transform used to end in .astype(np.int32): every placement re-snapped
    # the outline to whole pixels, and a display scaling a small object up turned each 1-px step into
    # a visible stair. A fractional pose must now survive to the caller intact.
    fp = _square(10, 10, 10)
    placed = place_footprint(fp, (10.0, 10.0), (100.25, 50.75, 0.0))
    assert placed.dtype == np.float32
    assert np.abs(placed - np.rint(placed)).max() > 0.2      # genuinely BETWEEN the pixels


def test_place_footprint_is_a_rigid_motion():
    # rotate + translate must not scale or shear: every pairwise distance is preserved. Cheap
    # invariant that catches a sign flip or a transposed rotation matrix at any angle.
    fp = _square(10, 10, 10)
    for th in (0.0, 0.3, math.pi / 2, 2.0, math.pi):
        placed = place_footprint(fp, (10.0, 10.0), (100.0, 50.0, th))
        assert np.allclose(_pairwise(fp), _pairwise(placed), atol=1e-3), f"not rigid at {th}"


def test_place_footprint_accepts_both_contour_layouts_and_returns_cv2_shape():
    # callers hand it cv2 (N,1,2) footprints AND bare (N,2) point sets; the reshape(-1, 2) up front
    # takes either, and the return is always cv2-shaped so outputs can hand it straight to cv2.
    fp = _square(10, 10, 10)
    a = place_footprint(fp, (10.0, 10.0), (100.0, 50.0, 0.4))
    b = place_footprint(fp.reshape(-1, 2), (10.0, 10.0), (100.0, 50.0, 0.4))
    assert a.shape == (4, 1, 2) and b.shape == (4, 1, 2)
    assert np.allclose(a, b)


def test_place_footprint_takes_float_footprints():
    # the wire contour reaching this function is itself float32 (perceptual_contour, 0.31.13) —
    # a float source must not be quantized on the way through.
    fp = _square(10.0, 10.0, 10.0, dtype=np.float32) + 0.5
    placed = place_footprint(fp, (10.5, 10.5), (100.0, 50.0, 0.0)).reshape(-1, 2)
    assert placed.dtype == np.float32
    assert np.allclose(placed, [[95, 45], [105, 45], [105, 55], [95, 55]])


def test_simplify_contour_passes_degenerate_input_through_untouched():
    # eps <= 0 is the documented "no simplification" switch (the perceptual path relies on it), and a
    # None / sub-triangle contour must never reach approxPolyDP, which would raise.
    fp = _square(10, 10, 10)
    assert simplify_contour(fp, 0.0) is fp
    assert simplify_contour(fp, -1.0) is fp
    assert simplify_contour(None, 0.75) is None
    two = np.array([[[0, 0]], [[1, 1]]], np.int32)
    assert simplify_contour(two, 0.75) is two


def test_simplify_contour_preserves_sub_pixel_dtype():
    # the ONE place sub-pixel can break ASYMMETRICALLY: oscout's fallback path composes
    # simplify_contour(place_footprint(...)) while the perceptual path skips simplify entirely, so a
    # quantizing approxPolyDP would leave the wire correct and stair-step only the features-off
    # fallback — invisible in every other test.
    dense = (_square(10, 10, 10, dtype=np.float32).reshape(-1, 2) + 0.25).reshape(-1, 1, 2)
    out = simplify_contour(dense, 0.1)
    assert out.dtype == np.float32
    assert np.abs(out - np.rint(out)).max() > 0.2


def test_simplify_contour_leaves_an_empty_contour_alone():
    # tuio2 calls place_footprint BEFORE its len(c) < 3 guard, so an empty contour reaches these
    # helpers in the ordinary course of events and must pass through rather than raise.
    empty = np.zeros((0, 1, 2), np.float32)
    assert place_footprint(empty, (0.0, 0.0), (1.0, 1.0, 0.5)).shape == (0, 1, 2)
    assert simplify_contour(empty, 0.75) is empty


def test_simplify_contour_drops_vertices_at_a_real_epsilon():
    # a densely sampled square is mostly collinear filler: a real epsilon must collapse it toward its
    # four corners without moving them (the fallback wire-quality path).
    h, pts = 5.0, []
    for i in range(40):
        f = i / 10.0 % 1.0
        side = i // 10
        pts.append([(-h + 2 * h * f, -h), (h, -h + 2 * h * f),
                    (h - 2 * h * f, h), (-h, h - 2 * h * f)][side])
    dense = (np.array(pts, np.float32) + 10.0).reshape(-1, 1, 2)
    simple = simplify_contour(dense, 1.0)
    assert len(simple) < len(dense) and len(simple) >= 4
    for corner in ([5, 5], [15, 5], [15, 15], [5, 15]):      # corners survive the collapse
        assert np.abs(np.asarray(simple, float).reshape(-1, 2) - corner).sum(axis=1).min() < 1.5
