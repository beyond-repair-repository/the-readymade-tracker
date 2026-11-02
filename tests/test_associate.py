"""observe._associate — the 4-pass blob→object matching, the heart of the hot loop.

The audit flagged it untested: only the edge gate and the reid veto had coverage. This drives
_associate directly with hand-built objects + blobs (no motion-model plumbing) to pin the load-bearing
rules: overlap continuity keeps an id frame-to-frame, a MOVING mass is never a match target and never
mints, and the AREA gate stops an arm-sized blob from stealing a small captured object's id.
"""
from types import SimpleNamespace

import cv2
import numpy as np

from lib.observe import Observation, Observe
from lib.scene import Status


def _observe():
    return Observe(SimpleNamespace(get=lambda k, d=None: {
        "min_area": 50, "fg_threshold": 40, "reassoc_dist": 64,
        "assoc_size_tol": 0.5, "relocate_dist": 160}.get(k, d)))


def _mask(x, y, w, h, W=200, H=200):
    m = np.zeros((H, W), np.uint8)
    m[y:y + h, x:x + w] = 255
    return m


def _sq_contour(x, y, w, h):
    return np.array([[[x, y]], [[x + w, y]], [[x + w, y + h]], [[x, y + h]]], np.int32)


def _blob(x, y, w, h, *, area=None, in_motion=False):
    mask = _mask(x, y, w, h)
    return {"centroid": (x + w / 2.0, y + h / 2.0), "bbox": (x, y, w, h),
            "area": float(area if area is not None else w * h),
            # "px" is the RASTERIZED count, "area" the shoelace one — Observe._blobs sets both and
            # they are NOT equal (px > area, by 2x for a ~1-px sliver). Counted from the fixture's
            # own mask so the two stay honest even when a test passes a synthetic `area`.
            "px": int(cv2.countNonZero(mask)),
            "mask": mask, "contour": _sq_contour(x, y, w, h), "in_motion": in_motion}


def _obj_live(oid, x, y, w, h):
    """An object SEEN last frame — its observation.mask is the continuity reference (no shape gate)."""
    obs = Observation(residual=0, missing=0, xor_prev=0, centroid=(x + w / 2.0, y + h / 2.0),
                      area=float(w * h), bbox=(x, y, w, h), mask=_mask(x, y, w, h))
    return SimpleNamespace(id=oid, status=Status.RESTING, observation=obs, footprint=None,
                           footprint_contour=None, footprint_area=None,
                           centroid=(x + w / 2.0, y + h / 2.0), embeds=None)


def _obj_captured(oid, x, y, w, h):
    """A captured object UNSEEN this frame — the overlap ref is its footprint (+ shape gate)."""
    return SimpleNamespace(id=oid, status=Status.RESTING, observation=None, footprint=_mask(x, y, w, h),
                           footprint_contour=_sq_contour(x, y, w, h), footprint_area=float(w * h),
                           centroid=(x + w / 2.0, y + h / 2.0), embeds=None)


def test_overlap_matches_a_stable_overlapping_blob():
    """Pass 1: a stable blob overlapping an object's last-frame mask keeps that object's id."""
    obs = _observe()
    obj = _obj_live(1, 40, 40, 20, 20)                       # area 400
    blob = _blob(44, 44, 20, 20)                             # shifted 4px, still overlapping, same area
    match, unmatched = obs._associate((obj,), [blob], None)
    assert match[1] is blob and unmatched == []              # id continuity, nothing left to mint


def test_in_motion_blob_never_matches_and_never_mints():
    """A moving mass (in_motion) is 'motion', not an object: it is never a match target for a nearby
    object AND never appears among the unmatched blobs that would mint."""
    obs = _observe()
    obj = _obj_live(1, 40, 40, 20, 20)
    moving = _blob(44, 44, 20, 20, in_motion=True)           # sits right on the object, but MOVING
    match, unmatched = obs._associate((obj,), [moving], None)
    assert match[1] is None                                  # object unseen this frame (not stolen by motion)
    assert unmatched == []                                   # and no spurious mint from the moving mass


def test_area_gate_blocks_an_arm_from_stealing_a_captured_id():
    """The decisive size gate: a huge stable mass (an arm) covering a small captured object cannot take
    its id through ANY pass (overlap area-gate + proximity area-gate) — it mints its own instead."""
    obs = _observe()
    obj = _obj_captured(1, 90, 90, 20, 20)                   # footprint area 400
    arm = _blob(20, 20, 160, 160)                            # a 160×160 mass (area 25600) covering it
    match, unmatched = obs._associate((obj,), [arm], None)
    assert match[1] is None                                  # the arm did NOT migrate onto the object's id
    assert arm in unmatched                                  # it's a fresh, stable blob → its OWN new id


def test_same_size_overlapping_blob_reassociates_a_captured_object():
    """The gate is size-similarity, not size-blindness: a same-size blob landing on the captured
    footprint DOES re-associate (shape matches too), so a briefly-lost object keeps its id."""
    obs = _observe()
    obj = _obj_captured(1, 90, 90, 20, 20)                   # footprint area 400
    ret = _blob(92, 92, 20, 20)                              # same size, overlapping, same square shape
    match, unmatched = obs._associate((obj,), [ret], None)
    assert match[1] is ret and unmatched == []               # re-associated, not re-minted


def test_detection_blur_kills_specks_without_touching_the_frame():
    """[observe] blur denoises ONLY the detection path: a 1-px salt speck above fg_threshold vanishes
    from the fg mask through _detection_gray, while ctx.gray itself is never modified (reid patches /
    textures stay sharp). blur = 0 is a true no-op (the SAME array back, no copy)."""
    obs = Observe(SimpleNamespace(get=lambda k, d=None: {
        "min_area": 50, "fg_threshold": 40, "blur": 0.6}.get(k, d)))
    gray = np.zeros((100, 100), np.uint8)
    gray[50, 50] = 255                                       # a single hot pixel (salt noise)
    raw_fg = obs._obj_fg(gray)
    assert raw_fg[50, 50] == 255                             # unblurred: the speck passes the cut
    det = obs._detection_gray(gray)
    assert det is not gray and gray[50, 50] == 255           # a blurred COPY — the source stays sharp
    assert obs._obj_fg(det)[50, 50] == 0                     # σ≈1.8 spreads it far below the cut
    obs0 = Observe(SimpleNamespace(get=lambda k, d=None: {"fg_threshold": 40}.get(k, d)))
    assert obs0._detection_gray(gray) is gray                # blur 0 (default) → the same array, no cost


def test_thin_sliver_blob_does_not_break_the_overlap_iou():
    """A ~1-px-wide blob (a wire, a felt rim, a specular edge) RASTERIZES to about twice its
    shoelace area — filling a contour lights every pixel the boundary touches. The IoU once mixed
    the two units, `ov / (ref_area + blob_area - ov)` with `ov` a countNonZero and the areas
    shoelace m00, so the denominator collapsed to 2A - 2A = 0 at exactly that 2x ratio and went
    NEGATIVE past it — a ZeroDivisionError on the crash path, and a negative IoU (sorting a real
    match last) on the near-miss path. Computed in pixel space the denominator is >= ov > 0."""
    w, h = 200, 1
    mask = _mask(10, 10, w, h)
    px = int(cv2.countNonZero(mask))
    shoelace = px / 2.0                                  # what m00 reports for a 1-px-wide contour
    obs_rec = Observation(residual=0, missing=0, xor_prev=0, centroid=(110.0, 10.5),
                          area=shoelace, bbox=(10, 10, w, h), mask=mask)
    obj = SimpleNamespace(id=1, status=Status.RESTING, observation=obs_rec, footprint=None,
                          footprint_contour=None, footprint_area=None,
                          centroid=(110.0, 10.5), embeds=None)
    blob = _blob(10, 10, w, h, area=shoelace)            # px=200, area=100 — the real 2x split
    assert blob["px"] == 2 * blob["area"]                # the fixture reproduces the ratio
    match, unmatched = _observe()._associate((obj,), [blob], None)
    assert match.get(1) is blob and unmatched == []      # associates, and does not raise
