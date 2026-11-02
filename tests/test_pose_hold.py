"""pose._place_at_settle — the low-confidence HOLD must hold the ACCUMULATED θ.

The committed pose θ is 0.0 by construction (the footprint is re-captured at the current
orientation), so the hold branch must read ``theta_cumulative`` — reading ``obj.pose[2]`` silently
RESET the reported absolute orientation to 0 on any poor re-settle fit (occlusion, shape damage).
"""
from types import SimpleNamespace

import numpy as np

from lib.pose import Pose


def _contour():
    return np.array([[[0, 0]], [[10, 0]], [[10, 4]], [[0, 4]]], np.int32)


def _obj(theta_cum):
    c = _contour()
    return SimpleNamespace(id=7, footprint_centroid=(50.0, 50.0), footprint_contour=c,
                           footprint=np.zeros((100, 100), np.uint8),
                           pose=(50.0, 50.0, 0.0),          # committed θ is ALWAYS 0 — the trap
                           pose_ref=(c, (40.0, 40.0)),      # existing canon → the re-settle branch
                           theta_cumulative=theta_cum, observation=None)


def _scene(sets):
    return SimpleNamespace(set=lambda i, f, v: sets.append((f, v)))


def _pose(score):
    p = Pose(None)
    p._conf_min = 0.4
    p._locked[7] = False                                    # skip the symmetry judgment
    p._measure_theta = lambda *a, **k: (0.123, score)       # stub the fit itself
    return p


def test_low_confidence_fit_holds_accumulated_theta():
    obj, sets = _obj(0.777), []
    _pose(score=0.1)._place_at_settle(_scene(sets), obj)    # poor fit → hold
    written = dict(sets)
    assert "theta_cumulative" not in written                # 0.777 held (unchanged → no write)


def test_confident_fit_updates_theta():
    obj, sets = _obj(0.777), []
    _pose(score=0.9)._place_at_settle(_scene(sets), obj)    # good fit → adopt the measurement
    assert dict(sets)["theta_cumulative"] == 0.123


def test_degenerate_measure_holds_accumulated_theta():
    """A missing reference/target returns (held θ, score 0) — not (0, 0)."""
    p = Pose(None)
    obj = _obj(0.777)
    theta, score = p._measure_theta(obj, None, 0.0, 0.0, 0.0, 0.0)
    assert theta == 0.777 and score == 0.0
