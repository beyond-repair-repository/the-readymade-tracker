"""The continuous edge gate: an object whose live blob reaches the frame border is flagged `ignored`
(an arm reaches in from off-frame; a real object sits interior). Extends the rim gate from a birth-only
gate to every frame, so a hand that mints mid-table and then extends to the edge is dropped."""
from types import SimpleNamespace

import numpy as np

from lib.observe import Observe
from lib.scene import Scene


def _ctx(gray, fi):
    return SimpleNamespace(gray=gray, bgr=None, frame_idx=fi, ts=float(fi))


def _frame(cx, cy, w=40, h=40, W=200, H=200):
    """A single bright blob centred at (cx,cy) on a dark frame."""
    g = np.zeros((H, W), np.uint8)
    x0, y0 = max(0, cx - w // 2), max(0, cy - h // 2)
    g[y0:y0 + h, x0:x0 + w] = 255
    return g


def _tick(obs, scene, gray, fi):
    scene.begin_tick(fi, ts=float(fi))
    with scene.writing("observe"):
        obs(_ctx(gray, fi), scene)
    snap = {o.id: o for o in scene.snapshot().objects}
    scene.end_tick()
    return snap


def _obs(key="edge_ignore_margin"):
    return Observe(SimpleNamespace(get=lambda k, d=None: {key: 0.06, "min_area": 100,
                                                          "fg_threshold": 40, "reassoc_dist": 80}.get(k, d)))



def test_object_extending_to_edge_is_ignored():
    obs, scene = _obs(), Scene()
    _tick(obs, scene, _frame(100, 60), 0)               # interior birth → mints (not at edge)
    snap = _tick(obs, scene, _frame(100, 60), 1)        # interior + stable → NOT ignored
    oid = next(iter(snap))
    assert snap[oid].ignored is False
    _tick(obs, scene, _frame(100, 15), 2)               # blob moves to the top border (in_motion → unmatched)
    snap = _tick(obs, scene, _frame(100, 15), 3)        # STABLE at the border → re-associates (a paused arm)
    assert snap[oid].ignored is True                    # edge-connected → flagged ignored (arm-like)


def test_interior_object_stays_visible():
    obs, scene = _obs(), Scene()
    _tick(obs, scene, _frame(100, 100), 0)
    snap = _tick(obs, scene, _frame(100, 100), 1)       # squarely interior across frames
    oid = next(iter(snap))
    assert snap[oid].ignored is False                   # never edge-connected → shown
