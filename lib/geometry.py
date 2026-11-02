"""Pure-functional geometry helpers — no dependency on Scene or any stage (a shared ``lib/`` util).
OpenCV ``(N,1,2)`` contour convention throughout; the placement output is **float32** (sub-pixel —
see ``place_footprint``), and callers that must DRAW with cv2 cast to int32 at the draw site.

  * ``place_footprint`` — put the captured REST outline at the object's live pose (the ONE
    rotate-about-centroid-then-translate every outline consumer shares: gui, tuio2, oscout, webgui).
  * ``simplify_contour`` — approxPolyDP at the global wire/UI contour-quality ε.

Settle-time rotation is measured by EFD phase correlation (``lib.efd.rotation_by_correlation``), used
directly by the pose stage. (The former masked-overlap brute-force fit is archived in the
development repository.)
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np


def place_footprint(footprint_contour: "np.ndarray", footprint_centroid: Any,
                    pose: Any) -> np.ndarray:
    """Place the captured REST outline at the object's LIVE pose — the SAME transform the renderer uses:
    rotate by θ about ``footprint_centroid``, then translate so the rest centroid lands on the pose
    ``(x, y)``. ``pose`` = ``(x, y, θ-radians)``. Returns an ``(N,1,2) float32`` contour in frame px —
    SUB-PIXEL: int-snapping here re-quantized every rotation and gave upscaled outlines a staircase
    look. Consumers that must draw with cv2 cast to int32 at the draw site; the wire (oscout /
    tuio2 / webgui JSON) ships the floats normalized.

    The ONE placement every consumer that shows the object's outline shares — the desktop gui, tuio2,
    oscout, the webgui overlay — so the shape TRACKS the pose (not frozen at the rest position where it
    was captured). Without this the outline sits where the object last RESTED and appears to lag behind."""
    pts = np.asarray(footprint_contour, np.float32).reshape(-1, 2)
    fx, fy = float(footprint_centroid[0]), float(footprint_centroid[1])
    x, y, th = float(pose[0]), float(pose[1]), float(pose[2])
    c, s = math.cos(th), math.sin(th)
    dx, dy = pts[:, 0] - fx, pts[:, 1] - fy
    xn = c * dx - s * dy + x
    yn = s * dx + c * dy + y
    return np.stack([xn, yn], axis=1).astype(np.float32).reshape(-1, 1, 2)


def simplify_contour(contour, eps: float):
    """approxPolyDP under the global wire/UI contour-quality ε ([observe] contour_epsilon) —
    the outline every output ships. eps <= 0 or a degenerate contour → returned as-is."""
    if contour is None or eps <= 0 or len(contour) < 3:
        return contour
    return cv2.approxPolyDP(np.asarray(contour), float(eps), True)
