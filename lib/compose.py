"""Pure cv2 view composition for the video outputs — ``input`` | ``debug`` | ``processed``.

Three single-panel views on a ``SceneSnapshot``:
  * ``input``     — the FULL pre-crop frame (crop-calibration view).
  * ``debug``     — the cropped input frame MASKED to-zero by observe's foreground (the original pixel
    where ``gray > fg_threshold``, else 0). observe already publishes that mask as ``snap.obj_fg``, so
    this reuses the exact threshold with no recompute. The webgui draws object/cluster/id overlays on
    top, client-side.
  * ``processed`` — the preprocess OUTPUT: flip + crop + lens + flatfield are already baked into
    ``snap.frame``, and the circular mask (normally applied to GRAY only) is baked into the COLOR
    here when the caller passes ``mask_circle``. A recording of this view replays as a test video
    with ``--no-preprocess`` and behaves exactly like the live run.

``render_view(snap, view)`` is the entry point; every surface (headless MJPEG, webgui underlay) renders
the same thing.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

VIEWS = ("input", "debug", "processed")


def to_bgr(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img


def label(panel: np.ndarray, name: str) -> np.ndarray:
    cv2.putText(panel, name, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return panel


def mask_circle_px(pcfg: Any) -> "tuple | None":
    """The circular-mask geometry in PROCESSED-frame px — ``(cx, cy, r)`` — or None when the mask is
    off. The mask centre is declared in RAW-frame px, so an active crop offsets it (the same shift
    the webgui overlay applies)."""
    if pcfg is None or not getattr(pcfg, "mask_enabled", False) or getattr(pcfg, "mask_radius", 0) <= 0:
        return None
    ox = pcfg.crop_top_x if pcfg.crop_enabled else 0
    oy = pcfg.crop_top_y if pcfg.crop_enabled else 0
    return (float(pcfg.mask_cx - ox), float(pcfg.mask_cy - oy), float(pcfg.mask_radius))


def render_view(snap: Any, view: str = "debug", with_label: bool = True,
                mask_circle: "tuple | None" = None) -> "np.ndarray | None":
    """The BGR canvas for ``view`` (one of ``VIEWS``). ``None`` when there's nothing to show.
    ``with_label`` burns the panel name in (the gui does; the MJPEG stream doesn't). ``mask_circle``
    ((cx, cy, r) px, from ``mask_circle_px``) is honored by the ``processed`` view only."""
    if snap is None or getattr(snap, "frame", None) is None:
        return None
    frame = np.asarray(snap.frame)
    if view == "input":                                   # the FULL pre-crop frame (crop calibration view)
        src = getattr(snap, "input_full", None)
        img = to_bgr((src if src is not None else frame).copy())
    elif view == "processed":                             # the preprocess OUTPUT (replayable with --no-preprocess)
        img = to_bgr(frame.copy())
        if mask_circle is not None:                       # bake the circular mask into COLOR (live: gray only)
            cx, cy, r = mask_circle
            m = np.zeros(img.shape[:2], np.uint8)
            cv2.circle(m, (int(round(cx)), int(round(cy))), int(round(r)), 255, -1)
            img = cv2.bitwise_and(img, img, mask=m)
    else:                                                 # debug (default) — input MASKED by observe's fg cut
        img = to_bgr(frame.copy())
        fg = getattr(snap, "obj_fg", None)                # gray > fg_threshold, published by observe
        if fg is not None and np.asarray(fg).shape[:2] == img.shape[:2]:
            img = cv2.bitwise_and(img, img, mask=np.asarray(fg))   # keep the original px where fg>0, else 0
    return label(img, view) if with_label else img
