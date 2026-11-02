"""The tiny message set.

There is **no event bus** — the core stages call one another directly, in order.
Only the I/O edges carry messages: ``FrameCaptured`` (INPUT backend → core) and
``SceneSnapshot`` (core → OUTPUTs; defined in ``lib.scene`` with the projection
it carries). Outputs derive everything they emit from the snapshot — there are
no per-transition event objects.

``readonly`` exposes a numpy array as a 0-copy read-only view so a stray
downstream write fails loudly instead of corrupting state the pipeline owns.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def readonly(arr: np.ndarray | None) -> np.ndarray | None:
    """0-copy **read-only view** of ``arr`` (``writeable=False``); ``None`` passes
    through. The pipeline must *replace* arrays on update (never mutate in place)
    to keep a published view valid."""
    if arr is None:
        return None
    view = np.asarray(arr).view()
    view.flags.writeable = False
    return view


@dataclass(frozen=True, slots=True)
class Event:
    """Base message. ``frame_idx`` is logs/ordering only — motion ``dt`` comes
    from the capture-time ``ts``, never the frame index."""

    frame_idx: int


@dataclass(frozen=True, slots=True, eq=False)
class FrameCaptured(Event):
    """An INPUT backend produced a preprocessed frame. ``ts`` is the capture
    stamp (video PTS / camera hardware), not processing wall-clock."""

    ts: float
    bgr: np.ndarray                             # the PROCESSED frame (flip → crop → undistort) — tracking + views
    gray: np.ndarray
    input_full: np.ndarray | None = None        # the pre-CROP (flipped, full) frame — the 'input' view / crop calib
