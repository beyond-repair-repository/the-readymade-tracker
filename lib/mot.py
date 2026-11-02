"""``--mot-export`` — the tracker's tracks per frame in **MOT16** format, the canonical dialect
py-motmetrics AND TrackEval (HOTA) both read, and the SAME format the ground-truth editor (in the
development repository) writes — so a tracker trace and a hand-corrected GT are byte-comparable.

One comma-separated line per CONFIRMED, non-ignored object per processed frame (9 columns):

    frame, id, bb_left, bb_top, bb_width, bb_height, not_ignored, class_id, visibility

- ``frame`` is 1-indexed (MOT convention; the tracker's frame_idx is 0-based).
- ``id`` is the tracker's persistent TRACK id — the churning identity we evaluate, NOT the recycled
  display ``slot`` (several tracks share a slot over time, which would corrupt the metrics).
- ``not_ignored`` = 1 (every reported row is scored) — py-motmetrics reads this column as Confidence.
- ``class_id`` = 10 (``other``): the tracker doesn't classify — the GROUP lives in the GT,
  which is what a scorer keys same-group swaps on. Map: 1 disc 2 tile 3 ring 4 mesh 5 nugget 6 bar
  7 stone 8 wire 9 hand 10 other 11 gear 12 compound (gear/compound appended so 1..10 stay put;
  ``compound`` = two objects touching and held as one merged blob).
- ``visibility`` = 1.0 (reported = visible).

Only the tracker's actual OUTPUT is written (confirmed + not ignored) — the same set OSC / TUIO2 / the
overlay emit. Dependency-free (plain CSV); run the metrics tool in a SEPARATE env (scipy/pandas) to keep this
repo numpy+OpenCV-only. Process EVERY frame — run with ``--pull`` (or ``--fps 0``) so none is skipped.
"""
from __future__ import annotations

from typing import Any

from lib.fsutil import own_to_dir_owner
from lib.logger import NULL_LOGGER


class MotWriter:
    """Appends MOTChallenge lines for each tick's confirmed tracks. Reliable per-frame: fed from
    ``pipeline.run`` (the snapshot, before the latest-wins bus that could drop frames)."""

    def __init__(self, path: str, *, logger: Any = None) -> None:
        self._path = path
        self._log = logger or NULL_LOGGER
        self._f = open(path, "w", encoding="utf-8", newline="\n")
        own_to_dir_owner(path)               # a sudo run must not root-own the export
        self.lines = 0
        self.frames = 0

    @property
    def path(self) -> str:
        return self._path

    def write(self, snap: Any) -> None:
        frame = int(getattr(snap, "frame_idx", 0)) + 1        # MOTChallenge is 1-indexed
        self.frames += 1
        out = []
        for o in snap.objects:
            if o.ignored or not getattr(o, "confirmed", True) or o.bbox is None:
                continue                                       # the tracker's reported OUTPUT only
            x, y, w, h = o.bbox
            out.append(f"{frame},{o.id},{x},{y},{w},{h},1,10,1")   # not_ignored=1, class=10 (other), vis=1
        if out:
            self._f.write("\n".join(out) + "\n")
            self.lines += len(out)

    def close(self) -> None:
        try:
            self._f.close()
            self._log.info(f"[mot] wrote {self.lines} rows over {self.frames} frames -> {self._path}")
        except Exception:                                      # noqa: BLE001 — a close hiccup must not crash teardown
            pass
