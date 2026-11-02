"""perf — machine-shaped settings that are not about what the tracker does, only how fast.

Currently one knob: how many threads OpenCV may split a single operation across.

WHY THIS EXISTS
  A frame is a long chain of SMALL operations — countNonZero, bitwise_and/not/xor, boundingRect,
  threshold — and OpenCV parallelises each of them across every core it can find. Splitting work
  that small is not free: the threads must be woken, handed a slice, and waited for, and that
  hand-off costs more than the slice. The cost GROWS with the core count, so a bigger machine is
  slower at this, at the same clock, purely from coordinating threads that had nothing to do.

  Measured (benchmarks/bench_threads.py, median ms per frame on hover1):
    Apple M1, 8 cores      1 thread 1.40 · 8 threads (default) 1.65   default is 1.18x the best
    Intel i9, 32 threads   2 threads 2.05 · 32 (default) 3.01         default is 1.47x the best
  On the i9 it degrades monotonically — 4: 2.17, 8: 2.43, 16: 2.79, 32: 3.01. Nothing in this
  pipeline scales with core COUNT; one frame is a dependent chain on one thread, and the other
  cores only ever help INSIDE a single call, which is exactly where they are hurting.

  So this is not a micro-optimisation, it is a correction: the default is wrong for this
  workload on every machine tested, and wrongest on the biggest one.

  It changes SPEED ONLY. cv2's parallel_for_ splits a loop over disjoint output regions, so the
  result of an operation does not depend on how many threads computed it.

DEFAULT 0 = LEAVE OPENCV ALONE, deliberately: shipping this knob changes nothing until someone
sets it, so an existing measurement stays comparable to the run that produced it. Find the right
number for a machine with bench_threads.py and put it in that machine's conf.
"""
from __future__ import annotations

from typing import Any

from lib.logger import NULL_LOGGER
from lib.params import ParamSpec, bind_params


def _apply_threads(self: "Perf", value: int) -> None:
    """Set OpenCV's thread limit now — cold start and a live /rmt/set take the same path."""
    self._cv2_threads = int(value)
    if self._cv2_threads <= 0:                      # 0 = don't touch it; whatever OpenCV chose stands
        return
    try:
        import cv2
        cv2.setNumThreads(self._cv2_threads)
    except Exception as e:                          # noqa: BLE001 — a perf knob must never stop the tracker
        self._log.warning(f"[perf] cv2.setNumThreads({self._cv2_threads}) failed: {e}")


class Perf:
    """Not a pipeline stage — a live param target only. Owns process-wide performance settings."""

    name = "perf"

    PARAMS = (
        ParamSpec("cv2_threads", int, 0, lo=0, hi=256, attr="_cv2_threads", apply=_apply_threads,
                  doc="How many threads OpenCV may split ONE operation across. 0 = leave it alone "
                      "(it will use every core, which is slower here). This pipeline's operations "
                      "are small, so splitting them costs more than it saves, and the cost grows "
                      "with core count: measured 1.18x on an 8-core M1 and 1.47x on a 32-thread "
                      "i9. Find this machine's best value with benchmarks/bench_threads.py - it "
                      "was 1 and 2 on those two. Affects speed only, never the result"),
    )

    def __init__(self, config: Any = None, *, logger: Any = None) -> None:
        self._log = logger or NULL_LOGGER
        self._cv2_threads = 0
        bind_params(self, config)                   # _cv2_threads ← PARAMS, and applies it
        if self._cv2_threads > 0:
            self._log.info(f"[perf] OpenCV limited to {self._cv2_threads} thread(s) per operation")
