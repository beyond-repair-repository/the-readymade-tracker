"""The core vision pipeline — the ordered stages, run per frame.

``Pipeline.run(fc) -> SceneSnapshot`` is the whole story: open a Scene tick on
this frame, run each stage in order (sharing the one Scene), publish the
read-only snapshot. No dispatcher, no events — a sequence of calls that reads
top-to-bottom. (Control commands never pass through here — the ControlRouter
drains them on the main-thread supervisor.)

The stage order is ``observe → classify → identity → pose`` (+ the optional
``grouper → features → clusters``: the grouper runs BEFORE features so compounds
get real features computed on their blended contour, and before clusters so
clusters form over the DISCRETE voices). A stage is a callable ``(ctx, scene) -> None`` with a
``.name`` (write provenance); the per-frame data it needs from earlier stages travels
in the shared ``FrameContext``.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, Sequence

import numpy as np

from lib.events import FrameCaptured
from lib.logger import NULL_LOGGER
from lib.scene import Scene, SceneSnapshot


@dataclass
class FrameContext:
    """Per-frame data the stages read/write, alongside the Scene. The pipeline
    builds it from ``FrameCaptured``; each stage fills its part, so a later stage
    reads an earlier one's output directly — no event. Pure pipeline-thread state.

    observe fills ``obj_fg``."""

    frame_idx: int
    ts: float
    bgr: np.ndarray
    gray: np.ndarray
    obj_fg: np.ndarray | None = None       # observe: the object foreground mask (gray > fg_threshold)


# A stage: ``(ctx, scene) -> None``, with a ``.name`` for write provenance.
Stage = Callable[["FrameContext", Scene], None]


class Pipeline:
    """The ordered vision pipeline over one Scene."""

    def __init__(self, scene: Scene, stages: Sequence[Stage] = (), *,
                 logger: Any = None, profiler: Any = None, audit: Any = None, mot: Any = None) -> None:
        self._scene = scene
        self._stages = tuple(stages)
        self._log = logger or NULL_LOGGER
        self._profiler = profiler
        self._audit = audit
        self._mot = mot                              # --mot-export: MOTChallenge track dump, fed the snapshot

    def run(self, fc: FrameCaptured) -> SceneSnapshot:
        """One tick: open the Scene tick on this frame, run the stages in order
        (each attributed via ``scene.writing`` for provenance), publish the snapshot.
        ``--profile`` / ``--debug`` share the instrumented path: per-stage timing to the
        profiler, lifecycle events to the audit."""
        prof, audit, mot = self._profiler, self._audit, self._mot
        ctx = FrameContext(frame_idx=fc.frame_idx, ts=fc.ts, bgr=fc.bgr, gray=fc.gray)
        if prof is None and audit is None:           # fast path — no per-stage instrumentation (mot is cheap)
            self._scene.begin_tick(fc.frame_idx, ts=fc.ts, frame=fc.bgr, input_full=fc.input_full)
            try:
                for stage in self._stages:
                    with self._scene.writing(getattr(stage, "name", "stage")):
                        stage(ctx, self._scene)
                snap = self._scene.snapshot()
                if mot is not None:
                    mot.write(snap)
                return snap
            finally:                                 # a stage (or the mot tap) raising is FATAL — the
                self._scene.end_tick()               # loop captures it and the orchestrator re-raises —
                #   but the tick still closes, so the store isn't left mid-tick with retired objects
                #   un-evicted for whatever reads the Scene during shutdown.
        # instrumented: ONE "pipeline" measurement brackets the whole pipeline-thread tick
        # (begin → stages → snapshot); each stage is timed once and fed to both consumers.
        pipe_t0 = perf_counter()
        self._scene.begin_tick(fc.frame_idx, ts=fc.ts, frame=fc.bgr, input_full=fc.input_full)
        if audit is not None:
            audit.begin_tick(fc.frame_idx, fc.ts)
        try:
            for stage in self._stages:
                name = getattr(stage, "name", "stage")
                t0 = perf_counter()
                with self._scene.writing(name):
                    stage(ctx, self._scene)
                dt = perf_counter() - t0
                if prof is not None:
                    prof.record(name, dt)
            t0 = perf_counter()
            snap = self._scene.snapshot()
            if prof is not None:
                prof.record("snapshot", perf_counter() - t0)
                prof.record("pipeline", perf_counter() - pipe_t0)
            if mot is not None:
                mot.write(snap)
            return snap
        finally:                                     # symmetry on the fatal path (see the fast path):
            self._scene.end_tick()                   #   the tick closes AND the audit's tick record is
            if audit is not None:                    #   terminated, so a post-mortem debug log doesn't
                audit.end_tick()                     #   end on an unclosed tick.
