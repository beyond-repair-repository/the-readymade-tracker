"""``--profile`` — a tiny per-span timing profiler.

The pipeline is a plain ordered call chain,
so profiling is just a context manager the pieces wrap themselves in:

    with profiler.span("observe"): ...

`Pipeline.run` wraps each stage (`<stage>`) + the whole tick (`pipeline`); `_serve_gui`
wraps `gui.render`. On exit the orchestrator prints `summary()` — a table sorted by
total time, so "where do the milliseconds go" is one line, not a manual benchmark.
Disabled (the default) → `span` is a no-op with ~zero overhead. Thread-safe (the pipeline
thread and the main/gui thread both record)."""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Iterator

# BENCH: reversible — per-SAMPLE stream for benchmarks/ (median/p95 need samples, not aggregates).
# OFF by default; enabled only when RMT_BENCH_SAMPLES names a file. Zero overhead when off
# (one `is not None` per record). Delete this block + the two lines in record() to revert.
import os as _os
_p = _os.environ.get("RMT_BENCH_SAMPLES")
_BENCH_SAMPLES = open(_p, "a", buffering=1) if _p else None


class Profiler:
    """Accumulates count / total / min / max wall-time per labelled span."""

    _GROUPS = ("input", "pipeline", "output")    # thread/category order in the summary

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._stats: dict[str, list] = {}        # label -> [count, total, min, max]
        self._group: dict[str, str] = {}         # label -> group (input | pipeline | output)
        self._lock = threading.Lock()

    def record(self, label: str, dt: float, group: str = "pipeline") -> None:
        """Accumulate one already-measured span (the guts of ``span``, exposed so a
        caller that times a region once can feed it here without a second clock)."""
        if not self.enabled:
            return
        if _BENCH_SAMPLES is not None:               # BENCH: reversible — stream every sample
            _BENCH_SAMPLES.write(f"{label},{group},{dt:.9f}\n")
        with self._lock:
            s = self._stats.get(label)
            if s is None:
                self._stats[label] = [1, dt, dt, dt]
                self._group[label] = group
            else:
                s[0] += 1
                s[1] += dt
                s[2] = dt if dt < s[2] else s[2]
                s[3] = dt if dt > s[3] else s[3]

    @contextmanager
    def span(self, label: str, group: str = "pipeline") -> Iterator[None]:
        if not self.enabled:
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.record(label, time.perf_counter() - t0, group)

    def summary(self, overall_fps: float | None = None) -> str:
        """Flat per-stage/op cost, grouped by thread — INPUT (capture), PIPELINE (core stages),
        OUTPUT (gui + the net outputs tuio2 / mjpeg, each on its OWN thread).

        ``ms/frame`` = wall time ÷ the pipeline's processed-frame count. Amortizing every row
        over the *same* denominator is what makes it **add up**: the stages + ``(untimed gap)``
        sum to the PIPELINE ``TOTAL`` even though each span *ran* a different number of times
        (a stage skips static frames). ``min``/``avg``/``max`` are per-CALL span durations (avg
        = total ÷ calls) — the spread of a single invocation; ``max`` marks the worst spike.
        The three threads run concurrently, so the per-group TOTALs do NOT sum to the frame time;
        achieved fps is gated by the slowest thread + cross-thread GIL contention. (INPUT
        ``grab`` folds in real-time pacing waits.)"""
        with self._lock:
            stats, grp = dict(self._stats), dict(self._group)
        if not stats:
            return "[profile] no spans recorded"
        ms = lambda v: v * 1000.0
        # the denominator: frames the pipeline PROCESSED (its tick count). One denominator
        # for every row ⇒ the ms/frame column adds up.
        nf = (stats["pipeline"][0] if "pipeline" in stats
              else max((s[0] for s in stats.values()), default=1)) or 1
        perf = lambda tot: ms(tot) / nf                  # ms per processed frame
        title = {"input": "INPUT - capture", "pipeline": "PIPELINE - core stages",
                 "output": "OUTPUT - gui + net (concurrent threads; TOTAL is not wall-clock)"}

        def row(label: str, *, disp: str | None = None, indent: str = "    ") -> str:
            n, tot, mn, mx = stats[label]
            w = 18 - (len(indent) - 4)
            avg = tot / n if n else 0.0
            return (f"{indent}{disp or label:{w}s} {perf(tot):9.2f} "
                    f"{ms(mn):9.2f} {ms(avg):9.2f} {ms(mx):9.2f}")

        def synth(label: str, tot: float, *, indent: str = "    ") -> str:
            """A computed row (TOTAL / untimed gap) - amortized only, no per-call spread."""
            w = 18 - (len(indent) - 4)
            return f"{indent}{label:{w}s} {perf(tot):9.2f} {'':>9s} {'':>9s} {'':>9s}"

        out = [f"[profile] ms/frame amortized over {nf} frames (adds up); min/avg/max = per-call span:",
               f"    {'span':18s} {'ms/frame':>9s} {'min':>9s} {'avg':>9s} {'max':>9s}"]
        for g in self._GROUPS:
            labels = [L for L in stats if grp.get(L, "pipeline") == g]
            if not labels:
                continue
            out.append(f"  {title[g]}")
            if g != "pipeline":                       # input/output: flat, then TOTAL = Σ
                for label in sorted(labels, key=lambda L: -stats[L][1]):
                    out.append(row(label))
                out.append(synth("TOTAL", sum(stats[L][1] for L in labels)))
                continue
            # PIPELINE: flat per-stage rows by cost, then the untimed gap + TOTAL.
            stage_total = 0.0
            for label in sorted((L for L in labels if L != "pipeline"), key=lambda L: -stats[L][1]):
                out.append(row(label))
                stage_total += stats[label][1]
            p = stats.get("pipeline")
            if p is not None:
                out.append(synth("(untimed gap)", p[1] - stage_total))
                out.append(row("pipeline", disp="TOTAL"))
            else:
                out.append(synth("TOTAL", stage_total))
        p = stats.get("pipeline")
        if p and p[1] > 0:
            # fps min/avg/max from the tick-time extremes: the slowest tick bounds the WORST-case
            # rate (fps min = 1/max_tick), the fastest bounds the best (fps max = 1/min_tick).
            avg = p[0] / p[1]
            lo = (1.0 / p[3]) if p[3] > 0 else avg
            hi = (1.0 / p[2]) if p[2] > 0 else avg
            line = f"  -> pipeline fps min/avg/max {lo:5.0f} /{avg:5.0f} /{hi:5.0f}  (capacity)"
            if overall_fps:
                line += (f"   -   overall {overall_fps:5.0f} fps (achieved)"
                         "   -   gap = input pacing + cross-thread GIL contention")
            out.append(line)
        return "\n".join(out)
