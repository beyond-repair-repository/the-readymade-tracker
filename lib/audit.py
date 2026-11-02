"""The audit sink — lifecycle + profile logs, written off the pipeline thread.

There is no dispatcher/event bus, so there's a single, honest tap point: the **Scene's
write-API** (the single writer). The Scene calls the ``Audit`` hooks
(``on_mint`` / ``on_state`` / ``on_rewrite`` / ``on_field``) as it mutates; the
``Pipeline`` frames each tick (``begin_tick`` / ``end_tick``). At ``end_tick`` it writes:

  * ``--debug`` → lifecycle events (mint / state-change / rewrite_id / capture) — low-volume,
    the id-churn diagnostic — to ``logs/debug_<dt>_<src>.log``;
  * ``--profile`` → every ``profile_every`` frames, the profiler's ms/frame table to
    ``logs/profile_<dt>_<src>.log`` (the on-exit console summary is unchanged).

Every logfile is written by a **daemon writer thread** (``BufferedSink``) — the pipeline
thread only queues a formatted string, never touches ``fflush``, so logging can't poison
the timings. Disabled (no flags) ⇒ every hook is a cheap
no-op; the Scene/Pipeline hold ``audit=None``.
"""

from __future__ import annotations

import math
import os
import queue
import threading
import time

from lib.fsutil import own_to_dir_owner
from typing import Any

import numpy as np


def log_path(kind: str, source: str, logdir: str = "logs", *, stamp: str | None = None) -> str:
    """``logs/<kind>_<YYYYMMDD-HHMMSS>_<source>.log``. ``source`` is the
    input location's basename, sans extension + sanitised."""
    os.makedirs(logdir, exist_ok=True)
    own_to_dir_owner(logdir)                 # a sudo run must not root-own logs/ for the next plain run
    ts = stamp or time.strftime("%Y%m%d-%H%M%S")
    src = os.path.splitext(os.path.basename(source or "src"))[0]
    src = "".join(c if c.isalnum() or c in "-." else "_" for c in src) or "src"
    return os.path.join(logdir, f"{kind}_{ts}_{src}.log")


class BufferedSink:
    """An append logfile with a lone daemon writer thread — producers ``write`` a string
    (thread-safe ``queue.put``), the writer owns the file handle. Idempotent ``close``."""

    _SENTINEL = object()

    def __init__(self, path: str) -> None:
        self.path = path
        self._q: queue.Queue = queue.Queue()
        # line-buffered so completed lines survive an abrupt exit. encoding=utf-8 is MANDATORY: the
        # profile blocks use box-drawing chars (─) and state lines use → — on Windows the
        # default cp1252 can't encode those, and the writer's broad except (UnicodeEncodeError IS a
        # ValueError) silently swallows the write → 0-byte logs. utf-8 makes every platform write them.
        self._f: Any = open(path, "w", buffering=1, encoding="utf-8")
        own_to_dir_owner(path)
        self._t = threading.Thread(target=self._loop, daemon=True, name="AuditWriter")
        self._t.start()

    def write(self, text: str) -> None:
        self._q.put(text if text.endswith("\n") else text + "\n")

    def _loop(self) -> None:
        while True:
            item = self._q.get()
            if item is self._SENTINEL:
                break
            try:
                self._f.write(item)
                if self._q.empty():                  # flush when caught up — line-buffering ISN'T
                    self._f.flush()                  #   reliable on Windows (empty log files), so
            except (ValueError, AttributeError):     #   force it; batches under load, prompt when idle
                pass

    def close(self) -> None:
        if self._f is None:
            return
        self._q.put(self._SENTINEL)
        self._t.join(timeout=5.0)
        if not self._f.closed:
            self._f.close()
        self._f = None


def _fmt(v: Any) -> str:
    """Compact value rendering for the log lines (round float tuples, summarise arrays)."""
    if v is None:
        return "None"
    if isinstance(v, np.ndarray):
        return f"<array {tuple(v.shape)}>"
    if isinstance(v, tuple):
        return "(" + ",".join(_fmt(x) for x in v) + ")"
    if isinstance(v, (float, np.floating)):
        return f"{v:.0f}" if abs(v) >= 100 else f"{v:.1f}"
    return str(v)


class Audit:
    """Assembles + writes the tick story. Fed by the Scene (write hooks) and the Pipeline
    (tick framing). All calls happen on the pipeline thread — no lock needed (the Scene is
    pipeline-thread state); only the disk write crosses to the writer thread."""

    def __init__(self, *, debug: BufferedSink | None = None,
                 profile: BufferedSink | None = None, profiler: Any = None,
                 profile_every: int = 100) -> None:
        self._debug = debug
        self._profile = profile
        self._profiler = profiler
        self._every = max(1, int(profile_every))
        self.enabled = bool(debug or profile)
        self._frame = -1
        self._ts: float | None = None
        self._events: list[str] = []               # this tick's lifecycle lines

    # -- pipeline framing --------------------------------------------------- #

    def begin_tick(self, frame_idx: int, ts: float | None) -> None:
        self._frame = frame_idx
        self._ts = ts
        self._events.clear()

    def end_tick(self) -> None:
        if self._debug is not None and self._events:
            pre = f"f{self._frame}"
            self._debug.write("\n".join(f"{pre} {t}" for t in self._events))
        if self._profile is not None and self._profiler is not None and self._frame % self._every == 0:
            self._profile.write(f"-- profile @ frame {self._frame} "
                                f"{'-' * 40}\n{self._profiler.summary()}")

    # -- Scene write hooks (called from the single writer) ------------------- #

    def on_mint(self, oid: int, det: Any, writer: str) -> None:
        self._events.append(f"[{writer}] mint #{oid} c={_fmt(det.centroid)} area={_fmt(det.area)}")

    def on_state(self, oid: int, old: Any, new: Any, writer: str) -> None:
        self._events.append(f"[{writer}] #{oid} {old.name}->{new.name}")

    def on_rewrite(self, oid: int, new_id: int, writer: str) -> None:
        self._events.append(f"[{writer}] rewrite #{oid}->#{new_id}")

    def on_field(self, oid: int, field: str, value: Any, writer: str) -> None:
        if field == "footprint" and value is not None:      # a footprint bake = a texture CAPTURE moment
            self._events.append(f"[{writer}] capture #{oid}")
            #                     (a SECOND `capture #id` for one id = a re-capture/overwrite — the tell)
        elif field == "theta_cumulative":                   # the settle's measured ABSOLUTE angle (only
            deg = math.degrees(float(value))                #   CHANGES are written — a held θ is silent)
            self._events.append(f"[{writer}] #{oid} theta {deg:+.1f}deg")
        elif field == "pose_conf" and value is not None and float(value) < 1.0:
            self._events.append(f"[{writer}] #{oid} settle-fit score {float(value):.3f}")

    # -- teardown ----------------------------------------------------------- #

    def close(self) -> None:
        """Write a final profile summary, then drain + close every sink."""
        if self._profile is not None and self._profiler is not None:
            self._profile.write(f"-- profile @ exit (frame {self._frame}) "
                                f"{'-' * 34}\n{self._profiler.summary()}")
        for sink in (self._debug, self._profile):
            if sink is not None:
                sink.close()
