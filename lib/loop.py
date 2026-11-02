"""Thread orchestration + the pipeline frame-loop.

The core owns **every** thread: the **input thread** (the
``InputRunner``: ``backend.grab → preprocess → FrameCaptured``), the **one
pipeline thread** (the direct stage sequence, ``lib.pipeline``), and the
main-thread gui pump (the orchestrator's ``serve()``). Plugins supply work
functions — a backend's ``grab``, an output's ``paint`` — and never spawn a
thread.

Concurrency lives only at narrow, lock/queue-guarded passes:
  * ``LatestFrameSlot`` — input thread → pipeline (one-deep, **latest-wins**: a
    slow tick drops stale frames, never builds a backlog; motion ``dt`` comes
    from the capture ``ts``, so a drop is harmless to correctness).
  * ``CommandInbox`` — control (gui keys / OSC-in) → pipeline, drained at
    tick-start so control enters the pass deterministically.
  * ``SnapshotBus`` — pipeline → outputs (the one channel *out*; the snapshot is
    an immutable read-only projection, so handing it across threads is 0-copy).

``PipelineLoop`` ties them together: one ``step()`` == take the newest frame →
``pipeline.run`` → publish the snapshot (control commands are drained by the
ControlRouter on the main-thread supervisor, never here). ``step()`` is factored
out so the whole loop is unit-testable with a synthetic frame + a no-op pipeline —
no real threads, fully deterministic.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from contextlib import nullcontext
from time import perf_counter
from typing import Any, Callable

from lib.events import FrameCaptured
from lib.logger import NULL_LOGGER
from lib.scene import SceneSnapshot


# --------------------------------------------------------------------------- #
# Latest-frame slot — input thread → pipeline (atomic, latest-wins)
# --------------------------------------------------------------------------- #

class LatestFrameSlot:
    """One-deep frame pass: input thread → pipeline.

    Default (**latest-wins**): ``put`` overwrites — a frame the consumer never took is
    *dropped*, so the pipeline always works on the freshest frame and a slow tick never
    queues a backlog. ``get`` blocks until a frame newer than the last returned is
    available (or ``timeout``)."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._item: Any = None
        self._seq = 0       # bumped on every put
        self._taken = 0     # seq of the last item handed out by get()
        self._dropped = 0   # puts that overwrote an untaken fresh frame
        self._closed = False

    def put(self, item: Any) -> None:
        with self._cond:
            if self._seq > self._taken:
                self._dropped += 1     # latest-wins: the previous fresh frame is dropped
            self._item = item
            self._seq += 1
            self._cond.notify_all()

    def get(self, timeout: float | None = None) -> Any:
        with self._cond:
            if self._seq == self._taken:
                self._cond.wait_for(lambda: self._seq != self._taken or self._closed, timeout)
            if self._seq == self._taken:
                return None            # timed out / closed with nothing new
            self._taken = self._seq    # consume freshness up to the newest
            self._cond.notify_all()
            return self._item

    def close(self) -> None:
        """Unblock any waiting ``get`` on shutdown so teardown can't deadlock."""
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    @property
    def dropped(self) -> int:
        with self._cond:
            return self._dropped

    @property
    def pending(self) -> bool:
        """True iff a fresh (not-yet-taken) frame is waiting — the orchestrator
        uses ``inputs-done and not pending`` for a graceful EOF shutdown."""
        with self._cond:
            return self._seq > self._taken

    # -- frame-channel protocol (shared with PullSlot) — the InputThread + PipelineLoop talk to a
    #    slot ONLY through these three, so the handoff (push / pull) is swappable. --
    def take(self, timeout: float | None = None) -> Any:
        return self.get(timeout)            # latest-wins: pass the freshest available frame
    def want(self, timeout: float | None = None) -> bool:
        return True                         # PUSH: the input grabs eagerly every loop (the slot drops)
    def give(self, item: Any) -> None:
        self.put(item)


class PullSlot:
    """Demand-driven frame pass: the pipeline REQUESTS a frame (``take``) after publishing the
    previous one; the input waits for that request (``want``), grabs a FRESH frame, and delivers it
    (``give``). Grab-AFTER-process → zero lookahead staleness (vs a blocking no-drop slot, which
    grabs eagerly so the frame goes ~one tick stale). Lossless + ordered for a file; freshest-on-
    demand for a live camera (the backend's ``grab`` reads the latest). One ``Condition``; never drops
    a frame the pipeline asked for. ``pending`` is always False — a delivered frame is consumed
    immediately by the waiting ``take``, so the EOF check reduces to ``input-done``."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._want = 0          # pipeline requests (bumped by take)
        self._served = 0        # input deliveries (caught up to _want on each give)
        self._item: Any = None
        self._closed = False

    def take(self, timeout: float | None = None) -> Any:
        """Pipeline side: request the next frame + block until the input delivers it (or timeout)."""
        with self._cond:
            self._want += 1
            self._cond.notify_all()                      # wake the input's want()
            self._cond.wait_for(lambda: self._served == self._want or self._closed, timeout)
            if self._closed or self._served != self._want:
                return None                              # timed out / closed with nothing delivered
            item, self._item = self._item, None
            return item

    def want(self, timeout: float | None = None) -> bool:
        """Input side: block until the pipeline has an outstanding request (or timeout / closed)."""
        with self._cond:
            self._cond.wait_for(lambda: self._want > self._served or self._closed, timeout)
            return (not self._closed) and (self._want > self._served)

    def give(self, item: Any) -> None:
        """Input side: deliver the freshly-grabbed frame, satisfying the outstanding request."""
        with self._cond:
            self._item = item
            self._served = self._want
            self._cond.notify_all()

    def close(self) -> None:
        """Unblock a waiting ``take`` / ``want`` on shutdown (the threads exit without a pass)."""
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    @property
    def pending(self) -> bool:
        return False            # demand-driven: a delivered frame is consumed at once by take()

    @property
    def dropped(self) -> int:
        return 0                # demand-driven never drops — uniform with LatestFrameSlot


# --------------------------------------------------------------------------- #
# Command inbox — control → pipeline (ordered, drained at tick-start)
# --------------------------------------------------------------------------- #

class CommandInbox:
    """Ordered, thread-safe command pass. Producers ``post`` a
    ``(name, args)`` command; the pipeline ``drain``s the whole queue FIFO at
    tick-start — never mid-run — so control is deterministic + ordered."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._q: list[tuple[str, tuple[Any, ...]]] = []

    def post(self, name: str, args: tuple[Any, ...] = ()) -> None:
        with self._lock:
            self._q.append((name, tuple(args)))

    def drain(self) -> list[tuple[str, tuple[Any, ...]]]:
        with self._lock:
            out, self._q = self._q, []
            return out

    def __len__(self) -> int:
        with self._lock:
            return len(self._q)


# --------------------------------------------------------------------------- #
# Snapshot bus — pipeline → outputs (lock-guarded, latest-wins)
# --------------------------------------------------------------------------- #

class SnapshotBus:
    """Lock-guarded latest-snapshot pass: pipeline → outputs. The single
    channel *out* of the core. The ``SceneSnapshot`` is an immutable projection
    with read-only array views, so handing the same object across threads is
    safe + 0-copy."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._snap: SceneSnapshot | None = None

    def publish(self, snap: SceneSnapshot) -> None:
        with self._cond:
            self._snap = snap
            self._cond.notify_all()               # wake a waiting output (event-driven present)

    def latest(self) -> SceneSnapshot | None:
        with self._cond:
            return self._snap

    def wait_fresh(self, last_idx: Any, timeout: float) -> "SceneSnapshot | None":
        """Block until a snapshot with ``frame_idx != last_idx`` is published, or ``timeout``
        elapses; return the latest either way. A consumer presents the INSTANT a frame lands
        instead of sleep-then-poll (a poll loop adds up to its poll interval of display latency)."""
        deadline = time.perf_counter() + timeout
        with self._cond:
            while True:
                snap = self._snap
                if snap is not None and getattr(snap, "frame_idx", None) != last_idx:
                    return snap
                remaining = deadline - time.perf_counter()
                if remaining <= 0.0:
                    return snap
                self._cond.wait(remaining)


def consume_bus(bus: Any, stop: "threading.Event", on_fresh: Callable[[Any], None], *,
                profiler: Any = None, name: str = "output", max_fps: float = 0.0,
                should_emit: "Callable[[Any], bool] | None" = None) -> None:
    """The ONE snapshot-bus consumer loop, shared by every output thread (tuio2 / oscout / mjpeg):
    block on ``wait_fresh`` (present the instant a frame lands), skip the None/stale returns, optionally
    skip a frame via ``should_emit`` (mjpeg: nobody watching → don't render), throttle to ``max_fps``
    by advancing past + sleeping the rest of the window (never busy-spin), then call ``on_fresh(snap)``
    inside the ``profiler`` span for each fresh frame. Loops until ``stop`` is set."""
    last_idx: Any = object()
    last_emit = 0.0
    min_dt = (1.0 / max_fps) if max_fps > 0 else 0.0
    while not stop.is_set():
        snap = bus.wait_fresh(last_idx, 0.5)
        if snap is None:
            continue
        idx = getattr(snap, "frame_idx", None)
        if idx == last_idx:                          # timed out with no fresh frame
            continue
        if should_emit is not None and not should_emit(snap):
            last_idx = idx                           # consumed (e.g. mjpeg unwatched) — don't re-poll it
            continue
        if min_dt:                                   # throttle: SKIP + BLOCK the rest of the window
            wait = min_dt - (perf_counter() - last_emit)
            if wait > 0:                             # advance last_idx so wait_fresh blocks (no busy-loop)
                last_idx = idx
                stop.wait(min(wait, 0.5))
                continue
        last_idx = idx
        last_emit = perf_counter()
        span = profiler.span(name, group="output") if profiler is not None else nullcontext()
        with span:                                   # --profile: the per-frame build+send cost
            on_fresh(snap)


# --------------------------------------------------------------------------- #
# Input runner — the core's grab → preprocess → FrameCaptured
# --------------------------------------------------------------------------- #

# BENCH: reversible — settle→OSC end-to-end wall stamps for benchmarks/ (paired with the
# oscout emit stamp). OFF by default; enabled only when RMT_BENCH_STAMPS names a file.
import os as _os
_bp = _os.environ.get("RMT_BENCH_STAMPS")
_BENCH_STAMPS = open(_bp, "a", buffering=1) if _bp else None


class InputRunner:
    """The core input-runner: the backend supplies frames (``grab``), the core
    runs ``grab → preprocess → FrameCaptured`` and owns the monotonic frame
    counter. ``produce`` is the work function the ``InputThread`` loops; it
    propagates ``StopIteration`` (source exhausted) and returns ``None`` for a
    transient 'no frame right now' (FPS-gated / paused)."""

    def __init__(self, backend: Any, preprocess: Any, *, logger: Any = None,
                 profiler: Any = None) -> None:
        self._backend = backend
        self._pp = preprocess
        self._log = logger or NULL_LOGGER
        self._count = 0
        self._prof = profiler

    def _span(self, label: str):
        return self._prof.span(label, group="input") if self._prof else nullcontext()

    def produce(self) -> FrameCaptured | None:
        with self._span("grab"):
            res = self._backend.grab()           # (bgr, ts) | None; StopIteration at EOF
        if res is None:
            return None
        bgr, ts = res
        with self._span("preprocess"):           # the core preprocess stage (input thread)
            bgr, gray = self._pp.process(bgr)
        # 0-BASED: the first decoded frame is frame_idx 0 — the SAME convention as a raw cv2
        # decode, so a per-frame index lines up with the source frame.
        fc = FrameCaptured(frame_idx=self._count, ts=ts, bgr=bgr, gray=gray,
                           input_full=self._pp.input_full)   # pre-crop frame → the 'input'/raw view
        if _BENCH_STAMPS is not None:                # BENCH: reversible — capture wall stamp
            _BENCH_STAMPS.write(f"cap,{fc.frame_idx},{perf_counter():.9f}\n")
        self._count += 1
        return fc


# --------------------------------------------------------------------------- #
# Input thread — framework-owned (plugins never spawn threads)
# --------------------------------------------------------------------------- #

class InputThread:
    """Framework-owned input thread: repeatedly call ``producer`` (an
    ``InputRunner.produce``) and drop each frame into the latest-frame ``slot``.
    A ``StopIteration`` means the source is exhausted (video EOF, no loop) → the
    thread stops + sets ``done``."""

    def __init__(self, producer: Callable[[], FrameCaptured | None],
                 slot: LatestFrameSlot, *, name: str = "input", logger: Any = None) -> None:
        self._producer = producer
        self._slot = slot
        self._log = logger or NULL_LOGGER
        self._stop = threading.Event()
        self._wait_timeout = 0.5                     # pull: re-check `stop` this often while idle-waiting
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self.done = False
        self.error: BaseException | None = None     # a producer fatal (NOT StopIteration) — the run ended
        #                                             abnormally, not at EOF; serve() surfaces it

    def start(self) -> "InputThread":
        self._thread.start()
        return self

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                if not self._slot.want(self._wait_timeout):
                    continue                    # pull: no request pending (or closed) → re-check stop
                try:
                    fc = self._producer()
                except StopIteration:
                    break                       # source exhausted
                except Exception as exc:        # noqa: BLE001 — a backend/preprocess fatal would otherwise
                    # end the run as a SILENT pseudo-EOF (traceback only on stderr; the tracker's own log
                    # and exit summary would read like a normal end-of-clip). Log it + mark it instead.
                    self.error = exc
                    self._log.error(f"[input] producer failed: {exc!r} - input thread stopping")
                    break
                if fc is not None:
                    self._slot.give(fc)
        finally:
            self.done = True

    def stop(self) -> None:
        self._stop.set()

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout)

    def is_alive(self) -> bool:
        return self._thread.is_alive()


# --------------------------------------------------------------------------- #
# The pipeline loop — the one deterministic thread
# --------------------------------------------------------------------------- #

class PipelineLoop:
    """The one pipeline thread: newest frame → ``pipeline.run`` → publish the
    snapshot, until stopped. ``pipeline`` is duck-typed (anything with
    ``run(fc) -> SceneSnapshot``), so the loop is unit-testable with a no-op
    pipeline."""

    def __init__(self, pipeline: Any, frames: LatestFrameSlot, inbox: CommandInbox,
                 snapshots: SnapshotBus, *, frame_timeout: float = 0.5,
                 max_frames: int | None = None, logger: Any = None) -> None:
        self._pipeline = pipeline
        self._frames = frames
        self._inbox = inbox
        self._snaps = snapshots
        self._frame_timeout = frame_timeout
        self._max_frames = max_frames or None       # stop after N ticks (--max-frames); None = ∞
        self._log = logger or NULL_LOGGER
        self._stop = threading.Event()
        self.ticks = 0
        self.error: BaseException | None = None    # set if a tick raised (fatal)
        self._tick_secs: deque[float] = deque(maxlen=60)
        self._total_secs = 0.0                      # whole-run pipeline-busy time → fps_stats (capacity)
        self._min_tick = float("inf")               # whole-run tick-time extremes → fps min/avg/max in
        self._max_tick = 0.0                        # the exit summary (the slowest tick bounds fps min)

    def step(self) -> SceneSnapshot | None:
        """One tick if a fresh frame is available, else ``None``. Take the newest
        frame, run the pipeline, publish the snapshot. (Control commands are drained
        + applied by the ControlRouter on the main-thread supervisor — NOT here: a
        paused pipeline gets no frame and returns early, so a consumer on this thread
        could never receive ``resume``.)"""
        fc: FrameCaptured | None = self._frames.take(self._frame_timeout)
        if fc is None:
            return None                          # no new frame within the timeout
        t0 = perf_counter()
        snap = self._pipeline.run(fc)
        dt = perf_counter() - t0
        self._tick_secs.append(dt)
        self._total_secs += dt
        if dt < self._min_tick:
            self._min_tick = dt
        if dt > self._max_tick:
            self._max_tick = dt
        self._snaps.publish(snap)
        self.ticks += 1
        return snap

    def run(self) -> None:
        """Thread body: ``step()`` until stopped. A tick that raises (a stage
        fatal) is captured to ``self.error`` and stops the loop; the
        orchestrator re-raises it on the main thread for a clean exit."""
        self._log.info("[loop] pipeline thread started")
        try:
            while not self._stop.is_set():
                self.step()
                if self._max_frames is not None and self.ticks >= self._max_frames:
                    self._log.info(f"[loop] reached --max-frames={self._max_frames}, stopping")
                    break
        except Exception as exc:                 # fatal
            self.error = exc
            self._log.error(f"[loop] fatal in tick: {exc}")
        finally:
            self._stop.set()
            self._log.info(f"[loop] pipeline thread stopped after {self.ticks} ticks")

    def stop(self) -> None:
        self._stop.set()

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    @property
    def fps(self) -> float:
        """PROCESS throughput: 1 / avg(pipeline.run time) over the recent window —
        real pipeline headroom, excluding the input FPS-gate."""
        if not self._tick_secs:
            return 0.0
        avg = sum(self._tick_secs) / len(self._tick_secs)
        return (1.0 / avg) if avg > 0 else 0.0

    @property
    def fps_stats(self) -> "tuple[float, float, float]":
        """Whole-run capacity fps as (min, avg, max): the slowest tick bounds fps min, the
        fastest bounds fps max. Zeros before the first tick."""
        if self.ticks == 0 or self._total_secs <= 0:
            return (0.0, 0.0, 0.0)
        avg = self.ticks / self._total_secs
        lo = (1.0 / self._max_tick) if self._max_tick > 0 else avg
        hi = (1.0 / self._min_tick) if self._min_tick > 0 else avg
        return (lo, avg, hi)
