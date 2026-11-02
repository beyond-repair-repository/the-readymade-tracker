"""``video`` INPUT backend — a recorded clip as the frame source.

A **pure source** the core's input-runner drives: ``open()`` →
``grab() → (bgr, ts) | None`` → ``close()``, plus a thread-safe ``Transport``
(pause / step / seek) the gui drives. ``grab`` returns ``None`` for *no frame this
call* (FPS-gated / paused / loop-rewind) and raises ``StopIteration`` at EOF with
no loop. The runner stamps ``frame_idx``, runs ``preprocess``, and emits
``FrameCaptured`` — the backend only decodes. PTS ``ts`` is forced monotonic across
a loop-rewind. ``cv2.VideoCapture`` is injectable (``capture_factory``) for tests.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Callable

import cv2

from lib.logger import NULL_LOGGER
from lib.params import ParamSpec


class Transport:
    """Thread-safe playback control (pause / step / seek / stop) for a seekable
    source. Written by the control path (gui keys / OSC-in), read by ``grab()`` on
    the input thread."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._paused = False
        self._step = 0
        self._seek: int | None = None
        self._refresh = False
        self._stopped = False

    def pause(self) -> None:
        with self._lock:
            self._paused = True

    def resume(self) -> None:
        with self._lock:
            self._paused = False

    def toggle_pause(self) -> None:
        with self._lock:
            self._paused = not self._paused

    def step_forward(self) -> None:
        with self._lock:
            self._step += 1

    def step_backward(self) -> None:
        with self._lock:
            self._step -= 1

    def seek(self, frame_idx: int) -> None:
        with self._lock:
            self._seek = max(0, int(frame_idx))

    def refresh(self) -> None:
        """Re-emit the CURRENT frame once — a PAUSED manual scan wants the frozen frame to go back
        through the pipeline so the outputs emit it (a paused source produces no frame otherwise).
        One-shot; cleared by the next ``take``."""
        with self._lock:
            self._refresh = True

    def stop(self) -> None:
        with self._lock:
            self._stopped = True

    @property
    def paused(self) -> bool:
        with self._lock:
            return self._paused

    @property
    def stopped(self) -> bool:
        with self._lock:
            return self._stopped

    def take(self) -> tuple[bool, int, int | None, bool]:
        """Atomically read + clear the pending step/seek/refresh; return (paused, step, seek, refresh)."""
        with self._lock:
            step, seek, refresh = self._step, self._seek, self._refresh
            self._step, self._seek, self._refresh = 0, None, False
            return self._paused, step, seek, refresh


class VideoBackend:
    """Frame source: a recorded clip."""

    # -- the [input] live parameters (registry anchor for the input backend). All bind the source at
    #    startup → needs-restart; bind=False (the backend/loader read the config themselves). --
    PARAMS = (
        ParamSpec("source", str, None, enum=("video", "camera"), hot=False, bind=False, doc="Which input backend supplies frames"),
        ParamSpec("location", str, None, hot=False, bind=False, doc="Path to the video clip file to play"),
        ParamSpec("fps", int, None, lo=0, hot=False, bind=False, doc="Playback fps override (0 = use clip's native fps)"),
        ParamSpec("loop", bool, None, hot=False, bind=False, doc="Rewind and replay the clip at end of stream"),
        ParamSpec("buffer_frames", int, 60, lo=2, hot=False, bind=False,
                  doc="Back-step replay ring size (frames) - recent decoded frames kept so a step back replays exactly (no cv2 keyframe seek)"),
    )

    def __init__(self, config: Any, *, capture_factory: Callable[[Any], Any] | None = None,
                 logger: Any = None) -> None:
        self._open_cap = capture_factory or cv2.VideoCapture
        self._log = logger or NULL_LOGGER
        self._path = config.get("location") if config is not None else None
        self._loop = bool(config.get("loop", False)) if config is not None else False
        self._cfg_fps = int(config.get("fps", 0)) if config is not None else 0
        self._cap: Any = None
        self._src_fps = 0.0
        self._frame_period = 0.0
        self._pos = -1
        self._next_due = 0.0
        self._last_ts = -1.0
        # backward-step ring buffer: cv2's CAP_PROP_POS_FRAMES seeks to keyframes, so a 1-frame
        # back-step reads the wrong frame. Replay recent decoded frames instead (no cv2 seek).
        buf = int(config.get("buffer_frames", 60)) if config is not None else 60
        self._buf: deque = deque(maxlen=max(2, buf))
        self._back = 0                          # how many frames behind the head we're showing
        self.transport = Transport()

    def open(self) -> None:
        if not self._path:
            raise ValueError("video backend requires 'location' (a clip path)")
        self._cap = self._open_cap(self._path)
        if not self._cap.isOpened():
            raise RuntimeError(f"video: cannot open source {self._path!r}")
        self._src_fps = float(self._cap.get(cv2.CAP_PROP_FPS)) or 0.0
        play_fps = float(self._cfg_fps) if self._cfg_fps > 0 else self._src_fps
        self._frame_period = (1.0 / play_fps) if play_fps > 0 else 0.0
        self._next_due = time.perf_counter()
        self._log.info(f"video source={self._path!r} src_fps={self._src_fps:.1f} "
                       f"play_fps={play_fps:.1f} loop={self._loop}")

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def grab(self) -> tuple[Any, float] | None:
        """One raw ``(bgr, ts)``, or ``None`` (gated / paused / rewound);
        ``StopIteration`` at EOF with no loop."""
        t = self.transport
        if t.stopped:
            raise StopIteration
        paused, step, seek, refresh = t.take()

        if refresh and step == 0 and seek is None and self._buf:   # PAUSED manual scan → re-emit the
            return self._buf[-(self._back + 1)]                    # frozen (currently shown) frame, no advance

        if paused and step == 0 and seek is None:
            time.sleep(0.01)
            self._next_due = time.perf_counter()
            return None

        if step == 0 and seek is None and self._frame_period > 0.0:
            now = time.perf_counter()
            if now < self._next_due:
                time.sleep(min(0.005, self._next_due - now))
                return None
            self._next_due += self._frame_period
            if self._next_due < now:             # fell far behind real-time (startup warmup, a slow
                self._next_due = now + self._frame_period   # stretch) → resync; DON'T burst frames to
                                                            # catch up (that drops the first second)

        if seek is not None:                     # explicit jump (rare; the gui uses step, not seek)
            self._buf.clear()
            self._back = 0
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, seek)
            self._pos = seek - 1
            return self._decode_next()

        move = step if step != 0 else 1          # playing / step-forward advances one frame
        new_back = self._back - move             # forward → toward the head (back↓); backward → back↑
        if new_back > 0 and self._buf:           # replay a buffered PAST frame — exact, no cv2 seek
            self._back = min(new_back, len(self._buf) - 1)
            return self._buf[-(self._back + 1)]
        self._back = 0                           # at / past the head → decode forward, show the latest
        frame = None
        for _ in range(max(1, -new_back)):
            frame = self._decode_next()
            if frame is None:                    # loop-rewind this call
                break
        return frame

    def _decode_next(self) -> tuple[Any, float] | None:
        """Decode + buffer the next sequential frame (so back-steps can replay it). ``None`` on a
        loop-rewind this call; ``StopIteration`` at EOF with no loop."""
        ts_msec = float(self._cap.get(cv2.CAP_PROP_POS_MSEC))
        ok, bgr = self._cap.read()
        if not ok or bgr is None:
            if self._loop:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self._pos = -1
                self._next_due = time.perf_counter()
                self._log.info("EOF -> rewind (loop)")
                return None
            self._log.info("end of stream")
            raise StopIteration
        self._pos += 1
        ts = ts_msec / 1000.0           # video PTS, forced monotonic across a loop-rewind
        if ts <= self._last_ts:
            ts = self._last_ts + ((1.0 / self._src_fps) if self._src_fps > 0 else (1.0 / 30.0))
        self._last_ts = ts
        self._buf.append((bgr, ts))
        return (bgr, ts)
