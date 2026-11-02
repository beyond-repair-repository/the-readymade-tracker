"""Recorder — rec/stop capture of a chosen view to ``recordings/<timestamp>.mp4``.

An OUTPUT plugin on the ``SnapshotBus`` with its OWN thread; the encode (``cv2.VideoWriter``) never
touches the pipeline. It is idle until toggled — rec/stop arrives as a thread-safe FLAG (the gui ``r``
key, OSCIn ``/rmt/rec start|stop``, or a webui button), and the recorder THREAD opens/closes the writer
on the transition, so every cv2 call stays on that one thread.

Backpressure is free: the bus is latest-wins, so if the encoder falls behind, the intermediate frames
are simply never handed out — they're counted as *dropped* (surfaced to the HUD / ``status()``), and the
pipeline never stalls. Records the PROCESSED view by default — the exact preprocess output (flip + crop +
lens + flatfield + circular mask baked into color), so a recording replays as a test video with
``--no-preprocess`` and behaves like the live run. ``[rec] view`` can instead pick input (the raw
pre-crop frame — full-fidelity source footage) or debug, via the shared ``lib.compose``.
"""

from __future__ import annotations

import os
import threading
import time
from contextlib import nullcontext
from typing import Any, Callable

import cv2

from lib.compose import mask_circle_px, render_view
from lib.fsutil import own_to_dir_owner
from lib.logger import NULL_LOGGER
from lib.params import ParamSpec, bind_params

_FOURCC = {"mp4v": "mp4v", "avc1": "avc1", "xvid": "XVID", "mjpg": "MJPG"}


class Recorder:
    name = "rec"

    # -- live parameters, declared WITH the plugin. dir/codec/view are cached attrs (codec/view lowercased
    #    via apply); fps ('or 30' on 0) and width/height (collapse into self._size) keep their setup reads
    #    (bind=False, registry-only) since they don't map 1:1 to a stored attr. --
    PARAMS = (
        ParamSpec("dir", str, "recordings", attr="_dir", doc="output directory for recordings"),
        ParamSpec("codec", str, "mp4v", enum=("mp4v", "avc1", "xvid", "mjpg"),
                  apply=lambda self, v: setattr(self, "_codec", str(v).lower()), doc="fourcc video codec"),
        ParamSpec("view", str, "processed", enum=("processed", "input", "debug"),
                  apply=lambda self, v: setattr(self, "_view", str(v).lower()),
                  doc="which view to record: processed (the preprocess output - replays with --no-preprocess) | input (raw pre-crop) | debug (fg-masked)"),
        ParamSpec("fps", int, 30, lo=1, bind=False, doc="recording playback frame rate"),
        ParamSpec("width", int, 0, lo=0, hot=False, bind=False, doc="output width px (0 = source size)"),
        ParamSpec("height", int, 0, lo=0, hot=False, bind=False, doc="output height px (0 = source size)"),
    )

    def __init__(self) -> None:
        self._log = NULL_LOGGER
        self._prof: Any = None
        self._bus: Any = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._writer: Any = None
        self._lock = threading.Lock()
        self._want = False                       # rec/stop REQUEST (set from any thread)
        self._path: str | None = None
        self._size_active: tuple[int, int] | None = None
        self._t0 = 0.0
        self._written = 0
        self._dropped = 0
        # config (filled in setup)
        self._dir = "recordings"
        self._codec = "mp4v"
        self._view = "processed"
        self._fps = 30.0
        self._size: tuple[int, int] | None = None

    def setup(self, config: Any, *, logger: Any = None, profiler: Any = None,
              send_fn: Callable | None = None, preprocess_cfg: Any = None) -> None:
        self._log = logger or NULL_LOGGER
        self._prof = profiler
        self._pp = preprocess_cfg          # live PreprocessConfig → the processed view's mask circle
        g = config.get if config is not None else (lambda k, d=None: d)
        bind_params(self, config)          # _dir / _codec / _view ← PARAMS (bind=True)
        self._fps = float(g("fps", 30.0)) or 30.0        # 0 → 30 (a writer needs a real rate); registry-only spec
        w, h = int(g("width", 0)), int(g("height", 0))   # collapse into _size (not self._width/_height)
        self._size = (w, h) if w > 0 and h > 0 else None

    # -- control (thread-safe flag; the recorder thread does the cv2 work) --- #

    def set_recording(self, on: bool) -> None:
        with self._lock:
            self._want = bool(on)

    def toggle(self) -> None:
        with self._lock:
            self._want = not self._want

    def is_recording(self) -> bool:
        return self._writer is not None

    def elapsed(self) -> float:
        return (time.perf_counter() - self._t0) if self._writer is not None else 0.0

    @property
    def dropped(self) -> int:
        return self._dropped

    def status(self) -> dict:
        return {"recording": self.is_recording(), "path": self._path,
                "elapsed": round(self.elapsed(), 1), "written": self._written, "dropped": self._dropped}

    # -- lifecycle ----------------------------------------------------------- #

    def start(self, bus: Any) -> "Recorder":
        self._bus = bus
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="rec", daemon=True)
        self._thread.start()
        self._log.info(f"[rec] ready - view={self._view} codec={self._codec} fps={self._fps:g} -> {self._dir}/")
        return self

    def stop(self) -> None:
        with self._lock:
            self._want = False                   # clear the rec INTENT so `output rec off`→`on` starts
        self._stop.set()                         # idle (else a live recording spontaneously resumes)
        alive = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            alive = self._thread.is_alive()      # join timed out → the encoder still owns _writer
            self._thread = None
        if not alive:                            # only finalize from here if the thread exited (its own
            self._close()                        # finally _close()s otherwise) — no race on _writer

    # -- the encode thread --------------------------------------------------- #

    def _run(self) -> None:
        last_idx: Any = object()
        while not self._stop.is_set():
            snap = self._bus.wait_fresh(last_idx, 0.5)
            if snap is None:
                continue
            idx = getattr(snap, "frame_idx", None)
            if idx == last_idx:
                continue
            with self._lock:
                want = self._want
            if want and self._writer is None:
                self._open(snap)
            elif not want and self._writer is not None:
                self._close()
            if self._writer is not None:
                if isinstance(idx, int) and isinstance(last_idx, int):
                    self._dropped += max(0, idx - last_idx - 1)   # bus skipped ahead → missed frames
                span = self._prof.span("rec", group="output") if self._prof is not None else nullcontext()
                try:
                    with span:
                        self._write(snap)
                except Exception as e:           # noqa: BLE001 — a bad frame must not kill the thread
                    self._log.warning(f"[rec] write failed: {e}")
            last_idx = idx
        self._close()

    def _open(self, snap: Any) -> None:
        mc = mask_circle_px(getattr(self, "_pp", None)) if self._view == "processed" else None
        img = render_view(snap, self._view, with_label=False, mask_circle=mc)
        if img is None:
            return
        h, w = img.shape[:2]
        size = self._size or (w, h)
        os.makedirs(self._dir, exist_ok=True)
        own_to_dir_owner(self._dir)          # a sudo run must not root-own recordings/ (locks later runs out)
        path = self._timestamp_path()
        fourcc = cv2.VideoWriter_fourcc(*_FOURCC.get(self._codec, "mp4v"))
        writer = cv2.VideoWriter(path, fourcc, self._fps, size)
        own_to_dir_owner(path)
        if not writer.isOpened():
            self._log.warning(f"[rec] cannot open a writer for {path} (codec {self._codec}) - not recording")
            with self._lock:
                self._want = False               # don't spin retrying a broken codec every frame
            return
        self._writer, self._path, self._size_active = writer, path, size
        self._t0, self._written, self._dropped = time.perf_counter(), 0, 0
        self._log.info(f"[rec] * recording -> {path} ({size[0]}x{size[1]} @ {self._fps:g})")

    def _write(self, snap: Any) -> None:
        mc = mask_circle_px(getattr(self, "_pp", None)) if self._view == "processed" else None
        img = render_view(snap, self._view, with_label=False, mask_circle=mc)
        if img is None:
            return
        if (img.shape[1], img.shape[0]) != self._size_active:
            img = cv2.resize(img, self._size_active)   # keep the writer's constant frame size
        self._writer.write(img)
        self._written += 1

    def _close(self) -> None:
        if self._writer is not None:
            elapsed, path, written, dropped = self.elapsed(), self._path, self._written, self._dropped
            self._writer.release()
            self._writer = None
            self._log.info(f"[rec] # stopped {path} - {written} frames, {dropped} dropped, {elapsed:.1f}s")

    def _timestamp_path(self) -> str:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path = os.path.join(self._dir, f"{stamp}.mp4")
        n = 2
        while os.path.exists(path):              # two starts in the same second → -2, -3, …
            path = os.path.join(self._dir, f"{stamp}-{n}.mp4")
            n += 1
        return path
