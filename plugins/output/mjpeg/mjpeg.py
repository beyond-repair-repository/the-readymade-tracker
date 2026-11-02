"""``mjpeg`` OUTPUT plugin — a live MJPEG video stream over HTTP.

A background HTTP server (Python stdlib ``http.server``, dependency-free) serving
``multipart/x-mixed-replace``: each part a JPEG of the tracker's current VIEW. Point a
browser, VLC, OBS, or ``tools/tuio2_monitor.py --mjpeg-stream PORT`` at
``http://host:port/`` and watch the pipeline live.

Decoupled from the pipeline via the ``SnapshotBus`` (the single channel out of the core):
ONE publisher thread waits on the bus, JPEG-encodes the chosen view ONCE per fresh frame,
and every connected client streams that shared buffer — so N viewers cost one encode, and
a slow/absent client never touches tick timing.

``view`` picks what to stream:
  * ``debug`` (default) — the cropped input frame MASKED to-zero by observe's fg (gray > fg_threshold).
  * ``input``           — the full pre-crop frame (crop-calibration view).
The overlay-with-object-ids view is reconstructed CLIENT-side by the tuio2_monitor (it draws
the TUIO contours/ids over this video), so the server stays a plain video source.

Enable with ``[mjpeg]`` in ``readymade.conf`` (or force it with the ``--mjpeg`` CLI flag; the
port is the section's ``port`` key). ``setup`` → ``start(bus)`` → ``stop``.
"""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable

import cv2

from lib.compose import VIEWS, render_view
from lib.logger import NULL_LOGGER
from lib.loop import consume_bus
from lib.params import ParamSpec, bind_params

_VIEW_NAMES = frozenset(VIEWS)                    # input | debug — the two single panels the stream serves

_BOUNDARY = "readymadeframe"          # the multipart part separator (any token; clients read it from the header)


def _multipart_chunk(jpeg: bytes, boundary: str = _BOUNDARY) -> bytes:
    """One ``multipart/x-mixed-replace`` part: the boundary + JPEG part-headers + the bytes.
    Pure (unit-tested) — the exact framing the HTTP handler writes per frame."""
    head = (f"--{boundary}\r\n"
            f"Content-Type: image/jpeg\r\n"
            f"Content-Length: {len(jpeg)}\r\n\r\n").encode("ascii")
    return head + jpeg + b"\r\n"


class MjpegOut:
    """MJPEG-over-HTTP video-out. ``setup`` → ``start(bus)`` → ``stop``."""

    # -- live parameters, declared WITH the plugin. host/port bind the HTTP server at setup → needs-restart.
    #    view clamps to input|debug; scale guards ≤0 → 1.0; both via apply setters. --
    PARAMS = (
        ParamSpec("host", str, "0.0.0.0", hot=False, doc="HTTP bind address for the MJPEG server (0.0.0.0 = all)"),
        ParamSpec("port", int, 8080, lo=0, hi=65535, hot=False, doc="TCP port the MJPEG-over-HTTP server listens on"),
        ParamSpec("view", str, "debug", enum=("input", "debug"),
                  apply=lambda self, v: setattr(self, "_view", str(v).lower() if str(v).lower() in _VIEW_NAMES else "debug"),
                  doc="which single view to stream: input (full pre-crop) | debug (input masked to fg)"),
        ParamSpec("scale", float, 1.0, lo=0.0, hi=1.0,
                  apply=lambda self, v: setattr(self, "_scale", v if v > 0 else 1.0),
                  doc="downscale factor for the stream"),
        ParamSpec("quality", int, 80, lo=1, hi=100, doc="JPEG quality"),
        ParamSpec("max_fps", float, 0.0, lo=0.0, hot=False, doc="stream fps cap (0 = every frame)"),
    )

    def __init__(self) -> None:
        self._log = NULL_LOGGER
        self._prof: Any = None                   # --profile: times the per-frame JPEG encode (OUTPUT)
        self._bus: Any = None
        self._srv: Any = None
        self._srv_thread: threading.Thread | None = None
        self._pub_thread: threading.Thread | None = None
        self._stop = threading.Event()
        # the shared latest-encoded-frame, guarded by a Condition (publisher notifies clients)
        self._cond = threading.Condition()
        self._jpeg: bytes | None = None
        self._seq = 0
        self._clients = 0                        # connected viewers; 0 → the publisher skips encode/render
        # config (filled in setup)
        self._host = "0.0.0.0"
        self._port = 8080
        self._view = "debug"
        self._quality = 80
        self._scale = 1.0
        self._max_fps = 0.0

    # -- lifecycle ----------------------------------------------------------- #

    def setup(self, config: Any, *, logger: Any = None,
              send_fn: Callable[[bytes], Any] | None = None, profiler: Any = None) -> None:
        # send_fn is accepted for output-plugin signature parity (tests); MJPEG has no packet sink.
        self._log = logger or NULL_LOGGER
        self._prof = profiler
        bind_params(self, config)          # _host / _port / _view / _scale / _quality / _max_fps ← PARAMS

    def set_view(self, view: str) -> None:
        """Runtime view switch (the ControlRouter / OSCIn ``/rmt/view``). One of input|debug. Unknown
        names are ignored. A single string write — atomic vs the publisher thread that reads it once per
        frame (no lock needed)."""
        v = str(view).lower()
        if v in _VIEW_NAMES:
            self._view = v
            self._log.info(f"[mjpeg] view -> {v}")
        else:
            self._log.warning(f"[mjpeg] unknown view: {view} (have {sorted(_VIEW_NAMES)})")

    def current_view(self) -> str:
        """The view the stream is currently showing (input|debug)."""
        return self._view

    def start(self, bus: Any) -> "MjpegOut":
        self._bus = bus
        self._stop.clear()
        self._srv = ThreadingHTTPServer((self._host, self._port), self._make_handler())
        self._srv.daemon_threads = True          # a client thread never blocks shutdown
        self._srv_thread = threading.Thread(target=self._srv.serve_forever,
                                            name="mjpeg-http", daemon=True)
        self._srv_thread.start()
        self._pub_thread = threading.Thread(target=self._publish, name="mjpeg-pub", daemon=True)
        self._pub_thread.start()
        self._log.info(f"[mjpeg] http://{self._host}:{self._port}/ "
                       f"view={self._view} scale={self._scale} quality={self._quality} "
                       f"fps={'max' if self._max_fps <= 0 else self._max_fps}")
        return self

    def stop(self) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()              # release any client blocked waiting for a frame
        if self._srv is not None:
            if self._srv_thread is not None and self._srv_thread.is_alive():
                self._srv.shutdown()             # unblock serve_forever — ONLY if it's actually running
            #                                      (shutdown() blocks forever if serve_forever never started)
            self._srv.server_close()
            self._srv = None
        for t in (self._pub_thread, self._srv_thread):
            if t is not None:
                t.join(timeout=1.0)
        self._pub_thread = self._srv_thread = None

    # -- the pure encode path (unit-tested; no server) ----------------------- #

    def _select(self, snap: Any) -> Any:
        """The BGR image for the current view (``None`` if unavailable). Uses the shared
        composer so the stream can serve input / debug. ``with_label=False``: no burnt-in panel text
        (a clean feed for a projector/recorder)."""
        return render_view(snap, self._view, with_label=False)

    def encode(self, snap: Any) -> bytes | None:
        """Pick the view image, optionally downscale by ``scale``, and JPEG-encode it at
        ``quality``. ``None`` when there's nothing to send (no frame, or cv2 failed). Pure
        — the whole payload path, testable."""
        img = self._select(snap)
        if img is None:
            return None
        if self._scale != 1.0:                           # downscale the stream (bandwidth / CPU)
            h, w = img.shape[:2]
            nw, nh = max(1, round(w * self._scale)), max(1, round(h * self._scale))
            if (nw, nh) != (w, h):
                interp = cv2.INTER_AREA if self._scale < 1.0 else cv2.INTER_LINEAR
                img = cv2.resize(img, (nw, nh), interpolation=interp)
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), self._quality])
        return buf.tobytes() if ok else None

    # -- client bookkeeping (on-demand: encode only while someone is watching) --- #

    def _client_join(self) -> None:
        with self._cond:             # pure increment — CANNOT raise, so it always pairs with the
            self._clients += 1       # _client_leave in the handler's finally (no counter leak). The
        #                              seed encode (which CAN raise) is a separate call, inside the try.

    def _client_leave(self) -> None:
        with self._cond:
            self._clients = max(0, self._clients - 1)

    def _seed_latest(self) -> None:
        """Encode the latest snapshot ONCE on connect, so a client that joins while the tracker is
        paused / between frames still sees the current image immediately (the publisher then takes
        over for subsequent frames). No bus / no frame yet → nothing to seed."""
        if self._bus is None:
            return
        snap = self._bus.latest()
        if snap is None:
            return
        jpeg = self.encode(snap)
        if jpeg is not None:
            with self._cond:
                self._jpeg = jpeg
                self._seq += 1
                self._cond.notify_all()

    # -- publisher thread: bus snapshot -> shared JPEG ----------------------- #

    def _publish(self) -> None:
        # NObody watching → skip render/encode entirely (view=debug costs nothing unwatched).
        consume_bus(self._bus, self._stop, self._render_publish, profiler=self._prof, name="mjpeg",
                    max_fps=self._max_fps, should_emit=lambda snap: self._clients > 0)

    def _render_publish(self, snap: Any) -> None:
        try:
            jpeg = self.encode(snap)             # --profile: the per-frame JPEG encode cost
        except Exception as e:                   # noqa: BLE001 — a bad frame must not kill the thread
            self._log.warning(f"[mjpeg] encode failed on frame {getattr(snap, 'frame_idx', '?')}: {e}")
            return
        if jpeg is None:
            return
        with self._cond:
            self._jpeg = jpeg
            self._seq += 1
            self._cond.notify_all()              # wake every streaming client

    # -- HTTP handler: stream the shared JPEG as multipart/x-mixed-replace ---- #

    def _make_handler(self):
        out = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"        # 1.0 + Connection: close — no chunked framing needed
            timeout = 30.0                       # socket read/write timeout: a slow-loris client that
            #                                      stops draining frames for 30s times out (socket.timeout
            #                                      → OSError below) instead of parking a thread + FD forever.

            def log_message(self, *a: Any) -> None:      # silence the default stderr access log
                pass

            def do_GET(self) -> None:            # noqa: N802 — BaseHTTPRequestHandler API
                if self.path.split("?", 1)[0] not in ("/", "/stream", "/stream.mjpg", "/mjpeg"):
                    self.send_error(404)
                    return
                out._client_join()                   # pure increment → always pairs with the finally leave
                try:
                    self.send_response(200)
                    self.send_header("Content-Type",
                                     f"multipart/x-mixed-replace; boundary={_BOUNDARY}")
                    self.send_header("Cache-Control", "no-cache, private")
                    self.send_header("Pragma", "no-cache")
                    self.send_header("Access-Control-Allow-Origin", "*")   # webgui (other port) draws the
                    self.send_header("Connection", "close")                # frame to a canvas for snapshots
                    self.end_headers()
                    out._seed_latest()               # immediate frame on connect (the encode that CAN raise)
                    last_seq = -1                    # -1 ⇒ send the current frame immediately on connect
                    while not out._stop.is_set():
                        with out._cond:
                            # the predicate MUST require an actual frame: with no frame yet
                            # (_jpeg None, _seq 0) a bare `_seq != last_seq` is trivially true
                            # (0 != -1) → wait_for returns without blocking → a 100%-CPU spin.
                            out._cond.wait_for(
                                lambda: (out._jpeg is not None and out._seq != last_seq)
                                        or out._stop.is_set(), timeout=1.0)
                            if out._stop.is_set():
                                break
                            if out._jpeg is None or out._seq == last_seq:
                                continue         # timed out with no fresh frame — block again
                            jpeg, last_seq = out._jpeg, out._seq
                        self.wfile.write(_multipart_chunk(jpeg))
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass                         # the viewer went away — end this connection quietly
                finally:
                    out._client_leave()          # last viewer gone → publisher stops encoding

        return Handler
