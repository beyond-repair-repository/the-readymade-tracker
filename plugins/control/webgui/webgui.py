"""WebGui — the browser control surface + the OSC↔HTTP bridge.

A dep-free ``http.server`` (like MJPEG), co-hosted in the tracker, that gives any browser on the LAN a
live view + a full control panel WITHOUT the browser speaking OSC:

  GET  /              → the single-page UI (static/); GET /<file> → its assets (traversal-guarded)
  GET  /info          → bootstrap JSON: the OSC prefix + the MJPEG port (so the <img> knows where to look)
  POST /osc           → {address, args} → lib.osc → a loopback UDP datagram to OSCIn → collect the reply
                        (until the /ack · /reloaded · /saved · /params terminator, or a ~2 s deadline —
                        inter-packet gaps are deliberately waited out, never treated as the end) → {messages}
  GET  /events        → Server-Sent Events: each fresh SceneSnapshot serialised to JSON (objects + pose +
                        contour + annotations), pushed live — no polling.

The bridge speaks ONLY OSC to the tracker (loopback UDP to OSCIn), so all the param logic — the rich
registry, clamp/validate, the reply framing — is reused, not duplicated. The SSE data comes straight off
the ``SnapshotBus`` (this is an in-process plugin), so the live overlay is real-time. The UI builds ITSELF
from ``/rmt/query`` — no per-parameter HTML — so a new @feature or knob just appears.
"""

from __future__ import annotations

import json
import os
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable

from lib.geometry import place_footprint, simplify_contour
from lib.logger import NULL_LOGGER
from lib.osc import decode_packet, encode_message
from lib.params import ParamSpec, bind_params

_HERE = os.path.dirname(os.path.abspath(__file__))
_STATIC = os.path.join(_HERE, "static")
_REPLY_VERBS = frozenset({"set", "reload", "query", "save"})  # these OSCIn verbs answer; the rest don't
_MIME = {".html": "text/html", ".js": "application/javascript", ".css": "text/css",
         ".json": "application/json", ".svg": "image/svg+xml", ".ico": "image/x-icon",
         ".png": "image/png", ".map": "application/json"}


def _mime(path: str) -> str:
    return _MIME.get(os.path.splitext(path)[1].lower(), "application/octet-stream")


class WebGui:
    name = "webgui"

    # -- live parameters: host/port bind the HTTP server → needs-restart. The OSC-bridge + video-underlay
    #    targets are NOT knobs here — build() single-sources them from the LIVE [oscin]/[mjpeg] via
    #    attach_bridge(), so a port/prefix change there can't desync a webgui mirror and kill the panel. --
    PARAMS = (
        ParamSpec("host", str, "0.0.0.0", hot=False, doc="UI HTTP bind address - restart"),
        ParamSpec("port", int, 8090, hot=False, doc="UI HTTP port - restart"),
    )

    def __init__(self) -> None:
        self._log = NULL_LOGGER
        self._bus: Any = None
        self._srv: Any = None
        self._srv_thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._rec: Any = None                          # recorder handle → rec status in the SSE payload
        self._mjpeg: Any = None                        # mjpeg handle → current stream view (overlay mapping)
        self._transport: Any = None                    # input transport → pause state for the browser
        self._stats_fn: Callable | None = None         # () -> pipeline fps (core.loop.fps), for the header
        self._preprocess: Any = None                   # live Preprocess → the mask-circle overlay geometry
        # HTTP bind (from [webgui])
        self._host = "0.0.0.0"
        self._port = 8090
        # bridge / underlay targets — attach_bridge() overwrites these from the LIVE [oscin]/[mjpeg] at
        # build. oscin_host is loopback ALWAYS: the browser bridge and OSCIn share this process/machine.
        self._oscin_host = "127.0.0.1"
        self._oscin_port = 3401
        self._prefix = "/rmt"
        self._mjpeg_port = 8080

    def setup(self, config: Any, *, logger: Any = None, profiler: Any = None,
              send_fn: Callable | None = None) -> None:
        self._log = logger or NULL_LOGGER
        bind_params(self, config)          # _host / _port ← PARAMS

    # -- wiring (set once by build(); all optional → graceful when absent) --- #

    def attach_recorder(self, rec: Any) -> None:
        """The [rec] recorder, so the SSE payload can carry live rec state to the browser."""
        self._rec = rec

    def attach_mjpeg(self, mjpeg: Any) -> None:
        """The [mjpeg] output, so the browser knows which view the stream is showing (the overlay
        maps onto the single-panel view, input|debug)."""
        self._mjpeg = mjpeg

    def attach_transport(self, transport: Any) -> None:
        """The input Transport, so pause state reaches the browser (head + the quiet-window ping) —
        the play/pause button must track a pause toggled from ANY surface (oscin, mcu, another tab)."""
        self._transport = transport

    def attach_stats(self, fn: Callable) -> None:
        """A ``() -> float`` returning the pipeline fps (core.loop.fps) for the header."""
        self._stats_fn = fn

    def attach_preprocess(self, pp: Any) -> None:
        """The live Preprocess, so the SSE payload can carry the mask CIRCLE (its effect is invisible in
        the frame — mask zeroes gray only — so it can only be shown as a drawn overlay for calibration)."""
        self._preprocess = pp

    def attach_bridge(self, *, oscin_port: int, prefix: str, mjpeg_port: int) -> None:
        """Single-source the OSC bridge + video underlay from the LIVE [oscin]/[mjpeg] endpoints (build()
        passes the actual configured values), so a port/prefix change there can't desync a webgui mirror
        knob and kill the control panel. oscin_host stays loopback — the bridge reaches the in-process OSCIn."""
        self._oscin_port = int(oscin_port)
        self._prefix = "/" + str(prefix).strip("/")
        self._mjpeg_port = int(mjpeg_port)

    # -- lifecycle ----------------------------------------------------------- #

    def start(self, bus: Any) -> "WebGui":
        self._bus = bus
        self._stop.clear()
        self._srv = ThreadingHTTPServer((self._host, self._port), self._make_handler())
        self._srv.daemon_threads = True
        self._srv_thread = threading.Thread(target=self._srv.serve_forever, name="webgui-http", daemon=True)
        self._srv_thread.start()
        self._log.info(f"[webgui] http://{self._host}:{self._port}/  -> OSCIn {self._oscin_host}:{self._oscin_port}")
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._srv is not None:
            if self._srv_thread is not None and self._srv_thread.is_alive():
                self._srv.shutdown()
            self._srv.server_close()
            self._srv = None
        if self._srv_thread is not None:
            self._srv_thread.join(timeout=1.0)
            self._srv_thread = None

    # -- POST /osc: browser → OSCIn (loopback UDP) → collect the reply ------- #

    def osc_roundtrip(self, address: str, args: list) -> list[dict]:
        """Send one OSC message to OSCIn and collect its reply (pure — unit-tested). A control verb
        (pause/step/view/output/rec) gets no answer → returns []; set/reload/save/query collect until
        the explicit terminator (/ack · /reloaded · /saved · /params) or the ~2 s deadline — NEVER on a
        mere inter-packet gap (see the comment in the loop)."""
        verb = address.rstrip("/").rsplit("/", 1)[-1]
        expect = verb in _REPLY_VERBS
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("127.0.0.1", 0))                           # OSCIn replies to THIS ephemeral socket
        try:
            sock.sendto(encode_message(address, args), (self._oscin_host, self._oscin_port))
            if not expect:
                return []
            out: list[dict] = []
            sock.settimeout(0.2)
            deadline = time.perf_counter() + 2.0
            # A /rmt/query reply is CHUNKED into several UDP datagrams (the manifest > one datagram).
            # Complete ONLY on the explicit terminator (/ack · /reloaded · /params) or the deadline —
            # NEVER on a mere inter-packet gap: under load the tracker can pace the chunks >0.2s apart,
            # and an early break there would silently truncate the manifest (dropping whole param rows,
            # incl. enum dropdowns). The terminator is the normal fast exit; the deadline bounds loss.
            while time.perf_counter() < deadline:
                try:
                    data, _ = sock.recvfrom(65535)
                except socket.timeout:
                    continue                                 # a gap is not the end — wait for terminator
                done = False
                for a, ar in decode_packet(data):
                    out.append({"address": a, "args": ar})
                    if a.rsplit("/", 1)[-1] in ("ack", "reloaded", "params", "saved"):
                        done = True                          # explicit terminator → reply complete
                if done:
                    break
            return out
        finally:
            sock.close()

    # -- GET /events: SnapshotBus → SSE ------------------------------------- #

    @staticmethod
    def _dims(snap: Any) -> tuple[int, int]:
        fr = getattr(snap, "frame", None)
        if fr is not None and getattr(fr, "shape", None):
            return int(fr.shape[1]), int(fr.shape[0])
        return 0, 0

    def snapshot_json(self, snap: Any) -> str:
        """Serialise a SceneSnapshot to compact JSON for the SSE overlay (coords normalized 0..1)."""
        fw, fh = self._dims(snap)
        nx = (lambda v: v / fw) if fw else (lambda v: v)
        ny = (lambda v: v / fh) if fh else (lambda v: v)
        objs = []
        hp = getattr(snap, "hidden_present", frozenset())

        def obj_dict(o, grp=False):
            x, y, th = o.pose
            # continuous absolute θ (pose-maintained theta_cumulative) — else the rotation tick snaps back
            # to 0 each time the object re-settles and re-bakes its footprint.
            cum = getattr(o, "theta_cumulative", None)
            # label = the TRACKER id: every surface (this overlay, TUIO, the OSC egress) speaks ONE id
            # space, so the webgui and a TUIO monitor always agree. The recycled display slots remain
            # in the Scene, unused by outputs.
            d: dict = {"id": o.id, "label": o.id, "state": o.status.name, "x": nx(x), "y": ny(y),
                       "confirmed": bool(getattr(o, "confirmed", True)),   # output overlay hides unconfirmed
                       "theta": float(cum) if cum is not None else float(th)}
            if o.status.name == "HIDDEN":
                d["present"] = o.id in hp           # covered (hand over it) vs GONE — the band keeps covered
            if getattr(o, "bbox", None) is not None:   # axis-aligned bounds of the last observed blob
                bx, by, bw, bh = o.bbox                # (x,w by width; y,h by height — like cluster bbox)
                d["bbox"] = [nx(float(bx)), ny(float(by)), nx(float(bw)), ny(float(bh))]
            fpc = o.footprint_contour
            fc = getattr(o, "footprint_centroid", None)
            wc = getattr(o, "wire_contour", None)       # the PERCEPTUAL outline — already wire quality
            if wc is not None and fc is not None:
                placed = place_footprint(wc, fc, o.pose)
                d["contour"] = [[nx(float(p[0][0])), ny(float(p[0][1]))] for p in placed]
            elif fpc is not None:
                # fallback (features stage off): place the rest outline AT the live pose (as the gui/tuio2
                # do) — else it stays frozen at the rest position and appears to lag the moving object.
                placed = place_footprint(fpc, fc, o.pose) if fc is not None else fpc
                placed = simplify_contour(placed, float(getattr(snap, "contour_epsilon", 0.75)))
                d["contour"] = [[nx(float(p[0][0])), ny(float(p[0][1]))] for p in placed]
            ef = getattr(o, "efd_full", None)           # COMPLETE ± spectrum (diagnostic) — the features
            if ef is not None:                          #   inspector draws the rejected bins dim; SSE-only,
                d["efd_full"] = [list(ef[0]), list(ef[1])]   # never on OSC/TUIO2
            ann = getattr(o, "annotations", None)
            if ann:
                d["ann"] = {k: dict(v) if hasattr(v, "items") else v for k, v in ann.items()}
            if grp:
                d["grp"] = True                     # a COMPOUND (fused emit view) — the features view
            return d                                #   shows it; the process canvas draws the overlay
        for o in snap.objects:
            if getattr(o, "ignored", False) or o.pose is None:
                continue
            objs.append(obj_dict(o))
        emit_ids = None
        if getattr(snap, "emit", None) is not None:    # fusion active: the FEATURES view follows the wire
            real = {o.id for o in snap.objects}
            emit_ids = [o.id for o in snap.emit if getattr(o, "pose", None) is not None]
            objs += [obj_dict(o, grp=True) for o in snap.emit
                     if o.id not in real and getattr(o, "pose", None) is not None]
        groups = []
        for g in getattr(snap, "groups", ()) or ():        # compound super-objects (Grouper) — over budget only
            gx, gy = g.centroid
            gd: dict = {"id": g.id, "members": list(g.members), "n": len(g.members),
                        "x": nx(gx), "y": ny(gy), "size": float(g.size),
                        "theta": float(g.theta) if g.theta is not None else None}
            if getattr(g, "contour", None) is not None:
                gd["contour"] = [[nx(float(p[0][0])), ny(float(p[0][1]))] for p in g.contour]
            groups.append(gd)
        clusters = []
        for c in getattr(snap, "clusters", ()) or ():      # spatial clusters (Clusters stage) — cluster-centric
            ccx, ccy, cr = c.circle
            bx, by, bw, bh = c.bbox
            cd: dict = {"id": c.id, "members": list(c.members), "n": len(c.members),
                        "cx": nx(c.centroid[0]), "cy": ny(c.centroid[1]),
                        "circle": [nx(ccx), ny(ccy), nx(cr)],   # radius by width (frames ~square)
                        "bbox": [nx(bx), ny(by), nx(bw), ny(bh)]}
            if getattr(c, "hull", None) is not None:
                cd["hull"] = [[nx(float(p[0][0])), ny(float(p[0][1]))] for p in c.hull]
            clusters.append(cd)
        head: dict = {"frame": getattr(snap, "frame_idx", 0), "ts": getattr(snap, "ts", None),
                      "objects": objs}
        if emit_ids is not None:
            head["emit"] = emit_ids                   # the wire's voice list (features view = this view)
        if groups:
            head["groups"] = groups
        if clusters:
            head["clusters"] = clusters
        if fw and fh:                                     # cropped frame dims → object overlays (output view)
            head["fdim"] = [fw, fh]
        full = getattr(snap, "input_full", None)          # the pre-crop frame dims → the crop/mask overlay on
        if full is not None and getattr(full, "shape", None):   # the INPUT view lands correctly (full-frame px)
            head["full_dim"] = [int(full.shape[1]), int(full.shape[0])]
        if self._stats_fn is not None:                    # pipeline fps (headroom) — matches the desktop HUD
            try:
                head["proc_fps"] = round(float(self._stats_fn()), 1)
            except Exception:                             # noqa: BLE001 — a stats hiccup must not kill the SSE
                pass
        if self._mjpeg is not None:                       # which view the <img> is streaming → overlay mapping
            head["view"] = self._mjpeg.current_view()
        if self._rec is not None:                         # {recording,path,elapsed,written,dropped} or absent
            head["rec"] = self._rec.status()
        if self._transport is not None:                   # play/pause button syncs to the REAL state
            head["paused"] = bool(self._transport.paused)
        head_mask = self._mask_circle(fw, fh)             # circular mask overlay (its effect is invisible in BGR)
        if head_mask is not None:
            head["mask"] = head_mask
        return json.dumps(head)

    def _mask_circle(self, fw: int, fh: int) -> "list | None":
        """The circular mask as [cx, cy, r] normalized to the DISPLAYED (post-crop) frame, or None. The frame
        the browser shows is already cropped, so the raw-pixel mask centre is offset by the crop origin and
        divided by the cropped frame size — the same fraction the object overlay uses."""
        pp = self._preprocess
        c = getattr(pp, "cfg", None)
        if c is None or not fw or not fh or not getattr(c, "mask_enabled", False) or c.mask_radius <= 0:
            return None
        ox = c.crop_top_x if c.crop_enabled else 0
        oy = c.crop_top_y if c.crop_enabled else 0
        return [(c.mask_cx - ox) / fw, (c.mask_cy - oy) / fh, c.mask_radius / fw]

    # -- the HTTP handler ---------------------------------------------------- #

    def _make_handler(self):
        out = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            timeout = 30.0

            def log_message(self, *a: Any) -> None:
                pass

            def _cors(self) -> None:
                self.send_header("Access-Control-Allow-Origin", "*")

            def _json(self, code: int, payload: Any) -> None:
                body = json.dumps(payload).encode() if not isinstance(payload, bytes) else payload
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self._cors()
                self.end_headers()
                self.wfile.write(body)

            def do_OPTIONS(self) -> None:                    # noqa: N802
                self.send_response(204)
                self._cors()
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.end_headers()

            def do_POST(self) -> None:                       # noqa: N802
                if self.path.split("?", 1)[0] != "/osc":
                    self._json(404, {"error": "not found"})
                    return
                try:
                    n = int(self.headers.get("Content-Length", 0))
                    req = json.loads(self.rfile.read(n) or b"{}")
                    address = str(req["address"])
                    args = _decode_args(req.get("args", []))
                except Exception:                            # noqa: BLE001 — bad request body
                    self._json(400, {"error": "bad request"})
                    return
                try:
                    msgs = out.osc_roundtrip(address, args)
                except Exception as e:                       # noqa: BLE001
                    out._log.warning(f"[webgui] /osc {address} failed: {e}")
                    self._json(502, {"error": "bridge failure"})
                    return
                self._json(200, {"messages": msgs})

            def do_GET(self) -> None:                        # noqa: N802
                path = self.path.split("?", 1)[0]
                if path == "/events":
                    self._events()
                elif path == "/info":
                    self._json(200, {"prefix": out._prefix, "mjpeg_port": out._mjpeg_port})
                else:
                    self._static(path)

            def _static(self, path: str) -> None:
                rel = "index.html" if path in ("/", "") else path.lstrip("/")
                full = os.path.normpath(os.path.join(_STATIC, rel))
                if not full.startswith(_STATIC + os.sep) or not os.path.isfile(full):   # traversal / missing
                #   (+ os.sep: a bare prefix test would also admit a SIBLING dir whose name extends "static")
                    self.send_error(404)
                    return
                with open(full, "rb") as f:
                    body = f.read()
                self.send_response(200)
                self.send_header("Content-Type", _mime(full))
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(body)

            def _events(self) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self._cors()
                self.end_headers()
                try:
                    last_idx: Any = object()
                    while not out._stop.is_set():
                        snap = out._bus.wait_fresh(last_idx, 0.5) if out._bus is not None else None
                        idx = getattr(snap, "frame_idx", None) if snap is not None else None
                        if snap is None or idx == last_idx:
                            # No FRESH frame this window. wait_fresh returns the LATEST snapshot on
                            # timeout once one exists, so a PAUSED tracker lands here forever — writing
                            # NOTHING here would leave the SSE socket indefinitely silent (proxy
                            # idle-timeouts kill it, and a closed tab's handler thread is never detected,
                            # since spotting a broken pipe needs a write). Ping in BOTH quiet cases.
                            # a DATA event (not an SSE comment): a paused tracker emits no frames, so
                            # this quiet-window ping is the ONLY channel that can tell the browser the
                            # pause state (the play/pause button must follow oscin/mcu/another tab).
                            paused = (bool(out._transport.paused)
                                      if out._transport is not None else None)
                            self.wfile.write(b"data: " + json.dumps(
                                {"ping": 1, "paused": paused}).encode() + b"\n\n")
                            self.wfile.flush()
                            continue
                        last_idx = idx
                        self.wfile.write(b"data: " + out.snapshot_json(snap).encode() + b"\n\n")
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass

        return Handler


def _decode_args(raw: list) -> list:
    """Browser JSON args → Python OSC args. Accepts plain numbers/strings/bools, or typed {t,v}."""
    args = []
    for a in raw:
        if isinstance(a, dict) and "v" in a:
            t, v = a.get("t"), a["v"]
            args.append(int(v) if t == "i" else float(v) if t == "f" else str(v))
        elif isinstance(a, bool):
            args.append(1 if a else 0)                       # OSC has no bool
        else:
            args.append(a)
    return args
