#!/usr/bin/env python3
"""tuio2_monitor — a standalone TUIO 2.0 receiver + visualizer.

The localhost verification companion to the ``tuio2`` plugin (roadmap B.1): bind a
UDP port, decode the TUIO 2.0 bundles with ``lib.osc``, and draw what arrives — each
object's **id** (+ 4-state + confidence from the ``/rmt/obj`` sidecar), its position
and orientation, and its **high-resolution contour outline** (``/tuio2/ocg``).

It is **tracker-agnostic** — it taps nothing internal and speaks only OSC/TUIO, so it
works with any sender of these protocols; the natural "does the wire actually carry
what I think?" check.

    # terminal 1: run the tracker with OSC out
    venv/bin/python readymade_tracker.py --source video clip.mp4          # with [tuio2] in readymade.conf
    # terminal 2: watch it
    venv/bin/python tools/tuio2_monitor.py                   # binds 0.0.0.0:3333
    venv/bin/python tools/tuio2_monitor.py --port 3333 --text # headless: print, no window
    venv/bin/python tools/tuio2_monitor.py --record cap.jsonl # log every received message
    venv/bin/python tools/tuio2_monitor.py --record-osc       # raw replayable tape (osc_tuio2_<ts>.bin)
    venv/bin/python tools/tuio2_monitor.py --replay osc_tuio2_20260718_192205.bin   # play a tape back
    venv/bin/python tools/tuio2_monitor.py --show-clusters    # + the oscout cluster rings

``--show-clusters`` draws the cluster rings + ``C<id>`` from the tracker's custom **oscout** stream
(``/rmt/cluster/*``). oscout runs on its own port (default 3400) — the monitor binds a SECOND socket
for it automatically, so no reconfiguration is needed; use ``--osc-port`` if you moved it. If you
pointed ``[oscout]`` at the tuio2 port (``[oscout] port = 3333``), the one socket carries both.

Keys (window mode):  q / Esc quit · space freeze/unfreeze receiving · r record/stop
(JSONL of the received messages) · c clear the scene · s save a PNG snapshot (s + freeze
capture a moment).

State parsing (``TuioState``) and rendering (``render_frame``) are pure and unit-
tested against ``tuio2``'s own output — the encode→decode loop is closed in-process,
so the wire contract is verified with nothing running. The socket + cv2 window loop
is the only untested part (it needs a display)."""

from __future__ import annotations

import argparse
import json
import socket
import struct
import sys
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.osc import TimeTag, decode_packet  # noqa: E402


def _jsonable(args: list) -> list:
    """Render decoded OSC args JSON-safe for the --record log (TimeTag → seconds, blob → hex)."""
    out: list = []
    for a in args:
        if isinstance(a, TimeTag):
            out.append({"timetag": a.seconds})
        elif isinstance(a, (bytes, bytearray)):
            out.append({"blob": bytes(a).hex()})
        else:
            out.append(a)
    return out


# --------------------------------------------------------------------------- #
# State — pure: fold decoded TUIO 2.0 messages into a live object table
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# OSC tape — RAW datagram record + paced replay (--record-osc / --replay)
# --------------------------------------------------------------------------- #
_TAPE_MAGIC = b"OSCTAPE1"


def tape_open(path: str):
    """Open a new OSC tape for writing (magic header first)."""
    f = open(path, "wb")
    f.write(_TAPE_MAGIC)
    return f


def tape_append(f, t_rel: float, src: int, data: bytes) -> None:
    """One raw datagram: ``<f64 seconds-since-start><u8 source><u32 len><bytes>`` — source 0 = the
    tuio2 socket, 1 = the oscout socket (kept so a replay is a faithful merge of both streams)."""
    f.write(struct.pack("<dBI", t_rel, src, len(data)))
    f.write(data)


def tape_read(path: str):
    """Yield ``(t_rel, src, data)`` from a tape. Raises ValueError on a non-tape file."""
    with open(path, "rb") as f:
        if f.read(len(_TAPE_MAGIC)) != _TAPE_MAGIC:
            raise ValueError(f"{path}: not an OSC tape (bad magic)")
        while True:
            head = f.read(13)
            if len(head) < 13:
                return
            t_rel, src, n = struct.unpack("<dBI", head)
            yield t_rel, src, f.read(n)


def replay_tape(state: "TuioState", path: str, stop) -> threading.Thread:
    """Feed a tape into ``state`` at the RECORDED pacing on a daemon thread (``stop`` = Event)."""
    def run():
        t0 = time.perf_counter()
        try:
            for t_rel, src, data in tape_read(path):
                while not stop.is_set():
                    dt = (t0 + t_rel) - time.perf_counter()
                    if dt <= 0:
                        break
                    time.sleep(min(dt, 0.05))
                if stop.is_set():
                    return
                state.update_packet(data, now=time.perf_counter(), src=src)
            print("[tuio2] replay finished")
        except (OSError, ValueError) as e:
            print(f"[tuio2] replay failed: {e}", file=sys.stderr)
    th = threading.Thread(target=run, daemon=True, name="OscTapeReplay")
    th.start()
    return th


class TuioState:
    """The current TUIO scene, folded from received packets. ``objects`` maps
    session id → a dict with x, y, angle, contour (list of (x, y)), state, conf.

    Robust to ``tuio2``'s multi-packet frames: every packet repeats ``/frm`` +
    the full ``/alv``, and components (``/tok`` / ``/ocg`` / ``/rmt/obj``) are
    split across packets. We fold each message as it arrives and prune to the
    alive set on every ``/alv`` — so a frame reassembles regardless of how it was
    chunked, and an object drops the instant it leaves the alive set."""

    def __init__(self) -> None:
        self.objects: dict[int, dict[str, Any]] = {}
        self.clusters: dict[int, dict[str, Any]] = {}  # spatial clusters (/rmt/cluster table), keyed by cid
        self.frame_id: int | None = None
        self.src_dim: tuple[int, int] = (0, 0)
        self.source: str = ""
        self.frames: int = 0          # distinct /frm ids seen (for fps/HUD)
        self.packets: int = 0
        self.last_recv: float = 0.0
        self._rec = None              # optional JSONL recording of the received message stream
        self._tape = None             # optional RAW datagram tape (replayable; see tape_open/replay_tape)
        self._tape_t0 = 0.0

    @property
    def recording(self) -> bool:
        return self._rec is not None

    def start_recording(self, path: str) -> None:
        """Log every received packet's decoded messages to ``path`` as JSONL — one line
        per packet: ``{"recv", "t", "msgs":[[address, args], …]}`` — for offline inspection."""
        self.stop_recording()
        self._rec = open(path, "w")

    def stop_recording(self) -> None:
        if self._rec is not None:
            self._rec.close()
            self._rec = None

    @property
    def taping(self) -> bool:
        return self._tape is not None

    def start_tape(self, path: str) -> None:
        """Record every RAW datagram to ``path`` — a tape ``--replay`` plays back at recorded pacing."""
        self.stop_tape()
        self._tape = tape_open(path)
        self._tape_t0 = time.perf_counter()

    def stop_tape(self) -> None:
        if self._tape is not None:
            self._tape.close()
            self._tape = None

    def reset(self) -> None:
        """Clear the live scene (the 'c' key) — empty the object table + counters
        without rebinding the socket, so stale objects from an old run disappear."""
        self.objects.clear()
        self.clusters.clear()
        self.frame_id = None
        self.frames = 0
        self.packets = 0

    def update_packet(self, data: bytes, *, now: float | None = None, src: int = 0) -> None:
        """Fold one received UDP payload into the state. Fail-soft: a malformed /
        non-OSC packet (raw UDP is untrusted) is ignored, never raised."""
        self.packets += 1
        self.last_recv = now if now is not None else self.last_recv
        if self._tape is not None:                    # raw tape BEFORE decode — replay is byte-faithful
            tape_append(self._tape, (now or time.perf_counter()) - self._tape_t0, src, data)
        try:
            messages = decode_packet(data)
        except Exception:          # noqa: BLE001 — garbage on the port must not crash the monitor
            return
        if self._rec is not None:
            # `t` is monotonic (ordering, gaps); `wall` is UNIX time, the only clock that can be
            # compared with the tracker's own timetags — without it a recording can say what
            # arrived and in what order, but never how long anything took.
            self._rec.write(json.dumps({"recv": self.packets, "t": now, "wall": time.time(),
                                        "msgs": [[a, _jsonable(args)] for a, args in messages]}) + "\n")
        for address, args in messages:
            handler = _HANDLERS.get(address)
            if handler is not None:
                handler(self, args)
            elif (address.startswith("/rmt/") and address.count("/") == 2
                  and address != "/rmt/alv" and args and isinstance(args[0], (int, float))):
                # GENERIC oscout per-object fold: /rmt/<field> <id> <v…> (features, pose, efd, …) —
                # whatever the tracker annotates lands on the object for the features panel.
                self._o(args[0]).setdefault("features", {})[address[5:]] = list(args[1:])

    def _o(self, sid: int) -> dict[str, Any]:
        return self.objects.setdefault(int(sid), {})

    def _on_frm(self, args: list) -> None:
        if not args:
            return
        f_id = args[0]
        if f_id != self.frame_id:
            self.frames += 1
        self.frame_id = f_id
        # /frm  f_id  <time>  dim  source — the TIME field is one OSC 't' (the TUIO 2.0 spec form)
        # or TWO int32s under [tuio2] timetag_mode=ints, so `dim` is NOT at a fixed index. Locate
        # both from the END: source is the trailing string, dim the int before it. Indexing args[2]
        # silently read the microseconds as dim in ints mode — an int, so a type guard could not
        # catch it (it reported a 1024x1024 frame as 9x16297).
        if len(args) >= 2 and isinstance(args[-1], str):
            self.source = args[-1]
            if len(args) >= 3 and isinstance(args[-2], int):
                dim = args[-2]
                self.src_dim = ((dim >> 16) & 0xFFFF, dim & 0xFFFF)

    def _on_tok(self, args: list) -> None:
        if len(args) < 6:
            return
        sid, _tu, _cid, x, y, ang = args[:6]   # tu_id/c_id are 0: the tracker is markerless
        o = self._o(sid)
        o.update(x=float(x), y=float(y), angle=float(ang))

    def _on_ocg(self, args: list) -> None:
        if len(args) < 3:
            return
        sid = args[0]
        pts = args[1:]
        self._o(sid)["contour"] = [(float(pts[i]), float(pts[i + 1]))
                                   for i in range(0, len(pts) - 1, 2)]

    def _on_bnd(self, args: list) -> None:
        if len(args) < 7:
            return
        sid, x, y, ang, w, h, area = args[:7]
        self._o(sid)["bnd"] = (float(x), float(y), float(ang), float(w), float(h), float(area))

    def _on_obj(self, args: list) -> None:
        if len(args) < 3:
            return
        sid, state, conf = args[:3]
        self._o(sid).update(state=str(state), conf=float(conf))

    def _on_osc_frm(self, args: list) -> None:
        """oscout's frame header (``/rmt/frm``) — a NEW derived-data frame. Clear the previous
        cluster table so a value the tracker stopped sending doesn't linger stale; the
        rest of this bundle re-folds the current set. (A compound super-object is an ORDINARY
        object on this wire — it needs no table of its own.)"""
        self.clusters.clear()

    def _c(self, cid: Any) -> dict[str, Any]:
        return self.clusters.setdefault(int(cid), {})

    def _on_cluster(self, args: list) -> None:       # /rmt/cluster <cid> <n> <cx> <cy>
        if len(args) < 4:
            return
        cid, n, cx, cy = args[:4]
        self._c(cid).update(n=int(n), cx=float(cx), cy=float(cy))

    def _on_cluster_members(self, args: list) -> None:   # /rmt/cluster/members <cid> <oid…>
        if not args:
            return
        self._c(args[0])["members"] = [int(m) for m in args[1:]]

    def _on_cluster_bbox(self, args: list) -> None:  # /rmt/cluster/bbox <cid> <x> <y> <w> <h>
        if len(args) < 5:
            return
        self._c(args[0])["bbox"] = tuple(float(v) for v in args[1:5])

    def _on_cluster_circle(self, args: list) -> None:  # /rmt/cluster/circle <cid> <cx> <cy> <r>
        if len(args) < 4:
            return
        self._c(args[0])["circle"] = (float(args[1]), float(args[2]), float(args[3]))

    def _on_cluster_hull(self, args: list) -> None:  # /rmt/cluster/hull <cid> <x> <y> …
        if not args:
            return
        pts = args[1:]
        self._c(args[0])["hull"] = [(float(pts[i]), float(pts[i + 1])) for i in range(0, len(pts) - 1, 2)]

    def _on_cluster_alv(self, args: list) -> None:   # /rmt/cluster/alv <cid…>
        alive = {int(c) for c in args}
        for cid in list(self.clusters):
            if cid not in alive:
                del self.clusters[cid]

    def _on_alv(self, args: list) -> None:
        alive = {int(s) for s in args}
        for sid in list(self.objects):
            if sid not in alive:
                del self.objects[sid]


_HANDLERS = {
    "/tuio2/frm": TuioState._on_frm,
    "/tuio2/tok": TuioState._on_tok,
    "/tuio2/ocg": TuioState._on_ocg,
    "/tuio2/bnd": TuioState._on_bnd,
    "/tuio2/alv": TuioState._on_alv,
    "/rmt/obj": TuioState._on_obj,
    "/rmt/frm": TuioState._on_osc_frm,       # oscout header → clear the previous derived-data frame
    "/rmt/cluster": TuioState._on_cluster,   # spatial-cluster table (summary + geometry, keyed by cid)
    "/rmt/cluster/members": TuioState._on_cluster_members,
    "/rmt/cluster/bbox": TuioState._on_cluster_bbox,
    "/rmt/cluster/circle": TuioState._on_cluster_circle,
    "/rmt/cluster/hull": TuioState._on_cluster_hull,
    "/rmt/cluster/alv": TuioState._on_cluster_alv,
}


# --------------------------------------------------------------------------- #
# Rendering — pure (draws to an ndarray; no window). cv2 + numpy only.
# --------------------------------------------------------------------------- #

# state → BGR, matching the tracker's own view colours so the two read alike.
_STATE_COLOR = {"RESTING": (0, 220, 0), "MOVED": (0, 140, 255), "SETTLING": (0, 220, 220),
                "HIDDEN": (160, 160, 160)}
_DEFAULT_COLOR = (200, 200, 200)
# per-cluster ring colours (BGR), cycled by cluster id — matches the tracker gui.
_CLUSTER_COLORS = [(220, 200, 80), (80, 200, 220), (200, 120, 220),
                   (120, 220, 140), (220, 140, 120), (160, 220, 220)]


def _draw_cluster_rings(cv2, np, img, state, width, height, font_scale) -> None:
    """Cluster circles (the min-enclosing circle) + a ``C<id>`` tag, one per cluster in the
    ``/rmt/cluster`` table. The tracker sends the geometry NORMALIZED (cx/w, cy/h, r/w — always,
    independent of pose coord_mode)."""
    for cid, c in state.clusters.items():
        circle = c.get("circle")
        if not circle:
            continue
        cx, cy, r = (int(round(circle[0] * width)), int(round(circle[1] * height)),
                     max(1, int(round(circle[2] * width))))
        col = _CLUSTER_COLORS[(cid - 1) % len(_CLUSTER_COLORS)]
        cv2.circle(img, (cx, cy), r, col, 2, cv2.LINE_AA)
        _id_chip(cv2, img, f"C{cid}", cx - 12, max(cy - r - 6, 16), col, font_scale)


def canvas_size(src_dim: tuple[int, int], size: tuple[int, int] | None, max_size: int) -> tuple[int, int]:
    """The render canvas: an explicit ``--size`` if given, else the SOURCE aspect ratio scaled
    to fit ``max_size`` (so a source that isn't the window aspect -- e.g. a square 1024x1024 --
    isn't squished)."""
    if size:
        return int(size[0]), int(size[1])
    sw, sh = src_dim
    if sw > 0 and sh > 0:
        s = min(max_size / sw, max_size / sh)
        return max(1, int(round(sw * s))), max(1, int(round(sh * s)))
    return max_size, int(max_size * 2 / 3)      # no frame dim yet → provisional 3:2


def _resolve_mode(state: "TuioState", coord_mode: str) -> str:
    if coord_mode in ("normalized", "pixel"):
        return coord_mode
    # auto: normalized coords live in [0,1]; anything bigger is pixel.
    mx = 0.0
    for o in state.objects.values():
        if "x" in o:
            mx = max(mx, abs(o["x"]), abs(o["y"]))
    return "pixel" if mx > 2.0 else "normalized"


def render_frame(state: "TuioState", width: int = 900, height: int = 600,
                 coord_mode: str = "auto", font_scale: float = 1.0,
                 background: Any = None, outline_width: int = 2,
                 show_clusters: bool = False, active: "bool | None" = None) -> Any:
    """Draw the current TUIO scene onto a BGR canvas: per object a state-coloured
    contour + centroid + orientation tick + a filled ID CHIP (id in black, like the
    tracker's GUI), plus a HUD. Pure — returns the image. The state→colour key is
    printed to the console at startup (``_print_legend``), not drawn on the canvas.
    ``font_scale`` (the ``--font-size`` knob) multiplies every label's size + weight
    and the chip metrics laid out around them. ``outline_width`` (``--outline-width``) is
    the contour line thickness in px (default 2; ≥1). ``background`` (the ``--mjpeg-stream``
    video frame) is resized to the canvas + drawn UNDER the overlay when present, else
    a flat dark canvas — the TUIO overlay aligns because coords are normalized to the
    same frame the tracker streamed."""
    import cv2
    import numpy as np

    if background is not None:
        img = cv2.resize(background, (width, height))     # a fresh array — safe to draw on
    else:
        img = np.full((height, width, 3), 28, np.uint8)
    mode = _resolve_mode(state, coord_mode)
    sw, sh = state.src_dim

    def to_px(x: float, y: float) -> tuple[int, int]:
        if mode == "pixel":
            fx = x / sw if sw > 0 else x / max(width, 1)
            fy = y / sh if sh > 0 else y / max(height, 1)
        else:
            fx, fy = x, y
        return int(round(fx * width)), int(round(fy * height))

    if show_clusters:                                       # cluster rings BEHIND the objects
        _draw_cluster_rings(cv2, np, img, state, width, height, font_scale)
    for sid, o in sorted(state.objects.items()):
        col = _STATE_COLOR.get(o.get("state"), _DEFAULT_COLOR)   # colour BY STATE (see legend)
        if o.get("contour"):
            pts = np.array([to_px(px, py) for px, py in o["contour"]], np.int32)
            cv2.polylines(img, [pts], True, col, max(1, int(outline_width)), cv2.LINE_AA)
        if "x" in o:
            cx, cy = to_px(o["x"], o["y"])
            cv2.circle(img, (cx, cy), 4, col, -1, cv2.LINE_AA)
            ang = o.get("angle", 0.0)                  # orientation ARROW: 0 = UP, clockwise-positive
            cv2.arrowedLine(img, (cx, cy),             #   (the TUIO/fiducial reading in the y-down frame)
                            (cx + int(26 * np.sin(ang)), cy - int(26 * np.cos(ang))),
                            col, 2, cv2.LINE_AA, tipLength=0.3)
            _id_chip(cv2, img, str(sid), cx + 8, cy - 8, col, font_scale)

    hud = (f"objs {len(state.objects)}   frame {state.frame_id}   "
           f"res {state.src_dim[0]}x{state.src_dim[1]}   coords {mode}   "
           f"pkts {state.packets}   source {state.source or '-'}")
    _text(cv2, img, hud, (10, int(round(22 * font_scale))), (210, 210, 210), 0.5 * font_scale)
    if active is not None:                                  # data-flow dot, top-right: green = packets
        col = (60, 220, 60) if active else (50, 50, 230)    # arriving, red = the stream is quiet
        cv2.circle(img, (width - 16, 16), max(5, int(round(6 * font_scale))), col, -1, cv2.LINE_AA)
    if not state.objects:
        _text(cv2, img, "waiting for TUIO... (is [tuio2] enabled + the tracker running?)",
              (10, height // 2), (120, 120, 120), 0.5 * font_scale)
    return img


def _id_chip(cv2, img, txt, x, y, col, font_scale: float = 1.0) -> None:
    """A filled state-coloured chip with the id in BLACK (legible on any background) — sized
    to the text and kept on-canvas. Mirrors the tracker GUI's id tag."""
    f, sc, th = cv2.FONT_HERSHEY_SIMPLEX, 0.55 * font_scale, max(1, int(round(font_scale)))
    (tw, tht), base = cv2.getTextSize(txt, f, sc, th)
    x = max(3, min(int(x), img.shape[1] - tw - 5))
    y = max(tht + 5, int(y))
    cv2.rectangle(img, (x - 3, y - tht - 4), (x + tw + 3, y + base + 1), col, -1)
    cv2.putText(img, txt, (x, y), f, sc, (0, 0, 0), th, cv2.LINE_AA)


def _print_legend() -> None:
    """Print the state→colour key to the console (the on-canvas legend was removed).
    A truecolor swatch per state when stdout is a TTY, plain names otherwise — the
    colours match the drawn contours/chips (and the tracker GUI)."""
    tty = sys.stdout.isatty()
    print("[tuio2] state colours:")
    for st, (b, g, r) in _STATE_COLOR.items():
        swatch = f"\x1b[48;2;{r};{g};{b}m  \x1b[0m " if tty else ""
        print(f"          {swatch}{st}")


def _text(cv2, img, s, org, col, scale: float = 0.5) -> None:
    """Crisp label: a 1px black drop-shadow (not a fat outline, which smears small
    glyphs) under the fill."""
    x, y = int(org[0]), int(org[1])
    cv2.putText(img, s, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, col, 1, cv2.LINE_AA)


# --------------------------------------------------------------------------- #
# MJPEG background — optional video underlay from the tracker's --mjpeg-stream
# --------------------------------------------------------------------------- #

_SOI = b"\xff\xd8"          # JPEG start-of-image marker
_EOI = b"\xff\xd9"          # JPEG end-of-image marker


def _extract_jpegs(buf: bytes) -> "tuple[list[bytes], bytes]":
    """Pull every COMPLETE JPEG (SOI..EOI) out of a raw MJPEG byte buffer; return the
    frames + the trailing remainder to carry into the next read. Boundary-agnostic — it
    scans the JPEG markers, so it reads any ``multipart/x-mixed-replace`` server. Pure
    (unit-tested); a partial trailing JPEG is preserved in the remainder."""
    frames: list[bytes] = []
    i = 0
    while True:
        a = buf.find(_SOI, i)
        if a < 0:
            break
        b = buf.find(_EOI, a + 2)
        if b < 0:                      # SOI with no EOI yet — keep from here as the remainder
            return frames, buf[a:]
        frames.append(buf[a:b + 2])
        i = b + 2
    return frames, buf[i:]


class MjpegClient:
    """Background MJPEG-over-HTTP reader: GET the stream, decode the latest JPEG to a BGR
    frame the monitor draws UNDER the TUIO overlay (the ``--mjpeg-stream`` video). Dep-free
    (urllib + cv2), tracker-agnostic, and auto-reconnecting — ``latest()`` returns the most
    recent frame or ``None`` (before the first frame / while disconnected)."""

    def __init__(self, url: str) -> None:
        self._url = url
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._frame: Any = None
        self._thread: threading.Thread | None = None
        self.frames = 0

    def start(self) -> "MjpegClient":
        self._thread = threading.Thread(target=self._run, name="mjpeg-client", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def latest(self) -> Any:
        with self._lock:
            return self._frame

    def _run(self) -> None:  # pragma: no cover — network loop (the parse is tested via _extract_jpegs)
        import urllib.request

        import cv2
        import numpy as np
        while not self._stop.is_set():
            try:
                with urllib.request.urlopen(self._url, timeout=3.0) as resp:  # timeout guards read too
                    buf = b""
                    while not self._stop.is_set():
                        chunk = resp.read1(65536)      # read1: return what's buffered (one read) —
                        if not chunk:                  #   read(n) would BLOCK for exactly n bytes on
                            break                      #   an unbounded stream. server closed → reconnect
                        frames, buf = _extract_jpegs(buf + chunk)
                        if frames:
                            img = cv2.imdecode(np.frombuffer(frames[-1], np.uint8), cv2.IMREAD_COLOR)
                            if img is not None:
                                with self._lock:
                                    self._frame = img
                                self.frames += 1
                        if len(buf) > 8_000_000:       # runaway guard (never found a frame boundary)
                            buf = b""
            except Exception:                          # noqa: BLE001 — connection issues → retry
                pass
            if not self._stop.is_set():
                self._stop.wait(1.0)                   # reconnect after a beat


# --------------------------------------------------------------------------- #
# Receiver loop
# --------------------------------------------------------------------------- #

def _open_socket(host: str, port: int) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    sock.setblocking(False)
    return sock


def _drain(sock: "socket.socket | None", state: TuioState, now: float, src: int = 0) -> None:
    """Fold every packet currently buffered on the socket (non-blocking). None = replay mode
    (a tape feeder thread folds packets instead — nothing to drain)."""
    if sock is None:
        return
    while True:
        try:
            data, _addr = sock.recvfrom(65535)
        except (BlockingIOError, OSError):
            return                          # nothing more buffered (or socket closed)
        state.update_packet(data, now=now, src=src)  # fail-soft: ignores a malformed packet


def _run_text(sock: socket.socket, state: TuioState,
              osc_sock: "socket.socket | None" = None) -> int:  # pragma: no cover
    print(f"[tuio2] listening (text mode). Ctrl-C to stop.")
    last = 0.0
    try:
        while True:
            now = time.perf_counter()
            _drain(sock, state, now)
            if osc_sock is not None:
                _drain(osc_sock, state, now, src=1)
            if now - last >= 1.0:                 # one summary line per second
                last = now
                ids = sorted(state.objects)
                det = ", ".join(
                    f"{i}:{state.objects[i].get('state', '?')}"
                    f"({len(state.objects[i].get('contour', []))}pts)" for i in ids)
                print(f"[tuio2] frame {state.frame_id} src {state.src_dim[0]}x{state.src_dim[1]} "
                      f"objs {len(ids)} [{det}]  pkts {state.packets}")
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\n[tuio2] interrupted - bye")
        return 0


def snapshot_path(directory: str, when: str | None = None) -> str:
    """A timestamped PNG path for the 's' snapshot, under ``directory``."""
    stamp = when if when is not None else time.strftime("%Y%m%d-%H%M%S")
    return str(Path(directory) / f"tuio2_{stamp}.png")


def record_path(directory: str, when: str | None = None) -> str:
    """A timestamped JSONL path for the 'r' record toggle, under ``directory``."""
    stamp = when if when is not None else time.strftime("%Y%m%d-%H%M%S")
    return str(Path(directory) / f"tuio2_{stamp}.jsonl")


def _run_window(sock: socket.socket, state: TuioState, args, mjpeg: "MjpegClient | None" = None,
                osc_sock: "socket.socket | None" = None) -> int:  # pragma: no cover
    import cv2

    win = "TUIO 2.0 monitor - q quit | space freeze | r record | c clear | s snapshot"
    try:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)   # keep aspect on resize
    except cv2.error as e:
        print(f"[tuio2] no display ({e}); falling back to --text", file=sys.stderr)
        return _run_text(sock, state)
    print(f"[tuio2] listening on {args.host}:{args.port}. "
          f"q/Esc quit | space freeze | r record | c clear | s snapshot.")
    frozen = False
    last_wh = None
    try:
        return _window_loop(sock, state, args, mjpeg, osc_sock, cv2, win, frozen, last_wh)
    except KeyboardInterrupt:                     # Ctrl-C in the terminal → exit as cleanly as 'q'
        print("\n[tuio2] interrupted - bye")
        return 0
    finally:
        cv2.destroyAllWindows()


def _window_loop(sock, state, args, mjpeg, osc_sock, cv2, win, frozen, last_wh) -> int:  # pragma: no cover
    while True:
        if not frozen:                            # freeze = STOP receiving so the view holds
            now = time.perf_counter()
            _drain(sock, state, now)                    # (buffered UDP is dropped while frozen;
            if osc_sock is not None:                    # the oscout stream (clusters), if separate port
                _drain(osc_sock, state, now, src=1)
        bg = mjpeg.latest() if mjpeg is not None else None      # the video underlay (or None)
        # size the window to the TUIO source aspect; before any /frm arrives, fall back to the
        # MJPEG frame's aspect so the video isn't stretched while we wait for TUIO.
        src = state.src_dim if state.src_dim != (0, 0) else \
            ((bg.shape[1], bg.shape[0]) if bg is not None else (0, 0))
        cw, ch = canvas_size(src, args.size, args.max_size)
        if (cw, ch) != last_wh:                   # snap the window to the (new) source aspect
            cv2.resizeWindow(win, cw, ch)
            last_wh = (cw, ch)
        fs = args.font_size
        active = state.packets > 0 and (time.perf_counter() - state.last_recv) < 1.0
        img = render_frame(state, cw, ch, args.coord_mode, fs, background=bg,
                           outline_width=args.outline_width, show_clusters=args.show_clusters,
                           active=active)
        if mjpeg is not None and bg is None:      # enabled but nothing decoded yet
            _text(cv2, img, "waiting for MJPEG video...", (10, ch - int(round(40 * fs))),
                  (120, 120, 120), 0.5 * fs)
        if state.recording:
            _text(cv2, img, "REC", (cw - int(round(60 * fs)), int(round(40 * fs))),
                  (60, 60, 255), 0.6 * fs)
        if frozen:
            _text(cv2, img, "FROZEN  (space to resume, s to snapshot)",
                  (10, ch - 14), (90, 200, 255), 0.6 * fs)
        cv2.imshow(win, img)
        key = cv2.waitKey(15) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord(" "):
            frozen = not frozen
            print(f"[tuio2] {'frozen' if frozen else 'resumed'}")
        elif key == ord("r"):
            if state.recording:
                state.stop_recording()
                print("[tuio2] recording stopped")
            else:
                path = args.record or record_path(args.snapshot_dir)
                state.start_recording(path)
                print(f"[tuio2] recording -> {path}")
        elif key == ord("c"):
            state.reset()
            print("[tuio2] cleared")
        elif key == ord("s"):
            path = snapshot_path(args.snapshot_dir)
            if cv2.imwrite(path, img):
                print(f"[tuio2] snapshot -> {path}")
            else:
                print(f"[tuio2] snapshot failed (cannot write {path})", file=sys.stderr)
    return 0


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="tuio2_monitor",
                                description="Receive + visualize TUIO 2.0 (roadmap B.1 verification)")
    p.add_argument("--host", default="0.0.0.0", help="bind address (default 0.0.0.0)")
    p.add_argument("--port", type=int, default=3333, help="UDP port (default 3333, the tuio2 default)")
    p.add_argument("--size", type=int, nargs=2, metavar=("W", "H"), default=None,
                   help="fixed canvas W H (default: match the source aspect ratio, fit to --max-size)")
    p.add_argument("--max-size", dest="max_size", type=int, default=900,
                   help="max canvas dimension when auto-sizing to the source aspect (default 900)")
    p.add_argument("--coord-mode", dest="coord_mode", default="auto",
                   choices=("auto", "normalized", "pixel"),
                   help="how to interpret coords (default auto-detect)")
    p.add_argument("--font-size", dest="font_size", type=float, default=1.0,
                   help="label/HUD font scale multiplier (default 1.0; try 1.5 or 2 for a big window)")
    p.add_argument("--outline-width", dest="outline_width", type=int, default=2, metavar="PX",
                   help="object contour line thickness in px (default 2; try 1 for thinner outlines)")
    p.add_argument("--show-clusters", dest="show_clusters", action="store_true",
                   help="draw cluster rings + C<id> from the tracker's [oscout] stream (see --osc-port)")
    p.add_argument("--osc-port", dest="osc_port", type=int, default=3400, metavar="PORT",
                   help="UDP port of the tracker's [oscout] stream for --show-clusters (default 3400, "
                        "the oscout default; ignored if it equals --port, i.e. oscout shares [tuio2]'s port)")
    p.add_argument("--snapshot-dir", dest="snapshot_dir", default=".",
                   help="where the 's' key writes tuio2_<timestamp>.png (default: cwd)")
    p.add_argument("--record", metavar="FILE",
                   help="log every received TUIO message to FILE as JSONL (for inspection)")
    p.add_argument("--record-osc", dest="record_osc", nargs="?", const="", metavar="FILE",
                   help="record the RAW OSC datagrams to a replayable tape "
                        "(default name osc_tuio2_<timestamp>.bin)")
    p.add_argument("--replay", metavar="FILE",
                   help="replay a --record-osc tape at its recorded pacing instead of listening "
                        "on the network (no sockets are bound)")
    p.add_argument("--text", action="store_true", help="headless: print a summary line/second, no window")
    p.add_argument("--mjpeg-stream", dest="mjpeg_stream", type=int, metavar="PORT",
                   help="also show the tracker's MJPEG video (http://<mjpeg-host>:PORT/) UNDER the "
                        "TUIO overlay - the port in the tracker's [mjpeg] section")
    p.add_argument("--mjpeg-host", dest="mjpeg_host", default="127.0.0.1", metavar="HOST",
                   help="host of the MJPEG stream (default 127.0.0.1; the tracker machine)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    args = _parse_args(list(sys.argv[1:] if argv is None else argv))
    sock = None
    if args.replay is None:
        try:
            sock = _open_socket(args.host, args.port)
        except OSError as e:
            print(f"[tuio2] cannot bind {args.host}:{args.port}: {e}", file=sys.stderr)
            return 3
    # --show-clusters needs the tracker's [oscout] stream. If oscout runs on its own port
    # (the default 3400, ≠ tuio2's 3333), bind a SECOND socket so cluster data arrives without
    # asking the user to reconfigure oscout onto the tuio2 port. Same port → one socket suffices.
    osc_sock = None
    if args.replay is None and args.show_clusters and args.osc_port != args.port:
        try:
            osc_sock = _open_socket(args.host, args.osc_port)
            print(f"[tuio2] cluster stream <- {args.host}:{args.osc_port} (oscout)")
        except OSError as e:
            print(f"[tuio2] cannot bind oscout port {args.osc_port}: {e} "
                  f"(clusters won't show; is another monitor already bound?)", file=sys.stderr)
    _print_legend()
    state = TuioState()
    stop_replay = threading.Event()
    if args.replay is not None:
        print(f"[tuio2] replaying {args.replay} (recorded pacing)")
        replay_tape(state, args.replay, stop_replay)
    if args.record_osc is not None:
        path = args.record_osc or time.strftime("osc_tuio2_%Y%m%d_%H%M%S.bin")
        state.start_tape(path)
        print(f"[tuio2] recording raw OSC -> {path}")
    if args.record:
        state.start_recording(args.record)
        print(f"[tuio2] recording messages -> {args.record}")
    mjpeg = None
    if args.mjpeg_stream is not None and not args.text:      # video underlay (window mode only)
        url = f"http://{args.mjpeg_host}:{args.mjpeg_stream}/"
        print(f"[tuio2] MJPEG underlay <- {url}")
        mjpeg = MjpegClient(url).start()
    try:
        return _run_text(sock, state, osc_sock) if args.text \
            else _run_window(sock, state, args, mjpeg, osc_sock)
    finally:
        stop_replay.set()
        if mjpeg is not None:
            mjpeg.stop()
        if osc_sock is not None:
            osc_sock.close()
        state.stop_recording()
        state.stop_tape()
        if sock is not None:
            sock.close()


if __name__ == "__main__":
    raise SystemExit(main())
