"""``oscout`` OUTPUT plugin — the ONE smart custom-OSC egress.

Not a sender per producer: a single output that **generically serializes whatever derived data
is on the snapshot** — the per-object ``annotations`` a stage wrote (``features``, ``efd``, …) plus
core ``pose``, and the scene-level ``clusters`` table — over UDP OSC, reusing the
dep-free ``lib.osc`` codec. A compound (the Grouper's fused emit view, over budget) IS an object
here: it replaces its members and rides the same per-object addresses.

"Smart" = generic + config-driven, so producers never touch it:
  * **Generic reflection** — it walks ``obj.annotations`` (``{producer: {field: [floats]}}``) and
    emits each field; a new ``@feature`` or a brand-new producer stage just *appears* on the wire.
    (The ``clusters`` table spans several objects, so it gets its own small emitter
    — ``_cluster_msgs`` — instead of riding the per-object reflection.)
  * **Self-describing address-per-field** — ``/<prefix>/<producer>/<field> <id> <v…>`` (e.g.
    ``/rmt/features/circularity 5 0.9``), so a Max/MSP / SuperCollider patch routes a value
    straight to a parameter by its address. A ``/<prefix>/frm`` header + ``/<prefix>/alv`` alive-set
    bound the frame; ``/<prefix>/pose <id> <x> <y> <θ>`` carries pose.
  * **Config-driven** (``[oscout]``): ``send`` (which producers, or ``all``), ``prefix``,
    ``coord_mode`` (pose normalized/pixel), ``max_fps``.

Coexists with ``tuio2`` (the STANDARD protocol) + ``mjpeg`` (video): three independent outputs, one
annotated snapshot. ``setup`` → ``start(bus)`` → ``stop``. The build path is socket-injectable
(``send_fn``) so it's unit-tested on localhost with no network.
"""

from __future__ import annotations

import errno
import socket
import threading
import time
from typing import Any, Callable, Mapping

# BENCH: reversible — settle→OSC end-to-end wall stamps (pairs with lib/loop.py's capture stamp).
# OFF by default; enabled only when RMT_BENCH_STAMPS names a file.
import os as _os
_bp = _os.environ.get("RMT_BENCH_STAMPS")
_BENCH_STAMPS = open(_bp, "a", buffering=1) if _bp else None

from lib.geometry import place_footprint, simplify_contour
from lib.logger import NULL_LOGGER
from lib.loop import consume_bus
from lib.osc import parse_destinations, TimeTag, encode_bundle, encode_message
from lib.params import ParamSpec, bind_params


# A UDP datagram can't exceed the OS limit (macOS net.inet.udp.maxdgram defaults to 9216B; a bigger
# sendto fails with EMSGSIZE / [Errno 40]). A frame with many objects × fat fields (e.g. efd_bins=64)
# blows past it, so build_packets SPLITS the frame across datagrams, each kept under this budget.
_MAX_DATAGRAM = 8192
_BUNDLE_OVERHEAD = 16                                   # "#bundle\0" (8) + timetag (8)
_ELEM_PREFIX = 4                                        # each bundle element carries a 4-byte size prefix


def _as_list(v: Any) -> list:
    """Coerce an annotation value to a list of numbers (a scalar → 1-elem)."""
    if isinstance(v, (list, tuple)):
        return [float(x) for x in v]
    return [float(v)]


def _as_fields(value: Any):
    """Normalize an annotation value to ``(field, [floats])`` pairs. The convention is
    ``{field: [floats]}`` (what the features stage emits); a bare list/scalar → field ``"v"``."""
    if isinstance(value, Mapping):
        for k, v in value.items():
            yield str(k), _as_list(v)
    else:
        yield "v", _as_list(value)


class OscOut:
    """The smart custom-OSC output. ``setup`` → ``start(bus)`` → ``stop``."""

    # -- live parameters, declared WITH the plugin. host/port bind the socket at setup → needs-restart.
    #    prefix (/-normalized) + coord_mode (→_pixel bool) via apply setters. `send` derives FOUR flags
    #    (_send_all/_send_pose/_send_contour/_send_keys), so it's bind=False and setup parses it itself. --
    PARAMS = (
        ParamSpec("destinations", str, "", attr="_dest_spec",
                  apply=lambda self, v: (setattr(self, "_dest_spec", str(v)),
                                         setattr(self, "_dests", parse_destinations(v, self._host, self._port))),
                  doc="THE destination setting, LIVE: comma list of host:port targets - EVERY packet goes to each (several clients watch one stream); add/remove on the fly; empty = 127.0.0.1:3400 (the legacy host/port pair)"),
        ParamSpec("prefix", str, "/rmt",
                  apply=lambda self, v: setattr(self, "_prefix", "/" + str(v).strip("/")), doc="OSC address prefix"),
        ParamSpec("send", str, "all", hot=False, bind=False, multi=True, expose="edit",
                  enum=("all", "pose", "contour", "features", "cluster", "obj"),
                  doc="What goes on the wire, comma list: 'all' = every producer annotation + the cluster table + pose - 'features'/'cluster' = just that producer/table - 'pose' = the /pose message - 'contour' = the per-object outline (bulky, NEVER part of 'all' - opt in explicitly)"),
        ParamSpec("coord_mode", str, "normalized", enum=("normalized", "pixel"),
                  apply=lambda self, v: setattr(self, "_pixel", str(v).lower() == "pixel"),
                  doc="Pose x,y as [0,1] frame-normalized or raw pixels"),
        ParamSpec("max_fps", float, 0.0, lo=0.0, hot=False, doc="Throttle send rate in fps (0 = every fresh frame)"),
    )

    def __init__(self) -> None:
        self._log = NULL_LOGGER
        self._prof: Any = None
        self._bus: Any = None
        self._sock: Any = None
        self._owns_socket = False
        self._send: Callable[[bytes], Any] | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._tt_base: float | None = None   # wall-clock anchor for bundle timetags (first frame)
        # config (filled in setup)
        self._host = "127.0.0.1"           # the empty-`destinations` fallback (constant, not a knob):
        self._port = 3400                  #   an empty destinations spec sends to this single default
        self._prefix = "/rmt"
        self._pixel = False
        self._send_all = True
        self._send_keys: set[str] = set()
        self._send_pose = True
        self._max_fps = 0.0

    # -- lifecycle ----------------------------------------------------------- #

    def setup(self, config: Any, *, logger: Any = None,
              profiler: Any = None, send_fn: Callable[[bytes], Any] | None = None) -> None:
        self._log = logger or NULL_LOGGER
        self._prof = profiler
        g = config.get if config is not None else (lambda k, d=None: d)
        bind_params(self, config)          # _dest_spec / _prefix / _pixel / _max_fps ← PARAMS
        toks = [s.strip().lower() for s in str(g("send", "all")).split(",") if s.strip()]
        self._send_all = ("all" in toks) or not toks
        self._send_pose = self._send_all or ("pose" in toks)
        self._send_contour = "contour" in toks          # opt-in ONLY (bulky) — not part of 'all'
        self._send_keys = {t for t in toks if t not in ("all", "pose", "contour")}
        if send_fn is not None:
            self._send = send_fn                          # tests inject a recorder; no socket
            self._owns_socket = False
        else:
            self._owns_socket = True
            self._open_socket()
            self._send = self._send_packet
            #             ^ reads self._dests at CALL time — a live `destinations` edit re-aims mid-run
        self._drops = 0                    # ENOBUFS drops since the last throttled report
        self._drop_logged = 0.0

    def _open_socket(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:                               # absorb egress bursts: macOS answers a FULL interface
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 << 20)   # queue (typically
        except OSError:                    # the Wi-Fi hop) with ENOBUFS instead of blocking — a
            pass                           # bigger send buffer rides the spikes out

    def _send_packet(self, data: bytes) -> None:
        """One datagram to EVERY destination, failures isolated per dest (a dead Wi-Fi peer must
        not starve localhost). ENOBUFS gets one 1 ms retry — this runs on the sender thread, the
        pipeline never waits — then counts into a THROTTLED warning (one line per 5 s, not one
        per drop: the wire re-states everything next frame, so a drop is a 1-frame hole)."""
        for d in self._dests:
            try:
                self._sock.sendto(data, d)
            except OSError as e:
                if getattr(e, "errno", None) == errno.ENOBUFS:
                    time.sleep(0.001)
                    try:
                        self._sock.sendto(data, d)
                        continue
                    except OSError as e2:
                        e = e2
                self._drops += 1
                now = time.monotonic()
                if now - self._drop_logged >= 5.0:
                    self._log.warning(f"[oscout] send failed ({len(data)}B to {d[0]}:{d[1]}): {e} — "
                                      f"{self._drops} datagram(s) dropped since last report")
                    self._drops = 0
                    self._drop_logged = now

    def start(self, bus: Any) -> "OscOut":
        self._bus = bus
        self._stop.clear()
        if self._owns_socket and self._sock is None:      # RESTART after stop() — recreate the socket
            self._open_socket()
        self._thread = threading.Thread(target=self._run, name="oscout", daemon=True)
        self._thread.start()
        self._log.info(f"[oscout] OSC -> {', '.join(f'{h}:{p}' for h, p in self._dests)} prefix={self._prefix} "
                       f"send={'all' if self._send_all else sorted(self._send_keys) or ['pose']} "
                       f"coord={'pixel' if self._pixel else 'normalized'}")
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    # -- the pure build path (unit-tested; no socket) ------------------------ #

    def _dims(self, snap: Any) -> tuple[int, int]:
        fr = getattr(snap, "frame", None)
        if fr is not None and getattr(fr, "shape", None):
            return int(fr.shape[1]), int(fr.shape[0])
        return 0, 0

    def _norm(self, x: float, y: float, fw: int, fh: int) -> tuple[float, float]:
        if self._pixel or fw <= 0 or fh <= 0:
            return float(x), float(y)
        return float(x) / fw, float(y) / fh

    def _object_msgs(self, o: Any, fw: int, fh: int) -> list:
        """Every OSC message this object contributes — one per (annotation field) + pose."""
        out: list = []
        oid = int(o.id)
        for producer, value in (getattr(o, "annotations", None) or {}).items():
            if not (self._send_all or producer in self._send_keys):
                continue
            for field, vals in _as_fields(value):
                out.append(encode_message(f"{self._prefix}/{producer}/{field}", [oid, *vals]))
        if self._send_pose and getattr(o, "pose", None) is not None:
            x, y, _rel = o.pose                         # the per-settle θ snaps to ~0 on every re-bake —
            #                                             report the ACCUMULATED orientation instead
            th = float(getattr(o, "theta_cumulative", 0.0) or 0.0)
            nx, ny = self._norm(x, y, fw, fh)
            out.append(encode_message(f"{self._prefix}/pose", [oid, nx, ny, float(th)]))
        if self._send_all or "obj" in self._send_keys:  # the settle lifecycle + pose confidence — the
            st = getattr(o, "status", None)             #   native stream is COMPLETE without tuio2
            out.append(encode_message(f"{self._prefix}/obj",
                                      [oid, st.name if st is not None else "RESTING",
                                       float(getattr(o, "pose_conf", None) or 0.0)]))
        if self._send_contour:                          # the rest outline PLACED at the live pose (as the
            fpc, fc = getattr(o, "footprint_contour", None), getattr(o, "footprint_centroid", None)
            wc = getattr(o, "wire_contour", None)       # the PERCEPTUAL outline (inverse EFD after the
            placed = None                               #   detail cut) — already wire quality, no approxPolyDP
            if fc is not None and getattr(o, "pose", None) is not None:
                if wc is not None:
                    placed = place_footprint(wc, fc, o.pose)
                elif fpc is not None:                   # fallback (features stage off): the legacy simplify path
                    placed = simplify_contour(place_footprint(fpc, fc, o.pose), self._ceps)
            if placed is not None:
                flat: list = []
                for p in placed:                               # gui/tuio2/webgui — tracks, not frozen at rest
                    cx, cy = self._norm(float(p[0][0]), float(p[0][1]), fw, fh)
                    flat += [cx, cy]
                out.append(encode_message(f"{self._prefix}/contour", [oid, *flat]))
        return out

    def _cluster_msgs(self, c: Any, fw: int, fh: int) -> list:
        """One CLUSTER's messages (a spatial grouping of nearby voices — the Clusters stage): a
        ``/cluster`` summary [id, n, cx, cy] + its member object ids + bbox / min-circle / convex-hull
        geometry. Membership + geometry ONLY — a client joins ``members`` to the ``/pose`` table for any
        ring/step layout it wants (the tracker ships facts, not the sequence; see ``lib/clusters.py``)."""
        out: list = []
        cid = int(c.id)
        cx, cy = self._norm(c.centroid[0], c.centroid[1], fw, fh)
        out.append(encode_message(f"{self._prefix}/cluster", [cid, len(c.members), cx, cy]))
        out.append(encode_message(f"{self._prefix}/cluster/members", [cid, *[int(m) for m in c.members]]))
        bx, by, bw, bh = c.bbox                          # axis-aligned: x,y by frame, w,h by their own axis
        nbx, nby = self._norm(bx, by, fw, fh)
        nbw, nbh = self._norm(bw, bh, fw, fh)
        out.append(encode_message(f"{self._prefix}/cluster/bbox", [cid, nbx, nby, nbw, nbh]))
        ccx, ccy, cr = c.circle                          # min-enclosing: radius normalized by width (~square)
        ncx, ncy = self._norm(ccx, ccy, fw, fh)
        nr = (cr / fw) if (not self._pixel and fw > 0) else float(cr)
        out.append(encode_message(f"{self._prefix}/cluster/circle", [cid, ncx, ncy, nr]))
        if getattr(c, "hull", None) is not None:
            flat: list = []
            for p in c.hull:
                hx, hy = self._norm(float(p[0][0]), float(p[0][1]), fw, fh)
                flat += [hx, hy]
            out.append(encode_message(f"{self._prefix}/cluster/hull", [cid, *flat]))
        return out

    def _frame_elements(self, snap: Any) -> tuple[bytes, list, bytes]:
        """The frame's OSC messages: the ``/frm`` header, every emitted object's annotation/pose
        messages, and the ``/alv`` alive-set — the raw material for one bundle or several. The
        population is the FUSED EMIT VIEW when the grouper is over budget (compounds stand in for
        their members, carrying real features), else the plain objects."""
        self._ceps = float(getattr(snap, "contour_epsilon", 0.75))   # the global wire contour ε
        fw, fh = self._dims(snap)
        elems: list = []
        emitted: list = []
        for o in (getattr(snap, "emit", None) or getattr(snap, "objects", ())):
            if getattr(o, "ignored", False):             # edge-connected arm / border residue — never emit
                continue
            if not getattr(o, "confirmed", True):        # withhold a fresh mint (same grace as render/tuio2)
                continue
            msgs = self._object_msgs(o, fw, fh)
            if msgs:
                emitted.append(int(o.id))
                elems.extend(msgs)
        clusters = getattr(snap, "clusters", ()) or ()   # spatial clusters (gated like the old annotation)
        if self._send_all or "cluster" in self._send_keys:
            for c in clusters:
                elems.extend(self._cluster_msgs(c, fw, fh))
            if clusters:                                 # the cluster alive-set (persistent ids present now)
                elems.append(encode_message(f"{self._prefix}/cluster/alv", [int(c.id) for c in clusters]))
        header = encode_message(f"{self._prefix}/frm",
                                [int(getattr(snap, "frame_idx", 0)) & 0x7FFFFFFF, len(emitted),
                                 int(fw), int(fh)])   # dims appended: a renderer needs the aspect
        #                          ratio to draw normalized coords unsquashed (pure-/rmt clients
        #                          have no /tuio2/frm to read it from)
        #                          masked to int32 like tuio2's f_id — OSC 'i' is 32-bit and an
        #                          installation left running crosses 2^31 frames in ~2.3 years @30fps,
        #                          after which an unmasked value makes struct.pack raise EVERY frame
        alive = encode_message(f"{self._prefix}/alv", emitted)
        return header, elems, alive

    def _timetag(self, snap: Any) -> TimeTag:
        """The bundle timetag, in REAL Unix time — what the OSC TimeTag contract encodes.

        ``snap.ts`` is CAPTURE time on an arbitrary base (video PTS / ``perf_counter``), NOT Unix
        time: sent raw, a clip's 3.4 s PTS encodes as NTP 1970-01-01 00:00:03. So the base is
        anchored to the wall clock ONCE, on the first frame, and every later tag is base + ts —
        the absolute value becomes real while consecutive frames keep their exact capture spacing
        (the relative timing a musical client consumes). Mirrors the tuio2 sender."""
        ts = getattr(snap, "ts", None)
        if ts is None:
            return TimeTag(None)                  # no capture stamp → OSC "immediate"
        if self._tt_base is None:
            self._tt_base = time.time() - float(ts)
        return TimeTag(self._tt_base + float(ts))

    def build_packet(self, snap: Any) -> bytes:
        """One OSC **bundle** for the frame: a ``/frm`` header, every object's annotation/pose
        messages, then an ``/alv`` alive-set. Pure — the whole logical frame in one bundle, testable.
        The wire path uses ``build_packets`` (this can exceed a UDP datagram on a busy frame)."""
        header, elems, alive = self._frame_elements(snap)
        return encode_bundle([header, *elems, alive], self._timetag(snap))

    def build_packets(self, snap: Any) -> list:
        """The frame as a LIST of OSC bundles, each ≤ ``_MAX_DATAGRAM`` — so a fat frame (many objects
        × wide fields like efd_bins=64) is SPLIT across UDP datagrams instead of overflowing one and
        failing with EMSGSIZE. ``/frm`` header goes first, ``/alv`` last; objects may span packets
        (each OSC message is self-contained, so a receiver handles them independently)."""
        header, elems, alive = self._frame_elements(snap)
        tt = self._timetag(snap)
        packets: list = []
        cur: list = []
        size = _BUNDLE_OVERHEAD
        for m in [header, *elems, alive]:
            need = len(m) + _ELEM_PREFIX
            if need + _BUNDLE_OVERHEAD > _MAX_DATAGRAM:    # one message alone can't fit any datagram
                self._log.warning(f"[oscout] a single {len(m)}B OSC message exceeds the "
                                  f"{_MAX_DATAGRAM}B datagram budget (likely a dense contour) — "
                                  f"sending it alone; the OS may drop it")
            if cur and size + need > _MAX_DATAGRAM:        # flush the current bundle, start a new one
                packets.append(encode_bundle(cur, tt))
                cur, size = [], _BUNDLE_OVERHEAD
            cur.append(m)
            size += need
        if cur:
            packets.append(encode_bundle(cur, tt))
        return packets

    # -- sender thread on the bus ------------------------------------------- #

    def _run(self) -> None:
        consume_bus(self._bus, self._stop, self._emit, profiler=self._prof, name="oscout",
                    max_fps=self._max_fps)

    def _emit(self, snap: Any) -> None:
        try:
            packets = self.build_packets(snap)
        except Exception as e:                            # noqa: BLE001 — a bad frame must not kill the thread
            self._log.warning(f"[oscout] build failed on frame {getattr(snap, 'frame_idx', '?')}: {e}")
            return
        for packet in packets:                            # a fat frame is several datagrams (all ≤ budget)
            try:
                self._send(packet)
            except OSError as e:
                self._log.warning(f"[oscout] send failed ({len(packet)}B): {e}")
        if _BENCH_STAMPS is not None:                     # BENCH: reversible — emit wall stamp (pairs
            import time as _t                             #   with the capture stamp in lib/loop.py)
            _BENCH_STAMPS.write(f"emit,{getattr(snap, 'frame_idx', -1)},{_t.perf_counter():.9f}\n")
