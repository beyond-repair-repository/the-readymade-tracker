"""``tuio2`` OUTPUT plugin — TUIO 2.0 over UDP.

A sender **thread** that waits on the ``SnapshotBus`` (the single channel out of the
core) and emits one OSC bundle per frame — decoupled from the pipeline, so a slow /
absent receiver never touches tick timing (UDP is fire-and-forget anyway).

Per frame the bundle carries (TUIO 2.0):
  · ``/tuio2/frm``  frame id + timetag + dimension + source
  · ``/tuio2/tok``  one per LIVE object — session id, type/user id, component id,
    x, y, angle (radians). The tracker is MARKERLESS, so type/user and component are
    both **0**: those fields identify a tag the object carries, and a readymade
    carries none. The object IS its session id.
  · ``/tuio2/bnd``  optional per-object bounds (x, y, angle, w, h, area)
  · ``/tuio2/ocg``  optional per-object **high-resolution outer-contour geometry** —
    the object's outline, ``approxPolyDP``-simplified by the GLOBAL ``[observe]
    contour_epsilon`` knob (published on the snapshot; one quality dial for every output)
  · ``/tuio2/alv``  the alive session-id set (bounds the frame)
  · ``/rmt/obj``   custom sidecar — our 4-state name + pose confidence + id (TUIO 2.0
    has no field for these)

Coordinates are normalized [0,1] (TUIO convention) or raw pixels (``coord_mode``). The
angle is the accumulated pose θ wrapped to **[0, 2π)** radians, clockwise-positive in the
y-down frame with 0 = the shape's OWN principal axis pointing screen-up (the shape-anchored
zero — the TUIO/fiducial reading; the raw unwrapped accumulation stays on ``/rmt/pose``).
Because high-res contours can be large, the frame is
split across **multiple UDP packets under a byte ceiling** (``max_packet_bytes``, default
8 KB — under macOS's ``net.inet.udp.maxdgram`` datagram floor of 9216 B) — each packet repeats
``/frm`` + the full ``/alv`` so it is a valid, self-describing TUIO frame and the
receiver merges component messages by session id.

The UDP socket is injectable (``send_fn``) so the whole build-and-emit path is
unit-testable on localhost with no network. Coordinate space: pose (x, y) lives in
POST-preprocess frame pixels — the SAME space as ``snap.frame`` — so normalizing by
the snapshot frame's dimensions is correct even under crop.
"""

from __future__ import annotations

import math
import socket
import threading
import time
from typing import Any, Callable

import cv2

from lib.geometry import place_footprint
from lib.logger import NULL_LOGGER
from lib.loop import consume_bus
from lib.osc import parse_destinations, TimeTag, encode_bundle, encode_message
from lib.params import ParamSpec, bind_params
from lib.scene import Status

_BUNDLE_OVERHEAD = 16          # "#bundle\0" (8) + timetag (8)
_ELEM_PREFIX = 4               # per-element int32 length prefix inside a bundle
_UDP_MAX = 65507               # theoretical max UDP payload (IPv4)
_TAU = 2.0 * math.pi           # the TUIO angle range: radians wrapped to [0, 2π)


class Tuio2Out:
    """TUIO 2.0 sender. ``setup`` → ``start(bus)`` → ``stop``."""

    # -- live parameters, declared WITH the plugin. host/port bind the socket at setup → needs-restart;
    #    max_packet_bytes (→ self._max_bytes, capped at _UDP_MAX) is read in setup (bind=False). coord_mode
    #    (→_pixel) + contour_source (lowercased) via apply setters; Outline quality = the GLOBAL [observe] contour_epsilon (snapshot). --
    PARAMS = (
        ParamSpec("destinations", str, "", attr="_dest_spec",
                  apply=lambda self, v: (setattr(self, "_dest_spec", str(v)),
                                         setattr(self, "_dests", parse_destinations(v, self._host, self._port))),
                  doc="THE destination setting, LIVE: comma list of host:port targets - EVERY packet goes to each (several clients watch one stream); add/remove on the fly; empty = 127.0.0.1:3333 (the legacy host/port pair)"),
        ParamSpec("coord_mode", str, "normalized", enum=("normalized", "pixel"),
                  apply=lambda self, v: setattr(self, "_pixel", str(v).lower() == "pixel"),
                  doc="Emit coords as normalized [0,1] or raw pixels"),
        ParamSpec("timetag_mode", str, "timetag", enum=("timetag", "ints"),
                  apply=lambda self, v: setattr(self, "_tt_ints", str(v).lower() == "ints"),
                  doc="/tuio2/frm time field: 'timetag' = one OSC 't' arg (TUIO 2.0 spec) - 'ints' = two int32 args, unix SECONDS + MICROSECONDS, for clients whose OSC parser rejects the 't' type tag. Same instant, no 't' anywhere; off-spec but lossless to ~1us (NOT the timetag's raw 32-bit halves - NTP seconds overflow a SIGNED int32)"),
        ParamSpec("send_contours", bool, True, doc="Emit per-object outer-contour geometry (/tuio2/ocg)"),
        ParamSpec("contour_source", str, "footprint", enum=("footprint", "auto", "live"),
                  apply=lambda self, v: setattr(self, "_contour_source", str(v).lower()),
                  doc="Which contour to send: rest footprint, live contour, or auto"),
        ParamSpec("send_bounds", bool, False, doc="Emit per-object bounds (/tuio2/bnd): pos, angle, w, h, area"),
        ParamSpec("include_custom", bool, False, doc="Emit the /rmt/obj sidecar (state name, pose confidence) INSIDE the TUIO bundle - OPT-IN since 0.32.124: the message's home is the native oscout stream; enable this bridge only on a TUIO-only rig that needs the lifecycle"),
        ParamSpec("source", str, "readymade", doc="TUIO source name reported in the /tuio2/frm message"),
        ParamSpec("max_packet_bytes", int, 8192, lo=1, hi=65507, hot=False, bind=False, expose="hidden",
                  doc="Per-UDP-packet byte ceiling (capped at IPv4 max 65507)"),
        ParamSpec("max_fps", float, 0.0, lo=0.0, hot=False, doc="Throttle send rate; 0 = unthrottled (every frame)"),
    )

    def __init__(self) -> None:
        self._log = NULL_LOGGER
        self._prof: Any = None                     # --profile: times _emit under the OUTPUT group
        self._bus: Any = None
        self._sock: Any = None
        self._owns_socket = False
        self._send: Callable[[bytes], Any] | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._tt_base: float | None = None         # wall-clock anchor for /frm timetags (first frame)
        # config (filled in setup)
        self._host = "127.0.0.1"           # the empty-`destinations` fallback (constant, not a knob):
        self._port = 3333                  #   an empty destinations spec sends to this single default
        self._pixel = False
        self._tt_ints = False              # /frm time as two int32s instead of an OSC 't' (compat)
        self._send_contours = True
        self._send_bounds = False
        self._include_custom = False
        self._contour_source = "footprint"
        self._source = "readymade"
        self._max_bytes = 8192            # under macOS's net.inet.udp.maxdgram default (9216)
        self._max_fps = 0.0
        self._dims_warned = False
        self._drop_warned = False          # one-shot: an unfittable contour repeats every frame

    # -- lifecycle ----------------------------------------------------------- #

    def setup(self, config: Any, *, logger: Any = None,
              send_fn: Callable[[bytes], Any] | None = None, profiler: Any = None) -> None:
        self._log = logger or NULL_LOGGER
        self._prof = profiler
        g = config.get if config is not None else (lambda k, d=None: d)
        bind_params(self, config)          # every param except max_packet_bytes ← PARAMS (see class attr)
        self._max_bytes = min(int(g("max_packet_bytes", 8192)), _UDP_MAX)    # IPv4 cap; DEFAULT stays under
        #   macOS's net.inet.udp.maxdgram (9216B) — a bigger sendto fails with EMSGSIZE [Errno 40]; raise
        #   this knob only after raising the sysctl on EVERY receiving host
        if send_fn is not None:
            self._send = send_fn                 # tests inject a recorder; no socket
            self._owns_socket = False
        else:
            self._owns_socket = True
            self._open_socket()                  # the send closure derefs self._sock at call time
            self._send = lambda data: [self._sock.sendto(data, d) for d in self._dests]
            #             ^ reads self._dests at CALL time — a live `destinations` edit re-aims mid-run
        self._log.info(f"[tuio2] TUIO 2.0 -> {', '.join(f'{h}:{p}' for h, p in self._dests)} "
                       f"coord={'pixel' if self._pixel else 'normalized'} "
                       f"contours={self._send_contours}({self._contour_source}) "
                       f"bounds={self._send_bounds}")

    def _open_socket(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def start(self, bus: Any) -> "Tuio2Out":
        self._bus = bus
        self._stop.clear()
        if self._owns_socket and self._sock is None:   # RESTART after stop() (output tuio2 on) — stop()
            self._open_socket()                         # closed + nulled it; recreate so sends don't NPE
        self._thread = threading.Thread(target=self._run, name="tuio2", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    def _run(self) -> None:
        self._log.info("[tuio2] sender thread started")
        consume_bus(self._bus, self._stop, self._emit, profiler=self._prof, name="tuio2",
                    max_fps=self._max_fps)

    def _emit(self, snap: Any) -> None:
        try:
            packets = self.build_packets(snap)
        except Exception as e:                   # noqa: BLE001 — a bad frame must not kill the thread
            self._log.warning(f"[tuio2] build failed on frame {getattr(snap, 'frame_idx', '?')}: {e}")
            return
        for p in packets:
            try:
                self._send(p)                    # UDP: fire-and-forget
            except OSError as e:
                self._log.warning(f"[tuio2] send failed ({len(p)}B): {e}")

    # -- the pure build path (unit-tested; no socket) ------------------------ #

    def build_packets(self, snap: Any) -> list[bytes]:
        """Snapshot → a list of UDP payloads (TUIO 2.0 bundles). Pure: no I/O, so the
        wire format is testable by decoding the returned bytes."""
        fw, fh = self._frame_dims(snap)
        eps = float(getattr(snap, "contour_epsilon", 0.75))
        f_id = int(getattr(snap, "frame_idx", 0) or 0) & 0x7FFFFFFF
        # snap.ts is CAPTURE time on an arbitrary base (video PTS / perf_counter) — NOT Unix time, which
        # is what the OSC TimeTag contract encodes. Anchor it to the wall clock ONCE, so the emitted
        # /frm timetag is real NTP-epoch time while consecutive frames keep their exact capture spacing
        # (the relative timing a musical client actually consumes).
        ts = getattr(snap, "ts", None)
        if ts is None:
            tt = TimeTag(None)
        else:
            if self._tt_base is None:
                self._tt_base = time.time() - float(ts)
            tt = TimeTag(self._tt_base + float(ts))
        # LIVE = an ESTABLISHED object with a position (captured: RESTING/MOVED), not ignored,
        # confirmed — the settled registry, the same set every other output ships. Over the voice
        # budget that set is the FUSED EMIT VIEW (snap.emit): a compound token stands in for its
        # members, with the mass-scaled blend as its footprint and the dominant member's angle.
        # Collect first so /alv is known before we size contours.
        live = [o for o in (getattr(snap, "emit", None) or getattr(snap, "objects", ()))
                if self._emitted(o, fw, fh)]
        # The ONLY OSC 't' type tag this plugin emits. Some clients' OSC parsers reject 't' outright
        # and drop the whole message — taking the frame header, and with it the dimensions every
        # normalized coordinate is read against. `timetag_mode = ints` sends the SAME INSTANT as two
        # plain int32s (typetag ',iiiis'), so no 't' appears and the time survives.
        #
        # Unix seconds + MICROSECONDS, not the timetag's own 32-bit halves: OSC 'i' is SIGNED, and
        # NTP seconds (~3.99e9 today) overflows int32 — splitting the raw timetag would either fail
        # to encode or force a negative bit-wrap that a naive client reads as garbage. The fraction
        # has the same problem for half of every second. seconds fits until 2038 and microseconds
        # are ~33000x finer than a 30 fps frame, so nothing a musical client can hear is lost.
        time_args = [tt] if not self._tt_ints else self._time_ints(tt)
        frm = encode_message("/tuio2/frm",
                             [f_id, *time_args, ((fw & 0xFFFF) << 16) | (fh & 0xFFFF), self._source])
        alv = encode_message("/tuio2/alv", [int(o.id) for o in live])
        # the byte budget a single component may occupy so a packet (frm + it + alv)
        # stays under the cap — a contour that would blow the UDP ceiling is coarsened
        # to fit rather than emitted as an undeliverable packet.
        budget = self._max_bytes - _BUNDLE_OVERHEAD - (_ELEM_PREFIX + len(frm)) \
            - (_ELEM_PREFIX + len(alv)) - _ELEM_PREFIX

        components: list[bytes] = []
        for o in live:
            x, y, _raw = self._object_xy(o)
            ang = self._angle(o)                 # ABSOLUTE (cumulative) angle for the token/bounds
            sid = int(o.id)                      # the TRACKER id — TUIO session ids are unique per
            #                                        session, NEVER recycled (spec) — and every surface
            #                                        (webgui badge, OSC egress) speaks this same id
            nx, ny = self._norm(x, y, fw, fh)
            # tu_id = c_id = 0: this tracker is MARKERLESS. tu_id is the type/user pair and c_id the
            # component (symbol) id — both describe a tag the object carries, and a readymade carries
            # none: nothing is ever decoded off it, and no /tuio2/sym is emitted. Sending the tracker id
            # as c_id (as this did) claimed a symbol identity that was never read, and one that CHANGES
            # whenever a re-acquire mints a new id — so a client dispatching on c_id, which is what the
            # field is for, would silently re-map the same physical object. The object's whole identity
            # is s_id, which every other surface (webgui badge, OSC egress) already speaks.
            components.append(encode_message("/tuio2/tok", [sid, 0, 0, nx, ny, float(ang)]))
            if self._send_bounds and o.bbox is not None:
                components.append(self._bnd_msg(sid, o, ang, fw, fh))
            if self._include_custom:
                components.append(encode_message(
                    "/rmt/obj", [sid, o.status.name, float(o.pose_conf or 0.0)]))
            if self._send_contours:
                ocg = self._ocg_msg(sid, o, fw, fh, budget, eps)
                if ocg is not None:
                    components.append(ocg)
        return self._chunk(frm, components, alv, tt)

    # -- assembly ------------------------------------------------------------ #

    def _chunk(self, frm: bytes, components: list[bytes], alv: bytes,
               tt: TimeTag) -> list[bytes]:
        """Greedily pack ``frm`` + components + ``alv`` into bundles under the byte
        cap. Every packet repeats ``frm`` + the full ``alv`` so each is a valid frame
        (the receiver merges components by session id across packets of one frame)."""
        fixed = _BUNDLE_OVERHEAD + (_ELEM_PREFIX + len(frm)) + (_ELEM_PREFIX + len(alv))
        packets: list[bytes] = []
        cur: list[bytes] = []
        size = fixed
        for c in components:
            need = _ELEM_PREFIX + len(c)
            if cur and size + need > self._max_bytes:
                packets.append(encode_bundle([frm, *cur, alv], tt))
                cur, size = [], fixed
            cur.append(c)
            size += need
        packets.append(encode_bundle([frm, *cur, alv], tt))   # always ≥1 packet (frm+alv, empty alive)
        return packets

    @staticmethod
    def _time_ints(tt: TimeTag) -> list:
        """The timetag as ``[unix_seconds:i, microseconds:i]`` — the 't'-free encoding of the SAME
        instant (see build_packets). ``[0, 0]`` when the snapshot carried no capture stamp, the
        int pair standing in for OSC's 'immediate'."""
        s = tt.seconds
        if s is None:
            return [0, 0]
        sec = int(s // 1)
        usec = int(round((s - sec) * 1_000_000))
        if usec >= 1_000_000:                 # rounding can carry into the next second
            sec, usec = sec + 1, 0
        return [sec, usec]

    def _bnd_msg(self, sid: int, o: Any, ang: float, fw: int, fh: int) -> bytes:
        x, y, w, h = o.bbox
        cx, cy = self._norm(x + w / 2.0, y + h / 2.0, fw, fh)
        nw, nh = self._norm(w, h, fw, fh)
        area = float(o.area or (w * h))
        narea = area / float(fw * fh) if not self._pixel and fw > 0 and fh > 0 else area
        return encode_message("/tuio2/bnd", [sid, cx, cy, float(ang), nw, nh, float(narea)])

    def _source_contour(self, o: Any) -> "tuple[Any, bool]":
        """The contour to emit, per ``contour_source`` → ``(contour, perceptual)``:
          · ``footprint`` (default) — the PERCEPTUAL rest outline (``wire_contour``: the inverse EFD
            of the detail-surviving components — already wire quality, so ε never re-simplifies it),
            placed at the object's current pose; falls back to the approx-poly'd captured footprint
            when the features stage hasn't produced one;
          · ``live`` — the live observed contour (``o.contour``), ε-simplified as always;
          · ``auto`` — footprint when available, else live.
        Footprint needs ``footprint_centroid`` + ``pose``; when missing (a still-SETTLING object
        not yet captured) it falls back to the live contour so a token always has a shape."""
        fp = getattr(o, "footprint_contour", None)
        fc = getattr(o, "footprint_centroid", None)
        wc = getattr(o, "wire_contour", None)
        pose = getattr(o, "pose", None)
        want_fp = self._contour_source in ("footprint", "auto")
        if want_fp and fc is not None and pose is not None:
            if wc is not None:
                return place_footprint(wc, fc, pose), True
            if fp is not None:
                return place_footprint(fp, fc, pose), False
        return o.contour, False

    def _ocg_msg(self, sid: int, o: Any, fw: int, fh: int, budget: int, eps: float) -> bytes | None:
        """Encode the object's contour as ``/tuio2/ocg``, at the GLOBAL [observe]
        ``contour_epsilon`` (published on the snapshot — the one contour-quality knob). If the
        message would exceed ``budget`` (so its packet would blow the UDP ceiling), the
        contour is progressively coarsened (ε doubled) until it fits — a single high-res
        contour can't otherwise be split across packets without breaking standard TUIO
        clients (they'd take the last /ocg, not concatenate). Fidelity degrades only for
        a pathologically complex single contour; everything else is untouched."""
        c, perceptual = self._source_contour(o)
        if c is None or len(c) < 3:
            return None
        cc = c if perceptual or eps <= 0 else cv2.approxPolyDP(c, eps, True)   # the perceptual outline
        #                                       is already wire quality (the detail knob shaped it)
        msg = self._encode_ocg(sid, cc, fw, fh)
        if len(msg) > budget:
            eps = eps if eps > 0 else 1.0
            for _ in range(12):                  # ε→3 pts collapses well within a dozen doublings
                eps *= 2.0
                cc = cv2.approxPolyDP(c, eps, True)
                msg = self._encode_ocg(sid, cc, fw, fh)
                if len(msg) <= budget or len(cc) <= 3:
                    break
            if len(msg) > budget:
                # STILL unfittable — either collapsed to 3 points and even that is too big, or the
                # budget is <= 0 because /frm + /alv alone fill the cap (a huge alive set, or an
                # absurdly small max_packet_bytes). DROP the contour: an oversized sendto fails with
                # EMSGSIZE and loses the WHOLE bundle — /tok positions and /alv with it — so shipping
                # the object's POSITION without its outline beats shipping nothing at all.
                if not self._drop_warned:
                    self._drop_warned = True     # once per run: this repeats every frame
                    self._log.warning(f"[tuio2] contour {sid} dropped: {len(msg)}B exceeds the "
                                      f"{budget}B per-component budget (max_packet_bytes="
                                      f"{self._max_bytes}) — positions still ship")
                return None
            self._log.info(f"[tuio2] contour {sid} coarsened to {len(cc)} pts "
                           f"(eps~{eps:.1f}) to fit {budget}B packet budget")
        return msg

    def _encode_ocg(self, sid: int, contour: Any, fw: int, fh: int) -> bytes:
        coords: list[Any] = [sid]
        for px, py in contour.reshape(-1, 2):
            nx, ny = self._norm(float(px), float(py), fw, fh)
            coords.append(nx)
            coords.append(ny)
        return encode_message("/tuio2/ocg", coords)

    # -- helpers ------------------------------------------------------------- #

    def _emitted(self, o: Any, fw: int, fh: int) -> bool:
        """Is this object emitted this frame? By default, only CAPTURED objects — those
        with a committed footprint — the same set every other output draws. Captured spans
        RESTING, MOVED, and HIDDEN (occluded but real). This drops the transient un-captured
        blobs: a fast-moving arm entering at the edge goes SETTLING→MOVED *without ever
        capturing* (capture happens only at SETTLING→RESTING), so a status check alone would
        leak it — the footprint is the reliable "is this a real object" signal.
        A not-yet-``confirmed`` fresh mint (still inside [classify].confirm_seconds) is withheld —
        the SAME grace every output honours, so a hover blip never leaks onto the wire."""
        xy = self._object_xy(o)
        if getattr(o, "ignored", False) or not getattr(o, "confirmed", True) or xy is None:
            return False
        return getattr(o, "footprint_contour", None) is not None

    def _object_xy(self, o: Any) -> tuple[float, float, float] | None:
        if o.pose is not None:
            return float(o.pose[0]), float(o.pose[1]), float(o.pose[2])
        if o.centroid is not None:
            return float(o.centroid[0]), float(o.centroid[1]), 0.0
        return None

    def _angle(self, o: Any) -> float:
        """The token/bounds angle: the ACCUMULATED orientation (``theta_cumulative``) — the one useful
        rotation. The pipeline maintains it continuously across re-settles; the per-settle pose θ
        snaps back to ~0 whenever the footprint is re-baked, so it is never reported anywhere.
        Wrapped to **[0, 2π)** — TUIO wants plain radians, and the accumulated angle may be negative
        or wind past a full turn (``/rmt/pose`` keeps the raw accumulated value for clients that
        integrate turns)."""
        cum = getattr(o, "theta_cumulative", None)
        if cum is not None:
            return float(cum) % _TAU
        return (float(o.pose[2]) % _TAU) if getattr(o, "pose", None) is not None else 0.0

    def _norm(self, x: float, y: float, fw: int, fh: int) -> tuple[float, float]:
        if self._pixel or fw <= 0 or fh <= 0:
            return float(x), float(y)
        return x / float(fw), y / float(fh)

    def _frame_dims(self, snap: Any) -> tuple[int, int]:
        frame = getattr(snap, "frame", None)
        if frame is not None:
            return int(frame.shape[1]), int(frame.shape[0])
        if not self._pixel and not self._dims_warned:
            self._log.warning("[tuio2] no frame dims for normalization -> sending pixel coords")
            self._dims_warned = True
        return 0, 0                              # _norm falls back to pixel when dims are 0
