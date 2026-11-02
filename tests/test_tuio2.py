"""Tuio2Out (TUIO 2.0) — build the per-frame bundle from a synthetic snapshot and
decode it back. No socket, no pipeline: `build_packets` is the pure path, so the
whole wire contract (tokens, contours, alive set, custom sidecar, coord modes,
packet chunking) is verified on localhost with nothing running.
"""

import math

import numpy as np
import pytest

from lib.osc import TimeTag, decode_bundle, decode_packet
from lib.scene import ObjectSnap, SceneSnapshot, Status
from plugins.output.tuio2.tuio2 import Tuio2Out


def _square(cx, cy, r):
    return np.array([[[cx - r, cy - r]], [[cx + r, cy - r]],
                     [[cx + r, cy + r]], [[cx - r, cy + r]]], dtype=np.int32)


_UNSET = object()   # sentinel: distinguish "no footprint arg given" from an explicit footprint=None


def _obj(oid, status=Status.RESTING, pose=(20.0, 17.5, 1.5), contour=None,
         ignored=False, conf=0.9, centroid=(20.0, 17.5), confirmed=True,
         footprint=_UNSET, footprint_centroid=None, theta_cumulative=None):
    # captured by default (has a footprint → emitted). Pass footprint=None for an
    # UN-captured object (SETTLING / a moving arm blob) — dropped by the default filter.
    if footprint is _UNSET:
        footprint = _square(20, 17, 8)
        if footprint_centroid is None:
            footprint_centroid = (20.0, 17.5)
    return ObjectSnap(id=oid, status=status, ignored=ignored, centroid=centroid, confirmed=confirmed,
                      bbox=(10, 10, 20, 15), area=300.0,
                      contour=_square(20, 17, 8) if contour is None else contour,
                      pose=pose, pose_conf=conf, theta_cumulative=theta_cumulative,
                      footprint_contour=footprint, footprint_centroid=footprint_centroid)


def _snap(objects, w=100, h=50, frame_idx=7, ts=1.5):
    return SceneSnapshot(frame_idx=frame_idx, ts=ts, objects=tuple(objects),
                         frame=np.zeros((h, w, 3), np.uint8))


def _out(**cfg):
    o = Tuio2Out()
    o.setup(cfg, send_fn=lambda data: None)      # config dict; no socket
    return o


def _msgs(packets):
    out = []
    for p in packets:
        out.extend(decode_packet(p))
    return out


def _by_addr(msgs, addr):
    return [args for a, args in msgs if a == addr]


# --------------------------------------------------------------------------- #
# frame + token + alive
# --------------------------------------------------------------------------- #

def test_frame_token_alive_present():
    out = _out()
    msgs = _msgs(out.build_packets(_snap([_obj(5)])))
    frm = _by_addr(msgs, "/tuio2/frm")
    assert len(frm) == 1
    assert frm[0][0] == 7                          # frame id
    assert isinstance(frm[0][1], TimeTag)          # time
    # snap.ts (PTS/perf base) is anchored to the WALL clock (the OSC TimeTag contract encodes Unix
    # time): absolute ≈ now, and consecutive frames keep the exact capture spacing.
    import time
    assert abs(frm[0][1].seconds - time.time()) < 5.0
    frm2 = _by_addr(_msgs(out.build_packets(_snap([_obj(5)], ts=1.7))), "/tuio2/frm")
    # abs tolerance: at wall-clock magnitude (~1.7e9 s) float64 ULP is ~2e-7, larger than approx's
    # default rel tolerance on 0.2 — the spacing is exact only to the anchor's representable grid.
    assert frm2[0][1].seconds - frm[0][1].seconds == pytest.approx(0.2, abs=1e-5)
    assert frm[0][2] == ((100 & 0xFFFF) << 16) | (50 & 0xFFFF)   # dim = w<<16|h
    assert frm[0][3] == "readymade"
    tok = _by_addr(msgs, "/tuio2/tok")
    assert len(tok) == 1
    sid, tu, cid, x, y, ang = tok[0]
    # MARKERLESS: the object is its SESSION id. tu_id (type/user) and c_id (component/symbol)
    # both describe a tag the object carries — a readymade carries none, and no /tuio2/sym is
    # ever sent, so echoing the tracker id into c_id would claim a symbol that was never read.
    assert sid == 5 and tu == 0 and cid == 0
    assert x == pytest.approx(20.0 / 100) and y == pytest.approx(17.5 / 50)   # normalized
    assert ang == pytest.approx(1.5)               # pose θ in radians, verbatim
    assert _by_addr(msgs, "/tuio2/alv")[0] == [5]


def test_tok_tag_fields_are_zero_for_every_object():
    # The tag fields must not be DERIVED from the tracker id: two objects with different session
    # ids still carry c_id 0. Echoing the id (as this once did) also made c_id change whenever a
    # re-acquire minted a new one, silently re-mapping the same physical object under a client
    # that dispatches on it.
    toks = _by_addr(_msgs(_out().build_packets(_snap([_obj(5), _obj(17)]))), "/tuio2/tok")
    assert sorted(t[0] for t in toks) == [5, 17]          # session ids DO distinguish objects
    assert {t[1] for t in toks} == {0} and {t[2] for t in toks} == {0}   # tu_id / c_id never do


def test_custom_sidecar_is_opt_in_since_the_native_stream_carries_state():
    plain = _msgs(_out().build_packets(_snap([_obj(5)])))
    assert _by_addr(plain, "/rmt/obj") == []              # default OFF: /rmt/obj's home is oscout
    out = _out(include_custom=True)                       # the TUIO-only bridge, explicitly enabled
    msgs = _msgs(out.build_packets(_snap([_obj(5, status=Status.MOVED, conf=0.42)])))
    assert _by_addr(msgs, "/rmt/obj") == [[5, "MOVED", pytest.approx(0.42)]]


def test_pixel_coord_mode():
    out = _out(coord_mode="pixel")
    tok = _by_addr(_msgs(out.build_packets(_snap([_obj(5)]))), "/tuio2/tok")[0]
    assert tok[3] == pytest.approx(20.0) and tok[4] == pytest.approx(17.5)   # raw px


def test_centroid_fallback_when_no_pose():
    out = _out(coord_mode="pixel")
    o = _obj(9, pose=None, centroid=(33.0, 12.0))
    tok = _by_addr(_msgs(out.build_packets(_snap([o]))), "/tuio2/tok")[0]
    assert tok[0] == 9 and tok[3] == pytest.approx(33.0) and tok[4] == pytest.approx(12.0)
    assert tok[5] == pytest.approx(0.0)            # no pose → angle 0


# --------------------------------------------------------------------------- #
# liveness filtering
# --------------------------------------------------------------------------- #

def test_only_captured_objects_emitted_by_default():
    # default: emit CAPTURED objects (footprint present) — RESTING/MOVED/HIDDEN alike —
    # matching the render. Drop ignored + every UN-captured object, INCLUDING a moving arm
    # blob that reached MOVED without ever capturing (the hands1 edge-blob bug).
    out = _out()
    objs = [_obj(1, status=Status.RESTING),                      # captured → emit
            _obj(2, status=Status.HIDDEN, footprint=None),       # un-captured ghost → drop
            _obj(3, ignored=True),                               # ignored → drop
            _obj(4, status=Status.SETTLING, footprint=None),     # un-captured settling → drop
            _obj(5, status=Status.MOVED),                        # captured (rested then moved) → emit
            _obj(6, status=Status.MOVED, footprint=None),        # arm blob: MOVED, never captured → DROP
            _obj(7, status=Status.HIDDEN)]                       # captured, occluded under hand → emit
    alv = _by_addr(_msgs(out.build_packets(_snap(objs))), "/tuio2/alv")[0]
    assert sorted(alv) == [1, 5, 7]


def test_unconfirmed_fresh_mint_is_withheld():
    # a captured but not-yet-confirmed object (still inside [classify].confirm_seconds) is withheld —
    # the SAME grace the render composite honours, so a hover blip never leaks onto the wire.
    out = _out()
    objs = [_obj(1, confirmed=True), _obj(9, confirmed=False)]   # #9 = a fresh mint
    alv = _by_addr(_msgs(out.build_packets(_snap(objs))), "/tuio2/alv")[0]
    assert sorted(alv) == [1]

def test_cumulative_angle_reports_pose_theta_cumulative():
    # the accumulated θ is maintained upstream by the pose stage (theta_cumulative); tuio2 just
    # REPORTS it. So across a re-settle (pose θ 1.4 → 0, footprint re-baked) the token angle stays 1.4.
    out = _out()
    tok_a = _by_addr(_msgs(out.build_packets(
        _snap([_obj(5, status=Status.MOVED, pose=(20.0, 17.5, 1.4), theta_cumulative=1.4)]))), "/tuio2/tok")[0]
    assert tok_a[5] == pytest.approx(1.4)
    tok_b = _by_addr(_msgs(out.build_packets(                    # pose[2] snapped to 0, but theta_cumulative held
        _snap([_obj(5, status=Status.RESTING, pose=(20.0, 17.5, 0.0), theta_cumulative=1.4)]))), "/tuio2/tok")[0]
    assert tok_b[5] == pytest.approx(1.4)                       # absolute — not reset to 0


def test_token_angle_wrapped_to_tuio_range():
    # TUIO wants plain radians in [0, 2π): the accumulated θ may go negative (counter-clockwise) or wind
    # past a full turn — the token reports it WRAPPED (the raw accumulation stays on /rmt/pose).
    out = _out()
    tok = _by_addr(_msgs(out.build_packets(
        _snap([_obj(5, pose=(20.0, 17.5, 0.0), theta_cumulative=-0.5)]))), "/tuio2/tok")[0]
    assert tok[5] == pytest.approx(2 * math.pi - 0.5)           # −0.5 → 5.783… (never negative on the wire)
    tok = _by_addr(_msgs(out.build_packets(
        _snap([_obj(5, pose=(20.0, 17.5, 0.0), theta_cumulative=7.0)]))), "/tuio2/tok")[0]
    assert tok[5] == pytest.approx(7.0 - 2 * math.pi)           # a wound-past-full turn wraps back into range


def test_empty_scene_still_emits_frame_and_empty_alive():
    out = _out()
    msgs = _msgs(out.build_packets(_snap([])))
    assert len(_by_addr(msgs, "/tuio2/frm")) == 1
    assert _by_addr(msgs, "/tuio2/alv")[0] == []   # nothing alive → release everything


# --------------------------------------------------------------------------- #
# contours (/tuio2/ocg)
# --------------------------------------------------------------------------- #

def test_contour_geometry_normalized():
    out = _out(contour_source="live")          # test the live contour path directly
    msgs = _msgs(out.build_packets(_snap([_obj(5, contour=_square(20, 17, 8))])))
    ocg = _by_addr(msgs, "/tuio2/ocg")
    assert len(ocg) == 1
    sid, *coords = ocg[0]
    assert sid == 5
    assert len(coords) == 8                        # 4 corners × (x, y)
    # first corner (12,9) normalized by (100,50)
    assert coords[0] == pytest.approx(12 / 100) and coords[1] == pytest.approx(9 / 50)


def test_contours_off():
    out = _out(send_contours=False)
    assert _by_addr(_msgs(out.build_packets(_snap([_obj(5)]))), "/tuio2/ocg") == []


def test_contour_epsilon_simplifies():
    # a many-vertex near-circle collapses to fewer points at a larger epsilon.
    theta = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    circle = np.stack([50 + 20 * np.cos(theta), 25 + 15 * np.sin(theta)], 1)
    circle = circle.round().astype(np.int32).reshape(-1, 1, 2)
    # the ε comes from the SNAPSHOT — the global [observe] contour_epsilon, one knob for every output
    out = _out(contour_source="live")
    fine_snap = _snap([_obj(5, contour=circle)]); object.__setattr__(fine_snap, "contour_epsilon", 0.1)
    coarse_snap = _snap([_obj(5, contour=circle)]); object.__setattr__(coarse_snap, "contour_epsilon", 5.0)
    n_fine = len(_by_addr(_msgs(out.build_packets(fine_snap)), "/tuio2/ocg")[0]) - 1
    n_coarse = len(_by_addr(_msgs(out.build_packets(coarse_snap)), "/tuio2/ocg")[0]) - 1
    assert n_coarse < n_fine                       # higher ε → fewer contour points


def test_degenerate_contour_skipped():
    out = _out(contour_source="live")
    tiny = np.array([[[1, 1]], [[2, 2]]], dtype=np.int32)   # < 3 points
    assert _by_addr(_msgs(out.build_packets(_snap([_obj(5, contour=tiny)]))), "/tuio2/ocg") == []


# --------------------------------------------------------------------------- #
# contour_source: STABLE footprint (default) placed at pose vs the live contour
# --------------------------------------------------------------------------- #

def test_footprint_is_default_and_placed_at_pose():
    # default contour_source=footprint: emit the captured rest outline at the pose,
    # NOT the noisy live contour. Identity pose → the footprint square itself.
    out = _out(coord_mode="pixel")
    fp = _square(50, 25, 10)                            # rest outline centered at (50,25)
    o = _obj(5, pose=(50.0, 25.0, 0.0), contour=_square(20, 17, 5),   # live is a DIFFERENT shape
             footprint=fp, footprint_centroid=(50.0, 25.0))
    ocg = _by_addr(_msgs(out.build_packets(_snap([o]))), "/tuio2/ocg")[0]
    corners = set(zip(ocg[1::2], ocg[2::2]))
    assert (40.0, 15.0) in corners and (60.0, 35.0) in corners    # the footprint square
    assert (15.0, 12.0) not in corners                            # NOT the live contour


def test_footprint_follows_pose_translation():
    out = _out(coord_mode="pixel")
    fp = _square(50, 25, 10)
    o = _obj(5, pose=(70.0, 25.0, 0.0), contour=fp, footprint=fp, footprint_centroid=(50.0, 25.0))
    ocg = _by_addr(_msgs(out.build_packets(_snap([o]))), "/tuio2/ocg")[0]
    xs = ocg[1::2]
    assert min(xs) == pytest.approx(60.0) and max(xs) == pytest.approx(80.0)   # shifted +20


def test_footprint_follows_pose_rotation():
    import math
    out = _out(coord_mode="pixel")
    fp = np.array([[[30, 20]], [[70, 20]], [[70, 30]], [[30, 30]]], np.int32)  # wide rest rect
    o = _obj(5, pose=(50.0, 25.0, math.pi / 2), contour=fp,
             footprint=fp, footprint_centroid=(50.0, 25.0))
    ocg = _by_addr(_msgs(out.build_packets(_snap([o]))), "/tuio2/ocg")[0]
    xs, ys = ocg[1::2], ocg[2::2]
    assert max(xs) - min(xs) == pytest.approx(10.0, abs=1)   # 90° → wide rect becomes tall
    assert max(ys) - min(ys) == pytest.approx(40.0, abs=1)


def test_contour_source_live_uses_live_even_with_footprint():
    out = _out(coord_mode="pixel", contour_source="live")
    o = _obj(5, pose=(50.0, 25.0, 0.0), contour=_square(20, 17, 5),
             footprint=_square(50, 25, 10), footprint_centroid=(50.0, 25.0))
    xs = _by_addr(_msgs(out.build_packets(_snap([o]))), "/tuio2/ocg")[0][1::2]
    assert min(xs) == pytest.approx(15.0)                   # the LIVE (20-5) square, not footprint

# --------------------------------------------------------------------------- #
# bounds + packet chunking
# --------------------------------------------------------------------------- #

def test_bounds_optional():
    assert _by_addr(_msgs(_out().build_packets(_snap([_obj(5)]))), "/tuio2/bnd") == []
    on = _out(send_bounds=True)
    bnd = _by_addr(_msgs(on.build_packets(_snap([_obj(5)]))), "/tuio2/bnd")
    assert len(bnd) == 1 and bnd[0][0] == 5        # sid, then x,y,angle,w,h,area


def test_packet_chunking_preserves_all_tokens():
    # a tiny cap forces multiple UDP packets; every token must still arrive, and
    # every packet must be a self-describing frame (its own /frm + /alv).
    objs = [_obj(i, contour=_square(20, 17, 8)) for i in range(1, 13)]
    out = _out(max_packet_bytes=250)
    packets = out.build_packets(_snap(objs))
    assert len(packets) > 1                        # actually split
    for p in packets:
        inner = decode_packet(p)
        assert any(a == "/tuio2/frm" for a, _ in inner)
        assert any(a == "/tuio2/alv" for a, _ in inner)
    all_tok = _by_addr(_msgs(packets), "/tuio2/tok")
    assert sorted(t[0] for t in all_tok) == list(range(1, 13))   # all 12 objects present


def test_an_unfittable_contour_is_dropped_rather_than_emitted_oversized():
    # The coarsening loop bottoms out (approxPolyDP can collapse a contour only so far). When even
    # that does not fit, the message must be DROPPED, not returned oversized: an oversized sendto
    # fails with EMSGSIZE and loses the WHOLE bundle — /tok positions and /alv with it — so shipping
    # the object's position without its outline beats shipping nothing at all.
    out = _out()
    o = _obj(1, contour=_square(20, 17, 8))
    assert out._ocg_msg(1, o, 100, 50, 8, 0.75) is None       # no contour fits 8 bytes
    # budget <= 0 means /frm + the full /alv already fill the cap (a huge alive set, or an absurd
    # max_packet_bytes). This case previously skipped the fit check ENTIRELY and emitted oversized.
    assert out._ocg_msg(1, o, 100, 50, 0, 0.75) is None
    assert out._ocg_msg(1, o, 100, 50, -20, 0.75) is None


def test_a_contour_that_can_fit_is_coarsened_and_still_sent():
    # the drop is the LAST resort — anything that can be made to fit is still emitted, so the guard
    # above cannot quietly degrade into "drop every contour".
    out = _out()
    o = _obj(1, contour=_square(20, 17, 8))
    roomy = out._ocg_msg(1, o, 100, 50, 4096, 0.75)
    assert roomy is not None and len(roomy) <= 4096
    tight = out._ocg_msg(1, o, 100, 50, 48, 0.75)              # forces the coarsening path
    assert tight is not None and len(tight) <= 48              # coarsened to fit, not dropped


def test_alive_set_repeated_in_every_packet():
    objs = [_obj(i, contour=_square(20, 17, 8)) for i in range(1, 13)]
    out = _out(max_packet_bytes=250)
    for p in out.build_packets(_snap(objs)):
        _tt, els = decode_bundle(p)
        alv = [decode_packet(e)[0] for e in els if decode_packet(e)[0][0] == "/tuio2/alv"]
        assert alv and sorted(alv[0][1]) == list(range(1, 13))   # full alive set in each packet


# --------------------------------------------------------------------------- #
# config-schema discipline (same guard as [camera]/[observe]: every bool declared)
# --------------------------------------------------------------------------- #

def test_tuio2_schema_bools_declared():
    import readymade_tracker
    sch = readymade_tracker.CONF_SCHEMAS["tuio2"]
    for b in ("send_contours", "send_bounds", "include_custom"):
        assert sch[b] is bool, f"[tuio2].{b} missing from SCHEMAS — a conf 'false' would read True"


def test_tuio2_config_coerces():
    import readymade_tracker
    from lib.config import build_config
    cfg = build_config({"tuio2": {"send_contours": "false", "max_packet_bytes": "9000",
                                    "coord_mode": "pixel"},
                        "observe": {"contour_epsilon": "0.5"}},
                       schemas=readymade_tracker.CONF_SCHEMAS)
    s = cfg.get("tuio2")
    assert s.get("send_contours") is False        # not the truthy string "false"
    assert s.get("max_packet_bytes") == 9000 and s.get("coord_mode") == "pixel"   # int coercion
    assert cfg.get("observe").get("contour_epsilon") == 0.5   # the GLOBAL contour-quality knob


def test_tuio2_default_budget_fits_a_macos_datagram():
    # macOS caps a UDP datagram at net.inet.udp.maxdgram (default 9216B) — a bigger sendto fails
    # with EMSGSIZE [Errno 40]. The default packet budget must sit UNDER that floor (like oscout's),
    # so fat perceptual /ocg bundles split instead of erroring on an untuned Mac.
    out = _out()                                  # no max_packet_bytes in config → the default
    assert out._max_bytes <= 9216
    assert out._max_bytes == 8192


def test_tuio2_enabled_by_section_presence():
    import readymade_tracker
    from lib.config import build_config
    assert build_config({"tuio2": {}}, schemas=readymade_tracker.CONF_SCHEMAS).is_enabled("tuio2")
    assert not build_config({"tuio2": {"enabled": "false"}},
                            schemas=readymade_tracker.CONF_SCHEMAS).is_enabled("tuio2")


# --------------------------------------------------------------------------- #
# review fixes: oversized-contour fit + no throttle busy-spin
# --------------------------------------------------------------------------- #

def test_oversized_contour_coarsened_to_fit():
    # a single high-res contour that would blow the UDP ceiling must be coarsened to
    # fit the packet budget — never emitted as an undeliverable >cap packet (which
    # sendto drops, silently losing the REQUIRED geometry).
    theta = np.linspace(0, 2 * np.pi, 5000, endpoint=False)
    big = np.stack([500 + 400 * np.cos(theta), 300 + 250 * np.sin(theta)], 1)
    big = big.round().astype(np.int32).reshape(-1, 1, 2)
    out = _out(contour_epsilon=0.0, max_packet_bytes=8000,   # eps 0 = no simplification → ~50 KB raw
               contour_source="live")
    packets = out.build_packets(_snap([_obj(5, contour=big)], w=1000, h=600))
    assert packets and all(len(p) <= 8000 for p in packets)  # every packet under the cap
    ocg = _by_addr(_msgs(packets), "/tuio2/ocg")
    assert len(ocg) == 1 and ocg[0][0] == 5                  # contour still present (coarsened)
    assert (len(ocg[0]) - 1) // 2 >= 3                       # at least a polygon


def test_throttle_does_not_busy_spin():
    # regression: with max_fps > 0 and frames flowing fast, the sender must BLOCK
    # between emits, not spin wait_fresh (the bug did ~1e6 calls / 0.5 s, pinning a core).
    import time

    class CountingBus:
        def __init__(self):
            self.calls = 0
            self._n = 0

        def wait_fresh(self, last_idx, timeout):     # always a NEW fresh frame, no blocking
            self.calls += 1
            self._n += 1
            return _snap([_obj(1)], frame_idx=self._n)

    bus = CountingBus()
    out = _out(max_fps=50)                            # 20 ms window
    out.start(bus)
    time.sleep(0.3)
    out.stop()
    assert bus.calls < 200, f"busy-spin regression: {bus.calls} wait_fresh calls in 0.3 s"


class _IdleBus:
    def wait_fresh(self, last_idx, timeout):
        import time
        time.sleep(min(timeout, 0.01))
        return None
    def latest(self):
        return None


def test_restart_recreates_the_socket():
    # regression: `output tuio2 off` (stop) then `on` (start) must recreate the socket, else the send
    # closure derefs None.sendto → AttributeError kills the sender thread while the router logs "ON".
    out = Tuio2Out()
    out.setup({"host": "127.0.0.1", "port": 34567})   # a real socket (no injected send_fn)
    assert out._sock is not None
    out.stop()
    assert out._sock is None                           # stop() closed + nulled it
    out.start(_IdleBus())
    try:
        assert out._sock is not None                   # start() recreated it
        out._send(b"\x00\x00\x00\x00")                 # the send closure works — no AttributeError
    finally:
        out.stop()


def test_session_id_is_the_tracker_id_even_when_a_slot_exists():
    """Owner decision: EVERY surface speaks the tracker id. TUIO session ids are unique per session
    and never recycled (spec), so the recycled display slot must NOT leak onto the wire."""
    from dataclasses import replace

    o = replace(_obj(17), slot=3)
    msgs = _msgs(_out().build_packets(_snap([o])))
    assert _by_addr(msgs, "/tuio2/tok")[0][0] == 17
    assert list(_by_addr(msgs, "/tuio2/alv")[0]) == [17]


from types import SimpleNamespace


def test_destinations_parse_and_fan_out():
    # 'h:p, h:p' fans every packet out to each target; empty falls back to the single host/port.
    from types import SimpleNamespace
    from lib.osc import parse_destinations
    assert parse_destinations("", "10.0.0.1", 3333) == [("10.0.0.1", 3333)]
    assert parse_destinations("127.0.0.1:3333, 192.168.1.20:3335", "x", 1) == \
        [("127.0.0.1", 3333), ("192.168.1.20", 3335)]
    # malformed tokens are SKIPPED (never raise — this parses live /rmt/set input): a port-less host,
    # a garbled port, and an out-of-range port each drop; the valid tokens still apply.
    assert parse_destinations("192.168.1.5", "10.0.0.1", 3333) == [("10.0.0.1", 3333)]   # → fallback
    assert parse_destinations("a:9000, 192.168.1.5, b:banana, c:70000, d:0", "x", 1) == [("a", 9000)]
    out = Tuio2Out()
    out.setup(SimpleNamespace(get=lambda k, d=None: {"destinations": "127.0.0.1:3333,127.0.0.1:3335"}.get(k, d)))
    sent = []
    out._sock = SimpleNamespace(sendto=lambda data, addr: sent.append((data, addr)))
    out._send(b"pkt")
    assert [a for _, a in sent] == [("127.0.0.1", 3333), ("127.0.0.1", 3335)]


def test_destinations_edit_live():
    # the param is HOT: its apply re-parses, and the send closure reads self._dests at call time —
    # a webgui /rmt/set adds or removes clients mid-run, no restart.
    out = _out()
    sent = []
    out._sock = SimpleNamespace(sendto=lambda data, addr: sent.append(addr))
    out._send = lambda data: [out._sock.sendto(data, d) for d in out._dests]
    spec = next(p for p in Tuio2Out.PARAMS if p.key == "destinations")
    spec.apply(out, "127.0.0.1:4000, 127.0.0.1:4001")
    out._send(b"x")
    spec.apply(out, "127.0.0.1:4000")                           # one client removed, live
    out._send(b"y")
    assert sent == [("127.0.0.1", 4000), ("127.0.0.1", 4001), ("127.0.0.1", 4000)]


# --------------------------------------------------------------------------- #
# timetag_mode — the OSC 't' type-tag compatibility escape hatch
# --------------------------------------------------------------------------- #

def test_frm_carries_an_osc_timetag_by_default():
    frm = _by_addr(_msgs(_out().build_packets(_snap([_obj(5)], ts=2.0))), "/tuio2/frm")[0]
    assert isinstance(frm[1], TimeTag)                 # spec: one 't' arg
    assert len(frm) == 4                               # f_id, time, dim, source


def test_timetag_mode_ints_emits_no_t_type_tag():
    # Some clients' OSC parsers reject the 't' type tag and drop the WHOLE message — which for
    # /frm means losing the frame dimensions every normalized coordinate is read against. In
    # `ints` mode the time field becomes unix SECONDS + MICROSECONDS as two plain int32s: the
    # same instant, nothing type-tagged 't'.
    import time as _t
    before = _t.time()
    packets = _out(timetag_mode="ints").build_packets(_snap([_obj(5)], ts=2.0))
    after = _t.time()
    frm = _by_addr(_msgs(packets), "/tuio2/frm")[0]
    assert len(frm) == 5                               # f_id, sec, usec, dim, source
    assert not any(isinstance(a, TimeTag) for a in frm)
    sec, usec = frm[1], frm[2]
    assert 0 <= usec < 1_000_000
    assert before <= sec + usec / 1e6 <= after         # the REAL anchored instant, not a sentinel
    # both halves must survive OSC's SIGNED int32 — NTP seconds would not have
    assert -2**31 <= sec < 2**31 and -2**31 <= usec < 2**31
    assert isinstance(frm[3], int) and isinstance(frm[4], str)   # dim + source still follow
    # and NO message in the frame carries a TimeTag argument
    assert not any(isinstance(a, TimeTag) for _addr, args in _msgs(packets) for a in args)


def test_timetag_mode_ints_preserves_capture_spacing():
    out = _out(timetag_mode="ints")
    def at(ts):
        frm = _by_addr(_msgs(out.build_packets(_snap([_obj(5)], ts=ts))), "/tuio2/frm")[0]
        return frm[1] + frm[2] / 1e6
    t0 = at(2.0)
    assert at(2.25) - t0 == pytest.approx(0.25, abs=2e-6)   # exact to the microsecond grid


def test_timetag_mode_ints_keeps_the_real_time_on_the_bundle():
    # the bundle's own timetag is a raw 8-byte header field, never type-tagged, so a client that
    # reads bundles still gets the anchored capture time even in the compat mode.
    out = _out(timetag_mode="ints")
    pkt = out.build_packets(_snap([_obj(5)], ts=2.0))[0]
    assert pkt.startswith(b"#bundle\0")
    assert pkt[8:16] != b"\0" * 8                      # not the 'immediate' 8 bytes


# --------------------------------------------------------------------------- #
# The FUSED EMIT VIEW — a compound is an ordinary token
# --------------------------------------------------------------------------- #

def test_emit_view_compound_is_an_ordinary_token():
    from dataclasses import replace as _replace
    a, b, lone = _obj(3), _obj(5), _obj(9)
    comp = _obj(12, pose=(40.0, 20.0, 0.0))                  # world-placed blend → identity placement
    comp = _replace(comp, theta_cumulative=1.25)
    snap = _replace(_snap([a, b, lone]), emit=(lone, comp))
    msgs = decode_packet(_out().build_packets(snap)[0])
    d = dict(msgs)
    assert d["/tuio2/alv"] == [9, 12]                        # members replaced by the compound
    toks = [v for adr, v in msgs if adr == "/tuio2/tok"]
    assert {v[0] for v in toks} == {9, 12}
    comp_tok = next(v for v in toks if v[0] == 12)
    assert comp_tok[5] == pytest.approx(1.25 % (2 * math.pi))   # tok angle = theta_cumulative wrapped


def test_no_emit_means_the_token_wire_is_unchanged():
    from dataclasses import replace as _replace
    snap = _snap([_obj(3), _obj(5)])                         # emit defaults to None
    d = dict(decode_packet(_out().build_packets(snap)[0]))
    assert d["/tuio2/alv"] == [3, 5]
