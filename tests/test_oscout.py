"""oscout — the smart custom-OSC egress. Generic serialization of obj.annotations + pose,
self-describing address-per-field, decoded back with lib.osc. Pure build path + a real UDP loopback."""

from types import SimpleNamespace

import numpy as np
import pytest

import readymade_tracker
from lib.config import build_config
from lib.osc import decode_packet
from lib.scene import ClusterSnap, ObjectSnap, Status
from plugins.output.oscout.oscout import OscOut, _as_fields


def _obj(oid, ann=None, pose=(50.0, 25.0, 0.0), theta_cumulative=0.5):
    # /rmt/pose theta reports the ACCUMULATED orientation (theta_cumulative), never pose[2]
    # (which snaps to ~0 on every re-settle by construction).
    return ObjectSnap(id=oid, status=Status.RESTING, ignored=False, centroid=(50.0, 25.0),
                      bbox=(40, 15, 20, 20), area=400.0, contour=None, pose=pose,
                      theta_cumulative=theta_cumulative, annotations=ann or {})


def _clu(cid, members, cx=50.0, cy=25.0, hull=None):
    return ClusterSnap(id=cid, members=tuple(members), centroid=(cx, cy),
                       bbox=(40.0, 15.0, 20.0, 20.0), circle=(cx, cy, 14.0), hull=hull)


def _snap(objs, emit=None, clusters=(), w=100, h=50, frame_idx=3, ts=1.5):
    return SimpleNamespace(objects=tuple(objs), emit=None if emit is None else tuple(emit),
                           clusters=tuple(clusters),
                           frame_idx=frame_idx, ts=ts, frame=np.zeros((h, w, 3), np.uint8))


class _Cfg:
    def __init__(self, **kw):
        self._d = kw
    def get(self, k, d=None):
        return self._d.get(k, d)


def _out(**cfg):
    o = OscOut()
    o.setup(_Cfg(**cfg), send_fn=lambda d: None)      # inject: no socket
    return o


def _decode(bundle):
    return decode_packet(bundle)                       # flat [(address, args), …]


# --------------------------------------------------------------------------- #
# build_packet — annotations, pose, header, alive
# --------------------------------------------------------------------------- #

def test_features_serialized_address_per_field():
    o = _out(send="all", coord_mode="pixel")
    d = dict(_decode(o.build_packet(_snap([_obj(5, ann={"features": {"circularity": [0.9],
                                                                     "solidity": [1.0]}})]))))
    assert d["/rmt/features/circularity"] == [5, pytest.approx(0.9)]
    assert d["/rmt/features/solidity"] == [5, pytest.approx(1.0)]
    assert d["/rmt/frm"] == [3, 1, 100, 50]           # frame_idx, emitted count, frame dims (aspect)
    assert d["/rmt/alv"] == [5]
    assert d["/rmt/pose"][:1] == [5]


def test_pose_normalized_by_frame_dims():
    d = dict(_decode(_out(coord_mode="normalized").build_packet(
        _snap([_obj(5, pose=(50.0, 25.0, 0.5))], w=100, h=50))))
    p = d["/rmt/pose"]
    assert p[1] == pytest.approx(0.5) and p[2] == pytest.approx(0.5) and p[3] == pytest.approx(0.5)


def test_send_filter_selects_producers_and_pose():
    o = _out(send="features")                          # only features; pose + cluster NOT listed → off
    addrs = {a for a, _ in _decode(o.build_packet(
        _snap([_obj(5, ann={"features": {"a": [1.0]}})], clusters=[_clu(3, [5])])))}
    assert "/rmt/features/a" in addrs
    assert "/rmt/cluster" not in addrs and "/rmt/pose" not in addrs   # cluster table gated like a producer


def test_contour_opt_in_and_placed_at_live_pose():
    fp = np.array([[[6, 6]], [[14, 6]], [[14, 14]], [[6, 14]]], np.int32)   # rest square around (10,10)
    o = ObjectSnap(id=7, status=Status.MOVED, ignored=False, centroid=(10.0, 10.0),
                   bbox=(6, 6, 8, 8), area=64.0, contour=None, pose=(50.0, 10.0, 0.0),
                   footprint_contour=fp, footprint_centroid=(10.0, 10.0), annotations={})
    all_addrs = {a for a, _ in _decode(_out(send="all").build_packet(_snap([o])))}
    assert "/rmt/contour" not in all_addrs                # opt-in ONLY, not part of 'all'
    d = dict(_decode(_out(send="pose,contour", coord_mode="pixel").build_packet(_snap([o]))))
    args = d["/rmt/contour"]
    assert args[0] == 7
    xs = args[1::2]                                        # [id, x0,y0,x1,y1,…]
    assert min(xs) >= 45 and max(xs) <= 55                 # centered on x=50 (the POSE), NOT x=10 (rest)


def test_send_all_includes_every_producer_and_pose():
    addrs = {a for a, _ in _decode(_out(send="all").build_packet(
        _snap([_obj(5, ann={"features": {"a": [1.0]}})], clusters=[_clu(3, [5])])))}
    assert {"/rmt/features/a", "/rmt/cluster", "/rmt/cluster/members", "/rmt/pose"} <= addrs


def test_object_with_no_requested_data_is_not_alive():
    o = _out(send="features")                          # pose off
    d = dict(_decode(o.build_packet(_snap([
        _obj(5, ann={}, pose=None),                    # no features + no pose → not emitted
        _obj(6, ann={"features": {"a": [1.0]}})]))))
    assert d["/rmt/alv"] == [6] and d["/rmt/frm"] == [3, 1, 100, 50]


def test_unconfirmed_fresh_mint_is_withheld():
    o = _out(send="all")
    fresh = ObjectSnap(id=9, status=Status.RESTING, ignored=False, centroid=(50.0, 25.0),
                       bbox=(40, 15, 20, 20), area=400.0, contour=None, pose=(50.0, 25.0, 0.5),
                       confirmed=False, annotations={"features": {"circularity": [0.9]}})
    d = dict(_decode(o.build_packet(_snap([fresh, _obj(6, ann={"features": {"circularity": [0.9]}})]))))
    assert d["/rmt/alv"] == [6]                         # #9 (fresh mint) withheld, same grace as render/tuio2


def test_prefix_is_configurable():
    addrs = {a for a, _ in _decode(_out(prefix="rt", send="features").build_packet(
        _snap([_obj(5, ann={"features": {"c": [0.5]}})])))}
    assert "/rt/features/c" in addrs and "/rt/frm" in addrs


def test_as_fields_shapes():
    assert list(_as_fields({"a": [1.0, 2.0], "b": 3})) == [("a", [1.0, 2.0]), ("b", [3.0])]
    assert list(_as_fields([1, 2])) == [("v", [1.0, 2.0])]


def test_build_packets_splits_fat_frame_under_datagram_limit():
    """A busy frame (many objects × a wide efd field) would overflow one UDP datagram (macOS caps at
    9216B → EMSGSIZE); build_packets must split it into datagrams that each fit, losing no message."""
    from plugins.output.oscout.oscout import _MAX_DATAGRAM
    o = _out(send="all", coord_mode="pixel")
    fat = [_obj(i, ann={"features": {"efd": [float(j) for j in range(64)]}}) for i in range(40)]
    snap = _snap(fat)
    assert len(o.build_packet(snap)) > _MAX_DATAGRAM               # one bundle WOULD overflow
    packets = o.build_packets(snap)
    assert len(packets) > 1                                        # so it's split
    assert all(len(p) <= _MAX_DATAGRAM for p in packets)          # every datagram fits the OS limit
    # no message lost: the union of the split bundles == the single-bundle content
    split = [a for p in packets for a, _ in _decode(p)]
    assert sorted(split) == sorted(a for a, _ in _decode(o.build_packet(snap)))
    assert _decode(packets[0])[0][0] == "/rmt/frm"                # header first
    assert _decode(packets[-1])[-1][0] == "/rmt/alv"             # alive last
    assert split.count("/rmt/frm") == 1 and split.count("/rmt/alv") == 1


# --------------------------------------------------------------------------- #
# The FUSED EMIT VIEW — a compound replaces its members on the object wire
# --------------------------------------------------------------------------- #

def test_emit_view_replaces_members_with_the_compound_on_the_wire():
    o = _out(coord_mode="normalized")
    # objects = the REAL scene (3 and 5 fused, 9 alone); emit = the wire population
    comp = _obj(12, ann={"features": {"size": [0.4]}}, theta_cumulative=1.25)
    d = dict(_decode(o.build_packet(_snap([_obj(3), _obj(5), _obj(9)], emit=[_obj(9), comp]))))
    alv = d["/rmt/alv"]
    assert alv == [9, 12]                                    # the compound stands in; members gone
    assert d["/rmt/frm"][1] == 2                             # /frm counts the EMITTED population
    dc = dict(_decode(o.build_packet(_snap([_obj(3), _obj(5)], emit=[comp]))))
    assert dc["/rmt/pose"][0] == 12
    assert dc["/rmt/pose"][3] == pytest.approx(1.25)         # theta rides theta_cumulative, like any object
    assert dc["/rmt/features/size"] == [12, pytest.approx(0.4)]   # REAL features, generic reflection
    assert not any(a.startswith("/rmt/grp") for a in dc)     # the /grp wire is gone


def test_no_emit_means_the_object_wire_is_unchanged():
    d = dict(_decode(_out().build_packet(_snap([_obj(3), _obj(5)]))))     # emit=None → plain objects
    assert d["/rmt/alv"] == [3, 5]
    addrs = {a for a, _ in _decode(_out().build_packet(_snap([_obj(3)])))}
    assert not any(a.startswith("/rmt/grp") for a in addrs)


# --------------------------------------------------------------------------- #
# clusters — the /rmt/cluster table (cluster-centric; mirrors /grp)
# --------------------------------------------------------------------------- #

def test_cluster_table_on_the_wire():
    o = _out(coord_mode="normalized")
    hull = np.array([[[40, 15]], [[60, 15]], [[60, 35]], [[40, 35]]], np.int32)
    c = _clu(4, [3, 5], cx=50.0, cy=25.0, hull=hull)
    d = dict(_decode(o.build_packet(_snap([_obj(3), _obj(5)], clusters=[c]))))
    assert d["/rmt/cluster"][:2] == [4, 2]                          # [cid, member count, …]
    assert d["/rmt/cluster"][2:4] == [pytest.approx(0.5), pytest.approx(0.5)]   # centroid normalized (50/100, 25/50)
    assert d["/rmt/cluster/members"] == [4, 3, 5]                   # cid + the member object ids
    assert d["/rmt/cluster/bbox"] == [4, pytest.approx(0.4), pytest.approx(0.3),
                                      pytest.approx(0.2), pytest.approx(0.4)]   # x/w, y/h — w by width, h by height
    assert d["/rmt/cluster/circle"][:3] == [4, pytest.approx(0.5), pytest.approx(0.5)]
    assert d["/rmt/cluster/circle"][3] == pytest.approx(0.14)       # r normalized by width (14/100)
    assert d["/rmt/cluster/hull"][0] == 4 and len(d["/rmt/cluster/hull"]) == 1 + 2 * len(hull)
    assert d["/rmt/cluster/alv"] == [4]                             # the cluster alive-set


def test_cluster_hull_omitted_when_absent():
    d = dict(_decode(_out().build_packet(_snap([_obj(3)], clusters=[_clu(4, [3], hull=None)]))))
    assert "/rmt/cluster" in d and "/rmt/cluster/hull" not in d     # degenerate → no hull message


def test_no_clusters_means_no_cluster_messages():
    addrs = {a for a, _ in _decode(_out().build_packet(_snap([_obj(3)])))}   # clusters defaults to ()
    assert not any(a.startswith("/rmt/cluster") for a in addrs)


# --------------------------------------------------------------------------- #
# real localhost UDP loopback
# --------------------------------------------------------------------------- #

def test_oscout_udp_loopback():
    import socket

    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("127.0.0.1", 0))
    rx.settimeout(2.0)
    port = rx.getsockname()[1]
    out = OscOut()
    out.setup(_Cfg(destinations=f"127.0.0.1:{port}", send="all", coord_mode="pixel"))   # real socket
    try:
        out._emit(_snap([_obj(7, ann={"features": {"circularity": [0.88]}})]))
        data, _ = rx.recvfrom(65535)
        d = dict(_decode(data))
        assert d["/rmt/features/circularity"] == [7, pytest.approx(0.88)]
        assert d["/rmt/pose"][0] == 7
    finally:
        out.stop()
        rx.close()


def test_setup_stores_profiler_and_stop_idempotent():
    from lib.profile import Profiler
    p = Profiler()
    out = OscOut()
    out.setup(_Cfg(), profiler=p)                    # no send → default fallback destination
    assert out._prof is p
    out.stop()
    out.stop()                                         # second stop must not raise


# --------------------------------------------------------------------------- #
# wiring: section presence enables it; --oscout/--no-oscout toggles it
# --------------------------------------------------------------------------- #

def test_oscout_enabled_by_section_presence():
    assert build_config({"oscout": {"destinations": "127.0.0.1:3400"}}, schemas=readymade_tracker.SCHEMAS).is_enabled("oscout")
    assert not build_config({"input": {"source": "video"}}, schemas=readymade_tracker.SCHEMAS).is_enabled("oscout")


def test_no_oscout_flag_disables_a_present_section():
    ov = readymade_tracker._cli_overrides(readymade_tracker._parse_args(["--no-oscout"]))
    assert ov["oscout"] == {"enabled": "false"}
    assert not build_config({"oscout": {"destinations": "127.0.0.1:3400"}},
                            schemas=readymade_tracker.SCHEMAS, overrides=ov).is_enabled("oscout")


class _IdleBus:
    def wait_fresh(self, last_idx, timeout):
        import time
        time.sleep(min(timeout, 0.01))
        return None
    def latest(self):
        return None


def test_restart_recreates_the_socket():
    # regression: output oscout off→on must recreate the socket (else the send closure NPEs the thread).
    out = OscOut()
    out.setup(_Cfg(destinations="127.0.0.1:34568"))
    assert out._sock is not None
    out.stop()
    assert out._sock is None
    out.start(_IdleBus())
    try:
        assert out._sock is not None
        out._send(b"\x00\x00\x00\x00")
    finally:
        out.stop()


def test_object_id_on_the_wire_is_the_tracker_id_even_when_a_slot_exists():
    """Owner decision: one id space everywhere — the tracker id, never the recycled slot."""
    from dataclasses import replace

    out = OscOut()
    out.setup(None, send_fn=lambda b: None)
    o = replace(_obj(42), slot=2)
    msgs = _decode(out.build_packet(_snap([o])))
    assert [m for m in msgs if m[0].endswith("/pose")][0][1][0] == 42
    assert list([m for m in msgs if m[0].endswith("/alv")][0][1]) == [42]


def test_contour_prefers_the_perceptual_wire_contour():
    # with a wire_contour present (the inverse-EFD perceptual outline), /rmt/contour emits IT placed
    # at the pose, with NO approxPolyDP — the [efd] detail knob already set its quality.
    wc = np.array([[[6, 6]], [[14, 6]], [[14, 14]], [[6, 14]]], np.int32)   # rest square around (10,10)
    o = ObjectSnap(id=7, status=Status.MOVED, ignored=False, centroid=(10.0, 10.0),
                   bbox=(6, 6, 8, 8), area=64.0, contour=None, pose=(50.0, 10.0, 0.0),
                   footprint_contour=None, footprint_centroid=(10.0, 10.0), wire_contour=wc,
                   annotations={})
    d = dict(_decode(_out(send="pose,contour", coord_mode="pixel").build_packet(_snap([o]))))
    args = d["/rmt/contour"]
    assert args[0] == 7 and len(args) == 1 + 2 * len(wc)   # every perceptual vertex survives (no simplify)
    xs = args[1::2]
    assert min(xs) >= 45 and max(xs) <= 55                 # placed at the POSE (x=50), not the rest spot


# --------------------------------------------------------------------------- #
# bundle timetag
# --------------------------------------------------------------------------- #

def test_bundle_timetag_is_real_unix_time_with_exact_frame_spacing():
    # snap.ts is CAPTURE time on an arbitrary base (video PTS / perf_counter), NOT Unix time —
    # which is what TimeTag encodes. Sent raw, a clip's 3.4 s PTS lands in 1970. The base is
    # anchored to the wall clock once, so the ABSOLUTE value is real and the RELATIVE spacing
    # between frames stays exact.
    import time as _t
    from lib.osc import TimeTag

    out = _out()
    before = _t.time()
    t0 = out._timetag(_snap([_obj(1)], ts=3.4)).seconds       # a video PTS near the clip start
    after = _t.time()
    assert before <= t0 <= after, "timetag is not real wall-clock time"

    t1 = out._timetag(_snap([_obj(1)], ts=3.4 + 0.25)).seconds
    assert t1 - t0 == pytest.approx(0.25, abs=1e-9)           # capture spacing preserved exactly

    # and it survives the encode: NTP seconds must land in the present, not 1970
    ntp_1970 = TimeTag(0.0).to_bytes()
    assert TimeTag(t0).to_bytes() != ntp_1970


def test_bundle_timetag_immediate_when_no_capture_stamp():
    assert _out()._timetag(_snap([_obj(1)], ts=None)).seconds is None


def test_obj_state_message_makes_the_native_stream_complete():
    """/rmt/obj (state + pose confidence) now rides oscout itself — before 0.32.123 it existed
    only inside the tuio2 bundle, so a pure-/rmt client could not tell RESTING from HIDDEN."""
    d = dict(_decode(_out().build_packet(_snap([_obj(3)]))))
    assert d["/rmt/obj"] == [3, "RESTING", 0.0]
    only_feats = dict(_decode(_out(send="features").build_packet(_snap([_obj(3)]))))
    assert "/rmt/obj" not in only_feats                   # gated: part of `all`, or the `obj` token


def test_enobufs_gets_one_retry_then_a_throttled_report():
    """macOS answers a full egress queue (Wi-Fi hop) with ENOBUFS. The sender must retry once
    after a 1 ms breather (it runs on its own thread), isolate failures per destination, and
    THROTTLE the warning - one line per window, not one per dropped datagram."""
    import errno as _errno
    import types

    o = OscOut.__new__(OscOut)
    o._dests = [("127.0.0.1", 9999), ("10.0.0.9", 9999)]
    o._drops = 0
    o._drop_logged = 0.0
    warned = []
    o._log = types.SimpleNamespace(warning=lambda m: warned.append(m))

    calls = []
    class _RetrySock:
        def sendto(self, data, dest):
            calls.append(dest)
            if len(calls) == 1:                       # first attempt to dest 1: queue full
                raise OSError(_errno.ENOBUFS, "No buffer space available")
    o._sock = _RetrySock()
    o._send_packet(b"x" * 16)
    assert len(calls) == 3                            # dest1 fail + dest1 retry OK + dest2
    assert o._drops == 0 and not warned               # the retry saved it: nothing to report

    class _DeadSock:
        def sendto(self, data, dest):
            raise OSError(_errno.ENOBUFS, "No buffer space available")
    o._sock = _DeadSock()
    for _ in range(5):
        o._send_packet(b"x" * 16)                     # 5 packets x 2 dests, all dropped
    assert len(warned) == 1                           # ONE throttled line, not ten
    assert "dropped since last report" in warned[0]
    assert o._drops < 10                              # the counter reset at the report
