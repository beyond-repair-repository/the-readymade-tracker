"""WebGui bridge (roadmap C) — the OSC round-trip (browser → OSCIn → collect), the SSE snapshot
serialization, and a real HTTP end-to-end (static + /info + POST /osc bridged to a fake OSCIn)."""

import json
import socket
import threading
import time
import urllib.request
from types import SimpleNamespace

import numpy as np
import pytest

from lib.osc import encode_message
from plugins.control.webgui.webgui import WebGui, _decode_args


class _Cfg:
    def __init__(self, **kw):
        self._d = kw
    def get(self, k, d=None):
        return self._d.get(k, d)


class _NullBus:
    def wait_fresh(self, last_idx, timeout):
        time.sleep(min(timeout, 0.01))
        return None
    def latest(self):
        return None


def _fake_oscin(reply_msgs):
    """A UDP socket that, on any datagram, replies with the given encoded messages to the sender.
    Returns (port, stop_fn)."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    srv.bind(("127.0.0.1", 0))
    srv.settimeout(0.3)
    port = srv.getsockname()[1]

    def run():
        while True:
            try:
                _data, addr = srv.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break
            for m in reply_msgs:
                srv.sendto(m, addr)

    threading.Thread(target=run, daemon=True).start()
    return port, srv.close


# --------------------------------------------------------------------------- #
# POST /osc → OSCIn round-trip
# --------------------------------------------------------------------------- #

def test_fire_and_forget_returns_immediately():
    w = WebGui()
    w.setup(_Cfg())
    w.attach_bridge(oscin_port=59999, prefix="/rmt", mjpeg_port=8080)   # nothing listening
    t0 = time.perf_counter()
    assert w.osc_roundtrip("/rmt/pause", []) == []    # a control verb waits for no reply
    assert time.perf_counter() - t0 < 0.5


def test_query_collects_until_the_params_terminator():
    port, close = _fake_oscin([
        encode_message("/rmt/param", ["observe", "k", "float", 1, "0.3", "0.3", "0.0", "1.0", "", "gate"]),
        encode_message("/rmt/params", [1]),
    ])
    w = WebGui()
    w.setup(_Cfg())
    w.attach_bridge(oscin_port=port, prefix="/rmt", mjpeg_port=8080)
    try:
        msgs = w.osc_roundtrip("/rmt/query", [])
        addrs = [m["address"] for m in msgs]
        assert "/rmt/param" in addrs and "/rmt/params" in addrs
        assert msgs[0]["args"][0] == "observe" and msgs[0]["args"][1] == "k"
    finally:
        close()


def test_query_collects_across_an_inter_packet_gap():
    """The chunked /query reply can arrive as several datagrams paced >0.2s apart under load; the
    collector must wait for the /params terminator, NOT break on the first gap (else enum rows drop)."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    srv.bind(("127.0.0.1", 0))
    srv.settimeout(0.5)
    port = srv.getsockname()[1]
    chunks = [encode_message("/rmt/param", ["gui", "view", "str", 1, "debug", "debug", "", "",
                                             "input,debug", "the view"]),
              encode_message("/rmt/params", [1])]

    def run():
        try:
            _data, addr = srv.recvfrom(65535)
        except OSError:
            return
        srv.sendto(chunks[0], addr)
        time.sleep(0.35)                                   # gap wider than the 0.2s socket timeout
        srv.sendto(chunks[1], addr)

    threading.Thread(target=run, daemon=True).start()
    w = WebGui()
    w.setup(_Cfg())
    w.attach_bridge(oscin_port=port, prefix="/rmt", mjpeg_port=8080)
    try:
        msgs = w.osc_roundtrip("/rmt/query", [])
        addrs = [m["address"] for m in msgs]
        assert "/rmt/param" in addrs and "/rmt/params" in addrs   # the delayed terminator survived
        enum = next(m for m in msgs if m["address"] == "/rmt/param")["args"][8]
        assert enum == "input,debug"                               # the enum row (dropdown) survived
    finally:
        srv.close()


def test_snapshot_json_places_footprint_at_the_live_pose():
    """The overlay contour must be the rest outline PLACED at the pose (as the gui/tuio2 do), not the
    frozen rest coordinates — else it appears to lag behind a moving object."""
    fp = np.array([[[6, 6]], [[14, 6]], [[14, 14]], [[6, 14]]], np.int32)   # rest square around (10,10)
    o = SimpleNamespace(id=1, ignored=False, pose=(50.0, 10.0, 0.0), status=SimpleNamespace(name="MOVED"),
                        footprint_contour=fp, footprint_centroid=(10.0, 10.0), annotations=None)
    snap = SimpleNamespace(frame_idx=1, ts=0.0, frame=np.zeros((100, 100, 3), np.uint8), objects=(o,))
    d = json.loads(WebGui().snapshot_json(snap))
    xs = [p[0] * 100 for p in d["objects"][0]["contour"]]   # denormalize x (frame width 100)
    assert min(xs) >= 45 and max(xs) <= 55                  # at x≈50 (the POSE), NOT x≈10 (the rest spot)


def test_snapshot_json_includes_stats_view_and_rec_when_attached():
    o = SimpleNamespace(id=1, ignored=False, pose=(10.0, 10.0, 0.0),
                        status=SimpleNamespace(name="RESTING"), footprint_contour=None, annotations=None)
    snap = SimpleNamespace(frame_idx=3, ts=123.5, frame=np.zeros((100, 100, 3), np.uint8), objects=(o,))
    w = WebGui()
    bare = json.loads(w.snapshot_json(snap))               # nothing attached → only scene fields
    assert "proc_fps" not in bare and "view" not in bare and "rec" not in bare
    assert bare["ts"] == 123.5
    w.attach_stats(lambda: 42.4)
    w.attach_mjpeg(SimpleNamespace(current_view=lambda: "debug"))
    w.attach_recorder(SimpleNamespace(status=lambda: {"recording": True, "path": "x.mp4",
                                                      "elapsed": 3.0, "written": 90, "dropped": 0}))
    d = json.loads(w.snapshot_json(snap))
    assert d["proc_fps"] == 42.4 and d["view"] == "debug"
    assert d["rec"]["recording"] is True and d["rec"]["elapsed"] == 3.0


def test_snapshot_json_serializes_compound_groups():
    """The Grouper's compound super-objects ride the SSE payload under `groups` (normalized centroid +
    members + merged contour) so the client can draw them. Absent when the scene is under budget."""
    o = SimpleNamespace(id=1, ignored=False, pose=(10.0, 10.0, 0.0),
                        status=SimpleNamespace(name="RESTING"), footprint_contour=None, annotations=None)
    frame = np.zeros((100, 200, 3), np.uint8)                          # h=100, w=200
    g = SimpleNamespace(id=5, members=(1, 2, 3), centroid=(100.0, 50.0), area=900.0, size=0.42, theta=1.5,
                        contour=np.array([[[80, 40]], [[120, 40]], [[120, 60]], [[80, 60]]], np.int32))
    snap = SimpleNamespace(frame_idx=1, ts=0.0, frame=frame, objects=(o,), groups=(g,))
    d = json.loads(WebGui().snapshot_json(snap))
    assert len(d["groups"]) == 1
    grp = d["groups"][0]
    assert grp["id"] == 5 and grp["members"] == [1, 2, 3] and grp["n"] == 3
    assert grp["x"] == pytest.approx(0.5) and grp["y"] == pytest.approx(0.5)   # (100/200, 50/100)
    assert grp["size"] == 0.42 and grp["theta"] == 1.5   # SIZE (0..1), not the raw summed area
    assert len(grp["contour"]) == 4 and grp["contour"][0] == [pytest.approx(0.4), pytest.approx(0.4)]
    # under budget → no groups → the key is absent (the client sees scene.groups undefined)
    bare = json.loads(WebGui().snapshot_json(SimpleNamespace(frame_idx=1, ts=0.0, frame=frame, objects=(o,))))
    assert "groups" not in bare


def test_decode_args_typed_and_plain():
    assert _decode_args([{"t": "i", "v": 3}, {"t": "f", "v": 0.5}, {"t": "s", "v": "x"}]) == [3, 0.5, "x"]
    assert _decode_args([1, "a", True]) == [1, "a", 1]     # plain pass-through; bool → int (no OSC bool)


def test_snapshot_json_includes_frame_dims():
    o = SimpleNamespace(id=1, ignored=False, pose=(5.0, 5.0, 0.0), status=SimpleNamespace(name="RESTING"),
                        footprint_contour=None, annotations=None)
    snap = SimpleNamespace(frame_idx=1, ts=0.0, frame=np.zeros((200, 400, 3), np.uint8), objects=(o,))
    d = json.loads(WebGui().snapshot_json(snap))
    assert d["fdim"] == [400, 200]                        # the client places the crop-rect overlay in raw px


def test_snapshot_json_mask_circle_from_live_preprocess():
    from lib.preprocess import Preprocess, PreprocessConfig
    o = SimpleNamespace(id=1, ignored=False, pose=(10.0, 10.0, 0.0), status=SimpleNamespace(name="RESTING"),
                        footprint_contour=None, annotations=None)
    snap = SimpleNamespace(frame_idx=1, ts=0.0, frame=np.zeros((200, 400, 3), np.uint8), objects=(o,))  # 400×200
    w = WebGui()
    assert "mask" not in json.loads(w.snapshot_json(snap))           # no preprocess attached → no mask
    # crop origin (40,20); the mask centre is offset by it and normalized to the CROPPED-frame space
    pp = Preprocess(PreprocessConfig(crop_enabled=True, crop_top_x=40, crop_top_y=20,
                                     mask_enabled=True, mask_cx=240, mask_cy=120, mask_radius=80))
    w.attach_preprocess(pp)
    d = json.loads(w.snapshot_json(snap))                            # (240-40)/400, (120-20)/200, 80/400
    assert d["mask"] == [pytest.approx(0.5), pytest.approx(0.5), pytest.approx(0.2)]
    pp.apply_param("mask_enabled", False)                            # live disable → the circle drops out
    assert "mask" not in json.loads(w.snapshot_json(snap))


# --------------------------------------------------------------------------- #
# SSE snapshot serialization
# --------------------------------------------------------------------------- #

def test_snapshot_json_normalizes_and_includes_annotations():
    o = SimpleNamespace(id=5, ignored=False, pose=(50.0, 25.0, 1.5),
                        status=SimpleNamespace(name="MOVED"), footprint_contour=None,
                        bbox=(40, 15, 20, 20),
                        annotations={"features": {"solidity": [0.9]}})
    snap = SimpleNamespace(frame_idx=7, frame=np.zeros((50, 100, 3), np.uint8), objects=(o,))
    d = json.loads(WebGui().snapshot_json(snap))
    assert d["frame"] == 7 and len(d["objects"]) == 1
    ob = d["objects"][0]
    assert ob["id"] == 5 and ob["state"] == "MOVED"
    assert abs(ob["x"] - 0.5) < 1e-9 and abs(ob["y"] - 0.5) < 1e-9    # 50/100, 25/50
    assert ob["bbox"] == [pytest.approx(0.4), pytest.approx(0.3),     # x,w ÷ width; y,h ÷ height
                          pytest.approx(0.2), pytest.approx(0.4)]
    assert ob["ann"]["features"]["solidity"] == [0.9]


def test_snapshot_json_includes_cluster_table():
    # clusters ride a top-level `clusters` array (cluster-centric), NOT a per-object annotation.
    from lib.scene import ClusterSnap
    hull = np.array([[[40, 15]], [[60, 15]], [[60, 35]]], np.int32)
    c = ClusterSnap(id=4, members=(5, 7), centroid=(50.0, 25.0),
                    bbox=(40.0, 15.0, 20.0, 20.0), circle=(50.0, 25.0, 14.0), hull=hull)
    snap = SimpleNamespace(frame_idx=7, frame=np.zeros((50, 100, 3), np.uint8), objects=(), clusters=(c,))
    d = json.loads(WebGui().snapshot_json(snap))
    assert len(d["clusters"]) == 1
    cl = d["clusters"][0]
    assert cl["id"] == 4 and cl["members"] == [5, 7] and cl["n"] == 2
    assert cl["cx"] == pytest.approx(0.5) and cl["cy"] == pytest.approx(0.5)   # 50/100, 25/50
    assert cl["circle"] == [pytest.approx(0.5), pytest.approx(0.5), pytest.approx(0.14)]   # r by width
    assert cl["bbox"] == [pytest.approx(0.4), pytest.approx(0.3), pytest.approx(0.2), pytest.approx(0.4)]
    assert len(cl["hull"]) == 3


# --------------------------------------------------------------------------- #
# the HTTP server end-to-end
# --------------------------------------------------------------------------- #

def test_http_serves_ui_info_and_bridges_a_set():
    port, close = _fake_oscin([encode_message("/rmt/ack", ["observe", "k", "ok", "0.5", 0])])
    w = WebGui()
    w.setup(_Cfg(host="127.0.0.1", port=0))
    w.attach_bridge(oscin_port=port, prefix="/rmt", mjpeg_port=8080)
    w.start(_NullBus())
    try:
        hp = w._srv.server_address[1]
        base = f"http://127.0.0.1:{hp}"
        info = json.loads(urllib.request.urlopen(base + "/info", timeout=2).read())
        assert info["mjpeg_port"] == 8080 and info["prefix"] == "/rmt"
        idx = urllib.request.urlopen(base + "/", timeout=2).read()
        assert b"ReadyMade Tracker" in idx                                          # index.html served
        req = urllib.request.Request(base + "/osc", method="POST",
            data=json.dumps({"address": "/rmt/set", "args": ["observe", "k", 0.5]}).encode(),
            headers={"Content-Type": "application/json"})
        resp = json.loads(urllib.request.urlopen(req, timeout=2).read())
        assert any(m["address"] == "/rmt/ack" and m["args"][2] == "ok" for m in resp["messages"])
    finally:
        w.stop()
        close()


def test_http_stop_is_idempotent():
    w = WebGui()
    w.setup(_Cfg(host="127.0.0.1", port=0))
    w.start(_NullBus())
    w.stop()
    w.stop()                                                          # second stop must not raise


def test_snapshot_json_prefers_the_wire_contour():
    wc = np.array([[[45, 20]], [[55, 20]], [[55, 30]], [[45, 30]]], np.int32)
    o = SimpleNamespace(id=5, ignored=False, pose=(50.0, 25.0, 0.0),
                        status=SimpleNamespace(name="RESTING"), footprint_contour=None,
                        footprint_centroid=(50.0, 25.0), wire_contour=wc, annotations={})
    snap = SimpleNamespace(frame_idx=1, frame=np.zeros((50, 100, 3), np.uint8), objects=(o,))
    d = json.loads(WebGui().snapshot_json(snap))
    assert len(d["objects"][0]["contour"]) == len(wc)      # the perceptual outline, un-simplified
    assert "efd_full" not in d["objects"][0]               # absent field → absent key (older snaps degrade)


def test_snapshot_json_carries_the_full_spectrum():
    # efd_full (the complete ± spectrum, the inspector's dim rejected bins) rides the SSE JSON —
    # the one channel it exists for; oscout/tuio2 never read it.
    full = ((1.0, 0.25, 0.004), (0.5, 0.0, 0.002))
    o = SimpleNamespace(id=5, ignored=False, pose=(50.0, 25.0, 0.0),
                        status=SimpleNamespace(name="RESTING"), footprint_contour=None,
                        footprint_centroid=(50.0, 25.0), wire_contour=None,
                        efd_full=full, annotations={})
    snap = SimpleNamespace(frame_idx=1, frame=np.zeros((50, 100, 3), np.uint8), objects=(o,))
    d = json.loads(WebGui().snapshot_json(snap))
    assert d["objects"][0]["efd_full"] == [[1.0, 0.25, 0.004], [0.5, 0.0, 0.002]]


def test_snapshot_json_carries_the_emit_view_for_the_features_inspector():
    """Owner decision: the FEATURES view follows the wire. When the grouper fuses, snapshot_json
    appends the compound as a grp-flagged object row and lists the emit ids — the client's
    inspector shows exactly the voice list, while scene.objects keeps the real scene."""
    import numpy as np
    from lib.scene import Detection, ObjectSnap, Scene, Status
    s = Scene()
    s.begin_tick(0, ts=0.0)
    ids = []
    with s.writing("setup"):
        for x in (100.0, 140.0):
            ids.append(s.mint(Detection((x, 100.0), (int(x) - 5, 95, 10, 10), 100.0, None)))
    with s.writing("pose"):
        for oid, x in zip(ids, (100.0, 140.0)):
            s.set(oid, "pose", (x, 100.0, 0.0))
    with s.writing("classify"):
        for oid in ids:
            s.set(oid, "confirmed", True)
    comp = ObjectSnap(id=99, status=Status.RESTING, ignored=False, centroid=(120.0, 100.0),
                      bbox=(110, 90, 20, 20), area=200.0, contour=None, pose=(120.0, 100.0, 0.0),
                      annotations={"features": {"size": [0.4]}})
    with s.writing("grouper"):
        s.set_fused(frozenset(ids), (comp,))
    snap = s.snapshot()
    import json
    d = json.loads(WebGui().snapshot_json(snap))
    assert d["emit"] == [99]                                  # the wire's voice list
    rows = {o["id"]: o for o in d["objects"]}
    assert rows[99].get("grp") is True                        # the compound, flagged for the client
    assert rows[99]["ann"]["features"]["size"] == [0.4]       # with its REAL features
    assert ids[0] in rows and "grp" not in rows[ids[0]]       # the real scene stays serialized too


def test_snapshot_json_marks_hidden_rows_covered_or_gone():
    """identity's covered/bared verdict rides the snapshot: a HIDDEN row carries present=true when
    a hand still occupies its spot (the band keeps it) and present=false when GONE. Non-HIDDEN
    rows carry no flag, and the output WIRES do not consume the channel at all (owner's rule)."""
    import json
    from lib.scene import Detection, Scene, Status
    s = Scene()
    s.begin_tick(0, ts=0.0)
    with s.writing("setup"):
        covered = s.mint(Detection((100.0, 100.0), (95, 95, 10, 10), 100.0, None))
        gone = s.mint(Detection((200.0, 100.0), (195, 95, 10, 10), 100.0, None))
        resting = s.mint(Detection((300.0, 100.0), (295, 95, 10, 10), 100.0, None))
        for oid in (covered, gone):                     # same writer as the mint: the single-writer
            s.set_status(oid, Status.HIDDEN)            #   guard is per (id, field, tick)
    with s.writing("pose"):
        for oid, x in ((covered, 100.0), (gone, 200.0), (resting, 300.0)):
            s.set(oid, "pose", (x, 100.0, 0.0))
    with s.writing("identity"):
        s.set_hidden_present(frozenset({covered}))
    d = json.loads(WebGui().snapshot_json(s.snapshot()))
    rows = {o["id"]: o for o in d["objects"]}
    assert rows[covered]["present"] is True
    assert rows[gone]["present"] is False
    assert "present" not in rows[resting]
