"""tuio2_monitor — the receiver, tested against tuio2's OWN output.

Closes the encode→decode loop in-process: Tuio2Out.build_packets → TuioState.update_packet,
so the wire contract (ids, positions, contours, state, alive pruning, multi-packet
reassembly) is verified with no socket and no pipeline. render_frame is smoke-tested
(it draws to an ndarray; the cv2 window loop is the only untested part).
"""

import numpy as np
import pytest

from lib.osc import encode_bundle, encode_message
from lib.scene import ObjectSnap, SceneSnapshot, Status
from plugins.output.tuio2.tuio2 import Tuio2Out
from tools.tuio2_monitor import _CLUSTER_COLORS, TuioState, canvas_size, record_path, render_frame, snapshot_path


def _osc(*msgs):
    return encode_bundle([encode_message(a, args) for a, args in msgs])


def _square(cx, cy, r):
    return np.array([[[cx - r, cy - r]], [[cx + r, cy - r]],
                     [[cx + r, cy + r]], [[cx - r, cy + r]]], dtype=np.int32)


_UNSET = object()


def _obj(oid, status=Status.RESTING, pose=(20.0, 17.5, 1.5), contour=None,
         ignored=False, conf=0.9, footprint=_UNSET):
    # captured by default (has a footprint → Tuio2Out emits it). footprint=None = un-captured.
    fp = _square(20, 17, 8) if footprint is _UNSET else footprint
    fpc = (20.0, 17.5) if fp is not None else None
    return ObjectSnap(id=oid, status=status, ignored=ignored, centroid=(20.0, 17.5),
                      bbox=(10, 10, 20, 15), area=300.0,
                      contour=_square(20, 17, 8) if contour is None else contour,
                      pose=pose, pose_conf=conf, footprint_contour=fp, footprint_centroid=fpc)


def _snap(objects, w=100, h=50, frame_idx=7, ts=1.5):
    return SceneSnapshot(frame_idx=frame_idx, ts=ts, objects=tuple(objects),
                         frame=np.zeros((h, w, 3), np.uint8))


def _out(**cfg):
    o = Tuio2Out()
    o.setup(cfg, send_fn=lambda data: None)
    return o


def _fold(state, packets):
    for p in packets:
        state.update_packet(p)


# --------------------------------------------------------------------------- #
# the encode → decode round trip
# --------------------------------------------------------------------------- #

def test_receives_token_state_and_contour():
    # include_custom: the sidecar is OPT-IN since 0.32.124 (its home is oscout; the monitor folds
    # /rmt/obj from either socket) — this test exercises the TUIO-only bridge explicitly.
    out = _out(contour_source="live", include_custom=True)   # assert on the live contour coords
    st = TuioState()
    _fold(st, out.build_packets(_snap([_obj(5, status=Status.MOVED, conf=0.42)])))
    assert set(st.objects) == {5}
    o = st.objects[5]
    assert o["x"] == pytest.approx(20.0 / 100) and o["y"] == pytest.approx(17.5 / 50)  # normalized
    assert o["angle"] == pytest.approx(1.5)                # radians, verbatim
    assert o["state"] == "MOVED" and o["conf"] == pytest.approx(0.42)
    assert len(o["contour"]) == 4                          # the square's 4 corners
    assert o["contour"][0] == (pytest.approx(12 / 100), pytest.approx(9 / 50))
    assert st.src_dim == (100, 50) and st.source == "readymade"


def test_multiple_objects():
    out = _out()
    st = TuioState()
    _fold(st, out.build_packets(_snap([_obj(1), _obj(2), _obj(3)])))
    assert set(st.objects) == {1, 2, 3}


def test_uncaptured_and_ignored_not_received():
    # a RESTING captured object is received; an un-captured HIDDEN ghost + an ignored one aren't.
    out = _out()
    st = TuioState()
    _fold(st, out.build_packets(_snap([_obj(1),
                                       _obj(2, status=Status.HIDDEN, footprint=None),
                                       _obj(3, ignored=True)])))
    assert set(st.objects) == {1}


def test_alive_pruning_drops_departed_object():
    out = _out()
    st = TuioState()
    _fold(st, out.build_packets(_snap([_obj(1), _obj(2)], frame_idx=1)))
    assert set(st.objects) == {1, 2}
    _fold(st, out.build_packets(_snap([_obj(1)], frame_idx=2)))   # 2 gone from the alive set
    assert set(st.objects) == {1}


def test_reassembly_across_chunked_packets():
    # a tiny cap splits one frame into many packets; the receiver must still end up
    # with every object + its contour (components fold, /alv prunes nothing here).
    out = _out(max_packet_bytes=250)
    objs = [_obj(i, contour=_square(20, 17, 8)) for i in range(1, 13)]
    packets = out.build_packets(_snap(objs))
    assert len(packets) > 1                                # actually chunked
    st = TuioState()
    _fold(st, packets)
    assert set(st.objects) == set(range(1, 13))
    assert all(len(st.objects[i]["contour"]) == 4 for i in range(1, 13))


def test_pixel_coord_mode_roundtrip():
    out = _out(coord_mode="pixel")
    st = TuioState()
    _fold(st, out.build_packets(_snap([_obj(5)])))
    assert st.objects[5]["x"] == pytest.approx(20.0) and st.objects[5]["y"] == pytest.approx(17.5)


def test_malformed_packet_ignored():
    st = TuioState()
    st.update_packet(b"not an osc packet at all")           # must not raise
    st.update_packet(b"")
    assert st.objects == {}


# --------------------------------------------------------------------------- #
# rendering (no window)
# --------------------------------------------------------------------------- #

def test_render_frame_shape_and_no_crash():
    out = _out()
    st = TuioState()
    _fold(st, out.build_packets(_snap([_obj(5, status=Status.MOVED)])))
    img = render_frame(st, 640, 480)
    assert img.shape == (480, 640, 3) and img.dtype == np.uint8
    assert img.sum() > 0                                    # something was drawn


def test_render_empty_state():
    img = render_frame(TuioState(), 320, 240)               # waiting screen, no objects
    assert img.shape == (240, 320, 3)


def test_outline_width_controls_contour_thickness():
    # thicker outlines paint more contour ink; default is 2px, floored at 1.
    out = _out(contour_source="live")
    st = TuioState()
    _fold(st, out.build_packets(_snap([_obj(5, status=Status.MOVED)])))
    thin = render_frame(st, 400, 300, outline_width=1)
    thick = render_frame(st, 400, 300, outline_width=6)
    assert thick.shape == thin.shape
    assert int(thick.sum()) > int(thin.sum())          # a heavier line = more coloured pixels


def test_font_scale_draws_bigger_labels():
    # --font-size multiplier: a larger scale paints more label ink (bigger glyphs + chips).
    out = _out()
    st = TuioState()
    _fold(st, out.build_packets(_snap([_obj(5, status=Status.MOVED)])))
    small = render_frame(st, 640, 480, font_scale=1.0)
    big = render_frame(st, 640, 480, font_scale=2.0)
    assert big.shape == small.shape
    assert int(big.sum()) > int(small.sum())


def test_canvas_size_matches_source_aspect():
    w, h = canvas_size((957, 1024), None, 900)          # portrait source → keep aspect, fit 900
    assert h == 900 and abs(w / h - 957 / 1024) < 0.01
    w2, h2 = canvas_size((1280, 720), None, 900)         # landscape → width caps at 900
    assert w2 == 900 and abs(w2 / h2 - 1280 / 720) < 0.01
    assert canvas_size((957, 1024), (512, 512), 900) == (512, 512)   # explicit --size wins
    assert canvas_size((0, 0), None, 900)[0] == 900      # no frame dim yet → provisional


def test_coord_mode_autodetect():
    from tools.tuio2_monitor import _resolve_mode
    norm = TuioState(); norm.objects = {1: {"x": 0.5, "y": 0.5}}
    px = TuioState(); px.objects = {1: {"x": 640.0, "y": 360.0}}
    assert _resolve_mode(norm, "auto") == "normalized"
    assert _resolve_mode(px, "auto") == "pixel"
    assert _resolve_mode(px, "normalized") == "normalized"  # explicit overrides autodetect


# --------------------------------------------------------------------------- #
# 'r' clear + 's' snapshot
# --------------------------------------------------------------------------- #

def test_reset_clears_scene():
    out = _out()
    st = TuioState()
    _fold(st, out.build_packets(_snap([_obj(1), _obj(2)])))
    assert st.objects and st.frame_id is not None
    st.reset()                                          # the 'r' key
    assert st.objects == {} and st.frame_id is None and st.packets == 0


def test_snapshot_path_is_timestamped_png(tmp_path):
    p = snapshot_path(str(tmp_path), when="20260708-131400")
    assert p == str(tmp_path / "tuio2_20260708-131400.png")
    assert p.endswith(".png")


def test_record_path_is_timestamped_jsonl(tmp_path):
    p = record_path(str(tmp_path), when="20260708-131400")
    assert p == str(tmp_path / "tuio2_20260708-131400.jsonl")


def test_recording_property_toggles(tmp_path):
    st = TuioState()
    assert st.recording is False
    st.start_recording(str(tmp_path / "a.jsonl"))
    assert st.recording is True
    st.stop_recording()
    assert st.recording is False


def test_recording_writes_decoded_jsonl(tmp_path):
    import json
    out = _out()
    st = TuioState()
    rec = tmp_path / "rec.jsonl"
    st.start_recording(str(rec))
    for p in out.build_packets(_snap([_obj(5, status=Status.MOVED)])):
        st.update_packet(p, now=1.5)
    st.stop_recording()
    lines = [json.loads(ln) for ln in rec.read_text().splitlines() if ln.strip()]
    assert lines
    addrs = {m[0] for rl in lines for m in rl["msgs"]}
    assert "/tuio2/frm" in addrs and "/tuio2/tok" in addrs
    frm = [m for rl in lines for m in rl["msgs"] if m[0] == "/tuio2/frm"][0]
    assert any(isinstance(a, dict) and "timetag" in a for a in frm[1])   # TimeTag serialized


def test_snapshot_roundtrips_a_rendered_frame(tmp_path):
    # what the 's' key does: render_frame -> cv2.imwrite -> a readable PNG.
    import cv2
    out = _out()
    st = TuioState()
    _fold(st, out.build_packets(_snap([_obj(5, status=Status.MOVED)])))
    img = render_frame(st, 320, 240)
    path = snapshot_path(str(tmp_path), when="t")
    assert cv2.imwrite(path, img)
    back = cv2.imread(path)
    assert back is not None and back.shape == (240, 320, 3)


# --------------------------------------------------------------------------- #
# oscout cluster overlay: /rmt/cluster/* (--show-clusters)
# --------------------------------------------------------------------------- #

def test_folds_cluster_messages():
    st = TuioState()
    st.update_packet(_osc(
        ("/rmt/frm", [3, 1]),
        ("/rmt/cluster", [2, 3, 0.5, 0.5]),               # cid=2, n=3, centroid (0.5, 0.5)
        ("/rmt/cluster/circle", [2, 0.5, 0.5, 0.2]),
        ("/rmt/cluster/members", [2, 5, 7, 9]),
    ))
    c = st.clusters[2]                                     # a table keyed by cluster id, not per-object
    assert c["n"] == 3 and c["cx"] == pytest.approx(0.5)
    assert c["circle"] == pytest.approx((0.5, 0.5, 0.2))
    assert c["members"] == [5, 7, 9]


def test_oscout_frame_clears_stale_cluster():
    st = TuioState()
    st.update_packet(_osc(("/rmt/frm", [1, 1]), ("/rmt/cluster", [3, 2, 0.4, 0.4])))
    assert 3 in st.clusters                                # folded
    st.update_packet(_osc(("/rmt/frm", [2, 0])))          # next frame: no cluster
    assert st.clusters == {}                               # stale table cleared, doesn't linger


def test_show_clusters_draws_ring_in_cluster_colour():
    st = TuioState()
    st.clusters = {1: {"n": 2, "cx": 0.5, "cy": 0.5, "circle": (0.5, 0.5, 0.2)}}
    img = render_frame(st, 200, 200, coord_mode="normalized", show_clusters=True)
    col = np.array(_CLUSTER_COLORS[0])                     # cluster id 1
    near = np.all(np.abs(img.astype(int) - col) < 60, axis=2)
    assert near.sum() > 50                                 # a circle @ (100,100) r40 + the C1 chip
    assert near[136:145, 95:106].any()                     # ring bottom ≈ (100,140), away from the chip


def test_show_clusters_off_draws_no_ring():
    st = TuioState()
    st.clusters = {1: {"circle": (0.5, 0.5, 0.2)}}
    img = render_frame(st, 200, 200, coord_mode="normalized", show_clusters=False)
    col = np.array(_CLUSTER_COLORS[0])
    assert not np.all(np.abs(img.astype(int) - col) < 60, axis=2).any()   # no ring when off


# --------------------------------------------------------------------------- #
# the activity dot (a compound is an ORDINARY object on this wire — no /grp table)
# --------------------------------------------------------------------------- #

def test_old_grp_addresses_fold_generically_not_fatally():
    st = TuioState()                                                    # the wire was deleted at 0.32.102;
    st.update_packet(encode_message("/rmt/grp", [5, 3, 0.4, 0.5, 1200.0, 0.7]), now=1.0)   # an old tape
    assert not hasattr(st, "groups")                                    # no dedicated table anymore — the
    #                                                                     generic /rmt/* folder may keep the
    #                                                                     value as a pseudo-feature; no crash


def test_render_draws_the_activity_dot():
    pytest.importorskip("cv2")
    st = TuioState()
    st.update_packet(encode_message("/tuio2/frm", [1, 0, (100 << 16) | 50, "x"]), now=0.0)
    for active in (True, False, None):
        img = render_frame(st, 200, 100, active=active)
        assert img.shape == (100, 200, 3)


def test_osc_tape_round_trip(tmp_path):
    # the raw tape replays byte-faithfully with source tags + monotonic times
    import tools.tuio2_monitor as m
    p = str(tmp_path / "t.bin")
    f = m.tape_open(p)
    m.tape_append(f, 0.0, 0, b"alpha")
    m.tape_append(f, 0.5, 1, b"beta-beta")
    m.tape_append(f, 1.25, 0, b"\x00\x01\x02")
    f.close()
    got = list(m.tape_read(p))
    assert got == [(0.0, 0, b"alpha"), (0.5, 1, b"beta-beta"), (1.25, 0, b"\x00\x01\x02")]


def test_osc_tape_rejects_a_non_tape_file(tmp_path):
    import pytest
    import tools.tuio2_monitor as m
    p = tmp_path / "junk.bin"
    p.write_bytes(b"not a tape at all")
    with pytest.raises(ValueError):
        list(m.tape_read(str(p)))


def test_state_tape_records_raw_packets(tmp_path):
    import tools.tuio2_monitor as m
    from lib.osc import encode_message
    st = m.TuioState()
    p = str(tmp_path / "cap.bin")
    st.start_tape(p)
    pkt = encode_message("/tuio2/alv", [1, 2])
    st.update_packet(pkt, now=1.0, src=0)
    st.stop_tape()
    (_, src, data), = list(m.tape_read(p))
    assert src == 0 and data == pkt


def test_generic_oscout_fold_lands_features_not_alv():
    # any /rmt/<field> <id> <v…> from oscout becomes a feature on the object; /rmt/alv never does.
    import tools.tuio2_monitor as m
    from lib.osc import encode_message
    st = m.TuioState()
    st.update_packet(encode_message("/rmt/eccentricity", [5, 0.42]), now=0.0, src=1)
    st.update_packet(encode_message("/rmt/pose", [5, 0.5, 0.5, 1.2]), now=0.0, src=1)
    st.update_packet(encode_message("/rmt/alv", [5, 9]), now=0.0, src=1)
    import pytest
    f = st.objects[5]["features"]
    assert f["eccentricity"][0] == pytest.approx(0.42) and f["pose"][2] == pytest.approx(1.2)
    assert "alv" not in f and 9 not in st.objects
