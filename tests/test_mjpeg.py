"""MJPEG video-out (roadmap B.2) — the encode/framing path, the enable-by-section wiring,
the monitor's JPEG extraction + background underlay, and a real localhost HTTP round-trip
(server thread → urllib client → a decodable JPEG), the analogue of the tuio2 UDP loopback.
"""

import time

import numpy as np
import pytest

import readymade_tracker
from lib.config import build_config
from plugins.output.mjpeg.mjpeg import MjpegOut, _multipart_chunk
from tools.tuio2_monitor import MjpegClient, _extract_jpegs, render_frame  # noqa: F401 (MjpegClient smoke)
from tools.tuio2_monitor import TuioState


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

class _Cfg:
    """A minimal Section stand-in: .get(key, default)."""
    def __init__(self, **kw):
        self._d = kw
    def get(self, k, d=None):
        return self._d.get(k, d)


class _Snap:
    def __init__(self, frame=None, composite=None, frame_idx=0, objects=()):
        self.frame, self.composite, self.frame_idx = frame, composite, frame_idx
        self.objects = objects


class _Bus:
    """A fake SnapshotBus: hands the snapshot out ONCE (fresh), then blocks-and-None like the
    real wait_fresh when the caller has already seen this frame_idx (no busy-spin)."""
    def __init__(self, snap):
        self._snap = snap
    def wait_fresh(self, last_idx, timeout):
        if last_idx != self._snap.frame_idx:
            return self._snap
        time.sleep(min(timeout, 0.02))
        return None
    def latest(self):
        return self._snap


def _out(**cfg):
    o = MjpegOut()
    o.setup(_Cfg(**cfg))
    return o


def _bgr(h, w, color):
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = color
    return img


# --------------------------------------------------------------------------- #
# encode: view selection + fallback + framing
# --------------------------------------------------------------------------- #

def test_encode_input_view_uses_raw_frame():
    o = _out(view="input")
    assert o.encode(_Snap(frame=_bgr(16, 16, (9, 9, 9)))) is not None
    assert o.encode(_Snap(frame=None)) is None                # None frame → nothing to send


def test_encode_none_when_no_image():
    assert _out(view="debug").encode(_Snap(frame=None, composite=None)) is None


def test_set_view_switches_the_streamed_view_at_runtime():
    o = _out(view="input")
    assert o._view == "input"
    o.set_view("debug")
    assert o._view == "debug"
    o.set_view("bogus")                               # unknown → ignored, view unchanged
    assert o._view == "debug"


def test_setup_rejects_an_unknown_view():
    assert _out(view="nonsense")._view == "debug"     # falls back to the default (debug)


def test_debug_view_encodes_single_width():
    import cv2
    snap = _Snap(frame=_bgr(48, 64, (10, 20, 30)))    # objects=() → overlay just draws the frame
    assert _out(view="debug").encode(snap)[:2] == b"\xff\xd8"          # a valid JPEG
    dbg = cv2.imdecode(np.frombuffer(_out(view="debug").encode(snap), np.uint8), cv2.IMREAD_COLOR)
    assert dbg.shape == (48, 64, 3)                    # single panel — NOT the 3-wide 3-fold


def test_scale_downsizes_the_streamed_frame():
    import cv2
    snap = _Snap(frame=_bgr(80, 120, (30, 60, 90)), composite=_bgr(80, 120, (30, 60, 90)))
    full = cv2.imdecode(np.frombuffer(_out(scale=1.0).encode(snap), np.uint8), cv2.IMREAD_COLOR)
    half = cv2.imdecode(np.frombuffer(_out(scale=0.5).encode(snap), np.uint8), cv2.IMREAD_COLOR)
    assert full.shape == (80, 120, 3)
    assert half.shape == (40, 60, 3)                 # 0.5 → half each dim
    assert _out(scale=0)._scale == 1.0               # non-positive scale → full size


def test_quality_is_clamped():
    assert _out(quality=9999)._quality == 100 and _out(quality=-5)._quality == 1


def test_multipart_chunk_framing():
    chunk = _multipart_chunk(b"\xff\xd8xx\xff\xd9", "bnd")
    assert chunk.startswith(b"--bnd\r\n")
    assert b"Content-Type: image/jpeg\r\n" in chunk
    assert b"Content-Length: 6\r\n\r\n" in chunk
    assert chunk.endswith(b"\r\n") and b"\xff\xd8xx\xff\xd9" in chunk


# --------------------------------------------------------------------------- #
# enable-by-section wiring (+ the --mjpeg-stream port override)
# --------------------------------------------------------------------------- #

def test_mjpeg_enabled_by_section_presence_and_port_coerced():
    cfg = build_config({"mjpeg": {"port": "4444"}}, schemas=readymade_tracker.CONF_SCHEMAS)
    assert cfg.is_enabled("mjpeg")
    assert cfg.get("mjpeg")["port"] == 4444 and isinstance(cfg.get("mjpeg")["port"], int)


def test_mjpeg_absent_section_is_disabled():
    cfg = build_config({"input": {"source": "video"}}, schemas=readymade_tracker.CONF_SCHEMAS)
    assert not cfg.is_enabled("mjpeg")


def test_mjpeg_override_without_section_is_noop():
    # overrides apply only to sections already in root → --mjpeg-stream alone can't enable it
    # (the no-op main() warns about). This is the deliberate "[mjpeg] lives in readymade.conf" design.
    cfg = build_config({"input": {"source": "video"}},
                       schemas=readymade_tracker.CONF_SCHEMAS, overrides={"mjpeg": {"port": "4444"}})
    assert not cfg.is_enabled("mjpeg")

def test_no_mjpeg_flag_no_override():
    assert "mjpeg" not in readymade_tracker._cli_overrides(readymade_tracker._parse_args([]))


# --------------------------------------------------------------------------- #
# monitor: JPEG extraction + background underlay
# --------------------------------------------------------------------------- #

def test_extract_jpegs_pulls_full_frames_and_keeps_partial():
    a, b = b"\xff\xd8AAA\xff\xd9", b"\xff\xd8BBBB\xff\xd9"
    frames, rem = _extract_jpegs(b"junk" + a + b + b"\xff\xd8partial")
    assert frames == [a, b]
    assert rem == b"\xff\xd8partial"          # the incomplete trailing JPEG is carried over


def test_extract_jpegs_no_markers_keeps_buffer():
    frames, rem = _extract_jpegs(b"no jpeg here")
    assert frames == [] and rem == b"no jpeg here"


def test_render_frame_uses_background_underlay():
    bg = _bgr(40, 60, (200, 200, 200))                    # bright video frame
    img = render_frame(TuioState(), 120, 80, background=bg)
    assert img.shape == (80, 120, 3)
    assert img.mean() > 120                               # background shows through
    dark = render_frame(TuioState(), 120, 80)             # no background → flat dark canvas
    assert dark.mean() < 60


# --------------------------------------------------------------------------- #
# end-to-end: HTTP server thread -> urllib client -> a decodable JPEG
# --------------------------------------------------------------------------- #

def test_mjpeg_server_streams_a_decodable_jpeg_over_http():
    import urllib.request

    import cv2

    frame = _bgr(32, 48, (0, 0, 255))
    out = MjpegOut()
    out.setup(_Cfg(host="127.0.0.1", port=0, view="debug", quality=60))   # port 0 = ephemeral
    out.start(_Bus(_Snap(frame=frame, composite=frame, frame_idx=1)))
    try:
        port = out._srv.server_address[1]
        deadline = time.time() + 4.0
        got = None
        while time.time() < deadline and got is None:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=2.0) as r:
                    buf = b""
                    while time.time() < deadline:
                        chunk = r.read1(65536)          # read1: available bytes (one read), not block-for-n
                        if not chunk:
                            break
                        frames, buf = _extract_jpegs(buf + chunk)
                        if frames:
                            got = frames[-1]
                            break
            except Exception:                    # noqa: BLE001 — server still warming up; retry
                time.sleep(0.1)
        assert got is not None, "no JPEG received from the MJPEG server"
        img = cv2.imdecode(np.frombuffer(got, np.uint8), cv2.IMREAD_COLOR)
        assert img is not None and img.shape == (32, 48, 3)
    finally:
        out.stop()


class _NeverBus:
    """A bus that never has a fresh frame — exercises the 'client connected, no frame yet' path."""
    def wait_fresh(self, last_idx, timeout):
        time.sleep(min(timeout, 0.1))
        return None
    def latest(self):
        return None


def test_client_before_first_frame_does_not_busy_spin():
    # regression: the handler predicate must require an actual frame, else `_seq(0) != last_seq(-1)`
    # is trivially true and wait_for never blocks → a pegged CPU core until the first frame.
    import socket as _socket

    out = MjpegOut()
    out.setup(_Cfg(host="127.0.0.1", port=0))
    out.start(_NeverBus())
    try:
        port = out._srv.server_address[1]
        s = _socket.create_connection(("127.0.0.1", port), timeout=2.0)
        s.sendall(b"GET / HTTP/1.0\r\nHost: x\r\n\r\n")
        time.sleep(0.2)                          # let the handler read the request + enter its loop
        cpu0 = time.process_time()
        time.sleep(0.5)                          # a window with a client attached but no frame
        cpu = time.process_time() - cpu0
        s.close()
        assert cpu < 0.3, f"handler busy-spins with no frame (cpu={cpu:.2f}s over 0.5s wall)"
    finally:
        out.stop()


def test_mjpeg_stop_is_idempotent():
    out = MjpegOut()
    out.setup(_Cfg(host="127.0.0.1", port=0))
    out.start(_Bus(_Snap(frame=_bgr(8, 8, (1, 1, 1)), frame_idx=1)))
    out.stop()
    out.stop()                                   # second stop must not raise


# --------------------------------------------------------------------------- #
# on-demand: the publisher renders/encodes ONLY while a client is connected
# --------------------------------------------------------------------------- #

class _LiveBus:
    """Always hands a FRESH frame (incrementing idx) — the publisher is never starved, so whether it
    encodes is decided purely by the client gate (not by frame availability)."""
    def __init__(self, snap):
        self._snap, self._i = snap, 0
    def wait_fresh(self, last_idx, timeout):
        self._i += 1
        self._snap.frame_idx = self._i
        return self._snap
    def latest(self):
        return self._snap


def _wait(pred, timeout=2.0):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.005)
    return False


def test_no_encode_while_unwatched():
    # frames flow, but with zero clients the publisher must render/encode NOTHING (view=debug is free)
    out = _out(host="127.0.0.1", port=0, view="debug")
    out.start(_LiveBus(_Snap(frame=_bgr(16, 16, (1, 2, 3)))))
    try:
        time.sleep(0.15)                             # many publisher iterations, no viewer
        assert out._jpeg is None                     # nothing encoded
    finally:
        out.stop()


def test_encodes_only_after_a_client_joins():
    out = _out(host="127.0.0.1", port=0, view="debug")
    out.start(_LiveBus(_Snap(frame=_bgr(16, 16, (1, 2, 3)), composite=_bgr(16, 16, (4, 5, 6)))))
    try:
        assert not _wait(lambda: out._jpeg is not None, 0.1)   # idle: no encode
        out._client_join()
        assert _wait(lambda: out._jpeg is not None)            # viewer joined → encoding starts
    finally:
        out._client_leave()
        out.stop()


# --------------------------------------------------------------------------- #
# --profile: outputs land in the OUTPUT group (their own threads)
# --------------------------------------------------------------------------- #

def test_setup_stores_profiler():
    from lib.profile import Profiler
    from plugins.output.tuio2.tuio2 import Tuio2Out
    prof = Profiler()
    m = MjpegOut(); m.setup(_Cfg(), profiler=prof); assert m._prof is prof
    t = Tuio2Out(); t.setup(_Cfg(), profiler=prof, send_fn=lambda d: None); assert t._prof is prof


def test_profiler_records_mjpeg_under_output_group():
    from lib.profile import Profiler
    prof = Profiler()
    out = MjpegOut()
    out.setup(_Cfg(host="127.0.0.1", port=0, view="debug"), profiler=prof)
    out._client_join()                               # simulate a viewer → the publisher encodes
    out.start(_Bus(_Snap(frame=_bgr(16, 16, (1, 2, 3)), composite=_bgr(16, 16, (4, 5, 6)), frame_idx=1)))
    deadline = time.time() + 3.0
    while time.time() < deadline and "mjpeg" not in prof._stats:      # wait for one encode
        time.sleep(0.02)
    out.stop()
    assert "mjpeg" in prof._stats and prof._group["mjpeg"] == "output"
    report = prof.summary()
    assert "OUTPUT" in report and "mjpeg" in report


def test_client_join_is_pure_increment_seed_is_separate():
    # regression: _client_join must not encode (the seed can raise) — else a seed failure leaks the
    # counter (join outside the handler's try/finally). Join increments; the seed is a separate call.
    out = _out(view="debug")
    out._bus = _Bus(_Snap(frame=_bgr(16, 16, (1, 2, 3)), composite=_bgr(16, 16, (1, 2, 3)), frame_idx=1))
    out._client_join()
    assert out._clients == 1 and out._jpeg is None      # join alone did NOT encode
    out._seed_latest()
    assert out._jpeg is not None                        # the seed is the separate, raise-able encode


def test_handler_has_a_socket_timeout():
    assert _out()._make_handler().timeout == 30.0       # slow-loris guard (stuck client times out)
