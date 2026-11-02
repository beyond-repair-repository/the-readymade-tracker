"""CameraBackend + control-shim unit tests — INJECTED capture, no hardware.

Track 1's verification contract (roadmap): the camera plugin is built cross-platform
and unit-tested against a fake ``cv2.VideoCapture`` (as ``VideoBackend`` is via
``capture_factory``); the owner validates pyuvc / exposure on real devices. So these
exercise the grab loop, the transport (pause / step / stop / live-seek no-op), the
control shim's cv2 path, and the fail-soft behaviour — none of which need a camera.

``_IS_LINUX`` is forced False where a control-value is asserted so the shim takes the
cv2 path deterministically (on a Linux host with v4l2-ctl it would otherwise shell out
to /dev/videoN — correct in production, untestable here).
"""

import time
from types import MappingProxyType, SimpleNamespace

import cv2
import numpy as np
import pytest

import lib.camera_controls as cc
from lib.camera_controls import CameraControls, _nearest_mode
from lib.config import Section
from lib.logger import NULL_LOGGER
from plugins.input.camera.camera import CameraBackend, _parse_device


class FakeCapture:
    """A cv2.VideoCapture-shaped stand-in. ``fail_reads`` frames read as misses
    (ok=False); ``reject`` props return False from ``set``; ``raise_on`` props raise."""

    def __init__(self, device, *, fail_reads=0, reject=(), raise_on=()):
        self.device = device
        self.opened = True
        self.sets: dict = {}
        self.n = 0
        self._fail_reads = fail_reads
        self._reject = set(reject)
        self._raise_on = set(raise_on)

    def isOpened(self):
        return self.opened

    def set(self, prop, val):
        if prop in self._raise_on:
            raise RuntimeError(f"prop {prop} unsupported")
        if prop in self._reject:
            return False
        self.sets[prop] = val
        return True

    def get(self, prop):
        return float(self.sets.get(prop, 0.0))

    def read(self):
        self.n += 1
        if self.n <= self._fail_reads:
            return False, None
        return True, np.full((4, 4, 3), self.n % 256, np.uint8)

    def release(self):
        self.opened = False


def _section(**kw):
    return Section("camera", MappingProxyType(kw))


def _backend(section, **cap_kw):
    caps = []

    def factory(device):
        c = FakeCapture(device, **cap_kw)
        caps.append(c)
        return c

    return CameraBackend(section, capture_factory=factory), caps


# --------------------------------------------------------------------------- #
# open: geometry + isOpened contract
# --------------------------------------------------------------------------- #

def test_open_sets_geometry(monkeypatch):
    monkeypatch.setattr(cc, "_IS_LINUX", False)
    b, caps = _backend(_section(device="0", width=640, height=480, fps=30))
    b.open()
    cap = caps[0]
    assert cap.sets[cv2.CAP_PROP_FRAME_WIDTH] == 640
    assert cap.sets[cv2.CAP_PROP_FRAME_HEIGHT] == 480
    assert cap.sets[cv2.CAP_PROP_FPS] == 30


def test_open_zero_geometry_left_at_driver_default(monkeypatch):
    monkeypatch.setattr(cc, "_IS_LINUX", False)
    b, caps = _backend(_section(device="0", width=0, height=0, fps=0))
    b.open()
    assert cv2.CAP_PROP_FRAME_WIDTH not in caps[0].sets   # 0 → don't touch the driver default


def test_open_raises_when_not_opened():
    def factory(device):
        c = FakeCapture(device)
        c.opened = False
        return c

    b = CameraBackend(_section(device="0"), capture_factory=factory)
    with pytest.raises(RuntimeError):
        b.open()


# --------------------------------------------------------------------------- #
# controls applied on open (cv2 path)
# --------------------------------------------------------------------------- #

def test_controls_applied_on_open(monkeypatch):
    monkeypatch.setattr(cc, "_IS_LINUX", False)
    b, caps = _backend(_section(device="0", auto_exposure=False, exposure=200,
                                brightness=140, gain=12))
    b.open()
    s = caps[0].sets
    assert s[cv2.CAP_PROP_EXPOSURE] == 200
    assert s[cv2.CAP_PROP_BRIGHTNESS] == 140
    assert s[cv2.CAP_PROP_GAIN] == 12
    # auto_exposure=False → the "manual" value for this platform (1 posix / 0.25 win)
    assert s[cv2.CAP_PROP_AUTO_EXPOSURE] in (1, 1.0, 0.25)


def test_absent_control_left_at_default(monkeypatch):
    monkeypatch.setattr(cc, "_IS_LINUX", False)
    # no controls in the section → nothing but geometry is set
    b, caps = _backend(_section(device="0", width=320, height=240))
    b.open()
    assert cv2.CAP_PROP_EXPOSURE not in caps[0].sets
    assert cv2.CAP_PROP_BRIGHTNESS not in caps[0].sets


def test_control_failure_never_crashes_open(monkeypatch):
    monkeypatch.setattr(cc, "_IS_LINUX", False)
    # a camera that raises on the exposure prop must not tear down the feed
    b, caps = _backend(_section(device="0", exposure=200, brightness=90),
                       raise_on=(cv2.CAP_PROP_EXPOSURE,))
    b.open()                                              # no exception
    assert caps[0].sets[cv2.CAP_PROP_BRIGHTNESS] == 90    # the other control still applied


# --------------------------------------------------------------------------- #
# grab loop + transport
# --------------------------------------------------------------------------- #

def test_grab_returns_frame_and_monotonic_ts(monkeypatch):
    monkeypatch.setattr(cc, "_IS_LINUX", False)
    b, _ = _backend(_section(device="0"))
    b.open()
    (bgr1, ts1) = b.grab()
    (bgr2, ts2) = b.grab()
    assert bgr1.shape == (4, 4, 3)
    assert ts2 >= ts1                                    # wall-clock PTS is monotonic
    assert int(bgr1[0, 0, 0]) != int(bgr2[0, 0, 0])      # advanced to a fresh frame


def test_pause_holds_then_step_advances(monkeypatch):
    monkeypatch.setattr(cc, "_IS_LINUX", False)
    b, _ = _backend(_section(device="0"))
    b.open()
    b.grab()
    b.transport.pause()
    assert b.grab() is None                              # paused → no new frame (holds last)
    b.transport.step_forward()
    assert b.grab() is not None                          # one fresh frame on step


def test_stop_raises_stopiteration(monkeypatch):
    monkeypatch.setattr(cc, "_IS_LINUX", False)
    b, _ = _backend(_section(device="0"))
    b.open()
    b.transport.stop()
    with pytest.raises(StopIteration):
        b.grab()


def test_transient_read_miss_returns_none_not_eof(monkeypatch):
    monkeypatch.setattr(cc, "_IS_LINUX", False)
    b, _ = _backend(_section(device="0"), fail_reads=1)
    b.open()
    assert b.grab() is None                              # a live source SKIPS a miss...
    assert b.grab() is not None                          # ...and keeps going (no StopIteration)


def test_live_seek_is_noop(monkeypatch):
    monkeypatch.setattr(cc, "_IS_LINUX", False)
    b, _ = _backend(_section(device="0"))
    b.open()
    b.transport.seek(100)
    assert b.grab() is not None                          # seek ignored, still grabs live


def test_close_releases(monkeypatch):
    monkeypatch.setattr(cc, "_IS_LINUX", False)
    b, caps = _backend(_section(device="0"))
    b.open()
    b.close()
    assert caps[0].opened is False


# --------------------------------------------------------------------------- #
# control shim — backend selection, conversions, fail-soft
# --------------------------------------------------------------------------- #

def test_shim_picks_cv2_for_plain_capture(monkeypatch):
    monkeypatch.setattr(cc, "_IS_LINUX", False)
    ctl = CameraControls(FakeCapture("0"), device="0")
    assert ctl.backend == "cv2"


def test_shim_auto_exposure_manual_vs_auto_differ(monkeypatch):
    monkeypatch.setattr(cc, "_IS_LINUX", False)
    cap = FakeCapture("cam")
    CameraControls(cap, device="cam").apply({"auto_exposure": True})
    auto = cap.sets[cv2.CAP_PROP_AUTO_EXPOSURE]
    cap2 = FakeCapture("cam")
    CameraControls(cap2, device="cam").apply({"auto_exposure": False})
    manual = cap2.sets[cv2.CAP_PROP_AUTO_EXPOSURE]
    assert auto != manual                                # the mode actually toggles


def test_shim_unknown_control_ignored(monkeypatch):
    monkeypatch.setattr(cc, "_IS_LINUX", False)
    cap = FakeCapture("cam")
    CameraControls(cap, device="cam").apply({"nonsense": 5})   # not in CONTROL_NAMES
    assert cap.sets == {}                                 # nothing applied, no raise


def test_shim_none_value_skipped(monkeypatch):
    monkeypatch.setattr(cc, "_IS_LINUX", False)
    cap = FakeCapture("cam")
    CameraControls(cap, device="cam").apply({"exposure": None, "gain": 7})
    assert cv2.CAP_PROP_EXPOSURE not in cap.sets          # None = leave the default
    assert cap.sets[cv2.CAP_PROP_GAIN] == 7


def test_shim_rejected_set_does_not_raise(monkeypatch):
    monkeypatch.setattr(cc, "_IS_LINUX", False)
    cap = FakeCapture("cam", reject=(cv2.CAP_PROP_GAMMA,))
    CameraControls(cap, device="cam").apply({"gamma": 100})   # set() returns False
    # fail-soft: logged, not raised; the value simply didn't take
    assert cv2.CAP_PROP_GAMMA not in cap.sets


def test_manual_exposure_settle_only_on_transition(monkeypatch):
    # engaging manual exposure + writing the value in ONE apply (the tracker's open) settles
    # so the driver leaves auto first; a standalone exposure tweak must NOT settle (responsive).
    monkeypatch.setattr(cc, "_IS_LINUX", False)
    slept: list = []
    monkeypatch.setattr(cc.time, "sleep", lambda t: slept.append(t))

    CameraControls(FakeCapture("cam"), device="cam").apply({"auto_exposure": False, "exposure": 200})
    assert slept == [cc._MANUAL_SETTLE_S]                 # transition → one settle

    slept.clear()
    CameraControls(FakeCapture("cam"), device="cam").apply({"exposure": 300})
    assert slept == []                                    # standalone tweak → no settle

    slept.clear()
    CameraControls(FakeCapture("cam"), device="cam").apply({"auto_exposure": True, "exposure": 300})
    assert slept == []                                    # staying in AUTO → no settle


def test_shim_white_balance_disables_auto_wb_first(monkeypatch):
    # manual WB only sticks if auto-WB is turned off first (mirrors the auto_exposure
    # handshake) — otherwise the driver holds the temperature control read-only.
    monkeypatch.setattr(cc, "_IS_LINUX", False)
    auto_wb = getattr(cv2, "CAP_PROP_AUTO_WB", None)
    if auto_wb is None:
        import pytest
        pytest.skip("this OpenCV build lacks CAP_PROP_AUTO_WB")
    cap = FakeCapture("cam")
    CameraControls(cap, device="cam").apply({"white_balance": 4600})
    assert cap.sets[auto_wb] == 0.0                       # auto-WB turned OFF
    assert cap.sets[cc._CV2_PROPS["white_balance"]] == 4600


def test_fit_uvc_clamps_and_snaps():
    from lib.camera_controls import _fit_uvc

    class Ctrl:
        def __init__(self, lo, hi, step=1):
            self.min_val, self.max_val, self.step = lo, hi, step

    assert _fit_uvc(Ctrl(100, 500), 41) == 100          # below min → clamp up (the gamma=41 case)
    assert _fit_uvc(Ctrl(0, 95, 5), 29) == 30           # off the step grid → snap (the contrast=29 case)
    assert _fit_uvc(Ctrl(0, 255), 16) == 16             # already legal → unchanged
    assert _fit_uvc(Ctrl(0, 95), 9999) == 95            # above max → clamp down

    class Bad:                                           # no range metadata → passthrough, never raise
        pass
    assert _fit_uvc(Bad(), 42) == 42


def test_shim_uvc_fits_out_of_range_value_instead_of_rejecting():
    # a stale/foreign conf value must be FITTED to the device's reported range (not written raw, where
    # libuvc rejects it with "Could not set Value" and never applies it).
    class Ctrl:
        def __init__(self, lo, hi, step=1):
            self.min_val, self.max_val, self.step, self.value = lo, hi, step, lo

    class UvcCap:
        def __init__(self):
            self._ctrls = {"Gamma": Ctrl(100, 500), "Contrast": Ctrl(0, 95, 5)}

        @property
        def uvc_controls(self):
            return self._ctrls

    cap = UvcCap()
    shim = CameraControls(cap, device="cam")
    assert shim.backend == "uvc"                          # picked via uvc_controls
    shim.apply({"gamma": 41, "contrast": 29})
    assert cap._ctrls["Gamma"].value == 100              # fitted up to the min
    assert cap._ctrls["Contrast"].value == 30            # snapped to step 5


def test_apply_device_ranges_bounds_and_clamps():
    # the webgui builds sliders from the param lo/hi; publish the DEVICE's real ranges so it can't
    # offer a value the device rejects, and clamp a stale conf value into range.
    import readymade_tracker as rt
    from lib.logger import NULL_LOGGER
    from lib.params import build_registry
    reg = build_registry(rt.SCHEMAS, rt.COMPONENTS)

    class FakeBackend:
        def control_range(self, name):
            return {"gamma": (100, 500, 1), "contrast": (0, 95, 5)}.get(name)   # others → None

    values = {("camera", "gamma"): 41, ("camera", "contrast"): 29,
              ("camera", "exposure"): 227, ("camera", "device"): "0"}
    rt._apply_device_ranges(reg, "camera", FakeBackend(), values, NULL_LOGGER)

    assert (reg["camera"]["gamma"].lo, reg["camera"]["gamma"].hi) == (100.0, 500.0)   # device range
    assert (reg["camera"]["contrast"].lo, reg["camera"]["contrast"].hi) == (0.0, 95.0)
    assert reg["camera"]["exposure"].hi == 5000                    # no device range → generic kept
    assert values[("camera", "gamma")] == 100                     # 41 below min → clamped up
    assert values[("camera", "contrast")] == 29                   # already in range → unchanged
    assert values[("camera", "device")] == "0"                    # non-numeric untouched


# --------------------------------------------------------------------------- #
# small pure helpers
# --------------------------------------------------------------------------- #

def test_parse_device():
    assert _parse_device("0") == 0
    assert _parse_device(3) == 3
    assert _parse_device("/dev/video1") == "/dev/video1"
    assert _parse_device(" 2 ") == 2


def test_v4l2_node():
    assert cc._v4l2_node("/dev/video0") == "/dev/video0"
    assert cc._v4l2_node(1) == "/dev/video1"
    assert cc._v4l2_node("2") == "/dev/video2"
    assert cc._v4l2_node("some-uvc-name") is None


def test_nearest_mode_picks_closest():
    modes = [(640, 480, 30), (1280, 720, 30), (1920, 1080, 60)]
    assert _nearest_mode(modes, 1280, 720, 30) == (1280, 720, 30)
    assert _nearest_mode(modes, 1300, 700, 30) == (1280, 720, 30)   # nearest by area
    assert _nearest_mode([], 640, 480, 30) is None


class _RingCapture:
    """libuvc's streaming ring: ONE buffer, refilled in place on every grab. pyuvc's
    ``Frame.bgr`` is a window onto it, so two grabs hand back the same memory."""

    def __init__(self, shape=(2, 2, 3)):
        self.buf = np.zeros(shape, np.uint8)
        self.n = 0

    def get_frame_robust(self):
        self.buf[:] = self.n                      # refill THE SAME memory
        self.n += 1
        return SimpleNamespace(bgr=self.buf)


def _uvc_shim(cap):
    """A ``UvcCapture`` bypassing ``__init__`` (it imports pyuvc, a macOS-only optional dep)."""
    u = object.__new__(cc.UvcCapture)
    u._cap = cap
    u._mode_set = True                            # skip _ensure_mode's frame-mode negotiation
    u._log = NULL_LOGGER
    return u


def test_uvc_read_copies_off_the_recycled_capture_buffer():
    # The snapshot contract (lib.events.readonly) is that a published frame is REPLACED, never
    # rewritten — outputs get a read-only VIEW, not a copy. Returning pyuvc's own array breaks
    # that: the ring refills it, so a slow output thread's snap.frame changes mid-read.
    u = _uvc_shim(_RingCapture())
    ok, first = u.read()
    assert ok and first[0, 0, 0] == 0
    u.read()                                      # the ring refills the same buffer with 1s
    assert first[0, 0, 0] == 0, "read() handed back pyuvc's live buffer, not a copy"


def test_uvc_read_miss_returns_none_without_copying():
    class _NoFrame:
        def get_frame_robust(self):
            return SimpleNamespace(bgr=None, img=None)

    assert _uvc_shim(_NoFrame()).read() == (False, None)


# --------------------------------------------------------------------------- #
# UVC stream death: a bounded wait, so the backend gets misses and can reopen
# --------------------------------------------------------------------------- #

class _DeadStream:
    """libuvc after every USB transfer has died: ``get_frame`` wakes without a frame (pyuvc's bare
    TimeoutError) or waits out its timeout. The unbounded ``get_frame_robust`` would block forever."""

    def __init__(self):
        self.calls = 0

    def get_frame(self, timeout=0):
        assert timeout > 0, "an unbounded wait never returns on a dead stream"
        self.calls += 1
        time.sleep(min(timeout, 0.01))
        raise TimeoutError

    def get_frame_robust(self):
        raise AssertionError("read() used the unbounded wait")


def test_uvc_read_returns_a_miss_on_a_dead_stream_instead_of_blocking():
    u = _uvc_shim(_DeadStream())
    u._min_frame_wait = 0.2
    t0 = time.perf_counter()
    assert u.read() == (False, None)
    assert time.perf_counter() - t0 < 1.0
    assert u._cap.calls > 1                       # wake-ups without a frame are retried to the deadline


def test_uvc_read_retries_a_spurious_wakeup_and_a_corrupt_frame():
    class StreamError(Exception):
        pass

    class _Flaky:
        def __init__(self):
            self.n = 0

        def get_frame(self, timeout=0):
            self.n += 1
            if self.n == 1:
                raise TimeoutError                # woken by one dying transfer
            if self.n == 2:
                raise StreamError("JPEG header corrupt.")
            return SimpleNamespace(bgr=np.full((2, 2, 3), 7, np.uint8))

    ok, frame = _uvc_shim(_Flaky()).read()
    assert ok and frame[0, 0, 0] == 7


def test_uvc_frame_wait_follows_the_mode_fps():
    assert _uvc_shim(SimpleNamespace(frame_mode=(640, 480, 2)))._frame_wait() == pytest.approx(1.5)
    assert _uvc_shim(SimpleNamespace(frame_mode=(640, 480, 30)))._frame_wait() == 1.0   # the floor


def test_backend_reopens_and_replays_controls_after_a_frameless_streak(monkeypatch):
    import plugins.input.camera.camera as cam
    monkeypatch.setattr(cc, "_IS_LINUX", False)
    clock = [100.0]
    monkeypatch.setattr(cam.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(cam.time, "sleep", lambda s: None)
    b, caps = _backend(_section(device="0", exposure=156), fail_reads=10**9)
    b.open()
    for _ in range(4):                            # misses at t = 0, 1, 2, 3 s
        assert b.grab() is None
        clock[0] += 1.0
    assert len(caps) == 2                         # reopened once the streak passed 2 s
    assert caps[0].opened is False                # the dead capture was released first
    assert caps[1].sets.get(cv2.CAP_PROP_EXPOSURE) == 156   # conf controls replayed on the new handle


def test_unplugged_uvc_camera_never_falls_back_to_the_builtin_cv2_camera(monkeypatch):
    """Unplug the USB camera mid-run: reopens must keep failing (and keep retrying) instead of
    falling back to cv2 index 0 = a laptop's built-in camera; replug → frames come back via pyuvc."""
    import plugins.input.camera.camera as cam

    class FakeUvc:
        plugged = True

        def __init__(self, device, logger=None):
            if not FakeUvc.plugged:
                raise IndexError("list index out of range")      # pyuvc: no such device
            self.live = True
            self.uvc_controls = {}

        def isOpened(self):
            return True

        def set(self, prop, val):
            return True

        def get(self, prop):
            return 0.0

        def read(self):
            if self.live and FakeUvc.plugged:
                return True, np.zeros((4, 4, 3), np.uint8)
            return False, None

        def release(self):
            self.live = False

    builtin_opened = []
    monkeypatch.setattr(cc, "UvcCapture", FakeUvc)
    monkeypatch.setattr(cam, "UvcCapture", FakeUvc)
    monkeypatch.setattr(cc, "_IS_MAC", True)
    monkeypatch.setattr(cc, "_uvc_available", lambda: True)
    monkeypatch.setattr(cc.cv2, "VideoCapture", lambda idx: builtin_opened.append(idx) or FakeCapture(idx))
    clock = [100.0]
    monkeypatch.setattr(cam.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(cam.time, "sleep", lambda s: None)

    b = CameraBackend(_section(device="0"))
    b.open()
    assert b.grab() is not None
    FakeUvc.plugged = False                       # cable out
    try:
        for _ in range(12):                       # 12 s unplugged → several failed reopens
            assert b.grab() is None
            clock[0] += 1.0
        assert builtin_opened == [], "an outage reopen fell back to the built-in cv2 camera"
        FakeUvc.plugged = True                    # cable back in
        frame = None
        for _ in range(5):
            frame = b.grab()
            if frame is not None:
                break
            clock[0] += 1.0
        assert frame is not None and isinstance(b._cap, FakeUvc)
    finally:
        FakeUvc.plugged = True
