"""``camera`` INPUT backend — a live camera as the frame source.

Mirrors ``VideoBackend`` (open / grab / close + the shared ``Transport``) so the
core's input-runner drives it identically: ``open()`` →
``grab() → (bgr, ts) | None`` → ``close()``. The differences a *live* source
brings:

  • **``seek`` is a no-op** — you can't rewind a live camera (logged once).
  • **``pause`` holds the last frame** — ``grab`` returns ``None`` while paused
    (no new frame; the pipeline keeps its last snapshot); ``step`` grabs one
    fresh frame.
  • **No EOF / loop** — a live source runs until the transport is stopped
    (``StopIteration``) or the input thread is told to quit. A *transient* read
    failure returns ``None`` (skip this call) rather than tearing down the app.
  • **PTS is the wall clock** — no container timestamps; ``ts`` is a monotonic
    ``perf_counter`` so the pipeline's motion ``dt`` is real elapsed time.

The tracker is the SOLE owner of the device (single-client cameras; libuvc claims
it exclusively). ``cv2.VideoCapture`` — or a pyuvc ``UvcCapture`` on macOS — is
injectable via ``capture_factory`` (a ``device -> cv2-shaped capture`` callable)
so this is unit-testable with a fake capture, no camera required. Resolution /
fps and the logical controls (brightness … exposure … white_balance) are applied
on ``open`` via ``lib.camera_controls`` (fail-soft per control).
"""

from __future__ import annotations

import time
from typing import Any, Callable

import cv2

from lib.camera_controls import CONTROL_NAMES, CameraControls, UvcCapture, open_capture
from lib.logger import NULL_LOGGER
from lib.params import ParamSpec
from plugins.input.video.video import Transport


def _cam_apply(name: str):
    """Setter for a HOT [camera] image control: live-apply it to the OPEN device via the backend's
    persistent CameraControls shim (webgui / oscin `set` → visible on the feed immediately)."""
    return lambda backend, value: backend.apply_control(name, value)


class CameraBackend:
    """Frame source: a live camera."""

    # -- the [camera] live parameters (registry anchor). The IMAGE controls (exposure … white_balance +
    #    auto_exposure) are HOT: a set is applied to the open device NOW via the shim (apply=_cam_apply)
    #    so a webgui tweak is visible live — and remembered so a reopen replays it. The MODE params
    #    (device/width/height/fps/backend) reopen the camera → needs-restart (bind=False: the backend
    #    reads its own config at construction; the store only drives live sets + save). --
    PARAMS = (
        ParamSpec("device", str, None, hot=False, bind=False, expose="edit", doc='Capture index ("0"), /dev/videoN path, or UVC name/UID'),
        ParamSpec("width", int, None, lo=0, hot=False, bind=False, expose="edit", doc="Requested capture width in px; 0 = driver default"),
        ParamSpec("height", int, None, lo=0, hot=False, bind=False, expose="edit", doc="Requested capture height in px; 0 = driver default"),
        ParamSpec("fps", int, None, lo=0, hot=False, bind=False, expose="edit", doc="Requested capture frame rate; 0 = driver default"),
        ParamSpec("backend", str, None, enum=("auto", "uvc", "cv2"), hot=False, bind=False, expose="edit", doc="Capture backend: auto picks pyuvc on macOS, else cv2"),
        ParamSpec("auto_exposure", bool, None, hot=True, bind=False, apply=_cam_apply("auto_exposure"), doc="true = camera auto-exposes; false = manual (exposure applies)"),
        ParamSpec("exposure", int, None, lo=0, hi=5000, hot=True, bind=False, apply=_cam_apply("exposure"), doc="Manual exposure time (device units); needs auto_exposure off"),
        ParamSpec("gain", int, None, lo=0, hi=255, hot=True, bind=False, apply=_cam_apply("gain"), doc="Sensor gain in device-native units"),
        ParamSpec("brightness", int, None, lo=0, hi=255, hot=True, bind=False, apply=_cam_apply("brightness"), doc="Image brightness in device-native units"),
        ParamSpec("contrast", int, None, lo=0, hi=255, hot=True, bind=False, apply=_cam_apply("contrast"), doc="Image contrast in device-native units"),
        ParamSpec("saturation", int, None, lo=0, hi=255, hot=True, bind=False, apply=_cam_apply("saturation"), doc="Color saturation in device-native units"),
        ParamSpec("gamma", int, None, lo=0, hi=500, hot=True, bind=False, apply=_cam_apply("gamma"), doc="Gamma correction in device-native units"),
        ParamSpec("white_balance", int, None, lo=2000, hi=8000, hot=True, bind=False, apply=_cam_apply("white_balance"), doc="Manual white-balance color temperature in Kelvin"),
    )

    def __init__(self, config: Any, *, capture_factory: Callable[[Any], Any] | None = None,
                 logger: Any = None) -> None:
        self._log = logger or NULL_LOGGER
        g = config.get if config is not None else (lambda k, d=None: d)
        self._device = _parse_device(g("device", 0))
        self._backend_pref = str(g("backend", "auto"))
        self._width = int(g("width", 0))
        self._height = int(g("height", 0))
        self._fps = int(g("fps", 0))
        # controls: only the ones actually present in the conf are applied (an
        # absent key = leave the camera's own default). auto_exposure is a bool;
        # the rest are device-native ints.
        self._controls = {name: g(name) for name in CONTROL_NAMES if config is not None
                          and name in config}
        # a caller-injected `device -> capture` factory (tests) or the real opener
        self._open_cap = capture_factory or self._default_capture
        self._cap: Any = None
        self._shim: CameraControls | None = None     # persistent → live control sets (apply_control)
        self._seek_warned = False
        self._last: tuple[Any, float] | None = None   # latest grabbed frame — re-emitted on a paused scan
        self._miss_since: float | None = None     # first ts of the current no-frame streak (outage detect)
        self._dead_warned = False                 # one warning per outage, not one per read
        self._retry_ts = 0.0                      # last reopen attempt (throttled)
        self._uvc_locked = False                  # once pyuvc has opened, every reopen must stay on pyuvc
        self.transport = Transport()

    def _default_capture(self, device: Any) -> Any:
        # strict once pyuvc worked: an outage reopen must not fall back to cv2, whose index 0 is the
        # laptop's built-in camera while the USB one is unplugged (a raise here = "retry in 3 s")
        cap = open_capture(device, self._backend_pref, self._log, strict=self._uvc_locked)
        if isinstance(cap, UvcCapture):
            self._uvc_locked = True
        return cap

    def open(self) -> None:
        self._cap = self._open_cap(self._device)
        if not self._cap.isOpened():
            raise RuntimeError(f"camera: cannot open device {self._device!r}")
        if self._width > 0:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        if self._height > 0:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        if self._fps > 0:
            self._cap.set(cv2.CAP_PROP_FPS, self._fps)
        self._shim = CameraControls(self._cap, device=self._device, logger=self._log)
        try:
            self._shim.apply(self._controls)      # bulk-apply the conf controls at open
        except Exception as e:                    # noqa: BLE001 — controls never block a live feed
            self._log.warning(f"[camera] control setup failed: {e} -> continuing")
        aw = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        ah = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self._log.info(f"camera device={self._device!r} backend={self._backend_pref} "
                       f"requested={self._width}x{self._height}@{self._fps} actual={aw}x{ah}")

    def control_range(self, name: str) -> "tuple[int, int, int] | None":
        """``(min, max, step)`` the OPEN device accepts for a control, or ``None`` — so the tracker can
        publish the REAL range to the webgui (an out-of-range value would be silently clamped)."""
        return self._shim.control_range(name) if self._shim is not None else None

    def apply_control(self, name: str, value: Any) -> None:
        """Live-apply ONE camera control to the OPEN device (a hot [camera] set from the webgui/oscin),
        and remember it so a reopen replays it. Fail-soft: a control never disturbs the live feed. To make
        a manual ``exposure`` stick, set ``auto_exposure`` false first (the shim engages manual mode)."""
        self._controls[name] = value
        if self._cap is None or self._shim is None:
            return
        try:
            self._shim.apply({name: value})
        except Exception as e:                    # noqa: BLE001 — a control must never crash the feed
            self._log.warning(f"[camera] live control {name}={value!r} failed: {e}")

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._shim = None

    def grab(self) -> tuple[Any, float] | None:
        """One raw ``(bgr, ts)``, or ``None`` (paused / transient read miss);
        ``StopIteration`` only when the transport is stopped. A live source never
        reaches EOF, so — unlike the video backend — it does not raise at end."""
        t = self.transport
        if t.stopped:
            raise StopIteration
        paused, step, seek, refresh = t.take()

        if seek is not None and not self._seek_warned:
            self._log.info("[camera] seek is a no-op on a live source")
            self._seek_warned = True

        if refresh and self._last is not None:    # PAUSED manual scan → re-emit the last live frame
            return self._last                     #   so the frozen table's scan reaches the outputs

        if paused and step == 0:                  # hold the last frame (no new grab)
            time.sleep(0.01)
            return None

        # no capture = a reopen failed (device still unplugged): count it as a miss, keep retrying
        ok, bgr = self._cap.read() if self._cap is not None else (False, None)
        if not ok or bgr is None:                 # transient miss — skip, don't tear down …
            now = time.perf_counter()
            if self._miss_since is None:
                self._miss_since = now
            elif now - self._miss_since > 2.0:    # … but a 2 s streak is an OUTAGE (cable/power/USB):
                if not self._dead_warned:         #     say so ONCE (a dead camera must not be silent-black
                    self._dead_warned = True      #     forever) and gently try to get the device back.
                    self._log.warning("[camera] no frames for 2 s - device lost? reopening every 3 s")
                if now - self._retry_ts > 3.0:
                    self._retry_ts = now
                    try:
                        self.close()
                        self.open()
                    except Exception as e:        # noqa: BLE001 — keep retrying, never tear down the loop
                        self._log.warning(f"[camera] reopen failed ({e}) - will retry")
            time.sleep(0.005)
            return None
        if self._dead_warned:
            self._log.info("[camera] frames back after outage")
        self._miss_since, self._dead_warned = None, False
        self._last = (bgr, time.perf_counter())   # cache the latest for a paused-scan refresh
        return self._last                         # wall-clock PTS (monotonic)


def _parse_device(device: Any) -> Any:
    """A numeric ``device`` (``0`` / ``"1"``) is a capture index; anything else
    (a ``/dev/videoN`` path, a UVC name/UID) passes through as a string."""
    if isinstance(device, int):
        return device
    s = str(device).strip()
    return int(s) if s.isdigit() else s
