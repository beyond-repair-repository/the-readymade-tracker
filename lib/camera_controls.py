"""Cross-platform camera **control shim**.

Maps a *logical* control name (``brightness``, ``exposure``, ``auto_exposure``,
…) to the right backend call, papering over the per-OS quirks. Backends, in
preference order:

  1. **pyuvc / libuvc** — UVC over libusb, *bypassing* AVFoundation, so
     exposure / gain / white-balance actually LOCK on **macOS** (OpenCV →
     AVFoundation exposes almost no settable ``CAP_PROP_*``: the macOS
     low-level-access wall). Claims the device exclusively — fine, the tracker
     is the SOLE camera owner.
  2. **v4l2-ctl** — Linux/V4L2 fallback (a subprocess to the same ``/dev/videoN``).
  3. **cv2.CAP_PROP_*** — last resort (Windows DirectShow/MSMF stick most).

An unsupported control **logs and continues** — the shim NEVER raises into the
pipeline (a camera that lacks a gamma control must not crash the tracker).

pyuvc is an OPTIONAL dependency (live-camera use only): every import of it here
is guarded, so this module loads cleanly on a host without it. The pyuvc + v4l2
paths need real hardware to exercise; the cv2 path is unit-tested with an
injected fake capture.

``open_capture(device, backend, log)`` returns a **cv2-``VideoCapture``-shaped**
object (``read`` / ``set`` / ``get`` / ``isOpened`` / ``release``) so the
``CameraBackend`` grab loop is uniform regardless of which backend actually
talks to the silicon; the pyuvc-ness lives entirely behind ``UvcCapture``.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import time
from typing import Any, Mapping

import cv2

from lib.logger import NULL_LOGGER

_IS_LINUX = platform.system() == "Linux"
_IS_MAC = platform.system() == "Darwin"
# settle after switching a UVC device out of auto-exposure before writing the manual
# value — some drivers drop a manual exposure written too soon after leaving auto.
_MANUAL_SETTLE_S = 0.1


# --------------------------------------------------------------------------- #
# The control table — one logical name, three backend spellings
# --------------------------------------------------------------------------- #
#
# Each entry: cv2 CAP_PROP_*, the V4L2 control id (v4l2-ctl `--set-ctrl`), and the
# ordered list of pyuvc `display_name`s to try (spellings vary by device/firmware).
# ``auto_exposure`` is special-cased (its *value* convention differs per backend).

_CV2_PROPS = {
    "brightness": cv2.CAP_PROP_BRIGHTNESS,
    "contrast": cv2.CAP_PROP_CONTRAST,
    "saturation": cv2.CAP_PROP_SATURATION,
    "gamma": cv2.CAP_PROP_GAMMA,
    "gain": cv2.CAP_PROP_GAIN,
    "exposure": cv2.CAP_PROP_EXPOSURE,
    # CAP_PROP_WB_TEMPERATURE exists from OpenCV 3.4; guard for older stubs.
    "white_balance": getattr(cv2, "CAP_PROP_WB_TEMPERATURE", 45),
    "auto_exposure": cv2.CAP_PROP_AUTO_EXPOSURE,
}

_V4L2_CTRLS = {
    "brightness": "brightness",
    "contrast": "contrast",
    "saturation": "saturation",
    "gamma": "gamma",
    "gain": "gain",
    "exposure": "exposure_time_absolute",
    "white_balance": "white_balance_temperature",
    "auto_exposure": "auto_exposure",
}

_UVC_NAMES = {
    "brightness": ("Brightness",),
    "contrast": ("Contrast",),
    "saturation": ("Saturation",),
    "gamma": ("Gamma",),
    "gain": ("Gain",),
    "exposure": ("Absolute Exposure Time", "Exposure Time, Absolute"),
    "white_balance": ("White Balance temperature", "White Balance Temperature"),
    "auto_exposure": ("Auto Exposure Mode",),
    "auto_white_balance": ("White Balance temperature,Auto",
                           "White Balance Temperature, Auto"),
}

# ``auto_exposure`` value conventions (the central per-OS quirk):
#   • V4L2 (Linux, and the Linux cv2/V4L2 backend): V4L2_CID_EXPOSURE_AUTO —
#     1 = MANUAL, 3 = APERTURE_PRIORITY (auto). (0/2 are reserved/rare.)
#   • Windows DirectShow/MSMF via cv2: normalized 0.25 = manual, 0.75 = auto.
#   • UVC auto-exposure-mode bitmap (pyuvc): 1 = Manual, 2 = Auto,
#     4 = Shutter-priority, 8 = Aperture-priority. Many cams expose only {1, 8},
#     so 8 is the safe "auto" (verify the exact value per device).
_AUTO_EXPOSURE = {
    "v4l2": {True: 3, False: 1},
    "cv2_posix": {True: 3, False: 1},        # Linux cv2 rides the V4L2 driver
    "cv2_win": {True: 0.75, False: 0.25},
    "uvc": {True: 8, False: 1},
}

# The logical controls this shim understands. Anything else
# in the config is ignored with a log line.
CONTROL_NAMES = ("brightness", "contrast", "saturation", "auto_exposure",
                 "exposure", "gamma", "gain", "white_balance")


# --------------------------------------------------------------------------- #
# The shim
# --------------------------------------------------------------------------- #

class CameraControls:
    """Apply logical camera controls to an open capture via the best available
    backend. Stateless apart from the resolved backend choice; every apply is
    fail-soft (an unsupported control logs and is skipped)."""

    def __init__(self, capture: Any, *, device: Any = None, logger: Any = None) -> None:
        self._cap = capture
        self._device = device
        self._log = logger or NULL_LOGGER
        self._backend = self._pick_backend()

    def _pick_backend(self) -> str:
        """pyuvc (a ``UvcCapture``) > v4l2-ctl (Linux + a resolvable /dev node) > cv2."""
        if isinstance(self._cap, UvcCapture) or hasattr(self._cap, "uvc_controls"):
            return "uvc"
        if _IS_LINUX and _v4l2_node(self._device) and shutil.which("v4l2-ctl"):
            return "v4l2"
        return "cv2"

    @property
    def backend(self) -> str:
        return self._backend

    def control_range(self, name: str) -> "tuple[int, int, int] | None":
        """``(min, max, step)`` the LIVE device accepts for a logical control, or ``None`` if unknown —
        so the webgui can bound a slider to the real range instead of a generic guess. Only the UVC
        backend reports per-control ranges reliably (v4l2 would need ``--list-ctrls``; cv2 exposes none)."""
        if self._backend != "uvc":
            return None
        controls = getattr(self._cap, "uvc_controls", None)
        if not isinstance(controls, dict):
            return None
        for display in _UVC_NAMES.get(name, ()):
            ctrl = controls.get(display)
            if ctrl is not None:
                try:
                    return int(ctrl.min_val), int(ctrl.max_val), int(getattr(ctrl, "step", 1) or 1)
                except (TypeError, ValueError, AttributeError):
                    return None
        return None

    def apply(self, controls: Mapping[str, Any]) -> None:
        """Apply every known, non-``None`` control in ``controls``. Order matters:
        ``auto_exposure`` first (manual mode must be engaged before a manual
        ``exposure`` value will stick), then the rest. When this call ENGAGES manual
        exposure AND also carries a manual ``exposure`` (the tracker's open-time bulk
        apply), wait a short settle after the mode switch before writing the value —
        several UVC drivers drop a manual exposure written too soon after leaving auto.
        A standalone exposure tweak (no ``auto_exposure`` in the dict) skips the
        settle, so the calibrator's slider stays responsive."""
        ordered = ["auto_exposure"] + [k for k in controls if k != "auto_exposure"]
        for name in ordered:
            if name not in controls:
                continue
            value = controls[name]
            if value is None:                    # unset in the conf → leave the camera default
                continue
            if name not in CONTROL_NAMES:
                self._log.info(f"[camera] ignoring unknown control {name!r}")
                continue
            try:
                self._apply_one(name, value)
            except Exception as e:               # noqa: BLE001 — a control must never crash the tracker
                self._log.warning(f"[camera] control {name}={value!r} failed "
                                  f"({self._backend}): {e} -> skipped")
            if name == "auto_exposure" and not bool(value) and controls.get("exposure") is not None:
                time.sleep(_MANUAL_SETTLE_S)     # let the driver leave auto before the manual write

    # -- per-control dispatch ------------------------------------------------ #

    def _apply_one(self, name: str, value: Any) -> None:
        if name == "auto_exposure":
            self._apply_auto_exposure(bool(value))
            return
        if name == "white_balance":          # manual-WB handshake: while auto-WB is engaged the
            self._disable_auto_wb()          # driver holds the temperature control read-only, so
            #                                  the write below is silently dropped (mirrors exposure)
        {"uvc": self._set_uvc, "v4l2": self._set_v4l2, "cv2": self._set_cv2}[self._backend](
            name, value)

    def _apply_auto_exposure(self, auto: bool) -> None:
        if self._backend == "uvc":
            self._set_uvc("auto_exposure", _AUTO_EXPOSURE["uvc"][auto])
        elif self._backend == "v4l2":
            self._set_v4l2("auto_exposure", _AUTO_EXPOSURE["v4l2"][auto])
        else:
            key = "cv2_win" if platform.system() == "Windows" else "cv2_posix"
            self._set_cv2("auto_exposure", _AUTO_EXPOSURE[key][auto])

    def _disable_auto_wb(self) -> None:
        """Turn auto white-balance OFF so a manual ``white_balance`` temperature sticks.
        Best-effort + fail-soft: a camera without an auto-WB control keeps its default
        (the temperature write then simply has no auto mode fighting it)."""
        try:
            if self._backend == "uvc":
                controls = self._cap.uvc_controls
                for display in _UVC_NAMES["auto_white_balance"]:
                    ctrl = controls.get(display)
                    if ctrl is not None:
                        ctrl.value = 0
                        self._log.info(f"[camera] uvc {display}=0 (manual WB)")
                        return
            elif self._backend == "v4l2":
                node = _v4l2_node(self._device)
                subprocess.run(["v4l2-ctl", "-d", node, "--set-ctrl",
                                "white_balance_temperature_auto=0"],
                               check=True, capture_output=True, text=True)
                self._log.info("[camera] v4l2-ctl white_balance_temperature_auto=0")
            else:
                prop = getattr(cv2, "CAP_PROP_AUTO_WB", None)
                if prop is not None:
                    self._cap.set(prop, 0.0)
                    self._log.info("[camera] cv2 CAP_PROP_AUTO_WB=0 (manual WB)")
        except Exception as e:               # noqa: BLE001 — never crash on a missing auto-WB control
            self._log.info(f"[camera] auto-WB disable unavailable ({self._backend}): {e}")

    # -- backends ------------------------------------------------------------ #

    def _set_cv2(self, name: str, value: Any) -> None:
        prop = _CV2_PROPS[name]
        ok = self._cap.set(prop, float(value))
        level = self._log.info if ok else self._log.warning
        level(f"[camera] cv2 set {name}={value} (prop {prop}) -> {'ok' if ok else 'rejected'}")

    def _set_v4l2(self, name: str, value: Any) -> None:
        node = _v4l2_node(self._device)
        ctrl = _V4L2_CTRLS[name]
        subprocess.run(["v4l2-ctl", "-d", node, "--set-ctrl", f"{ctrl}={int(round(float(value)))}"],
                       check=True, capture_output=True, text=True)
        self._log.info(f"[camera] v4l2-ctl {node} {ctrl}={value} -> ok")

    def _set_uvc(self, name: str, value: Any) -> None:
        controls = self._cap.uvc_controls           # {display_name: Control}
        for display in _UVC_NAMES[name]:
            ctrl = controls.get(display)
            if ctrl is not None:
                want = int(round(float(value)))
                v = _fit_uvc(ctrl, want)             # clamp to [min,max] + snap to step (device-reported)
                ctrl.value = v                       # so an out-of-range conf value doesn't get rejected
                note = "" if v == want else f" (clamped from {want} to {ctrl.min_val}..{ctrl.max_val})"
                self._log.info(f"[camera] uvc {display}={v}{note} -> ok")
                return
        raise KeyError(f"no UVC control for {name} (tried {_UVC_NAMES[name]})")


def _fit_uvc(ctrl: Any, v: int) -> int:
    """Coerce ``v`` into a UVC control's settable domain: clamp to ``[min_val, max_val]`` and snap to
    ``step`` (both device-reported). A value outside the range — or off the step grid — makes libuvc
    reject the write ('Could not set Value'), so a stale/foreign conf value would spam the console and
    never apply; fitting it first means the closest LEGAL value is written instead. Missing/garbage
    range metadata → pass ``v`` through unchanged (best-effort, never raise)."""
    try:
        lo, hi = int(ctrl.min_val), int(ctrl.max_val)
    except (TypeError, ValueError, AttributeError):
        return v
    if hi < lo:
        return v
    v = max(lo, min(hi, v))
    try:
        step = int(getattr(ctrl, "step", 1) or 1)
    except (TypeError, ValueError):
        step = 1
    if step > 1:                                     # snap to the nearest lo + k·step inside [lo, hi]
        v = lo + ((v - lo + step // 2) // step) * step
        v = max(lo, min(hi, v))
    return v


def _v4l2_node(device: Any) -> str | None:
    """A ``/dev/videoN`` path if ``device`` names or indexes one, else ``None``."""
    if isinstance(device, str) and device.startswith("/dev/video"):
        return device
    if isinstance(device, int):
        return f"/dev/video{device}"
    if isinstance(device, str) and device.isdigit():
        return f"/dev/video{int(device)}"
    return None


# --------------------------------------------------------------------------- #
# Capture opening — cv2 everywhere, pyuvc where it's the right tool (macOS/UVC)
# --------------------------------------------------------------------------- #

def open_capture(device: Any, backend: str = "auto", logger: Any = None, *,
                 strict: bool = False) -> Any:
    """Open the camera and return a cv2-``VideoCapture``-shaped object.

    ``backend``: ``auto`` (default) picks pyuvc on macOS when it's importable
    (the only way to lock exposure there), else cv2; ``uvc`` / ``cv2`` force it.
    A pyuvc failure falls back to cv2 (never leaves the tracker with no camera),
    unless ``strict``: then it raises. A REOPEN after an outage is strict, because
    while the USB camera is unplugged cv2's index 0 is a laptop's BUILT-IN camera,
    and the tracker would silently carry on with the wrong camera.
    """
    log = logger or NULL_LOGGER
    want_uvc = backend == "uvc" or (backend == "auto" and _IS_MAC and _uvc_available())
    if want_uvc:
        try:
            return UvcCapture(device, logger=log)
        except Exception as e:                   # noqa: BLE001 — degrade to cv2, don't abort
            if strict:
                raise
            log.warning(f"[camera] pyuvc open failed ({e}); falling back to cv2")
    return cv2.VideoCapture(_cv2_index(device))


def _cv2_index(device: Any) -> Any:
    """cv2.VideoCapture takes an int index or a str path/URL; coerce a numeric
    string to an int so ``device = 0`` in the conf opens index 0 (not a file)."""
    if isinstance(device, str) and device.isdigit():
        return int(device)
    return device


def _uvc_available() -> bool:
    try:
        import uvc  # noqa: F401
        return True
    except Exception:                            # noqa: BLE001
        return False


class UvcCapture:
    """A thin cv2-``VideoCapture``-shaped adapter over ``pyuvc`` — so the
    ``CameraBackend`` and the control shim treat a libuvc device exactly like a
    cv2 capture. Exercised only on real hardware; import-guarded so a host
    without pyuvc never trips over it.

    Resolution / fps requested via ``set(CAP_PROP_FRAME_WIDTH/HEIGHT/FPS, …)`` are
    buffered and resolved to the *nearest available UVC frame mode* on the first
    ``read`` (pyuvc sets all three atomically as ``frame_mode``)."""

    def __init__(self, device: Any, *, logger: Any = None) -> None:
        import uvc                               # noqa: F401 — guarded import
        self._uvc = uvc
        self._log = logger or NULL_LOGGER
        self._cap = uvc.Capture(self._resolve_uid(device))
        self._want = {"w": None, "h": None, "fps": None}
        self._mode_set = False
        self._opened = True

    def _resolve_uid(self, device: Any) -> Any:
        """Map a config ``device`` (index / name substring / UID) to a pyuvc UID."""
        devs = self._uvc.device_list()
        if isinstance(device, int) or (isinstance(device, str) and device.isdigit()):
            return devs[int(device)]["uid"]
        for d in devs:                           # a name substring, or an exact UID
            if device in (d.get("uid"), d.get("name")) or (
                    isinstance(device, str) and device.lower() in str(d.get("name", "")).lower()):
                return d["uid"]
        raise ValueError(f"no UVC device matches {device!r} (have "
                         f"{[d.get('name') for d in devs]})")

    @property
    def uvc_controls(self) -> dict:
        return {c.display_name: c for c in self._cap.controls}

    @property
    def available_modes(self) -> list:
        """The device's (width, height, fps, …) capture modes — for a mode picker."""
        return list(getattr(self._cap, "available_modes", []) or [])

    @property
    def frame_mode(self) -> Any:
        return getattr(self._cap, "frame_mode", None)

    @frame_mode.setter
    def frame_mode(self, mode: Any) -> None:
        self._cap.frame_mode = mode          # direct mode set (bypasses the lazy _ensure_mode)
        self._mode_set = True

    def set(self, prop: int, value: float) -> bool:
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            self._want["w"] = int(value)
        elif prop == cv2.CAP_PROP_FRAME_HEIGHT:
            self._want["h"] = int(value)
        elif prop == cv2.CAP_PROP_FPS:
            self._want["fps"] = int(value)
        else:
            return False
        self._mode_set = False
        return True

    def get(self, prop: int) -> float:
        mode = getattr(self._cap, "frame_mode", None)
        if mode is not None:
            w, h, fps = mode[0], mode[1], (mode[2] if len(mode) > 2 else 0)
            return {cv2.CAP_PROP_FRAME_WIDTH: float(w), cv2.CAP_PROP_FRAME_HEIGHT: float(h),
                    cv2.CAP_PROP_FPS: float(fps)}.get(prop, 0.0)
        return 0.0

    def _ensure_mode(self) -> None:
        if self._mode_set:
            return
        modes = list(getattr(self._cap, "available_modes", []) or [])
        w = self._want["w"] or (modes[0][0] if modes else 640)
        h = self._want["h"] or (modes[0][1] if modes else 480)
        fps = self._want["fps"] or 30
        best = _nearest_mode(modes, w, h, fps)
        if best is not None:
            self._cap.frame_mode = best
            self._log.info(f"[camera] uvc frame_mode {best} (requested {w}x{h}@{fps})")
        self._mode_set = True

    # Floor for one read's wait. A healthy stream delivers well inside this; a dead one makes
    # ``read`` return a miss after it, so the backend's outage logic can reopen the device.
    _min_frame_wait = 1.0

    def _frame_wait(self) -> float:
        """How long one ``read`` may wait: three frame intervals of the active mode, at least 1 s."""
        mode = getattr(self._cap, "frame_mode", None)
        try:
            fps = float(mode[2]) if mode is not None and len(mode) > 2 else 0.0
        except (TypeError, ValueError):
            fps = 0.0
        return max(self._min_frame_wait, 3.0 / fps) if fps > 0 else self._min_frame_wait

    def _grab(self) -> Any:
        """One pyuvc frame with a BOUNDED wait.

        pyuvc's ``get_frame()`` / ``get_frame_robust()`` wait with no timeout, and a dead stream
        never wakes them: libuvc frees a USB transfer that ends in ERROR / NO_DEVICE and never
        resubmits it, so after a cable glitch, a USB reset or another process touching the
        device, every transfer dies and the input thread blocks forever (the tracker "runs" with
        no frames, and the backend's reopen logic never gets a miss to count). Each dying transfer
        also wakes the waiter once WITHOUT a frame, which pyuvc reports as a bare ``TimeoutError``,
        and a corrupt MJPEG frame raises ``StreamError``; both are retried until the deadline.
        Captures without ``get_frame`` (fakes, older bindings) keep ``get_frame_robust``."""
        get_frame = getattr(self._cap, "get_frame", None)
        if get_frame is None:
            return self._cap.get_frame_robust()
        wait = self._frame_wait()
        deadline = time.perf_counter() + wait
        last: Exception | None = None
        while True:
            left = deadline - time.perf_counter()
            if left <= 0:
                why = f"; last error {type(last).__name__}: {last}" if last is not None and str(last) else ""
                raise TimeoutError(f"no frame within {wait:.1f} s{why}")
            try:
                return get_frame(timeout=max(0.05, left))
            except TimeoutError as e:            # woken by a dying transfer, or the wait ran out
                last = e
            except Exception as e:               # noqa: BLE001 — corrupt frame / stream won't start
                last = e
                time.sleep(0.02)

    def read(self) -> tuple[bool, Any]:
        self._ensure_mode()
        try:
            frame = self._grab()
        except Exception as e:                   # noqa: BLE001 — transient, mirror cv2's (False, None)
            if not getattr(self, "_read_warned", False):   # ONE warning per outage — a dead camera at
                self._read_warned = True                   #   ~30 attempts/s must not flood the log
                self._log.warning(f"[camera] uvc get_frame failed: {type(e).__name__}: {e} "
                                  f"- muting until frames return")
            return False, None
        self._read_warned = False
        bgr = getattr(frame, "bgr", None)
        if bgr is None:
            bgr = getattr(frame, "img", None)
        # COPY, and never return pyuvc's own array: its buffer belongs to the libuvc streaming
        # ring and is recycled under us, whereas the snapshot contract (lib.events.readonly) is
        # that a published frame is replaced, never rewritten — `readonly()` hands outputs a VIEW,
        # so a recycled buffer would change under a slow reader mid-frame. cv2.VideoCapture.read
        # already allocates per call; this is the one source that doesn't. One memcpy per grab.
        return (bgr is not None), (None if bgr is None else bgr.copy())

    def isOpened(self) -> bool:                  # noqa: N802 — cv2 API parity
        return self._opened

    def release(self) -> None:
        if self._opened:
            self._cap.close()
            self._opened = False


def _nearest_mode(modes: list, w: int, h: int, fps: int):
    """Pick the available (w,h,fps) mode closest to the request (area, then fps)."""
    if not modes:
        return None

    def cost(m):
        mw, mh = m[0], m[1]
        mfps = m[2] if len(m) > 2 else fps
        return (abs(mw * mh - w * h), abs(mfps - fps))

    return min(modes, key=cost)
