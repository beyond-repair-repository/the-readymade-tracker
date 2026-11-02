#!/usr/bin/env python3
"""calibrate_camera — interactive camera + preprocess calibration (I/O roadmap D).

A **dearpygui** tool: a live video texture
with a floating, tabbed control HUD — **Hardware · Processing · Lens Calibration**. It is
the COMPANION to the `camera` input backend (A.1): run it FIRST (it owns the camera), tune,
Save, quit — then run the tracker (`readymade_tracker.py --source camera camera1.conf`), which reads the
conf. They never hold the device at once (single-client cameras).

Wired to the tracker so what you tune is what the tracker runs:
  • **Hardware** — the resolution picker + the camera controls (brightness / contrast /
    saturation / auto_exposure / exposure / gamma / gain / white_balance) applied LIVE through
    the SAME `lib.camera_controls` shim the tracker uses (slider ranges come from the device's
    real UVC min/max when available). A read-only "all device controls" list shows everything
    the UVC device exposes.
  • **Processing** — the `[preprocess]` core stage previewed with the REAL `lib.preprocess`:
    lens undistort, crop ROI, circular mask, and an N-frame flatfield capture. View toggles
    between the processed frame and the raw frame with crop/mask guides.
  • **Lens Calibration** — intrinsic **undistort calibration** in-tool: grab chessboard views,
    `cv2.calibrateCamera`, write the `lens.yaml` (`camera_matrix` + `dist_coeffs`) that
    `lib.preprocess` consumes, and flip undistort on.

Save writes `camera1.conf` ([camera] + [preprocess]) + `lens.yaml` beside it; a re-run RESUMES
from an existing `camera1.conf`.

    venv/bin/python tools/calibrate_camera.py --list-devices
    venv/bin/python tools/calibrate_camera.py --device 0 --width 1280 --height 720
    venv/bin/python tools/calibrate_camera.py --device /dev/video0 --chessboard 9x6

Keys:  s save · space grab chessboard view · c solve lens · d delete last view · p toggle HUD ·
       q / Esc quit.

The conf/lens I/O + control-range helpers (`format_camera_conf`, `write_camera_conf`,
`seed_from_conf`, `write_lens_yaml`, `chessboard_object_points`, `parse_chessboard`,
`logical_control_ranges`) are pure and unit-tested; the dpg loop needs a display + camera.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.camera_controls import CONTROL_NAMES, _UVC_NAMES  # noqa: E402
from lib.config import parse_ini_file  # noqa: E402


# Default control ranges (device-dependent; used when the device exposes no UVC min/max —
# exposure up to 5000). auto_exposure is a bool toggle, not in here.
_CONTROL_RANGES: dict[str, tuple[int, int]] = {
    "brightness": (0, 255), "contrast": (0, 255), "saturation": (0, 255),
    "gamma": (0, 500), "gain": (0, 255), "exposure": (0, 5000),
    "white_balance": (2000, 8000),
}

# The [preprocess] keys the tool controls, in write order.
_PREPROCESS_KEYS = ("preprocess", "lens_enabled", "lens_path", "flip", "crop_enabled",
                    "crop_top_x", "crop_top_y", "crop_width", "crop_height",
                    "mask_enabled", "mask_cx", "mask_cy", "mask_radius",
                    "flatfield_enabled", "flatfield_path")

_FLIP_CHOICES = ("none", "x", "y", "xy")   # x = horizontal mirror, y = vertical, xy = both
_FLIP_CV = {"x": 1, "y": 0, "xy": -1}      # cv2.flip codes (matches lib.preprocess); "none" → no flip

_FLATFIELD_FRAMES = 30          # N frames averaged into the flatfield


# --------------------------------------------------------------------------- #
# Conf I/O — pure, unit-tested (round-trips through the loader)
# --------------------------------------------------------------------------- #

def format_camera_conf(camera: Mapping[str, Any], preprocess: Mapping[str, Any]) -> str:
    """Render a ``camera1.conf`` string: a ``[camera]`` section (device / geometry /
    backend + the tuned controls) and a ``[preprocess]`` section. Only the camera
    controls present in ``camera`` are written (an absent control = leave the camera
    default). Booleans render as ``true``/``false`` so they coerce back correctly."""
    def val(v: Any) -> str:
        if isinstance(v, bool):
            return "true" if v else "false"
        return str(v)

    lines = ["# camera1.conf - written by tools/calibrate_camera.py (roadmap D).",
             "# Consume with:  readymade_tracker.py --source camera camera1.conf",
             "", "[camera]"]
    for key in ("device", "width", "height", "fps", "backend"):
        if key in camera:
            lines.append(f"{key} = {val(camera[key])}")
    if "auto_exposure" in camera:
        lines.append(f"auto_exposure = {val(camera['auto_exposure'])}")
    for key in CONTROL_NAMES:
        if key == "auto_exposure":
            continue
        if key in camera and camera[key] is not None:
            lines.append(f"{key} = {val(camera[key])}")

    lines += ["", "[preprocess]"]
    for key in _PREPROCESS_KEYS:
        if key in preprocess and preprocess[key] not in (None, ""):
            lines.append(f"{key} = {val(preprocess[key])}")
    return "\n".join(lines) + "\n"


def _atomic_write(p: Path, text: str) -> None:
    """Write ``text`` to ``p`` atomically (temp + fsync + os.replace), preserving the file's existing
    permission bits and — when running as root (this tool needs root for a pyuvc camera) — handing
    ownership back to whoever owns the parent directory. Without this, a sudo run would leave the conf
    0600/root, locking a later non-root run (and the webgui) out of reading + re-saving it."""
    import os
    import tempfile
    try:
        keep_mode = os.stat(p).st_mode & 0o777
    except OSError:
        keep_mode = 0o664
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=p.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, keep_mode)
        try:
            if hasattr(os, "geteuid") and os.geteuid() == 0:
                d = os.stat(p.parent)
                os.chown(tmp, d.st_uid, d.st_gid)
        except OSError:
            pass
        os.replace(tmp, str(p))
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def write_camera_conf(path: str | Path, camera: Mapping[str, Any],
                      preprocess: Mapping[str, Any]) -> None:
    _atomic_write(Path(path), format_camera_conf(camera, preprocess))


def seed_from_conf(path: str | Path) -> tuple[dict, dict]:
    """Read an existing ``camera1.conf`` back into ``(camera, preprocess)`` raw-string
    dicts, so a re-run RESUMES from the last calibration instead of the defaults.
    Missing / unparseable file → two empty dicts (start fresh)."""
    p = Path(path)
    if not p.is_file():
        return {}, {}
    try:
        raw = parse_ini_file(p)
    except Exception:                            # noqa: BLE001 — a broken conf just means "start fresh"
        return {}, {}
    return dict(raw.get("camera", {})), dict(raw.get("preprocess", {}))


def _as_int(raw: Mapping[str, Any], key: str, default: int) -> int:
    try:
        return int(str(raw[key]).strip())
    except (KeyError, ValueError):
        return default


def _as_bool(raw: Mapping[str, Any], key: str, default: bool) -> bool:
    v = str(raw.get(key, "")).strip().lower()
    if v in ("true", "yes", "on", "1"):
        return True
    if v in ("false", "no", "off", "0"):
        return False
    return default


def logical_control_ranges(cap: Any) -> dict[str, tuple[int, int, int]]:
    """For each ranged logical control, ``(min, max, current)`` — pulled from the
    device's real UVC control (via the shim's name map) when the capture exposes
    ``uvc_controls``, else a sane default range. Drives the Hardware-tab sliders so
    they match the actual device. auto_exposure (a bool toggle) is excluded."""
    uvc = getattr(cap, "uvc_controls", None)
    uvc = uvc if isinstance(uvc, dict) else None
    out: dict[str, tuple[int, int, int]] = {}
    for name, (lo, hi) in _CONTROL_RANGES.items():
        cur = (lo + hi) // 2
        if uvc:
            for disp in _UVC_NAMES.get(name, ()):
                c = uvc.get(disp)
                if c is not None:
                    lo = int(getattr(c, "min_val", lo))
                    hi = int(getattr(c, "max_val", hi))
                    cur = int(getattr(c, "value", cur))
                    break
        if hi <= lo:
            hi = lo + 1
        out[name] = (lo, hi, max(lo, min(hi, cur)))
    return out


# --------------------------------------------------------------------------- #
# Intrinsic (undistort) calibration — pure helpers, unit-tested through preprocess
# --------------------------------------------------------------------------- #

def parse_chessboard(spec: str) -> tuple[int, int]:
    """``"9x6"`` -> ``(9, 6)`` = the INNER-corner count (cols, rows) of the
    calibration board (a printed N×M-square board has (N-1)×(M-1) inner corners)."""
    try:
        cols, rows = (int(x) for x in str(spec).lower().split("x"))
    except (ValueError, TypeError) as e:
        raise ValueError(f"bad --chessboard {spec!r}; expected COLSxROWS like 9x6") from e
    if cols < 2 or rows < 2:
        raise ValueError(f"--chessboard {spec!r}: need >=2 inner corners per side")
    return cols, rows


def chessboard_object_points(cols: int, rows: int, square_size: float = 1.0):
    """The board's 3-D model points (z=0 plane), one per inner corner, spaced
    ``square_size`` apart — the ``objectPoints`` fed to ``cv2.calibrateCamera``.
    ``square_size`` only scales the (unused-for-undistort) extrinsics; the distortion
    coefficients + normalized intrinsics are scale-free, so 1.0 is fine for undistort."""
    import numpy as np
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    return objp * float(square_size)


def write_lens_yaml(path: str | Path, camera_matrix, dist_coeffs, *,
                    image_size: tuple[int, int] | None = None, rms: float | None = None) -> None:
    """Write a lens calibration YAML in the exact shape ``lib.preprocess`` reads
    (``camera_matrix`` + ``dist_coeffs`` nodes via ``cv2.FileStorage``). Distortion
    is written as a row vector so ``.mat()`` round-trips cleanly."""
    import cv2
    import numpy as np
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    fs.write("camera_matrix", np.asarray(camera_matrix, dtype=np.float64))
    fs.write("dist_coeffs", np.asarray(dist_coeffs, dtype=np.float64).reshape(1, -1))
    if image_size is not None:
        fs.write("image_width", int(image_size[0]))
        fs.write("image_height", int(image_size[1]))
    if rms is not None:
        fs.write("avg_reprojection_error", float(rms))
    fs.release()


# --------------------------------------------------------------------------- #
# Interactive dearpygui app (not unit-tested — needs a display + a camera)
# --------------------------------------------------------------------------- #

class _CalibApp:  # pragma: no cover
    """The dearpygui calibration app. Holds camera + processing + calibration state,
    builds the tabbed HUD, runs the render/capture loop, and writes camera1.conf +
    lens.yaml. Modules are imported lazily so importing this file stays display-free."""

    def __init__(self, args: argparse.Namespace) -> None:
        import threading

        import cv2
        import dearpygui.dearpygui as dpg
        import numpy as np

        from lib.camera_controls import CameraControls, open_capture
        from lib.logger import Logger
        from lib.preprocess import Preprocess, PreprocessConfig

        self.dpg, self.cv2, self.np, self.threading = dpg, cv2, np, threading
        self._Preprocess, self._PreprocessConfig = Preprocess, PreprocessConfig
        self.args = args
        self.log = Logger(level="info", stderr_level="warning")

        # RESUME everything from the supplied conf (--conf / --out) FIRST, so device / backend /
        # resolution / fps / controls all default to the prior calibration. Precedence per field:
        # an explicit CLI flag > the conf value > the hardcoded default (the geometry/device/backend
        # CLI args default to None precisely so "not given" is distinguishable from a real value).
        seed_cam, seed_pp = seed_from_conf(args.out)
        self.seed_cam = seed_cam
        self._device_str = args.device if args.device is not None else str(seed_cam.get("device", "0"))
        self.device = int(self._device_str) if str(self._device_str).isdigit() else self._device_str
        self._backend = args.backend if args.backend is not None else str(seed_cam.get("backend", "auto"))
        req_w = args.width if args.width is not None else _as_int(seed_cam, "width", 1280)
        req_h = args.height if args.height is not None else _as_int(seed_cam, "height", 720)
        req_fps = args.fps if args.fps is not None else _as_int(seed_cam, "fps", 30)

        self.out = args.out
        self.out_dir = Path(args.out).resolve().parent
        self.lens_out = (str(Path(args.lens_out).resolve()) if args.lens_out
                         else str(self.out_dir / "lens.yaml"))

        self.cap = open_capture(self.device, self._backend, self.log)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open device {self.device!r}")
        self.shim = CameraControls(self.cap, device=self.device, logger=self.log)
        self._apply_geometry(req_w, req_h, req_fps)

        # controls SAVED to camera1.conf = only what's carried over from a prior calibration
        # (seed_cam) PLUS what the user explicitly touches here. An untouched control stays
        # ABSENT (so the tracker leaves the camera default) — never a silent midpoint.
        self.controls = {n: _as_int(seed_cam, n, 0) for n in CONTROL_NAMES
                         if n != "auto_exposure" and n in seed_cam}
        self.auto_exposure = _as_bool(seed_cam, "auto_exposure", True)
        self.proc = {
            "flip": (str(seed_pp.get("flip", "") or args.flip or "none").lower()
                     if (str(seed_pp.get("flip", "") or args.flip or "none").lower() in _FLIP_CHOICES)
                     else "none"),
            # crop stored as X / Y / Width / Height (converted to the preprocess corner model at the
            # boundary — see _crop_corners). A seeded bottom of 0 means "to edge"; W/H of 0 here is
            # resolved to the full remaining extent once the real frame size is known (_resolve_crop).
            "crop_enabled": _as_bool(seed_pp, "crop_enabled", False),
            "crop_x": _as_int(seed_pp, "crop_top_x", 0),
            "crop_y": _as_int(seed_pp, "crop_top_y", 0),
            "crop_w": _as_int(seed_pp, "crop_width", 0),
            "crop_h": _as_int(seed_pp, "crop_height", 0),
            "mask_enabled": _as_bool(seed_pp, "mask_enabled", False),
            "mask_cx": _as_int(seed_pp, "mask_cx", 0),
            "mask_cy": _as_int(seed_pp, "mask_cy", 0),
            "mask_radius": _as_int(seed_pp, "mask_radius", 0),
        }
        self.lens_path = args.calibration or seed_pp.get("lens_path", "")
        self.lens_on = _as_bool(seed_pp, "lens_enabled", False) and bool(self.lens_path)
        self.flatfield_path = seed_pp.get("flatfield_path", "") or args.flatfield
        self.view = "processed"     # or "raw"

        self._ff = {"cap": False, "buf": None, "n": 0}
        self._pp_cache = {"sig": None, "pre": None}
        self.cb_cols, self.cb_rows = parse_chessboard(args.chessboard)
        self.calib = {"active": False, "objp": [], "imgp": [], "size": None,
                      "found": False, "corners": None, "gray": None,
                      "rms": None, "K": None, "D": None}

        self._reader = None
        self._pending_mode = None   # a resolution the combo picked; the main loop applies it
        self._tex = "calib_tex"
        self._tw = self._th = 0     # current texture (raw capture) size
        # the resolution/fps actually in effect (updated from the first frame + mode switches);
        # this — NOT the CLI defaults — is what gets saved so the tracker runs the SAME mode.
        self.width, self.height, self.fps = int(req_w), int(req_h), int(req_fps)

        # apply the resumed calibration to the device NOW so the live preview shows what the
        # tracker will reproduce (the tracker applies these same values on open).
        self._safe_apply({**self.controls, "auto_exposure": self.auto_exposure})

    # -- camera helpers ------------------------------------------------------ #

    def _apply_geometry(self, w: int, h: int, fps: int) -> None:
        cv2 = self.cv2
        if w:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        if h:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        if fps:
            self.cap.set(cv2.CAP_PROP_FPS, fps)

    class _Reader:
        def __init__(self, cap, threading):
            self._cap = cap
            self._lock = threading.Lock()
            self._run = True
            ok, frame = cap.read()
            self._frame = frame if ok else None
            self._t = threading.Thread(target=self._loop, daemon=True)
            self._t.start()

        def _loop(self):
            import time
            while self._run:
                ok, frame = self._cap.read()
                if ok and frame is not None:
                    with self._lock:
                        self._frame = frame
                else:
                    time.sleep(0.005)

        def read(self):
            with self._lock:
                return None if self._frame is None else self._frame.copy()

        def stop(self):
            self._run = False
            self._t.join(timeout=1.0)

    # -- live processing (real lib.preprocess, rebuilt only on geometry change) - #

    def _crop_pp(self) -> dict:
        """The crop X/Y/W/H as the preprocess keys (origin + width/height). W/H of 0 = to the edge."""
        return {"crop_top_x": int(self.proc["crop_x"]), "crop_top_y": int(self.proc["crop_y"]),
                "crop_width": int(self.proc["crop_w"]), "crop_height": int(self.proc["crop_h"])}

    def _crop_rect_corners(self, fw: int, fh: int) -> tuple[int, int, int, int]:
        """Display corners (x0, y0, x1, y1) for drawing the ROI rectangle — a 0 width/height extends to
        the frame edge (fw/fh)."""
        x, y = int(self.proc["crop_x"]), int(self.proc["crop_y"])
        w, h = int(self.proc["crop_w"]), int(self.proc["crop_h"])
        return x, y, (x + w if w > 0 else fw), (y + h if h > 0 else fh)

    def _flip_frame(self, bgr):
        """Apply the current FLIP (mirror) to a frame — returns a fresh array so the caller can draw on
        it without touching the source raw frame. 'none' → a plain copy."""
        code = _FLIP_CV.get(self.proc["flip"])
        return self.cv2.flip(bgr, code) if code is not None else bgr.copy()

    def _detect_frame(self, bgr):
        """The frame the LENS undistorts = flip → crop of the raw frame. The pipeline is
        flip → crop → undistort, so the lens is calibrated at the CROP size — chessboard detection runs on
        THIS frame, and the solved K/D matches what the tracker applies. Returns (frame, (ox, oy)) where
        the offset is the crop origin (0,0 when crop is off) — used to place the overlay in the raw view."""
        f = self._flip_frame(bgr)
        ox = oy = 0
        if self.proc["crop_enabled"]:
            x0, y0, x1, y1 = self._crop_rect_corners(f.shape[1], f.shape[0])
            x0, y0 = max(0, int(x0)), max(0, int(y0))
            x1, y1 = min(f.shape[1], int(x1)), min(f.shape[0], int(y1))
            if x1 > x0 and y1 > y0:
                f, ox, oy = f[y0:y1, x0:x1], x0, y0
        return f, (ox, oy)

    def _corners_for_raw_view(self, corners):
        """Corners are detected on the flipped + CROPPED frame (the undistort input). The processed view
        IS that frame so it draws them as-is; the RAW view shows the full flipped frame, so shift them
        back by the crop origin so they land on the board there too."""
        if not self.proc["crop_enabled"]:
            return corners
        out = corners.copy()
        out[:, 0, 0] += int(self.proc["crop_x"])
        out[:, 0, 1] += int(self.proc["crop_y"])
        return out

    def _resolve_crop(self, fw: int, fh: int) -> None:
        """Fill an unset (0) crop width/height with the full remaining extent, now that the real frame
        size is known — so the number fields always show concrete pixels, never an ambiguous 0."""
        if int(self.proc["crop_w"]) <= 0:
            self.proc["crop_w"] = max(1, fw - int(self.proc["crop_x"]))
        if int(self.proc["crop_h"]) <= 0:
            self.proc["crop_h"] = max(1, fh - int(self.proc["crop_y"]))

    def _pp_map(self, flatfield_on: bool) -> dict:
        m = {"preprocess": True, "lens_enabled": self.lens_on, "lens_path": self.lens_path,
             "flip": self.proc["flip"],
             "crop_enabled": self.proc["crop_enabled"], "mask_enabled": self.proc["mask_enabled"],
             **self._crop_pp(),
             "flatfield_enabled": flatfield_on and bool(self.flatfield_path),
             "flatfield_path": self.flatfield_path}
        for k in ("mask_cx", "mask_cy", "mask_radius"):
            m[k] = self.proc[k]
        return m

    def _process(self, bgr, flatfield_on: bool):
        m = self._pp_map(flatfield_on)
        sig = tuple(sorted(m.items()))
        if sig != self._pp_cache["sig"]:
            cfg = self._PreprocessConfig.from_mapping(m)   # no base_dir → resolve like the tracker
            cfg.load_artifacts(self.log)
            self._pp_cache["sig"], self._pp_cache["pre"] = sig, self._Preprocess(cfg, logger=self.log)
        proc, _gray = self._pp_cache["pre"].process(bgr)
        return proc

    # -- the render/capture loop --------------------------------------------- #

    def _await_first_frame(self) -> tuple[int, int]:
        import time
        deadline = time.perf_counter() + 3.0
        while time.perf_counter() < deadline:
            frame = self._reader.read()
            if frame is not None:
                return frame.shape[1], frame.shape[0]
            time.sleep(0.02)
        return self.width or 1280, self.height or 720   # never delivered → best guess

    def run(self) -> int:
        dpg = self.dpg
        dpg.create_context()
        dpg.add_texture_registry(tag="calib_tex_reg")
        with dpg.handler_registry():
            for key in ("Q", "Escape", "S", "P", "Spacebar", "C", "D"):
                dpg.add_key_press_handler(getattr(dpg, f"mvKey_{key}"), callback=self._on_key)
        # size the texture from a REAL frame (a UVC device resolves its mode only on the
        # first read, so cap.get() up-front can be 0 / the wrong mode).
        self._reader = self._Reader(self.cap, self.threading)
        w, h = self._await_first_frame()
        self.width, self.height = w, h            # the resolution actually in effect
        scale = min(1.0, 1280.0 / w)
        dpg.create_viewport(title="ReadyMade Tracker camera calibration",
                            width=int(w * scale) + 40, height=int(h * scale) + 60, vsync=False)
        dpg.setup_dearpygui()
        self._build_ui(w, h)
        dpg.show_viewport()

        while dpg.is_dearpygui_running():
            if self._pending_mode is not None:    # resolution picked in the combo — switch on THIS
                self._apply_mode(self._pending_mode)      # (main) thread, never in the callback thread
                self._pending_mode = None
                continue
            frame = self._reader.read() if self._reader else None
            if frame is not None:
                if frame.shape[1] != self._tw or frame.shape[0] != self._th:
                    self.width, self.height = frame.shape[1], frame.shape[0]
                    self._build_ui(frame.shape[1], frame.shape[0])   # self-heal on any size change
                self._render(frame)
            dpg.render_dearpygui_frame()

        if self._reader:
            self._reader.stop()
        self.cap.release()
        dpg.destroy_context()
        self.log.close()
        return 0

    def _apply_mode(self, mode) -> None:
        """Switch the live capture mode (main thread): stop the reader so no two threads
        grab the non-thread-safe pyuvc device at once, set the mode, invalidate stale
        calibration + preprocess caches, rebuild the UI, restart the reader."""
        if self._reader:
            self._reader.stop()
            self._reader = None
        try:
            self.cap.frame_mode = mode
        except Exception as e:                    # noqa: BLE001
            self.log.warning(f"[calibrate] mode switch failed: {e}")
        w = int(getattr(mode, "width", 0)) or self.width
        h = int(getattr(mode, "height", 0)) or self.height
        self.width, self.height = w, h
        self.fps = int(getattr(mode, "fps", 0)) or self.fps
        self.calib.update(objp=[], imgp=[], size=(w, h), found=False, corners=None, gray=None)
        self._pp_cache["sig"] = None
        self._reader = self._Reader(self.cap, self.threading)
        self._await_first_frame()
        self._build_ui(w, h)

    def _render(self, bgr) -> None:
        dpg, cv2, np = self.dpg, self.cv2, self.np
        h, w = bgr.shape[:2]
        self._pull_crop_fields()      # live crop values → ROI overlay + preprocess update as you type

        # Process ONCE per frame. During a flatfield capture, preview the PRE-flatfield
        # geometry (exactly what's being averaged) so the capture + display share one
        # preprocess signature (no per-frame lens-remap rebuild thrash).
        capturing = self._ff["cap"]
        proc = self._process(bgr, flatfield_on=not capturing)
        if capturing:
            geom = proc.astype(np.float32)
            if self._ff["buf"] is not None and geom.shape != self._ff["buf"].shape:
                self._ff.update(buf=None, n=0)
            self._ff["buf"] = geom if self._ff["buf"] is None else self._ff["buf"] + geom
            self._ff["n"] += 1
            self._set_text("ff_status", f"capturing {self._ff['n']}/{_FLATFIELD_FRAMES}")
            if self._ff["n"] >= _FLATFIELD_FRAMES:
                avg = (self._ff["buf"] / _FLATFIELD_FRAMES).astype(np.uint8)
                path = self.flatfield_path or str(self.out_dir / "flatfield.png")
                cv2.imwrite(path, avg)
                self.flatfield_path = path
                self._ff.update(cap=False, buf=None)
                self._pp_cache["sig"] = None
                self._set_text("ff_status", f"saved {Path(path).name} (applied)")

        # chessboard detection on the flip+CROP frame — the exact frame the lens undistorts (pipeline is
        # flip → crop → undistort), so the solve produces K/D at the crop size the tracker applies.
        cal = self.calib
        if cal["active"]:
            det, _ = self._detect_frame(bgr)
            gray = cv2.cvtColor(det, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(
                gray, (self.cb_cols, self.cb_rows),
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK)
            cal["found"] = found
            if found:
                crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
                cal["corners"] = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), crit)
                cal["gray"], cal["size"] = gray, (det.shape[1], det.shape[0])
            self._set_text("calib_live", f"views {len(cal['imgp'])}/{self.args.min_views}  "
                           f"board {'FOUND' if found else '-'}")

        # build the display: processed frame, or raw + crop/mask guides
        if self.view == "processed":
            disp = np.zeros((h, w, 3), np.uint8)
            ph, pw = proc.shape[:2]
            disp[:ph, :pw] = proc[:h, :w]         # place (post-crop is ≤ raw); top-left, black pad
            if self.proc["mask_enabled"] and self.proc["mask_radius"] > 0:
                # mask affects the downstream GRAY only (not the visible bgr) — show its ROI as a
                # guide, in POST-crop coords (matching how lib.preprocess shifts the centre).
                ox = int(self.proc["crop_x"]) if self.proc["crop_enabled"] else 0
                oy = int(self.proc["crop_y"]) if self.proc["crop_enabled"] else 0
                cv2.circle(disp, (self.proc["mask_cx"] - ox, self.proc["mask_cy"] - oy),
                           self.proc["mask_radius"], (0, 255, 0), 1)
        else:
            disp = self._flip_frame(bgr)     # raw view respects FLIP (crop coords live in the flipped frame)
            # ALWAYS draw the crop rectangle as a guide (bright when enabled, dim otherwise) so it can
            # be positioned before enabling — switch to this 'raw' view to place it.
            x0, y0, x1, y1 = self._crop_rect_corners(w, h)
            col = (0, 255, 255) if self.proc["crop_enabled"] else (0, 150, 150)
            cv2.rectangle(disp, (max(0, x0), max(0, y0)),
                          (min(x1, w - 1), min(y1, h - 1)), col, 2)
            if self.proc["mask_enabled"] and self.proc["mask_radius"] > 0:
                cv2.circle(disp, (self.proc["mask_cx"], self.proc["mask_cy"]),
                           self.proc["mask_radius"], (0, 255, 0), 2)
        if cal["active"] and cal["corners"] is not None:
            # corners are in flip+crop coords (the detect frame). The processed view IS that frame → draw
            # them as-is; the raw view shows the full flipped frame → shift back by the crop origin.
            oc = (cal["corners"] if self.view == "processed"
                  else self._corners_for_raw_view(cal["corners"]))
            cv2.drawChessboardCorners(disp, (self.cb_cols, self.cb_rows), oc, cal["found"])

        rgba = cv2.cvtColor(disp, cv2.COLOR_BGR2RGBA).astype(np.float32) / 255.0
        dpg.set_value(self._tex, rgba.ravel())

    # -- UI ------------------------------------------------------------------ #

    def _build_ui(self, w: int, h: int) -> None:
        dpg, np = self.dpg, self.np
        self._tw, self._th = w, h
        if dpg.does_item_exist(self._tex):
            dpg.delete_item(self._tex)
        dpg.add_dynamic_texture(width=w, height=h, default_value=np.zeros(w * h * 4, np.float32),
                                tag=self._tex, parent="calib_tex_reg")
        scale = min(1.0, 1280.0 / w)
        dw, dh = int(w * scale), int(h * scale)

        if dpg.does_item_exist("video_win"):
            dpg.delete_item("video_win")
        with dpg.window(tag="video_win", pos=(0, 0), width=dw, height=dh, no_title_bar=True,
                        no_move=True, no_resize=True, no_scrollbar=True,
                        no_bring_to_front_on_focus=True):
            dpg.add_image(self._tex, width=dw, height=dh)

        if dpg.does_item_exist("hud"):
            dpg.delete_item("hud")
        with dpg.window(tag="hud", label="Calibration", width=430, height=min(dh - 20, 760),
                        pos=(max(10, dw - 450), 20), no_collapse=True):
            with dpg.tab_bar():
                self._tab_hardware()
                self._tab_processing(w, h)
                self._tab_lens()

    def _tab_hardware(self) -> None:
        dpg = self.dpg
        with dpg.tab(label="Hardware"):
            dpg.add_spacer(height=4)
            self._resolution_combo()
            dpg.add_button(label="Save camera1.conf  (S)", callback=self._save_conf)
            dpg.add_text("", tag="save_status", color=(120, 255, 120), wrap=400)
            dpg.add_separator()
            dpg.add_text("Controls (only ones you touch - or a resumed conf - are saved):",
                         color=(120, 200, 255))
            dpg.add_radio_button(("Manual", "Auto"), horizontal=True,
                                 default_value="Auto" if self.auto_exposure else "Manual",
                                 callback=self._on_auto_exposure, label="auto_exposure")
            # slider default = the saved/resumed value if any, else the device's current value
            # (UVC) or a midpoint (cv2). Only touching a slider (or resuming a conf) puts a
            # control into self.controls, so an untouched control is NOT written — the tracker
            # then leaves the camera default (matching the [camera] "absent = default" rule).
            for name, (lo, hi, cur) in logical_control_ranges(self.cap).items():
                dpg.add_slider_int(label=name, min_value=lo, max_value=hi,
                                   default_value=self.controls.get(name, cur),
                                   callback=self._on_control, user_data=name, width=200)
            self._all_device_controls()

    def _all_device_controls(self) -> None:
        dpg = self.dpg
        uvc = getattr(self.cap, "uvc_controls", None)
        if not isinstance(uvc, dict) or not uvc:
            return
        with dpg.collapsing_header(label="All device controls (live only - NOT saved)"):
            for disp, ctrl in uvc.items():
                try:
                    lo = int(getattr(ctrl, "min_val", 0))
                    hi = int(getattr(ctrl, "max_val", 255))
                    val = int(getattr(ctrl, "value", lo))
                except (TypeError, ValueError):
                    continue
                if hi <= lo:
                    hi = lo + 1
                if lo == 0 and hi == 1:
                    dpg.add_radio_button(("Off", "On"), horizontal=True, label=disp,
                                         default_value="On" if val else "Off",
                                         callback=self._on_raw_ctrl,
                                         user_data=(ctrl, {"Off": 0, "On": 1}))
                else:
                    dpg.add_slider_int(label=disp, min_value=lo, max_value=hi,
                                       default_value=max(lo, min(hi, val)), width=190,
                                       callback=self._on_raw_ctrl, user_data=(ctrl, None))

    def _resolution_combo(self) -> None:
        dpg = self.dpg
        modes = list(getattr(self.cap, "available_modes", []) or [])
        if not modes:
            return
        self._mode_map, items, default = {}, [], ""
        cur = getattr(self.cap, "frame_mode", None)
        for m in modes:
            w, h = getattr(m, "width", "?"), getattr(m, "height", "?")
            fps = getattr(m, "fps", "?")
            fmt = str(getattr(m, "format_name", "")).replace("FrameFormat.", "")
            s = f"{w}x{h} @ {fps} [{fmt}]"
            self._mode_map[s] = m
            items.append(s)
            if m == cur:
                default = s
        dpg.add_combo(items, label="Resolution", default_value=default or (items[0] if items else ""),
                      callback=self._on_mode, width=230)

    def _tab_processing(self, w: int, h: int) -> None:
        dpg = self.dpg
        self._resolve_crop(w, h)                 # fill an unset W/H with the full extent (concrete px)
        with dpg.tab(label="Processing"):
            dpg.add_spacer(height=4)
            dpg.add_radio_button(("processed", "raw"), horizontal=True, default_value=self.view,
                                 label="view", callback=lambda s, a: setattr(self, "view", a))
            dpg.add_checkbox(label="Lens undistort", default_value=self.lens_on,
                             callback=self._on_lens_toggle)
            dpg.add_combo(_FLIP_CHOICES, label="Flip (mirror)", default_value=self.proc["flip"],
                          callback=self._on_flip, width=120)
            dpg.add_separator()
            dpg.add_text("Crop ROI  -  X, Y, Width, Height in raw px", color=(120, 200, 255))
            dpg.add_text("(pick 'raw' view above to see the rectangle)", color=(150, 150, 150))
            dpg.add_checkbox(label="Enable crop", default_value=self.proc["crop_enabled"],
                             callback=lambda s, a: self._set_proc("crop_enabled", a))
            self._crop_field("X", "crop_x", w)
            self._crop_field("Y", "crop_y", h)
            self._crop_field("Width", "crop_w", w)
            self._crop_field("Height", "crop_h", h)
            dpg.add_separator()
            dpg.add_text("Circular mask", color=(120, 200, 255))
            dpg.add_checkbox(label="Enable mask", default_value=self.proc["mask_enabled"],
                             callback=lambda s, a: self._set_proc("mask_enabled", a))
            self._proc_slider("Center X", "mask_cx", w)
            self._proc_slider("Center Y", "mask_cy", h)
            self._proc_slider("Radius", "mask_radius", max(w, h))
            dpg.add_separator()
            dpg.add_text("Flatfield", color=(120, 200, 255))
            dpg.add_button(label=f"Capture flatfield ({_FLATFIELD_FRAMES}-frame avg)",
                           callback=self._on_ff_capture)
            dpg.add_text("idle", tag="ff_status", color=(180, 180, 180))

    def _proc_slider(self, label: str, key: str, hi: int) -> None:
        self.dpg.add_slider_int(label=label, min_value=0, max_value=hi,
                                default_value=max(0, min(hi, self.proc[key])),
                                callback=lambda s, a, k=key: self._set_proc(k, a))

    @staticmethod
    def _cropf_tag(key: str) -> str:
        return "cropf_" + key

    def _crop_field(self, label: str, key: str, hi: int) -> None:
        """A typed NUMBER field (not a slider) for a crop dimension — clamped to [0, hi]. Tagged so the
        ROI overlay + save can read it LIVE (dpg may not fire the change callback until focus leaves)."""
        self.dpg.add_input_int(label=label, tag=self._cropf_tag(key),
                               default_value=max(0, min(hi, int(self.proc[key]))),
                               min_value=0, max_value=hi, min_clamped=True, max_clamped=True,
                               step=1, step_fast=10, width=160,
                               callback=lambda s, a, k=key: self._set_proc(k, a))

    def _pull_crop_fields(self) -> None:
        """Read the crop number fields straight from their widgets into ``self.proc`` — so the ROI
        rectangle, the live preprocess, and save all reflect what's currently TYPED, without waiting for
        dpg to fire the field's change callback (it can defer until the field loses focus / Enter)."""
        dpg = self.dpg
        for key in ("crop_x", "crop_y", "crop_w", "crop_h"):
            tag = self._cropf_tag(key)
            if dpg.does_item_exist(tag):
                try:
                    self.proc[key] = max(0, int(dpg.get_value(tag)))
                except (TypeError, ValueError):
                    pass

    def _tab_lens(self) -> None:
        dpg = self.dpg
        with dpg.tab(label="Lens Calibration"):
            dpg.add_spacer(height=4)
            dpg.add_checkbox(label="Detect chessboard (overlay)", default_value=False,
                             callback=lambda s, a: self.calib.update(active=a))
            dpg.add_text(f"board: {self.cb_cols}x{self.cb_rows} inner corners "
                         f"(--chessboard), need {self.args.min_views} views")
            dpg.add_text("views 0  board -", tag="calib_live", color=(255, 255, 120))
            dpg.add_text("", tag="calib_status", color=(120, 255, 120))
            with dpg.group(horizontal=True):
                dpg.add_button(label="Grab view (Space)", callback=self._on_grab)
                dpg.add_button(label="Delete last (D)", callback=self._on_del)
            with dpg.group(horizontal=True):
                dpg.add_button(label="Solve (C)", callback=self._on_solve)
                dpg.add_button(label="Save camera1.conf (S)", callback=self._save_conf)

    # -- callbacks ----------------------------------------------------------- #

    def _on_control(self, sender, value, name) -> None:
        self.controls[name] = int(value)
        self._safe_apply({name: int(value)})

    def _on_auto_exposure(self, sender, value) -> None:
        self.auto_exposure = (value == "Auto")
        self._safe_apply({"auto_exposure": self.auto_exposure})

    def _on_raw_ctrl(self, sender, value, user_data) -> None:
        ctrl, vmap = user_data
        try:
            ctrl.value = vmap[value] if vmap else int(value)
        except Exception as e:                   # noqa: BLE001
            self.log.info(f"[calibrate] control set failed: {e}")

    def _safe_apply(self, controls: dict) -> None:
        try:
            self.shim.apply(controls)
        except Exception as e:                   # noqa: BLE001 — never let a control crash the UI
            self.log.warning(f"[calibrate] apply failed: {e}")

    def _set_proc(self, key: str, value: Any) -> None:
        # bool BEFORE int (bool is an int subclass) so an enable checkbox stores True/False,
        # not 1/0 — matching lib.config._coerce_scalar and keeping the conf `crop_enabled = true`.
        self.proc[key] = bool(value) if isinstance(value, bool) else int(value)

    def _on_flip(self, sender, value) -> None:
        self.proc["flip"] = value if value in _FLIP_CHOICES else "none"
        self._pp_cache["sig"] = None            # rebuild the preview pipeline with the new mirror

    def _on_lens_toggle(self, sender, value) -> None:
        if value and not self.lens_path:
            self._set_text("calib_status", "no lens.yaml yet - calibrate first (or --calibration)")
            self.dpg.set_value(sender, False)
            return
        self.lens_on = bool(value)
        self._pp_cache["sig"] = None

    def _on_ff_capture(self, sender, app, user) -> None:
        self._ff.update(cap=True, n=0, buf=None)

    def _on_mode(self, sender, value) -> None:
        # DON'T touch the reader from this (callback) thread — hand the mode to the main
        # loop, which stops/restarts the reader so two threads never grab the (non-thread-
        # safe) pyuvc device at once. See run() / _apply_mode.
        m = getattr(self, "_mode_map", {}).get(value)
        if m is not None:
            self._pending_mode = m

    def _on_grab(self, *_) -> None:
        cal = self.calib
        if not cal["active"]:
            self._set_text("calib_status", "enable 'Detect chessboard' first")
            return
        if cal["found"] and cal["corners"] is not None:
            cal["objp"].append(chessboard_object_points(self.cb_cols, self.cb_rows, self.args.square_size))
            cal["imgp"].append(cal["corners"])
            self._set_text("calib_status", f"grabbed view {len(cal['imgp'])}")
        else:
            self._set_text("calib_status", "no board in view")

    def _on_del(self, *_) -> None:
        cal = self.calib
        if cal["imgp"]:
            cal["objp"].pop()
            cal["imgp"].pop()
            self._set_text("calib_status", f"deleted — {len(cal['imgp'])} left")

    def _on_solve(self, *_) -> None:
        cal, cv2 = self.calib, self.cv2
        if len(cal["imgp"]) < self.args.min_views:
            self._set_text("calib_status", f"need {self.args.min_views} views, have {len(cal['imgp'])}")
            return
        self._set_text("calib_status", "solving...")
        try:
            rms, K, D, _r, _t = cv2.calibrateCamera(cal["objp"], cal["imgp"], cal["size"], None, None)
        except Exception as e:                   # noqa: BLE001
            self._set_text("calib_status", f"solve error: {e}")
            return
        cal.update(rms=rms, K=K, D=D)
        write_lens_yaml(self.lens_out, K, D, image_size=cal["size"], rms=rms)
        self.lens_path, self.lens_on = self.lens_out, True
        self._pp_cache["sig"] = None
        self._set_text("calib_status", f"RMS {rms:.3f}px → {Path(self.lens_out).name} (undistort ON)")

    def _save_conf(self, *_) -> None:
        cv2 = self.cv2
        self._pull_crop_fields()      # capture typed-but-not-yet-committed crop values before writing
        # Save the resolution ACTUALLY in effect (live capture → tracked mode → CLI request),
        # NOT the CLI default — so a resolution picked in the combo, or the mode the device
        # snapped to, is what the tracker requests (and what lens/crop/mask were calibrated at).
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) or self.width
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) or self.height
        fps = int(self.cap.get(cv2.CAP_PROP_FPS) or 0) or self.fps
        camera = {"device": self._device_str, "width": w, "height": h, "fps": fps,
                  "backend": self._backend, "auto_exposure": self.auto_exposure}
        camera.update({k: int(v) for k, v in self.controls.items()})
        preprocess = {"preprocess": True, "lens_enabled": self.lens_on, "lens_path": self.lens_path,
                      "flip": self.proc["flip"], "crop_enabled": self.proc["crop_enabled"],
                      **self._crop_pp(),
                      "mask_enabled": self.proc["mask_enabled"], "mask_cx": self.proc["mask_cx"],
                      "mask_cy": self.proc["mask_cy"], "mask_radius": self.proc["mask_radius"],
                      "flatfield_enabled": bool(self.flatfield_path),
                      "flatfield_path": self.flatfield_path}
        path = str(Path(self.out).resolve())
        try:
            write_camera_conf(path, camera, preprocess)
            msg = f"SAVED  {path}  ({Path(path).stat().st_size} B)"
            self.log.info(f"[calibrate] {msg}")
        except Exception as e:                        # noqa: BLE001 — surface it, never crash the UI
            msg = f"SAVE FAILED: {type(e).__name__}: {e}"
            self.log.warning(f"[calibrate] {msg}")
        self._set_text("calib_status", msg)           # Lens tab
        self._set_text("save_status", msg)            # Hardware tab

    def _on_key(self, sender, key) -> None:
        dpg = self.dpg
        if key in (dpg.mvKey_Q, dpg.mvKey_Escape):
            dpg.stop_dearpygui()
        elif key == dpg.mvKey_S:
            self._save_conf()
        elif key == dpg.mvKey_Spacebar:
            self._on_grab()
        elif key == dpg.mvKey_C:
            self._on_solve()
        elif key == dpg.mvKey_D:
            self._on_del()
        elif key == dpg.mvKey_P and dpg.does_item_exist("hud"):
            dpg.configure_item("hud", show=not dpg.is_item_shown("hud"))

    def _set_text(self, tag: str, msg: str) -> None:
        if self.dpg.does_item_exist(tag):
            self.dpg.set_value(tag, msg)


def _list_devices() -> int:
    try:
        import uvc
    except Exception:                            # noqa: BLE001
        print("pyuvc not installed (pip install pupil-labs-uvc) - UVC device listing unavailable. "
              "cv2 devices are indices 0,1,2,... (try --device 0).")
        return 0
    devs = uvc.device_list()
    print(f"UVC devices: {len(devs)}")
    for i, d in enumerate(devs):
        print(f"  [{i}] {d.get('name')}  (uid {d.get('uid')})")
    return 0


def _run_interactive(args: argparse.Namespace) -> int:  # pragma: no cover
    try:
        app = _CalibApp(args)
    except RuntimeError as e:
        print(f"[calibrate] {e}", file=sys.stderr)
        return 3
    try:
        return app.run()
    except Exception as e:                       # noqa: BLE001 — surface a display/dpg failure cleanly
        print(f"[calibrate] GUI error: {type(e).__name__}: {e}", file=sys.stderr)
        return 4


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="calibrate_camera",
                                description="Interactive camera + preprocess calibration (roadmap D, dearpygui)")
    p.add_argument("--list-devices", action="store_true", help="list UVC devices + exit")
    # geometry / device / backend default to None so "not passed" is distinguishable from a real value
    # — an unpassed flag falls back to the --conf's saved value, else the hardcoded default (see __init__).
    p.add_argument("--device", default=None, help="capture index / /dev/videoN / UVC name (conf, else 0)")
    p.add_argument("--width", type=int, default=None, help="requested capture width (conf, else 1280)")
    p.add_argument("--height", type=int, default=None, help="requested capture height (conf, else 720)")
    p.add_argument("--fps", type=int, default=None, help="requested capture fps (conf, else 30)")
    p.add_argument("--backend", default=None, choices=("auto", "uvc", "cv2"),
                   help="camera open backend (conf, else auto = pyuvc on macOS, else cv2)")
    p.add_argument("--flip", default="none", choices=_FLIP_CHOICES,
                   help="mirror the frame: none / x (horizontal) / y (vertical) / xy (both)")
    p.add_argument("--calibration", "-c", default="", help="seed an existing lens YAML")
    p.add_argument("--flatfield", default="", help="seed a flatfield PNG path")
    p.add_argument("--chessboard", default="9x6", help="board INNER corners COLSxROWS (default 9x6)")
    p.add_argument("--square-size", dest="square_size", type=float, default=1.0,
                   help="chessboard square edge (scale-free for undistort, default 1.0)")
    p.add_argument("--min-views", dest="min_views", type=int, default=12,
                   help="chessboard views required before Solve (default 12)")
    p.add_argument("--lens-out", dest="lens_out", default="",
                   help="lens YAML output (default: lens.yaml next to --out)")
    p.add_argument("--out", "--conf", dest="out", default="camera1.conf",
                   help="conf to edit/create - loaded to RESUME its settings if it exists, and saved back (S)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(sys.argv[1:] if argv is None else argv))
    if args.list_devices:
        return _list_devices()
    return _run_interactive(args)


if __name__ == "__main__":
    raise SystemExit(main())
