#!/usr/bin/env python3
"""calibrate_camera_web — browser-based camera + preprocess calibration.

A **zero-dependency-GUI** rewrite of ``calibrate_camera.py`` (which needs dearpygui).
Instead of a native window, this serves a local web page on ``http://127.0.0.1:PORT``
with an MJPEG live stream and HTML/JS controls: the same three-tab workflow
(Hardware · Processing · Lens Calibration), the same conf/lens output, the same
keyboard shortcuts.

Dependencies: only Python stdlib + OpenCV + numpy (no dearpygui, no Flask).

    venv/bin/python tools/calibrate_camera_web.py --list-devices
    venv/bin/python tools/calibrate_camera_web.py --device 0 --width 1280 --height 720
    venv/bin/python tools/calibrate_camera_web.py --device 0 --chessboard 9x6 --port 8088
    sudo venv/bin/python tools/calibrate_camera_web.py --backend uvc     # macOS: full UVC control

Keys (in the browser, when no field has focus):
    S save · Space grab chessboard view · C solve lens · D delete last view ·
    1/2/3 switch tabs · Q quit (stops the server).

Notes
-----
* The HTTP server is THREADED: the MJPEG stream is an endless response, so a
  single-threaded server would serve the stream and nothing else.
* On macOS without pyuvc, OpenCV's AVFoundation backend accepts almost none of the
  picture controls. The Hardware tab probes each control and marks the ones the
  driver rejects, rather than pretending they work.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.camera_controls import CONTROL_NAMES, _UVC_NAMES  # noqa: E402,F401
from lib.config import parse_ini_file  # noqa: E402


# ────────────────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────────────────

# Generic slider bounds, used when the device reports no range (cv2 backends).
_CONTROL_RANGES: dict[str, tuple[int, int]] = {
    "exposure": (0, 5000), "gain": (0, 255), "brightness": (0, 255),
    "contrast": (0, 255), "saturation": (0, 255), "gamma": (0, 500),
    "white_balance": (2000, 8000),
}

# The seven picture controls, in the order they appear in the UI and in the conf.
_SLIDER_ORDER = ("exposure", "gain", "brightness", "contrast", "saturation", "gamma",
                 "white_balance")

# Seed values when neither the conf nor the device supplies one.
_DEFAULT_CONTROLS = {"exposure": 156, "gain": 0, "brightness": 128, "contrast": 128,
                     "saturation": 128, "gamma": 100, "white_balance": 4600}

_PREPROCESS_KEYS = ("preprocess", "lens_enabled", "lens_path", "flip", "crop_enabled",
                    "crop_top_x", "crop_top_y", "crop_width", "crop_height",
                    "mask_enabled", "mask_cx", "mask_cy", "mask_radius",
                    "flatfield_enabled", "flatfield_path")

_FLIP_CHOICES = ("none", "x", "y", "xy")
_FLIP_CV = {"x": 1, "y": 0, "xy": -1}
_VIEWS = ("raw", "processed", "gray")

_FLATFIELD_FRAMES = 30

# Probed when the camera offers no mode list (cv2 backend, no ffmpeg).
_COMMON_RESOLUTIONS = [
    (640, 480), (800, 600), (960, 540), (1024, 768), (1280, 720), (1280, 960),
    (1600, 1200), (1920, 1080), (2560, 1440), (3840, 2160),
]
_COMMON_FPS = [5, 10, 15, 20, 24, 25, 30, 50, 60, 90, 120]


# ────────────────────────────────────────────────────────────────────────────
# Conf I/O (pure)
# ────────────────────────────────────────────────────────────────────────────

def format_camera_conf(camera: Mapping[str, Any], preprocess: Mapping[str, Any]) -> str:
    def val(v: Any) -> str:
        if isinstance(v, bool):
            return "true" if v else "false"
        return str(v)
    lines = ["# camera1.conf - written by tools/calibrate_camera_web.py",
             "# Consume with:  readymade_tracker.py --source camera camera1.conf",
             "", "[camera]"]
    for key in ("device", "width", "height", "fps", "backend"):
        if key in camera:
            lines.append(f"{key} = {val(camera[key])}")
    if "auto_exposure" in camera:
        lines.append(f"auto_exposure = {val(camera['auto_exposure'])}")
    for key in _SLIDER_ORDER:
        if key in camera and camera[key] is not None:
            lines.append(f"{key} = {val(camera[key])}")
    lines += ["", "[preprocess]"]
    for key in _PREPROCESS_KEYS:
        if key in preprocess and preprocess[key] not in (None, ""):
            lines.append(f"{key} = {val(preprocess[key])}")
    return "\n".join(lines) + "\n"


def _atomic_write(p: Path, text: str) -> None:
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
            if hasattr(os, "geteuid") and os.geteuid() == 0:   # under sudo: hand back to dir owner
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


def write_camera_conf(path, camera, preprocess):
    _atomic_write(Path(path), format_camera_conf(camera, preprocess))


def seed_from_conf(path) -> tuple[dict, dict]:
    p = Path(path)
    if not p.is_file():
        return {}, {}
    try:
        raw = parse_ini_file(p)
    except Exception:
        return {}, {}
    return dict(raw.get("camera", {})), dict(raw.get("preprocess", {}))


def _as_int(raw, key, default):
    try:
        return int(float(str(raw[key]).strip()))
    except (KeyError, ValueError):
        return default


def _as_bool(raw, key, default):
    v = str(raw.get(key, "")).strip().lower()
    if v in ("true", "yes", "on", "1"):
        return True
    if v in ("false", "no", "off", "0"):
        return False
    return default


def logical_control_ranges(cap) -> dict[str, tuple[int, int, int | None]]:
    """``{name: (lo, hi, device_current_or_None)}`` for the seven sliders. UVC devices
    report real ranges + current values; everything else gets the generic bounds."""
    uvc = getattr(cap, "uvc_controls", None)
    uvc = uvc if isinstance(uvc, dict) else None
    out: dict[str, tuple[int, int, int | None]] = {}
    for name in _SLIDER_ORDER:
        lo, hi = _CONTROL_RANGES[name]
        cur = None
        if uvc:
            for disp in _UVC_NAMES.get(name, ()):
                c = uvc.get(disp)
                if c is not None:
                    lo = int(getattr(c, "min_val", lo))
                    hi = int(getattr(c, "max_val", hi))
                    cur = int(getattr(c, "value", (lo + hi) // 2))
                    break
        if hi <= lo:
            hi = lo + 1
        out[name] = (lo, hi, cur)
    return out


# ────────────────────────────────────────────────────────────────────────────
# Mode discovery helpers
# ────────────────────────────────────────────────────────────────────────────

def _mode_whf(m) -> tuple[int, int, int]:
    """(w, h, fps) from a pyuvc mode (namedtuple or plain tuple)."""
    try:
        return int(m.width), int(m.height), int(round(float(m.fps)))
    except AttributeError:
        return int(m[0]), int(m[1]), (int(round(float(m[2]))) if len(m) > 2 else 0)


def _sort_modes(modes: dict[str, list[int]]) -> list[dict]:
    def area(res):
        w, h = (int(x) for x in res.split("x"))
        return (w * h, w)
    return [{"res": r, "fps": sorted(set(f for f in modes[r] if f > 0))} for r in sorted(modes, key=area)]


_AVF_MODE_RE = re.compile(r"(\d+)x(\d+)@\[([\d.]+)\s+([\d.]+)\]fps")


def ffmpeg_avfoundation_modes(device: Any) -> dict[str, list[int]]:
    """Native capture modes of a macOS camera, read from ffmpeg's avfoundation input.

    Asking ffmpeg for an impossible 1x1 mode makes it print the device's
    ``Supported modes:`` list and exit. Returns ``{}`` off macOS, without ffmpeg,
    or when the output can't be parsed."""
    exe = shutil.which("ffmpeg")
    if not exe or platform.system() != "Darwin":
        return {}
    try:
        r = subprocess.run(
            [exe, "-hide_banner", "-f", "avfoundation", "-framerate", "1",
             "-video_size", "1x1", "-i", f"{device}:none", "-frames:v", "1", "-f", "null", "-"],
            capture_output=True, text=True, timeout=10)
    except Exception:
        return {}
    return parse_avfoundation_modes(r.stderr or "")


def parse_avfoundation_modes(text: str) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for m in _AVF_MODE_RE.finditer(text):
        w, h, lo, hi = int(m[1]), int(m[2]), float(m[3]), float(m[4])
        fps = out.setdefault(f"{w}x{h}", [])
        for f in _COMMON_FPS + [int(round(hi))]:
            if lo - 0.5 <= f <= hi + 0.5 and f not in fps:          # 59.94 counts as 60
                fps.append(f)
    return out


# ────────────────────────────────────────────────────────────────────────────
# Lens calibration helpers (pure)
# ────────────────────────────────────────────────────────────────────────────

def parse_chessboard(spec):
    try:
        cols, rows = (int(x) for x in str(spec).lower().split("x"))
    except (ValueError, TypeError) as e:
        raise ValueError(f"bad --chessboard {spec!r}; expected COLSxROWS like 9x6") from e
    if cols < 2 or rows < 2:
        raise ValueError(f"--chessboard {spec!r}: need >=2 inner corners per side")
    return cols, rows


def chessboard_object_points(cols, rows, square_size=1.0):
    import numpy as np
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    return objp * float(square_size)


def find_chessboard(cv2, gray, size):
    """Locate a chessboard with ``size`` = (cols, rows) INNER corners.

    Tries the sector-based detector first (more tolerant of blur, uneven light and
    perspective), then the classic one + sub-pixel refinement. ``(found, corners)``."""
    if hasattr(cv2, "findChessboardCornersSB"):
        try:
            found, corners = cv2.findChessboardCornersSB(gray, size, cv2.CALIB_CB_NORMALIZE_IMAGE)
            if found and corners is not None:
                return True, corners.reshape(-1, 1, 2).astype("float32")
        except cv2.error:
            pass
    found, corners = cv2.findChessboardCorners(
        gray, size, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
    if found:
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), crit)
        return True, corners
    return False, None


def write_lens_yaml(path, camera_matrix, dist_coeffs, *, image_size=None, rms=None):
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


# ────────────────────────────────────────────────────────────────────────────
# The inline HTML/JS UI
# ────────────────────────────────────────────────────────────────────────────

_HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Camera Calibration</title>
<style>
:root {
    --bg: #14161c; --surface: #1c1f27; --raised: #252a35; --border: #313848;
    --text: #e3e6ec; --dim: #8b93a4; --accent: #6cc4ff; --ok: #5cd69a;
    --warn: #f0c050; --bad: #ff7b72; --input: #11131a;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: var(--bg); color: var(--text); font: 13px/1.5 -apple-system, system-ui, sans-serif; }
.layout { display: flex; height: 100vh; }
.video-pane { flex: 1; display: flex; align-items: center; justify-content: center; background: #000; min-width: 0; }
.video-pane img { max-width: 100%; max-height: 100%; object-fit: contain; }
.controls { width: 420px; min-width: 360px; overflow-y: auto; background: var(--surface);
            border-left: 1px solid var(--border); padding: 12px 14px 20px; }
.conn { font-size: 11px; color: var(--dim); margin-bottom: 8px; }
.conn.bad { color: var(--bad); }

.tabs { display: flex; gap: 2px; margin-bottom: 12px; }
.tab { flex: 1; padding: 8px 4px; background: var(--raised); border: none; color: var(--dim);
       cursor: pointer; border-radius: 6px 6px 0 0; font-size: 12px; font-weight: 600; margin: 0; }
.tab.active { background: var(--bg); color: var(--accent); }
.tab-panel { display: none; }
.tab-panel.active { display: block; }

h3 { font-size: 11px; text-transform: uppercase; letter-spacing: 1px; color: var(--accent); margin: 14px 0 6px; }
h3:first-child { margin-top: 0; }
label { display: flex; align-items: center; gap: 8px; margin: 5px 0; font-size: 12px; }
.lbl { min-width: 104px; color: var(--dim); flex-shrink: 0; }
input[type=range] { flex: 1; min-width: 0; accent-color: var(--accent); }
input[type=number], input[type=text] { width: 72px; background: var(--input); border: 1px solid var(--border);
    color: var(--text); padding: 3px 6px; border-radius: 3px; font-size: 12px; }
input[type=checkbox] { accent-color: var(--ok); width: 16px; height: 16px; }
select { background: var(--input); border: 1px solid var(--border); color: var(--text);
         padding: 4px 6px; border-radius: 3px; font-size: 12px; flex: 1; min-width: 0; }
.val { width: 58px; text-align: right; font-variant-numeric: tabular-nums; }
.ctl.unsupported .lbl { color: var(--bad); }
.ctl.unsupported input[type=range] { opacity: .45; }
button { padding: 6px 12px; background: var(--raised); border: 1px solid var(--border); color: var(--text);
         border-radius: 4px; cursor: pointer; font-size: 12px; margin: 3px 2px; }
button:hover { border-color: var(--accent); }
button:active { background: var(--accent); color: var(--bg); }
.status { color: var(--ok); font-size: 12px; min-height: 18px; margin: 6px 0; }
.status.warn { color: var(--warn); }
.note { font-size: 11px; color: var(--dim); margin: 4px 0 6px; }
.note.warn { color: var(--warn); }
.row { display: flex; gap: 6px; flex-wrap: wrap; align-items: center; }
.sep { border-top: 1px solid var(--border); margin: 12px 0; }
.keys { font-size: 11px; color: var(--dim); margin-top: 16px; padding-top: 10px; border-top: 1px solid var(--border); line-height: 1.9; }
kbd { background: var(--raised); padding: 1px 5px; border-radius: 3px; font-size: 11px; border: 1px solid var(--border); }
.pill { display: inline-block; padding: 0 6px; border-radius: 8px; background: var(--raised); color: var(--dim);
        font-size: 10px; letter-spacing: 0; text-transform: none; }
</style>
</head>
<body>
<div class="layout">
  <div class="video-pane"><img id="stream" src="/stream" alt="camera stream"></div>
  <div class="controls">
    <div id="conn" class="conn">connecting…</div>
    <div class="tabs">
      <button class="tab active" onclick="switchTab(0)">Hardware <kbd>1</kbd></button>
      <button class="tab" onclick="switchTab(1)">Processing <kbd>2</kbd></button>
      <button class="tab" onclick="switchTab(2)">Lens <kbd>3</kbd></button>
    </div>

    <!-- ─── Tab 0: Hardware ─── -->
    <div id="tab0" class="tab-panel active">
      <div class="row"><button onclick="api('save')">Save conf <kbd>S</kbd></button>
        <span id="conf_path" class="note"></span></div>
      <div id="save_status" class="status"></div>

      <h3>Capture mode</h3>
      <label><span class="lbl">Resolution</span>
        <select id="res_picker" onchange="applyMode(true); this.blur()"><option>loading…</option></select></label>
      <label><span class="lbl">FPS</span>
        <select id="fps_picker" onchange="applyMode(false); this.blur()"><option>loading…</option></select></label>
      <div id="mode_info" class="note"></div>
      <div id="mode_status" class="status"></div>

      <div class="sep"></div>
      <h3>Camera controls <span id="backend_pill" class="pill"></span></h3>
      <div id="ctrl_note" class="note"></div>
      <label><span class="lbl">auto_exposure</span>
        <select id="ctrl_auto_exposure" onchange="api('control',{name:'auto_exposure',value:this.value==='Auto'}); this.blur()">
          <option>Manual</option><option>Auto</option></select></label>
      <div id="hw_sliders"><p class="note">loading camera controls…</p></div>
      <div id="ctrl_status" class="status"></div>
    </div>

    <!-- ─── Tab 1: Processing ─── -->
    <div id="tab1" class="tab-panel">
      <h3>View</h3>
      <label><span class="lbl">Stream shows</span>
        <select id="view_mode" onchange="api('proc',{key:'view',value:this.value}); this.blur()">
          <option value="raw">Raw camera + crop/mask guides</option>
          <option value="processed">Processed (flip, crop, lens, flatfield)</option>
          <option value="gray">Tracker input (gray, masked)</option>
        </select></label>
      <p class="note">With crop, mask, lens and flatfield all off, Raw and Processed look the same. The tag in the
        stream's top-left corner always names the active view.</p>

      <div class="sep"></div>
      <h3>[preprocess]</h3>
      <label><input type="checkbox" id="pp_master" onchange="api('proc',{key:'preprocess',value:this.checked})">
        <span>preprocess (master switch; off = plain BGR to gray)</span></label>
      <label><span class="lbl">flip</span>
        <select id="flip_mode" onchange="api('proc',{key:'flip',value:this.value}); this.blur()">
          <option value="none">none</option>
          <option value="x">x: mirror left/right</option>
          <option value="y">y: mirror top/bottom</option>
          <option value="xy">xy: both (= 180° rotation)</option>
        </select></label>
      <label><input type="checkbox" id="lens_on" onchange="api('proc',{key:'lens_enabled',value:this.checked})">
        <span>lens_enabled</span> <span id="lens_path" class="note"></span></label>

      <div class="sep"></div>
      <h3>Crop ROI (raw px, in the flipped frame)</h3>
      <label><input type="checkbox" id="crop_enabled" onchange="api('proc',{key:'crop_enabled',value:this.checked})">
        <span>crop_enabled</span></label>
      <label><span class="lbl">crop_top_x</span><input type="number" id="crop_x" min="0" onchange="api('proc',{key:'crop_x',value:+this.value})"></label>
      <label><span class="lbl">crop_top_y</span><input type="number" id="crop_y" min="0" onchange="api('proc',{key:'crop_y',value:+this.value})"></label>
      <label><span class="lbl">crop_width</span><input type="number" id="crop_w" min="0" onchange="api('proc',{key:'crop_w',value:+this.value})"></label>
      <label><span class="lbl">crop_height</span><input type="number" id="crop_h" min="0" onchange="api('proc',{key:'crop_h',value:+this.value})"></label>
      <p class="note">0 width or height = to the frame edge. The rectangle is drawn in the Raw view.</p>

      <div class="sep"></div>
      <h3>Circular mask</h3>
      <label><input type="checkbox" id="mask_enabled" onchange="api('proc',{key:'mask_enabled',value:this.checked})">
        <span>mask_enabled</span></label>
      <div id="mask_sliders"></div>
      <p class="note">The mask zeroes the GRAY frame only; see its effect in the "Tracker input" view.</p>

      <div class="sep"></div>
      <h3>Flatfield</h3>
      <label><input type="checkbox" id="ff_on" onchange="api('proc',{key:'flatfield_enabled',value:this.checked})">
        <span>flatfield_enabled</span> <span id="ff_path" class="note"></span></label>
      <button onclick="api('flatfield')">Capture flatfield (""" + str(_FLATFIELD_FRAMES) + r"""-frame average)</button>
      <div id="ff_status" class="status">idle</div>
    </div>

    <!-- ─── Tab 2: Lens Calibration ─── -->
    <div id="tab2" class="tab-panel">
      <h3>Chessboard</h3>
      <label><span class="lbl">Inner corners</span>
        <input type="number" id="cb_cols" min="2" max="30"> &times;
        <input type="number" id="cb_rows" min="2" max="30">
        <button onclick="api('board',{cols:+cbv('cb_cols'),rows:+cbv('cb_rows')})">Set</button></label>
      <p class="note">Count the corners where four squares meet, not the squares: a board of 10 &times; 7 squares
        has 9 &times; 6 inner corners. Keep a white margin around the board.</p>
      <div class="row"><button onclick="api('board_auto')">Auto-detect board size</button></div>
      <div id="auto_status" class="status"></div>
      <label><input type="checkbox" id="detect_cb" onchange="api('detect',{active:this.checked})">
        <span>Detect chessboard (live)</span></label>
      <div id="calib_live" class="status warn">views 0 · board —</div>
      <div class="row">
        <button onclick="api('grab')">Grab view <kbd>Space</kbd></button>
        <button onclick="api('delete')">Delete last <kbd>D</kbd></button>
        <button onclick="api('solve')">Solve <kbd>C</kbd></button>
      </div>
      <div id="calib_status" class="status"></div>
    </div>

    <div class="keys">
      <kbd>S</kbd> save &nbsp; <kbd>Space</kbd> grab &nbsp; <kbd>C</kbd> solve &nbsp; <kbd>D</kbd> delete &nbsp;
      <kbd>1</kbd><kbd>2</kbd><kbd>3</kbd> tabs &nbsp; <kbd>Q</kbd> quit
    </div>
  </div>
</div>

<script>
let S = null;              // last state
let hwSig = '', modeSig = '', maskSig = '';

const $ = id => document.getElementById(id);
const focused = el => el && el === document.activeElement;
function setVal(id, v) { const el = $(id); if (el && !focused(el) && v !== undefined && v !== null) el.value = v; }
function setChk(id, v) { const el = $(id); if (el && !focused(el)) el.checked = !!v; }
function setTxt(id, v) { const el = $(id); if (el && v !== undefined && v !== null) el.textContent = v; }
function cbv(id) { return $(id).value; }

function switchTab(n) {
  document.querySelectorAll('.tab').forEach((t, i) => t.classList.toggle('active', i === n));
  document.querySelectorAll('.tab-panel').forEach((p, i) => p.classList.toggle('active', i === n));
}

async function api(action, data) {
  try {
    const r = await fetch('/api/' + action, {
      method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(data || {})});
    const j = await r.json();
    if (j.status_msg !== undefined) {
      const el = $(j.status_el || 'calib_status');
      if (el) { el.textContent = j.status_msg; el.classList.toggle('warn', j.ok === false); }
    }
    refreshState();
    return j;
  } catch (e) { console.error(action, e); return {}; }
}

function buildHwSliders(s) {
  const c = $('hw_sliders');
  c.innerHTML = '';
  for (const [name, lo, hi] of s.control_ranges) {
    const sup = s.ctrl_support[name];
    const val = s.controls[name];
    const lbl = document.createElement('label');
    lbl.className = 'ctl' + (sup === false ? ' unsupported' : '');
    if (sup === false) lbl.title = 'the camera driver rejected this control';
    lbl.innerHTML =
      '<span class="lbl">' + name + (sup === false ? ' ✕' : '') + '</span>' +
      '<input type="range" id="sl_' + name + '" min="' + lo + '" max="' + hi + '" value="' + val + '">' +
      '<input type="number" class="val" id="nv_' + name + '" min="' + lo + '" max="' + hi + '" value="' + val + '">';
    const sl = lbl.querySelector('input[type=range]'), nv = lbl.querySelector('input[type=number]');
    sl.addEventListener('input', () => { nv.value = sl.value; });
    sl.addEventListener('change', () => api('control', {name, value: +sl.value}));
    nv.addEventListener('change', () => { sl.value = nv.value; api('control', {name, value: +nv.value}); });
    c.appendChild(lbl);
  }
}

function buildModes(s) {
  const rp = $('res_picker');
  rp.innerHTML = '';
  const list = s.modes.slice();
  if (!list.some(m => m.res === s.current_res)) list.unshift({res: s.current_res, fps: []});
  for (const m of list) {
    const o = document.createElement('option');
    o.value = m.res; o.text = m.res + (m.res === s.current_res ? '  (current)' : '');
    rp.appendChild(o);
  }
  rp.value = s.current_res;
  fillFps(s, s.current_res);
}

function fpsFor(s, res) {
  const m = s.modes.find(m => m.res === res);
  return (m && m.fps.length) ? m.fps : s.fallback_fps;
}

function fillFps(s, res) {
  const fp = $('fps_picker');
  const list = fpsFor(s, res).slice();
  if (res === s.current_res && s.current_fps && !list.includes(s.current_fps)) list.push(s.current_fps);
  list.sort((a, b) => a - b);
  fp.innerHTML = '';
  for (const f of list) { const o = document.createElement('option'); o.value = f; o.text = f + ' fps'; fp.appendChild(o); }
  let want = (res === s.current_res && s.current_fps) ? s.current_fps : null;
  if (want === null) { const le30 = list.filter(f => f <= 30); want = le30.length ? Math.max(...le30) : list[0]; }
  fp.value = String(want);
}

function applyMode(resChanged) {
  if (!S) return;
  const res = $('res_picker').value;
  if (resChanged) fillFps(S, res);
  const [w, h] = res.split('x').map(Number);
  setTxt('mode_status', 'switching…');
  api('mode', {width: w, height: h, fps: +$('fps_picker').value}).then(() => { modeSig = ''; });
}

function buildMaskSliders(s) {
  const c = $('mask_sliders');
  c.innerHTML = '';
  const items = [['mask_cx', s.frame_w], ['mask_cy', s.frame_h], ['mask_radius', Math.max(s.frame_w, s.frame_h)]];
  for (const [key, hi] of items) {
    const val = s.proc[key] || 0;
    const lbl = document.createElement('label');
    lbl.innerHTML = '<span class="lbl">' + key + '</span>' +
      '<input type="range" id="sl_' + key + '" min="0" max="' + hi + '" value="' + val + '">' +
      '<input type="number" class="val" id="nv_' + key + '" min="0" max="' + hi + '" value="' + val + '">';
    const sl = lbl.querySelector('input[type=range]'), nv = lbl.querySelector('input[type=number]');
    sl.addEventListener('input', () => { nv.value = sl.value; api('proc', {key, value: +sl.value}); });
    nv.addEventListener('change', () => { sl.value = nv.value; api('proc', {key, value: +nv.value}); });
    c.appendChild(lbl);
  }
}

let polling = false;
async function refreshState() {
  if (polling) return;
  polling = true;
  try {
    const r = await fetch('/api/state', {cache: 'no-store'});
    if (!r.ok) throw new Error('HTTP ' + r.status);
    const s = await r.json();
    S = s;
    const conn = $('conn');
    conn.className = 'conn';
    conn.textContent = 'device ' + s.device + ' · ' + s.frame_w + '×' + s.frame_h + ' · ' +
                       s.measured_fps.toFixed(1) + ' fps measured';

    // hardware
    setTxt('conf_path', s.conf_path);
    const hs = JSON.stringify([s.control_ranges, s.ctrl_support]);
    if (hs !== hwSig) { buildHwSliders(s); hwSig = hs; }
    for (const [name] of s.control_ranges) { setVal('sl_' + name, s.controls[name]); setVal('nv_' + name, s.controls[name]); }
    setVal('ctrl_auto_exposure', s.auto_exposure ? 'Auto' : 'Manual');
    setTxt('backend_pill', 'backend: ' + s.backend);
    const cn = $('ctrl_note'); cn.textContent = s.ctrl_note; cn.className = 'note' + (s.ctrl_note_warn ? ' warn' : '');
    const ms = JSON.stringify([s.modes, s.current_res, s.current_fps]);
    if (ms !== modeSig && !focused($('res_picker')) && !focused($('fps_picker'))) { buildModes(s); modeSig = ms; }
    setTxt('mode_info', s.modes_note);

    // processing
    setVal('view_mode', s.view);
    setChk('pp_master', s.pp_master);
    setVal('flip_mode', s.proc.flip);
    setChk('lens_on', s.lens_on);
    setTxt('lens_path', s.lens_path ? '(' + s.lens_path.split('/').pop() + ')' : '(no lens.yaml yet)');
    setChk('crop_enabled', s.proc.crop_enabled);
    setVal('crop_x', s.proc.crop_x); setVal('crop_y', s.proc.crop_y);
    setVal('crop_w', s.proc.crop_w); setVal('crop_h', s.proc.crop_h);
    setChk('mask_enabled', s.proc.mask_enabled);
    const mk = s.frame_w + 'x' + s.frame_h;
    if (mk !== maskSig) { buildMaskSliders(s); maskSig = mk; }
    for (const k of ['mask_cx', 'mask_cy', 'mask_radius']) { setVal('sl_' + k, s.proc[k]); setVal('nv_' + k, s.proc[k]); }
    setChk('ff_on', s.flatfield_on);
    setTxt('ff_path', s.flatfield_path ? '(' + s.flatfield_path.split('/').pop() + ')' : '(none captured)');
    setTxt('ff_status', s.ff_status);

    // lens
    setVal('cb_cols', s.cb_cols); setVal('cb_rows', s.cb_rows);
    setChk('detect_cb', s.calib_active);
    const cl = $('calib_live');
    cl.textContent = 'views ' + s.calib_views + '/' + s.min_views + ' · ' +
      (s.calib_active ? (s.calib_found ? 'board FOUND (' + s.cb_cols + '×' + s.cb_rows + ')'
                                       : 'no ' + s.cb_cols + '×' + s.cb_rows + ' board in view')
                      : 'detection off') + (s.rms !== null ? ' · last RMS ' + s.rms.toFixed(3) + ' px' : '');
    cl.className = 'status' + (s.calib_found ? '' : ' warn');
    if (s.auto_status) setTxt('auto_status', s.auto_status);
  } catch (e) {
    const conn = $('conn'); conn.className = 'conn bad'; conn.textContent = 'server not responding (' + e.message + ')';
  } finally { polling = false; }
}

document.addEventListener('keydown', e => {
  if (['INPUT', 'SELECT', 'TEXTAREA'].includes(document.activeElement.tagName)) return;
  switch (e.key.toLowerCase()) {
    case 's': api('save'); break;
    case ' ': e.preventDefault(); api('grab'); break;
    case 'c': api('solve'); break;
    case 'd': api('delete'); break;
    case 'q': api('quit'); break;
    case '1': switchTab(0); break;
    case '2': switchTab(1); break;
    case '3': switchTab(2); break;
  }
});

refreshState();
setInterval(refreshState, 500);
</script>
</body>
</html>
"""


# ────────────────────────────────────────────────────────────────────────────
# The calibration app
# ────────────────────────────────────────────────────────────────────────────

class _CalibApp:
    """Opens the camera, runs a capture loop + a chessboard worker, and answers
    the browser's JSON API. Heavy modules are imported lazily so importing this
    file stays camera-free (for tests)."""

    def __init__(self, args: argparse.Namespace) -> None:
        import cv2
        import numpy as np

        from lib import camera_controls as _cc
        from lib.logger import Logger
        from lib.preprocess import Preprocess, PreprocessConfig

        self.cv2, self.np = cv2, np
        self._Preprocess, self._PreprocessConfig = Preprocess, PreprocessConfig
        self.args = args
        self.log = Logger(level="info", stderr_level="warning")

        seed_cam, seed_pp = seed_from_conf(args.out)
        self._device_str = args.device if args.device is not None else str(seed_cam.get("device", "0"))
        self.device = int(self._device_str) if str(self._device_str).isdigit() else self._device_str
        self._backend_req = args.backend if args.backend is not None else str(seed_cam.get("backend", "auto"))
        req_w = args.width if args.width is not None else _as_int(seed_cam, "width", 1280)
        req_h = args.height if args.height is not None else _as_int(seed_cam, "height", 720)
        req_fps = args.fps if args.fps is not None else _as_int(seed_cam, "fps", 30)

        self.out = args.out
        self.out_dir = Path(args.out).resolve().parent
        self.lens_out = (str(Path(args.lens_out).resolve()) if args.lens_out
                         else str(self.out_dir / "lens.yaml"))

        # every cap.read / cap.set / cap.get goes through this lock: the capture loop,
        # the HTTP threads and the control shim all touch the same device handle
        self._cap_lock = threading.RLock()
        self._frame_lock = threading.Lock()

        self.cap = _cc.open_capture(self.device, self._backend_req, self.log)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open device {self.device!r}")
        self.shim = _cc.CameraControls(self.cap, device=self.device, logger=self.log)
        self.req_w, self.req_h, self.req_fps = int(req_w), int(req_h), int(req_fps)
        self._set_geometry(self.req_w, self.req_h, self.req_fps)

        # ── picture controls: conf > device-reported (UVC) > defaults, clamped ──
        self._control_ranges = logical_control_ranges(self.cap)
        self.controls: dict[str, int] = {}
        for name, (lo, hi, dev_cur) in self._control_ranges.items():
            if name in seed_cam:
                v = _as_int(seed_cam, name, _DEFAULT_CONTROLS[name])
            elif dev_cur is not None:
                v = dev_cur
            else:
                v = _DEFAULT_CONTROLS[name]
            self.controls[name] = max(lo, min(hi, int(v)))
        self.auto_exposure = _as_bool(seed_cam, "auto_exposure", False)

        # ── preprocess state ──
        fl = str(seed_pp.get("flip") or args.flip or "none").strip().lower()
        self.proc = {
            "flip": fl if fl in _FLIP_CHOICES else "none",
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
        self.pp_master = _as_bool(seed_pp, "preprocess", True)
        self.lens_path = args.calibration or seed_pp.get("lens_path", "")
        self.lens_on = _as_bool(seed_pp, "lens_enabled", False) and bool(self.lens_path)
        self.flatfield_path = seed_pp.get("flatfield_path", "") or args.flatfield
        self.flatfield_on = (_as_bool(seed_pp, "flatfield_enabled", bool(self.flatfield_path))
                             and bool(self.flatfield_path))
        self.view = "raw"

        self._ff = {"cap": False, "buf": None, "n": 0, "status": "idle"}
        self._pp_cache = {"sig": None, "pre": None}

        self.cb_cols, self.cb_rows = parse_chessboard(args.chessboard)
        self.calib = {"active": False, "objp": [], "imgp": [], "size": None,
                      "found": False, "corners": None, "rms": None, "K": None, "D": None}
        self._auto = {"running": False, "status": ""}

        # ── discovery ──
        self._modes, self._modes_note = self._discover_modes()
        self._ctrl_support = self._probe_control_support()

        self._safe_apply({**self.controls, "auto_exposure": self.auto_exposure})

        # ── runtime ──
        self.width, self.height = self.req_w, self.req_h      # replaced by the real frame size
        self._last_bgr = None
        self._jpeg_buf: bytes = b""
        self._jpeg_seq = 0
        self._measured_fps = 0.0
        self._running = True

    # ── device access (always under the lock) ──────────────────────────────

    def _set_geometry(self, w, h, fps):
        cv2 = self.cv2
        with self._cap_lock:
            if w:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            if h:
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            if fps:
                self.cap.set(cv2.CAP_PROP_FPS, fps)

    def _cap_fps(self) -> int:
        with self._cap_lock:
            try:
                return int(round(float(self.cap.get(self.cv2.CAP_PROP_FPS) or 0)))
            except Exception:
                return 0

    def _safe_apply(self, controls):
        with self._cap_lock:
            try:
                self.shim.apply(controls)
            except Exception as e:
                self.log.warning(f"[calibrate] apply failed: {e}")

    def _discover_modes(self) -> tuple[list[dict], str]:
        """The capture modes to offer, best source first: the UVC mode list, then
        ffmpeg's avfoundation listing (macOS), then a set/read-back probe."""
        cv2 = self.cv2
        uvc_modes = list(getattr(self.cap, "available_modes", []) or [])
        if uvc_modes:
            modes: dict[str, list[int]] = {}
            for m in uvc_modes:
                try:
                    w, h, f = _mode_whf(m)
                except Exception:
                    continue
                if w and h:
                    modes.setdefault(f"{w}x{h}", []).append(f)
            if modes:
                return _sort_modes(modes), f"{len(modes)} native modes reported by the UVC device"

        if platform.system() == "Darwin":
            modes = ffmpeg_avfoundation_modes(self.device)
            if modes:
                return _sort_modes(modes), (f"{len(modes)} native modes listed by ffmpeg/avfoundation; "
                                            f"OpenCV may still pick the nearest one")

        # probe: set each size, keep what the driver reports back
        found: dict[str, list[int]] = {}
        with self._cap_lock:
            orig_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            orig_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            for tw, th in _COMMON_RESOLUTIONS:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, tw)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, th)
                gw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                gh = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                if gw > 0 and gh > 0:
                    found.setdefault(f"{gw}x{gh}", [])
            if orig_w and orig_h:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, orig_w)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, orig_h)
        note = "sizes the driver accepted when probed"
        if platform.system() == "Darwin":
            note += ("; on macOS OpenCV can scale to any size, so not all may be native "
                     "(install ffmpeg to list the real modes)")
        return _sort_modes(found), note

    def _probe_control_support(self) -> dict[str, bool | None]:
        """Which picture controls the device accepts. UVC: those it lists. cv2 on
        macOS: write each control's own current value back and keep the driver's
        verdict (AVFoundation rejects nearly all). Elsewhere: unknown (None)."""
        uvc = getattr(self.cap, "uvc_controls", None)
        if isinstance(uvc, dict):
            return {n: any(d in uvc for d in _UVC_NAMES.get(n, ())) for n in _SLIDER_ORDER}
        if self.shim.backend == "cv2" and platform.system() == "Darwin":
            from lib.camera_controls import _CV2_PROPS
            out: dict[str, bool | None] = {}
            with self._cap_lock:
                for n in _SLIDER_ORDER:
                    prop = _CV2_PROPS.get(n)
                    try:
                        out[n] = bool(self.cap.set(prop, float(self.cap.get(prop))))
                    except Exception:
                        out[n] = False
            return out
        return {n: None for n in _SLIDER_ORDER}

    # ── frame processing ───────────────────────────────────────────────────

    def _crop_pp(self):
        return {"crop_top_x": int(self.proc["crop_x"]), "crop_top_y": int(self.proc["crop_y"]),
                "crop_width": int(self.proc["crop_w"]), "crop_height": int(self.proc["crop_h"])}

    def _crop_rect_corners(self, fw, fh):
        x, y = int(self.proc["crop_x"]), int(self.proc["crop_y"])
        w, h = int(self.proc["crop_w"]), int(self.proc["crop_h"])
        return x, y, (x + w if w > 0 else fw), (y + h if h > 0 else fh)

    def _flip_frame(self, bgr):
        code = _FLIP_CV.get(self.proc["flip"]) if self.pp_master else None
        return self.cv2.flip(bgr, code) if code is not None else bgr.copy()

    def _detect_frame(self, bgr):
        """The frame the lens is calibrated on: flip, then crop (as the pipeline does)."""
        f = self._flip_frame(bgr)
        if self.pp_master and self.proc["crop_enabled"]:
            x0, y0, x1, y1 = self._crop_rect_corners(f.shape[1], f.shape[0])
            x0, y0 = max(0, int(x0)), max(0, int(y0))
            x1, y1 = min(f.shape[1], int(x1)), min(f.shape[0], int(y1))
            if x1 > x0 and y1 > y0:
                f = f[y0:y1, x0:x1]
        return f

    def _corners_for_raw_view(self, corners):
        if not (self.pp_master and self.proc["crop_enabled"]):
            return corners
        out = corners.copy()
        out[:, 0, 0] += int(self.proc["crop_x"])
        out[:, 0, 1] += int(self.proc["crop_y"])
        return out

    def _resolve_crop(self, fw, fh):
        if int(self.proc["crop_w"]) <= 0:
            self.proc["crop_w"] = max(1, fw - int(self.proc["crop_x"]))
        if int(self.proc["crop_h"]) <= 0:
            self.proc["crop_h"] = max(1, fh - int(self.proc["crop_y"]))

    def _pp_map(self, flatfield_on):
        m = {"preprocess": self.pp_master, "lens_enabled": self.lens_on, "lens_path": self.lens_path,
             "flip": self.proc["flip"],
             "crop_enabled": self.proc["crop_enabled"], "mask_enabled": self.proc["mask_enabled"],
             **self._crop_pp(),
             "flatfield_enabled": flatfield_on and bool(self.flatfield_path),
             "flatfield_path": self.flatfield_path}
        for k in ("mask_cx", "mask_cy", "mask_radius"):
            m[k] = self.proc[k]
        return m

    def _process(self, bgr, flatfield_on):
        m = self._pp_map(flatfield_on)
        sig = tuple(sorted(m.items()))
        if sig != self._pp_cache["sig"]:
            cfg = self._PreprocessConfig.from_mapping(m)
            cfg.load_artifacts(self.log)
            self._pp_cache["sig"], self._pp_cache["pre"] = sig, self._Preprocess(cfg, logger=self.log)
        return self._pp_cache["pre"].process(bgr)          # (bgr, gray)

    @staticmethod
    def _tag(cv2, img, text):
        """A legible label in the stream's top-left corner."""
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (0, 0), (tw + 16, th + base + 12), (0, 0, 0), -1)
        cv2.putText(img, text, (8, th + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # ── capture loop (thread) ──────────────────────────────────────────────

    def _camera_loop(self):
        cv2 = self.cv2
        last_t = None
        while self._running:
            with self._cap_lock:
                ok, bgr = self.cap.read()
            if not ok or bgr is None:
                time.sleep(0.01)
                continue
            now = time.perf_counter()
            if last_t is not None and now > last_t:
                inst = 1.0 / (now - last_t)
                self._measured_fps = inst if self._measured_fps == 0 else 0.9 * self._measured_fps + 0.1 * inst
            last_t = now

            h, w = bgr.shape[:2]
            if (w, h) != (self.width, self.height):
                self.width, self.height = w, h
                self._pp_cache["sig"] = None
            with self._frame_lock:
                self._last_bgr = bgr
            self._resolve_crop(w, h)

            try:
                disp = self._render(bgr, w, h)
            except Exception as e:                       # never let a bad setting kill the stream
                disp = bgr.copy()
                self._tag(cv2, disp, f"render error: {type(e).__name__}: {e}"[:90])
            ok, jpeg = cv2.imencode(".jpg", disp, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                self._jpeg_buf = jpeg.tobytes()
                self._jpeg_seq += 1

    def _render(self, bgr, w, h):
        cv2, np = self.cv2, self.np
        capturing = self._ff["cap"]
        proc, gray = self._process(bgr, flatfield_on=self.flatfield_on and not capturing)
        if capturing:
            self._accumulate_flatfield(proc)

        if self.view == "raw":
            disp = self._flip_frame(bgr)
            x0, y0, x1, y1 = self._crop_rect_corners(w, h)
            col = (0, 255, 255) if self.proc["crop_enabled"] else (0, 140, 140)
            cv2.rectangle(disp, (max(0, x0), max(0, y0)), (min(x1, w) - 1, min(y1, h) - 1), col, 2)
            if self.proc["mask_radius"] > 0:
                mcol = (0, 255, 0) if self.proc["mask_enabled"] else (0, 120, 0)
                cv2.circle(disp, (self.proc["mask_cx"], self.proc["mask_cy"]), self.proc["mask_radius"], mcol, 2)
            tag = "RAW + guides"
        else:
            src = proc if self.view == "processed" else cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            disp = np.zeros((max(h, src.shape[0]), max(w, src.shape[1]), 3), np.uint8)
            disp[:src.shape[0], :src.shape[1]] = src
            if self.view == "processed" and self.proc["mask_enabled"] and self.proc["mask_radius"] > 0:
                crop = self.pp_master and self.proc["crop_enabled"]
                ox = int(self.proc["crop_x"]) if crop else 0
                oy = int(self.proc["crop_y"]) if crop else 0
                cv2.circle(disp, (self.proc["mask_cx"] - ox, self.proc["mask_cy"] - oy),
                           self.proc["mask_radius"], (0, 255, 0), 1)
            tag = (f"PROCESSED {src.shape[1]}x{src.shape[0]}" if self.view == "processed"
                   else f"TRACKER INPUT (gray) {src.shape[1]}x{src.shape[0]}")

        cal = self.calib
        corners = cal["corners"]
        if cal["active"] and corners is not None:
            oc = corners if self.view != "raw" else self._corners_for_raw_view(corners)
            cv2.drawChessboardCorners(disp, (self.cb_cols, self.cb_rows), oc, bool(cal["found"]))
        if self._ff["cap"]:
            tag += f"  |  flatfield {self._ff['n']}/{_FLATFIELD_FRAMES}"
        self._tag(cv2, disp, tag)
        return disp

    def _accumulate_flatfield(self, proc):
        cv2, np = self.cv2, self.np
        geom = proc.astype(np.float32)
        if self._ff["buf"] is not None and geom.shape != self._ff["buf"].shape:
            self._ff.update(buf=None, n=0)
        self._ff["buf"] = geom if self._ff["buf"] is None else self._ff["buf"] + geom
        self._ff["n"] += 1
        self._ff["status"] = f"capturing {self._ff['n']}/{_FLATFIELD_FRAMES}"
        if self._ff["n"] >= _FLATFIELD_FRAMES:
            avg = (self._ff["buf"] / _FLATFIELD_FRAMES).astype(np.uint8)
            path = self.flatfield_path or str(self.out_dir / "flatfield.png")
            cv2.imwrite(path, avg)
            self.flatfield_path, self.flatfield_on = path, True
            self._ff.update(cap=False, buf=None)
            self._pp_cache["sig"] = None
            self._ff["status"] = f"saved {Path(path).name} (applied)"

    # ── chessboard worker (thread): detection never stalls the stream ─────

    def _detect_loop(self):
        cv2 = self.cv2
        while self._running:
            cal = self.calib
            if not cal["active"]:
                cal["found"], cal["corners"] = False, None
                time.sleep(0.1)
                continue
            with self._frame_lock:
                bgr = self._last_bgr
            if bgr is None:
                time.sleep(0.05)
                continue
            size = (self.cb_cols, self.cb_rows)
            det = self._detect_frame(bgr)
            gray = cv2.cvtColor(det, cv2.COLOR_BGR2GRAY)
            found, corners = find_chessboard(cv2, gray, size)
            if size != (self.cb_cols, self.cb_rows):      # board size changed mid-detection
                continue
            if found:
                cal["size"] = (det.shape[1], det.shape[0])
            cal["corners"], cal["found"] = (corners if found else None), found
            time.sleep(0.02)

    def _auto_board(self):
        """Try every inner-corner count from 15x11 down to 3x3 on the current frame;
        the largest pattern that is found is the board."""
        cv2 = self.cv2
        try:
            with self._frame_lock:
                bgr = self._last_bgr
            if bgr is None:
                self._auto["status"] = "no frame yet"
                return
            gray = cv2.cvtColor(self._detect_frame(bgr), cv2.COLOR_BGR2GRAY)
            scale = min(1.0, 1000.0 / max(gray.shape[:2]))
            if scale < 1.0:
                gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            sizes = sorted(((c, r) for r in range(3, 12) for c in range(r, 16)),
                           key=lambda s: -(s[0] * s[1]))
            for i, size in enumerate(sizes):
                if not self._running:
                    return
                self._auto["status"] = f"trying {size[0]}x{size[1]}  ({i + 1}/{len(sizes)})"
                ok, _ = find_chessboard(cv2, gray, size)
                if ok:
                    self._set_board(*size)
                    self.calib["active"] = True
                    self._auto["status"] = (f"found {size[0]}x{size[1]} inner corners "
                                            f"({size[0] + 1}x{size[1] + 1} squares); detection on")
                    return
            self._auto["status"] = ("no chessboard found at any size. Check focus and glare, keep the whole "
                                    "board plus a white margin in view, and check the crop.")
        finally:
            self._auto["running"] = False

    def _set_board(self, cols, rows):
        if (cols, rows) != (self.cb_cols, self.cb_rows):
            self.cb_cols, self.cb_rows = int(cols), int(rows)
            self.calib.update(objp=[], imgp=[], found=False, corners=None)

    # ── API ────────────────────────────────────────────────────────────────

    def handle_api(self, action: str, data: dict) -> dict:
        handlers = {
            "state": lambda d: self._get_state(),
            "control": self._api_control,
            "proc": self._api_proc,
            "mode": self._api_mode,
            "save": lambda d: self._api_save(),
            "grab": lambda d: self._api_grab(),
            "delete": lambda d: self._api_delete(),
            "solve": lambda d: self._api_solve(),
            "detect": self._api_detect,
            "board": self._api_board,
            "board_auto": lambda d: self._api_board_auto(),
            "flatfield": lambda d: self._api_flatfield(),
            "quit": lambda d: self._api_quit(),
        }
        fn = handlers.get(action)
        if fn is None:
            return {"ok": False, "error": f"unknown action: {action}"}
        try:
            return fn(data or {})
        except Exception as e:
            self.log.warning(f"[calibrate] api {action} failed: {e}")
            return {"ok": False, "status_msg": f"{action} failed: {type(e).__name__}: {e}"}

    def _get_state(self):
        cal = self.calib
        sup = self._ctrl_support
        rejected = [n for n, v in sup.items() if v is False]
        if self.shim.backend == "uvc":
            note, warn = "UVC backend: controls go straight to the camera.", False
        elif rejected and len(rejected) == len(sup):
            note, warn = ("The camera driver (OpenCV on macOS = AVFoundation) rejected ALL picture controls, "
                          "so these values are saved to the conf but cannot change the live image here. "
                          "For real control use pyuvc:  pip install pupil-labs-uvc,  then run with "
                          "sudo and --backend uvc."), True
        elif rejected:
            note, warn = f"Driver rejected: {', '.join(rejected)} (marked ✕). Values are still saved.", True
        else:
            note, warn = "All values are saved to the conf. Set auto_exposure to Manual for exposure to apply.", False
        return {
            "device": self._device_str,
            "conf_path": str(Path(self.out).resolve()),
            "backend": self.shim.backend,
            "auto_exposure": self.auto_exposure,
            "controls": dict(self.controls),
            "control_ranges": [[n, lo, hi] for n, (lo, hi, _c) in self._control_ranges.items()],
            "ctrl_support": sup,
            "ctrl_note": note, "ctrl_note_warn": warn,
            "modes": self._modes, "modes_note": self._modes_note,
            "fallback_fps": _COMMON_FPS,
            "current_res": f"{self.width}x{self.height}",
            "current_fps": self._cap_fps() or self.req_fps,
            "measured_fps": round(self._measured_fps, 2),
            "frame_w": self.width, "frame_h": self.height,
            "view": self.view,
            "pp_master": self.pp_master,
            "lens_on": self.lens_on, "lens_path": self.lens_path,
            "flatfield_on": self.flatfield_on, "flatfield_path": self.flatfield_path,
            "proc": dict(self.proc),
            "ff_status": self._ff.get("status", "idle"),
            "cb_cols": self.cb_cols, "cb_rows": self.cb_rows,
            "calib_active": cal["active"], "calib_views": len(cal["imgp"]),
            "calib_found": bool(cal["found"]), "min_views": self.args.min_views,
            "rms": cal["rms"],
            "auto_status": self._auto["status"],
        }

    def _api_control(self, data):
        name = data.get("name", "")
        value = data.get("value")
        if name == "auto_exposure":
            self.auto_exposure = bool(value)
            self._safe_apply({"auto_exposure": self.auto_exposure})
            return {"ok": True, "status_msg": f"auto_exposure = {self.auto_exposure}", "status_el": "ctrl_status"}
        if name in self._control_ranges:
            lo, hi, _ = self._control_ranges[name]
            v = max(lo, min(hi, int(float(value))))
            self.controls[name] = v
            self._safe_apply({name: v})
            msg = f"{name} = {v}"
            if self._ctrl_support.get(name) is False:
                msg += "  (saved; the driver ignores it)"
            elif name == "exposure" and self.auto_exposure:
                msg += "  (auto_exposure is on, so the camera ignores it)"
            return {"ok": True, "status_msg": msg, "status_el": "ctrl_status"}
        return {"ok": False, "status_msg": f"unknown control {name!r}", "status_el": "ctrl_status"}

    def _api_proc(self, data):
        key = data.get("key", "")
        value = data.get("value")
        if key == "view":
            self.view = value if value in _VIEWS else "raw"
        elif key == "preprocess":
            self.pp_master = bool(value)
        elif key == "lens_enabled":
            if value and not self.lens_path:
                return {"ok": False, "status_msg": "no lens.yaml yet: calibrate in the Lens tab first",
                        "status_el": "calib_status"}
            self.lens_on = bool(value)
        elif key == "flatfield_enabled":
            if value and not self.flatfield_path:
                return {"ok": False, "status_msg": "no flatfield yet: capture one first", "status_el": "ff_status"}
            self.flatfield_on = bool(value)
        elif key == "flip":
            self.proc["flip"] = value if value in _FLIP_CHOICES else "none"
        elif key in ("crop_enabled", "mask_enabled"):
            self.proc[key] = bool(value)
        elif key in ("crop_x", "crop_y", "crop_w", "crop_h", "mask_cx", "mask_cy", "mask_radius"):
            self.proc[key] = max(0, int(float(value)))
        else:
            return {"ok": False, "error": f"unknown key {key!r}"}
        return {"ok": True}

    def _api_mode(self, data):
        w, h = int(data.get("width", 0)), int(data.get("height", 0))
        fps = int(data.get("fps", 0)) or self.req_fps
        if w <= 0 or h <= 0:
            return {"ok": False, "status_msg": "invalid resolution", "status_el": "mode_status"}
        old = (self.width, self.height)
        self.req_w, self.req_h, self.req_fps = w, h, fps
        self._set_geometry(w, h, fps)
        # wait until frames arrive at the new size (UVC switches on the next read)
        seq0, deadline = self._jpeg_seq, time.perf_counter() + 3.0
        while time.perf_counter() < deadline:
            n = self._jpeg_seq - seq0
            if (n >= 2 and (self.width, self.height) != old) or n >= 6:
                break
            time.sleep(0.05)
        aw, ah = self.width, self.height
        if not self.proc["crop_enabled"]:           # an unused crop follows the new full frame
            self.proc.update(crop_x=0, crop_y=0, crop_w=0, crop_h=0)
        self.calib.update(objp=[], imgp=[], found=False, corners=None)
        self._pp_cache["sig"] = None
        msg = f"now {aw}x{ah} @ {self._cap_fps() or fps} fps"
        if (aw, ah) != (w, h):
            msg += f"  (asked {w}x{h}; the driver chose the nearest mode)"
        if self.lens_on:
            msg += "  ·  lens.yaml was solved at another size: recalibrate"
        return {"ok": True, "status_msg": msg, "status_el": "mode_status"}

    def _api_detect(self, data):
        self.calib["active"] = bool(data.get("active", False))
        return {"ok": True}

    def _api_board(self, data):
        try:
            cols, rows = int(data.get("cols", 0)), int(data.get("rows", 0))
        except (TypeError, ValueError):
            cols = rows = 0
        if not (2 <= cols <= 30 and 2 <= rows <= 30):
            return {"ok": False, "status_msg": "inner corners must be 2 to 30 per side", "status_el": "auto_status"}
        had = len(self.calib["imgp"])
        self._set_board(cols, rows)
        msg = f"board set to {cols}x{rows} inner corners"
        if had and not self.calib["imgp"]:
            msg += f" ({had} grabbed views cleared)"
        return {"ok": True, "status_msg": msg, "status_el": "auto_status"}

    def _api_board_auto(self):
        if self._auto["running"]:
            return {"ok": False, "status_msg": "already searching…", "status_el": "auto_status"}
        self._auto.update(running=True, status="searching…")
        threading.Thread(target=self._auto_board, daemon=True).start()
        return {"ok": True}

    def _api_flatfield(self):
        self._ff.update(cap=True, n=0, buf=None, status="capturing…")
        return {"ok": True}

    def _api_quit(self):
        self._running = False
        return {"ok": True, "status_msg": "server stopped; you can close this tab", "status_el": "save_status"}

    def _api_save(self):
        camera = {"device": self._device_str, "width": self.width, "height": self.height,
                  "fps": self._cap_fps() or self.req_fps, "backend": self._backend_req,
                  "auto_exposure": self.auto_exposure}
        camera.update({k: int(v) for k, v in self.controls.items()})
        preprocess = {"preprocess": self.pp_master, "lens_enabled": self.lens_on, "lens_path": self.lens_path,
                      "flip": self.proc["flip"], "crop_enabled": self.proc["crop_enabled"],
                      **self._crop_pp(),
                      "mask_enabled": self.proc["mask_enabled"], "mask_cx": self.proc["mask_cx"],
                      "mask_cy": self.proc["mask_cy"], "mask_radius": self.proc["mask_radius"],
                      "flatfield_enabled": self.flatfield_on, "flatfield_path": self.flatfield_path}
        path = str(Path(self.out).resolve())
        try:
            write_camera_conf(path, camera, preprocess)
            msg, ok = f"saved {Path(path).name} ({Path(path).stat().st_size} B)", True
            self.log.info(f"[calibrate] saved {path}")
        except Exception as e:
            msg, ok = f"SAVE FAILED: {type(e).__name__}: {e}", False
            self.log.warning(f"[calibrate] {msg}")
        return {"ok": ok, "status_msg": msg, "status_el": "save_status"}

    def _api_grab(self):
        cal = self.calib
        if not cal["active"]:
            return {"ok": False, "status_msg": "turn on 'Detect chessboard' first", "status_el": "calib_status"}
        corners = cal["corners"]
        if cal["found"] and corners is not None:
            cal["objp"].append(chessboard_object_points(self.cb_cols, self.cb_rows, self.args.square_size))
            cal["imgp"].append(corners.copy())
            return {"ok": True, "status_msg": f"grabbed view {len(cal['imgp'])}", "status_el": "calib_status"}
        return {"ok": False, "status_msg": "no board in view", "status_el": "calib_status"}

    def _api_delete(self):
        cal = self.calib
        if cal["imgp"]:
            cal["objp"].pop()
            cal["imgp"].pop()
            return {"ok": True, "status_msg": f"deleted; {len(cal['imgp'])} left", "status_el": "calib_status"}
        return {"ok": False, "status_msg": "nothing to delete", "status_el": "calib_status"}

    def _api_solve(self):
        cal, cv2 = self.calib, self.cv2
        if len(cal["imgp"]) < self.args.min_views:
            return {"ok": False, "status_msg": f"need {self.args.min_views} views, have {len(cal['imgp'])}",
                    "status_el": "calib_status"}
        rms, K, D, _r, _t = cv2.calibrateCamera(cal["objp"], cal["imgp"], cal["size"], None, None)
        cal.update(rms=float(rms), K=K, D=D)
        write_lens_yaml(self.lens_out, K, D, image_size=cal["size"], rms=rms)
        self.lens_path, self.lens_on = self.lens_out, True
        self._pp_cache["sig"] = None
        return {"ok": True, "status_msg": f"RMS {rms:.3f} px, wrote {Path(self.lens_out).name} (undistort on)",
                "status_el": "calib_status"}

    # ── run ────────────────────────────────────────────────────────────────

    def run(self, port: int, open_browser: bool = True) -> int:
        threading.Thread(target=self._camera_loop, daemon=True).start()
        threading.Thread(target=self._detect_loop, daemon=True).start()

        deadline = time.perf_counter() + 3.0
        while time.perf_counter() < deadline and not self._jpeg_buf:
            time.sleep(0.05)
        if not self._jpeg_buf:
            print("[calibrate] warning: no frame from the camera yet (still waiting)", file=sys.stderr)

        server = None
        for p in range(port, port + 10):
            try:
                server = ThreadingHTTPServer(("127.0.0.1", p), partial(_Handler, self))
                port = p
                break
            except OSError:
                continue
        if server is None:
            print(f"[calibrate] no free port in {port}..{port + 9}", file=sys.stderr)
            return 5
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, daemon=True).start()

        url = f"http://127.0.0.1:{port}"
        print(f"[calibrate] serving on {url}  (Q in the browser or Ctrl-C to quit)")
        print(f"[calibrate] backend {self.shim.backend} · {self._modes_note}")
        if open_browser:
            webbrowser.open(url)
        try:
            while self._running:
                time.sleep(0.2)
        except KeyboardInterrupt:
            self._running = False
        print("[calibrate] shutting down…")
        time.sleep(0.3)                               # let the stream handlers see _running=False
        server.shutdown()
        with self._cap_lock:
            self.cap.release()
        self.log.close()
        return 0


# ────────────────────────────────────────────────────────────────────────────
# HTTP handler (one thread per request: the MJPEG stream never blocks the API)
# ────────────────────────────────────────────────────────────────────────────

class _Handler(BaseHTTPRequestHandler):
    def __init__(self, app: _CalibApp, *args, **kwargs):
        self._app = app
        super().__init__(*args, **kwargs)

    def log_message(self, fmt, *args):
        pass

    def _json(self, obj, code=200):
        payload = json.dumps(obj, default=_json_default).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            payload = _HTML_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        elif path == "/stream":
            self._serve_mjpeg()
        elif path == "/snapshot.jpg":
            jpeg = self._app._jpeg_buf
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(jpeg)))
            self.end_headers()
            self.wfile.write(jpeg)
        elif path.startswith("/api/"):
            self._json(self._app.handle_api(path[5:], {}))
        else:
            self.send_error(404)

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if not path.startswith("/api/"):
            self.send_error(404)
            return
        n = int(self.headers.get("Content-Length", 0) or 0)
        body = self.rfile.read(n) if n else b""
        try:
            data = json.loads(body) if body.strip() else {}
        except json.JSONDecodeError:
            data = {}
        self._json(self._app.handle_api(path[5:], data if isinstance(data, dict) else {}))

    def _serve_mjpeg(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.end_headers()
        last = -1
        try:
            while self._app._running:
                seq = self._app._jpeg_seq
                if seq == last:
                    time.sleep(0.01)
                    continue
                last = seq
                jpeg = self._app._jpeg_buf
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass


def _json_default(o):
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
    except Exception:
        pass
    return str(o)


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

def _list_devices() -> int:
    try:
        import uvc
    except Exception:
        print("pyuvc not installed: UVC device listing unavailable. "
              "cv2 devices are indices 0,1,2,... (try --device 0).")
        if platform.system() == "Darwin" and shutil.which("ffmpeg"):
            r = subprocess.run(["ffmpeg", "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
                               capture_output=True, text=True)
            print("\nffmpeg/avfoundation devices:")
            for line in (r.stderr or "").splitlines():
                if "] [" in line:
                    print("  " + line.split("] ", 1)[1])
        return 0
    devs = uvc.device_list()
    print(f"UVC devices: {len(devs)}")
    for i, d in enumerate(devs):
        print(f"  [{i}] {d.get('name')}  (uid {d.get('uid')})")
    return 0


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="calibrate_camera_web",
        description="Browser-based camera + preprocess calibration (no dearpygui needed)")
    p.add_argument("--list-devices", action="store_true", help="list devices + exit")
    p.add_argument("--device", default=None, help="capture index / /dev/videoN / UVC name (conf, else 0)")
    p.add_argument("--width", type=int, default=None, help="requested capture width (conf, else 1280)")
    p.add_argument("--height", type=int, default=None, help="requested capture height (conf, else 720)")
    p.add_argument("--fps", type=int, default=None, help="requested capture fps (conf, else 30)")
    p.add_argument("--backend", default=None, choices=("auto", "uvc", "cv2"),
                   help="camera open backend (conf, else auto)")
    p.add_argument("--flip", default="", choices=("",) + _FLIP_CHOICES,
                   help="initial mirror when the conf has none: none / x / y / xy")
    p.add_argument("--calibration", "-c", default="", help="seed an existing lens YAML")
    p.add_argument("--flatfield", default="", help="seed a flatfield PNG path")
    p.add_argument("--chessboard", default="9x6", help="board INNER corners COLSxROWS (default 9x6)")
    p.add_argument("--square-size", dest="square_size", type=float, default=1.0,
                   help="chessboard square edge (default 1.0)")
    p.add_argument("--min-views", dest="min_views", type=int, default=12,
                   help="chessboard views required before Solve (default 12)")
    p.add_argument("--lens-out", dest="lens_out", default="",
                   help="lens YAML output (default: lens.yaml next to --out)")
    p.add_argument("--out", "--conf", dest="out", default="camera1.conf",
                   help="conf to edit/create (loaded to resume, saved back with S)")
    p.add_argument("--port", type=int, default=8088, help="HTTP server port (default 8088)")
    p.add_argument("--no-browser", dest="no_browser", action="store_true",
                   help="don't open a browser tab automatically")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(sys.argv[1:] if argv is None else argv))
    if args.list_devices:
        return _list_devices()
    try:
        app = _CalibApp(args)
    except (RuntimeError, ValueError) as e:
        print(f"[calibrate] {e}", file=sys.stderr)
        return 3
    return app.run(port=args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    raise SystemExit(main())
