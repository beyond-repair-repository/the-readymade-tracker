"""[camera] config coercion + camera1.conf layering (I/O roadmap A.1 / D).

Same discipline as test_config_seg: every [camera] BOOL must be schema-declared or a
conf `false` returns the raw string and `bool("false")` is True (this bit
hand_clean/fit_crop). [camera] has one bool — auto_exposure — and a manual `exposure`
that only takes effect when it's off, so a stuck-True auto_exposure would silently
override the whole manual-exposure calibration.

Also covers the D→A.1 seam: `--source camera camera1.conf` layers the calibration
tool's output over readymade.conf (materializing [camera], overriding [preprocess]), and
the conf-writer round-trips through the loader.
"""

from pathlib import Path

import numpy as np

import readymade_tracker
from lib.preprocess import PreprocessConfig
from tools.calibrate_camera import (chessboard_object_points, format_camera_conf,
                                     logical_control_ranges, parse_chessboard, seed_from_conf,
                                     write_camera_conf, write_lens_yaml)


# --------------------------------------------------------------------------- #
# schema coercion  (via a real camera1.conf — the production path: a [camera]
# section reaches the manifest through readymade.conf or the layered camera1.conf, not
# via a CLI override, which only edits sections already present)
# --------------------------------------------------------------------------- #

def _load_camera(tmp_path, camera_body, **override):
    loc = tmp_path / "camera1.conf"
    loc.write_text(f"[camera]\n{camera_body}")
    ov = {"input": {"source": "camera", "location": str(loc)}, **override}
    return readymade_tracker._load(readymade_tracker.DEFAULT_CONF, ov).get("camera")


def test_camera_bool_coerces_not_truthy_string(tmp_path):
    cam = _load_camera(tmp_path, "auto_exposure = false\n")
    assert cam.get("auto_exposure") is False        # not the truthy string "false"


def test_camera_numeric_controls_coerce_to_int(tmp_path):
    cam = _load_camera(tmp_path, "width = 640\nheight = 480\nexposure = 300\ngain = 12\n")
    assert cam.get("width") == 640 and isinstance(cam.get("width"), int)
    assert cam.get("exposure") == 300 and cam.get("gain") == 12


def test_every_camera_bool_is_declared():
    # any [camera] knob read via bool(...) must be a declared bool in the coercion schema, or a
    # conf `false` silently becomes True. camera is plugin-owned now → CameraBackend.PARAMS via CONF_SCHEMAS.
    schema = readymade_tracker.CONF_SCHEMAS["camera"]
    assert schema.get("auto_exposure") is bool, "[camera].auto_exposure must be a schema bool"


def test_camera_defaults_from_plugin_conf(tmp_path):
    # even an empty [camera] section inherits device/backend/fps from
    # plugins/input/camera/camera.conf (the plugin-default layer).
    cam = _load_camera(tmp_path, "")
    assert cam is not None
    assert cam.get("backend") == "auto"
    assert cam.get("fps") == 30


# --------------------------------------------------------------------------- #
# camera1.conf layering (the D→A.1 seam)
# --------------------------------------------------------------------------- #

def _write(tmp_path, text) -> str:
    p = tmp_path / "camera1.conf"
    p.write_text(text)
    return str(p)


def test_camera1_conf_layered_materializes_section(tmp_path):
    loc = _write(tmp_path, "[camera]\ndevice = 2\nexposure = 250\nauto_exposure = false\n"
                           "[preprocess]\npreprocess = true\ncrop_enabled = true\ncrop_top_y = 16\n")
    cfg = readymade_tracker._load(readymade_tracker.DEFAULT_CONF, {"input": {"source": "camera", "location": loc}})
    cam = cfg.get("camera")
    assert cam is not None
    assert cam.get("device") == "2"
    assert cam.get("exposure") == 250
    assert cam.get("auto_exposure") is False
    assert cam.get("fps") == 30                     # inherited from the plugin default
    pp = cfg.get("preprocess")
    assert pp.get("preprocess") is True             # camera1.conf overrode readymade.conf's default
    assert pp.get("crop_enabled") is True and pp.get("crop_top_y") == 16


def test_cli_override_wins_over_camera1_conf(tmp_path):
    loc = _write(tmp_path, "[camera]\ndevice = 2\n[preprocess]\npreprocess = true\n")
    cfg = readymade_tracker._load(readymade_tracker.DEFAULT_CONF, {"input": {"source": "camera", "location": loc},
                                         "camera": {"device": "9"},
                                         "preprocess": {"preprocess": "false"}})
    assert cfg.get("camera").get("device") == "9"       # CLI beats camera1.conf
    assert cfg.get("preprocess").get("preprocess") is False


def test_missing_camera1_conf_is_harmless(tmp_path):
    # a location that doesn't exist must not blow up loading (the tracker falls back to
    # the [camera] plugin defaults / backend defaults).
    cfg = readymade_tracker._load(readymade_tracker.DEFAULT_CONF,
                     {"input": {"source": "camera", "location": str(tmp_path / "nope.conf")}})
    # [camera] absent (no file, none in readymade.conf) → backend uses its own defaults; no crash
    assert cfg.get("input").get("source") == "camera"


def test_video_source_unaffected_by_camera_layer():
    # the default (video) path must load exactly as before — no camera layering.
    cfg = readymade_tracker._load(readymade_tracker.DEFAULT_CONF, {})
    assert cfg.get("input").get("source") == "video"
    assert cfg.get("camera") is None


# --------------------------------------------------------------------------- #
# the calibration conf-writer round-trips through the loader
# --------------------------------------------------------------------------- #

def test_conf_writer_roundtrips(tmp_path):
    cam = {"device": "0", "width": 1280, "height": 720, "fps": 30, "backend": "auto",
           "auto_exposure": False, "exposure": 300, "brightness": 140, "gain": 12}
    pp = {"preprocess": True, "crop_enabled": True, "crop_top_x": 10, "crop_top_y": 20,
          "crop_width": 990, "crop_height": 0, "mask_enabled": True,
          "mask_cx": 500, "mask_cy": 400, "mask_radius": 350}
    out = tmp_path / "camera1.conf"
    write_camera_conf(out, cam, pp)

    cfg = readymade_tracker._load(readymade_tracker.DEFAULT_CONF, {"input": {"source": "camera", "location": str(out)}})
    c = cfg.get("camera")
    assert c.get("device") == "0" and c.get("auto_exposure") is False
    assert c.get("exposure") == 300 and c.get("gain") == 12 and c.get("brightness") == 140
    p = cfg.get("preprocess")
    assert p.get("crop_enabled") is True and p.get("mask_enabled") is True
    assert p.get("mask_radius") == 350 and p.get("crop_top_x") == 10


def test_conf_writer_bools_render_lowercase():
    txt = format_camera_conf({"device": "0", "auto_exposure": True},
                             {"preprocess": True, "crop_enabled": False})
    assert "auto_exposure = true" in txt
    assert "crop_enabled = false" in txt
    assert "[camera]" in txt and "[preprocess]" in txt


def test_seed_from_conf_resumes_written_values(tmp_path):
    # a re-run of the calibrator resumes from the last camera1.conf it wrote.
    cam = {"device": "1", "width": 640, "height": 480, "fps": 30, "backend": "cv2",
           "auto_exposure": False, "exposure": 275, "gain": 8}
    pp = {"preprocess": True, "crop_enabled": True, "crop_top_x": 5, "mask_enabled": False,
          "flatfield_enabled": True, "flatfield_path": "flatfield.png"}
    out = tmp_path / "camera1.conf"
    write_camera_conf(out, cam, pp)
    seed_cam, seed_pp = seed_from_conf(out)
    assert seed_cam["device"] == "1" and seed_cam["exposure"] == "275"      # raw strings, for trackbar seeding
    assert seed_cam["auto_exposure"] == "false"
    assert seed_pp["crop_enabled"] == "true" and seed_pp["crop_top_x"] == "5"
    assert seed_pp["flatfield_path"] == "flatfield.png"


def test_seed_from_conf_missing_file_is_empty(tmp_path):
    seed_cam, seed_pp = seed_from_conf(tmp_path / "does-not-exist.conf")
    assert seed_cam == {} and seed_pp == {}


# --------------------------------------------------------------------------- #
# intrinsic (undistort) calibration output — the lens YAML the tool writes MUST be
# consumable by lib.preprocess (the tracker's actual lens stage), or undistort is a
# silent no-op. This is the load-bearing seam between D (produces) and the pipeline.
# --------------------------------------------------------------------------- #

def test_parse_chessboard():
    assert parse_chessboard("9x6") == (9, 6)
    assert parse_chessboard("7X5") == (7, 5)
    for bad in ("9", "axb", "1x9", "9x1", ""):
        try:
            parse_chessboard(bad)
            assert False, f"expected ValueError for {bad!r}"
        except ValueError:
            pass


def test_chessboard_object_points_grid():
    op = chessboard_object_points(9, 6, 2.5)
    assert op.shape == (54, 3)
    assert (op[:, 2] == 0).all()                     # planar board (z = 0)
    assert op[1, 0] - op[0, 0] == 2.5                # spacing = square_size


def test_logical_control_ranges_defaults_without_uvc():
    # a plain (cv2-like) capture with no uvc_controls → the tool's default ranges.
    class Cap:
        pass
    r = logical_control_ranges(Cap())
    assert set(r) == {"brightness", "contrast", "saturation", "gamma", "gain",
                      "exposure", "white_balance"}                 # auto_exposure excluded (bool)
    assert r["exposure"][0] == 0 and r["exposure"][1] == 5000
    lo, hi, cur = r["brightness"]
    assert lo <= cur <= hi


def test_logical_control_ranges_from_uvc_device():
    # a UvcCapture-like device → real per-control min/max/current, mapped by display name.
    class Ctrl:
        def __init__(self, mn, mx, v):
            self.min_val, self.max_val, self.value = mn, mx, v
    cap = type("Cap", (), {"uvc_controls": {"Brightness": Ctrl(-64, 64, 10),
                                            "Gain": Ctrl(0, 100, 30)}})()
    r = logical_control_ranges(cap)
    assert r["brightness"] == (-64, 64, 10)       # device range + current value used
    assert r["gain"] == (0, 100, 30)
    assert r["exposure"][1] == 5000               # not exposed by this device → default


def test_lens_yaml_roundtrips_through_preprocess(tmp_path):
    # the exact seam: write a lens map, then load it via the REAL PreprocessConfig the
    # tracker uses. If preprocess disables it (missing/wrong keys), undistort silently
    # never happens — this guards the format contract.
    K = np.array([[900.0, 0, 640], [0, 900, 360], [0, 0, 1]])
    D = np.array([-0.32, 0.12, 0.001, -0.002, 0.0])
    p = tmp_path / "lens.yaml"
    write_lens_yaml(p, K, D, image_size=(1280, 720), rms=0.21)

    cfg = PreprocessConfig(lens_enabled=True, lens_path=str(p))
    cfg.load_artifacts()
    assert cfg.lens_enabled is True                  # NOT disabled -> preprocess accepted it
    assert cfg.lens_camera_matrix is not None and cfg.lens_dist_coeffs is not None
    np.testing.assert_allclose(cfg.lens_camera_matrix, K, rtol=1e-4)
    np.testing.assert_allclose(cfg.lens_dist_coeffs.ravel(), D, rtol=1e-4)


def test_flatfield_capture_and_delete(tmp_path):
    from lib.preprocess import Preprocess, PreprocessConfig
    pp = Preprocess(PreprocessConfig(flatfield_path=str(tmp_path / "ff.png")))
    pp.capture_flatfield(2)                              # average the next 2 frames
    assert pp.cfg.flatfield_enabled is False             # subtraction is OFF while capturing the reference
    pp.process(np.full((10, 10, 3), 100, np.uint8))
    pp.process(np.full((10, 10, 3), 200, np.uint8))      # → avg 150, capture completes
    assert pp.cfg.flatfield_enabled is True and pp.cfg.flatfield_arr is not None
    assert int(pp.cfg.flatfield_arr[0, 0, 0]) == 150
    assert (tmp_path / "ff.png").exists()                # persisted for SAVE + restart
    pp.clear_flatfield()
    assert pp.cfg.flatfield_enabled is False and pp.cfg.flatfield_arr is None


def test_flip_mirrors_the_frame_and_is_live():
    from lib.preprocess import Preprocess, PreprocessConfig
    frame = np.zeros((20, 30, 3), np.uint8)
    frame[:, 0] = 255                                    # a white LEFT column
    pp = Preprocess(PreprocessConfig(flip="x"))          # horizontal mirror → column moves RIGHT
    bgr, _ = pp.process(frame)
    assert bgr[0, -1, 0] == 255 and bgr[0, 0, 0] == 0
    pp.apply_param("flip", "none")                       # live toggle off → back to the left
    assert pp.process(frame)[0][0, 0, 0] == 255
    pp.apply_param("flip", "y")                          # vertical mirror → top row's column stays x=0, rows swap
    assert pp.process(frame)[0][0, 0, 0] == 255          # (a full-height column is unchanged by a vertical flip)


def test_apply_param_is_live_and_invalidates_caches():
    # the webgui calibration path: a live [preprocess] set writes the cfg AND drops the cached artifacts,
    # so the very next process() re-resolves the crop rect / mask circle (no restart).
    from lib.preprocess import Preprocess
    pp = Preprocess(PreprocessConfig(crop_enabled=True, crop_top_x=10, crop_top_y=10,
                                     crop_width=80, crop_height=80,
                                     mask_enabled=True, mask_cx=50, mask_cy=50, mask_radius=20))
    frame = np.full((100, 100, 3), 120, np.uint8)
    bgr, _ = pp.process(frame)
    assert bgr.shape[:2] == (80, 80)                              # cropped 10..90 (h, w)
    assert pp._crop_rect == (10, 10, 90, 90) and pp._mask_circle is not None

    pp.apply_param("crop_width", 50)                              # live-narrow the crop (10..60)
    assert pp.cfg.crop_width == 50
    assert pp._crop_rect is None and pp._mask_circle is None      # a crop change drops the mask cache too
    bgr2, _ = pp.process(frame)
    assert bgr2.shape[:2] == (80, 50) and pp._mask_circle is not None   # re-resolved: now 10..60 wide

    pp.apply_param("mask_radius", 5)                             # a mask change drops ONLY the mask cache
    assert pp.cfg.mask_radius == 5 and pp._mask_circle is None and pp._crop_rect is not None

    pp.apply_param("preprocess", False)                          # master switch off → full-frame passthrough
    assert pp.cfg.enabled is False
    assert pp.process(frame)[0].shape[:2] == (100, 100)
