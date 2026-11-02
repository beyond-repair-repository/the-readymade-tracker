"""Frame preprocessing — lens / crop / flatfield / gray / mask — a CORE stage.

Preprocess is a **core stage run on the input thread**: the core input-runner
calls ``process(bgr) -> (bgr, gray)`` after ``backend.grab()`` and before
``FrameCaptured`` — off the pipeline budget. Pure math: no threads, no I/O, no
``VideoCapture`` (that's the backend), no ``SharedState``.

Stage order ``lens → crop → flatfield → gray → mask`` (mask applies to GRAY only —
the contour pipeline reads gray, so a zeroed-outside-ROI gray suppresses spurious
contours while BGR stays full). Artifacts (lens maps / crop rect / flatfield array
/ circular mask) resolve lazily on the first frame, then cache. The master switch
``preprocess = false`` (or ``--no-preprocess``) skips every stage to a plain
BGR→gray (an already-conditioned source); a degenerate stage disables itself.

**Fast path:** when BOTH lens-undistort + crop are on they are *fused* into one
``remap`` whose output is only the crop window — no compute on undistorted pixels
outside the crop, provably identical to undistort-then-crop (a flatfield at the raw
size is likewise cropped once + cached).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Mapping

import cv2
import numpy as np

from lib.fsutil import own_to_dir_owner
from lib.logger import NULL_LOGGER
from lib.params import ParamSpec


# cv2.flip codes: 1 = horizontal (mirror L↔R), 0 = vertical (mirror ↑↕↓), -1 = both. None = no flip.
_FLIP_CODES = {"x": 1, "y": 0, "xy": -1}


# --------------------------------------------------------------------------- #
# Config + artifacts
# --------------------------------------------------------------------------- #

def _pp_apply(key: str):
    """Bind a [preprocess] param key to the running Preprocess instance's live setter. The apply TARGET
    is the Preprocess (which owns the caches), not this config dataclass that merely declares PARAMS."""
    return lambda pp, v: pp.apply_param(key, v)


@dataclass
class PreprocessConfig:
    """The preprocess settings (all coords in RAW frame pixels) + the resolved
    artifacts. Built explicitly (tests) or via ``from_mapping`` (a plugin's
    config ``Section``); call ``load_artifacts`` to read the lens YAML +
    flatfield (an image — ``.png``/``.tif`` — or a ``.npy``) from disk. A stage
    with its flag off is a no-op."""

    # -- the [preprocess] live parameters (registry anchor). All needs-restart
    #    (the crop/mask/lens/flatfield are resolved at startup) → bind=False; from_mapping reads the config.
    #    Unannotated class attr → NOT a dataclass field. --
    #    SIZE-CHANGING params (crop + the master switch that gates it) are needs-restart: they resize the
    #    frame, and a live resize would strand every stateful downstream stage's frame-sized buffers
    #    (a crash). They EDIT → STAGE (⟳ save + restart to apply). The rest — lens/mask/flatfield —
    #    keep the SAME frame size so they stay HOT (live): apply_param writes the cfg + drops the artifact
    #    caches so the next frame re-resolves. bind=False → cold start still reads from_mapping.
    PARAMS = (
        ParamSpec("preprocess", bool, None, hot=False, bind=False, expose="edit", doc="Master switch: off skips all stages, just does BGR to gray (needs restart - changes frame size)"),
        ParamSpec("lens_enabled", bool, None, hot=True, bind=False, apply=_pp_apply("lens_enabled"), doc="Enable lens undistort remap from calibration file"),
        ParamSpec("lens_path", str, None, hot=True, bind=False, apply=_pp_apply("lens_path"), doc="Path to lens calibration YAML (camera matrix + dist coeffs)"),
        ParamSpec("flip", str, "none", enum=("none", "x", "y", "xy"), hot=True, bind=False, apply=_pp_apply("flip"), doc="Mirror the frame: none / x (horizontal) / y (vertical) / xy (both) - same size, so live"),
        ParamSpec("crop_enabled", bool, None, hot=False, bind=False, expose="edit", doc="Enable rectangular crop to a ROI (needs restart - changes frame size)"),
        ParamSpec("crop_top_x", int, None, lo=0, hot=False, bind=False, expose="edit", doc="Crop ROI origin X (left edge) in raw pixels (needs restart)"),
        ParamSpec("crop_top_y", int, None, lo=0, hot=False, bind=False, expose="edit", doc="Crop ROI origin Y (top edge) in raw pixels (needs restart)"),
        ParamSpec("crop_width", int, None, lo=0, hot=False, bind=False, expose="edit", doc="Crop ROI width in raw pixels, 0 = to the frame edge (needs restart)"),
        ParamSpec("crop_height", int, None, lo=0, hot=False, bind=False, expose="edit", doc="Crop ROI height in raw pixels, 0 = to the frame edge (needs restart)"),
        ParamSpec("mask_enabled", bool, None, hot=True, bind=False, apply=_pp_apply("mask_enabled"), doc="Enable circular ROI mask zeroing gray outside the circle"),
        ParamSpec("mask_cx", int, None, lo=0, hot=True, bind=False, apply=_pp_apply("mask_cx"), doc="Circular mask center X in raw pixels"),
        ParamSpec("mask_cy", int, None, lo=0, hot=True, bind=False, apply=_pp_apply("mask_cy"), doc="Circular mask center Y in raw pixels"),
        ParamSpec("mask_radius", int, None, lo=0, hot=True, bind=False, apply=_pp_apply("mask_radius"), doc="Circular mask radius in pixels (0 disables the mask)"),
        ParamSpec("flatfield_enabled", bool, None, hot=True, bind=False, apply=_pp_apply("flatfield_enabled"), doc="Enable flatfield dark-frame subtraction"),
        ParamSpec("flatfield_path", str, None, hot=True, bind=False, apply=_pp_apply("flatfield_path"), doc="Path to flatfield dark-frame image or .npy to subtract"),
    )

    # Master switch (config key ``preprocess``): when False, ``process`` skips
    # every stage and just does BGR→gray — for a source that's already
    # conditioned (a pre-processed clip). Set per-section, or globally via the
    # ``--no-preprocess`` CLI flag. Overrides the individual stage flags.
    enabled: bool = True

    lens_enabled: bool = False
    lens_path: str = ""
    lens_camera_matrix: np.ndarray | None = None
    lens_dist_coeffs: np.ndarray | None = None

    flip: str = "none"          # none / x / y / xy — a frame mirror (after lens, before crop)

    crop_enabled: bool = False
    crop_top_x: int = 0      # ROI origin X (left edge)
    crop_top_y: int = 0      # ROI origin Y (top edge)
    crop_width: int = 0      # 0 → "to the right frame edge"  (resolved on the first frame)
    crop_height: int = 0     # 0 → "to the bottom frame edge"

    mask_enabled: bool = False
    mask_cx: int = 0
    mask_cy: int = 0
    mask_radius: int = 0

    flatfield_enabled: bool = False
    flatfield_path: str = ""
    flatfield_arr: np.ndarray | None = None  # the dark-frame to subtract

    @classmethod
    def from_mapping(cls, m: Mapping[str, Any], *, base_dir: str | None = None
                     ) -> "PreprocessConfig":
        """Build from a config ``Section`` / dict (casts its own bool/int per key).
        Relative ``lens_path`` / ``flatfield_path`` are resolved against
        ``base_dir`` (the plugin folder / repo root) when given."""
        def path(p: str) -> str:
            if p and base_dir and not os.path.isabs(p):
                return os.path.join(base_dir, p)
            return p
        return cls(
            enabled=bool(m.get("preprocess", True)),
            lens_enabled=bool(m.get("lens_enabled", False)),
            lens_path=path(m.get("lens_path", "")),
            flip=str(m.get("flip", "none")).lower(),
            crop_enabled=bool(m.get("crop_enabled", False)),
            crop_top_x=int(m.get("crop_top_x", 0)),
            crop_top_y=int(m.get("crop_top_y", 0)),
            crop_width=max(0, int(m.get("crop_width", 0))),
            crop_height=max(0, int(m.get("crop_height", 0))),
            mask_enabled=bool(m.get("mask_enabled", False)),
            mask_cx=int(m.get("mask_cx", 0)),
            mask_cy=int(m.get("mask_cy", 0)),
            mask_radius=int(m.get("mask_radius", 0)),
            flatfield_enabled=bool(m.get("flatfield_enabled", False)),
            flatfield_path=path(m.get("flatfield_path", "")),
        )

    def load_artifacts(self, logger: Any = None) -> None:
        """Load the lens calibration YAML + flatfield (image or ``.npy``) if
        enabled, disabling the stage (with a warning) on any error — fail-soft, via the logger."""
        log = logger or NULL_LOGGER

        if self.lens_enabled and self.lens_path:
            if not os.path.isfile(self.lens_path):
                log.warning(f"[preprocess] lens calibration not found: "
                            f"{self.lens_path} -> disabling")
                self.lens_enabled = False
            else:
                try:
                    fs = cv2.FileStorage(self.lens_path, cv2.FILE_STORAGE_READ)
                    mat = _read_node(fs, ("camera_matrix", "cameraMatrix",
                                          "K", "intrinsic_matrix"))
                    dist = _read_node(fs, ("distortion_coefficients", "dist_coeffs",
                                           "distCoeffs", "D"))
                    fs.release()
                    if mat is None or dist is None:
                        log.warning(f"[preprocess] lens calibration missing "
                                    f"matrix/coeffs in {self.lens_path} -> disabling")
                        self.lens_enabled = False
                    else:
                        self.lens_camera_matrix = mat.astype(np.float32)
                        self.lens_dist_coeffs = dist.astype(np.float32)
                except Exception as e:  # noqa: BLE001 — fail soft
                    log.warning(f"[preprocess] lens load failed: {e} -> disabling")
                    self.lens_enabled = False

        if self.flatfield_enabled and self.flatfield_path:
            if not os.path.isfile(self.flatfield_path):
                log.warning(f"[preprocess] flatfield not found: "
                            f"{self.flatfield_path} -> disabling")
                self.flatfield_enabled = False
                return
            try:
                ext = os.path.splitext(self.flatfield_path)[1].lower()
                if ext in (".npy", ".npz"):
                    arr = np.load(self.flatfield_path)
                else:                            # an image (png/tif/jpg) — the natural flatfield
                    arr = cv2.imread(self.flatfield_path, cv2.IMREAD_COLOR)
                    if arr is None:
                        raise ValueError(f"cv2.imread could not read {self.flatfield_path!r}")
                if arr.dtype != np.uint8:        # uint8 for the saturating subtract
                    arr = np.clip(arr, 0, 255).astype(np.uint8)
                self.flatfield_arr = arr
            except Exception as e:  # noqa: BLE001
                log.warning(f"[preprocess] flatfield load failed: {e} -> disabling")
                self.flatfield_enabled = False


def _read_node(fs: "cv2.FileStorage", keys: tuple[str, ...]) -> np.ndarray | None:
    """Return the first matrix found under any of ``keys`` (key-spelling fallback)."""
    for k in keys:
        n = fs.getNode(k)
        if not n.empty():
            return n.mat()
    return None


# --------------------------------------------------------------------------- #
# The preprocessor — lazy artifacts resolved on the first frame, then cached
# --------------------------------------------------------------------------- #

class Preprocess:
    """Runs ``lens → crop → flatfield → gray → mask`` on each BGR frame, lazily
    building the remap maps / crop rect / flatfield array / circular mask on the
    first frame (when the raw size is known) and caching them. Stateful only in
    those caches — no threads, no I/O, no ``SharedState``."""

    def __init__(self, cfg: PreprocessConfig, *, logger: Any = None) -> None:
        self.cfg = cfg
        self.log = logger or NULL_LOGGER
        self._lens_mapx: np.ndarray | None = None
        self._lens_mapy: np.ndarray | None = None
        self._lens_map_size: tuple[int, int] | None = None   # (w, h) the lens map was built for (= crop size)
        self._crop_rect: tuple[int, int, int, int] | None = None
        self._input_full: np.ndarray | None = None      # the flipped, PRE-crop frame — the 'input'/raw view
        self._raw_hw: tuple[int, int] | None = None     # (h, w) of the incoming frame
        self._flatfield_ready: np.ndarray | None = None
        self._mask_circle: np.ndarray | None = None
        self._ff_cap: dict | None = None                # in-progress flatfield capture (average N frames)

    # -- flatfield capture / delete (webgui buttons via the /rmt/flatfield control verb) -------------- #

    def capture_flatfield(self, n: int = 30) -> None:
        """Start averaging the next ``n`` post-crop frames into a NEW flatfield (the static background /
        vignetting to subtract — capture it over an EMPTY table). Disables subtraction while capturing so
        the reference is clean. Runs the accumulation on the input thread via ``process``."""
        self.cfg.flatfield_enabled = False
        self._flatfield_ready = None
        self._ff_cap = {"n": max(1, int(n)), "acc": None, "count": 0}
        self.log.info(f"[preprocess] flatfield capture started ({self._ff_cap['n']} frames)")

    def clear_flatfield(self) -> None:
        """Delete the current flatfield: disable + drop the array (the file, if any, is left on disk)."""
        self.cfg.flatfield_enabled = False
        self.cfg.flatfield_arr = None
        self._flatfield_ready = None
        self._ff_cap = None
        self.log.info("[preprocess] flatfield cleared")

    def _accumulate_flatfield(self, bgr: np.ndarray) -> None:
        cap = self._ff_cap                              # read ONCE: capture_flatfield/clear_flatfield run on
        if cap is None:                                 # the CONTROL thread and may null it between process()'s
            return                                      # check and this call — without the local ref that race
        #                                                 TypeErrors and kills the input thread (a silent EOF)
        if cap["acc"] is None or cap["acc"].shape != bgr.shape:
            cap["acc"] = np.zeros(bgr.shape, np.float64)
        cap["acc"] += bgr
        cap["count"] += 1
        if cap["count"] >= cap["n"]:                    # done → average → the new flatfield (live + saved)
            avg = np.clip(cap["acc"] / cap["count"], 0, 255).astype(np.uint8)
            self.cfg.flatfield_arr = avg
            self._flatfield_ready = None
            self.cfg.flatfield_enabled = True
            self._ff_cap = None
            path = self.cfg.flatfield_path or "flatfield.png"
            try:
                cv2.imwrite(path, avg)                  # persist so SAVE + restart can reload it
                own_to_dir_owner(path)                  # a sudo run must not root-own the flatfield
                self.log.info(f"[preprocess] flatfield captured ({cap['count']} frames) -> {path}")
            except Exception as e:                      # noqa: BLE001 — a bad path must not kill the pipeline
                self.log.warning(f"[preprocess] flatfield write failed ({path}): {e}")

    def apply_param(self, key: str, value: Any) -> None:
        """Live-set a [preprocess] param: write the cfg, then invalidate the artifact caches the key feeds
        so the next ``process`` re-resolves them (the same cache-invalidation pattern ``features`` uses for
        efd_detail). Runs on the CONTROL thread while ``process`` runs on the input thread — clearing a
        cache only forces a re-resolve next frame, and a torn multi-key edit self-heals within a frame.
        This is what makes crop/mask/flatfield calibration LIVE from the webgui."""
        c = self.cfg
        setattr(c, "enabled" if key == "preprocess" else key, value)   # 'preprocess' master switch → cfg.enabled
        if key == "preprocess" or key.startswith("crop"):              # crop feeds the rect, the crop-sized
            self._crop_rect = None                                     #   lens map, + flatfield/mask translations
            self._lens_mapx = self._lens_mapy = None                   # crop size changed → rebuild undistort map
            self._flatfield_ready = self._mask_circle = None
        if key.startswith("lens"):
            self._lens_mapx = self._lens_mapy = None
        if key.startswith("mask"):
            self._mask_circle = None
        if key.startswith("flatfield"):
            self._flatfield_ready = None
        if key in ("lens_enabled", "lens_path", "flatfield_enabled", "flatfield_path"):
            c.load_artifacts(self.log)                                 # (re)read the file-backed artifacts

    @property
    def input_full(self) -> "np.ndarray | None":
        """The flipped, PRE-crop frame from the last ``process`` — the 'input'/raw view (crop + mask
        overlays are drawn on THIS, in full-frame coords, so they land correctly)."""
        return self._input_full

    def process(self, bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Preprocess one BGR frame → ``(bgr, gray)``. Pipeline order: FLIP → CROP → lens UNDISTORT →
        flatfield → gray → mask. The lens undistort runs on the flipped, CROPPED frame, so it is
        calibrated at the crop size (crop selects the working window; undistort straightens just that
        window). Crop/mask coords live in the flipped frame; the crop rect is resolved once + cached."""
        raw_h, raw_w = bgr.shape[:2]
        self._raw_hw = (raw_h, raw_w)
        c = self.cfg
        if not c.enabled:                      # master switch off → just gray
            self._input_full = bgr
            return bgr, self._gray(bgr)
        flip = _FLIP_CODES.get(str(c.flip).lower())     # None / 1 / 0 / -1
        if flip is not None:
            bgr = cv2.flip(bgr, flip)                    # 1. mirror the raw frame
        # the flipped, PRE-crop frame — the 'input'/raw view + the crop-calibration overlay live in THIS
        # space (crop/mask coords are flipped-full px), so a crop rectangle drawn on it lands correctly.
        self._input_full = bgr
        bgr = self._crop(bgr, raw_w, raw_h)              # 2. crop the flipped frame (coords in flipped space)
        bgr = self._lens(bgr)                            # 3. undistort the cropped frame (map @ crop size)
        if self._ff_cap is not None:                     # averaging N post-undistort frames into a new flatfield
            self._accumulate_flatfield(bgr)
        bgr = self._flatfield(bgr)
        gray = self._gray(bgr)
        gray = self._mask(gray, bgr.shape)
        return bgr, gray

    # -- stages ------------------------------------------------------------- #

    def _lens(self, bgr: np.ndarray) -> np.ndarray:
        """Undistort the (already flipped + cropped) frame. The map is built for the CURRENT frame size —
        the crop size — and rebuilt whenever that changes, so the lens must be calibrated at the crop size."""
        c = self.cfg
        if not (c.lens_enabled and c.lens_camera_matrix is not None
                and c.lens_dist_coeffs is not None):
            return bgr
        h, w = bgr.shape[:2]
        # READ-INTO-LOCALS discipline: a hot /rmt/set preprocess lens_* (control thread) nulls these
        # caches to force a rebuild — re-reading the attrs after the None-check would hand cv2.remap a
        # None mid-swap and kill the input thread as a pseudo-EOF. Locals make check + use one snapshot.
        mapx, mapy = self._lens_mapx, self._lens_mapy
        if mapx is None or mapy is None or self._lens_map_size != (w, h):
            K, D = c.lens_camera_matrix, c.lens_dist_coeffs
            mapx, mapy = cv2.initUndistortRectifyMap(K, D, None, K, (w, h), cv2.CV_16SC2)
            self._lens_mapx, self._lens_mapy = mapx, mapy
            self._lens_map_size = (w, h)
            self.log.info(f"[preprocess] lens maps built for {w}x{h} (post-crop)")
        return cv2.remap(bgr, mapx, mapy, cv2.INTER_LINEAR)

    def _resolve_crop_rect(self, raw_w: int, raw_h: int
                           ) -> tuple[int, int, int, int] | None:
        """Resolve + cache the crop rect on the first frame (clamped to the frame;
        a ``0`` bottom coord → full extent). A degenerate rect disables crop and
        returns ``None``. Shared by the crop-only path and the fused lens+crop."""
        if self._crop_rect is not None:
            return self._crop_rect
        c = self.cfg
        tx, ty = max(0, c.crop_top_x), max(0, c.crop_top_y)
        bx = min(raw_w, (tx + c.crop_width) if c.crop_width else raw_w)
        by = min(raw_h, (ty + c.crop_height) if c.crop_height else raw_h)
        if bx <= tx or by <= ty:
            self.log.warning(f"[preprocess] crop degenerate ({tx},{ty})-({bx},{by}) "
                             f"-> disabling crop")
            c.crop_enabled = False
            return None
        self._crop_rect = (tx, ty, bx, by)
        self.log.info(f"[preprocess] crop rect ({tx},{ty})-({bx},{by})")
        return self._crop_rect

    def _crop(self, bgr: np.ndarray, raw_w: int, raw_h: int) -> np.ndarray:
        if not self.cfg.crop_enabled:
            return bgr
        rect = self._resolve_crop_rect(raw_w, raw_h)
        if rect is None:
            return bgr
        x0, y0, x1, y1 = rect
        return bgr[y0:y1, x0:x1].copy()   # contiguous copy (cache-friendly downstream)

    def _flatfield(self, bgr: np.ndarray) -> np.ndarray:
        c = self.cfg
        if not (c.flatfield_enabled and c.flatfield_arr is not None):
            return bgr
        # READ-INTO-LOCAL discipline (same race as _lens): capture_flatfield/clear_flatfield (control
        # thread) null this cache; check + use must be one snapshot or cv2.subtract gets a None.
        ready = self._flatfield_ready
        if ready is None:
            arr = c.flatfield_arr
            if arr.shape[:2] != bgr.shape[:2]:
                # Also accept a flatfield sized to the PRE-crop (raw) frame: crop
                # it to the rect ONCE + cache, so the per-frame subtract runs on
                # the small post-crop buffer (no wasted CPU).
                if (c.crop_enabled and self._crop_rect is not None
                        and self._raw_hw is not None and arr.shape[:2] == self._raw_hw):
                    x0, y0, x1, y1 = self._crop_rect
                    arr = arr[y0:y1, x0:x1]
                    self.log.info(f"[preprocess] flatfield cropped to {arr.shape[:2]}")
                else:
                    self.log.warning(f"[preprocess] flatfield shape {arr.shape} != "
                                     f"frame {bgr.shape} -> disabling flatfield")
                    c.flatfield_enabled = False
                    return bgr
            ready = cv2.merge([arr, arr, arr]) if arr.ndim == 2 else arr
            self._flatfield_ready = ready
            self.log.info(f"[preprocess] flatfield ready shape={ready.shape}")
        return cv2.subtract(bgr, ready)  # saturating uint8 subtract

    def _gray(self, bgr: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    def _mask(self, gray: np.ndarray, bgr_shape: tuple[int, ...]) -> np.ndarray:
        c = self.cfg
        if not (c.mask_enabled and c.mask_radius > 0):
            return gray
        if self._mask_circle is None:
            # Translate the mask centre from raw coords into post-crop coords.
            if c.crop_enabled and self._crop_rect is not None:
                cx, cy = c.mask_cx - self._crop_rect[0], c.mask_cy - self._crop_rect[1]
            else:
                cx, cy = c.mask_cx, c.mask_cy
            self._mask_circle = np.zeros(bgr_shape[:2], dtype=np.uint8)
            cv2.circle(self._mask_circle, (int(cx), int(cy)), int(c.mask_radius),
                       255, thickness=-1)
            self.log.info(f"[preprocess] circular mask center=({cx},{cy}) "
                          f"r={c.mask_radius} on {bgr_shape[:2]}")
        return cv2.bitwise_and(gray, gray, mask=self._mask_circle)
