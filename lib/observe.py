"""``observe`` — blobs · association · the Observation record.

Per frame, ONCE:
    obj_fg = fg (gray > fg_threshold)     # bright objects on dark felt — NO dilation
    blobs  = findContours(RETR_EXTERNAL)  # one external contour per blob; NEVER connectedComponents
and per blob ONCE: moments/centroid, boundingRect, contourArea, the raster mask.

Each blob is **associated** to the object whose footprint (or, before capture, last
mask) it overlaps — association is NOT a fit: the registration lives only in pose (at
settle). Unmatched blobs mint a new object into SETTLING (the birth debounce). For every
object observe writes ONE immutable ``Observation`` read-only downstream (classify reads it):

    residual = area(mask − footprint)   # object px off the footprint → MOVED (in motion)
    missing  = area(footprint − mask)   # footprint px that should show, don't → HIDDEN

The **per-object contour-change gate** — the heart of partial update: observe reassigns an
object's live geometry ONLY when the mask changed ≥ ``k·area`` from its held reference
(frame-over-frame XOR). Sub-``k`` jitter holds the geometry verbatim (the camera-noise
deadband); a moving blob lights up XOR → it never settles. Each object gates independently,
so a quiet object updates while another is still being moved. ``k`` is the ONE threshold
scale (≈3× the ±1px contour jitter).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from lib.logger import NULL_LOGGER
from lib.params import ParamSpec, bind_params
from lib.reid import ReidParams, best_cosine
from lib.scene import Detection, Status

_FG_THRESHOLD = 40      # gray > this is foreground (THRESH_BINARY, fixed — not Otsu)
_MIN_AREA = 300         # min contour area (px²) — drops noise specks
_K = 0.05               # the ONE threshold scale: R = M = k·area, and the change-gate — one global k,
#                         ≈3× the thin-wire residual noise floor (~1.6% of area) so contour jitter never trips it.
_MOTION_THRESHOLD = 25  # frame-diff (|gray − prev_gray|) cut: a pixel changing more than this = "moving"
_MOTION_FRAC = 0.10     # ≥ this fraction of a region's px changed frame-over-frame ⇒ IN MOTION
_ASSOC_SIZE_TOL = 0.5   # reject an association when 1−min/max of blob-vs-object area exceeds this
_RELOC_SHAPE_TOL = 1.0  # max fold-change in elongation OR fill for a claimed blob to shape-match a footprint
#                         (1.0 ⇒ within 2×): rejects a gross mismatch (a solid rectangle can't claim a wire)


@dataclass(frozen=True, slots=True)
class Observation:
    """One frame's set-comparison for one object. Immutable; written by observe, read-only
    downstream. ``mask`` is the visible object mask, baked as the footprint at capture and
    used as next frame's stability reference."""

    residual: int                          # area(mask − footprint) — off_footprint (→ moved)
    missing: int                           # area(footprint − mask) — rest_hidden (→ hidden)
    xor_prev: int                          # frame-over-frame XOR area (per-object stationarity)
    centroid: tuple[float, float] | None
    area: float
    bbox: tuple[int, int, int, int] | None
    mask: np.ndarray | None            # the visible object mask (uint8 0/255)
    under_motion: bool = False         # a moving mass overlaps this object's rest region → classify HOLDS


def observation_terms(footprint: np.ndarray, mask: np.ndarray) -> tuple[int, int]:
    """The two spatial signals, as pure mask arithmetic — ``(residual, missing)``.

    All masks uint8 0/255, same shape. residual is object px outside the footprint (→ MOVED);
    missing is footprint px that aren't there (→ HIDDEN). (Camera noise → small residual < R; a
    shadow removes px → missing up, never residual up — so a shadow can't fake MOVED.)"""
    residual = int(cv2.countNonZero(cv2.bitwise_and(mask, cv2.bitwise_not(footprint))))
    missing = int(cv2.countNonZero(cv2.bitwise_and(footprint, cv2.bitwise_not(mask))))
    return residual, missing


def _bbox_intersect(a: tuple, b: tuple) -> tuple | None:
    """Intersection of two ``(x, y, w, h)`` bboxes, or None if disjoint — the mask overlap is 0
    outside it, so association counts only there (byte-identical, but skips disjoint pairs entirely)."""
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    return (ix0, iy0, ix1 - ix0, iy1 - iy0) if ix1 > ix0 and iy1 > iy0 else None


class Observe:
    """The observe stage: obj_fg → blobs → association → the Observation record."""

    name = "observe"

    # -- live parameters, declared WITH the stage. reassoc_dist = re-associate a lost object's blob within
    #    this of its held last centroid. edge_ignore_margin = the RIM gate, one knob for both boundaries:
    #    a NEW blob within this frac of a frame edge is NOT minted (an arm at a border, the felt rim) and a
    #    blob whose contour reaches within this frac of the circular [preprocess] mask radius is NOT minted
    #    (clipped by the mask → shape/pose unreliable, exactly like a frame-edge clip). Only births are
    #    gated; a live blob reaching the frame edge also flags the object `ignored`. 0 = off. --
    PARAMS = (
        ParamSpec("fg_threshold", int, _FG_THRESHOLD, lo=0, hi=255, doc="foreground/bg cut on the diff image"),
        ParamSpec("blur", float, 0.0, lo=0.0, hi=1.0,
                  doc="Camera-noise denoise for the DETECTION path only (0 = off ... 1 ~ 3 px Gaussian sigma): smooths the gray feeding the fg + motion masks; the frame, reid patches and textures stay sharp"),
        ParamSpec("min_area", int, _MIN_AREA, lo=1, doc="smallest blob area (px) kept as an object (>=1: the m00>0 invariant the centroid division relies on)"),
        ParamSpec("max_area", int, 0, lo=0, doc="reject MINTING a blob larger than this px^2 (0 = off) - keeps a stopped arm from becoming an object"),
        ParamSpec("k", float, _K, lo=0.0, hi=1.0, doc="change-gate: min contour change as a fraction of object area"),
        ParamSpec("reassoc_dist", float, 64.0, lo=0, doc="Max px from held centroid to re-associate a lost object's blob"),
        ParamSpec("relocate_dist", float, 160.0, lo=0, attr="_reloc_dist",
                  doc="THE set-down re-link REACH - max px a returning object may reach to re-link geometrically. ONE knob for BOTH surfaces: observe's coasting-husk relocation AND identity's gone-pool re-acquire (build single-sources it into identity)"),
        ParamSpec("contour_epsilon", float, 0.75, lo=0.0, attr="_contour_eps",
                  doc="THE contour-quality knob for every emitted outline (tuio2 /ocg, oscout /rmt/contour, webgui): approxPolyDP eps in px - 0 = full resolution. Internal consumers (EFD, pose, coverage) always use the full-resolution contour"),
        ParamSpec("reloc_shape_tol", float, _RELOC_SHAPE_TOL, lo=0.0, attr="_reloc_shape",
                  doc="Max fold-change in elongation OR fill for a claimed blob to shape-match the footprint (1.0 = within 2x; 0 = off/any shape)"),
        ParamSpec("motion_threshold", int, _MOTION_THRESHOLD, lo=0, hi=255,
                  doc="frame-diff (abs-diff gray) cut: a pixel changing more than this vs last frame is 'moving'"),
        ParamSpec("motion_frac", float, _MOTION_FRAC, lo=0.0, hi=1.0,
                  doc="fraction of a blob's px in the motion mask to call it in-motion (never associates / never mints)"),
        ParamSpec("assoc_size_tol", float, _ASSOC_SIZE_TOL, lo=0.0, hi=1.0,
                  doc="reject an association when 1-min/max of blob-vs-object area exceeds this (stops id migration onto a merged/foreign blob)"),
        ParamSpec("edge_ignore_margin", float, 0.0, lo=0.0, hi=1.0, attr="_edge_ignore",
                  doc="Rim ignore margin - ONE knob for both boundaries: reject a BIRTH whose bbox comes within this frac of a frame edge (and continuously flag such an object `ignored`), and reject a blob whose contour reaches within this frac of the circular [preprocess] mask radius (a clipped shape is unreliable). 0 = off"),
    )

    def __init__(self, config: Any = None, *, logger: Any = None, preprocess_cfg: Any = None,
                 embedder: Any = None, reid_params: Any = None) -> None:
        self._log = logger or NULL_LOGGER
        bind_params(self, config)      # _fg_threshold/_min_area/_max_area/_k/_reassoc_dist/_reloc_dist/_motion_*/_assoc_size_tol/_edge_ignore
        self._reidp = reid_params if reid_params is not None else ReidParams(None)
        #                              # (classify + identity read k / relocate_dist LIVE via the
        #                              # properties below — observe is the single source for both)
        #                              # the shared [reid] appearance policy (cosine gates, twin guard),
        #                              # read LIVE — one panel owns it; a bare construction gets defaults
        self._pp = preprocess_cfg      # the live PreprocessConfig — for the mask-boundary reject (mask is hot)
        self._embedder = embedder         # classical appearance embedder (or None) — appearance-assisted relocation
        self._prev_gray: "np.ndarray | None" = None   # previous frame's gray, for the per-blob motion signal (survives ticks)

    # -- the two single-sourced values, read LIVE by their second consumers (classify / identity) —
    #    a /rmt/set observe … reaches both stages mid-run instead of splitting at a build-time copy --
    @property
    def k(self) -> float:
        return self._k

    @property
    def relocate_dist(self) -> float:
        return self._reloc_dist

    def _embed_blob(self, ctx: Any, blob: dict) -> "np.ndarray | None":
        """Appearance fingerprint of a blob's clean crop (BGR bbox masked) — for matching a settled blob to a
        HIDDEN object's accumulated prints. Cached on the blob dict (embedded at most once per blob)."""
        if self._embedder is None or getattr(ctx, "bgr", None) is None:
            return None
        if "embed" in blob:
            return blob["embed"]
        x, y, w, h = blob["bbox"]
        patch = ctx.bgr[max(0, y):y + h, max(0, x):x + w]
        e = self._embedder.embed(patch, blob["mask"][max(0, y):y + h, max(0, x):x + w])
        blob["embed"] = e
        return e

    # -- pixels → blobs ----------------------------------------------------- #

    def _detection_gray(self, gray: np.ndarray) -> np.ndarray:
        """The gray the DETECTION masks see: optionally Gaussian-denoised (``blur`` 0..1 → σ = 3·blur px).
        Camera noise bites exactly here — a speck flickering past ``fg_threshold``, and DOUBLED sensor
        noise in the ``|gray − prev|`` motion diff (two noisy frames) keeping a region "unquiet" so it
        never settles. The blur never touches ``ctx.gray`` itself: reid patches, captures and the views
        read the original, so appearance/texture stays sharp. 0 = off (returns gray uncopied)."""
        if self._blur <= 0.0:
            return gray
        return cv2.GaussianBlur(gray, (0, 0), 3.0 * float(self._blur))

    def _obj_fg(self, gray: np.ndarray) -> np.ndarray:
        """The object foreground: ``gray > fg_threshold`` — bright objects on dark felt, NO dilation."""
        _, fg = cv2.threshold(gray, self._fg_threshold, 255, cv2.THRESH_BINARY)
        return fg

    def _motion_mask(self, gray: np.ndarray) -> "np.ndarray | None":
        """Full-frame frame-diff mask (``|gray − prev| > motion_threshold``) — the per-blob motion signal.
        ``None`` on the first frame and after a live crop-size change → everything reads STABLE that tick
        (births still mint). Stores a COPY of gray as next frame's reference (defensive: a future buffer
        reuse in preprocess must not silently alias prev==current → an all-zero motion mask)."""
        prev = self._prev_gray
        self._prev_gray = gray.copy()
        if prev is None or prev.shape != gray.shape:
            return None
        _, motion = cv2.threshold(cv2.absdiff(gray, prev), self._motion_threshold, 255, cv2.THRESH_BINARY)
        return motion

    def _tag_motion(self, blobs: list[dict], motion: "np.ndarray | None") -> None:
        """Tag each blob ``in_motion = (moving px / area) > motion_frac``, counted in the blob's own bbox
        ROI. ``motion`` None (first frame) → all stable. A moving blob is 'motion': it never associates
        (never hijacks an id) and never mints — the decisive arm defeat, upstream of association."""
        for b in blobs:
            if motion is None:
                b["in_motion"] = False
                continue
            x, y, w, h = b["bbox"]
            moving = int(cv2.countNonZero(cv2.bitwise_and(
                motion[y:y + h, x:x + w], b["mask"][y:y + h, x:x + w])))
            b["in_motion"] = (moving / b["area"]) > self._motion_frac

    def _area_ok(self, a: "float | None", b: "float | None") -> bool:
        """Area-similarity gate: the smaller area must be ≥ (1 − assoc_size_tol) of the larger, else
        it's a foreign / merged blob (an arm-sized mass vs a small object) → never let its id migrate.
        Unknown size (pre-capture) → don't block."""
        if not a or not b:
            return True
        lo, hi = (a, b) if a < b else (b, a)
        return (1.0 - lo / hi) <= self._assoc_size_tol

    def _area_gate(self, obj: Any, blob_area: "float | None") -> bool:
        """Relocation-claim area gate: the blob passes if it matches the object's footprint area OR its
        CANON (true, largest-clean) area. A settle that captured a half-occluded footprint (baked at ~½
        the true size) sits its area gate right at the corruption ratio, rejecting the object's OWN full
        husk on return → a fresh mint. Testing canon too heals that (union → can only ADD matches)."""
        if self._area_ok(obj.footprint_area or obj.area, blob_area):
            return True
        ca = getattr(obj, "canon_area", None)
        return ca is not None and self._area_ok(ca, blob_area)

    @staticmethod
    def _shape_desc(contour: np.ndarray) -> "tuple[float, float]":
        """Two numerically-STABLE, rotation-invariant shape descriptors from the oriented box:
        elongation (long/short side, ≥1) and fill (area ÷ oriented-box area, 0..1). Unlike Hu-moment
        ``matchShapes`` — whose 1÷log(moment) form explodes for small compact silhouettes, scoring an
        object AGAINST ITSELF as a gross mismatch (blocking legitimate re-acquisition) — these stay
        well-behaved for tiny blobs, so a sparse/wiry object (elongated + low-fill) and a solid
        rectangle (compact + high-fill) separate cleanly."""
        (_, _), (w, h), _ = cv2.minAreaRect(contour)
        lo, hi = (w, h) if w < h else (h, w)
        elong = hi / lo if lo > 1e-3 else 1.0
        box = w * h
        fill = float(cv2.contourArea(contour)) / box if box > 1e-3 else 0.0
        return elong, fill

    def _shape_ok(self, obj: Any, blob: dict) -> bool:
        """Shape-similarity gate for a CLAIM: the blob's silhouette must match the object's footprint in
        BOTH elongation and fill, each within a factor of ``1 + reloc_shape_tol`` — so a solid rectangle
        can't claim a sparse wire, and vice-versa. Off (True) when reloc_shape_tol is 0 or a contour is
        missing. Stable for small blobs (see ``_shape_desc`` for why these descriptors, not Hu moments)."""
        return self.shape_claim_ok(obj.footprint_contour, blob.get("contour"))

    def shape_claim_ok(self, ref_contour: Any, claim_contour: Any) -> bool:
        """PUBLIC form of the claim shape gate, on raw contours — shared by observe's relocation pass
        (via ``_shape_ok``) and identity's geometric pool re-acquire, the two halves of the ONE
        set-down reach: the same reach must apply the same gates on both surfaces, or the pooled-vs-husk
        timing (early-GONE) silently decides whether a different-shape claim is allowed."""
        if self._reloc_shape <= 0.0:
            return True
        if ref_contour is None or claim_contour is None or len(ref_contour) < 3 or len(claim_contour) < 3:
            return True
        try:
            fe, ff = self._shape_desc(ref_contour)
            be, bf = self._shape_desc(claim_contour)
        except cv2.error:
            return True
        f = 1.0 + self._reloc_shape                       # max fold-change allowed either way
        for a, b in ((fe, be), (ff, bf)):                 # elongation, then fill
            lo, hi = (a, b) if a < b else (b, a)
            if hi > lo * f:
                return False
        return True

    def _appearance_ok(self, ctx: Any, obj: Any, blob: dict) -> bool:
        """Appearance VETO for a GEOMETRY-based claim (proximity / relocation): reject when the blob's
        reid embedding is clearly DIFFERENT from the object's accumulated fingerprints (best cosine below
        ``reloc_cosine_reject``). reid reliably separates DISSIMILAR objects (a stone from a gear), so this
        stops a place-swap handing one's id to the other; look-alikes (reid's weak spot) still swap — the
        accepted trade. Off (True) when the veto is 0, reid is absent, or the object has no prints yet."""
        if self._reidp.reloc_cosine_reject <= 0.0 or self._embedder is None or not obj.embeds or ctx is None:
            return True
        q = self._embed_blob(ctx, blob)
        if q is None:
            return True
        return best_cosine(q, obj.embeds) >= self._reidp.reloc_cosine_reject

    def _under_motion(self, obj: Any, motion: "np.ndarray | None") -> bool:
        """True if a moving mass overlaps the object's REST region (footprint if captured, else last
        mask) beyond motion_frac — the HOLD signal classify reads to keep a covered RESTING object
        pinned instead of dropping it to HIDDEN (a HIDDEN round-trip would re-capture = wobble)."""
        if motion is None:
            return False
        ref = obj.footprint
        if ref is not None:
            ref_bbox = (cv2.boundingRect(obj.footprint_contour)
                        if obj.footprint_contour is not None else cv2.boundingRect(ref))
            ref_area = float(obj.footprint_area or 0)
        else:                                          # pre-capture: fall back to the last mask
            oobs = obj.observation
            ref = oobs.mask if oobs is not None else None
            ref_bbox = oobs.bbox if oobs is not None else None
            ref_area = 0.0
        if ref is None or ref_bbox is None:
            return False
        x, y, w, h = ref_bbox
        ref_roi = ref[y:y + h, x:x + w]
        if ref_area <= 0:
            ref_area = float(cv2.countNonZero(ref_roi))
        if ref_area <= 0:
            return False
        moving = int(cv2.countNonZero(cv2.bitwise_and(motion[y:y + h, x:x + w], ref_roi)))
        return (moving / ref_area) > self._motion_frac

    def _blobs(self, obj_fg: np.ndarray) -> list[dict]:
        """``findContours(RETR_EXTERNAL)`` → one blob per external contour, computed ONCE:
        centroid (moments), bbox, area, contour, and the mask (filled contour AND
        obj_fg). Sorted row-major ``(round(cy),
        round(cx))`` so mint order + byte-repro are contour-order-independent."""
        cnts, _ = cv2.findContours(obj_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blobs: list[dict] = []
        for c in cnts:
            m = cv2.moments(c)
            area = m["m00"]                              # == contourArea (same shoelace) — one C-call,
            if area < self._min_area:                    #   and m00 <= 0 is subsumed (min_area > 0)
                continue
            x, y, w, h = cv2.boundingRect(c)
            mask = np.zeros(obj_fg.shape, np.uint8)          # full-frame (downstream reads frame coords)
            cv2.drawContours(mask, [c], -1, 255, cv2.FILLED)   # fills only the contour bbox
            roi = mask[y:y + h, x:x + w]                     # intersect with obj_fg inside the bbox
            cv2.bitwise_and(roi, obj_fg[y:y + h, x:x + w], dst=roi)   # only (0 outside the bbox anyway)
            blobs.append({
                "centroid": (m["m10"] / m["m00"], m["m01"] / m["m00"]),
                "bbox": (int(x), int(y), int(w), int(h)), "area": float(area),
                # "px": the RASTERIZED pixel count — NOT interchangeable with "area", which is the
                # shoelace (m00) area of the polygon. Filling a contour lights every pixel the
                # boundary touches, so px > area always, and by 2x for a ~1-px-wide sliver. Any
                # formula mixing a countNonZero overlap with a shoelace area is unit-broken (see
                # _overlap_pass). Free here: the bbox roi is already materialized.
                "px": int(cv2.countNonZero(roi)),
                "contour": c, "mask": mask})
        blobs.sort(key=lambda b: (round(b["centroid"][1]), round(b["centroid"][0])))
        return blobs

    # -- association (NOT a fit) --------------------------------------------- #

    def _associate(self, objs: tuple, blobs: list[dict], ctx: Any = None
                   ) -> tuple[dict[int, dict | None], list[dict]]:
        """Match each object to the free blob it most overlaps, ranked by CONTOUR **IoU** (not raw
        pixels — a huge blob can't win a small footprint by sheer count) and gated by AREA similarity
        (``_area_ok`` — the arm 53k can never take the stone's 2.9k id). Reference = the object's
        last-frame mask (continuity as it moves) or its footprint. A **moving** blob (in_motion) is
        never a match target and never mints — it is 'motion'. Returns ``{oid: blob|None}`` + the list
        of unmatched STABLE blobs (→ mint).

        Four passes, strongest signal first, each claiming from the still-unmatched pool
        (``match``/``used_o``/``used_b`` accumulate across them):
        overlap IoU → proximity fallback → appearance relocation → geometric relocation."""
        match: dict[int, dict | None] = {}
        used_o: set[int] = set()
        used_b: set[int] = set()
        self._overlap_pass(objs, blobs, match, used_o, used_b)
        self._proximity_pass(objs, blobs, ctx, match, used_o, used_b)
        self._appearance_pass(objs, blobs, ctx, match, used_o, used_b)
        self._relocation_pass(objs, blobs, ctx, match, used_o, used_b)
        for obj in objs:
            match.setdefault(obj.id, None)               # unseen this frame
        unmatched = [blobs[bi] for bi in range(len(blobs))
                     if bi not in used_b and not blobs[bi].get("in_motion")]
        return match, unmatched

    def _overlap_pass(self, objs: tuple, blobs: list[dict], match: dict,
                      used_o: set[int], used_b: set[int]) -> None:
        """Pass 1 — the contour-IoU overlap match (see ``_associate`` for the ranking + gates)."""
        # cache each object's (ref mask, ref bbox, ref area, shape-gate?) ONCE. use_shape = the object is
        # re-associating against its STABLE captured FOOTPRINT (it was unseen last frame — HIDDEN/covered),
        # so the claimed blob must SHAPE-match the footprint: this stops a different-shaped blob (a wire /
        # a rectangle settling over the spot) from hijacking the id. Continuity (last-mask) is NOT gated —
        # a moving/partial object's shape varies frame-to-frame.
        refs: list[tuple[int, Any, tuple, float, bool]] = []
        for oi, obj in enumerate(objs):
            oobs = obj.observation
            mask = oobs.mask if oobs is not None else None
            if mask is not None and oobs.area and oobs.area > 0:
                ref, ref_bbox, ref_area, use_shape = mask, oobs.bbox, float(oobs.area), False
            elif obj.footprint is not None:
                ref = obj.footprint
                ref_bbox = (cv2.boundingRect(obj.footprint_contour)
                            if obj.footprint_contour is not None else cv2.boundingRect(ref))
                ref_area = float(obj.footprint_area or cv2.countNonZero(ref))
                use_shape = obj.footprint_contour is not None
            else:
                continue
            if ref_bbox is None or ref_area <= 0:
                continue
            # the reference's RASTERIZED pixel count, for the IoU below — ref_area is a shoelace
            # area on the oobs path (blob["area"] = m00) and a pixel count on the footprint path,
            # so it cannot be used against a countNonZero overlap. Counted in the bbox roi.
            rx, ry, rw, rh = ref_bbox
            ref_px = int(cv2.countNonZero(ref[ry:ry + rh, rx:rx + rw])) if rw > 0 and rh > 0 else 0
            if ref_px <= 0:
                continue
            refs.append((oi, ref, ref_bbox, ref_area, ref_px, use_shape))
        pairs: list[tuple[float, int, int]] = []
        for oi, ref, ref_bbox, ref_area, ref_px, use_shape in refs:
            for bi, blob in enumerate(blobs):
                if blob.get("in_motion"):                        # a moving mass is never a match target
                    continue
                if ref.shape[:2] != blob["mask"].shape[:2]:
                    continue
                if not self._area_ok(ref_area, blob["area"]):    # SIZE GATE — the decisive arm reject
                    continue
                if use_shape and not self._shape_ok(objs[oi], blob):   # SHAPE GATE — no cross-object hijack
                    continue
                inter = _bbox_intersect(ref_bbox, blob["bbox"])   # disjoint bboxes → 0 overlap, skip
                if inter is None:
                    continue
                ix, iy, iw, ih = inter                            # overlap is 0 outside → count in the ROI
                ov = int(cv2.countNonZero(cv2.bitwise_and(ref[iy:iy + ih, ix:ix + iw],
                                                          blob["mask"][iy:iy + ih, ix:ix + iw])))
                if ov <= 0:
                    continue
                # IoU entirely in PIXEL space (ref_px / blob["px"] / ov all countNonZero), so the
                # denominator is >= max(ref_px, blob_px) >= ov > 0 — it cannot vanish. Mixing the
                # shoelace areas in here made it 2A-ov, which goes NEGATIVE for a ~1-px-wide sliver
                # (a wire, a rim) and divides by zero at exactly 2x.
                iou = ov / float(ref_px + blob["px"] - ov)       # normalized: a big blob can't win by px
                pairs.append((iou, oi, bi))
        pairs.sort(reverse=True)
        for _iou, oi, bi in pairs:
            if oi in used_o or bi in used_b:
                continue
            match[objs[oi].id] = blobs[bi]
            used_o.add(oi)
            used_b.add(bi)

    def _proximity_pass(self, objs: tuple, blobs: list[dict], ctx: Any, match: dict,
                        used_o: set[int], used_b: set[int]) -> None:
        """Pass 2 — PROXIMITY fallback: a lost object (no overlap match: its last mask went empty,
        e.g. briefly occluded mid-drag) whose HELD last-known centroid is near an unmatched STABLE
        blob re-associates (size + shape gated): the object reappeared, so it keeps its id instead
        of re-minting. The SHAPE gate here is what stops a captured object claiming a
        different-shaped blob that merely settled nearby."""
        prox: list[tuple[float, int, int]] = []
        for oi, obj in enumerate(objs):
            if oi in used_o or obj.centroid is None:
                continue
            for bi, blob in enumerate(blobs):
                if bi in used_b or blob.get("in_motion"):
                    continue
                if not self._area_gate(obj, blob["area"]):   # footprint OR canon (heals half-capture reject)
                    continue
                if obj.footprint_contour is not None and not self._shape_ok(obj, blob):
                    continue                                 # captured shape must match — no cross-object grab
                if not self._appearance_ok(ctx, obj, blob):  # reid veto — no dissimilar place-swap
                    continue
                bc = blob["centroid"]
                d = ((bc[0] - obj.centroid[0]) ** 2 + (bc[1] - obj.centroid[1]) ** 2) ** 0.5
                if d < self._reassoc_dist:
                    prox.append((d, oi, bi))
        prox.sort()
        for _d, oi, bi in prox:
            if oi in used_o or bi in used_b:
                continue
            match[objs[oi].id] = blobs[bi]
            used_o.add(oi)
            used_b.add(bi)

    def _appearance_pass(self, objs: tuple, blobs: list[dict], ctx: Any, match: dict,
                         used_o: set[int], used_b: set[int]) -> None:
        """Pass 3 — APPEARANCE RELOCATION (reid, optional): match a SETTLED blob to a HIDDEN object
        by its accumulated multi-view fingerprints (best cosine), at ANY distance. This is the
        ADDITIONAL signal: it disambiguates crossing paths and catches a FAR set-down the geometric
        pass can't. Candidates are HIDDEN only (never a settled/RESTING object). A confident,
        uniquely-best match claims the blob → the object relocates (HIDDEN→MOVED→…→RESTING, snap)
        keeping its id + prints."""
        if self._embedder is not None and ctx is not None:
            hidden = [(oi, obj) for oi, obj in enumerate(objs)
                      if oi not in used_o and obj.status is Status.HIDDEN and obj.embeds]
            if hidden:
                app: list[tuple[float, int, int]] = []
                for bi, blob in enumerate(blobs):
                    if bi in used_b or blob.get("in_motion"):
                        continue
                    cands = [(oi, obj) for oi, obj in hidden
                             if self._area_gate(obj, blob["area"])
                             and self._shape_ok(obj, blob)]
                    if not cands:
                        continue
                    q = self._embed_blob(ctx, blob)
                    if q is None:
                        continue
                    scored = sorted(((best_cosine(q, obj.embeds), oi) for oi, obj in cands), reverse=True)
                    top = scored[0]
                    if top[0] >= self._reidp.cosine_min and \
                            (len(scored) == 1 or top[0] - scored[1][0] >= self._reidp.uniqueness_margin):
                        app.append((top[0], top[1], bi))   # (cosine, object index, blob index)
                app.sort(reverse=True)                     # strongest appearance match wins first (greedy)
                for _cos, oi, bi in app:
                    if oi in used_o or bi in used_b:
                        continue
                    match[objs[oi].id] = blobs[bi]
                    used_o.add(oi)
                    used_b.add(bi)

    def _relocation_pass(self, objs: tuple, blobs: list[dict], ctx: Any, match: dict,
                         used_o: set[int], used_b: set[int]) -> None:
        """Pass 4 — GEOMETRIC RELOCATION (stay-then-snap): a HIDDEN object stayed drawn at its old
        rest spot through the carry (nothing claimed it while its blob was in motion). Now a SETTLED
        blob has appeared within relocate_dist of the object's last-known centroid + of similar size
        → the object CLAIMS it and relocates (HIDDEN→MOVED→…→RESTING, snapping to the new spot at
        capture). This is the long-range bridge (>> reassoc_dist, a deliberate move) that keeps the
        SAME object — no new mint, no gap."""
        reloc: list[tuple[float, int, int]] = []
        for oi, obj in enumerate(objs):
            if oi in used_o or obj.status is not Status.HIDDEN:
                continue
            oc = obj.centroid or obj.footprint_centroid
            if oc is None:
                continue
            for bi, blob in enumerate(blobs):
                if bi in used_b or blob.get("in_motion") or not self._area_gate(obj, blob["area"]):
                    continue                             # footprint OR canon (heals half-capture reject)
                if not self._shape_ok(obj, blob):        # a small blob can't claim a wire / a rectangle
                    continue
                if not self._appearance_ok(ctx, obj, blob):   # reid veto — no dissimilar place-swap
                    continue
                bc = blob["centroid"]
                d = ((bc[0] - oc[0]) ** 2 + (bc[1] - oc[1]) ** 2) ** 0.5
                if d < self._reloc_dist:
                    reloc.append((d, oi, bi))
        reloc.sort()                                     # nearest shape/size match claims the drop first
        for _d, oi, bi in reloc:
            if oi in used_o or bi in used_b:
                continue
            match[objs[oi].id] = blobs[bi]
            used_o.add(oi)
            used_b.add(bi)

    # -- the stage ---------------------------------------------------------- #

    def __call__(self, ctx: Any, scene: Any) -> None:
        det = self._detection_gray(ctx.gray)          # optional denoise for the MASKS only — ctx.gray
        obj_fg = self._obj_fg(det)                    #   itself stays sharp (reid patches / textures)
        ctx.obj_fg = obj_fg
        scene.set_obj_fg(obj_fg)
        motion = self._motion_mask(det)               # full-frame frame-diff (None on frame 1 / after resize)
        scene.set_contour_epsilon(self._contour_eps)  # publish the wire/UI contour-quality knob (live)
        ctx.motion = motion                           # identity's early-GONE reads the spot + table activity
        ctx.motion_frac = (cv2.countNonZero(motion) / motion.size) if motion is not None else 1.0
        blobs = self._blobs(obj_fg)
        self._tag_motion(blobs, motion)               # per-blob in_motion (a moving mass is 'motion')
        objs = scene.objects()
        match, unmatched = self._associate(objs, blobs, ctx)   # skips in_motion blobs (never match / never mint)
        for obj in objs:
            self._observe_object(ctx, scene, obj, match.get(obj.id), obj_fg.shape, motion)
        for blob in unmatched:                        # unmatched is stable-only → moving blobs never mint
            # a border birth (arm entering) OR a blob clipped by the circular mask OR an oversize mass
            # (a stopped arm, > max_area) → drop, don't mint
            if self._at_edge(blob["bbox"], obj_fg.shape) or self._at_mask_boundary(blob["contour"]):
                continue
            if self._max_area and blob["area"] > self._max_area:
                continue
            self._mint(scene, blob, ctx.frame_idx)

    def _at_edge(self, bbox: tuple, shape: tuple) -> bool:
        """True if ``bbox`` comes within ``edge_ignore_margin`` of any frame border (off when 0)."""
        if self._edge_ignore <= 0.0:
            return False
        x, y, w, h = bbox
        fh, fw = shape[:2]
        mx, my = self._edge_ignore * fw, self._edge_ignore * fh
        return x < mx or y < my or (x + w) > (fw - mx) or (y + h) > (fh - my)

    def _at_mask_boundary(self, contour: np.ndarray) -> bool:
        """True if the blob's contour reaches within ``edge_ignore_margin`` of the circular [preprocess] mask
        boundary — i.e. the mask CLIPS it (its shape/pose would be unreliable), the circular analogue of
        ``_at_edge``. Off when the margin is 0 / no preprocess cfg / mask disabled. Reads the live cfg."""
        if self._edge_ignore <= 0.0 or self._pp is None:
            return False
        pp = self._pp
        if not getattr(pp, "mask_enabled", False) or pp.mask_radius <= 0:
            return False
        # mask centre in POST-crop coords (preprocess draws the circle offset by the crop origin)
        cx = pp.mask_cx - (pp.crop_top_x if pp.crop_enabled else 0)
        cy = pp.mask_cy - (pp.crop_top_y if pp.crop_enabled else 0)
        pts = contour.reshape(-1, 2)
        d2 = (pts[:, 0].astype(np.float64) - cx) ** 2 + (pts[:, 1].astype(np.float64) - cy) ** 2
        lim = (pp.mask_radius * (1.0 - self._edge_ignore)) ** 2
        return bool(d2.max() > lim)

    def _union_roi(self, obj: Any, blob_bbox: tuple | None, prev_bbox: tuple | None,
                   shape: tuple, margin: int = 2) -> tuple | None:
        """The bbox covering the footprint + this-frame mask + prev mask (from their stored
        bboxes — NO full-frame scan), padded and clamped. The residual/missing/xor arithmetic
        is zero outside it, so cropping every mask to this ROI is byte-identical at a tiny
        fraction of the pixels."""
        boxes = []
        if obj.footprint_contour is not None:
            boxes.append(cv2.boundingRect(obj.footprint_contour))
        if blob_bbox is not None:
            boxes.append(blob_bbox)
        if prev_bbox is not None:
            boxes.append(prev_bbox)
        if not boxes:
            return None
        x0 = min(b[0] for b in boxes); y0 = min(b[1] for b in boxes)
        x1 = max(b[0] + b[2] for b in boxes); y1 = max(b[1] + b[3] for b in boxes)
        h, w = shape
        x0, y0 = max(0, x0 - margin), max(0, y0 - margin)
        x1, y1 = min(w, x1 + margin), min(h, y1 + margin)
        return (x0, y0, x1 - x0, y1 - y0) if x1 > x0 and y1 > y0 else None

    def _observe_object(self, ctx: Any, scene: Any, obj: Any, blob: dict | None,
                        shape: tuple, motion: "np.ndarray | None") -> None:
        """Compute + write one object's Observation, applying the change-gate to the live
        geometry (reassign only on a ≥ k·area change; hold otherwise). The set-diff arithmetic runs
        cropped to the object's union ROI — byte-identical, far fewer pixels."""
        if blob is not None:
            mask = blob["mask"]
            centroid, area, bbox = blob["centroid"], blob["area"], blob["bbox"]
        else:                                            # unseen this frame → empty mask
            mask = None
            centroid, area, bbox = None, 0.0, None

        prev = obj.observation.mask if obj.observation is not None else None
        prev_bbox = obj.observation.bbox if obj.observation is not None else None
        roi = self._union_roi(obj, bbox, prev_bbox, shape)
        if roi is not None:                              # crop every mask to the object's ROI
            rx, ry, rw, rh = roi
            fp_r = obj.footprint[ry:ry + rh, rx:rx + rw] if obj.footprint is not None else None
            mask_r = (mask[ry:ry + rh, rx:rx + rw] if mask is not None
                      else np.zeros((rh, rw), np.uint8))
            prev_r = prev[ry:ry + rh, rx:rx + rw] if prev is not None else np.zeros((rh, rw), np.uint8)
        else:                                            # no geometry at all → full-frame fallback
            fp_r = obj.footprint
            mask_r = mask if mask is not None else np.zeros(shape, np.uint8)
            prev_r = prev if prev is not None else np.zeros(shape, np.uint8)

        if fp_r is not None:
            residual, missing = observation_terms(fp_r, mask_r)
        else:                                            # pre-capture (SETTLING): no footprint yet
            residual, missing = 0, 0
        xor_prev = int(cv2.countNonZero(cv2.bitwise_xor(mask_r, prev_r)))

        # the per-object contour-change gate: reassign the live geometry ONLY on a real change
        # (mask moved ≥ k·area from the held reference); sub-k jitter holds it verbatim.
        ref_area = obj.footprint_area or obj.area or max(1.0, area)
        if blob is not None and xor_prev >= self._k * ref_area:
            scene.match(Detection(centroid=centroid, bbox=bbox, area=area,
                                  contour=blob["contour"]), obj.id)
        obs = Observation(residual=residual, missing=missing,
                          xor_prev=xor_prev, centroid=centroid, area=area, bbox=bbox,
                          mask=mask, under_motion=self._under_motion(obj, motion))
        scene.set(obj.id, "observation", obs)
        # CONTINUOUS EDGE GATE: an object whose LIVE blob reaches the frame border is (part of) an arm
        # reaching in from off-frame — a real object sits interior. Flag it `ignored` (excluded from every
        # output + overlay) while edge-connected; a no-blob (HIDDEN) object keeps its last flag, so a hand
        # that minted mid-table and then extended to the edge stays out. `_at_edge` extends the birth-only
        # edge_ignore_margin to every frame. Objects placed AT the edge are dropped too — the accepted trade.
        if blob is not None:
            edge = self._at_edge(bbox, shape)
            if edge != obj.ignored:
                scene.set(obj.id, "ignored", edge)

    def _mint(self, scene: Any, blob: dict, frame_idx: int) -> None:
        """A blob matching no object → a new SETTLING object (the birth debounce). Seeds an
        initial Observation (mask, zero set-diffs) so next frame associates by continuity."""
        oid = scene.mint(Detection(centroid=blob["centroid"], bbox=blob["bbox"],
                                   area=blob["area"], contour=blob["contour"]))
        obs = Observation(residual=0, missing=0,
                          xor_prev=int(cv2.countNonZero(blob["mask"])),
                          centroid=blob["centroid"], area=blob["area"], bbox=blob["bbox"],
                          mask=blob["mask"])
        scene.set(oid, "observation", obs)
