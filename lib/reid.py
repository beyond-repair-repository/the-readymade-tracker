"""``reid`` — the classical appearance descriptor (color + shape + size), the re-id tie-break.

No neural net, no onnxruntime, no model file. Each clean settled crop → a 21-d feature vector
(canonical rotation → contour mask → saturation hist + Hu-moments + shape(ellipse/circle variance,
defect depth) + size), z-scored by a fixed calibration + L2-normalized so cosine == dot. The MATCHING lives in the
``identity`` stage: appearance is a **closed-set tie-break** over the small gone-pool, not the core
matcher (geometry is primary) — it catches a FAR / rotated set-down the geometric gate misses.

A leakage-free re-id study (settle-based crops; in the development repository) picked this basis:
rank-1 0.789 / look-alike EER 0.059 — it EDGES a wider 47-d HSV+shape vector (0.771 / 0.085) and
lands within ~2 pts of neural embedders (osnet / mobilenet / DINOv2) at 0.5 ms/crop (~11-600×
cheaper), and — being nearly free — is the only signal cheap enough to also run during motion.
Because the tracker is settle-based (full, clean crops, no occlusion) the illumination/occlusion
invariance a neural embedding buys isn't needed. Hue is dropped (unreliable across the tabletop
lighting for the look-alikes).
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from lib.features import _feat_input, registry
from lib.logger import NULL_LOGGER
from lib.params import ParamSpec, bind_params

_REG = registry()      # the shared shape-feature registry (ellipse_variance / circle_variance / defect_depth)


def _canonical(patch: np.ndarray, mask: "np.ndarray | None"):
    """Rotate ``patch`` (+ ``mask``) so the object's MAJOR axis is horizontal — a portrait and a
    landscape crop of one object converge to the same view (rotation-invariant descriptor)."""
    m = (np.asarray(mask) > 0) if mask is not None else np.asarray(patch).any(axis=2)
    ys, xs = np.nonzero(m)
    if xs.size < 10:
        return patch, mask
    (_, _), (w, h), ang = cv2.minAreaRect(np.column_stack([xs, ys]).astype(np.float32))
    if w < h:
        ang += 90.0
    H, W = patch.shape[:2]
    M = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), ang, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nW, nH = int(H * sin + W * cos), int(H * cos + W * sin)
    M[0, 2] += nW / 2.0 - W / 2.0
    M[1, 2] += nH / 2.0 - H / 2.0
    pc = cv2.warpAffine(patch, M, (nW, nH), flags=cv2.INTER_LINEAR)
    mc = (cv2.warpAffine(np.asarray(mask), M, (nW, nH), flags=cv2.INTER_NEAREST)
          if mask is not None else None)
    return pc, mc


# The fixed calibration (z-score MU/SD for the 21-d vector) — a DEFAULT fit on a 721-crop settle dataset
# that generalises across scenes; re-fit with the development repository's re-id study utilities
# if the object palette shifts. It gives each feature ~unit contribution so
# cosine isn't dominated by the largest-scale block.
_CLASSIC_MU = np.array([0.0401, 0.4155, 0.3915, 0.0863, 0.0324, 0.0112, 0.0087, 0.0143, 3.1451, 8.3438, 11.593, 13.6427, 1.9646, 4.5546, -17.1506, 0.0716, 0.0968, 0.0967, 7.6403, 5.2518, 1.1913], np.float32)
_CLASSIC_SD = np.array([0.0858, 0.3603, 0.3257, 0.1695, 0.1294, 0.0572, 0.0521, 0.0856, 0.1483, 0.9639, 1.0473, 1.4106, 26.4341, 17.7476, 19.9857, 0.05, 0.0687, 0.1278, 0.6011, 0.3516, 0.3662], np.float32)


def _classic_features(patch_bgr: "np.ndarray | None", mask: "np.ndarray | None") -> "np.ndarray | None":
    """The raw 21-d appearance vector from a canonically-rotated, contour-masked crop:
    saturation hist(8) · Hu-moments(7, log) · shape(ellipse_variance, circle_variance, defect_depth) ·
    size(log-area, log-perimeter, w/h). Colour is saturation-ONLY — hue is unreliable across the
    tabletop lighting for the look-alikes (see the module docstring for the study numbers). All blocks
    are rotation-invariant (or canonicalized). Returns None for a degenerate/empty crop."""
    if patch_bgr is None or mask is None:
        return None
    p, mc = _canonical(np.asarray(patch_bgr), np.asarray(mask))
    if mc is None or cv2.countNonZero(mc) < 30:
        return None
    cnts, _ = cv2.findContours(mc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 20 or len(c) < 5:
        return None
    hsv = cv2.cvtColor(p, cv2.COLOR_BGR2HSV)
    sat = cv2.calcHist([hsv], [1], mc, [8], [0, 256]).flatten()
    ssum = float(sat.sum())
    sat = sat / ssum if ssum else sat
    hu = cv2.HuMoments(cv2.moments(mc)).flatten()
    hu = np.array([-np.sign(v) * np.log10(abs(v) + 1e-30) for v in hu], np.float32)  # log-scale (huge range)
    x = _feat_input(c, None)                       # ellipse/circle_variance + defect_depth read contour only
    shape = np.array([_REG[nm].fn(x)[0] for nm in ("ellipse_variance", "circle_variance", "defect_depth")],
                     np.float32)
    area, per = cv2.contourArea(c), cv2.arcLength(c, True)
    (_, _), (w, h), _ = cv2.minAreaRect(c)
    w, h = max(w, h), (min(w, h) or 1.0)
    size = np.array([np.log(area + 1), np.log(per + 1), w / h], np.float32)
    return np.concatenate([sat, hu, shape, size]).astype(np.float32)


class ClassicalEmbedder:
    """A neural-free appearance embedder — the 21-d saturation+shape+size vector, z-scored by the fixed
    calibration + L2-normalized so cosine (``best_cosine``) works as a similarity. ``embed(patch, mask)
    -> (21,)`` or None. No onnxruntime, no model file; canonical rotation is built in."""

    canonical = True                                  # canonicalization lives inside _classic_features

    def __init__(self, logger: Any = None) -> None:
        self._log = logger or NULL_LOGGER
        if logger is not None:
            logger.info("[reid] classical appearance descriptor (saturation+shape+size, 21-d, no onnxruntime)")

    def embed(self, patch_bgr: np.ndarray, mask: "np.ndarray | None" = None) -> "np.ndarray | None":
        f = _classic_features(patch_bgr, mask)
        if f is None:
            return None
        z = (f - _CLASSIC_MU) / _CLASSIC_SD           # fixed calibration → unit contribution per feature
        n = float(np.linalg.norm(z))
        return (z / n).astype(np.float32) if n >= 1e-8 else None


def best_cosine(query: "np.ndarray | None", prints: Any) -> float:
    """Best cosine of ``query`` against a set of L2-normalized fingerprints (a tuple/list, or a single
    vector) — the MAX similarity over the object's accumulated multi-view prints. 0.0 if either is empty.
    Cosine == dot product since both sides are L2-normalized."""
    if query is None or prints is None:
        return 0.0
    seq = prints if isinstance(prints, (list, tuple)) else (prints,)
    best = 0.0
    for p in seq:
        if p is not None:
            best = max(best, float(np.dot(query, p)))
    return best


_COSINE_MIN = 0.6         # min appearance cosine for ANY appearance adoption (husk claim / pool re-adopt)
_UNIQUENESS = 0.05        # twin guard: the best cosine must beat the 2nd-best by this, both adoption paths
_SIZE_TOL = 0.5           # appearance-candidate size gate: max 1 − min/max area
_FP_MAX = 8               # accumulated fingerprints per object (most-recent views kept)
_FP_NEW_DIST = 25.0       # min centroid move (px) for a settle to count as a NEW position (→ new print)


class ReidParams:
    """The ``[reid]`` section — the APPEARANCE-POLICY home. The classical descriptor itself has no
    knobs (its calibration is baked); everything here decides what the pipeline DOES with its
    cosines: the claim/adopt gates, the twin guard, the candidate size gate, and the fingerprint
    accumulation. Declared ONCE here and read LIVE by observe (husk relocation) and identity
    (pool re-acquire + fingerprinting), which are constructed with this instance — so one panel
    owns the whole appearance policy and a live ``/rmt/set reid …`` reaches both consumers."""

    PARAMS = (
        ParamSpec("enabled", bool, None, hot=False, bind=False, expose="edit",
                  doc="Master switch for the appearance re-id tie-break (classical color+shape+size descriptor)"),
        ParamSpec("reloc_cosine_reject", float, 0.0, lo=0.0, hi=1.0, attr="_reloc_cosine_reject",
                  doc="Appearance VETO on a GEOMETRIC/proximity claim: reject it if the blob's cosine to the object's fingerprint is below this (0 = off). Stops a stone stealing a gear's id on a place-swap"),
        ParamSpec("cosine_min", float, _COSINE_MIN, lo=0.0, hi=1.0, attr="_cosine_min",
                  doc="Min appearance cosine for ANY appearance adoption - a HIDDEN husk claiming a settled blob (observe, any distance) or a pooled identity being re-adopted (identity). One gate, one descriptor scale"),
        ParamSpec("uniqueness_margin", float, _UNIQUENESS, lo=0.0, hi=1.0, attr="_uniqueness_margin",
                  doc="Twin guard, BOTH adoption paths: the best cosine must beat the 2nd-best by this margin"),
        ParamSpec("size_tol", float, _SIZE_TOL, lo=0.0, hi=1.0, attr="_size_tol",
                  doc="Max relative area difference (1 - min/max) for an appearance candidate"),
        ParamSpec("fp_max_prints", int, _FP_MAX, lo=1, attr="_fp_max_prints",
                  doc="Max accumulated appearance fingerprints per object (most-recent views kept)"),
        ParamSpec("fp_new_dist", float, _FP_NEW_DIST, lo=0.0, attr="_fp_new_dist",
                  doc="Min centroid move (px) for a settle to count as a NEW position and append a fingerprint"),
    )

    def __init__(self, config: Any = None, *, logger: Any = None) -> None:
        bind_params(self, config)      # _cosine_min / … / _fp_new_dist ← PARAMS

    # read-only accessors for the LIVE consumers (observe / identity) — mirror efd.bins/detail
    @property
    def reloc_cosine_reject(self) -> float:
        return self._reloc_cosine_reject

    @property
    def cosine_min(self) -> float:
        return self._cosine_min

    @property
    def uniqueness_margin(self) -> float:
        return self._uniqueness_margin

    @property
    def size_tol(self) -> float:
        return self._size_tol

    @property
    def fp_max_prints(self) -> int:
        return self._fp_max_prints

    @property
    def fp_new_dist(self) -> float:
        return self._fp_new_dist


def load_embedder(config: Any = None, *, logger: Any = None) -> "ClassicalEmbedder | None":
    """Build the classical appearance embedder from a ``[reid]`` config, or None when disabled.
    No model file, no onnxruntime — enabling ``[reid]`` costs nothing but the descriptor compute."""
    log = logger or NULL_LOGGER
    g = config.get if config is not None else (lambda k, d=None: d)
    if not bool(g("enabled", True)):
        return None
    return ClassicalEmbedder(logger=log)
