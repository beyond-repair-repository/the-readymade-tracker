"""EFD — the elliptic-Fourier subsystem, shared across the whole tracker.

ONE Kuhl-Giardina integration per contour (MEMOIZED by contour data), reused by every consumer of the same
shape: the ``efd`` display feature (the sonification spectrum), the pose rotation fit (phase-correlation),
the reflectional-symmetry feature, and the grouper compound blend. Rotation, reflection and start-point are
all cheap *phase* operations on these coefficients, so once the spectrum exists everything else is cheap.

A **hardcoded ``MIN_BINS`` floor** guarantees the INTERNAL consumers (pose / symmetry / grouper) always get
enough harmonics, independent of the user-facing display ``bins`` (which a sonifier may set small). The cache
computes at ``max(bins, MIN_BINS)`` and hands each caller its requested SUBSET — harmonics are independent,
so a subset is exact.

The ``[efd]`` config section (``bins`` + ``detail``) is anchored by the ``Efd`` class below.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from lib.logger import NULL_LOGGER
from lib.params import ParamSpec, bind_params

MIN_BINS = 24              # HARDCODED floor — the most any INTERNAL consumer needs (pose fit 24, symmetry 10, grouper 12)
MAX_BINS = 256             # the `bins` param's ceiling: a full-floor spectrum ships 2×bins triples in ONE
                           #   OSC message, and 2×256×12B ≈ 6.2 KB still fits the 8192-byte datagram budget
                           #   (512 would overflow it). The shapes themselves decide how much arrives.
_ROT_N = 256               # sample count for the phase-correlation FFT
_SELECT_FLOOR_DB = 100.0   # detail knob sweeps the adaptive dB cut over this range below the loudest pair magnitude
_EPS = 1e-9

# -- the ONE shared coefficient cache -------------------------------------------------------------- #
_compute_k = MIN_BINS                # harmonics the cache computes = max(display bins, MIN_BINS); set at build
_cache: "dict[bytes, tuple]" = {}    # normalized-contour-bytes -> (coeff_dict @ _compute_k, centroid)
_CACHE_CAP = 128


def set_compute_bins(bins: int) -> None:
    """Set the shared cache's harmonic depth to ``max(bins, MIN_BINS)`` — called once at build from
    ``[efd].bins``, so a small display ``bins`` never starves pose's rotation fit. Clears the cache."""
    global _compute_k
    _compute_k = max(int(bins), MIN_BINS)
    _cache.clear()


def compute_bins() -> int:
    """The harmonic depth the shared cache currently computes at (= ``max(bins, MIN_BINS)``)."""
    return _compute_k


def efd_coeffs(contour, maxK):
    """Elliptic Fourier coefficients ``c[±n]`` (n=1..maxK) + arc-length centroid — MEMOIZED per contour, so
    the ONE Kuhl-Giardina integration is SHARED across every consumer of the same contour. The cache holds
    the full spectrum at ``compute_bins()`` and returns the requested ``maxK`` SUBSET (harmonics are
    independent → a subset is EXACT). Key = the contour's normalized float64 bytes, so different array
    shapes/dtypes of the same points share the entry. The returned dict is READ-ONLY (shared with the
    cache); no caller mutates it."""
    xy = np.ascontiguousarray(np.asarray(contour, np.float64).reshape(-1, 2))
    if maxK > _compute_k:                     # (no caller exceeds the floor/ceiling; guard) → compute uncached
        return _efd_kg(xy, maxK)
    key = xy.tobytes()
    hit = _cache.get(key)
    if hit is None:
        hit = _efd_kg(xy, _compute_k)
        if len(_cache) >= _CACHE_CAP:
            _cache.pop(next(iter(_cache)))    # bound (FIFO) — footprints churn as objects come + go
        _cache[key] = hit
    c, centroid = hit
    if maxK >= _compute_k:
        return c, centroid
    return {k: v for k, v in c.items() if abs(k) <= maxK}, centroid


def _efd_kg(contour, maxK):
    """Exact elliptic Fourier descriptors (Kuhl & Giardina 1982) — computed DIRECTLY on the raw polygon
    vertices, NO resampling.

    KG integrates the piecewise-linear contour segment by segment, so the coefficients are EXACT for the
    polygon it is handed (any vertex count, incl. sharp corners) — resampling would only cost + round corners.
    Returns the complex pair form the rest of the pipeline uses: ``c[+n], c[-n]`` for n in 1..maxK, ``c[0]=0``
    (the DC is returned separately as the ARC-LENGTH-weighted centroid — the true A0/C0, unbiased by the
    non-uniform vertex spacing of CHAIN_APPROX; a plain point-mean would pull toward dense corners).

    Per harmonic n the real coefficients are a_n,b_n for x(t), c_n,d_n for y(t); the complex pair from z=x+iy:
        c[+n] = ((a_n + d_n) + i(c_n - b_n)) / 2 ;  c[-n] = ((a_n - d_n) + i(b_n + c_n)) / 2
    """
    xy = np.asarray(contour, float)
    if len(xy) < 3:
        return {0: 0 + 0j}, (xy.mean(axis=0) if len(xy) else np.zeros(2))
    pts = np.vstack([xy, xy[0]])                 # close the contour
    d = np.diff(pts, axis=0)                      # segment deltas
    dt = np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-12)
    t = np.concatenate([[0.0], np.cumsum(dt)])    # cumulative arc length
    T = t[-1]
    if T <= 1e-9:                                 # coincident vertices → no arc → no spectrum
        return {0: 0 + 0j}, xy.mean(axis=0)
    mid = (pts[:-1] + pts[1:]) * 0.5              # arc-length-weighted centroid = Σ midpoint·seglen / T
    centroid = (mid * dt[:, None]).sum(axis=0) / T
    c = {0: 0 + 0j}
    for n in range(1, maxK + 1):
        phi = 2.0 * np.pi * n / T
        cs, sn = np.cos(phi * t), np.sin(phi * t)
        dcos, dsin = cs[1:] - cs[:-1], sn[1:] - sn[:-1]
        k = T / (2.0 * n * n * np.pi * np.pi)
        an = k * np.sum(d[:, 0] / dt * dcos)
        bn = k * np.sum(d[:, 0] / dt * dsin)
        cn = k * np.sum(d[:, 1] / dt * dcos)
        dn = k * np.sum(d[:, 1] / dt * dsin)
        c[n] = complex(an + dn, cn - bn) / 2.0
        c[-n] = complex(an - dn, bn + cn) / 2.0
    return c, centroid


_TIE_FRAC = 0.995        # the half-turn alternative within this fraction of the max = a genuine 2-fold
#                          TIE → resolved by hysteresis toward prev_theta. Discriminates cleanly: a
#                          2-fold shape's EVEN harmonics all re-align at the half turn (alternative
#                          ≈ 1.0), while an asymmetric shape drops there by ~2× its detail share
#                          (a stone ~0.974, a lumpy blob ~0.93) — no overlap with the window


def _corr_from_coeffs(cr, ct, max_k, samples, prev_theta=None):
    """Circular phase-correlation of two EFD coefficient dicts → ``(theta_rad, score)``. Peak LOCATION
    aligns the start-point, peak PHASE = the rotation, peak HEIGHT (Parseval-normalized) = 0..1 match.

    ``prev_theta``: 2-fold hysteresis. A bar-like shape correlates almost equally at θ and θ+180°
    (its even harmonics re-align at the half turn) and capture noise picks the winner — the measured
    angle flips at random between settles. When the HALF-TURN alternative peak is within ``_TIE_FRAC``
    of the max, the candidate whose rotation is circularly closest to ``prev_theta`` wins
    (continuity-preferred; the two silhouettes are physically indistinguishable). Only the specific
    N/2 shift is tested — never the main peak's own shoulder, whose phase reads skewed. None →
    plain argmax."""
    zr = np.zeros(samples, complex)                  # EFD pairs → a DFT spectrum (DC dropped ⇒ centroid-free)
    zt = np.zeros(samples, complex)
    for n in range(1, max_k + 1):                    # .get: a degenerate contour yields only the DC key
        zr[n] = cr.get(n, 0j); zr[samples - n] = cr.get(-n, 0j)
        zt[n] = ct.get(n, 0j); zt[samples - n] = ct.get(-n, 0j)
    er = float(np.vdot(zr, zr).real); et = float(np.vdot(zt, zt).real)
    if er <= 1e-20 or et <= 1e-20:                   # coincident-vertex / empty contour → no spectrum
        return 0.0, 0.0
    r = np.fft.ifft(zr * np.conj(zt))                # circular cross-correlation (zr, zt already freq-domain)
    mag = np.abs(r)
    m = int(np.argmax(mag))
    if prev_theta is not None:
        m2 = (m + samples // 2) % samples            # the half-turn alternative (θ + 180°)
        if mag[m2] >= _TIE_FRAC * mag[m]:            # a genuine 2-fold tie → prefer continuity
            cand = np.array([m, m2])
            phases = (-np.angle(r[cand])) % (2.0 * math.pi)
            dist = np.abs((phases - float(prev_theta) + math.pi) % (2.0 * math.pi) - math.pi)
            m = int(cand[int(np.argmin(dist))])
    theta = (-float(np.angle(r[m]))) % (2.0 * math.pi)
    score = float(mag[m] * samples / (er * et) ** 0.5)   # Parseval-normalized peak → ~1 for a match
    return theta, min(score, 1.0)


def rotation_by_correlation(ref_contour, tgt_contour, max_k: int = MIN_BINS, samples: int = _ROT_N,
                            prev_theta=None):
    """Absolute rotation of ``tgt_contour`` relative to ``ref_contour`` via EFD PHASE CORRELATION — the fast,
    contour-native alternative to a brute-force masked-overlap angle sweep.

    A rigid rotation ψ multiplies every complex EFD coefficient by ``e^{iψ}``; the START-POINT offset is a
    cyclic shift of the samples. The circular cross-correlation ``IFFT(Z_ref · conj(Z_tgt))`` peaks at the
    shift that best aligns the two, and the COMPLEX value at that peak carries the global phase → the
    rotation. So ONE IFFT recovers both the start alignment (peak location) and the rotation (peak phase), at
    CONTINUOUS resolution (no angle grid).

    Returns ``(theta_rad, score)`` — ``theta_rad`` in [0, 2π); ``score`` a 0..1 normalized peak height (1 = a
    perfect shape match at that rotation, low = a dissimilar / poor fit → gate on it like the overlap
    ``conf_min``). ``(0.0, 0.0)`` on a degenerate contour. Both contours must be traced in the SAME direction
    (cv2 findContours is consistent) — a reversed trace reads as a reflection. ``prev_theta`` (rad):
    tie hysteresis for near-symmetric shapes — see ``_corr_from_coeffs``."""
    cr, _ = efd_coeffs(np.asarray(ref_contour, float).reshape(-1, 2), max_k)   # accept cv2 (N,1,2) or (N,2)
    ct, _ = efd_coeffs(np.asarray(tgt_contour, float).reshape(-1, 2), max_k)
    return _corr_from_coeffs(cr, ct, max_k, samples, prev_theta)


def rotation_information(contour, max_k: int = MIN_BINS) -> float:
    """How much of a contour's spectral energy actually carries ORIENTATION (0..1). A perfect circle
    puts ALL its energy into a single fundamental coefficient whose phase is absorbed by the
    start-point shift — its rotation is UNDEFINED, and any measured θ is noise (the correlation
    ``score`` stays ~1: the shape matches itself at EVERY angle, so score cannot catch this). The
    energy OUTSIDE that one coefficient is what a rotation estimate stands on: measured on real
    captures, circles / rings / discs sit ≤ 0.004 while even mildly elongated or lumpy shapes sit
    ≥ 0.009. Gate rotation on it (pose ``symmetry_lock``). 0.0 on a degenerate contour."""
    c, _ = efd_coeffs(np.asarray(contour, float).reshape(-1, 2), max_k)
    energy = sum(abs(v) ** 2 for k, v in c.items() if k != 0)
    if energy <= 1e-20:
        return 0.0
    return 1.0 - max(abs(c.get(1, 0j)), abs(c.get(-1, 0j))) ** 2 / energy


def reflection_score(contour, max_k: int = 10, samples: int = 128) -> float:
    """Mirror-symmetry score (0..1) of a contour — how well it matches its own reflection over the best
    axis. The reflected shape's coefficients are ``mirror_c[k] = -conj(c[k])`` (a vertical flip + traversal
    reversal), DERIVED analytically from the shape's coeffs — so NO second EFD integration, it reuses the
    ONE shared (cached) spectrum. 1 = mirror-symmetric; sonify ``1 − score`` for an 'asymmetry' feature."""
    c, _ = efd_coeffs(np.asarray(contour, float).reshape(-1, 2), max_k)
    cm = {k: -np.conj(v) for k, v in c.items()}      # reflection = negate + conjugate every coefficient
    _theta, score = _corr_from_coeffs(c, cm, max_k, samples)
    return score


def pair_magnitudes(c, maxK):
    """Return (ks, a) where ``a_k = |c_+k| + |c_-k|`` for k = 1..maxK. Rotation-invariant (|c| is unchanged
    by a whole-shape rotation). (The SUM of a harmonic's two conjugate bins — a per-harmonic magnitude;
    ``select_detail`` thresholds the individual bins, see ``component_magnitudes``.)"""
    ks = np.arange(1, maxK + 1)
    a = np.array([abs(c[k]) + abs(c[-k]) for k in ks])
    return ks, a


def component_magnitudes(c, maxK):
    """Return (ks, mpos, mneg) with ``mpos[k-1] = |c_+k|`` and ``mneg[k-1] = |c_-k|`` for k = 1..maxK — the
    INDIVIDUAL conjugate-component magnitudes (the ± bins of the diverging spectrum). Rotation-invariant.
    ``select_detail`` runs its adaptive ± threshold over these."""
    ks = np.arange(1, maxK + 1)
    mpos = np.array([abs(c[k]) for k in ks])
    mneg = np.array([abs(c[-k]) for k in ks])
    return ks, mpos, mneg


def select_detail(mpos, mneg, details_level, max_bins):
    """Adaptive dB ± THRESHOLD driven by the 0..1 detail knob — PER COMPONENT. The whole spectrum is
    normalized to the loudest single BIN (0 dB = ``max`` over all ``|c±ₙ|``); the knob sets a cut
    ``cut_db = −detail·_SELECT_FLOOR_DB`` below it, and EACH conjugate bin survives on its own
    (``|c₊ₙ| ≥ cut`` → keep +n; ``|c₋ₙ| ≥ cut`` → keep −n). So a circle keeps ONE bin (its single winding),
    a busy contour keeps many, and a SPLIT harmonic whose two bins each fall below the cut drops entirely.
    The loudest bin always survives (cut ≤ 0 dB). detail 0 → only the loudest bin; detail 1 → down to the
    floor. Returns ``(keep_pos, keep_neg, cut_amp)`` — 1-based harmonic index sets, one per sign."""
    details_level = float(np.clip(details_level, 0.0, 1.0))
    amax = max(float(mpos.max()), float(mneg.max())) if len(mpos) else 0.0
    if amax <= _EPS:                                  # degenerate (a point) — nothing to keep
        return set(), set(), 0.0
    cut_amp = amax * (10.0 ** (-details_level * _SELECT_FLOOR_DB / 20.0))   # dB cut → linear bin magnitude
    keep_pos = set((np.nonzero(mpos >= cut_amp)[0] + 1).tolist())          # the loudest bin (= amax) always
    keep_neg = set((np.nonzero(mneg >= cut_amp)[0] + 1).tolist())          # clears the cut → never empty
    return keep_pos, keep_neg, cut_amp


def perceptual_contour(contour, max_bins: int, detail: float) -> "np.ndarray | None":
    """The PERCEPTUAL outline: the inverse EFD of exactly the ± components that survive the ``detail``
    cut (``select_detail``) — so the emitted contour IS the shape the emitted spectrum describes, and
    the ``[efd] detail`` knob is the one quality dial for both. Reconstructed with the RAW
    (non-canonicalized) coefficients around the contour's own arc-length centroid, so it lies exactly
    where the source contour lies, at its true orientation. Sample count follows the kept bandwidth
    (~8 points per highest kept harmonic, clamped 32..512 — the grouper's sampling policy). Returns a
    cv2 ``(N,1,2) float32`` contour — SUB-PIXEL, never snapped to the integer grid: on a small object
    adjacent samples are fractions of a pixel apart, and int-rounding here is what turned the smooth
    inverse-EFD curve into staircases once a display scaled it up. Ints only at cv2-DRAW boundaries.
    None for a degenerate source (< 3 points)."""
    if contour is None or len(contour) < 3:
        return None
    c, centroid = efd_coeffs(contour, max_bins)
    _ks, mpos, mneg = component_magnitudes(c, max_bins)
    keep_pos, keep_neg, _cut = select_detail(mpos, mneg, detail, max_bins)
    if not keep_pos and not keep_neg:
        return None
    n_max = max(max(keep_pos, default=1), max(keep_neg, default=1))
    samples = int(np.clip(8 * n_max, 32, 512))
    f = np.arange(samples) / samples
    z = np.full(samples, complex(centroid[0], centroid[1]))
    for n in sorted(keep_pos):                       # z(t) = centroid + Σ c[+n]e^{iθ} + Σ c[-n]e^{-iθ},
        z += c[n] * np.exp(1j * 2.0 * np.pi * n * f)    # summed over the SURVIVING bins only
    for n in sorted(keep_neg):
        z += c[-n] * np.exp(-1j * 2.0 * np.pi * n * f)
    return np.stack([z.real, z.imag], axis=1).astype(np.float32).reshape(-1, 1, 2)


class Efd:
    """``[efd]`` config anchor — the SHARED elliptic-Fourier spectrum's live params.

    ``bins`` = the display harmonic width (the ``efd`` shape feature + the webgui inspector). ``detail`` = the
    live 0..1 adaptive-selection knob. The shared cache computes at ``max(bins, MIN_BINS)``, so the INTERNAL
    consumers (pose's rotation fit, the symmetry feature, the grouper) always have ≥ ``MIN_BINS`` no matter
    how small ``bins`` is set for sonification. NOT a pipeline stage — a config holder the Features stage
    reads (as ``FeatInput.cfg``); a live ``detail`` change bumps ``version`` so Features recomputes the efd."""

    name = "efd"

    PARAMS = (
        ParamSpec("bins", int, 64, lo=1, hi=MAX_BINS, hot=False, attr="_bins",
                  doc=f"EFD harmonics for the display spectrum (the efd feature width, <= {MAX_BINS} - the datagram budget's edge); compute cost scales linearly, the detail cut keeps the wire sparse; internal consumers always get >= {MIN_BINS}"),
        ParamSpec("detail", float, 0.5, lo=0.0, hi=1.0, attr="_detail",
                  apply=lambda self, v: self._set_detail(v),
                  doc="EFD detail 0..1 - adaptive per-component +- dB cut, normalized to the loudest single bin: each |c+-n| survives on its own above -detail-100 dB; 0 = only the loudest bin, 1 = down to the floor (the kept count follows the shape)"),
    )

    def __init__(self, config: Any = None, *, logger: Any = None) -> None:
        self._log = logger or NULL_LOGGER
        self._version = 0
        bind_params(self, config)                    # _bins / _detail (detail's apply bumps _version to 1)
        set_compute_bins(self._bins)                 # cache depth = max(bins, MIN_BINS)
        self._log.info(f"[efd] bins={self._bins} detail={self._detail} "
                       f"(cache computes {compute_bins()} harmonics, floor {MIN_BINS})")

    def _set_detail(self, v: float) -> None:
        self._detail = float(v)
        self._version += 1                           # → Features re-selects the efd on the next tick

    @property
    def bins(self) -> int:
        return self._bins

    @property
    def detail(self) -> float:
        return self._detail

    @property
    def version(self) -> int:
        return self._version
