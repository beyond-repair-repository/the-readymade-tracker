"""``features`` — per-object geometric shape descriptors (an optional pipeline STAGE).

A small **registry** (`@feature`) + a stage that computes the *enabled* features for each
CAPTURED object and **annotates** them onto the scene (`scene.annotate(id, "features", …)`),
where the smart ``oscout`` serializes them and the gui draws them. Adding a feature is one
`@feature` function — the stage / cache / output / viz are all generic over the registry.

The seed set is the cheap, **pose-invariant** shape scalars (circularity, rectangularity,
convexity, solidity, eccentricity, circle_variance — each one `cv2` primitive). Because they
don't change when an object moves or rotates, only when its *shape* changes, a feature is
computed **once, at capture**, cached on the footprint-contour identity, and reused. A
resting scene costs ~0. (Pose-*variant* / dynamic features — speed, dwell, density — are
supported via `invariant=False`, recomputed per frame; none are registered here.)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from types import MappingProxyType, SimpleNamespace
from typing import Any, Callable

import cv2
import numpy as np

from lib.efd import component_magnitudes, efd_coeffs, perceptual_contour, reflection_score, select_detail
from lib.logger import NULL_LOGGER
from lib.params import ParamSpec
from lib.events import readonly
from lib.scene import rekey

# --------------------------------------------------------------------------- #
# The registry: a feature is one function name -> compute
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Feature:
    """A registered descriptor. ``fn`` maps a ``FeatInput`` → a list of floats (1 = a scalar,
    N = a vector, e.g. EFD). ``invariant`` = pose-invariant (compute once at capture + cache).
    ``span`` = the expected value range, for a client's 0..1 normalization."""

    name: str
    fn: Callable[["FeatInput"], "list[float]"]
    invariant: bool = True
    span: tuple[float, float] = (0.0, 1.0)
    internal: bool = False                       # system-level (reid's shape block) — never user-enable-able,
    #                                              never annotated/emitted, hidden from the webgui enum


_REGISTRY: dict[str, Feature] = {}


def feature(name: str, *, invariant: bool = True, span: tuple[float, float] = (0.0, 1.0),
            internal: bool = False):
    """Register a compute function under ``name``. Register it → it's *available*; enable its
    name in ``[features] enable`` → it's computed + sent + drawn. ``internal`` features exist only
    for system machinery (reid's shape block) — excluded from the user enum and never emitted."""
    def deco(fn: Callable[["FeatInput"], "list[float]"]) -> Callable:
        _REGISTRY[name] = Feature(name, fn, invariant, span, internal)
        return fn
    return deco


def registry() -> dict[str, Feature]:
    """The registered features (read-only view for callers / the config error)."""
    return dict(_REGISTRY)


@dataclass
class FeatInput:
    """What a compute fn gets — decoupled from ObjectSnap internals, with the common
    contour quantities precomputed ONCE so each feature is a one-liner."""

    contour: np.ndarray            # (N,1,2) int32 — the footprint outline (absolute frame coords)
    pts: np.ndarray                # (N,2) float64 — the same points, for covariance / radial stats
    area: float
    perimeter: float
    hull_area: float
    hull_perimeter: float
    centroid: tuple[float, float]
    cfg: Any                       # the [efd] config (the ``Efd`` component): ``.bins`` / ``.detail`` for the efd feature
    frame_area: "float | None" = None   # frame w·h px² — the `size` normalizer (None = raw px)


def _feat_input(contour: np.ndarray, cfg: Any, frame_area: "float | None" = None) -> FeatInput:
    # ROUND to int32 (the wire contour is float32/sub-pixel since 0.31.13): the int-picky cv2 ops
    # (convexityDefects) need it, and shape SCALARS don't gain from sub-pixel input.
    c = np.rint(np.asarray(contour)).reshape(-1, 1, 2).astype(np.int32)
    pts = c.reshape(-1, 2).astype(np.float64)
    area = float(cv2.contourArea(c))
    perim = float(cv2.arcLength(c, True))
    hull = cv2.convexHull(c)
    m = cv2.moments(c)
    m00 = m["m00"] or 1.0
    centroid = (m["m10"] / m00, m["m01"] / m00)
    return FeatInput(contour=c, pts=pts, area=area, perimeter=perim,
                     hull_area=float(cv2.contourArea(hull)),
                     hull_perimeter=float(cv2.arcLength(hull, True)),
                     centroid=centroid, cfg=cfg, frame_area=frame_area)


# --------------------------------------------------------------------------- #
# The seed set — cheap, pose-invariant shape scalars
# --------------------------------------------------------------------------- #

_EPS = 1e-9


@feature("circularity", span=(0.0, 1.0))            # 4πA/P² — round vs not (1 = perfect disc)
def _circularity(x: FeatInput) -> list:
    return [min(1.0, 4.0 * math.pi * x.area / (x.perimeter ** 2 + _EPS))]


_RECT_FLOOR = math.pi / 4.0    # a circle — and EVERY ellipse — fills exactly pi/4 of its min-area rect


@feature("rectangularity", span=(0.0, 1.0))         # BOX-likeness: 0 = round, 1 = a filled rectangle
def _rectangularity(x: FeatInput) -> list:
    """How box-like the shape is, over the range a shape can actually occupy.

    The raw fill fraction ``A / min-area-rect A`` is bounded BELOW by pi/4 = 0.785 for a disc — and
    for every ellipse, at any elongation — so read raw it says "79% rectangular" for a circle and
    leaves only the top fifth of 0..1 in use. Rescaling against that floor makes the number mean what
    its name says: a disc reads 0, a filled rectangle reads 1, at any rotation (the rect is
    min-AREA, so it follows the shape).

    Shapes sparser than a disc (a star, a torn edge) sit below the floor and clamp to 0: this axis is
    box-vs-round ONLY. Their sparseness is `solidity`'s to report and their points are `spikiness`'s."""
    (w, h) = cv2.minAreaRect(x.contour)[1]
    fill = x.area / (w * h + _EPS)
    return [float(np.clip((fill - _RECT_FLOOR) / (1.0 - _RECT_FLOOR), 0.0, 1.0))]


@feature("solidity", span=(0.0, 1.0))               # A / hull area — concavity / bulk
def _solidity(x: FeatInput) -> list:
    return [min(1.0, x.area / (x.hull_area + _EPS))]


@feature("eccentricity", span=(0.0, 1.0))           # STANDARD eccentricity: 0 = circle, →1 = a line
def _eccentricity(x: FeatInput) -> list:
    """Eccentricity of the ellipse with the same second moments — the TEXTBOOK definition,
    ``e = sqrt(1 - b²/a²)``, computed as ``sqrt(1 - λ₂/λ₁)`` from the boundary-point covariance
    (for an ellipse the eigenvalue ratio IS b²/a²). 0 = a circle, approaching 1 = a line.

    Note for anyone mapping this to a control: e rises very fast and then saturates — a barely
    elongated 1.5:1 ellipse already reads ~0.75, and everything from 2:1 to 10:1 is packed into
    0.87..0.995. That is the standard definition behaving as it always does, not a defect here."""
    if len(x.pts) < 3:
        return [0.0]                                 # no measurable elongation (NOT 1.0 — that would
        #                                              report a degenerate contour as maximally elongated)
    cov = np.cov(x.pts.T)
    ev = np.linalg.eigvalsh(cov)                     # ascending → [λ₂, λ₁]
    lam2, lam1 = float(ev[0]), float(ev[1])
    return [math.sqrt(max(0.0, 1.0 - lam2 / (lam1 + _EPS)))]


@feature("circle_variance", span=(0.0, 0.6), internal=True)   # reid shape block ONLY        # σ_R/μ_R radial-distance CV (0 = true circle)
def _circle_variance(x: FeatInput) -> list:
    gx, gy = x.centroid
    d = np.hypot(x.pts[:, 0] - gx, x.pts[:, 1] - gy)
    mu = float(d.mean())
    return [float(d.std()) / (mu + _EPS)]


@feature("ellipse_variance", span=(0.0, 0.6), internal=True)  # reid shape block ONLY       # radial CV in the shape's OWN covariance (Mahalanobis) metric —
def _ellipse_variance(x: FeatInput) -> list:        # deviation from a perfect ELLIPSE. 0 = clean ellipse.
    if len(x.pts) < 3:
        return [0.0]
    v = x.pts - np.asarray(x.centroid)
    cov = np.cov(x.pts.T)
    if not np.all(np.isfinite(cov)) or abs(np.linalg.det(cov)) < _EPS:
        return [0.0]                                 # degenerate (collinear) → no ellipse metric
    inv = np.linalg.inv(cov)
    d = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", v, inv, v), 0.0))   # Mahalanobis radius per boundary point
    mu = float(d.mean())
    return [float(d.std()) / (mu + _EPS)]


_SIZE_REF = 0.35    # 1.0 = an object whose equivalent diameter spans this fraction of the frame. Set from
#                     measurement: rigid tabletop objects run 3-16% of the frame width (0.09..0.46 at this
#                     reference), and MALLEABLE material goes bigger — the crumbs clip's sculpting sheet is
#                     Ø26% of the frame, which the old quarter-frame reference CLIPPED at 1.0, freezing its
#                     size (and any size-driven mapping) no matter how it was sculpted. At 0.35 the sheet
#                     reads ~0.74 and compressing its footprint MOVES the number. Raise again if objects
#                     routinely clip at 1.0. (0.25 -> 0.35 at 0.32.132: a BREAKING recalibration — every
#                     published size shrinks by 5/7.)


def size_fraction(area: float, frame_area: "float | None") -> float:
    """THE size definition, shared so that everything reporting a size means the same thing: the
    EQUIVALENT DIAMETER (the width of the disc of that area) over a reference extent.

    Linear in apparent size, which is also how people compare sizes, so the number moves the way the
    eye does. An AREAL fraction does not: it is quadratic in size, so real tabletop objects span
    0.0009..0.0198 — under 2% of a 0..1 range — while differing 22.6x in true area, and everything
    reads as "about zero". That is why the areal form is not published.

    ``sqrt(frame_area)`` is the frame's characteristic extent (exactly the side on a square frame, the
    geometric mean otherwise). With no frame known, returns the raw equivalent diameter in px."""
    d = math.sqrt(4.0 * max(area, 0.0) / math.pi)
    if not frame_area:
        return d
    return min(1.0, d / (_SIZE_REF * math.sqrt(frame_area)))


@feature("size", span=(0.0, 1.0))                   # PERCEPTUAL size: linear extent, not area
def _size(x: FeatInput) -> list:
    """How big the object looks, as a fraction of a reference extent. See ``size_fraction``."""
    return [size_fraction(x.area, x.frame_area)]


@feature("lobe_count", span=(0.0, 16.0))            # dominant frequency of the radial signature R(θ) — the m-fold
def _lobe_count(x: FeatInput) -> list:              # symmetry / #prongs (star→5, gear teeth→N, ellipse→2, blob→small).
    if len(x.pts) < 8:
        return [0.0]
    gx, gy = x.centroid
    ang = np.arctan2(x.pts[:, 1] - gy, x.pts[:, 0] - gx)
    rad = np.hypot(x.pts[:, 0] - gx, x.pts[:, 1] - gy)
    o = np.argsort(ang)                             # uniform-angle radial signature (star-convex assumption; robust enough)
    r = np.interp(np.linspace(-math.pi, math.pi, 256), ang[o], rad[o], period=2.0 * math.pi)
    mu = float(r.mean())
    if mu < _EPS or float(r.std()) / mu < 0.02:     # a near-constant radius (a disc / round blob) → no real lobes
        return [0.0]                                #   (else pixel-grid noise on a disc reads as a phantom count)
    band = np.abs(np.fft.rfft(r - mu))[1:24]        # 1..23-fold periodicity
    peak = int(np.argmax(band))
    if band[peak] < 3.0 * float(np.median(band)):   # no ONE prominent frequency (amorphous) → no lobes
        return [0.0]
    return [float(peak + 1)]


_SPIKE_WINDOW = 0.08     # coarse curvature support, as a FRACTION of perimeter — scale-relative by
#                          construction. Over it a circle turns pi*w, the zero point below.
_SPIKE_RATIO = 3.0       # fine window = coarse / this. A CORNER turns as much in the fine window as in
#                          the coarse one; a smooth arc turns proportionally to window size. That ratio
#                          is what separates a sharp point from a merely tight bend (see below).
_SPIKE_P = 6.0           # top-2 combine exponent: large p means the SHARPEST peak dominates, so many
#                          blunt points cannot accumulate (Huang: spikiness is driven nonlinearly by one
#                          or two very sharp spikes, not linearly by their number)


def _turning(p: np.ndarray, k: int, outward: float) -> np.ndarray:
    """Outward turning angle at every vertex over a support of ``k`` vertices either side.

    The perceptual contour is sampled uniformly in the EFD parameter, so a vertex offset IS an
    arc-length fraction (``2k/n`` of the perimeter) — which is what makes this scale-relative. Taking
    the support in vertices rather than by arc-length search also keeps it from collapsing to
    zero width on a coarse outline, where a fraction-of-perimeter window would round to nothing."""
    v_in, v_out = p - np.roll(p, k, axis=0), np.roll(p, -k, axis=0) - p
    ang = np.arctan2(v_in[:, 0] * v_out[:, 1] - v_in[:, 1] * v_out[:, 0],
                     v_in[:, 0] * v_out[:, 0] + v_in[:, 1] * v_out[:, 1])
    return np.maximum(ang * outward, 0.0)                    # convex only — inward sharpness is solidity's


@feature("spikiness", span=(0.0, 1.0))              # sharpness of the one or two sharpest OUTWARD points —
def _spikiness(x: FeatInput) -> list:               # the third SCI shape dimension beside lobe_count + solidity
    """How sharply the shape's sharpest OUTWARD protrusions come to a point (0 = a disc, 1 = a needle).

    Two-scale turning angle. Measuring the turn over ONE window cannot tell a sharp point from a tight
    bend — a 9-lobe flower turns as much over a lobe as a 9-point star does over a spike. So the turn is
    measured over a coarse window AND a fine one: a true corner concentrates its whole turn at the apex
    (fine ~ coarse), while a smooth arc spreads it (fine ~ coarse / ratio). Sharpness multiplies that
    CONCENTRATION by the coarse turn's MAGNITUDE above the circle baseline (a circle turns pi*w over the
    window), so a disc reads 0 and a needle reads 1. Both terms are angles and arc-length fractions,
    hence scale- and rotation-invariant.

    The two sharpest peaks — each counted once, via arc-length non-maximum suppression — combine under a
    p-norm, so two sharp points outrank many blunt ones instead of accumulating with count.

    Reads the PERCEPTUAL reconstruction, like every other geometric feature — so it describes the same
    outline the outputs emit, and moves with ``[efd] detail``: a coarser reconstruction really is a
    rounder object, and reports as less spiky. That band-limit is also the feature's ceiling — a tip
    finer than the kept harmonics resolve cannot be seen, so very high lobe counts compress."""
    p = x.pts
    n = len(p)
    k = int(round(0.5 * _SPIKE_WINDOW * n))
    kf = int(round(k / _SPIKE_RATIO))
    if n < 24 or kf < 1 or kf >= k:      # too coarse an outline to hold two separable scales — and a
        return [0.0]                     #   shape reconstructed from that few harmonics IS smooth
    shoelace = float(np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1]))
    outward = -1.0 if shoelace < 0.0 else 1.0    # winding-agnostic: + is always the OUTWARD turn
    coarse = _turning(p, k, outward)
    fine = _turning(p, kf, outward)
    ratio = kf / k
    conc = np.clip((fine / (coarse + _EPS) - ratio) / (1.0 - ratio), 0.0, 1.0)  # 1 = corner, 0 = arc
    base = math.pi * (2.0 * k / n)                           # what a circle turns over the same support
    mag = np.clip((coarse - base) / (math.pi - base), 0.0, 1.0)
    sharp = conc * mag
    if not sharp.any():
        return [0.0]
    # ONE spike spans several vertices — keep only its apex, so a peak counts once. Two apexes closer
    # together than the support are not separable by it anyway.
    peaks: list[float] = []
    taken: list[int] = []
    for i in np.argsort(sharp)[::-1]:
        if sharp[i] <= 0.0 or len(peaks) == 2:               # only the top two are ever needed
            break
        if any(min(abs(int(i) - j), n - abs(int(i) - j)) <= k for j in taken):
            continue
        peaks.append(float(sharp[i]))
        taken.append(int(i))
    a1 = peaks[0] if peaks else 0.0
    a2 = peaks[1] if len(peaks) > 1 else 0.0
    return [float(min(1.0, (a1 ** _SPIKE_P + a2 ** _SPIKE_P) ** (1.0 / _SPIKE_P)))]


@feature("defect_depth", span=(0.0, 1.0), internal=True)      # reid shape block ONLY           # deepest convexity notch / equivalent radius — concavity DEPTH (a
def _defect_depth(x: FeatInput) -> list:            # star's spikes, a gear's gullets). 0 = convex. Depth, vs convexity's raggedness.
    c = x.contour
    if len(c) < 4 or x.area < _EPS:
        return [0.0]
    try:
        hull = cv2.convexHull(c, returnPoints=False)
        if hull is None or len(hull) < 3:
            return [0.0]
        defects = cv2.convexityDefects(c, hull)      # raises if the contour isn't monotone along the hull
    except cv2.error:
        return [0.0]
    if defects is None:
        return [0.0]
    deepest = float(defects.reshape(-1, 4)[:, 3].max()) / 256.0   # fixed-point depth (×256) → px (shape (N,1,4)|(N,4))
    return [min(1.0, deepest / (math.sqrt(x.area / math.pi) + _EPS))]   # / equivalent radius → scale-invariant


@feature("reflectional_symmetry", span=(0.0, 1.0))  # match of the shape to its MIRROR (best reflection axis) via the EFD
def _reflectional_symmetry(x: FeatInput) -> list:   # phase-correlation. 1 = mirror-symmetric; sonify 1−v for 'asymmetry'.
    if len(x.pts) < 3:
        return [0.0]                                # no measurable symmetry (NOT 1.0 — that would report a
        #                                             degenerate contour as PERFECTLY symmetric, outranking
        #                                             every real object; same reasoning as eccentricity's
        #                                             guard, and the house rule for every guard here: an
        #                                             unmeasurable input returns the span MINIMUM)
    return [float(reflection_score(x.pts))]         # mirror derived analytically from the SHARED cached spectrum


# --------------------------------------------------------------------------- #
# EFD — the `efd` VECTOR feature: the shape's normalized harmonic spectrum, adaptively
# truncated by the [efd].detail knob. The EFD machinery (efd_coeffs, rotation_by_correlation,
# pair_magnitudes, select_detail + the shared cache + MIN_BINS) lives in lib/efd.py. Opt-in.
# --------------------------------------------------------------------------- #


@feature("efd", span=(0.0, 1.0))                    # sparse complex spectrum: per SURVIVING ±bin, (±bin, amp, phase)
def _efd(x: FeatInput) -> list:
    """The detail-truncated EFD as SIGNED-BIN TRIPLES: for each SURVIVING conjugate bin (+n and/or −n, in
    ascending harmonic order) a ``(signed_bin, amplitude, phase)`` — amplitude normalized to the loudest
    single BIN (the max ``|c±ₙ|`` over the whole spectrum, FOUND BY SEARCH — it may be +1, −1, or any
    harmonic, NOT assumed to be the fundamental — so the loudest bin = 1.0), phase in radians (−π…π). The
    signed bin distinguishes +n (|c₊ₙ|, the ellipse's forward winding) from −n (|c₋ₙ|, the reverse); their
    asymmetry is the harmonic's eccentricity and the phase is what a magnitude-only spectrum drops — a
    sonifier reads the flat list in triples. The detail knob is a per-COMPONENT ± threshold (see
    select_detail): each bin survives on its own, so a harmonic may emit BOTH bins, just one, or neither (a
    near-circle emits a lone forward winding) — sparse + adaptive."""
    max_bins = int(getattr(x.cfg, "bins", 12))       # x.cfg is the [efd] config (the Efd component)
    detail = float(getattr(x.cfg, "detail", 0.5))
    if max_bins < 1 or len(x.pts) < 3:
        return []
    c, _ = efd_coeffs(x.pts, max_bins)
    _ks, mpos, mneg = component_magnitudes(c, max_bins)
    amax = max(float(mpos.max()), float(mneg.max()))     # the loudest single BIN (searched over ALL ± bins —
    if amax <= _EPS:                                     #   may be +1, −1, or any n) — the 0 dB scale reference
        return []                                        # degenerate (a point) → no descriptor
    keep_pos, keep_neg, _cut = select_detail(mpos, mneg, detail, max_bins)   # per-component ± threshold
    out: list = []
    for k in range(1, max_bins + 1):                 # ascending harmonic order; +k before −k
        if k in keep_pos:
            out += [float(k), float(abs(c[k])) / amax, float(np.angle(c[k]))]
        if k in keep_neg:
            out += [float(-k), float(abs(c[-k])) / amax, float(np.angle(c[-k]))]
    return out


_DEFAULT_ENABLE = ("circularity, rectangularity, solidity, eccentricity, "
                   "size, lobe_count, reflectional_symmetry, spikiness")


# --------------------------------------------------------------------------- #
# The stage: compute enabled features at capture, cache per footprint identity
# --------------------------------------------------------------------------- #

class Features:
    """Pipeline stage: for each CAPTURED object, compute the enabled features from its footprint
    contour and ``scene.annotate`` them under ``"features"``. Invariant features are cached on the
    footprint-contour identity — recomputed + re-annotated only when the object (re)captures a new
    shape, so a resting scene costs nothing."""

    name = "features"

    # -- live parameters, declared WITH the stage. ``enable`` is construction-consumed (→ the feature-fn list),
    #    so bind=False (registry-only: coercion + /query + the dynamic multiselect); __init__ reads it itself.
    #    The EFD width/detail knobs live in the [efd] section (the ``Efd`` config, shared with pose etc.) —
    #    this stage reads them via the ``efd`` reference and re-selects the efd feature when detail changes. --
    PARAMS = (
        ParamSpec("enable", str, _DEFAULT_ENABLE, hot=False, bind=False, multi=True, expose="edit",
                  enum_fn=lambda: tuple(n for n, f in _REGISTRY.items() if not f.internal), doc="enabled feature list - restart to change"),
    )

    def __init__(self, config: Any = None, *, efd: Any = None, logger: Any = None) -> None:
        self._log = logger or NULL_LOGGER
        g = config.get if config is not None else (lambda k, d=None: d)
        names = [s.strip() for s in str(g("enable", _DEFAULT_ENABLE)).split(",") if s.strip()]
        usable = []
        for n in names:                     # warn+skip (never crash): a stale conf keeps the stage alive
            f = _REGISTRY.get(n)
            if f is None:
                self._log.warning(f"[features] enable: unknown feature '{n}' - skipped (available: "
                                  f"{', '.join(sorted(k for k, v in _REGISTRY.items() if not v.internal))})")
            elif f.internal:
                self._log.warning(f"[features] enable: '{n}' is system-internal (reid) - skipped")
            else:
                usable.append(f)
        self._features = usable
        # the [efd] config the `efd` feature reads (bins + the live detail knob) — the SAME shared spectrum
        # pose / symmetry / grouper use. A live detail change bumps efd.version → we re-select next tick.
        self._efd = efd if efd is not None else SimpleNamespace(bins=12, detail=0.5, version=0)
        self._efd_ver = getattr(self._efd, "version", 0)
        self._cache: dict[int, tuple[Any, dict]] = {}   # id -> (footprint_contour identity, {name: [floats]})
        self._comp_cache: dict[int, tuple] = {}         # gid -> (blend identity, ANNOTATED compound snap)
        self._log.info(f"[features] {', '.join(n.name for n in self._features) or '(none)'}")

    def __call__(self, ctx: Any, scene: Any) -> None:
        ver = getattr(self._efd, "version", 0)              # the [efd].detail knob changed (live) → re-select
        if ver != self._efd_ver:                            #   every object's efd on this tick (reassign, don't
            self._efd_ver, self._cache = ver, {}            #   mutate a dict a concurrent tick may be iterating)
            self._comp_cache = {}
        for old, new in scene.recent_rewrites():            # a re-id moves an object → re-key the cache
            rekey(old, new, self._cache)
        live = set()
        for obj in scene.objects():
            live.add(obj.id)
            if getattr(obj, "ignored", False):              # edge/arm residue — never described (nor emitted)
                continue
            fp = obj.footprint_contour                      # features are FOOTPRINT (shape) properties
            if fp is None:
                continue                                    # un-captured → nothing to describe yet
            cached = self._cache.get(obj.id)
            if cached is not None and cached[0] is fp:      # same captured shape → reuse; DON'T re-annotate
                continue                                    #   (invariant; re-annotating would rebuild the snap)
            g = getattr(ctx, "gray", None)
            fa = float(g.shape[0] * g.shape[1]) if g is not None else None
            feats, wire, full = self._compute(fp, fa)
            self._cache[obj.id] = (fp, feats)
            scene.set(obj.id, "wire_contour", wire)         # the PERCEPTUAL outline every output emits
            scene.set(obj.id, "efd_full", full)             # the COMPLETE ± spectrum — webgui SSE only, so the
            scene.annotate(obj.id, "features", feats)       # annotate ON CHANGE only    inspector can draw the
            #                                                 detail-REJECTED bins dim under the wire's survivors
        for i in [i for i in self._cache if i not in live]:  # drop departed ids
            self._cache.pop(i, None)
        self._describe_compounds(ctx, scene)

    def _describe_compounds(self, ctx: Any, scene: Any) -> None:
        """The COMPOUND pass (grouper active, over budget): the SAME computation, same knobs, on the
        mass-scaled blend — the contour carries the whole group's mass, so ``size`` and every other
        descriptor fall out of the normal path with no special case. A synthetic id is not in the
        scene store (``scene.annotate`` would raise), so the results live ON the snap: the compound
        is republished annotated via ``set_fused``. Cached on blend identity exactly like objects —
        the grouper reuses the same snap while a group is stable, so a settled scene costs nothing.
        ``efd_full`` rides along too: the webgui's FEATURES view follows the wire (owner decision),
        so the inspector needs the compound's complete spectrum to dim the detail-rejected bins."""
        claimed, comps = scene.fused()
        if not comps:
            if self._comp_cache:
                self._comp_cache = {}
            return
        g = getattr(ctx, "gray", None)
        fa = float(g.shape[0] * g.shape[1]) if g is not None else None
        out: list = []
        for snap in comps:
            fp = snap.footprint_contour
            cached = self._comp_cache.get(snap.id)
            if cached is not None and cached[0] is fp:      # same blend → the same annotated snap
                out.append(cached[1])
                continue
            feats, wire, full = self._compute(fp, fa)
            new = replace(snap, annotations=MappingProxyType({"features": feats}),
                          wire_contour=readonly(wire), efd_full=full)
            self._comp_cache[snap.id] = (fp, new)
            out.append(new)
        scene.set_fused(claimed, tuple(out))
        live_g = {s.id for s in comps}
        self._comp_cache = {k: v for k, v in self._comp_cache.items() if k in live_g}

    # EFD-native features select their own components — they read the RAW footprint; every other
    # (geometric) feature reads the PERCEPTUAL reconstruction, so the numbers describe the same shape
    # the efd spectrum ships and the outputs draw.
    _SPECTRAL = frozenset({"efd", "reflectional_symmetry"})

    def _compute(self, contour: np.ndarray, frame_area: "float | None" = None) -> "tuple[dict, Any, Any]":
        bins = int(getattr(self._efd, "bins", 12))
        detail = float(getattr(self._efd, "detail", 0.5))
        wire = perceptual_contour(contour, bins, detail)    # inverse EFD of the detail-surviving components
        xg = _feat_input(wire if wire is not None else contour, self._efd, frame_area)
        xs = _feat_input(contour, self._efd, frame_area)
        out: dict[str, list] = {}
        for f in self._features:
            try:
                out[f.name] = [float(v) for v in f.fn(xs if f.name in self._SPECTRAL else xg)]
            except Exception as e:                          # noqa: BLE001 — a bad contour must not kill the tick
                self._log.warning(f"[features] {f.name} failed: {e}")
        return out, wire, self._full_spectrum(contour, bins)

    @staticmethod
    def _full_spectrum(contour: np.ndarray, bins: int) -> "tuple | None":
        """The COMPLETE ± magnitude spectrum BEFORE the detail cut — ``((|c₊₁|…|c₊ₙ|), (|c₋₁|…|c₋ₙ|))``,
        normalized to the loudest single bin (the SAME 0 dB reference the efd annotation's triples use, so
        the inspector can overlay them). Detail-independent by construction. This is an OPERATOR diagnostic:
        it rides the webgui snapshot/SSE only (``ObjectSnap.efd_full``) and never touches OSC/TUIO2 — the
        art wire stays perceptual (survivors only)."""
        if bins < 1 or len(contour) < 3:
            return None
        c, _ = efd_coeffs(contour, bins)                    # memoized — the efd feature already paid for this
        _ks, mpos, mneg = component_magnitudes(c, bins)
        amax = max(float(mpos.max()), float(mneg.max()))
        if amax <= _EPS:
            return None                                     # degenerate (a point) → nothing to show
        return (tuple(round(float(v) / amax, 4) for v in mpos),
                tuple(round(float(v) / amax, 4) for v in mneg))
