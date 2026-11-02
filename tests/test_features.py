"""features stage + registry — the six seed shape scalars, computed at capture, cached per
footprint identity, annotated under "features". Values checked against known shapes."""

import math
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import readymade_tracker
from lib.config import build_config
from lib.features import Features, _feat_input, efd_coeffs, registry
from lib.scene import Detection, Scene


# --- known shapes (dense contours via findContours) ------------------------ #

def _contour(draw, size=240):
    img = np.zeros((size, size), np.uint8)
    draw(img)
    cnts, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return max(cnts, key=cv2.contourArea)


CIRCLE = _contour(lambda im: cv2.circle(im, (120, 120), 80, 255, -1))
SQUARE = _contour(lambda im: cv2.rectangle(im, (40, 40), (200, 200), 255, -1))
RECT = _contour(lambda im: cv2.rectangle(im, (20, 100), (220, 140), 255, -1))   # elongated
ELLIPSE = _contour(lambda im: cv2.ellipse(im, (120, 120), (90, 45), 0, 0, 360, 255, -1))


def _spiky(contour):
    return registry()["spikiness"].fn(_feat_input(contour, SimpleNamespace(bins=12, detail=0.5)))[0]


def _val(name, contour):
    return registry()[name].fn(_feat_input(contour, SimpleNamespace(efd_bins=12)))[0]


class _Cfg:
    def __init__(self, **kw):
        self._d = kw
    def get(self, k, d=None):
        return self._d.get(k, d)


def _snap(scene, oid):
    return {o.id: o for o in scene.snapshot().objects}[oid]


def _captured(scene, contour, **det):
    scene.begin_tick(0, ts=0.0)
    with scene.writing("setup"):
        oid = scene.mint(Detection((120.0, 120.0), (40, 40, 160, 160), 20000.0, None))
    with scene.writing("classify"):
        scene.set(oid, "footprint_contour", contour)
    return oid


# --------------------------------------------------------------------------- #
# registry + feature values on known shapes
# --------------------------------------------------------------------------- #

def test_person_facing_and_internal_features_registered():
    r = registry()
    person = {n for n, f in r.items() if not f.internal}
    assert person == {"circularity", "rectangularity", "solidity", "eccentricity",
                      "size", "lobe_count", "reflectional_symmetry", "spikiness", "efd"}
    # the reid shape block stays registered but SYSTEM-INTERNAL (never user-facing)
    assert {n for n, f in r.items() if f.internal} == {
        "ellipse_variance", "circle_variance", "defect_depth"}


def test_circle_values():
    assert _val("circularity", CIRCLE) > 0.85            # ~0.89 on a pixelated circle (staircase perim)
    assert _val("solidity", CIRCLE) > 0.95
    assert _val("circle_variance", CIRCLE) < 0.05        # ~0 for a true circle (internal, reid's)
    assert _val("eccentricity", CIRCLE) < 0.1            # STANDARD e: a circle is 0


def test_square_values():
    assert _val("rectangularity", SQUARE) > 0.95         # fills its min-area rect
    assert 0.70 < _val("circularity", SQUARE) < 0.82     # 4π·s²/(4s)² = π/4 ≈ 0.785
    assert _val("solidity", SQUARE) > 0.95
    assert _val("eccentricity", SQUARE) < 0.1            # isotropic too → 0


def test_elongated_rect_values():
    assert _val("eccentricity", RECT) > 0.9              # elongated → e approaches 1
    assert _val("rectangularity", RECT) > 0.9
    assert _val("circularity", RECT) < 0.6               # low for a long thin shape


def test_feature_spans_declared():
    assert registry()["circle_variance"].span == (0.0, 0.6)
    assert registry()["circularity"].invariant is True


# --------------------------------------------------------------------------- #
# the added geometric descriptors (survey §2 + fractal dimension)
# --------------------------------------------------------------------------- #

def test_ellipse_variance_measures_deviation_from_an_ellipse_not_a_circle():
    assert _val("ellipse_variance", CIRCLE) < 0.1        # a circle IS an ellipse → ~0
    # the discriminating case: an ELLIPSE is FAR from a circle but IS an ellipse
    assert _val("circle_variance", ELLIPSE) > 0.15       # circle_variance sees it as non-round
    assert _val("ellipse_variance", ELLIPSE) < 0.1       # ellipse_variance correctly sees a clean ellipse


def _radial(fn, n=600, cx=300, cy=300):
    """A closed contour from a radius function r(theta) — dense, so the reconstruction the feature
    actually reads is the thing under test rather than the sampling."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = fn(t)
    return np.stack([cx + r * np.cos(t), cy + r * np.sin(t)], 1).reshape(-1, 1, 2).astype(np.int32)


def _spikestar(k, inner, a=100.0, phase=0.0, seg=60):
    """A STRAIGHT-SIDED star: k outer vertices at ``a``, k inner at ``a*inner``, joined by straight
    edges. Apex sharpness is set by ``inner`` ALONE, independent of ``k`` — which a radial triangular
    wave does not give you (there the tip angle tightens as k rises, so 'sharper' and 'more points'
    become the same variable and no test can separate them)."""
    m = k * 2
    th = np.linspace(0, 2 * np.pi, m, endpoint=False) + phase
    r = np.where(np.arange(m) % 2 == 0, a, a * inner)
    v = np.stack([r * np.cos(th), r * np.sin(th)], 1)
    edge = [v[i] + (v[(i + 1) % m] - v[i]) * u
            for i in range(m) for u in np.linspace(0, 1, seg, endpoint=False)]
    return (np.array(edge) + 300).reshape(-1, 1, 2).astype(np.int32)


def test_spikiness_is_driven_by_SHARPNESS_not_by_the_number_of_points():
    # Huang's constraint: one or two very sharp spikes outrank many blunt ones.
    two_sharp = _spiky(_spikestar(2, 0.35))
    nine_blunt = _spiky(_radial(lambda t: 100 * (1 + 0.30 * np.cos(9 * t))))
    assert two_sharp > nine_blunt, f"two sharp {two_sharp:.3f} !> nine blunt {nine_blunt:.3f}"
    # and sharpness, not count, is what moves it: the SAME 5 points blunted must drop a long way
    assert _spiky(_spikestar(5, 0.30)) > 0.7
    assert _spiky(_spikestar(5, 0.85)) < 0.6


def test_spikiness_is_near_zero_for_round_shapes():
    assert _spiky(CIRCLE) < 0.15
    assert _spiky(ELLIPSE) < 0.25                    # an egg is not spiky, however elongated


def test_spikiness_counts_only_OUTWARD_turns():
    # Inward sharpness is concavity — solidity's dimension, not this one. Asserted on the estimator
    # itself: on a star, every OUTER vertex must register a positive turn and every INNER vertex
    # (equally sharp, but pointing the wrong way) must be clamped to zero. A shape-level comparison
    # cannot test this cleanly, because cutting a sharp notch into a disc also creates two sharp
    # OUTWARD shoulders at its rim.
    from lib.features import _turning
    c = _spikestar(5, 0.30).reshape(-1, 2).astype(float)
    rad = np.hypot(*(c - 300.0).T)
    turn = _turning(c, 24, 1.0 if np.sum(c[:, 0] * np.roll(c[:, 1], -1)
                                        - np.roll(c[:, 0], -1) * c[:, 1]) > 0 else -1.0)
    outer = turn[rad > 0.95 * rad.max()]
    inner = turn[rad < 1.05 * rad.min()]
    assert outer.max() > 0.5, "an outward apex must register a turn"
    assert inner.max() == 0.0, "an inward apex must be clamped away"


def test_spikiness_is_scale_and_rotation_invariant():
    base = _spiky(_spikestar(5, 0.30))
    assert abs(_spiky(_spikestar(5, 0.30, a=40.0)) - base) < 0.15      # 0.4x apparent size
    for deg in (13.0, 47.0, 90.0):
        v = _spiky(_spikestar(5, 0.30, phase=math.radians(deg)))
        assert abs(v - base) < 0.2, f"rotation {deg}deg: {v:.3f} vs {base:.3f}"


def test_size_reports_linear_extent_over_the_range_an_areal_fraction_wastes():
    """`size` is the ONE size definition. The areal fraction it replaced was correct and unusable:
    quadratic in size, so real tabletop objects (918..20,786 px^2 — a 22.6x spread) landed inside 2%
    of a 0..1 range and all read "about zero"."""
    from lib.features import size_fraction
    FA = 1024.0 * 1024.0

    def disc(px):
        r = math.sqrt(px / math.pi)
        t = np.linspace(0, 2 * math.pi, 400, endpoint=False)
        c = np.stack([500 + r * np.cos(t), 500 + r * np.sin(t)], 1)
        return _feat_input(c.reshape(-1, 1, 2).astype(np.int32), None, frame_area=FA)

    small, large = disc(1000), disc(20000)          # the measured range of real objects
    s_small = registry()["size"].fn(small)[0]
    s_large = registry()["size"].fn(large)[0]
    assert (large.area / FA) - (small.area / FA) < 0.03    # the areal fraction: inside 3% of 0..1
    assert s_large - s_small > 0.3                         # size: the SAME objects over ~a third of it
    #                                                         (0.346 at the 0.35 reference of 0.32.132)
    # linear in EXTENT, not area: 4x the area is 2x the size
    assert abs(registry()["size"].fn(disc(4000))[0] / registry()["size"].fn(disc(1000))[0] - 2.0) < 0.05
    assert registry()["size"].fn(disc(int(FA)))[0] == 1.0  # clamped at the top
    # the grouper reports a compound's mass through the SAME function, so both mean one thing
    assert size_fraction(large.area, FA) == s_large   # the polygon's MEASURED area, not the nominal


# --------------------------------------------------------------------------- #
# the stage
# --------------------------------------------------------------------------- #

def test_stage_annotates_captured_object():
    s = Scene()
    oid = _captured(s, CIRCLE)
    with s.writing("features"):
        Features(_Cfg(enable="circularity, solidity"))(None, s)
    ann = _snap(s, oid).annotations["features"]
    assert set(ann) == {"circularity", "solidity"}
    assert ann["circularity"][0] > 0.85 and len(ann["circularity"]) == 1


def test_stage_default_enable_is_the_seed_set():
    s = Scene()
    oid = _captured(s, SQUARE)
    with s.writing("features"):
        Features(_Cfg())(None, s)                        # no enable → the seed scalars
    assert set(_snap(s, oid).annotations["features"]) == {
        "circularity", "rectangularity", "solidity", "eccentricity",
        "size", "lobe_count", "reflectional_symmetry", "spikiness"}


def test_stage_skips_uncaptured_object():
    s = Scene()
    s.begin_tick(0, ts=0.0)
    with s.writing("setup"):
        oid = s.mint(Detection((50.0, 50.0), (30, 30, 40, 40), 1600.0, None))   # no footprint
    with s.writing("features"):
        Features(_Cfg())(None, s)
    assert "features" not in _snap(s, oid).annotations


def test_stage_skips_ignored_object():
    # an edge/border-residue blob flagged `ignored` (a hand/arm reaching in) is never described —
    # excluded from the derived features just as it is from every output.
    s = Scene()
    oid = _captured(s, CIRCLE)
    with s.writing("observe"):
        s.set(oid, "ignored", True)
    with s.writing("features"):
        Features(_Cfg())(None, s)
    assert "features" not in _snap(s, oid).annotations


def test_stage_recomputes_on_recapture():
    s = Scene()
    stage = Features(_Cfg())
    oid = _captured(s, CIRCLE)
    with s.writing("features"):
        stage(None, s)
    assert _snap(s, oid).annotations["features"]["rectangularity"][0] < 0.10   # circle -> ~0 since 0.31.x
    s.begin_tick(1, ts=1.0)
    with s.writing("classify"):
        s.set(oid, "footprint_contour", SQUARE)          # re-capture a new shape (new array identity)
    with s.writing("features"):
        stage(None, s)
    assert _snap(s, oid).annotations["features"]["rectangularity"][0] > 0.85   # now square


def test_stage_caches_stable_footprint():
    # same footprint array across ticks → the cache holds one entry, value unchanged
    s = Scene()
    stage = Features(_Cfg(enable="circularity"))
    oid = _captured(s, CIRCLE)
    with s.writing("features"):
        stage(None, s)
    v1 = _snap(s, oid).annotations["features"]["circularity"][0]
    s.begin_tick(1, ts=1.0)
    with s.writing("features"):
        stage(None, s)                                   # cache hit → no recompute, no re-annotate
    assert _snap(s, oid).annotations["features"]["circularity"][0] == v1
    assert len(stage._cache) == 1


def test_unknown_or_internal_names_warn_and_skip():
    # a stale conf (a removed feature name) must DEGRADE, not crash the stage; the internal reid
    # trio is likewise refused from the user surface.
    stage = Features(_Cfg(enable="circularity, bogus_metric, defect_depth"))
    assert [f.name for f in stage._features] == ["circularity"]


def test_stage_publishes_the_perceptual_wire_contour():
    s = Scene()
    oid = _captured(s, SQUARE)
    with s.writing("features"):
        Features(_Cfg(enable="circularity"))(None, s)
    wc = _snap(s, oid).wire_contour
    assert wc is not None and len(wc) >= 32              # the inverse-EFD outline outputs will emit
    assert wc.dtype == np.float32                        # SUB-PIXEL — int-snapping made upscaled outlines stairs


def test_stage_publishes_the_full_spectrum():
    # efd_full: the COMPLETE ± magnitude spectrum (pre-cut), loudest-bin-normalized — the webgui
    # inspector's dim "rejected bins". Detail-independent: the cut only chooses survivors.
    s = Scene()
    oid = _captured(s, SQUARE)
    with s.writing("features"):
        Features(_Cfg(enable="circularity"))(None, s)
    full = _snap(s, oid).efd_full
    assert full is not None and len(full) == 2
    pos, neg = full
    assert len(pos) == len(neg) == 12                    # one magnitude per ± bin (bins=12)
    allv = list(pos) + list(neg)
    assert max(allv) == 1.0                              # normalized to the loudest single bin (0 dB)
    assert all(0.0 <= v <= 1.0 for v in allv)


def test_perceptual_contour_respects_the_detail_knob():
    from lib.efd import perceptual_contour
    from lib.features import _feat_input
    lo = perceptual_contour(SQUARE, 12, 0.0)             # detail 0 → only the loudest bin → an ellipse-oid
    hi = perceptual_contour(SQUARE, 12, 1.0)             # detail 1 → the full kept tail → near the square
    assert lo is not None and hi is not None
    rect_lo = registry()["rectangularity"].fn(_feat_input(lo, None))[0]
    rect_hi = registry()["rectangularity"].fn(_feat_input(hi, None))[0]
    assert rect_hi > 0.8                                 # full detail reconstructs the box
    assert rect_lo < rect_hi                             # low detail is rounder — features follow the knob


# --------------------------------------------------------------------------- #
# EFD — the complex vector feature: SIGNED-BIN TRIPLES (±bin, amplitude, phase),
# sparse over the budget-kept harmonics + the adaptive detail knob.
# --------------------------------------------------------------------------- #

def _efd(contour, detail=1.0, bins=8):
    return registry()["efd"].fn(_feat_input(contour, SimpleNamespace(bins=bins, detail=detail)))


def _triples(v):
    """The flat EFD list → {signed_bin: (amplitude, phase)}."""
    return {v[i]: (v[i + 1], v[i + 2]) for i in range(0, len(v), 3)}


def test_efd_registered_and_triple_structure():
    assert "efd" in registry()
    v = _efd(SQUARE, detail=1.0, bins=8)
    assert len(v) % 3 == 0 and 3 <= len(v) <= 6 * 8      # (signed_bin, amp, phase) per SURVIVING ± bin
    bins = v[0::3]
    assert all(abs(b) >= 1 for b in bins) and len(set(bins)) == len(bins)   # distinct signed bins, |bin| ≥ 1
    assert abs(v[0]) == 1.0                              # the first (lowest-harmonic) surviving bin is the fundamental
    assert max(v[1::3]) == pytest.approx(1.0)            # amplitudes ÷ the loudest BIN (searched) → the max bin = 1.0


def test_efd_circle_is_pure_fundamental():
    t = _triples(_efd(CIRCLE, detail=1.0, bins=8))       # a circle = a single winding of the fundamental
    vals = sorted((a for _b, (a, _p) in t.items()), reverse=True)
    assert vals[0] > 0.95                                # ONE bin (the single winding) carries ~all the energy
    assert len(vals) == 1 or vals[1] < 0.05             # everything else (opposite winding + higher harmonics) ~0


def test_efd_square_has_odd_harmonics_only():
    t = _triples(_efd(SQUARE, detail=1.0, bins=8))       # 4-fold symmetry → energy at 3,5,7; ~0 at 2,4,6
    pair = lambda k: t.get(float(k), (0, 0))[0] + t.get(float(-k), (0, 0))[0]
    assert pair(3) > 0.05                                # harmonic 3 is the strong one
    assert pair(2) < 1e-2 and pair(4) < 1e-2            # even harmonics vanish


def test_efd_amplitudes_rotation_invariant_phase_is_not():
    sq = SQUARE.reshape(-1, 2).astype(float)
    c = sq.mean(0)
    R = np.array([[np.cos(0.5), -np.sin(0.5)], [np.sin(0.5), np.cos(0.5)]])
    rot = ((sq - c) @ R.T + c).astype(np.int32).reshape(-1, 1, 2)
    a, b = _efd(SQUARE, bins=8), _efd(rot, bins=8)
    assert np.allclose(a[1::3], b[1::3], atol=0.05)      # |c| amplitudes unchanged by a whole-shape rotation
    assert not np.allclose(a[2::3], b[2::3], atol=0.05)  # phases DO rotate (the info a magnitude spectrum drops)


def test_efd_detail_threshold_is_shape_adaptive():
    # the detail knob is an adaptive dB THRESHOLD, not a fixed count: deeper detail keeps ≥ bins, and
    # DIFFERENT shapes keep DIFFERENT numbers at the same detail (the count reflects the shape).
    coarse = _efd(SQUARE, detail=0.0, bins=12)           # cut at 0 dB → only the single loudest BIN survives
    assert len(coarse) == 3 and abs(coarse[0]) == 1.0    # one component, the fundamental (+1 or −1)
    deep = _efd(SQUARE, detail=1.0, bins=12)             # deeper cut → at least as many, and more than coarse
    assert len(deep) >= len(coarse) and len(deep) > 6
    # a circle concentrates energy in the fundamental → keeps FEWER bins than a busy square at the same detail
    assert len(_efd(CIRCLE, detail=0.6, bins=12)) < len(_efd(SQUARE, detail=0.6, bins=12))


def test_efd_degenerate_contour_is_empty():
    assert _efd(np.array([[[5, 5]], [[5, 5]]], np.int32)) == []   # < 3 pts → no descriptor, no crash


def _kg_reference(poly, maxK):
    """Independent from-scratch Kuhl-Giardina (the C++ SpectralProcessor::computeKuhlGiardina) — raw
    vertices, no resample — to pin efd_coeffs to the canonical algorithm the sonifier shares."""
    N = len(poly)
    dx = [poly[(i + 1) % N][0] - poly[i][0] for i in range(N)]
    dy = [poly[(i + 1) % N][1] - poly[i][1] for i in range(N)]
    dt = [max((dx[i] ** 2 + dy[i] ** 2) ** 0.5, 1e-12) for i in range(N)]
    cum = [0.0]
    for v in dt:
        cum.append(cum[-1] + v)
    T = cum[-1]
    out = {}
    for k in range(1, maxK + 1):
        a = b = c = d = 0.0
        for i in range(N):
            p0, p1 = 2 * np.pi * k * cum[i] / T, 2 * np.pi * k * cum[i + 1] / T
            dc, ds = np.cos(p1) - np.cos(p0), np.sin(p1) - np.sin(p0)
            a += dx[i] / dt[i] * dc; b += dx[i] / dt[i] * ds
            c += dy[i] / dt[i] * dc; d += dy[i] / dt[i] * ds
        cc = T / (2.0 * k * k * np.pi * np.pi)
        a, b, c, d = cc * a, cc * b, cc * c, cc * d
        out[k], out[-k] = complex(a + d, c - b) / 2, complex(a - d, b + c) / 2
    return out


def test_efd_coeffs_match_kuhl_giardina_reference_on_raw_vertices():
    rng = np.random.default_rng(7)                       # an UNEVEN-spacing polygon exercises no-resample + centroid
    th = np.sort(rng.uniform(0, 2 * np.pi, 11)); r = rng.uniform(20, 60, 11)
    poly = np.column_stack([200 + r * np.cos(th), 150 + r * np.sin(th)])
    c, _ = efd_coeffs(poly, 8)
    ref = _kg_reference([tuple(p) for p in poly], 8)
    assert max(abs(c[n] - ref[n]) for n in list(range(1, 9)) + list(range(-8, 0))) < 1e-9


def test_efd_centroid_is_arc_length_weighted_not_point_mean():
    rng = np.random.default_rng(3)                       # dense vertices on one side would bias a point-mean
    th = np.sort(rng.uniform(0, 2 * np.pi, 13)); r = rng.uniform(15, 55, 13)
    poly = np.column_stack([120 + r * np.cos(th), 90 + r * np.sin(th)])
    _, centroid = efd_coeffs(poly, 4)
    loop = np.vstack([poly, poly[0]])
    seg = np.hypot(*np.diff(loop, axis=0).T)
    arclen = ((loop[:-1] + loop[1:]) * 0.5 * seg[:, None]).sum(0) / seg.sum()
    assert np.allclose(centroid, arclen, atol=1e-9)      # matches the reference DC
    assert not np.allclose(centroid, poly.mean(0), atol=1.0)   # and is genuinely different from point-mean


def test_efd_detail_hot_setter_invalidates_cache():
    from lib.efd import Efd
    s = Scene()
    efd = Efd(_Cfg(bins=8, detail=1.0))
    stage = Features(_Cfg(enable="efd"), efd=efd)
    oid = _captured(s, SQUARE)
    with s.writing("features"):
        stage(None, s)
    assert len(_snap(s, oid).annotations["features"]["efd"]) > 6        # detail 1.0 → several harmonics survive
    efd._set_detail(0.0)                                 # the live [efd].detail knob → bumps efd.version
    s.begin_tick(1, ts=1.0)
    with s.writing("features"):
        stage(None, s)                                   # version changed → Features clears its cache + recomputes
    assert len(_snap(s, oid).annotations["features"]["efd"]) == 3       # detail 0 → cut at 0 dB → only the loudest bin


# --------------------------------------------------------------------------- #
# wiring (section presence enables the stage; schema coerces)
# --------------------------------------------------------------------------- #

def test_features_enabled_by_section_presence():
    cfg = build_config({"features": {"enable": "circularity"}, "efd": {"bins": "16"}},
                       schemas=readymade_tracker.CONF_SCHEMAS)
    assert cfg.is_enabled("features")
    assert cfg.get("features")["enable"] == "circularity"
    assert cfg.get("efd")["bins"] == 16                  # [efd] owns the harmonic width now (coerced to int)


def test_features_absent_section_disabled():
    assert not build_config({"input": {"source": "video"}}, schemas=readymade_tracker.SCHEMAS).is_enabled("features")


def test_no_features_flag_disables_a_present_section():
    ov = readymade_tracker._cli_overrides(readymade_tracker._parse_args(["--no-features"]))
    assert ov["features"] == {"enabled": "false"}
    assert not build_config({"features": {"enable": "circularity"}},
                            schemas=readymade_tracker.SCHEMAS, overrides=ov).is_enabled("features")


# --------------------------------------------------------------------------- #
# rotation_by_correlation — EFD phase-correlation rotation fit (pose's fast path)
# --------------------------------------------------------------------------- #

def _rot(pts, deg):
    th = np.radians(deg)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return (pts - pts.mean(0)) @ R.T


_BOOT = np.array([[0, 0], [60, 0], [60, 20], [25, 20], [25, 70], [10, 70], [10, 20], [0, 20]], float)


def test_rotation_by_correlation_recovers_known_angle():
    from lib.efd import rotation_by_correlation
    for deg in (0, 17, 45, 90, 133, 200, 271, 349):
        # a DIFFERENT start vertex + noise → the phase-correlation must still recover ψ (not a trivial phase)
        tgt = np.roll(_rot(_BOOT, deg), 3, axis=0) + np.random.default_rng(deg).normal(0, 0.4, _BOOT.shape)
        th, score = rotation_by_correlation(_BOOT, tgt)
        got = np.degrees(th)
        err = min((got - deg) % 360, (deg - got) % 360)
        assert err < 2.0, f"{deg}° -> {got:.1f}° (err {err:.1f}°)"
        assert score > 0.9


def test_rotation_by_correlation_accepts_cv2_contour_shape():
    from lib.efd import rotation_by_correlation
    ref = _BOOT.reshape(-1, 1, 2).astype(np.int32)             # cv2 (N,1,2)
    tgt = _rot(_BOOT, 30).reshape(-1, 1, 2).astype(np.int32)
    th, score = rotation_by_correlation(ref, tgt)
    err = min((np.degrees(th) - 30) % 360, (30 - np.degrees(th)) % 360)
    assert err < 2.0 and score > 0.9


def test_rotation_by_correlation_degenerate_is_zero():
    from lib.efd import rotation_by_correlation
    assert rotation_by_correlation(np.array([[0, 0], [1, 1]], float), _BOOT) == (0.0, 0.0)
    assert rotation_by_correlation(np.zeros((0, 2)), _BOOT) == (0.0, 0.0)


# --------------------------------------------------------------------------- #
# new features: lobe_count · defect_depth · reflectional_symmetry
# --------------------------------------------------------------------------- #

def _poly_contour(pts, size=280):
    img = np.zeros((size, size), np.uint8)
    cv2.fillPoly(img, [np.asarray(pts, np.int32)], 255)
    cnts, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return max(cnts, key=cv2.contourArea)


def _regular(n, cx=140, cy=140, r=100):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([cx + r * np.cos(t), cy + r * np.sin(t)], 1)


def _star(p=5, cx=140, cy=140, r1=105, r2=46):
    t = np.linspace(-np.pi / 2, 1.5 * np.pi, 2 * p, endpoint=False)
    rr = np.where(np.arange(2 * p) % 2 == 0, r1, r2)
    return np.stack([cx + rr * np.cos(t), cy + rr * np.sin(t)], 1)


def test_lobe_count_reads_prong_count():
    assert _val("lobe_count", _poly_contour(_regular(3))) == 3     # triangle
    assert _val("lobe_count", _poly_contour(_regular(6))) == 6     # hexagon
    assert _val("lobe_count", _poly_contour(_star(5))) == 5        # 5-point star
    assert _val("lobe_count", CIRCLE) <= 1                         # a disc has no lobes


def test_defect_depth_deep_for_star_zero_for_convex():
    assert _val("defect_depth", CIRCLE) < 0.05                     # convex → ~0
    assert _val("defect_depth", SQUARE) < 0.05
    assert _val("defect_depth", _poly_contour(_star(5))) > 0.4     # deep spikes


def test_reflectional_symmetry_ranks_symmetry():
    sym = _val("reflectional_symmetry", _poly_contour(_regular(6)))
    asym = _val("reflectional_symmetry",
                _poly_contour([(40, 70), (230, 45), (200, 160), (110, 235), (60, 150)]))  # irregular pentagon
    assert sym > 0.95                                              # near-perfect mirror match
    assert asym < sym                                             # asymmetric shape scores lower
