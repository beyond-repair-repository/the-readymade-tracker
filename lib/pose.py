"""``pose`` — one-shot placement at settle. No live tracking, no mask matching.

An object's pose is ``(x, y, θ)``. pose is TOLD, never decides:

  * SETTLING (a fresh mint, no footprint yet) → the provisional blob centroid, θ = 0.
  * RESTING, first time it settles          → this shape is the object's REFERENCE. Remember its
                                               footprint as the canonical shape; θ = the shape's OWN
                                               orientation, in the 0-is-up clockwise convention:
                                               its dominant ASYMMETRY — the handle — when one exists
                                               (``asymmetry_axis_up``; look-alike objects agree and
                                               the arrow points where the handle points), else its
                                               principal (elongation) axis (``principal_axis_up``).
                                               So the reported θ is SHAPE-ANCHORED: 0 means "anchor
                                               feature pointing screen-up". Near-isotropic shapes
                                               (no meaningful axis) anchor at 0.
  * RESTING, re-settled (edge into RESTING)  → measure the rotation SINCE CAPTURE once by EFD phase
                                               correlation — the canonical footprint OUTLINE vs the
                                               freshly-settled outline — and report axis + rotation
                                               (the anchor is a constant offset; the differential
                                               fit carries all the robustness).
  * RESTING steady / MOVED / HIDDEN          → HOLD the last pose (frozen — partial update: outputs
                                               keep emitting the last settled value while a region
                                               is in motion, and update only when it settles).

The fit runs at most ONCE per settle (on the not-RESTING → RESTING edge), against the object's OWN
outline — never per frame. There is no smoothing, Kalman, median, or hold band by construction:
a settled object is unoccluded and still, so a single EFD phase-correlation fit is exact.

Rotationally symmetric shapes (a circle, a ring, a disc) have NO measurable orientation — the
correlation matches at every angle, so a fitted θ would be pure noise. Such objects are detected
ONCE from their canonical contour (``efd.rotation_information`` under ``symmetry_lock``) and their
θ is held at a constant 0; the settle fit is skipped entirely.

Pose-private state (per object id): the canonical ``(footprint_contour, centroid)`` captured at the
first settle, its principal-axis anchor, the symmetry-lock verdict, and the last status (to detect
the settle edge). Re-keyed on a ``rewrite_id``, dropped when the object dies.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from lib.efd import rotation_by_correlation, rotation_information
from lib.logger import NULL_LOGGER
from lib.params import ParamSpec, bind_params
from lib.scene import Status, rekey

_CONF_MIN = 0.4          # accept the settle rotation fit at/above this score; below → hold the last θ
_SYM_LOCK = 0.005        # rotation-information floor: below it a shape is rotationally symmetric →
#                          θ locked to 0 (real captures: circles/rings/discs ≤0.004, others ≥0.009)
_AXIS_MIN_ANISO = 1e-3   # relative second-moment anisotropy under which a shape has NO meaningful
#                          principal axis (a circle, a regular polygon) → anchor at 0 (arrow up)
_ASYM_UNIQ_MIN = 0.06    # the far tip must stand out from the OPPOSITE sector by this fraction of
#                          its radius to count as a unique bump (a cup's handle ≈ 0.14; an ellipse
#                          ≈ 0 — both ends tie, so elongated-but-symmetric shapes keep the axis)


def principal_axis_up(contour: Any) -> float:
    """The contour's OWN orientation: its principal (elongation) axis from the second central
    moments, expressed in the tracker's angle convention (0 = screen-up, clockwise-positive in the
    y-down frame), choosing the axis END nearest up → result in (−π/2, π/2]. An axis is a LINE
    (defined mod π), so the end pick is a one-time convention — the differential settle fit keeps it
    consistent afterwards. A near-isotropic shape (relative anisotropy < ``_AXIS_MIN_ANISO``: a
    circle, a square) has no meaningful axis → 0.0 (behaves like the old placement anchor)."""
    if contour is None or len(contour) < 3:
        return 0.0
    m = cv2.moments(np.asarray(contour, np.float32).reshape(-1, 1, 2))
    denom = m["mu20"] + m["mu02"]
    aniso = math.hypot(2.0 * m["mu11"], m["mu20"] - m["mu02"])
    if m["m00"] == 0.0 or denom <= 0.0 or aniso < _AXIS_MIN_ANISO * denom:
        return 0.0
    alpha = 0.5 * math.atan2(2.0 * m["mu11"], m["mu20"] - m["mu02"])   # major axis from +x, (−π/2, π/2]
    phi = alpha + math.pi / 2.0                        # → the 0-is-up clockwise convention
    if phi > math.pi / 2.0:
        phi -= math.pi                                 # pick the end nearest screen-up
    return phi


def asymmetry_axis_up(contour: Any) -> "float | None":
    """The direction of the contour's dominant ASYMMETRY — for a handled object, the handle — in
    the 0-is-up clockwise convention, or ``None`` when no unique bump exists.

    The farthest boundary point from the centroid marks the bump, and it counts only if it stands
    out from the OPPOSITE sector (the uniqueness gate): a 2-fold shape (an ellipse, a stick) ties
    at both ends and falls through to the principal axis. The tip ARC — every near-max point in
    the bump's sector — is vector-averaged so a one-pixel spike can't steer the anchor. The
    person-facing rationale: look-alike objects AGREE on zero, and the arrow points where the
    handle points."""
    if contour is None or len(contour) < 3:
        return None
    pts = np.asarray(contour, np.float32).reshape(-1, 2)
    m = cv2.moments(pts.reshape(-1, 1, 2))
    c = (np.array([m["m10"], m["m01"]], np.float32) / m["m00"]) if m["m00"] else pts.mean(axis=0)
    d = pts - c
    r = np.hypot(d[:, 0], d[:, 1])
    i = int(np.argmax(r))
    rmax = float(r[i])
    if rmax <= 0.0:
        return None
    u = d[i] / rmax                                    # unit direction of the tip
    cosang = (d @ u) / np.maximum(r, 1e-9)             # each vertex's angular closeness to the tip
    opp = r[cosang < -0.866]                           # the sector within 30° of OPPOSITE
    if opp.size == 0 or (rmax - float(opp.max())) / rmax < _ASYM_UNIQ_MIN:
        return None                                    # two-ended / ambiguous → caller falls back
    near = (cosang > 0.866) & (r > 0.75 * rmax + 0.25 * float(np.median(r)))
    tip = d[near].sum(axis=0) if near.any() else d[i]
    return float(math.atan2(tip[0], -tip[1]))          # (dx, −dy) → 0-is-up, clockwise-positive


class Pose:
    """The pose stage: one-shot (x, y, θ) at settle; hold otherwise."""

    name = "pose"

    PARAMS = (
        ParamSpec("measure_rotation", bool, True, attr="_measure_rot",
                  apply=lambda self, v: setattr(self, "_measure_rot", bool(v)),
                  doc="Measure each object's absolute rotation at settle (the EFD phase-correlation fit - the priciest per-settle work). OFF: placement, ids and the emitted outline are unaffected and theta is reported as 0, which is CORRECT - the footprint is re-captured at the object's current orientation, so the turn is already baked into the shape every consumer draws"),
        ParamSpec("conf_min", float, _CONF_MIN, lo=0.0, hi=1.0, attr="_conf_min",
                  doc="Min settle-fit score (EFD phase-correlation confidence) to accept the rotation; below -> hold the last theta"),
        ParamSpec("max_fits_per_frame", int, 1, lo=1, attr="_max_fits",
                  doc="Cap on the rotation fits per frame; extra simultaneous re-settles defer to the next frame(s), bounding the worst-frame spike (a settled object is still, so a 1-2 frame theta delay is imperceptible)"),
        ParamSpec("symmetry_lock", float, _SYM_LOCK, lo=0.0, hi=1.0, attr="_sym_lock",
                  apply=lambda self, v: (setattr(self, "_sym_lock", float(v)), self._locked.clear()),
                  doc="Rotation-information floor: a shape whose contour carries less orientation signal than this (a circle / ring / disc) has no measurable rotation - its theta is locked to a constant 0 and the settle fit is skipped. 0 disables the lock"),
    )

    def __init__(self, config: Any = None, *, logger: Any = None) -> None:
        self._log = logger or NULL_LOGGER
        self._locked: dict[int, bool] = {}   # id → symmetry-lock verdict (before bind: apply clears it)
        bind_params(self, config)          # _conf_min / _max_fits / _sym_lock ← PARAMS
        self._canon: dict[int, tuple] = {}   # id → (footprint_contour, (cx, cy)) — the reference shape
        self._axis0: dict[int, float] = {}   # id → the canonical shape's principal-axis anchor (θ zero)
        self._last: dict[int, Status] = {}   # id → last status (to detect the settle edge)

    def __call__(self, ctx: Any, scene: Any) -> None:
        for old, new in scene.recent_rewrites():
            rekey(old, new, self._canon, self._axis0, self._last, self._locked)
        live = set()
        fits = 0                                           # expensive rotation fits done THIS frame (budget)
        for obj in scene.objects():
            live.add(obj.id)
            if getattr(obj, "ignored", False):             # edge/arm residue — spend no rotation fit on it
                continue
            st = obj.status
            if st is Status.RESTING and self._last.get(obj.id) is not Status.RESTING:
                # The not-RESTING → RESTING edge. A RE-settle (a canonical reference already exists) runs
                # the EFD rotation fit; a FIRST settle (no reference) is θ=0 with no fit. Budget only
                # the expensive re-fits: if this frame is already at the cap, DEFER by leaving _last at its
                # pre-RESTING value so the edge re-fires next frame — bounding the worst-frame spike (many
                # objects settling at once) without changing any θ (a settled object is still).
                refit = self._canon.get(obj.id) is not None or obj.pose_ref is not None
                if refit and (self._locked.get(obj.id) or not self._measure_rot):
                    refit = False                          # θ locked, or rotation off → no fit will run,
                    #                                        so it must neither defer nor spend budget
                if refit and fits >= self._max_fits:
                    continue                               # over budget → defer (do NOT update _last)
                self._place_at_settle(scene, obj)          # fit ONCE
                if refit:
                    fits += 1
            elif st is Status.SETTLING and obj.footprint is None:
                self._place_provisional(scene, obj)        # a fresh mint being placed for the first time
            # RESTING steady / MOVED / HIDDEN → hold the last pose (do nothing)
            self._last[obj.id] = st
        for d in (self._canon, self._axis0, self._last, self._locked):  # drop state for dead objects
            for i in [i for i in d if i not in live]:
                d.pop(i, None)

    def _place_provisional(self, scene: Any, obj: Any) -> None:
        """A brand-new object mid-settle has no footprint yet — show it at its blob centroid, θ = 0.
        Outputs that gate on a committed footprint won't emit it until it actually settles (RESTING)."""
        obs = obj.observation
        c = obs.centroid if obs is not None else None
        if c is None:
            return
        target = (float(c[0]), float(c[1]), 0.0)
        if obj.pose != target:
            scene.set(obj.id, "pose", target)

    def _place_at_settle(self, scene: Any, obj: Any) -> None:
        """The object just settled: place it at its footprint centroid and measure its absolute
        rotation ONCE by fitting the canonical footprint onto the whole freshly-settled silhouette.

        The measured angle is the object's ABSOLUTE orientation vs its original (canonical) shape — it
        goes to ``theta_cumulative`` (what OUTPUTs report). The COMMITTED pose angle is 0: the footprint
        (and patch) are captured FRESH at the current orientation, so a consumer drawing the captured
        shape must not rotate it again — that would DOUBLE the rotation. The turn is already baked in;
        θ is the reported number."""
        c = obj.footprint_centroid
        if c is None:
            return
        cx, cy = float(c[0]), float(c[1])
        canon = self._canon.get(obj.id)
        if canon is None and obj.pose_ref is not None:     # RELOCATED: identity re-seeded the reference
            canon = obj.pose_ref                           #   through the gone-pool → restore it so θ
            self._canon[obj.id] = canon                    #   stays absolute vs the ORIGINAL orientation
        if not self._measure_rot:
            # Rotation measurement off: θ = 0, and that is CORRECT, not a degradation. The footprint is
            # re-captured at the object's CURRENT orientation on every settle, and the emitted outline is
            # placed with pose[2] (always 0) — so the turn is already baked into the shape every consumer
            # draws. Reporting a held θ would name a rotation the outline ALREADY carries, double-counting
            # it for anyone who rotates by it, and would go stale the moment the object is turned.
            if canon is None:                              # still seed the reference, so flipping the knob
                canon = (obj.footprint_contour, (cx, cy))  #   back on measures against the ORIGINAL shape
                self._canon[obj.id] = canon                #   instead of re-anchoring to wherever it now lies
                scene.set(obj.id, "pose_ref", canon)
            theta, score = 0.0, 1.0
        elif canon is None:                                # genuine first settle → become the ref; θ = the
            canon = (obj.footprint_contour, (cx, cy))      #   shape's OWN axis (shape-anchored zero)
            self._canon[obj.id] = canon
            scene.set(obj.id, "pose_ref", canon)           # persist for a future relocation (written ONCE)
            locked = self._rotation_locked(obj.id, obj.footprint_contour)   # decide the lock at capture
            theta = 0.0 if locked else self._axis_of(obj.id, obj.footprint_contour)
            score = 1.0
        else:
            ref_contour, (rcx, rcy) = canon
            if self._rotation_locked(obj.id, ref_contour):
                theta, score = 0.0, 1.0                    # symmetric shape: θ is undefined → constant 0
            else:
                theta, score = self._measure_theta(obj, ref_contour, float(rcx), float(rcy), cx, cy)
                if score < self._conf_min:                 # a poor fit (occlusion / shape change) → hold θ
                    theta = float(obj.theta_cumulative or 0.0)   # the ACCUMULATED absolute angle — obj.pose[2]
                    #                                              is 0.0 by construction (committed θ is
                    #                                              always 0), so reading it would RESET the
                    #                                              reported orientation instead of holding it
        scene.set(obj.id, "pose", (cx, cy, 0.0))           # committed θ is 0: the captured shape already
        #                                                    sits at this orientation (no double-rotation)
        scene.set(obj.id, "pose_conf", float(score))
        if theta != obj.theta_cumulative:                  # the ABSOLUTE orientation OUTPUTs report (accumulates)
            scene.set(obj.id, "theta_cumulative", float(theta))

    def _axis_of(self, oid: int, ref_contour: Any) -> float:
        """The object's shape anchor in the 0-is-up clockwise convention, computed once and cached
        (also lazily after a relocation restores the canon through ``pose_ref``: the anchor is a
        pure function of that contour). ASYMMETRY FIRST: a unique bump (a cup's handle) defines
        zero, so look-alike objects agree and the reported θ points where the handle points; a
        shape with no unique bump (an ellipse, a stick) keeps the principal-axis anchor."""
        axis = self._axis0.get(oid)
        if axis is None:
            asym = asymmetry_axis_up(ref_contour)
            axis = asym if asym is not None else principal_axis_up(ref_contour)
            self._axis0[oid] = axis
        return axis

    def _rotation_locked(self, oid: int, ref_contour: Any) -> bool:
        """Whether this object's rotation is UNMEASURABLE (a rotationally symmetric shape). Decided
        once per object from its canonical contour and cached; a live ``symmetry_lock`` change clears
        the cache so every object is re-judged at its next settle."""
        locked = self._locked.get(oid)
        if locked is None:
            locked = (self._sym_lock > 0.0 and ref_contour is not None
                      and rotation_information(ref_contour) < self._sym_lock)
            self._locked[oid] = locked
        return locked

    def _measure_theta(self, obj: Any, ref_contour: Any, rcx: float, rcy: float,
                       cx: float, cy: float) -> "tuple[float, float]":
        """The reported absolute orientation (θ_rad, score): the shape's ANCHOR (asymmetry/handle, or
        principal axis — ``_axis_of``) plus the rotation SINCE CAPTURE, measured by EFD PHASE
        CORRELATION of the canonical outline against the fresh settled OUTLINE (``footprint_contour``):
        one FFT, continuous angle, no per-frame spike. The differential fit carries all the robustness;
        the anchor is a per-object constant offset so θ = 0 means "anchor feature screen-up" (the
        handle, for a handled object) instead of "as first placed". ``score`` is the
        correlation confidence (gated against ``conf_min`` by the caller). (The ``rcx/rcy/cx/cy`` seed
        args were the removed overlap fit's.)"""
        if ref_contour is None:
            return float(obj.theta_cumulative or 0.0), 0.0
        tgt = obj.footprint_contour                        # the fresh settled OUTLINE (not the mask)
        if tgt is None:
            return float(obj.theta_cumulative or 0.0), 0.0
        axis0 = self._axis_of(obj.id, ref_contour)
        # prev θ = tie hysteresis: a 2-fold shape (a bar) correlates ~equally at θ and θ+180°; prefer
        # the peak nearest the last reported orientation over a noise-picked flip. The hysteresis (and
        # the correlation) live in DELTA space — strip the anchor going in, add it back on the result.
        prev_delta = float(obj.theta_cumulative or 0.0) - axis0
        delta, score = rotation_by_correlation(ref_contour, tgt, prev_theta=prev_delta)
        return axis0 + delta, score
