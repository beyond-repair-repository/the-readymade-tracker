"""``classify`` — the per-object comparison → status; capture at settle.

Reads each object's immutable ``Observation`` (observe wrote it) and applies the 4-state
transition table over the two spatial signals ``{residual, missing}`` on the ONE scale ``k``
(``R = M = k·area``) plus the **stationarity** settle-trigger (the mask is stable
frame-over-frame: ``xor_prev < k·area``). The capture gate is purely **stable for N seconds →
capture the footprint** — each object settles on its OWN timer, so a quiet object updates while
another is still being moved (partial update). Pure comparison — the only cv2 is at capture
(rasterizing the new footprint's contour); the transition logic is mask-free.

The load-bearing line is **RESTING → else → RESTING**: a settled object satisfies
``!off_footprint`` AND ``!rest_hidden`` so no branch fires and it stays RESTING by falling
through — its pose is never re-measured while at rest (no wobble).

N (settle) and TTL (carry) are SECONDS of wall-clock ``ts``. HIDDEN coast + the TTL/GONE eviction live in identity; here HIDDEN only re-acquires
(reappeared → SETTLING; off-footprint → MOVED).
"""

from __future__ import annotations

from typing import Any

import cv2

from lib.logger import NULL_LOGGER
from lib.params import ParamSpec, bind_params
from lib.scene import Status

_K = 0.05                # single-sourced from [observe].k in build()
_SETTLE_SECONDS = 0.3    # the capture debounce (N) — stable this long → capture
_MOVE_DEBOUNCE = 0.1     # RESTING→MOVED entry debounce (s) — off_footprint must persist this long,
#                          so a 1-frame occlusion-bridge residual spike can't flicker a resting object
_HOLD_MAX = 2.0          # max seconds a RESTING object is pinned under a moving mass — a watchdog so
#                          persistent motion over a footprint can't hold a stale husk forever
_CONFIRM_SECONDS = 1.0   # persist this long after mint → `confirmed`; until then withheld from ALL
_GLOBAL_QUIET = 1.5      # [commit_mode=global] the table must be this quiet, this long, to commit
_GLOBAL_FRAC = 0.002     # moving-pixel fraction of the WHOLE frame below which the table is 'idle'
#                          (the same scale identity uses for its table-quiet test)
#                          outputs (OSC/TUIO2/the output overlay), so a hover blip never leaks out


class Classify:
    """The classify stage: Observation → status transition, with capture at SETTLING→RESTING."""

    name = "classify"

    # -- live parameters, declared WITH the stage (the renamed attrs _settle/_move_debounce are owned
    #    here). `k` is single-sourced from [observe].k (injected in build), NOT a [classify] live param →
    #    read directly below, not in PARAMS. The HIDDEN coast/eviction is identity's (one bared-anchored
    #    memory clock, [identity].pool_ttl_seconds) — classify no longer owns a ttl_seconds. --
    PARAMS = (
        ParamSpec("settle_seconds", float, _SETTLE_SECONDS, lo=0.0, attr="_settle", doc="dwell before SETTLING->RESTING"),
        ParamSpec("move_debounce_seconds", float, _MOVE_DEBOUNCE, lo=0, attr="_move_debounce",
                  doc="Seconds off-footprint must persist for RESTING to become MOVED"),
        ParamSpec("hold_max_seconds", float, _HOLD_MAX, lo=0, attr="_hold_max",
                  doc="Max seconds a RESTING object stays pinned under a moving mass before the hold releases"),
        ParamSpec("commit_mode", str, "partial", enum=("partial", "global", "manual"), attr="_commit_mode",
                  apply=lambda self, v: (setattr(self, "_commit_mode", str(v).lower()),
                                         setattr(self, "_quiet_since", None),
                                         setattr(self, "_committed", False)),
                  doc="WHEN the registry updates. 'partial' (default) = per-object settle: each object commits on its OWN settle_seconds dwell, so a quiet object updates while another is still handled - the partial-update paradigm. 'global' = the whole TABLE must go quiet for global_quiet_seconds, then EVERY settled object commits in one tick and nothing commits in between: the arrangement speaks as one event (the automatic form of the /rmt/trigger pedal). 'manual' = nothing commits on its own: settling objects hold until a /rmt/trigger (pedal / OSC / webgui scan) commits the scene - the explicit-shutter form"),
        ParamSpec("global_quiet_seconds", float, _GLOBAL_QUIET, lo=0.0, attr="_quiet_secs",
                  doc="[commit_mode=global] how long the whole table must stay idle before the one global commit fires. Fires ONCE per quiet period; re-arms when the table moves again"),
        ParamSpec("global_quiet_frac", float, _GLOBAL_FRAC, lo=0.0, hi=1.0, attr="_quiet_frac",
                  doc="[commit_mode=global] moving-pixel fraction of the WHOLE frame below which the table counts as idle. Raise it if camera noise never lets the frame go still"),
        ParamSpec("confirm_seconds", float, _CONFIRM_SECONDS, lo=0.0, attr="_confirm",
                  doc="Persist this long after mint before an object is shown in the output/OSC/TUIO2 (a fresh hover blip is withheld; debug shows it at once); 0 = show immediately"),
    )

    def __init__(self, config: Any = None, *, logger: Any = None, scan_trigger: Any = None,
                 observe: Any = None) -> None:
        self._log = logger or NULL_LOGGER
        g = config.get if config is not None else (lambda k, d=None: d)
        self._obs = observe                # the Observe stage — the LIVE source of the ONE scale k
        self._k_cold = float(g("k", _K))   # fallback for a bare/test construction (no observe attached)
        self._scan = scan_trigger          # ScanTrigger (/rmt/trigger) — the manual scan; None = auto only
        self._quiet_since: float | None = None   # [global] ts the table went idle (None = busy/re-armed)
        self._committed = False                  # [global] fired for THIS quiet period? (one commit per stillness)
        bind_params(self, config)          # _settle / _move_debounce / _hold_max / _confirm / _commit_* ← PARAMS

    @property
    def _k(self) -> float:
        """The ONE threshold scale k — read LIVE from observe (same instance the registry writes), so a
        /rmt/set observe k reaches BOTH consumers mid-run; a cold copy would silently split the scale."""
        return self._obs.k if self._obs is not None else self._k_cold

    def __call__(self, ctx: Any, scene: Any) -> None:
        # /rmt/trigger (pedal): for THIS tick, read the table as it stands — capture every settling object
        # without its dwell, re-bake every quiet RESTING object from the live frame (flushing sub-k drift),
        # and confirm fresh mints, so the operator's "read it now" lands in the outputs immediately.
        # Taken once per fire.
        forced = bool(self._scan is not None and self._scan.take())
        if forced:
            self._log.info("[classify] manual scan - capturing settled objects now")
        elif self._commit_mode == "global" and self._global_commit_due(ctx):
            forced = True                  # the SAME one-tick capture-everything path as the pedal,
            #                                fired by table stillness instead of a foot
            self._log.info("[classify] global commit - table idle %.2fs, capturing every settled "
                           "object now", self._quiet_secs)
        minted = scene.minted_this_tick()
        for obj in scene.objects():
            if obj.id in minted:           # brand-new: no history + status just written by mint
                continue
            self._latch_confirmed(ctx, scene, obj, forced)
            self._classify(ctx, scene, obj, forced)

    def _global_commit_due(self, ctx: Any) -> bool:
        """[commit_mode=global] True on the ONE tick where the table has been idle long enough.

        Reads observe's whole-frame moving-pixel fraction (``ctx.motion_frac``, published every tick).
        Any motion re-arms: the dwell restarts AND the fired-latch clears, so one period of stillness
        yields exactly one commit no matter how long it lasts — otherwise a table left alone would
        re-capture every ``global_quiet_seconds`` forever. Frame 1 reports 1.0 (no previous frame to
        diff), which reads as busy, so a run never commits before it has seen the table move."""
        ts = getattr(ctx, "ts", None)
        if ts is None:
            return False
        if getattr(ctx, "motion_frac", 1.0) >= self._quiet_frac:      # the table is BUSY → re-arm
            self._quiet_since = None
            self._committed = False
            return False
        if self._quiet_since is None:                                 # first idle tick → start the dwell
            self._quiet_since = ts
        if self._committed or ts - self._quiet_since < self._quiet_secs:
            return False
        self._committed = True                                        # fire once for this stillness
        return True

    def _latch_confirmed(self, ctx: Any, scene: Any, obj: Any, forced: bool = False) -> None:
        """Latch `confirmed` once the object has persisted `confirm_seconds` since mint — the ONE gate
        every output honours (OSC/TUIO2, the output-view overlay), so a fresh hover
        blip that mints then vanishes never leaks out. Write-once; born_ts only grows, so it stays set.
        A manual scan (``forced``) confirms at once — the operator asked for it to go out."""
        if obj.confirmed or obj.born_ts is None or ctx.ts is None:
            return
        if forced or self._confirm <= 0.0 or ctx.ts - obj.born_ts >= self._confirm:
            scene.set(obj.id, "confirmed", True)

    def _classify(self, ctx: Any, scene: Any, obj: Any, forced: bool = False) -> None:
        obs = obj.observation
        if obs is None:
            return
        k = self._k
        ref_area = obj.footprint_area or obs.area or 1.0
        thr = k * ref_area                              # the ONE change threshold, shared by all three tests
        has_fp = obj.footprint is not None
        off_footprint = has_fp and obs.residual > thr
        rest_hidden = has_fp and obs.missing > thr
        stable = obs.xor_prev < thr
        gone = (obs.area or 0) <= 0
        # COVERED (→ HIDDEN) vs RELOCATED: `missing` (rest_hidden) fires for BOTH a covered object AND
        # one that just moved away (its footprint is empty either way). `residual` (off_footprint)
        # tells them apart — a relocated object's mask is PRESENT off the footprint. So the object
        # is truly hidden only when its footprint is missing AND nothing is off it (or the mask
        # vanished). Without this, an object held still at a NEW spot flickers MOVED↔HIDDEN forever
        # (its old footprint reads empty) and never re-settles → never snaps to the new position.
        covered = gone or (rest_hidden and not off_footprint)
        st = obj.status

        if st is Status.SETTLING:
            if covered:                                 # covered / lifted mid-settle (NOT merely
                self._to(ctx, scene, obj, Status.HIDDEN)     #   relocated — a re-settle at a new spot has an
                #                                           empty old footprint but a present mask)
            elif not stable:                            # still in motion → not settled yet
                self._to(ctx, scene, obj, Status.MOVED)
            else:                                       # stable + motion-free → run the N-sec timer
                if forced:                              # manual scan: it IS at rest — skip the dwell
                    self._capture(ctx, scene, obj, obs)
                elif obj.settle_since is None:
                    scene.set(obj.id, "settle_since", ctx.ts)
                elif ctx.ts - obj.settle_since >= self._settle and self._commit_mode == "partial":
                    self._capture(ctx, scene, obj, obs)   # PARTIAL mode: each object commits on its own
                    #   dwell. In GLOBAL/MANUAL mode nothing commits here — the object stays SETTLING,
                    #   holding its settle_since, until the whole table goes idle (global) or the pedal
                    #   fires (manual). The timer keeps running so switching the knob back to partial
                    #   commits it immediately.
        elif st is Status.RESTING:
            # a MOVING mass over the footprint (an arm crossing / grabbing) HOLDS the object pinned so a
            # transient occlusion doesn't flicker RESTING→HIDDEN→…→RESTING (a re-capture wobble). BOUNDED
            # by hold_max_seconds: persistent motion over a spot can't pin a stale husk forever.
            if obs.under_motion and obj.under_motion_since is None:
                scene.set(obj.id, "under_motion_since", ctx.ts)      # start the hold window
            elif not obs.under_motion and obj.under_motion_since is not None:
                scene.set(obj.id, "under_motion_since", None)        # occlusion cleared → reset the window
            holding = obs.under_motion and (obj.under_motion_since is None
                                            or ctx.ts - obj.under_motion_since <= self._hold_max)
            if (forced and stable and not covered and not off_footprint
                    and not obs.under_motion and obs.mask is not None):
                # manual scan = "read the table NOW": re-bake the quiet object from the live mask,
                # flushing the sub-k drift (nudges, light creep) the change-gate deliberately holds.
                # The capture invariant stands — moving, covered, or under-a-hand keeps the old footprint.
                self._capture(ctx, scene, obj, obs)
            elif holding:                               # HOLD pinned (its residual/missing are the mass's)
                if obj.off_since is not None:
                    scene.set(obj.id, "off_since", None)
            elif rest_hidden:
                self._to(ctx, scene, obj, Status.HIDDEN)
            elif off_footprint:                         # entry debounce: off_footprint must PERSIST
                if obj.off_since is None:                #   move_debounce_seconds → a 1-frame spike
                    scene.set(obj.id, "off_since", ctx.ts)   #   (occlusion bridge) can't flicker MOVED
                elif ctx.ts - obj.off_since >= self._move_debounce:
                    self._to(ctx, scene, obj, Status.MOVED)
            elif obj.off_since is not None:              # off_footprint cleared within the window → reset
                scene.set(obj.id, "off_since", None)
            # else: the hover — no branch fires, stay RESTING (θ untouched → zero wobble)
        elif st is Status.MOVED:
            if covered:                                 # mask gone / footprint empty w/ nothing off
                self._to(ctx, scene, obj, Status.HIDDEN)     #   it → lifted or covered (NOT relocated)
            elif has_fp and not off_footprint:          # back on the ORIGINAL footprint (never really
                self._to(ctx, scene, obj, Status.RESTING)    #   relocated) → resume rest, keep the footprint.
                #   MUST require a footprint: a footprint-less object has off_footprint==False too, and
                #   would bypass capture (RESTING with no footprint → nothing for an output to draw).
            elif stable:                                # relocated / fresh + settled → capture via SETTLING
                self._to(ctx, scene, obj, Status.SETTLING)
        elif st is Status.HIDDEN:
            if off_footprint:
                self._to(ctx, scene, obj, Status.MOVED)
            elif not rest_hidden:                       # the footprint shows again → genuinely reappeared
                self._to(ctx, scene, obj, Status.SETTLING)   #   → re-confirm via the settle timer
            # GONE eviction → identity (one bared-anchored memory clock: coast + reid pool)

    def _to(self, ctx: Any, scene: Any, obj: Any, status: Status) -> None:
        """Transition, (re)stamping the per-state timers (any edge invalidates the old ones). The HIDDEN
        coast/eviction is identity's job now — gated on the object's rest FOOTPRINT baring (a bared-
        anchored memory clock), not a classify entry stamp — so there is nothing to stamp here for it."""
        scene.set_status(obj.id, status)
        if obj.settle_since is not None:
            scene.set(obj.id, "settle_since", None)
        if obj.off_since is not None:
            scene.set(obj.id, "off_since", None)
        if obj.under_motion_since is not None:
            scene.set(obj.id, "under_motion_since", None)

    def _capture(self, ctx: Any, scene: Any, obj: Any, obs: Any) -> None:
        """SETTLING → RESTING: bake the WHOLE clean mask as the new footprint, rebasing θ to 0 at this
        rest. The continuous absolute angle (theta_cumulative) is maintained by the pose stage — it
        accumulates the footprint-relative θ across settles. (Appearance consumers — reid — crop their
        own patches from the live frame; the capture stores no texture.)

        Rigid capture only: at settle the object is re-captured whole from a clean, unoccluded view,
        so any bend is already baked into the fresh footprint."""
        mask = obs.mask
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(cnts, key=cv2.contourArea) if cnts else None
        fine = cv2.approxPolyDP(contour, 0.8, True) if contour is not None else None
        scene.set(obj.id, "footprint", mask.copy())
        scene.set(obj.id, "footprint_contour", fine)
        fp_area = int(cv2.countNonZero(mask))
        scene.set(obj.id, "footprint_area", fp_area)
        scene.set(obj.id, "canon_area", max(obj.canon_area or 0, fp_area))   # true size only ever grows
        scene.set(obj.id, "footprint_centroid", obs.centroid)
        scene.set(obj.id, "settle_since", None)
        scene.set_status(obj.id, Status.RESTING)
