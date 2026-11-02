"""``identity`` — cross-settle identity: HIDDEN retirement + closed-set re-acquire.

Runs after classify, before pose. Two jobs:

  * **Retire HIDDEN objects** to a gone-pool: a HIDDEN object whose rest FOOTPRINT has BARED (the
    object was carried off) goes to the pool; while the footprint is still covered (a hand hovering
    over it) it is held — occlusion is not proof of absence. A ghost (a HIDDEN blob that never
    captured a footprint) is retired immediately, NOT pooled. The identity is remembered for ONE
    window — ``pool_ttl_seconds``, measured from the moment the spot BARES (NOT from when the object
    went HIDDEN) and FROZEN while occluded — spanning both the drawn stay-then-snap coast and the
    gone-pool, so "how long is a lifted object remembered?" has a single answer. EARLY GONE: once the
    spot is bared, still, and the WHOLE table idle for ``away_quiet_seconds``, it pools at once (a
    removed object disappears in ~half a second) instead of waiting the window out; a carry (a hand
    always moving somewhere) never trips it.

  * **Re-acquire** a re-emerged object: a fresh (footprint-less) SETTLING blob near a pooled
    object's last rest position re-adopts that identity (`rewrite_id`) — so a lift-and-replace
    keeps the same id and the churn stays bounded.

Identity is CLOSED-SET: a set-down is 1-of-N over the small gone-pool, not open-set matching
against all objects. This stage does the geometric (near set-down) closed-set match; the
optional classical appearance tie-break ([reid]) catches a FAR set-down.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from lib.logger import NULL_LOGGER
from lib.params import ParamSpec, bind_params
from lib.reid import ReidParams, best_cosine
from lib.scene import Status, rekey

_REACQUIRE_DIST = 160.0   # default set-down re-link reach (px) — read LIVE from [observe].relocate_dist via
#                           the attached observe; this is only the fallback for a bare construction
_POOL_TTL = 5.0           # a pooled identity expires after this long unclaimed (s)
_CLEAR_FRAC = 0.02        # last-known area foreground fraction below this = "clear/empty" (gone)
_AWAY_QUIET = 0.5         # bared + motionless spot + an idle table held this long → pool NOW (early GONE)
_TABLE_QUIET = 0.002      # global moving-pixel fraction below which the table counts as idle
_SPOT_MOTION = 0.10       # footprint moving-pixel fraction above which the spot counts as active


class Identity:
    """The identity stage — HIDDEN retirement to the gone-pool + geometric closed-set re-acquire."""

    name = "identity"

    # -- live parameters, declared WITH the stage. The APPEARANCE policy (cosine gates, twin guard,
    #    size gate, fingerprint accumulation) lives in [reid] (lib.reid.ReidParams), read live. --
    PARAMS = (
        ParamSpec("pool_ttl_seconds", float, _POOL_TTL, lo=0, attr="_ttl", doc="Seconds an identity is remembered after its rest spot BARES - ONE clock across the drawn coast and the gone-pool, anchored at baring (occlusion does not count) then to re-acquire/expiry"),
        ParamSpec("clear_frac", float, _CLEAR_FRAC, lo=0.0, hi=1.0, attr="_clear", doc="footprint-coverage frac below which a HIDDEN object's spot counts as bared (the contour-level GONE test)"),
        ParamSpec("away_quiet_seconds", float, _AWAY_QUIET, lo=0.0, attr="_away_quiet", doc="Bared spot + no motion there + an idle table held this long -> the HIDDEN object pools at once instead of coasting out the full ttl (early GONE; 0 disables)"),
        ParamSpec("table_quiet_frac", float, _TABLE_QUIET, lo=0.0, hi=1.0, attr="_table_quiet", doc="Global moving-pixel fraction below which the table counts as idle - the early-GONE guard (a carried object always keeps a hand moving somewhere)"),
    )

    def __init__(self, config: Any = None, *, logger: Any = None, embedder: Any = None,
                 reid_params: Any = None, observe: Any = None) -> None:
        self._log = logger or NULL_LOGGER
        g = config.get if config is not None else (lambda k, d=None: d)
        bind_params(self, config)      # _ttl / _clear / _away_quiet / _table_quiet ← PARAMS
        self._obs = observe            # the Observe stage — the LIVE source of the set-down re-link reach
        self._reacq_cold = float(g("reacquire_dist", _REACQUIRE_DIST))   # bare/test fallback (no observe)
        self._reidp = reid_params if reid_params is not None else ReidParams(None)
        #                              # the shared [reid] appearance policy, read LIVE (defaults when bare)
        self._embedder = embedder                # classical appearance embedder, or None (graceful)
        self._pool: dict[int, dict] = {}      # id -> {"c","ts","embeds","area","canon"}; "ts" is the BARING
        #                                       time, so the pool's _ttl expiry is measured from baring too
        self._reacquired: set[int] = set()    # ids that re-adopted a pooled identity THIS settle cycle —
        #                                       re-adopt AT MOST ONCE (a SETTLING/footprint-None object
        #                                       re-enters _reacquire each tick and would otherwise hop
        #                                       through successive pool entries, corrupting ids)
        self._fp_pos: dict[int, tuple] = {}   # id -> the last-fingerprinted (cx, cy): append a new print
        #                                       only when the object settles at a NEW position
        self._away: dict[int, float] = {}     # id -> ts the spot went bared+quiet+idle-table (the early-GONE
        #                                       confirm clock); absent = not currently holding
        self._bared: dict[int, float] = {}    # id -> ts the spot FIRST bared — the MEMORY anchor: the
        #                                       identity is remembered _ttl s from here, across coast + pool.
        #                                       Cleared on re-occlusion, so occlusion never ages the memory.

    @property
    def _reacq_dist(self) -> float:
        """The ONE set-down re-link reach — read LIVE from observe's relocate_dist (the same instance the
        registry writes), so a /rmt/set observe relocate_dist reaches BOTH geometric surfaces mid-run;
        the former build-time cold copy silently diverged after a live set."""
        return self._obs.relocate_dist if self._obs is not None else self._reacq_cold

    def __call__(self, ctx: Any, scene: Any) -> None:
        for old, new in scene.recent_rewrites():
            rekey(old, new, self._pool, self._reacquired, self._fp_pos, self._away, self._bared)
        objs = scene.objects()                # materialize ONCE — the ordered passes below reuse it
        #                                       (membership is stable within the tick) instead of
        #                                       allocating a fresh tuple(_objs.values()) per pass.
        if self._embedder is not None:           # ACCUMULATE a fingerprint per NEW-position settle (build a
            for obj in objs:                  #   multi-view appearance model → better re-id persistence)
                if obj.status is Status.RESTING:
                    self._fingerprint(ctx, scene, obj)
        ctx.hidden_present = set()            # HIDDEN ids still PRESENT — their spot is occluded OR ACTIVE
        #                                       (a hand at it, so 'gone' can't be confirmed); grouper/clusters
        #                                       keep these as voices. Only a BARED-and-QUIET spot is dropped.
        for obj in objs:
            if obj.status is Status.HIDDEN:
                self._retire_maybe(ctx, scene, obj)
        if hasattr(scene, "set_hidden_present"):    # publish the covered/bared verdict on the snapshot
            scene.set_hidden_present(frozenset(ctx.hidden_present))   # (webgui reads it; the WIRE does not)
        settling = {obj.id for obj in objs if obj.status is Status.SETTLING and obj.footprint is None}
        for obj in objs:
            if obj.id in settling and obj.id not in self._reacquired:
                self._reacquire(ctx, scene, obj)      # re-adopt at most once per settle cycle
        # Drop guard entries for ids that captured (RESTING) / died → free to re-adopt later. The set must
        # be REBUILT here: _reacquire rewrites obj.id in place (mint → adopted pid), so the pre-loop
        # `settling` holds the OLD ids — intersecting with it would empty the guard the same tick and let a
        # still-settling blob hop through successive pool entries, corrupting ids.
        self._reacquired &= {o.id for o in objs if o.status is Status.SETTLING and o.footprint is None}
        expired = [i for i, e in self._pool.items() if ctx.ts - e["ts"] > self._ttl]
        for i in expired:                     # pop the expired (rare) instead of rebuilding the whole
            del self._pool[i]                 #   pool dict every tick
        live = {o.id for o in objs}
        for d in (self._fp_pos, self._away, self._bared):  # drop per-id state for departed ids
            for i in [i for i in d if i not in live]:
                d.pop(i, None)

    def _fingerprint(self, ctx: Any, scene: Any, obj: Any) -> None:
        """Append an appearance fingerprint when the object settles at a NEW position (moved > fp_new_dist
        from the last-fingerprinted spot, or the first settle) — building a multi-view dictionary per
        object. Capped at fp_max_prints (most-recent views). NOT done for a re-settle at the same spot."""
        c = obj.footprint_centroid
        if c is None:
            return
        last = self._fp_pos.get(obj.id)
        if last is not None:
            d = ((c[0] - last[0]) ** 2 + (c[1] - last[1]) ** 2) ** 0.5
            if d <= self._reidp.fp_new_dist:
                return                        # same position → don't append a redundant print
        e = self._embed(ctx, obj)
        if e is None:
            return
        prints = tuple(obj.embeds or ())
        prints = (prints + (e,))[-self._reidp.fp_max_prints:]   # append + cap to the most-recent N views
        scene.set(obj.id, "embeds", prints)
        self._fp_pos[obj.id] = (float(c[0]), float(c[1]))

    def _embed(self, ctx: Any, obj: Any) -> "np.ndarray | None":
        """Appearance embedding of the object's clean crop (BGR bbox masked by its mask)."""
        if self._embedder is None or getattr(ctx, "bgr", None) is None:
            return None
        obs = obj.observation
        bbox = obj.bbox or (obs.bbox if obs is not None else None)
        if bbox is None:
            return None
        x, y, w, h = bbox
        patch = ctx.bgr[max(0, y):y + h, max(0, x):x + w]
        omask = obs.mask if obs is not None else None
        mask = omask[max(0, y):y + h, max(0, x):x + w] if omask is not None else None
        return self._embedder.embed(patch, mask)

    def _footprint_cov(self, obj: Any, fg: Any) -> float:
        """Fraction of the object's rest FOOTPRINT — its exact rasterized silhouette, NOT the bounding
        box — currently covered by foreground. Contour-level: the footprint mask ∩ the live obj_fg over
        the footprint area. ~0 = the spot bared (the object was carried off → GONE); high = something
        (the object still, or a hand hovering over it) sits on the spot → occluded, NOT gone."""
        fp = obj.footprint
        if fp is None or fg is None or fp.shape != fg.shape:
            return 1.0                             # can't test → treat as covered (don't retire via GONE)
        area = obj.footprint_area or int(cv2.countNonZero(fp))
        if area <= 0:
            return 1.0
        x, y, w, h = cv2.boundingRect(fp)          # ∩ only inside the footprint's bbox — the mask is 0
        if w <= 0 or h <= 0:                        #   outside it, so this is byte-identical to the
            return 1.0                             #   full-frame ∩ but skips scanning the empty frame
        ov = int(cv2.countNonZero(cv2.bitwise_and(fp[y:y + h, x:x + w], fg[y:y + h, x:x + w])))
        return ov / float(area)

    def _footprint_motion(self, obj: Any, motion: Any) -> float:
        """Fraction of the footprint's pixels in the live motion mask — is anything HAPPENING at the
        spot? Unknown (no mask / shape mismatch / empty) → 1.0: treat as active, never retire blind."""
        fp = obj.footprint
        if motion is None or fp is None or motion.shape != fp.shape:
            return 1.0
        x, y, w, h = cv2.boundingRect(fp)
        if w <= 0 or h <= 0:
            return 1.0
        roi = fp[y:y + h, x:x + w]
        area = obj.footprint_area or int(cv2.countNonZero(roi))
        if area <= 0:
            return 1.0
        return int(cv2.countNonZero(cv2.bitwise_and(motion[y:y + h, x:x + w], roi))) / float(area)

    def _retire_maybe(self, ctx: Any, scene: Any, obj: Any) -> None:
        if obj.footprint is None:                          # ghost (never captured) → drop, don't pool
            self._drop_state(obj.id)
            scene.retire(obj.id)
            return
        # STAY-THEN-SNAP: a captured HIDDEN object stays drawn at its old rest spot as a RELOCATION
        # CANDIDATE — observe's relocation pass claims it (→ MOVED) when it re-settles nearby, so the SAME
        # object snaps to the new spot with no blink.
        #
        # THE MEMORY CLOCK, anchored at BARING. The identity is remembered for _ttl (pool_ttl_seconds)
        # AFTER its rest FOOTPRINT first BARES — the first moment we can tell the object has left the spot.
        # One clock spans the whole life (the drawn coast AND the gone-pool), so a lifted object is
        # remembered for exactly _ttl from baring, no matter which path it takes. The GONE test is
        # contour-level: the footprint's rasterized silhouette ∩ the live foreground (obj_fg), NOT a bbox.
        #   * footprint COVERED (a hand hovering) or untestable → NOT gone: hold it HIDDEN at its spot,
        #     mark it PRESENT, and FREEZE the clock (drop the baring anchor — the next baring restarts it).
        #     Occlusion is never proof of absence, so it must never age a still-present object out. When the
        #     cover lifts, observe's proximity pass re-associates the reveal (distance ~0), id intact — no
        #     pool round-trip, no re-acquire gamble, no fresh mint.
        #   * footprint BARED → the object left the spot; the memory clock runs (anchored at the FIRST
        #     bared tick) while the coast keeps the husk claimable, until early-GONE or the window ends.
        fg = getattr(ctx, "obj_fg", None)
        present = getattr(ctx, "hidden_present", None)          # published each tick by __call__
        # GONE needs a BARED **and** QUIET spot: nothing on the rest footprint AND no motion there AND an
        # idle table (a carry always keeps a hand moving somewhere). Motion makes the verdict robust to a
        # 1-frame flicker of the pure coverage test.
        spot_active = (getattr(ctx, "motion_frac", 1.0) >= self._table_quiet
                       or self._footprint_motion(obj, getattr(ctx, "motion", None)) > _SPOT_MOTION)
        if fg is None or self._footprint_cov(obj, fg) >= self._clear:
            if present is not None:
                present.add(obj.id)                             # occluded/untestable → bodily present
            self._away.pop(obj.id, None)
            self._bared.pop(obj.id, None)                       # freeze the memory clock; next baring restarts
            return
        # BARED: the spot is empty. Anchor the memory clock at the FIRST bared tick — it persists through
        # bared-active and bared-quiet alike, until a re-occlusion clears it above.
        bared_ts = self._bared.setdefault(obj.id, ctx.ts)
        if spot_active:
            if present is not None:
                present.add(obj.id)                             # bared but ACTIVE (a hand at the spot) → present
            self._away.pop(obj.id, None)                        # activity → restart the quiet-confirm clock
        elif self._away_quiet > 0.0 and ctx.ts - self._away.setdefault(obj.id, ctx.ts) >= self._away_quiet:
            # BARED + QUIET + idle table held ≥ away_quiet_seconds → confident GONE: pool NOW for re-acquire,
            # long before the window ends. The pool entry still expires _ttl from BARING (bared_ts below).
            self._pool_and_retire(ctx, scene, obj, bared_ts)
            return
        # BACKSTOP: bared (quiet-but-brief, quiet disabled, or a table that never settles) for the whole
        # memory window since baring → the identity is forgotten. Pool it (a set-down THIS tick can still
        # re-acquire) with the baring stamp, so the pool expiry drops it right away.
        if ctx.ts - bared_ts >= self._ttl:
            self._pool_and_retire(ctx, scene, obj, bared_ts)

    def _drop_state(self, oid: int) -> None:
        """Forget the per-id trackers for a retiring id. MUST run at retire time: retire defers eviction to
        end_tick, so the id is still 'live' at this tick's departed-ids prune — and if a settling blob
        re-acquires the id next tick, a leftover ``_bared`` anchor from the FIRST lift would instantly
        expire the SECOND lift's memory window (and a stale ``_fp_pos`` would suppress its next print)."""
        for d in (self._fp_pos, self._away, self._bared):
            d.pop(oid, None)

    def _pool_and_retire(self, ctx: Any, scene: Any, obj: Any, bared_ts: "float | None" = None) -> None:
        """Move a GONE object to the gone-pool (carry its fingerprint dict + canon size + θ reference for
        a later re-acquire) and retire it from the live scene. ``ts`` is the BARING time (not now), so the
        one memory window — the pool's ``_ttl`` expiry — runs from baring, unbroken across coast → pool."""
        self._pool[obj.id] = {"c": obj.centroid or obj.footprint_centroid,
                              "ts": bared_ts if bared_ts is not None else ctx.ts,
                              "embeds": obj.embeds, "area": obj.area, "canon_area": obj.canon_area,
                              "canon": obj.pose_ref}
        self._drop_state(obj.id)
        scene.retire(obj.id)

    def _reacquire(self, ctx: Any, scene: Any, obj: Any) -> None:
        obs = obj.observation
        c = obs.centroid if obs is not None else obj.centroid
        if c is None or not self._pool:
            return
        # APPEARANCE (preferred — catches a FAR set-down): the best-cosine pooled entry, above the
        # floor, clearly beating the 2nd-best (twin guard), within the size gate. A closed-set
        # tie-break over the small pool — NOT open-set matching.
        emb = self._embed(ctx, obj) if self._embedder is not None else None
        area = obs.area if obs is not None else obj.area
        # a pool id RETIRED THIS TICK is still live until end_tick (retire defers eviction) — adopting
        # it now would collide in rewrite_id (target #id already exists). Skip it; it frees next tick.
        adoptable = {pid: e for pid, e in self._pool.items() if scene.get(pid) is None}
        if emb is not None:
            sims = []
            for pid, e in adoptable.items():
                if not e.get("embeds") or not self._size_gate(area, e):
                    continue
                sims.append((best_cosine(emb, e["embeds"]), pid))   # best over the pooled multi-view prints
            sims.sort(reverse=True)
            if sims and sims[0][0] >= self._reidp.cosine_min and \
                    (len(sims) == 1 or sims[0][0] - sims[1][0] >= self._reidp.uniqueness_margin):
                pid = sims[0][1]
                scene.rewrite_id(obj.id, pid)
                self._restore_canon(scene, pid, adoptable[pid])
                self._pool.pop(pid, None)
                self._reacquired.add(pid)                 # adopted once — don't hop pool entries next tick
                return
        # GEOMETRIC fallback — the nearest pooled rest position within the gate AND of similar size
        # (the size gate closes the re-acquire door a paused arm could otherwise use to re-hijack an id)
        # AND of similar SHAPE: observe's relocation pass (the husk half of the same single-sourced
        # reach) demands the fold-change elongation/fill gate, so the pool half must too — else whether
        # a different-shape set-down can steal the id depends on whether early-GONE pooled it first.
        # The settling blob's contour comes from its Observation MASK (rare path: only footprint-less
        # settling frames reach here, so the findContours is a few calls per set-down, not per frame).
        mask = getattr(obs, "mask", None)                 # Observation carries a mask (Detection doesn't)
        blob_contour = None
        if self._obs is not None and mask is not None:
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if cnts:
                blob_contour = max(cnts, key=cv2.contourArea)
        best = None
        for pid, e in adoptable.items():
            ec = e["c"]
            if ec is None or not self._size_gate(area, e):
                continue
            canon = e.get("canon")                        # (footprint_contour, centroid) from pose_ref
            if (self._obs is not None and canon is not None
                    and not self._obs.shape_claim_ok(canon[0], blob_contour)):
                continue
            d = ((c[0] - ec[0]) ** 2 + (c[1] - ec[1]) ** 2) ** 0.5
            if d < self._reacq_dist and (best is None or d < best[1]):
                best = (pid, d)
        if best is not None:
            pid = best[0]
            scene.rewrite_id(obj.id, pid)
            self._restore_canon(scene, pid, adoptable[pid])
            self._pool.pop(pid, None)
            self._reacquired.add(pid)                     # adopted once — don't hop pool entries next tick

    @staticmethod
    def _restore_canon(scene: Any, pid: int, entry: dict) -> None:
        """Put pose's canonical reference back on the re-acquired object so its absolute rotation
        (theta_cumulative) survives the relocation instead of re-basing to 0. (rewrite_id has just made
        pid the live object, so scene.set(pid, …) is valid.)"""
        canon = entry.get("canon")
        if canon is not None:
            scene.set(pid, "pose_ref", canon)

    def _size_ok(self, a: "float | None", b: "float | None") -> bool:
        if not a or not b:
            return True
        return (1.0 - min(a, b) / max(a, b)) <= self._reidp.size_tol

    def _size_gate(self, area: "float | None", e: dict) -> bool:
        """Pool re-acquire size gate: the returning blob passes if it matches the pooled object's stored
        area OR its canon (true, largest-clean) size — heals a husk pooled with a half-occluded footprint."""
        return self._size_ok(area, e.get("area")) or self._size_ok(area, e.get("canon_area"))
