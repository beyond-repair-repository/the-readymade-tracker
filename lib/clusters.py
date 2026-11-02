"""``clusters`` — spatial grouping of DISCRETE voices (a relational / *dynamic* pipeline STAGE).

Per frame: group nearby entities (proximity union-find), **persist the cluster id across frames**
(membership overlap), and publish one ``ClusterSnap`` per group via ``scene.set_clusters`` — a
cluster-centric TABLE (mirroring the Grouper's ``set_groups``), not a per-object annotation.

The population is the DISCRETE output voices, so it runs AFTER the Grouper. When the Grouper is active
(over the voice budget) every emitted voice is a compound group → clusters form over the GROUPS, each
one entity; a cluster's ``members`` are then the VOICE ids — a compound's gid, NOT its underlying
object ids (the fused emit view: every member id is on the ``/pose`` wire, absorbed objects are not).
Under budget the entities are the plain objects, and the cluster is a SETTLE-HELD set (mirroring the
object registry): an object JOINS by settling (RESTING), is HELD while PRESENT (occluded or its spot
is active — identity's motion-aware verdict), and LEAVES only when GONE (its spot bared AND quiet).
No debounce timers — see ``_entities``. Membership is DISJOINT: each object in ≤ 1 cluster.

**Interpretation-agnostic** — the tracker ships FACTS, the client ships MEANING. A ``ClusterSnap`` is
membership + geometry only: ``{id, members, centroid, bbox, min-circle, convex-hull}`` (raw px; the
outputs normalize). It carries NO ring/step layout — a radial sequencer joins ``members`` to the
``/pose`` table and computes its own angle→step / radius→velocity. ``oscout`` serializes the table
(``/rmt/cluster`` + ``/members`` + ``/bbox`` + ``/circle`` + ``/hull`` + ``/alv``), a viz strokes the
circle/hull. All cheap (O(n²) over a handful of objects + one enclosing circle per group), computed
ONCE — not per client.

Dynamic ⇒ ``invariant=False`` in spirit: it recomputes when objects MOVE. On a
still frame nothing moved, so the clusters are unchanged and the snapshot repeats (no rebuild)."""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from lib.events import readonly
from lib.logger import NULL_LOGGER
from lib.params import ParamSpec, bind_params
from lib.scene import ClusterSnap, Status, rekey

def _pos(o: Any) -> "tuple[float, float] | None":
    """The object's clustering position — its committed pose (x, y), else its centroid."""
    if getattr(o, "pose", None) is not None:
        return float(o.pose[0]), float(o.pose[1])
    if getattr(o, "centroid", None) is not None:
        return float(o.centroid[0]), float(o.centroid[1])
    return None


def _circle_groups(px: list, py: list, radii: list, gap: float, min_objects: int) -> list:
    """Union-find on ENCLOSING-CIRCLE proximity: two objects link when the gap between their circle
    EDGES — centre distance − r_i − r_j — is below ``gap`` px (so overlapping circles, gap < 0, always
    link). Size drives grouping, not raw centre distance: two big objects touch sooner than their
    centres suggest, and a small object beside a big one joins it. Groups of ≥ ``min_objects`` are
    clusters. Returns a list of index-lists. O(n²) over the (few) objects."""
    n = len(px)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        for j in range(i + 1, n):
            d = math.hypot(px[i] - px[j], py[i] - py[j])
            if d - radii[i] - radii[j] < gap:            # edge-to-edge gap (overlap → negative)
                union(i, j)
    groups: dict[int, list] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [g for g in groups.values() if len(g) >= min_objects]


class Clusters:
    """Pipeline stage: group nearby objects → publish a ``ClusterSnap`` table via ``scene.set_clusters``."""

    name = "clusters"

    # -- the live parameters, declared WITH the stage (harvested into the registry + config coercion).
    #   NO debounce timers: a cluster is a SETTLE-HELD set (see _entities) — an object JOINS by settling,
    #   is HELD while present, LEAVES when GONE. The ``apply`` setter owns the _min floor so cold-start
    #   bind_params and a live /rmt/set can't diverge. --
    PARAMS = (
        ParamSpec("enabled", bool, True, attr="_enabled",
                  apply=lambda self, v: setattr(self, "_enabled", bool(v)),
                  doc="Enable spatial clustering (live on/off; off = no cluster table published, wire and overlay show none)"),
        ParamSpec("radius", float, 0.12, lo=0.0, hi=1.0,
                  doc="Max gap between object enclosing circles to link them (fraction of max frame side); overlapping circles always link"),
        ParamSpec("min_objects", int, 2, lo=2, attr="_min",
                  apply=lambda self, v: setattr(self, "_min", max(2, int(v))),
                  doc="Min members for a group to count as a cluster (floor 2)"),
    )

    def __init__(self, config: Any = None, *, logger: Any = None,
                 enabled: "bool | None" = None) -> None:
        self._log = logger or NULL_LOGGER
        bind_params(self, config)                      # _enabled / _radius / _min ← PARAMS
        if enabled is not None:                        # the config loader POPS `enabled` (it drives the
            self._enabled = bool(enabled)              # manifest) — build() passes the manifest state, so the
        #                                                stage can exist DISABLED and go live via
        #                                                /rmt/set clusters enabled 1 (mirrors the grouper)
        # entity key = the voice id (int): an object id, or a compound's gid — one shared,
        # never-recycled sequence (scene.reserve_id), so the two can never collide.
        self._prev: dict = {}                          # entity key -> cluster id (last frame): BOTH id
        #                                                persistence AND the HELD-member set (its keys are
        #                                                the members of last frame's clusters).
        self._encr: dict = {}                          # entity key -> (area signature, enclosing radius)
        self._next_id = 1
        self._log.info(f"[clusters] proximity radius={self._radius} min={self._min}")

    def _entities(self, ctx: Any, scene: Any) -> list:
        """The candidate voices to cluster, as ``(key, geometry_source, member_ids)`` triples. The key
        IS the voice id (an object id, or a compound's gid when fused — one shared, never-recycled
        sequence) and ``member_ids`` is just ``(key,)``: a compound contributes its gid, NOT its
        underlying object ids (the FUSED EMIT VIEW below). Plain objects join under the SETTLE-HELD
        rule (a cluster mirrors the object registry: settled, held in motion):
          · JOIN by settling — a plain object is a candidate when it is RESTING (settled). A blob still in
            motion (a wire carried across the scene) is not RESTING and not yet a member, so it never joins
            or flashes; the old ``persist_seconds`` debounce falls out for free.
          · HELD while present — an object already in a cluster last frame (a key in ``self._prev``) stays a
            candidate through occlusion, a re-settle, or being carried (its pose is held), as long as it is
            PRESENT: not HIDDEN, or HIDDEN with identity's motion-aware verdict in ``ctx.hidden_present``.
          · LEAVE when GONE — a HIDDEN member whose spot is bared AND quiet is not in hidden_present, so it
            drops (dissolving the ring if that takes it below min_objects); this replaces ``cover_grace``.
        `ignored` blobs (edge/arm residue) and UNCONFIRMED objects (withheld from every output during
        confirm_seconds — an invisible voice must not take a ring index) are excluded here as everywhere.
        Grouper active (over the voice budget) → the FUSED EMIT VIEW: each compound joins as ONE plain
        entity (its shared-sequence gid; RESTING by construction) and its absorbed members are skipped —
        the ring forms over exactly the voices the output wires ship."""
        claimed, comps = scene.fused() if hasattr(scene, "fused") else (frozenset(), ())
        retiring = scene.retiring()
        present = getattr(ctx, "hidden_present", ())
        held = self._prev                                  # keys = the members of last frame's clusters
        out = [(int(c.id), c, (int(c.id),)) for c in comps if _pos(c) is not None]
        for o in scene.objects():
            if o.id in claimed:
                continue                                    # absorbed into a compound — the compound is the voice
            if o.id in retiring or getattr(o, "ignored", False) or _pos(o) is None:
                continue
            if not getattr(o, "confirmed", False):
                continue                                    # withheld from every output (confirm_seconds) —
                #                                             an invisible voice must not take a ring index
                #                                             or inflate n (confirmed latches, so held
                #                                             members are never re-dropped by this)
            if o.status is Status.HIDDEN and o.id not in present:
                continue                                    # HIDDEN and GONE (bared + quiet) → drop the member
            if o.status is not Status.RESTING and o.id not in held:
                continue                                    # in motion and not yet a member → don't JOIN
            out.append((o.id, o, (o.id,)))
        return out

    def __call__(self, ctx: Any, scene: Any) -> None:
        if not self._enabled:                              # runtime off → publish nothing (cheap no-op);
            if self._prev:                                 #   drop persistence so a re-enable re-mints ids
                self._prev, self._encr = {}, {}
            scene.set_clusters(())
            return
        for old, new in scene.recent_rewrites():           # a re-id moves an object → re-key persistence
            rekey(old, new, self._prev, self._encr)
        bgr = getattr(ctx, "bgr", None)
        h, w = (bgr.shape[0], bgr.shape[1]) if bgr is not None else (0, 0)
        ents = self._entities(ctx, scene)
        pos = [_pos(src) for _k, src, _m in ents]
        px, py = [p[0] for p in pos], [p[1] for p in pos]
        radii = [self._radius_of(k, src) for k, src, _m in ents]   # enclosing-circle radius (cached)
        gap = self._radius * max(w, h, 1)                  # frac → px (max edge-to-edge gap to link)
        groups = _circle_groups(px, py, radii, gap, self._min)
        present = {k for k, _s, _m in ents}                # drop cached radii for gone entities (no leak)
        for k in [k for k in self._encr if k not in present]:
            self._encr.pop(k, None)
        cids = self._assign_ids(ents, groups)              # persistent cluster ids; sets self._prev to now
        snaps = []
        for group, cid in zip(groups, cids):
            # every grouped member is included — the JOIN / HOLD / GONE gating already happened in
            # _entities (settled to join, present to stay), so there is NO per-member debounce here. A
            # member held through motion keeps its last pose, so the cluster geometry stays put.
            conf = group
            # enclose the member BODIES (each entity's bbox corners), not just their centres, so the
            # cluster shape CONTAINS them instead of cutting through them (fall back to the centre).
            corners = []
            for i in conf:
                b = getattr(ents[i][1], "bbox", None)
                if b is not None:
                    bx, by, bw, bh = b
                    corners += [(bx, by), (bx + bw, by), (bx, by + bh), (bx + bw, by + bh)]
                else:
                    corners.append((px[i], py[i]))
            pts = np.array(corners, np.float32).reshape(-1, 1, 2)
            (ccx, ccy), rr = cv2.minEnclosingCircle(pts)   # rotation-invariant enclosing circle
            bx, by, bw, bh = cv2.boundingRect(pts)         # axis-aligned bounds
            hull = cv2.convexHull(pts) if len(corners) >= 3 else None   # tight outer outline
            cxm = sum(px[i] for i in conf) / len(conf)     # cluster centroid = mean member position
            cym = sum(py[i] for i in conf) / len(conf)
            # DISJOINT membership → the VOICE ids (an object id, or a compound's gid when fused);
            # every one is on the /pose wire, so a client can join to compute a layout.
            members = tuple(int(oid) for i in conf for oid in ents[i][2])
            snaps.append(ClusterSnap(id=int(cid), members=members,
                                     centroid=(float(cxm), float(cym)),
                                     bbox=(float(bx), float(by), float(bw), float(bh)),
                                     circle=(float(ccx), float(ccy), float(rr)),
                                     hull=readonly(hull)))
        scene.set_clusters(tuple(snaps))                   # the scene-level channel resets each tick — no
        #                                                    stale-clear needed (unlike the old annotations)

    def _radius_of(self, key: Any, src: Any) -> float:
        """The entity's enclosing-circle RADIUS, cached by entity key + a shape signature:
        ``minEnclosingCircle`` of the SETTLED footprint (a shape size, position-independent; 0 before
        capture → it links by pure centre distance until it settles). A compound's footprint is the
        mass-scaled blend; it carries no ``footprint_area``, so its summed ``area`` is the
        re-blend signal instead (membership change → new mass → recompute)."""
        fc = getattr(src, "footprint_contour", None)
        if fc is None or len(fc) < 1:
            return 0.0
        fa = float(getattr(src, "footprint_area", 0) or 0.0)
        sig = fa if fa > 0 else float(getattr(src, "area", 0) or 0.0)
        cached = self._encr.get(key)
        if cached is None or cached[0] != sig:             # first sight, re-settle, or re-blend
            (_, _), r = cv2.minEnclosingCircle(fc)
            self._encr[key] = (sig, float(r))
        return self._encr[key][1]

    def _assign_ids(self, ents: list, groups: list) -> list:
        """Match each group to a PREVIOUS cluster by membership overlap of ENTITY keys (reuse its id
        if the overlap is ≥ half the group), else mint a new id — so a cluster keeps its id as
        entities jiggle. (A grouper on/off regime flip changes the key space, so rings re-mint.)"""
        prev, used, out = self._prev, set(), []
        for group in groups:
            member_keys = {ents[i][0] for i in group}
            counts: dict[int, int] = {}
            for k in member_keys:
                pcid = prev.get(k)
                if pcid is not None:
                    counts[pcid] = counts.get(pcid, 0) + 1
            best_cid, best = None, 0
            for cid, c in counts.items():
                if cid not in used and c > best:
                    best, best_cid = c, cid
            if best_cid is not None and best >= max(1, len(member_keys) // 2):
                cid = best_cid
            else:
                cid = self._next_id
                self._next_id += 1
            used.add(cid)
            out.append(cid)
        self._prev = {ents[i][0]: out[gi] for gi, group in enumerate(groups) for i in group}
        return out
