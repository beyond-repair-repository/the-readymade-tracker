"""``grouper`` — OUTPUT-side budget grouping.

When the scene holds more objects than the output voice budget (``max_objects``), collapse them into
``≤ N`` **merged super-objects** the outputs emit — no object is dropped; nearby ones fuse into one.
Because the tracker has persistent identity, groups keep STABLE ids across frames (membership
overlap) — a per-frame re-slot would jitter the voices.

Two layers here. The pure **EFD merge primitive**: blend several contours into one representative
"averaged" shape via a weighted sum of their complex elliptic-Fourier coefficients — a complex-plane
phasor blend — then inverse-transform. And the ``Grouper`` STAGE (clustering + budget + the
``scene.groups`` channel) built on it.

    merged = IFFT( Σ_i  w_i · EFD(contour_i) )      # a shape between the members, weighted

The coefficients are computed by ``lib.efd.efd_coeffs`` (exact Kuhl & Giardina), which already
returns the complex pairs ``c[+n], c[-n]`` — so nothing about the ``efd`` feature has to change.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from lib.efd import efd_coeffs
from lib.features import size_fraction
from lib.geometry import place_footprint
from lib.logger import NULL_LOGGER
from lib.params import ParamSpec, bind_params
from lib.events import readonly
from lib.scene import GroupSnap, ObjectSnap, Status, rekey


def _canon(c: dict, max_k: int) -> "tuple[np.ndarray, np.ndarray]":
    """Kuhl-Giardina normalization → (cp, cm) arrays with the STARTING POINT + ROTATION canonicalized
    (the first harmonic's c[+1], c[-1] made real-positive). Without this, contours with different start
    points / orientations blend DESTRUCTIVELY (the harmonics cancel → a collapsed shape). Scale is left
    ALONE (size preserved — the caller weights by area). Under a rotation ψ and start-shift α:
    ``c[+n]→c[+n]·e^{i(ψ+nα)}``, ``c[-n]→c[-n]·e^{i(ψ−nα)}``; solving c[±1]→real+ gives ψ, α below."""
    a1, am1 = np.angle(c[1]), np.angle(c[-1])
    psi, alpha = -(a1 + am1) / 2.0, (am1 - a1) / 2.0
    cp = np.zeros(max_k + 1, complex)
    cm = np.zeros(max_k + 1, complex)
    for n in range(1, max_k + 1):
        cp[n] = c[n] * np.exp(1j * (psi + n * alpha))
        cm[n] = c[-n] * np.exp(1j * (psi - n * alpha))
    return cp, cm


def merge_efd(contours: "list", weights: "list | None" = None,
              max_k: int = 12, samples: int = 180) -> np.ndarray:
    """Blend ``contours`` into ONE representative 'averaged' contour via a weighted sum of their
    CANONICALIZED complex EFD coefficients (see ``_canon``), then inverse-transform to ``samples`` points.

    ``contours`` — each cv2 ``(N,1,2)`` or ``(N,2)``. ``weights`` — per-contour (need not sum to 1;
    ``None`` = equal). Returns a cv2-shaped ``(samples,1,2)`` float32 (SUB-PIXEL) contour, centred on the weight-blended
    centroid, in CANONICAL orientation — major axis along +x (the Grouper stage rotates it to the
    members' union principal axis, ``_union_axis``). A single
    contour → (an EFD reconstruction of) itself; ``w=[1,0,…]`` → the first. Empty/degenerate → ``None``."""
    contours = [np.asarray(c, float).reshape(-1, 2) for c in contours if c is not None and len(c) >= 3]
    if not contours:
        return None
    w = np.ones(len(contours)) if weights is None else np.asarray(weights[:len(contours)], float)
    s = w.sum()
    w = (w / s) if s > 0 else np.full(len(contours), 1.0 / len(contours))

    cp = np.zeros(max_k + 1, complex)          # blended (canonicalized) c[+n], n = 1..max_k
    cm = np.zeros(max_k + 1, complex)          # blended c[-n]
    cx = cy = 0.0
    for contour, wi in zip(contours, w):
        c, centroid = efd_coeffs(contour, max_k)
        cx += wi * centroid[0]; cy += wi * centroid[1]
        ncp, ncm = _canon(c, max_k)            # align start-point + rotation before summing (else they cancel)
        cp += wi * ncp
        cm += wi * ncm

    f = np.arange(samples) / samples           # inverse EFD: z(t) = centroid + Σ_n c[+n]e^{iθ} + c[-n]e^{-iθ}
    z = np.full(samples, complex(cx, cy))
    for n in range(1, max_k + 1):
        ph = np.exp(1j * 2.0 * np.pi * n * f)
        z += cp[n] * ph + cm[n] / ph           # e^{-iθ} = 1/e^{iθ}
    return np.stack([z.real, z.imag], axis=1).astype(np.float32).reshape(-1, 1, 2)


# --------------------------------------------------------------------------- #
# The Grouper STAGE — cluster-to-budget + merge into compound super-objects.
# --------------------------------------------------------------------------- #

def _union_axis(world: "list", gcx: float, gcy: float) -> "float | None":
    """The members' UNION principal axis (radians from +x, image coords): area-weighted filled-shape
    central moments PLUS the parallel-axis terms of each member's offset from the group centroid —
    so both a member's own elongation AND the stacking direction count (two round blobs stacked
    vertically are a VERTICAL union). ``world`` = the (contour, area) pairs handed to the blend.
    ``None`` when the union is near-isotropic (a round pile has no meaningful axis)."""
    mu20 = mu11 = mu02 = 0.0
    for c, _a in world:
        m = cv2.moments(np.asarray(c, np.float32).reshape(-1, 1, 2))
        if m["m00"] <= 0:
            continue
        dx, dy = m["m10"] / m["m00"] - gcx, m["m01"] / m["m00"] - gcy
        mu20 += m["mu20"] + m["m00"] * dx * dx
        mu11 += m["mu11"] + m["m00"] * dx * dy
        mu02 += m["mu02"] + m["m00"] * dy * dy
    denom = mu20 + mu02
    if denom <= 0 or math.hypot(2.0 * mu11, mu20 - mu02) < 1e-3 * denom:
        return None
    return 0.5 * math.atan2(2.0 * mu11, mu20 - mu02)


def _pos(o: Any) -> "tuple[float, float] | None":
    """The object's grouping position — its committed pose (x, y), else its centroid (same rule as
    clusters, so the two stages agree on where an object 'is')."""
    if getattr(o, "pose", None) is not None:
        return float(o.pose[0]), float(o.pose[1])
    if getattr(o, "centroid", None) is not None:
        return float(o.centroid[0]), float(o.centroid[1])
    return None


def _radius_of(o: Any) -> float:
    """The object's enclosing-circle radius — ``minEnclosingCircle`` of its rest footprint, else an
    area-equivalent radius (``√(area/π)``) for a still-settling object with no footprint yet, else 0."""
    fc = getattr(o, "footprint_contour", None)
    if fc is not None and len(fc) >= 1:
        (_, _), r = cv2.minEnclosingCircle(fc)
        return float(r)
    a = float(getattr(o, "area", 0) or 0.0)
    return math.sqrt(a / math.pi) if a > 0 else 0.0


def _world_contour(o: Any) -> "np.ndarray | None":
    """The member's contour in WORLD coords (for the complex-EFD blend): its clean rest footprint placed
    at the pose, else the live contour. World-frame inputs make ``merge_efd`` centre the compound shape
    at the members' true position."""
    fc = getattr(o, "footprint_contour", None)
    fcen = getattr(o, "footprint_centroid", None)
    pose = getattr(o, "pose", None)
    if fc is not None and fcen is not None and pose is not None:
        return place_footprint(fc, fcen, pose)
    return getattr(o, "contour", None)


def _member_key(o: Any) -> tuple:
    """The cache signature of ONE member's contribution to a compound — EVERYTHING ``_build_compound`` +
    ``merge_efd`` read from it: the id, the source-contour IDENTITY, pose, footprint centroid, area, θ.
    ``id(src)`` is stable frame-to-frame while the object rests (the Scene reuses the same read-only
    contour array — it's only re-allocated on a re-capture), and CHANGES on a re-settle, so a signature
    match means the merged shape is byte-identical and the EFD blend can be skipped."""
    fc = getattr(o, "footprint_contour", None)
    fcen = getattr(o, "footprint_centroid", None)
    pose = getattr(o, "pose", None)
    src = fc if (fc is not None and fcen is not None and pose is not None) else getattr(o, "contour", None)
    return (o.id, id(src), pose, fcen, float(getattr(o, "area", 0) or 0.0),
            getattr(o, "theta_cumulative", None))


def _agglomerate(px: list, py: list, r: list, area: list, budget: int,
                 shape: "list | None" = None, spatial_w: float = 0.5,
                 scale: float = 1448.0) -> list:
    """Agglomerative merge-to-budget over a BLENDED cost, not raw distance. Pure closest-pair-first
    snowballs: a merged group's enclosing circle bounds all its members, so a big group's EDGE keeps
    getting closer to everything else and it swallows object after object — one 6-member monster of
    unlike things while the rest stay singletons. And because a raw pixel gap spans orders of
    magnitude, bounded similarity terms can never outvote it — so both terms are NORMALIZED to 0..1
    and blended by ``spatial_w``:

        gap    = edge-to-edge circle gap / frame diagonal          (0..1 — proximity)
        dissim = (area dissimilarity + roundness mismatch) / 2     (0..1 — un-likeness;
                 the roundness half is inert when a member has no contour)
        cost   = (spatial_w·gap + (1−spatial_w)·dissim + ε) · (n_a + n_b)

    ``spatial_w`` 1 → group purely by proximity; 0 → purely by LIKENESS (size + shape), anywhere on
    the table — like objects share a voice across distance and the gap only breaks ties. The final
    ``n_a + n_b`` factor is BALANCE: growing an already-big group costs proportionally more, so
    merges spread instead of feeding one compound. Fuse the cheapest pair until ``≤ budget`` groups
    remain. Returns member-index lists. O(g²) per fuse over the (few) groups."""
    groups = [[i] for i in range(len(px))]
    gcx, gcy, gr, ga = list(px), list(py), list(r), list(area)     # per-group bounding circle + total area
    gsh = list(shape) if shape is not None else [None] * len(px)   # per-group blended roundness (or None)
    sw = min(1.0, max(0.0, float(spatial_w)))
    while len(groups) > budget:
        best, bc = None, None
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                gap = math.hypot(gcx[a] - gcx[b], gcy[a] - gcy[b]) - gr[a] - gr[b]
                gap = min(1.0, max(0.0, gap) / scale)
                size = 1.0 - min(ga[a], ga[b]) / max(ga[a], ga[b], 1e-9)
                shp = min(1.0, abs(gsh[a] - gsh[b])) if gsh[a] is not None and gsh[b] is not None else 0.0
                dissim = 0.5 * (size + shp)
                cost = (sw * gap + (1.0 - sw) * dissim + 1e-3) * (len(groups[a]) + len(groups[b]))
                if bc is None or cost < bc:
                    bc, best = cost, (a, b)
        a, b = best
        wa, wb = max(ga[a], 1e-9), max(ga[b], 1e-9)                # fuse b into a: area-weighted centre …
        ncx = (gcx[a] * wa + gcx[b] * wb) / (wa + wb)
        ncy = (gcy[a] * wa + gcy[b] * wb) / (wa + wb)
        ngr = max(math.hypot(gcx[a] - ncx, gcy[a] - ncy) + gr[a],  # … + a circle bounding both members
                  math.hypot(gcx[b] - ncx, gcy[b] - ncy) + gr[b])
        if gsh[a] is None:
            gsh[a] = gsh[b]
        elif gsh[b] is not None:                                   # blend the group roundness, area-weighted
            gsh[a] = (gsh[a] * wa + gsh[b] * wb) / (wa + wb)
        groups[a] += groups[b]; ga[a] += ga[b]
        gcx[a], gcy[a], gr[a] = ncx, ncy, ngr
        for lst in (groups, gcx, gcy, gr, ga, gsh):
            del lst[b]
    return groups


def _shape_score(o: Any) -> "float | None":
    """Roundness of the member's outline — radial std/mean about the centroid: ~0 for a disc, higher
    for elongated or spiky shapes. The one cheap scalar the merge cost uses to keep like with like.
    None when the object has no usable contour (the cost term then stays inert)."""
    c = _world_contour(o)
    if c is None or len(c) < 8:
        return None
    p = c.reshape(-1, 2).astype(float)
    g = p.mean(0)
    d = np.hypot(p[:, 0] - g[0], p[:, 1] - g[1])
    m = float(d.mean())
    return float(d.std() / m) if m > 1e-9 else None


class Grouper:
    """OUTPUT-side budget grouping. When the confirmed, output-bound objects outnumber the
    voice budget, fuse the nearest into ``≤ max_objects`` COMPOUND super-objects and publish the
    FUSED EMIT VIEW (``scene.set_fused``: compounds replace their members on the output wires) plus
    the ``groups`` overlay channel (``scene.set_groups``, webgui); otherwise publish neither (outputs
    emit objects as usual). Runs after pose and BEFORE features (after every
    stage that sets pose / footprint), reading the same object set the outputs will emit."""

    name = "grouper"

    # -- live parameters, declared WITH the stage (harvested into the registry + config coercion). --
    PARAMS = (
        ParamSpec("enabled", bool, True, attr="_enabled",
                  apply=lambda self, v: setattr(self, "_enabled", bool(v)),
                  doc="Enable budget grouping (off = never fuse; outputs always emit the raw objects)"),
        ParamSpec("max_objects", int, 8, lo=1, attr="_budget",
                  apply=lambda self, v: setattr(self, "_budget", max(1, int(v))),
                  doc="Voice budget: with more objects present, fuse the nearest into <= this many compound super-objects"),
        ParamSpec("spatial_weight", float, 0.5, lo=0.0, hi=1.0, attr="_spatial_w",
                  doc="How groups form: 1 = purely by proximity (nearest fuse); 0 = purely by LIKENESS (size + shape - alike objects share a voice anywhere on the table); between = blend"),
    )

    # Compound-contour reconstruction resolution (the outline the CLIENT receives as the
    # compound's own contour, the drawn hull, and the clusters enclosing-circle input) — DERIVED, not a
    # knob: the blend truncates every member's spectrum to the K harmonics the detail knob keeps,
    # so the compound curve is BAND-LIMITED at K and the K-th harmonic makes K wiggles around the
    # outline. ~8 points per wiggle is a faithful polyline; sampling beyond that manufactures no
    # information (member count / size cannot add detail past K). Clamped: 32 = a decent ellipse at
    # detail 0; 512 keeps a full-detail contour message inside the 8192-byte datagram budget.
    _PTS_PER_HARMONIC = 8
    _SAMPLES_LO, _SAMPLES_HI = 32, 512
    _DEFAULT_K = 12          # blend harmonics when no [efd] stage is wired (bare test constructions)

    def __init__(self, config: Any = None, *, logger: Any = None, enabled: "bool | None" = None,
                 efd: Any = None) -> None:
        self._log = logger or NULL_LOGGER
        bind_params(self, config)                      # _enabled / _budget ← PARAMS
        if enabled is not None:                        # the config loader POPS `enabled` out of the section
            self._enabled = bool(enabled)              # (it drives the manifest), so bind_params can't see a
        #                                                disabled section — build() passes the manifest state,
        #                                                letting the stage exist DISABLED and go live via
        #                                                /rmt/set grouper enabled 1 (the documented live on/off)
        self._efd = efd                                # the shared [efd] stage: its bins + LIVE detail knob
        #                                                set the blend budget (no grouper-private EFD knobs)
        self._prev: dict[int, int] = {}                # obj id -> group id (last frame) — STABLE group ids
        self._gcache: dict[int, tuple] = {}            # gid -> (signature, GroupSnap) — skip the EFD blend
        #                                                when a group is unchanged (the common settled case)
        self._log.info(f"[grouper] enabled={self._enabled} max_objects={self._budget}")

    def _blend_k(self) -> int:
        """Blend harmonics for the compound shape, from the SHARED [efd] settings: the linear bin
        budget ``round(1 + detail·(bins−1))`` — the same economy the detail knob applies to the
        sonification spectrum, so turning detail down simplifies the compounds too. Falls back to a
        fixed default when no [efd] stage is wired."""
        efd = self._efd
        if efd is None:
            return self._DEFAULT_K
        bins = int(getattr(efd, "bins", self._DEFAULT_K))
        detail = float(getattr(efd, "detail", 1.0))
        return max(1, round(1 + detail * (bins - 1)))

    def _contour_samples(self, k: int) -> int:
        """Reconstruction points for a blend of ``k`` harmonics: ~8 per wiggle of the finest
        harmonic, clamped — the outline the client receives carries exactly the detail the knob
        keeps, no padding at detail 0, no faceting at full detail."""
        return max(self._SAMPLES_LO, min(self._SAMPLES_HI, self._PTS_PER_HARMONIC * int(k)))

    def __call__(self, ctx: Any, scene: Any) -> None:
        if not self._enabled:                              # runtime off → never fuse (cheap no-op)
            if self._prev:
                self._prev, self._gcache = {}, {}
            scene.set_groups(())
            return
        for old, new in scene.recent_rewrites():           # a re-id moves an object → re-key persistence
            rekey(old, new, self._prev)
        retiring = scene.retiring()
        # count only what the outputs actually emit: CONFIRMED, non-ignored, PRESENT objects with a
        # position. A HIDDEN object whose spot is still COVERED (a hand hovering over it) is bodily
        # present and keeps its voice; a BARED one (truly away) is excluded — it must not consume the
        # budget and must never be fused into a compound. identity publishes the verdict per tick.
        covered = getattr(ctx, "hidden_present", ())
        objs = [o for o in scene.objects()
                if o.id not in retiring and getattr(o, "confirmed", False)
                and not getattr(o, "ignored", False)
                and (o.status is not Status.HIDDEN or o.id in covered)
                and _pos(o) is not None]
        if len(objs) <= self._budget:
            scene.set_groups(())
            self._prev, self._gcache = {}, {}              # under budget → no groups; drop persistence + cache
            return
        pos = [_pos(o) for o in objs]
        bgr = getattr(ctx, "bgr", None)
        scale = math.hypot(bgr.shape[1], bgr.shape[0]) if bgr is not None else 1448.0
        groups = _agglomerate([p[0] for p in pos], [p[1] for p in pos],
                              [_radius_of(o) for o in objs],
                              [float(getattr(o, "area", 0) or 0.0) for o in objs], self._budget,
                              shape=[_shape_score(o) for o in objs],
                              spatial_w=self._spatial_w, scale=scale)
        gids = self._assign_ids(objs, groups, scene)
        g = getattr(ctx, "gray", None)
        fa = float(g.shape[0] * g.shape[1]) if g is not None else None   # size needs the frame
        pairs = [self._compound(objs, members, gid, fa) for members, gid in zip(groups, gids)]
        self._gcache = {g: self._gcache[g] for g in gids if g in self._gcache}   # evict departed groups
        scene.set_groups(tuple(g for g, _ in pairs))
        # The FUSED EMIT VIEW: a multi-member compound REPLACES its members on the output wires
        # (a contour-less degenerate blend claims nobody — its members stay visible as themselves);
        # singletons pass through as the untouched real object, so a lone id never changes mid-scene.
        claimed: set = set()
        compounds: list = []
        for g, o in pairs:
            if o is not None:
                compounds.append(o)
                claimed.update(g.members)
        scene.set_fused(frozenset(claimed), tuple(compounds))

    def _compound(self, objs: list, members: list, gid: int, frame_area: "float | None") -> tuple:
        """One COMPOUND super-object, CACHED. The signature captures everything ``_build_compound`` reads
        (per-member id / contour identity / pose / centroid / area / θ, plus the EFD budget), so on a frame
        where the group is unchanged — the common fully-settled case — the expensive ``merge_efd`` blend is
        skipped and last frame's byte-identical (GroupSnap, ObjectSnap) pair is reused — SAME Python
        objects, so downstream caches keyed on contour identity stay warm. The signature includes each member id,
        so a re-used gid that now covers a DIFFERENT membership misses the cache and rebuilds."""
        ms = [objs[i] for i in members]
        k = self._blend_k()
        sig = (k, self._contour_samples(k), frame_area, tuple(_member_key(o) for o in ms))
        cached = self._gcache.get(gid)
        if cached is not None and cached[0] == sig:
            return cached[1]
        pair = self._build_compound(ms, gid, frame_area)
        self._gcache[gid] = (sig, pair)
        return pair

    def _build_compound(self, ms: list, gid: int, frame_area: "float | None") -> tuple:
        """Compute a compound from scratch: area-weighted centroid, the EFD blend of the members' world
        contours (``merge_efd``) MASS-SCALED so the blend's area IS the summed member area, and the
        area-dominant member's absolute θ. Returns ``(GroupSnap, ObjectSnap | None)`` — the GroupSnap
        feeds the webgui overlay channel; the ObjectSnap is the compound's stand-in on the output
        wires (None for singletons and contour-less degenerate blends).

        The mass scale is the owner's design: the harmonic blend is area-WEIGHTED (big members shape
        the compound) but averages out at a typical member's SCALE — rescaling the reconstruction
        about its centroid by sqrt(Σarea / blend_area) makes the compound contour a REAL footprint of
        the whole mass, so the features stage re-derives everything (size included) from the contour
        alone, exactly as for any object — no downstream special case."""
        areas = [float(getattr(o, "area", 0) or 0.0) for o in ms]
        total = float(sum(areas))
        ws = areas if total > 0 else [1.0] * len(ms)
        sw = sum(ws)
        cx = sum(w * _pos(o)[0] for w, o in zip(ws, ms)) / sw
        cy = sum(w * _pos(o)[1] for w, o in zip(ws, ms)) / sw
        world = [(_world_contour(o), a) for o, a in zip(ms, areas)]
        wc = [(c, a) for c, a in world if c is not None]
        k = self._blend_k()
        contour = merge_efd([c for c, _ in wc], weights=[a for _, a in wc] or None,
                            max_k=k, samples=self._contour_samples(k)) if wc else None
        if contour is not None:
            phi = _union_axis(wc, cx, cy)      # merge_efd returns the blend CANONICAL (major axis
            if phi is not None:                #   along +x) — rotate it to the members' union axis,
                rot = np.array([[math.cos(phi), -math.sin(phi)],     # so the silhouette follows the
                                [math.sin(phi), math.cos(phi)]], np.float32)   # physical mass instead
                ctr = np.array([cx, cy], dtype=np.float32)           # of lying flat on the screen
                contour = ((contour.reshape(-1, 2) - ctr) @ rot.T + ctr).reshape(-1, 1, 2).astype(np.float32)
        if contour is not None and len(ms) > 1 and total > 0:
            blend_area = float(cv2.contourArea(contour))
            if blend_area > 1e-6:
                s = math.sqrt(total / blend_area)
                ctr = np.array([cx, cy], dtype=np.float32)
                contour = ((contour.reshape(-1, 2) - ctr) * s + ctr).reshape(-1, 1, 2).astype(np.float32)
        bbox = None
        if contour is not None:
            x, y, w, h = cv2.boundingRect(contour)
            bbox = (int(x), int(y), int(w), int(h))
        dom = max(ms, key=lambda o: float(getattr(o, "area", 0) or 0.0))
        ro = readonly(contour)                       # immutable across the output-thread boundary
        gsnap = GroupSnap(id=gid, members=tuple(o.id for o in ms),
                          centroid=(float(cx), float(cy)),
                          area=total,                                  # raw, for clusters
                          size=size_fraction(total, frame_area),       # what the webgui overlay shows
                          bbox=bbox, contour=ro,
                          theta=getattr(dom, "theta_cumulative", None))
        osnap = None
        if len(ms) > 1 and contour is not None:
            # The wire stand-in: a hand-built ObjectSnap (the blessed pattern — scene.ObjectSnap
            # docstring). pose = the centroid with θ=0 and the contour already world-placed, so
            # place_footprint is the identity; the tok/pose angle rides theta_cumulative.
            osnap = ObjectSnap(
                id=gid, status=Status.RESTING, ignored=False,
                centroid=(float(cx), float(cy)), bbox=bbox, area=total,
                contour=ro, pose=(float(cx), float(cy), 0.0),
                confirmed=True,
                theta_cumulative=getattr(dom, "theta_cumulative", None),
                footprint_contour=ro, footprint_centroid=(float(cx), float(cy)),
                pose_conf=getattr(dom, "pose_conf", None))
        return gsnap, osnap

    def _assign_ids(self, objs: list, groups: list, scene: Any) -> list:
        """Match each group to a PREVIOUS group by membership overlap (reuse its id if the overlap is ≥
        half the group), else mint a new id — so a compound keeps its id as its members jiggle. Fresh
        ids come from ``scene.reserve_id()``: the SAME never-recycled sequence real objects mint from,
        so a compound id can never alias a live-or-future object id on any wire."""
        prev, used, out = self._prev, set(), []
        for group in groups:
            member_ids = {objs[i].id for i in group}
            counts: dict[int, int] = {}
            for oid in member_ids:
                pg = prev.get(oid)
                if pg is not None:
                    counts[pg] = counts.get(pg, 0) + 1
            best_g, best = None, 0
            for g, c in counts.items():
                if g not in used and c > best:
                    best, best_g = c, g
            if best_g is not None and best >= max(1, len(member_ids) // 2):
                gid = best_g
            else:
                gid = scene.reserve_id()
            used.add(gid)
            out.append(gid)
        self._prev = {objs[i].id: out[gi] for gi, group in enumerate(groups) for i in group}
        return out
