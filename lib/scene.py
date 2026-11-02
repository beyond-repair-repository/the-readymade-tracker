"""The data model + single-writer store — the anti-god-object core.

A ``TrackedObject``'s fields are **read-only properties**; they change *only*
through the audited ``Scene`` write-API (``mint`` / ``match`` / ``set_status`` /
``set`` / ``rewrite_id`` / ``retire``). No raw ``obj.field = …`` anywhere; that one
rule is the whole contract.

Design notes:
  * **``Status`` has 4 values** — ``{SETTLING, RESTING, MOVED, HIDDEN}``,
    each an *output* of the one set-comparison. No ``BORN`` (mint
    goes straight to ``SETTLING`` — the birth debounce) and no ``REMOVED`` enum
    (retirement is a per-tick eviction, not a fifth state).
  * **no separate visibility/engagement enums** — "visible?" and "being handled?"
    both fall out of the one set-comparison, so no extra noisy state is stored.
  * **single-writer HARDENED** — a second stage writing a field it doesn't own is
    a **hard ``SceneError``** (warn-only would hide the bug). No stage can re-derive another's
    verdict because it cannot write the field.

Pure pipeline-thread state — no threads, no I/O. Unit-testable by driving ticks
directly (``begin_tick`` → writes → ``snapshot`` → ``end_tick``).
"""

from __future__ import annotations

import enum
import heapq
from contextlib import contextmanager
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterator, Mapping

import numpy as np

from lib.events import Event, readonly
from lib.logger import NULL_LOGGER


def rekey(old: int, new: int, *stores) -> None:
    """Re-key a stage's private id-keyed stores after a ``rewrite_id`` — dicts
    pop-move, sets discard-add. ONE module-level helper instead of a custom
    loop per stage. A pure function (not Scene state) so synthetic stub scenes in
    tests need nothing extra."""
    for st in stores:
        if isinstance(st, set):
            if old in st:
                st.discard(old)
                st.add(new)
        elif old in st:
            st[new] = st.pop(old)


class SceneError(Exception):
    """A write violated an invariant (illegal transition, dup id, a second writer
    to a field, a write to a retired object). Raised so the run stops with context."""


class Status(enum.Enum):
    """The one history bit a per-frame set-comparison can't recover.
    Each value is an *output* of the ``{residual, missing}`` comparison, not a
    stored guess. There is no ``BORN`` (mint → ``SETTLING``) and no ``REMOVED``
    (retirement evicts; it is not a state)."""

    SETTLING = enum.auto()   # the N-seconds capture debounce (also the mint state)
    RESTING = enum.auto()    # footprint trustworthy & current — the product state; pose pinned
    MOVED = enum.auto()      # off_footprint: a real mask to register — pose HELD at the last settled
    #                          value (outputs keep emitting it); measured ONCE when it re-settles
    HIDDEN = enum.auto()     # rest_hidden, no residual — carried/lifted; hold last pose


# Legal status transitions. One machine — there
# are only objects. Eviction (HIDDEN past TTL) is ``retire()``, not an edge here.
_LEGAL: dict[Status, frozenset[Status]] = {
    Status.SETTLING: frozenset({Status.SETTLING, Status.RESTING, Status.MOVED, Status.HIDDEN}),
    Status.RESTING: frozenset({Status.RESTING, Status.MOVED, Status.HIDDEN}),
    # MOVED→RESTING: an object that never actually relocated (residual dropped back under R —
    # a false-MOVED from a transient occlusion bridge, or a return-to-origin) resumes its rest
    # directly, keeping the footprint (no re-capture). A truly relocated object stays
    # off_footprint and settles elsewhere via SETTLING → capture.
    Status.MOVED: frozenset({Status.MOVED, Status.SETTLING, Status.RESTING, Status.HIDDEN}),
    Status.HIDDEN: frozenset({Status.HIDDEN, Status.MOVED, Status.SETTLING}),
}


@dataclass(frozen=True, slots=True)
class Detection:
    """One object-foreground blob this frame — ``observe`` produces these from
    ``findContours(RETR_EXTERNAL)`` on the object mask; ``mint`` / ``match`` consume
    them."""

    centroid: tuple[float, float]
    bbox: tuple[int, int, int, int]
    area: float
    contour: np.ndarray | None = None


# field -> default. The Scene seeds + mutates these; readers see read-only attrs.
# One registry keeps the defaults + the snapshot in lockstep. Every field traces
# to a live writing stage — no speculative slots.
_FIELDS: dict[str, Any] = {
    "id": None,
    "status": Status.SETTLING,
    "ignored": False,           # border/edge residue — excluded from the output
    "born_ts": None,            # ts this object was minted — drives the confirm-grace (classify latches
    #                             `confirmed` once it has persisted [classify].confirm_seconds)
    "confirmed": False,         # latched by classify once the object has persisted confirm_seconds. A
    #                             fresh mint (a hover blip that mints then vanishes) stays UNconfirmed and
    #                             is withheld from ALL outputs (OSC/TUIO2 + the webgui overlay); the
    #                             DEBUG masked view shows the raw foreground regardless.
    # geometry (live) — written by observe, and ONLY when the blob changed ≥ k·area from the
    # held reference (the per-object contour-change gate); sub-k jitter holds these verbatim.
    "centroid": None,
    "bbox": None,
    "contour": None,
    "area": None,
    # the REST reference — written by classify at capture; observe reads it
    # to compute residual/missing. None until the first RESTING birth.
    "footprint": None,          # rasterized rest-reference mask (uint8 0/255)
    "footprint_contour": None,  # contour_fine (approxPolyDP ε=0.8) — OSC + the overlays
    "footprint_area": None,     # int px² of the footprint (R = M = k·area uses this)
    "canon_area": None,         # LARGEST clean area ever seen (mint + captures) — the object's TRUE size.
    #                             The area gates pass a blob matching footprint_area OR canon_area, so a
    #                             half-occluded capture (footprint baked at ~½ size) can't reject the real
    #                             full husk on return (the dominant fresh-mint cause). Only ever grows.
    "footprint_centroid": None, # (x, y) rest centroid — the pinned RESTING pose (pose reads it)
    "wire_contour": None,       # the PERCEPTUAL rest outline — inverse EFD of the detail-surviving
    #                             components (features stage writes it at capture / on a detail change);
    #                             outputs place THIS at the live pose instead of approx-poly'ing the
    #                             footprint, so the wire shape IS the wire spectrum's shape
    "efd_full": None,           # the COMPLETE ± magnitude spectrum ((pos…), (neg…)), loudest-bin-normalized —
    #                             OPERATOR diagnostic (features stage writes it with wire_contour): the webgui
    #                             inspector draws the detail-REJECTED bins dim under the wire's survivors.
    #                             Rides the webgui snapshot/SSE ONLY — oscout/tuio2 never emit it.
    "embeds": None,             # tuple of appearance fingerprints (L2-normalized) — ONE per new-position settle,
    #                             building a multi-view appearance model; matched only vs HIDDEN/gone candidates
    "theta_cumulative": None,   # absolute orientation vs the canonical footprint → continuous OSC θ
    "settle_since": None,       # ts the current capture-ready streak began (the ONE N-sec timer)
    "off_since": None,          # ts off_footprint began — the RESTING→MOVED entry debounce (anti-flicker)
    "under_motion_since": None, # ts a moving mass began covering the RESTING footprint — bounds the HOLD
    # per-frame Observation record (the ONE rule's set-diffs) — written by observe, read-only
    # downstream (classify). One immutable record; no stage re-derives it.
    "observation": None,
    # pose (from the pose stage)
    "pose": None,               # (x, y, theta) — last committed pose; what outputs draw; held in HIDDEN
    "pose_conf": None,          # the accepted measurement's conf (diagnostics)
    "pose_ref": None,           # pose's canonical (contour, centroid) reference for ABSOLUTE θ — written
    #                             ONCE at the genuine first settle, carried through identity's gone-pool so
    #                             a re-acquired (relocated) id keeps its ORIGINAL orientation reference.
    #                             pose-internal (NOT projected to ObjectSnap — outputs read theta_cumulative).
}


# the fields ``_to_snap`` projects into an ObjectSnap — a write to ANY of these invalidates the
# object's cached snap (a write to a non-projected field, e.g. ``observation`` every tick, does not).
_PROJECTED_FIELDS = frozenset((
    "id", "status", "ignored", "confirmed", "centroid", "bbox", "area", "contour", "pose",
    "footprint_contour", "footprint_centroid", "wire_contour", "efd_full", "pose_conf",
    "theta_cumulative"))

# The shared, immutable "no annotations" default — the common case (every object with no
# derived data yet), so an un-annotated snapshot allocates nothing extra.
_EMPTY_ANN: Mapping[str, Any] = MappingProxyType({})


class TrackedObject:
    """One tracked object. All fields are **read-only** to everyone but the Scene
    (which mutates ``_f`` directly). Any ``obj.<field> = …`` raises — write through
    the Scene API."""

    __slots__ = ("_f", "_snap", "_ann")

    def __init__(self, **fields: Any) -> None:
        f = dict(_FIELDS)
        f.update(fields)
        object.__setattr__(self, "_f", f)
        object.__setattr__(self, "_snap", None)   # cached ObjectSnap projection; None = (re)build.
        #                                           Invalidated on any PROJECTED-field write / rewrite_id.
        object.__setattr__(self, "_ann", {})      # producer-namespaced derived data (scene.annotate)

    def __getattr__(self, name: str) -> Any:
        if name == "_f":                          # guard: pre-__init__ access
            raise AttributeError(name)
        try:
            return self._f[name]
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(
            f"TrackedObject.{name} is read-only — write via the Scene API")

    def __repr__(self) -> str:
        return f"TrackedObject(#{self._f['id']} {self._f['status'].name})"


@dataclass(frozen=True, slots=True)
class ObjectSnap:
    """Frozen per-object record for one frame. Arrays are read-only views."""

    id: int
    status: Status
    ignored: bool
    centroid: tuple[float, float] | None
    bbox: tuple[int, int, int, int] | None
    area: float | None
    contour: np.ndarray | None
    pose: tuple[float, float, float] | None
    slot: int | None = None                     # small recycled label (1,2,3…), or None if unassigned. NOT
    #   emitted anywhere today — every surface shows the tracker id (TUIO session ids must never be
    #   recycled). Kept as ready machinery for a future fixed voice-pool mode (see Scene._slot_of)
    confirmed: bool = True                      # persisted ≥ [classify].confirm_seconds → exposed to outputs;
    #   a fresh mint is withheld from OSC/TUIO2/the output overlay (debug shows all). Real objects carry
    #   the scene's explicit value (_to_snap); the default True is only for hand-built snaps (assume established)
    theta_cumulative: float | None = None       # CONTINUOUS absolute θ (pose[2] is footprint-relative and
    #   snaps to 0 on re-settle) — the one source of truth for orientation, drawn/emitted by every output
    footprint_contour: np.ndarray | None = None   # the clean rest outline — gui draws it @ pose
    footprint_centroid: tuple[float, float] | None = None
    wire_contour: np.ndarray | None = None      # the PERCEPTUAL outline (inverse EFD after the detail cut),
    #   float32 SUB-PIXEL — what every output emits/draws placed at the live pose; None → outputs fall
    #   back to the approx-poly'd footprint (features stage disabled / not yet computed)
    efd_full: tuple | None = None               # COMPLETE ± spectrum ((pos…), (neg…)), loudest-bin-normalized —
    #   webgui-SSE-only diagnostic (inspector's dim rejected bins); oscout/tuio2 never read it
    pose_conf: float | None = None              # last measurement conf — diagnostics
    annotations: Mapping[str, Any] = _EMPTY_ANN  # producer-namespaced per-object derived data (features, efd)


@dataclass(frozen=True, slots=True)
class GroupSnap:
    """One COMPOUND object — several nearby objects fused into one when the scene exceeds the output
    voice budget (``[grouper].max_objects``). Written by the Grouper stage (``scene.set_groups``),
    read by budget-aware outputs. A super-object carries the same fields an output emits per object,
    so a consumer treats it exactly like an ``ObjectSnap``: ``id`` is a STABLE group id (membership
    overlap), ``contour`` is the complex-EFD blend of the members (``lib.grouper.merge_efd``)."""

    id: int
    members: tuple[int, ...]                # the object ids fused into this group (n = len(members))
    centroid: tuple[float, float]           # area-weighted blend of member positions
    area: float                             # summed member area, RAW px² — INTERNAL, like ObjectSnap.area:
    #                                         clusters reads it for the enclosing-radius fallback and its
    #                                         cache signature. Never published.
    size: float                             # what the wire carries: the same summed mass expressed through
    #                                         the ONE size definition (lib.features.size_fraction), so a
    #                                         compound's size means what an object's size means. NOT the
    #                                         merged contour's extent — merge_efd builds an AVERAGED
    #                                         representative shape, so its area is a typical member's.
    bbox: tuple[int, int, int, int] | None  # from the merged contour
    contour: np.ndarray | None = None       # the merged compound shape (cv2 (N,1,2)), or None if degenerate
    theta: float | None = None              # the area-dominant member's absolute θ (compound orientation)


@dataclass(frozen=True, slots=True)
class ClusterSnap:
    """One CLUSTER — a spatial grouping of nearby discrete voices (proximity union-find; ``id`` persists
    across frames by membership overlap). Written by the Clusters stage (``scene.set_clusters``), read by
    the outputs (oscout's ``/rmt/cluster`` table, the webgui overlay). Cluster-centric FACTS only —
    membership + geometry — never the ring/step MEANING: a client reads ``members`` and joins them to the
    ``/pose`` table to compute its own layout. Membership is DISJOINT (each object in ≤ 1 cluster).
    Coords are RAW px (each output normalizes at emit), mirroring ``GroupSnap``."""

    id: int
    members: tuple[int, ...]                  # member object ids — DISJOINT (each object in ≤ 1 cluster)
    centroid: tuple[float, float]             # mean member position (raw px)
    bbox: tuple[float, float, float, float]   # axis-aligned x, y, w, h over the member bodies (raw px)
    circle: tuple[float, float, float]        # min-enclosing cx, cy, r over the member bodies (raw px)
    hull: np.ndarray | None = None            # convex hull polygon (cv2 (N,1,2), raw px), None if degenerate


@dataclass(frozen=True, slots=True, eq=False)
class SceneSnapshot(Event):
    """Read-only projection of the Scene at tick end — the single cross-thread
    channel to OUTPUTs. ``frame`` is the raw frame (gui background)."""

    ts: float | None
    objects: tuple[ObjectSnap, ...]
    frame: np.ndarray | None
    obj_fg: np.ndarray | None = None       # observe's object mask (bright fg) — the 'debug' masked view
    input_full: np.ndarray | None = None   # the pre-crop (flipped, full) frame — the 'input'/raw view
    groups: tuple[GroupSnap, ...] = ()      # compound super-objects when over the voice budget (Grouper); else empty
    clusters: tuple[ClusterSnap, ...] = ()  # spatial groupings of nearby voices (Clusters stage); else empty
    contour_epsilon: float = 0.75           # THE wire/UI contour-quality knob ([observe] contour_epsilon)
    hidden_present: frozenset = frozenset() # HIDDEN ids still bodily present (identity's covered verdict)
    # The FUSED EMIT VIEW (Grouper, over budget): compounds + the real objects no compound absorbed —
    # what the OUTPUT wires ship. None = no fusion this tick → outputs read ``objects`` unchanged.
    # ``objects`` itself always stays the REAL scene (webgui / MOT truth).
    emit: "tuple[ObjectSnap, ...] | None" = None

    def emit_objects(self) -> "tuple[ObjectSnap, ...]":
        """What the output wires iterate: the fused emit view when fusion is active, else the
        real objects — one call site per output, one fallback rule."""
        return self.emit if self.emit is not None else self.objects


class Scene:
    """The single-writer store.

    Per frame: ``begin_tick`` → stages write inside ``with scene.writing(stage):``
    → ``snapshot()`` → ``end_tick()`` (evicts retired objects, bounding the store).
    A write records its provenance (the writing stage) and **hard-errors** on a
    cross-stage overwrite of the same field — the stages run in a fixed order and
    own distinct fields, so a second writer is a build bug, not a runtime event."""

    def __init__(self, *, logger: Any = None, audit: Any = None) -> None:
        self._objs: dict[int, TrackedObject] = {}
        self._next_id = 1
        # DISPLAY SLOTS — a small, RECYCLED label per confirmed object (1,2,3…), shown by the outputs
        # instead of the ever-climbing raw id. ~14 objects → labels 1..~16 forever; a freed slot (an
        # object evicted) is the lowest reused, so a churned re-mint often lands the same number, and
        # sonification gets a fixed voice pool. Managed here (not a TrackedObject field): id → slot.
        self._slot_of: dict[int, int] = {}
        self._free_slots: list[int] = []      # min-heap of freed slot numbers (recycle lowest first)
        self._next_slot = 1
        self.log = logger or NULL_LOGGER
        self._audit = audit              # --debug tap; None = no-op, zero cost
        self._frame_idx = -1
        self._ts: float | None = None
        self._frame: np.ndarray | None = None
        self._input_full: np.ndarray | None = None       # pre-crop frame → the 'input'/raw view
        self._obj_fg: np.ndarray | None = None
        self._groups: tuple = ()                          # compound super-objects (Grouper) — scene-level channel
        self._fused: tuple = ()                           # compound ObjectSnaps (Grouper) — the emit view's synthetic half
        self._claimed: frozenset = frozenset()            # real ids absorbed into compounds this tick
        self._hidden_present: frozenset = frozenset()     # HIDDEN ids still bodily PRESENT (identity's verdict)
        self._clusters: tuple = ()                        # spatial clusters (Clusters) — scene-level channel
        self._contour_eps: float = 0.75                   # published by observe each tick (live knob)
        self._writer = "core"
        self._field_writer: dict[tuple[int, str], str] = {}
        self._minted: set[int] = set()                      # ids minted THIS tick (classify skips them)
        self._evict: set[int] = set()                       # retired THIS tick → dropped at end_tick
        self._rewrites: list[tuple[int, int]] = []          # (old, new) rewrite_ids THIS tick
        self._last_rewrites: tuple[tuple[int, int], ...] = ()   # …from the PREVIOUS tick (re-key)

    # -- lifecycle ---------------------------------------------------------- #

    def begin_tick(self, frame_idx: int, ts: float | None = None,
                   frame: np.ndarray | None = None, input_full: np.ndarray | None = None) -> None:
        """Open a tick: set the frame context, clear per-tick bookkeeping."""
        self._frame_idx = frame_idx
        self._ts = ts
        self._frame = frame
        self._input_full = input_full
        self._obj_fg = None
        self._groups = ()
        self._fused = ()
        self._claimed = frozenset()
        self._hidden_present = frozenset()
        self._clusters = ()
        self._field_writer.clear()
        self._minted.clear()
        # roll this tick's rewrites → "last tick's", for stages to re-key their private id maps
        # (identity runs AFTER classify, so a re-id's rewrite reaches an earlier stage next tick).
        self._last_rewrites = tuple(self._rewrites)
        self._rewrites = []

    @contextmanager
    def writing(self, stage: str) -> Iterator[None]:
        """Attribute writes inside this block to ``stage`` (provenance for the
        trace + the conflict-error). The pipeline wraps each stage in this."""
        prev = self._writer
        self._writer = stage
        try:
            yield
        finally:
            self._writer = prev

    def snapshot(self) -> SceneSnapshot:
        """Build the read-only projection of the live Scene (incl. this tick's
        retired objects, still present until ``end_tick`` so identity / the
        outputs can act on them one last time)."""
        objs = tuple(self._to_snap(o) for o in self._objs.values())
        emit = None
        if self._fused:                       # fusion active → the wires get compounds + unabsorbed reals
            emit = tuple(o for o in objs if o.id not in self._claimed) + self._fused
        return SceneSnapshot(
            frame_idx=self._frame_idx, ts=self._ts, objects=objs,
            frame=readonly(self._frame),
            obj_fg=readonly(self._obj_fg), input_full=readonly(self._input_full),
            groups=self._groups, clusters=self._clusters, contour_epsilon=self._contour_eps,
            hidden_present=self._hidden_present, emit=emit)

    def end_tick(self) -> None:
        """Close a tick: evict objects retired this tick (so the store stays bounded)."""
        for i in self._evict:
            self._objs.pop(i, None)
            s = self._slot_of.pop(i, None)        # free the display slot for reuse (lowest recycled first)
            if s is not None:
                heapq.heappush(self._free_slots, s)
        self._evict.clear()

    def set_obj_fg(self, mask: np.ndarray | None) -> None:
        """Publish this frame's object mask on the snapshot — the 'objects' view: what
        observe/pose actually see (the object pieces)."""
        self._obj_fg = mask

    def set_groups(self, groups: tuple) -> None:
        """Publish this frame's COMPOUND super-objects on the snapshot (the Grouper stage). A
        scene-level channel (like ``obj_fg``) — the groups aren't ``TrackedObject``s but a computed
        aggregate. Empty when the scene is under the voice budget (outputs then emit objects as usual)."""
        self._groups = tuple(groups)

    def set_fused(self, claimed: frozenset, compounds: tuple) -> None:
        """Publish this frame's FUSED EMIT VIEW halves (the Grouper stage, over budget only):
        ``claimed`` = real object ids absorbed into compounds, ``compounds`` = their hand-built
        ``ObjectSnap`` stand-ins. Snapshot composes ``emit`` from these; empty = no fusion."""
        self._claimed = frozenset(claimed)
        self._fused = tuple(compounds)

    def set_hidden_present(self, ids: frozenset) -> None:
        """Publish identity's covered/bared verdict: HIDDEN ids whose spot is still occupied (a hand
        over them — bodily present). The webgui distinguishes covered from GONE with it; the output
        WIRES deliberately do not consume it (they stream the raw state, owner's rule)."""
        self._hidden_present = frozenset(ids)

    def set_clusters(self, clusters: tuple) -> None:
        """Publish this frame's spatial CLUSTERS on the snapshot (the Clusters stage). A scene-level
        channel (like ``groups``) — a cluster isn't a ``TrackedObject`` but a computed aggregate over
        them. Empty when no group reaches ``min_objects``."""
        self._clusters = tuple(clusters)

    def set_contour_epsilon(self, eps: float) -> None:
        """Publish the live wire/UI contour-quality ε (observe's knob) on the snapshot."""
        self._contour_eps = float(eps)

    def groups(self) -> tuple:
        """This frame's compound super-objects (set by the Grouper; empty under the voice budget).
        A read getter for LATER stages — clusters rings over these as single discrete entities."""
        return self._groups

    def fused(self) -> "tuple[frozenset, tuple]":
        """The raw emit-view halves ``(claimed ids, compound snaps)`` — for the features stage,
        which computes on the compounds and republishes them annotated via ``set_fused``."""
        return self._claimed, self._fused

    def emit_view(self) -> tuple:
        """The live FUSED EMIT VIEW for LATER stages (features / clusters): compound snaps + the
        real objects no compound absorbed — or every live object as snaps when no fusion is active.
        Mirrors ``SceneSnapshot.emit_objects()`` so a stage and the wire agree on the population."""
        snaps = tuple(self._to_snap(o) for o in self._objs.values())
        if not self._fused:
            return snaps
        return tuple(o for o in snaps if o.id not in self._claimed) + self._fused

    # -- read getters (free) ------------------------------------------------ #

    def get(self, id: int) -> TrackedObject | None:
        return self._objs.get(id)

    def objects(self) -> tuple[TrackedObject, ...]:
        """Every live object (there are only objects)."""
        return tuple(self._objs.values())

    def recent_rewrites(self) -> tuple[tuple[int, int], ...]:
        """``(old_id, new_id)`` pairs from ``rewrite_id`` calls in the PREVIOUS tick —
        a stage with a private id-keyed map reads this at the start of its tick and
        re-keys, so an identity reclaim (a backward edge) doesn't orphan it."""
        return self._last_rewrites

    def retiring(self) -> frozenset[int]:
        """Ids retired THIS tick (evicted at ``end_tick``). They linger in the snapshot for
        identity / the outputs but are LEAVING — a dynamic stage (clusters) excludes them so a departing
        object doesn't stay in a group, while an occluded-but-present HIDDEN object still does."""
        return frozenset(self._evict)

    def minted_this_tick(self) -> frozenset[int]:
        """Ids minted since ``begin_tick`` — classify skips them (a brand-new object has no
        history, and its status was just written by observe's mint, so a same-tick transition
        would be a single-writer conflict). They are classified from the next tick."""
        return frozenset(self._minted)

    # -- write API (audited) ------------------------------------------------ #

    def reserve_id(self) -> int:
        """Draw one id from the SAME never-recycled sequence ``mint`` uses, WITHOUT creating an
        object — for synthetic entities (the grouper's compounds) that must share the id space with
        real objects so no wire ever aliases the two. Pipeline-thread only, like every write here."""
        oid = self._next_id
        self._next_id += 1
        return oid

    def mint(self, det: Detection) -> int:
        """Create a new object from a detection, in ``SETTLING`` (the birth
        debounce). Returns its id."""
        oid = self._next_id
        self._next_id += 1
        self._objs[oid] = TrackedObject(
            id=oid, status=Status.SETTLING, centroid=det.centroid, bbox=det.bbox,
            area=det.area, contour=det.contour, born_ts=self._ts,   # born_ts → the confirm grace
            canon_area=det.area)                                    # seed the true-size reference (grows at capture)
        for fld in ("status", "centroid", "bbox", "area", "contour"):
            self._field_writer[(oid, fld)] = self._writer
        self._minted.add(oid)
        if self._audit is not None:
            self._audit.on_mint(oid, det, self._writer)
        return oid

    def match(self, det: Detection, id: int) -> None:
        """Update a live object's geometry from a detection (no status change)."""
        self._require_live(id)
        self._write_field(id, "bbox", det.bbox)
        self._write_field(id, "area", det.area)
        if det.contour is not None:
            self._write_field(id, "contour", det.contour)
        self._write_field(id, "centroid", det.centroid)

    def set_status(self, id: int, new_status: Status) -> None:
        """Transition the status, validating the machine."""
        obj = self._require_live(id)
        if new_status not in _LEGAL.get(obj.status, frozenset()):
            raise SceneError(
                f"illegal transition #{id}: {obj.status.name} -> {new_status.name}")
        old = obj.status
        self._write_field(id, "status", new_status)
        if self._audit is not None:
            self._audit.on_state(id, old, new_status, self._writer)

    def set(self, id: int, field: str, value: Any) -> None:
        """Generic field write. ``id``/``status`` are rejected (use ``rewrite_id`` /
        ``set_status``); an unknown field is rejected (drift guard)."""
        self._require_live(id)
        if field in ("id", "status"):
            raise SceneError(f"set() cannot write '{field}' — use rewrite_id / set_status")
        if field not in _FIELDS:
            raise SceneError(f"set() unknown field '{field}'")
        self._write_field(id, field, value)

    def annotate(self, id: int, producer: str, value: Any) -> None:
        """Attach derived data to an object under a **producer-namespaced key** (``features``,
        ``cluster``, ``dyn``, …) — the shared bus a smart output (``oscout``) serializes and the
        gui draws. Distinct producers write distinct keys and never conflict; the SAME key written
        by two stages within one tick is a single-writer violation (a build bug), just like a field.

        Annotations PERSIST across ticks (like object fields) and project into the snapshot. So a
        stage that CACHES (pose-invariant features) should ``annotate`` only WHEN THE VALUE CHANGES
        — re-annotating an unchanged value every tick needlessly rebuilds the object's snapshot;
        a resting object annotated once at capture then costs nothing.

        A **retired** object (still in this tick's snapshot for identity/render, evicted at
        ``end_tick``) accepts no writes — but a derived annotation on a dying object is a harmless
        **no-op**, not an error (unlike a core-state write). A truly MISSING id is still a bug."""
        obj = self._objs.get(id)
        if obj is None:
            raise SceneError(f"#{id} not in scene")
        if id in self._evict:
            return                                        # retired this tick → skip (derived data on a dying obj)
        key = (id, "@" + producer)                        # namespaced so it can't collide with a field
        holder = self._field_writer.get(key)
        if holder is not None and holder != self._writer:
            raise SceneError(
                f"[conflict] #{id} annotation '{producer}': {self._writer} would overwrite "
                f"{holder} — single-writer violation")
        obj._ann[producer] = value
        object.__setattr__(obj, "_snap", None)            # annotations project into the snap → rebuild
        self._field_writer[key] = self._writer
        if self._audit is not None:
            self._audit.on_field(id, "@" + producer, value, self._writer)

    def clear_annotation(self, id: int, producer: str) -> None:
        """Remove a producer's annotation — the counterpart to ``annotate`` for a DYNAMIC producer that
        must retract a key once it no longer applies, else the stale value persists across ticks like any
        object field. No-op if absent / the object is retired or gone. (No stage clears today — clusters,
        the former user, moved to the scene-level ``clusters`` channel — but the API stays for symmetry.)"""
        obj = self._objs.get(id)
        if obj is None or id in self._evict or producer not in obj._ann:
            return
        del obj._ann[producer]
        object.__setattr__(obj, "_snap", None)
        self._field_writer.pop((id, "@" + producer), None)

    def rewrite_id(self, id: int, new_id: int) -> None:
        """Transfer an object to a new id (re-ID). Target must be free."""
        obj = self._require_live(id)
        if new_id in self._objs:
            raise SceneError(f"rewrite_id: target #{new_id} already exists")
        del self._objs[id]
        # NOTE: _field_writer entries stay keyed by the OLD id for the rest of this tick, so the
        # single-writer conflict guard doesn't cover a same-tick second write under the NEW id. Stages
        # run in fixed order and none does that today; re-keying here would buy nothing but bookkeeping.
        obj._f["id"] = new_id
        object.__setattr__(obj, "_snap", None)       # 'id' is projected → drop the stale cached snap
        self._objs[new_id] = obj
        s = self._slot_of.pop(id, None)              # a re-acquired object KEEPS its display slot (label stays)
        if s is not None:
            self._slot_of[new_id] = s
        self._rewrites.append((id, new_id))      # → recent_rewrites() next tick (stages re-key)
        if self._audit is not None:
            self._audit.on_rewrite(id, new_id, self._writer)

    def retire(self, id: int) -> None:
        """Retire an object (HIDDEN past TTL → the gone-pool). It stays
        readable + in this tick's snapshot (identity/render act on it once more),
        then is evicted at ``end_tick``. Further writes to it raise (frozen)."""
        self._require_live(id)
        self._evict.add(id)

    # -- internals ---------------------------------------------------------- #

    def _require_live(self, id: int) -> TrackedObject:
        obj = self._objs.get(id)
        if obj is None:
            raise SceneError(f"#{id} not in scene")
        if id in self._evict:
            raise SceneError(f"#{id} is retired (frozen)")
        return obj

    def _write_field(self, id: int, field: str, value: Any) -> None:
        key = (id, field)
        holder = self._field_writer.get(key)
        if holder is not None and holder != self._writer:
            raise SceneError(
                f"[conflict] #{id}.{field}: {self._writer} would overwrite {holder} "
                f"— single-writer violation")
        obj = self._objs[id]
        obj._f[field] = value
        if field in _PROJECTED_FIELDS:                # drop the cached snap so it rebuilds next snapshot
            object.__setattr__(obj, "_snap", None)
        self._field_writer[key] = self._writer
        if self._audit is not None and field != "status":   # 'status' → richer on_state line
            self._audit.on_field(id, field, value, self._writer)

    def _slot_for(self, oid: int, confirmed: bool, ignored: bool) -> "int | None":
        """The recycled display slot for an object. Assign one the first time a CONFIRMED, non-ignored
        object needs a label (lowest free); un-assigned otherwise. Freed at eviction (end_tick)."""
        s = self._slot_of.get(oid)
        if s is not None:
            return s
        if not confirmed or ignored:
            return None
        s = heapq.heappop(self._free_slots) if self._free_slots else self._next_slot
        if s == self._next_slot:
            self._next_slot += 1
        self._slot_of[oid] = s
        return s

    def _to_snap(self, o: TrackedObject) -> ObjectSnap:
        # NOTE: slot is NOT cached on the object — a lazily-assigned slot must appear even when no
        # projected field changed, so we rebuild when the object has no slot yet but now qualifies.
        f = o._f
        slot = self._slot_for(f["id"], f["confirmed"], f["ignored"])
        if o._snap is not None and o._snap.slot == slot:   # no projected field changed + slot stable → reuse
            return o._snap
        snap = ObjectSnap(
            id=f["id"], status=f["status"], ignored=f["ignored"], confirmed=f["confirmed"], slot=slot,
            centroid=f["centroid"], bbox=f["bbox"], area=f["area"],
            contour=readonly(f["contour"]), pose=f["pose"], theta_cumulative=f["theta_cumulative"],
            footprint_contour=readonly(f["footprint_contour"]),
            footprint_centroid=f["footprint_centroid"],
            wire_contour=readonly(f["wire_contour"]),
            efd_full=f["efd_full"],            # tuple-of-tuples — already immutable, no readonly() copy
            pose_conf=f["pose_conf"],
            # frozen copy so a later annotate can't mutate an already-published (cross-thread) snap;
            # the shared empty for the common un-annotated object (no allocation).
            annotations=MappingProxyType(dict(o._ann)) if o._ann else _EMPTY_ANN)
        object.__setattr__(o, "_snap", snap)
        return snap
