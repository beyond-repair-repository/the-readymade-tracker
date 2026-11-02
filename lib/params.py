"""Live parameters — the registry + store behind ``/rmt/set``, ``/rmt/reload``, ``/rmt/query``.

Every live parameter is a rich spec: type, numeric range, string enum, default, whether it **hot-applies**
(a threshold read each tick) or **needs a restart** (a bound socket, a loaded model, a built kernel), and a
one-line doc. ONE registry powers all three verbs —

  * ``set``    coerces → clamps (numeric) / validates (enum) → writes the running value (via the plugin's setter);
  * ``reload`` re-reads the conf files and applies every *changed* hot key, reporting the rest as restart-needed;
  * ``query``  serialises the specs + current values back out, so a client (a webui, SuperCollider) builds
               its UI automatically — sliders from ``[lo,hi]``, dropdowns from ``enum``, a restart badge from ``hot``.

The registry is DERIVED from each stage/plugin's own class-attr ``PARAMS`` tuple (``ParamSpec``): the ONE
place a param's type/default/range/enum/hot-ness/storage-attr/value-transform live, co-located with the code
that uses it (see ``bind_params``). ``COMPONENTS`` (in ``readymade_tracker.py``) maps a section to its owning class; that
same declaration also drives the INI coercion (``collect_schema``). No central schema, no metadata overlay.
"""

from __future__ import annotations

import os
import re
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from lib.fsutil import own_to_dir_owner
from lib.logger import NULL_LOGGER

_SECTION_RE = re.compile(r"^\s*\[([^\]]+)\]\s*$")
_KEY_RE = re.compile(r"^(\s*)([^#;=\s][^=]*?)\s*=\s*(.*)$")


def _as_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v != 0
    return str(v).strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Param:
    section: str
    key: str
    type: type                                   # int | float | str | bool
    default: Any = None
    lo: float | None = None                      # numeric min (clamp)
    hi: float | None = None                      # numeric max (clamp)
    enum: tuple[str, ...] | None = None          # allowed string values (reject others)
    hot: bool = True                             # hot-apply vs needs-rebuild
    attr: str = ""                               # storage attr on the target ("" → "_"+key)
    apply: Callable[[Any, Any], None] | None = None  # (owner, coerced) → None — the plugin's OWN setter
    doc: str = ""
    multi: bool = False                          # a COMMA-list drawn from ``enum`` (e.g. features.enable)
    synthetic: bool = False                      # not a real target attr — conf-only (the section `enabled`)
    expose: str = "auto"                         # UI exposure. "auto" (default) → hot params are EDITABLE
    #   (real-time-safe), needs-restart params are READ-ONLY (config-file territory). "edit" forces editable
    #   (a needs-restart param worth changing via the UI + restart); "readonly" forces locked; "hidden" →
    #   off the panel. A `get`-backed (computed) param is read-only too.
    get: Callable[[Any], Any] | None = None      # (live target) → the EFFECTIVE runtime value, for /query's
    #   read-only `effective` view — a computed/resolved param whose conf value is blank/derived (e.g. a
    #   value the runtime negotiates at startup: the conf holds "auto" but the effective view shows what was
    #   actually chosen). Never used by /set or save (those keep the RAW conf value).

    @property
    def target_attr(self) -> str:
        return self.attr or ("_" + self.key)


@dataclass(frozen=True)
class ParamSpec:
    """A single live parameter DECLARED on the owning stage/plugin class, in a ``PARAMS`` tuple — the ONE
    place a param's type, default, range, enum, hot-ness, storage attr, and value-transform live, next to
    the code that uses it. ``section`` is stamped on at harvest (the owner is implicit — its key in the
    ``COMPONENTS`` map). If ``apply`` is set, BOTH cold-start ``bind_params`` and a live ``/rmt/set`` call
    it, so the private attr + transform can never diverge between setup and set."""
    key: str
    type: type                                   # int | float | str | bool
    default: Any                                 # the REAL default (drives config fallback + /query)
    lo: float | None = None
    hi: float | None = None
    enum: tuple[str, ...] | None = None
    hot: bool = True                             # hot-apply vs needs-restart (staged)
    attr: str = ""                               # storage attr ("" → "_"+key)
    apply: Callable[[Any, Any], None] | None = None   # (owner, coerced) → None; supersedes plain attr setattr
    doc: str = ""
    multi: bool = False                          # comma-list drawn from enum (features.enable)
    enum_fn: Callable[[], tuple[str, ...]] | None = None   # LAZY dynamic enum (features registry)
    get: Callable[[Any], Any] | None = None      # (live target) → EFFECTIVE runtime value for /query's
    #   read-only view (a computed/resolved param whose live value the runtime negotiates at startup).
    expose: str = "auto"                         # UI exposure: "auto" (hot⇒edit, needs-restart⇒readonly) |
    #   "edit" (force editable) | "readonly" (force locked) | "hidden" (off-panel). get-backed ⇒ readonly.
    bind: bool = True                            # False → registry-only: bind_params SKIPS it (the owner
    #   reads this value itself at construction — it's threaded into a sub-object / local, not a cached
    #   attr, e.g. camera's device/backend → open_capture). Still in the registry (coercion + /query + staging).

    @property
    def target_attr(self) -> str:
        return self.attr or ("_" + self.key)


# sections that can be turned on/off from the panel (a synthetic `enabled` bool per section, current =
# config.is_enabled). The CORE pipeline (observe/classify/pose/identity) + input are always on.
_TOGGLEABLE = {"reid", "features", "clusters", "tuio2", "mjpeg", "oscout", "oscin", "rec", "webgui"}


def build_registry(schemas: dict[str, dict[str, Any]],
                   components: dict[str, Any] | None = None) -> dict[str, dict[str, Param]]:
    """Assemble the param registry by HARVESTING each component's ``PARAMS`` tuple (the plugin-owned
    model — the ONE place a param's type/default/range/enum/hot/attr/apply/doc live). A synthetic
    per-section ``enabled`` toggle is added for the toggleable sections. ``schemas`` is a legacy
    ``{section: {key: type}}`` fallback for sections WITHOUT a component (type-only, hot=True) — the
    app passes ``{}`` (every section is a component); it survives only for lightweight test fixtures."""
    components = components or {}
    reg: dict[str, dict[str, Param]] = {}
    for section in dict.fromkeys([*schemas.keys(), *components.keys()]):   # union, order-preserving
        cls = components.get(section)
        specs = getattr(cls, "PARAMS", None) if cls is not None else None
        if specs:                                           # -- plugin-owned: harvest ParamSpecs -- #
            params = {}
            for spec in specs:
                enum = spec.enum_fn() if spec.enum_fn is not None else spec.enum
                params[spec.key] = Param(
                    section=section, key=spec.key, type=spec.type, default=spec.default,
                    lo=spec.lo, hi=spec.hi, enum=tuple(enum) if enum else None, hot=spec.hot,
                    attr=spec.attr, apply=spec.apply, doc=spec.doc, multi=spec.multi, get=spec.get,
                    expose=spec.expose)
        else:                                               # -- bare fallback: type-only, hot -- #
            params = {key: Param(section=section, key=key, type=typ)
                      for key, typ in schemas.get(section, {}).items() if isinstance(typ, type)}
        if section in _TOGGLEABLE and "enabled" not in params:   # a section on/off toggle (conf-only)
            params["enabled"] = Param(section=section, key="enabled", type=bool, hot=False, synthetic=True,
                                      expose="edit",   # a meaningful UI action (turn a section on/off + restart)
                                      doc="enable / disable this section - restart to apply")
        if "enabled" in params:                        # the on/off switch always LEADS its section — in
            params = {"enabled": params.pop("enabled"), **params}   # /query order and therefore every panel
        if params:
            reg[section] = params
    return reg


def collect_schema(components: dict[str, Any] | None) -> dict[str, dict[str, type]]:
    """The INI coercion schema derived from each component's ``PARAMS`` — ``{section: {key: type}}`` — so
    ONE source (the specs) drives both config typing and the registry. This IS the coercion schema
    (the app's central schema dict is empty)."""
    return {section: {spec.key: spec.type for spec in getattr(cls, "PARAMS", ())}
            for section, cls in (components or {}).items() if getattr(cls, "PARAMS", None)}


def bind_params(owner: Any, config: Any) -> None:
    """Cold-start binding: read the owner's class ``PARAMS`` from ``config``, coerce per spec, and apply
    each ONCE — the SAME setter a live ``/rmt/set`` runs (``spec.apply`` if set, else ``setattr`` on the
    storage attr). Replaces the ``g("key", default)`` block in a component's ``__init__``/``setup``."""
    g = config.get if config is not None else (lambda k, d=None: d)
    for spec in getattr(type(owner), "PARAMS", ()):
        if not spec.bind:                        # registry-only spec — the owner reads it itself
            continue
        raw = g(spec.key, spec.default)
        val = raw if raw is None else _clamp(spec, _coerce(spec, raw))   # coerce+clamp = the live-set path
        if spec.apply is not None:
            spec.apply(owner, val)
        else:
            setattr(owner, spec.target_attr, val)


def seed_values(registry: dict[str, dict[str, Param]], config: Any) -> dict[tuple[str, str], Any]:
    """The initial 'current value' shadow — each param's value from the loaded config (coerced), or its
    default when the section/key is absent. Query reports from here; set/reload keep it in step."""
    values: dict[tuple[str, str], Any] = {}
    is_enabled = getattr(config, "is_enabled", None)
    for section, params in registry.items():
        sec = config.get(section) if config is not None and section in config else None
        for key, p in params.items():
            if key == "enabled" and is_enabled is not None:
                # the config loader POPS `enabled` out of each section (it drives is_enabled), so the
                # section view no longer has it — read the section's live on/off state directly.
                values[(section, key)] = bool(is_enabled(section))
                continue
            raw = sec.get(key, p.default) if sec is not None else p.default
            try:
                # coerce + CLAMP — the exact bind_params path, so /query's 'current' matches the value
                # the stage actually bound (an out-of-range conf entry used to be reported unclamped)
                values[(section, key)] = _clamp(p, _coerce(p, raw)) if raw is not None else None
            except (ValueError, TypeError):
                values[(section, key)] = raw
    return values


def _coerce(p: Param, value: Any) -> Any:
    return _as_bool(value) if p.type is bool else p.type(value)


def _clamp(p: Param, v: Any) -> Any:
    if p.type in (int, float):
        if p.lo is not None:
            v = max(v, p.type(p.lo))
        if p.hi is not None:
            v = min(v, p.type(p.hi))
    return v


def _fmt_conf(p: Param, value: Any) -> str:
    """A param value → its conf string (bool → true/false; None → empty; else str). Defence-in-depth:
    reject a newline so a value can never split a conf line (set() already guards the entry point)."""
    if p.type is bool:
        return "true" if _as_bool(value) else "false"
    s = "" if value is None else str(value)
    if "\n" in s or "\r" in s:
        raise ValueError("newline in conf value")
    return s


def _write_conf_inplace(path: Any, changes: dict[str, dict[str, str]]) -> int:
    """Surgically rewrite an INI conf: replace the value on each existing ``key = ...`` line, insert a
    missing key at the end of its section, append an absent section at EOF. Whole-line comments, blank
    lines, order, and every untouched section/key are preserved (configparser has no inline comments,
    so the whole post-``=`` text is the value). Returns the number of keys written."""
    p = Path(path)
    src = p.read_text(encoding="utf-8").splitlines()
    remaining = {s: dict(kv) for s, kv in changes.items() if kv}
    out: list[str] = []
    written = 0
    cur: str | None = None

    def flush(sec: str | None) -> None:                  # a section's still-missing keys → insert them
        nonlocal written
        kv = remaining.get(sec) if sec is not None else None
        if kv:
            blanks = []                                  # insert BEFORE the section's trailing blank
            while out and out[-1].strip() == "":         # lines, so a new key stays in ITS section
                blanks.append(out.pop())                 # instead of visually attaching to the next one
            out.extend(f"{k} = {v}" for k, v in kv.items())
            out.extend(reversed(blanks))
            written += len(kv)
            remaining[sec] = {}

    for line in src:
        m = _SECTION_RE.match(line)
        if m:
            flush(cur)                                   # leaving a section → flush its inserts first
            cur = m.group(1).strip()
            out.append(line)
            continue
        km = _KEY_RE.match(line)
        if km and cur in remaining and km.group(2).strip() in remaining[cur]:
            key = km.group(2).strip()
            out.append(f"{km.group(1)}{key} = {remaining[cur].pop(key)}")   # in-place value swap
            written += 1
        else:
            out.append(line)
    flush(cur)                                           # the file's last section

    for sec, kv in remaining.items():                    # sections not present in the file → append
        if kv:
            out.append(f"[{sec}]")
            out.extend(f"{k} = {v}" for k, v in kv.items())
            written += len(kv)

    # ATOMIC write — temp file in the same dir, fsync, then os.replace. A crash / kill / ENOSPC during
    # the write only ever touches the temp file; the user's readymade.conf is the complete old OR new file.
    data = "\n".join(out) + "\n"
    # mkstemp makes the temp 0600 owned by the runner — a plain os.replace would then STAMP the conf
    # 0600/root whenever the app runs under sudo (a camera backend may need root), locking a later
    # non-root run out of both reading and re-saving it. So carry the existing conf's permission bits
    # onto the temp, and when we're root hand the file back to the user who owns the repo directory.
    try:
        keep_mode = os.stat(p).st_mode & 0o777           # preserve the current conf's perms
    except OSError:
        keep_mode = 0o664                                # brand-new conf: group/world-readable default
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=p.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, keep_mode)
        own_to_dir_owner(tmp)                            # root → return ownership to the dir's owner
        os.replace(tmp, str(p))                          # atomic swap; the original survives until here
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return written


class ParamStore:
    """Applies the three verbs against the LIVE stage/plugin objects. Pure of any network/OSC — the
    transport layer (OSCIn) formats + sends the replies these return.

    THREAD-SAFE: TWO threads can drive the store (the OSCIn receiver and the mcu serial reader), so
    the verbs serialize on one lock. Without it, ``save()``'s iteration over ``_dirty`` races another
    thread's FIRST-EVER ``set()`` of a key ("Set changed size during iteration"), which unwinds the
    OSCIn receive loop and kills all remote control. The verbs are control-rate (never called from
    the pipeline thread), so the lock costs nothing where it matters."""

    def __init__(self, registry: dict[str, dict[str, Param]], targets: dict[str, Any],
                 values: dict[tuple[str, str], Any], *, reload_fn: Callable[[], Any] | None = None,
                 conf_path: Any = None, logger: Any = None,
                 layer_path: Any = None, layer_sections: "frozenset[str]" = frozenset()) -> None:
        self._reg = registry
        self._targets = dict(targets)                 # section → the running object (may omit inactive)
        self._current = dict(values)                  # (section, key) → current value (query source)
        self._reload_fn = reload_fn                    # () → a fresh Config re-read from disk
        self._conf_path = conf_path                    # readymade.conf path — where save() writes
        # a layered conf (e.g. camera1.conf) OVERRIDES readymade.conf for these sections, so save()
        # MUST write their edits there or the change is masked on the next load (the edit would look
        # unsaved). Empty layer_sections / no layer_path → everything goes to conf_path.
        self._layer_path = layer_path
        self._layer_sections = layer_sections
        self._dirty: set[tuple[str, str]] = set()      # (section, key) changed this session → save() targets
        self._staged: set[tuple[str, str]] = set()     # changed non-hot → not live yet (needs restart)
        self.on_applied: Any = None                    # optional () -> None, fired after a successful HOT
        #                                                live apply (every route: oscin / webgui / mcu) —
        #                                                build wires the paused-transport refresh here
        self._lock = threading.Lock()                  # serializes set/reload/save/describe (see class doc)
        self._log = logger or NULL_LOGGER

    def get_param(self, section: str, key: str) -> Param | None:
        return self._reg.get(section, {}).get(key)

    def current(self, section: str, key: str) -> Any:
        return self._current.get((section, key))

    # -- set --------------------------------------------------------------- #

    def set(self, section: str, key: str, value: Any) -> dict:
        """Apply one param. A HOT param is written to the live object; a needs-restart param is STAGED
        (remembered for save() + restart, not applied live). Returns a result dict (ok / reason / value
        / clamped / staged). Serialized on the store lock (multiple control threads)."""
        with self._lock:
            return self._set(section, key, value)

    def _set(self, section: str, key: str, value: Any) -> dict:
        res = {"section": section, "key": key}
        p = self.get_param(section, key)
        if p is None:
            return {**res, "ok": False, "reason": "unknown"}
        if isinstance(value, str) and ("\n" in value or "\r" in value):
            return {**res, "ok": False, "reason": "bad_value"}   # a newline would split/corrupt a conf line
        if p.multi:                                    # a comma-list drawn from enum (features.enable)
            items = [s.strip() for s in str(value).split(",") if s.strip()]
            if p.enum is not None and any(it not in p.enum for it in items):
                return {**res, "ok": False, "reason": "not_in_enum"}
            clamped = ",".join(items)
            return self._apply(p, clamped, clamped, res)
        try:
            coerced = _coerce(p, value)
        except (ValueError, TypeError):
            return {**res, "ok": False, "reason": "bad_value"}
        if p.enum is not None and coerced not in p.enum:
            return {**res, "ok": False, "reason": "not_in_enum"}
        return self._apply(p, coerced, _clamp(p, coerced), res)

    def _apply(self, p: Param, coerced: Any, clamped: Any, res: dict) -> dict:
        """Commit a validated value: STAGE it if the param needs a restart, else write it live."""
        key = (p.section, p.key)
        if not p.hot:                                  # needs-restart → stage (save() persists, restart applies)
            self._current[key] = clamped
            self._dirty.add(key)
            self._staged.add(key)
            return {**res, "ok": True, "value": clamped, "clamped": clamped != coerced, "staged": True}
        target = self._targets.get(p.section)
        if target is None and not p.synthetic:
            return {**res, "ok": False, "reason": "inactive"}
        if not p.synthetic:
            self._write(p, target, clamped)
        else:
            self._current[key] = clamped
        self._dirty.add(key)
        if not p.synthetic and self.on_applied is not None:
            try:                                       # a LIVE apply landed — poke the observer (build wires
                self.on_applied()                      #   the paused-transport refresh here, so a paused
            except Exception:                          #   operator SEES the knob's effect); a poke must
                pass                                   #   never fail the set itself
        return {**res, "ok": True, "value": clamped, "clamped": clamped != coerced, "staged": False}

    def _write(self, p: Param, target: Any, value: Any) -> None:
        # APPLY FIRST, record after: if the plugin's setter raises (e.g. a value its own parse rejects),
        # nothing is recorded — /query keeps telling the truth about what the plugin actually runs with.
        if p.apply is not None:
            p.apply(target, value)                          # the plugin's OWN setter (attr + transform in ONE place)
        else:
            setattr(target, p.target_attr, value)           # plain "_"+key idiom
        self._current[(p.section, p.key)] = value           # RAW value → /query never sees an apply-transform

    # -- reload ------------------------------------------------------------ #

    def reload(self) -> dict:
        """Re-read the conf from disk and apply every CHANGED hot key; report changed rebuild keys.
        Serialized on the store lock."""
        with self._lock:
            return self._reload()

    def _reload(self) -> dict:
        if self._reload_fn is None:
            return {"ok": False, "reason": "no_reload"}
        try:
            cfg = self._reload_fn()
        except Exception as exc:                       # noqa: BLE001 — a bad conf edit must not crash
            self._log.warning(f"[params] reload failed: {exc}")
            return {"ok": False, "reason": "load_error"}
        applied: list[str] = []
        restart: list[str] = []
        for section, params in self._reg.items():
            sec = cfg.get(section) if section in cfg else None
            for key, p in params.items():
                raw = sec.get(key, p.default) if sec is not None else p.default
                if raw is None:
                    continue
                try:
                    coerced = _coerce(p, raw)
                except (ValueError, TypeError):
                    continue
                if coerced == self._current.get((section, key)):
                    continue                           # unchanged
                if p.hot and section in self._targets:
                    self._write(p, self._targets[section], _clamp(p, coerced))
                    applied.append(f"{section}.{key}")
                else:
                    restart.append(f"{section}.{key}")
        return {"ok": True, "applied": applied, "restart": restart}

    # -- save -------------------------------------------------------------- #

    def save(self) -> dict:
        """Persist every value changed this session to readymade.conf (surgical in-place: only the changed
        keys' values, comments/order/other sections preserved). After save, staged needs-restart
        changes are on disk — a restart applies them. Returns {ok, written, path}. Serialized on the
        store lock — a concurrent set() cannot mutate ``_dirty`` mid-iteration."""
        with self._lock:
            return self._save()

    def _save(self) -> dict:
        if self._conf_path is None:
            return {"ok": False, "reason": "no_conf"}
        if not self._dirty:
            return {"ok": True, "written": 0, "path": str(self._conf_path)}
        main: dict[str, dict[str, str]] = {}
        layer: dict[str, dict[str, str]] = {}
        for (sec, key) in self._dirty:
            p = self.get_param(sec, key)
            if p is None:
                continue
            v = _fmt_conf(p, self._current.get((sec, key)))
            # sections a layered conf owns (camera/preprocess in camera mode) must be written to that
            # layer, else its values win on the next load and the edit looks unsaved.
            bucket = layer if (self._layer_path is not None and sec in self._layer_sections) else main
            bucket.setdefault(sec, {})[key] = v
        written = 0
        try:
            if main:
                written += _write_conf_inplace(self._conf_path, main)
            if layer:
                written += _write_conf_inplace(self._layer_path, layer)
        except Exception as exc:                           # noqa: BLE001 — a write hiccup must not crash
            self._log.warning(f"[params] save failed: {exc}")
            return {"ok": False, "reason": "write_error"}
        paths = [str(self._conf_path)] + ([str(self._layer_path)] if layer else [])
        self._log.info(f"[params] saved {written} key(s) -> {', '.join(paths)}")
        return {"ok": True, "written": written, "path": str(self._layer_path) if layer and not main
                else str(self._conf_path)}

    # -- query ------------------------------------------------------------- #

    def describe(self, section: str | None = None) -> list[dict]:
        """The self-description: one dict per param (type, current, default, range, enum, hot, multi,
        dirty, doc, effective). ``dirty`` = changed-but-unsaved this session (the UI badges it).
        ``effective`` = a read-only computed/resolved value (``p.get`` on the live target, e.g. a value
        the runtime negotiates at startup) — ``None`` for a normal editable param. It NEVER replaces
        ``current`` (which stays the RAW conf value save() writes), so a computed value can't be clobbered.
        Serialized on the store lock (a consistent manifest under concurrent sets)."""
        with self._lock:
            return self._describe(section)

    def _describe(self, section: str | None = None) -> list[dict]:
        out: list[dict] = []
        for sec, params in self._reg.items():
            if section is not None and sec != section:
                continue
            for key, p in params.items():
                out.append({
                    "section": sec, "key": key, "type": p.type.__name__,
                    "current": self._current.get((sec, key)), "default": p.default,
                    "lo": p.lo, "hi": p.hi, "enum": list(p.enum) if p.enum else None,
                    "hot": p.hot, "multi": p.multi, "dirty": (sec, key) in self._dirty, "doc": p.doc,
                    "effective": self._effective(sec, p),
                    # real-time policy: editable = hot (auto) or force-"edit"; else show, edit-in-conf
                    "readonly": p.get is not None or p.expose == "readonly" or (p.expose == "auto" and not p.hot),
                    "hidden": p.expose == "hidden",                            # config-file territory — off the panel
                })
        return out

    def _effective(self, section: str, p: Param) -> Any:
        """A read-only computed value for /query, or None. Fail-soft: a ``get`` reaching into a live
        (possibly torn-down) session must NEVER crash the manifest."""
        if p.get is None:
            return None
        target = self._targets.get(section)
        if target is None:
            return None
        try:
            return p.get(target)
        except Exception:                                    # noqa: BLE001 — /query must not raise
            return None
