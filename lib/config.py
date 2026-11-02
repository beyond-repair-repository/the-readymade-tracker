"""Typed config loading for ReadyMade Tracker.

ONE INI-style ``readymade.conf`` is the manifest: a ``[section]`` per core module and
per *enabled* plugin. Each plugin also ships a default ``<name>.conf`` (its own
``[<name>]`` section); the loader reads those defaults first, then overlays the
matching section from ``readymade.conf``. A section runs iff it is present in
``readymade.conf`` and not ``enabled = false`` — so the section list *is* the plugin
manifest.

INI values are all strings, so each module/plugin declares a small *schema*
(key -> type) and this module coerces. The vocabulary is deliberately tiny
(int / float / bool / str / list) — extend only when a plugin needs it.

The result is read-only (``Section.values`` is a ``MappingProxyType``), matching
the read-only ``config`` handed to plugins via ``ctx``.

This module is pure (no plugin imports, no global state): feed it strings/dicts,
get a typed manifest — which is what makes it unit-testable in isolation.
"""

from __future__ import annotations

import configparser
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence


class ConfigError(Exception):
    """A config file is missing/unparseable, or a value fails its schema type."""


# --------------------------------------------------------------------------- #
# Type coercion (INI strings -> typed values, per a schema)
# --------------------------------------------------------------------------- #
#
# A *schema* maps key -> type-spec. A type-spec is one of:
#   int | float | bool | str    a scalar
#   list                        comma-separated list of `str`
#   (list, elem)                comma-separated list of `elem` (a scalar type)
#
# Keys present in the config but absent from the schema pass through unchanged
# (as `str`): the schema types what it knows and stays out of the way of the
# rest — config keys are intentionally permissive.

_BOOL_TRUE = frozenset({"true", "yes", "on", "1"})
_BOOL_FALSE = frozenset({"false", "no", "off", "0"})
_SCALARS = (int, float, bool, str)


def _coerce_bool(raw: str, where: str) -> bool:
    s = raw.strip().lower()
    if s in _BOOL_TRUE:
        return True
    if s in _BOOL_FALSE:
        return False
    raise ConfigError(
        f"{where}: expected bool (true/false/yes/no/on/off/1/0), got {raw!r}"
    )


def _coerce_scalar(raw: str, typ: type, where: str) -> Any:
    s = raw.strip()
    if typ is bool:  # before int — bool is an int subclass
        return _coerce_bool(s, where)
    if typ is str:
        return s
    if typ in (int, float):
        try:
            return typ(s)
        except ValueError as e:
            raise ConfigError(f"{where}: expected {typ.__name__}, got {raw!r}") from e
    raise ConfigError(f"{where}: unsupported scalar type {typ!r}")


def coerce_value(raw: str, spec: Any, where: str = "") -> Any:
    """Coerce one INI string ``raw`` to the type given by ``spec`` (see module doc)."""
    if spec is list:
        return [x.strip() for x in raw.split(",") if x.strip()]
    if isinstance(spec, tuple) and len(spec) == 2 and spec[0] is list:
        elem = spec[1]
        if elem not in _SCALARS:
            raise ConfigError(
                f"{where}: list element type must be a scalar, got {elem!r}"
            )
        items = [x.strip() for x in raw.split(",")]
        items = [x for x in items if x]  # drop empties (trailing comma etc.)
        return [_coerce_scalar(x, elem, f"{where}[{i}]") for i, x in enumerate(items)]
    if spec in _SCALARS:
        return _coerce_scalar(raw, spec, where)
    raise ConfigError(f"{where}: unsupported type-spec {spec!r}")


def coerce_section(
    raw: Mapping[str, str], schema: Mapping[str, Any], where: str = ""
) -> dict[str, Any]:
    """Coerce every key in ``raw`` per ``schema``; unknown keys pass through as str."""
    out: dict[str, Any] = {}
    for key, val in raw.items():
        if key in schema:
            out[key] = coerce_value(val, schema[key], f"{where}.{key}" if where else key)
        else:
            out[key] = val
    return out


# --------------------------------------------------------------------------- #
# INI parsing (stdlib configparser, no surprises)
# --------------------------------------------------------------------------- #

def _make_parser() -> configparser.ConfigParser:
    cp = configparser.ConfigParser(interpolation=None)  # no %-interpolation
    cp.optionxform = str  # preserve key case (default lowercases)
    return cp


def parse_ini_string(text: str) -> dict[str, dict[str, str]]:
    """Parse INI text -> ``{section: {key: raw_str_value}}`` (values uncoerced)."""
    cp = _make_parser()
    try:
        cp.read_string(text)
    except configparser.Error as e:
        raise ConfigError(f"INI parse error: {e}") from e
    return {name: dict(cp[name]) for name in cp.sections()}


def parse_ini_file(path: str | Path) -> dict[str, dict[str, str]]:
    """Parse an INI file from disk. Missing file -> ``ConfigError``."""
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"config file not found: {p}")
    # confs ship em-dashes / → / · — read UTF-8 explicitly (params.py already WRITES utf-8), so a
    # non-UTF-8 locale (Windows) can't double-decode the same file the saver wrote.
    return parse_ini_string(p.read_text(encoding="utf-8"))


def merge_layers(
    *layers: Mapping[str, Mapping[str, str]],
) -> dict[str, dict[str, str]]:
    """Merge raw layers per-section per-key; later layers override earlier ones."""
    merged: dict[str, dict[str, str]] = {}
    for layer in layers:
        for section, kv in layer.items():
            merged.setdefault(section, {}).update(kv)
    return merged


# --------------------------------------------------------------------------- #
# The loaded manifest
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Section:
    """Read-only typed view of one ``[section]``."""

    name: str
    values: Mapping[str, Any]  # MappingProxyType — already coerced

    def __getitem__(self, key: str) -> Any:
        return self.values[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)

    def __contains__(self, key: str) -> bool:
        return key in self.values

    def __iter__(self):
        return iter(self.values)

    def __len__(self) -> int:
        return len(self.values)


@dataclass(frozen=True)
class Config:
    """The manifest: each ``[section]`` present in ``readymade.conf`` + its enabled state."""

    sections: Mapping[str, Section]  # MappingProxyType, in readymade.conf order
    enabled: frozenset[str]
    unknown: Mapping[str, tuple[str, ...]] = field(default_factory=lambda: MappingProxyType({}))
    #   section -> keys present in the OPERATOR's conf (readymade.conf + extra_roots) that no component
    #   declares — schema-unknown, so silently ignored at load. Surfaced so the orchestrator can warn
    #   ("[conf] unknown key …"): a stale/typo'd key during live tuning is a no-op otherwise (bit us before).

    def __getitem__(self, name: str) -> Section:
        return self.sections[name]

    def get(self, name: str, default: Any = None) -> Section | None:
        return self.sections.get(name, default)

    def __contains__(self, name: str) -> bool:
        return name in self.sections

    def is_enabled(self, name: str) -> bool:
        return name in self.enabled


_ENABLED_KEY = "enabled"


def build_config(
    root: Mapping[str, Mapping[str, str]],
    *,
    defaults: Mapping[str, Mapping[str, str]] | None = None,
    schemas: Mapping[str, Mapping[str, Any]] | None = None,
    overrides: Mapping[str, Mapping[str, str]] | None = None,
) -> Config:
    """Assemble the manifest from already-parsed raw layers.

    ``root``      sections parsed from ``readymade.conf``. Presence here = part of the
                  manifest (the section list IS the manifest).
    ``defaults``  per-section default key/values (each plugin's ``<name>.conf``);
                  overlaid *under* ``root`` (root wins per key).
    ``schemas``   per-section key -> type-spec for coercion.
    ``overrides`` top-priority per-section key/values (e.g. CLI ``--video-file``);
                  overlaid *over* ``root`` (override wins). Applied only to
                  sections already in ``root`` — they never add to the manifest.

    Only sections present in ``root`` are loaded; a default-only section (a
    plugin whose ``[section]`` isn't in ``readymade.conf``) is simply absent.
    """
    defaults = defaults or {}
    schemas = schemas or {}
    overrides = overrides or {}
    sections: dict[str, Section] = {}
    enabled: set[str] = set()
    unknown: dict[str, tuple[str, ...]] = {}

    for name, root_kv in root.items():
        # precedence: overrides > root > defaults
        raw = {**defaults.get(name, {}), **root_kv, **overrides.get(name, {})}
        is_enabled = _coerce_bool(raw.pop(_ENABLED_KEY, "true"), f"{name}.{_ENABLED_KEY}")
        sch = schemas.get(name, {})
        # keys the OPERATOR wrote (root layer, not shipped defaults/CLI) that no component declares —
        # schema-unknown, silently ignored. Collected so the orchestrator can warn once at startup.
        unk = tuple(k for k in root_kv if k != _ENABLED_KEY and k not in sch)
        if unk:
            unknown[name] = unk
        coerced = coerce_section(raw, sch, where=name)
        sections[name] = Section(name, MappingProxyType(coerced))
        if is_enabled:
            enabled.add(name)

    return Config(MappingProxyType(sections), frozenset(enabled), MappingProxyType(unknown))


def load_config(
    root_path: str | Path,
    *,
    plugin_defaults: Mapping[str, str | Path] | None = None,
    schemas: Mapping[str, Mapping[str, Any]] | None = None,
    overrides: Mapping[str, Mapping[str, str]] | None = None,
    extra_roots: Sequence[str | Path] | None = None,
) -> Config:
    """Load ``readymade.conf`` (+ optional plugin default confs) from disk into a ``Config``.

    ``plugin_defaults`` maps section-name -> path of that plugin's ``<name>.conf``
    (which holds its default ``[<name>]`` section). Discovering those paths from
    the ``plugins/`` folder is the orchestrator's job; this stays pure.
    ``overrides`` is a top-priority layer (CLI flags) passed straight through.

    ``extra_roots`` are additional INI files merged *into* the root layer, in
    order, each overriding the ones before it (so the last wins) and all winning
    over ``root_path``. This is how a produced side-conf overlays the master —
    e.g. ``calibrate_camera.py`` writes ``camera1.conf`` ([camera] + [preprocess])
    and ``--source camera camera1.conf`` layers it in: its sections join the
    manifest (a ``[camera]`` absent from ``readymade.conf`` becomes present) and its
    keys override ``readymade.conf`` — while CLI ``overrides`` still win over it.
    """
    root = parse_ini_file(root_path)
    for extra in (extra_roots or ()):
        root = merge_layers(root, parse_ini_file(extra))
    defaults: dict[str, dict[str, str]] = {}
    if plugin_defaults:
        for name, path in plugin_defaults.items():
            parsed = parse_ini_file(path)
            if name in parsed:  # a <name>.conf missing its own [name] contributes nothing
                defaults[name] = parsed[name]
    return build_config(root, defaults=defaults, schemas=schemas, overrides=overrides)
