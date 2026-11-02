"""Minimal, dependency-free OSC 1.0 codec (encode + decode).

Just enough Open Sound Control to speak TUIO 2.0 out (``plugins/output/tuio2``)
and OSC control in (``plugins/control/oscin``). Custom-written rather than pulling a dep: the
wire format is small and fixed, and a self-contained codec keeps the I/O plugins
dependency-light (the rest of the codebase rolls its own small formats too). Round-trip
tested, and cross-checked against ``python-osc`` in the test suite for interop.

Supported argument types (inferred from Python type on encode):
    int   -> 'i'  int32           float -> 'f'  float32
    str   -> 's'  OSC-string      bytes -> 'b'  blob
    TimeTag       -> 't'  OSC 64-bit NTP timetag  (frame time in /tuio2/frm)

A *packet* is either a single message or a ``#bundle`` (timetag + length-prefixed
elements, each itself a message or nested bundle). All integers/floats are
big-endian; strings + blobs are null-terminated and padded to 4-byte boundaries.
"""

from __future__ import annotations

import struct
from typing import Any

# Seconds between the NTP epoch (1900-01-01) and the Unix epoch (1970-01-01).
_NTP_DELTA = 2208988800
# The OSC "immediate" timetag — "dispatch now" (63 zero bits + 1).
_IMMEDIATE = b"\x00\x00\x00\x00\x00\x00\x00\x01"
_BUNDLE_TAG = b"#bundle\x00"


class TimeTag:
    """An OSC timetag. ``seconds`` is Unix time (``time.time()``); ``None`` = immediate."""

    __slots__ = ("seconds",)

    def __init__(self, seconds: float | None = None) -> None:
        self.seconds = seconds

    def to_bytes(self) -> bytes:
        if self.seconds is None:
            return _IMMEDIATE
        ntp = self.seconds + _NTP_DELTA
        sec = int(ntp)
        frac = int((ntp - sec) * (1 << 32)) & 0xFFFFFFFF
        return struct.pack(">II", sec & 0xFFFFFFFF, frac)

    @classmethod
    def from_bytes(cls, b: bytes) -> "TimeTag":
        if b == _IMMEDIATE:
            return cls(None)
        sec, frac = struct.unpack(">II", b)
        return cls((sec - _NTP_DELTA) + frac / (1 << 32))


# --------------------------------------------------------------------------- #
# Primitive (de)serialization
# --------------------------------------------------------------------------- #

def _pad(n: int) -> int:
    """Bytes needed to round ``n`` up to a 4-byte boundary."""
    return (-n) % 4


def _enc_string(s: str) -> bytes:
    raw = s.encode("utf-8") + b"\x00"
    return raw + b"\x00" * _pad(len(raw))


def _enc_blob(b: bytes) -> bytes:
    return struct.pack(">i", len(b)) + bytes(b) + b"\x00" * _pad(len(b))


def _dec_string(data: bytes, i: int) -> tuple[str, int]:
    end = data.index(b"\x00", i)
    s = data[i:end].decode("utf-8")
    return s, end + 1 + _pad(end + 1 - i)


def _dec_blob(data: bytes, i: int) -> tuple[bytes, int]:
    (n,) = struct.unpack_from(">i", data, i)
    i += 4
    if n < 0 or i + n > len(data):               # size is a SIGNED int32 on the wire — a negative or
        raise ValueError(f"OSC blob length out of range: {n}")   # oversized length must not be trusted
    return bytes(data[i:i + n]), i + n + _pad(n)


# --------------------------------------------------------------------------- #
# Messages + bundles — encode
# --------------------------------------------------------------------------- #

def encode_message(address: str, args: "list[Any] | tuple[Any, ...]" = ()) -> bytes:
    """Encode one OSC message. Arg types are inferred (see module doc)."""
    tags = ","
    payload = b""
    for a in args:
        if isinstance(a, TimeTag):
            tags += "t"
            payload += a.to_bytes()
        elif isinstance(a, bool):                 # guard: bool is an int subclass — reject, not 'i'
            raise TypeError("OSC has no bool type; pass int 0/1")
        elif isinstance(a, int):
            tags += "i"
            payload += struct.pack(">i", a)
        elif isinstance(a, float):
            tags += "f"
            payload += struct.pack(">f", a)
        elif isinstance(a, str):
            tags += "s"
            payload += _enc_string(a)
        elif isinstance(a, (bytes, bytearray)):
            tags += "b"
            payload += _enc_blob(a)
        else:
            raise TypeError(f"unsupported OSC arg type {type(a).__name__}: {a!r}")
    return _enc_string(address) + _enc_string(tags) + payload


def encode_bundle(elements: "list[bytes]", timetag: TimeTag | float | None = None) -> bytes:
    """Encode a ``#bundle`` from already-encoded ``elements`` (messages or nested
    bundles). ``timetag`` may be a ``TimeTag``, Unix seconds, or ``None`` (immediate)."""
    tt = timetag if isinstance(timetag, TimeTag) else TimeTag(timetag)
    out = _BUNDLE_TAG + tt.to_bytes()
    for el in elements:
        out += struct.pack(">i", len(el)) + el
    return out


# --------------------------------------------------------------------------- #
# Decode
# --------------------------------------------------------------------------- #

def decode_message(data: bytes) -> tuple[str, list[Any]]:
    """Decode one OSC message -> ``(address, args)``."""
    address, i = _dec_string(data, 0)
    tags, i = _dec_string(data, i)
    if not tags.startswith(","):
        raise ValueError(f"OSC type tag string must start with ',': {tags!r}")
    args: list[Any] = []
    for t in tags[1:]:
        if t == "i":
            args.append(struct.unpack_from(">i", data, i)[0]); i += 4
        elif t == "f":
            args.append(struct.unpack_from(">f", data, i)[0]); i += 4
        elif t == "s":
            s, i = _dec_string(data, i); args.append(s)
        elif t == "b":
            b, i = _dec_blob(data, i); args.append(b)
        elif t == "t":
            args.append(TimeTag.from_bytes(data[i:i + 8])); i += 8
        else:
            raise ValueError(f"unsupported OSC type tag {t!r}")
    return address, args


def is_bundle(data: bytes) -> bool:
    return data[:8] == _BUNDLE_TAG


def decode_bundle(data: bytes) -> tuple[TimeTag, list[bytes]]:
    """Decode a ``#bundle`` -> ``(timetag, raw_elements)``; each element is an
    undecoded packet (feed back to ``decode_packet``)."""
    tt = TimeTag.from_bytes(data[8:16])
    elements: list[bytes] = []
    i = 16
    while i < len(data):
        (n,) = struct.unpack_from(">i", data, i)
        i += 4
        # the element length is a SIGNED int32 on the wire. A NEGATIVE n would move the cursor
        # BACKWARD (`i += n`) → the loop never terminates + `elements` grows unbounded (a single-packet
        # DoS that wedges the receiver thread at 100% CPU). Reject n<0 and any overrun.
        if n < 0 or i + n > len(data):
            raise ValueError(f"OSC bundle element length out of range: {n}")
        elements.append(bytes(data[i:i + n]))
        i += n
    return tt, elements


def decode_packet(data: bytes, _depth: int = 0) -> list[tuple[str, list[Any]]]:
    """Decode any OSC packet -> a flat list of ``(address, args)`` messages
    (recursing into bundles). Bundle timetags are dropped (use ``decode_bundle``
    if you need them). ``_depth`` caps nested-bundle recursion — a maliciously
    deep #bundle-in-#bundle can't blow the stack."""
    if _depth > 8:
        return []
    if is_bundle(data):
        _tt, elements = decode_bundle(data)
        out: list[tuple[str, list[Any]]] = []
        for el in elements:
            out.extend(decode_packet(el, _depth + 1))
        return out
    return [decode_message(data)]


def parse_destinations(spec, host: str, port: int) -> "list[tuple[str, int]]":
    """Parse a ``destinations`` config value — ``"host:port, host:port, …"`` — into UDP send targets.
    Empty/absent → the single ``(host, port)`` default, so the plain host/port pair keeps working.
    Every packet is sent to EVERY destination (UDP fire-and-forget), which is how several clients
    (a SuperCollider patch + a monitor + another machine) watch one stream at once.

    Malformed tokens (no port, non-numeric or out-of-range port) are SKIPPED, not raised: this parses
    live ``/rmt/set … destinations`` input, and an exception here would wedge the param half-applied —
    the store's ``_current`` already updated (so ``/rmt/query`` lies) while the plugin keeps the old
    list. A bad token degrades to "that client doesn't receive"; the valid tokens still apply."""
    out = []
    for tok in str(spec or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        h, _, pp = tok.rpartition(":")
        try:
            pnum = int(pp)
        except ValueError:
            continue                                   # "host" with no/garbled port → skip the token
        if not 1 <= pnum <= 65535:
            continue
        out.append((h.strip() or "127.0.0.1", pnum))
    return out or [(str(host), int(port))]
