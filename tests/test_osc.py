"""lib.osc — the dependency-free OSC 1.0 codec (encode/decode round-trips).

Guards the wire format the TUIO 2.0 output (and, later, OSC-in) rides on: 4-byte
padding, big-endian ints/floats, null-terminated strings, blobs, timetags, and
nested bundles. A `python-osc` interop check (skipped if absent) confirms our bytes
decode in the de-facto reference library — the real safety net for TUIO clients.
"""

import struct

import pytest

from lib import osc
from lib.osc import (TimeTag, decode_bundle, decode_message, decode_packet,
                     encode_bundle, encode_message)


def test_message_roundtrip_all_types():
    addr, args = decode_message(encode_message("/a/b", [1, -2, 3.5, "hi", b"\x00\x01\x02"]))
    assert addr == "/a/b"
    assert args[0] == 1 and args[1] == -2
    assert abs(args[2] - 3.5) < 1e-6
    assert args[3] == "hi"
    assert args[4] == b"\x00\x01\x02"


def test_empty_arg_message():
    addr, args = decode_message(encode_message("/ping"))
    assert addr == "/ping" and args == []


@pytest.mark.parametrize("s", ["", "a", "ab", "abc", "abcd", "abcde"])
def test_string_padding_is_4_byte_aligned(s):
    # OSC strings are null-terminated then padded to a 4-byte boundary.
    enc = encode_message("/x", [s])
    assert len(enc) % 4 == 0
    _addr, args = decode_message(enc)
    assert args[0] == s


def test_blob_padding_and_length():
    for n in range(0, 9):
        enc = encode_message("/b", [bytes(range(n))])
        assert len(enc) % 4 == 0
        assert decode_message(enc)[1][0] == bytes(range(n))


def test_bool_rejected():
    # bool is an int subclass; OSC has no bool type — must not silently encode as 'i'.
    with pytest.raises(TypeError):
        encode_message("/x", [True])


def test_bundle_roundtrip_and_flatten():
    m1 = encode_message("/tuio2/frm", [7, TimeTag(None), 42, "readymade"])
    m2 = encode_message("/tuio2/tok", [5, 0, 5, 0.25, 0.5, 1.5])
    bundle = encode_bundle([m1, m2], timetag=None)
    assert osc.is_bundle(bundle)
    msgs = decode_packet(bundle)
    assert [a for a, _ in msgs] == ["/tuio2/frm", "/tuio2/tok"]
    # timetag arg survives as a TimeTag
    assert isinstance(msgs[0][1][1], TimeTag)


def test_nested_bundle():
    inner = encode_bundle([encode_message("/deep", [9])], timetag=None)
    outer = encode_bundle([inner, encode_message("/top", [1])], timetag=None)
    msgs = decode_packet(outer)
    assert sorted(a for a, _ in msgs) == ["/deep", "/top"]


def test_timetag_roundtrip():
    tt = TimeTag(1_700_000_000.5)
    back = TimeTag.from_bytes(tt.to_bytes())
    assert back.seconds is not None and abs(back.seconds - 1_700_000_000.5) < 1e-3
    assert TimeTag.from_bytes(TimeTag(None).to_bytes()).seconds is None


def test_decode_bundle_returns_elements():
    m = encode_message("/x", [1])
    tt, els = decode_bundle(encode_bundle([m, m], timetag=None))
    assert len(els) == 2 and els[0] == m


def test_pythonosc_interop():
    # cross-check: our bytes decode in the reference library the owner's TUIO clients use.
    pytest.importorskip("pythonosc")
    from pythonosc.osc_bundle import OscBundle
    from pythonosc.osc_message import OscMessage

    m = encode_message("/tuio2/tok", [5, 0, 5, 0.25, 0.5, 1.5])
    pm = OscMessage(m)
    assert pm.address == "/tuio2/tok"
    assert list(pm.params)[:3] == [5, 0, 5]
    assert abs(list(pm.params)[3] - 0.25) < 1e-6

    b = encode_bundle([m, encode_message("/tuio2/alv", [5])], timetag=None)
    pb = OscBundle(b)
    assert pb.num_contents == 2
    assert pb.content(0).address == "/tuio2/tok"


def test_big_endian_encoding():
    # sanity: an int32 is packed big-endian after the (padded) address+typetags.
    enc = encode_message("/i", [0x01020304])
    assert enc.endswith(struct.pack(">i", 0x01020304))


# --------------------------------------------------------------------------- #
# hardening: a crafted length must never wedge the decoder (single-packet DoS)
# --------------------------------------------------------------------------- #

def test_negative_bundle_element_length_is_rejected_not_looping():
    import struct
    from lib.osc import decode_bundle
    # a #bundle whose first element length is NEGATIVE would move the cursor backward → infinite loop
    # + unbounded memory. It must raise (fail-soft), not hang.
    pkt = b"#bundle\x00" + b"\x00" * 8 + struct.pack(">i", -4)
    with pytest.raises(ValueError):            # raises (fail-soft) instead of hanging — the receiver's
        decode_bundle(pkt)                      # feed() try/except then drops it; no infinite loop
    with pytest.raises(ValueError):
        decode_packet(pkt)


def test_oversized_bundle_element_length_is_rejected():
    import struct
    from lib.osc import decode_bundle
    pkt = b"#bundle\x00" + b"\x00" * 8 + struct.pack(">i", 1 << 20)   # claims 1MB in a tiny packet
    with pytest.raises(ValueError):
        decode_bundle(pkt)


def test_negative_blob_length_is_rejected():
    import struct
    # a valid ",b" message, then corrupt its blob-length field to negative
    good = encode_message("/x", [b"\x01\x02\x03\x04"])
    hdr = good[:good.index(b",b") + 4]                  # up to + incl the ",b" typetag (4-byte aligned)
    pkt = hdr + struct.pack(">i", -8)
    with pytest.raises(ValueError):
        decode_message(pkt)


def test_deeply_nested_bundles_are_depth_capped():
    # a #bundle nested inside itself many times must not blow the recursion — capped, returns [].
    from lib.osc import encode_bundle
    pkt = encode_message("/x", [1])
    for _ in range(30):
        pkt = encode_bundle([pkt])
    assert decode_packet(pkt) == []            # past the depth cap → empty (no RecursionError)
    shallow = encode_bundle([encode_bundle([encode_message("/y", [2])])])
    assert decode_packet(shallow) == [("/y", [2])]   # a few levels still decode
