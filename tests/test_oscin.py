"""OSCIn control receiver — translation (pure) + the UDP receive path + an end-to-end
localhost round-trip through the ControlRouter. Reuses lib.osc to build the packets."""

import socket
import time

from lib.osc import decode_packet, encode_bundle, encode_message
from plugins.control.oscin.oscin import OscIn, translate


class _Cfg:
    def __init__(self, **kw):
        self._d = kw
    def get(self, k, d=None):
        return self._d.get(k, d)


class FakeInbox:
    def __init__(self):
        self.posts = []
    def post(self, name, args=()):
        self.posts.append((name, tuple(args)))
    def __len__(self):
        return len(self.posts)


class FakeTransport:
    def __init__(self):
        self.calls = []
    def pause(self):
        self.calls.append("pause")
    def step_forward(self):
        self.calls.append("fwd")
    def step_backward(self):
        self.calls.append("back")


def _wait(pred, timeout=2.0):
    end = time.perf_counter() + timeout
    while time.perf_counter() < end:
        if pred():
            return True
        time.sleep(0.005)
    return False


# --------------------------------------------------------------------------- #
# translate() — pure address/args → (command, args)
# --------------------------------------------------------------------------- #

def test_translate_transport_commands():
    assert translate("/rmt/pause", []) == ("pause", ())
    assert translate("/rmt/resume", []) == ("resume", ())
    assert translate("/rmt/toggle_pause", []) == ("toggle_pause", ())
    assert translate("/rmt/toggle", []) == ("toggle_pause", ())          # alias → canonical


def test_translate_step():
    assert translate("/rmt/step", []) == ("step", ())                    # bare → router defaults +1
    assert translate("/rmt/step", [3]) == ("step", (3,))
    assert translate("/rmt/step", [-2]) == ("step", (-2,))
    assert translate("/rmt/step", ["abc"]) is None                       # non-int → dropped


def test_translate_seek():
    # the webgui ⏮ button POSTs /rmt/seek 0 through this table — the entry MUST exist (was silently
    # dropped: the whole downstream chain, ControlRouter._seek + Transport.seek, was wired and dead).
    assert translate("/rmt/seek", []) == ("seek", ())                    # bare → router defaults to 0
    assert translate("/rmt/seek", [0]) == ("seek", (0,))
    assert translate("/rmt/seek", [120]) == ("seek", (120,))
    assert translate("/rmt/seek", ["abc"]) is None                       # non-int → dropped


def test_translate_view_and_output():
    assert translate("/rmt/view", ["debug"]) == ("view", ("debug",))
    assert translate("/rmt/view", []) is None                            # needs a name
    assert translate("/rmt/output", ["tuio2", "off"]) == ("output", ("tuio2", "off"))
    assert translate("/rmt/output", ["tuio2"]) is None                   # needs name + state


def test_translate_rec():
    assert translate("/rmt/rec", ["start"]) == ("rec", ("start",))
    assert translate("/rmt/rec", ["stop"]) == ("rec", ("stop",))
    assert translate("/rmt/rec", []) == ("rec", ("toggle",))             # bare = toggle


def test_translate_rejects_non_commands():
    assert translate("/rmt/frobnicate", []) is None                      # unknown suffix
    assert translate("/rmt/features/circularity", [5, 0.9]) is None      # a DATA message, not control
    assert translate("/other/pause", []) is None                         # wrong prefix
    assert translate("/rmtpause", []) is None                           # prefix must be followed by /


def test_translate_custom_prefix():
    assert translate("/rig/pause", [], prefix="/rig") == ("pause", ())
    assert translate("/rig/pause", [], prefix="/rig/") == ("pause", ())   # trailing slash tolerated
    assert translate("/rmt/pause", [], prefix="/rig") is None


# --------------------------------------------------------------------------- #
# feed() — decode a datagram + post recognized commands
# --------------------------------------------------------------------------- #

def test_feed_posts_recognized_command():
    o = OscIn()
    o._inbox = FakeInbox()
    o.feed(encode_message("/rmt/step", [2]))
    assert o._inbox.posts == [("step", (2,))]


def test_feed_folds_a_bundle_in_order():
    o = OscIn()
    o._inbox = FakeInbox()
    o.feed(encode_bundle([encode_message("/rmt/pause", []),
                          encode_message("/rmt/output", ["mjpeg", "off"])]))
    assert o._inbox.posts == [("pause", ()), ("output", ("mjpeg", "off"))]


def test_feed_ignores_garbage_and_unknown():
    o = OscIn()
    o._inbox = FakeInbox()
    o.feed(b"not an osc packet")                                          # must not raise
    o.feed(encode_message("/rmt/features/circularity", [5, 0.9]))        # data, not a command
    assert o._inbox.posts == []


# --------------------------------------------------------------------------- #
# end-to-end: real UDP → OscIn thread → CommandInbox → ControlRouter → transport
# --------------------------------------------------------------------------- #

def test_udp_round_trip_through_router():
    from lib.control import ControlRouter
    from lib.loop import CommandInbox

    inbox = CommandInbox()
    oscin = OscIn()
    oscin.setup(_Cfg(host="127.0.0.1", port=0))                           # port 0 → OS assigns
    oscin.start(inbox)
    try:
        port = oscin._port                                               # resolved ephemeral port
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.sendto(encode_message("/rmt/pause", []), ("127.0.0.1", port))
        s.sendto(encode_message("/rmt/step", [2]), ("127.0.0.1", port))
        assert _wait(lambda: len(inbox) >= 2), "commands never arrived on the inbox"
    finally:
        oscin.stop()
    t = FakeTransport()
    ControlRouter(transport=t).drain(inbox)
    assert t.calls == ["pause", "fwd", "fwd"]                             # applied in order


def test_stop_is_clean_and_idempotent():
    inbox = FakeInbox()
    oscin = OscIn()
    oscin.setup(_Cfg(port=0))
    oscin.start(inbox)
    oscin.stop()
    oscin.stop()                                                         # second stop must not raise


# --------------------------------------------------------------------------- #
# param verbs: /rmt/set, /rmt/reload, /rmt/query — applied here + replied to the sender
# --------------------------------------------------------------------------- #

class _FakeSock:
    def __init__(self):
        self.sent = []
    def sendto(self, data, addr):
        self.sent.append((data, addr))


class _FakeStore:
    def __init__(self):
        self.calls = []
    def set(self, s, k, v):
        self.calls.append(("set", s, k, v))
        return {"ok": True, "section": s, "key": k, "value": v, "clamped": False}
    def reload(self):
        self.calls.append(("reload",))
        return {"ok": True, "applied": ["observe.k"], "restart": ["preprocess.crop_top_x"]}
    def describe(self, section=None):
        self.calls.append(("describe", section))
        return [{"section": "observe", "key": "k", "type": "float", "current": 0.3, "default": 0.3,
                 "lo": 0.0, "hi": 1.0, "enum": None, "hot": True, "doc": "the change gate"}]


def _oscin_with(store):
    o = OscIn()
    o._params, o._sock = store, _FakeSock()
    return o


def test_set_verb_calls_store_and_acks_the_sender():
    o = _oscin_with(_FakeStore())
    o.feed(encode_message("/rmt/set", ["observe", "k", 0.5]), ("1.2.3.4", 9000))
    assert ("set", "observe", "k", 0.5) in o._params.calls
    (data, addr), = o._sock.sent
    (a, args), = decode_packet(data)
    assert a == "/rmt/ack" and args[:4] == ["observe", "k", "ok", "0.5"] and addr == ("1.2.3.4", 9000)


def test_reload_verb_replies_counts_and_restart_keys():
    o = _oscin_with(_FakeStore())
    o.feed(encode_message("/rmt/reload", []), ("1.2.3.4", 9000))
    assert ("reload",) in o._params.calls
    (data, _), = o._sock.sent
    (a, args), = decode_packet(data)
    assert a == "/rmt/reloaded" and args[0] == 1 and args[1] == 1 and args[2] == "preprocess.crop_top_x"


def test_query_verb_replies_a_manifest_bundle():
    o = _oscin_with(_FakeStore())
    o.feed(encode_message("/rmt/query", []), ("1.2.3.4", 9000))
    assert ("describe", None) in o._params.calls
    (data, _), = o._sock.sent
    msgs = decode_packet(data)                                  # one bundle → flat message list
    addrs = [a for a, _ in msgs]
    assert addrs.count("/rmt/param") == 1 and "/rmt/params" in addrs
    param = next(args for a, args in msgs if a == "/rmt/param")
    assert param[0] == "observe" and param[1] == "k" and param[2] == "float" and param[3] == 1
    assert next(args for a, args in msgs if a == "/rmt/params")[0] == 1     # the count


def test_query_with_a_section_filter():
    o = _oscin_with(_FakeStore())
    o.feed(encode_message("/rmt/query", ["pose"]), ("1.2.3.4", 9000))
    assert ("describe", "pose") in o._params.calls


def test_param_verbs_noop_without_a_store():
    o = OscIn()
    o._sock = _FakeSock()                                       # no _params attached
    o.feed(encode_message("/rmt/set", ["observe", "k", 0.5]), ("1.2.3.4", 9000))
    assert o._sock.sent == []                                   # not handled, nothing replied


def test_udp_set_applies_live_and_acks_the_real_sender():
    from types import SimpleNamespace

    from lib.loop import CommandInbox
    from lib.params import ParamStore, build_registry

    reg = build_registry({"observe": {"k": float}})            # bare fallback: a type-only hot param
    obs = SimpleNamespace(_k=0.3)
    store = ParamStore(reg, {"observe": obs}, {("observe", "k"): 0.3})
    oscin = OscIn()
    oscin.setup(_Cfg(host="127.0.0.1", port=0))
    oscin.attach_params(store)
    oscin.start(CommandInbox())
    client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    client.bind(("127.0.0.1", 0))                              # a fixed addr to receive the ack
    client.settimeout(2.0)
    try:
        client.sendto(encode_message("/rmt/set", ["observe", "k", 0.75]),
                      ("127.0.0.1", oscin._port))
        data, _ = client.recvfrom(65535)
        (a, args), = decode_packet(data)
        assert a == "/rmt/ack" and args[2] == "ok"
        assert _wait(lambda: obs._k == 0.75)                   # applied to the live stage
    finally:
        oscin.stop()
        client.close()


def test_query_reply_chunks_a_large_manifest_under_the_udp_limit():
    # regression (C blocker): the full manifest is ~17 KB — one bundle would exceed macOS's default
    # 9216-B datagram and be dropped silently. It must chunk into multiple sub-limit datagrams.
    from plugins.control.oscin.oscin import _QUERY_CHUNK_BYTES

    class _BigStore:
        def describe(self, section=None):
            return [{"section": "s", "key": f"k{i}", "type": "float", "current": 0.5, "default": 0.5,
                     "lo": 0.0, "hi": 1.0, "enum": None, "hot": True, "doc": "x" * 40} for i in range(200)]

    o = OscIn()
    o._params, o._sock = _BigStore(), _FakeSock()
    o.feed(encode_message("/rmt/query", []), ("1.2.3.4", 9000))
    sent = o._sock.sent
    assert len(sent) > 1                                           # chunked into multiple datagrams
    assert all(len(data) <= _QUERY_CHUNK_BYTES + 200 for data, _ in sent)   # each under the limit
    msgs = [m for data, _ in sent for m in decode_packet(data)]    # reassemble across chunks
    params = [a for addr, a in msgs if addr == "/rmt/param"]
    term = [a for addr, a in msgs if addr == "/rmt/params"]
    assert len(params) == 200 and term and term[0][0] == 200       # all params + the count arrive


# --------------------------------------------------------------------------- #
# fail-soft: a handler exception must never kill the receive loop
# --------------------------------------------------------------------------- #

def test_feed_survives_a_raising_param_store():
    """A store/reply bug used to unwind feed() -> _run and silently kill the oscin thread (all OSC
    control + the webgui bridge dead until restart). feed() now fail-softs per message."""
    o = OscIn()

    class BadStore:
        def save(self):
            raise RuntimeError("boom")

    o._params = BadStore()
    o.feed(encode_message("/rmt/save", []), ("127.0.0.1", 9999))   # must NOT raise


def test_feed_one_bad_message_does_not_skip_the_rest_of_the_bundle():
    o = OscIn()

    class BadStore:                                   # save raises; set works
        def __init__(self):
            self.sets = []

        def save(self):
            raise RuntimeError("boom")

        def set(self, section, key, value):
            self.sets.append((section, key, value))
            return {"ok": True, "section": section, "key": key, "value": value}

    o._params = BadStore()
    bundle = encode_bundle([encode_message("/rmt/save", []),
                            encode_message("/rmt/set", ["observe", "min_area", 42])])
    o.feed(bundle, None)                              # no reply addr → replies skipped, set still lands
    assert o._params.sets == [("observe", "min_area", 42)]
