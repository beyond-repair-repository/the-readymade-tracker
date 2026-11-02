"""OSCIn — the control-input plugin.

The network half of the tracker's control surface: a blocking-UDP receiver thread decodes OSC with
``lib.osc`` and translates ``<prefix>/<command>`` messages into ``CommandInbox`` posts, which the
``ControlRouter`` (on the main-thread supervisor) applies. This plugin only PRODUCES commands; it
never touches the transport / gui / outputs directly — that separation is what lets the same command
arrive from an OSC packet or the webgui bridge and take the identical path.

Command set (core control, `/rmt/*` by default — matches the oscout egress namespace):
    /rmt/pause                          → pause
    /rmt/resume                         → resume
    /rmt/toggle_pause                   → toggle_pause   (alias: /rmt/toggle)
    /rmt/step        [n]                → step (n)       (default +1; n<0 steps back)
    /rmt/seek        [n]                → seek (n)       (jump to frame n; bare = 0, first frame)
    /rmt/view        <name>             → view (name)    (input|debug|processed)
    /rmt/output      <name> <on|off>    → output (name, state)
    /rmt/rec         [start|stop]       → rec (verb)     (bare = toggle; [rec] recorder)
    /rmt/trigger                        → trigger        (manual scan: capture+emit NOW, skip the dwell)

Live parameters (need a ParamStore attached; each REPLIES to the sender's address):
    /rmt/set    <section> <key> <value> → set a live param; ack /rmt/ack <section> <key> <ok|err> <…>
    /rmt/reload                         → re-read the conf; ack /rmt/reloaded <n_applied> <n_restart> <keys>
    /rmt/query  [section]               → the self-description: a /rmt/param per knob + /rmt/params <count>

Everything else is ignored (a stray packet / a data message like /rmt/features/* on the same port is
silently skipped). It is NOT a frame source, so it's wired alongside the pipeline (not in _BACKENDS):
``start(inbox)`` / ``stop()``, mirroring the output plugins' lifecycle but posting to the inbox instead
of reading the bus. The set/reload/query verbs make it BIDIRECTIONAL (it replies over its own socket).
"""

from __future__ import annotations

import socket
import threading
from typing import Any

from lib.logger import NULL_LOGGER
from lib.osc import decode_packet, encode_bundle, encode_message
from lib.params import ParamSpec, bind_params

_PARAM_CMDS = frozenset({"set", "reload", "query", "save"})       # handled here (need a reply), not the inbox
_QUERY_CHUNK_BYTES = 8000     # each /rmt/query reply bundle stays under the smallest default UDP datagram


def _chunk_bundles(msgs: list, limit: int) -> list:
    """Group encoded OSC messages into ``#bundle``s each ≤ ``limit`` bytes (bundle = 16-byte header +
    a 4-byte length prefix per element). A single oversized message still ships in its own bundle."""
    out: list = []
    batch: list = []
    size = 20                                                     # 16-byte header + slack
    for m in msgs:
        if batch and size + len(m) + 4 > limit:
            out.append(encode_bundle(batch))
            batch, size = [], 20
        batch.append(m)
        size += len(m) + 4
    if batch:
        out.append(encode_bundle(batch))
    return out

# command suffix → how to shape its OSC args into the (name, args) the ControlRouter expects.
# The command NAME is the address suffix itself (pause/resume/step/view/output) — one namespace,
# both directions. A shaper returning None (bad/missing args) drops the message.
_COMMANDS: dict[str, Any] = {
    "pause": lambda a: (),
    "resume": lambda a: (),
    "toggle_pause": lambda a: (),
    "toggle": lambda a: (),                                        # alias → emitted as toggle_pause
    "step": lambda a: (int(a[0]),) if a else (),                  # /step or /step n
    "seek": lambda a: (int(a[0]),) if a else (),                  # /seek n (bare = 0: reset to first frame)
    "view": lambda a: (str(a[0]),) if a else None,               # needs a view name
    "output": lambda a: (str(a[0]), str(a[1])) if len(a) >= 2 else None,   # needs name + state
    "rec": lambda a: (str(a[0]),) if a else ("toggle",),         # /rec start|stop|toggle (bare = toggle)
    "trigger": lambda a: (),                                     # manual scan (pedal): capture+emit NOW
    "flatfield": lambda a: (str(a[0]), int(a[1])) if len(a) >= 2 else ((str(a[0]),) if a else None),  # capture [n] | delete
    "restart": lambda a: (),                                     # re-exec the tracker (apply saved conf)
    "quit": lambda a: (),                                        # graceful shutdown + exit
}
_ALIAS = {"toggle": "toggle_pause"}                               # suffix → canonical command name


def translate(address: str, args: Any, prefix: str = "/rmt") -> "tuple[str, tuple] | None":
    """Pure: map one OSC ``(address, args)`` to a ``(command, args)`` tuple, or ``None`` to skip.
    Unknown addresses, the wrong prefix, and malformed args all return ``None`` (never raise)."""
    head = prefix.rstrip("/") + "/"
    if not address.startswith(head):
        return None
    suffix = address[len(head):]
    shaper = _COMMANDS.get(suffix)
    if shaper is None:
        return None
    try:
        shaped = shaper(list(args))
    except (ValueError, TypeError, IndexError):                  # e.g. /step "abc" → int() fails
        return None
    if shaped is None:
        return None
    return _ALIAS.get(suffix, suffix), tuple(shaped)


class OscIn:
    """UDP OSC control receiver → CommandInbox. One daemon thread; stop() closes + joins it."""

    # -- live parameters, declared WITH the plugin. All bind the listening socket at setup → needs-restart.
    #    prefix is stored /-normalized via its apply setter. --
    PARAMS = (
        ParamSpec("host", str, "0.0.0.0", hot=False, doc="Bind address (0.0.0.0 = any LAN host, 127.0.0.1 = localhost)"),
        ParamSpec("port", int, 3401, lo=0, hi=65535, hot=False, doc="UDP port to listen on for OSC control (0 = ephemeral)"),
        ParamSpec("prefix", str, "/rmt", hot=False,
                  apply=lambda self, v: setattr(self, "_prefix", "/" + str(v).strip("/")),
                  doc="OSC address prefix the control commands live under"),
    )

    def __init__(self) -> None:
        self._log = NULL_LOGGER
        self._host = "0.0.0.0"
        self._port = 3401
        self._prefix = "/rmt"
        self._sock: socket.socket | None = None
        self._inbox: Any = None
        self._params: Any = None                                 # optional ParamStore (set/reload/query)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    # -- lifecycle ----------------------------------------------------------- #

    def setup(self, config: Any, *, logger: Any = None, profiler: Any = None) -> None:
        self._log = logger or NULL_LOGGER
        bind_params(self, config)          # _host / _port / _prefix(/-normalized) ← PARAMS

    def attach_params(self, store: Any) -> None:
        """Wire the live-parameter store (built once the stages exist). Absent → /rmt/set|reload|query
        are ignored. Enables the OSC-OUT reply path (this receiver answers the querying client)."""
        self._params = store

    def start(self, inbox: Any) -> "OscIn":
        """Bind the UDP port and spawn the receiver thread. ``inbox`` is the shared CommandInbox."""
        self._inbox = inbox
        self._stop.clear()
        if self._sock is None:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._sock.bind((self._host, self._port))
            self._port = self._sock.getsockname()[1]             # resolve an ephemeral (port 0) bind
        self._sock.settimeout(0.5)                               # wake to check _stop
        self._thread = threading.Thread(target=self._run, name="oscin", daemon=True)
        self._thread.start()
        self._log.info(f"[oscin] listening on {self._host}:{self._port} (prefix {self._prefix})")
        return self

    def stop(self) -> None:
        self._stop.set()
        sock, self._sock = self._sock, None
        if sock is not None:
            try:
                sock.close()                                     # unblocks a waiting recvfrom → OSError
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    # -- receive ------------------------------------------------------------- #

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                data, addr = self._sock.recvfrom(65535)           # type: ignore[union-attr]
            except socket.timeout:
                continue
            except OSError:
                break                                            # socket closed by stop()
            try:
                self.feed(data, addr)
            except Exception as exc:                             # noqa: BLE001 — the receive loop must SURVIVE
                # any handler bug: losing this thread silently kills ALL remote control (OSC + the whole
                # webgui bridge) for the rest of the run. feed() already guards per-message; this is the
                # last-resort backstop.
                self._log.error(f"[oscin] handler failed: {exc!r} - receiver continues")

    def feed(self, data: bytes, addr: Any = None) -> None:
        """Decode one datagram. Param verbs (set/reload/query) are applied HERE + replied to ``addr``
        (they need a response); control verbs post to the inbox. Public for tests; fail-soft on garbage
        AND on handler errors — a bad packet or a store/reply exception must not take down the receive
        loop (one bad message doesn't skip the rest of its bundle either)."""
        try:
            messages = decode_packet(data)
        except Exception:                                        # noqa: BLE001 — garbage on the wire
            return
        for address, args in messages:
            try:
                if self._params is not None and self._handle_param(address, args, addr):
                    continue                                     # a param verb — handled + replied
                cmd = translate(address, args, self._prefix)
                if cmd is not None and self._inbox is not None:
                    self._inbox.post(cmd[0], cmd[1])
                    self._log.info(f"[oscin] {address} {list(args)} -> {cmd[0]}{cmd[1]}")
            except Exception as exc:                             # noqa: BLE001 — fail-soft per message
                self._log.error(f"[oscin] {address} failed: {exc!r} - skipped")

    # -- param verbs: /rmt/set, /rmt/reload, /rmt/query (reply to the sender) --- #

    def _handle_param(self, address: str, args: Any, addr: Any) -> bool:
        """Handle a param verb; return True if ``address`` was one (so the caller stops)."""
        head = self._prefix.rstrip("/") + "/"
        if not address.startswith(head):
            return False
        verb = address[len(head):]
        if verb not in _PARAM_CMDS:
            return False
        args = list(args)
        if verb == "set":
            if len(args) >= 3:                                   # /rmt/set <section> <key> <value>
                res = self._params.set(str(args[0]), str(args[1]), args[2])
                self._reply_set(res, addr)
        elif verb == "reload":
            self._reply_reload(self._params.reload(), addr)
        elif verb == "save":                                     # /rmt/save → persist to readymade.conf
            self._reply_save(self._params.save(), addr)
        elif verb == "query":                                    # /rmt/query [section]
            self._reply_query(self._params.describe(str(args[0]) if args else None), addr)
        return True

    def _reply(self, addr: Any, address: str, args: list) -> None:
        sock = self._sock                            # local ref: stop() may null self._sock concurrently
        if addr is None or sock is None:             # (TOCTOU) — a closed socket raises OSError, caught
            return
        try:
            sock.sendto(encode_message(address, args), addr)
        except OSError:
            pass

    def _reply_set(self, res: dict, addr: Any) -> None:
        p, sec, key = self._prefix, res.get("section", ""), res.get("key", "")
        if res.get("ok"):
            self._reply(addr, p + "/ack", [sec, key, "ok", str(res.get("value")), int(bool(res.get("clamped")))])
        else:
            self._reply(addr, p + "/ack", [sec, key, "err", str(res.get("reason", "error"))])

    def _reply_reload(self, res: dict, addr: Any) -> None:
        if not res.get("ok"):
            self._reply(addr, self._prefix + "/reloaded", [-1, -1, str(res.get("reason", "error"))])
            return
        applied, restart = res.get("applied", []), res.get("restart", [])
        self._reply(addr, self._prefix + "/reloaded", [len(applied), len(restart), ",".join(restart)])

    def _reply_save(self, res: dict, addr: Any) -> None:
        if not res.get("ok"):
            self._reply(addr, self._prefix + "/saved", [-1, str(res.get("reason", "error"))])
            return
        self._reply(addr, self._prefix + "/saved", [int(res.get("written", 0)), str(res.get("path", ""))])

    def _reply_query(self, manifest: list, addr: Any) -> None:
        """Reply with the full self-description as ONE bundle: a /rmt/param per knob + a /rmt/params
        terminator with the count. Scalar fields are stringified (the client parses by the `type`
        field) so the message stays uniform regardless of int/float/str/bool."""
        sock = self._sock                            # local ref (stop() may null self._sock)
        if addr is None or sock is None:
            return

        def _s(v: Any) -> str:
            return "" if v is None else str(v)

        def _eff(v: Any) -> str:                                       # effective: list → comma-string, else str
            return ",".join(str(x) for x in v) if isinstance(v, (list, tuple)) else _s(v)

        p = self._prefix
        msgs = [encode_message(p + "/param", [
            d["section"], d["key"], d["type"], int(bool(d["hot"])),
            _s(d["default"]), _s(d["current"]), _s(d["lo"]), _s(d["hi"]),
            ",".join(d["enum"]) if d["enum"] else "", d["doc"] or "",
            int(bool(d.get("multi"))), int(bool(d.get("dirty"))),      # multi=checkbox-group, dirty=unsaved
            _eff(d.get("effective")),                                  # effective=read-only resolved value ("" if none)
            int(bool(d.get("readonly"))), int(bool(d.get("hidden"))),  # readonly=show-but-locked, hidden=off-panel
        ]) for d in manifest]
        msgs.append(encode_message(p + "/params", [len(manifest)]))   # terminator: the client folds
        # CHUNK into bundles under a safe UDP payload — the full 126-param manifest is ~17 KB, which
        # exceeds macOS's default net.inet.udp.maxdgram (9216) → one sendto would be dropped SILENTLY.
        # Small bundles fit every default limit AND survive IP packet loss (a lost chunk = a few
        # missing params + a re-query, not a blank panel). The /rmt/params <count> terminator lets the
        # client know when it has them all.
        for bundle in _chunk_bundles(msgs, _QUERY_CHUNK_BYTES):
            try:
                sock.sendto(bundle, addr)
            except OSError:
                pass
