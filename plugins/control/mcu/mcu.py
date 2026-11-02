"""``mcu`` — a serial (microcontroller) control input: a physical panel drives the tracker.

Reads newline-terminated lines from a serial port and applies each to the SAME surfaces the OSC control
input drives — the shared ``CommandInbox`` (control verbs) and the ``ParamStore`` (live parameters). A
foot pedal and a couple of knobs therefore need no new plumbing: they ride the machinery ``/rmt/*``
already uses, and anything the store exposes as a live param is reachable.

Line protocol — ONE command per line, case-insensitive, whitespace-tolerant, fail-soft::

    T                 → trigger            the manual scan: capture every settled object NOW and emit
    D <0.0..1.0>      → set [efd] detail   the EFD OSC-out detail level (the store clamps to range)

A line that is empty, unknown, or malformed is ignored — a noisy wire, a boot banner, or a half-line at
connect must never take the tracker down.

NO DEPENDENCY ON POSIX. A serial port is just a tty: on Linux/macOS the stdlib opens it and ``termios``
puts it in raw mode at the right baud — everything pyserial would do here. A foot pedal should not cost a
dependency, so that is the DEFAULT path. ``pyserial`` is only the fallback (Windows, or an exotic device
the stdlib path can't open), imported lazily; absent, the plugin logs and stays off while the tracker
still runs on numpy + OpenCV alone.

Lifecycle mirrors the OSC control input: ``setup(config)`` → ``attach_params(store)`` → ``start(inbox)``
→ ``stop()``. One daemon reader thread; ``stop()`` closes the port and joins it.
"""

from __future__ import annotations

import os
import threading
from typing import Any

from lib.logger import NULL_LOGGER
from lib.params import ParamSpec, bind_params

# verb → what it drives. The command NAME is the first token; the rest are its args.
_TRIGGER = "T"          # no args  → the ControlRouter's `trigger` verb (via the inbox)
_DETAIL = "D"           # one float → [efd] detail (via the ParamStore)


def translate(line: str) -> "tuple[str, tuple] | None":
    """Pure: map one serial LINE to a ``(kind, args)`` intent, or ``None`` to skip.

    ``("trigger", ())`` — post to the CommandInbox.
    ``("detail", (v,))`` — set [efd] detail to the float v.
    Unknown verbs, empty lines and malformed numbers all return ``None`` (never raise)."""
    parts = line.strip().split()
    if not parts:
        return None
    verb = parts[0].upper()
    if verb == _TRIGGER:
        return "trigger", ()
    if verb == _DETAIL:
        if len(parts) < 2:
            return None
        try:
            return "detail", (float(parts[1]),)
        except ValueError:                       # "D banana" → drop the line, don't raise
            return None
    return None


class _PosixSerial:
    """A dependency-free serial reader for POSIX (Linux / macOS) — the stdlib does all of it.

    Opens the tty, puts it in RAW mode (no canonical line editing, no echo, no CR/NL translation, no flow
    control) at ``baud``, with a read TIMEOUT so the reader thread can poll its stop flag instead of
    blocking forever. ``readline()`` returns one whole line, or ``b""`` on a timeout tick. This is exactly
    what pyserial would do for a USB-CDC board; doing it here keeps the pedal dependency-free."""

    def __init__(self, port: str, baud: int, timeout: float = 0.2) -> None:
        import termios
        rate = getattr(termios, f"B{baud}", None)
        if rate is None:                                  # a non-standard baud → let pyserial try
            raise ValueError(f"termios has no B{baud} (non-standard baud)")
        # O_NONBLOCK on OPEN so a port with no carrier can't hang us; then back to blocking so the
        # VMIN/VTIME timeout below governs reads.
        self._fd = os.open(port, os.O_RDONLY | os.O_NOCTTY | os.O_NONBLOCK)
        try:
            os.set_blocking(self._fd, True)
            a = termios.tcgetattr(self._fd)               # [iflag, oflag, cflag, lflag, ispeed, ospeed, cc]
            a[0] = 0                                      # iflag: no CR/NL xlate, no parity/flow mangling
            a[1] = 0                                      # oflag: no output post-processing
            a[3] = 0                                      # lflag: raw — no canonical mode, echo or signals
            a[2] |= termios.CLOCAL | termios.CREAD        # ignore modem lines, enable the receiver
            a[4] = a[5] = rate                            # ispeed / ospeed
            a[6][termios.VMIN] = 0                        # return as soon as…
            a[6][termios.VTIME] = max(1, int(timeout * 10))   # …VTIME deciseconds elapse (the timeout)
            termios.tcsetattr(self._fd, termios.TCSANOW, a)
            termios.tcflush(self._fd, termios.TCIFLUSH)   # drop whatever the board said before we listened
        except Exception:
            os.close(self._fd)
            raise
        self._buf = b""

    def readline(self) -> bytes:
        """One whole line (with its newline), or ``b""`` when the read timed out with none pending."""
        while True:
            if b"\n" in self._buf:
                line, _, self._buf = self._buf.partition(b"\n")
                return line + b"\n"
            chunk = os.read(self._fd, 256)
            if not chunk:                                 # VTIME timeout → let the caller poll _stopping
                return b""
            self._buf += chunk
            if len(self._buf) > 8192:                     # a newline-free stream (wrong baud / binary
                self._buf = self._buf[-256:]              #   noise) must not grow without bound — keep
                #                                             the tail; a real line ends it anyway

    def close(self) -> None:
        os.close(self._fd)


class Mcu:
    """Serial control receiver → CommandInbox + ParamStore. One daemon thread; stop() closes + joins it."""

    # -- live parameters, declared WITH the plugin. port/baud open the device at start() → needs-restart. --
    PARAMS = (
        ParamSpec("enabled", bool, None, hot=False, bind=False, expose="edit",
                  doc="Master switch for the serial (microcontroller) control input"),
        ParamSpec("port", str, "", hot=False,
                  doc="Serial device to read (e.g. /dev/tty.usbmodem1101 or COM3); empty = disabled"),
        ParamSpec("baud", int, 115200, lo=300, hi=2000000, hot=False,
                  doc="Serial baud rate - must match the board's firmware"),
    )

    def __init__(self) -> None:
        self._log = NULL_LOGGER
        self._port = ""
        self._baud = 115200
        self._inbox: Any = None
        self._params: Any = None
        self._ser: Any = None
        self._stopping = threading.Event()
        self._thread: threading.Thread | None = None

    def setup(self, config: Any, *, logger: Any = None, profiler: Any = None) -> None:
        self._log = logger or NULL_LOGGER
        bind_params(self, config)          # _port / _baud ← PARAMS

    def attach_params(self, store: Any) -> None:
        """The live-parameter store — the D verb sets [efd] detail through it (same path as /rmt/set)."""
        self._params = store

    def _open(self, quiet: bool = False) -> Any:
        """Open the port: the STDLIB on POSIX (no dependency), pyserial as the fallback (Windows, or a
        device/baud the stdlib path can't take). Returns a reader with ``readline()``/``close()``, or None
        — every failure degrades to "logged and off", never fatal. ``quiet`` mutes the per-attempt
        logging (the reopen retry loop polls every 5 s — one line per failure would flood an 8-h log)."""
        log = NULL_LOGGER if quiet else self._log
        if os.name == "posix":
            try:
                r = _PosixSerial(self._port, self._baud)
                self._log.info(f"[mcu] reading {self._port} @ {self._baud} (stdlib termios, no dependency)")
                return r
            except Exception as e:                           # missing / busy / permissions / odd baud
                log.warning(f"[mcu] stdlib open of {self._port} failed "
                            f"({type(e).__name__}: {e}) — trying pyserial")
        try:
            import serial                                    # OPTIONAL dep, only reached as the fallback
        except ImportError:
            log.error(f"[mcu] cannot open {self._port}: no stdlib path and pyserial is not "
                      f"installed (pip install pyserial) — not started")
            return None
        try:
            r = serial.Serial(self._port, self._baud, timeout=0.2)
        except Exception as e:                               # busy / absent / permission — degrade, don't die
            log.error(f"[mcu] cannot open {self._port} @ {self._baud}: {type(e).__name__}: {e}")
            return None
        self._log.info(f"[mcu] reading {self._port} @ {self._baud} (pyserial)")
        return r

    def start(self, inbox: Any) -> "Mcu":
        """Open the serial port and spawn the reader thread. ``inbox`` is the shared CommandInbox.
        An unopenable port is logged and left off — never fatal."""
        self._inbox = inbox
        if not self._port:
            self._log.warning("[mcu] no port configured ([mcu] port) - not started")
            return self
        self._ser = self._open()
        if self._ser is None:
            return self
        self._stopping.clear()
        self._thread = threading.Thread(target=self._run, name="mcu", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        """Close the port + join the reader. Idempotent."""
        self._stopping.set()
        ser, self._ser = self._ser, None
        if ser is not None:
            try:
                ser.close()
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _run(self) -> None:
        while not self._stopping.is_set():
            try:
                raw = self._ser.readline()                   # type: ignore[union-attr]
            except Exception:                                # unplugged mid-read → try to get it back:
                if self._stopping.is_set():                  #   a replugged pedal must NOT need a full
                    return                                   #   tracker restart (8-hour installation)
                self._log.warning("[mcu] serial read failed (unplugged?) - retrying every 5 s")
                if not self._reopen_until_stopped():
                    return
                continue
            if not raw:
                continue                                     # timeout tick — just poll _stopping again
            self.feed(raw.decode("utf-8", "replace"))

    def _reopen_until_stopped(self) -> bool:
        """Close the dead port, then retry ``_open`` every 5 s (quietly) until the device is back or
        ``stop()`` is called. True = reopened; False = stopping."""
        ser, self._ser = self._ser, None
        try:
            if ser is not None:
                ser.close()
        except Exception:                                    # noqa: BLE001 — it is already dead
            pass
        while not self._stopping.wait(5.0):
            ser = self._open(quiet=True)
            if ser is not None:
                self._ser = ser
                self._log.info("[mcu] serial port reopened")
                return True
        return False

    def feed(self, line: str) -> None:
        """Apply ONE serial line. Public for tests (no port needed). Fail-soft on anything unexpected."""
        cmd = translate(line)
        if cmd is None:
            return
        kind, args = cmd
        if kind == "trigger":
            if self._inbox is not None:
                self._inbox.post("trigger", ())
                self._log.info("[mcu] T -> trigger")
        elif kind == "detail":
            if self._params is None:
                self._log.warning("[mcu] D: no parameter store attached")
                return
            res = self._params.set("efd", "detail", args[0])
            if res.get("ok"):
                self._log.info(f"[mcu] D {args[0]} -> efd.detail = {res.get('value')}")
            else:
                self._log.warning(f"[mcu] D {args[0]} rejected: {res.get('reason', 'error')}")
