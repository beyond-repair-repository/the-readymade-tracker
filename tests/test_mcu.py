"""``mcu`` — the serial control input: a physical panel drives the tracker through the SAME surfaces
the OSC control input uses (the CommandInbox + the ParamStore).

Line protocol:  T → trigger (the manual scan)   ·   D <0.0..1.0> → [efd] detail
Everything else is ignored — a boot banner, line noise or a half-line at connect must never raise.

The POSIX path is dependency-free (stdlib open + termios). The tty tests below run against a real pty,
so they exercise that path end to end with no board and no pyserial."""

import os
import time
from types import SimpleNamespace

import pytest

from plugins.control.mcu.mcu import Mcu, _PosixSerial, translate

posix_only = pytest.mark.skipif(os.name != "posix", reason="stdlib termios path is POSIX-only")


class _Inbox:
    def __init__(self):
        self.posted = []

    def post(self, name, args):
        self.posted.append((name, args))


class _Store:
    def __init__(self, ok=True, value=None, reason="bad_value"):
        self.sets = []
        self._ok, self._value, self._reason = ok, value, reason

    def set(self, section, key, value):
        self.sets.append((section, key, value))
        return ({"ok": True, "value": self._value if self._value is not None else value}
                if self._ok else {"ok": False, "reason": self._reason})


def _mcu(inbox=None, store=None):
    m = Mcu()
    m.setup(SimpleNamespace(get=lambda k, d=None: {"port": "", "baud": 115200}.get(k, d)))
    m._inbox = inbox
    m._params = store
    return m


# --------------------------------------------------------------------------- #
# the pure line parser
# --------------------------------------------------------------------------- #

def test_translate_the_two_verbs():
    assert translate("T") == ("trigger", ())
    assert translate("D 0.4") == ("detail", (0.4,))


def test_translate_is_case_and_whitespace_tolerant():
    assert translate(" t \r\n") == ("trigger", ())
    assert translate("d   0.75\n") == ("detail", (0.75,))


def test_translate_drops_anything_it_does_not_know():
    for junk in ("", "   ", "\n", "X", "hello world", "D", "D banana", "TT", "0.5"):
        assert translate(junk) is None, junk


# --------------------------------------------------------------------------- #
# T → the trigger verb, on the shared inbox
# --------------------------------------------------------------------------- #

def test_T_posts_the_trigger_command():
    inbox = _Inbox()
    _mcu(inbox=inbox).feed("T\n")
    assert inbox.posted == [("trigger", ())]          # the SAME verb /rmt/trigger posts


def test_T_without_an_inbox_is_a_no_op():
    _mcu().feed("T\n")                                # never raises


# --------------------------------------------------------------------------- #
# D → [efd] detail, through the live-parameter store
# --------------------------------------------------------------------------- #

def test_D_sets_efd_detail():
    store = _Store()
    _mcu(store=store).feed("D 0.42\n")
    assert store.sets == [("efd", "detail", 0.42)]    # the SAME path /rmt/set efd detail uses


def test_D_out_of_range_is_left_to_the_store_to_clamp():
    store = _Store(value=1.0)                          # the store clamps to the ParamSpec's 0..1
    _mcu(store=store).feed("D 5\n")
    assert store.sets == [("efd", "detail", 5.0)]      # mcu forwards; it does not second-guess range


def test_D_rejected_by_the_store_is_logged_not_raised():
    _mcu(store=_Store(ok=False)).feed("D 0.5\n")       # never raises


def test_D_without_a_store_is_a_no_op():
    _mcu().feed("D 0.5\n")                             # never raises


# --------------------------------------------------------------------------- #
# fail-soft: junk on the wire must not disturb anything
# --------------------------------------------------------------------------- #

def test_junk_lines_touch_neither_inbox_nor_store():
    inbox, store = _Inbox(), _Store()
    m = _mcu(inbox=inbox, store=store)
    for junk in ("", "\n", "boot ok\n", "D\n", "D nope\n", "?\n", "TRIGGER\n"):
        m.feed(junk)
    assert inbox.posted == [] and store.sets == []


def test_a_stream_of_mixed_lines_applies_only_the_valid_ones():
    inbox, store = _Inbox(), _Store()
    m = _mcu(inbox=inbox, store=store)
    for line in ("boot\n", "T\n", "D 0.1\n", "junk\n", "t\n", "D 0.9\n", "D x\n"):
        m.feed(line)
    assert inbox.posted == [("trigger", ()), ("trigger", ())]
    assert store.sets == [("efd", "detail", 0.1), ("efd", "detail", 0.9)]


# --------------------------------------------------------------------------- #
# lifecycle: no port / no pyserial degrades quietly, never fatal
# --------------------------------------------------------------------------- #

def test_start_without_a_port_does_not_start_or_raise():
    m = _mcu()
    m.start(_Inbox())                                  # port = "" → logged, stays off
    assert m._thread is None
    m.stop()                                           # idempotent, no thread to join


def test_start_on_a_bogus_port_degrades_and_does_not_raise():
    m = Mcu()
    m.setup(SimpleNamespace(get=lambda k, d=None: {"port": "/dev/does-not-exist", "baud": 115200}.get(k, d)))
    m.start(_Inbox())                                  # stdlib fails → pyserial absent/fails → off
    assert m._thread is None
    m.stop()


# --------------------------------------------------------------------------- #
# the dependency-free POSIX reader, against a REAL tty (a pty) — no board, no pyserial
# --------------------------------------------------------------------------- #

@posix_only
def test_posix_serial_reads_whole_lines_from_a_real_tty():
    master, slave = os.openpty()
    try:
        r = _PosixSerial(os.ttyname(slave), 115200)
        os.write(master, b"T\n")
        assert r.readline() == b"T\n"
        os.write(master, b"D 0.4\n")
        assert r.readline() == b"D 0.4\n"
        r.close()
    finally:
        os.close(master)
        os.close(slave)


@posix_only
def test_posix_serial_reassembles_a_split_write_and_splits_a_batched_one():
    master, slave = os.openpty()
    try:
        r = _PosixSerial(os.ttyname(slave), 115200)
        os.write(master, b"D 0.")                       # a line arriving in pieces
        time.sleep(0.05)
        os.write(master, b"75\n")
        assert r.readline() == b"D 0.75\n"
        os.write(master, b"T\nD 1.0\n")                 # two lines in ONE write
        assert r.readline() == b"T\n"
        assert r.readline() == b"D 1.0\n"
        r.close()
    finally:
        os.close(master)
        os.close(slave)


@posix_only
def test_posix_serial_times_out_instead_of_blocking_forever():
    master, slave = os.openpty()
    try:
        r = _PosixSerial(os.ttyname(slave), 115200, timeout=0.2)
        t0 = time.time()
        assert r.readline() == b""                      # nothing pending → a timeout TICK, not a hang
        assert time.time() - t0 < 1.0                   # (so the reader thread can poll its stop flag)
        r.close()
    finally:
        os.close(master)
        os.close(slave)


@posix_only
def test_mcu_drives_the_inbox_and_store_from_a_real_tty():
    """The whole plugin over a real tty: open → reader thread → parse → inbox + store. No pyserial."""
    master, slave = os.openpty()
    inbox, store = _Inbox(), _Store()
    m = Mcu()
    m.setup(SimpleNamespace(get=lambda k, d=None: {"port": os.ttyname(slave), "baud": 115200}.get(k, d)))
    m._params = store
    try:
        m.start(inbox)
        assert m._thread is not None                    # the STDLIB path opened it (pyserial not needed)
        os.write(master, b"boot ok\nT\nD 0.4\njunk\n")
        deadline = time.time() + 3.0
        while time.time() < deadline and not (inbox.posted and store.sets):
            time.sleep(0.02)
    finally:
        m.stop()
        os.close(master)
        os.close(slave)
    assert inbox.posted == [("trigger", ())]            # T → the manual scan
    assert store.sets == [("efd", "detail", 0.4)]       # D → [efd] detail; the junk lines are ignored


def test_posix_serial_buffer_is_bounded(monkeypatch):
    # a newline-free stream (wrong baud / binary noise) must not grow the line buffer without bound —
    # an 8-hour installation would otherwise leak memory at the serial data rate.
    ser = _PosixSerial.__new__(_PosixSerial)
    ser._fd, ser._buf = 99, b""
    chunks = [b"x" * 256] * 100 + [b""]                      # 25.6 kB of garbage, then a timeout tick
    monkeypatch.setattr(os, "read", lambda fd, n: chunks.pop(0))
    assert ser.readline() == b""                             # timeout with no line
    assert len(ser._buf) <= 8192                             # capped, not 25.6 kB


def test_reader_survives_a_dead_port_and_reopens():
    # one serial error must NOT kill the reader for good — a replugged pedal resumes without a restart.
    mcu = Mcu()

    class _DeadOnce:
        def readline(self):
            raise OSError("unplugged")
        def close(self):
            pass

    reopened = SimpleNamespace(lines=[b"T\n", b""], closed=False)

    class _Back:
        def readline(self):
            if reopened.lines:
                line = reopened.lines.pop(0)
                if not line:
                    mcu._stopping.set()                      # end the loop after the replayed line
                return line
            mcu._stopping.set()
            return b""
        def close(self):
            pass

    mcu._ser = _DeadOnce()
    mcu._stopping.clear()
    mcu._open = lambda quiet=False: _Back()                  # the retry loop finds the port again
    mcu._stopping.wait = lambda t: False                     # don't actually sleep 5 s in the test
    fed = []
    mcu.feed = lambda line: fed.append(line)
    mcu._run()
    assert fed == ["T\n"]                                    # the line after the reopen was processed
