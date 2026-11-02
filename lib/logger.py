"""Shared logging service.

File-based with a daemon writer thread: producers format a timestamped line and pass it to a
``queue.Queue``; one writer thread owns the file handle, so the pipeline thread never touches disk
(keeps it out of the ``fflush`` hot path). The interface is the standard severity levels the spine
calls — ``debug`` / ``info`` / ``warning`` / ``error`` — handed to each plugin read-only via
``ctx.logger``, auto-prefixed by ``_PrefixLogger``.

  * **async file write** — the writer thread owns the file handle; the pipeline thread never
    touches disk.
  * **level-gated** — a message is emitted iff its level >= the configured threshold.
  * **inert when off** — no file path and nothing to mirror => every call is a no-op (no thread,
    no file).
  * **stderr mirror** — messages at >= ``stderr_level`` (default ``warning``) also go to stderr
    synchronously, so real problems surface without trawling the log file.
  * **no module global** — constructed once by the orchestrator and threaded through ``ctx``.
  * **close()** — drain + stop the writer + close the file; idempotent.
"""

from __future__ import annotations

import queue
import sys
import threading
import time
from typing import Any

# Severity levels (the standard ladder; values mirror stdlib logging for sanity).
DEBUG, INFO, WARNING, ERROR = 10, 20, 30, 40
_NAMES = {"debug": DEBUG, "info": INFO, "warning": WARNING, "error": ERROR}


class NullLogger:
    """A no-op logger — the inert default when none is supplied. Matches the
    ``Logger`` level interface so any module can fall back to it. Stateless, so
    use the shared ``NULL_LOGGER`` singleton rather than allocating one."""

    def debug(self, *a: Any, **k: Any) -> None: ...
    def info(self, *a: Any, **k: Any) -> None: ...
    def warning(self, *a: Any, **k: Any) -> None: ...
    def error(self, *a: Any, **k: Any) -> None: ...


NULL_LOGGER = NullLogger()


def _as_level(x: Any) -> int:
    """Coerce a level given as an int or a name (``"info"``) to its int value."""
    if isinstance(x, bool):  # guard: bool is an int subclass
        raise ValueError(f"log level must be an int or name, not bool {x!r}")
    if isinstance(x, int):
        return x
    try:
        return _NAMES[str(x).strip().lower()]
    except KeyError:
        raise ValueError(
            f"unknown log level {x!r} (use one of {sorted(_NAMES)} or an int)"
        ) from None


class Logger:
    """File + stderr logger with severity levels and an async file writer.

    Producers (``debug`` / ``info`` / ``warning`` / ``error``) can be called from
    any thread — ``queue.Queue.put`` is thread-safe and the stderr write is a
    single ``write`` call. The lone daemon writer thread owns the file handle.
    """

    _SENTINEL = None  # tells the writer thread to drain + exit

    def __init__(
        self,
        level: int | str = "info",
        path: str | None = None,
        *,
        stderr_level: int | str | None = "warning",
    ) -> None:
        self.level = _as_level(level)
        self.stderr_level = None if stderr_level is None else _as_level(stderr_level)
        self.path = path
        self._f = None
        self._q: queue.Queue = queue.Queue()
        self._writer: threading.Thread | None = None
        if path is not None:
            try:
                # line-buffered: each completed line is flushed by the OS without
                # an explicit fflush; survives an abrupt exit better than block-buffered.
                # encoding=utf-8: log lines can carry non-ASCII (→, box chars) and Windows'
                # default cp1252 would raise UnicodeEncodeError mid-write (silently, on Windows).
                self._f = open(path, "w", buffering=1, encoding="utf-8")
                self._writer = threading.Thread(
                    target=self._writer_loop, daemon=True, name="LoggerWriter"
                )
                self._writer.start()
            except OSError as e:
                print(f"[log] failed to open {path}: {e}", file=sys.stderr)
                self._f = None

    # -- the level methods (what the spine + _PrefixLogger call) ------------- #

    def debug(self, msg: object, *a: Any, **k: Any) -> None:
        self._emit(DEBUG, msg)

    def info(self, msg: object, *a: Any, **k: Any) -> None:
        self._emit(INFO, msg)

    def warning(self, msg: object, *a: Any, **k: Any) -> None:
        self._emit(WARNING, msg)

    def error(self, msg: object, *a: Any, **k: Any) -> None:
        self._emit(ERROR, msg)

    # -- internals ----------------------------------------------------------- #

    def _emit(self, lvl: int, msg: object) -> None:
        if lvl < self.level:
            return
        line = f'{time.strftime("%H:%M:%S")} {msg}\n'
        if self._f is not None:
            self._q.put(line)  # async — the writer thread does the disk write
        if self.stderr_level is not None and lvl >= self.stderr_level:
            sys.stderr.write(line)

    def _writer_loop(self) -> None:
        """Drain queued lines to the file until the sentinel arrives. Swallows
        write errors so one bad line can't take down the run."""
        while True:
            item = self._q.get()
            if item is self._SENTINEL:
                break
            try:
                self._f.write(item)
                if self._q.empty():               # force the flush — buffering=1 line-buffering
                    self._f.flush()               #   isn't reliable on Windows (empty log files)
            except (ValueError, AttributeError):  # closed / None mid-flight
                pass

    def close(self) -> None:
        """Drain the queue, stop the writer, close the file. Idempotent / a no-op
        when inert. Blocks up to 5 s for the writer to flush."""
        if self._f is None:
            return
        self._q.put(self._SENTINEL)
        if self._writer is not None:
            self._writer.join(timeout=5.0)
        if not self._f.closed:
            self._f.close()
        self._f = None
