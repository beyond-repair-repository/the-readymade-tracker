"""ControlRouter — the ``CommandInbox`` consumer.

Producers (the OSCIn plugin) ``post`` ``(name, args)`` commands to the shared ``CommandInbox``; this
router ``drain``s it and applies each command to the tracker's EXISTING control surfaces — the input
``Transport`` (pause / step), the view-aware outputs (the MJPEG stream), and the output threads
(start / stop). It invents no parallel state; it is the single place a control command becomes an action.

**Where it runs — the MAIN-THREAD supervisor loop, NOT the pipeline thread.** A paused pipeline
starves ``PipelineLoop.step`` (no frame → early return before any drain), so a consumer there could
never receive ``resume``. The supervisor (`serve`'s `_serve_headless`) polls on the main thread
regardless of frame flow, so control stays responsive while paused.

Unknown commands / missing surfaces (e.g. ``view`` with no MJPEG) are logged and skipped, never raised —
a stray OSC packet must not take down the tracker.
"""

from __future__ import annotations

import threading
from typing import Any

from lib.logger import NULL_LOGGER

_TRUTHY = frozenset({"on", "true", "1", "yes", "enable", "enabled"})


class ScanTrigger:
    """A thread-safe one-shot 'read the table NOW' flag — the manual-scan pedal.

    ``/rmt/trigger`` FIRES it (on the supervisor thread, via the router); the ``classify`` stage TAKES it
    on the PIPELINE thread and, for that one tick, captures every settling object immediately instead of
    waiting out its ``settle_seconds`` dwell, and confirms it so the outputs emit at once. The operator
    asserts "the arrangement is set — read it", rather than letting each object's timer decide.

    It only bypasses the DWELL, never the stability test: an object still genuinely in motion is not
    captured (that would bake a smeared/occluded footprint). Fire-and-forget — firing twice before a tick
    consumes it is one scan, which is what a stomped pedal should mean."""

    def __init__(self) -> None:
        self._flag = threading.Event()

    def fire(self) -> None:
        """Request a scan on the next pipeline tick (idempotent until taken)."""
        self._flag.set()

    def take(self) -> bool:
        """Test-and-clear: True at most once per ``fire()``. Called by the single consumer (classify)."""
        if self._flag.is_set():
            self._flag.clear()
            return True
        return False


class ControlRouter:
    """Drain the command inbox and apply each ``(name, args)`` to the live control surfaces.

    Held surfaces are all optional (a bare source may have no ``transport``) — every handler guards its
    own surface and no-ops when absent."""

    def __init__(self, *, transport: Any = None,
                 outputs: dict[str, Any] | None = None, bus: Any = None,
                 preprocess: Any = None, scan_trigger: Any = None, logger: Any = None) -> None:
        self._transport = transport
        self._outputs = dict(outputs or {})           # section name → output plugin
        self._bus = bus
        self._preprocess = preprocess                 # for the /rmt/flatfield capture|delete verb
        self._scan = scan_trigger                     # ScanTrigger — the /rmt/trigger manual scan
        self._log = logger or NULL_LOGGER
        self._active = set(self._outputs)             # every enabled output starts running
        self._lifecycle: str | None = None            # None | "restart" | "quit" — the supervisor acts on it

    @property
    def lifecycle(self) -> str | None:
        """A pending whole-process action the SUPERVISOR (serve) must carry out after teardown:
        ``"restart"`` (re-exec, re-reading the saved conf) or ``"quit"`` (exit). Set by /rmt/restart|quit."""
        return self._lifecycle

    # -- entry points ------------------------------------------------------- #

    def drain(self, inbox: Any) -> None:
        """Apply every command currently queued on ``inbox`` (called once per supervisor iteration)."""
        self.dispatch(inbox.drain())

    def dispatch(self, commands: Any) -> None:
        for name, args in commands:
            handler = self._HANDLERS.get(name)
            if handler is None:
                self._log.warning(f"[control] unknown command: {name}")
                continue
            try:
                handler(self, tuple(args))
            except Exception as exc:                  # noqa: BLE001 — one bad command can't kill the loop
                self._log.warning(f"[control] {name}{tuple(args)} failed: {exc}")

    # -- handlers ----------------------------------------------------------- #

    def _pause(self, args: tuple) -> None:
        if self._transport is not None:
            self._transport.pause()

    def _resume(self, args: tuple) -> None:
        if self._transport is not None:
            self._transport.resume()

    def _toggle_pause(self, args: tuple) -> None:
        if self._transport is not None:
            self._transport.toggle_pause()

    def _step(self, args: tuple) -> None:
        """``step [n]`` — advance ``n`` frames (default +1; n<0 steps back). The ``Transport`` step
        counter accumulates, so N ±1 calls = an N-frame step on the next ``take``."""
        if self._transport is None:
            return
        n = int(args[0]) if args else 1
        if n == 0:
            return
        step = self._transport.step_forward if n > 0 else self._transport.step_backward
        for _ in range(abs(n)):
            step()

    def _seek(self, args: tuple) -> None:
        """``seek [n]`` — jump to frame ``n`` (default 0 = the first frame / reset to the beginning).
        Seekable sources only; a live camera logs it as a no-op."""
        if self._transport is not None:
            self._transport.seek(int(args[0]) if args else 0)

    def _view(self, args: tuple) -> None:
        """``view <input|debug>`` — a single GLOBAL view every view-aware surface follows: any output
        with a ``set_view`` (the MJPEG stream). Each surface validates the name."""
        if not args:
            return
        v = str(args[0]).lower()
        applied = False
        for out in self._outputs.values():            # view-aware outputs (mjpeg) follow the same view
            if hasattr(out, "set_view"):
                out.set_view(v)
                applied = True
        if not applied:
            self._log.info(f"[control] view {v}: no view-aware surface (no mjpeg)")

    def _output(self, args: tuple) -> None:
        """``output <name> <on|off>`` — start/stop an output thread by its config-section name.
        Idempotent: toggling an already-on output on (or off off) is a no-op."""
        if len(args) < 2:
            return
        name = str(args[0]).lower()
        out = self._outputs.get(name)
        if out is None:
            self._log.warning(f"[control] no such output: {name} (have {sorted(self._outputs)})")
            return
        on = str(args[1]).lower() in _TRUTHY
        if on and name not in self._active:
            out.start(self._bus)
            self._active.add(name)
            self._log.info(f"[control] output {name} ON")
        elif not on and name in self._active:
            out.stop()
            self._active.discard(name)
            self._log.info(f"[control] output {name} OFF")

    def _rec(self, args: tuple) -> None:
        """``rec [start|stop|toggle]`` — rec/stop the video recorder ([rec] output). Distinct from
        ``output rec on|off`` (which starts/stops the whole recorder THREAD): this toggles CAPTURE."""
        rec = self._outputs.get("rec")
        if rec is None or not hasattr(rec, "set_recording"):
            self._log.warning("[control] no recorder ([rec] not enabled)")
            return
        verb = str(args[0]).lower() if args else "toggle"
        if verb == "toggle":
            rec.toggle()
        else:
            rec.set_recording(verb in _TRUTHY or verb == "start")

    def _flatfield(self, args: tuple) -> None:
        """``flatfield capture [n]`` — average the next n frames (default 30) into a new flatfield;
        ``flatfield delete`` — drop it. A webgui calibration button acting on the live Preprocess."""
        pp = self._preprocess
        if pp is None or not hasattr(pp, "capture_flatfield"):
            self._log.warning("[control] no preprocess for flatfield")
            return
        verb = str(args[0]).lower() if args else ""
        if verb == "capture":
            pp.capture_flatfield(int(args[1]) if len(args) > 1 else 30)
        elif verb == "delete":
            pp.clear_flatfield()

    def _trigger(self, args: tuple) -> None:
        """``trigger`` — the manual scan (a pedal / OSC client): capture every settling object on the next
        pipeline tick without waiting out its settle dwell, and confirm it so the outputs emit at once.
        No-op (warned) when no ScanTrigger is wired.

        When PAUSED, the source produces no frame, so the scan would never be consumed and the outputs
        would emit nothing. So a paused trigger ALSO asks the transport to re-emit the frozen frame once —
        one pipeline pass on it (with the scan set) → the settled reading goes out over the wire."""
        if self._scan is None:
            self._log.warning("[control] trigger: no scan trigger wired")
            return
        self._scan.fire()
        if self._transport is not None and getattr(self._transport, "paused", False):
            refresh = getattr(self._transport, "refresh", None)
            if callable(refresh):
                refresh()                              # frozen frame → one more pass → outputs emit
        self._log.info("[control] trigger: scan requested")

    def _restart(self, args: tuple) -> None:
        """``restart`` — re-exec the whole tracker after a clean teardown, re-reading readymade.conf so
        SAVEd needs-restart params take effect. The supervisor loop sees ``lifecycle`` and acts."""
        self._lifecycle = "restart"
        self._log.info("[control] restart requested")

    def _quit(self, args: tuple) -> None:
        """``quit`` — graceful shutdown + process exit (same teardown as restart, minus the re-exec)."""
        self._lifecycle = "quit"
        self._log.info("[control] quit requested")

    _HANDLERS = {
        "pause": _pause, "resume": _resume, "toggle_pause": _toggle_pause,
        "step": _step, "seek": _seek, "view": _view, "output": _output, "rec": _rec,
        "trigger": _trigger, "flatfield": _flatfield, "restart": _restart, "quit": _quit,
    }
