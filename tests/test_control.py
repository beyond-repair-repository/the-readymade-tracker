"""ControlRouter — the CommandInbox consumer. Drive it with fakes for the control surfaces
(transport / outputs) and assert each (name, args) hits the right existing mechanism.
No network, no pipeline — the router is pure dispatch."""

from lib.control import ControlRouter


class FakeTransport:
    def __init__(self):
        self.calls = []
    def pause(self):
        self.calls.append("pause")
    def resume(self):
        self.calls.append("resume")
    def toggle_pause(self):
        self.calls.append("toggle")
    def step_forward(self):
        self.calls.append("fwd")
    def step_backward(self):
        self.calls.append("back")


class FakeOutput:
    def __init__(self):
        self.events = []
    def start(self, bus):
        self.events.append(("start", bus))
    def stop(self):
        self.events.append("stop")


def _router(**kw):
    return ControlRouter(**kw)


# --------------------------------------------------------------------------- #
# transport: pause / resume / step
# --------------------------------------------------------------------------- #

def test_pause_resume_toggle():
    t = FakeTransport()
    r = _router(transport=t)
    r.dispatch([("pause", ()), ("resume", ()), ("toggle_pause", ())])
    assert t.calls == ["pause", "resume", "toggle"]


def test_restart_and_quit_set_lifecycle():
    r = _router()
    assert r.lifecycle is None
    r.dispatch([("restart", ())])
    assert r.lifecycle == "restart"                  # the supervisor re-execs after teardown
    r.dispatch([("quit", ())])
    assert r.lifecycle == "quit"


def test_step_defaults_to_one_forward():
    t = FakeTransport()
    _router(transport=t).dispatch([("step", ())])
    assert t.calls == ["fwd"]


def test_step_n_forward_and_back():
    t = FakeTransport()
    _router(transport=t).dispatch([("step", (3,)), ("step", (-2,))])
    assert t.calls == ["fwd", "fwd", "fwd", "back", "back"]


def test_step_zero_is_noop():
    t = FakeTransport()
    _router(transport=t).dispatch([("step", (0,))])
    assert t.calls == []


def test_transport_commands_noop_without_transport():
    _router().dispatch([("pause", ()), ("step", (2,))])          # must not raise


# --------------------------------------------------------------------------- #
# view
# --------------------------------------------------------------------------- #

class FakeViewOutput(FakeOutput):
    def __init__(self):
        super().__init__()
        self.views = []
    def set_view(self, v):
        self.views.append(v)


def test_view_without_a_view_output_is_noop():
    _router().dispatch([("view", ("debug",))])                   # no view-aware surface: ignored, no crash


def test_view_drives_view_aware_outputs():
    vo = FakeViewOutput()
    plain = FakeOutput()                                          # no set_view → untouched by view
    _router(outputs={"mjpeg": vo, "tuio2": plain}, bus=object()).dispatch([("view", ("debug",))])
    assert vo.views == ["debug"]                                 # the view-aware output (mjpeg) follows
    assert plain.events == []                                    # a non-view output is left alone


def test_view_switches_the_stream():
    vo = FakeViewOutput()
    _router(outputs={"mjpeg": vo}, bus=object()).dispatch([("view", ("input",))])
    assert vo.views == ["input"]


def test_view_without_arg_is_noop():
    vo = FakeViewOutput()
    _router(outputs={"mjpeg": vo}).dispatch([("view", ())])
    assert vo.views == []


# --------------------------------------------------------------------------- #
# output on/off (start/stop, idempotent)
# --------------------------------------------------------------------------- #

def test_output_off_then_on():
    o = FakeOutput()
    bus = object()
    r = _router(outputs={"tuio2": o}, bus=bus)                   # starts ACTIVE
    r.dispatch([("output", ("tuio2", "off"))])
    assert o.events == ["stop"]
    r.dispatch([("output", ("tuio2", "on"))])
    assert o.events == ["stop", ("start", bus)]


def test_output_toggle_is_idempotent():
    o = FakeOutput()
    r = _router(outputs={"tuio2": o}, bus=object())
    r.dispatch([("output", ("tuio2", "off")), ("output", ("tuio2", "off"))])   # 2nd off = no-op
    assert o.events == ["stop"]
    r.dispatch([("output", ("tuio2", "on")), ("output", ("tuio2", "on"))])     # 2nd on = no-op
    assert o.events.count("stop") == 1 and sum(e != "stop" for e in o.events) == 1


class FakeRecorder:
    def __init__(self):
        self.events = []
    def set_recording(self, on):
        self.events.append(("set", on))
    def toggle(self):
        self.events.append("toggle")


def test_rec_command_starts_stops_toggles():
    rec = FakeRecorder()
    r = _router(outputs={"rec": rec}, bus=object())
    r.dispatch([("rec", ("start",)), ("rec", ("stop",)), ("rec", ())])   # bare = toggle
    assert rec.events == [("set", True), ("set", False), "toggle"]


def test_rec_command_without_a_recorder_is_noop():
    _router(outputs={}, bus=object()).dispatch([("rec", ("start",))])    # no [rec] → no crash


def test_output_unknown_name_is_noop():
    o = FakeOutput()
    _router(outputs={"tuio2": o}, bus=object()).dispatch([("output", ("mjpeg", "off"))])
    assert o.events == []                                        # only the named output is touched


def test_output_truthy_variants():
    o = FakeOutput()
    r = _router(outputs={"x": o}, bus=object())
    r.dispatch([("output", ("x", "off"))])
    r.dispatch([("output", ("x", "1"))])                        # "1" reads as on
    assert o.events[-1][0] == "start"


# --------------------------------------------------------------------------- #
# robustness + the drain() entry point
# --------------------------------------------------------------------------- #

def test_unknown_command_is_skipped():
    t = FakeTransport()
    _router(transport=t).dispatch([("frobnicate", ()), ("pause", ())])
    assert t.calls == ["pause"]                                  # unknown skipped, rest applied


def test_a_failing_handler_does_not_stop_the_batch():
    class Boom(FakeTransport):
        def pause(self):
            raise RuntimeError("boom")
    t = Boom()
    _router(transport=t).dispatch([("pause", ()), ("resume", ())])   # pause raises, resume still runs
    assert t.calls == ["resume"]


def test_drain_pulls_from_the_inbox():
    from lib.loop import CommandInbox
    t = FakeTransport()
    inbox = CommandInbox()
    inbox.post("pause")
    inbox.post("step", (2,))
    r = _router(transport=t)
    r.drain(inbox)
    assert t.calls == ["pause", "fwd", "fwd"]
    r.drain(inbox)                                              # inbox now empty → no-op
    assert t.calls == ["pause", "fwd", "fwd"]


def test_serve_headless_applies_queued_control():
    # the wiring: the main-thread supervisor loop drains the router each poll (so control works
    # even when the pipeline is idle/paused). Fake just enough Core to run one iteration.
    from types import SimpleNamespace

    import readymade_tracker
    from lib.loop import CommandInbox

    t = FakeTransport()
    inbox = CommandInbox()
    inbox.post("pause")
    core = SimpleNamespace(loop=SimpleNamespace(stopped=False),
                           control=_router(transport=t), inbox=inbox,
                           slot=SimpleNamespace(pending=False))
    it = SimpleNamespace(done=True)                            # source exhausted → loop breaks after 1 drain
    readymade_tracker._serve_headless(core, it)
    assert t.calls == ["pause"]                                # the queued command was applied
