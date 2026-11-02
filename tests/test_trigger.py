"""``/rmt/trigger`` — the manual scan (a pedal / OSC client): capture every SETTLING object on the next
tick WITHOUT waiting out its ``settle_seconds`` dwell, and confirm it so the outputs emit at once.

The operator asserts "the arrangement is set — read it" instead of letting each object's timer decide.
It bypasses the DWELL only, never the stability test: an object still genuinely in motion is not
captured (that would bake a smeared/occluded footprint)."""

from types import SimpleNamespace

from lib.classify import Classify
from lib.control import ControlRouter, ScanTrigger
from lib.scene import Detection, Scene, Status
from plugins.control.oscin.oscin import translate


def _cfg(**kw):
    return SimpleNamespace(get=lambda k, d=None: kw.get(k, d))


def _obs(*, xor_prev=0.0, area=100.0, residual=0.0, missing=0.0, under_motion=False):
    return SimpleNamespace(xor_prev=xor_prev, area=area, residual=residual,
                           missing=missing, under_motion=under_motion, mask=None)


def _settling_object(scene, *, ts=0.0, obs=None):
    """Mint an object and leave it SETTLING with an observation, then open a FRESH tick — so it is no
    longer minted-this-tick and classify (not observe) owns the fields it writes."""
    scene.begin_tick(0, ts=ts)
    with scene.writing("observe"):
        oid = scene.mint(Detection((50.0, 50.0), (45, 45, 10, 10), 100.0, None))
    scene.begin_tick(1, ts=ts)                       # a new tick → no longer minted_this_tick
    with scene.writing("observe"):
        scene.set(oid, "observation", obs if obs is not None else _obs())
        scene.set_status(oid, Status.SETTLING)
    scene.begin_tick(2, ts=ts)                       # fresh tick → classify may own status/settle_since
    return oid


# --------------------------------------------------------------------------- #
# the ScanTrigger primitive
# --------------------------------------------------------------------------- #

def test_scan_trigger_is_a_one_shot_flag():
    t = ScanTrigger()
    assert t.take() is False              # nothing requested yet
    t.fire()
    assert t.take() is True               # consumed once…
    assert t.take() is False              # …and only once
    t.fire()
    t.fire()
    assert t.take() is True               # two stomps before a tick = ONE scan
    assert t.take() is False


# --------------------------------------------------------------------------- #
# the wire: OSC address → command → router → trigger
# --------------------------------------------------------------------------- #

def test_oscin_translates_the_trigger_address():
    assert translate("/rmt/trigger", []) == ("trigger", ())
    assert translate("/rmt/trigger", [], prefix="/rmt") == ("trigger", ())
    assert translate("/other/trigger", []) is None          # wrong prefix → ignored


def test_router_trigger_fires_the_scan():
    t = ScanTrigger()
    ControlRouter(scan_trigger=t)._trigger(())
    assert t.take() is True


def test_router_trigger_without_a_scan_trigger_is_a_no_op():
    ControlRouter()._trigger(())                            # warns, never raises


def test_trigger_is_a_registered_handler():
    assert "trigger" in ControlRouter._HANDLERS


# --------------------------------------------------------------------------- #
# the behaviour: bypass the dwell, but never the stability test
# --------------------------------------------------------------------------- #

def test_trigger_captures_a_settled_object_without_its_dwell(monkeypatch):
    s = Scene()
    t = ScanTrigger()
    cls = Classify(_cfg(settle_seconds=999.0, k=0.05), scan_trigger=t)   # auto-settle can never fire
    oid = _settling_object(s, obs=_obs(xor_prev=0.0))                    # stable + at rest
    got = []
    monkeypatch.setattr(cls, "_capture", lambda ctx, scene, obj, obs: got.append(obj.id))

    with s.writing("classify"):
        cls(SimpleNamespace(ts=1.0), s)
    assert got == []                       # dwell 999s not elapsed → no capture

    t.fire()
    with s.writing("classify"):
        cls(SimpleNamespace(ts=1.1), s)
    assert got == [oid]                    # manual scan → captured NOW, dwell skipped


def test_trigger_does_not_capture_an_object_still_in_motion(monkeypatch):
    s = Scene()
    t = ScanTrigger()
    cls = Classify(_cfg(settle_seconds=999.0, k=0.05), scan_trigger=t)
    # xor_prev way over k*area → NOT stable (still moving)
    oid = _settling_object(s, obs=_obs(xor_prev=1000.0))
    got = []
    monkeypatch.setattr(cls, "_capture", lambda ctx, scene, obj, obs: got.append(obj.id))

    t.fire()
    with s.writing("classify"):
        cls(SimpleNamespace(ts=1.0), s)
    assert got == []                       # a moving object is never captured, trigger or not
    assert s.get(oid).status is Status.MOVED


def test_trigger_is_consumed_so_the_next_tick_is_normal(monkeypatch):
    s = Scene()
    t = ScanTrigger()
    cls = Classify(_cfg(settle_seconds=999.0, k=0.05), scan_trigger=t)
    _settling_object(s, obs=_obs(xor_prev=0.0))
    got = []
    monkeypatch.setattr(cls, "_capture", lambda ctx, scene, obj, obs: got.append(obj.id))

    t.fire()
    with s.writing("classify"):
        cls(SimpleNamespace(ts=1.0), s)
    with s.writing("classify"):
        cls(SimpleNamespace(ts=1.1), s)    # trigger already consumed → back to the dwell
    assert len(got) == 1


def test_no_scan_trigger_wired_keeps_the_normal_dwell(monkeypatch):
    s = Scene()
    cls = Classify(_cfg(settle_seconds=999.0, k=0.05))    # no scan_trigger at all
    _settling_object(s, obs=_obs(xor_prev=0.0))
    got = []
    monkeypatch.setattr(cls, "_capture", lambda ctx, scene, obj, obs: got.append(obj.id))
    with s.writing("classify"):
        cls(SimpleNamespace(ts=1.0), s)
    assert got == []                       # unchanged behaviour when the pedal isn't wired


# --------------------------------------------------------------------------- #
# a manual scan emits at once (bypasses the confirm grace)
# --------------------------------------------------------------------------- #

def test_trigger_confirms_immediately_so_outputs_emit():
    s = Scene()
    t = ScanTrigger()
    cls = Classify(_cfg(confirm_seconds=10.0), scan_trigger=t)      # long grace: would normally withhold
    s.begin_tick(0, ts=0.0)
    with s.writing("observe"):
        oid = s.mint(Detection((50.0, 50.0), (45, 45, 10, 10), 100.0, None))
    assert s.get(oid).confirmed is False

    with s.writing("classify"):
        cls._latch_confirmed(SimpleNamespace(ts=0.1), s, s.get(oid), False)
    assert s.get(oid).confirmed is False                 # age 0.1s < 10s grace → still withheld

    with s.writing("classify"):
        cls._latch_confirmed(SimpleNamespace(ts=0.1), s, s.get(oid), True)
    assert s.get(oid).confirmed is True                  # manual scan → confirmed at once
