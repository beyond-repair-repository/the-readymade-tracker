"""classify latches `confirmed` once an object has persisted [classify].confirm_seconds after mint —
the ONE grace every output honours (render composite, OSC/TUIO2, the output-view overlay), so a fresh
hover blip that mints then vanishes never leaks out. The debug overlay shows every object regardless."""

from types import SimpleNamespace

from lib.classify import Classify
from lib.scene import Detection, Scene


def _cfg(**kw):
    return SimpleNamespace(get=lambda k, d=None: kw.get(k, d))


def _mint(s, ts):
    s.begin_tick(0, ts=ts)
    with s.writing("observe"):
        oid = s.mint(Detection((50.0, 50.0), (45, 45, 10, 10), 100.0, None))
    return oid


def test_confirmed_latches_after_confirm_seconds():
    s = Scene()
    cls = Classify(_cfg(confirm_seconds=1.0))
    oid = _mint(s, ts=0.0)
    assert s.get(oid).born_ts == 0.0 and s.get(oid).confirmed is False   # fresh mint starts unconfirmed
    with s.writing("classify"):
        cls._latch_confirmed(SimpleNamespace(ts=0.5), s, s.get(oid))
    assert s.get(oid).confirmed is False                                  # age 0.5s < 1.0s grace
    with s.writing("classify"):
        cls._latch_confirmed(SimpleNamespace(ts=1.5), s, s.get(oid))
    assert s.get(oid).confirmed is True                                   # age 1.5s ≥ grace → latched


def test_confirm_seconds_zero_is_immediate():
    s = Scene()
    cls = Classify(_cfg(confirm_seconds=0.0))
    oid = _mint(s, ts=5.0)
    with s.writing("classify"):
        cls._latch_confirmed(SimpleNamespace(ts=5.0), s, s.get(oid))
    assert s.get(oid).confirmed is True                                   # grace 0 → confirmed at once


def test_confirmed_is_projected_to_the_snapshot():
    s = Scene()
    cls = Classify(_cfg(confirm_seconds=1.0))
    oid = _mint(s, ts=0.0)
    with s.writing("classify"):
        cls._latch_confirmed(SimpleNamespace(ts=2.0), s, s.get(oid))
    snap = {o.id: o for o in s.snapshot().objects}[oid]
    assert snap.confirmed is True                                         # ObjectSnap carries it to outputs
