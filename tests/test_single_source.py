"""The two single-sourced observe values reach their second consumers LIVE.

[observe].k is THE threshold scale (classify shares it) and [observe].relocate_dist is THE set-down
re-link reach (identity's gone-pool re-acquire shares it). Both used to be COLD copies injected at
build — a /rmt/set observe … updated observe only, silently splitting the "one knob" mid-run (proved
by the 2026-07-20 audit). Now classify/identity hold the Observe instance and read live.
"""
from lib.classify import Classify
from lib.identity import Identity
from lib.observe import Observe
from lib.params import ParamStore, build_registry


def _store(observe):
    reg = build_registry({}, components={"observe": Observe})
    return ParamStore(reg, {"observe": observe}, {})


def test_classify_reads_k_live_from_observe():
    obs = Observe(None)
    cls = Classify(None, observe=obs)
    assert cls._k == obs._k                                 # same value at rest
    res = _store(obs).set("observe", "k", 0.12)             # the REAL live-set path
    assert res["ok"] and obs._k == 0.12
    assert cls._k == 0.12                                   # classify sees it the same tick


def test_identity_reads_reach_live_from_observe():
    obs = Observe(None)
    idn = Identity(None, observe=obs)
    assert idn._reacq_dist == obs._reloc_dist == 160.0
    res = _store(obs).set("observe", "relocate_dist", 300.0)
    assert res["ok"] and obs._reloc_dist == 300.0
    assert idn._reacq_dist == 300.0                         # the ONE reach holds under live tuning


def test_bare_constructions_fall_back_to_cold_values():
    """Tests / standalone constructions (no observe attached) keep working off config/default."""
    assert Classify({"k": 0.07})._k == 0.07
    assert Identity(None)._reacq_dist == 160.0
