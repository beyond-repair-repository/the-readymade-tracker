"""[reid] — the appearance-POLICY home. The cosine gates / twin guard / size gate / fingerprint
knobs are declared once on ReidParams and read LIVE by observe (husk relocation) and identity
(pool re-acquire). Pre-0.24 confs that kept them in [observe]/[identity] are lifted with a warning."""

from types import SimpleNamespace

from lib.identity import Identity
from lib.observe import Observe
from lib.reid import ReidParams


class _Sec:
    def __init__(self, d):
        self._d = d
    def get(self, k, d=None):
        return self._d.get(k, d)


class _Cfg:
    def __init__(self, m):
        self._m = m
    def get(self, s):
        return _Sec(self._m[s]) if s in self._m else None


class _Log:
    def __init__(self):
        self.warnings = []
    def warning(self, msg):
        self.warnings.append(msg)
    def info(self, msg):
        pass


def test_defaults_when_bare():
    p = ReidParams(None)
    assert p.cosine_min == 0.6
    assert p.uniqueness_margin == 0.05 and p.size_tol == 0.5
    assert p.fp_max_prints == 8 and p.fp_new_dist == 25.0




def test_one_instance_read_live_by_both_consumers():
    # observe and identity hold the SAME ReidParams — a live /rmt/set reid … (a plain attr write,
    # as the param store applies it) is visible to both on their next read.
    p = ReidParams(None)
    obs = Observe(None, reid_params=p)
    idn = Identity(None, reid_params=p)
    assert obs._reidp is p and idn._reidp is p
    p._cosine_min = 0.9
    p._reloc_cosine_reject = 0.33
    assert idn._reidp.cosine_min == 0.9
    assert obs._reidp.reloc_cosine_reject == 0.33
