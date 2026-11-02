"""observe._appearance_ok — the reid VETO on a geometric/proximity claim: reject a blob whose
appearance cosine to the object's fingerprint is below reloc_cosine_reject (a stone can't steal a
gear's id on a place-swap). Off when the veto is 0, reid is absent, or the object has no prints.
Uses a stub embedder returning controlled unit vectors so the cosine threshold is deterministic."""
from types import SimpleNamespace

import numpy as np

from lib.observe import Observe


def _unit(a, b):
    v = np.array([a, b], np.float32)
    return v / np.linalg.norm(v)


class _StubEmb:
    """Returns a fixed embedding for any blob (canonical stays True — reid contract)."""
    canonical = True

    def __init__(self, vec):
        self._vec = vec

    def embed(self, patch, mask):
        return self._vec


REF = _unit(1.0, 0.0)                       # the object's fingerprint
SIMILAR = _unit(0.9, np.sqrt(1 - 0.81))     # cosine 0.9 vs REF
DIFFERENT = _unit(0.3, np.sqrt(1 - 0.09))   # cosine 0.3 vs REF


def _obs(reject, blob_vec):
    from lib.reid import ReidParams
    reidp = ReidParams(SimpleNamespace(get=lambda k, d=None: {"reloc_cosine_reject": reject}.get(k, d)))
    return Observe(None, embedder=_StubEmb(blob_vec), reid_params=reidp)


def _ctx():
    return SimpleNamespace(bgr=np.zeros((20, 20, 3), np.uint8))


def _blob():
    return {"bbox": (2, 2, 10, 10), "mask": np.ones((20, 20), np.uint8) * 255}


def test_veto_rejects_dissimilar_allows_similar():
    obj = SimpleNamespace(embeds=(REF,))
    assert _obs(0.45, SIMILAR)._appearance_ok(_ctx(), obj, _blob()) is True     # cosine 0.9 ≥ 0.45 → keep
    assert _obs(0.45, DIFFERENT)._appearance_ok(_ctx(), obj, _blob()) is False  # cosine 0.3 < 0.45 → veto


def test_veto_off_or_no_prints_allows_all():
    obj = SimpleNamespace(embeds=(REF,))
    assert _obs(0.0, DIFFERENT)._appearance_ok(_ctx(), obj, _blob()) is True    # veto off
    obj_no = SimpleNamespace(embeds=None)
    assert _obs(0.45, DIFFERENT)._appearance_ok(_ctx(), obj_no, _blob()) is True  # no prints yet
