"""lib.reid — the classical (neural-free) appearance embedder + the load_embedder on/off switch.
No model files, no onnxruntime: the classical descriptor is the only backend and works standalone."""
import cv2
import numpy as np

from lib.reid import ClassicalEmbedder, best_cosine, load_embedder


def _shape(color, kind="ellipse", rot=0):
    """A filled colored shape on black → (bgr, mask), for a synthetic 'object' crop."""
    img = np.zeros((90, 90, 3), np.uint8)
    if kind == "ellipse":
        cv2.ellipse(img, (45, 45), (32, 18), rot, 0, 360, color, -1)
    else:                                              # rotated rectangle
        box = cv2.boxPoints(((45, 45), (54, 26), rot)).astype(np.int32)
        cv2.fillPoly(img, [box], color)
    mask = ((img.max(2) > 8).astype(np.uint8)) * 255
    return img, mask


def test_classical_embed_shape_norm_and_degenerate():
    emb = ClassicalEmbedder()
    v = emb.embed(*_shape((0, 0, 200)))
    assert v is not None and v.shape == (21,)
    assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-4          # L2-normalized → cosine == dot
    assert emb.embed(np.zeros((40, 40, 3), np.uint8), np.zeros((40, 40), np.uint8)) is None   # empty → None
    assert emb.embed(None, None) is None


def test_classical_separates_and_is_rotation_invariant():
    emb = ClassicalEmbedder()
    a0 = emb.embed(*_shape((0, 0, 200), "ellipse", 0))        # red ellipse
    a1 = emb.embed(*_shape((0, 0, 200), "ellipse", 55))       # SAME object, rotated → canonicalized
    b = emb.embed(*_shape((200, 120, 0), "rect", 0))          # blue rectangle (different color + shape)
    assert best_cosine(a0, a1) > 0.9                           # canonical rotation → same object stays same
    assert best_cosine(a0, a1) > best_cosine(a0, b)            # same-object > different-object


def _cfg(d):
    class C:
        def get(self, k, default=None):
            return d.get(k, default)
    return C()


def test_load_embedder_enable_switch():
    assert load_embedder(_cfg({"enabled": False})) is None                       # off → no embedder
    assert type(load_embedder(_cfg({"enabled": True}))).__name__ == "ClassicalEmbedder"
    assert type(load_embedder(_cfg({}))).__name__ == "ClassicalEmbedder"          # default enabled = on
    assert load_embedder(None) is not None                                       # None config → default on
