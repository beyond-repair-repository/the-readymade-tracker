"""[preprocess] smooth_alpha — the temporal EMA noise low-pass on gray.

Three contracts: (1) it actually suppresses per-pixel sensor noise on a static
scene, (2) off-values (0 and 1) are byte-identical passthrough, (3) a frame-size
change reseeds the accumulator instead of crashing or blending across sizes.
"""

import numpy as np

from lib.preprocess import Preprocess, PreprocessConfig


def _gray3(gray: np.ndarray) -> np.ndarray:
    """A BGR frame whose cv2 gray equals ``gray`` exactly (equal channels)."""
    return np.repeat(gray[:, :, None], 3, axis=2)


def _pp(alpha: float) -> Preprocess:
    return Preprocess(PreprocessConfig(smooth_alpha=alpha))


def test_static_scene_noise_is_reduced():
    rng = np.random.default_rng(7)
    base = np.tile(np.linspace(40, 200, 64).astype(np.uint8), (48, 1))
    pp = _pp(0.08)
    raw_resid = ema_resid = 0.0
    tail = 0
    for i in range(80):
        noise = rng.integers(-25, 26, base.shape, dtype=np.int16)
        noisy = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        _, gray = pp.process(_gray3(noisy))
        if i >= 60:                      # steady state only
            raw_resid += float(np.mean(np.abs(noisy.astype(np.float32) - base)))
            ema_resid += float(np.mean(np.abs(gray.astype(np.float32) - base)))
            tail += 1
    assert tail > 0
    # the EMA output must sit far closer to the true scene than the raw frames do
    assert ema_resid < raw_resid * 0.6, (ema_resid / tail, raw_resid / tail)


def test_alpha_zero_and_one_are_byte_identical_passthrough():
    rng = np.random.default_rng(11)
    frames = [rng.integers(0, 256, (32, 40), dtype=np.uint8) for _ in range(5)]
    for alpha in (0.0, 1.0):
        pp = _pp(alpha)
        for f in frames:
            _, gray = pp.process(_gray3(f))
            assert np.array_equal(gray, f), f"alpha={alpha} altered the frame"
    # and the config key round-trips through from_mapping
    assert PreprocessConfig.from_mapping({"smooth_alpha": 0.25}).smooth_alpha == 0.25
    assert PreprocessConfig.from_mapping({}).smooth_alpha == 0.0


def test_frame_size_change_reseeds_the_accumulator():
    pp = _pp(0.2)
    small = np.full((32, 32), 100, np.uint8)
    big = np.full((64, 48), 200, np.uint8)
    pp.process(_gray3(small))
    pp.process(_gray3(small))            # accumulator live at 32x32
    _, gray = pp.process(_gray3(big))    # size change -> reseed, first frame passes through
    assert gray.shape == big.shape
    assert np.array_equal(gray, big)
    # and the reseeded accumulator keeps smoothing at the new size
    darker = np.full((64, 48), 100, np.uint8)
    _, gray2 = pp.process(_gray3(darker))
    assert 100 < int(gray2.mean()) < 200   # blended, not passthrough
