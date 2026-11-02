"""lib.compose — the dpg-free view composer. Two single panels: ``input`` (full pre-crop frame) and
``debug`` (the cropped input MASKED to-zero by observe's fg). The MJPEG stream + webgui underlay render
through it; the webgui draws object/cluster/id overlays client-side. Pure cv2 on a SceneSnapshot."""

import numpy as np

from lib.compose import VIEWS, render_view
from lib.scene import SceneSnapshot


def _snap(frame, obj_fg=None, input_full=None):
    return SceneSnapshot(frame_idx=1, ts=0.0, objects=(), frame=frame, obj_fg=obj_fg, input_full=input_full)


def test_views_are_input_and_debug():
    assert VIEWS == ("input", "debug", "processed")


def test_every_view_renders_a_bgr_image():
    snap = _snap(np.full((60, 80, 3), 120, np.uint8))
    for v in VIEWS:
        img = render_view(snap, v)
        assert img is not None and img.ndim == 3 and img.shape[2] == 3


def test_input_uses_the_pre_crop_frame_when_present():
    full = np.full((60, 100, 3), 90, np.uint8)
    img = render_view(_snap(np.zeros((40, 40, 3), np.uint8), input_full=full), "input", with_label=False)
    assert img.shape[:2] == (60, 100) and int(img.mean()) == 90          # uses input_full, not frame


def test_input_falls_back_to_frame_without_input_full():
    img = render_view(_snap(np.full((40, 40, 3), 77, np.uint8)), "input", with_label=False)
    assert img.shape == (40, 40, 3) and int(img.mean()) == 77


def test_debug_masks_the_frame_to_observe_fg():
    # keep the ORIGINAL pixel where obj_fg > 0, else zero it (to-zero mask).
    frame = np.full((10, 10, 3), 200, np.uint8)
    fg = np.zeros((10, 10), np.uint8)
    fg[2:5, 2:5] = 255                                                    # a 3×3 fg patch
    dbg = render_view(_snap(frame, obj_fg=fg), "debug", with_label=False)
    assert (dbg[2:5, 2:5] == 200).all()                                  # fg region keeps the original value
    dbg[2:5, 2:5] = 0
    assert int(dbg.sum()) == 0                                           # everything outside fg is zeroed


def test_debug_without_fg_is_the_plain_frame():
    dbg = render_view(_snap(np.full((10, 10, 3), 50, np.uint8)), "debug", with_label=False)
    assert int(dbg.mean()) == 50


def test_with_label_burns_in_text():
    snap = _snap(np.zeros((60, 80, 3), np.uint8), obj_fg=np.full((60, 80), 255, np.uint8))
    assert int(render_view(snap, "debug", with_label=True).sum()) > \
        int(render_view(snap, "debug", with_label=False).sum())          # the caption is ink


def test_none_when_no_frame():
    assert render_view(None) is None


def test_processed_view_is_the_bare_frame_without_a_mask():
    # flip/crop/lens/flatfield are already baked into snap.frame — with no mask circle the
    # processed view IS that frame (so a recording replays with --no-preprocess).
    import numpy as np
    snap = _snap(np.random.default_rng(3).integers(0, 255, (40, 60, 3), np.uint8))
    img = render_view(snap, "processed", with_label=False)
    assert np.array_equal(img, np.asarray(snap.frame))


def test_processed_view_bakes_the_mask_circle():
    import numpy as np
    from lib.compose import render_view
    snap = _snap(np.full((40, 60, 3), 200, np.uint8))
    h, w = np.asarray(snap.frame).shape[:2]
    img = render_view(snap, "processed", with_label=False, mask_circle=(w / 2, h / 2, 5))
    assert img[0, 0].max() == 0                            # a corner is outside the circle → black
    assert np.array_equal(img[h // 2, w // 2], np.asarray(snap.frame)[h // 2, w // 2])


def test_mask_circle_px_offsets_by_the_crop():
    from types import SimpleNamespace
    from lib.compose import mask_circle_px
    pc = SimpleNamespace(mask_enabled=True, mask_radius=100.0, mask_cx=500.0, mask_cy=300.0,
                         crop_enabled=True, crop_top_x=480, crop_top_y=16)
    assert mask_circle_px(pc) == (20.0, 284.0, 100.0)      # raw-px centre shifted into cropped coords
    pc.mask_enabled = False
    assert mask_circle_px(pc) is None
