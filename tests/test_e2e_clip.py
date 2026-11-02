"""End-to-end gate: build() the REAL core, run a checked-in clip, assert ID stability vs its hand-fixed
MOT ground truth. The audit flagged that build() is never constructed in tests and the clips + .MOT
files go unused by pytest — so a whole-pipeline integration break (a stage miswired, a snapshot field
dropped, a settle regression) could stay green. This automates the manual GT bench for one clip.

Tracking-critical params are pinned via overrides so the result does NOT depend on the operator's live
readymade.conf (which they tune — e.g. fg_threshold) — the clips are already 1024² preprocessed, so
preprocess/crop are off and the committed thresholds are forced.
"""
import sys
from pathlib import Path

import readymade_tracker as rt
from lib.logger import NULL_LOGGER

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from tools.mot_analyze import churn, load_mot16, reuse   # noqa: E402


def _run_clip(clip: Path, pred: Path) -> None:
    """Build the core on ``clip`` with pinned tracking params + all I/O off, run to EOF, export MOT."""
    ov = {
        "input": {"source": "video", "location": str(clip), "fps": "1000"},
        "preprocess": {"crop_enabled": "false"},           # the test clips are 1024² preprocessed already
        "observe": {"fg_threshold": "60", "min_area": "200", "k": "0.05"},
        "classify": {"settle_seconds": "0.5", "confirm_seconds": "0.7"},
        "webgui": {"enabled": "false"}, "mjpeg": {"enabled": "false"}, "tuio2": {"enabled": "false"},
        "oscout": {"enabled": "false"}, "oscin": {"enabled": "false"}, "rec": {"enabled": "false"},
        "mcu": {"enabled": "false"},
    }
    core = rt.build(rt._load(rt.DEFAULT_CONF, ov), logger=NULL_LOGGER, mot_export=str(pred))
    try:
        while True:
            try:
                fc = core.runner.produce()
            except StopIteration:
                break
            if fc is not None:
                core.pipeline.run(fc)
    finally:
        core.mot.close()
        core.backend.close()


def test_hands2_end_to_end_is_one_to_one_stable(tmp_path):
    """hands2 (8 objects) tracked end-to-end must land 1:1 on the ground truth: every GT object mapped to
    exactly ONE tracker id, zero switches, and no tracker id shared across GT objects."""
    clip = REPO / "tests" / "videos" / "hands2.mp4"
    gt = REPO / "tests" / "videos" / "hands2.mp4.MOT"
    if not clip.is_file() or not gt.is_file():
        import pytest
        pytest.skip("clip / GT not present")
    pred = tmp_path / "hands2.mot"
    _run_clip(clip, pred)

    cm = churn(load_mot16(str(gt)), load_mot16(str(pred)))
    assert len(cm) == 8                                    # all 8 GT objects present in the analysis
    for gid, (_first, _last, seen, switches) in cm.items():
        assert len(seen) == 1 and switches == 0, f"GT#{gid} churned: tracker ids {seen}, {switches} switches"
    assert reuse(cm) == {}                                 # no tracker id spans two GT objects
