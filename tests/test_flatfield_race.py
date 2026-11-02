"""The flatfield capture/clear race: capture_flatfield()/clear_flatfield() run on the CONTROL thread
while process() accumulates on the INPUT thread. Clearing between process()'s check and the
accumulate call used to TypeError and kill the input thread (the run ended as a silent EOF)."""

import numpy as np

from lib.preprocess import Preprocess, PreprocessConfig


def _pp():
    return Preprocess(PreprocessConfig.from_mapping({"preprocess": "true"}))


def test_accumulate_survives_a_concurrent_clear():
    pp = _pp()
    pp.capture_flatfield(3)
    pp._ff_cap = None                                  # the control thread cleared it mid-window
    pp._accumulate_flatfield(np.zeros((8, 8, 3), np.uint8))   # must NOT raise


def test_capture_still_completes_normally(tmp_path):
    pp = _pp()
    pp.cfg.flatfield_path = str(tmp_path / "ff.png")   # keep the persist write out of the repo
    pp.capture_flatfield(2)
    f = np.full((8, 8, 3), 100, np.uint8)
    pp.process(f)
    pp.process(f)
    assert pp._ff_cap is None                          # capture finished
    assert pp.cfg.flatfield_enabled and pp.cfg.flatfield_arr is not None
    assert int(pp.cfg.flatfield_arr[0, 0, 0]) == 100
