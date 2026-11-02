"""Recorder (roadmap B.3) — rec/stop flag + the encode thread + a real file write. Uses fake buses +
a tmp dir; skips the file assertions if the mp4v codec isn't in this OpenCV build (headless CI)."""

import os
import re
import time
from types import SimpleNamespace

import numpy as np
import pytest

from plugins.output.rec.rec import Recorder


class _Cfg:
    def __init__(self, **kw):
        self._d = kw
    def get(self, k, d=None):
        return self._d.get(k, d)


def _snap(frame_idx=1, w=32, h=24):
    return SimpleNamespace(frame=np.zeros((h, w, 3), np.uint8), frame_idx=frame_idx,
                           composite=None, objects=())


class _JumpBus:
    """Hands a fresh frame whose idx advances by ``step`` — step>1 models the bus skipping ahead
    (the recorder should count the skipped frames as dropped)."""
    def __init__(self, snap, step=1):
        self._snap, self._i, self._step = snap, 0, step
    def wait_fresh(self, last_idx, timeout):
        self._i += self._step
        self._snap.frame_idx = self._i
        return self._snap
    def latest(self):
        return self._snap


def _wait(pred, timeout=2.0):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.005)
    return False


def _rec(tmp_path, **cfg):
    r = Recorder()
    r.setup(_Cfg(dir=str(tmp_path), fps=10, **cfg))
    return r


# --------------------------------------------------------------------------- #
# control flag + state (no thread, no codec)
# --------------------------------------------------------------------------- #

def test_toggle_and_set_recording_flag(tmp_path):
    r = _rec(tmp_path)
    assert r.is_recording() is False and r._want is False
    r.set_recording(True); assert r._want is True
    r.toggle(); assert r._want is False
    r.toggle(); assert r._want is True
    assert r.status()["recording"] is False        # still no writer until the thread opens it


def test_timestamp_filename(tmp_path):
    p = _rec(tmp_path)._timestamp_path()
    assert re.search(r"[/\\]\d{8}-\d{6}\.mp4$", p)


# --------------------------------------------------------------------------- #
# the encode path — a real file (skip if the codec is unavailable)
# --------------------------------------------------------------------------- #

def test_open_write_close_writes_a_file(tmp_path):
    r = _rec(tmp_path)
    r._open(_snap())
    if r._writer is None:
        pytest.skip("no mp4v VideoWriter in this OpenCV build")
    path = r._path
    for i in range(5):
        r._write(_snap(frame_idx=i))
    assert r.status()["written"] == 5 and r.is_recording()
    r._close()
    assert not r.is_recording() and os.path.exists(path) and os.path.getsize(path) > 0


def test_view_config_records_a_resized_frame(tmp_path):
    r = _rec(tmp_path, width=16, height=16)               # force output size
    r._open(_snap(w=48, h=36))
    if r._writer is None:
        pytest.skip("no mp4v VideoWriter in this OpenCV build")
    r._write(_snap(w=48, h=36))                            # bigger frame → resized to 16x16, no crash
    assert r.status()["written"] == 1
    r._close()


# --------------------------------------------------------------------------- #
# the thread — rec/stop + dropped-frame counting
# --------------------------------------------------------------------------- #

def test_thread_records_then_stops_and_counts_dropped(tmp_path):
    r = _rec(tmp_path)
    r.start(_JumpBus(_snap(), step=3))                    # idx jumps by 3 → 2 dropped per written frame
    try:
        r.set_recording(True)
        if not _wait(lambda: r.is_recording(), 1.0):
            pytest.skip("no mp4v VideoWriter in this OpenCV build")
        assert _wait(lambda: r.status()["written"] >= 3)
        r.set_recording(False)
        assert _wait(lambda: not r.is_recording())        # thread closed the writer on the flag flip
    finally:
        r.stop()
    assert list(tmp_path.glob("*.mp4")) and r.status()["dropped"] > 0


def test_stop_is_idempotent_and_finalises(tmp_path):
    r = _rec(tmp_path)
    r.start(_JumpBus(_snap()))
    r.stop()
    r.stop()                                              # second stop must not raise


def test_unknown_codec_falls_back_to_default(tmp_path):
    r = _rec(tmp_path, codec="zzzz")                      # unrecognised name → falls back to mp4v
    r._open(_snap())
    if r._writer is None:
        pytest.skip("no mp4v VideoWriter in this OpenCV build")
    assert r.is_recording()                              # opened via the fallback, no crash
    r._close()


def test_stop_clears_the_rec_intent(tmp_path):
    # `output rec off` (stop) must clear _want, else `output rec on` (start) spontaneously re-records.
    r = _rec(tmp_path)
    r.set_recording(True)
    r.start(_JumpBus(_snap()))
    r.stop()
    assert r._want is False
