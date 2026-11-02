"""lib/pipeline — the core tick: open a Scene tick, run the stages in declared order under their
write provenance, publish the snapshot. The suite reached this module only through test_e2e_clip,
which builds the REAL core and therefore always takes the FAST path (build() is called without
--profile/--debug, so profiler and audit are both None): the instrumented branch — a second, hand-
duplicated copy of the same sequence — was never executed by any test. These tests drive both
branches and assert they agree, plus the contracts nothing else pins: stage order, the shared
FrameContext, the `.name` provenance fallback, and the lifecycle order around the snapshot."""

import numpy as np
import pytest

from lib.events import FrameCaptured
from lib.pipeline import FrameContext, Pipeline
from lib.scene import Scene


def _fc(frame_idx=1, ts=0.5, full=False):
    bgr = np.zeros((8, 8, 3), np.uint8)
    return FrameCaptured(frame_idx=frame_idx, ts=ts, bgr=bgr, gray=np.zeros((8, 8), np.uint8),
                         input_full=np.ones((16, 16, 3), np.uint8) if full else None)


class _Stage:
    """A recording stand-in for a real stage: `(ctx, scene) -> None` with a `.name`, per the Stage
    contract. Appends (name, ctx, scene) so a test can assert order AND what each stage was handed."""

    def __init__(self, name, log):
        self.name, self._log = name, log

    def __call__(self, ctx, scene):
        self._log.append((self.name, ctx, scene))


class _SpyScene(Scene):
    """Real Scene (so the write-audit is real), instrumented to record the tick lifecycle and every
    write-provenance name the Pipeline opens."""

    def __init__(self):
        super().__init__()
        self.writers, self.lifecycle = [], []

    def begin_tick(self, *a, **kw):
        self.lifecycle.append("begin_tick")
        return super().begin_tick(*a, **kw)

    def writing(self, name):
        self.writers.append(name)
        return super().writing(name)

    def snapshot(self, *a, **kw):
        self.lifecycle.append("snapshot")
        return super().snapshot(*a, **kw)

    def end_tick(self, *a, **kw):
        self.lifecycle.append("end_tick")
        return super().end_tick(*a, **kw)


class _Prof:
    def __init__(self):
        self.records = []

    def record(self, label, dt, group="pipeline"):
        self.records.append((label, dt))


class _Audit:
    def __init__(self):
        self.calls = []

    def begin_tick(self, frame_idx, ts):
        self.calls.append(("begin_tick", frame_idx, ts))

    def end_tick(self):
        self.calls.append(("end_tick",))


class _Mot:
    def __init__(self):
        self.written = []

    def write(self, snap):
        self.written.append(snap)


def _stages(log, names=("observe", "classify", "identity", "pose")):
    return [_Stage(n, log) for n in names]


# --------------------------------------------------------------------------- #
# stage order + what a stage is handed
# --------------------------------------------------------------------------- #

def test_stages_run_in_declared_order():
    # the paradigm depends on this sequence: observe's masks feed classify's verdicts, which decide
    # what identity may claim, which fixes what pose measures. Nothing else in the suite asserts it —
    # a reorder would surface downstream as a churned id, naming the wrong culprit.
    log = []
    Pipeline(_SpyScene(), _stages(log)).run(_fc())
    assert [c[0] for c in log] == ["observe", "classify", "identity", "pose"]


def test_every_stage_shares_ONE_FrameContext_built_from_the_frame():
    # stages talk to each other through the ctx, not through events (observe fills obj_fg, later
    # stages read it) — so it must be the SAME object all the way down, carrying this frame's data.
    log = []
    Pipeline(_SpyScene(), _stages(log)).run(_fc(frame_idx=7, ts=1.25))
    ctxs = [c[1] for c in log]
    assert all(c is ctxs[0] for c in ctxs)
    assert isinstance(ctxs[0], FrameContext)
    assert ctxs[0].frame_idx == 7 and ctxs[0].ts == 1.25
    assert ctxs[0].obj_fg is None                     # observe's slot, empty until observe fills it


def test_each_tick_gets_a_fresh_context():
    # ctx is per-frame scratch: leaking one tick's obj_fg into the next would let a stale mask decide
    # the current frame's verdicts.
    log = []
    p = Pipeline(_SpyScene(), _stages(log, ("observe",)))
    p.run(_fc(frame_idx=1))
    p.run(_fc(frame_idx=2))
    assert log[0][1] is not log[1][1]
    assert [c[1].frame_idx for c in log] == [1, 2]


def test_every_stage_gets_the_pipelines_own_scene():
    log = []
    scene = _SpyScene()
    Pipeline(scene, _stages(log)).run(_fc())
    assert all(c[2] is scene for c in log)


# --------------------------------------------------------------------------- #
# write provenance + tick lifecycle
# --------------------------------------------------------------------------- #

def test_each_stage_runs_under_its_own_write_provenance():
    # every stage's writes are attributed to its `.name` — that attribution is what lets the Scene
    # catch two stages writing the same field in one tick.
    log = []
    scene = _SpyScene()
    Pipeline(scene, _stages(log)).run(_fc())
    assert scene.writers == ["observe", "classify", "identity", "pose"]


def test_a_stage_without_a_name_falls_back_to_the_literal_stage():
    # `getattr(stage, "name", "stage")` — every production stage sets `.name`, so this default is
    # never taken in the real system and would rot unnoticed.
    scene = _SpyScene()
    Pipeline(scene, [lambda ctx, s: None]).run(_fc())
    assert scene.writers == ["stage"]


def test_the_tick_is_bracketed_begin_then_snapshot_then_end():
    # the snapshot must be taken INSIDE the tick (before end_tick evicts retired objects), else a
    # retirement would vanish from the frame that reported it.
    scene = _SpyScene()
    snap = Pipeline(scene, _stages([])).run(_fc())
    assert scene.lifecycle == ["begin_tick", "snapshot", "end_tick"]
    assert snap is not None and snap.frame_idx == 1


def test_the_frame_reaches_the_snapshot_and_input_full_is_withheld_from_ctx():
    # the pre-crop 'input' view rides the Scene tick (outputs read it off the snapshot), NOT the ctx —
    # stages track in post-preprocess space only.
    log = []
    snap = Pipeline(_SpyScene(), _stages(log, ("observe",))).run(_fc(full=True))
    assert not hasattr(log[0][1], "input_full")
    assert snap.input_full is not None and snap.input_full.shape == (16, 16, 3)


# --------------------------------------------------------------------------- #
# the instrumented branch — a hand-duplicated copy of the same sequence
# --------------------------------------------------------------------------- #

def _run(**kw):
    log = []
    scene = _SpyScene()
    snap = Pipeline(scene, _stages(log), **kw).run(_fc())
    return [c[0] for c in log], scene.writers, scene.lifecycle, snap


def test_instrumented_and_fast_paths_do_the_same_work():
    # lib/pipeline.py keeps TWO copies of the tick — the fast path and the timed one — and only the
    # fast path runs in production and in test_e2e_clip. Any edit to one must land on the other, so
    # the three collaborator combinations are compared against the uninstrumented baseline.
    base = _run()
    assert _run(profiler=_Prof()) [:3] == base[:3]
    assert _run(audit=_Audit())   [:3] == base[:3]
    assert _run(profiler=_Prof(), audit=_Audit())[:3] == base[:3]


def test_an_audit_alone_switches_to_the_instrumented_path():
    # the branch is `prof is None AND audit is None` — an audit with no profiler must still take the
    # instrumented route (and must not trip over the profiler being absent).
    audit = _Audit()
    Pipeline(_SpyScene(), _stages([]), audit=audit).run(_fc(frame_idx=3, ts=2.0))
    assert audit.calls == [("begin_tick", 3, 2.0), ("end_tick",)]


def test_the_profiler_is_fed_every_stage_plus_snapshot_and_the_whole_tick():
    prof = _Prof()
    Pipeline(_SpyScene(), _stages([]), profiler=prof).run(_fc())
    labels = [r[0] for r in prof.records]
    assert labels == ["observe", "classify", "identity", "pose", "snapshot", "pipeline"]
    assert all(dt >= 0.0 for _l, dt in prof.records)
    # the bracket measurement must cover the parts it brackets
    spans = dict(prof.records)
    assert spans["pipeline"] >= spans["snapshot"]


# --------------------------------------------------------------------------- #
# the MOT export tap
# --------------------------------------------------------------------------- #

def test_the_mot_writer_is_fed_the_published_snapshot_on_both_paths():
    # --mot-export is how every GT/benchmark number is produced; it must see exactly the snapshot the
    # outputs see, on whichever path the run happens to take.
    for kw in ({}, {"profiler": _Prof()}):
        mot = _Mot()
        snap = Pipeline(_SpyScene(), _stages([]), mot=mot, **kw).run(_fc())
        assert mot.written == [snap]


def test_no_mot_writer_is_simply_no_tap():
    snap = Pipeline(_SpyScene(), _stages([])).run(_fc())     # mot=None — must not raise
    assert snap is not None


# --------------------------------------------------------------------------- #
# degenerate construction
# --------------------------------------------------------------------------- #

def _boom(ctx, scene):
    raise RuntimeError("stage fatal")


_boom.name = "boom"


def test_a_raising_stage_still_closes_the_tick():
    # a stage fatal propagates (the loop captures it and the orchestrator re-raises on the main
    # thread — that is deliberate), but the tick must still CLOSE, so nothing reading the Scene
    # during shutdown sees a store left mid-tick with retired objects un-evicted.
    scene = _SpyScene()
    with pytest.raises(RuntimeError, match="stage fatal"):
        Pipeline(scene, [_Stage("observe", []), _boom]).run(_fc())
    assert scene.lifecycle == ["begin_tick", "end_tick"]     # no snapshot — it raised before it


def test_a_raising_mot_tap_still_closes_the_tick_and_the_audit_record():
    # --mot-export is a diagnostic side-channel; a disk-full write raising must not leave the tick
    # open NOR the audit's tick record unterminated (a post-mortem debug log ending on an unclosed
    # tick is exactly when you need it to be readable).
    class _BadMot:
        def write(self, snap):
            raise OSError("no space left on device")

    scene, audit = _SpyScene(), _Audit()
    with pytest.raises(OSError):
        Pipeline(scene, _stages([]), mot=_BadMot(), audit=audit).run(_fc())
    assert scene.lifecycle == ["begin_tick", "snapshot", "end_tick"]
    assert [c[0] for c in audit.calls] == ["begin_tick", "end_tick"]


def test_a_pipeline_with_no_stages_still_publishes_a_snapshot():
    # the stage list is optional (Pipeline(scene) is legal) — an empty tick must still open, publish
    # and close, so an all-optional-stages build can't wedge the loop.
    scene = _SpyScene()
    snap = Pipeline(scene).run(_fc())
    assert scene.lifecycle == ["begin_tick", "snapshot", "end_tick"]
    assert snap.objects == ()


def test_the_stage_sequence_is_frozen_at_construction():
    # stages are tuple()'d in __init__ — mutating the caller's list afterwards must not change what
    # the pipeline runs (build() hands in a list it goes on to use).
    log = []
    stages = _stages(log, ("observe",))
    p = Pipeline(_SpyScene(), stages)
    stages.append(_Stage("intruder", log))
    p.run(_fc())
    assert [c[0] for c in log] == ["observe"]
