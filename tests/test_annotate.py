"""Scene.annotate — the producer-namespaced derived-data bus (features / cluster / dyn …)
that a smart output serializes and the gui draws. Single-writer-safe, persists across ticks,
projects (frozen) into the snapshot. Default-empty ⇒ the trio pipeline is byte-identical."""

import numpy as np
import pytest

from lib.scene import Detection, ObjectSnap, Scene, SceneError

_DET = Detection((80.0, 80.0), (60, 60, 40, 40), 1600.0, None)


def _scene_with_one(**ann):
    """A Scene with one minted object; optionally annotate it (producer=value) as 'setup'."""
    s = Scene()
    s.begin_tick(0, ts=0.0)
    with s.writing("setup"):
        oid = s.mint(_DET)
        for k, v in ann.items():
            s.annotate(oid, k, v)
    return s, oid


def _snap_of(s, oid):
    return {o.id: o for o in s.snapshot().objects}[oid]


# --------------------------------------------------------------------------- #

def test_annotation_appears_in_snapshot():
    s, oid = _scene_with_one(features={"circularity": [0.9]})
    o = _snap_of(s, oid)
    assert o.annotations["features"] == {"circularity": [0.9]}


def test_distinct_producers_coexist_no_conflict():
    s, oid = _scene_with_one()
    s.begin_tick(1, ts=1.0)
    with s.writing("features"):
        s.annotate(oid, "features", {"solidity": [1.0]})
    with s.writing("clusters"):                       # a DIFFERENT key, different stage → fine
        s.annotate(oid, "cluster", {"id": 3})
    o = _snap_of(s, oid)
    assert o.annotations["features"] == {"solidity": [1.0]}
    assert o.annotations["cluster"] == {"id": 3}


def test_same_key_two_writers_in_one_tick_conflicts():
    s, oid = _scene_with_one()
    s.begin_tick(1, ts=1.0)
    with s.writing("features"):
        s.annotate(oid, "features", {"a": [1.0]})
    with s.writing("impostor"):
        with pytest.raises(SceneError, match="single-writer"):
            s.annotate(oid, "features", {"a": [2.0]})   # same key, different stage → violation


def test_same_writer_may_update_across_ticks():
    s, oid = _scene_with_one(features={"speed": [0.0]})
    s.begin_tick(1, ts=1.0)                            # new tick clears the per-tick writer map
    with s.writing("setup"):
        s.annotate(oid, "features", {"speed": [5.0]})   # same writer, next tick → OK, updates
    assert _snap_of(s, oid).annotations["features"] == {"speed": [5.0]}


def test_annotations_persist_across_ticks_without_reannotation():
    s, oid = _scene_with_one(features={"circularity": [0.9]})   # annotated tick 0 only
    s.begin_tick(1, ts=1.0)                            # tick 1: nobody re-annotates
    s.end_tick()
    assert _snap_of(s, oid).annotations["features"] == {"circularity": [0.9]}   # still there


def test_unannotated_object_has_empty_annotations():
    s, oid = _scene_with_one()
    assert _snap_of(s, oid).annotations == {}


def test_snapshot_annotations_are_read_only():
    s, oid = _scene_with_one(features={"x": [1.0]})
    o = _snap_of(s, oid)
    with pytest.raises(TypeError):                     # MappingProxyType — a published snap is frozen
        o.annotations["features"] = {"x": [9.0]}


def test_rewrite_id_carries_annotations():
    s, oid = _scene_with_one(features={"solidity": [1.0]})
    s.begin_tick(1, ts=1.0)
    with s.writing("identity"):
        s.rewrite_id(oid, 99)
    o = _snap_of(s, 99)
    assert o.annotations["features"] == {"solidity": [1.0]}


def test_annotate_retired_object_is_a_noop():
    # a retired object lingers in this tick's snapshot (for identity/render); a stage that annotates
    # every frame (clusters) will hit it — a derived annotation on a dying object must NOT raise.
    s, oid = _scene_with_one()
    s.begin_tick(1, ts=1.0)
    with s.writing("clusters"):
        s.retire(oid)
        s.annotate(oid, "cluster", {"id": [1.0]})       # frozen → silently skipped, no error
    o = _snap_of(s, oid)
    assert "cluster" not in o.annotations                # not written


def test_clear_annotation_removes_the_key():
    s, oid = _scene_with_one(features={"x": [1.0]})
    s.begin_tick(1, ts=1.0)
    with s.writing("clusters"):
        s.clear_annotation(oid, "features")
    assert "features" not in _snap_of(s, oid).annotations


def test_clear_annotation_absent_or_retired_is_noop():
    s, oid = _scene_with_one()
    with s.writing("clusters"):
        s.clear_annotation(oid, "cluster")              # absent → no error
        s.retire(oid)
        s.clear_annotation(oid, "cluster")              # retired → no error
    assert _snap_of(s, oid).annotations == {}


def test_annotate_missing_id_still_raises():
    s, _oid = _scene_with_one()
    with s.writing("clusters"):
        with pytest.raises(SceneError, match="not in scene"):
            s.annotate(999, "cluster", {"id": [1.0]})   # a real bug — no such object


def test_objectsnap_annotations_default_is_empty():
    # ObjectSnap can still be built without annotations (tests / other producers rely on the default)
    o = ObjectSnap(id=1, status=None, ignored=False, centroid=None, bbox=None, area=None,
                   contour=None, pose=None)
    assert o.annotations == {}
