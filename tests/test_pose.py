"""pose — the per-frame settle-fit BUDGET. The expensive rotation fit runs at most `max_fits_per_frame`
times per frame; extra simultaneous re-settles defer to the next frame(s) so a burst of settles can't
spike one frame. (The one-shot fit itself is exercised via the geometry tests.)"""

from types import SimpleNamespace

from lib.pose import Pose
from lib.scene import Status


class _Cfg:
    def __init__(self, **kw):
        self._d = kw
    def get(self, k, d=None):
        return self._d.get(k, d)


class _CountingPose(Pose):
    """Count the expensive-fit calls instead of running the real rotation fit."""
    def __init__(self, cfg):
        super().__init__(cfg)
        self.placed = []
    def _place_at_settle(self, scene, obj):
        self.placed.append(obj.id)


class _FakeScene:
    def __init__(self, objs):
        self._objs = objs
    def recent_rewrites(self):
        return ()
    def objects(self):
        return self._objs


def _resettling(*ids):
    # RESTING with a pose_ref present → refit=True (an expensive re-settle, budgeted)
    return [SimpleNamespace(id=i, status=Status.RESTING, footprint=object(), pose_ref=object()) for i in ids]


def test_three_simultaneous_resettles_are_spread_over_frames():
    p = _CountingPose(_Cfg(max_fits_per_frame=1))
    objs = _resettling(1, 2, 3)
    for o in objs:
        p._last[o.id] = Status.MOVED                 # pre-RESTING → the settle edge fires
    scene = _FakeScene(objs)
    p(SimpleNamespace(), scene); assert p.placed == [1]          # budget 1 → one fit this frame
    p(SimpleNamespace(), scene); assert p.placed == [1, 2]       # deferred #2 re-fires
    p(SimpleNamespace(), scene); assert p.placed == [1, 2, 3]    # all measured within 3 frames
    p(SimpleNamespace(), scene); assert p.placed == [1, 2, 3]    # steady RESTING → no more fits


def test_budget_two_fits_two_per_frame():
    p = _CountingPose(_Cfg(max_fits_per_frame=2))
    objs = _resettling(1, 2, 3)
    for o in objs:
        p._last[o.id] = Status.MOVED
    scene = _FakeScene(objs)
    p(SimpleNamespace(), scene); assert p.placed == [1, 2]       # two fits this frame
    p(SimpleNamespace(), scene); assert p.placed == [1, 2, 3]    # the third next frame


def test_ignored_object_gets_no_pose_fit():
    # an edge/border-residue blob flagged `ignored` (a hand/arm reaching in) must not consume a
    # rotation fit — it is excluded from pose just as from every output.
    p = _CountingPose(_Cfg(max_fits_per_frame=4))
    objs = _resettling(1, 2)
    objs[0].ignored = True                               # the first is edge residue
    for o in objs:
        p._last[o.id] = Status.MOVED                     # both sit at the settle edge
    p(SimpleNamespace(), _FakeScene(objs))
    assert p.placed == [2]                               # only the non-ignored object is fitted


def test_first_settle_is_not_budgeted():
    # a FIRST settle (no canon reference, no pose_ref) is θ=0 with no rotation fit → it must NOT consume
    # the budget, so several first-settles in one frame all place immediately.
    p = _CountingPose(_Cfg(max_fits_per_frame=1))
    objs = [SimpleNamespace(id=i, status=Status.RESTING, footprint=object(), pose_ref=None) for i in (1, 2, 3)]
    for o in objs:
        p._last[o.id] = Status.SETTLING
    p(SimpleNamespace(), _FakeScene(objs))
    assert p.placed == [1, 2, 3]                                 # all placed — first settles aren't capped
