"""lib.mot.MotWriter — the MOTChallenge track export: 1-indexed frame, the track id (not the display
slot), only the tracker's reported output (confirmed + non-ignored + has a bbox), and the 10-col CSV shape."""
from types import SimpleNamespace

from lib.mot import MotWriter


def _obj(oid, bbox, confirmed=True, ignored=False):
    return SimpleNamespace(id=oid, bbox=bbox, confirmed=confirmed, ignored=ignored)


def _snap(fi, objs):
    return SimpleNamespace(frame_idx=fi, objects=objs)


def test_mot_line_format_and_1_indexing(tmp_path):
    p = tmp_path / "m.txt"
    w = MotWriter(str(p))
    w.write(_snap(0, [_obj(7, (10, 20, 30, 40))]))               # frame_idx 0 → MOT frame 1
    w.write(_snap(1, [_obj(7, (11, 20, 30, 40)), _obj(9, (100, 100, 20, 20))]))
    w.close()
    lines = p.read_text().strip().split("\n")
    assert lines[0] == "1,7,10,20,30,40,1,10,1"                 # MOT16: frame,id,l,t,w,h,not_ignored,class,vis
    assert lines[1] == "2,7,11,20,30,40,1,10,1"                 # class 10 = other (tracker doesn't classify)
    assert lines[2] == "2,9,100,100,20,20,1,10,1"


def test_mot_excludes_unconfirmed_ignored_and_bboxless(tmp_path):
    p = tmp_path / "m.txt"
    w = MotWriter(str(p))
    w.write(_snap(0, [_obj(1, (0, 0, 5, 5), confirmed=False),    # unconfirmed → not reported
                      _obj(2, (0, 0, 5, 5), ignored=True),       # ignored (edge/arm) → not reported
                      _obj(3, None),                             # no bbox yet → skip
                      _obj(4, (1, 2, 3, 4))]))                   # the only reported track
    w.close()
    assert [l for l in p.read_text().strip().split("\n") if l] == ["1,4,1,2,3,4,1,10,1"]
    assert w.lines == 1


def test_mot_frame_and_row_counts(tmp_path):
    w = MotWriter(str(tmp_path / "m.txt"))
    w.write(_snap(0, [_obj(1, (0, 0, 1, 1)), _obj(2, (0, 0, 1, 1))]))
    w.write(_snap(1, []))                                        # an empty frame still counts as processed
    w.close()
    assert w.frames == 2 and w.lines == 2
