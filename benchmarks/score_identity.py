"""score_identity.py — scoring tracks against the ground truth (needs motmetrics; install it
with `venv/bin/pip install -r benchmarks/requirements.txt`).

analyze.py imports the functions below to write identity.csv from a run's exported tracks;
running this file directly re-scores benchmarks/runs/<machine>/tracks_<clip>.txt in BOTH
scoring modes (analyze.py reports only the primary one). IDF1 primary; MOTA, IDSW;
settle-success % over every GT ONSET (placements + moves).

Conventions:
- GT is REST-ANCHORED (measured): boxes hold the last rest position during manipulation and
  TELEPORT to the new rest — the same convention the tracker emits. So `all_frames` scoring is
  already rest-based; the residual mismatch is the settle/confirm LAG around each GT teleport.
- scoring_mode `rest_settled` (PRIMARY): around every GT ONSET a ±G-frame window (G=24 ≈ 0.8 s
  @30fps) is DON'T-CARE for that id — its GT rows are dropped and any pred box overlapping the
  id's old OR new rest box (IoU ≥ thresh) is dropped too. An onset is a MOVE (per-id center step
  > 8 px) or a PLACEMENT (an id first appearing after the clip opens, which has no old rest).
- scoring_mode `all_frames`: no exclusion (transparency row).
- Compound GT ids (class 12 — grouper compounds; heavy clip only) are excluded from scoring,
  from scoring. gt_objs counts primary ids.
- settle_success: a MOVE succeeds iff within `SETTLE_WINDOW` frames the tracker reports a box on
  the NEW rest position (IoU ≥ thresh) carrying the SAME tracker id it held before the move; a
  PLACEMENT succeeds iff it reports ANY box there in time (no prior identity exists to carry).
  This is the paper's "settle-success %".
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# motmetrics 1.4.0 still calls np.asfarray (removed in NumPy 2.0); numpy<2 has no
# Python-3.14 wheels, so restore the alias before importing it. Behaviour-identical.
if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)

import motmetrics as mm  # noqa: E402

BENCH_DIR = Path(__file__).resolve().parent
REPO = BENCH_DIR.parent
sys.path.insert(0, str(BENCH_DIR))
sys.path.insert(0, str(REPO))

IOU_THRESH = 0.5          # the match gate, stated + parameterized (one place)
TELEPORT_PX = 8.0         # per-id center step that counts as a rearrangement event
G_WINDOW = 24             # ± frames excluded around a teleport in rest_settled mode
SETTLE_WINDOW = 150       # frames a teleport has to re-settle before counting as a failure
COMPOUND_CLASS = 12


def load_mot(path):
    """MOT16 → {frame: [(id, l, t, w, h, cls)]}"""
    out = defaultdict(list)
    for ln in open(path):
        v = ln.strip().split(",")
        if len(v) < 6:
            continue
        cls = int(v[7]) if len(v) >= 8 else -1
        out[int(v[0])].append((int(v[1]), float(v[2]), float(v[3]), float(v[4]), float(v[5]), cls))
    return dict(out)


def iou(a, b):
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    ua = a[2] * a[3] + b[2] * b[3] - inter
    return inter / ua if ua > 0 else 0.0


def _tracks(gt):
    """{gid: {frame: box}} over PRIMARY ids only — class-12 grouper compounds are alternate
    groupings of objects already counted, not objects, so they generate no onsets: a compound
    "appearing" is the grouper fusing two things that were both already on the table. Excluded
    here for the same reason gt_objs and score_mode exclude them. (Empirically this changes only
    arrivals — compounds appear and vanish rather than teleport — so teleport_events is unchanged
    and stays comparable across the suite's history.)"""
    tracks = defaultdict(dict)
    for f, dets in gt.items():
        for gid, l, t, w, h, c in dets:
            if c == COMPOUND_CLASS:
                continue
            tracks[gid][f] = (l, t, w, h)
    return tracks


def teleports(gt):
    """Per-id MOVE events: [(gid, f_jump, box_old, box_new)] — the object was already on the table
    and was relocated. The rest-anchored GT holds the box still, then teleports it to the new rest,
    so ``f_jump`` is the frame the object counts as LANDED (not as picked up)."""
    ev = []
    for gid, tr in _tracks(gt).items():
        fs = sorted(tr)
        for a, b in zip(fs, fs[1:]):
            if b - a != 1:
                continue
            (l0, t0, w0, h0), (l1, t1, w1, h1) = tr[a], tr[b]
            if np.hypot((l1 + w1 / 2) - (l0 + w0 / 2), (t1 + h1 / 2) - (t0 + h0 / 2)) > TELEPORT_PX:
                ev.append((gid, b, tr[a], tr[b]))
    return ev


def arrivals(gt):
    """Per-id PLACEMENT events: [(gid, f_first, None, box_new)] — an id whose first GT frame is after
    the clip opens, i.e. an object carried in and set down. ``box_old`` is None because there is no
    previous rest to jump from; that None is what marks an arrival to the two consumers below.

    A placement is NOT a teleport (nothing to displace from), so ``teleports`` cannot see it — yet it
    is exactly the onset an instrument fires on, and the case a settle-based tracker most has to get
    right. Objects present in the opening frame are the initial arrangement, not events."""
    tracks = _tracks(gt)
    first_frame = min((f for tr in tracks.values() for f in tr), default=None)
    ev = []
    for gid, tr in tracks.items():
        f0 = min(tr)
        if first_frame is not None and f0 > first_frame:   # appeared after the clip opened
            ev.append((gid, f0, None, tr[f0]))
    return ev


def rest_events(gt):
    """Every GT onset — placements AND moves — in frame order. The rest-anchored convention makes
    both a LANDING instant, so this is the full set of moments the tracker is expected to report."""
    return sorted(arrivals(gt) + teleports(gt), key=lambda e: (e[1], e[0]))


def settle_success(gt, pred, events):
    """% of onsets the tracker lands. For a MOVE, success = it re-reports the SAME id on the new rest
    within the window (identity carried across the rearrangement). For a PLACEMENT (``box_old`` None)
    there is no prior identity to carry, so success = it reports ANY box on the new rest in time —
    the object was seen and published at all."""
    ok = 0
    for gid, fj, box_old, box_new in events:
        pre_id = None                                    # the tracker id holding the object before the move
        for f in range(fj - 1, fj - 11, -1):
            if box_old is None:                          # a placement has no previous rest to scan
                break
            best = max(((iou(box_old, p[1:5]), p[0]) for p in pred.get(f, ())), default=(0, None))
            if best[0] >= IOU_THRESH:
                pre_id = best[1]
                break
        got = False
        for f in range(fj, fj + SETTLE_WINDOW):
            best = max(((iou(box_new, p[1:5]), p[0]) for p in pred.get(f, ())), default=(0, None))
            if best[0] >= IOU_THRESH:
                got = best[1] == pre_id if pre_id is not None else True
                break
        ok += bool(got)
    return 100.0 * ok / len(events) if events else 100.0, len(events)


def score_mode(gt, pred, mode, events):
    """One motmetrics pass → (idf1, mota, idsw)."""
    excl = defaultdict(list)                             # frame -> [(gid, box_old, box_new)]
    if mode == "rest_settled":
        for gid, fj, box_old, box_new in events:
            for f in range(fj - G_WINDOW, fj + G_WINDOW + 1):
                excl[f].append((gid, box_old, box_new))
    acc = mm.MOTAccumulator(auto_id=False)
    for f in sorted(set(gt) | set(pred)):
        grows = [(gid, (l, t, w, h)) for gid, l, t, w, h, c in gt.get(f, ()) if c != COMPOUND_CLASS]
        prows = [(pid, (l, t, w, h)) for pid, l, t, w, h, _c in pred.get(f, ())]
        for gid, box_old, box_new in excl.get(f, ()):    # don't-care: drop the id's GT row + any pred
            grows = [(g, b) for g, b in grows if g != gid]           # box sitting on its old/new rest
            prows = [(p, b) for p, b in prows                        # (a placement has no old rest)
                     if (box_old is None or iou(b, box_old) < IOU_THRESH)
                     and iou(b, box_new) < IOU_THRESH]
        gids = [g for g, _ in grows]
        pids = [p for p, _ in prows]
        dist = mm.distances.iou_matrix([b for _, b in grows], [b for _, b in prows],
                                       max_iou=1.0 - IOU_THRESH)     # distance = 1−IoU; > this → no match
        acc.update(gids, pids, dist, frameid=f)
    mh = mm.metrics.create()
    s = mh.compute(acc, metrics=["idf1", "mota", "num_switches"], name="run")
    return (float(s["idf1"].iloc[0]), float(s["mota"].iloc[0]), int(s["num_switches"].iloc[0]))


def main():
    """Re-score every machine's exported tracks, both modes, to stdout.

    The retired harness aggregated a CSV here; the shipped flow's identity.csv is written by
    analyze.py. This entry point exists for the second scoring mode (all_frames, the
    transparency row) and for re-scoring by hand without a full analyze pass.
    """
    runs = BENCH_DIR / "runs"
    machines = sorted(p for p in runs.iterdir() if p.is_dir()) if runs.exists() else []
    if not machines:
        print("no runs found - do this first:  venv/bin/python benchmarks/run.py", file=sys.stderr)
        return 1
    for mdir in machines:
        for mot in sorted(mdir.glob("tracks_*.txt")):
            clip = mot.stem[len("tracks_"):]
            gt_path = REPO / "tests" / "videos" / f"{clip}.mp4.MOT"
            if not gt_path.exists():
                continue                       # a performance-only clip has no annotation
            gt = load_mot(gt_path)
            pred = load_mot(mot)
            ev = rest_events(gt)
            ss, n_ev = settle_success(gt, pred, ev)
            for mode in ("rest_settled", "all_frames"):
                idf1, mota, idsw = score_mode(gt, pred, mode, ev)
                print(f"  {mdir.name:14s} {clip:22s} {mode:13s} IDF1={idf1:.4f} "
                      f"MOTA={mota:.4f} IDSW={idsw:3d} settle={ss:5.1f}% ({n_ev} onsets)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
