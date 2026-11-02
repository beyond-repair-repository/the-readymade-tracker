"""analyze.py — turn what a run wrote into the tables. Reads logs, computes nothing itself.

    venv/bin/python benchmarks/run.py          first: play the clips (~2.5 min; --gt ~5)
    venv/bin/python benchmarks/analyze.py      then: logs -> CSV

Every number here was written by the tracker or by a real client. This file parses and joins;
it never times anything, never drives the pipeline, and never reaches inside it. If a figure
looks wrong the run is wrong, not the analysis — which is the point of doing it this way.

WHAT IT READS, PER MACHINE, FROM benchmarks/runs/<machine>/
    profile_<clip>.log   the tracker's own --profile summary. Its "@ exit" block is the whole
                         run: every step's cost in ms per frame, grouped by the thread that
                         pays it, plus the frame rate achieved.
    debug_<clip>.log     every lifecycle event with its frame number, so "how many objects, how
                         many came to rest, when" is counted rather than assumed.
    client_<clip>.jsonl  what a real TUIO client received, and the wall-clock instant it arrived.
    tracks_<clip>.txt    the tracker's own tracks, to score against the hand-made ground truth.

WHAT IT WRITES, into benchmarks/results/<machine>/ — one folder per machine, so collecting a
second machine is copying a folder and nothing can overwrite anyone else's numbers:
    steps.csv       machine · clip · step -> ms per frame. The stage table.
    summary.csv     machine · clip -> the three thread totals, frame rate, and what was in the
                    scene. One line per clip per machine.
    latency.csv     machine · clip -> object at rest to first message AT THE CLIENT.
    identity.csv    machine · clip -> did it get the objects RIGHT, against the ground truth.
                    The only correctness table here; everything else is speed.

THE LATENCY IS MEASURED HONESTLY, and this is the only part that needs explaining. Each TUIO
bundle carries the frame number it describes and the wall-clock instant that frame was CAPTURED
(its timetag). The client records the instant it RECEIVED that bundle. So for a moment the
ground truth says an object came to rest, we can say exactly when a listening patch first heard
about something at that spot — start and end both real, both outside the tracker's own opinion.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
REPO = BENCH_DIR.parent
sys.path.insert(0, str(BENCH_DIR))

RUNS = BENCH_DIR / "runs"          # the logs: big, local, gitignored
RESULTS = BENCH_DIR / "results"    # the tables: small, one folder per machine, committed
VIDEOS = REPO / "tests" / "videos"
FRAME_PX = 1024.0                      # the clips' size; TUIO positions are 0..1 of it

# A profile row: four leading spaces, a name, then ms/frame and optionally min/avg/max.
# The name is separated from the numbers by TWO OR MORE spaces, not by "the first digit" — a
# step called tuio2 ends in one, and matching on digits dropped it from every table silently.
ROW = re.compile(r"^ {4}(\S.*?)\s{2,}([\d.]+)(?:\s+([\d.]+)\s+([\d.]+)\s+([\d.]+))?\s*$")
GROUP = re.compile(r"^ {2}([A-Z]+) - ")
FPS = re.compile(r"pipeline fps min/avg/max\s+(\d+) /\s*(\d+) /\s*(\d+)")
ACHIEVED = re.compile(r"overall\s+(\d+) fps \(achieved\)")
FRAMES = re.compile(r"amortized over (\d+) frames")


def parse_profile(path: Path) -> dict:
    """The LAST '@ exit' block of a --profile log: {group: {step: (ms_frame, min, avg, max)}}."""
    text = path.read_text(errors="replace")
    i = text.rfind("profile @ exit")
    if i < 0:
        # The "@ exit" block is the WHOLE run; the periodic ones are snapshots taken every 100
        # frames. Falling back to a snapshot would report part of a run as if it were all of it,
        # and nothing in the table would say so. A missing block means the tracker did not shut
        # down cleanly — re-run that clip rather than publishing a number from the middle of it.
        print(f"  ! {path.name} has no '@ exit' summary — the run did not finish cleanly. "
              f"SKIPPED (re-run that clip).", file=sys.stderr)
        return {}
    block = text[i:]
    out: dict = {"groups": {}, "frames": None, "capacity_fps": None, "achieved_fps": None}
    group = None
    for ln in block.splitlines():
        if m := GROUP.match(ln):
            group = m.group(1).lower()
            out["groups"][group] = {}
            continue
        if m := FRAMES.search(ln):
            out["frames"] = int(m.group(1))
        if m := FPS.search(ln):
            out["capacity_fps"] = int(m.group(2))
        if m := ACHIEVED.search(ln):
            out["achieved_fps"] = int(m.group(1))
        if group and (m := ROW.match(ln)):
            name = m.group(1).strip()
            vals = tuple(float(v) if v else None for v in m.groups()[1:])
            out["groups"][group][name] = vals
    return out


# "[readymade] 425 frames processed, 264 dropped (38.3%) · pipeline fps ... · 26.5 fps (achieved)"
CONSOLE = re.compile(r"\[readymade\] (\d+) frames processed, (\d+) dropped")
CONSOLE_FPS = re.compile(r"([\d.]+) fps \(achieved\)")


def parse_console(path: Path) -> dict:
    """Frames DROPPED and the rate achieved — the two numbers only the console carries.

    A drop is not an error: the frame slot is latest-wins, so a tick that overruns the budget
    discards the next frame rather than falling behind. It is the honest "did not keep up"
    signal, and it exists only in a paced run.
    """
    text = path.read_text(errors="replace")
    out: dict = {}
    if m := CONSOLE.search(text):
        processed, dropped = int(m.group(1)), int(m.group(2))
        out["processed"] = processed
        out["dropped"] = dropped
        total = processed + dropped
        out["dropped_pct"] = round(100.0 * dropped / total, 1) if total else 0.0
    if m := CONSOLE_FPS.search(text):
        out["achieved_fps"] = float(m.group(1))
    return out


def parse_debug(path: Path) -> dict:
    """Count what happened, from the tracker's own lifecycle lines."""
    n = {"objects": 0, "captures": 0, "rested": 0, "moved": 0}
    seen = set()
    for ln in path.read_text(errors="replace").splitlines():
        if " mint #" in ln:
            seen.add(ln.split("mint #")[1].split()[0])
        elif " capture #" in ln:
            n["captures"] += 1
        elif "→RESTING" in ln:
            n["rested"] += 1
        elif "→MOVED" in ln or "→MOVING" in ln:
            n["moved"] += 1
    n["objects"] = len(seen)
    return n


def client_frames(path: Path):
    """[(frame_idx, captured_wall, arrived_wall, [(x_px, y_px), ...])] from the client's recording.

    Each bundle says which frame it describes and when that frame was captured (the TUIO timetag,
    UNIX seconds); the client stamped when it arrived. Both ends real, and on the same clock.
    """
    out = []
    for ln in path.read_text(errors="replace").splitlines():
        try:
            rec = json.loads(ln)
        except Exception:                          # noqa: BLE001 — a truncated last line is normal
            continue
        arrived, frame, captured, toks = rec.get("wall"), None, None, []
        if arrived is None:                        # recorded before the monitor stamped wall time
            continue
        for addr, args in rec.get("msgs", []):
            if addr.endswith("/frm") and len(args) >= 2:
                frame = int(args[0])
                if isinstance(args[1], dict):
                    captured = args[1].get("timetag")
            elif addr.endswith("/tok") and len(args) >= 5:
                toks.append((float(args[3]) * FRAME_PX, float(args[4]) * FRAME_PX))
        if frame is not None and captured:
            out.append((frame, captured, arrived, toks))
    return out


def rest_to_report(clip: str, bundles) -> dict | None:
    """Ground-truth landing instant -> the first message placing something there, at the client."""
    try:
        from score_identity import load_mot, rest_events
        gt = load_mot(VIDEOS / f"{clip}.mp4.MOT")
        events = rest_events(gt)
    except Exception:                              # noqa: BLE001 — no ground truth for this clip
        return None
    if not events or not bundles:
        return None
    captured = {f: c for f, c, _a, _t in bundles}
    lat = []
    for _gid, landing, _old, (bx, by, bw, bh) in events:
        t0 = captured.get(landing)
        if t0 is None:                             # that frame was never emitted (a drop)
            continue
        for frame, _c, arrived, toks in bundles:
            if frame < landing:
                continue
            if any(bx <= x <= bx + bw and by <= y <= by + bh for x, y in toks):
                lat.append((arrived - t0) * 1000.0)
                break
    if not lat:
        return None
    lat.sort()
    return {"onsets": len(events), "matched": len(lat),
            "median_ms": round(lat[len(lat) // 2], 1),
            "p95_ms": round(lat[min(len(lat) - 1, int(0.95 * len(lat)))], 1)}


def score_tracks(clip: str, mot: Path) -> dict | None:
    """The tracker's tracks against the hand-made ground truth. None if it cannot be scored.

    `rest_settled` is the mode the design deserves: a settle-based tracker is not chasing, so
    the frames around a landing are DON'T-CARE — scoring them would penalise it for the dwell it
    takes on purpose. settle_success is the plainer question: of every moment the ground truth
    says an object came to rest, how many did it report.
    """
    try:
        from score_identity import load_mot, rest_events, score_mode, settle_success
        gt, pred = load_mot(VIDEOS / f"{clip}.mp4.MOT"), load_mot(mot)
        ev = rest_events(gt)
        ss, n_ev = settle_success(gt, pred, ev)
        idf1, mota, idsw = score_mode(gt, pred, "rest_settled", ev)
    except Exception as e:                             # noqa: BLE001 — no GT, or motmetrics absent
        print(f"  (no identity score for {clip}: {e})")
        return None
    return {"onsets": n_ev, "settle_success_pct": round(ss, 1),
            "idf1": round(idf1, 4), "mota": round(mota, 4), "id_switches": idsw}


def write(path: Path, rows: list[dict], note: str) -> None:
    """One table, one machine, one file. No merging, because there is nothing to merge.

    Each machine owns benchmarks/results/<machine>/ and writes only there, so collecting a
    second machine is copying a folder — a few kilobytes — and nothing can silently overwrite
    or half-update anyone else's numbers. The logs that produced them stay put; they are ~90 MB
    a machine, nearly all of it the client capture.
    """
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        fh.write(f"# {note}\n")
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"  {path.parent.name}/{path.name:<14} {len(rows):>3} rows")


def combine() -> None:
    """Print every machine that has results here, side by side.

    Regenerated from the per-machine folders every time and never stored, so it cannot drift
    from them: a machine whose folder is not present simply is not in the table, which is the
    honest answer rather than a stale row nobody notices.
    """
    machines = sorted(p for p in RESULTS.iterdir() if p.is_dir()) if RESULTS.exists() else []
    rows = []
    for m in machines:
        f = m / "summary.csv"
        if not f.exists():
            continue
        with open(f) as fh:
            rows += [r for r in csv.DictReader(ln for ln in fh if not ln.startswith("#"))]
    if len(machines) < 2:
        print(f"\n  {len(machines)} machine so far. Copy another machine's "
              f"benchmarks/results/<name>/ folder here to compare them.")
        return
    cols = ["machine", "clip", "dropped", "achieved_fps", "capacity_fps", "pipeline_ms",
            "output_ms"]
    w = max(len(r.get("machine", "")) for r in rows) + 2
    print(f"\n  ALL MACHINES ({len(machines)})")
    print("  " + f"{'machine':<{w}}" + "".join(f"{c:>16}" for c in cols[1:]))
    for r in sorted(rows, key=lambda r: (r.get("clip", ""), r.get("machine", ""))):
        print("  " + f"{r.get('machine', ''):<{w}}"
              + "".join(f"{r.get(c, '-'):>16}" for c in cols[1:]))


def main() -> int:
    if not RUNS.exists() or not any(RUNS.iterdir()):
        print("no runs found — do this first:  venv/bin/python benchmarks/run.py", file=sys.stderr)
        return 1

    steps, summary, latency, identity = [], [], [], []
    machine_name = ""
    for machine_dir in sorted(p for p in RUNS.iterdir() if p.is_dir()):
        machine = machine_name = machine_dir.name
        # what this machine measured WITH — python/numpy/cv2 are the whole dependency surface,
        # so a row without them cannot be compared with one measured on another interpreter.
        ef = machine_dir / "env.json"
        env = json.loads(ef.read_text()) if ef.exists() else {}
        env = {k: env.get(k, "") for k in ("python", "numpy", "cv2")}
        for prof in sorted(machine_dir.glob("profile_*.log")):
            clip = prof.stem[len("profile_"):]
            p = parse_profile(prof)
            if not p:                              # truncated log — already reported
                continue
            for group, rows in p["groups"].items():
                for step, (msf, lo, avg, hi) in rows.items():
                    if step == "TOTAL":
                        continue
                    steps.append({"machine": machine, "clip": clip, "thread": group,
                                  "step": step, "ms_per_frame": msf,
                                  "per_call_min_ms": lo, "per_call_avg_ms": avg,
                                  "per_call_max_ms": hi})
            row = {"machine": machine, "clip": clip, "frames": p["frames"], **env}
            for group in ("input", "pipeline", "output"):
                tot = p["groups"].get(group, {}).get("TOTAL")
                row[f"{group}_ms"] = tot[0] if tot else None
            row["capacity_fps"] = p["capacity_fps"]
            con = machine_dir / f"console_{clip}.log"
            if con.exists():
                row.update(parse_console(con))

            dbg = machine_dir / f"debug_{clip}.log"
            if dbg.exists():
                row.update(parse_debug(dbg))
            summary.append(row)

            cli = machine_dir / f"client_{clip}.jsonl"
            if cli.exists():
                bundles = client_frames(cli)
                row["bundles_received"] = len(bundles)
                if (lat := rest_to_report(clip, bundles)):
                    latency.append({"machine": machine, "clip": clip, **lat})


        # Identity is keyed off the TRACKS files, not the profile logs: on a non-reference
        # machine the validation pass is unpaced tracks ONLY (no profile log), and its score is
        # the whole point — it is what proves the performance row ran correct code on correct
        # bytes. Hanging this off the profile loop silently skipped exactly that machine.
        for mot in sorted(machine_dir.glob("tracks_*.txt")):
            clip = mot.stem[len("tracks_"):]
            if (sc := score_tracks(clip, mot)):
                identity.append({"machine": machine, "clip": clip, **sc})

    # ONE FOLDER PER MACHINE, partitioned by each row's own machine column. The old tail wrote
    # every row under the LAST machine analysed — correct on a machine that only has its own
    # runs/ folder, silently wrong the moment a second machine's logs are copied here: both
    # machines' rows landed in one folder and the other folder was never written.
    print("wrote:")
    notes = {
        "steps": "what each step of a frame costs, in ms per frame. From the tracker's own "
                 "--profile. thread: input = getting the frame in - pipeline = the work that "
                 "must fit the frame budget - output = sending it out, on its own thread",
        "summary": "one line per clip per machine. *_ms are the three threads' totals in ms per "
                   "frame; capacity_fps is what the pipeline could sustain, achieved_fps what "
                   "the run did at real-time pacing; objects/captures/rested/moved are counted "
                   "from the debug log",
        "latency": "object at rest -> first message received BY A CLIENT, in ms. From the "
                   "ground truth's landing frame to the arrival of a bundle placing something "
                   "there. Both ends measured outside the tracker",
        "identity": "did it get the objects RIGHT, against the hand-made ground truth. "
                    "settle_success_pct: of every moment the GT says an object came to rest, "
                    "how many the tracker reported. idf1/mota/id_switches: scored with the "
                    "frames around a landing treated as don't-care, since a settle-based "
                    "tracker deliberately does not chase",
    }
    machines = sorted({r["machine"] for rows in (steps, summary, latency, identity) for r in rows})
    for m in machines:
        for name, rows in (("steps", steps), ("summary", summary),
                           ("latency", latency), ("identity", identity)):
            write(RESULTS / m / f"{name}.csv", [r for r in rows if r["machine"] == m],
                  notes[name])
    combine()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
