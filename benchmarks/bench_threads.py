"""bench_threads.py — what each thread group costs at each OpenCV thread count.

    venv/bin/python benchmarks/bench_threads.py            (macOS / Linux / Pi)
    venv\\Scripts\\python benchmarks\\bench_threads.py       (Windows)

Run it when a machine's numbers look wrong, or to find the right [perf] cv2_threads for a new
one. ~2 minutes.

IT MEASURES THE TRACKER, NOT A PROXY. Each thread count plays the same clip through the real
readymade_tracker.py with --profile, and the number compared is the tracker's own PIPELINE
TOTAL: what one frame's work actually cost. Nothing here times an OpenCV call of its own
choosing — a synthetic operation can scale completely differently from this pipeline, and the
whole point is that THIS pipeline behaves unlike the benchmark OpenCV was tuned for.

WHAT TO EXPECT
  A frame is a long chain of SMALL operations - countNonZero, bitwise_and, boundingRect,
  threshold - and OpenCV splits each of them across every core it can find. Waking N threads to
  hand each a sliver can cost more than the sliver, and that cost grows with the core count. So
  a many-core machine can be SLOWER at this than a laptop at the same clock, purely from
  coordinating threads that had almost nothing to do.

  MEASURED, idle machines, this script, the raw clip through benchmark.conf (input =
  decode + the full preprocess; pipeline = the tick; output = the OSC/TUIO threads):

                    input 1t -> 2t      pipeline 1t -> 2t -> max-t
    8-core M1        6.12 -> 2.87 ms     0.64 -> 0.77 -> 0.83 (8t)
    4-core Pi 5     25.50 -> 16.76 ms    2.27 -> 2.33 -> 2.57 (4t)
    32-thread i9     9.27 -> 6.97 ms     1.69 -> 1.63 -> 2.09 (32t)   (8P+16E, Windows)

  The same tension on all three: the PIPELINE prefers few threads (its operations are too
  small to split), the INPUT thread wants 2+ (undistort + flatfield are exactly the big
  operations threads help, and on the Pi that thread is the one near the 33.3 ms budget).
  2 buys the input drop for a ~0-0.15 ms pipeline penalty - which is why 2 is the shipped
  value. So READ ALL THREE COLUMNS against the 33.3 ms frame budget; there is deliberately
  no "fastest" verdict, because one column's winner is another's loss.

  The i9 row settles what the retired rig claimed (1.47x pipeline penalty at full thread
  count, degrading monotonically per doubling): measured, the penalty is 1.24x at 32t and
  NOT monotonic - flat through 4t, then the cliff at 8t+, consistent with the worker pool
  spilling onto the E-cores. Direction confirmed, magnitude halved, shape refuted.

  NOTE the sweep is UNPACED, so its absolute ms are not the paced benchmark row's: at replay
  speed the wall-clock settle dwells barely fire, so each frame does less settle work. Thread
  counts compare against each other; the paced run is the deployment number.

WHAT IT CANNOT SEE, so check by hand if the answer looks wrong:
  * the Windows power plan ("Balanced" holds the clock down - use High performance)
  * whether the process was parked on efficiency cores (Task Manager > Details > Efficiency mode)
  * that nothing in this pipeline scales with core COUNT at all: one frame is a dependent chain
    on one thread, and the other cores only ever help INSIDE a single call. A big machine losing
    to a laptop here is not a fault.

CHANGING THE VALUE INVALIDATES A COMPARISON. Every machine's committed row was measured at
cv2_threads = 2; if this says a machine prefers something else, that is a finding to report, not
a setting to change before re-running one row.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
REPO = BENCH_DIR.parent
sys.path.insert(0, str(BENCH_DIR))

CONF = REPO / "benchmark.conf"
LOGS = REPO / "logs"
VIDEOS = REPO / "tests" / "videos"
# The RAW recording through benchmark.conf - the same workload as the performance pass, so
# the sweep answers the question the benchmark actually asks. (hands2 was the old default;
# it is 1024x1024 already, and benchmark.conf now crops at (460,20) - out of bounds for it.)
DEFAULT_CLIP, DEFAULT_FRAMES = "20260714-164508_raw", 600


def groups_ms(n: int, clip: str, frames: int, conf: Path = None) -> dict:
    """Play the clip through the REAL tracker at n threads; return {group: TOTAL ms/frame}.

    All THREE threads are reported, because the knob does not reach them equally: the pipeline
    and preprocess are OpenCV and respond to it, while building OSC and TUIO messages is pure
    Python and cannot care. Seeing output stay flat while pipeline moves is how you know the
    measurement is doing what it claims.

    The ONLY difference between runs is --cv2-threads, which overrides [perf] cv2_threads for
    that process. Same conf, same clip, same everything else - so the difference in the tick is
    the thread count and nothing else.
    """
    from analyze import parse_profile
    conf = conf or CONF
    before = {p.name for p in LOGS.glob("profile_*.log")} if LOGS.exists() else set()
    # --pull: demand-driven, so every frame is processed and none dropped - the tick cost IS the
    # measurement. --fps 1000: unpaced, or pacing would just add idle waiting to `grab`. The
    # outputs stay ON: they are part of what a deployment runs, and their flatness is the control.
    clip_path = Path(clip) if Path(clip).exists() else VIDEOS / f"{clip}.mp4"
    r = subprocess.run([sys.executable, str(REPO / "readymade_tracker.py"), str(conf),
                        "--source", "video", str(clip_path),
                        "--cv2-threads", str(n),
                        "--fps", "1000", "--pull", "--max-frames", str(frames), "--profile",
                        "--no-webgui", "--no-mjpeg", "--no-oscin"],
                       cwd=REPO, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"    ! tracker exited {r.returncode} at {n} threads:\n{r.stderr[-400:]}",
              file=sys.stderr)
        return {}
    fresh = [p for p in LOGS.glob(f"profile_*_{clip_path.stem}.log") if p.name not in before]
    if not fresh:
        print(f"    ! no profile log for {n} threads", file=sys.stderr)
        return {}
    log = max(fresh, key=lambda p: p.stat().st_mtime)
    prof = parse_profile(log)
    log.unlink(missing_ok=True)                # scratch: this run is a probe, not evidence
    g = prof.get("groups", {}) if prof else {}
    return {k: g[k]["TOTAL"][0] for k in ("input", "pipeline", "output")
            if k in g and "TOTAL" in g[k]}


def main() -> int:
    ap = argparse.ArgumentParser(description="Per-thread-group cost at each OpenCV thread count.")
    ap.add_argument("--clip", default=DEFAULT_CLIP,
                    help="a name in tests/videos/, or a path to any .mp4 (e.g. a RAW recording)")
    ap.add_argument("--conf", default=None,
                    help="config to sweep with (default benchmark.conf: the full preprocess "
                         "chain on the raw clip - the same workload the performance pass times)")
    ap.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    ap.add_argument("--reps", type=int, default=2, help="play-throughs per thread count (default 2)")
    args = ap.parse_args()

    import cv2
    import platform
    cores = os.cpu_count() or 1
    print("== per-thread-group cost at each OpenCV thread count ==")
    print(f"  {platform.node().split('.')[0]} | {platform.processor() or platform.machine()} | "
          f"{platform.system()} | {cores} cores | OpenCV {cv2.__version__}")
    print(f"  {args.clip}, {args.frames} frames through the real tracker, x{args.reps}"
          f"{'' if args.conf is None else f' - conf {args.conf}'}")
    print("  Close everything else - a busy machine hides the effect.\n")

    conf = Path(args.conf) if args.conf else CONF
    if not conf.exists():
        raise SystemExit(f"no such config: {conf}")
    counts = sorted({1, 2, 4, 8, 16, cores} & set(range(1, cores + 1)))
    rows: dict[int, dict] = {}
    for n in counts:
        runs = [g for _ in range(args.reps)
                if (g := groups_ms(n, args.clip, args.frames, conf))]
        if runs:                               # the FLOOR per group: least-contended play-through
            rows[n] = {k: min(r[k] for r in runs if k in r)
                       for k in ("input", "pipeline", "output") if any(k in r for r in runs)}
    if not rows:
        print("  nothing measured - did the tracker run?", file=sys.stderr)
        return 1
    # No verdict. The three threads pull in different directions (the pipeline prefers few,
    # the input thread prefers more), so a single "fastest" would be judged on one column and
    # recommend the wrong number on an input-bound machine - it did, on the Pi. The table is
    # the result; the docstring says how to read it.
    print(f"  {'threads':>8} {'input':>9} {'pipeline':>10} {'output':>9}   (ms/frame, best of {args.reps} runs)")
    for n, v in rows.items():
        cells = "".join(f"{v[k]:>9.3f}" if k in v else f"{'-':>9}"
                        for k in ("input", "pipeline", "output"))
        print(f"  {n:>8}{cells[:9]}{cells[9:19]}{cells[19:]}")
    print("\n  input = decode + preprocess; pipeline = the tick; output = the OSC/TUIO threads.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
