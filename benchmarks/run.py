"""run.py — the benchmark: ONE raw clip for performance, scored ground truth for correctness.

    venv/bin/python benchmarks/run.py --name <machine>          every machine   (~2.5 min)
    venv/bin/python benchmarks/run.py --name MBA_M1 --gt        reference machine (~5 min)

Then: `venv/bin/python benchmarks/analyze.py` turns what this wrote into the tables.

WHAT RUNS WHERE
  EVERY MACHINE — two passes of the SAME recording (tests/videos/20260714-164508_table.mp4,
  1920x1080, the camera frame with everything outside the table masked):

    PERFORMANCE   benchmark.conf — the FULL preprocess chain as deployed (flip, crop, lens
                  undistort, flatfield; calibration committed under benchmarks/calib/), paced
                  at real-time 30 fps, with a real TUIO+OSC client consuming both output
                  streams. This is the performance table: per-step ms, thread totals, dropped
                  frames, achieved fps.
    VALIDATION    benchmark_gt.conf — the ANNOTATED view of the same file (plain crop, no
                  undistort), unpaced + --pull so every frame is processed, tracks exported
                  and scored against the untouched hand-made .MOT. The run PASSES when it
                  reads 9 ids / idf1 ~0.972 / 0 switches / 100% settle — same code, same
                  bytes on every machine, so anything else means a broken setup, not a slower
                  machine. (Known tolerance: idf1's last digit tracks the OpenCV build's SIMD
                  dispatch, not the OS — measured 0.9723 on M1 macOS AND Windows x86, 0.9725
                  on Pi 5 Linux; all pass.)

  REFERENCE MACHINE ONLY (--gt) — adds the two other annotated clips, hover1 and hands2.
  They were recorded THROUGH the preprocess (already flipped, cropped, undistorted,
  flatfielded), so they run with --no-preprocess. Each gets the paced+client pass (their
  latency numbers) and the unpaced tracks pass (their identity scores); the big clip's
  validation pass is upgraded to paced+client too, so the reference machine also owns the
  38-onset latency median. These describe the TRACKER, not the hardware — once is enough.

WHY THE VALIDATION IS A DIFFERENT VIEW THAN THE PERFORMANCE PASS
  The ground truth was annotated on the unwarped crop. Lens undistortion moves silhouettes by
  a sub-pixel amount, which flips the sign of near-zero Hu moments inside the reid descriptor
  and can cost an identity (measured: cosine 0.95 -> 0.53 on the same object). So correctness
  is scored in the annotated space, and the full chain is what gets TIMED. The timing is
  unaffected either way.

MECHANICS
  Nothing here re-implements the tracker or reaches inside it — every number comes from the
  tracker's own --profile / --debug / --mot-export and from what a real UDP client received.
  Paced passes measure "does it keep up" (drops are the honest signal); the tracks pass is
  --pull because a paced run drops frames exactly AT settles, understating identity by a
  third (measured). Whole clips, never a prefix: the GT counts onsets over the whole file.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
REPO = BENCH_DIR.parent
sys.path.insert(0, str(BENCH_DIR))

CONF = REPO / "benchmark.conf"           # PERFORMANCE: full chain, as deployed
GT_CONF = REPO / "benchmark_gt.conf"     # VALIDATION:  the annotated view (crop only)
LOGS = REPO / "logs"
RUNS = BENCH_DIR / "runs"                # one folder per machine, holding that machine's logs
VIDEOS = REPO / "tests" / "videos"

RAW = "20260714-164508_table"            # the recording: 1920x1080, table only (rest masked)
GT_LABEL = "20260714-164508"             # the same file through the annotated view -> its .MOT
RAW_FRAMES = 2289                        # 9 objects, 38 onsets
GT_CLIPS = [("hover1", 425),             # 8 objects, nothing put down or moved: the usual case
            ("hands2", 447)]             # 8 objects, 1 put down + 4 moved


def machine() -> str:
    import platform
    return platform.node().split(".")[0]


def write_env(out: Path) -> dict:
    """Record what this row was measured WITH, next to the logs.

    numpy and OpenCV are the tracker's entire dependency surface and CPython is the interpreter
    running it, so all three are part of the result rather than context for it.
    """
    import json
    import platform
    import cv2
    import numpy
    env = {"python": platform.python_version(), "numpy": numpy.__version__,
           "cv2": cv2.__version__, "os": platform.platform(),
           "cpu": platform.processor() or platform.machine(), "cores": os.cpu_count()}
    (out / "env.json").write_text(json.dumps(env, indent=1))
    return env


def run_clip(label: str, src: str, frames: int, conf: Path, out: Path, *,
             monitor: bool, no_preprocess: bool, paced: bool, tracks: bool) -> None:
    """One benchmark entry. ``label`` names every artifact (and, for a scored clip, selects its
    ground truth: tracks_<label>.txt is scored against <label>.mp4.MOT).

    The paced pass is "does it keep up": real time, latest-wins slot, drops counted, a real
    client stamping arrivals. The tracks pass is --pull, because a paced run drops frames
    exactly AT settles — scoring identity off it understates the tracker by a third (measured).
    """
    base = [sys.executable, str(REPO / "readymade_tracker.py"), str(conf),
            "--source", "video", str(VIDEOS / src), "--max-frames", str(frames)]
    if no_preprocess:                    # hover1/hands2 were RECORDED through the preprocess —
        base.append("--no-preprocess")   # already flipped/cropped/undistorted/flatfielded

    if paced:
        mon = None
        if monitor:
            cap = out / f"client_{label}.jsonl"
            # BOTH streams get a real consumer: --show-clusters makes the monitor bind a second
            # socket for [oscout], and both sockets fold through the same recorder.
            mon = subprocess.Popen(
                [sys.executable, str(REPO / "tools" / "tuio2_monitor.py"), "--text",
                 "--record", str(cap), "--show-clusters", "--osc-port", "3400"],
                cwd=REPO, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(1.0)              # let it bind before the tracker starts sending
        print(f"  {label}: {frames} frames at 30 fps (~{frames / 30:.0f}s) ", end="", flush=True)
        t0 = time.perf_counter()
        r = subprocess.run(base + ["--profile", "--debug"], cwd=REPO,
                           capture_output=True, text=True)
        print(f"done in {time.perf_counter() - t0:.0f}s", end="", flush=True)
        # KEEP THE CONSOLE OUTPUT: frames DROPPED and achieved fps exist only there.
        (out / f"console_{label}.log").write_text(r.stdout + r.stderr, encoding="utf-8")
        if r.returncode != 0:
            print(f"\n    ! tracker exited {r.returncode}:\n{r.stderr[-800:]}", file=sys.stderr)
        if mon is not None:
            mon.terminate()
            try:
                mon.wait(timeout=5)
            except subprocess.TimeoutExpired:
                mon.kill()
            if not cap.exists() or cap.stat().st_size == 0:
                print(f"\n    ! the client recorded NOTHING for {label} - it could not bind "
                      f"port 3333.\n      Something else is on it (often a monitor left over "
                      f"from an interrupted run):\n        pkill -f tuio2_monitor.py",
                      file=sys.stderr)
        # the tracker names its logs by timestamp + source; adopt THIS run's under the label
        stem = Path(src).stem
        for kind in ("profile", "debug"):
            found = sorted(LOGS.glob(f"{kind}_*_{stem}.log"), key=lambda p: p.stat().st_mtime)
            if found:
                shutil.move(str(found[-1]), out / f"{kind}_{label}.log")
            else:
                print(f"    ! no {kind} log for {label} — did the tracker start?", file=sys.stderr)

    if tracks:
        print(f"{' ' if paced else f'  {label}:'} tracks (unpaced, every frame)", flush=True)
        subprocess.run(base + ["--fps", "1000", "--pull",
                               "--mot-export", str(out / f"tracks_{label}.txt")],
                       cwd=REPO, capture_output=True, text=True)
        for junk in LOGS.glob(f"*_{Path(src).stem}.log"):   # the unpaced pass writes no evidence
            junk.unlink(missing_ok=True)
    elif not paced:
        print(f"  {label}: nothing to do", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="One clip for performance; ground truth for correctness.")
    ap.add_argument("--name", default=None,
                    help="what to call this machine in every table (MBA_M1, LabPC_win11, Pi5). "
                         "Default: the hostname, which a paper reader rarely recognises")
    ap.add_argument("--gt", action="store_true",
                    help="reference machine: add hover1 + hands2 and the paced latency pass. "
                         "These describe the tracker, not the hardware — run them ONCE, on the M1")
    ap.add_argument("--no-client", action="store_true",
                    help="skip the TUIO client (then there is no arrival-latency data)")
    args = ap.parse_args()

    for f in (CONF, GT_CONF):
        if not f.exists():
            print(f"missing {f} — the benchmark configs are committed files", file=sys.stderr)
            return 1
    out = RUNS / (args.name or machine())
    if out.exists():
        shutil.rmtree(out)               # a re-run REPLACES this machine's logs, never mixes them
    out.mkdir(parents=True)
    LOGS.mkdir(exist_ok=True)

    env = write_env(out)
    print(f"== {args.name or machine()} - performance + validation{' + GT suite' if args.gt else ''} ==")
    print(f"   python {env['python']} - numpy {env['numpy']} - OpenCV {env['cv2']} "
          f"- {env['cores']} cores")
    print(f"   logs -> {out.relative_to(REPO)}")
    print("   Close everything else: the paced pass measures whether the machine keeps up.\n")

    # 1. PERFORMANCE — the full chain, paced, client on. The number the paper is about.
    run_clip(RAW, f"{RAW}.mp4", RAW_FRAMES, CONF, out,
             monitor=not args.no_client, no_preprocess=False, paced=True, tracks=False)
    # 2. VALIDATION — same file, annotated view. Paced too on the reference machine (latency).
    run_clip(GT_LABEL, f"{RAW}.mp4", RAW_FRAMES, GT_CONF, out,
             monitor=args.gt and not args.no_client, no_preprocess=False,
             paced=args.gt, tracks=True)
    # 3. GT SUITE (reference only) — the other two annotated clips, recorded through the
    #    preprocess, so they run with it off.
    if args.gt:
        for clip, frames in GT_CLIPS:
            run_clip(clip, f"{clip}.mp4", frames, GT_CONF, out,
                     monitor=not args.no_client, no_preprocess=True, paced=True, tracks=True)

    print(f"\n   {len(list(out.iterdir()))} files written.")
    print("   Now: venv/bin/python benchmarks/analyze.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
