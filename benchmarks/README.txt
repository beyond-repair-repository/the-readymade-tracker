HOW TO BENCHMARK, ON EACH OF THE THREE MACHINES
===============================================

One raw recording is both the performance workload and the correctness check:

    tests/videos/20260714-164508_raw.mp4    1920x1080, the original camera bytes
    benchmark.conf        PERFORMANCE - the full preprocess chain as deployed
                          (crop, lens undistort, flatfield; calibration in benchmarks/calib/)
    benchmark_gt.conf     VALIDATION - the annotated view (plain crop, no undistort),
                          scored against the untouched hand annotation

Every machine runs BOTH: the paced performance pass produces the table, the unpaced
validation pass proves the run executed correct code on correct bytes.


SETUP (once per machine - see INSTALL.txt "Install with a PINNED Python")
-------------------------------------------------------------------------
Same platform everywhere, or the table compares interpreters instead of hardware:

    python 3.14  -  numpy 2.3.5  -  OpenCV 5.0.0      (uv install, ~1 minute, no compiling)
    venv/bin/pip install -r requirements.txt -r benchmarks/requirements.txt

Check before measuring anything:

    venv/bin/python -V && venv/bin/python -c "import numpy,cv2;print(numpy.__version__,cv2.__version__)"


RUN
---
Close everything else - the paced pass measures whether the machine keeps up, and a busy
machine answers a different question.

    EVERY machine (~2.5 min):
        venv/bin/python benchmarks/run.py --name <MBA_M1 | LabPC_win11 | Pi5>
        venv/bin/python benchmarks/analyze.py

    REFERENCE machine only, the M1 (~5 min) - adds hover1 + hands2 and the latency pass;
    they describe the tracker, not the hardware, so once is enough:
        venv/bin/python benchmarks/run.py --name MBA_M1 --gt
        venv/bin/python benchmarks/analyze.py

    (Windows: venv\Scripts\python benchmarks\run.py --name LabPC_win11)


THE PASS CRITERION - read this before trusting any row
------------------------------------------------------
analyze.py writes identity.csv. The clip 20260714-164508 row MUST read:

    idf1 0.972x   0 switches   settle 100.0%

Same code, same bytes on every machine - so anything else means a broken setup (wrong
versions, edited conf, truncated copy), NOT a slower machine, and that machine's
performance row is void. Known tolerance: the LAST DIGIT of idf1 (and mota, in tandem)
varies with the OpenCV build's SIMD dispatch, not the OS - measured 0.9723 (M1 macOS,
Windows x86) and 0.9725 (Pi 5 Linux); onsets, switches and settle are identical
everywhere, and every value in that range passes.

Dropped frames are NOT a failure - they are the honest "did not keep up" signal, and they
appear in summary.csv as their own column.


COLLECT
-------
Each machine owns benchmarks/results/<name>/ (four small CSVs). Copy that folder to the
reference machine and run analyze.py there - it prints every machine present side by side.
The ~90 MB of logs in benchmarks/runs/<name>/ stay where they were produced.

    steps.csv     per-step ms per frame (input / pipeline / output threads)
    summary.csv   totals, capacity fps, DROPPED, achieved fps, python/numpy/cv2, scene counts
    latency.csv   object at rest -> first message at a real client (reference machine)
    identity.csv  the pass criterion above

If a machine's numbers look wrong, benchmarks/bench_threads.py sweeps the OpenCV thread
count through the real tracker (~2 min). All committed rows use [perf] cv2_threads = 2.
