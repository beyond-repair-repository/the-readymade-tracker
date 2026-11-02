"""The teardown watchdog must end the process even when a native call HOLDS THE GIL.

That is how pyuvc's ``Capture.close()`` hangs on macOS: libuvc's ``uvc_stream_stop`` waits for
cancelled USB transfers with the GIL held, so no Python code (Quit, Ctrl-C, a Python timer) can
run. ``ctypes.PyDLL`` calls a C function WITHOUT releasing the GIL, so ``sleep`` through it is an
exact stand-in. Run in a subprocess: the watchdog ``_exit``s the interpreter it lives in."""

import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(sys.platform == "win32", reason="libc sleep via PyDLL(None) is POSIX-only")
def test_watchdog_exits_a_process_wedged_in_native_code_holding_the_gil():
    code = (
        "import ctypes, readymade_tracker as rt\n"
        "rt._arm_shutdown_watchdog(1.0)\n"
        "ctypes.PyDLL(None).sleep(30)\n"            # GIL held for 30 s: nothing in Python can run
        "print('unreachable')\n"
    )
    t0 = time.perf_counter()
    r = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, timeout=25)
    took = time.perf_counter() - t0
    assert r.returncode == 1, (r.returncode, r.stderr[-500:])
    assert took < 15, f"watchdog fired late ({took:.1f} s)"
    assert "Timeout" in r.stderr                   # faulthandler's header, followed by the stacks
    assert "unreachable" not in r.stdout


_CLOSE_SCRIPT = (
    "import ctypes, sys\n"
    "sys.path.insert(0, {repo!r})\n"
    "import readymade_tracker as rt\n"
    "rt._CAMERA_CLOSE_TIMEOUT_S = 1.0\n"
    "class Backend:\n"
    "    def close(self):\n"
    "        {body}\n"
    "rt._close_backend_guarded(Backend(), is_camera=True, logdir='logs')\n"
    "print('returned')\n"
)


@pytest.mark.skipif(sys.platform == "win32", reason="libc sleep via PyDLL(None) is POSIX-only")
def test_hung_camera_close_ends_the_process_and_writes_the_stacks_to_logs(tmp_path):
    code = _CLOSE_SCRIPT.format(repo=str(REPO), body="ctypes.PyDLL(None).sleep(30)   # pyuvc-style hang, GIL held")
    t0 = time.perf_counter()
    r = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True, timeout=25)
    assert r.returncode == 1 and time.perf_counter() - t0 < 10, (r.returncode, r.stderr[-400:])
    assert "releasing the camera" in r.stdout and "returned" not in r.stdout
    assert "Timeout" not in r.stderr                       # the stacks go to the file, not the console
    files = list((tmp_path / "logs").glob("shutdown-hang_*.txt"))
    assert len(files) == 1
    text = files[0].read_text()
    assert "did not return" in text and "Timeout" in text and "close" in text   # names the hung call


def test_normal_camera_close_returns_and_leaves_no_hang_file(tmp_path):
    code = _CLOSE_SCRIPT.format(repo=str(REPO), body="pass")
    r = subprocess.run([sys.executable, "-c", code + "import time; time.sleep(1.5)\n"],
                       cwd=tmp_path, capture_output=True, text=True, timeout=25)
    assert r.returncode == 0 and "returned" in r.stdout, r.stderr[-400:]
    assert not list((tmp_path / "logs").glob("shutdown-hang_*.txt"))


def test_disarmed_watchdog_lets_a_normal_teardown_exit_cleanly():
    code = (
        "import time, readymade_tracker as rt\n"
        "rt._arm_shutdown_watchdog(1.0)\n"
        "time.sleep(0.2)\n"
        "rt._disarm_shutdown_watchdog()\n"
        "time.sleep(1.5)\n"                         # past the original deadline: must NOT fire
        "print('clean')\n"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, timeout=25)
    assert r.returncode == 0 and "clean" in r.stdout, r.stderr[-500:]
