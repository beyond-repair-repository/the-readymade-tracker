"""``fsutil`` — the shared give-files-back-when-root helper.

On macOS the pyuvc camera path runs the tracker under ``sudo`` (claiming the USB device needs it),
and every file root creates — recordings, logs, a saved conf, a captured flatfield, a MOT export —
would stay root-owned, locking the NEXT plain run out of its own output dirs (the classic symptom:
"[rec] cannot open a writer"). ``own_to_dir_owner`` hands a freshly-created path back to whoever
owns its parent directory — the user whose repo checkout this is. A no-op when not running as root
(and on non-POSIX); it never raises — ownership is a courtesy, not a requirement."""

from __future__ import annotations

import os


def own_to_dir_owner(path: str) -> None:
    """chown ``path`` to its parent directory's owner — no-op unless running as root (POSIX)."""
    try:
        if not hasattr(os, "geteuid") or os.geteuid() != 0:
            return
        d = os.stat(os.path.dirname(os.path.abspath(str(path))) or ".")
        os.chown(str(path), d.st_uid, d.st_gid)
    except OSError:
        pass
