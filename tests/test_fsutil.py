"""fsutil — a sudo run hands every file it creates back to the repo's user (own_to_dir_owner)."""

import os
from types import SimpleNamespace

from lib.fsutil import own_to_dir_owner


def test_owns_to_the_parent_dirs_owner_when_root(monkeypatch):
    calls = []
    monkeypatch.setattr(os, "geteuid", lambda: 0, raising=False)
    monkeypatch.setattr(os, "stat", lambda p: SimpleNamespace(st_uid=501, st_gid=20))
    monkeypatch.setattr(os, "chown", lambda p, u, g: calls.append((p, u, g)), raising=False)
    own_to_dir_owner("/repo/recordings/take.mp4")
    assert calls == [("/repo/recordings/take.mp4", 501, 20)]


def test_no_op_when_not_root(monkeypatch):
    calls = []
    monkeypatch.setattr(os, "geteuid", lambda: 501, raising=False)
    monkeypatch.setattr(os, "chown", lambda p, u, g: calls.append((p, u, g)), raising=False)
    own_to_dir_owner("/repo/recordings/take.mp4")
    assert calls == []


def test_never_raises(monkeypatch):
    monkeypatch.setattr(os, "geteuid", lambda: 0, raising=False)
    def boom(p): raise OSError("nope")
    monkeypatch.setattr(os, "stat", boom)
    own_to_dir_owner("/nowhere/x")                       # ownership is a courtesy, not a crash
