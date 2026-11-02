"""ReadyMade Tracker core: the data model + the settle pipeline + shared utils.

The core IS the algorithm — the ordered stages (observe · classify · identity ·
pose, plus the optional features / clusters / grouper) wired by ``pipeline.py``,
plus the single-writer ``Scene`` (a ``_field_writer`` conflict is a HARD ERROR),
the typed config, the thread runners, and shared utils (geometry / preprocess).
No dispatcher, no inter-stage events. The only plugins are I/O; they never reach
into the core or each other — they read the ``SceneSnapshot`` and own no shared
state.
"""
