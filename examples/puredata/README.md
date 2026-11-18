# Pure Data example — `readymade_knob.pd`

**What it does:** token + constraint — a physical object IS a knob, and WHERE it
rests decides WHAT it turns. A cup on the left half of the table moves a slider
with its angle (up = 50 %, right → 100 %, left → 0 %); a cup on the right half
paints the same slider (angle → hue). Values commit at settle; ids don't matter —
the slot is the identity.

> The demo clip `two_cups.mp4` is not in the repository yet. Until it is, use a live camera or a
> clip from `tests/videos/` (run as in the main README, without `--no-preprocess`).

## Install

Vanilla Pure Data ≥ 0.46 (no externals; tested with 0.56):
<https://puredata.info/downloads> — macOS/Windows installers;
Linux / Raspberry Pi: `sudo apt install puredata`.

## Run

Tracker first (TUIO 2.0 on :3333 is on by default; the demo clip is a tracker
recording, so replay without preprocessing):

    venv/bin/python readymade_tracker.py --source video examples/puredata/two_cups.mp4 --loop --no-preprocess

Then open `examples/puredata/readymade_knob.pd` in Pd. The slider jumps when the
left cup settles and recolors when the right one does — one committed step per
settle, never a glide.

Note: on load the patch asks the tracker for the timetag dialect vanilla
`[oscparse]` can read (`/rmt/set tuio2 timetag_mode ints`, port 3401). If you
start or restart the tracker AFTER opening the patch, click the
`[list tuio2 timetag_mode ints(` message once.
