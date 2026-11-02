# SuperCollider example — `readymade_pad.scd`

**What it does:** malleable material as sound. One settled object = one pad
voice: the shape's EFD spectrum IS the sound's spectrum (one sine partial per
surviving harmonic), `size` → pitch (quantized to a pentatonic degree — big =
LOW, crumple the sheet tighter and it rises), `x` → pan, and the contour's
unwrapped radial scan breathes the voice as a slow self-AM (5 s per revolution
of the silhouette). Push crumbs together and the tracker fuses their voices
into one deeper compound.

## Install

SuperCollider (tested with 3.14): <https://supercollider.github.io/downloads>
— macOS/Windows installers; Linux / Raspberry Pi: `sudo apt install supercollider`.
Vanilla only, no quarks.

## Run

Tracker first — the native stream plus the contour opt-in (the radial scan is
cut from it; without the flag the example still runs, just without the breath):

    venv/bin/python readymade_tracker.py --source video examples/supercollider/crumbs.mp4 --loop --no-preprocess --oscout --set oscout send "all, contour"

Then open `examples/supercollider/readymade_pad.scd` in the SC IDE and evaluate
the block (or `sclang examples/supercollider/readymade_pad.scd`). First check
`s.options.outDevice` near the top — it pins the rig's output; change it to
yours (list devices with `ServerOptions.devices`).

Live, over the control port (oscin, :3401):

    /rmt/set grouper max_objects 3       # aggressive fusion: push crumbs together, hear voices merge
    /rmt/set efd detail 0.2              # simplify every shape spectrum -> purer tones
