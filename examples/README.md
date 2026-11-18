# Example clients

Three minimal clients — one per interaction paradigm, one per platform.

- [`puredata/readymade_knob.pd`](puredata/README.md) — **token + constraint** (vanilla Pd, no externals)
- [`supercollider/readymade_pad.scd`](supercollider/README.md) — **malleable tangibles** (vanilla SC)
- [`vvvv/readymade-demo_2.vl`](vvvv/README.md) — **perceptual-property physics** (vvvv gamma)

Each example runs against a live camera. `camera1.conf` is written by the calibration tool
(`tools/calibrate_camera_web.py`). To try a client without a camera, start the tracker on one
of the clips in `tests/videos/` as in the main README; the objects in those clips are not the
ones each example was designed for.
