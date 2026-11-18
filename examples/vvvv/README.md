# vvvv example — perceptual physics with skins

**What it does:** measured properties BECOME physical material. Every settled
object spawns a 2D physics body — `circularity³` → bounciness (a disc is a
superball, a crumple thuds), `spikiness` → friction, `size` → mass — and each
body wears **the photograph of its own object**: a sprite cut from the live
MJPEG stream at the moment it settled. Drop a paper crumb and watch it bounce
as itself.

Watch it run: [`demo.mp4`](demo.mp4), a short captioned screen recording.

## Install

- **vvvv gamma** (tested with 2025.7): <https://visualprogramming.net/>
- Get `VL.AetherPhysics` (the Aether.Physics2D wrapper) from
  <https://github.com/elektromeier/VL.AetherPhysics> and fix the document dependency
  path in `readymade-demo_2.vl` if yours differs. The zip is not bundled here.
- The C# project (`csharp/RmtWorld/`) builds on first patch open; its
  `SkiaSharp` package version **must match gamma's own** (2.88.8 for
  gamma 2025.7) or the image pins won't connect.

## Run

Tracker on the table's machine, replaying the recorded clip `physics.mp4` (a tracker
recording, so it replays without preprocessing) — native stream + contour + the MJPEG server;
put the vvvv PC's IP in `destinations`:

    venv/bin/python readymade_tracker.py --source video examples/vvvv/physics.mp4 --loop --no-preprocess \
        --oscout --mjpeg \
        --set oscout send "all, contour" \
        --set oscout destinations "127.0.0.1:3400, <VVVV-PC-IP>:3400" \
        --set efd bins 48

For a live table, replace `--source video examples/vvvv/physics.mp4 --loop --no-preprocess` with
`--source camera camera1.conf` (on macOS, prefix the command with `sudo`).

Then open `examples/vvvv/readymade-demo_2.vl` in gamma and set the
`MjpegClient` host pin to the tracker machine's IP (port 8080) — Create-group
pins default to empty, not to the C# defaults. A freshly settled object's
sprite appears exactly on the object in the video, then physics takes it.

Builder notes (hard-won): ids are identities, not slots — on a looped clip the
same objects re-acquire the SAME ids every pass; join body → object by id
(`RmtWorld.TryGet`), never by spread index. VL rotation pins are in CYCLES —
divide radians by 2π. Never park a background thread in hot-reloaded C#.
