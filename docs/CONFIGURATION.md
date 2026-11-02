# The tuning surface — every group and knob, five questions each

This document is organized exactly as the webgui's tabs — INPUT · PREPROCESS · PROCESS · OUTPUT (· CONTROL) — section by section, in
pipeline order. Values in brackets are (type, range, default, live/restart) as declared in the code (`PARAMS`, 112 knobs at v0.32.88).
Live knobs apply on the next frame — including while PAUSED: a live set re-emits the frozen frame through one pipeline pass, so the operator sees the effect immediately (the same one-shot a paused manual trigger uses).

A per-section `enabled` switch exists for every optional stage/plugin (reid, features, clusters, grouper, tuio2, mjpeg, oscout, oscin, rec, webgui): presence of the `[section]` in the conf is the manifest, `enabled = false` keeps it loaded but inert — the core pipeline and input cannot be disabled.

---

# TAB: INPUT

## `[input]` — the frame source (video backend)
**Why:** the tracker must run identically off a live camera and off recorded footage, so the source is a swappable backend behind one interface.
**What:** selects and configures what supplies frames — a video file for development/replay, the camera for the installation.
**Impact:** everything downstream is source-agnostic; deterministic clip replay is what makes the whole evaluation reproducible.
**How:** a backend factory (`video`/`camera`) exposing `grab/seek/close` plus a transport (pause/step/seek) the controls drive.
**Limits:** one source at a time; switching source or file needs a restart.

- **`source`** (str: video|camera, restart) — **Why:** one conf key must pick the backend. **What:** chooses video-file vs camera capture. **Impact:** decides which section ([input] vs [camera]) configures capture. **How:** keys a backend-factory map at build. **Limits:** restart to switch.
- **`location`** (str, restart) — **Why:** the clip path must live in conf, not code. **What:** the video file to play. **Impact:** selects the material every stage sees. **How:** handed to `cv2.VideoCapture`. **Limits:** video mode only.
- **`fps`** (int ≥0, restart) — **Why:** replay must run either real-time (as at the table) or unpaced (for scoring). **What:** playback-rate override; 0 = the clip's native rate. **Impact:** real-time pacing vs maximum-speed batch runs (the benchmarks use unpaced). **How:** a frame-release governor in the grab loop. **Limits:** pacing only — it never drops or duplicates frames.
- **`loop`** (bool, restart) — **Why:** unattended demos need endless footage. **What:** rewinds and replays at end-of-stream. **Impact:** turns a clip into an infinite source. **How:** seek-to-0 on EOF instead of StopIteration. **Limits:** objects persist across the wrap, so identities may re-link across the seam.
- **`buffer_frames`** (int ≥2, default 60, restart) — **Why:** cv2 seeks land on keyframes, so a one-frame step back would be inexact. **What:** a ring of the most recent decoded frames for exact back-stepping. **Impact:** frame-accurate reverse stepping in the webgui transport. **How:** replays from the ring instead of seeking. **Limits:** memory proportional to ring size; back-steps beyond it fall back to a coarse seek.

## `[camera]` — live capture
**Why:** installation lighting is controlled, so the camera must be *lockable* — auto-anything makes the whole frame breathe and the diff stage sees it as change.
**What:** selects the device and pins its exposure/gain/color controls.
**Impact:** the noise floor and stability of every downstream threshold are set here — manual, locked capture is the first and cheapest noise mitigation.
**How:** pyuvc on macOS (full UVC control) or cv2 `CAP_PROP_*` elsewhere, chosen by `backend=auto`.
**Limits:** which controls actually apply depends on the driver/backend; units are device-native, not physical.

- **`device`** (str, restart) — **Why:** machines enumerate cameras differently. **What:** capture index, /dev path, or UVC name/UID. **Impact:** which physical camera feeds the tracker. **How:** resolved by the chosen backend. **Limits:** hot-plug re-enumeration needs a restart.
- **`width` / `height` / `fps`** (int ≥0, restart) — **Why:** the mode must be pinned, not driver-chosen. **What:** requested capture mode; 0 = driver default. **Impact:** resolution sets the px scale every threshold/distance knob lives in. **How:** requested at open; the driver may negotiate. **Limits:** unsupported modes silently fall back — verify in the webgui HUD.
- **`backend`** (str: auto|uvc|cv2, restart) — **Why:** macOS cv2 cannot set exposure reliably. **What:** picks the capture stack. **Impact:** decides whether the manual controls below actually work. **How:** auto = pyuvc if importable on macOS, else cv2. **Limits:** pyuvc is a macOS-only optional extra.
- **`auto_exposure`** (bool, live) — **Why:** auto-exposure fights the diff stage. **What:** toggles camera AE; off makes `exposure` apply. **Impact:** the single most important stability switch on a live rig. **How:** UVC/CAP_PROP write-through. **Limits:** some drivers ignore it.
- **`exposure`** (int 0–5000, live) — **Why:** exposure fixes the signal level. **What:** manual exposure time, device units. **Impact:** with light level, sets contrast vs motion blur. **How:** applied when AE is off. **Limits:** device-native units; range varies per camera.
- **`gain`** (int 0–255, live) — **Why:** gain is the sensor-noise dial. **What:** analog/digital sensor gain. **Impact:** high gain is the main source of the speckle the detection thresholds then fight — prefer more light, less gain. **How:** UVC/CAP_PROP. **Limits:** device-native scale.
- **`brightness` / `contrast` / `saturation` / `gamma`** (int, live) — **Why:** last-resort image shaping when optics/light can't change. **What:** the driver's picture controls. **Impact:** shift the gray histogram the fg threshold cuts. **How:** UVC/CAP_PROP write-through. **Limits:** nonlinear, device-specific; prefer fixing light.
- **`white_balance`** (int 2000–8000 K, live) — **Why:** auto-WB drifts hue, which the reid saturation histogram would see as appearance change. **What:** manual color temperature. **Impact:** keeps the appearance descriptor stable over hours. **How:** Kelvin write-through when supported. **Limits:** driver support varies.

## `[mcu]` — pedal / board control input
**Why:** a performer needs hands-free triggers (a scan pedal) without OS dependencies.
**What:** reads a serial microcontroller and injects control commands (e.g. `/rmt/trigger`).
**Impact:** manual re-scan and transport control from hardware; zero effect when absent.
**How:** dependency-free POSIX serial (`os.open` + `termios`); pyserial only on Windows.
**Limits:** the board's firmware defines the protocol; baud must match.

- **`enabled`** (bool, restart) — the master switch for the serial input.
- **`port`** (str, restart) — **What:** the tty/COM device to read; empty = disabled. **Limits:** device naming is per-OS.
- **`baud`** (int 300–2M, default 115200, restart) — **What:** line speed. **Limits:** must equal the firmware's rate or reads are garbage.

---

# TAB: PREPROCESS

## `[preprocess]` — geometric + photometric conditioning
**Why:** the pipeline wants a stable, table-aligned, evenly-lit gray frame; everything correctable once belongs before perception.
**What:** lens undistortion, mirroring, rectangular crop to the table ROI, a circular mask, and dark-frame (flatfield) subtraction, in one fixed-order pass; the master switch off reduces it to BGR→gray.
**Impact:** defines the coordinate frame every knob's pixels live in (crop changes frame size — restart) and removes fixed-pattern light/vignette the thresholds would otherwise misread as objects.
**How:** cv2 remap/flip/slice/mask/subtract chained on the input thread, cached artifacts (lens YAML, dark frame) loaded once.
**Limits:** all corrections are static calibrations — nothing adapts at runtime; test clips ship pre-conditioned so `preprocess=false` replays them correctly.

- **`preprocess`** (bool, restart) — **Why:** conditioned recordings must replay without double-processing. **What:** master switch; off = BGR→gray only. **Impact:** on live rigs on, on conditioned clips off. **How:** skips every sub-stage. **Limits:** restart (frame size may change).
- **`lens_enabled` / `lens_path`** (bool/str, live) — **Why:** wide lenses bend the table's geometry. **What:** undistort via a calibration YAML (camera matrix + distortion). **Impact:** straightens positions so px distances mean the same everywhere. **How:** a precomputed `cv2.remap`. **Limits:** needs the offline calibration (`calibrate_camera.py`); wrong files distort worse.
- **`flip`** (str: none|x|y|xy, live) — **Why:** mirror/projection rigs invert axes. **What:** mirrors the frame. **Impact:** aligns screen axes with the physical table. **How:** `cv2.flip`; size-preserving, hence live. **Limits:** none.
- **`crop_enabled` / `crop_top_x` / `crop_top_y` / `crop_width` / `crop_height`** (bool/int, restart) — **Why:** the sensor sees more than the table. **What:** rectangular ROI cut (0 width/height = to the edge). **Impact:** defines the 1024² working frame; off-table motion vanishes from every stage. **How:** an array slice on the input thread. **Limits:** changes frame size → restart; all px-denominated knobs implicitly assume the cropped frame.
- **`mask_enabled` / `mask_cx` / `mask_cy` / `mask_radius`** (bool/int, live) — **Why:** round tables have corners the crop can't remove. **What:** zeroes gray outside a circle. **Impact:** kills border intrusions (knees, cables) at the source; pairs with observe's rim gate. **How:** a multiply by a cached circle mask. **Limits:** the effect is invisible in the BGR views (the webgui draws the circle as an overlay instead).
- **`flatfield_enabled` / `flatfield_path`** (bool/str, live) — **Why:** vignetting and hot pixels are fixed-pattern noise no global threshold can absorb. **What:** subtracts a captured dark/empty-table frame. **Impact:** evens the field so ONE `fg_threshold` works everywhere. **How:** per-pixel subtraction of the stored image/`.npy`. **Limits:** stale dark frames (moved lamp) reintroduce bias — recapture after any light change.

---

# TAB: PROCESS

## `[observe]` (panel: CONTOURS) — per-frame perception
**Why:** the settle paradigm forbids per-frame inference, so each frame gets only a cheap segmentation-and-association pass.
**What:** builds the object-foreground and motion masks, extracts one external contour per blob, associates blobs to known objects across four passes, and mints unmatched stable blobs.
**Impact:** every downstream stage sees only what observe segments and associates — its thresholds are the system's sensitivity-vs-noise dial and the first identity gate.
**How:** global brightness threshold (bright objects on dark felt, no dilation) + frame-diff motion mask + `findContours(RETR_EXTERNAL)` + overlap/footprint/relocation/size association passes.
**Limits:** assumes bright-on-dark contrast; two touching objects merge into one blob; a moving blob is deliberately ignored until it rests.

- **`fg_threshold`** (int 0–255, default 40, live) — **Why:** objects must separate from felt without background modelling. **What:** the gray cut above which a pixel is foreground. **Impact:** the primary sensitivity dial — too low mints phantom noise objects, too high loses dim ones. **How:** `cv2.threshold(gray, v, 255, BINARY)` on the (optionally denoised) detection gray. **Limits:** one global cut presumes even illumination (flatfield/mask compensate, not this).
- **`blur`** (float 0–1, default 0, live) — **Why:** sensor noise flickers pixels past the two cuts, and the frame-diff doubles it. **What:** Gaussian denoise (σ up to ~3 px) of the *detection path only*. **Impact:** melts speckle from the fg/motion masks and lets regions settle, without touching the frame reid's textures come from. **How:** blurs a local copy feeding only the two thresholds; 0 returns the original uncopied. **Limits:** also attenuates very small blobs' peaks — compensate with `fg_threshold` if tiny objects flicker out.
- **`min_area`** (int ≥1, default 300, live) — **Why:** dust and speckle survive any threshold. **What:** the smallest blob area (px²) kept. **Impact:** the speck floor; also the floor that keeps centroid math valid. **How:** area test per contour. **Limits:** an object smaller than it is invisible.
- **`max_area`** (int ≥0, default 0=off, live) — **Why:** a stopped forearm is a huge stable blob. **What:** rejects *minting* blobs larger than this. **Impact:** arms/bags can't become objects. **How:** area gate on the mint path only (existing objects still track). **Limits:** must exceed the largest legitimate object.
- **`k`** (float 0–1, default 0.05, live) — **Why:** partial update needs one scale for "changed enough to matter". **What:** the change-gate: minimum contour change as a fraction of object area. **Impact:** the ONE change scale — governs when a region counts as disturbed vs still. **How:** per-object residual/missing fractions compared against it. **Limits:** very low values make breathing shadows look like change.
- **`reassoc_dist`** (float px, default 64, live) — **Why:** a briefly-lost blob (flicker, short occlusion) should re-join its object. **What:** max centroid distance for re-associating a lost object's blob. **Impact:** bridges 1–2-frame dropouts without opening the door to swaps. **How:** nearest-centroid gate in the association pass. **Limits:** deliberately short — long reaches belong to `relocate_dist`.
- **`relocate_dist`** (float px, default 160, live) — **Why:** a picked-up object sets down somewhere else; the re-link reach must be ONE knob. **What:** the maximum distance a returning object may reach to re-link geometrically — shared by observe's husk relocation *and* identity's gone-pool re-acquire (single-sourced at build). **Impact:** the radius of "same object, moved"; beyond it only appearance can re-link. **How:** distance gate on relocation claims, shape-checked by `reloc_shape_tol`. **Limits:** large values invite wrong claims among look-alikes (measured: 340 px regressed the stress clip).
- **`contour_epsilon`** (float px, default 0.75, live) — **Why:** simplified outlines must be cheap where the perceptual pipeline doesn't apply. **What:** the approx-poly ε for the LIVE-source contour and the fallback path (features stage off); the FOOTPRINT outline every output normally emits is the perceptual inverse-EFD reconstruction, whose quality dial is `[efd] detail`. **Impact:** internal-machinery / fallback outline quality; zero effect on tracking. **How:** `approxPolyDP(ε)` on the original contour; internal consumers (EFD, pose, coverage) always use full resolution. **Limits:** display/wire only.
- **`reloc_shape_tol`** (float ≥0, default 1.0, live) — **Why:** geometric relocation lies during rearrangement. **What:** max fold-change in elongation or fill for a claimed blob to shape-match the footprint (1.0 = within 2×). **Impact:** stops a bar's id landing on a disc at the same spot. **How:** two shape ratios compared pre-claim. **Limits:** loose by design — twins pass; appearance handles them.
- **`motion_threshold`** (int 0–255, default 25, live) — **Why:** "moving" must be decidable per pixel — a hand at work is a moving mass, and the mass must be recognizable as such. **What:** the |gray − previous-gray| cut marking a pixel as moving. **Impact:** feeds the per-blob motion verdict, the settle machinery's quiet test, and the hover hold (motion over a resting footprint pins the object). **How:** threshold on the frame difference. **Limits:** shares the noise floor with `fg_threshold` (the `blur` knob serves both).
- **`motion_frac`** (float 0–1, default 0.1, live) — **Why:** a blob is "in motion" only if enough of it moves. **What:** the moving-pixel fraction above which a region counts as a moving mass. **Impact:** in-motion blobs never associate and never mint — the rule that keeps hands and carried objects out of the registry; the same verdict over a RESTING footprint triggers the hover hold. **How:** per-region count over the motion mask. **Limits:** very slow drags can stay under it.
- **`assoc_size_tol`** (float 0–1, default 0.5, live) — **Why:** ids must not migrate onto merged or foreign blobs. **What:** rejects an association when relative area difference (1 − min/max) exceeds it. **Impact:** an object+hand fusion can't inherit the object's id. **How:** area-ratio gate per association. **Limits:** objects that legitimately change apparent size (tilting) need it loose.
- **`edge_ignore_margin`** (float 0–1 of frame/mask, default 0=off, live) — **Why:** arms enter from the rim; clipped shapes are unreliable. **What:** one rim band for both boundaries — births near a frame edge are rejected/flagged `ignored`, blobs reaching the circular mask's rim likewise. **Impact:** reaching arms never mint and never claim; rim residue stays out of every output. **How:** bbox/contour proximity tests against the margin. **Limits:** real objects placed in the band are ignored too.

## `[classify]` (panel: STATE) — the settle state machine
**Why:** "settled" must be an explicit, debounced state — the entire paradigm hangs on knowing when an object is at rest.
**What:** runs each object's RESTING/MOVED/SETTLING/HIDDEN transitions from observe's change verdicts and captures the canonical footprint at settle.
**Impact:** decides *when* capture, pose measurement, feature extraction and output visibility happen; its timers are the system's felt responsiveness.
**How:** a per-object transition table over {residual, missing} on the change scale `k`, plus a stationarity trigger; a moving mass over a resting footprint PINS the object (the hover hold — its residual/missing spikes are attributed to the mass, no transition fires); capture (the only cv2 here) fires once per settle.
**Limits:** everything is time-debounced — reaction is deliberately traded for stability; hovering and manipulation are discriminated temporally, not photometrically.

- **`commit_mode`** (str: partial|global, default partial, live) — **Why:** *when* the registry updates is a compositional choice, not only an engineering one. **What:** `partial` commits each object on its own `settle_seconds` dwell — the partial-update paradigm, where a quiet object updates while another is still being handled. `global` commits nothing until the WHOLE table has been idle for `global_quiet_seconds`, then captures every settled object in one tick. **Impact:** two different instruments. Partial: each object speaks as it lands. Global: you arrange in silence and the whole configuration sounds as ONE event, with no intermediate fumbles on the wire — the automatic form of the `/rmt/trigger` pedal, whose intent is exactly *"the arrangement is set, read it"*. **How:** the same one-tick capture-everything path the pedal fires, triggered by observe's whole-frame moving-pixel fraction going quiet instead of by a foot; fires once per stillness and re-arms on motion. **Limits:** global spends the very property the tracker is built on — a mass commit re-scans everything at once, which is the worst-frame cost the `[pose] max_fits_per_frame` budget exists to spread. It also commits at precisely the moment a resting hand looks most like an object (§ the still-hand failure mode), so the exposure there is higher than in partial.
- **`global_quiet_seconds`** (float ≥0, default 1.5, live) — **What:** how long the whole table must stay idle before the one global commit fires. **Impact:** the felt latency between "I stepped back" and "it sounds". **Limits:** `commit_mode = global` only.
- **`global_quiet_frac`** (float 0–1, default 0.002, live) — **What:** the moving-pixel fraction of the whole frame below which the table counts as idle — the same scale identity uses for its table-quiet test. **Impact:** too low and camera noise means the table is never still enough to commit. **Limits:** `commit_mode = global` only.
- **`settle_seconds`** (float ≥0, default 0.3, live) — **Why:** capture must wait out landing wobble. **What:** the dwell before SETTLING→RESTING (and capture). **Impact:** THE responsiveness knob: shorter = snappier but risks capturing mid-nudge; the installation runs 0.72 s. **How:** a per-object quiet timer. **Limits:** adds exactly this latency to every settle→output path.
- **`move_debounce_seconds`** (float ≥0, default 0.1, live) — **Why:** only evidence that survives the motion's clearing may change state — one noisy frame must not un-settle an object. **What:** how long a mask must stay off its footprint before RESTING→MOVED. **Impact:** the temporal half of the hover/manipulation discrimination: a hover leaves no trace in the output, while a real displacement registers the moment the hand withdraws (plus this debounce). **How:** persistence timer on the change verdict. **Limits:** too long and quick pick-place cycles blur together.
- **`hold_max_seconds`** (float ≥0, default 2.0, live) — **Why:** a hand at work is a moving mass, and an object under it must not churn states. **What:** the ceiling on how long a RESTING object stays pinned under a moving mass. **Impact:** while pinned, the residual/missing spikes are attributed to the mass, no transition fires, and the outputs keep emitting the settled pose — so hovering and working hands never disturb the wire. **How:** a pin that expires — persistent motion cannot pin a stale entry forever. **Limits:** past the ceiling, prolonged cover becomes ordinary HIDDEN handling.
- **`confirm_seconds`** (float ≥0, default 1.0, live) — **Why:** outputs must not sound a blip that vanishes. **What:** withholds a fresh mint from every output until it persists this long (debug view shows it immediately). **Impact:** phantom-free OSC/TUIO at the cost of first-appearance latency. **How:** a `confirmed` latch outputs gate on. **Limits:** genuinely new objects reach clients late by exactly this.

## `[pose]` — one-shot placement at settle
**Why:** live pose tracking is what the paradigm removes; position and rotation are measured once, when they are measurable.
**What:** places each object at its footprint centroid and measures absolute rotation once per settle; holds the last pose otherwise.
**Impact:** the θ every output carries; zero per-frame cost between settles.
**How:** EFD phase correlation of the canonical footprint outline against the fresh settled outline — differential, continuous, with 2-fold hysteresis; the reported zero is shape-anchored (θ=0 = the shape's anchor feature screen-up: its dominant asymmetry — the handle — when one exists, else its principal axis; measured once at capture).
**Limits:** no orientation while moving (by design); rotational near-symmetry bounds what any fit can read.

- **`measure_rotation`** (bool, default true, live) — **Why:** the settle rotation fit is the priciest per-settle work, and an installation whose clients never read θ should not pay for it. **What:** switches the EFD fit off; placement, ids and the emitted outline are untouched and θ is reported as 0. **Impact:** the cost of a settle drops to capture + placement. **How:** an early branch that also releases the per-frame fit budget, so a skipped fit never defers a real one. **Limits:** none semantically — θ=0 is *correct* here, not degraded: the footprint is re-captured at the object's current orientation every settle and the outline is placed with pose[2]=0, so the turn is already carried by the shape itself. Reporting a held angle would double-count it.
- **`conf_min`** (float 0–1, default 0.4, live) — **Why:** a bad fit is worse than no update. **What:** the minimum correlation confidence to accept a settle rotation. **Impact:** below it θ holds — stale but never wild. **How:** score gate on the phase-correlation peak. **Limits:** high values freeze θ on deformed/occluded captures.
- **`max_fits_per_frame`** (int ≥1, default 1, live) — **Why:** many simultaneous settles would spike one frame. **What:** caps rotation fits per frame, deferring the rest. **Impact:** bounds the worst-frame cost; a 1–2 frame θ delay on a *still* object is imperceptible. **How:** a per-frame budget that re-fires the settle edge next frame. **Limits:** mass rearrangements stagger their θ updates.
- **`symmetry_lock`** (float 0–1, default 0.005, live) — **Why:** a circle's rotation is pure noise. **What:** the rotation-information floor under which a shape's θ locks to 0 and the fit is skipped. **Impact:** discs/rings report a clean constant instead of jitter, and cost nothing. **How:** judged once per object from `efd.rotation_information` of the canonical contour. **Limits:** near the floor, genuinely oriented but almost-round shapes lose their θ.

## `[identity]` — persistence across absence
**Why:** an object picked up and set down must come back as itself.
**What:** manages HIDDEN objects (occluded vs bared), retires the truly gone into a memory pool, and re-acquires returning objects geometrically or by appearance.
**Impact:** identity continuity across manipulation — the property the whole instrument mapping depends on.
**How:** footprint-coverage and motion evidence decide occluded/bared; one bared-anchored clock spans coast and pool; re-acquire = relocation gates + the reid tie-break.
**Limits:** memory is time-bounded; among look-alikes, geometry lies and appearance saturates (the stress clip's residual churn).

- **`pool_ttl_seconds`** (float ≥0, default 5 — installation runs 18, live) — **Why:** memory must outlive a human pick-and-place. **What:** how long an identity is remembered after its spot BARES (one clock: drawn coast + gone-pool). **Impact:** the re-acquire window; past it a return is a new object. **How:** anchored at baring — occlusion time does not count (a 20 s hover costs nothing). **Limits:** long TTLs grow the candidate set and with it wrong-claim risk.
- **`clear_frac`** (float 0–1, default 0.02, live) — **Why:** "the hand left" must be decidable from pixels. **What:** the footprint-coverage fraction below which a HIDDEN spot counts as bared. **Impact:** the occluded/gone pivot — starts the memory clock. **How:** foreground coverage inside the stored footprint. **Limits:** shadows near the threshold can flicker the verdict (motion awareness bridges it).
- **`away_quiet_seconds`** (float ≥0, default 0.5, live) — **Why:** an object visibly gone should not coast for the full TTL. **What:** bared + no motion there + an idle table held this long → pool immediately (early GONE). **Impact:** husks vanish promptly on a calm table. **How:** three quiet conditions ANDed over a timer; 0 disables. **Limits:** guarded by table-quiet — never fires mid-manipulation.
- **`table_quiet_frac`** (float 0–1, default 0.002, live) — **Why:** early-GONE must be impossible while anything is being carried. **What:** the global moving-pixel fraction under which the table counts as idle. **Impact:** the safety interlock for `away_quiet_seconds` (a carried object keeps a hand moving somewhere). **How:** whole-frame motion-mask fraction. **Limits:** camera noise inflates it — the `blur` knob keeps it honest.

## `[reid]` — the classical appearance tie-break
**Why:** geometry cannot distinguish look-alikes or long-range returns; a texture/shape/size fingerprint can — without an ML runtime.
**What:** a 21-d classical descriptor (canonically rotated crop → saturation histogram + Hu moments + shape/size scalars) used to veto wrong geometric claims and to adopt returning identities.
**Impact:** measured on the benchmark: on the rearrangement clip it is the difference between IDF1 0.97 and 0.72; chosen because it lands within ~2 pts of small deep re-id nets at 0.48 ms/crop, CPU-only.
**How:** fingerprints accumulate per object across settles; cosine similarity with a floor and a twin margin gates every adoption.
**Limits:** appearance saturates on visually identical objects — the stress-clip bound; color features assume stable lighting/WB.

- **`enabled`** (bool, restart) — **Why:** the tracker must run descriptor-free for ablation/minimal rigs. **What:** master switch for the appearance tie-break. **Impact:** off = geometry-only identity (the ablation column). **How:** stage constructed but inert when off. **Limits:** restart.
- **`reloc_cosine_reject`** (float 0–1, default 0=off, live) — **Why:** a confident *geometric* claim can still be wrong on a place-swap. **What:** appearance VETO: reject a proximity claim whose cosine falls below this. **Impact:** stops a stone stealing a gear's id at the gear's old spot. **How:** post-gate check on relocation claims. **Limits:** 0 by default — enable when swaps are common.
- **`cosine_min`** (float 0–1, default 0.6, live) — **Why:** appearance adoption needs a floor. **What:** the minimum cosine for ANY appearance-based adoption (husk claim or pool re-adopt) — one gate, one scale. **Impact:** below it, no appearance re-link ever happens (open-world safety). **How:** cosine against the object's fingerprint set. **Limits:** high floors turn borderline returns into fresh mints.
- **`uniqueness_margin`** (float 0–1, default 0.05, live) — **Why:** among twins, best-match is a coin flip. **What:** the best cosine must beat the runner-up by this margin, both adoption paths. **Impact:** refuses ambiguous claims instead of guessing — the twin guard. **How:** margin test over the candidate ranking. **Limits:** with many look-alikes it (correctly) refuses often → fresh mints.
- **`size_tol`** (float 0–1, default 0.5, live) — **Why:** appearance candidates must be physically plausible. **What:** max relative area difference for an appearance candidate. **Impact:** a thimble never adopts a plate's identity however similar the texture. **How:** area-ratio pre-filter. **Limits:** none in practice.
- **`fp_max_prints`** (int ≥1, default 8, live) — **Why:** an object's look varies with pose/position. **What:** how many fingerprints accumulate per object (most recent kept). **Impact:** robustness to viewpoint variety vs dilution of the match. **How:** a bounded per-object deque. **Limits:** memory/compare cost linear in it.
- **`fp_new_dist`** (float px ≥0, default 25, live) — **Why:** re-settling in place shouldn't duplicate fingerprints. **What:** minimum centroid move for a settle to append a new fingerprint. **Impact:** the fingerprint set spans *positions* (lighting variation), not copies. **How:** distance test against the last print's position. **Limits:** pure in-place rotations don't add views.

## `[efd]` — the shared elliptic-Fourier subsystem
**Why:** one exact spectral shape representation serves display, rotation fitting, symmetry and grouping — computed once, shared.
**What:** Kuhl–Giardina EFD on raw contour vertices (no resampling), memoized per contour.
**Impact:** feeds the rotation fit, the `efd` feature on the wire, reflectional symmetry, and the grouper's contour blend from a single integration.
**How:** complex conjugate-pair coefficients c(±n); selection by an adaptive per-component ± dB threshold normalized to the loudest single bin.
**Limits:** contour-only (no interior texture); internal consumers are guaranteed ≥24 harmonics regardless of the display width.

- **`bins`** (int 1–64, default 64, restart) — **Why:** the wire spectrum's width must be bounded. **What:** max harmonics for the emitted `efd` feature. **Impact:** wire payload size and client spectrum resolution. **How:** truncates the shared spectrum at emit; internals keep their floor. **Limits:** restart (buffers are sized by it).
- **`detail`** (float 0–1, default 0.5, live) — **Why:** clients want a shape-adaptive, not fixed-size, spectrum. **What:** the adaptive dB cut: each conjugate bin |c±n| survives on its own, thresholded against the loudest single bin (0 dB) at −detail·100 dB. **Impact:** the ONE wire-quality dial — it selects both the emitted spectrum AND the emitted outline (the wire contour is the inverse EFD of exactly the surviving components, sampled sub-pixel in float32 — never snapped to the pixel grid); 0 = only the loudest bin, 1 = the full tail; a circle emits ~1 bin, a busy contour many. **How:** per-COMPONENT ± selection (a harmonic may keep both bins, one, or none), amplitudes ÷ the max bin; the webgui inspector shows the FULL spectrum with the cut bins dimmed (a browser-only diagnostic — the OSC wire carries survivors only). **Limits:** split harmonics whose two bins are individually quiet drop entirely (by design).

## `[features]` — per-object PERCEPTUAL shape scalars
**Why:** sonification maps need pose-invariant, person-meaningful descriptors, computed only when they can change.
**What:** the perceptual set — linear extent (`size`), elongation (`eccentricity`, the standard 0=circle definition), circle-likeness (`circularity`), box-likeness (`rectangularity`, rescaled so a disc reads 0 rather than the raw pi/4 fill), filled-ness (`solidity`), points/prongs (`lobe_count`), spike sharpness (`spikiness`), mirror symmetry (`reflectional_symmetry`) — plus the `efd` spectrum, computed once at capture and annotated onto the object.
**Impact:** the `/rmt/features/*` vocabulary clients map from — all 0..1 except `lobe_count` (a count); the areal form was removed at v0.32.10 — quadratic, and real objects used under 2% of its range; zero per-frame cost (cache by footprint identity, refreshed live when `[efd] detail` changes).
**How:** geometric features are computed on the PERCEPTUAL contour — the inverse EFD of the detail-surviving components (the same shape the spectrum ships and the outputs draw); EFD-native features (`efd`, `reflectional_symmetry`) read the raw footprint. Three system-internal features (`ellipse_variance`, `circle_variance`, `defect_depth`) exist only for reid's shape block — never user-facing.
**Limits:** contour-derived only; values refresh on re-capture or a detail change.

- **`enable`** (str list from the registry, restart) — **Why:** most rigs need a handful, not all. **What:** the comma-list of computed person-facing features. **Impact:** wire vocabulary + capture cost scale with it. **How:** registry lookup at build; unknown or internal names WARN and are skipped (a stale conf degrades, never crashes). **Limits:** restart to change.

## `[grouper]` — the output voice budget
**Why:** a synth has finitely many voices; fifteen objects must not demand fifteen.
**What:** when over budget, fuses the nearest/likest objects into ≤ `max_objects` compound objects that REPLACE their members on every output wire (the fused emit view) — same per-object addresses, ids from the same never-recycled sequence, real features measured on the mass-scaled blend.
**Impact:** outputs and clients see a bounded voice count with stable compound ids and zero client-side grouping logic; a no-op under budget. The webgui process view keeps the real objects (compounds as overlay).
**How:** merge cost blends normalized proximity with likeness (size+roundness) under `spatial_weight`; compound contours are complex-EFD blends of members.
**Limits:** a compound's θ is its dominant member's; membership shifts as objects move (id kept by overlap).

- **`enabled`** (bool, default true, live) — the budget switch; off = outputs always carry raw objects.
- **`max_objects`** (int ≥1, default 8, live) — **Why:** the budget is the client's polyphony. **What:** the compound-count ceiling when over budget. **Impact:** the maximum voices any output ever carries. **How:** fuse-nearest until ≤ N. **Limits:** small budgets fuse aggressively — spatial meaning coarsens.
- **`spatial_weight`** (float 0–1, default 0.5, live) — **Why:** "which objects share a voice" is an aesthetic choice. **What:** 1 = fuse by proximity, 0 = by likeness (alike objects share a voice anywhere), between = blend. **Impact:** the character of compound voices under load. **How:** the blend factor in the merge cost. **Limits:** likeness uses size+roundness only.

## `[clusters]` — spatial grouping of voices
**Why:** clients want "which objects sit together" as a fact, without inventing semantics.
**What:** proximity union-find over the discrete voices (objects, or compounds when the grouper is active) → a per-cluster table (id, members, centroid, bbox, min-circle, hull) with persistent ids.
**Impact:** the `/rmt/cluster` wire a client joins to `/pose` for its own layouts (e.g. a radial sequencer); membership is settle-held — objects join by settling, hold while present, leave when gone.
**How:** circles link when edge-to-edge gap < radius; ids persist by membership overlap.
**Limits:** a DISJOINT partition — an object belongs to at most one cluster; overlapping/shared clusters would need a different formulation.

- **`radius`** (float 0–1 of max frame side, default 0.12, live) — **Why:** "together" needs a scale. **What:** the max gap between enclosing circles that links two objects (overlap always links). **Impact:** the cluster granularity dial. **How:** size-aware edge-gap union-find (big objects touch sooner than centers suggest). **Limits:** one global scale for the whole table.
- **`min_objects`** (int ≥2, default 2, live) — **Why:** a lone object is not a group. **What:** minimum members for a cluster. **Impact:** filters accidental pairings when raised. **How:** post-partition size filter. **Limits:** floor 2 by construction.

---

# TAB: OUTPUT

## `[mjpeg]` — the video monitor stream
**Why:** operators need to *see* what the tracker sees, in any browser, without a desktop window.
**What:** an MJPEG-over-HTTP server streaming one selected view.
**Impact:** pure observability; when nobody watches, frames are never even encoded.
**How:** a bus consumer that JPEG-encodes on demand; the webgui embeds it and draws overlays on top.
**Limits:** MJPEG is bandwidth-heavy; one view at a time.

- **`host` / `port`** (str/int, restart) — bind address/port of the HTTP server (default 0.0.0.0:8080).
- **`view`** (str: input|debug, live) — **What:** which view streams — raw pre-crop vs foreground-masked. **Impact:** debug is the noise-tuning view (the fg mask made visible).
- **`scale`** (float 0–1, default 1, live) — downscales the stream; bandwidth vs legibility.
- **`quality`** (int 1–100, default 80, live) — JPEG quality; bandwidth vs artifacts.
- **`max_fps`** (float ≥0, default 0=every frame, restart) — stream cap, decoupling monitor load from pipeline rate.

## `[tuio2]` — the standard protocol egress
**Why:** existing TUIO clients must work unmodified.
**What:** TUIO 2.0 bundles — /frm, /tok (position + angle; both tag fields 0, see below), optional /ocg contours and /bnd bounds, /alv — plus a small /rmt/obj sidecar.
**Impact:** plug-and-play with the TUIO ecosystem; the token angle is wrapped to [0,2π), clockwise, 0 = the shape's own axis up. An object IS its session id — clients key on that alone.
**How:** a bus consumer building per-frame bundles, split under a packet ceiling.
**Markerless by contract (test-enforced):** /tok's `tu_id` (type/user) and `c_id` (component/symbol) are BOTH 0 for every object. Those fields identify a *tag the object carries*; a readymade carries none — nothing is ever decoded off it, and no /tuio2/sym is sent. Neither is derived from the tracker id: echoing it into `c_id` would assert a symbol that was never read, and one that changes whenever a re-acquire mints a new id, silently re-mapping the same physical object under any client that dispatches on it.
**Limits:** TUIO's schema has no state/confidence fields — hence the sidecar; contours can be heavy.

- **`destinations`** (str list, live) — **Why:** several clients watch one stream. **What:** THE live target list (host:port, comma-separated; empty = 127.0.0.1:3333). **Impact:** add/remove listeners mid-performance. **How:** every packet sent to each. **Limits:** UDP — no delivery guarantee.
- **`coord_mode`** (str: normalized|pixel, live) — [0,1] vs raw px on the wire; normalized is the TUIO convention.
- **`timetag_mode`** (str: timetag|ints, default timetag, live) — **Why:** some clients' OSC parsers reject the `t` type tag and drop the whole message; for `/tuio2/frm` that also loses `dim`, which every normalized coordinate is read against. **What:** `ints` replaces the one `t` argument with TWO int32s carrying real capture time — unix seconds and microseconds — so the type-tag string is `,iiiis` and no `t` appears anywhere. **Impact:** an otherwise-incompatible client works. **How:** the time argument is swapped at message build. **Limits:** off-spec — a conforming TUIO client expects the `t` — and `/frm` then carries no capture time, though the bundle's own timetag (never type-tagged) still does.
- **`send_contours`** (bool, default true, live) — emits /ocg outlines: normally the perceptual inverse-EFD contour (quality = `[efd] detail`, sub-pixel float coords); `contour_epsilon` applies only to the live-source/fallback path. Heavy but what visual clients want.
- **`contour_source`** (str: footprint|auto|live, live) — **What:** rest outline (clean, placed at live pose) vs live blob outline vs auto. **Impact:** footprint = clean shapes that track; live = raw segmentation truth.
- **`send_bounds`** (bool, default false, live) — optional /bnd (pos, angle, w, h, area) for box-based clients.
- **`include_custom`** (bool, default true, live) — the /rmt/obj sidecar TUIO lacks, in wire order: `id:i`, `state_name:s`, `pose_conf:f`. The confidence is the settle fit's Parseval-normalised EFD correlation peak (0..1) — but reads 1.0 both for a perfect fit and for the two cases where no fit ran at all (a first settle, which *becomes* the reference, and a symmetry-locked shape whose rotation is undefined).
- **`source`** (str, default "readymade", live) — the source name in /frm, distinguishing multiple trackers on one network.
- **`max_packet_bytes`** (int ≤65507, default 8192, restart) — **Why:** a fat frame must not overflow a UDP datagram. **What:** the per-packet ceiling bundles split under. **Impact:** a contour too big for one packet is progressively coarsened to fit, and if even the coarsest cannot fit it is DROPPED — because an oversized send fails outright and would take the whole bundle with it, positions included, so the object's pose still ships without its outline (logged once). **Limits:** the OS datagram cap — macOS's `net.inet.udp.maxdgram` defaults to 9216 B (bigger sends fail EMSGSIZE), hence the under-floor default; raising the knob needs the sysctl raised on every receiving host. TUIO also requires `/frm` + the full `/alv` in every packet, so the minimum packet grows with the alive set — a ceiling below that floor cannot be honoured by any setting.
- **`max_fps`** (float ≥0, default 0, restart) — send-rate throttle; 0 = every fresh frame.

## `[oscout]` — the rich custom-OSC egress
**Why:** the features/EFD/cluster vocabulary exceeds TUIO's schema.
**What:** self-describing per-field messages (`/rmt/<producer>/<field> id v…`), /rmt/pose (unwrapped shape-anchored θ), the cluster table, opt-in contours.
**Impact:** the wire the SC/Max/vvvv examples map from — a new feature appears on the wire without touching this plugin.
**How:** generic reflection over per-object annotations + dedicated emitters for the scene-level group/cluster tables; frames split across datagrams under a byte budget.
**Limits:** UDP; `contour` is never part of `all` (opt-in — bulky).

- **`destinations`** (str list, live) — as tuio2's; empty = 127.0.0.1:3400.
- **`prefix`** (str, default /rmt, live) — the address namespace; lets several trackers or versions coexist.
- **`send`** (str list: all|pose|contour|features|cluster, restart) — **Why:** clients differ in appetite. **What:** the emission allow-list ('all' = every producer + cluster table + pose; 'contour' strictly opt-in). **Impact:** wire volume and client parsing surface.
- **`coord_mode`** (str: normalized|pixel, live) — pose coordinate convention.
- **`max_fps`** (float ≥0, default 0, restart) — egress throttle.

## `[rec]` — the frame recorder
**Why:** every session should be able to become tomorrow's test clip.
**What:** writes a selected view to video files.
**Impact:** produces the conditioned clips the whole evaluation pipeline replays (record `processed`, replay with `--no-preprocess`).
**How:** a bus consumer feeding `cv2.VideoWriter`.
**Limits:** codec availability is platform-dependent; recording competes for disk I/O.

- **`dir`** (str, default recordings, live) — output directory.
- **`codec`** (str: mp4v|avc1|xvid|mjpg, live) — fourcc; availability varies by platform build.
- **`view`** (str: processed|input|debug, live) — **What:** which view is written. **Impact:** `processed` gives replayable conditioned footage — the corpus recipe.
- **`fps`** (int ≥1, default 30, live) — the container's playback rate (should match capture).
- **`width` / `height`** (int, 0 = source, restart) — optional re-scale on write.

---

# TAB: PERF (conf-only; hidden from the panel)

## `[perf]` — process-wide performance
**Why:** OpenCV parallelises every operation across all cores by default, and this pipeline's operations are small enough that the splitting can cost more than the work — while the preprocess (lens undistort, flatfield) is exactly the big work threads help.
**What:** `cv2_threads` (int, 0–256, default 0 = leave OpenCV alone, live) — how many threads OpenCV may split ONE operation across.
**Impact:** speed only, never the result; the benchmark pins 2 on every machine so rows compare.
**How:** applied via `cv2.setNumThreads` at bind and on every live set; `--cv2-threads N` overrides per run.
**Limits:** reaches only OpenCV — the pure-Python output threads cannot respond to it. Measure a machine's own preference with `benchmarks/bench_threads.py`.

# TAB: CONTROL (conf-only; hidden from the panel)

## `[oscin]` — the OSC control input
**Why:** every runtime control (set/query/trigger/pause/…) arrives as OSC, so any client — including the webgui via its bridge — speaks one protocol.
**What:** the UDP listener routing `/rmt/*` commands into the tracker.
**Impact:** THE control plane; the webgui is just a browser client of it.
**How:** OSC decode → command inbox → applied between ticks (thread-safe by construction).
**Limits:** plumbing only: `host`/`port`/`prefix` (default 0.0.0.0:3401, `/rmt`), all restart — you are connected *through* it, so it cannot re-bind itself live.

## `[webgui]` — the browser control surface
**Why:** the tracker is browser-only (no OS window); operators need panel + overlays anywhere.
**What:** serves the UI and bridges HTTP↔OSC; the panel builds itself from the live parameter registry.
**Impact:** every knob in this document is live-editable there (this whole catalog mirrors its tabs).
**How:** static files + POST /osc bridge + an SSE snapshot stream; overlays render client-side.
**Limits:** plumbing only: `host`/`port` (default 0.0.0.0:8090), restart; it deliberately hides its own and oscin's bind config from the panel.
