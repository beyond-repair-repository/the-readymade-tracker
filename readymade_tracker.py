#!/usr/bin/env python3
"""ReadyMade Tracker — orchestrator for the settle-based, partial-update object tracker.

Parse ``readymade.conf`` → build the core (``Scene`` + the ordered ``Pipeline``) → wire
the I/O (the input backend → ``InputRunner`` → ``InputThread``) → ``serve()`` the
threaded loop. The core owns every thread; ``serve()`` *is* the main thread — a
headless supervisor (the browser control surface is the only UI; no OS window).

    venv/bin/python readymade_tracker.py                            # play the configured input
    venv/bin/python readymade_tracker.py --source video clip.mp4    # override the source
    venv/bin/python readymade_tracker.py --fps 0                    # max speed (no real-time pacing)

No per-frame inference and no live pose: the tracker holds the last settled registry
while blobs are in motion, and re-scans only the regions that settle (auto-settle or a
manual ``/rmt/trigger``) — classic reactable / partial-update operation.
"""

from __future__ import annotations

__version__ = "0.32.165"           # MAJOR.MINOR.BUILD — bump BUILD per minor commit, MINOR per bigger change

import argparse
import errno
import faulthandler
import os
import sys
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from lib.audit import Audit, BufferedSink, log_path
from lib.classify import Classify
from lib.clusters import Clusters
from lib.config import Config, ConfigError, load_config
from lib.control import ControlRouter, ScanTrigger
from lib.efd import Efd
from lib.features import Features
from lib.grouper import Grouper
from lib.identity import Identity
from lib.loop import (CommandInbox, InputRunner, InputThread, LatestFrameSlot,
                      PipelineLoop, PullSlot, SnapshotBus)
from lib.logger import NULL_LOGGER, Logger
from lib.mot import MotWriter
from lib.observe import Observe
from lib.params import ParamStore, build_registry, collect_schema, seed_values
from lib.pipeline import Pipeline
from lib.perf import Perf
from lib.pose import Pose
from lib.preprocess import Preprocess, PreprocessConfig
from lib.profile import Profiler
from lib.reid import ReidParams, load_embedder
from lib.scene import Scene
from plugins.control.mcu.mcu import Mcu
from plugins.control.oscin.oscin import OscIn
from plugins.control.webgui.webgui import WebGui
from plugins.input.camera.camera import CameraBackend
from plugins.input.video.video import VideoBackend
from plugins.output.mjpeg.mjpeg import MjpegOut
from plugins.output.oscout.oscout import OscOut
from plugins.output.rec.rec import Recorder
from plugins.output.tuio2.tuio2 import Tuio2Out

HERE = Path(__file__).resolve().parent
DEFAULT_CONF = HERE / "readymade.conf"

# source → backend factory. A backend supplies open/grab/seek/close + a transport.
# A backend that keeps its config in its OWN section (camera → [camera]) is listed
# in _BACKEND_SECTION; the rest read [input] (video: source + location + fps).
_BACKENDS = {"video": VideoBackend, "camera": CameraBackend}
_BACKEND_SECTION = {"camera": "camera"}

# Per-section typed schemas. [input] = the source; [preprocess] = the core preprocess stage.
# EVERY section is plugin-owned: each stage/plugin/backend declares its params
# as a class-attr ``PARAMS`` tuple, and COMPONENTS maps its section to that class. SCHEMAS stays
# empty — the app declares no central schema (build_registry accepts a bare dict only for
# lightweight test fixtures).
SCHEMAS: dict[str, dict[str, Any]] = {}

# section → the owning class for PLUGIN-OWNED params. A component declares its params as a class-attr
# ``PARAMS`` tuple; that ONE declaration drives BOTH the live-param registry (build_registry harvests it)
# AND the INI coercion (collect_schema derives {key: type}). This is the SINGLE source of truth per param.
COMPONENTS: dict[str, Any] = {
    "input": VideoBackend, "camera": CameraBackend, "preprocess": PreprocessConfig,
    "observe": Observe, "classify": Classify, "reid": ReidParams,
    "identity": Identity, "pose": Pose, "efd": Efd, "features": Features, "clusters": Clusters,
    "grouper": Grouper, "perf": Perf,
    "tuio2": Tuio2Out, "mjpeg": MjpegOut, "oscout": OscOut,
    "rec": Recorder, "oscin": OscIn, "mcu": Mcu, "webgui": WebGui,
}

# the coercion schema handed to load_config — derived entirely from the components' PARAMS.
CONF_SCHEMAS: dict[str, dict[str, Any]] = {**SCHEMAS, **collect_schema(COMPONENTS)}


@dataclass
class Core:
    """The wired spine + the thread-orchestration passes."""

    scene: Scene
    pipeline: Pipeline
    backend: Any
    runner: InputRunner
    slot: LatestFrameSlot
    inbox: CommandInbox
    bus: SnapshotBus
    loop: PipelineLoop
    profiler: Any = None
    audit: Any = None
    mot: Any = None              # --mot-export MOTChallenge track writer (closed at shutdown)
    outputs: tuple = ()          # background OUTPUT threads (tuio2, …) waiting on the bus
    control: Any = None          # ControlRouter — drains the inbox on the supervisor thread
    controls: tuple = ()         # control-INPUT threads (oscin, …) posting to the inbox


def build(config: Config, *, logger: Any = None, paused: bool = False,
          profile: bool = False, max_frames: int | None = None,
          debug: bool = False, profile_every: int = 100,
          pull: bool = False, mot_export: str | None = None,
          conf_path: Path | None = None, overrides: dict | None = None) -> Core:
    """Construct + wire the core: the input backend (by ``source``) + preprocess +
    runner; the ``Scene`` + ``Pipeline`` (observe → classify → identity → pose)
    + the loop machinery. Outputs (webgui / mjpeg / tuio2 / oscout / rec) and the control
    inputs (oscin, mcu) start iff their section is enabled. ``paused`` holds the source on frame 1;
    ``max_frames`` stops the loop after N processed frames. ``debug``/``profile``
    open buffered ``logs/`` audit sinks."""
    icfg = config.get("input")
    source = icfg.get("source", "video") if icfg is not None else "video"
    factory = _BACKENDS.get(source)
    if factory is None:
        raise ValueError(f"unknown source {source!r} (have {sorted(_BACKENDS)})")
    # a backend with its own section (camera → [camera]) reads that; video reads [input].
    bcfg = config.get(_BACKEND_SECTION.get(source)) if source in _BACKEND_SECTION else icfg
    backend = factory(bcfg, logger=logger)
    backend.open()
    if paused:                                    # --paused: hold on frame 1
        backend.transport.pause()
        backend.transport.step_forward()
    ppcfg = config.get("preprocess")             # the preprocess CORE stage, own section
    pcfg = PreprocessConfig.from_mapping(ppcfg if ppcfg is not None else {})
    pcfg.load_artifacts(logger)
    profiler = Profiler() if profile else None
    # a label for logs / the gui: the clip name (video) or device (camera).
    if source in _BACKEND_SECTION:
        loc = f"{source}{bcfg.get('device') if bcfg is not None else ''}"
    else:
        loc = icfg.get("location", source) if icfg is not None else source
    audit = None
    if debug or profile:
        audit = Audit(
            debug=BufferedSink(log_path("debug", loc)) if debug else None,
            profile=BufferedSink(log_path("profile", loc)) if profile else None,
            profiler=profiler, profile_every=profile_every)
    preprocess = Preprocess(pcfg, logger=logger)            # captured: it's a live param target (calibration)
    runner = InputRunner(backend, preprocess, logger=logger, profiler=profiler)
    scene = Scene(logger=logger, audit=audit)
    stages, param_targets, scan_trigger = _build_stages(config, logger, backend=backend,
                                                        source=source, preprocess=preprocess, pcfg=pcfg)
    mot = None
    if mot_export is not None:                               # --mot-export → MOTChallenge track dump (every frame)
        mot_path = mot_export or log_path("mot", loc)[:-4] + ".txt"   # given path, else logs/mot_<dt>_<src>.txt
        mot = MotWriter(mot_path, logger=logger)
        if pcfg.enabled and (pcfg.crop_enabled or pcfg.lens_enabled or str(pcfg.flip).lower() != "none"):
            (logger or NULL_LOGGER).warning(                # coords would be in the cropped/warped INPUT space,
                "[mot] preprocess (crop/lens/flip) is ACTIVE - MOT coordinates are in the tracker's INPUT "
                "space, NOT the raw source; they will NOT line up with a raw-space GT. Run with "
                "--no-preprocess for a benchmark (already-cropped clips need no preprocess).")
    pipeline = Pipeline(scene, stages=tuple(stages),
                        logger=logger, profiler=profiler, audit=audit, mot=mot)
    slot = PullSlot() if pull else LatestFrameSlot()
    inbox, bus = CommandInbox(), SnapshotBus()
    loop = PipelineLoop(pipeline, slot, inbox, bus, max_frames=max_frames, logger=logger)
    output_map = _build_outputs(config, logger, profiler, pcfg)  # section name → output (for runtime on/off)
    control = ControlRouter(transport=getattr(backend, "transport", None),
                            outputs=output_map, bus=bus, preprocess=preprocess,
                            scan_trigger=scan_trigger, logger=logger)
    if "webgui" in output_map:                               # feed the browser: rec state, stream view, fps
        wg = output_map["webgui"]
        wg.attach_stats(lambda: loop.fps)                    # pipeline fps (headroom) → header
        wg.attach_transport(backend.transport)               # pause state → play/pause button sync
        wg.attach_preprocess(preprocess)                     # the mask-circle overlay reads the live cfg
        # single-source the OSC bridge + video underlay from the LIVE [oscin]/[mjpeg] — no mirror knobs to
        # desync (a port/prefix change there follows automatically instead of silently breaking the panel).
        oin, mj = config.get("oscin"), config.get("mjpeg")
        wg.attach_bridge(oscin_port=int(oin.get("port", 3401)) if oin is not None else 3401,
                         prefix=str(oin.get("prefix", "/rmt")) if oin is not None else "/rmt",
                         mjpeg_port=int(mj.get("port", 8080)) if mj is not None else 8080)
        if "rec" in output_map:
            wg.attach_recorder(output_map["rec"])            # ● rec ↔ ■ stop, driven off the live status
        if "mjpeg" in output_map:
            wg.attach_mjpeg(output_map["mjpeg"])             # so the overlay maps onto the streamed view
    param_targets.update(output_map)                         # tuio2 / mjpeg / oscout / rec are param targets too
    controls = _build_control(config, logger, profiler, source=source, backend=backend,
                              param_targets=param_targets, conf_path=conf_path, overrides=overrides)
    return Core(scene, pipeline, backend, runner, slot, inbox, bus, loop,
                profiler, audit, mot, tuple(output_map.values()), control, tuple(controls))


def _build_stages(config: Config, logger: Any, *, backend: Any, source: str,
                  preprocess: Preprocess, pcfg: PreprocessConfig
                  ) -> tuple[list[Any], dict[str, Any], ScanTrigger]:
    """The processing stages (observe → classify → identity → pose, plus the optional
    grouper/features/clusters — in that order) and the live-param target map for the store. Returns
    ``(stages, param_targets, scan_trigger)`` — the trigger is shared with ControlRouter."""
    # reid is OPTIONAL and NEURAL-FREE: the classical color+shape+size descriptor (no onnxruntime, no
    # model file). Absent [reid] → None, and identity/observe fall back to shape + position. The ONE
    # embedder is shared by observe (appearance-assisted relocation) + identity (fingerprint pool re-id).
    embedder = load_embedder(config.get("reid"), logger=logger) if config.is_enabled("reid") else None
    # the [reid] APPEARANCE POLICY (cosine gates, twin guard, size gate, fingerprint accumulation) —
    # declared once, read LIVE by observe + identity.
    reid_params = ReidParams(config.get("reid"), logger=logger)
    observe = Observe(config.get("observe"), logger=logger,   # blobs · association · Observation
                      preprocess_cfg=pcfg, embedder=embedder,  # live cfg → reject blobs clipped by the mask
                      reid_params=reid_params)
    # the manual-scan pedal: /rmt/trigger fires it on the supervisor thread, classify takes it on the
    # pipeline thread → capture the settling objects that tick without their dwell, and emit at once.
    scan_trigger = ScanTrigger()
    # observe single-sources TWO values to their second consumers, read LIVE off the attached instance
    # (the same object the param registry writes) — so /rmt/set observe k / relocate_dist reaches both
    # stages mid-run: classify shares the ONE threshold scale k; identity shares the ONE set-down
    # re-link reach (its gone-pool re-acquire = the pool half of observe's husk relocation).
    classify = Classify(config.get("classify"), logger=logger, scan_trigger=scan_trigger,
                        observe=observe)                        # per-object comparison → status
    identity = Identity(config.get("identity"), logger=logger, embedder=embedder,
                        reid_params=reid_params, observe=observe)   # cross-settle id + GONE-gated retirement
    pose = Pose(config.get("pose"), logger=logger)          # one-shot placement at settle
    stages: list[Any] = [observe, classify, identity, pose]
    # section → the live object, for the param store (/rmt/set|reload|query) — hot stages only.
    param_targets: dict[str, Any] = {"observe": observe,
                                     "classify": classify, "identity": identity, "pose": pose,
                                     "reid": reid_params,        # the appearance policy, live for both readers
                                     "preprocess": preprocess}   # live crop/mask/flatfield calibration
    if source in _BACKEND_SECTION:                              # a live camera → its image controls are hot
        param_targets[_BACKEND_SECTION[source]] = backend      # (exposure/brightness/… apply to the device now)

    # the SHARED elliptic-Fourier config ([efd]: bins + the live detail knob). Always built — pose's rotation
    # fit, the symmetry feature and the grouper use the shared spectrum even when the efd feature is off; its
    # __init__ sets the cache depth to max(bins, MIN_BINS) so those internal consumers never run short.
    efd = Efd(config.get("efd"), logger=logger)
    param_targets["efd"] = efd                              # the live detail knob (/rmt/set efd detail …)
    # NOT a stage — process-wide performance settings, built unconditionally so the knob is live
    # even with no [perf] section (it then holds its 0 = leave-OpenCV-alone default).
    param_targets["perf"] = Perf(config.get("perf"), logger=logger)
    if "grouper" in config:                                 # optional: fuse >max_objects into compound super-objects.
        # Runs BEFORE features so the compounds get REAL features computed on their mass-scaled
        # blended contour, and before clusters so clusters form over the DISCRETE voices — when the
        # grouper is active, a compound clusters as ONE entity, never its member pieces.
        # Constructed whenever the SECTION is present — even `enabled = false` — because `enabled` is a
        # LIVE param (its conf comment promises on/off without a restart). A disabled stage is a cheap
        # per-frame no-op; a stage built only-when-enabled would make `/rmt/set grouper enabled 1`
        # report "inactive" (no live target), forcing a conf edit + restart for off→on while the
        # webgui showed an editable checkbox that always errors.
        grouper_stage = Grouper(config.get("grouper"), logger=logger,
                                enabled=config.is_enabled("grouper"), efd=efd)
        stages.append(grouper_stage)
        param_targets["grouper"] = grouper_stage
    if config.is_enabled("features"):                       # optional: annotate obj.features (shape descriptors)
        features_stage = Features(config.get("features"), efd=efd, logger=logger)
        stages.append(features_stage)
        param_targets["features"] = features_stage
    if "clusters" in config:                                # optional: publish the spatial-cluster table.
        # Constructed whenever the SECTION is present — even `enabled = false` — so the toggle is LIVE
        # (mirrors the grouper): a disabled stage is a cheap per-frame no-op that publishes an empty
        # table, and /rmt/set clusters enabled 1 switches it on without a restart.
        clusters_stage = Clusters(config.get("clusters"), logger=logger,
                                  enabled=config.is_enabled("clusters"))
        stages.append(clusters_stage)
        param_targets["clusters"] = clusters_stage
    return stages, param_targets, scan_trigger


def _build_outputs(config: Config, logger: Any, profiler: Any, pcfg: Any = None) -> dict[str, Any]:
    """Construct + ``setup`` every enabled OUTPUT plugin. Returns ``{section: output}`` —
    the map ControlRouter uses for runtime on/off (started later by ``serve``)."""
    output_map: dict[str, Any] = {}
    if config.is_enabled("tuio2"):
        tuio2 = Tuio2Out()
        tuio2.setup(config.get("tuio2"), logger=logger, profiler=profiler)
        output_map["tuio2"] = tuio2
    if config.is_enabled("mjpeg"):
        mjpeg = MjpegOut()
        mjpeg.setup(config.get("mjpeg"), logger=logger, profiler=profiler)
        output_map["mjpeg"] = mjpeg
    if config.is_enabled("oscout"):
        oscout = OscOut()
        oscout.setup(config.get("oscout"), logger=logger, profiler=profiler)
        output_map["oscout"] = oscout
    if config.is_enabled("rec"):
        recorder = Recorder()
        recorder.setup(config.get("rec"), logger=logger, profiler=profiler,
                       preprocess_cfg=pcfg)   # the processed view bakes the live circular mask
        output_map["rec"] = recorder
    if config.is_enabled("webgui"):
        webgui = WebGui()
        webgui.setup(config.get("webgui"), logger=logger, profiler=profiler)
        output_map["webgui"] = webgui
    return output_map


def _build_control(config: Config, logger: Any, profiler: Any, *, source: str, backend: Any,
                   param_targets: dict[str, Any], conf_path: Path | None,
                   overrides: dict | None) -> list[Any]:
    """Construct + ``setup`` the control-INPUT plugins (oscin, mcu) around the ONE shared
    ``ParamStore`` (built only when a control input is enabled). Mutates ``param_targets``
    (adds the mcu). Returns the control-input threads (they post to the inbox)."""
    controls: list[Any] = []                                 # control-input threads (post to the inbox)
    if config.is_enabled("oscin") or config.is_enabled("mcu"):
        # ONE live-parameter store, shared by EVERY control input (OSC /rmt/set|reload|query, the mcu's
        # knobs). reload_fn re-reads the SAME layered conf from disk (plugin defaults < readymade.conf
        # < camera < CLI overrides).
        registry = build_registry(SCHEMAS, COMPONENTS)
        values = seed_values(registry, config)
        if source in _BACKEND_SECTION and hasattr(backend, "control_range"):
            # bound the [camera] control params to the OPEN device's REAL ranges, so /rmt/query → the
            # webgui can't offer a value the device would reject (and the store clamps to it).
            _apply_device_ranges(registry, _BACKEND_SECTION[source], backend, values, logger)
        reload_fn = (lambda: _load(conf_path, overrides or {})) if conf_path is not None else None
        # in camera mode camera1.conf OVERRIDES [camera]/[preprocess]; route those saves there so a
        # webgui crop/exposure edit isn't masked on the next load (see ParamStore.save layering).
        layer_roots = _camera_layer(conf_path, overrides or {}) if conf_path is not None else []
        layer_path = Path(layer_roots[-1]) if layer_roots else None
        store = ParamStore(registry, param_targets, values,
                           reload_fn=reload_fn, conf_path=conf_path, logger=logger,
                           layer_path=layer_path,
                           layer_sections=frozenset({"camera", "preprocess"}) if layer_path else frozenset())
        _tr = backend.transport

        def _poke_paused() -> None:
            # a LIVE knob changed while PAUSED: re-emit the frozen frame so ONE pipeline pass shows the
            # effect (the same one-shot a paused /rmt/trigger uses) — else a paused operator tuning e.g.
            # [efd] detail or [observe] fg_threshold sees nothing until unpause/step. Frames flowing →
            # the next tick shows it anyway and the refresh flag is a no-op.
            if getattr(_tr, "paused", False):
                refresh = getattr(_tr, "refresh", None)
                if callable(refresh):
                    refresh()
        store.on_applied = _poke_paused
        if config.is_enabled("oscin"):
            oscin = OscIn()
            oscin.setup(config.get("oscin"), logger=logger, profiler=profiler)
            oscin.attach_params(store)
            controls.append(oscin)
        if config.is_enabled("mcu"):                          # serial panel: pedal → trigger, knob → detail
            mcu = Mcu()
            mcu.setup(config.get("mcu"), logger=logger, profiler=profiler)
            mcu.attach_params(store)
            controls.append(mcu)
            param_targets["mcu"] = mcu
    return controls


# --------------------------------------------------------------------------- #
# serve — the main thread (a headless supervisor)
# --------------------------------------------------------------------------- #

def _safe_start(plugin: Any, arg: Any, logger: Any) -> None:
    """Start one output/control, but never let its failure crash the whole tracker. A bound port
    (``EADDRINUSE`` — almost always another tracker already running) or any other start error is
    logged at ERROR and skipped, so the pipeline + the plugins that DID start keep running.
    Teardown's guarded ``stop()`` already tolerates a plugin that never started."""
    name = type(plugin).__name__
    try:
        plugin.start(arg)
    except OSError as exc:
        if exc.errno == errno.EADDRINUSE:
            (logger or NULL_LOGGER).error(
                f"[readymade] {name} could not bind its port — already in use. Is another tracker "
                f"running? Skipping {name}; the rest of the tracker continues.")
        else:
            (logger or NULL_LOGGER).error(f"[readymade] {name}.start() failed: {exc!r} - skipping it.")
    except Exception as exc:                       # noqa: BLE001 — a plugin start must never be fatal
        (logger or NULL_LOGGER).error(f"[readymade] {name}.start() failed: {exc!r} - skipping it.")


# Upper bound on the whole teardown (Quit / Ctrl-C / EOF / restart). Normal teardown takes ~2-4 s
# (two 2 s joins at most). A native call that never returns can stall it forever: pyuvc's
# Capture.close() runs libuvc's uvc_stream_stop WITH the GIL held, and that loops until every
# cancelled USB transfer reports back, which on macOS after a USB hiccup can be never. With the GIL
# held no Python code runs at all, so neither Quit, Ctrl-C nor a Python-level timeout can end it.
_SHUTDOWN_TIMEOUT_S = 8.0


def _arm_shutdown_watchdog(timeout: float = _SHUTDOWN_TIMEOUT_S) -> None:
    """Arm faulthandler's watchdog for the teardown. It runs on its own NATIVE thread that never needs
    the GIL, so it fires even while a C call holds it: it prints every thread's stack to stderr (which
    names the call that hung) and then ``_exit(1)``s. The OS releases the camera when the process ends."""
    try:
        faulthandler.dump_traceback_later(timeout, exit=True, file=sys.__stderr__)
    except Exception:                             # noqa: BLE001 — no usable stderr fd: run unguarded
        pass


def _disarm_shutdown_watchdog() -> None:
    try:
        faulthandler.cancel_dump_traceback_later()
    except Exception:                             # noqa: BLE001
        pass


# The camera release gets its OWN, shorter budget. Observed on macOS + pyuvc: Capture.close() sometimes
# never returns (libuvc's stream stop waits for cancelled USB transfers, or uvc_close joins libusb's event
# thread, which wakes on its own only every 60 s) — with the GIL held. A normal release takes well under 1 s.
_CAMERA_CLOSE_TIMEOUT_S = 3.0


def _close_backend_guarded(backend: Any, *, is_camera: bool, logdir: str = "logs") -> None:
    """Close the frame source under a dedicated native watchdog. If the close hangs, the watchdog writes
    every thread's stack to ``logs/shutdown-hang_<stamp>.txt`` (not the console) and ends the process
    with status 1; macOS frees the camera when the process ends. If it returns, the file is removed."""
    hang_path = hang_file = None
    try:
        os.makedirs(logdir, exist_ok=True)
        hang_path = os.path.join(logdir, time.strftime("shutdown-hang_%Y%m%d-%H%M%S.txt"))
        hang_file = open(hang_path, "w", encoding="utf-8")
        hang_file.write(f"The input close did not return within {_CAMERA_CLOSE_TIMEOUT_S:.0f} s, so the "
                        f"process was ended (exit status 1). Thread stacks at that moment:\n\n")
        hang_file.flush()
        try:
            from lib.fsutil import own_to_dir_owner
            own_to_dir_owner(hang_path)          # a sudo run must not leave a root-owned file in logs/
        except Exception:                        # noqa: BLE001
            pass
        faulthandler.dump_traceback_later(_CAMERA_CLOSE_TIMEOUT_S, exit=True, file=hang_file)
    except Exception:                            # noqa: BLE001 — no log file: fall back to the console
        hang_file = None
        _arm_shutdown_watchdog(_CAMERA_CLOSE_TIMEOUT_S)
    if is_camera:
        print(f"[readymade] releasing the camera (if the driver hangs, the tracker ends itself after "
              f"{_CAMERA_CLOSE_TIMEOUT_S:.0f} s and the stacks go to {logdir}/)", flush=True)
    try:
        backend.close()
    finally:
        _disarm_shutdown_watchdog()
        if hang_file is not None:
            hang_file.close()
            try:
                os.remove(hang_path)             # it returned: nothing to report
            except OSError:
                pass


def serve(core: Core, *, logger: Any = None) -> int:
    """Start the input + pipeline background threads, then run the main thread as a
    bare headless supervisor idling to EOF. Stops + joins + closes on EOF / Ctrl-C;
    re-raises a captured pipeline fatal on the main thread."""
    pipe = threading.Thread(target=core.loop.run, name="pipeline", daemon=True)
    pipe.start()                                  # pipeline up first → blocked waiting on the slot
    it = InputThread(core.runner.produce, core.slot, name="input", logger=logger).start()
    t0 = time.perf_counter()
    try:
        for o in core.outputs:                    # start INSIDE the try so a failure still hits teardown.
            _safe_start(o, core.bus, logger)      # one output's bad port (e.g. mjpeg EADDRINUSE — usually
        for c in core.controls:                   # another tracker already running) must NOT crash the
            _safe_start(c, core.inbox, logger)    # tracker: log it loudly + keep the rest running.
        _serve_headless(core, it)                 # headless: the browser control surface is the only UI
    except KeyboardInterrupt:
        pass
    finally:
        _arm_shutdown_watchdog()                  # a hung camera driver must not wedge the process
        core.loop.stop()
        it.stop()
        core.slot.close()                         # release the input slot so the producer can exit
        for x in (*core.controls, *core.outputs):  # stop control-inputs first (no more posts), then outputs.
            try:                                   # guard each: a partial-start (start() raised) plugin's
                x.stop()                           # stop() must not abort the rest of teardown.
            except Exception as exc:               # noqa: BLE001
                (logger or NULL_LOGGER).warning(f"[readymade] {type(x).__name__}.stop() failed: {exc}")
        pipe.join(timeout=2.0)
        it.join(timeout=2.0)
        for th, nm in ((pipe, "pipeline"), (it, "input")):
            if th.is_alive():                     # a wedged thread outliving its join is about to have
                (logger or NULL_LOGGER).warning(  #   backend/mot/audit closed under it — say so instead
                    f"[readymade] {nm} thread did not stop within 2 s — closing resources under it")
        elapsed = time.perf_counter() - t0
        processed, dropped = core.loop.ticks, core.slot.dropped
        produced = processed + dropped
        achieved = processed / elapsed if elapsed > 0 else 0.0
        pct = f" ({100.0 * dropped / produced:.1f}%)" if produced else ""
        lo, avg, hi = core.loop.fps_stats
        print(f"[readymade] {processed} frames processed, {dropped} dropped{pct} - "
              f"pipeline fps min/avg/max {lo:.0f}/{avg:.0f}/{hi:.0f} (capacity) - "
              f"{achieved:.1f} fps (achieved)")
        if getattr(it, "error", None) is not None:     # the input thread died on a fatal, NOT at EOF —
            print(f"[readymade] ABNORMAL END: the input thread failed: {it.error!r}", file=sys.stderr)
        # CLOSE THE SINKS FIRST. audit.close() is what writes the "@ exit" profile block, and it
        # used to run AFTER the summary print — so on a console that could not encode the
        # summary (Windows cp1252 and the "→" in it), the print raised, the block was never
        # written, and every log came out truncated. Nothing that only DISPLAYS may come before
        # the writes that are the actual result.
        if core.audit is not None:
            core.audit.close()
        if core.mot is not None:
            core.mot.close()
        if core.profiler is not None:
            print(core.profiler.summary(overall_fps=achieved))
        if core.mot is not None:
            print(f"[readymade] MOT tracks -> {core.mot.path}")
        # the device LAST: everything that records the run is written + flushed before the one call
        # that can hang in native code (the watchdog's forced exit does not flush buffers)
        sys.stdout.flush()
        sys.stderr.flush()
        _close_backend_guarded(core.backend, is_camera=isinstance(core.backend, CameraBackend))
    if core.loop.error is not None:
        raise core.loop.error
    if core.control is not None and core.control.lifecycle == "restart":
        _reexec(logger)                               # teardown is done → replace this process (never returns)
    return 0


def _reexec(logger: Any = None) -> None:
    """Replace this process with a fresh tracker so a SAVEd needs-restart param takes effect: serve()'s
    teardown has already released the backend, sockets/ports, ML session and gui, so os.execv can rebind
    everything cleanly and re-read readymade.conf. Reuses argv (same source/flags). Never returns on success."""
    (logger or NULL_LOGGER).info("[readymade] restarting - re-exec")
    sys.stdout.flush()
    sys.stderr.flush()
    try:
        os.execv(sys.executable, [sys.executable, *sys.argv])
    except OSError as exc:                             # execv failed → don't leave a half-dead process
        (logger or NULL_LOGGER).warning(f"[readymade] re-exec failed: {exc}")


def _exhausted(core: Core, it: InputThread) -> bool:
    """EOF: the source is done AND the pipeline has drained the last frame."""
    return it.done and not core.slot.pending


def _serve_headless(core: Core, it: InputThread) -> None:
    """No gui: idle until the source is exhausted + drained, or the loop stops. Still applies any
    queued control (OSCIn pause/step/output on|off) each poll — control works headless too."""
    while not core.loop.stopped:
        if core.control is not None:
            core.control.drain(core.inbox)
            if core.control.lifecycle:                # /rmt/restart | /rmt/quit → teardown + act
                break
        if _exhausted(core, it):
            break
        time.sleep(0.02)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def _cli_overrides(args: argparse.Namespace) -> dict[str, dict[str, str]]:
    """Map CLI flags to a per-section override layer (strings — the schema coerces)."""
    out: dict[str, dict[str, str]] = {}
    inp: dict[str, str] = {}
    if args.source is not None:
        inp["source"], inp["location"] = args.source
    if args.fps is not None:
        inp["fps"] = args.fps
    if args.loop:                       # store_true: present → force loop on; absent → keep [input] loop
        inp["loop"] = "true"
    if inp:
        out["input"] = inp
    if args.cv2_threads is not None:    # --cv2-threads: sweep [perf] without editing a conf file
        out["perf"] = {"cv2_threads": str(args.cv2_threads)}
    pre = args.preprocess if args.preprocess is not None else (False if args.skip_preprocess else None)
    if pre is not None:                                # --preprocess / --no-preprocess (master switch);
        out["preprocess"] = {"preprocess": "true" if pre else "false"}   # --skip-preprocess = legacy alias
    # per-plugin enable/disable → flip the section's `enabled` key. None = not given (leave conf).
    # Overrides apply only to sections ALREADY in readymade.conf, so --tuio2/--gui/--mjpeg can only
    # FLIP a present section (a --enable of an absent section is a no-op; main() warns).
    for section in ("tuio2", "mjpeg", "oscout", "oscin", "mcu", "rec", "webgui", "features", "clusters",
                    "grouper"):
        val = getattr(args, section, None)
        if val is not None:
            out.setdefault(section, {})["enabled"] = "true" if val else "false"
    for sec, key, val in (args.set_params or ()):       # --set SECTION KEY VALUE: the /rmt/set grammar
        out.setdefault(sec, {})[key] = val              #   at boot — generic, repeatable, schema-coerced
    return out


def _plugin_defaults(conf_path: Path, overrides: dict[str, dict[str, str]]) -> dict[str, str]:
    """Each plugin ships its default ``<name>.conf`` next to it; ``readymade.conf``
    overrides per key. The input backend's conf is picked by ``source``."""
    src = "video"
    pre = load_config(conf_path, schemas=CONF_SCHEMAS, overrides=overrides)   # pass 1: read source
    if "input" in pre:
        src = pre["input"].get("source", "video")
    d: dict[str, str] = {}
    # each plugin lives in its OWN folder: plugins/<cat>/<name>/<name>.conf (matches the imports).
    for section, path in (("input", HERE / "plugins" / "input" / src / f"{src}.conf"),
                          ("tuio2", HERE / "plugins" / "output" / "tuio2" / "tuio2.conf"),
                          ("mjpeg", HERE / "plugins" / "output" / "mjpeg" / "mjpeg.conf"),
                          ("oscout", HERE / "plugins" / "output" / "oscout" / "oscout.conf"),
                          ("rec", HERE / "plugins" / "output" / "rec" / "rec.conf"),
                          ("webgui", HERE / "plugins" / "control" / "webgui" / "webgui.conf"),
                          ("oscin", HERE / "plugins" / "control" / "oscin" / "oscin.conf"),
                          ("mcu", HERE / "plugins" / "control" / "mcu" / "mcu.conf")):
        if path.is_file():
            d[section] = str(path)
    # camera keeps its config in a [camera] section (not [input]); ship its defaults
    # under whatever camera1.conf / readymade.conf provides (present-section-only, per the
    # loader). So `--source camera camera1.conf` inherits device/backend/auto_exposure.
    if src == "camera":
        cam = HERE / "plugins" / "input" / "camera" / "camera.conf"
        if cam.is_file():
            d["camera"] = str(cam)
    return d


def _apply_device_ranges(registry: dict, section: str, backend: Any,
                         values: dict, logger: Any) -> None:
    """Rewrite the [camera] control params' ``lo``/``hi`` in the registry with the OPEN device's REAL
    ranges (via ``backend.control_range``), and clamp each seeded current value into its range. So
    /rmt/query → the webgui bounds every control to what the device accepts (no out-of-range slider),
    the store clamps a set to it, and /query reports the value the device actually holds. Controls the
    backend can't range (or a non-UVC backend) are left at their generic ParamSpec bounds."""
    sec = registry.get(section)
    if not sec:
        return
    for key, p in list(sec.items()):
        try:
            rng = backend.control_range(key)
        except Exception:                        # noqa: BLE001 — ranging must never block startup
            rng = None
        if rng is None:
            continue
        lo, hi, _step = rng
        if hi <= lo:
            continue
        sec[key] = replace(p, lo=float(lo), hi=float(hi))
        cur = values.get((section, key))
        if isinstance(cur, (int, float)) and not isinstance(cur, bool):
            values[(section, key)] = min(max(int(cur), lo), hi)
        logger.info(f"[camera] {key} range {lo}..{hi} (device-reported)")


def _camera_layer(conf_path: Path, overrides: dict[str, dict[str, str]]) -> list[str]:
    """When the source is ``camera``, layer the ``location`` conf (the calibration
    tool's ``camera1.conf``) into the root: its ``[camera]`` + ``[preprocess]``
    override ``readymade.conf`` (and, being present in the root, materialize a
    ``[camera]`` section the manifest otherwise wouldn't have). The calibration
    tool writes it; the tracker layers it in. CLI ``overrides`` still win over it."""
    pre = load_config(conf_path, schemas=CONF_SCHEMAS, overrides=overrides)
    icfg = pre.get("input")
    if icfg is None or icfg.get("source") != "camera":
        return []
    loc = icfg.get("location")
    return [loc] if loc and Path(loc).is_file() else []


def _load(conf_path: Path, overrides: dict[str, dict[str, str]]) -> Config:
    """Layer: plugin ``<name>.conf`` defaults < ``readymade.conf`` < ``camera1.conf``
    (camera only) < CLI ``overrides``."""
    return load_config(conf_path, plugin_defaults=_plugin_defaults(conf_path, overrides),
                       schemas=CONF_SCHEMAS, overrides=overrides,
                       extra_roots=_camera_layer(conf_path, overrides))


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="readymade_tracker",
                                description="ReadyMade Tracker - settle-based partial-update object tracker")
    p.add_argument("conf", nargs="?", default=str(DEFAULT_CONF), help="path to readymade.conf")
    p.add_argument("--source", nargs=2, metavar=("KIND", "LOCATION"),
                   help="override the input: KIND (video) + LOCATION (a clip path)")
    p.add_argument("--fps", metavar="FPS", help="override [input] fps (0 = real-time)")
    p.add_argument("--loop", action="store_true",
                   help="loop a video file: rewind to the start at EOF ([input] source=video only)")
    p.add_argument("--preprocess", action=argparse.BooleanOptionalAction, default=None,
                   help="force the [preprocess] master switch: --no-preprocess = plain BGR->gray for a "
                        "conditioned source (e.g. a recording of the processed view)")
    p.add_argument("--skip-preprocess", dest="skip_preprocess", action="store_true",
                   help=argparse.SUPPRESS)      # deprecated alias for --no-preprocess
    p.add_argument("--paused", action="store_true",
                   help="start the source paused on frame 1")
    p.add_argument("--set", dest="set_params", action="append", nargs=3, default=None,
                   metavar=("SECTION", "KEY", "VALUE"),
                   help="override ONE conf value for this run (repeatable) — the /rmt/set grammar "
                        "at boot, e.g. --set efd bins 64; coerced/clamped by the same schema, and "
                        "it reaches restart-only knobs a live /rmt/set cannot")
    p.add_argument("--cv2-threads", dest="cv2_threads", type=int, default=None, metavar="N",
                   help="override [perf] cv2_threads for this run (0 = leave OpenCV alone). "
                        "How many threads OpenCV may split ONE operation across; "
                        "benchmarks/bench_threads.py sweeps it to find this machine's best")
    p.add_argument("--profile", action="store_true",
                   help="time each pipeline stage; print a summary on exit")
    p.add_argument("--max-frames", dest="max_frames", type=int, metavar="N",
                   help="stop after N processed frames (quick profiling / eye-checks)")
    p.add_argument("--debug", action="store_true",
                   help="audit lifecycle (mint/status/rewrite) -> logs/debug_<dt>_<src>.log")
    p.add_argument("--mot-export", dest="mot_export", nargs="?", const="", metavar="PATH",
                   help="write the tracker's confirmed tracks per frame in MOTChallenge format for standard "
                        "eval (default logs/mot_<dt>_<src>.txt; give PATH to override). Use with --pull so "
                        "no frame is skipped.")
    p.add_argument("--profile-every", dest="profile_every", type=int, default=100, metavar="N",
                   help="with --profile, dump the ms/frame table every N frames (default 100)")
    p.add_argument("--pull", action="store_true",
                   help="demand-driven input: grab-after-process, no drops, freshest frame")
    # per-plugin enable/disable — flip the section's `enabled` key (only if it's in readymade.conf).
    # BooleanOptionalAction gives --X / --no-X; default None = leave the conf's `enabled` alone.
    p.add_argument("--grouper", action=argparse.BooleanOptionalAction, default=None,
                   help="enable/disable the [grouper] voice-budget fusion (compounds replace members on the wires)")
    p.add_argument("--tuio2", action=argparse.BooleanOptionalAction, default=None,
                   help="enable/disable the [tuio2] TUIO 2.0 output")
    p.add_argument("--mjpeg", action=argparse.BooleanOptionalAction, default=None,
                   help="enable/disable the [mjpeg] MJPEG video output")
    p.add_argument("--oscout", action=argparse.BooleanOptionalAction, default=None,
                   help="enable/disable the [oscout] custom-OSC output (features/annotations + pose)")
    p.add_argument("--oscin", action=argparse.BooleanOptionalAction, default=None,
                   help="enable/disable the [oscin] OSC control input (pause/step/view/output on|off)")
    p.add_argument("--mcu", action=argparse.BooleanOptionalAction, default=None,
                   help="enable/disable the [mcu] serial control input (a board: T = trigger, D <0..1> = efd detail)")
    p.add_argument("--rec", action=argparse.BooleanOptionalAction, default=None,
                   help="enable/disable the [rec] video recorder (rec/stop -> recordings/<timestamp>.mp4)")
    p.add_argument("--webgui", action=argparse.BooleanOptionalAction, default=None,
                   help="enable/disable the [webgui] browser control surface + OSC<->HTTP bridge")
    p.add_argument("--features", action=argparse.BooleanOptionalAction, default=None,
                   help="enable/disable the [features] shape-descriptor stage (annotates obj.features)")
    p.add_argument("--clusters", action=argparse.BooleanOptionalAction, default=None,
                   help="enable/disable the [clusters] spatial-clustering stage (publishes the cluster table)")
    return p.parse_args(argv)


def _utf8_console() -> None:
    """Never let the terminal's encoding decide whether the tracker runs.

    The logs are already UTF-8 by construction; only the CONSOLE varies, and on Windows it
    defaults to cp1252, which cannot encode the box-drawing and arrow characters the profile
    summary is made of. `errors="replace"` is deliberate over a narrower fix: a diagnostic that
    prints a "?" is a diagnostic; one that raises is a crash in the middle of shutdown.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:                          # noqa: BLE001 — pipes/IDEs may not support it
            pass


def main(argv: list[str] | None = None) -> int:
    _utf8_console()
    print(f"ReadyMade Tracker v{__version__}")
    args = _parse_args(list(sys.argv[1:] if argv is None else argv))
    overrides = _cli_overrides(args)
    try:
        config = _load(Path(args.conf), overrides)
    except ConfigError as e:
        print(f"[readymade] config error: {e}", file=sys.stderr)
        return 2
    logger = Logger(level="info", stderr_level="warning")
    # surface conf keys no component declares — a stale/mistyped key is silently ignored otherwise, so
    # a live-tuning typo (or a key left behind by a removed feature) reads as "no effect" with no clue.
    for section in config.unknown:
        for key in config.unknown[section]:
            logger.warning(f"[conf] unknown key [{section}] {key} - ignored (not a live parameter)")
    # warn on a CLI enable that can't take effect: overrides only flip sections PRESENT in readymade.conf,
    # so --gui/--tuio2/--mjpeg on an absent section is a silent no-op otherwise.
    asked_on = set()
    asked_on |= {s for s in ("tuio2", "mjpeg", "oscout", "oscin", "mcu", "rec", "webgui", "features",
                             "clusters", "grouper")
                 if getattr(args, s, None) is True}
    for section in sorted(asked_on):
        if not config.is_enabled(section):
            logger.warning(f"[readymade] asked to enable [{section}] via CLI, but [{section}] is not in "
                           f"readymade.conf → no effect (add a [{section}] section to enable it)")
    try:
        core = build(config, logger=logger, paused=args.paused, profile=args.profile,
                     max_frames=args.max_frames,
                     debug=args.debug, profile_every=args.profile_every,
                     pull=args.pull, mot_export=args.mot_export,
                     conf_path=Path(args.conf), overrides=overrides)
    except Exception as e:                       # bad source / open failure — fatal
        print(f"[readymade] startup error: {type(e).__name__}: {e}", file=sys.stderr)
        logger.close()
        return 3
    icfg = config.get("input")
    print(f"[readymade] serving: source={icfg.get('source') if icfg else '?'} "
          f"location={icfg.get('location') if icfg else '?'} - Ctrl-C to stop.")
    if config.is_enabled("webgui"):              # no OS window, but there IS a UI — say where it lives
        wcfg = config.get("webgui")
        wport = int(wcfg.get("port", 8090)) if wcfg is not None else 8090
        print(f"[readymade] control surface: open http://localhost:{wport}/ in a browser")
    try:
        serve(core, logger=logger)
        return 0
    finally:
        logger.close()


if __name__ == "__main__":
    raise SystemExit(main())
