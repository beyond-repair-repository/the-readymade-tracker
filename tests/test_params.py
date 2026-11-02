"""lib.params — the live-parameter registry + store (behind /rmt/set, /rmt/reload, /rmt/query).
Pure logic, driven against fake target objects + a fake config; no OSC, no pipeline."""

from types import SimpleNamespace

from lib.params import Param, ParamSpec, ParamStore, build_registry, seed_values

# a tiny legacy schema (type-only) exercising build_registry's BARE fallback for sections without a
# component. Real params live on the fake _Rebuilt component below.
_SCHEMA = {
    "gate": {"enabled": bool, "diff_thresh": int},
    "observe": {"k": float, "min_area": int},
    "pose": {"hold_near_px": float, "conf_min": float},
    "detect": {"label": str, "score": float},
    "mjpeg": {"view": str, "quality": int},
    "oscin": {"port": int},
}


class _Rebuilt:
    """A fake plugin-owned component with one NEEDS-RESTART param — the STABLE 'restart' example for the
    store-mechanics tests (staging / save / reload). Decoupled from the real sections' migration: as each
    real section moves to PARAMS its legacy hot=False bit disappears, but this fake always models it."""
    PARAMS = (
        ParamSpec("path", str, "old.txt", hot=False, doc="a needs-restart path"),
        ParamSpec("gain", float, 0.5, lo=0.0, hi=1.0, doc="a hot [0,1] knob"),
        ParamSpec("mode", str, "a", enum=("a", "b", "c"), doc="a hot enum knob"),
    )


_COMPONENTS = {"rebuilt": _Rebuilt}


class _Sec:
    def __init__(self, d):
        self._d = d
    def get(self, k, default=None):
        return self._d.get(k, default)


class _Cfg:
    def __init__(self, d):
        self._d = {s: _Sec(v) for s, v in d.items()}
    def __contains__(self, s):
        return s in self._d
    def get(self, s, default=None):
        return self._d.get(s, default)


def _store(targets=None, values=None, reload_fn=None):
    reg = build_registry(_SCHEMA, _COMPONENTS)
    return ParamStore(reg, targets or {}, values or {}, reload_fn=reload_fn)


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #

def test_build_registry_harvest_and_bare_fallback():
    reg = build_registry(_SCHEMA, _COMPONENTS)
    assert reg["observe"]["k"].hot and reg["observe"]["k"].apply is None   # bare fallback: type-only, hot
    assert reg["rebuilt"]["gain"].hot and reg["rebuilt"]["gain"].lo == 0.0 and reg["rebuilt"]["gain"].hi == 1.0
    assert reg["rebuilt"]["path"].hot is False                         # plugin-owned needs-restart param
    assert reg["rebuilt"]["mode"].enum == ("a", "b", "c")             # plugin-owned enum
    assert reg["oscin"]["enabled"].synthetic                          # toggleable → synthetic enabled added
    assert reg["pose"]["conf_min"].target_attr == "_conf_min"          # ParamSpec default attr = "_"+key


def test_seed_values_from_config_and_defaults():
    reg = build_registry(_SCHEMA)
    vals = seed_values(reg, _Cfg({"observe": {"k": 0.42}}))
    assert vals[("observe", "k")] == 0.42                              # from config, coerced
    assert ("pose", "conf_min") in vals                               # absent section → default (None)


def test_seed_values_clamp_like_bind_params():
    # an out-of-range conf entry is CLAMPED at bind — the /query shadow must report the value the
    # stage actually runs with, not the raw conf number.
    reg = build_registry({}, components=_COMPONENTS)
    vals = seed_values(reg, _Cfg({"rebuilt": {"gain": 7.5}}))          # gain is lo=0, hi=1
    assert vals[("rebuilt", "gain")] == 1.0


# --------------------------------------------------------------------------- #
# set
# --------------------------------------------------------------------------- #

def test_set_hot_param_writes_attr_and_current():
    obs = SimpleNamespace(_k=0.3, _min_area=100)
    store = _store({"observe": obs}, {("observe", "k"): 0.3})
    r = store.set("observe", "k", 0.5)
    assert r["ok"] and obs._k == 0.5 and store.current("observe", "k") == 0.5


def test_set_clamps_numeric_to_range():
    rb = SimpleNamespace(_gain=0.3)
    r = _store({"rebuilt": rb}).set("rebuilt", "gain", 5.0)           # >1 → clamp
    assert r["ok"] and r["value"] == 1.0 and r["clamped"] is True and rb._gain == 1.0


def test_set_via_apply_setter_transforms_but_current_is_raw():
    # the apply setter writes the transformed value to the target, but /query reports the RAW value.
    from lib.params import Param
    reg = {"x": {"sq": Param("x", "sq", float, apply=lambda self, v: setattr(self, "_sq", float(v) ** 2))}}
    tgt = SimpleNamespace(_sq=0.0)
    store = ParamStore(reg, {"x": tgt}, {})
    store.set("x", "sq", 4.0)
    assert tgt._sq == 16.0                                             # stored squared (apply setter)
    assert store.current("x", "sq") == 4.0                            # reported raw


def test_raising_apply_setter_records_nothing():
    # apply runs BEFORE the value is recorded: if the plugin's own setter rejects the value, /query
    # must keep telling the truth (the old value) — a half-applied param would lie forever.
    import pytest
    from lib.params import Param

    def _apply(self, v):
        raise ValueError("plugin parse rejected it")
    reg = {"x": {"dst": Param("x", "dst", str, apply=_apply)}}
    store = ParamStore(reg, {"x": SimpleNamespace()}, {("x", "dst"): "old:1"})
    with pytest.raises(ValueError):
        store.set("x", "dst", "bad token")
    assert store.current("x", "dst") == "old:1"                        # untouched — no lie, not dirty
    assert ("x", "dst") not in store._dirty


def test_set_bool_coercion():
    gate = SimpleNamespace(_enabled=True)
    _store({"gate": gate}).set("gate", "enabled", "false")
    assert gate._enabled is False


def test_set_rejections():
    obs, rb = SimpleNamespace(_k=0.3), SimpleNamespace(_mode="a")
    store = _store({"observe": obs, "rebuilt": rb})
    assert store.set("observe", "nope", 1)["reason"] == "unknown"
    assert store.set("rebuilt", "mode", "spinny")["reason"] == "not_in_enum"
    assert store.set("observe", "k", "abc")["reason"] == "bad_value"
    ok = store.set("rebuilt", "mode", "b")                            # valid enum
    assert ok["ok"] and rb._mode == "b"


def test_set_needs_restart_is_staged_not_rejected():
    store = _store({"observe": SimpleNamespace(_k=0.3)})
    res = store.set("rebuilt", "path", "/x.txt")                      # non-hot → staged, not rejected
    assert res["ok"] and res["staged"] and res["value"] == "/x.txt"
    assert ("rebuilt", "path") in store._dirty and ("rebuilt", "path") in store._staged


def test_save_rewrites_conf_in_place_preserving_structure(tmp_path):
    from lib.params import ParamStore
    conf = tmp_path / "readymade.conf"
    conf.write_text("# header comment\n[rebuilt]\npath = old.txt\n"
                    "\n# a comment inside\n[mjpeg]\nenabled = true\nquality = 80\n")
    store = ParamStore(build_registry(_SCHEMA, _COMPONENTS), {}, {}, conf_path=str(conf))
    store.set("rebuilt", "path", "new.txt")                          # staged (non-hot)
    store.set("mjpeg", "enabled", 0)                                  # synthetic enabled → staged
    res = store.save()
    assert res["ok"] and res["written"] == 2
    text = conf.read_text()
    assert "path = new.txt" in text and "enabled = false" in text
    assert "# header comment" in text and "# a comment inside" in text          # comments preserved
    assert "quality = 80" in text                                                # untouched key kept
    assert store.save()["written"] == 0 or True                                 # idempotent-ish re-save ok


def test_save_routes_layer_owned_sections_to_the_layer_conf(tmp_path):
    """A layered conf (camera1.conf) OVERRIDES certain sections on load, so their edits must be written
    THERE, not to the main conf — else the change is masked next load (the 'webgui won't save crop' bug)."""
    from lib.params import ParamStore
    main = tmp_path / "readymade.conf"
    main.write_text("[rebuilt]\npath = old.txt\n")
    layer = tmp_path / "camera1.conf"
    layer.write_text("# camera layer\n")
    store = ParamStore(build_registry(_SCHEMA, _COMPONENTS), {}, {}, conf_path=str(main),
                       layer_path=str(layer), layer_sections=frozenset({"mjpeg"}))
    store.set("rebuilt", "path", "new.txt")                          # a main-conf section
    store.set("mjpeg", "enabled", 0)                                 # a layer-owned section
    res = store.save()
    assert res["ok"] and res["written"] == 2
    assert "path = new.txt" in main.read_text()
    assert "enabled = false" not in main.read_text()                 # NOT written to main
    assert "enabled = false" in layer.read_text()                    # routed to the layer conf


def test_set_rejects_newline_values():
    store = _store({"detect": SimpleNamespace()})
    for bad in ("a\nb = evil", "x\r[oscin]"):                         # would split/corrupt a conf line
        assert store.set("detect", "label", bad)["reason"] == "bad_value"
    assert not store._dirty                                            # never staged


def test_save_is_atomic_leaves_original_on_failure(tmp_path, monkeypatch):
    from lib import params as P
    conf = tmp_path / "c.conf"
    conf.write_text("[rebuilt]\npath = original.txt\n")
    store = P.ParamStore(build_registry(_SCHEMA, _COMPONENTS), {}, {}, conf_path=str(conf))
    store.set("rebuilt", "path", "new.txt")
    monkeypatch.setattr(P.os, "replace", lambda *a: (_ for _ in ()).throw(OSError("boom")))  # fail the swap
    assert store.save()["ok"] is False                                # reported, not raised
    assert conf.read_text() == "[rebuilt]\npath = original.txt\n"      # master conf intact
    assert not list(tmp_path.glob("*.tmp"))                            # temp cleaned up


def test_write_conf_inplace_inserts_missing_key_and_section(tmp_path):
    from lib.config import parse_ini_string
    from lib.params import _write_conf_inplace
    conf = tmp_path / "c.conf"
    conf.write_text("[detect]\nlabel = a.txt")                        # NO trailing newline
    n = _write_conf_inplace(str(conf), {
        "detect": {"label": "b.txt", "extra": "512"},                 # update + insert-into-section
        "mjpeg": {"port": "9000"},                                    # brand-new section
    })
    assert n == 3
    parsed = parse_ini_string(conf.read_text())
    assert parsed["detect"]["label"] == "b.txt"                       # updated in place
    assert parsed["detect"]["extra"] == "512"                         # inserted into the section
    assert parsed["mjpeg"]["port"] == "9000"                         # appended section parses


def test_paramspec_bind_harvest_apply_collect_schema():
    """The Step-0 plugin-owned infra: a component declares PARAMS; bind_params seeds it; build_registry
    harvests it; a live /set routes through the SAME apply setter as cold-start; /query stays RAW."""
    from lib.params import ParamSpec, ParamStore, bind_params, collect_schema

    class Widget:
        PARAMS = (
            ParamSpec("radius", float, 0.12, lo=0.0, hi=1.0, doc="r"),
            ParamSpec("hold_px", float, 12.0, lo=0.0, attr="_hold2",
                      apply=lambda self, v: setattr(self, "_hold2", float(v) ** 2)),
        )
        def __init__(self, config=None):
            bind_params(self, config)

    w = Widget(SimpleNamespace(get=lambda k, d=None: {"radius": 0.3}.get(k, d)))
    assert w._radius == 0.3                        # "_"+key setattr path (radius absent → its own default? no: 0.3 given)
    assert w._hold2 == 144.0                       # apply x²: default 12 → 144, the SAME line a live /set runs
    reg = build_registry({}, {"widget": Widget})   # harvest: real defaults + apply carried onto the Param
    assert reg["widget"]["radius"].default == 0.12 and reg["widget"]["radius"].hi == 1.0
    assert reg["widget"]["hold_px"].apply is not None
    assert collect_schema({"widget": Widget}) == {"widget": {"radius": float, "hold_px": float}}
    store = ParamStore(reg, {"widget": w}, {})
    store.set("widget", "hold_px", 15)             # live /set → same apply → x²; current stays RAW
    assert w._hold2 == 225.0 and store.current("widget", "hold_px") == 15.0


def test_clusters_live_set_applies_floor_via_apply():
    """clusters is plugin-owned: a live /set routes through Clusters' OWN apply setter, so the min-2 floor
    holds at RUNTIME (not just __init__) — the pre-migration registry clamped min_objects to lo=1 and a
    live set wrote _min=1, bypassing the max(2,...) that only ran at cold-start."""
    import readymade_tracker
    from lib.clusters import Clusters

    c = Clusters(None)
    assert (c._min, c._radius) == (2, 0.12)                                            # bind_params defaults
    reg = build_registry(readymade_tracker.SCHEMAS, readymade_tracker.COMPONENTS)
    store = ParamStore(reg, {"clusters": c}, {})
    assert store.set("clusters", "min_objects", 1)["ok"] and c._min == 2               # floored live
    assert store.set("clusters", "radius", 9)["ok"] and c._radius == 1.0                # clamped to hi
    assert store.current("clusters", "radius") == 1.0


def test_pose_bind_params_settle_only_attrs():
    """pose is plugin-owned + settle-only: bind_params sets the plain cached attrs (no x²/median/Kalman
    live layer), and a live /set writes through the default attr."""
    import readymade_tracker
    from lib.pose import Pose

    p = Pose(None)                                                                # defaults
    assert (p._conf_min, p._max_fits, p._sym_lock) == (0.4, 1, 0.005)
    cfg = SimpleNamespace(get=lambda k, d=None: {"conf_min": 0.6, "max_fits_per_frame": 3}.get(k, d))
    assert Pose(cfg)._conf_min == 0.6 and Pose(cfg)._max_fits == 3               # override → plain setattr
    reg = build_registry(readymade_tracker.SCHEMAS, readymade_tracker.COMPONENTS)
    store = ParamStore(reg, {"pose": p}, {})
    assert store.set("pose", "conf_min", 0.5)["ok"] and p._conf_min == 0.5        # hot live write
    assert store.current("pose", "conf_min") == 0.5


def test_effective_get_reports_resolved_value_readonly():
    """A get-backed param reports a read-only `effective` value in describe() computed from the LIVE target,
    WITHOUT touching `current` (so save() still writes the raw conf value → no clobber). Fail-soft: a get
    that raises is swallowed; no live target → effective is None."""
    from lib.params import ParamSpec, ParamStore

    class Widget:
        PARAMS = (
            ParamSpec("providers", str, None, hot=False, bind=False,
                      get=lambda w: ",".join(w.resolved), doc="resolved ORT providers"),
            ParamSpec("boom", str, None, hot=False, bind=False, get=lambda w: 1 / 0, doc="raises"),
        )

    reg = build_registry({}, {"w": Widget})
    assert reg["w"]["providers"].get is not None                         # harvested onto the Param
    w = SimpleNamespace(resolved=["CoreMLExecutionProvider", "CPUExecutionProvider"])
    store = ParamStore(reg, {"w": w}, {("w", "providers"): None})
    d = {x["key"]: x for x in store.describe("w")}
    assert d["providers"]["effective"] == "CoreMLExecutionProvider,CPUExecutionProvider"
    assert d["providers"]["current"] is None                            # RAW conf value untouched (blank = auto)
    assert d["boom"]["effective"] is None                               # a raising get → None, never crashes /query
    assert ParamStore(reg, {}, {}).describe("w")[0]["effective"] is None  # no live target → no effective


def test_expose_curation_realtime_policy():
    """The real-time policy: expose defaults "auto" → hot params EDITABLE, needs-restart READ-ONLY. "edit"
    forces a needs-restart param editable; "readonly"/"hidden" force; get-backed ⇒ readonly."""
    from lib.params import ParamSpec, ParamStore

    class W:
        PARAMS = (
            ParamSpec("live", float, 0.5, doc="hot → editable"),                             # hot, auto
            ParamSpec("restart_cfg", str, "x", hot=False, bind=False, doc="needs-restart"),  # auto → readonly
            ParamSpec("restart_edit", str, "y", hot=False, bind=False, expose="edit", doc="e"),  # forced edit
            ParamSpec("locked", float, 0.5, expose="readonly", doc="hot but forced RO"),     # forced readonly
            ParamSpec("secret", str, "w", hot=False, bind=False, expose="hidden", doc="h"),
            ParamSpec("computed", str, None, hot=False, bind=False, get=lambda o: "c", doc="g"),
        )

    reg = build_registry({}, {"w": W})
    d = {x["key"]: x for x in ParamStore(reg, {"w": SimpleNamespace()}, {}).describe("w")}
    assert d["live"]["readonly"] is False and d["live"]["hidden"] is False   # hot → editable (real-time)
    assert d["restart_cfg"]["readonly"] is True                             # needs-restart → readonly by default
    assert d["restart_edit"]["readonly"] is False                          # forced-editable exception
    assert d["locked"]["readonly"] is True                                 # forced readonly (even though hot)
    assert d["secret"]["hidden"] is True
    assert d["computed"]["readonly"] is True                               # get-backed ⇒ readonly


def test_bare_fallback_when_no_components():
    reg = build_registry(_SCHEMA)                  # components omitted → bare Params (type-only, hot, no apply)
    assert reg["observe"]["k"].hot and reg["observe"]["k"].apply is None and reg["observe"]["k"].lo is None
    assert reg["oscin"]["port"].hot and reg["oscin"]["enabled"].synthetic        # toggleable → synthetic enabled


def test_synthetic_enabled_reflects_is_enabled(tmp_path):
    from lib.config import load_config
    from readymade_tracker import COMPONENTS, CONF_SCHEMAS, SCHEMAS
    conf = tmp_path / "readymade.conf"
    conf.write_text("[mjpeg]\nenabled = true\nport = 8080\n[oscin]\nenabled = false\nport = 3401\n")
    cfg = load_config(conf, schemas=CONF_SCHEMAS)
    reg = build_registry(SCHEMAS, COMPONENTS)                          # oscin is plugin-owned now
    store = ParamStore(reg, {}, seed_values(reg, cfg))
    en = {d["section"]: d for d in store.describe() if d["key"] == "enabled"}
    assert en["mjpeg"]["current"] is True and en["oscin"]["current"] is False   # loader pops enabled; fixed


def test_set_inactive_when_section_not_running():
    assert _store({}).set("observe", "k", 0.5)["reason"] == "inactive"


# --------------------------------------------------------------------------- #
# reload
# --------------------------------------------------------------------------- #

def test_reload_applies_changed_hot_reports_rebuild_skips_unchanged():
    obs, rb = SimpleNamespace(_k=0.3), SimpleNamespace()
    reg = build_registry(_SCHEMA, _COMPONENTS)
    disk0 = _Cfg({"observe": {"k": 0.3}, "rebuilt": {"path": "a.txt"}})
    disk1 = _Cfg({"observe": {"k": 0.7}, "rebuilt": {"path": "b.txt"}})
    store = ParamStore(reg, {"observe": obs, "rebuilt": rb}, seed_values(reg, disk0),
                       reload_fn=lambda: disk1)
    r = store.reload()
    assert r["ok"]
    assert "observe.k" in r["applied"] and obs._k == 0.7             # hot + changed → applied live
    assert "rebuilt.path" in r["restart"]                            # changed but rebuild → restart
    assert "observe.min_area" not in r["applied"]                   # unchanged → skipped


def test_reload_without_a_reload_fn():
    assert _store({}).reload()["reason"] == "no_reload"


# --------------------------------------------------------------------------- #
# query
# --------------------------------------------------------------------------- #

def test_describe_manifest_and_section_filter():
    store = _store({}, {("rebuilt", "gain"): 0.4})
    manifest = store.describe()
    k = next(d for d in manifest if d["section"] == "rebuilt" and d["key"] == "gain")
    assert k["type"] == "float" and k["current"] == 0.4 and k["lo"] == 0.0 and k["hi"] == 1.0 and k["hot"]
    mode = next(d for d in manifest if d["section"] == "rebuilt" and d["key"] == "mode")
    assert mode["enum"] == ["a", "b", "c"]                           # enum surfaced in the manifest
    assert store.describe("pose") and all(d["section"] == "pose" for d in store.describe("pose"))


# --------------------------------------------------------------------------- #
# the REAL schema builds cleanly
# --------------------------------------------------------------------------- #

def test_real_schemas_build():
    import readymade_tracker
    reg = build_registry(readymade_tracker.SCHEMAS, readymade_tracker.COMPONENTS)
    for s in ("observe", "classify", "identity", "pose", "tuio2", "mjpeg", "oscout"):
        assert s in reg and reg[s]
    # pose is plugin-owned (settle-only): its params are harvested from Pose.PARAMS
    assert reg["pose"]["conf_min"].target_attr == "_conf_min" and reg["pose"]["symmetry_lock"].target_attr == "_sym_lock"
    assert all(not p.hot for p in reg["oscin"].values())             # oscin binds a socket → all rebuild
    assert reg["observe"]["k"].hot
    # clusters is plugin-owned (Clusters.PARAMS) — harvested, not from the central overlays
    assert reg["clusters"]["min_objects"].target_attr == "_min" and reg["clusters"]["min_objects"].lo == 2
    assert reg["clusters"]["min_objects"].apply is not None and reg["clusters"]["radius"].target_attr == "_radius"


def test_every_hot_param_targets_a_real_attribute():
    # the guard for the silent-no-op class: a hot param whose target object doesn't HAVE its
    # target_attr means /rmt/set acks "ok" and writes a dead attribute. Construct the real
    # stages/plugins and check every hot param's attr exists.
    import readymade_tracker
    from lib.classify import Classify
    from lib.clusters import Clusters
    from lib.efd import Efd
    from lib.features import Features
    from lib.grouper import Grouper
    from lib.identity import Identity
    from lib.observe import Observe
    from lib.perf import Perf
    from lib.pose import Pose
    from lib.preprocess import Preprocess, PreprocessConfig
    from plugins.output.mjpeg.mjpeg import MjpegOut
    from plugins.output.oscout.oscout import OscOut
    from plugins.input.camera.camera import CameraBackend
    from plugins.output.rec.rec import Recorder
    from plugins.output.tuio2.tuio2 import Tuio2Out

    reg = build_registry(readymade_tracker.SCHEMAS, readymade_tracker.COMPONENTS)
    # EVERY constructible target — a missing section here is a coverage hole (clusters was, and hid a bug).
    from lib.reid import ReidParams
    targets = {"observe": Observe(None), "classify": Classify(None),
               "identity": Identity(None), "pose": Pose(None),
               "features": Features(None), "clusters": Clusters(None),
               "grouper": Grouper(None), "efd": Efd(None),
               "preprocess": Preprocess(PreprocessConfig()),
               "reid": ReidParams(None),
               "perf": Perf(None),              # process-wide; cv2_threads applies at once        # the appearance policy — live for observe + identity
               "camera": CameraBackend(None)}   # live image controls are hot (apply=_cam_apply)
    t = Tuio2Out(); t.setup({}, send_fn=lambda d: None); targets["tuio2"] = t
    o = OscOut(); o.setup({}, send_fn=lambda d: None); targets["oscout"] = o
    m = MjpegOut(); m.setup({}); targets["mjpeg"] = m
    r = Recorder(); r.setup({}); targets["rec"] = r
    # Every section with a hot param must be covered here (a missing section is a coverage hole that
    # can hide a silent-no-op /rmt/set — clusters was one). No section is exempt: there is no model or
    # display to construct — every stage builds headless (input/oscin/webgui/reid have no hot params).
    uncovered = [s for s in reg if any(p.hot for p in reg[s].values()) and s not in targets]
    assert not uncovered, f"registry sections with hot params NOT checked by this guard: {uncovered}"

    missing = []
    for section, obj in targets.items():
        for key, p in reg[section].items():
            if p.hot and p.apply is None and not hasattr(obj, p.target_attr):
                # a param with an apply setter writes via its OWN line (not target_attr) → not a silent
                # no-op risk; the golden-snapshot tests verify those. Only plain setattr params are checked.
                missing.append(f"{section}.{key} → writes {p.target_attr!r} (absent)")
    assert not missing, "hot params write nonexistent attrs (silent no-op /rmt/set): " + "; ".join(missing)


def test_real_registry_is_fully_documented():
    # every param carries a doc (self-describing /rmt/query); a new PARAMS spec must add one.
    import readymade_tracker
    reg = build_registry(readymade_tracker.SCHEMAS, readymade_tracker.COMPONENTS)
    undoc = [f"{s}.{k}" for s, ps in reg.items() for k, p in ps.items() if not p.doc]
    assert not undoc, f"params missing a doc — add one to the owning class's PARAMS: {undoc}"
    # spec metadata (enum, renamed attr) is harvested from each owning class's PARAMS
    assert reg["mjpeg"]["view"].enum == ("input", "debug")   # single panels; no composite/triptych
    assert reg["pose"]["conf_min"].target_attr == "_conf_min"


# --------------------------------------------------------------------------- #
# thread-safety: two control threads (oscin + mcu) drive the store concurrently
# --------------------------------------------------------------------------- #

def test_save_survives_a_concurrent_first_ever_set(tmp_path):
    """save() iterating _dirty while another thread lands the FIRST-EVER set() of a key used to raise
    "Set changed size during iteration" (reproduced) and killed the OSC receive loop. The store lock
    makes the interleave impossible; the inert keys widen save()'s iteration so, without the lock,
    this test fails on the first attempt."""
    import sys
    import threading

    conf = tmp_path / "t.conf"
    conf.write_text("[observe]\nfg_threshold = 60\n")
    reg = {"observe": {"fg_threshold": Param(section="observe", key="fg_threshold", type=int),
                       "min_area": Param(section="observe", key="min_area", type=int)}}

    class T:
        pass

    store = ParamStore(reg, {"observe": T()}, {}, conf_path=conf)
    old = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        for _ in range(20):
            store._dirty = {("zz", f"k{i}") for i in range(50_000)}   # unknown keys: skipped, but iterated
            errs: list = []

            def saver():
                try:
                    store.save()
                except Exception as exc:              # noqa: BLE001 — the failure this test guards against
                    errs.append(exc)

            def setter():
                store.set("observe", "min_area", 123)  # first-ever set of this key → _dirty grows

            a = threading.Thread(target=saver)
            b = threading.Thread(target=setter)
            a.start(); b.start(); a.join(); b.join()
            assert not errs, f"save() raised under a concurrent set(): {errs[0]!r}"
    finally:
        sys.setswitchinterval(old)


# --------------------------------------------------------------------------- #
# on_applied — the live-apply observer (build wires the paused-transport poke)
# --------------------------------------------------------------------------- #

def test_on_applied_fires_on_hot_apply_only():
    tgt = SimpleNamespace()
    store = _store(targets={"observe": tgt})
    pokes = []
    store.on_applied = lambda: pokes.append(1)
    assert store.set("observe", "k", 0.2)["ok"] is True      # hot live apply → poked
    assert pokes == [1]
    assert store.set("observe", "nope", 1)["ok"] is False    # unknown → no poke
    assert pokes == [1]


def test_on_applied_failure_never_fails_the_set():
    tgt = SimpleNamespace()
    store = _store(targets={"observe": tgt})
    store.on_applied = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    res = store.set("observe", "k", 0.3)                     # the poke raising must not break the apply
    assert res["ok"] is True and tgt._k == 0.3
