"""Per-plugin enable/disable: the [section] ``enabled`` conf key + the --X / --no-X CLI toggles
(tuio2, mjpeg, webgui). The ``enabled`` key is handled by the config loader (popped + coerced), and
the CLI flags flip it via the override layer — which only touches sections PRESENT in readymade.conf."""

import readymade_tracker
from lib.config import build_config


def _cfg(root, overrides=None):
    return build_config(root, schemas=readymade_tracker.CONF_SCHEMAS, overrides=overrides or {})


# --------------------------------------------------------------------------- #
# the conf `enabled` key
# --------------------------------------------------------------------------- #

def test_enabled_false_in_conf_disables_a_present_section():
    assert not _cfg({"tuio2": {"enabled": "false", "destinations": "127.0.0.1:3333"}}).is_enabled("tuio2")
    assert _cfg({"tuio2": {"destinations": "127.0.0.1:3333"}}).is_enabled("tuio2")        # present, no key → on
    assert not _cfg({"webgui": {"enabled": "false"}}).is_enabled("webgui")
    assert not _cfg({"mjpeg": {"enabled": "off", "port": "8080"}}).is_enabled("mjpeg")   # bool-coerced


def test_enabled_is_meta_not_a_section_value():
    # `enabled` is popped before coercion → it is NOT exposed among the Section's values
    cfg = _cfg({"mjpeg": {"enabled": "true", "port": "8080"}})
    assert cfg.is_enabled("mjpeg")
    assert "enabled" not in cfg.get("mjpeg") and cfg.get("mjpeg")["port"] == 8080


# --------------------------------------------------------------------------- #
# unknown-key surfacing: a conf key no component declares is silently ignored (still
# passed through as str), so a stale/typo'd operator key reads as "no effect". Config
# records it on `.unknown` so the orchestrator can warn once at startup.
# --------------------------------------------------------------------------- #

def test_unknown_operator_key_is_surfaced_but_still_passes_through():
    cfg = _cfg({"clusters": {"frobnicate": "7"}})
    assert cfg.unknown.get("clusters") == ("frobnicate",)      # announced for the startup warning
    assert cfg.get("clusters")["frobnicate"] == "7"            # zero behavior change — still loaded as str


def test_declared_key_and_enabled_are_never_unknown():
    cfg = _cfg({"tuio2": {"enabled": "true", "destinations": "127.0.0.1:3333"}})
    assert "tuio2" not in cfg.unknown          # `destinations` is a declared param; `enabled` is meta


def test_unknown_judges_the_operator_layer_only_not_cli_overrides():
    # only the operator's conf (root) is judged — a CLI override key is not a live-tuning typo
    cfg = _cfg({"tuio2": {"destinations": "127.0.0.1:3333"}}, overrides={"tuio2": {"frobnicate": "1"}})
    assert "tuio2" not in cfg.unknown


# --------------------------------------------------------------------------- #
# the CLI toggles
# --------------------------------------------------------------------------- #

def test_no_flag_leaves_no_override():
    ov = readymade_tracker._cli_overrides(readymade_tracker._parse_args([]))
    assert "gui" not in ov and "tuio2" not in ov and "mjpeg" not in ov


def test_disable_flags_map_to_enabled_false():
    assert readymade_tracker._cli_overrides(readymade_tracker._parse_args(["--no-tuio2"]))["tuio2"] == {"enabled": "false"}
    assert readymade_tracker._cli_overrides(readymade_tracker._parse_args(["--no-webgui"]))["webgui"] == {"enabled": "false"}
    assert readymade_tracker._cli_overrides(readymade_tracker._parse_args(["--no-mjpeg"]))["mjpeg"] == {"enabled": "false"}


def test_enable_flags_map_to_enabled_true():
    assert readymade_tracker._cli_overrides(readymade_tracker._parse_args(["--tuio2"]))["tuio2"] == {"enabled": "true"}
    assert readymade_tracker._cli_overrides(readymade_tracker._parse_args(["--webgui"]))["webgui"] == {"enabled": "true"}


def test_disable_override_switches_off_a_present_section():
    ov = readymade_tracker._cli_overrides(readymade_tracker._parse_args(["--no-tuio2"]))
    assert not _cfg({"tuio2": {"destinations": "127.0.0.1:3333"}}, ov).is_enabled("tuio2")


def test_enable_override_flips_a_disabled_section_on():
    ov = readymade_tracker._cli_overrides(readymade_tracker._parse_args(["--tuio2"]))
    assert _cfg({"tuio2": {"enabled": "false", "destinations": "127.0.0.1:3333"}}, ov).is_enabled("tuio2")


def test_enable_override_cannot_materialize_an_absent_section():
    # overrides touch only present sections — --webgui on a config without [webgui] is a no-op (main warns)
    ov = readymade_tracker._cli_overrides(readymade_tracker._parse_args(["--webgui"]))
    assert not _cfg({"input": {"source": "video"}}, ov).is_enabled("webgui")


def test_cli_set_is_a_generic_conf_override():
    """--set SECTION KEY VALUE — the /rmt/set grammar at boot: repeatable, lands in the override
    layer as strings for the schema to coerce, and reaches restart-only knobs (e.g. [efd] bins)."""
    ov = readymade_tracker._cli_overrides(readymade_tracker._parse_args(
        ["--set", "efd", "bins", "64", "--set", "grouper", "enabled", "true"]))
    assert ov["efd"] == {"bins": "64"}
    assert ov["grouper"]["enabled"] == "true"


def test_cli_grouper_flag_flips_the_section():
    ov = readymade_tracker._cli_overrides(readymade_tracker._parse_args(["--no-grouper"]))
    assert ov["grouper"]["enabled"] == "false"
