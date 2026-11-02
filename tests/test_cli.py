"""CLI flags — the preprocess master-switch pair + its legacy alias."""


def test_no_preprocess_flag_and_legacy_alias():
    # --preprocess/--no-preprocess force the [preprocess] master switch (BooleanOptionalAction,
    # consistent with the plugin toggles); --skip-preprocess survives as a hidden legacy alias.
    import readymade_tracker as rt
    for argv, want in ((["c", "--no-preprocess"], "false"), (["c", "--preprocess"], "true"),
                       (["c", "--skip-preprocess"], "false"), (["c"], None)):
        ov = rt._cli_overrides(rt._parse_args(argv))
        assert (ov.get("preprocess") or {}).get("preprocess") == want
