"""Derived comparison tables. Every number here must be reproducible from per-run files alone."""

import math

import pytest

from hallm.reports import (hc_verdict, hl_verdict, hs_verdict, index_runs, mean_se, paired,
                            parse_probe_id, parse_run_id, regress, tax)


def test_parse_run_id_and_reject_malformed():
    assert parse_run_id("L16-A2-s1339") == {"depth": 16, "arm": "A2", "seed": 1339}
    assert parse_run_id("L8-A2ffn-s1337")["arm"] == "A2ffn"
    assert parse_run_id("smoke-A0-s7") is None
    assert parse_run_id("nonsense") is None


def test_index_runs_groups_arms_by_rung_and_seed():
    rows = [{"run": "L8-A0-s1337", "test_ppl": 26.06}, {"run": "L8-A2-s1337", "test_ppl": 29.68},
            {"run": "L8-A2ffn-s1337", "test_ppl": 27.0}]
    idx = index_runs(rows)
    assert set(idx[(8, 1337)]) == {"A0", "A2", "A2ffn"}


def test_tax_matches_the_published_experiment_1_number():
    assert abs(tax(29.68, 26.06) - 13.891) < 0.01


def test_mean_se_single_value_has_no_standard_error():
    m, se = mean_se([13.9])
    assert m == 13.9 and math.isnan(se)


def test_regress_recovers_a_known_negative_slope():
    xs = [1.0, 2.0, 3.0, 4.0]
    ys = [10.0, 8.0, 6.0, 4.0]
    r = regress(xs, ys)
    assert abs(r["slope"] + 2.0) < 1e-9
    assert r["ci_lo"] < 0 and r["ci_hi"] < 0
    assert hs_verdict(r).startswith("SUPPORTED")


def test_regress_reports_inconclusive_when_ci_spans_zero():
    r = regress([1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 4.0, 6.5])
    assert hs_verdict(r).startswith("INCONCLUSIVE")


def test_regress_needs_at_least_three_points():
    assert hs_verdict(regress([1.0, 2.0], [3.0, 4.0])) == "insufficient data"


def test_memory_table_excludes_probe_run_ids():
    """Probe runs (4-part ids) must stay out of every generated table, not just ladder/mechanism."""
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("build_reports", Path("scripts/build_reports.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mem = {"weight_bytes_bf16": 1e8, "kv_bytes_ctx512_b1": 8e6, "kv_bytes_ctx2048_b8": 2.6e8,
           "weight_frac_of_total_ctx512_b1": 0.9}
    text = mod.build_memory([{"run": "L8-A0-s1337", **mem}, {"run": "L8-A0-s1337-lr2x", **mem}])
    assert "| L8-A0-s1337 |" in text
    assert "lr2x" not in text


def _build_reports():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("build_reports", Path("scripts/build_reports.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_parse_probe_id():
    assert parse_probe_id("L8-A0-s1337-lr2x") == {"depth": 8, "arm": "A0", "seed": 1337, "suffix": "lr2x"}
    assert parse_probe_id("L8-A0-s1337") is None
    assert parse_probe_id("x-A0-s1-fw") is None


def test_paired_signs_mean_a_is_better_when_negative():
    by_id = {"L8-A1u4-s1337": {"val_ppl": 29.0, "acc": 0.5}, "L8-A2-s1337": {"val_ppl": 29.7, "acc": 0.4},
             "L8-A1u4-s1338": {"val_ppl": 29.0}, "L8-A1u4-s1337-lr2x": {"val_ppl": 1.0}}
    assert [s for s, _ in paired(by_id, "L8-A1u4", "L8-A2", "val_ppl")] == [1337]  # 1338 unpaired, probe ignored
    assert paired(by_id, "L8-A1u4", "L8-A2", "val_ppl")[0][1] < 0
    assert abs(paired(by_id, "L8-A1u4", "L8-A2", "acc", higher_is_better=True)[0][1] + 10.0) < 1e-9


def test_paired_scale_controls_the_higher_is_better_unit():
    by_id = {"L8-a-s1337": {"copy_gain": 1.50}, "L8-b-s1337": {"copy_gain": 1.30}}
    # default scale (100) treats the metric as a [0,1] fraction -> percentage points
    assert abs(paired(by_id, "L8-a", "L8-b", "copy_gain", higher_is_better=True)[0][1] + 20.0) < 1e-9
    # scale=1.0 keeps the diff a plain nats difference
    assert abs(paired(by_id, "L8-a", "L8-b", "copy_gain",
                      higher_is_better=True, scale=1.0)[0][1] + 0.20) < 1e-9


def test_hc_verdict_needs_every_seed():
    assert hc_verdict([]) == "pending"
    assert hc_verdict([-1.0, -2.0]) == "SUPPORTED"
    assert hc_verdict([-1.0, 0.5]) == "NOT SUPPORTED"


def test_hl_verdict_needs_three_agreeing_seeds_beyond_two_se():
    assert hl_verdict([-2.0, -2.1], "a", "b") == "pending (2/3 seeds)"
    assert hl_verdict([-2.36, -2.35, -2.61], "a", "b") == "a better"
    assert hl_verdict([2.36, 2.35, 2.61], "a", "b") == "b better"
    assert hl_verdict([-1.0, 1.0, -1.0], "a", "b") == "no difference detected"


def test_mechanism_verdict_is_per_rung_not_pooled():
    """Pooled across rungs this data names A2ffn; the pre-registered L8 reading is A2attn."""
    rows = [{"run": "L8-A0-s1337", "val_ppl": 26.0}, {"run": "L8-A2ffn-s1337", "val_ppl": 28.34},
            {"run": "L8-A2attn-s1337", "val_ppl": 27.04},
            {"run": "L4-A0-s1337", "val_ppl": 29.0}, {"run": "L4-A2ffn-s1337", "val_ppl": 31.32},
            {"run": "L4-A2attn-s1337", "val_ppl": 30.74}]
    text = _build_reports().build_mechanism(rows)
    l4, l8 = text.split("### L4")[1].split("### L8")
    assert "**A2ffn**" in l4 and "**A2attn**" in l8


def test_iso_storage_table_applies_the_pre_registered_rules():
    mem_a = {"weight_bytes_bf16": 1e8, "kv_bytes_ctx512_b1": 2e6, "kv_bytes_ctx2048_b8": 8e7}
    mem_b = {"weight_bytes_bf16": 1.2e8, "kv_bytes_ctx512_b1": 1e6, "kv_bytes_ctx2048_b8": 4e7}
    rows = [{"run": "L8-A0-s1337", "val_ppl": 26.0, **mem_b}, {"run": "L8-A0-s1338", "val_ppl": 26.0},
            {"run": "L16-A1u8-s1337", "val_ppl": 25.5, **mem_a}, {"run": "L16-A1u8-s1338", "val_ppl": 26.2}]
    for s, (a, b) in zip((1337, 1338, 1339), ((29.0, 29.7), (29.1, 29.8), (29.0, 29.8))):
        rows += [{"run": f"L8-A1u4-s{s}", "val_ppl": a}, {"run": f"L8-A2-s{s}", "val_ppl": b}]
    lines = _build_reports().build_iso_storage(rows).splitlines()
    hc = next(l for l in lines if l.startswith("| L16-A1u8 | L8-A0 |"))
    hl = next(l for l in lines if l.startswith("| L8-A1u4 | L8-A2 |"))
    assert "NOT SUPPORTED" in hc and "| 2 |" in hc
    assert "L8-A1u4 better" in hl
    # F3: total MB (weight + KV) at both fixed-storage context/batch points, both sides.
    assert "| 102.0 / 121.0 |" in hc  # total MB @512×1: (100MB+2MB) / (120MB+1MB)
    assert "| 180.0 / 160.0 |" in hc  # total MB @2048×8: (100MB+80MB) / (120MB+40MB)
    assert "KV MB @2048" not in _build_reports().build_iso_storage(rows)


def test_probes_report_lists_only_four_part_ids():
    mem = {"weight_bytes_bf16": 1e8, "kv_bytes_ctx512_b1": 8e6, "kv_bytes_ctx2048_b8": 2.6e8}
    text = _build_reports().build_probes([{"run": "L8-A0-s1337", "val_ppl": 26.0, **mem},
                                          {"run": "L4-A0-s1337-d720", "val_ppl": 26.25, **mem}])
    assert "| L4-A0-s1337-d720 |" in text and "| L8-A0-s1337 |" not in text
    assert "| 108.0 |" in text  # weights + KV @512×1, MB


def test_split_check_flags_a_verdict_that_changes():
    rows = []
    for s in (1337, 1338):
        rows += [{"run": f"L16-A2-s{s}", "val_ppl": 27.0, "test_ppl": 25.0},
                 {"run": f"L8-A0-s{s}", "val_ppl": 26.0, "test_ppl": 26.0}]
    line = next(l for l in _build_reports().build_split_check(rows).splitlines()
                if l.startswith("| L16-A2 | L8-A0 |"))
    assert "NOT SUPPORTED" in line and "SUPPORTED" in line and "**NO**" in line


def test_split_check_matches_val_and_test_verdicts_on_the_same_seeds():
    """F2: before the fix, the val verdict pooled ALL val seeds while the test verdict used only the
    seeds carrying test_ppl — a seed present on one side but not the other could manufacture a false
    disagreement. Here 3 seeds have val_ppl, only 2 of them also have test_ppl, and those 2 agree —
    the verdict comparison must use only the matched 2, giving 'yes' rather than a spurious NO."""
    rows = [
        {"run": "L16-A2-s1337", "val_ppl": 27.0, "test_ppl": 27.0},
        {"run": "L8-A0-s1337", "val_ppl": 28.0, "test_ppl": 28.0},
        {"run": "L16-A2-s1338", "val_ppl": 27.0, "test_ppl": 27.0},
        {"run": "L8-A0-s1338", "val_ppl": 28.0, "test_ppl": 28.0},
        {"run": "L16-A2-s1339", "val_ppl": 29.0},  # no test_ppl; would flip the pooled val verdict
        {"run": "L8-A0-s1339", "val_ppl": 28.0},
    ]
    line = next(l for l in _build_reports().build_split_check(rows).splitlines()
                if l.startswith("| L16-A2 | L8-A0 |"))
    cells = [c.strip() for c in line.strip("|").split("|")]
    # a | b | kind | verdict(val, all seeds) | n matched | val verdict(matched) | test verdict(matched) | same
    assert cells[2] == "iso-storage"
    assert cells[3] == "NOT SUPPORTED"  # headline over all 3 val seeds stays visible
    assert cells[4] == "2"              # n matched
    assert cells[5] == "SUPPORTED" and cells[6] == "SUPPORTED"
    assert cells[7] == "yes"


def test_split_check_missing_test_data_is_not_a_disagreement():
    """No test_ppl anywhere for a comparison must not render as a bolded NO — that's absence of
    data, not a verdict that depends on the split."""
    rows = [{"run": "L16-A2-s1337", "val_ppl": 27.0}, {"run": "L8-A0-s1337", "val_ppl": 26.0},
            {"run": "L16-A2-s1338", "val_ppl": 27.1}, {"run": "L8-A0-s1338", "val_ppl": 26.05}]
    line = next(l for l in _build_reports().build_split_check(rows).splitlines()
                if l.startswith("| L16-A2 | L8-A0 |"))
    assert "no test data" in line
    assert "**NO**" not in line
    assert "| — |" in line


def test_build_reports_fails_clearly_on_a_pre_migration_row():
    """F13: a results/runs row from before the 2026-09-14 val/test rename lacks `val_ppl` — that
    must fail with a message naming the run and the migration script, not a bare KeyError deep in
    some build_* function."""
    rows = [{"run": "L8-A0-s1337", "test_ppl": 26.06}]  # pre-migration: no val_ppl
    with pytest.raises(ValueError, match="migrate_ppl_split.*L8-A0-s1337|L8-A0-s1337.*migrate_ppl_split"):
        _build_reports().check_val_ppl_present(rows)


def test_capability_report_gives_hr_verdicts_per_metric():
    rows = []
    for s in (1337, 1338):
        rows += [{"run": f"L16-A1u8-s{s}", "copy_gain": 1.36, "late_loss": 3.0},
                 {"run": f"L8-A0-s{s}", "copy_gain": 1.28, "late_loss": 2.9}]
    lines = _build_reports().build_capability(rows).splitlines()
    cg = next(l for l in lines if l.startswith("| L16-A1u8 | L8-A0 | iso-storage | copy_gain |"))
    late = next(l for l in lines if l.startswith("| L16-A1u8 | L8-A0 | iso-storage | late_loss |"))
    assert cg.endswith("| SUPPORTED |") and late.endswith("| NOT SUPPORTED |")


def test_capability_report_renders_copy_gain_verdicts_in_nats():
    """F14 (probe amendment 2026-09-14): copy_gain is in nats, not [0, 1] — the higher_is_better
    branch of `paired` must not turn its diff into centi-nats via the ×100 used for accuracies."""
    rows = []
    for s in (1337, 1338, 1339):
        rows += [{"run": f"L16-A1u8-s{s}", "copy_gain": 1.50}, {"run": f"L8-A0-s{s}", "copy_gain": 1.30}]
    text = _build_reports().build_capability(rows)
    assert "nats for copy_gain" in text
    line = next(l for l in text.splitlines()
                if l.startswith("| L16-A1u8 | L8-A0 | iso-storage | copy_gain |"))
    # plain nats difference (-0.20), not the accuracy-style ×100 (-20.00)
    assert "| -0.20 |" in line


def test_transposed_loop_rows_use_the_hl_rule():
    rows = []
    for s, (t, loop) in zip((1337, 1338, 1339), ((27.8, 28.2), (27.9, 28.2), (27.8, 28.2))):
        rows += [{"run": f"L8-A1u4t-s{s}", "val_ppl": t}, {"run": f"L8-A1u4-s{s}", "val_ppl": loop}]
    lines = _build_reports().build_iso_storage(rows).splitlines()
    ht = next(l for l in lines if l.startswith("| L8-A1u4t | L8-A1u4 | matched |"))
    assert "L8-A1u4t better" in ht and "| 3 |" in ht
    for a in ("L8-A1u4t", "L8-A1u4n", "L8-A1u4a"):
        for b in ("L8-A1u4", "L4-A0", "L8-A2"):
            assert any(l.startswith(f"| {a} | {b} |") for l in lines), (a, b)
