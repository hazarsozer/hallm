import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "bvr", Path(__file__).resolve().parents[1] / "scripts" / "build_vision_report.py")
bvr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bvr)


def rows():
    return [
        {"run": "V4-A0-s1337", "arm": "A0", "n_layer": 4, "top1": 0.400, "val_loss": 2.60,
         "non_embedding_params_M": 12.59, "dataset": "imagenet-100"},
        {"run": "V8-A0-s1337", "arm": "A0", "n_layer": 8, "top1": 0.450, "val_loss": 2.40,
         "non_embedding_params_M": 25.17, "dataset": "imagenet-100"},
        {"run": "V8-A2-s1337", "arm": "A2", "n_layer": 8, "top1": 0.410, "val_loss": 2.55,
         "non_embedding_params_M": 12.59, "dataset": "imagenet-100"},
        {"run": "V8-A1u4-s1337", "arm": "A1u4", "n_layer": 8, "top1": 0.430, "val_loss": 2.45,
         "non_embedding_params_M": 12.59, "dataset": "imagenet-100"},
    ]


def test_header_states_one_seed_and_no_verdict():
    text = bvr.vision_report(rows())
    assert "1 seed" in text and "descriptive" in text.lower()
    assert "SUPPORTED" not in text          # verdicts are a seeded-pass concept


def test_pairs_report_top1_point_differences():
    text = bvr.vision_report(rows())
    # V8-A1u4 (0.430) vs V4-A0 (0.400) = +3.0 points for the looped arm
    assert "+3.0" in text and "V8-A1u4" in text
    # V8-A2 (0.410) vs V4-A0 (0.400) = +1.0
    assert "+1.0" in text
    # V8-A1u4 (0.430) vs V8-A2 (0.410) = +2.0
    assert "+2.0" in text
    # V8-A1u4t vs V8-A1u4: V8-A1u4t is absent, so this pair reports pending, not a delta
    assert "| V8-A1u4t | V8-A1u4 | matched storage and compute | — | — | pending |" in text


def test_depth_gate_is_reported():
    text = bvr.vision_report(rows())
    assert "depth gate" in text.lower() and "pass" in text.lower()


def test_depth_gate_fails_when_deeper_is_not_better():
    bad = rows()
    bad[1]["top1"] = 0.350                  # V8-A0 below V4-A0
    text = bvr.vision_report(bad)
    assert "FAIL" in text


def test_missing_runs_are_shown_as_pending():
    text = bvr.vision_report(rows())        # V8-A1u4t absent
    assert "pending" in text and "V8-A1u4t" in text


def test_storage_column_flags_an_unequal_pairing():
    broken = rows()
    broken[3]["non_embedding_params_M"] = 20.0
    text = bvr.vision_report(broken)
    assert "storage mismatch" in text.lower()


def test_followup_seed_rule_triggers_above_one_point():
    r = rows()
    r[2]["top1"] = 0.420                    # V8-A2 (0.420) vs V4-A0 (0.400) → gap = 2.0 >= 1.0
    text = bvr.vision_report(r)
    assert "|Δ| = 2.0 points" in text
    assert "triggered — run seeds 1338/1339 for V8-A2, V4-A0 and V8-A1u4" in text


def test_followup_seed_rule_not_triggered_below_one_point():
    r = rows()
    r[2]["top1"] = 0.405                    # V8-A2 vs V4-A0 gap = 0.5 < 1.0
    text = bvr.vision_report(r)
    assert "|Δ| = 0.5 points" in text
    assert "not triggered — reported as no difference detected at 1 seed" in text


def test_followup_seed_rule_triggers_either_direction():
    # V8-A2 below V4-A0 by 1.5 points (negative raw diff) must trigger exactly like +1.5 would —
    # the rule reads on the absolute gap, not the sign.
    r = rows()
    r[2]["top1"] = 0.385                    # V8-A2 (0.385) vs V4-A0 (0.400) → raw diff -1.5
    text = bvr.vision_report(r)
    assert "|Δ| = 1.5 points" in text
    assert "triggered — run seeds 1338/1339 for V8-A2, V4-A0 and V8-A1u4" in text


def test_followup_seed_rule_pending_when_an_input_is_missing():
    r = [row for row in rows() if row["run"] != "V4-A0-s1337"]   # V4-A0 absent
    text = bvr.vision_report(r)
    assert "**Follow-up seed rule (spec §3):** pending" in text


def test_empty_rows_degrade_to_all_pending_without_raising():
    text = bvr.vision_report([])
    assert "pending" in text
    assert "**Depth gate (spec §3):** pending" in text
    assert "**Follow-up seed rule (spec §3):** pending" in text
    for arm in bvr.ARMS:
        assert arm in text


def test_missing_field_raises_informative_error():
    r = rows()
    del r[1]["top1"]                        # V8-A0 row missing a field the table reads
    try:
        bvr.vision_report(r)
        assert False, "expected a ValueError for the missing field"
    except ValueError as e:
        assert "V8-A0-s1337" in str(e) and "top1" in str(e)
