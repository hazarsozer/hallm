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
