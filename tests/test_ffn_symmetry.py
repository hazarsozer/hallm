import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "ffn_symmetry", Path(__file__).resolve().parents[1] / "scripts" / "ffn_symmetry.py")
ffn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ffn)


# --- fix 2a: the --hallm-vit docstring pattern must actually match the real (case-sensitive)
# directory names and must not exclude V4-A0. ---

def test_docstring_hallm_vit_pattern_is_case_correct_and_includes_all_five_arms():
    assert "runs/vision/V*-s1337/*.pt" in ffn.__doc__
    assert "runs/vision/v8-*" not in ffn.__doc__   # the old, zero-matching pattern


# --- fix 2b: write_results refuses (non-zero exit) when a requested row group is empty, instead
# of silently writing an empty section or truncating an existing file. ---

def test_write_results_refuses_when_requested_group_is_empty(tmp_path):
    out, rpt = tmp_path / "out.json", tmp_path / "report.md"
    with pytest.raises(SystemExit):
        ffn.write_results([], out, rpt, group=("hallm-vit", ["runs/vision/v8-*/*.pt"], []))
    assert not out.exists()
    assert not rpt.exists()


def test_write_results_refuses_without_truncating_existing_rows(tmp_path):
    out, rpt = tmp_path / "out.json", tmp_path / "report.md"
    existing = [{"model": "m1", "kind": "lm", "arm": "A0", "per_layer": [0.1], "mean": 0.1,
                "per_layer_xhat": [0.1], "mean_xhat": 0.1}]
    ffn.write_results(existing, out, rpt)   # first write succeeds, no group given
    assert json.loads(out.read_text()) == existing

    # Simulate main()'s next call: existing rows plus zero new rows from a mistyped --hallm-vit
    # glob. Must refuse rather than overwrite `out` with just `existing` truncated further, or
    # (worse) accept a call that silently drops half the comparison.
    with pytest.raises(SystemExit):
        ffn.write_results(existing, out, rpt, group=("hallm-vit", ["runs/vision/v8-*/*.pt"], []))
    # the file from the successful write must be untouched
    assert json.loads(out.read_text()) == existing


def test_write_results_succeeds_when_group_not_requested():
    # patterns == [] means the flag was never passed — that's a normal, not-an-error case (e.g.
    # running without --deit), so it must not be confused with "requested but matched nothing".
    ffn.report([])  # sanity: report() tolerates empty rows
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        out, rpt = Path(d) / "out.json", Path(d) / "report.md"
        ffn.write_results([{"model": "random", "kind": "reference", "arm": "—", "per_layer": [0.5],
                            "mean": 0.5, "per_layer_xhat": [0.5], "mean_xhat": 0.5}],
                          out, rpt, group=("hallm-vit", [], []))
        assert out.exists() and rpt.exists()


def test_write_results_succeeds_when_group_produced_rows():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        out, rpt = Path(d) / "out.json", Path(d) / "report.md"
        row = {"model": "m1", "kind": "vision-ours", "arm": "A0", "per_layer": [0.2], "mean": 0.2,
              "per_layer_xhat": [0.2], "mean_xhat": 0.2}
        ffn.write_results([row], out, rpt, group=("hallm-vit", ["runs/vision/V*-s1337/*.pt"], [row]))
        assert json.loads(out.read_text()) == [row]


# --- fix 3: hallm_vit_rows must draw a seeded random sample of images, not the eval path's
# first-N-in-file-order slice, so it is controlled the same way lm_rows/deit_row are. Tested at
# the level of the pure index-selection helper: no checkpoint, model or GPU needed. ---

def test_seeded_val_indices_is_deterministic():
    a = ffn._seeded_val_indices(5000, 64, seed=0)
    b = ffn._seeded_val_indices(5000, 64, seed=0)
    assert np.array_equal(a, b)


def test_seeded_val_indices_is_not_the_first_n_in_file_order():
    # A class-ordered validation split (the concern the finding raises): the first 64 of 5000
    # images would all share one class. The seeded sample must not be indices 0..63.
    idx = ffn._seeded_val_indices(5000, 64, seed=0)
    assert not np.array_equal(idx, np.arange(64))


def test_seeded_val_indices_has_no_duplicates_and_is_sorted_in_range():
    idx = ffn._seeded_val_indices(5000, 256)
    assert len(idx) == 256
    assert len(set(idx.tolist())) == 256          # no duplicates (replace=False)
    assert np.array_equal(idx, np.sort(idx))       # sorted, like deit_row's `sorted(pick)`
    assert idx.min() >= 0 and idx.max() < 5000


def test_seeded_val_indices_caps_at_available_images():
    idx = ffn._seeded_val_indices(10, 64)
    assert len(idx) == 10
