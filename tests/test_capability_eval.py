"""scripts/capability_eval.py's per-run write path. CPU-only, network-free: no checkpoints, no
tiktoken — exercises `save_row` directly via importlib, the same pattern test_reports.py uses for
scripts/build_reports.py."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from hallm.results import read_run_results


def _capability_eval():
    spec = importlib.util.spec_from_file_location("capability_eval", Path("scripts/capability_eval.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_save_row_merges_into_an_existing_run_instead_of_overwriting_it(tmp_path):
    mod = _capability_eval()
    mod.save_row(tmp_path, {"run": "L8-A0-s1337", "induction_acc": 0.9, "recall_acc": 0.8})
    mod.save_row(tmp_path, {"run": "L8-A0-s1337", "lambada_acc": 0.5})

    (row,) = read_run_results(tmp_path)
    assert row["induction_acc"] == 0.9 and row["recall_acc"] == 0.8 and row["lambada_acc"] == 0.5

    on_disk = json.loads((tmp_path / "L8-A0-s1337.json").read_text())
    assert on_disk == row
