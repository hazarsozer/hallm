"""Post-hoc split evaluation merges into existing per-run results and never invents a run."""
from __future__ import annotations

import torch

from hallm.data import make_synthetic_data
from hallm.model import GPT, SHAPES
from hallm.results import read_run_results, write_run_result
from hallm.train import TrainConfig, save_checkpoint
from scripts.eval_split import eval_split

CFG = SHAPES["smoke"]


def _checkpoint(tmp_path, name="smoke-A0-s7"):
    torch.manual_seed(0)
    ck = tmp_path / f"{name}.pt"
    save_checkpoint(GPT(CFG), CFG, TrainConfig(), ck)
    data = tmp_path / "test.bin"
    make_synthetic_data(CFG.vocab_size, 1024, seed=1).tofile(data)
    return ck, data


def test_eval_split_merges_test_ppl(tmp_path):
    ck, data = _checkpoint(tmp_path)
    write_run_result(tmp_path / "runs", {"run": "smoke-A0-s7", "val_ppl": 1.0})
    out = eval_split([str(ck)], data, "test", tmp_path / "runs", "cpu")
    row = read_run_results(tmp_path / "runs")[0]
    assert row["val_ppl"] == 1.0
    assert row["test_ppl"] == out["smoke-A0-s7"] and "test_split_sha256" in row


def test_eval_split_skips_checkpoints_without_a_result(tmp_path):
    ck, data = _checkpoint(tmp_path)
    assert eval_split([str(ck)], data, "test", tmp_path / "runs", "cpu") == {}


def test_eval_split_skips_a_row_whose_dataset_differs(tmp_path):
    """F4: a fineweb-edu result row must not be scored under the default wikitext-103 --dataset."""
    ck, data = _checkpoint(tmp_path)
    write_run_result(tmp_path / "runs", {"run": "smoke-A0-s7", "val_ppl": 1.0, "dataset": "fineweb-edu"})
    out = eval_split([str(ck)], data, "test", tmp_path / "runs", "cpu")  # default dataset=wikitext-103
    assert out == {}
    row = read_run_results(tmp_path / "runs")[0]
    assert "test_ppl" not in row


def test_eval_split_treats_a_row_with_no_dataset_field_as_wikitext_103(tmp_path):
    """F4: every legacy and pilot row predates the dataset field and must default to wikitext-103."""
    ck, data = _checkpoint(tmp_path)
    write_run_result(tmp_path / "runs", {"run": "smoke-A0-s7", "val_ppl": 1.0})  # no dataset field
    out = eval_split([str(ck)], data, "test", tmp_path / "runs", "cpu")
    assert "smoke-A0-s7" in out


def test_eval_split_scores_a_matching_dataset_explicitly(tmp_path):
    ck, data = _checkpoint(tmp_path)
    write_run_result(tmp_path / "runs", {"run": "smoke-A0-s7", "val_ppl": 1.0, "dataset": "fineweb-edu"})
    out = eval_split([str(ck)], data, "test", tmp_path / "runs", "cpu", dataset="fineweb-edu")
    assert "smoke-A0-s7" in out
