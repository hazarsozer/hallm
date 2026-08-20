"""Results are one file per run. A single append-only ledger is not the source of truth:
it cannot be re-derived, a crashed write corrupts every later row, and re-running one run
either duplicates or silently conflicts with its old row."""

import json

import pytest
import yaml

from hallm.data import make_synthetic_data
from hallm.model import SHAPES
from hallm.results import read_run_results, write_run_result
from hallm.runqueue import drain

SMOKE_TRAIN = dict(
    max_steps=4, warmup_steps=1, batch_size=2, grad_accum=1, dtype="float32",
    eval_iters=2, log_interval=1, eval_interval=2, checkpoint_interval=2, seed=7,
)


def _setup(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for split in ("train", "val"):
        make_synthetic_data(SHAPES["smoke"].vocab_size, 2048, seed=0).tofile(data_dir / f"{split}.bin")
    cfgs = []
    for arm in ("A0", "A2"):
        name = f"smoke-{arm}-s7"
        spec = {"shape": "smoke", "arm": arm,
                "train": {**SMOKE_TRAIN, "out_dir": str(tmp_path / "runs" / name)}}
        p = tmp_path / f"{name}.yaml"
        p.write_text(yaml.safe_dump(spec))
        cfgs.append(p)
    queue = tmp_path / "queue.txt"
    queue.write_text("\n".join(str(c) for c in cfgs) + "\n")
    return data_dir, cfgs, queue


def test_write_run_result_is_one_file_per_run(tmp_path):
    write_run_result(tmp_path, {"run": "L8-A0-s1337", "test_ppl": 26.06})
    write_run_result(tmp_path, {"run": "L8-A2-s1337", "test_ppl": 29.68})
    assert (tmp_path / "L8-A0-s1337.json").exists()
    assert (tmp_path / "L8-A2-s1337.json").exists()


def test_rerunning_a_run_replaces_only_its_own_file(tmp_path):
    """Idempotence: re-running one run must not duplicate or disturb any other run's result."""
    write_run_result(tmp_path, {"run": "L8-A0-s1337", "test_ppl": 26.06})
    write_run_result(tmp_path, {"run": "L8-A2-s1337", "test_ppl": 29.68})
    write_run_result(tmp_path, {"run": "L8-A0-s1337", "test_ppl": 26.10})
    rows = read_run_results(tmp_path)
    assert len(rows) == 2
    assert next(r for r in rows if r["run"] == "L8-A0-s1337")["test_ppl"] == 26.10


def test_write_run_result_requires_a_run_id(tmp_path):
    with pytest.raises(ValueError):
        write_run_result(tmp_path, {"test_ppl": 1.0})


def test_read_run_results_is_sorted_and_ignores_non_json(tmp_path):
    write_run_result(tmp_path, {"run": "L8-A2-s1337"})
    write_run_result(tmp_path, {"run": "L4-A0-s1337"})
    (tmp_path / "notes.md").write_text("ignore me")
    rows = read_run_results(tmp_path)
    assert [r["run"] for r in rows] == ["L4-A0-s1337", "L8-A2-s1337"]


def test_drain_writes_per_run_files_not_an_appended_ledger(tmp_path):
    data_dir, _, queue = _setup(tmp_path)
    results_dir = tmp_path / "results" / "runs"
    rows = drain(queue, data_dir, results_dir, device="cpu")
    assert len(rows) == 2
    written = sorted(p.name for p in results_dir.glob("*.json"))
    assert written == ["smoke-A0-s7.json", "smoke-A2-s7.json"]
    payload = json.loads((results_dir / "smoke-A0-s7.json").read_text())
    assert payload["run"] == "smoke-A0-s7" and "test_ppl" in payload
