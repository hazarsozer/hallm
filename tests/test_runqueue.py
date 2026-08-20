"""Queue semantics (spec 06 §8.3): drain in order, freeze manifest at first launch, skip finished
runs, resume interrupted ones. GPU-free: smoke shape, synthetic bins, 4 steps, CPU float32."""
from __future__ import annotations

import json

import pytest
import yaml

from hallm.data import make_synthetic_data
from hallm.model import SHAPES
from hallm.runqueue import PAUSED, drain, run_one

SMOKE_TRAIN = dict(
    max_steps=4, warmup_steps=1, batch_size=2, grad_accum=1, dtype="float32",
    eval_iters=2, log_interval=1, checkpoint_interval=2, seed=7,
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


def test_run_one_trains_freezes_and_skips(tmp_path):
    data_dir, cfgs, _ = _setup(tmp_path)
    row = run_one(cfgs[0], data_dir, device="cpu")
    out = tmp_path / "runs" / "smoke-A0-s7"
    assert (out / "smoke-A0-s7.pt").exists()
    assert (out / "manifest.json").exists()
    assert row["run"] == "smoke-A0-s7" and "test_ppl" in row
    manifest = json.loads((out / "manifest.json").read_text())
    assert set(manifest["data_sha256"]) == {"train.bin", "val.bin"}
    assert run_one(cfgs[0], data_dir, device="cpu") is None  # done ⇒ skip


def test_run_one_resumes_after_interrupt(tmp_path):
    data_dir, cfgs, _ = _setup(tmp_path)
    assert run_one(cfgs[0], data_dir, device="cpu", stop_step=2) is PAUSED  # interrupted ⇒ paused
    out = tmp_path / "runs" / "smoke-A0-s7"
    assert (out / "resume.pt").exists() and not (out / "smoke-A0-s7.pt").exists()
    row = run_one(cfgs[0], data_dir, device="cpu")  # second session finishes it
    assert row is not None and (out / "smoke-A0-s7.pt").exists()


def test_drain_writes_one_result_file_per_run(tmp_path):
    data_dir, _, queue = _setup(tmp_path)
    results = tmp_path / "results" / "runs"
    rows = drain(queue, data_dir, results, device="cpu")
    assert [r["run"] for r in rows] == ["smoke-A0-s7", "smoke-A2-s7"]
    written = sorted(p.name for p in results.glob("*.json"))
    assert written == ["smoke-A0-s7.json", "smoke-A2-s7.json"]
    assert json.loads((results / "smoke-A2-s7.json").read_text())["arm"] == "A2"
    assert drain(queue, data_dir, results, device="cpu") == []  # everything done ⇒ no-op


def test_drain_per_entry_error_isolation(tmp_path):
    data_dir, cfgs, _ = _setup(tmp_path)
    results = tmp_path / "results" / "runs"
    # Create queue with nonexistent config on first line, valid config on second
    queue = tmp_path / "queue_with_error.txt"
    queue.write_text(f"{tmp_path}/nonexistent.yaml\n{cfgs[0]}\n")
    rows = drain(queue, data_dir, results, device="cpu")
    # Verify drain completed despite first entry error, finished the valid run
    assert len(rows) == 1 and rows[0]["run"] == "smoke-A0-s7"
    assert sorted(p.name for p in results.glob("*.json")) == ["smoke-A0-s7.json"]


def test_drain_stops_at_stop_step_without_starting_next_entry(tmp_path):
    # A --stop-step session must bound the ONE run active when it ends, not start every queued
    # run for one step each: drain must BREAK (not `continue`) when a run pauses.
    data_dir, cfgs, queue = _setup(tmp_path)
    results = tmp_path / "results" / "runs"
    rows = drain(queue, data_dir, results, device="cpu", stop_step=2)
    assert rows == []
    out0 = tmp_path / "runs" / "smoke-A0-s7"
    out1 = tmp_path / "runs" / "smoke-A2-s7"
    assert (out0 / "resume.pt").exists()  # first entry paused mid-run
    assert not out1.exists()  # second entry never started
    assert list(results.glob("*.json")) == []


def test_run_one_rejects_resume_with_changed_config(tmp_path):
    data_dir, cfgs, _ = _setup(tmp_path)
    cfg_path = cfgs[0]
    assert run_one(cfg_path, data_dir, device="cpu", stop_step=2) is PAUSED
    out = tmp_path / "runs" / "smoke-A0-s7"
    assert (out / "resume.pt").exists()

    # Rewrite the config with a changed train value (lr) between sessions.
    spec = yaml.safe_load(cfg_path.read_text())
    spec["train"]["lr"] = spec["train"].get("lr", 6e-4) * 2 + 1.0
    cfg_path.write_text(yaml.safe_dump(spec))

    with pytest.raises(RuntimeError, match="lr"):
        run_one(cfg_path, data_dir, device="cpu")


def test_drain_reports_resume_config_mismatch_as_failure(tmp_path):
    data_dir, cfgs, queue = _setup(tmp_path)
    results = tmp_path / "results" / "runs"
    assert run_one(cfgs[0], data_dir, device="cpu", stop_step=2) is PAUSED

    spec = yaml.safe_load(cfgs[0].read_text())
    spec["train"]["lr"] = spec["train"].get("lr", 6e-4) * 2 + 1.0
    cfgs[0].write_text(yaml.safe_dump(spec))

    failures: list[str] = []
    rows = drain(queue, data_dir, results, device="cpu", failures=failures)
    # First entry fails config validation; drain isolates the error and continues to the second.
    assert [r["run"] for r in rows] == ["smoke-A2-s7"]
    assert len(failures) == 1 and "lr" in failures[0]


# --- P0 item 1/3/4 wiring: every run must leave a metrics trail and an enriched row ---------

def test_run_one_writes_metrics_and_enriched_row(tmp_path):
    data_dir, cfgs, _ = _setup(tmp_path)
    row = run_one(cfgs[0], data_dir, device="cpu")
    out = tmp_path / "runs" / "smoke-A0-s7"

    metrics = out / "metrics.jsonl"
    assert metrics.exists(), "no per-run metrics.jsonl written"
    recs = [json.loads(x) for x in metrics.read_text().splitlines() if x.strip()]
    assert recs and all("step" in r and "loss" in r for r in recs)
    assert any("val_loss" in r for r in recs), "no val loss recorded during training"

    assert "final_train_loss" in row
    assert "final_val_loss" in row
    assert "kv_bytes_ctx512_b1" in row and "weight_bytes_bf16" in row


def test_metrics_jsonl_survives_resume_without_truncation(tmp_path):
    """A resumed run must append to its metrics trail, not start a fresh one."""
    data_dir, cfgs, _ = _setup(tmp_path)
    run_one(cfgs[0], data_dir, device="cpu", stop_step=2)
    out = tmp_path / "runs" / "smoke-A0-s7"
    first = len([x for x in (out / "metrics.jsonl").read_text().splitlines() if x.strip()])
    run_one(cfgs[0], data_dir, device="cpu")
    second = len([x for x in (out / "metrics.jsonl").read_text().splitlines() if x.strip()])
    assert second > first, "metrics trail was truncated on resume"
