import json

import numpy as np
import pytest
import torch
import yaml
from hallm.experiment import load_vision_experiment
from hallm.vision_runqueue import drain_vision, run_one_vision


def write_cfg(tmp_path, name, arm, n_layer, max_steps=2):
    cfg = {
        "vshape": "v4" if n_layer == 4 else "v8",
        "arm": arm,
        "train": {"max_steps": max_steps, "warmup_steps": 1, "batch_size": 4, "crop_size": 32,
                  "eval_interval": 0, "log_interval": 1, "dtype": "float32",
                  "deterministic": False, "checkpoint_interval": 0,
                  "out_dir": str(tmp_path / "runs" / name)},
    }
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def tiny_data(tmp_path, n=8):
    d = tmp_path / "data"
    d.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    for split in ("train", "val"):
        rng.integers(0, 255, (n, 3, 128, 128), dtype=np.uint8).tofile(d / f"{split}.bin")
        np.save(d / f"{split}_labels.npy", rng.integers(0, 100, n).astype(np.int64))
    return d


def test_load_vision_experiment_resolves_shape_and_arm(tmp_path):
    path = write_cfg(tmp_path, "V8-A1u4-s1337", "A1u4", 8)
    model_cfg, train_cfg = load_vision_experiment(path)
    assert (model_cfg.n_layer, model_cfg.n_unique_blocks) == (8, 4)
    assert model_cfg.block_size == 50 and model_cfg.causal is False
    assert train_cfg.crop_size == 32


def test_run_one_writes_result_and_manifest(tmp_path):
    data = tiny_data(tmp_path)
    results = tmp_path / "results"
    path = write_cfg(tmp_path, "V4-A0-s1337", "A0", 4)
    status = run_one_vision(path, data, results, device="cpu")
    assert status == "ok"
    row = json.loads((results / "V4-A0-s1337.json").read_text())
    assert row["run"] == "V4-A0-s1337" and row["arm"] == "A0"
    assert 0.0 <= row["top1"] <= 1.0
    assert row["dataset"] == "imagenet-100" and row["n_classes"] == 100
    assert row["non_embedding_params_M"] > 0
    assert (tmp_path / "runs" / "V4-A0-s1337" / "manifest.json").exists()


def test_looped_and_shallow_rows_agree_on_stored_weights(tmp_path):
    data = tiny_data(tmp_path)
    results = tmp_path / "results"
    run_one_vision(write_cfg(tmp_path, "V4-A0-s1337", "A0", 4), data, results, device="cpu")
    run_one_vision(write_cfg(tmp_path, "V8-A1u4-s1337", "A1u4", 8), data, results, device="cpu")
    a = json.loads((results / "V4-A0-s1337.json").read_text())
    b = json.loads((results / "V8-A1u4-s1337.json").read_text())
    assert a["non_embedding_params_M"] == pytest.approx(b["non_embedding_params_M"], rel=1e-6)


def test_drain_runs_every_queued_config_in_order(tmp_path):
    data = tiny_data(tmp_path)
    results = tmp_path / "results"
    queue = tmp_path / "queue.txt"
    paths = [write_cfg(tmp_path, "V4-A0-s1337", "A0", 4),
             write_cfg(tmp_path, "V8-A0-s1337", "A0", 8)]
    queue.write_text("\n".join(str(p) for p in paths), encoding="utf-8")
    rows = drain_vision(queue, data, results, device="cpu")
    assert [r["run"] for r in rows] == ["V4-A0-s1337", "V8-A0-s1337"]


def test_finished_runs_are_skipped_on_a_second_drain(tmp_path):
    data = tiny_data(tmp_path)
    results = tmp_path / "results"
    queue = tmp_path / "queue.txt"
    queue.write_text(str(write_cfg(tmp_path, "V4-A0-s1337", "A0", 4)), encoding="utf-8")
    drain_vision(queue, data, results, device="cpu")
    assert drain_vision(queue, data, results, device="cpu") == []


# --- fix-round tests: the paused path, queue-file filtering, per-entry isolation, config drift ---

def test_run_one_pauses_at_stop_step_and_saves_resume(tmp_path):
    data = tiny_data(tmp_path)
    results = tmp_path / "results"
    path = write_cfg(tmp_path, "V4-A0-s1337", "A0", 4, max_steps=4)
    status = run_one_vision(path, data, results, device="cpu", stop_step=2)
    assert status == "paused"
    assert not (results / "V4-A0-s1337.json").exists()
    assert (tmp_path / "runs" / "V4-A0-s1337" / "resume.pt").exists()


def test_run_one_notices_stop_step_already_reached_without_training_further(tmp_path):
    """A resume checkpoint already at/past `stop_step` must be recognized before any training
    step runs — re-draining with the same --stop-step must not train one extra step first."""
    data = tiny_data(tmp_path)
    results = tmp_path / "results"
    path = write_cfg(tmp_path, "V4-A0-s1337", "A0", 4, max_steps=4)
    run_one_vision(path, data, results, device="cpu", stop_step=2)
    out = tmp_path / "runs" / "V4-A0-s1337"
    metrics_before = (out / "metrics.jsonl").read_text()

    status = run_one_vision(path, data, results, device="cpu", stop_step=2)

    assert status == "paused"
    assert (out / "metrics.jsonl").read_text() == metrics_before  # no extra step was taken


def test_run_one_rejects_resume_with_changed_config(tmp_path):
    """A YAML edited between sessions must not silently continue training under a different
    recipe than the frozen manifest attests (mirrors runqueue.py's run_one)."""
    data = tiny_data(tmp_path)
    results = tmp_path / "results"
    path = write_cfg(tmp_path, "V4-A0-s1337", "A0", 4, max_steps=4)
    assert run_one_vision(path, data, results, device="cpu", stop_step=2) == "paused"
    assert (tmp_path / "runs" / "V4-A0-s1337" / "resume.pt").exists()

    spec = yaml.safe_load(path.read_text())
    spec["train"]["batch_size"] = spec["train"]["batch_size"] + 1
    path.write_text(yaml.safe_dump(spec), encoding="utf-8")

    with pytest.raises(RuntimeError, match="batch_size"):
        run_one_vision(path, data, results, device="cpu")


def test_drain_stops_at_paused_run_without_starting_the_next_entry(tmp_path):
    data = tiny_data(tmp_path)
    results = tmp_path / "results"
    queue = tmp_path / "queue.txt"
    paths = [write_cfg(tmp_path, "V4-A0-s1337", "A0", 4, max_steps=4),
             write_cfg(tmp_path, "V8-A0-s1337", "A0", 8)]
    queue.write_text("\n".join(str(p) for p in paths), encoding="utf-8")

    rows = drain_vision(queue, data, results, device="cpu", stop_step=2)

    assert rows == []
    assert (tmp_path / "runs" / "V4-A0-s1337" / "resume.pt").exists()
    assert not (tmp_path / "runs" / "V8-A0-s1337").exists()  # never started
    assert not (results / "V8-A0-s1337.json").exists()


def test_drain_ignores_comments_and_blank_lines_in_the_queue_file(tmp_path):
    data = tiny_data(tmp_path)
    results = tmp_path / "results"
    queue = tmp_path / "queue.txt"
    path = write_cfg(tmp_path, "V4-A0-s1337", "A0", 4)
    queue.write_text(f"# a comment line\n\n{path}\n\n# trailing comment\n", encoding="utf-8")

    rows = drain_vision(queue, data, results, device="cpu")

    assert [r["run"] for r in rows] == ["V4-A0-s1337"]


def test_drain_per_entry_error_isolation(tmp_path):
    """One bad queue entry must not cost the runs after it — the five arms are independent runs
    writing independent result files (mirrors runqueue.py's drain)."""
    data = tiny_data(tmp_path)
    results = tmp_path / "results"
    queue = tmp_path / "queue.txt"
    good = write_cfg(tmp_path, "V4-A0-s1337", "A0", 4)
    queue.write_text(f"{tmp_path / 'nonexistent.yaml'}\n{good}\n", encoding="utf-8")

    rows = drain_vision(queue, data, results, device="cpu")

    assert [r["run"] for r in rows] == ["V4-A0-s1337"]
    assert sorted(p.name for p in results.glob("*.json")) == ["V4-A0-s1337.json"]
