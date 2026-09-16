import json

import numpy as np
import pytest
import torch
import yaml
from hallm.experiment import load_vision_experiment
from hallm.vision_runqueue import drain_vision, run_one_vision


def write_cfg(tmp_path, name, arm, n_layer):
    cfg = {
        "vshape": "v4" if n_layer == 4 else "v8",
        "arm": arm,
        "train": {"max_steps": 2, "warmup_steps": 1, "batch_size": 4, "crop_size": 32,
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
