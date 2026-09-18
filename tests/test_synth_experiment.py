"""Config-driven loading for Track 2 synth experiments (mirrors test_experiment-style coverage
for hallm.experiment.load_experiment, for the synth pipeline's own loader)."""

from __future__ import annotations

from pathlib import Path

import pytest

from hallm.synth.experiment import load_synth_experiment
from hallm.synth.harness import SynthTrainConfig


def _write_config(tmp_path: Path, **overrides) -> Path:
    body = {
        "task": "p_hop_induction",
        "task_kwargs": {"vocab_size": 32, "seq_len_margin": 3},
        "difficulty": 2,
        "arm": "A0",
        "model": {"n_embd": 16, "n_layer": 2, "n_head": 4, "ffn_mult": 4, "block_size": 20},
        "train": {"max_steps": 5, "batch_size": 4, "seed": 7},
    }
    body.update(overrides)
    import yaml

    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(body), encoding="utf-8")
    return path


def test_load_synth_experiment_wires_task_kwargs_and_vocab(tmp_path):
    path = _write_config(tmp_path)
    model_cfg, train_cfg, task, difficulty, run_id = load_synth_experiment(path)

    assert task.name == "p_hop_induction"
    assert task.vocab_size == 32  # from task_kwargs, not the class default
    assert model_cfg.vocab_size == 32  # model's vocab_size is synced to the task's, not user-set
    assert model_cfg.n_layer == 2
    assert difficulty == 2
    assert isinstance(train_cfg, SynthTrainConfig)
    assert train_cfg.seed == 7
    assert run_id == "synth-p_hop_induction-L2-A0-s7"  # default derivation, since not given


def test_load_synth_experiment_uses_explicit_run_id(tmp_path):
    path = _write_config(tmp_path, run_id="my-custom-run")
    *_, run_id = load_synth_experiment(path)
    assert run_id == "my-custom-run"


def test_load_synth_experiment_arm_flags_wired_correctly(tmp_path):
    path = _write_config(tmp_path, arm="A1u1")
    model_cfg, *_ = load_synth_experiment(path)
    assert model_cfg.share_cross_layer is True
    assert model_cfg.n_unique_blocks == 1


def test_load_synth_experiment_unknown_task_raises(tmp_path):
    path = _write_config(tmp_path)
    body = path.read_text(encoding="utf-8").replace("p_hop_induction", "not_a_real_task")
    path.write_text(body, encoding="utf-8")
    with pytest.raises(KeyError):
        load_synth_experiment(path)
