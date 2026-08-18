"""Data / metrics / train / eval / config-loading coverage."""

from __future__ import annotations

import glob
import os
import tempfile

import torch

from hallm.data import get_batch, iter_eval_batches, make_synthetic_data
from hallm.eval import comparison_table, evaluate_arm, evaluate_perplexity
from hallm.experiment import load_experiment
from hallm.metrics import analytic_forward_gflops, count_parameters
from hallm.model import GPT, SHAPES, arm_config
from hallm.train import TrainConfig, build_model_from_checkpoint, save_checkpoint, train

SMOKE = SHAPES["smoke"]


def test_get_batch_shapes_and_target_shift() -> None:
    data = make_synthetic_data(SMOKE.vocab_size, 2000, seed=1)
    gen = torch.Generator().manual_seed(0)
    x, y = get_batch(data, 64, 4, generator=gen)
    assert x.shape == (4, 64) and y.shape == (4, 64)
    assert torch.equal(x[:, 1:], y[:, :-1])  # y is x shifted by one (next-token target)


def test_eval_batches_cover_stream() -> None:
    data = make_synthetic_data(SMOKE.vocab_size, 2000, seed=1)
    n = sum(1 for _ in iter_eval_batches(data, 64, 8))
    assert n >= 1


def test_gflops_flat_across_arms() -> None:
    flops = {a: analytic_forward_gflops(arm_config(SMOKE, a)) for a in ["A0", "A1", "A2", "A3"]}
    assert len(set(flops.values())) == 1  # sharing compresses storage, not compute


def test_non_embedding_params_shrink_with_sharing() -> None:
    n = {a: count_parameters(GPT(arm_config(SMOKE, a)))["non_embedding"] for a in ["A0", "A1", "A2", "A3"]}
    assert n["A1"] < n["A0"] and n["A2"] < n["A0"] and n["A3"] < n["A1"]


def test_train_runs_and_perplexity_finite() -> None:
    data = make_synthetic_data(SMOKE.vocab_size, 4000, seed=2)
    model = GPT(arm_config(SMOKE, "A0"))
    tcfg = TrainConfig(block_size=64, batch_size=4, max_steps=5, warmup_steps=1, dtype="float32", deterministic=False)
    history = train(model, tcfg, data, device="cpu")
    assert len(history) >= 1
    assert all(torch.isfinite(torch.tensor(h["loss"])) for h in history)
    ppl = evaluate_perplexity(model, data, 64, batch_size=8, device="cpu")
    assert ppl > 0 and torch.isfinite(torch.tensor(ppl))


def test_comparison_table_renders() -> None:
    data = make_synthetic_data(SMOKE.vocab_size, 2000, seed=3)
    rows = [
        evaluate_arm(GPT(arm_config(SMOKE, a)), arm_config(SMOKE, a), data, batch_size=8, device="cpu")
        for a in ["A0", "A1", "A2", "A3"]
    ]
    table = comparison_table(rows)
    assert "test_ppl" in table and "A3" in table


def test_checkpoint_round_trip() -> None:
    model = GPT(arm_config(SMOKE, "A2"))
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "m.pt")
        save_checkpoint(model, arm_config(SMOKE, "A2"), TrainConfig(block_size=64), path)
        restored, cfg = build_model_from_checkpoint(path)
    assert cfg.arm == "A2"
    assert all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), restored.state_dict().values()))


def test_all_configs_load_and_build() -> None:
    files = sorted(glob.glob("configs/*.yaml"))
    assert files, "no config files found"
    for f in files:
        model_cfg, train_cfg = load_experiment(f)
        GPT(model_cfg)  # must instantiate
        assert train_cfg.block_size == model_cfg.block_size
