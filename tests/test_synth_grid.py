"""Track 2 p-hop grid configs (spec 2026-09-14 §5): the arms the comparison rests on must store
the same block weights, and nothing but arm, depth, difficulty and seed may differ between runs."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import yaml

from hallm.model.gpt import GPT
from hallm.synth.experiment import load_synth_experiment
from scripts.gen_synth_grid import ARMS, LEVELS, QUEUE, SEEDS, generate


def _block_matrix(model: GPT) -> int:
    return sum(p.numel() for p in model.blocks.parameters() if p.ndim == 2)


def test_grid_is_seed_major_and_complete(tmp_path):
    queue = generate(tmp_path)
    assert len(queue) == len(SEEDS) * len(LEVELS) * len(ARMS) == 54
    names = [Path(q).name for q in queue]
    assert all("s1337" in n for n in names[:18]) and all("s1339" in n for n in names[-18:])
    assert names[:6] == [f"grid-p_hop-p2-L{layers}-{arm}-s1337.yaml" for arm, layers in ARMS]
    assert (tmp_path / QUEUE).read_text(encoding="utf-8").splitlines() == queue


def test_grid_run_ids_are_unique(tmp_path):
    run_ids = [load_synth_experiment(q)[4] for q in generate(tmp_path)]
    assert len(set(run_ids)) == 54
    assert "synth-p_hop-p8-L4-A1u1-s1339" in run_ids and "synth-p_hop-p2-L2-A2-s1337" in run_ids


def test_same_storage_arms_store_identical_block_weights(tmp_path):
    generate(tmp_path)
    stored = {}
    for arm, layers in ARMS:
        model_cfg, *_ = load_synth_experiment(tmp_path / f"grid-p_hop-p4-L{layers}-{arm}-s1337.yaml")
        stored[(arm, layers)] = _block_matrix(GPT(model_cfg))
    base = stored[("A0", 1)]
    assert stored[("A2", 2)] == stored[("A1u1", 2)] == stored[("A1u1", 4)] == base
    assert stored[("A0", 2)] == 2 * base and stored[("A0", 4)] == 4 * base


def test_grid_runs_differ_only_in_arm_depth_difficulty_and_seed(tmp_path):
    ref_model, ref_train, ref_task, _, _ = load_synth_experiment(generate(tmp_path)[0])
    for q in (tmp_path / QUEUE).read_text(encoding="utf-8").splitlines():
        model, train, task, difficulty, _ = load_synth_experiment(q)
        same_shape = dataclasses.replace(
            model, n_layer=ref_model.n_layer, share_cross_layer=False, n_unique_blocks=None,
            share_intra_ffn=False, share_intra_attn=False)
        assert same_shape == ref_model, q
        assert dataclasses.replace(train, seed=ref_train.seed) == ref_train, q
        assert (task.vocab_size, task.seq_len(difficulty), task.dense) == (4, 64, True), q
        assert model.block_size == task.seq_len(difficulty) + 1


def test_committed_grid_configs_match_the_generator(tmp_path):
    """configs/ is never hand-edited: the committed files must be exactly what the generator emits."""
    for q in generate(tmp_path):
        committed = Path("configs/synth") / Path(q).name
        assert yaml.safe_load(committed.read_text(encoding="utf-8")) == yaml.safe_load(Path(q).read_text(encoding="utf-8"))
