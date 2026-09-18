"""Track 2 training/eval harness (spec 2026-09-14 §5): batch masking correctness, greedy-decode
eval plumbing, and one real (fast, CPU) learnability check on a task trivial enough to converge in
a handful of steps -- everything else here checks shapes and invariants, not "does it learn"."""

from __future__ import annotations

import dataclasses
import random

import torch

from hallm.model import SHAPES, arm_config
from hallm.model.gpt import GPT
from hallm.synth.data import make_batch, make_problems
from hallm.synth.harness import SynthTrainConfig, evaluate_exact_match, train_synth
from hallm.synth.tasks import AdditionTask, BindingChainTask, PHopInductionTask


def _tiny_model(vocab_size: int, block_size: int) -> GPT:
    base = dataclasses.replace(SHAPES["smoke"], vocab_size=vocab_size, block_size=block_size)
    return GPT(arm_config(base, "A0"))


# --- batch masking -----------------------------------------------------------------------------

def test_make_batch_masks_everything_before_the_answer():
    task = PHopInductionTask(vocab_size=32)
    rng = random.Random(0)
    problems = make_problems(task, 2, 8, rng)
    x, y = make_batch(problems)

    answer_start = problems[0].answer_start
    cutoff = answer_start - 1
    assert (y[:, :cutoff] == -1).all()
    assert (y[:, cutoff:] != -1).all()
    # y is x shifted by one: y[:, cutoff:] must equal the true next tokens (the answer itself).
    expected_answer = torch.tensor([p.answer for p in problems], dtype=torch.long)
    assert torch.equal(y[:, cutoff:], expected_answer)


def test_make_batch_rejects_mismatched_lengths():
    task_a = PHopInductionTask(vocab_size=32)
    task_b = AdditionTask()
    rng = random.Random(0)
    mixed = make_problems(task_a, 2, 2, rng) + make_problems(task_b, 3, 2, rng)
    try:
        make_batch(mixed)
        assert False, "expected ValueError for mismatched problem lengths"
    except ValueError:
        pass


def test_make_batch_shape_matches_token_length_minus_one():
    task = BindingChainTask(max_vars=16)
    rng = random.Random(1)
    problems = make_problems(task, 3, 5, rng)
    x, y = make_batch(problems)
    total_len = len(problems[0].tokens)
    assert x.shape == (5, total_len - 1)
    assert y.shape == (5, total_len - 1)


# --- eval plumbing -------------------------------------------------------------------------------

def test_evaluate_exact_match_runs_and_returns_a_fraction():
    task = PHopInductionTask(vocab_size=16)
    model = _tiny_model(vocab_size=task.vocab_size, block_size=64)
    acc = evaluate_exact_match(model, task, difficulty=1, n_problems=17, seed=0, batch_size=8)
    assert 0.0 <= acc <= 1.0


def test_evaluate_exact_match_handles_multi_token_answers():
    task = AdditionTask()
    model = _tiny_model(vocab_size=task.vocab_size, block_size=64)
    # digits=2 -> 3-digit answer; just checking the autoregressive loop produces a well-formed
    # accuracy without crashing on a multi-step generation.
    acc = evaluate_exact_match(model, task, difficulty=2, n_problems=9, seed=0, batch_size=4)
    assert 0.0 <= acc <= 1.0


def test_evaluate_exact_match_restores_training_mode():
    task = PHopInductionTask(vocab_size=16)
    model = _tiny_model(vocab_size=task.vocab_size, block_size=64)
    model.train()
    evaluate_exact_match(model, task, difficulty=1, n_problems=4, seed=0, batch_size=4)
    assert model.training is True

    model.eval()
    evaluate_exact_match(model, task, difficulty=1, n_problems=4, seed=0, batch_size=4)
    assert model.training is False


# --- training loop -------------------------------------------------------------------------------

def test_train_synth_history_shape_and_no_nans():
    task = PHopInductionTask(vocab_size=16, seq_len_margin=3)
    model = _tiny_model(vocab_size=task.vocab_size, block_size=32)
    cfg = SynthTrainConfig(max_steps=10, batch_size=4, log_interval=2, eval_interval=0, warmup_steps=2)
    history = train_synth(model, cfg, task, difficulty=1, device="cpu")
    # steps 0, 2, 4, 6, 8 (log_interval=2) plus the final step (9), always logged.
    assert len(history) == 6
    assert all(torch.isfinite(torch.tensor(r["loss"])) for r in history)


def test_train_synth_records_accuracy_at_eval_interval():
    task = PHopInductionTask(vocab_size=16, seq_len_margin=3)
    model = _tiny_model(vocab_size=task.vocab_size, block_size=32)
    cfg = SynthTrainConfig(
        max_steps=6, batch_size=4, log_interval=1, eval_interval=3, eval_problems=8, warmup_steps=1
    )
    history = train_synth(model, cfg, task, difficulty=1, device="cpu")
    with_acc = [r for r in history if "accuracy" in r]
    assert len(with_acc) >= 1
    assert all(0.0 <= r["accuracy"] <= 1.0 for r in with_acc)


def test_train_synth_learns_the_trivial_zero_hop_task():
    """p=0 means 'answer = the last prompt token' -- copy the previous token, the easiest possible
    induction task. A real (if small) model should clear ~90% within a couple hundred CPU steps;
    if it can't, something is wrong with the masking/training wiring, not with the task's
    difficulty."""
    task = PHopInductionTask(vocab_size=8, seq_len_margin=2)
    model = _tiny_model(vocab_size=task.vocab_size, block_size=16)
    cfg = SynthTrainConfig(
        max_steps=300, batch_size=32, lr=3e-3, warmup_steps=10,
        log_interval=300, eval_interval=0,
    )
    train_synth(model, cfg, task, difficulty=0, device="cpu")
    acc = evaluate_exact_match(model, task, difficulty=0, n_problems=200, seed=999)
    assert acc > 0.9, f"expected the model to learn 0-hop copying, got {acc:.2f} accuracy"
