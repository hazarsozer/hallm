"""Capability metrics (spec 06 §5) — verified against hand computations on the smoke shape.
Network-free: fake byte-level encoder, synthetic data, no tiktoken."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from hallm.capeval import (
    blimp_accuracy,
    copy_gain,
    frequency_bucket_loss,
    frequency_buckets,
    greedy_continuation,
    lambada_accuracy,
    load_blimp_file,
    load_lambada,
    position_loss,
    repeat_batch,
    sequence_nll,
    sliced_perplexity,
)
from hallm.data import make_synthetic_data
from hallm.model import GPT, SHAPES

CFG = SHAPES["smoke"]


def _model() -> GPT:
    torch.manual_seed(0)
    return GPT(CFG)


def test_sequence_nll_matches_manual():
    model = _model()
    ids = list(range(10))
    x = torch.tensor([ids[:-1]])
    with torch.no_grad():
        logits, _ = model(x, torch.tensor([ids[1:]]))
    manual = F.cross_entropy(
        logits[0], torch.tensor(ids[1:]), reduction="sum"
    ).item()
    assert abs(sequence_nll(model, ids) - manual) < 1e-3


def test_blimp_tie_counts_as_wrong():
    model = _model()
    s = list(range(8))
    assert blimp_accuracy(model, [(s, s)]) == 0.0


def test_blimp_prefers_learned_sequence():
    model = _model()
    good = [3] * 16  # trivially learnable
    torch.manual_seed(0)
    bad = torch.randint(0, CFG.vocab_size, (16,)).tolist()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    x, y = torch.tensor([good[:-1]]), torch.tensor([good[1:]])
    for _ in range(50):
        opt.zero_grad()
        loss = model(x, y)[1]
        loss.backward()
        opt.step()
    assert blimp_accuracy(model, [(good, bad)]) == 1.0


def test_greedy_matches_argmax():
    model = _model()
    ctx = list(range(12))
    with torch.no_grad():
        logits, _ = model(torch.tensor([ctx]))
    assert greedy_continuation(model, ctx, 1) == [int(logits[0, -1].argmax())]


def test_lambada_exact_match_semantics():
    model = _model()
    ctx = list(range(12))
    target = greedy_continuation(model, ctx, 2)
    assert lambada_accuracy(model, [(ctx, target)]) == 1.0
    wrong = [(target[0] + 1) % CFG.vocab_size, target[1]]
    assert lambada_accuracy(model, [(ctx, wrong)]) == 0.0


def test_sliced_perplexity():
    model = _model()
    data = make_synthetic_data(CFG.vocab_size, 4096, seed=0)
    slices = sliced_perplexity(model, data, block_size=CFG.block_size, n_slices=4, batch_size=2)
    assert len(slices) == 4
    assert all(s > 0 and s == s for s in slices)  # positive, not NaN


def test_jsonl_loaders(tmp_path):
    encode = lambda s: [ord(c) % 256 for c in s]  # fake byte encoder — no tiktoken in tests
    lam = tmp_path / "lambada.jsonl"
    lam.write_text(json.dumps({"text": "the quick brown fox"}) + "\n")
    (ctx, tgt), = load_lambada(lam, encode)
    assert ctx == encode("the quick brown") and tgt == encode(" fox")

    bl = tmp_path / "anaphor.jsonl"
    bl.write_text(json.dumps({"sentence_good": "he ran", "sentence_bad": "he run"}) + "\n")
    (good, bad), = load_blimp_file(bl, encode)
    assert good == encode("he ran") and bad == encode("he run")


def test_training_mode_restored():
    """Capeval functions must restore model.training state after calling model.eval()."""
    model = _model()
    ctx = list(range(12))

    # Test sequence_nll: call in training mode, assert training is restored
    model.train()
    assert model.training
    sequence_nll(model, ctx)
    assert model.training, "sequence_nll should restore training mode"

    # Test greedy_continuation: call in training mode, assert training is restored
    model.train()
    assert model.training
    greedy_continuation(model, ctx, 1)
    assert model.training, "greedy_continuation should restore training mode"

    # Test that eval mode is preserved when already in eval mode
    model.eval()
    assert not model.training
    sequence_nll(model, ctx)
    assert not model.training, "sequence_nll should preserve eval mode"

    model.eval()
    assert not model.training
    greedy_continuation(model, ctx, 1)
    assert not model.training, "greedy_continuation should preserve eval mode"


from hallm.eval import evaluate_perplexity


class _CopyOracle(nn.Module):
    """Predicts x[p - half + 1] at position p: a perfect induction head on [r, r] sequences. Logits
    are scaled so the predicted class dominates the softmax, giving near-zero cross-entropy where
    the prediction is right (not just the right argmax)."""

    def __init__(self, half: int) -> None:
        super().__init__()
        self.cfg = CFG
        self.half = half

    def forward(self, x, targets=None):
        idx = (torch.arange(x.shape[1]) - self.half + 1).clamp(min=0)
        return F.one_hot(x[:, idx], CFG.vocab_size).float() * 30.0, None


def test_position_buckets_average_to_the_overall_loss():
    model, data = _model(), make_synthetic_data(CFG.vocab_size, 2048, seed=3)
    buckets = position_loss(model, data, CFG.block_size, n_buckets=4)
    ppl = evaluate_perplexity(model, data, CFG.block_size)
    assert len(buckets) == 4 and abs(sum(buckets) / 4 - math.log(ppl)) < 1e-4


def test_frequency_buckets_split_by_training_mass():
    train = np.array([0] * 50 + [1] * 30 + [2] * 20, dtype=np.uint16)
    assert frequency_buckets(train, vocab_size=4, n_buckets=2).tolist() == [0, 1, 1, 1]


def test_frequency_buckets_does_not_copy_the_training_array_to_int64(monkeypatch):
    """F10: np.bincount must be called on the array as stored (uint16 accepted), not on a copy
    forced to int64 — the training array can be large, and the cast was a needless full copy."""
    seen_dtypes = []
    real_bincount = np.bincount

    def spy(arr, **kwargs):
        seen_dtypes.append(np.asarray(arr).dtype)
        return real_bincount(arr, **kwargs)

    monkeypatch.setattr(np, "bincount", spy)
    train = np.array([0] * 50 + [1] * 30 + [2] * 20, dtype=np.uint16)
    result = frequency_buckets(train, vocab_size=4, n_buckets=2)
    assert seen_dtypes and seen_dtypes[0] == np.uint16
    assert result.tolist() == [0, 1, 1, 1]  # behaviour unchanged


def test_single_frequency_bucket_is_the_overall_loss():
    model, data = _model(), make_synthetic_data(CFG.vocab_size, 2048, seed=3)
    zeros = np.zeros(CFG.vocab_size, dtype=np.int64)
    (loss,) = frequency_bucket_loss(model, data, zeros, 1, CFG.block_size)
    assert abs(loss - math.log(evaluate_perplexity(model, data, CFG.block_size))) < 1e-4


def test_repeat_batch_halves_are_identical_and_deterministic():
    data = np.arange(1000, dtype=np.uint16)
    seq_a = repeat_batch(data, n_seqs=4, half=8, seed=0)
    seq_b = repeat_batch(data, n_seqs=4, half=8, seed=0)
    assert seq_a.shape == (4, 16)
    assert seq_a.dtype == torch.int64
    assert torch.equal(seq_a, seq_b)  # deterministic for a given seed
    assert torch.equal(seq_a[:, :8], seq_a[:, 8:])  # second half is a copy of the first


def test_copy_gain_perfect_copier_has_near_zero_second_loss_and_positive_gain():
    data = np.arange(2000, dtype=np.uint16) % CFG.vocab_size
    result = copy_gain(_CopyOracle(half=8), data, n_seqs=16, half_len=8, seed=0, batch_size=4)
    assert result["copy_loss_second"] < 1e-3
    assert result["copy_gain"] > 0
    assert result["copy_gain"] == result["copy_loss_first"] - result["copy_loss_second"]


def test_copy_gain_raises_when_the_repeated_window_exceeds_block_size():
    model = _model()  # CFG.block_size == 64
    data = np.arange(2000, dtype=np.uint16) % CFG.vocab_size
    with pytest.raises(ValueError):
        copy_gain(model, data, n_seqs=2, half_len=40, seed=0)  # 2*40-1 = 79 > 64
