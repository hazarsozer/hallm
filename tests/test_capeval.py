"""Capability metrics (spec 06 §5) — verified against hand computations on the smoke shape.
Network-free: fake byte-level encoder, synthetic data, no tiktoken."""
from __future__ import annotations

import json

import torch
import torch.nn.functional as F

from hallm.capeval import (
    blimp_accuracy,
    greedy_continuation,
    lambada_accuracy,
    load_blimp_file,
    load_lambada,
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
