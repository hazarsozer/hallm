"""Capability evals beyond perplexity (spec 06 §5, Tier 1.5).

Inference-only over existing checkpoints — no training, no GPU lockdown. Chosen to discriminate at
12–100M scale: LAMBADA (long-range final-word prediction), BLiMP (grammatical minimal pairs), and
per-slice PPL over the eval stream (a coarse per-domain proxy; contiguous slices ≈ article groups).
The question: is the sharing tax uniform, or does the PPL average hide a lopsided deficit?

Loaders take an `encode` callable (e.g. a tiktoken encoder's) so tests inject a fake encoder and
never touch the network."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from hallm.eval import evaluate_perplexity
from hallm.model.gpt import GPT


@torch.no_grad()
def sequence_nll(model: GPT, ids: list[int], device: str = "cpu") -> float:
    """Sum NLL (nats) of ids[1:] given their prefixes. Left-truncates to block_size + 1 tokens."""
    model.eval()
    ids = list(ids)[-(model.cfg.block_size + 1):]
    x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
    y = torch.tensor([ids[1:]], dtype=torch.long, device=device)
    _, loss = model(x, y)  # mean CE over the sequence
    return loss.item() * (len(ids) - 1)


@torch.no_grad()
def blimp_accuracy(model: GPT, pairs, device: str = "cpu") -> float:
    """Fraction of (good_ids, bad_ids) pairs with NLL(good) < NLL(bad). Strict: a tie is wrong."""
    correct = sum(
        1 for good, bad in pairs
        if sequence_nll(model, good, device) < sequence_nll(model, bad, device)
    )
    return correct / len(pairs)


@torch.no_grad()
def greedy_continuation(model: GPT, context_ids, n_tokens: int, device: str = "cpu") -> list[int]:
    """Argmax-decode n_tokens after the context (sliding window at block_size)."""
    model.eval()
    ids = list(context_ids)
    for _ in range(n_tokens):
        x = torch.tensor([ids[-model.cfg.block_size:]], dtype=torch.long, device=device)
        logits, _ = model(x)  # inference path: logits at the last position only
        ids.append(int(logits[0, -1].argmax()))
    return ids[len(context_ids):]


@torch.no_grad()
def lambada_accuracy(model: GPT, examples, device: str = "cpu") -> float:
    """examples: (context_ids, target_ids). Correct iff greedy continuation matches target exactly."""
    correct = sum(
        1 for ctx, tgt in examples
        if greedy_continuation(model, ctx, len(tgt), device) == list(tgt)
    )
    return correct / len(examples)


@torch.no_grad()
def sliced_perplexity(
    model: GPT, data: np.ndarray, block_size: int, n_slices: int = 10,
    batch_size: int = 8, device: str = "cpu",
) -> list[float]:
    """PPL per contiguous slice of the eval stream. Slices too short for one window are skipped."""
    bounds = np.linspace(0, len(data), n_slices + 1, dtype=int)
    return [
        evaluate_perplexity(model, data[a:b], block_size, batch_size, device)
        for a, b in zip(bounds[:-1], bounds[1:])
        if b - a > block_size
    ]


# --- jsonl loaders (encode injected; real callers pass tiktoken's encode_ordinary) ---

def load_lambada(path: str | Path, encode) -> list[tuple[list[int], list[int]]]:
    """LAMBADA jsonl ({"text": ...}): context = all but the last word, target = " " + last word."""
    examples = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        text = json.loads(line)["text"].strip()
        context, _, last = text.rpartition(" ")
        if not context:
            continue
        examples.append((list(encode(context)), list(encode(" " + last))))
    return examples


def load_blimp_file(path: str | Path, encode) -> list[tuple[list[int], list[int]]]:
    """One BLiMP paradigm jsonl → (sentence_good_ids, sentence_bad_ids) pairs."""
    pairs = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        pairs.append((list(encode(d["sentence_good"])), list(encode(d["sentence_bad"]))))
    return pairs
