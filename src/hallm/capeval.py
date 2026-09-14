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
import torch.nn.functional as F

from hallm.data import iter_eval_batches
from hallm.eval import evaluate_perplexity
from hallm.model import GPT


@torch.no_grad()
def sequence_nll(model: GPT, ids: list[int], device: str = "cpu") -> float:
    """Sum NLL (nats) of ids[1:] given their prefixes. Left-truncates to block_size + 1 tokens."""
    was_training = model.training
    model.eval()
    ids = list(ids)[-(model.cfg.block_size + 1):]
    x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
    y = torch.tensor([ids[1:]], dtype=torch.long, device=device)
    _, loss = model(x, y)  # mean CE over the sequence
    result = loss.item() * (len(ids) - 1)
    if was_training:
        model.train()
    return result


@torch.no_grad()
def blimp_accuracy(model: GPT, pairs, device: str = "cpu") -> float:
    """Fraction of (good_ids, bad_ids) pairs with NLL(good) < NLL(bad). Strict: a tie is wrong."""
    if not pairs:
        return float("nan")
    correct = sum(
        1 for good, bad in pairs
        if sequence_nll(model, good, device) < sequence_nll(model, bad, device)
    )
    return correct / len(pairs)


@torch.no_grad()
def greedy_continuation(model: GPT, context_ids, n_tokens: int, device: str = "cpu") -> list[int]:
    """Argmax-decode n_tokens after the context (sliding window at block_size)."""
    was_training = model.training
    model.eval()
    ids = list(context_ids)
    for _ in range(n_tokens):
        x = torch.tensor([ids[-model.cfg.block_size:]], dtype=torch.long, device=device)
        logits, _ = model(x)  # inference path: logits at the last position only
        ids.append(int(logits[0, -1].argmax()))
    result = ids[len(context_ids):]
    if was_training:
        model.train()
    return result


@torch.no_grad()
def lambada_accuracy(model: GPT, examples, device: str = "cpu") -> float:
    """examples: (context_ids, target_ids). Correct iff greedy continuation matches target exactly."""
    if not examples:
        return float("nan")
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


# --- Track 1 probes (spec 2026-09-14 §4.2) ---------------------------------------------------

@torch.no_grad()
def token_losses(model: GPT, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Per-token NLL (nats), shape (B, T)."""
    logits, _ = model(x, y)
    return F.cross_entropy(logits.transpose(1, 2), y, reduction="none")


@torch.no_grad()
def position_loss(model: GPT, data: np.ndarray, block_size: int, n_buckets: int = 8,
                  batch_size: int = 8, device: str = "cpu") -> list[float]:
    """Mean NLL by position in the context window, in n_buckets equal-width buckets. Does the shared
    model fall further behind late in the window, where long-range context is usable?"""
    was_training = model.training
    model.eval()
    sums, n = torch.zeros(block_size, dtype=torch.float64), 0
    for x, y in iter_eval_batches(data, block_size, batch_size, device):
        sums += token_losses(model, x, y).double().sum(0).cpu()
        n += x.shape[0]
    if was_training:
        model.train()
    return [float(c.mean()) for c in (sums / max(n, 1)).chunk(n_buckets)]


def frequency_buckets(train: np.ndarray, vocab_size: int, n_buckets: int = 10) -> np.ndarray:
    """Token id → bucket 0 (most frequent) … n_buckets−1 (rarest); buckets hold roughly equal shares
    of training-token mass. Ids never seen in training land in the rarest bucket."""
    counts = np.bincount(np.asarray(train), minlength=vocab_size)[:vocab_size]
    order = np.argsort(-counts, kind="stable")
    before = (np.cumsum(counts[order]) - counts[order]) / max(int(counts.sum()), 1)
    buckets = np.empty(vocab_size, dtype=np.int64)
    buckets[order] = np.minimum((before * n_buckets).astype(np.int64), n_buckets - 1)
    return buckets


@torch.no_grad()
def frequency_bucket_loss(model: GPT, data: np.ndarray, buckets: np.ndarray, n_buckets: int,
                          block_size: int, batch_size: int = 8, device: str = "cpu") -> list[float]:
    """Mean NLL of target tokens per frequency bucket (NaN for an empty bucket)."""
    was_training = model.training
    model.eval()
    b = torch.from_numpy(np.asarray(buckets, dtype=np.int64))
    sums = torch.zeros(n_buckets, dtype=torch.float64)
    cnt = torch.zeros(n_buckets, dtype=torch.float64)
    for x, y in iter_eval_batches(data, block_size, batch_size, device):
        nll = token_losses(model, x, y).double().cpu().flatten()
        by = b[y.cpu().flatten()]
        sums.index_add_(0, by, nll)
        cnt.index_add_(0, by, torch.ones_like(nll))
    if was_training:
        model.train()
    return [float(s / c) if c else float("nan") for s, c in zip(sums, cnt)]


def induction_batch(n_seqs: int, half: int, lo: int, hi: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """[r, r] sequences of random tokens in [lo, hi). Returns (x, y) with y the next-token targets;
    positions half… of y are fully determined by the first copy."""
    g = torch.Generator().manual_seed(seed)
    r = torch.randint(lo, hi, (n_seqs, half), generator=g)
    seq = torch.cat([r, r], dim=1)
    return seq[:, :-1], seq[:, 1:]


@torch.no_grad()
def induction_accuracy(model: GPT, n_seqs: int = 256, half_len: int | None = None, lo: int = 0,
                       hi: int | None = None, seed: int = 0, batch_size: int = 8,
                       device: str = "cpu") -> float:
    """Greedy next-token accuracy over the second copy — what an induction head ('find the previous
    occurrence, copy what followed') solves."""
    half = half_len or model.cfg.block_size // 2
    x, y = induction_batch(n_seqs, half, lo, hi or model.cfg.vocab_size, seed)
    was_training = model.training
    model.eval()
    correct = total = 0
    for i in range(0, n_seqs, batch_size):
        xb, yb = x[i:i + batch_size].to(device), y[i:i + batch_size].to(device)
        logits, _ = model(xb, yb)
        pred = logits.argmax(-1)
        correct += int((pred[:, half:] == yb[:, half:]).sum())
        total += yb[:, half:].numel()
    if was_training:
        model.train()
    return correct / total


def recall_batch(n_seqs: int, n_pairs: int, key_range: tuple[int, int], val_range: tuple[int, int],
                 seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Sequences k1 v1 … kn vn kq with distinct keys (key and value pools disjoint); target = the
    value paired with the query key kq."""
    g = torch.Generator().manual_seed(seed)
    rows, targets = [], []
    for _ in range(n_seqs):
        keys = key_range[0] + torch.randperm(key_range[1] - key_range[0], generator=g)[:n_pairs]
        vals = torch.randint(val_range[0], val_range[1], (n_pairs,), generator=g)
        q = int(torch.randint(0, n_pairs, (1,), generator=g))
        rows.append(torch.cat([torch.stack([keys, vals], 1).flatten(), keys[q:q + 1]]))
        targets.append(vals[q])
    return torch.stack(rows), torch.stack(targets)


@torch.no_grad()
def recall_accuracy(model: GPT, n_seqs: int = 256, n_pairs: int = 32,
                    key_range: tuple[int, int] = (1000, 11000), val_range: tuple[int, int] = (11000, 21000),
                    seed: int = 0, batch_size: int = 64, device: str = "cpu") -> float:
    """Greedy accuracy of predicting the value for the query key (in-context associative recall)."""
    x, target = recall_batch(n_seqs, n_pairs, key_range, val_range, seed)
    was_training = model.training
    model.eval()
    correct = 0
    for i in range(0, n_seqs, batch_size):
        logits, _ = model(x[i:i + batch_size].to(device))  # inference path: last position only
        correct += int((logits[:, -1].argmax(-1).cpu() == target[i:i + batch_size]).sum())
    if was_training:
        model.train()
    return correct / n_seqs
