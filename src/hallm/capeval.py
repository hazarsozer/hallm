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


# --- Track 1 probes (spec 2026-09-14 §4.2, amended 2026-09-14) -------------------------------
# Per-position loss, loss by token-frequency decile, plus copy_gain: repeat a real validation
# passage and measure the NLL drop from its first copy to its second — the in-context copying
# signal an induction head would produce. Replaces the random-token induction and random-BPE
# associative-recall probes, both of which sat at the floor on a seed-1337 sanity check (see the
# amendment note in the spec).

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


def repeat_batch(data: np.ndarray, n_seqs: int, half: int, seed: int) -> torch.Tensor:
    """n_seqs windows of `half` consecutive tokens from `data`, each concatenated with itself:
    shape (n_seqs, 2*half), int64. Start positions drawn uniformly over valid starts with
    `np.random.default_rng(seed)`."""
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(data) - half + 1, size=n_seqs)
    windows = np.stack([np.asarray(data[s:s + half]) for s in starts])
    return torch.from_numpy(np.concatenate([windows, windows], axis=1).astype(np.int64))


@torch.no_grad()
def copy_gain(model: GPT, data: np.ndarray, n_seqs: int = 256, half_len: int = 128, seed: int = 0,
             batch_size: int = 8, device: str = "cpu") -> dict:
    """Repeat a real passage and measure the NLL drop from the first copy to the second — the
    in-context copying signal an induction head would produce, on real text instead of the
    random-token/random-BPE probes that sat at the floor (spec amendment 2026-09-14).

    Returns {"copy_loss_first", "copy_loss_second", "copy_gain"} (nats); gain = first - second,
    higher meaning more in-context copying. copy_loss_first is the mean NLL over the first copy's
    targets (excluding its unpredictable first token); copy_loss_second is the mean NLL over the
    second copy's targets that are fully determined by the first copy (excluding the boundary
    token, predictable only from the whole first copy having just been seen)."""
    half = half_len
    if 2 * half - 1 > model.cfg.block_size:
        raise ValueError(
            f"2*half_len - 1 ({2 * half - 1}) exceeds model block_size ({model.cfg.block_size})"
        )
    seq = repeat_batch(data, n_seqs, half, seed)
    was_training = model.training
    model.eval()
    first_sum = first_n = second_sum = second_n = 0.0
    for i in range(0, n_seqs, batch_size):
        s = seq[i:i + batch_size].to(device)
        x, y = s[:, :-1], s[:, 1:]
        nll = token_losses(model, x, y).double()
        first, second = nll[:, :half - 1], nll[:, half:]
        first_sum += float(first.sum().cpu())
        first_n += first.numel()
        second_sum += float(second.sum().cpu())
        second_n += second.numel()
    if was_training:
        model.train()
    first_loss = first_sum / first_n
    second_loss = second_sum / second_n
    return {"copy_loss_first": first_loss, "copy_loss_second": second_loss,
            "copy_gain": first_loss - second_loss}
