"""Data pipeline: matched-budget batch sampling + GPT-2 BPE tokenization (FR-4).

Two paths:
  • Tests / smoke (network-free): `make_synthetic_data` returns a deterministic uint16 token stream;
    `get_batch` samples contiguous (x, y) windows from it. No download, no tiktoken.
  • Real runs (Term-2, NOT run by the overnight loop): `prepare_bin` tokenizes a WikiText-103 text
    file with the GPT-2 BPE (tiktoken) into a uint16 `.bin`; `load_bin` memmaps it.

`get_batch` is seeded via a `torch.Generator`, so the data order is reproducible and can be held
IDENTICAL across the four arms (matched-budget control, NFR-1).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

GPT2_VOCAB_SIZE = 50257
_TOKEN_DTYPE = np.uint16  # vocab <= 65536 fits in uint16 (true for GPT-2 and all smoke shapes)


def make_synthetic_data(vocab_size: int, n_tokens: int, seed: int = 0) -> np.ndarray:
    """Deterministic random token stream for tests/smoke (no network)."""
    if vocab_size > 65536:
        raise ValueError("synthetic helper supports vocab_size <= 65536 (uint16)")
    rng = np.random.default_rng(seed)
    return rng.integers(0, vocab_size, size=n_tokens, dtype=_TOKEN_DTYPE)


def get_batch(
    data: np.ndarray,
    block_size: int,
    batch_size: int,
    device: str | torch.device = "cpu",
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a batch of contiguous (input, next-token-target) windows.

    x = data[i : i+T], y = data[i+1 : i+1+T]. Reproducible given a seeded `generator`.
    """
    n = len(data)
    if n < block_size + 1:
        raise ValueError(f"data length {n} too short for block_size {block_size}")
    max_start = n - block_size - 1
    ix = torch.randint(0, max_start + 1, (batch_size,), generator=generator)
    x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


def iter_eval_batches(data: np.ndarray, block_size: int, batch_size: int, device="cpu"):
    """Non-overlapping sequential windows over the whole stream — for deterministic perplexity eval."""
    n_full = (len(data) - 1) // block_size
    starts = [i * block_size for i in range(n_full)]
    for s in range(0, len(starts), batch_size):
        chunk = starts[s : s + batch_size]
        x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in chunk])
        y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in chunk])
        yield x.to(device), y.to(device)


# --- real-data path (Term-2; lazy tiktoken import so tests never need it) ---

def gpt2_encode(text: str) -> np.ndarray:
    """Encode text to GPT-2 BPE ids (uint16). Requires `tiktoken` (downloads vocab on first use)."""
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    return np.array(enc.encode_ordinary(text), dtype=_TOKEN_DTYPE)


def prepare_bin(text_path: str | Path, out_path: str | Path) -> int:
    """Tokenize a WikiText-103 text file → uint16 `.bin`. Returns the token count. Real-data step."""
    text = Path(text_path).read_text(encoding="utf-8")
    ids = gpt2_encode(text)
    ids.tofile(out_path)
    return int(ids.size)


def load_bin(path: str | Path) -> np.ndarray:
    """Memory-map a prepared `.bin` token stream (read-only)."""
    return np.memmap(path, dtype=_TOKEN_DTYPE, mode="r")
