"""FineWeb-Edu → uint16 token bins (spec 2026-09-14 §4.3, §7).

Documents are GPT-2-BPE encoded and each is followed by the end-of-text token, the nanoGPT
convention for document corpora: a training window may span a boundary, and EOT marks it. Held-out
splits come from a shard never used for training, so no document is on both sides. When a budget
is reached mid-batch, the rest of that batch is dropped — deterministic, and it keeps sequential
splits drawn from one stream disjoint.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np

EOT = 50256  # GPT-2 <|endoftext|>


def write_until(batches: Iterable[list[str]], encode_batch: Callable[[list[str]], list[list[int]]],
                out_path: str | Path, max_tokens: int) -> int:
    """Encode documents in order and write them (EOT after each) to `out_path` until exactly
    `max_tokens` tokens are written, or the stream runs out. Returns the token count."""
    n = 0
    with open(out_path, "wb") as f:
        for texts in batches:
            for ids in encode_batch(texts):
                arr = np.asarray(list(ids) + [EOT], dtype=np.uint16)[: max_tokens - n]
                arr.tofile(f)
                n += arr.size
                if n >= max_tokens:
                    return n
    return n
