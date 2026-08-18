"""Decoder-only GPT assembled from the sharing-aware sublayers.

The model is deliberately nanoGPT-class and minimal (ROADMAP.md §4): token + learned-position
embeddings, L pre-LN transformer blocks, a final LayerNorm, and a weight-tied LM head. The only thing
that varies across the four arms is which weights are shared — all handled inside `sharing.py` and the
`build_blocks` cross-layer helper, so this file is identical for every arm (NFR-6).

NOTE (Term-2 refinement, not needed for the smoke test): residual-projection weights are left at the
plain N(0, 0.02) init; the 1/sqrt(2L) GPT-2 residual-scaling could be added for real-training quality,
with a decision on how to handle it for the shared (transposed) projections.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from hallm.model.config import ModelConfig
from hallm.model.sharing import CausalSelfAttention, MLP, build_blocks


class Block(nn.Module):
    """Pre-LN transformer block: x + attn(LN(x)); x + mlp(LN(x))."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = build_blocks(cfg, Block)   # 1 block (cross-layer) or L blocks
        self.ln_f = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.head.weight = self.tok_emb.weight   # weight tying (counted once)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        if T > self.cfg.block_size:
            raise ValueError(f"sequence length {T} exceeds block_size {self.cfg.block_size}")
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))

        n_unique = len(self.blocks)
        for i in range(self.cfg.n_layer):
            x = self.blocks[i % n_unique](x)   # cross-layer: n_unique==1 ⇒ block 0 reused L times
        x = self.ln_f(x)

        if targets is not None:
            logits = self.head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1
            )
            return logits, loss
        logits = self.head(x[:, [-1], :])   # only the last position is needed at inference
        return logits, None

    def num_parameters(self, non_embedding: bool = False) -> int:
        """Total unique parameters (tied/shared tensors counted once via the id memo in parameters())."""
        seen: set[int] = set()
        total = 0
        for p in self.parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            total += p.numel()
        if non_embedding:
            total -= self.tok_emb.weight.numel() + self.pos_emb.weight.numel()
        return total
