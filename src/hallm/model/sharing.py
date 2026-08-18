"""The weight-sharing mechanisms — THE scientific crux of the thesis.

Two orthogonal axes (ROADMAP.md §2-§3, roadmap/01-mechanism.md, roadmap/02-arms-and-paramcount.md):

  • Intra-layer (HaLViT, *width* axis): within one sublayer, the second matrix is the transpose of
    the first, so only ONE matrix is stored. A nonlinearity sits between the two applications so the
    transpose is a genuinely independent transformation (the column-space argument).
      - FFN:        FFN(x) = Wᵀ · GELU(W · x)            8d² → 4d²   (−50%)
      - Attention:  K = W_kv·x, V = W_kvᵀ·x ; Q = W_q·x, Out = W_qᵀ·x̂   4d² → 2d²   (−50%)

  • Cross-layer (ALBERT, *depth* axis): one block's weights are reused for all L layers — handled by
    `build_blocks` (one block instead of L). See gpt.py for how the GPT forward applies it L times.

PyTorch orientation note (this is where sharing bugs hide):
  `F.linear(x, W)` computes `x @ Wᵀ`. So for a stored `W` of shape (out, in):
    - "forward" use  `F.linear(x, W)`     = x @ Wᵀ   maps in→out
    - "transpose" use `F.linear(a, W.t())` = a @ W    maps out→in   (the SAME tensor, no second param)
  The transpose path references the identical `nn.Parameter`, so `parameters()` counts it once and the
  gradient accumulates contributions from BOTH roles (verified in tests/test_sharing.py).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from hallm.model.config import ModelConfig


class MLP(nn.Module):
    """Position-wise feed-forward block.

    Standard:  y = W2 · GELU(W1 · x)        — two matrices  (8d²)
    HaLViT:    y = W1ᵀ · GELU(W1 · x)       — one matrix W1 (4d²), down-projection is its transpose
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        d, h = cfg.n_embd, cfg.ffn_hidden
        self.shared = cfg.share_intra_ffn
        self.dropout = nn.Dropout(cfg.dropout)
        if self.shared:
            # single stored matrix W of shape (h, d): up uses W, down uses Wᵀ
            self.w = nn.Parameter(torch.empty(h, d))
            nn.init.normal_(self.w, mean=0.0, std=0.02)
            self.up_bias = nn.Parameter(torch.zeros(h)) if cfg.bias else None
            self.down_bias = nn.Parameter(torch.zeros(d)) if cfg.bias else None
        else:
            self.fc = nn.Linear(d, h, bias=cfg.bias)    # W1: d→h
            self.proj = nn.Linear(h, d, bias=cfg.bias)  # W2: h→d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.shared:
            a = F.gelu(F.linear(x, self.w, self.up_bias))      # up:   x @ Wᵀ  → (..., h)
            y = F.linear(a, self.w.t(), self.down_bias)        # down: a @ W   → (..., d)  (Wᵀ applied)
        else:
            a = F.gelu(self.fc(x))
            y = self.proj(a)
        return self.dropout(y)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention.

    Standard:  four (d×d) projections Q, K, V, O                          (4d²)
    HaLViT:    store W_q and W_kv; V = W_kvᵀ·x, Out = W_qᵀ·x̂              (2d²)

    The K/V pair has no nonlinearity between them (both linear in x); the only mixing is the causal
    softmax — so this path is the *weak* one and the likely failure mode (Risk R1). Toggled by
    `cfg.share_intra_attn`; the ablation arms isolate it from the FFN path.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        d = cfg.n_embd
        self.n_head = cfg.n_head
        self.head_dim = cfg.head_dim
        self.shared = cfg.share_intra_attn
        self.dropout_p = cfg.dropout
        self.resid_drop = nn.Dropout(cfg.dropout)
        if self.shared:
            self.w_q = nn.Parameter(torch.empty(d, d))   # Q;  Out = w_qᵀ
            self.w_kv = nn.Parameter(torch.empty(d, d))  # K;  V   = w_kvᵀ
            nn.init.normal_(self.w_q, std=0.02)
            nn.init.normal_(self.w_kv, std=0.02)
            self.qkv_bias = nn.Parameter(torch.zeros(3 * d)) if cfg.bias else None
            self.out_bias = nn.Parameter(torch.zeros(d)) if cfg.bias else None
        else:
            self.q = nn.Linear(d, d, bias=cfg.bias)
            self.k = nn.Linear(d, d, bias=cfg.bias)
            self.v = nn.Linear(d, d, bias=cfg.bias)
            self.proj = nn.Linear(d, d, bias=cfg.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        if self.shared:
            qb = self.qkv_bias[:C] if self.qkv_bias is not None else None
            kb = self.qkv_bias[C : 2 * C] if self.qkv_bias is not None else None
            vb = self.qkv_bias[2 * C :] if self.qkv_bias is not None else None
            q = F.linear(x, self.w_q, qb)        # Q   = W_q · x
            k = F.linear(x, self.w_kv, kb)       # K   = W_kv · x
            v = F.linear(x, self.w_kv.t(), vb)   # V   = W_kvᵀ · x   (same tensor, transposed)
        else:
            q, k, v = self.q(x), self.k(x), self.v(x)

        # (B, T, C) → (B, nh, T, hd)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout_p if self.training else 0.0
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        if self.shared:
            y = F.linear(y, self.w_q.t(), self.out_bias)   # Out = W_qᵀ · ŷ
        else:
            y = self.proj(y)
        return self.resid_drop(y)


def build_blocks(cfg: ModelConfig, block_cls: type[nn.Module]) -> nn.ModuleList:
    """Cross-layer (ALBERT) sharing.

    If ``cfg.share_cross_layer`` is set, return a ModuleList with ONE block (reused for all L layers
    by the GPT forward, which indexes ``blocks[i % len(blocks)]``). Otherwise return L distinct blocks.
    One shared block ⇒ block parameters are independent of depth L (the A1/A3 invariant).
    """
    n_unique = 1 if cfg.share_cross_layer else cfg.n_layer
    return nn.ModuleList([block_cls(cfg) for _ in range(n_unique)])
