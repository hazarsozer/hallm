"""Compression metrics — parameters, size, analytic FLOPs (ITU-T F.748.11; FR-5).

All metrics are reported SEPARATELY (no opaque composite) per roadmap/05-eval-protocol.md.

FLOPs framing (important): `analytic_forward_gflops` depends only on the SHAPE and the unrolled depth
L — NOT on the sharing flags. All four arms execute the identical matmul graph (A2 still does two FFN
matmuls; A1 still applies its shared block L times), so GFLOPs are ~equal across arms. Weight sharing
compresses storage/memory, not compute (ROADMAP.md §6).
"""

from __future__ import annotations

import torch.nn as nn

from hallm.model.config import ModelConfig


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Unique parameter counts (tied/shared tensors counted once)."""
    seen: set[int] = set()
    total = 0
    for p in model.parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        total += p.numel()
    embedding = model.tok_emb.weight.numel() + model.pos_emb.weight.numel()
    return {"total": total, "embedding": embedding, "non_embedding": total - embedding}


def model_size_mb(model: nn.Module, bytes_per_param: int = 4) -> float:
    """Weight storage in MB. bytes_per_param: 4 (FP32) or 2 (BF16)."""
    return count_parameters(model)["total"] * bytes_per_param / (1024**2)


def analytic_forward_gflops(cfg: ModelConfig, batch_size: int = 1) -> float:
    """Analytic GFLOPs for one forward pass over a full block_size sequence.

    Per token, per block: QKV 6d² + out-proj 2d² + attention 4·T·d + FFN 4·d·h.
    Plus the LM head 2·d·V per token. Multiplied by T tokens and the batch.
    Uses the UNROLLED depth (cfg.n_layer), so the result is identical for every arm (see module note).
    """
    d, h, T, V, L = cfg.n_embd, cfg.ffn_hidden, cfg.block_size, cfg.vocab_size, cfg.n_layer
    per_token_block = 6 * d * d + 2 * d * d + 4 * T * d + 4 * d * h
    per_token = L * per_token_block + 2 * d * V
    flops = per_token * T * batch_size
    return flops / 1e9


def metrics_row(model: nn.Module, cfg: ModelConfig) -> dict[str, float | int | str]:
    """Assemble one comparison-table row (params separate from size separate from FLOPs)."""
    pc = count_parameters(model)
    return {
        "arm": cfg.arm,
        "non_embedding_params_M": round(pc["non_embedding"] / 1e6, 4),
        "total_params_M": round(pc["total"] / 1e6, 4),
        "size_fp32_MB": round(model_size_mb(model, 4), 3),
        "size_bf16_MB": round(model_size_mb(model, 2), 3),
        "fwd_gflops": round(analytic_forward_gflops(cfg), 4),
    }
