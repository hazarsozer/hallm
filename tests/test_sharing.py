"""Parameter-count invariants — the proof that the sharing mechanism is implemented correctly (FR-7).

Counts only 2-D ("matrix") parameters inside the transformer blocks, which are exactly the d² terms
(attention + FFN weights); LayerNorm gains are 1-D and excluded, so these checks match the closed-form
formulas in roadmap/02-arms-and-paramcount.md exactly.
"""

from __future__ import annotations

import dataclasses

import pytest

from hallm.model import GPT, SHAPES, arm_config

SMOKE = SHAPES["smoke"]
D, L = SMOKE.n_embd, SMOKE.n_layer


def block_matrix_params(model: GPT) -> int:
    return sum(p.numel() for p in model.blocks.parameters() if p.ndim == 2)


@pytest.mark.parametrize(
    "arm,coef,one_block",
    [("A0", 12, False), ("A1", 12, True), ("A2", 6, False), ("A3", 6, True)],
)
def test_block_matrix_param_counts(arm: str, coef: int, one_block: bool) -> None:
    """A0: L·12d² · A1: 12d² · A2: L·6d² · A3: 6d² (roadmap formulas)."""
    n_blocks = 1 if one_block else L
    assert block_matrix_params(GPT(arm_config(SMOKE, arm))) == coef * D * D * n_blocks


def test_ablation_param_counts() -> None:
    # FFN-only sharing: attn standard (4d²) + ffn shared (4d²) = 8d² per block
    assert block_matrix_params(GPT(arm_config(SMOKE, "A2-ffn"))) == (4 + 4) * D * D * L
    # attn-only sharing: attn shared (2d²) + ffn standard (8d²) = 10d² per block
    assert block_matrix_params(GPT(arm_config(SMOKE, "A2-attn"))) == (2 + 8) * D * D * L


@pytest.mark.parametrize("arm", ["A1", "A3"])
def test_cross_layer_params_independent_of_depth(arm: str) -> None:
    p2 = GPT(arm_config(dataclasses.replace(SMOKE, n_layer=2), arm)).num_parameters()
    p8 = GPT(arm_config(dataclasses.replace(SMOKE, n_layer=8), arm)).num_parameters()
    assert p2 == p8


def test_intra_layer_halves_block() -> None:
    a0 = block_matrix_params(GPT(arm_config(SMOKE, "A0")))
    a2 = block_matrix_params(GPT(arm_config(SMOKE, "A2")))
    assert a2 * 2 == a0


def test_halvit_ffn_stores_single_matrix() -> None:
    """HaLViT FFN must store exactly ONE 2-D weight (down-projection is its transpose, same tensor)."""
    mlp = GPT(arm_config(SMOKE, "A2")).blocks[0].mlp
    assert len([p for p in mlp.parameters() if p.ndim == 2]) == 1


def test_halvit_attn_stores_two_matrices() -> None:
    attn = GPT(arm_config(SMOKE, "A2")).blocks[0].attn
    assert len([p for p in attn.parameters() if p.ndim == 2]) == 2  # W_q, W_kv (V/O are transposes)
