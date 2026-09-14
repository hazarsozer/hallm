"""Parameter-count invariants — the proof that the sharing mechanism is implemented correctly (FR-7).

Counts only 2-D ("matrix") parameters inside the transformer blocks, which are exactly the d² terms
(attention + FFN weights); LayerNorm gains are 1-D and excluded, so these checks match the closed-form
formulas in roadmap/02-arms-and-paramcount.md exactly.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from hallm.model import GPT, SHAPES, arm_config
from hallm.model.config import ModelConfig

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


def _deep(L: int):
    return dataclasses.replace(SMOKE, n_layer=L)


@pytest.mark.parametrize("L", [4, 8])
def test_looped_arm_stores_k_blocks_independent_of_depth(L: int) -> None:
    """A1u<k> stores k blocks (12d²·k) whatever the unrolled depth."""
    assert block_matrix_params(GPT(arm_config(_deep(L), "A1u2"))) == 12 * D * D * 2


def test_looped_arm_tag_and_block_count() -> None:
    cfg = arm_config(_deep(4), "A1u2")
    assert cfg.arm == "A1u2" and cfg.share_cross_layer and cfg.n_unique_blocks == 2
    assert len(GPT(cfg).blocks) == 2
    assert arm_config(_deep(4), "A1u1").arm == "A1"


def test_looped_with_k_equal_depth_matches_unshared_storage() -> None:
    assert block_matrix_params(GPT(arm_config(_deep(4), "A1u4"))) == \
        block_matrix_params(GPT(arm_config(_deep(4), "A0")))


def test_looped_cycles_blocks_in_order() -> None:
    """Layer i must use block i mod k (Saunshi et al.'s k-layer block looped L/k times)."""
    model = GPT(arm_config(_deep(4), "A1u2"))
    order: list[int] = []
    for j, block in enumerate(model.blocks):
        block.register_forward_hook(lambda m, i, o, j=j: order.append(j))
    model(torch.zeros(1, 8, dtype=torch.long))
    assert order == [0, 1, 0, 1]


@pytest.mark.parametrize("k", [0, 3, 5])
def test_looped_rejects_k_that_does_not_divide_depth(k: int) -> None:
    with pytest.raises(ValueError):
        arm_config(_deep(4), f"A1u{k}")


def test_n_unique_blocks_requires_the_cross_layer_flag() -> None:
    with pytest.raises(ValueError):
        dataclasses.replace(SMOKE, n_unique_blocks=2)


def test_non_looped_arms_reset_n_unique_blocks() -> None:
    looped = arm_config(_deep(4), "A1u2")
    assert arm_config(looped, "A0").n_unique_blocks is None


def test_checkpoint_config_without_the_new_field_still_loads() -> None:
    d = dataclasses.asdict(SMOKE)
    d.pop("n_unique_blocks")
    assert ModelConfig(**d).n_unique_blocks is None
