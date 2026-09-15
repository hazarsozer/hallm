"""Transposed loop (spec 2026-09-15): config, sublayer transposition, pass routing, α, and the
invariant that every pre-existing arm is untouched."""

from __future__ import annotations

import dataclasses

import pytest
import torch
import torch.nn.functional as F

import prepatch_fixture
from hallm.model import GPT, SHAPES, arm_config
from hallm.model.config import ModelConfig
from hallm.model.sharing import MLP, CausalSelfAttention

SMOKE = SHAPES["smoke"]
D = SMOKE.n_embd


def _deep(L: int) -> ModelConfig:
    return dataclasses.replace(SMOKE, n_layer=L)


def test_default_path_is_bit_identical():
    """loop_pass2=None ⇒ the exact logits the pre-amendment code produced (CPU, float32)."""
    golden = torch.load(prepatch_fixture.FIXTURE, weights_only=True)
    now = prepatch_fixture.make()
    assert set(golden) == set(now) == set(prepatch_fixture.CASES)
    for arm in golden:
        assert torch.equal(golden[arm], now[arm]), arm


@pytest.mark.parametrize("suffix,mode", [("t", "transpose"), ("n", "negate"), ("a", "scaled")])
def test_tag_round_trip(suffix, mode):
    cfg = arm_config(_deep(4), f"A1u2{suffix}")
    assert cfg.loop_pass2 == mode and cfg.n_unique_blocks == 2 and cfg.share_cross_layer
    assert cfg.arm == f"A1u2{suffix}"


def test_k1_transposed_tag_keeps_its_suffix():
    assert arm_config(SMOKE, "A1u1t").arm == "A1u1t"
    assert arm_config(SMOKE, "A1u1").arm == "A1"   # unchanged plain behaviour


def test_plain_loop_has_no_pass2():
    cfg = arm_config(_deep(4), "A1u2")
    assert cfg.loop_pass2 is None and cfg.arm == "A1u2"


def test_loop_pass2_requires_a_looped_arm():
    with pytest.raises(ValueError):
        dataclasses.replace(SMOKE, loop_pass2="transpose")


def test_loop_pass2_rejects_intra_layer_sharing():
    looped = arm_config(_deep(4), "A1u2")
    with pytest.raises(ValueError):
        dataclasses.replace(looped, share_intra_ffn=True, loop_pass2="transpose")


def test_loop_pass2_rejects_biases():
    looped = arm_config(dataclasses.replace(_deep(4), bias=True), "A1u2")
    with pytest.raises(ValueError):
        dataclasses.replace(looped, loop_pass2="transpose")


def test_loop_pass2_rejects_unknown_mode():
    looped = arm_config(_deep(4), "A1u2")
    with pytest.raises(ValueError):
        dataclasses.replace(looped, loop_pass2="sideways")


def test_non_looped_arm_resets_loop_pass2():
    assert arm_config(arm_config(_deep(4), "A1u2t"), "A0").loop_pass2 is None
    assert arm_config(arm_config(_deep(4), "A1u2t"), "A1u2").loop_pass2 is None


def test_pre_amendment_config_still_loads():
    d = dataclasses.asdict(SMOKE)
    d.pop("loop_pass2")
    assert ModelConfig(**d).loop_pass2 is None


def test_transposed_mlp_matches_manual():
    torch.manual_seed(0)
    mlp = MLP(SMOKE)
    x = torch.randn(2, 5, D)
    expected = F.gelu(x @ mlp.proj.weight) @ mlp.fc.weight   # up = W_downᵀ, down = W_upᵀ
    assert torch.allclose(mlp(x, transposed=True), expected, atol=1e-6)


def test_untransposed_mlp_unchanged():
    torch.manual_seed(0)
    mlp = MLP(SMOKE)
    x = torch.randn(2, 5, D)
    assert torch.equal(mlp(x), mlp(x, transposed=False))


def test_transposed_attention_matches_manual():
    torch.manual_seed(0)
    attn = CausalSelfAttention(SMOKE)
    x = torch.randn(2, 5, D)
    B, T, C, H = 2, 5, D, SMOKE.n_head

    def heads(t):
        return t.view(B, T, H, C // H).transpose(1, 2)

    q, k, v = (heads(x @ lin.weight) for lin in (attn.q, attn.k, attn.v))   # x @ W = F.linear(x, Wᵀ)
    y = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, C)
    expected = y @ attn.proj.weight
    assert torch.allclose(attn(x, transposed=True), expected, atol=1e-6)


def test_transposed_path_rejects_halvit_sublayers():
    x = torch.randn(1, 3, D)
    with pytest.raises(ValueError):
        MLP(dataclasses.replace(SMOKE, share_intra_ffn=True))(x, transposed=True)
    with pytest.raises(ValueError):
        CausalSelfAttention(dataclasses.replace(SMOKE, share_intra_attn=True))(x, transposed=True)
