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
from hallm.model.gpt import Block
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


def _params(model):
    return model.num_parameters()


@pytest.mark.parametrize("suffix", ["t", "n"])
def test_zero_param_variants_match_the_plain_loop(suffix):
    assert _params(GPT(arm_config(_deep(4), f"A1u2{suffix}"))) == _params(GPT(arm_config(_deep(4), "A1u2")))


def test_scaled_variant_adds_two_scalars_per_block():
    plain = _params(GPT(arm_config(_deep(4), "A1u2")))
    assert _params(GPT(arm_config(_deep(4), "A1u2a"))) == plain + 2 * 2


def test_odd_passes_are_transposed():
    model = GPT(arm_config(_deep(4), "A1u2t"))
    calls = []
    for j, block in enumerate(model.blocks):
        block.register_forward_pre_hook(
            lambda m, args, kwargs, j=j: calls.append((j, kwargs.get("transposed", False))),
            with_kwargs=True)
    model(torch.zeros(1, 8, dtype=torch.long))
    assert calls == [(0, False), (1, False), (0, True), (1, True)]


def test_plain_loop_never_transposes():
    model = GPT(arm_config(_deep(4), "A1u2"))
    calls = []
    for block in model.blocks:
        block.register_forward_pre_hook(
            lambda m, args, kwargs: calls.append(kwargs.get("transposed", False)), with_kwargs=True)
    model(torch.zeros(1, 8, dtype=torch.long))
    assert calls == [False] * 4


def _pair(a: str, b: str):
    """Two models with identical stored weights (b gets a's tensors; α stays at init)."""
    torch.manual_seed(0)
    ma = GPT(arm_config(_deep(4), a))
    mb = GPT(arm_config(_deep(4), b))
    mb.load_state_dict(ma.state_dict(), strict=False)
    return ma, mb


def _logits(model):
    x = torch.randint(0, SMOKE.vocab_size, (2, 12), generator=torch.Generator().manual_seed(3))
    return model(x, x)[0]


def test_transposed_pass_changes_the_function():
    plain, t = _pair("A1u2", "A1u2t")
    assert not torch.allclose(_logits(plain), _logits(t))


def test_scaled_equals_transpose_at_init():
    t, a = _pair("A1u2t", "A1u2a")
    assert torch.equal(_logits(t), _logits(a))


def test_negate_subtracts_each_pass2_update():
    torch.manual_seed(0)
    blk = Block(arm_config(_deep(4), "A1u2n"))
    x = torch.randn(2, 6, D)
    x1 = x - blk.attn(blk.ln1(x), transposed=True)
    expected = x1 - blk.mlp(blk.ln2(x1), transposed=True)
    assert torch.allclose(blk(x, transposed=True), expected, atol=1e-6)


def test_alpha_is_pinned_to_its_own_sublayer():
    """A single shared α (or one swapped to the wrong sublayer) would still pass
    `test_scaled_equals_transpose_at_init` (both α start at 1) and `test_loop_scales_reports_alpha_only_for_scaled`
    (which only checks presence/naming, not which forward use). Distinct, non-default values pin
    each α to its own sublayer's update."""
    torch.manual_seed(0)
    blk = Block(arm_config(_deep(4), "A1u2a"))
    with torch.no_grad():
        blk.alpha_attn.fill_(0.5)
        blk.alpha_mlp.fill_(-2.0)
    x = torch.randn(2, 6, D)
    x1 = x + 0.5 * blk.attn(blk.ln1(x), transposed=True)
    expected = x1 - 2 * blk.mlp(blk.ln2(x1), transposed=True)
    assert torch.allclose(blk(x, transposed=True), expected, atol=1e-6)


@pytest.mark.parametrize("suffix", ["t", "n", "a"])
def test_gradients_reach_every_stored_matrix(suffix):
    torch.manual_seed(0)
    model = GPT(arm_config(_deep(4), f"A1u2{suffix}"))
    x = torch.randint(0, SMOKE.vocab_size, (2, 12))
    model(x, x)[1].backward()
    for name, p in model.blocks.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, name


def test_gradient_actually_requires_pass_two_not_just_pass_one():
    """`test_gradients_reach_every_stored_matrix` only asserts grad != 0, which pass 1 alone already
    guarantees — a `.detach()` on pass 2's weights in the transposed MLP branch would still pass it.
    This isolates pass 2's marginal contribution to `fc.weight.grad`: same model, same forward
    values (`.detach()` doesn't change values, only cuts the graph), pass 2's use of the shared
    matrix detached or not. If sharing.py's transposed MLP branch is itself sabotaged with a
    `.detach()` (e.g. `self.proj.weight.t().detach()`), `real_forward` below already produces the
    sabotaged gradient and this test fails (verified manually, then reverted — see the fix report)."""
    torch.manual_seed(0)
    model = GPT(arm_config(_deep(2), "A1u1t"))          # L=2: layer 0 plain (W), layer 1 transposed (Wᵀ)
    x = torch.randint(0, SMOKE.vocab_size, (2, 12))
    real_forward = MLP.forward

    def sabotaged(self, inp, transposed=False):
        if transposed:
            a = F.gelu(F.linear(inp, self.proj.weight.t().detach()))
            return self.dropout(F.linear(a, self.fc.weight.t().detach()))
        return real_forward(self, inp, transposed)

    model.zero_grad()
    model(x, x)[1].backward()
    grad_real = model.blocks[0].mlp.fc.weight.grad.clone()

    model.zero_grad()
    MLP.forward = sabotaged
    try:
        model(x, x)[1].backward()
    finally:
        MLP.forward = real_forward
    grad_sabotaged = model.blocks[0].mlp.fc.weight.grad.clone()

    assert not torch.allclose(grad_real, grad_sabotaged)


def test_loop_scales_reports_alpha_only_for_scaled():
    scales = GPT(arm_config(_deep(4), "A1u2a")).loop_scales()
    assert scales == {"alpha_attn_0": 1.0, "alpha_mlp_0": 1.0, "alpha_attn_1": 1.0, "alpha_mlp_1": 1.0}
    assert GPT(arm_config(_deep(4), "A1u2t")).loop_scales() == {}
    assert GPT(arm_config(SMOKE, "A0")).loop_scales() == {}
