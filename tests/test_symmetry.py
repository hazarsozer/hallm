"""FFN Jacobian symmetry (spec 2026-09-15 §5)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from hallm.model import GPT, SHAPES, arm_config
from hallm.symmetry import (antisymmetric_share, effective_ffn_weights, gelu_grad,
                            hallm_rotation_shares, rotation_share)

SMOKE = SHAPES["smoke"]


def test_gelu_grad_matches_autograd():
    z = torch.linspace(-6, 6, 101, dtype=torch.float64, requires_grad=True)
    (g,) = torch.autograd.grad(F.gelu(z).sum(), z)
    assert torch.allclose(gelu_grad(z.detach()), g, atol=1e-10)


def test_antisymmetric_share_extremes():
    a = torch.randn(3, 8, 8, dtype=torch.float64)
    assert torch.allclose(antisymmetric_share(a - a.transpose(1, 2)), torch.ones(3, dtype=torch.float64))
    assert torch.allclose(antisymmetric_share(a + a.transpose(1, 2)), torch.zeros(3, dtype=torch.float64))


def test_halvit_ffn_is_exactly_symmetric():
    torch.manual_seed(0)
    w = torch.randn(64, 16)            # (h, d): up = W, down = Wᵀ
    u = torch.randn(40, 16)
    assert rotation_share(w, w.t(), u) < 1e-20


def test_random_ffn_is_about_half_rotation():
    torch.manual_seed(0)
    share = rotation_share(torch.randn(256, 64), torch.randn(64, 256), torch.randn(64, 64))
    assert 0.4 < share < 0.6


def test_bias_shifts_the_gate():
    torch.manual_seed(0)
    w_up, w_down, u = torch.randn(64, 16), torch.randn(16, 64), torch.randn(20, 16)
    assert rotation_share(w_up, w_down, u) != rotation_share(w_up, w_down, u, b_up=torch.full((64,), 3.0))


def test_collector_matches_a_direct_computation():
    torch.manual_seed(0)
    model = GPT(arm_config(SMOKE, "A0")).eval()
    idx = torch.randint(0, SMOKE.vocab_size, (2, 10))
    shares = hallm_rotation_shares(model, idx, n_positions=20)
    with torch.no_grad():
        x = model.tok_emb(idx) + model.pos_emb(torch.arange(10))
        for layer, block in enumerate(model.blocks):
            x = x + block.attn(block.ln1(x))
            u = block.ln2(x).reshape(-1, SMOKE.n_embd)
            direct = rotation_share(block.mlp.fc.weight, block.mlp.proj.weight, u)
            assert abs(shares[layer] - direct) < 1e-9
            x = x + block.mlp(block.ln2(x))


def test_collector_gives_zero_for_a_w_plus_wt_model():
    model = GPT(arm_config(SMOKE, "A2")).eval()
    shares = hallm_rotation_shares(model, torch.randint(0, SMOKE.vocab_size, (2, 10)), n_positions=20)
    assert len(shares) == SMOKE.n_layer and max(shares) < 1e-20


def test_collector_uses_transposed_weights_on_pass_two():
    model = GPT(arm_config(SMOKE, "A1u1t")).eval()          # L=2: layer 0 W, layer 1 Wᵀ
    mlp = model.blocks[0].mlp
    up, down = effective_ffn_weights(mlp, transposed=True)
    assert torch.equal(up, mlp.proj.weight.t()) and torch.equal(down, mlp.fc.weight.t())
    shares = hallm_rotation_shares(model, torch.randint(0, SMOKE.vocab_size, (2, 10)), n_positions=20)
    assert len(shares) == 2
