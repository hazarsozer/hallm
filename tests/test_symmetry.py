"""FFN Jacobian symmetry (spec 2026-09-15 §5)."""

from __future__ import annotations

import json

import numpy as np
import pytest
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


def test_in_scale_breaks_symmetry_for_a_w_plus_wt_layer():
    """Spec 2026-09-15 §5 (added after the final code review): symmetry is basis-dependent. A
    W+Wᵀ FFN is exactly symmetric w.r.t. u = LN(x) (in_scale=None, or a uniform gain), but not
    w.r.t. the pre-gain x̂ once the LayerNorm gain is non-uniform."""
    torch.manual_seed(0)
    w = torch.randn(64, 16)            # (h, d): up = W, down = Wᵀ
    u = torch.randn(40, 16)
    assert rotation_share(w, w.t(), u) < 1e-20
    assert rotation_share(w, w.t(), u, in_scale=torch.ones(16)) < 1e-20
    assert rotation_share(w, w.t(), u, in_scale=torch.linspace(0.5, 2.0, 16)) > 0


def test_collector_xhat_basis_matches_a_direct_computation_with_ln2_gain():
    torch.manual_seed(0)
    model = GPT(arm_config(SMOKE, "A0")).eval()
    with torch.no_grad():
        for block in model.blocks:
            block.ln2.weight.copy_(torch.linspace(0.5, 1.5, SMOKE.n_embd))
    idx = torch.randint(0, SMOKE.vocab_size, (2, 10))
    shares = hallm_rotation_shares(model, idx, n_positions=20, basis="xhat")
    with torch.no_grad():
        x = model.tok_emb(idx) + model.pos_emb(torch.arange(10))
        for layer, block in enumerate(model.blocks):
            x = x + block.attn(block.ln1(x))
            u = block.ln2(x).reshape(-1, SMOKE.n_embd)
            direct = rotation_share(block.mlp.fc.weight, block.mlp.proj.weight, u, in_scale=block.ln2.weight)
            assert abs(shares[layer] - direct) < 1e-9
            x = x + block.mlp(block.ln2(x))


def test_hallm_rotation_shares_rejects_an_unknown_basis():
    model = GPT(arm_config(SMOKE, "A0")).eval()
    with pytest.raises(ValueError):
        hallm_rotation_shares(model, torch.randint(0, SMOKE.vocab_size, (1, 4)), basis="bogus")


def test_vit_collector_accepts_the_xhat_basis():
    transformers = pytest.importorskip("transformers")
    from hallm.symmetry import vit_rotation_shares

    cfg = transformers.ViTConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                                 intermediate_size=64, image_size=32, patch_size=8, hidden_act="gelu")
    torch.manual_seed(0)
    vit = transformers.ViTModel(cfg).eval()
    shares = vit_rotation_shares(vit, torch.randn(3, 3, 32, 32), n_positions=30, basis="xhat")
    assert len(shares) == 2 and all(0.0 <= s <= 1.0 for s in shares)


def test_deit_row_on_a_tiny_local_vit(tmp_path):
    """Covers `deit_row`'s code path (finding: AutoImageProcessor needs torchvision, not installed)
    entirely offline: a tiny ViTModel saved with `save_pretrained`, a DeiT-style preprocessor built
    directly (not downloaded) and saved alongside it, and 2 generated JPEGs."""
    transformers = pytest.importorskip("transformers")
    from PIL import Image
    from transformers import DeiTImageProcessorPil, ViTConfig, ViTModel

    from scripts.ffn_symmetry import deit_row

    model_dir = tmp_path / "model"
    cfg = ViTConfig(hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
                    intermediate_size=32, image_size=32, patch_size=8, hidden_act="gelu")
    torch.manual_seed(0)
    ViTModel(cfg).save_pretrained(model_dir)
    DeiTImageProcessorPil(size={"height": 32, "width": 32},
                          crop_size={"height": 32, "width": 32}).save_pretrained(model_dir)

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    rng = np.random.default_rng(0)
    for i in range(2):
        img = Image.fromarray((rng.random((32, 32, 3)) * 255).astype("uint8"))
        img.save(images_dir / f"img{i}.JPEG")

    row = deit_row(str(images_dir), n_images=2, n_positions=4, device="cpu", model_name=str(model_dir))
    assert row["kind"] == "vision" and row["arm"] == "A0"
    assert len(row["per_layer"]) == 2 and len(row["per_layer_xhat"]) == 2
    assert "mean" in row and "mean_xhat" in row


def test_collector_gives_zero_for_a_w_plus_wt_model():
    model = GPT(arm_config(SMOKE, "A2")).eval()
    shares = hallm_rotation_shares(model, torch.randint(0, SMOKE.vocab_size, (2, 10)), n_positions=20)
    assert len(shares) == SMOKE.n_layer and max(shares) < 1e-20


def test_collector_uses_transposed_weights_on_pass_two():
    model = GPT(arm_config(SMOKE, "A1u1t")).eval()          # L=2: layer 0 W, layer 1 Wᵀ
    mlp = model.blocks[0].mlp
    up, down, b_up = effective_ffn_weights(mlp, transposed=True)
    assert torch.equal(up, mlp.proj.weight.t()) and torch.equal(down, mlp.fc.weight.t()) and b_up is None
    shares = hallm_rotation_shares(model, torch.randint(0, SMOKE.vocab_size, (2, 10)), n_positions=20)
    assert len(shares) == 2


def test_hallm_shares_match_autograd_jacobian_oracle():
    """Oracle check against torch.autograd.functional.jacobian: pins the w_up/w_down orientation
    inside rotation_share (gating with u @ w_down instead of u @ w_up.T would fail this) and the
    transposed flag threaded through the collector's hook (hard-coding transposed=False would fail
    the A1u1t layer-1 case)."""
    torch.manual_seed(0)
    for arm in ("A0", "A1u1t"):
        model = GPT(arm_config(SMOKE, arm)).double().eval()
        with torch.no_grad():
            for p in model.parameters():
                if p.dim() == 2:
                    p.mul_(20.0)                    # push GELU gates away from ~0.5
        idx = torch.randint(0, SMOKE.vocab_size, (2, 10))
        n_pos = 2 * 10                              # = B*T: every position, matching the collector
        shares = hallm_rotation_shares(model, idx, n_positions=n_pos)
        calls: list[tuple[torch.nn.Module, torch.Tensor, bool]] = []
        hooks = [block.mlp.register_forward_pre_hook(
                     lambda m, args, kwargs: calls.append(
                         (m, args[0].detach(), kwargs.get("transposed", args[1] if len(args) > 1 else False))),
                     with_kwargs=True)
                 for block in model.blocks]
        with torch.no_grad():
            model(idx)
        for h in hooks:
            h.remove()
        for layer, (mlp, x, tr) in enumerate(calls):
            u = x.reshape(-1, x.shape[-1])
            jac = torch.stack([torch.autograd.functional.jacobian(
                lambda v: mlp(v, transposed=tr), u[i]) for i in range(u.shape[0])])
            oracle = float(antisymmetric_share(jac).mean())
            assert abs(shares[layer] - oracle) < 1e-9


def test_vit_collector_on_a_tiny_random_vit():
    transformers = pytest.importorskip("transformers")
    from hallm.symmetry import vit_rotation_shares

    cfg = transformers.ViTConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                                 intermediate_size=64, image_size=32, patch_size=8, hidden_act="gelu")
    torch.manual_seed(0)
    vit = transformers.ViTModel(cfg).eval()
    shares = vit_rotation_shares(vit, torch.randn(3, 3, 32, 32), n_positions=30)
    assert len(shares) == 2 and all(0.0 < s < 1.0 for s in shares)


def test_vit_collector_rejects_a_non_gelu_model():
    transformers = pytest.importorskip("transformers")
    from hallm.symmetry import vit_rotation_shares

    cfg = transformers.ViTConfig(hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                                 intermediate_size=64, image_size=32, patch_size=8, hidden_act="relu")
    with pytest.raises(ValueError):
        vit_rotation_shares(transformers.ViTModel(cfg).eval(), torch.randn(1, 3, 32, 32))


def test_script_writes_json_and_report_for_an_lm_checkpoint(tmp_path):
    from hallm.data import make_synthetic_data
    from hallm.train import TrainConfig, save_checkpoint
    from scripts.ffn_symmetry import main

    cfg = arm_config(SMOKE, "A0")
    save_checkpoint(GPT(cfg), cfg, TrainConfig(), tmp_path / "smoke-A0-s7.pt")
    data = tmp_path / "data"
    data.mkdir()
    make_synthetic_data(SMOKE.vocab_size, 4096, seed=0).tofile(data / "val.bin")
    out, rep = tmp_path / "sym.json", tmp_path / "sym.md"
    main(["--checkpoints", str(tmp_path / "*.pt"), "--data", str(data), "--n-positions", "32",
          "--windows", "2", "--out", str(out), "--report", str(rep), "--device", "cpu"])
    rows = json.loads(out.read_text())
    lm = next(r for r in rows if r["model"] == "smoke-A0-s7")
    assert lm["kind"] == "lm" and len(lm["per_layer"]) == SMOKE.n_layer
    assert any(r["model"] == "random" for r in rows)
    assert "smoke-A0-s7" in rep.read_text()
