import pytest
import torch
from hallm.model.config import VSHAPES, VisionConfig, arm_config
from hallm.model.vit import ViT


def tiny(**kw):
    base = dict(image_size=32, patch_size=16, n_classes=10, n_embd=32, n_layer=4, n_head=2,
                block_size=5, vocab_size=1, tie_embeddings=False)
    base.update(kw)
    return VisionConfig(**base)


def test_block_size_must_match_patch_count_plus_class_token():
    with pytest.raises(ValueError, match="block_size"):
        tiny(block_size=7)


def test_image_size_must_divide_by_patch_size():
    with pytest.raises(ValueError, match="divisible"):
        tiny(image_size=30, block_size=5)


def test_forward_returns_class_logits_and_loss():
    model = ViT(tiny()).eval()
    pixels = torch.randn(3, 3, 32, 32)
    targets = torch.tensor([1, 2, 3])
    logits, loss = model(pixels, targets)
    assert logits.shape == (3, 10)
    assert loss.ndim == 0 and loss.item() > 0


def test_attention_is_not_causal():
    """A ViT must mix both ways: perturbing the last patch changes the first token's output."""
    torch.manual_seed(0)
    model = ViT(tiny()).eval()
    pixels = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        a, _ = model(pixels)
        pixels2 = pixels.clone()
        pixels2[0, :, -16:, -16:] += 5.0
        b, _ = model(pixels2)
    assert not torch.allclose(a, b, atol=1e-5)


def test_arms_build_over_the_same_sharing_code():
    for arm, n_blocks in [("A0", 4), ("A2", 4), ("A1u2", 2)]:
        model = ViT(arm_config(tiny(), arm))
        assert len(model.blocks) == n_blocks


def test_looped_arm_stores_what_the_shallower_unshared_model_stores():
    looped = ViT(arm_config(tiny(n_layer=4), "A1u2"))
    shallow = ViT(arm_config(tiny(n_layer=2, block_size=5), "A0"))
    assert looped.num_parameters(non_embedding=True) == shallow.num_parameters(non_embedding=True)


def test_non_embedding_excludes_patch_pos_head_and_class_token():
    model = ViT(tiny())
    blocks_only = sum(p.numel() for p in model.blocks.parameters())
    # final LayerNorm rides with the blocks; patch/pos/head/cls do not
    assert model.num_parameters(non_embedding=True) == blocks_only + sum(
        p.numel() for p in model.ln_f.parameters()
    )


def test_transposed_loop_reports_alphas():
    model = ViT(arm_config(tiny(n_layer=4), "A1u2a"))
    scales = model.loop_scales()
    assert set(scales) == {"alpha_attn_0", "alpha_mlp_0", "alpha_attn_1", "alpha_mlp_1"}


def test_vshapes_match_the_spec_geometry():
    for key, n_layer in [("v4", 4), ("v8", 8)]:
        cfg = VSHAPES[key]
        assert (cfg.image_size, cfg.patch_size, cfg.n_embd, cfg.n_head) == (112, 16, 512, 8)
        assert cfg.n_layer == n_layer
        assert cfg.block_size == (112 // 16) ** 2 + 1 == 50
