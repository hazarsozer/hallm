import torch
from hallm.model.config import VisionConfig, arm_config
from hallm.model.vit import ViT
from hallm.symmetry import hallm_rotation_shares


def tiny(arm):
    cfg = VisionConfig(image_size=32, patch_size=16, n_classes=4, n_embd=32, n_layer=2,
                       n_head=2, block_size=5, vocab_size=1, tie_embeddings=False)
    return ViT(arm_config(cfg, arm))


def test_rotation_shares_run_on_a_vit_and_return_one_per_layer():
    torch.manual_seed(0)
    model = tiny("A0")
    pixels = torch.randn(4, 3, 32, 32)
    shares = hallm_rotation_shares(model, pixels, n_positions=8)
    assert len(shares) == 2
    assert all(0.0 <= s <= 1.0 for s in shares)


def test_w_plus_wt_vit_is_exactly_symmetric_in_the_u_basis():
    torch.manual_seed(0)
    model = tiny("A2")
    pixels = torch.randn(4, 3, 32, 32)
    shares = hallm_rotation_shares(model, pixels, n_positions=8)
    assert max(shares) < 1e-10          # the §3 sanity gate, as for L8-A2


def test_eval_mode_is_restored_after_measurement():
    model = tiny("A0")
    model.train()
    hallm_rotation_shares(model, torch.randn(2, 3, 32, 32), n_positions=4)
    assert model.training
