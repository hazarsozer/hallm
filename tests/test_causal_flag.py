import torch
from hallm.model.config import ModelConfig
from hallm.model.gpt import GPT
from hallm.model.sharing import CausalSelfAttention


def test_causal_defaults_to_true():
    assert ModelConfig().causal is True


def test_non_causal_attention_sees_later_positions():
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=16, block_size=8, n_embd=16, n_layer=1, n_head=2, causal=False)
    attn = CausalSelfAttention(cfg).eval()
    x = torch.randn(1, 8, 16)
    base = attn(x)
    x2 = x.clone()
    x2[0, -1] += 10.0            # perturb the LAST position only
    perturbed = attn(x2)
    # non-causal: position 0 must react to a change at the last position
    assert not torch.allclose(base[0, 0], perturbed[0, 0], atol=1e-5)


def test_causal_attention_ignores_later_positions():
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=16, block_size=8, n_embd=16, n_layer=1, n_head=2)
    attn = CausalSelfAttention(cfg).eval()
    x = torch.randn(1, 8, 16)
    base = attn(x)
    x2 = x.clone()
    x2[0, -1] += 10.0
    perturbed = attn(x2)
    assert torch.allclose(base[0, 0], perturbed[0, 0], atol=1e-6)


def test_lm_forward_unchanged_by_the_new_field():
    """Sanity check only: the default (`causal` omitted) matches an explicit `causal=True` config,
    both otherwise identical, so this is tautological rather than proof the field leaves existing
    LM runs bit-identical. That property is actually proven in
    `tests/test_transposed_loop.py::test_default_path_is_bit_identical`, which compares A0/A2/
    A2-attn/A1u2 against golden logits captured from the pre-amendment code."""
    def logits_for(cfg):
        torch.manual_seed(1234)
        model = GPT(cfg).eval()
        idx = torch.randint(0, cfg.vocab_size, (2, 16), generator=torch.Generator().manual_seed(7))
        with torch.no_grad():
            out, _ = model(idx)
        return out

    a = logits_for(ModelConfig(vocab_size=64, block_size=32, n_embd=32, n_layer=2, n_head=2))
    b = logits_for(ModelConfig(vocab_size=64, block_size=32, n_embd=32, n_layer=2, n_head=2, causal=True))
    assert torch.equal(a, b)
