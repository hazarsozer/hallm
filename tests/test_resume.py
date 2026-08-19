"""Resume-equivalence: interrupted-and-resumed training must equal an unbroken run (spec §8.1)."""
from __future__ import annotations

from dataclasses import replace

import torch

from hallm.data import make_synthetic_data
from hallm.model import GPT, SHAPES
from hallm.train import (
    TrainConfig,
    load_resume_checkpoint,
    save_resume_checkpoint,
    set_seed,
    train,
)

CFG = SHAPES["smoke"]
TC = TrainConfig(
    max_steps=8, warmup_steps=2, batch_size=4, grad_accum=1, block_size=CFG.block_size,
    dtype="float32", seed=7, deterministic=True, checkpoint_interval=4, log_interval=1,
)
DATA = make_synthetic_data(CFG.vocab_size, 4096, seed=0)


def _fresh_model() -> GPT:
    set_seed(TC.seed, TC.deterministic)
    return GPT(CFG)


def test_resume_checkpoint_roundtrip(tmp_path):
    model = _fresh_model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(7)
    path = tmp_path / "resume.pt"
    save_resume_checkpoint(path, model, TC, opt, gen, step=3)
    ckpt = load_resume_checkpoint(path)  # must survive weights_only=True
    assert ckpt["step"] == 3
    assert ckpt["model_cfg"]["n_layer"] == CFG.n_layer
    assert ckpt["train_cfg"]["max_steps"] == TC.max_steps
    assert torch.equal(ckpt["gen_state"], gen.get_state())


def test_stop_step_writes_checkpoint(tmp_path):
    path = tmp_path / "resume.pt"
    model = _fresh_model()
    history = train(model, TC, DATA, device="cpu", resume_path=str(path), stop_step=4)
    assert path.exists()
    assert load_resume_checkpoint(path)["step"] == 4
    assert max(h["step"] for h in history) == 3  # steps 0..3 ran


def test_resume_equals_unbroken_run(tmp_path):
    # unbroken 8-step run
    m_full = _fresh_model()
    train(m_full, TC, DATA, device="cpu")
    # same run interrupted at step 4, then resumed by a FRESH process (fresh model object)
    path = tmp_path / "resume.pt"
    m_a = _fresh_model()
    train(m_a, TC, DATA, device="cpu", resume_path=str(path), stop_step=4)
    m_b = GPT(CFG)  # arbitrary init — resume must overwrite it entirely
    train(m_b, TC, DATA, device="cpu", resume_path=str(path))
    sd_full, sd_res = m_full.state_dict(), m_b.state_dict()
    assert sd_full.keys() == sd_res.keys()
    for k in sd_full:
        assert torch.equal(sd_full[k], sd_res[k]), f"param {k} diverged after resume"
