"""Smoke tests — every arm instantiates, forwards with finite loss, and can learn (FR-7).

The overfit-a-batch decrease is the end-to-end proof that gradients flow correctly through the shared
W/Wᵀ transpose path (no detached tensors / stop-grad), which is the subtle failure mode for A2/A3.
"""

from __future__ import annotations

import pytest
import torch

from hallm.model import GPT, SHAPES, arm_config
from hallm.train import configure_optimizer

SMOKE = SHAPES["smoke"]
ARMS = ["A0", "A1", "A2", "A3"]


@pytest.mark.parametrize("arm", ARMS)
def test_forward_finite(arm: str) -> None:
    torch.manual_seed(0)
    model = GPT(arm_config(SMOKE, arm))
    x = torch.randint(0, SMOKE.vocab_size, (2, 16))
    y = torch.randint(0, SMOKE.vocab_size, (2, 16))
    logits, loss = model(x, y)
    assert logits.shape == (2, 16, SMOKE.vocab_size)
    assert torch.isfinite(loss)


@pytest.mark.parametrize("arm", ARMS)
def test_overfit_a_batch_decreases(arm: str) -> None:
    torch.manual_seed(0)
    model = GPT(arm_config(SMOKE, arm))
    x = torch.randint(0, SMOKE.vocab_size, (4, 32))
    y = torch.randint(0, SMOKE.vocab_size, (4, 32))
    opt = configure_optimizer(model, weight_decay=0.0, lr=1e-2, betas=(0.9, 0.95))

    loss0 = model(x, y)[1].item()
    for _ in range(50):
        opt.zero_grad()
        loss = model(x, y)[1]
        loss.backward()
        opt.step()
    loss1 = model(x, y)[1].item()

    assert torch.isfinite(torch.tensor(loss1))
    assert loss1 < 0.8 * loss0, f"{arm}: loss did not clearly decrease ({loss0:.3f} -> {loss1:.3f})"
