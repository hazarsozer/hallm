"""Transposed loop (spec 2026-09-15): config, sublayer transposition, pass routing, α, and the
invariant that every pre-existing arm is untouched."""

from __future__ import annotations

import torch

import prepatch_fixture


def test_default_path_is_bit_identical():
    """loop_pass2=None ⇒ the exact logits the pre-amendment code produced (CPU, float32)."""
    golden = torch.load(prepatch_fixture.FIXTURE, weights_only=True)
    now = prepatch_fixture.make()
    assert set(golden) == set(now) == set(prepatch_fixture.CASES)
    for arm in golden:
        assert torch.equal(golden[arm], now[arm]), arm
