"""Transposed loop (spec 2026-09-15): config, sublayer transposition, pass routing, α, and the
invariant that every pre-existing arm is untouched."""

from __future__ import annotations

import dataclasses

import pytest
import torch

import prepatch_fixture
from hallm.model import GPT, SHAPES, arm_config
from hallm.model.config import ModelConfig

SMOKE = SHAPES["smoke"]


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
