"""P0 item 3: the thesis's memory claim must be measured, not inferred from parameter counts."""

from hallm.metrics import kv_cache_bytes, memory_row
from hallm.model.config import SHAPES, arm_config
from hallm.model.gpt import GPT


def test_kv_cache_bytes_is_two_d_l_per_token():
    cfg = SHAPES["s30"]  # d=512, L=8
    assert kv_cache_bytes(cfg, ctx=1, batch=1, bytes_per_elem=2) == 2 * 512 * 8 * 2


def test_kv_cache_is_independent_of_sharing():
    """Sharing compresses weights only — this is why depth re-investment pays the saving back."""
    base = SHAPES["s30"]
    a0, a2 = arm_config(base, "A0"), arm_config(base, "A2")
    assert kv_cache_bytes(a0, 512, 1) == kv_cache_bytes(a2, 512, 1)


def test_kv_cache_doubles_with_depth():
    d8, d16 = SHAPES["s30"], SHAPES["s30x2"]
    assert kv_cache_bytes(d16, 512, 1) == 2 * kv_cache_bytes(d8, 512, 1)


def test_memory_row_reports_weight_fraction_below_one():
    cfg = SHAPES["smoke"]
    row = memory_row(GPT(cfg), cfg)
    assert 0.0 < row["weight_frac_of_total_ctx512_b1"] <= 1.0
    assert row["weight_bytes_bf16"] > 0
    assert row["kv_bytes_ctx2048_b8"] > row["kv_bytes_ctx512_b1"]
