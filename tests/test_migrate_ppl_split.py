"""One-time split-label migration (2026-09-14): every recorded test_ppl was scored on val.bin."""
from __future__ import annotations

from pathlib import Path

from scripts.migrate_ppl_split import migrate_row


def test_renames_test_ppl_to_val_ppl():
    row = migrate_row({"run": "L8-A0-s1337", "test_ppl": 26.061, "weight_bytes_bf16": 1}, None)
    assert row["val_ppl"] == 26.061 and "test_ppl" not in row


def test_is_idempotent_and_keeps_a_true_test_ppl():
    row = {"run": "L8-A0-s1337", "val_ppl": 26.0, "test_ppl": 26.5, "weight_bytes_bf16": 1}
    assert migrate_row(dict(row), None) == row


def test_backfills_memory_fields_from_the_run_config():
    row = migrate_row({"run": "L4-A0-s1337", "test_ppl": 29.1429},
                      Path("configs/runs/L4-A0-s1337.yaml"))
    assert row["kv_bytes_ctx512_b1"] == 2 * 512 * 4 * 512 * 1 * 2  # 2·d·L·ctx·batch·bf16
    assert row["weight_bytes_bf16"] > 0 and "nonemb_weight_bytes_bf16" in row
