"""One-time migration (2026-09-14): every `test_ppl` recorded before this date was computed on
WikiText-103 val.bin — the runner never read test.bin. Rename it to `val_ppl`, and backfill the
measured-memory fields that pre-P0 rows lack by rebuilding the model from the run's config.

Idempotent: a row that already carries `val_ppl` keeps its PPL keys untouched, so a true `test_ppl`
written later by scripts/eval_split.py survives a re-run.

Usage: uv run python scripts/migrate_ppl_split.py [--results-dir results/runs] [--configs configs/runs]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from hallm.results import read_run_results, write_run_result


def migrate_row(row: dict, config_path: Path | None) -> dict:
    row = dict(row)
    if "val_ppl" not in row and "test_ppl" in row:
        row["val_ppl"] = row.pop("test_ppl")
    if "weight_bytes_bf16" not in row and config_path is not None and config_path.exists():
        from hallm.experiment import load_experiment
        from hallm.metrics import memory_row
        from hallm.model import GPT

        model_cfg, _ = load_experiment(config_path)
        row.update(memory_row(GPT(model_cfg), model_cfg))
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default="results/runs")
    ap.add_argument("--configs", default="configs/runs")
    args = ap.parse_args()
    for row in read_run_results(args.results_dir):
        cfg = Path(args.configs) / f"{row['run']}.yaml"
        new = migrate_row(row, cfg)
        if new != row:
            write_run_result(args.results_dir, new)
            print(f"migrated {row['run']}")


if __name__ == "__main__":
    main()
