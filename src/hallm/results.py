"""Per-run result files.

One JSON file per run, keyed by run ID — NOT a single append-only ledger. The ledger form was
rejected (Hazar, 2026-08-20) because it cannot be re-derived from the runs it describes, a crashed
or interleaved write corrupts every row after it, and re-running one run either duplicates its row
or silently conflicts with the old one. Per-run files make each write independent and idempotent:
a run owns exactly one file and touching it cannot disturb any other run.

Consolidated views (comparison tables, ledgers) are GENERATED from these files by
`scripts/build_reports.py` and are always safe to delete and rebuild.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def result_path(results_dir: str | Path, run_id: str) -> Path:
    return Path(results_dir) / f"{run_id}.json"


def write_run_result(results_dir: str | Path, row: dict) -> Path:
    """Write one run's result atomically. Overwriting the SAME run ID is intended and safe."""
    run_id = row.get("run")
    if not run_id:
        raise ValueError("result row must carry a 'run' id")
    d = Path(results_dir)
    d.mkdir(parents=True, exist_ok=True)
    p = result_path(d, run_id)
    tmp = str(p) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(row, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, p)  # atomic: a crash mid-write never leaves a half-file in place
    return p


def read_run_results(results_dir: str | Path) -> list[dict]:
    """Every run result, sorted by run ID. Missing directory ⇒ empty list."""
    d = Path(results_dir)
    if not d.is_dir():
        return []
    rows = []
    for p in sorted(d.glob("*.json")):
        try:
            rows.append(json.loads(p.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    return sorted(rows, key=lambda r: r.get("run", ""))
