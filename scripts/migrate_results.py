"""One-pass, idempotent migration of legacy result files into per-run result files.

Three naming eras produced results in three shapes (artifact-layout spec 2026-08-19):
  results/ladder.jsonl            append-only ledger, rows already carry a "run" id
  results/four-arm-results.json   Experiment 1, keyed by bare arm at L8 seed 1337
  results/iso-*-results.json      Experiment 2, keyed by bare arm at L16 seed 1337

All fold into results/runs/<run-id>.json. Safe to re-run: each run's file is rewritten from its
own source, so nothing accumulates or duplicates.

Usage: uv run python scripts/migrate_results.py [--results results] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hallm.results import write_run_result

# Legacy files map bare arm tags onto the run-ID grammar. These are the unqueued seed-1337
# slot-ins of the ladder, so the mapping is exact and collision-free.
LEGACY = {
    "four-arm-results.json": {"A0": "L8-A0-s1337", "A1": "L8-A1-s1337",
                              "A2": "L8-A2-s1337", "A3": "L8-A3-s1337"},
    "iso-a2-results.json": {"A2": "L16-A2-s1337"},
    "iso-a0deep-results.json": {"A0": "L16-A0-s1337"},
}


def migrate(results_dir: str | Path, dry_run: bool = False,
            ledger_name: str = "ladder.jsonl") -> dict[str, str]:
    results_dir = Path(results_dir)
    runs_dir = results_dir / "runs"
    written: dict[str, str] = {}

    # `ledger_name` lets a collaborator convert their own ledger (e.g. ladder-alper.jsonl)
    # without renaming it first.
    ledger = results_dir / ledger_name
    if ledger.exists():
        for line in ledger.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not row.get("run"):
                continue
            if not dry_run:
                write_run_result(runs_dir, row)
            written[row["run"]] = ledger_name

    for fname, arm_map in LEGACY.items():
        p = results_dir / fname
        if not p.exists():
            continue
        for row in json.loads(p.read_text(encoding="utf-8")):
            run_id = arm_map.get(row.get("arm"))
            if not run_id:
                continue
            # Marked so a reader never mistakes a backfilled row for one this pipeline produced:
            # these predate per-run metrics, memory accounting and manifest linkage.
            row = {**row, "run": run_id, "retroactive": True, "source_file": fname}
            if not dry_run:
                write_run_result(runs_dir, row)
            written[run_id] = fname
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--ledger", default="ladder.jsonl",
                    help="ledger filename inside --results to convert (e.g. ladder-alper.jsonl)")
    args = ap.parse_args()
    written = migrate(args.results, args.dry_run, ledger_name=args.ledger)
    verb = "would write" if args.dry_run else "wrote"
    print(f"{verb} {len(written)} per-run result file(s) to {args.results}/runs/")
    for run_id, src in sorted(written.items()):
        print(f"  {run_id:<20} <- {src}")


if __name__ == "__main__":
    main()
