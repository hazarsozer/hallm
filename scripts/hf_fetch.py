"""Download finished checkpoints from the HF store into the local run layout. READ-ONLY on the Hub.

    checkpoints/<run-id>/model.pt  →  runs/ladder/<run-id>/<run-id>.pt

Default: every run that has a result file. Runs with no weights on the Hub (4 known, written off
2026-09-14) are reported and skipped.

Usage: uv run --group data python scripts/hf_fetch.py [--runs L8-A0-s1337 ...]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

REPO_ID = "hallm-thesis/hallm-wikitext103"


def main() -> None:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--results-dir", default="results/runs")
    ap.add_argument("--dest", default="runs/ladder")
    args = ap.parse_args()
    ids = args.runs or sorted(p.stem for p in Path(args.results_dir).glob("*.json"))
    staging = Path(args.dest) / ".hf-staging"
    for rid in ids:
        dest = Path(args.dest) / rid / f"{rid}.pt"
        if dest.exists():
            continue
        try:
            got = hf_hub_download(REPO_ID, f"checkpoints/{rid}/model.pt", local_dir=staging)
        except EntryNotFoundError:
            print(f"[missing] {rid}: no weights on the Hub")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        os.replace(got, dest)
        print(f"[fetched] {rid}")


if __name__ == "__main__":
    main()
