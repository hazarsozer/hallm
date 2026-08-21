"""Upload finished run artifacts to the HuggingFace checkpoint store.

ADD-ONLY BY CONSTRUCTION. This script never deletes, moves, renames or overwrites anything on the
Hub: it writes only to `checkpoints/<run-id>/` paths and skips any run already present there. The
old flat layout (`A0/`, `A2-iso/`, `runs/ladder/...`) is left untouched — folding those into the
run-addressed scheme is a separate, deliberate migration that must verify hashes before removing
anything (artifact-layout spec, 2026-08-19).

Target layout per run:
    checkpoints/<run-id>/model.pt
    checkpoints/<run-id>/manifest.json
    checkpoints/<run-id>/config.yaml

Usage:
    uv run --with huggingface_hub python scripts/hf_sync.py --dry-run
    uv run --with huggingface_hub python scripts/hf_sync.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ID = "hallm-thesis/hallm-wikitext103"


def discover_runs(runs_dir: Path, configs_dir: Path) -> list[dict]:
    """Every run with a FINAL checkpoint (not resume.pt) plus its manifest and config."""
    found = []
    for d in sorted(runs_dir.glob("*")):
        if not d.is_dir():
            continue
        run_id = d.name
        ckpt = d / f"{run_id}.pt"
        if not ckpt.exists():
            continue  # unfinished run — nothing to publish
        found.append({
            "run_id": run_id,
            "model": ckpt,
            "manifest": d / "manifest.json",
            "config": configs_dir / f"{run_id}.yaml",
        })
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default="runs/ladder")
    ap.add_argument("--configs", default="configs/runs")
    ap.add_argument("--repo", default=REPO_ID)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from huggingface_hub import HfApi

    api = HfApi()
    info = api.model_info(args.repo)
    remote = {s.rfilename for s in info.siblings}
    print(f"repo {args.repo} — private={info.private}, {len(remote)} files present\n")

    runs = discover_runs(Path(args.runs), Path(args.configs))
    todo, skipped = [], []
    for r in runs:
        if f"checkpoints/{r['run_id']}/model.pt" in remote:
            skipped.append(r["run_id"])
        else:
            todo.append(r)

    if skipped:
        print(f"already on the Hub ({len(skipped)}): {', '.join(skipped)}\n")
    if not todo:
        print("nothing to upload.")
        return 0

    print(f"to upload ({len(todo)}):")
    total = 0
    for r in todo:
        size = r["model"].stat().st_size
        total += size
        missing = [k for k in ("manifest", "config") if not r[k].exists()]
        note = f"  [WARN missing: {', '.join(missing)}]" if missing else ""
        print(f"   {r['run_id']:<20} {size/1e6:8.1f} MB{note}")
    print(f"\n   total {total/1e9:.2f} GB")

    if args.dry_run:
        print("\n--dry-run: nothing uploaded.")
        return 0

    for r in todo:
        rid = r["run_id"]
        for local, remote_name in ((r["model"], "model.pt"),
                                   (r["manifest"], "manifest.json"),
                                   (r["config"], "config.yaml")):
            if not local.exists():
                print(f"   skip {rid}/{remote_name} (local file absent)")
                continue
            api.upload_file(
                path_or_fileobj=str(local),
                path_in_repo=f"checkpoints/{rid}/{remote_name}",
                repo_id=args.repo,
                repo_type="model",
            )
            print(f"   uploaded checkpoints/{rid}/{remote_name}")

    after = {s.rfilename for s in api.model_info(args.repo).siblings}
    lost = remote - after
    if lost:
        print(f"\n!! {len(lost)} previously present path(s) are GONE — investigate immediately:")
        for p in sorted(lost):
            print("   ", p)
        return 1
    print(f"\nverified: {len(after)} files now present, none of the previous {len(remote)} lost.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
