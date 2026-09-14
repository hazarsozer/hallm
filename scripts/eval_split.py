"""Score finished checkpoints on a held-out split and merge `<split>_ppl` into each run's result.

Inference-only. Defaults to CPU so it can run while the GPU trains.

Usage: uv run python scripts/eval_split.py --split test --data data \
           --checkpoints 'runs/ladder/*/*.pt' [--device cpu]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import torch

from hallm.data import load_bin
from hallm.eval import evaluate_perplexity
from hallm.manifest import file_sha256
from hallm.results import result_path, update_run_result
from hallm.train import build_model_from_checkpoint


def eval_split(checkpoints: list[str], data_path: Path, split: str, results_dir: Path,
               device: str, batch_size: int = 8, dataset: str = "wikitext-103") -> dict[str, float]:
    data, sha = load_bin(data_path), file_sha256(data_path)
    out: dict[str, float] = {}
    for path in checkpoints:
        name = Path(path).stem
        if name == "resume":
            continue
        rp = result_path(results_dir, name)
        if not rp.exists():
            print(f"[skip] {name}: no result file")
            continue
        # F4: a row with no "dataset" field predates the field — every legacy and pilot row — and
        # is wikitext-103.
        row_dataset = json.loads(rp.read_text(encoding="utf-8")).get("dataset", "wikitext-103")
        if row_dataset != dataset:
            print(f"[skip] {name}: dataset '{row_dataset}' != requested '{dataset}'")
            continue
        model, cfg = build_model_from_checkpoint(path, map_location=device)
        model.to(device)
        ppl = round(evaluate_perplexity(model, data, cfg.block_size, batch_size, device), 4)
        update_run_result(results_dir, name, {f"{split}_ppl": ppl, f"{split}_split_sha256": sha})
        out[name] = ppl
        print(f"{name}: {split}_ppl {ppl}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--data", default="data")
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--results-dir", default="results/runs")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dataset", default="wikitext-103")
    args = ap.parse_args()
    if args.device == "cpu":
        torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))  # leave the trainer its data loader
    paths = sorted(p for pattern in args.checkpoints for p in glob.glob(pattern))
    eval_split(paths, Path(args.data) / f"{args.split}.bin", args.split, Path(args.results_dir),
               args.device, dataset=args.dataset)


if __name__ == "__main__":
    main()
