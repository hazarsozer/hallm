"""Real-training driver — trains all four arms on WikiText-103 under a matched budget and emits the
four-way comparison table (FR-3/4/5/6). This is a Term-2 / GPU step.

>>> NOT executed by the overnight loop. <<<  It is provided READY to run when you choose to.

Usage
-----
1) Get WikiText-103 (e.g. from Hugging Face `wikitext` / `wikitext-103-raw-v1`) as plain-text files,
   then tokenize them to uint16 token bins:

     uv run python scripts/run_real_training.py prepare \
         --train wiki.train.tokens --val wiki.valid.tokens --out data/

2) Train + evaluate every arm config in `configs/arm*.yaml` and write the comparison table:

     uv run python scripts/run_real_training.py run --data data/ --configs configs --out runs/

All four arms share one TrainConfig and differ only in the sharing flags (matched budget), so any
perplexity gap is attributable to the sharing scheme.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import torch

from hallm.data import load_bin, prepare_bin
from hallm.eval import comparison_table, evaluate_arm
from hallm.experiment import load_experiment
from hallm.model import GPT
from hallm.train import save_checkpoint, set_seed, train


def cmd_prepare(args: argparse.Namespace) -> None:
    Path(args.out).mkdir(parents=True, exist_ok=True)
    n_train = prepare_bin(args.train, os.path.join(args.out, "train.bin"))
    n_val = prepare_bin(args.val, os.path.join(args.out, "val.bin"))
    print(f"prepared: train={n_train:,} tokens, val={n_val:,} tokens → {args.out}/")


def cmd_run(args: argparse.Namespace) -> None:
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if os.environ.get("SHAREDLM_NO_FLASH_SDP"):
        # workaround for wedged flash-attention backward kernels (2026-08-18 hangs);
        # math/mem-efficient SDP computes the identical attention, just slower
        torch.backends.cuda.enable_flash_sdp(False)
        print("flash SDP disabled (SHAREDLM_NO_FLASH_SDP)")
    train_data = load_bin(os.path.join(args.data, "train.bin"))
    val_data = load_bin(os.path.join(args.data, "val.bin"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for cfg_file in sorted(glob.glob(os.path.join(args.configs, "arm*.yaml"))):
        model_cfg, train_cfg = load_experiment(cfg_file)
        set_seed(train_cfg.seed, train_cfg.deterministic)
        model = GPT(model_cfg)
        print(f"\n=== {model_cfg.arm} ({Path(cfg_file).name}) — {train_cfg.max_steps} steps on {device} ===")
        train(model, train_cfg, train_data, device=device, progress=True)
        save_checkpoint(model, model_cfg, train_cfg, out / f"{model_cfg.arm}.pt")
        row = evaluate_arm(model, model_cfg, val_data, batch_size=8, device=device)
        rows.append(row)
        print(row)

    table = comparison_table(rows)
    (out / "comparison.md").write_text(table + "\n", encoding="utf-8")
    (out / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("\nFour-way comparison (also written to", out / "comparison.md", "):\n")
    print(table)


def main() -> None:
    parser = argparse.ArgumentParser(description="hallm real-training driver (Term-2, GPU)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_prep = sub.add_parser("prepare", help="tokenize WikiText-103 text → uint16 bins")
    p_prep.add_argument("--train", required=True, help="training text file")
    p_prep.add_argument("--val", required=True, help="validation text file")
    p_prep.add_argument("--out", default="data", help="output dir for train.bin / val.bin")
    p_prep.set_defaults(func=cmd_prepare)

    p_run = sub.add_parser("run", help="train + evaluate all arms")
    p_run.add_argument("--data", default="data", help="dir with train.bin / val.bin")
    p_run.add_argument("--configs", default="configs", help="dir with arm*.yaml")
    p_run.add_argument("--out", default="runs", help="output dir for checkpoints + results")
    p_run.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    p_run.set_defaults(func=cmd_run)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
