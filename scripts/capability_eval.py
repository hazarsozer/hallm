"""Tier-1.5 capability evals over trained checkpoints (spec 06 §5). Inference-only; runs on CPU or
GPU without locking the box for long.

One-time data fetch (manual, like the WikiText prepare step):
  LAMBADA (OpenAI variant, jsonl):
    curl -L -o data/lambada_test.jsonl \
      https://openaipublic.blob.core.windows.net/gpt-2/data/lambada_test.jsonl
  BLiMP (67 paradigm jsonl files):
    git clone --depth 1 https://github.com/alexwarstadt/blimp /tmp/blimp && \
      mkdir -p data/blimp && cp /tmp/blimp/data/*.jsonl data/blimp/

Usage:
  uv run python scripts/capability_eval.py --checkpoints 'runs/ladder/*/*.pt' \
      --lambada data/lambada_test.jsonl --blimp data/blimp --data data/ --probes
Use --limit N to subsample each benchmark for a quick pass.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import torch

from hallm.capeval import (
    blimp_accuracy,
    frequency_bucket_loss,
    frequency_buckets,
    induction_accuracy,
    lambada_accuracy,
    load_blimp_file,
    load_lambada,
    position_loss,
    recall_accuracy,
    sliced_perplexity,
)
from hallm.data import GPT2_VOCAB_SIZE, load_bin
from hallm.results import result_path, update_run_result, write_run_result
from hallm.train import build_model_from_checkpoint


def save_row(out_dir: str | Path, row: dict) -> Path:
    """Create `<out_dir>/<run>.json` if it doesn't exist yet, else merge `row`'s fields into it.

    Each of --lambada/--blimp/--data/--probes is independently toggleable, so two passes over the
    same checkpoint (e.g. a --probes-only pass followed by a --lambada pass) must not have the
    second overwrite-and-erase the first's fields."""
    if result_path(out_dir, row["run"]).exists():
        return update_run_result(out_dir, row["run"], row)
    return write_run_result(out_dir, row)


def main() -> None:
    ap = argparse.ArgumentParser(description="hallm capability evals (LAMBADA / BLiMP / sliced PPL / Track 1 probes)")
    ap.add_argument("--checkpoints", nargs="+", required=True, help="glob(s) of .pt checkpoints")
    ap.add_argument("--lambada", default=None, help="lambada_test.jsonl (skip if omitted)")
    ap.add_argument("--blimp", default=None, help="dir of BLiMP paradigm .jsonl files (skip if omitted)")
    ap.add_argument("--data", default=None, help="dir with val.bin for sliced PPL (skip if omitted)")
    ap.add_argument("--n-slices", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None, help="subsample each benchmark to N examples")
    ap.add_argument("--probes", action="store_true",
                    help="position/frequency loss (needs --data with train.bin + val.bin), induction, recall")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="results/capability")
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    lam, blimp = None, {}
    if args.lambada or args.blimp:
        import tiktoken  # lazy: only needed when text benchmarks are requested (repo rule)

        encode = tiktoken.get_encoding("gpt2").encode_ordinary
        if args.lambada:
            lam = load_lambada(args.lambada, encode)[: args.limit]
        if args.blimp:
            blimp_files = sorted(glob.glob(str(Path(args.blimp) / "*.jsonl")))
            blimp = {Path(f).stem: load_blimp_file(f, encode)[: args.limit] for f in blimp_files}
    val = load_bin(Path(args.data) / "val.bin") if args.data else None

    buckets = None
    if args.probes and args.data:
        buckets = frequency_buckets(load_bin(Path(args.data) / "train.bin"), GPT2_VOCAB_SIZE, 10)

    paths = sorted(p for pattern in args.checkpoints for p in glob.glob(pattern))
    for path in paths:
        name = Path(path).stem
        if name == "resume":
            print(f"[skip] {path}: resume checkpoint, not a finished run")
            continue
        model, cfg = build_model_from_checkpoint(path, map_location=device)
        model.to(device).eval()
        row = {"run": name, "arm": cfg.arm, "n_layer": cfg.n_layer}
        if lam is not None:
            row["lambada_acc"] = round(lambada_accuracy(model, lam, device), 4)
        if blimp:
            per_task = {k: round(blimp_accuracy(model, v, device), 4) for k, v in blimp.items()}
            row["blimp_per_task"] = per_task
            row["blimp_macro"] = round(sum(per_task.values()) / len(per_task), 4)
        if val is not None:
            row["sliced_ppl"] = [round(p, 3) for p in
                                 sliced_perplexity(model, val, cfg.block_size, args.n_slices, device=device)]
        if args.probes:
            row["induction_acc"] = round(induction_accuracy(model, device=device), 4)
            row["recall_acc"] = round(recall_accuracy(model, device=device), 4)
            if val is not None:
                pl = position_loss(model, val, cfg.block_size, device=device)
                row["position_loss"], row["late_loss"] = [round(v, 4) for v in pl], round(pl[-1], 4)
            if val is not None and buckets is not None:
                fl = frequency_bucket_loss(model, val, buckets, 10, cfg.block_size, device=device)
                row["freq_loss"], row["rare_loss"] = [round(v, 4) for v in fl], round(fl[-1], 4)
        save_row(args.out, row)
        print(f"{name}: " + json.dumps({k: v for k, v in row.items() if k != 'blimp_per_task'}))

    print(f"\nwrote per-run results to {args.out}/")


if __name__ == "__main__":
    main()
