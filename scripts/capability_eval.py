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
  uv run python scripts/capability_eval.py --checkpoints 'runs/*.pt' 'runs/ladder/*/*.pt' \
      --lambada data/lambada_test.jsonl --blimp data/blimp --data data/ --out results/
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
    lambada_accuracy,
    load_blimp_file,
    load_lambada,
    sliced_perplexity,
)
from hallm.data import load_bin
from hallm.train import build_model_from_checkpoint


def main() -> None:
    ap = argparse.ArgumentParser(description="hallm capability evals (LAMBADA / BLiMP / sliced PPL)")
    ap.add_argument("--checkpoints", nargs="+", required=True, help="glob(s) of .pt checkpoints")
    ap.add_argument("--lambada", default=None, help="lambada_test.jsonl (skip if omitted)")
    ap.add_argument("--blimp", default=None, help="dir of BLiMP paradigm .jsonl files (skip if omitted)")
    ap.add_argument("--data", default=None, help="dir with val.bin for sliced PPL (skip if omitted)")
    ap.add_argument("--n-slices", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None, help="subsample each benchmark to N examples")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="results")
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

    paths = sorted(p for pattern in args.checkpoints for p in glob.glob(pattern))
    results: dict[str, dict] = {}
    for path in paths:
        name = Path(path).stem
        if name == "resume":
            print(f"[skip] {path}: resume checkpoint, not a finished run")
            continue
        model, cfg = build_model_from_checkpoint(path, map_location=device)
        model.to(device).eval()
        row: dict = {"arm": cfg.arm, "n_layer": cfg.n_layer}
        if lam is not None:
            row["lambada_acc"] = round(lambada_accuracy(model, lam, device), 4)
        if blimp:
            per_task = {k: round(blimp_accuracy(model, v, device), 4) for k, v in blimp.items()}
            row["blimp_per_task"] = per_task
            row["blimp_macro"] = round(sum(per_task.values()) / len(per_task), 4)
        if val is not None:
            row["sliced_ppl"] = [round(p, 3) for p in
                                 sliced_perplexity(model, val, cfg.block_size, args.n_slices, device=device)]
        results[name] = row
        print(f"{name}: " + json.dumps({k: v for k, v in row.items() if k != 'blimp_per_task'}))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "capability.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    lines = ["| run | arm | L | lambada | blimp (macro) | sliced PPL min–max |",
             "|-----|-----|---|---------|---------------|--------------------|"]
    for name, r in sorted(results.items()):
        sl = r.get("sliced_ppl")
        sl_txt = f"{min(sl)}–{max(sl)}" if sl else "—"
        lines.append(f"| {name} | {r['arm']} | {r['n_layer']} | {r.get('lambada_acc', '—')} | "
                     f"{r.get('blimp_macro', '—')} | {sl_txt} |")
    (out / "capability.md").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {out / 'capability.json'} and {out / 'capability.md'}")


if __name__ == "__main__":
    main()
