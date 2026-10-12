"""Track 2 sizing-pilot sweep on the p-hop task (PHopTask): accuracy by depth, arm and hop count.

Exploratory: it prints a table and appends one JSON line per run to --log. It writes nothing under
results/ and is not a committed experiment — the locked recipe goes into configs/synth/.

    uv run python scripts/synth_pilot_calibrate.py --p-values 2 4 8 --arms A0:1 A0:2 A0:4 --steps 20000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from hallm.model import ModelConfig, arm_config
from hallm.model.gpt import GPT
from hallm.synth.harness import SynthTrainConfig, evaluate_exact_match, train_synth
from hallm.synth.tasks import PHopTask


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p-values", type=int, nargs="+", required=True)
    ap.add_argument("--arms", nargs="+", default=["A0:1", "A0:2", "A0:4"], help="ARM:LAYERS, e.g. A1u1:4")
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--vocab", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--evals", type=int, default=10, help="accuracy checks during training")
    ap.add_argument("--dense", action="store_true", help="train with hop_p targets at every position")
    ap.add_argument("--log", default="runs/synth_pilot/sweep.jsonl")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    task = PHopTask(vocab_size=args.vocab, seq_len=args.seq_len, dense=args.dense)
    log = Path(args.log)
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"p_hop vocab={args.vocab} seq_len={args.seq_len} d={args.d} steps={args.steps} "
          f"batch={args.batch} lr={args.lr} seed={args.seed} dense={args.dense}", flush=True)

    for p in args.p_values:
        for spec in args.arms:
            arm, layers = spec.split(":")
            base = ModelConfig(vocab_size=args.vocab, block_size=args.seq_len + 1, n_embd=args.d,
                               n_layer=int(layers), n_head=args.heads, ffn_mult=4)
            model = GPT(arm_config(base, arm))
            cfg = SynthTrainConfig(
                max_steps=args.steps, batch_size=args.batch, lr=args.lr, seed=args.seed,
                warmup_steps=min(500, args.steps // 20), log_interval=args.steps // args.evals,
                eval_interval=args.steps // args.evals, eval_problems=500,
            )
            t0 = time.time()
            history = train_synth(model, cfg, task, p, device=device)
            seconds = time.time() - t0
            acc = evaluate_exact_match(model, task, p, 2000, 999, device=device)
            curve = [round(r["accuracy"], 3) for r in history if "accuracy" in r]
            rec = {"p": p, "arm": arm, "layers": int(layers), "accuracy": acc, "curve": curve,
                   "seconds": round(seconds), **{k: getattr(args, k) for k in
                                                 ("d", "heads", "vocab", "seq_len", "steps", "batch", "lr", "seed", "dense")}}
            with log.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"p={p:2d} {arm:>5s}@L{layers}  acc {acc:.3f}  ({seconds:4.0f}s)  curve {curve}", flush=True)


if __name__ == "__main__":
    main()
