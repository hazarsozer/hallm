"""T-006 sizing-pilot calibration (spec 2026-09-14 §5): find a (d, k, steps, p-range) where the
unshared deep reference (A0 @ 4k layers) clears >=90% on p-hop at a difficulty where the shallow
reference (A0 @ k layers) is <=50% -- the brief's "depth matters on this task" precondition.

Not a committed experiment script -- exploratory, meant to be edited/rerun while calibrating.
Prints a table; does not write results/ or checkpoints.
"""

from __future__ import annotations

import argparse
import dataclasses
import time

import torch

from hallm.model import SHAPES, arm_config
from hallm.model.gpt import GPT
from hallm.synth.harness import SynthTrainConfig, evaluate_exact_match, train_synth
from hallm.synth.tasks import PHopInductionTask


def run_one(d: int, layers: int, vocab_size: int, p: int, steps: int, seed: int) -> tuple[float, float]:
    task = PHopInductionTask(vocab_size=vocab_size, seq_len_margin=4)
    base = dataclasses.replace(
        SHAPES["smoke"], vocab_size=vocab_size, block_size=task.seq_len(p) + 4,
        n_embd=d, n_layer=layers, n_head=max(1, d // 32), ffn_mult=4,
    )
    model = GPT(arm_config(base, "A0"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = SynthTrainConfig(
        max_steps=steps, batch_size=64, lr=3e-3, warmup_steps=max(10, steps // 20),
        log_interval=steps, eval_interval=0, seed=seed,
    )
    t0 = time.time()
    train_synth(model, cfg, task, difficulty=p, device=device)
    elapsed = time.time() - t0
    acc = evaluate_exact_match(model, task, difficulty=p, n_problems=500, seed=seed + 999, device=device)
    return acc, elapsed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--vocab", type=int, default=32)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--p-values", type=int, nargs="+", default=[2, 4, 6, 8, 10, 12, 16])
    args = ap.parse_args()

    print(f"d={args.d} k={args.k} (deep=4k={4*args.k}) vocab={args.vocab} steps={args.steps}\n")
    print(f"{'p':>4} | {'shallow acc':>11} | {'deep acc':>9} | {'shallow s':>9} | {'deep s':>7}")
    for p in args.p_values:
        acc_shallow, t_shallow = run_one(args.d, args.k, args.vocab, p, args.steps, args.seed)
        acc_deep, t_deep = run_one(args.d, 4 * args.k, args.vocab, p, args.steps, args.seed)
        flag = "  <-- target band" if acc_shallow <= 0.5 and acc_deep >= 0.9 else ""
        print(f"{p:>4} | {acc_shallow:>11.3f} | {acc_deep:>9.3f} | {t_shallow:>8.1f}s | {t_deep:>6.1f}s{flag}")


if __name__ == "__main__":
    main()
