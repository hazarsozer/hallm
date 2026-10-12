"""Generate the Track 2 p-hop grid configs (spec 2026-09-14 §5, amendment 2026-10-12).

Six arms x three difficulty levels x three seeds, on PHopTask (alphabet 4, length 64, dense
training targets). Four arms store the same 12d² of block weights (k = 1 block); two unshared arms
are the compute references:

    A0   @ L1   unshared, k layers                    12d²   (same-storage baseline)
    A2   @ L2   W+Wᵀ at 2k layers                     12d²
    A1u1 @ L2   looped, k blocks x 2                  12d²
    A1u1 @ L4   looped, k blocks x 4                  12d²
    A0   @ L2   unshared, 2k layers                   24d²   (compute reference)
    A0   @ L4   unshared, 4k layers                   48d²   (compute reference)

The queue is seed-major, so every seed that finishes is a complete comparison at all levels.

Usage:
    uv run python scripts/gen_synth_grid.py            # writes configs/synth/grid-*.yaml + queue
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

TASK = "p_hop"
LEVELS = (2, 4, 8)
SEEDS = (1337, 1338, 1339)
ARMS = (("A0", 1), ("A2", 2), ("A1u1", 2), ("A1u1", 4), ("A0", 2), ("A0", 4))
TASK_KWARGS = {"vocab_size": 4, "seq_len": 64, "dense": True}
MODEL = {"n_embd": 128, "n_head": 4, "ffn_mult": 4, "block_size": 65}
TRAIN = {"max_steps": 20000, "batch_size": 128, "lr": 3.0e-3, "warmup_steps": 500,
         "eval_interval": 2000, "eval_problems": 500, "log_interval": 2000}
QUEUE = "queue-grid-p_hop.txt"


def generate(out_dir: str | Path) -> list[str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    queue: list[str] = []
    for seed in SEEDS:
        for p in LEVELS:
            for arm, layers in ARMS:
                tag = f"p{p}-L{layers}-{arm}-s{seed}"
                spec = {
                    "task": TASK,
                    "task_kwargs": dict(TASK_KWARGS),
                    "difficulty": p,
                    "arm": arm,
                    "model": {**MODEL, "n_layer": layers},
                    "train": {**TRAIN, "seed": seed},
                    "run_id": f"synth-{TASK}-{tag}",
                }
                path = out / f"grid-{TASK}-{tag}.yaml"
                path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
                queue.append(path.as_posix())
    (out / QUEUE).write_text("\n".join(queue) + "\n", encoding="utf-8")
    return queue


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="configs/synth")
    args = ap.parse_args()
    queue = generate(args.out)
    print(f"wrote {len(queue)} configs to {args.out}/ and {QUEUE}")


if __name__ == "__main__":
    main()
