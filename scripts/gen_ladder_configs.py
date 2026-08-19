"""Generate the Tier-1 ladder configs (wiki/roadmap/06-scaling-campaign.md §4).

18 configs = 3 rungs (L4/L8/L16) × 2 arms (A0/A2) × 3 seeds; the 4 pairs already trained in
Experiment 1 / the iso 2×2 (seed 1337 at L8 and L16) are generated for the record but excluded
from queue.txt. Protocol constants are the Experiment-1 recipe verbatim — never edit them here.

Usage: uv run python scripts/gen_ladder_configs.py [--out configs/ladder]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

RUNGS = {"L4": "s30h", "L8": "s30", "L16": "s30x2"}  # order = drain order (cheapest evidence first)
ARMS = ["A0", "A2"]
SEEDS = [1337, 1338, 1339]
EXISTING = {"L8-A0-s1337", "L8-A2-s1337", "L16-A0-s1337", "L16-A2-s1337"}  # runs/{A0,A2}.pt, runs_iso/

TRAIN = dict(
    lr=6.0e-4, min_lr=6.0e-5, warmup_steps=200, max_steps=50_000, weight_decay=0.1,
    grad_clip=1.0, batch_size=12, grad_accum=2, dtype="bfloat16", deterministic=True,
    eval_interval=1000, checkpoint_interval=1000,
)


def generate(out_dir: str | Path) -> list[str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    queue: list[str] = []
    for rung, shape in RUNGS.items():
        for seed in SEEDS:
            for arm in ARMS:
                name = f"{rung}-{arm}-s{seed}"
                spec = {
                    "shape": shape,
                    "arm": arm,
                    "train": {**TRAIN, "seed": seed, "out_dir": f"runs/ladder/{name}"},
                }
                (out / f"{name}.yaml").write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
                if name not in EXISTING:
                    queue.append(str(out / f"{name}.yaml"))
    (out / "queue.txt").write_text("\n".join(queue) + "\n", encoding="utf-8")
    return queue


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="configs/ladder")
    args = ap.parse_args()
    queue = generate(args.out)
    print(f"wrote 18 configs to {args.out}/, queue.txt has {len(queue)} pending runs")


if __name__ == "__main__":
    main()
