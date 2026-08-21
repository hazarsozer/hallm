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



# --- P1 mechanism decomposition (program spec P1) -------------------------------------------
# Which sublayer's sharing causes the ~14% tax? The FFN path has a genuine nonlinearity between
# W and W-transpose (strong column-space argument); the attention path has K and V both linear in
# the same x (weak path, risk R1). These ARMS entries are implemented and unit-tested but have
# never been run. Run-ID arm tags carry no hyphen, per the artifact-layout run-ID grammar.
P1_ARMS = {"A2ffn": "A2-ffn", "A2attn": "A2-attn"}
# Seed 1339 included so P1 matches the seed coverage of the main ladder. NOTE: its tax
# needs L8-A0-s1339 as the baseline, which is assigned to Alper on issue #1 — the s1339
# ablations are therefore queued LAST, so seeds 1337/1338 (whose baselines we already
# hold) produce a complete, self-contained decomposition first.
P1_SEEDS = [1337, 1338, 1339]


def generate_ablations(out_dir: str | Path, rung: str = "L8", seeds: list[int] | None = None,
                       queue_name: str | None = None) -> list[str]:
    """Generate the mechanism-decomposition configs at one rung; queue entries in drain order.

    Protocol constants come from TRAIN verbatim, because an ablation is only meaningful against
    the already-completed A0 baseline at the same rung and seed — any recipe drift invalidates it.

    Seed-major ordering: both arms of a seed land together, so a complete decomposition exists as
    early as possible rather than only after every run finishes.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    shape = RUNGS[rung]
    seeds = seeds if seeds is not None else list(P1_SEEDS)
    queue: list[str] = []
    for seed in seeds:
        for tag, arm in P1_ARMS.items():
            name = f"{rung}-{tag}-s{seed}"
            spec = {
                "shape": shape,
                "arm": arm,
                "train": {**TRAIN, "seed": seed, "out_dir": f"runs/ladder/{name}"},
            }
            (out / f"{name}.yaml").write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
            queue.append(str(out / f"{name}.yaml"))
    qn = queue_name or f"queue-ablations-{rung.lower()}.txt"
    (out / qn).write_text("\n".join(queue) + "\n", encoding="utf-8")
    return queue


def generate_p1(out_dir: str | Path) -> list[str]:
    """The original P1 cohort: mechanism decomposition at L8, three seeds."""
    return generate_ablations(out_dir, rung="L8", seeds=list(P1_SEEDS), queue_name="queue-p1.txt")

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="configs/runs")
    ap.add_argument("--p1", action="store_true",
                    help="generate the P1 mechanism-decomposition configs + queue-p1.txt instead")
    args = ap.parse_args()
    if args.p1:
        queue = generate_p1(args.out)
        print(f"wrote {len(queue)} P1 configs to {args.out}/, queue-p1.txt lists them in drain order")
        return
    queue = generate(args.out)
    print(f"wrote 18 configs to {args.out}/, queue.txt has {len(queue)} pending runs")


if __name__ == "__main__":
    main()
