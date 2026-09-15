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


# --- Fixed-storage pilot (spec 2026-09-14 §4.1) --------------------------------------------
# Every run here is compared against an EXISTING run at the same stored size (and, for the looped
# arms, the same compute), so the recipe is TRAIN verbatim. Seed-major: each comparison gets a
# complete seed-1337 point before any second seed starts.
PILOT = [  # (rung, shape, run-ID arm tag, seeds)
    ("L8", "s30", "A1u4", [1337, 1338, 1339]),     # vs L8-A2 (matched) and L4-A0 (iso-storage)
    ("L16", "s30x2", "A1u8", [1337, 1338, 1339]),  # vs L16-A2 (matched) and L8-A0 (iso-storage)
    ("L9", "s30l9", "A2attn", [1337]),             # vs L8-A0 (iso-storage, −6%)
    # vs L8-A0 (iso-storage, +4%). Seeds 1338/1339 added 2026-09-15: seed 1337 landed within 1% of
    # L8-A0 (+0.05%), the §4.1 trigger for extra A2attn seeds.
    ("L10", "s30l10", "A2attn", [1337, 1338, 1339]),
]
ARM_FOR_TAG = {"A2attn": "A2-attn"}  # run-ID tags carry no hyphen; ARMS keys do


def generate_pilot(out_dir: str | Path) -> list[str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    queue: list[str] = []
    for seed in (1337, 1338, 1339):
        for rung, shape, tag, seeds in PILOT:
            if seed not in seeds:
                continue
            name = f"{rung}-{tag}-s{seed}"
            spec = {"shape": shape, "arm": ARM_FOR_TAG.get(tag, tag),
                    "train": {**TRAIN, "seed": seed, "out_dir": f"runs/ladder/{name}"}}
            (out / f"{name}.yaml").write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
            queue.append(str(out / f"{name}.yaml"))
    (out / "queue-pilot.txt").write_text("\n".join(queue) + "\n", encoding="utf-8")
    return queue


# --- FineWeb-Edu bridge (spec 2026-09-14 §4.3) ---------------------------------------------
# Experiment 2's iso-storage pair re-trained on FineWeb-Edu with the WikiText recipe verbatim, so
# the WikiText numbers connect to the 124M rung's corpus. The `-fw` segment keeps these IDs out of
# the WikiText ladder tables (README run-ID grammar). Drain with `--data data/fineweb`.
BRIDGE = [("L8", "s30", "A0"), ("L16", "s30x2", "A2")]


def generate_bridge(out_dir: str | Path) -> list[str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    queue: list[str] = []
    for rung, shape, arm in BRIDGE:
        name = f"{rung}-{arm}-s1337-fw"
        spec = {"shape": shape, "arm": arm,
                "train": {**TRAIN, "dataset": "fineweb-edu", "seed": 1337, "out_dir": f"runs/ladder/{name}"}}
        (out / f"{name}.yaml").write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
        queue.append(str(out / f"{name}.yaml"))
    (out / "queue-bridge.txt").write_text("\n".join(queue) + "\n", encoding="utf-8")
    return queue


# --- Transposed loop (spec 2026-09-15 §2, exploratory) ----------------------------------------
# Three pass-2 variants of the plain loop L8-A1u4 at identical storage and FLOPs, the pilot recipe
# verbatim. Seed 1337 only: follow-up seeds are added by a later commit if spec §3's rule fires.
TRANSPOSED = [("L8", "s30", "A1u4t"), ("L8", "s30", "A1u4n"), ("L8", "s30", "A1u4a")]


def generate_transposed(out_dir: str | Path) -> list[str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    queue: list[str] = []
    for rung, shape, arm in TRANSPOSED:
        name = f"{rung}-{arm}-s1337"
        spec = {"shape": shape, "arm": arm,
                "train": {**TRAIN, "seed": 1337, "out_dir": f"runs/ladder/{name}"}}
        (out / f"{name}.yaml").write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
        queue.append(str(out / f"{name}.yaml"))
    (out / "queue-transposed.txt").write_text("\n".join(queue) + "\n", encoding="utf-8")
    return queue


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="configs/runs")
    ap.add_argument("--p1", action="store_true",
                    help="generate the P1 mechanism-decomposition configs + queue-p1.txt instead")
    ap.add_argument("--pilot", action="store_true", help="generate the fixed-storage pilot configs + queue-pilot.txt")
    ap.add_argument("--bridge", action="store_true", help="generate the FineWeb-Edu bridge configs + queue-bridge.txt")
    ap.add_argument("--transposed", action="store_true",
                    help="generate the transposed-loop configs + queue-transposed.txt")
    args = ap.parse_args()
    if args.transposed:
        queue = generate_transposed(args.out)
        print(f"wrote {len(queue)} transposed-loop configs to {args.out}/, queue-transposed.txt lists them")
        return
    if args.bridge:
        queue = generate_bridge(args.out)
        print(f"wrote {len(queue)} bridge configs to {args.out}/, queue-bridge.txt lists them in drain order")
        return
    if args.pilot:
        queue = generate_pilot(args.out)
        print(f"wrote {len(queue)} pilot configs to {args.out}/, queue-pilot.txt lists them in drain order")
        return
    if args.p1:
        queue = generate_p1(args.out)
        print(f"wrote {len(queue)} P1 configs to {args.out}/, queue-p1.txt lists them in drain order")
        return
    queue = generate(args.out)
    print(f"wrote 18 configs to {args.out}/, queue.txt has {len(queue)} pending runs")


if __name__ == "__main__":
    main()
