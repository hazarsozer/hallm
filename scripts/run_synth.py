"""Run one Track 2 synth experiment from a config file and write results/synth/<run-id>.json.

Usage:
    uv run python scripts/run_synth.py --config configs/synth/pilot-p4-A0-L1.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from hallm.model.gpt import GPT
from hallm.synth.experiment import load_synth_experiment
from hallm.synth.harness import evaluate_exact_match, train_synth


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--results-dir", default="results/synth")
    ap.add_argument("--eval-problems-final", type=int, default=1500)
    ap.add_argument("--eval-seed", type=int, default=999)
    args = ap.parse_args()

    model_cfg, train_cfg, task, difficulty, run_id = load_synth_experiment(args.config)
    model = GPT(model_cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    history = train_synth(model, train_cfg, task, difficulty, device=device, progress=True)
    final_acc = evaluate_exact_match(
        model, task, difficulty, args.eval_problems_final, args.eval_seed, device=device
    )

    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "run_id": run_id,
        "task": task.name,
        "difficulty": difficulty,
        "arm": model_cfg.arm,
        "n_layer": model_cfg.n_layer,
        "n_embd": model_cfg.n_embd,
        "vocab_size": model_cfg.vocab_size,
        "seed": train_cfg.seed,
        "max_steps": train_cfg.max_steps,
        "accuracy": final_acc,
        "final_train_loss": history[-1]["loss"] if history else None,
        "config_path": str(args.config),
    }
    out_path = out_dir / f"{run_id}.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_path}  accuracy={final_acc:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
