"""Run Track 2 synth experiments from config files and write results/synth/<run-id>.json.

Usage:
    uv run python scripts/run_synth.py --config configs/synth/pilot-p4-A0-L1.yaml
    uv run python scripts/run_synth.py --queue configs/synth/queue-grid-p4.txt --eval-problems-final 2000

A queue is a text file of config paths, one per line, run in order. A run whose result file
already exists is skipped, so relaunching the same command after an interruption continues with
the first unfinished run (a synth run has no mid-run checkpoint: an interrupted run restarts).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from hallm.model.gpt import GPT
from hallm.synth.experiment import load_synth_experiment
from hallm.synth.harness import evaluate_exact_match, train_synth
from hallm.train import save_checkpoint


def run_one(config: str, results_dir: Path, ckpt_dir: Path, eval_problems: int, eval_seed: int) -> bool:
    """Train + evaluate one config. Returns False if its result already existed (skipped)."""
    model_cfg, train_cfg, task, difficulty, run_id = load_synth_experiment(config)
    out_path = results_dir / f"{run_id}.json"
    if out_path.exists():
        print(f"[skip] {run_id}: {out_path} exists", flush=True)
        return False

    print(f"[run ] {run_id}", flush=True)
    model = GPT(model_cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    history = train_synth(model, train_cfg, task, difficulty, device=device, progress=True)
    train_seconds = time.time() - t0
    final_acc = evaluate_exact_match(model, task, difficulty, eval_problems, eval_seed, device=device)

    run_ckpt = ckpt_dir / run_id
    run_ckpt.mkdir(parents=True, exist_ok=True)
    save_checkpoint(model, model_cfg, train_cfg, run_ckpt / "model.pt")

    results_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "run_id": run_id,
        "task": task.name,
        "difficulty": difficulty,
        "arm": model_cfg.arm,
        "n_layer": model_cfg.n_layer,
        "n_unique_blocks": model_cfg.n_unique_blocks,
        "n_embd": model_cfg.n_embd,
        "vocab_size": model_cfg.vocab_size,
        "non_embedding_params": model.num_parameters(non_embedding=True),
        "seed": train_cfg.seed,
        "max_steps": train_cfg.max_steps,
        "accuracy": final_acc,
        "eval_problems_final": eval_problems,
        "eval_seed": eval_seed,
        "final_train_loss": history[-1]["loss"] if history else None,
        "accuracy_curve": [{"step": r["step"], "accuracy": r["accuracy"]} for r in history if "accuracy" in r],
        "train_seconds": round(train_seconds, 1),
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "config_path": Path(config).as_posix(),
    }
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"[done] {run_id}  accuracy={final_acc:.4f}  ({train_seconds:.0f}s)", flush=True)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--config")
    src.add_argument("--queue")
    ap.add_argument("--results-dir", default="results/synth")
    ap.add_argument("--ckpt-dir", default="runs/synth")
    ap.add_argument("--eval-problems-final", type=int, default=1500)
    ap.add_argument("--eval-seed", type=int, default=999)
    args = ap.parse_args()

    if args.config:
        configs = [args.config]
    else:
        lines = Path(args.queue).read_text(encoding="utf-8").splitlines()
        configs = [ln.strip() for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]

    ran = 0
    for config in configs:
        ran += run_one(config, Path(args.results_dir), Path(args.ckpt_dir),
                       args.eval_problems_final, args.eval_seed)
    print(f"session complete: {ran} run(s) finished, {len(configs) - ran} skipped", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
