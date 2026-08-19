"""Drain-the-queue runner (spec 06 §8.3): one ssh command starts "whatever is next", and
interruption is always safe. Run state is inferred from the filesystem — final checkpoint exists ⇒
done; resume.pt exists ⇒ continue; else fresh. The manifest is frozen at FIRST launch and never
rewritten, so it describes what the whole (possibly multi-session) run trained."""

from __future__ import annotations

import json
from pathlib import Path

from hallm.data import load_bin
from hallm.eval import evaluate_arm
from hallm.experiment import load_experiment
from hallm.manifest import build_manifest, write_manifest
from hallm.model import GPT
from hallm.train import save_checkpoint, set_seed, train


def run_one(cfg_path: str | Path, data_dir: str | Path, device: str, stop_step: int | None = None) -> dict | None:
    """Train one queue entry. Returns the eval row, or None if already done / stopped early."""
    cfg_path, data_dir = Path(cfg_path), Path(data_dir)
    model_cfg, train_cfg = load_experiment(cfg_path)
    name = cfg_path.stem
    out = Path(train_cfg.out_dir)
    final = out / f"{name}.pt"
    if final.exists():
        print(f"[skip] {name}: final checkpoint exists")
        return None
    out.mkdir(parents=True, exist_ok=True)

    manifest_path = out / "manifest.json"
    if not manifest_path.exists():
        write_manifest(
            build_manifest(model_cfg, train_cfg, config_path=cfg_path,
                           data_files=[data_dir / "train.bin", data_dir / "val.bin"]),
            manifest_path,
        )

    resume = out / "resume.pt"
    set_seed(train_cfg.seed, train_cfg.deterministic)  # seeds init; resume overwrites weights if present
    model = GPT(model_cfg)
    print(f"[run ] {name}: {'resuming' if resume.exists() else 'fresh'} on {device}")
    train(model, train_cfg, load_bin(data_dir / "train.bin"), device=device, progress=True,
          resume_path=str(resume), stop_step=stop_step)
    if stop_step is not None and stop_step < train_cfg.max_steps:
        print(f"[stop] {name}: paused at step {stop_step} (resume.pt saved)")
        return None

    save_checkpoint(model, model_cfg, train_cfg, final)
    row = evaluate_arm(model, model_cfg, load_bin(data_dir / "val.bin"), batch_size=8, device=device)
    row["run"] = name
    return row


def drain(queue_file: str | Path, data_dir: str | Path, results_path: str | Path, device: str,
          max_runs: int | None = None, stop_step: int | None = None) -> list[dict]:
    """Process queue entries in order; append each finished run's eval row to `results_path`."""
    entries = [l.strip() for l in Path(queue_file).read_text().splitlines()
               if l.strip() and not l.startswith("#")]
    results_path = Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for entry in entries:
        try:
            row = run_one(entry, data_dir, device, stop_step=stop_step)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"[fail] {entry}: {type(e).__name__}: {e}")
            continue
        if row is None:
            continue
        with results_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        rows.append(row)
        if max_runs is not None and len(rows) >= max_runs:
            break
    return rows
