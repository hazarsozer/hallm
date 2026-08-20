"""Drain-the-queue runner (spec 06 §8.3): one ssh command starts "whatever is next", and
interruption is always safe. Run state is inferred from the filesystem — final checkpoint exists ⇒
done; resume.pt exists ⇒ continue; else fresh. The manifest is frozen at FIRST launch and never
rewritten, so it describes what the whole (possibly multi-session) run trained."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch

from hallm.data import load_bin
from hallm.eval import evaluate_arm
from hallm.experiment import load_experiment
from hallm.manifest import build_manifest, write_manifest
from hallm.metrics import memory_row
from hallm.results import write_run_result
from hallm.model import GPT
from hallm.train import load_resume_checkpoint, save_checkpoint, set_seed, train

# Distinct from None: `run_one` returns this when a run paused early (--stop-step) rather than
# finished or was already done. `drain` must BREAK on this signal (not advance to the next queue
# entry) — otherwise a bounded GPU session would start every remaining run for one step each
# instead of stopping after the one run active when the session ends.
PAUSED = object()


def run_one(cfg_path: str | Path, data_dir: str | Path, device: str, stop_step: int | None = None):
    """Train one queue entry. Returns the eval row, None if already done, or PAUSED if training
    stopped early at `stop_step` without finishing (an absolute step index, per train())."""
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
    if resume.exists():
        # Config validation (spec 06 §8.1): a config edited between sessions must not silently
        # continue training under different hyperparameters than the frozen manifest attests.
        ckpt = load_resume_checkpoint(resume, map_location="cpu")
        mismatches = [
            k for k, (a, b) in {
                **{f"model_cfg.{k}": (v, asdict(model_cfg).get(k)) for k, v in ckpt["model_cfg"].items()},
                **{f"train_cfg.{k}": (v, asdict(train_cfg).get(k)) for k, v in ckpt["train_cfg"].items()},
            }.items()
            if a != b
        ]
        if mismatches:
            raise RuntimeError(
                f"{name}: resume.pt was trained under a different config than {cfg_path} now "
                f"describes — differing keys: {mismatches}"
            )
        if stop_step is not None and int(ckpt["step"]) >= stop_step:
            print(f"[stop] {name}: resume checkpoint already at step {ckpt['step']} >= stop_step {stop_step}")
            return PAUSED

    val_data = load_bin(data_dir / "val.bin")
    metrics_path = out / "metrics.jsonl"   # appended to across sessions; never truncated on resume
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    set_seed(train_cfg.seed, train_cfg.deterministic)  # seeds init; resume overwrites weights if present
    model = GPT(model_cfg)
    print(f"[run ] {name}: {'resuming' if resume.exists() else 'fresh'} on {device}")
    history = train(model, train_cfg, load_bin(data_dir / "train.bin"), device=device, progress=True,
                    resume_path=str(resume), stop_step=stop_step,
                    val_data=val_data, metrics_path=str(metrics_path))
    if stop_step is not None and stop_step < train_cfg.max_steps:
        print(f"[stop] {name}: paused at step {stop_step} (resume.pt saved)")
        return PAUSED

    save_checkpoint(model, model_cfg, train_cfg, final)
    row = evaluate_arm(model, model_cfg, val_data, batch_size=8, device=device)
    row["run"] = name
    # Measured memory + the train/val endpoints, so the generalisation gap is recoverable later
    # without re-reading a log that may not survive the session (spec P0 items 1, 3, 4).
    row.update(memory_row(model, model_cfg))
    if history:
        row["final_train_loss"] = round(history[-1]["loss"], 4)
        vals = [h["val_loss"] for h in history if "val_loss" in h]
        if vals:
            row["final_val_loss"] = round(vals[-1], 4)
    if torch.cuda.is_available():
        row["peak_vram_bytes"] = int(torch.cuda.max_memory_allocated())
    return row


def drain(queue_file: str | Path, data_dir: str | Path, results_dir: str | Path, device: str,
          max_runs: int | None = None, stop_step: int | None = None,
          failures: list[str] | None = None) -> list[dict]:
    """Process queue entries in order; write each finished run's eval row to its OWN file under
    `results_dir` (see hallm.results — one file per run, never a shared append-only ledger).

    Stops (without starting the next entry) as soon as a run pauses at `stop_step` — a bounded
    session must bound the ONE run active when it ends, not every queued run. If `failures` is
    given, the queue entry (and error) for each failed run is appended to it."""
    entries = [line.strip() for line in Path(queue_file).read_text().splitlines()]
    entries = [line for line in entries if line and not line.startswith("#")]
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for entry in entries:
        try:
            row = run_one(entry, data_dir, device, stop_step=stop_step)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"[fail] {entry}: {type(e).__name__}: {e}")
            if failures is not None:
                failures.append(f"{entry}: {type(e).__name__}: {e}")
            continue
        if row is PAUSED:
            break
        if row is None:
            continue
        write_run_result(results_dir, row)
        rows.append(row)
        if max_runs is not None and len(rows) >= max_runs:
            break
    return rows
