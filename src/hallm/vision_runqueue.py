"""Vision queue runner (spec 2026-09-16 §7-§8).

Mirrors `runqueue.py`: one config per line, each run resumable, each result written on landing so
an interrupted session loses at most the current run's progress since its last checkpoint.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from hallm.data.imagenet import load_images, load_labels
from hallm.experiment import load_vision_experiment
from hallm.manifest import build_manifest, write_manifest
from hallm.model.vit import ViT
from hallm.results import result_path, write_run_result
from hallm.train import load_resume_checkpoint, set_seed
from hallm.vision_train import evaluate_top1, train_vision

OK, PAUSED = "ok", "paused"


def _split_for_periodic_eval(val_images, val_labels, per_class: int, seed: int):
    """(images, labels) for the periodic eval: the whole split when per_class <= 0, else a fixed
    stratified subsample materialised into RAM (10 per class over 1000 classes is ~491 MB)."""
    if per_class <= 0:
        return val_images, val_labels
    from hallm.data.imagenet import stratified_subsample

    idx = stratified_subsample(val_labels, per_class, seed)
    return np.ascontiguousarray(val_images[idx]), val_labels[idx]


def _best_from_metrics(metrics_path: Path) -> tuple[float | None, int | None]:
    """Highest periodic-eval top-1 and the step it landed on, or (None, None) if none recorded."""
    if not metrics_path.exists():
        return None, None
    best, best_step = None, None
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        # the metrics file's key is "val_top1", NOT "top1" — verified against
        # runs/vision/V4-A0-s1337-p5k/metrics.jsonl before this plan was written
        t = row.get("val_top1")
        if t is not None and (best is None or t > best):
            best, best_step = float(t), int(row.get("step", -1))
    return best, best_step


def _result_row(model: ViT, model_cfg, train_cfg, name: str, top1: float, val_loss: float,
                top1_best: float | None = None, best_step: int | None = None,
                n_train: int | None = None) -> dict:
    nonemb = model.num_parameters(non_embedding=True)
    return {
        "run": name,
        "arm": model_cfg.arm,
        "dataset": train_cfg.dataset,
        "corpus": train_cfg.dataset,
        "n_classes": model_cfg.n_classes,
        "top1": round(top1, 6),
        "top1_best": None if top1_best is None else round(top1_best, 6),
        "best_step": best_step,
        # passes = how many times the run sweeps its corpus; the axis spec 2026-09-18 §1 found
        # unmatched between the ladder (10.31) and the first vision pass (101.03).
        "passes": None if not n_train else round(train_cfg.max_steps * train_cfg.batch_size / n_train, 3),
        "val_loss": round(val_loss, 4),
        "n_layer": model_cfg.n_layer,
        "non_embedding_params_M": round(nonemb / 1e6, 4),
        "total_params_M": round(model.num_parameters() / 1e6, 4),
        "nonemb_weight_bytes_bf16": nonemb * 2,
        **model.loop_scales(),
    }


def run_one_vision(cfg_path, data_dir, results_dir, device, stop_step: int | None = None) -> str:
    cfg_path, data_dir, results_dir = Path(cfg_path), Path(data_dir), Path(results_dir)
    model_cfg, train_cfg = load_vision_experiment(cfg_path)
    name = cfg_path.stem
    out = Path(train_cfg.out_dir)
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
        # Config validation (mirrors runqueue.py's run_one): a config edited between sessions must
        # not silently continue training under different hyperparameters than the frozen manifest
        # attests — these are 50k-step runs explicitly designed to resume across sessions, so a
        # touched YAML must be caught before it trains under a config the manifest no longer
        # describes.
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

    images = load_images(data_dir / "train.bin")
    labels = load_labels(data_dir / "train_labels.npy")
    val_images = load_images(data_dir / "val.bin")
    val_labels = load_labels(data_dir / "val_labels.npy")

    periodic_images, periodic_labels = _split_for_periodic_eval(
        val_images, val_labels, train_cfg.eval_subsample, train_cfg.seed)
    set_seed(train_cfg.seed, train_cfg.deterministic)
    model = ViT(model_cfg)
    print(f"[run ] {name}: {'resuming' if resume.exists() else 'fresh'} on {device}"
          f" | periodic eval on {len(periodic_images)} of {len(val_images)} val images")
    train_vision(model, train_cfg, images, labels, device=device, progress=True,
                 resume_path=str(resume), stop_step=stop_step, val_images=periodic_images,
                 val_labels=periodic_labels, metrics_path=str(out / "metrics.jsonl"))
    if stop_step is not None and stop_step < train_cfg.max_steps:
        print(f"[stop] {name}: paused at step {stop_step} (resume.pt saved)")
        return PAUSED

    top1, val_loss = evaluate_top1(model, val_images, val_labels, train_cfg, device)
    # dict-of-primitives, as `hallm.train.save_checkpoint` does: stays weights_only=True-loadable,
    # so nothing downstream has to unpickle arbitrary objects to read a checkpoint.
    torch.save(
        {"model": model.state_dict(), "model_cfg": asdict(model_cfg), "train_cfg": asdict(train_cfg)},
        out / f"{name}.pt",
    )
    top1_best, best_step = _best_from_metrics(out / "metrics.jsonl")
    write_run_result(results_dir, _result_row(model, model_cfg, train_cfg, name, top1, val_loss,
                                              top1_best=top1_best, best_step=best_step,
                                              n_train=len(images)))
    print(f"[done] {name}: top1 {top1:.4f} | val_loss {val_loss:.4f}")
    return OK


def drain_vision(queue_path, data_dir, results_dir, device, max_runs: int | None = None,
                 stop_step: int | None = None) -> list[dict]:
    rows: list[dict] = []
    lines = [ln.strip() for ln in Path(queue_path).read_text(encoding="utf-8").splitlines()
             if ln.strip() and not ln.startswith("#")]
    for line in lines:
        name = Path(line).stem
        if result_path(results_dir, name).exists():
            print(f"[skip] {name}: result already written")
            continue
        try:
            status = run_one_vision(line, data_dir, results_dir, device, stop_step)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            # Per-entry isolation (mirrors runqueue.py's drain): the five arms are independent runs
            # writing independent result files, so a transient failure on one must not cost the
            # queue every run after it — this runs unattended overnight.
            print(f"[fail] {line}: {type(e).__name__}: {e}")
            continue
        if status == PAUSED:
            break
        rows.append(json.loads(result_path(results_dir, name).read_text(encoding="utf-8")))
        if max_runs is not None and len(rows) >= max_runs:
            break
    print(f"session complete: {len(rows)} run(s) finished → {results_dir}/<run-id>.json")
    return rows
