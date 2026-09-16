"""Vision queue runner (spec 2026-09-16 §7-§8).

Mirrors `runqueue.py`: one config per line, each run resumable, each result written on landing so
an interrupted session loses at most the current run's progress since its last checkpoint.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch

from hallm.data.imagenet import load_images, load_labels
from hallm.experiment import load_vision_experiment
from hallm.manifest import build_manifest, write_manifest
from hallm.model.vit import ViT
from hallm.results import result_path, write_run_result
from hallm.train import set_seed
from hallm.vision_train import evaluate_top1, train_vision

OK, PAUSED = "ok", "paused"


def _result_row(model: ViT, model_cfg, train_cfg, name: str, top1: float, val_loss: float) -> dict:
    nonemb = model.num_parameters(non_embedding=True)
    return {
        "run": name,
        "arm": model_cfg.arm,
        "dataset": train_cfg.dataset,
        "n_classes": model_cfg.n_classes,
        "top1": round(top1, 6),
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

    images = load_images(data_dir / "train.bin")
    labels = load_labels(data_dir / "train_labels.npy")
    val_images = load_images(data_dir / "val.bin")
    val_labels = load_labels(data_dir / "val_labels.npy")

    set_seed(train_cfg.seed, train_cfg.deterministic)
    model = ViT(model_cfg)
    resume = out / "resume.pt"
    print(f"[run ] {name}: {'resuming' if resume.exists() else 'fresh'} on {device}")
    train_vision(model, train_cfg, images, labels, device=device, progress=True,
                 resume_path=str(resume), stop_step=stop_step, val_images=val_images,
                 val_labels=val_labels, metrics_path=str(out / "metrics.jsonl"))
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
    write_run_result(results_dir, _result_row(model, model_cfg, train_cfg, name, top1, val_loss))
    print(f"[done] {name}: top1 {top1:.4f} | val_loss {val_loss:.4f}")
    return OK


def drain_vision(queue_path, data_dir, results_dir, device, max_runs: int | None = None,
                 stop_step: int | None = None) -> list[dict]:
    import json

    rows: list[dict] = []
    lines = [ln.strip() for ln in Path(queue_path).read_text(encoding="utf-8").splitlines()
             if ln.strip() and not ln.startswith("#")]
    for line in lines:
        name = Path(line).stem
        if result_path(results_dir, name).exists():
            print(f"[skip] {name}: result already written")
            continue
        if run_one_vision(line, data_dir, results_dir, device, stop_step) == PAUSED:
            break
        rows.append(json.loads(result_path(results_dir, name).read_text(encoding="utf-8")))
        if max_runs is not None and len(rows) >= max_runs:
            break
    print(f"session complete: {len(rows)} run(s) finished → {results_dir}/<run-id>.json")
    return rows
