"""Vision training loop (spec 2026-09-16 §4).

A sibling of `train.py`, not a replacement: the LM loop is mid-campaign and token-batch specific.
Everything that defines the recipe — optimizer, schedule, seeding, resume format, metrics file — is
imported from `train.py` so the two domains cannot silently drift apart.

One deliberate divergence from `train.py`'s checkpoint condition: this loop also saves a resume
checkpoint on natural completion (`done == train_cfg.max_steps`), which `train.py` does not. That
guarantees a completed run always leaves a checkpoint at its final step even when
`checkpoint_interval` doesn't divide `max_steps` evenly; harmless otherwise. Noted here so the
difference is visible rather than accidental.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import torch

from hallm.data.imagenet import get_image_batch
from hallm.model.vit import ViT
from hallm.train import (TrainConfig, configure_optimizer, cosine_lr, load_resume_checkpoint,
                         save_resume_checkpoint)


@dataclass
class VisionTrainConfig(TrainConfig):
    dataset: str = "imagenet-100"
    crop_size: int = 112
    label_smoothing: float = 0.1
    eval_batch: int = 250          # evaluation batch; the whole val split is scored
    batch_size: int = 256
    # TrainConfig.block_size's comment reads "must equal ModelConfig.block_size" — for a ViT that
    # is 49 patches + class token = 50, not the LM default of 512. Nothing in this module reads
    # block_size (get_image_batch is driven by crop_size), but it is copied verbatim into every run
    # manifest, so the inherited 512 would be a lie in the permanent record.
    block_size: int = 50


@torch.no_grad()
def evaluate_top1(model: ViT, images: np.ndarray, labels: np.ndarray, cfg: VisionTrainConfig,
                  device) -> tuple[float, float]:
    """Top-1 fraction and mean loss over the whole split, center-cropped, in eval mode.

    The loss is plain cross-entropy WITHOUT label smoothing: model.label_smoothing is saved, forced
    to 0.0 for the duration of the evaluation, and restored in a `finally` (so an exception can't
    leave the model mis-configured for subsequent training). A smoothed CE's floor depends on the
    smoothing constant, so it isn't comparable across configs; unsmoothed CE is. Top-1, the primary
    metric, is unaffected either way.
    """
    was_training = model.training
    saved_smoothing = model.label_smoothing
    model.eval()
    model.label_smoothing = 0.0
    correct, total, loss_sum = 0, 0, 0.0
    try:
        for start in range(0, len(images), cfg.eval_batch):
            stop = min(start + cfg.eval_batch, len(images))
            x, y = get_image_batch(images[start:stop], labels[start:stop], stop - start,
                                   cfg.crop_size, device, None, train=False)
            logits, loss = model(x, y)
            correct += int((logits.argmax(-1) == y).sum())
            loss_sum += float(loss) * (stop - start)
            total += stop - start
    finally:
        model.label_smoothing = saved_smoothing
        model.train(was_training)
    return correct / max(1, total), loss_sum / max(1, total)


def train_vision(
    model: ViT,
    train_cfg: VisionTrainConfig,
    images: np.ndarray,
    labels: np.ndarray,
    device: str | torch.device | None = None,
    progress: bool = False,
    resume_path: str | None = None,
    stop_step: int | None = None,
    val_images: np.ndarray | None = None,
    val_labels: np.ndarray | None = None,
    metrics_path: str | None = None,
) -> list[dict]:
    """Run the matched-budget vision loop. Mirrors `hallm.train.train`, including mid-run resume."""
    if stop_step is not None and not resume_path:
        raise ValueError("stop_step requires resume_path")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.label_smoothing = train_cfg.label_smoothing
    model.train()
    opt = configure_optimizer(model, train_cfg.weight_decay, train_cfg.lr,
                              (train_cfg.beta1, train_cfg.beta2))
    gen = torch.Generator().manual_seed(train_cfg.seed)

    start_step = 0
    if resume_path and os.path.exists(resume_path):
        ckpt = load_resume_checkpoint(resume_path)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        gen.set_state(ckpt["gen_state"])
        torch.set_rng_state(ckpt["torch_rng"])
        if ckpt.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(ckpt["cuda_rng"])
        start_step = int(ckpt["step"])

    use_amp = str(device) != "cpu" and train_cfg.dtype in ("bfloat16", "float16")
    amp_dtype = torch.bfloat16 if train_cfg.dtype == "bfloat16" else torch.float16

    history: list[dict] = []
    for step in range(start_step, train_cfg.max_steps):
        lr = cosine_lr(step, train_cfg.warmup_steps, train_cfg.max_steps, train_cfg.lr,
                       train_cfg.min_lr)
        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for _ in range(train_cfg.grad_accum):
            x, y = get_image_batch(images, labels, train_cfg.batch_size, train_cfg.crop_size,
                                   device, gen, train=True)
            if use_amp:
                with torch.autocast(device_type=str(device).split(":")[0], dtype=amp_dtype):
                    _, loss = model(x, y)
            else:
                _, loss = model(x, y)
            loss = loss / train_cfg.grad_accum
            loss.backward()
            loss_accum += loss.item()

        if train_cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        opt.step()

        if step % train_cfg.log_interval == 0 or step == train_cfg.max_steps - 1:
            rec = {"step": step, "loss": loss_accum, "lr": lr}
            rec.update(model.loop_scales())
            if (
                val_images is not None
                and train_cfg.eval_interval > 0
                and (step % train_cfg.eval_interval == 0 or step == train_cfg.max_steps - 1)
            ):
                top1, vloss = evaluate_top1(model, val_images, val_labels, train_cfg, device)
                rec["val_top1"], rec["val_loss"] = round(top1, 6), vloss
            history.append(rec)
            if metrics_path:
                with open(metrics_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
            if progress:
                extra = f" | top1 {rec['val_top1']:.4f}" if "val_top1" in rec else ""
                print(f"step {step:6d} | loss {loss_accum:.4f} | lr {lr:.2e}{extra}")

        done = step + 1
        at_interval = train_cfg.checkpoint_interval > 0 and done % train_cfg.checkpoint_interval == 0
        stopping = stop_step is not None and done >= stop_step
        if resume_path and (at_interval or stopping or done == train_cfg.max_steps):
            # signature: (path, model, train_cfg, opt, gen, step) — it reads model.cfg itself
            save_resume_checkpoint(resume_path, model, train_cfg, opt, gen, done)
        if stopping:
            break
    return history
