"""Matched-budget training loop (FR-3, NFR-1/3/4).

The four arms share ONE `TrainConfig` and differ only in the `ModelConfig.share_*` flags, so any
perplexity gap is attributable to the sharing scheme. Reproducibility is enforced by a single seed
that drives both initialization and the data-sampling generator, so the data order is identical
across arms (roadmap/04-training-protocol.md).

NO real training is launched by the overnight loop — this module is exercised only by the CPU smoke
test (a few steps) and by the Term-2 `scripts/run_real_training.py` (GPU, manual).
"""

from __future__ import annotations

import math
import os
import random
from dataclasses import asdict, dataclass

import numpy as np
import torch

from hallm.model.config import ModelConfig
from hallm.model.gpt import GPT


@dataclass
class TrainConfig:
    # data
    dataset: str = "wikitext-103"
    tokenizer: str = "gpt2"
    # optimization
    lr: float = 6e-4
    min_lr: float = 6e-5
    warmup_steps: int = 200
    max_steps: int = 50_000          # fixes tokens-seen → identical across arms
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    # batching
    batch_size: int = 24
    grad_accum: int = 1
    block_size: int = 512            # must equal ModelConfig.block_size
    # precision / memory (NFR-3)
    dtype: str = "bfloat16"          # bfloat16 | float16 | float32
    grad_checkpoint: bool = False    # if used, ON FOR ALL ARMS (comparability)
    # reproducibility (NFR-1)
    seed: int = 1337
    deterministic: bool = True
    # logging / eval (NFR-7)
    eval_interval: int = 1000
    eval_iters: int = 100
    log_interval: int = 50
    out_dir: str = "runs"


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)


def cosine_lr(step: int, warmup: int, max_steps: int, lr: float, min_lr: float) -> float:
    """Linear warmup then cosine decay to min_lr."""
    if step < warmup:
        return lr * (step + 1) / max(1, warmup)
    if step >= max_steps:
        return min_lr
    ratio = (step - warmup) / max(1, max_steps - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return min_lr + coeff * (lr - min_lr)


def configure_optimizer(model: torch.nn.Module, weight_decay: float, lr: float, betas) -> torch.optim.Optimizer:
    """AdamW with weight decay on 2-D weights (matmuls/embeddings) only, not on biases / LayerNorm."""
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (decay if p.ndim >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=betas)


@torch.no_grad()
def estimate_loss(model: GPT, data: np.ndarray, cfg: TrainConfig, device, generator) -> float:
    """Average loss over `eval_iters` random batches (a cheap training-curve signal, not full PPL)."""
    from hallm.data import get_batch

    model.eval()
    losses = []
    for _ in range(cfg.eval_iters):
        x, y = get_batch(data, cfg.block_size, cfg.batch_size, device, generator)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def train(
    model: GPT,
    train_cfg: TrainConfig,
    train_data: np.ndarray,
    device: str | torch.device | None = None,
    progress: bool = False,
) -> list[dict]:
    """Run the matched-budget loop. Returns a history of {step, loss, lr} log dicts."""
    from hallm.data import get_batch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    opt = configure_optimizer(model, train_cfg.weight_decay, train_cfg.lr, (train_cfg.beta1, train_cfg.beta2))
    gen = torch.Generator().manual_seed(train_cfg.seed)  # data order — identical across arms

    use_amp = device != "cpu" and train_cfg.dtype in ("bfloat16", "float16")
    amp_dtype = torch.bfloat16 if train_cfg.dtype == "bfloat16" else torch.float16

    history: list[dict] = []
    for step in range(train_cfg.max_steps):
        lr = cosine_lr(step, train_cfg.warmup_steps, train_cfg.max_steps, train_cfg.lr, train_cfg.min_lr)
        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for _ in range(train_cfg.grad_accum):
            x, y = get_batch(train_data, train_cfg.block_size, train_cfg.batch_size, device, gen)
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
            history.append({"step": step, "loss": loss_accum, "lr": lr})
            if progress:
                print(f"step {step:6d} | loss {loss_accum:.4f} | lr {lr:.2e}")
    return history


def save_checkpoint(model: GPT, model_cfg: ModelConfig, train_cfg: TrainConfig, path: str | os.PathLike) -> None:
    """Save weights + configs. Configs are stored as plain dicts so the checkpoint can be loaded with
    the SAFE ``weights_only=True`` (no arbitrary-object unpickling / code-execution surface)."""
    torch.save(
        {"model": model.state_dict(), "model_cfg": asdict(model_cfg), "train_cfg": asdict(train_cfg)},
        path,
    )


def load_checkpoint(path: str | os.PathLike, map_location="cpu") -> dict:
    """Load a checkpoint written by `save_checkpoint`. Safe: only tensors + primitive dicts."""
    return torch.load(path, map_location=map_location, weights_only=True)


def build_model_from_checkpoint(path: str | os.PathLike, map_location="cpu") -> tuple[GPT, ModelConfig]:
    """Reconstruct the GPT and its config from a checkpoint."""
    ckpt = load_checkpoint(path, map_location)
    cfg = ModelConfig(**ckpt["model_cfg"])
    model = GPT(cfg)
    model.load_state_dict(ckpt["model"])
    return model, cfg
