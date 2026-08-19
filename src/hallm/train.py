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
    checkpoint_interval: int = 1000  # steps between resume-checkpoint writes (0 = never)


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
    resume_path: str | None = None,
    stop_step: int | None = None,
) -> list[dict]:
    """Run the matched-budget loop. Returns a history of {step, loss, lr} log dicts.

    If `resume_path` is given, training state is periodically saved there (every
    `checkpoint_interval` steps) and, when the file already exists, restored from it — so an
    interrupted run continues exactly where it stopped (spec 06 §8.1). `stop_step` ends the
    session after that step (checkpoint saved), for bounded GPU windows.

    `stop_step` requires `resume_path`: an early stop with nowhere to save state would silently
    discard the partial run."""
    from hallm.data import get_batch

    if stop_step is not None and not resume_path:
        raise ValueError("stop_step requires resume_path")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    opt = configure_optimizer(model, train_cfg.weight_decay, train_cfg.lr, (train_cfg.beta1, train_cfg.beta2))
    gen = torch.Generator().manual_seed(train_cfg.seed)  # data order — identical across arms

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

    use_amp = device != "cpu" and train_cfg.dtype in ("bfloat16", "float16")
    amp_dtype = torch.bfloat16 if train_cfg.dtype == "bfloat16" else torch.float16

    history: list[dict] = []
    for step in range(start_step, train_cfg.max_steps):
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

        done = step + 1
        at_interval = train_cfg.checkpoint_interval > 0 and done % train_cfg.checkpoint_interval == 0
        stopping = stop_step is not None and done >= stop_step
        if resume_path and (at_interval or stopping):
            save_resume_checkpoint(resume_path, model, train_cfg, opt, gen, done)
        if stopping:
            break
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


def save_resume_checkpoint(
    path: str | os.PathLike,
    model: GPT,
    train_cfg: TrainConfig,
    opt: torch.optim.Optimizer,
    gen: torch.Generator,
    step: int,
) -> None:
    """Full training state for exact resume (spec 06 §8.1): weights + AdamW moments + data-order
    generator + global RNG (+ CUDA RNG, when available) + step. Atomic (tmp + replace) so an
    interrupt never corrupts the file. Stays `weights_only=True`-loadable: tensors and primitive
    containers only.

    The CUDA RNG state is a list of ByteTensors (one per device) and is only saved/restored when
    CUDA is available, so it can never be regression-tested on the CPU-only test suite — it matters
    for bit-exact resume of any run with dropout > 0 (or other CUDA-side stochastic ops), where a
    fresh CUDA RNG stream after resume would silently diverge from the interrupted run."""
    tmp = str(path) + ".tmp"
    payload = {
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "gen_state": gen.get_state(),
        "torch_rng": torch.get_rng_state(),
        "step": step,
        "model_cfg": asdict(model.cfg),
        "train_cfg": asdict(train_cfg),
    }
    if torch.cuda.is_available():
        payload["cuda_rng"] = torch.cuda.get_rng_state_all()
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_resume_checkpoint(path: str | os.PathLike, map_location="cpu") -> dict:
    """Load a resume checkpoint written by `save_resume_checkpoint` (safe: weights_only)."""
    return torch.load(path, map_location=map_location, weights_only=True)
