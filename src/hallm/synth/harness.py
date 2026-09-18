"""Track 2 training/eval loop: masked-loss training on generated problems, exact-match accuracy
on held-out ones (spec 2026-09-14 §5). Deliberately separate from `hallm.train.train()` -- that
loop is tightly coupled to sampling fixed-corpus windows via `hallm.data.get_batch` and is the
shared, protocol-critical path every perplexity-based experiment depends on. Reusing its pure
helpers (`configure_optimizer`, `cosine_lr`, `set_seed`) costs nothing; reshaping its batch loop
to also serve on-the-fly generated data would risk that shared path for no benefit.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch

from hallm.model.gpt import GPT
from hallm.synth.data import make_batch, make_problems
from hallm.synth.tasks import SynthTask
from hallm.train import configure_optimizer, cosine_lr, set_seed


@dataclass
class SynthTrainConfig:
    lr: float = 6e-4
    min_lr: float = 6e-5
    warmup_steps: int = 100
    max_steps: int = 2000
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    batch_size: int = 64
    seed: int = 1337
    deterministic: bool = True
    log_interval: int = 50
    eval_interval: int = 200
    eval_problems: int = 500  # held-out problems per accuracy check during training


def train_synth(
    model: GPT,
    cfg: SynthTrainConfig,
    task: SynthTask,
    difficulty: int,
    device: str | torch.device | None = None,
    progress: bool = False,
) -> list[dict]:
    """Train on freshly generated problems (one fresh batch per step -- no epoch, no fixed
    dataset, matching Saunshi et al.'s setup). Held-out accuracy checks during training use a
    seed offset from the training seed, so they never overlap with training problems (same
    'separate seed' discipline as the corpus held-out split elsewhere in this repo)."""
    set_seed(cfg.seed, cfg.deterministic)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    opt = configure_optimizer(model, cfg.weight_decay, cfg.lr, (cfg.beta1, cfg.beta2))

    train_rng = random.Random(cfg.seed)
    eval_rng_base = cfg.seed + 10_000  # separate seed space, same convention as hallm.train

    history: list[dict] = []
    for step in range(cfg.max_steps):
        lr = cosine_lr(step, cfg.warmup_steps, cfg.max_steps, cfg.lr, cfg.min_lr)
        for g in opt.param_groups:
            g["lr"] = lr

        problems = make_problems(task, difficulty, cfg.batch_size, train_rng)
        x, y = make_batch(problems, device)
        opt.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        if step % cfg.log_interval == 0 or step == cfg.max_steps - 1:
            rec = {"step": step, "loss": loss.item(), "lr": lr}
            if cfg.eval_interval > 0 and (step % cfg.eval_interval == 0 or step == cfg.max_steps - 1):
                rec["accuracy"] = evaluate_exact_match(
                    model, task, difficulty, cfg.eval_problems, eval_rng_base + step, device
                )
            history.append(rec)
            if progress:
                extra = f" | acc {rec['accuracy']:.3f}" if "accuracy" in rec else ""
                print(f"step {step:6d} | loss {loss.item():.4f} | lr {lr:.2e}{extra}")
    return history


@torch.no_grad()
def evaluate_exact_match(
    model: GPT,
    task: SynthTask,
    difficulty: int,
    n_problems: int,
    seed: int,
    device: str | torch.device | None = None,
    batch_size: int = 128,
) -> float:
    """Exact-match accuracy on `n_problems` freshly generated problems, greedily decoded token by
    token (teacher forcing isn't available at eval time -- the model has to actually produce every
    answer token itself, not just predict the next one given the true previous ones). Correct for
    both single-token answers (p-hop, binding-chain) and multi-token ones (addition): the loop
    just runs `answer_len` times, trivially once for the single-token tasks.

    `device` defaults to the model's own device (not "cpu") -- a model trained on CUDA and then
    evaluated with no device argument would otherwise crash on a device mismatch rather than just
    working, which is exactly the footgun a default should avoid."""
    if device is None:
        device = next(model.parameters()).device
    rng = random.Random(seed)
    was_training = model.training
    model.eval()

    problems = make_problems(task, difficulty, n_problems, rng)
    correct = 0
    for start in range(0, len(problems), batch_size):
        batch = problems[start : start + batch_size]
        prompt_len = batch[0].answer_start
        answer_len = len(batch[0].tokens) - prompt_len

        cur = torch.tensor([p.tokens[:prompt_len] for p in batch], dtype=torch.long, device=device)
        generated = []
        for _ in range(answer_len):
            logits, _ = model(cur)  # (B, 1, V) -- inference path only returns the last position
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated.append(next_tok)
            cur = torch.cat([cur, next_tok], dim=1)

        gen_answer = torch.cat(generated, dim=1)
        true_answer = torch.tensor([p.answer for p in batch], dtype=torch.long, device=device)
        correct += (gen_answer == true_answer).all(dim=1).sum().item()

    if was_training:
        model.train()
    return correct / len(problems)
