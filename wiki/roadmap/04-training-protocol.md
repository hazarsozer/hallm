# 04 — Training Protocol & Config Schema

> Companion to `ROADMAP.md` §5. The matched-budget mechanics, hyperparameters, and the `TrainConfig`
> schema `train.py` consumes (FR-3, NFR-1/3/4, C1/C3). Real runs are Term-2 only; overnight the
> harness is smoke-tested on CPU.

## 1. The matched-budget invariant (the core control)

All four arms must see an **identical training budget** so perplexity differences are attributable to
sharing alone. "Identical" means, byte-for-byte equal across arms:
- model **shape** (d, L, heads, h, block_size) and vocab,
- optimizer + schedule + **all** hyperparameters,
- **total tokens seen** (`max_steps × batch × block_size × grad_accum`),
- **data order** (same seed → same shuffder/sampler sequence),
- dropout, init scheme, precision, grad-clip.

Only the three `share_*` flags differ. `train.py` asserts arms share one `TrainConfig` and differ only
in the `ModelConfig.share_*` fields (logged at startup, NFR-7).

## 2. `TrainConfig` schema

```
TrainConfig:
  # data
  dataset: str = "wikitext-103"     # tiny in-repo sample used for smoke/tests
  tokenizer: str = "gpt2"           # tiktoken
  # optimization
  optimizer: "adamw"
  lr: float = 6e-4                  # peak; cosine-decayed
  min_lr: float = 6e-5              # = lr/10
  warmup_steps: int = 200
  max_steps: int = 50000            # → fixes tokens-seen; identical across arms
  weight_decay: float = 0.1
  betas: [0.9, 0.95]
  grad_clip: float = 1.0
  # batching
  batch_size: int = 24
  grad_accum: int = 1
  block_size: int = 512             # must equal ModelConfig.block_size
  # precision / memory (NFR-3, single 4070 12GB)
  dtype: "bfloat16"
  grad_checkpoint: bool = false     # if on, ON FOR ALL ARMS (comparability)
  # reproducibility (NFR-1)
  seed: int = 1337
  deterministic: bool = true
  # logging (NFR-7)
  eval_interval: int = 1000
  log_interval: int = 50
  out_dir: "runs/<arm>"
```

## 3. Schedule & optimizer
- **AdamW** (β=0.9/0.95, wd=0.1, decoupled), **cosine** decay from `lr`→`min_lr` after linear
  `warmup_steps`. Grad-clip at 1.0. These are nanoGPT-class defaults, sound for 10–124M from scratch.
- **Seeds:** one `seed` drives Python/NumPy/torch + CUDA; data sampler is seeded from it so order is
  reproducible. `torch.use_deterministic_algorithms(True)` when `deterministic`.

## 4. Single-GPU envelope (NFR-3/4, R2)
- Target peak activation **< 11 GB** on the 4070. bf16 + SDPA attention. If a shape nears the limit,
  enable `grad_checkpoint` **for all four arms** (never one).
- Indicative budget **≤ 48 h/arm**; confirmed by the Term-2 tokens/sec pilot before locking size.

## 5. Stability hooks (R3 — off by default, reported if used)
- `sharing_warmup_steps`: initialize unconstrained, enforce W=Wᵀ ties after N steps (lets the model
  settle in a stable region first).
- Optional reduced LR multiplier on shared layers. Any such intervention is applied identically across
  arms and disclosed in the run log.

## 6. Smoke vs real
- **smoke.yaml**: micro shape, `max_steps≈5`, batch≈4, CPU, tiny in-repo corpus → used by
  `tests/test_smoke.py` (loss finite + decreases on overfit-a-batch).
- **run_real_training**: ready-to-run script that trains all four arms on WikiText-103 on the GPU.
  **Created but NOT executed by the overnight loop** — Term-2 manual launch only.
