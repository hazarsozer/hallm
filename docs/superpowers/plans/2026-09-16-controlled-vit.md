# Controlled ViT Study Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train the existing sharing arms (unshared, W+Wᵀ, looped, transposed-loop) as vision transformers on ImageNet-100 with the LM recipe, so the "sharing fails in language but works in vision" question can be answered with a controlled comparison instead of an off-the-shelf DeiT.

**Architecture:** The arms are `ModelConfig` flags over `model/sharing.py`; a ViT reuses that code unchanged and adds only a patch embedding, a class token and a classifier head. One flag (`causal`) becomes configurable so the same attention module serves both domains. The LM trainer is left untouched — vision gets its own loop, its own queue runner and its own report, all writing the artifact shapes the LM side already uses.

**Tech Stack:** PyTorch (CUDA 13, bf16), numpy memmaps for data, `hf` CLI + `pyarrow` for the corpus, pytest, uv.

**Spec:** `docs/superpowers/specs/2026-09-16-controlled-vit-design.md`

## Global Constraints

- Every existing LM run must stay bit-identical: `ModelConfig.causal` defaults to `True` and no LM code path changes behaviour.
- All five arms share one shape: `n_embd=512`, `n_head=8`, `ffn_mult=4`, `patch_size=16`, `image_size=112` → 49 patch tokens + 1 class token = `block_size=50`.
- One recipe for every arm: AdamW, cosine with `warmup_steps=200`, bf16 autocast, `batch_size=256`, `max_steps=50000`, `lr=6e-4`, `min_lr=6e-5`, `weight_decay=0.1`, `grad_clip=1.0`, label smoothing 0.1.
- Augmentation floor, identical across arms: random 112×112 crop from stored 128×128 plus horizontal flip. No mixup, cutmix, RandAugment, erasing or EMA.
- Seed 1337 for all five runs. One seed is descriptive; no verdict is claimed from this pass.
- Storage accounting counts transformer blocks only; patch embedding, positional embedding, class token and head are excluded.
- Run IDs are `V<layers>-<arm>-s<seed>`: `V4-A0-s1337`, `V8-A0-s1337`, `V8-A2-s1337`, `V8-A1u4-s1337`, `V8-A1u4t-s1337`.
- Corpus: `clane9/imagenet-100`, revision SHA recorded in `data/in100/SOURCE.json`.
- Tests run with `uv run pytest`; the analysis group (`uv run --group analysis`) is only for the symmetry script.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/hallm/model/config.py` (modify) | add `causal: bool = True`; add `VisionConfig` subclass with `image_size`, `patch_size`, `n_classes`; add `VSHAPES` |
| `src/hallm/model/sharing.py` (modify, line 128) | use the configured `causal` flag in SDPA |
| `src/hallm/model/vit.py` (create) | patch embedding, class token, positional embedding, blocks from `build_blocks`, classifier head, vision parameter counting |
| `src/hallm/data/imagenet.py` (create) | memmap loading, batch sampling, GPU-side crop/flip/normalize |
| `scripts/prepare_imagenet100.py` (create) | download parquet shards → `data/in100/{train,val}.bin`, `labels.npy`, `SOURCE.json` |
| `src/hallm/vision_train.py` (create) | `VisionTrainConfig`, the vision training loop with resume, top-1 evaluation |
| `src/hallm/vision_runqueue.py` (create) | per-run orchestration and queue drain, writing result + manifest |
| `scripts/run_vision_queue.py` (create) | CLI entry point for the queue |
| `configs/runs/V*.yaml`, `configs/runs/queue-vision.txt` (create) | the five runs and their order |
| `scripts/build_vision_report.py` (create) | `results/reports/vision.md` from `results/runs/V*.json` |
| `scripts/ffn_symmetry.py` (modify) | `--hallm-vit` pattern support using image inputs |
| `tests/test_causal_flag.py`, `test_vit.py`, `test_imagenet_data.py`, `test_vision_train.py`, `test_vision_runqueue.py`, `test_vision_report.py`, `test_symmetry_vit.py` (create) | per-task coverage |

---

### Task 1: Make causality configurable

**Files:**
- Modify: `src/hallm/model/config.py` (ModelConfig fields)
- Modify: `src/hallm/model/sharing.py:100-130` (CausalSelfAttention)
- Test: `tests/test_causal_flag.py`

**Interfaces:**
- Consumes: nothing
- Produces: `ModelConfig.causal: bool = True`; `CausalSelfAttention.causal` attribute read in `forward`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_causal_flag.py
import torch
from hallm.model.config import ModelConfig
from hallm.model.gpt import GPT
from hallm.model.sharing import CausalSelfAttention


def test_causal_defaults_to_true():
    assert ModelConfig().causal is True


def test_non_causal_attention_sees_later_positions():
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=16, block_size=8, n_embd=16, n_layer=1, n_head=2, causal=False)
    attn = CausalSelfAttention(cfg).eval()
    x = torch.randn(1, 8, 16)
    base = attn(x)
    x2 = x.clone()
    x2[0, -1] += 10.0            # perturb the LAST position only
    perturbed = attn(x2)
    # non-causal: position 0 must react to a change at the last position
    assert not torch.allclose(base[0, 0], perturbed[0, 0], atol=1e-5)


def test_causal_attention_ignores_later_positions():
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=16, block_size=8, n_embd=16, n_layer=1, n_head=2)
    attn = CausalSelfAttention(cfg).eval()
    x = torch.randn(1, 8, 16)
    base = attn(x)
    x2 = x.clone()
    x2[0, -1] += 10.0
    perturbed = attn(x2)
    assert torch.allclose(base[0, 0], perturbed[0, 0], atol=1e-6)


def test_lm_forward_unchanged_by_the_new_field():
    """The flag must not perturb an existing LM: same seed, same logits."""
    def logits_for(cfg):
        torch.manual_seed(1234)
        model = GPT(cfg).eval()
        idx = torch.randint(0, cfg.vocab_size, (2, 16), generator=torch.Generator().manual_seed(7))
        with torch.no_grad():
            out, _ = model(idx)
        return out

    a = logits_for(ModelConfig(vocab_size=64, block_size=32, n_embd=32, n_layer=2, n_head=2))
    b = logits_for(ModelConfig(vocab_size=64, block_size=32, n_embd=32, n_layer=2, n_head=2, causal=True))
    assert torch.equal(a, b)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_causal_flag.py -v`
Expected: FAIL — `ModelConfig.__init__() got an unexpected keyword argument 'causal'`

- [ ] **Step 3: Add the field**

In `src/hallm/model/config.py`, in the `# --- shape ---` block of `ModelConfig`, after `ffn_mult`:

```python
    causal: bool = True            # LM masks future positions; a ViT attends both ways
```

- [ ] **Step 4: Thread it into attention**

In `src/hallm/model/sharing.py`, in `CausalSelfAttention.__init__`, next to the other cached config reads:

```python
        self.causal = cfg.causal
```

and at line 128 replace `is_causal=True` with `is_causal=self.causal`:

```python
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=self.causal, dropout_p=self.dropout_p if self.training else 0.0
        )
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_causal_flag.py -v`
Expected: 4 passed

- [ ] **Step 6: Run the full suite — no LM regression**

Run: `uv run pytest -q`
Expected: all pre-existing tests still pass

- [ ] **Step 7: Commit**

```bash
git add src/hallm/model/config.py src/hallm/model/sharing.py tests/test_causal_flag.py
git commit -m "feat(model): configurable causality, default unchanged for LMs"
```

---

### Task 2: The ViT model

**Files:**
- Modify: `src/hallm/model/config.py` (add `VisionConfig`, `VSHAPES`)
- Create: `src/hallm/model/vit.py`
- Test: `tests/test_vit.py`

**Interfaces:**
- Consumes: `ModelConfig.causal` (Task 1), `build_blocks`, `Block`
- Produces:
  - `VisionConfig(ModelConfig)` with `image_size: int = 112`, `patch_size: int = 16`, `n_classes: int = 100`
  - `VSHAPES: dict[str, VisionConfig]` with keys `"v4"`, `"v8"`
  - `ViT(cfg: VisionConfig)` with `forward(pixels: Tensor, targets: Tensor | None = None) -> tuple[Tensor, Tensor | None]`, `num_parameters(non_embedding: bool = False) -> int`, `loop_scales() -> dict[str, float]`, attribute `blocks`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_vit.py
import pytest
import torch
from hallm.model.config import VSHAPES, VisionConfig, arm_config
from hallm.model.vit import ViT


def tiny(**kw):
    base = dict(image_size=32, patch_size=16, n_classes=10, n_embd=32, n_layer=4, n_head=2,
                block_size=5, vocab_size=1, tie_embeddings=False)
    base.update(kw)
    return VisionConfig(**base)


def test_block_size_must_match_patch_count_plus_class_token():
    with pytest.raises(ValueError, match="block_size"):
        tiny(block_size=7)


def test_image_size_must_divide_by_patch_size():
    with pytest.raises(ValueError, match="divisible"):
        tiny(image_size=30, block_size=5)


def test_forward_returns_class_logits_and_loss():
    model = ViT(tiny()).eval()
    pixels = torch.randn(3, 3, 32, 32)
    targets = torch.tensor([1, 2, 3])
    logits, loss = model(pixels, targets)
    assert logits.shape == (3, 10)
    assert loss.ndim == 0 and loss.item() > 0


def test_attention_is_not_causal():
    """A ViT must mix both ways: perturbing the last patch changes the first token's output."""
    torch.manual_seed(0)
    model = ViT(tiny()).eval()
    pixels = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        a, _ = model(pixels)
        pixels2 = pixels.clone()
        pixels2[0, :, -16:, -16:] += 5.0
        b, _ = model(pixels2)
    assert not torch.allclose(a, b, atol=1e-5)


def test_arms_build_over_the_same_sharing_code():
    for arm, n_blocks in [("A0", 4), ("A2", 4), ("A1u2", 2)]:
        model = ViT(arm_config(tiny(), arm))
        assert len(model.blocks) == n_blocks


def test_looped_arm_stores_what_the_shallower_unshared_model_stores():
    looped = ViT(arm_config(tiny(n_layer=4), "A1u2"))
    shallow = ViT(arm_config(tiny(n_layer=2, block_size=5), "A0"))
    assert looped.num_parameters(non_embedding=True) == shallow.num_parameters(non_embedding=True)


def test_non_embedding_excludes_patch_pos_head_and_class_token():
    model = ViT(tiny())
    blocks_only = sum(p.numel() for p in model.blocks.parameters())
    # final LayerNorm rides with the blocks; patch/pos/head/cls do not
    assert model.num_parameters(non_embedding=True) == blocks_only + sum(
        p.numel() for p in model.ln_f.parameters()
    )


def test_transposed_loop_reports_alphas():
    model = ViT(arm_config(tiny(n_layer=4), "A1u2a"))
    scales = model.loop_scales()
    assert set(scales) == {"alpha_attn_0", "alpha_mlp_0", "alpha_attn_1", "alpha_mlp_1"}


def test_vshapes_match_the_spec_geometry():
    for key, n_layer in [("v4", 4), ("v8", 8)]:
        cfg = VSHAPES[key]
        assert (cfg.image_size, cfg.patch_size, cfg.n_embd, cfg.n_head) == (112, 16, 512, 8)
        assert cfg.n_layer == n_layer
        assert cfg.block_size == (112 // 16) ** 2 + 1 == 50
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_vit.py -v`
Expected: FAIL — `cannot import name 'VisionConfig'`

- [ ] **Step 3: Add `VisionConfig` and `VSHAPES`**

In `src/hallm/model/config.py`, after the `ModelConfig` definition:

```python
@dataclass(frozen=True)
class VisionConfig(ModelConfig):
    """A ViT's shape. Inherits every sharing flag, so `arm_config` and `build_blocks` work unchanged.

    `block_size` is the token count (patches + class token) and `vocab_size`/`tie_embeddings` are
    unused by `ViT` — they stay on the base config so manifests keep one schema.
    """

    image_size: int = 112
    patch_size: int = 16
    n_classes: int = 100
    causal: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.image_size % self.patch_size:
            raise ValueError(
                f"image_size ({self.image_size}) must be divisible by patch_size ({self.patch_size})"
            )
        n_tokens = (self.image_size // self.patch_size) ** 2 + 1
        if self.block_size != n_tokens:
            raise ValueError(
                f"block_size ({self.block_size}) must equal patches + class token ({n_tokens})"
            )
```

and after `SHAPES`:

```python
# Controlled ViT study (spec 2026-09-16 §2): one geometry, two depths. 49 patches + class token.
VSHAPES: dict[str, VisionConfig] = {
    "v4": VisionConfig(vocab_size=1, block_size=50, n_embd=512, n_layer=4, n_head=8,
                       tie_embeddings=False),
    "v8": VisionConfig(vocab_size=1, block_size=50, n_embd=512, n_layer=8, n_head=8,
                       tie_embeddings=False),
}
```

Export both from `src/hallm/model/__init__.py` by adding `VSHAPES` and `VisionConfig` to the import line and to `__all__`.

- [ ] **Step 4: Write the model**

```python
# src/hallm/model/vit.py
"""Vision transformer over the sharing-aware sublayers (spec 2026-09-16).

Identical to `gpt.py` in everything that matters for the study: the same `Block`, the same
`build_blocks` cross-layer helper and the same transposed-loop pass schedule. Only the ends differ —
a patch embedding and a class token instead of token embeddings, and a classifier head instead of a
weight-tied LM head — so an arm gap here is comparable with the same arm gap in language.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from hallm.model.config import VisionConfig
from hallm.model.gpt import Block
from hallm.model.sharing import build_blocks


class ViT(nn.Module):
    def __init__(self, cfg: VisionConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.patch = nn.Conv2d(3, cfg.n_embd, kernel_size=cfg.patch_size, stride=cfg.patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.n_embd))
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = build_blocks(cfg, Block)
        self.ln_f = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.head = nn.Linear(cfg.n_embd, cfg.n_classes, bias=True)
        self.label_smoothing = 0.0   # set by the trainer; kept here so forward stays self-contained

        self.apply(self._init_weights)
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, pixels: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B = pixels.shape[0]
        x = self.patch(pixels).flatten(2).transpose(1, 2)          # (B, n_patches, C)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        pos = torch.arange(x.shape[1], device=pixels.device)
        x = self.drop(x + self.pos_emb(pos))

        n_unique = len(self.blocks)
        for i in range(self.cfg.n_layer):
            block = self.blocks[i % n_unique]
            if self.cfg.loop_pass2 is not None and (i // n_unique) % 2 == 1:
                x = block(x, transposed=True)
            else:
                x = block(x)
        x = self.ln_f(x)
        logits = self.head(x[:, 0])                                 # class token
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits, targets, label_smoothing=self.label_smoothing)
        return logits, loss

    def loop_scales(self) -> dict[str, float]:
        if self.cfg.loop_pass2 != "scaled":
            return {}
        out: dict[str, float] = {}
        for j, block in enumerate(self.blocks):
            out[f"alpha_attn_{j}"] = round(block.alpha_attn.item(), 6)
            out[f"alpha_mlp_{j}"] = round(block.alpha_mlp.item(), 6)
        return out

    def num_parameters(self, non_embedding: bool = False) -> int:
        """Unique parameters; `non_embedding` counts transformer blocks + final LN only, the vision
        analogue of the LM's non-embedding count (spec 2026-09-16 §2)."""
        seen: set[int] = set()
        modules = [self.blocks, self.ln_f] if non_embedding else [self]
        total = 0
        for m in modules:
            for p in m.parameters():
                if id(p) in seen:
                    continue
                seen.add(id(p))
                total += p.numel()
        return total
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/test_vit.py -v`
Expected: 9 passed

- [ ] **Step 6: Commit**

```bash
git add src/hallm/model/vit.py src/hallm/model/config.py src/hallm/model/__init__.py tests/test_vit.py
git commit -m "feat(vit): vision transformer over the shared sublayers"
```

---

### Task 3: Corpus preparation and image batches

**Files:**
- Create: `src/hallm/data/imagenet.py`
- Create: `scripts/prepare_imagenet100.py`
- Modify: `src/hallm/data/__init__.py` (re-export)
- Test: `tests/test_imagenet_data.py`

**Interfaces:**
- Consumes: nothing from earlier tasks
- Produces:
  - `preprocess_image(img: "PIL.Image.Image", size: int = 128) -> np.ndarray` — uint8 `(3, size, size)`
  - `load_images(path: str | Path) -> np.memmap` — uint8 `(N, 3, 128, 128)`
  - `load_labels(path: str | Path) -> np.ndarray` — int64 `(N,)`
  - `get_image_batch(images, labels, batch_size: int, crop: int, device, generator: torch.Generator, train: bool = True) -> tuple[torch.Tensor, torch.Tensor]` — float32 normalized `(B, 3, crop, crop)` and int64 `(B,)`
  - `IMAGENET_MEAN`, `IMAGENET_STD` (tuples of three floats)

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_imagenet_data.py
import numpy as np
import torch
from PIL import Image
from hallm.data.imagenet import (IMAGENET_MEAN, get_image_batch, load_images, load_labels,
                                 preprocess_image)


def _fixture(tmp_path, n=8):
    images = np.random.default_rng(0).integers(0, 255, (n, 3, 128, 128), dtype=np.uint8)
    labels = np.arange(n, dtype=np.int64)
    images.tofile(tmp_path / "train.bin")
    np.save(tmp_path / "train_labels.npy", labels)
    return tmp_path


def test_preprocess_resizes_shorter_side_and_center_crops():
    img = Image.new("RGB", (320, 160), color=(10, 20, 30))
    out = preprocess_image(img, size=128)
    assert out.shape == (3, 128, 128)
    assert out.dtype == np.uint8
    assert int(out[0].mean()) == 10


def test_preprocess_handles_grayscale_input():
    out = preprocess_image(Image.new("L", (200, 200), color=128), size=128)
    assert out.shape == (3, 128, 128)


def test_load_images_reads_the_memmap_shape(tmp_path):
    d = _fixture(tmp_path)
    images = load_images(d / "train.bin")
    assert images.shape == (8, 3, 128, 128) and images.dtype == np.uint8
    assert load_labels(d / "train_labels.npy").shape == (8,)


def test_batch_shapes_and_normalization(tmp_path):
    d = _fixture(tmp_path)
    images, labels = load_images(d / "train.bin"), load_labels(d / "train_labels.npy")
    gen = torch.Generator().manual_seed(0)
    x, y = get_image_batch(images, labels, batch_size=4, crop=112, device="cpu", generator=gen)
    assert x.shape == (4, 3, 112, 112) and x.dtype == torch.float32
    assert y.shape == (4,) and y.dtype == torch.int64
    assert abs(float(x.mean())) < 3.0        # normalized, not raw 0-255
    assert float(x.max()) < 10.0


def test_batches_are_reproducible_given_a_seeded_generator(tmp_path):
    d = _fixture(tmp_path)
    images, labels = load_images(d / "train.bin"), load_labels(d / "train_labels.npy")
    a = get_image_batch(images, labels, 4, 112, "cpu", torch.Generator().manual_seed(7))
    b = get_image_batch(images, labels, 4, 112, "cpu", torch.Generator().manual_seed(7))
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_eval_batches_are_deterministic_center_crops(tmp_path):
    d = _fixture(tmp_path)
    images, labels = load_images(d / "train.bin"), load_labels(d / "train_labels.npy")
    a = get_image_batch(images, labels, 4, 112, "cpu", torch.Generator().manual_seed(1), train=False)
    b = get_image_batch(images, labels, 4, 112, "cpu", torch.Generator().manual_seed(2), train=False)
    assert torch.equal(a[0], b[0])           # no crop jitter, no flip, no shuffling
    assert torch.equal(a[1], torch.arange(4))


def test_mean_constant_is_the_standard_imagenet_one():
    assert IMAGENET_MEAN == (0.485, 0.456, 0.406)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_imagenet_data.py -v`
Expected: FAIL — `No module named 'hallm.data.imagenet'`

- [ ] **Step 3: Write the data module**

```python
# src/hallm/data/imagenet.py
"""ImageNet-100 as a uint8 memmap (spec 2026-09-16 §4).

Same shape of interface as `wikitext.py`: one `.bin` per split, opened as a memmap, sampled with a
seeded generator so the data order is identical across arms. Images are stored at 128x128 and
cropped to 112 on the GPU, which keeps augmentation off the CPU and out of the input pipeline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
STORED = 128


def preprocess_image(img, size: int = STORED) -> np.ndarray:
    """PIL image → uint8 (3, size, size): resize shorter side to `size`, then center crop."""
    from PIL import Image

    img = img.convert("RGB")
    w, h = img.size
    scale = size / min(w, h)
    img = img.resize((max(size, round(w * scale)), max(size, round(h * scale))), Image.BICUBIC)
    w, h = img.size
    left, top = (w - size) // 2, (h - size) // 2
    img = img.crop((left, top, left + size, top + size))
    return np.asarray(img, dtype=np.uint8).transpose(2, 0, 1)


def load_images(path: str | Path) -> np.memmap:
    arr = np.memmap(path, dtype=np.uint8, mode="r")
    n = arr.size // (3 * STORED * STORED)
    return arr.reshape(n, 3, STORED, STORED)


def load_labels(path: str | Path) -> np.ndarray:
    return np.load(path).astype(np.int64)


def get_image_batch(
    images: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    crop: int,
    device: str | torch.device = "cpu",
    generator: torch.Generator | None = None,
    train: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a batch. Training draws random indices, random crops and random flips; evaluation
    takes the first `batch_size` images with a fixed center crop and no flip."""
    n = len(images)
    if train:
        idx = torch.randint(0, n, (batch_size,), generator=generator)
    else:
        idx = torch.arange(min(batch_size, n))
    raw = torch.from_numpy(np.ascontiguousarray(images[idx.numpy()]))
    x = raw.to(device, non_blocking=True).float().div_(255.0)

    margin = STORED - crop
    if train and margin > 0:
        oy, ox = (torch.randint(0, margin + 1, (2,), generator=generator)).tolist()
    else:
        oy = ox = margin // 2
    x = x[:, :, oy : oy + crop, ox : ox + crop]
    if train:
        flip = torch.rand(batch_size, generator=generator) < 0.5
        x[flip.to(device)] = x[flip.to(device)].flip(-1)

    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    x = (x - mean) / std
    y = torch.from_numpy(np.ascontiguousarray(labels[idx.numpy()])).to(device)
    return x, y
```

Add to `src/hallm/data/__init__.py`:

```python
from hallm.data.imagenet import get_image_batch, load_images, load_labels, preprocess_image
```

and the four names to `__all__`.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_imagenet_data.py -v`
Expected: 7 passed

- [ ] **Step 5: Write the prepare script**

```python
# scripts/prepare_imagenet100.py
"""Build data/in100/{train,val}.bin from clane9/imagenet-100 (spec 2026-09-16 §4).

One-time, ~26 GB of parquet downloaded to the HF cache, ~6.6 GB written here.

Usage:
  uv run python scripts/prepare_imagenet100.py --out data/in100
  uv run python scripts/prepare_imagenet100.py --out /tmp/in100-probe --limit 64   # smoke check
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np

from hallm.data.imagenet import STORED, preprocess_image

REPO = "clane9/imagenet-100"


def convert(split: str, out_dir: Path, limit: int | None) -> int:
    from datasets import load_dataset

    ds = load_dataset(REPO, split=split, streaming=True)
    bin_path, labels = out_dir / f"{split}.bin", []
    with open(bin_path, "wb") as f:
        for i, row in enumerate(ds):
            if limit is not None and i >= limit:
                break
            img = row["image"]
            if isinstance(img, (bytes, bytearray)):
                from PIL import Image

                img = Image.open(io.BytesIO(img))
            f.write(preprocess_image(img, STORED).tobytes())
            labels.append(int(row["label"]))
            if i % 5000 == 0:
                print(f"{split}: {i} images", flush=True)
    np.save(out_dir / f"{split}_labels.npy", np.asarray(labels, dtype=np.int64))
    return len(labels)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/in100")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    from huggingface_hub import HfApi

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sha = HfApi().dataset_info(REPO).sha
    counts = {split: convert(split, out, args.limit) for split in ("train", "validation")}
    (out / "validation.bin").rename(out / "val.bin")
    (out / "validation_labels.npy").rename(out / "val_labels.npy")
    (out / "SOURCE.json").write_text(
        json.dumps(
            {"repo": REPO, "revision": sha, "stored_size": STORED, "counts": counts,
             "preprocess": "resize shorter side then center crop, uint8 CHW"},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {out}/train.bin, val.bin, labels, SOURCE.json — {counts}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Add the `datasets` dependency**

Run: `uv add --group analysis datasets`
Expected: `pyproject.toml` and `uv.lock` updated

- [ ] **Step 7: Smoke-check the script against the real repo (64 images, no GPU)**

Run: `uv run --group analysis python scripts/prepare_imagenet100.py --out /tmp/in100-probe --limit 64`
Expected: writes `/tmp/in100-probe/train.bin` (64 × 3 × 128 × 128 = 3,145,728 bytes), `val.bin`, both label files and `SOURCE.json` with a 40-character revision SHA

- [ ] **Step 8: Commit**

```bash
git add src/hallm/data/imagenet.py src/hallm/data/__init__.py scripts/prepare_imagenet100.py \
        tests/test_imagenet_data.py pyproject.toml uv.lock
git commit -m "feat(data): ImageNet-100 memmap pipeline and prepare script"
```

---

### Task 4: The vision training loop

**Files:**
- Create: `src/hallm/vision_train.py`
- Test: `tests/test_vision_train.py`

**Interfaces:**
- Consumes: `ViT` (Task 2), `get_image_batch`/`load_images`/`load_labels` (Task 3), `cosine_lr`, `configure_optimizer`, `set_seed`, `save_resume_checkpoint`/`load_resume_checkpoint` from `hallm.train`
- Produces:
  - `VisionTrainConfig(TrainConfig)` adding `crop_size: int = 112`, `label_smoothing: float = 0.1`, `eval_batch: int = 250`, with `dataset: str = "imagenet-100"`
  - `evaluate_top1(model, images, labels, cfg, device) -> tuple[float, float]` — (top-1 fraction, mean loss) over the whole split
  - `train_vision(model, cfg, images, labels, device=None, progress=False, resume_path=None, stop_step=None, val_images=None, val_labels=None, metrics_path=None) -> list[dict]`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_vision_train.py
import numpy as np
import torch
from hallm.model.config import VisionConfig, arm_config
from hallm.model.vit import ViT
from hallm.vision_train import VisionTrainConfig, evaluate_top1, train_vision


def tiny_cfg():
    return VisionConfig(image_size=32, patch_size=16, n_classes=4, n_embd=32, n_layer=2,
                        n_head=2, block_size=5, vocab_size=1, tie_embeddings=False)


def fake_data(n=16, seed=0):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 4, n).astype(np.int64)
    images = np.zeros((n, 3, 128, 128), dtype=np.uint8)
    for i, y in enumerate(labels):           # class-coded images: learnable in a few steps
        images[i, int(y) % 3] = 200
    return images, labels


def test_defaults_match_the_spec_recipe():
    cfg = VisionTrainConfig()
    assert (cfg.crop_size, cfg.label_smoothing, cfg.dataset) == (112, 0.1, "imagenet-100")
    assert (cfg.warmup_steps, cfg.max_steps, cfg.lr) == (200, 50_000, 6e-4)


def test_training_reduces_loss_on_a_learnable_toy_set(tmp_path):
    torch.manual_seed(0)
    model = ViT(tiny_cfg())
    cfg = VisionTrainConfig(max_steps=30, warmup_steps=2, batch_size=8, crop_size=32,
                            log_interval=1, eval_interval=0, dtype="float32", deterministic=False)
    images, labels = fake_data()
    history = train_vision(model, cfg, images, labels, device="cpu")
    assert history[-1]["loss"] < history[0]["loss"]


def test_label_smoothing_reaches_the_model():
    model = ViT(tiny_cfg())
    cfg = VisionTrainConfig(max_steps=1, warmup_steps=1, batch_size=4, crop_size=32,
                            label_smoothing=0.25, dtype="float32", deterministic=False)
    images, labels = fake_data(8)
    train_vision(model, cfg, images, labels, device="cpu")
    assert model.label_smoothing == 0.25


def test_evaluate_top1_is_a_fraction_and_uses_eval_mode():
    model = ViT(tiny_cfg())
    model.train()
    images, labels = fake_data(8)
    cfg = VisionTrainConfig(crop_size=32, eval_batch=4, dtype="float32")
    top1, loss = evaluate_top1(model, images, labels, cfg, "cpu")
    assert 0.0 <= top1 <= 1.0 and loss > 0
    assert model.training      # evaluation restores the previous mode


def test_resume_continues_from_the_checkpoint(tmp_path):
    images, labels = fake_data()
    resume = tmp_path / "resume.pt"
    cfg = VisionTrainConfig(max_steps=10, warmup_steps=1, batch_size=8, crop_size=32,
                            checkpoint_interval=1, log_interval=1, eval_interval=0,
                            dtype="float32", deterministic=False)
    torch.manual_seed(0)
    first = train_vision(ViT(tiny_cfg()), cfg, images, labels, device="cpu",
                         resume_path=str(resume), stop_step=5)
    assert first[-1]["step"] == 4
    second = train_vision(ViT(tiny_cfg()), cfg, images, labels, device="cpu",
                          resume_path=str(resume))
    assert second[0]["step"] == 5 and second[-1]["step"] == 9


def test_metrics_file_records_top1(tmp_path):
    import json

    images, labels = fake_data()
    metrics = tmp_path / "metrics.jsonl"
    cfg = VisionTrainConfig(max_steps=4, warmup_steps=1, batch_size=8, crop_size=32,
                            log_interval=1, eval_interval=2, eval_batch=8,
                            dtype="float32", deterministic=False)
    train_vision(ViT(tiny_cfg()), cfg, images, labels, device="cpu", val_images=images,
                 val_labels=labels, metrics_path=str(metrics))
    rows = [json.loads(line) for line in metrics.read_text().splitlines()]
    assert any("val_top1" in r and "val_loss" in r for r in rows)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_vision_train.py -v`
Expected: FAIL — `No module named 'hallm.vision_train'`

- [ ] **Step 3: Write the trainer**

```python
# src/hallm/vision_train.py
"""Vision training loop (spec 2026-09-16 §4).

A sibling of `train.py`, not a replacement: the LM loop is mid-campaign and token-batch specific.
Everything that defines the recipe — optimizer, schedule, seeding, resume format, metrics file — is
imported from `train.py` so the two domains cannot silently drift apart.
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


@torch.no_grad()
def evaluate_top1(model: ViT, images: np.ndarray, labels: np.ndarray, cfg: VisionTrainConfig,
                  device) -> tuple[float, float]:
    """Top-1 fraction and mean loss over the whole split, center-cropped, in eval mode."""
    was_training = model.training
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    for start in range(0, len(images), cfg.eval_batch):
        stop = min(start + cfg.eval_batch, len(images))
        x, y = get_image_batch(images[start:stop], labels[start:stop], stop - start,
                               cfg.crop_size, device, None, train=False)
        logits, loss = model(x, y)
        correct += int((logits.argmax(-1) == y).sum())
        loss_sum += float(loss) * (stop - start)
        total += stop - start
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
```

- [ ] **Step 4: Confirm the resume helpers behave as assumed**

Run: `uv run python -c "import inspect, hallm.train as t; print(inspect.signature(t.save_resume_checkpoint)); print(inspect.signature(t.load_resume_checkpoint))"`
Expected: `(path, model, train_cfg, opt, gen, step)` and `(path, map_location='cpu')`. `save_resume_checkpoint` stores `asdict(model.cfg)` itself and loads with `weights_only=True`, so a `VisionConfig` round-trips as a plain dict and no unpickling surface is added. If the signatures differ, fix the call sites in `vision_train.py` — do not change `train.py`.

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/test_vision_train.py -v`
Expected: 6 passed

- [ ] **Step 6: Commit**

```bash
git add src/hallm/vision_train.py tests/test_vision_train.py
git commit -m "feat(vision): training loop with top-1 eval and mid-run resume"
```

---

### Task 5: Configs, queue runner and result rows

**Files:**
- Modify: `src/hallm/experiment.py` (add `load_vision_experiment`)
- Create: `src/hallm/vision_runqueue.py`
- Create: `scripts/run_vision_queue.py`
- Create: `configs/runs/V4-A0-s1337.yaml`, `V8-A0-s1337.yaml`, `V8-A2-s1337.yaml`, `V8-A1u4-s1337.yaml`, `V8-A1u4t-s1337.yaml`, `configs/runs/queue-vision.txt`
- Test: `tests/test_vision_runqueue.py`

**Interfaces:**
- Consumes: `VSHAPES`, `arm_config` (Task 2), `train_vision`, `evaluate_top1`, `VisionTrainConfig` (Task 4), `build_manifest`/`write_manifest`, `write_run_result`
- Produces:
  - `load_vision_experiment(path) -> tuple[VisionConfig, VisionTrainConfig]`
  - `run_one_vision(cfg_path, data_dir, results_dir, device, stop_step=None) -> str` (`"ok"` / `"paused"`)
  - `drain_vision(queue_path, data_dir, results_dir, device, max_runs=None, stop_step=None) -> list[dict]`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_vision_runqueue.py
import json

import numpy as np
import pytest
import torch
import yaml
from hallm.experiment import load_vision_experiment
from hallm.vision_runqueue import drain_vision, run_one_vision


def write_cfg(tmp_path, name, arm, n_layer):
    cfg = {
        "vshape": "v4" if n_layer == 4 else "v8",
        "arm": arm,
        "train": {"max_steps": 2, "warmup_steps": 1, "batch_size": 4, "crop_size": 32,
                  "eval_interval": 0, "log_interval": 1, "dtype": "float32",
                  "deterministic": False, "checkpoint_interval": 0,
                  "out_dir": str(tmp_path / "runs" / name)},
    }
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def tiny_data(tmp_path, n=8):
    d = tmp_path / "data"
    d.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    for split in ("train", "val"):
        rng.integers(0, 255, (n, 3, 128, 128), dtype=np.uint8).tofile(d / f"{split}.bin")
        np.save(d / f"{split}_labels.npy", rng.integers(0, 100, n).astype(np.int64))
    return d


def test_load_vision_experiment_resolves_shape_and_arm(tmp_path):
    path = write_cfg(tmp_path, "V8-A1u4-s1337", "A1u4", 8)
    model_cfg, train_cfg = load_vision_experiment(path)
    assert (model_cfg.n_layer, model_cfg.n_unique_blocks) == (8, 4)
    assert model_cfg.block_size == 50 and model_cfg.causal is False
    assert train_cfg.crop_size == 32


def test_run_one_writes_result_and_manifest(tmp_path):
    data = tiny_data(tmp_path)
    results = tmp_path / "results"
    path = write_cfg(tmp_path, "V4-A0-s1337", "A0", 4)
    status = run_one_vision(path, data, results, device="cpu")
    assert status == "ok"
    row = json.loads((results / "V4-A0-s1337.json").read_text())
    assert row["run"] == "V4-A0-s1337" and row["arm"] == "A0"
    assert 0.0 <= row["top1"] <= 1.0
    assert row["dataset"] == "imagenet-100" and row["n_classes"] == 100
    assert row["non_embedding_params_M"] > 0
    assert (tmp_path / "runs" / "V4-A0-s1337" / "manifest.json").exists()


def test_looped_and_shallow_rows_agree_on_stored_weights(tmp_path):
    data = tiny_data(tmp_path)
    results = tmp_path / "results"
    run_one_vision(write_cfg(tmp_path, "V4-A0-s1337", "A0", 4), data, results, device="cpu")
    run_one_vision(write_cfg(tmp_path, "V8-A1u4-s1337", "A1u4", 8), data, results, device="cpu")
    a = json.loads((results / "V4-A0-s1337.json").read_text())
    b = json.loads((results / "V8-A1u4-s1337.json").read_text())
    assert a["non_embedding_params_M"] == pytest.approx(b["non_embedding_params_M"], rel=1e-6)


def test_drain_runs_every_queued_config_in_order(tmp_path):
    data = tiny_data(tmp_path)
    results = tmp_path / "results"
    queue = tmp_path / "queue.txt"
    paths = [write_cfg(tmp_path, "V4-A0-s1337", "A0", 4),
             write_cfg(tmp_path, "V8-A0-s1337", "A0", 8)]
    queue.write_text("\n".join(str(p) for p in paths), encoding="utf-8")
    rows = drain_vision(queue, data, results, device="cpu")
    assert [r["run"] for r in rows] == ["V4-A0-s1337", "V8-A0-s1337"]


def test_finished_runs_are_skipped_on_a_second_drain(tmp_path):
    data = tiny_data(tmp_path)
    results = tmp_path / "results"
    queue = tmp_path / "queue.txt"
    queue.write_text(str(write_cfg(tmp_path, "V4-A0-s1337", "A0", 4)), encoding="utf-8")
    drain_vision(queue, data, results, device="cpu")
    assert drain_vision(queue, data, results, device="cpu") == []
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_vision_runqueue.py -v`
Expected: FAIL — `cannot import name 'load_vision_experiment'`

- [ ] **Step 3: Add the config loader**

Append to `src/hallm/experiment.py`:

```python
def load_vision_experiment(path: str | Path) -> tuple["VisionConfig", "VisionTrainConfig"]:
    """Return (vision_config, vision_train_config) from a YAML experiment file.

    Mirrors `load_experiment`: `vshape` names an entry in `VSHAPES`, `arm` selects the sharing
    flags through the same `arm_config` used by the LM ladder.
    """
    from hallm.model.config import VSHAPES, VisionConfig
    from hallm.vision_train import VisionTrainConfig

    spec = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if "vshape" in spec:
        base = VSHAPES[spec["vshape"]]
    elif "model" in spec:
        base = VisionConfig(**spec["model"])
    else:
        raise ValueError(f"{path}: vision config must define either `vshape` or `model`")
    model_cfg = arm_config(base, spec.get("arm", "A0"))
    train_cfg = VisionTrainConfig(**spec.get("train", {}))
    return model_cfg, train_cfg
```

- [ ] **Step 4: Write the queue runner**

```python
# src/hallm/vision_runqueue.py
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
```

- [ ] **Step 5: Write the CLI entry point**

```python
# scripts/run_vision_queue.py
"""Drain a vision queue file (spec 2026-09-16).

Usage:
  uv run python scripts/run_vision_queue.py --queue configs/runs/queue-vision.txt \
      --data data/in100 --results-dir results/runs
"""

from __future__ import annotations

import argparse

import torch

from hallm.vision_runqueue import drain_vision


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True)
    ap.add_argument("--data", default="data/in100")
    ap.add_argument("--results-dir", default="results/runs")
    ap.add_argument("--max-runs", type=int, default=None)
    ap.add_argument("--stop-step", type=int, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    drain_vision(args.queue, args.data, args.results_dir, device, args.max_runs, args.stop_step)


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Write the five configs**

`configs/runs/V4-A0-s1337.yaml` (repeat for the others, changing only `vshape`, `arm` and `out_dir`):

```yaml
vshape: v4
arm: A0
train:
  lr: 0.0006
  min_lr: 6.0e-05
  warmup_steps: 200
  max_steps: 50000
  weight_decay: 0.1
  grad_clip: 1.0
  batch_size: 256
  grad_accum: 1
  crop_size: 112
  label_smoothing: 0.1
  dtype: bfloat16
  deterministic: true
  eval_interval: 1000
  checkpoint_interval: 1000
  seed: 1337
  out_dir: runs/vision/V4-A0-s1337
```

The four others: `V8-A0-s1337` (`vshape: v8`, `arm: A0`), `V8-A2-s1337` (`vshape: v8`, `arm: A2`), `V8-A1u4-s1337` (`vshape: v8`, `arm: A1u4`), `V8-A1u4t-s1337` (`vshape: v8`, `arm: A1u4t`). Every other key is identical — that is the matched-budget control.

`configs/runs/queue-vision.txt`:

```
configs/runs/V4-A0-s1337.yaml
configs/runs/V8-A0-s1337.yaml
configs/runs/V8-A2-s1337.yaml
configs/runs/V8-A1u4-s1337.yaml
configs/runs/V8-A1u4t-s1337.yaml
```

- [ ] **Step 7: Verify every config loads and the arms are what the spec says**

Run:
```bash
uv run python -c "
from hallm.experiment import load_vision_experiment
import glob
for p in sorted(glob.glob('configs/runs/V*.yaml')):
    m, t = load_vision_experiment(p)
    print(p.split('/')[-1], m.arm, 'L' + str(m.n_layer), 'unique=' + str(len(range(m.n_unique_blocks or m.n_layer))), t.batch_size, t.max_steps)
"
```
Expected: five lines; `V8-A1u4` and `V8-A1u4t` show 4 unique blocks, the rest show their depth; every row `256 50000`

- [ ] **Step 8: Run the tests**

Run: `uv run pytest tests/test_vision_runqueue.py -v`
Expected: 5 passed

- [ ] **Step 9: Commit**

```bash
git add src/hallm/vision_runqueue.py src/hallm/experiment.py scripts/run_vision_queue.py \
        configs/runs/V*.yaml configs/runs/queue-vision.txt tests/test_vision_runqueue.py
git commit -m "feat(vision): queue runner, configs and result rows for the five arms"
```

---

### Task 6: Symmetry measurement over our own ViTs

**Files:**
- Modify: `scripts/ffn_symmetry.py` (add `--hallm-vit`)
- Test: `tests/test_symmetry_vit.py`

**Interfaces:**
- Consumes: `ViT` (Task 2), `load_images`/`get_image_batch` (Task 3), `hallm_rotation_shares` (existing — it hooks `model.blocks[*].mlp` and calls `model(inputs)`, so it works unchanged on a `ViT`)
- Produces: `hallm_vit_rows(patterns: list[str], data_dir: str, n_positions: int, n_images: int, device: str) -> list[dict]` in `scripts/ffn_symmetry.py`, rows shaped like the existing ones with `kind="vision-ours"`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_symmetry_vit.py
import torch
from hallm.model.config import VisionConfig, arm_config
from hallm.model.vit import ViT
from hallm.symmetry import hallm_rotation_shares


def tiny(arm):
    cfg = VisionConfig(image_size=32, patch_size=16, n_classes=4, n_embd=32, n_layer=2,
                       n_head=2, block_size=5, vocab_size=1, tie_embeddings=False)
    return ViT(arm_config(cfg, arm))


def test_rotation_shares_run_on_a_vit_and_return_one_per_layer():
    torch.manual_seed(0)
    model = tiny("A0")
    pixels = torch.randn(4, 3, 32, 32)
    shares = hallm_rotation_shares(model, pixels, n_positions=8)
    assert len(shares) == 2
    assert all(0.0 <= s <= 1.0 for s in shares)


def test_w_plus_wt_vit_is_exactly_symmetric_in_the_u_basis():
    torch.manual_seed(0)
    model = tiny("A2")
    pixels = torch.randn(4, 3, 32, 32)
    shares = hallm_rotation_shares(model, pixels, n_positions=8)
    assert max(shares) < 1e-10          # the §3 sanity gate, as for L8-A2


def test_eval_mode_is_restored_after_measurement():
    model = tiny("A0")
    model.train()
    hallm_rotation_shares(model, torch.randn(2, 3, 32, 32), n_positions=4)
    assert model.training
```

- [ ] **Step 2: Run to verify failure or pass**

Run: `uv run --group analysis pytest tests/test_symmetry_vit.py -v`
Expected: PASS if `hallm_rotation_shares` is already model-agnostic; FAIL with a shape or attribute error if it assumes token input. If it fails, fix `hallm_rotation_shares` to call `model(inputs)` with whatever tensor it is given and to take `n_total` from the hooked activation — do not special-case vision.

- [ ] **Step 3: Add the ViT rows to the script**

In `scripts/ffn_symmetry.py`, alongside `lm_rows` and `deit_row`:

```python
def hallm_vit_rows(patterns: list[str], data_dir: str, n_positions: int, n_images: int,
                   device: str) -> list[dict]:
    """Rotation shares for ViTs we trained ourselves — the controlled half of the LM/vision
    comparison (spec 2026-09-16 §5). Inputs are held-out ImageNet-100 images, center-cropped."""
    from hallm.data.imagenet import get_image_batch, load_images, load_labels
    from hallm.model.config import VisionConfig
    from hallm.model.vit import ViT

    images = load_images(Path(data_dir) / "val.bin")
    labels = load_labels(Path(data_dir) / "val_labels.npy")
    rows = []
    for path in sorted(p for pat in patterns for p in glob.glob(pat)):
        if Path(path).stem == "resume":
            continue
        ckpt = torch.load(path, map_location=device, weights_only=True)   # dict-of-primitives
        model = ViT(VisionConfig(**ckpt["model_cfg"]))
        model.load_state_dict(ckpt["model"])
        model.to(device).eval()
        x, _ = get_image_batch(images, labels, n_images, model.cfg.image_size, device, None,
                               train=False)
        shares_u = hallm_rotation_shares(model, x, n_positions, basis="u")
        shares_xhat = hallm_rotation_shares(model, x, n_positions, basis="xhat")
        rows.append(_row(Path(path).stem, "vision-ours", model.cfg.arm, shares_u, shares_xhat))
        print(rows[-1]["model"], rows[-1]["mean"])
    return rows
```

and wire the flag in `main`:

```python
    ap.add_argument("--hallm-vit", nargs="*", default=[])
    ap.add_argument("--vit-data", default="data/in100")
    ap.add_argument("--n-images-ours", type=int, default=64)
```

calling `rows += hallm_vit_rows(args.hallm_vit, args.vit_data, args.n_positions, args.n_images_ours, device)` where the other row builders are called. In the report header, label the DeiT-small row "uncontrolled reference (other recipe, scale and data)".

- [ ] **Step 4: Run the tests**

Run: `uv run --group analysis pytest tests/test_symmetry_vit.py tests/test_symmetry.py -v`
Expected: the new 3 pass and the existing symmetry tests still pass

- [ ] **Step 5: Commit**

```bash
git add scripts/ffn_symmetry.py tests/test_symmetry_vit.py src/hallm/symmetry.py
git commit -m "feat(symmetry): rotation share over ViTs trained with our recipe"
```

---

### Task 7: The vision report

**Files:**
- Create: `scripts/build_vision_report.py`
- Test: `tests/test_vision_report.py`

**Interfaces:**
- Consumes: `read_run_results` (existing), result rows from Task 5
- Produces: `vision_report(rows: list[dict]) -> str` and `PAIRS: list[tuple[str, str, str]]` in `scripts/build_vision_report.py`; writes `results/reports/vision.md`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_vision_report.py
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "bvr", Path(__file__).resolve().parents[1] / "scripts" / "build_vision_report.py")
bvr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bvr)


def rows():
    return [
        {"run": "V4-A0-s1337", "arm": "A0", "n_layer": 4, "top1": 0.400, "val_loss": 2.60,
         "non_embedding_params_M": 12.59, "dataset": "imagenet-100"},
        {"run": "V8-A0-s1337", "arm": "A0", "n_layer": 8, "top1": 0.450, "val_loss": 2.40,
         "non_embedding_params_M": 25.17, "dataset": "imagenet-100"},
        {"run": "V8-A2-s1337", "arm": "A2", "n_layer": 8, "top1": 0.410, "val_loss": 2.55,
         "non_embedding_params_M": 12.59, "dataset": "imagenet-100"},
        {"run": "V8-A1u4-s1337", "arm": "A1u4", "n_layer": 8, "top1": 0.430, "val_loss": 2.45,
         "non_embedding_params_M": 12.59, "dataset": "imagenet-100"},
    ]


def test_header_states_one_seed_and_no_verdict():
    text = bvr.vision_report(rows())
    assert "1 seed" in text and "descriptive" in text.lower()
    assert "SUPPORTED" not in text          # verdicts are a seeded-pass concept


def test_pairs_report_top1_point_differences():
    text = bvr.vision_report(rows())
    # V8-A1u4 (0.430) vs V4-A0 (0.400) = +3.0 points for the looped arm
    assert "+3.0" in text and "V8-A1u4" in text


def test_depth_gate_is_reported():
    text = bvr.vision_report(rows())
    assert "depth gate" in text.lower() and "pass" in text.lower()


def test_depth_gate_fails_when_deeper_is_not_better():
    bad = rows()
    bad[1]["top1"] = 0.350                  # V8-A0 below V4-A0
    text = bvr.vision_report(bad)
    assert "FAIL" in text


def test_missing_runs_are_shown_as_pending():
    text = bvr.vision_report(rows())        # V8-A1u4t absent
    assert "pending" in text and "V8-A1u4t" in text


def test_storage_column_flags_an_unequal_pairing():
    broken = rows()
    broken[3]["non_embedding_params_M"] = 20.0
    text = bvr.vision_report(broken)
    assert "storage mismatch" in text.lower()
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_vision_report.py -v`
Expected: FAIL — the script does not exist

- [ ] **Step 3: Write the report builder**

```python
# scripts/build_vision_report.py
"""Regenerate results/reports/vision.md from results/runs/V*.json (spec 2026-09-16 §8).

Idempotent and disposable, like build_reports.py: the run files are the source of truth. One seed
means point differences only — no verdict machinery lives here.

Usage: uv run python scripts/build_vision_report.py [--results results]
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from hallm.results import read_run_results

STAMP = "<!-- GENERATED by scripts/build_vision_report.py from results/runs/V*.json — do not edit -->"

ARMS = ["V4-A0", "V8-A0", "V8-A2", "V8-A1u4", "V8-A1u4t"]
# (a, b, kind): a's top-1 minus b's, in percentage points. Positive = a is better.
PAIRS = [
    ("V8-A1u4", "V4-A0", "iso-storage"),
    ("V8-A2", "V4-A0", "iso-storage"),
    ("V8-A1u4", "V8-A2", "matched storage and compute"),
    ("V8-A1u4t", "V8-A1u4", "matched storage and compute"),
]


def _by_arm(rows: list[dict]) -> dict[str, dict]:
    return {r["run"].rsplit("-s", 1)[0]: r for r in rows if r["run"].startswith("V")}


def vision_report(rows: list[dict]) -> str:
    seen = _by_arm(rows)
    out = [STAMP, "# Controlled ViT study (spec 2026-09-16)", ""]
    out.append("ImageNet-100 @112px, one recipe across arms, **1 seed (1337) — descriptive only, "
               "no verdict is claimed**. Δ is a top-1 point difference; positive means the first "
               "model is better.")
    out += ["", "| run | arm | layers | stored non-emb M | top-1 | val loss |", "|---|---|---|---|---|---|"]
    for name in ARMS:
        r = seen.get(name)
        if r is None:
            out.append(f"| {name} | — | — | — | pending | pending |")
        else:
            out.append(f"| {r['run']} | {r['arm']} | {r['n_layer']} | "
                       f"{r['non_embedding_params_M']:.2f} | {100 * r['top1']:.1f} | "
                       f"{r['val_loss']:.4f} |")

    out += ["", "## Pairs", "", "| a | b | kind | top-1 a / b | Δ points | note |",
            "|---|---|---|---|---|---|"]
    for a, b, kind in PAIRS:
        ra, rb = seen.get(a), seen.get(b)
        if ra is None or rb is None:
            out.append(f"| {a} | {b} | {kind} | — | — | pending |")
            continue
        delta = 100 * (ra["top1"] - rb["top1"])
        note = ""
        if abs(ra["non_embedding_params_M"] - rb["non_embedding_params_M"]) > 0.01:
            note = "**storage mismatch** — the pairing is not iso-storage"
        out.append(f"| {a} | {b} | {kind} | {100 * ra['top1']:.1f} / {100 * rb['top1']:.1f} | "
                   f"{delta:+.1f} | {note} |")

    deep, shallow = seen.get("V8-A0"), seen.get("V4-A0")
    if deep and shallow:
        ok = deep["top1"] > shallow["top1"]
        verdict = "pass" if ok else "**FAIL — the setup cannot say anything about sharing**"
        out += ["", f"**Depth gate (spec §3):** V8-A0 {100 * deep['top1']:.1f} vs V4-A0 "
                    f"{100 * shallow['top1']:.1f} → {verdict}"]
    else:
        out += ["", "**Depth gate (spec §3):** pending"]

    follow = ""
    a2, v4 = seen.get("V8-A2"), seen.get("V4-A0")
    if a2 and v4:
        gap = abs(100 * (a2["top1"] - v4["top1"]))
        follow = ("triggered — run seeds 1338/1339 for V8-A2, V4-A0 and V8-A1u4"
                  if gap >= 1.0 else "not triggered — reported as no difference detected at 1 seed")
        out += ["", f"**Follow-up seed rule (spec §3):** |Δ| = {gap:.1f} points → {follow}"]
    out += ["", f"<!-- built {datetime.now(timezone.utc).isoformat(timespec='seconds')} -->"]
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    rows = read_run_results(Path(args.results) / "runs")
    path = Path(args.results) / "reports" / "vision.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(vision_report(rows), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_vision_report.py -v`
Expected: 6 passed

- [ ] **Step 5: Run the whole suite before the GPU work starts**

Run: `uv run pytest -q`
Expected: everything passes, LM tests included

- [ ] **Step 6: Commit**

```bash
git add scripts/build_vision_report.py tests/test_vision_report.py
git commit -m "feat(reports): vision table with the pairs, depth gate and seed trigger"
```

---

### Task 8: Timing probe and queue sizing (GPU)

**Files:**
- Modify: `docs/superpowers/specs/2026-09-16-controlled-vit-design.md` (record the measured pace)
- No source changes

**Interfaces:**
- Consumes: everything above
- Produces: a measured steps/min figure and a go/no-go on queue size

- [ ] **Step 1: Prepare the corpus (one-time, ~40 min, no GPU)**

Run: `uv run --group analysis python scripts/prepare_imagenet100.py --out data/in100`
Expected: `data/in100/train.bin` ≈ 6.4 GB, `val.bin` ≈ 245 MB, both label files, `SOURCE.json` with the revision SHA

- [ ] **Step 2: Sanity-check what was written**

Run:
```bash
uv run python -c "
from hallm.data.imagenet import load_images, load_labels
for s in ('train', 'val'):
    im, lb = load_images(f'data/in100/{s}.bin'), load_labels(f'data/in100/{s}_labels.npy')
    print(s, im.shape, lb.shape, int(lb.min()), int(lb.max()))
"
```
Expected: train ≈ (130000, 3, 128, 128), val ≈ (5000, 3, 128, 128), labels 0–99, image and label counts equal per split

- [ ] **Step 3: Time 200 steps of the heaviest arm**

Run:
```bash
cd ~/Dev/hallm && time uv run python scripts/run_vision_queue.py \
  --queue <(echo configs/runs/V8-A0-s1337.yaml) --data data/in100 \
  --results-dir /tmp/vision-probe --stop-step 200
```
Expected: prints `[stop] V8-A0-s1337: paused at step 200`; note the wall-clock

- [ ] **Step 4: Record the pace and size the queue**

Compute steps/min from step 3 and write one line into the spec's §4 timing paragraph: the measured pace, the implied hours per 50k-step run, and the implied total for five runs. If a single run exceeds 8 hours, stop and report — the spec's §9 R4 says the 2026-10-05 gate work has priority, and the queue is re-planned rather than launched.

- [ ] **Step 5: Discard the probe state**

Run: `rm -rf /tmp/vision-probe runs/vision/V8-A0-s1337`
Expected: the probe's partial checkpoint is gone so the real run starts fresh

- [ ] **Step 6: Commit the measurement**

```bash
git add docs/superpowers/specs/2026-09-16-controlled-vit-design.md
git commit -m "docs(spec): record the measured vision training pace"
```

- [ ] **Step 7: Launch the queue (only after step 4 says it fits)**

```bash
cd ~/Dev/hallm && systemd-inhibit --what=sleep:idle --who=hallm --why=vision sh -c \
  'uv run python scripts/run_vision_queue.py --queue configs/runs/queue-vision.txt \
   --data data/in100 --results-dir results/runs >> runs-vision.log 2>&1'
```

As each run lands: copy `runs/vision/<name>/manifest.json` to `results/manifests/`, run `uv run python scripts/build_vision_report.py`, and commit the result, manifest and report together — the same landing ritual the LM queues use.

---

## Self-Review

**Spec coverage:** §2 arms → Tasks 2 and 5 · §3 rules, gates and the seed trigger → Task 7 (report) with the gate computed from the rows · §4 data and recipe → Tasks 3 and 5 configs · §4 timing probe → Task 8 · §5 symmetry → Task 6 · §6 predictions → no code needed; scored after the runs land, as the transposed-loop predictions were · §7 reuse boundary → Tasks 1, 2, 4 · §8 artifacts → Tasks 5 and 7 · §9 R5 (new-trainer bugs) → Task 4 tests plus the Task 7 full-suite run.

**Placeholders:** none — every step carries the code or the command it needs.

**Type consistency:** `VisionConfig` fields (`image_size`, `patch_size`, `n_classes`) are used identically in Tasks 2, 5 and 6. `get_image_batch`'s signature is fixed in Task 3 and called with the same argument order in Tasks 4 and 6. `evaluate_top1` returns `(top1, loss)` in Task 4 and is unpacked that way in Task 5. Result-row keys written in Task 5 (`top1`, `val_loss`, `n_layer`, `non_embedding_params_M`, `arm`, `run`) are exactly the keys Task 7's report reads.

**Known soft spot:** Task 4 step 4 verifies the resume-helper names in `train.py` rather than assuming them, because the plan was written from a partial read of that file.
