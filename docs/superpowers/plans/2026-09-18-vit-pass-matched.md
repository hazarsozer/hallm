# ViT Pass-Matched Re-Run Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-run the five controlled ViT arms on ImageNet-1k at 9.99 passes instead of ImageNet-100 at 101.03, so the vision arms match the LM ladder's data reuse and the depth gate can be tested on a corpus that supports it.

**Architecture:** Nothing about the model, sharing code or training loop changes. The work is a new corpus (a parallel prepare script writing a 62.97 GB uint8 memmap), a class-count of 1000 threaded through two new `VSHAPES` entries, a stratified validation subsample so periodic eval does not cost 10× the first pass's, and report columns that keep the two passes distinguishable. Execution is staged: a probe already running gates the corpus swap, and the depth gate on stage 2 gates the three sharing arms.

**Tech Stack:** Python 3.14, uv, PyTorch (CUDA 13), NumPy memmaps, `datasets` + `huggingface_hub` (group `data`), PIL (group `analysis`), pytest.

**Spec:** `docs/superpowers/specs/2026-09-18-vit-pass-matched-design.md`

## Global Constraints

- **Corpus:** `benjamin-paine/imagenet-1k-128x128`, 1,281,167 train / 50,000 validation. The test split has no public labels and is not used.
- **Passes:** 50,000 steps × batch 256 ÷ 1,281,167 = **9.99**. The LM ladder is 10.31. Do not change `max_steps` or `batch_size` to "round" this.
- **Unchanged from the first pass, in every task:** `lr` 6e-4, `min_lr` 6e-5, `warmup_steps` 200, `weight_decay` 0.1, `grad_clip` 1.0, `batch_size` 256, `grad_accum` 1, `crop_size` 112, `label_smoothing` 0.1, `dtype` bfloat16, `deterministic` true, seed 1337, `n_embd` 512, `n_head` 8, `ffn_mult` 4, patch 16, `block_size` 50.
- **No augmentation is added.** Random 112 crop from 128 + horizontal flip + label smoothing 0.1 is the whole floor. Any change here invalidates the study.
- **Stored format:** uint8 memmap `[N, 3, 128, 128]`, CHW, matching `hallm.data.imagenet.STORED = 128`.
- **The ImageNet-100 path must stay bit-identical.** Every change to `data/imagenet.py`, `vision_train.py` and `vision_runqueue.py` defaults to the existing behaviour, and Task 2 carries the regression test that proves it.
- **`--ipv4` is required on this host** for anything touching the HF CDN: the IPv6 route blackholes.
- **Run IDs:** `V{4,8}-{arm}-s1337-in1k`. Probe runs are `-p5k` and never enter a comparison table.
- **Commit message trailer:** `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`
- **Branch:** `vit-pass-matched`. Do not merge to `main` or push; Hazar reviews first.

## File Structure

| File | Responsibility |
|---|---|
| `src/hallm/model/config.py` | add `v4-in1k` / `v8-in1k` to `VSHAPES`, derived from `v4` / `v8` so they cannot drift |
| `src/hallm/data/imagenet.py` | add `stratified_subsample`; everything else untouched |
| `src/hallm/vision_train.py` | add `eval_subsample` field to `VisionTrainConfig` (default 0 = whole split) |
| `src/hallm/vision_runqueue.py` | apply the subsample to periodic eval only; add `top1_best`, `best_step`, `corpus`, `passes` to the result row |
| `scripts/prepare_imagenet1k.py` | new: parallel shard decode into a preallocated memmap |
| `scripts/build_vision_report.py` | two labelled blocks, corpus/pass columns, overfitting flag |
| `configs/runs/V*-s1337-in1k.yaml`, `queue-vision-in1k.txt` | the five arms, split into gate pair and sharing arms |
| `tests/test_vision_in1k.py` | new tests for all of the above |

---

### Task 1: Vision shapes for 1000 classes, and the new result-row fields

**Files:**
- Modify: `src/hallm/model/config.py` (after `VSHAPES`, line ~200)
- Modify: `src/hallm/vision_runqueue.py:26-40` (`_result_row`), `src/hallm/vision_runqueue.py:96-105` (call site)
- Test: `tests/test_vision_in1k.py`

**Interfaces:**
- Consumes: `VSHAPES`, `VisionConfig`, `arm_config` from `hallm.model.config`.
- Produces: `VSHAPES["v4-in1k"]`, `VSHAPES["v8-in1k"]` (both `n_classes=1000`); `_result_row(model, model_cfg, train_cfg, name, top1, val_loss, top1_best=None, best_step=None, n_train=None)` returning the existing keys plus `top1_best: float | None`, `best_step: int | None`, `corpus: str`, `passes: float | None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_vision_in1k.py
from dataclasses import replace

from hallm.model.config import VSHAPES


def test_in1k_shapes_differ_from_base_only_in_class_count():
    for base, derived in (("v4", "v4-in1k"), ("v8", "v8-in1k")):
        assert VSHAPES[derived].n_classes == 1000
        assert VSHAPES[base].n_classes == 100
        # every other field identical — this is the whole point of deriving them
        assert replace(VSHAPES[derived], n_classes=100) == VSHAPES[base]
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd ~/Dev/hallm && uv run pytest tests/test_vision_in1k.py -q`
Expected: FAIL with `KeyError: 'v4-in1k'`

- [ ] **Step 3: Implement**

```python
# src/hallm/model/config.py, immediately after the VSHAPES dict literal
# ImageNet-1k arms (spec 2026-09-18 §2): derived with dataclasses.replace rather than written out,
# so the only field that can ever differ from the 100-class arms is the class count.
VSHAPES["v4-in1k"] = _dc_replace(VSHAPES["v4"], n_classes=1000)
VSHAPES["v8-in1k"] = _dc_replace(VSHAPES["v8"], n_classes=1000)
```

Add `from dataclasses import replace as _dc_replace` to the imports at the top of the file if `replace` is not already imported under another name.

- [ ] **Step 4: Run it to verify it passes**

Run: `cd ~/Dev/hallm && uv run pytest tests/test_vision_in1k.py -q`
Expected: PASS

- [ ] **Step 5: Write the failing test for the result row**

```python
# append to tests/test_vision_in1k.py
from hallm.model.config import VSHAPES
from hallm.model.vit import ViT
from hallm.vision_runqueue import _result_row
from hallm.vision_train import VisionTrainConfig


def test_best_from_metrics_reads_val_top1(tmp_path):
    from hallm.vision_runqueue import _best_from_metrics

    p = tmp_path / "metrics.jsonl"
    p.write_text(
        '{"step": 1000, "loss": 2.0, "val_top1": 0.40, "val_loss": 2.5}\n'
        '{"step": 2000, "loss": 1.9}\n'                      # a step with no eval
        '{"step": 3000, "loss": 1.8, "val_top1": 0.43, "val_loss": 2.4}\n'
        '{"step": 4000, "loss": 1.7, "val_top1": 0.41, "val_loss": 2.6}\n',
        encoding="utf-8")
    assert _best_from_metrics(p) == (0.43, 3000)
    assert _best_from_metrics(tmp_path / "absent.jsonl") == (None, None)


def test_result_row_carries_corpus_passes_and_best():
    cfg = VSHAPES["v4-in1k"]
    model = ViT(cfg)
    tcfg = VisionTrainConfig(max_steps=50000, batch_size=256, dataset="imagenet-1k")
    row = _result_row(model, cfg, tcfg, "V4-A0-s1337-in1k", 0.4, 3.0,
                      top1_best=0.41, best_step=48000, n_train=1281167)
    assert row["corpus"] == "imagenet-1k"
    assert row["n_classes"] == 1000
    assert row["top1_best"] == 0.41
    assert row["best_step"] == 48000
    assert abs(row["passes"] - 9.99) < 0.01


def test_result_row_without_best_stays_backward_compatible():
    cfg = VSHAPES["v4"]
    model = ViT(cfg)
    tcfg = VisionTrainConfig(max_steps=50000, batch_size=256)
    row = _result_row(model, cfg, tcfg, "V4-A0-s1337", 0.4814, 2.5436)
    assert row["top1_best"] is None and row["best_step"] is None and row["passes"] is None
    assert row["corpus"] == "imagenet-100"
    assert row["top1"] == 0.4814
```

- [ ] **Step 6: Run it to verify it fails**

Run: `cd ~/Dev/hallm && uv run pytest tests/test_vision_in1k.py -q`
Expected: FAIL with `TypeError: _result_row() got an unexpected keyword argument 'top1_best'`

- [ ] **Step 7: Implement**

```python
# src/hallm/vision_runqueue.py — replace _result_row
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
```

Then add a `_best_from_metrics` helper and use it at the call site:

```python
# src/hallm/vision_runqueue.py — new helper above run_one_vision
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
```

```python
# src/hallm/vision_runqueue.py — replace the write_run_result call in run_one_vision
    top1_best, best_step = _best_from_metrics(out / "metrics.jsonl")
    write_run_result(results_dir, _result_row(model, model_cfg, train_cfg, name, top1, val_loss,
                                              top1_best=top1_best, best_step=best_step,
                                              n_train=len(images)))
```

- [ ] **Step 8: Run the tests**

Run: `cd ~/Dev/hallm && uv run pytest tests/test_vision_in1k.py -q`
Expected: PASS (4 tests)

- [ ] **Step 9: Run the full suite to check nothing regressed**

Run: `cd ~/Dev/hallm && uv run pytest -q`
Expected: all pass (203 at branch point, plus the new ones)

- [ ] **Step 10: Commit**

```bash
git add src/hallm/model/config.py src/hallm/vision_runqueue.py tests/test_vision_in1k.py
git commit -m "feat(vision): 1000-class shapes and corpus/passes/best-checkpoint result fields

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: Stratified validation subsample for periodic eval

Periodic eval currently scores the **whole** validation split (`evaluate_top1` loops over all of it). At 50,000 images that is 10× the first pass's cost for no added resolution on the training curve. Spec §7: periodic eval on a fixed stratified 10,000-image subsample, final number on all 50,000.

**Files:**
- Modify: `src/hallm/data/imagenet.py` (new function at end)
- Modify: `src/hallm/vision_train.py:30-40` (`VisionTrainConfig`)
- Modify: `src/hallm/vision_runqueue.py:82-93` (`run_one_vision` val wiring)
- Test: `tests/test_vision_in1k.py`

**Interfaces:**
- Produces: `stratified_subsample(labels: np.ndarray, per_class: int, seed: int) -> np.ndarray` returning sorted int64 indices; `VisionTrainConfig.eval_subsample: int = 0` (0 = whole split).

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_vision_in1k.py
import numpy as np

from hallm.data.imagenet import stratified_subsample


def test_stratified_subsample_is_balanced_deterministic_and_sorted():
    labels = np.repeat(np.arange(100), 50)          # 5000 images, 50 per class
    idx = stratified_subsample(labels, per_class=10, seed=1337)
    assert len(idx) == 1000
    assert np.array_equal(idx, np.sort(idx))        # sorted => contiguous memmap reads
    counts = np.bincount(labels[idx], minlength=100)
    assert set(counts.tolist()) == {10}
    assert np.array_equal(idx, stratified_subsample(labels, per_class=10, seed=1337))
    assert not np.array_equal(idx, stratified_subsample(labels, per_class=10, seed=1338))


def test_stratified_subsample_takes_all_when_class_is_smaller_than_quota():
    labels = np.array([0, 0, 1, 1, 1, 2])
    idx = stratified_subsample(labels, per_class=2, seed=1337)
    counts = np.bincount(labels[idx], minlength=3)
    assert counts.tolist() == [2, 2, 1]


def test_eval_subsample_defaults_to_whole_split():
    from hallm.vision_train import VisionTrainConfig
    assert VisionTrainConfig().eval_subsample == 0
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd ~/Dev/hallm && uv run pytest tests/test_vision_in1k.py -q`
Expected: FAIL with `ImportError: cannot import name 'stratified_subsample'`

- [ ] **Step 3: Implement the sampler**

```python
# src/hallm/data/imagenet.py, appended
def stratified_subsample(labels: np.ndarray, per_class: int, seed: int) -> np.ndarray:
    """Indices of `per_class` images from each class, drawn once with `seed`.

    Returned sorted, so scoring the subsample reads the memmap front-to-back rather than jumping
    (at 62.97 GB the validation split does not fit in page cache). A class with fewer than
    `per_class` members contributes all of them rather than raising.
    """
    rng = np.random.default_rng(seed)
    picks = []
    for cls in np.unique(labels):
        members = np.flatnonzero(labels == cls)
        take = min(per_class, len(members))
        picks.append(rng.choice(members, size=take, replace=False))
    return np.sort(np.concatenate(picks)).astype(np.int64)
```

- [ ] **Step 4: Add the config field**

```python
# src/hallm/vision_train.py, inside VisionTrainConfig
    # 0 keeps the first pass's behaviour exactly: periodic eval scores the whole split. A positive
    # value is images PER CLASS for the periodic eval only — the final reported number in
    # vision_runqueue always scores the full split (spec 2026-09-18 §7).
    eval_subsample: int = 0
```

- [ ] **Step 5: Run to verify the tests pass**

Run: `cd ~/Dev/hallm && uv run pytest tests/test_vision_in1k.py -q`
Expected: PASS

- [ ] **Step 6: Write the failing wiring test**

```python
# append to tests/test_vision_in1k.py
def test_run_one_vision_uses_subsample_for_periodic_and_full_split_for_final(tmp_path, monkeypatch):
    """The periodic eval sees 2 images per class; the final number sees all 6."""
    import hallm.vision_runqueue as vrq

    labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    seen = {}

    def fake_train_vision(model, train_cfg, images, lbls, **kw):
        seen["periodic_n"] = len(kw["val_images"])
        return []

    def fake_evaluate_top1(model, images, lbls, cfg, device):
        seen["final_n"] = len(images)
        return 0.5, 1.0

    monkeypatch.setattr(vrq, "train_vision", fake_train_vision)
    monkeypatch.setattr(vrq, "evaluate_top1", fake_evaluate_top1)
    n = vrq._split_for_periodic_eval(np.zeros((6, 3, 128, 128), np.uint8), labels, 2, 1337)
    assert len(n[0]) == 4 and len(n[1]) == 4
    assert np.bincount(n[1]).tolist() == [2, 2]
```

- [ ] **Step 7: Run to verify it fails**

Run: `cd ~/Dev/hallm && uv run pytest tests/test_vision_in1k.py::test_run_one_vision_uses_subsample_for_periodic_and_full_split_for_final -q`
Expected: FAIL with `AttributeError: module 'hallm.vision_runqueue' has no attribute '_split_for_periodic_eval'`

- [ ] **Step 8: Implement the wiring**

```python
# src/hallm/vision_runqueue.py — new helper above run_one_vision
def _split_for_periodic_eval(val_images, val_labels, per_class: int, seed: int):
    """(images, labels) for the periodic eval: the whole split when per_class <= 0, else a fixed
    stratified subsample materialised into RAM (10 per class over 1000 classes is ~491 MB)."""
    if per_class <= 0:
        return val_images, val_labels
    from hallm.data.imagenet import stratified_subsample

    idx = stratified_subsample(val_labels, per_class, seed)
    return np.ascontiguousarray(val_images[idx]), val_labels[idx]
```

```python
# src/hallm/vision_runqueue.py — in run_one_vision, replace the train_vision call's val arguments
    periodic_images, periodic_labels = _split_for_periodic_eval(
        val_images, val_labels, train_cfg.eval_subsample, train_cfg.seed)
    set_seed(train_cfg.seed, train_cfg.deterministic)
    model = ViT(model_cfg)
    print(f"[run ] {name}: {'resuming' if resume.exists() else 'fresh'} on {device}"
          f" | periodic eval on {len(periodic_images)} of {len(val_images)} val images")
    train_vision(model, train_cfg, images, labels, device=device, progress=True,
                 resume_path=str(resume), stop_step=stop_step, val_images=periodic_images,
                 val_labels=periodic_labels, metrics_path=str(out / "metrics.jsonl"))
```

The existing `evaluate_top1(model, val_images, val_labels, train_cfg, device)` line below it is **not** changed — the final number stays on the full split.

- [ ] **Step 9: Run the tests**

Run: `cd ~/Dev/hallm && uv run pytest tests/test_vision_in1k.py -q`
Expected: PASS

- [ ] **Step 10: Write the ImageNet-100 regression test**

This is the Global Constraint that matters most: the existing corpus path must be untouched.

```python
# append to tests/test_vision_in1k.py
import pytest

from hallm.data.imagenet import get_image_batch, load_images, load_labels

IN100 = Path("data/in100")


@pytest.mark.skipif(not (IN100 / "train.bin").exists(), reason="ImageNet-100 memmaps not present")
def test_in100_batches_unchanged_by_this_pass():
    """Same seed, same loader, same bytes — the first pass's data order must not move."""
    import torch

    images, labels = load_images(IN100 / "train.bin"), load_labels(IN100 / "train_labels.npy")
    g1 = torch.Generator().manual_seed(1337)
    x1, y1 = get_image_batch(images, labels, 8, 112, "cpu", g1, train=True)
    g2 = torch.Generator().manual_seed(1337)
    x2, y2 = get_image_batch(images, labels, 8, 112, "cpu", g2, train=True)
    assert torch.equal(x1, x2) and torch.equal(y1, y2)
    assert x1.shape == (8, 3, 112, 112)
    # the recorded first-batch labels for seed 1337 — a canary on the data order itself
    assert y1.tolist() == RECORDED_IN100_SEED1337_LABELS
```

Before running it, produce `RECORDED_IN100_SEED1337_LABELS` from the current code **on the pre-change commit** and paste the literal in:

```bash
git stash && uv run python -c "
import torch
from hallm.data.imagenet import get_image_batch, load_images, load_labels
g = torch.Generator().manual_seed(1337)
im = load_images('data/in100/train.bin'); lb = load_labels('data/in100/train_labels.npy')
print(get_image_batch(im, lb, 8, 112, 'cpu', g, train=True)[1].tolist())
" && git stash pop
```

Add the printed list as a module-level constant `RECORDED_IN100_SEED1337_LABELS = [...]`.

- [ ] **Step 11: Run it**

Run: `cd ~/Dev/hallm && uv run pytest tests/test_vision_in1k.py -q`
Expected: PASS — if the canary fails, a change in this task moved the ImageNet-100 data order and must be reverted before going further.

- [ ] **Step 12: Run the full suite**

Run: `cd ~/Dev/hallm && uv run pytest -q`
Expected: all pass

- [ ] **Step 13: Commit**

```bash
git add src/hallm/data/imagenet.py src/hallm/vision_train.py src/hallm/vision_runqueue.py tests/test_vision_in1k.py
git commit -m "feat(vision): stratified eval subsample for periodic eval, full split for the final number

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: `scripts/prepare_imagenet1k.py`

The ImageNet-100 script streams single-threaded; at its observed rate 1,281,167 images would take hours. This one downloads the 14 parquet shards, reads each shard's row count from its footer, preallocates the output file, and decodes shards in parallel with each worker writing at its own byte offset — so there is no concatenation pass and no doubled disk.

**Files:**
- Create: `scripts/prepare_imagenet1k.py`
- Test: `tests/test_prepare_imagenet1k.py`

**Interfaces:**
- Consumes: `hallm.data.imagenet.STORED`, `preprocess_image`, and `_force_ipv4` (copied, not imported — `prepare_imagenet100.py` is a script, not a module).
- Produces: `decode_row(raw: bytes | PIL.Image) -> np.ndarray` of shape `(3, 128, 128)` uint8; `shard_offsets(counts: list[int]) -> list[int]`; files `data/in1k/{train,val}.bin`, `{train,val}_labels.npy`, `SOURCE.json`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_prepare_imagenet1k.py
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

spec = importlib.util.spec_from_file_location(
    "prep1k", Path(__file__).parent.parent / "scripts" / "prepare_imagenet1k.py")
prep1k = importlib.util.module_from_spec(spec)
sys.modules["prep1k"] = prep1k
spec.loader.exec_module(prep1k)


def test_shard_offsets_are_cumulative_byte_positions():
    # 3 shards of 2, 3, 1 images at 3*128*128 bytes each
    per = 3 * 128 * 128
    assert prep1k.shard_offsets([2, 3, 1]) == [0, 2 * per, 5 * per]


def test_decode_row_passes_through_an_already_128px_image_without_resampling():
    arr = np.random.default_rng(0).integers(0, 256, (128, 128, 3), dtype=np.uint8)
    out = prep1k.decode_row(Image.fromarray(arr))
    assert out.shape == (3, 128, 128) and out.dtype == np.uint8
    # exact passthrough: no bicubic resample may touch an image already at the stored size
    assert np.array_equal(out, arr.transpose(2, 0, 1))


def test_decode_row_resizes_and_centre_crops_a_non_square_image():
    arr = np.random.default_rng(1).integers(0, 256, (200, 300, 3), dtype=np.uint8)
    out = prep1k.decode_row(Image.fromarray(arr))
    assert out.shape == (3, 128, 128) and out.dtype == np.uint8


def test_decode_row_accepts_encoded_bytes():
    import io
    buf = io.BytesIO()
    Image.fromarray(np.zeros((128, 128, 3), np.uint8)).save(buf, format="PNG")
    out = prep1k.decode_row(buf.getvalue())
    assert out.shape == (3, 128, 128)
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd ~/Dev/hallm && uv run --group data --group analysis pytest tests/test_prepare_imagenet1k.py -q`
Expected: FAIL — the script does not exist.

- [ ] **Step 3: Implement the script**

```python
"""Build data/in1k/{train,val}.bin from benjamin-paine/imagenet-1k-128x128 (spec 2026-09-18 §4).

One-time: ~6.14 GB of parquet downloaded to the HF cache, 62.97 GB (train) + 2.46 GB (val)
written here. Shards are decoded in parallel, each worker writing at its own byte offset into a
preallocated file, so there is no concatenation pass and no doubled disk.

Usage:
  uv run --group data --group analysis python scripts/prepare_imagenet1k.py --out data/in1k --ipv4
  uv run --group data --group analysis python scripts/prepare_imagenet1k.py --out /tmp/in1k-probe --limit 10000 --ipv4

`--limit N` processes only the first N rows of the first shard of each split — the timing slice
spec 2026-09-18 §4 requires before the full run is started.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import socket
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("PYTHONWARNINGS", "ignore::UserWarning:multiprocessing.resource_tracker")

import numpy as np

from hallm.data.imagenet import STORED, preprocess_image

REPO = "benjamin-paine/imagenet-1k-128x128"
BYTES_PER_IMAGE = 3 * STORED * STORED


def _force_ipv4() -> None:
    """Resolve IPv4 only, for this process. This host's IPv6 route to the HF CDN blackholes; see
    scripts/prepare_imagenet100.py for the full account."""
    _getaddrinfo = socket.getaddrinfo

    def _ipv4_only(host, port, family=0, type=0, proto=0, flags=0):
        return _getaddrinfo(host, port, socket.AF_INET, type, proto, flags)

    socket.getaddrinfo = _ipv4_only


def decode_row(img) -> np.ndarray:
    """Row's image → uint8 (3, 128, 128). An image already at the stored size is passed through
    without resampling; anything else goes through the same resize-and-centre-crop the
    ImageNet-100 prepare applied, so the two corpora are preprocessed identically."""
    from PIL import Image

    if isinstance(img, (bytes, bytearray)):
        img = Image.open(io.BytesIO(img))
    img = img.convert("RGB")
    if img.size == (STORED, STORED):
        return np.asarray(img, dtype=np.uint8).transpose(2, 0, 1)
    return preprocess_image(img, STORED)


def shard_offsets(counts: list[int]) -> list[int]:
    """Byte offset at which each shard's images start in the output file."""
    out, running = [], 0
    for c in counts:
        out.append(running * BYTES_PER_IMAGE)
        running += c
    return out


def _shard_counts(paths: list[Path]) -> list[int]:
    import pyarrow.parquet as pq

    return [pq.ParquetFile(p).metadata.num_rows for p in paths]


def _decode_shard(args) -> tuple[int, list[int], int]:
    """Worker: decode one shard into `bin_path` at `offset`. Returns (shard_index, labels, n_resized)."""
    shard_index, path, bin_path, offset, limit = args
    import pyarrow.parquet as pq
    from PIL import Image

    table = pq.ParquetFile(path).read()
    images, labels_col = table.column("image"), table.column("label")
    labels, n_resized = [], 0
    with open(bin_path, "r+b") as f:
        f.seek(offset)
        for i in range(len(images)):
            if limit is not None and i >= limit:
                break
            cell = images[i].as_py()
            raw = cell["bytes"] if isinstance(cell, dict) else cell
            img = Image.open(io.BytesIO(raw)) if isinstance(raw, (bytes, bytearray)) else raw
            img = img.convert("RGB")
            if img.size != (STORED, STORED):
                n_resized += 1
            f.write(decode_row(img).tobytes())
            labels.append(int(labels_col[i].as_py()))
    return shard_index, labels, n_resized


def convert(split: str, files: list[Path], out_dir: Path, limit: int | None, workers: int):
    counts = _shard_counts(files)
    if limit is not None:
        counts = [min(c, limit) for c in counts[:1]]
        files = files[:1]
    total = sum(counts)
    bin_path = out_dir / f"{split}.bin"
    with open(bin_path, "wb") as f:                      # preallocate, no concatenation pass
        f.truncate(total * BYTES_PER_IMAGE)
    offsets = shard_offsets(counts)
    jobs = [(i, files[i], bin_path, offsets[i], limit) for i in range(len(files))]
    per_shard: dict[int, list[int]] = {}
    resized = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for shard_index, labels, n_resized in ex.map(_decode_shard, jobs):
            per_shard[shard_index] = labels
            resized += n_resized
            print(f"{split}: shard {shard_index} done ({len(labels)} images)", flush=True)
    ordered = [lab for i in range(len(files)) for lab in per_shard[i]]
    np.save(out_dir / f"{split}_labels.npy", np.asarray(ordered, dtype=np.int64))
    return total, resized


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/in1k")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))
    ap.add_argument("--ipv4", action="store_true",
                    help="Force IPv4-only DNS resolution (this host's IPv6 route to the HF CDN "
                         "blackholes). Off by default.")
    args = ap.parse_args()
    if args.ipv4:
        _force_ipv4()

    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    info = api.dataset_info(REPO)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    wanted = {"train": [], "validation": []}
    for sib in sorted(info.siblings, key=lambda s: s.rfilename):
        for split in wanted:
            if sib.rfilename.startswith(f"data/{split}-"):
                wanted[split].append(sib.rfilename)

    counts, resized_counts = {}, {}
    for split, names in wanted.items():
        paths = [Path(hf_hub_download(REPO, n, repo_type="dataset")) for n in names]
        print(f"{split}: {len(paths)} shard(s) cached", flush=True)
        counts[split], resized_counts[split] = convert(split, paths, out, args.limit, args.workers)

    (out / "validation.bin").rename(out / "val.bin")
    (out / "validation_labels.npy").rename(out / "val_labels.npy")
    (out / "SOURCE.json").write_text(
        json.dumps(
            {"repo": REPO, "revision": info.sha, "stored_size": STORED, "counts": counts,
             "n_classes": 1000, "workers": args.workers,
             "resized_at_prepare": resized_counts,
             "preprocess": "passthrough when already 128x128, else resize shorter side then "
                           "center crop, uint8 CHW"},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {out} — {counts}, resized {resized_counts}", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)          # same fsspec non-daemon-thread reason as prepare_imagenet100.py


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests**

Run: `cd ~/Dev/hallm && uv run --group data --group analysis pytest tests/test_prepare_imagenet1k.py -q`
Expected: PASS (4 tests)

- [ ] **Step 5: Run the timing slice (spec §4 R3 — this is a gate, not a smoke test)**

Run:
```bash
cd ~/Dev/hallm && time uv run --group data --group analysis python scripts/prepare_imagenet1k.py \
  --out /tmp/in1k-probe --limit 10000 --workers 8 --ipv4
```
Expected: `/tmp/in1k-probe/train.bin` at exactly `10000 * 49152` bytes, `SOURCE.json` written, and a wall-clock number. Extrapolate to 1,281,167 images. **If the extrapolation exceeds ~3 h, stop and report the number instead of starting the full run** (spec §12 R3).

- [ ] **Step 6: Verify the slice decodes to real images**

```bash
cd ~/Dev/hallm && uv run --group analysis python -c "
import numpy as np, json
a = np.memmap('/tmp/in1k-probe/train.bin', dtype=np.uint8, mode='r').reshape(-1,3,128,128)
l = np.load('/tmp/in1k-probe/train_labels.npy')
print('images', a.shape, 'labels', l.shape, 'label range', l.min(), l.max())
print('per-image mean spread', float(a[:64].reshape(64,-1).mean(1).std()))
print(json.load(open('/tmp/in1k-probe/SOURCE.json')))
"
```
Expected: shape `(10000, 3, 128, 128)`; a non-zero mean spread (uniform zeros would mean the offset writes went wrong); `resized_at_prepare` recorded. **Record whether it is 0** — that answers whether the mirror is exactly 128×128, which the design left open.

- [ ] **Step 7: Commit**

```bash
git add scripts/prepare_imagenet1k.py tests/test_prepare_imagenet1k.py
git commit -m "feat(data): parallel ImageNet-1k prepare writing a preallocated uint8 memmap

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: Report with both passes, pass counts, and the overfitting flag

**Files:**
- Modify: `scripts/build_vision_report.py`
- Test: `tests/test_vision_in1k.py`

**Interfaces:**
- Produces: `overfit_flag(top1: float, top1_best: float | None, threshold: float = 0.01) -> str` returning `"still overfitting"` or `""`; the report groups rows by `corpus`.

- [ ] **Step 1: Read the current report builder**

Run: `cd ~/Dev/hallm && cat scripts/build_vision_report.py`
Note its row-loading and table-emitting functions; the next steps extend them rather than replacing them.

- [ ] **Step 2: Write the failing test**

```python
# append to tests/test_vision_in1k.py
def test_overfit_flag_fires_only_above_one_point():
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location(
        "bvr", Path("scripts/build_vision_report.py"))
    bvr = importlib.util.module_from_spec(spec); sys.modules["bvr"] = bvr
    spec.loader.exec_module(bvr)
    assert bvr.overfit_flag(0.4696, 0.4806) == "still overfitting"   # 1.10 points
    assert bvr.overfit_flag(0.4814, 0.4842) == ""                    # 0.28 points
    assert bvr.overfit_flag(0.4814, None) == ""                      # no periodic evals recorded
    assert bvr.overfit_flag(0.4814, 0.4714) == ""                    # final above best: no flag
```

- [ ] **Step 3: Run to verify it fails**

Run: `cd ~/Dev/hallm && uv run pytest tests/test_vision_in1k.py::test_overfit_flag_fires_only_above_one_point -q`
Expected: FAIL with `AttributeError: module 'bvr' has no attribute 'overfit_flag'`

- [ ] **Step 4: Implement**

```python
# scripts/build_vision_report.py, near the other helpers
def overfit_flag(top1: float, top1_best: float | None, threshold: float = 0.01) -> str:
    """Spec 2026-09-18 §7: an arm whose best periodic top-1 exceeds its final by more than
    `threshold` (1.0 point) is disclosed as still overfitting rather than reported at its peak."""
    if top1_best is None:
        return ""
    return "still overfitting" if (top1_best - top1) > threshold else ""
```

Then, in the table-emitting function, add columns `corpus`, `passes`, `top1_best`, `best_step`, `flag` (from `overfit_flag`), and group rows into one block per distinct `corpus` value, each block preceded by:

```python
    lines.append(f"### {corpus} — {n_classes}-way, {passes} passes over the corpus")
    lines.append("")
    lines.append("> 1 seed (1337), descriptive. Absolute accuracies are NOT comparable across "
                 "blocks: the blocks differ in corpus, class count and pass count. Only arm gaps "
                 "within a block are claimed.")
```

Rows missing `corpus` (the first pass's results, written before Task 1) fall back to `"imagenet-100"`, and rows missing `passes` print `—`.

- [ ] **Step 5: Run the test**

Run: `cd ~/Dev/hallm && uv run pytest tests/test_vision_in1k.py -q`
Expected: PASS

- [ ] **Step 6: Rebuild the existing report and check it did not lose the first pass**

Run: `cd ~/Dev/hallm && uv run python scripts/build_vision_report.py && head -40 results/reports/vision.md && git diff --stat results/reports/vision.md`
Expected: the five ImageNet-100 arms still present with their original `top1` values (48.14 / 46.96 / 47.32 / 47.36 / 47.28), now under a labelled block.

- [ ] **Step 7: Run the full suite and commit**

```bash
cd ~/Dev/hallm && uv run pytest -q
git add scripts/build_vision_report.py results/reports/vision.md tests/test_vision_in1k.py
git commit -m "feat(report): per-corpus blocks, pass counts and the overfitting flag

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: The five ImageNet-1k configs and the staged queues

**Files:**
- Create: `configs/runs/V4-A0-s1337-in1k.yaml`, `V8-A0-s1337-in1k.yaml`, `V8-A2-s1337-in1k.yaml`, `V8-A1u4-s1337-in1k.yaml`, `V8-A1u4t-s1337-in1k.yaml`
- Create: `configs/runs/queue-vision-in1k-gate.txt`, `configs/runs/queue-vision-in1k-sharing.txt`
- Test: `tests/test_vision_in1k.py`

**Interfaces:**
- Consumes: `load_vision_experiment` from `hallm.experiment`, `VSHAPES["v4-in1k"]` / `["v8-in1k"]` from Task 1, `VisionTrainConfig.eval_subsample` from Task 2.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_vision_in1k.py
from hallm.experiment import load_vision_experiment

IN1K_CONFIGS = ["V4-A0-s1337-in1k", "V8-A0-s1337-in1k", "V8-A2-s1337-in1k",
                "V8-A1u4-s1337-in1k", "V8-A1u4t-s1337-in1k"]


@pytest.mark.parametrize("name", IN1K_CONFIGS)
def test_in1k_configs_match_the_first_pass_except_corpus_and_classes(name):
    model_cfg, train_cfg = load_vision_experiment(f"configs/runs/{name}.yaml")
    base_name = name.replace("-in1k", "")
    base_model, base_train = load_vision_experiment(f"configs/runs/{base_name}.yaml")
    assert model_cfg.n_classes == 1000 and base_model.n_classes == 100
    assert replace(model_cfg, n_classes=100) == base_model        # arms identical otherwise
    for field in ("lr", "min_lr", "warmup_steps", "max_steps", "weight_decay", "grad_clip",
                  "batch_size", "grad_accum", "crop_size", "label_smoothing", "dtype",
                  "deterministic", "seed"):
        assert getattr(train_cfg, field) == getattr(base_train, field), field
    assert train_cfg.dataset == "imagenet-1k"
    assert train_cfg.eval_subsample == 10                          # 10 per class = 10,000 images
    assert train_cfg.out_dir == f"runs/vision/{name}"
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd ~/Dev/hallm && uv run pytest tests/test_vision_in1k.py -q -k in1k_configs`
Expected: FAIL — the config files do not exist.

- [ ] **Step 3: Write the configs**

Generate all five from their first-pass counterparts so nothing can drift by hand:

```bash
cd ~/Dev/hallm
for n in V4-A0 V8-A0 V8-A2 V8-A1u4 V8-A1u4t; do
  sed -e 's/^vshape: v4$/vshape: v4-in1k/' \
      -e 's/^vshape: v8$/vshape: v8-in1k/' \
      -e "s|^  out_dir: runs/vision/${n}-s1337$|  out_dir: runs/vision/${n}-s1337-in1k|" \
      configs/runs/${n}-s1337.yaml > configs/runs/${n}-s1337-in1k.yaml
  printf '  dataset: imagenet-1k\n  eval_subsample: 10\n' >> configs/runs/${n}-s1337-in1k.yaml
done
cat configs/runs/V8-A1u4t-s1337-in1k.yaml
```

Verify by eye that `dataset` and `eval_subsample` landed inside the `train:` block (they are appended at the end of the file, and `train:` is the last block in these configs — check the indentation is two spaces).

- [ ] **Step 4: Write the queue files**

```bash
cd ~/Dev/hallm
printf 'configs/runs/V4-A0-s1337-in1k.yaml\nconfigs/runs/V8-A0-s1337-in1k.yaml\n' \
  > configs/runs/queue-vision-in1k-gate.txt
printf 'configs/runs/V8-A2-s1337-in1k.yaml\nconfigs/runs/V8-A1u4-s1337-in1k.yaml\nconfigs/runs/V8-A1u4t-s1337-in1k.yaml\n' \
  > configs/runs/queue-vision-in1k-sharing.txt
```

Two files, not one: the sharing queue must not be launchable by accident before the depth gate is read (spec §5).

- [ ] **Step 5: Run the test**

Run: `cd ~/Dev/hallm && uv run pytest tests/test_vision_in1k.py -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add configs/runs/V*-in1k.yaml configs/runs/queue-vision-in1k-*.txt tests/test_vision_in1k.py
git commit -m "feat(configs): five ImageNet-1k arms and the two staged queues

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 6: Prepare the corpus and measure read throughput

Not a code task — the operational stage 1. It runs only if the §3 probe passed.

- [ ] **Step 1: Run the full prepare**

```bash
cd ~/Dev/hallm && nohup systemd-inhibit --what=sleep:idle --who=hallm --why=prep1k \
  sh -c 'uv run --group data --group analysis python scripts/prepare_imagenet1k.py \
  --out data/in1k --workers 8 --ipv4 >> prep-in1k.log 2>&1' &
```

- [ ] **Step 2: Verify the output**

```bash
cd ~/Dev/hallm && ls -la data/in1k/ && cat data/in1k/SOURCE.json && uv run python -c "
import numpy as np, json
s = json.load(open('data/in1k/SOURCE.json'))
for split, n in (('train', s['counts']['train']), ('val', s['counts']['validation'])):
    a = np.memmap(f'data/in1k/{split}.bin', dtype=np.uint8, mode='r')
    assert a.size == n * 3 * 128 * 128, (split, a.size, n)
    l = np.load(f'data/in1k/{split}_labels.npy')
    assert len(l) == n and l.min() == 0 and l.max() == 999, (split, len(l), l.min(), l.max())
    print(split, n, 'ok')
"
```
Expected: train 1,281,167 and val 50,000, labels spanning 0–999, byte counts exact.

- [ ] **Step 3: Measure random-read throughput (spec §12 R4)**

```bash
cd ~/Dev/hallm && uv run python -c "
import numpy as np, time
from hallm.data.imagenet import load_images
im = load_images('data/in1k/train.bin')
rng = np.random.default_rng(0)
t = time.time(); n = 20
for _ in range(n):
    idx = np.sort(rng.integers(0, len(im), 256))
    _ = np.ascontiguousarray(im[idx])
dt = (time.time() - t) / n
print(f'{dt*1000:.1f} ms per 256-image batch -> {256*49152/dt/1e6:.0f} MB/s, {1/dt:.1f} batches/s')
"
```
Expected: comfortably above 13.0 batches/s (the measured training rate), so the loader is not the bottleneck. **If it is below 13.0, report the number and stop** — the queue would be I/O-bound and the timing estimates in the spec would be wrong.

- [ ] **Step 4: Commit the SOURCE record**

```bash
cd ~/Dev/hallm && git add -f data/in1k/SOURCE.json && git commit -m "data: ImageNet-1k prepared — 1,281,167 train / 50,000 val at 128px

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

(`data/` is gitignored; `-f` records only the provenance file, never the memmaps.)

---

### Task 7: Stage 2 — the gate pair, and the gate decision

- [ ] **Step 1: Launch the gate queue**

```bash
cd ~/Dev/hallm && nohup systemd-inhibit --what=sleep:idle --who=hallm --why=in1k-gate \
  sh -c 'uv run python scripts/run_vision_queue.py --queue configs/runs/queue-vision-in1k-gate.txt \
  --data data/in1k --results-dir results/runs >> runs-in1k-gate.log 2>&1' &
```
Expected: ~1.6 h for both arms.

- [ ] **Step 2: Read the gate**

```bash
cd ~/Dev/hallm && uv run python -c "
import json
r = {n: json.load(open(f'results/runs/{n}.json')) for n in
     ('V4-A0-s1337-in1k', 'V8-A0-s1337-in1k')}
v4, v8 = r['V4-A0-s1337-in1k']['top1'], r['V8-A0-s1337-in1k']['top1']
print(f'V4-A0 {v4:.4f}  V8-A0 {v8:.4f}  gap {100*(v8-v4):+.2f} points')
print('DEPTH GATE:', 'PASS' if v8 > v4 else 'FAIL')
for n, row in r.items():
    print(n, 'best', row['top1_best'], '@', row['best_step'], 'passes', row['passes'])
"
```

- [ ] **Step 3: Act on it**

- **PASS** → commit both runs and proceed to Task 8.
- **FAIL** → **stop.** Do not launch the sharing queue. Per spec §6, write `docs/analysis/2026-09-18-vit-pass-matched-outcome.md` reporting depth-doesn't-pay, score predictions 1–6, and stop there. Also check the chance floor: a top-1 near 0.1% means under-training (spec §12 R1), which is a different finding from a clean gate failure and must be reported as such.

- [ ] **Step 4: Commit the landed runs**

```bash
cd ~/Dev/hallm && uv run python scripts/build_vision_report.py
git add results/runs/V4-A0-s1337-in1k.json results/runs/V8-A0-s1337-in1k.json \
        results/manifests/ results/reports/vision.md
git commit -m "results: ImageNet-1k gate pair — depth gate <PASS|FAIL>

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 8: Stage 3 — the three sharing arms

Runs only on a passed depth gate.

- [ ] **Step 1: Launch**

```bash
cd ~/Dev/hallm && nohup systemd-inhibit --what=sleep:idle --who=hallm --why=in1k-sharing \
  sh -c 'uv run python scripts/run_vision_queue.py --queue configs/runs/queue-vision-in1k-sharing.txt \
  --data data/in1k --results-dir results/runs >> runs-in1k-sharing.log 2>&1' &
```
Expected: ~3.2 h for three arms.

- [ ] **Step 2: Check the follow-up trigger (spec §6)**

```bash
cd ~/Dev/hallm && uv run python -c "
import json
g = lambda n: json.load(open(f'results/runs/{n}.json'))['top1']
gap = 100 * (g('V8-A2-s1337-in1k') - g('V4-A0-s1337-in1k'))
print(f'V8-A2 vs V4-A0: {gap:+.2f} points')
print('TRIGGER:', 'FIRES — seeds 1338/1339 for V8-A2, V4-A0 and V8-A1u4' if abs(gap) >= 1.0
      else 'does not fire — report as no difference detected at one seed')
"
```
If it fires, report it to Hazar with the cost (~4.8 h for six more runs) rather than launching them unattended — that is new GPU time beyond the plan he approved.

- [ ] **Step 3: Commit each landed run with the rebuilt report**

```bash
cd ~/Dev/hallm && uv run python scripts/build_vision_report.py
git add results/runs/V8-A2-s1337-in1k.json results/runs/V8-A1u4-s1337-in1k.json \
        results/runs/V8-A1u4t-s1337-in1k.json results/manifests/ results/reports/vision.md
git commit -m "results: ImageNet-1k sharing arms landed

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 9: Symmetry re-measurement

- [ ] **Step 1: Run the existing measurement over the new checkpoints**

```bash
cd ~/Dev/hallm && uv run --group analysis python scripts/ffn_symmetry.py --help
```
Read its arguments, then run it over the five `runs/vision/V*-s1337-in1k/*.pt` checkpoints with the ImageNet-1k validation split as the input source, appending to `results/analysis/ffn-symmetry.json`. **Do not truncate existing rows** — the first pass's session lost 20 rows to exactly this and it was caught in review; confirm the row count before and after.

```bash
cd ~/Dev/hallm && uv run python -c "
import json; print('rows before:', len(json.load(open('results/analysis/ffn-symmetry.json'))))"
```

- [ ] **Step 2: Check sanity gate 3**

`V8-A2-s1337-in1k`'s rotation share in the `u` basis must be 0.0000 to float precision. If it is not, the measurement or the arm is wrong, and nothing else in the symmetry table should be believed.

- [ ] **Step 3: Check prediction 6**

The `-in1k` ViTs' rotation share should land within ±0.03 of the first pass's 0.416–0.422. Record the actual band whether or not it does.

- [ ] **Step 4: Commit**

```bash
cd ~/Dev/hallm && git add results/analysis/ffn-symmetry.json results/reports/ffn-symmetry.md
git commit -m "results: FFN symmetry over the ImageNet-1k ViTs

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 10: Outcome document

- [ ] **Step 1: Write `docs/analysis/2026-09-18-vit-pass-matched-outcome.md`**

Follow the shape of `docs/analysis/2026-09-17-controlled-vit-outcome.md`: one-line summary, the arm table, the gate verdict, the diagnostic contrast against the LM ladder, the symmetry table, **predictions 1–6 scored one by one including any that fail**, what the pass does and does not license, and what is open.

Mandatory contents:
- The probe's numbers and whether prediction 5 held in both halves.
- The 101-pass vs 10-pass contrast as a result about data reuse.
- If the trigger in Task 8 fired but the seeds were not run, say so explicitly rather than reporting one seed as if it settled the comparison.

- [ ] **Step 2: Commit**

```bash
cd ~/Dev/hallm && git add docs/analysis/2026-09-18-vit-pass-matched-outcome.md
git commit -m "docs(analysis): pass-matched ViT outcome, predictions scored

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage.** §1 → Task 10's contrast section. §2 → Tasks 1 and 5. §3 probe → already running before this plan was written; its verdict gates Task 6. §4 data → Tasks 3 and 6. §5 staging → Tasks 6–8, two queue files. §6 rules and gates → Task 7 step 3, Task 8 step 2, Task 9 step 2. §7 metrics → Tasks 1, 2, 4. §8 symmetry → Task 9. §9 predictions → Task 10. §10 implementation → the file table matches. §11 artifacts → Tasks 7–10. §12 risks: R1 Task 7 step 3, R2 Task 10, R3 Task 3 step 5, R4 Task 6 step 3, R5 Task 8 step 2, R6 noted in Global Constraints.

**Type consistency.** `_result_row` gains three keyword-only-by-convention parameters with defaults, so the existing two-positional call sites keep working; `stratified_subsample(labels, per_class, seed)` is called with keywords in tests and positionally in `_split_for_periodic_eval`, matching its signature; `overfit_flag(top1, top1_best, threshold=0.01)` takes fractions, not points, and the test asserts the 1.0-point boundary in fractions (0.01).

**Known gap, deliberate.** Task 4 step 4 describes the table-emitting change in prose rather than a literal diff, because `build_vision_report.py`'s current structure has to be read first (step 1). Its test is exact, and the acceptance check in step 6 is exact.
