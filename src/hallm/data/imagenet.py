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
        # One random crop offset per BATCH, not per sample (deliberate — see spec 2026-09-16 §4 /
        # task-3-brief decision 3). Over ~98 epochs each image lands in ~98 different batches and
        # so sees ~98 different offsets, so per-image augmentation diversity is essentially
        # unchanged; only within-batch offset correlation differs, and that correlation is
        # identical across all five arms, which is what the controlled comparison needs. Do not
        # "fix" this to per-sample without checking that reasoning first.
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
