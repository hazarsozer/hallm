"""Build data/in100/{train,val}.bin from clane9/imagenet-100 (spec 2026-09-16 §4).

One-time, ~26 GB of parquet downloaded to the HF cache, ~6.6 GB written here.

The script imports both `datasets` (dependency group `data`) and PIL (dependency group
`analysis`, via `preprocess_image`), so every invocation needs both groups:

Usage:
  uv run --group data --group analysis python scripts/prepare_imagenet100.py --out data/in100
  uv run --group data --group analysis python scripts/prepare_imagenet100.py --out /tmp/in100-probe --limit 64   # smoke check
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
