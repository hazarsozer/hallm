"""Build data/in100/{train,val}.bin from clane9/imagenet-100 (spec 2026-09-16 §4).

One-time, ~26 GB of parquet downloaded to the HF cache, ~6.6 GB written here.

The script imports both `datasets` (dependency group `data`) and PIL (dependency group
`analysis`, via `preprocess_image`), so every invocation needs both groups:

Usage:
  uv run --group data --group analysis python scripts/prepare_imagenet100.py --out data/in100
  uv run --group data --group analysis python scripts/prepare_imagenet100.py --out /tmp/in100-probe --limit 64   # smoke check

`--ipv4` is an opt-in workaround for hosts whose IPv6 route to the HF CDN blackholes — symptom is a
multi-minute stall with a socket parked in SYN-SENT to a CDN IPv6 address. Off by default: other
machines' IPv6 may work fine, so this only applies to a process that actually hits the fault.
"""

from __future__ import annotations

import argparse
import io
import json
import socket
from pathlib import Path

import numpy as np

from hallm.data.imagenet import STORED, preprocess_image

REPO = "clane9/imagenet-100"


def _force_ipv4() -> None:
    """Monkeypatch `socket.getaddrinfo` to resolve IPv4 addresses only, for this process.

    Workaround for a machine where the IPv6 path to the HF CDN is blackholed: glibc's resolver
    prefers IPv6 (RFC 6724), the SYN to the CDN's IPv6 address gets no response, and only after a
    multi-minute connect timeout does it fall back to IPv4 — every request pays that stall. Forcing
    `family=AF_INET` up front skips the IPv6 attempt entirely. Opt-in via `--ipv4`; never applied
    unconditionally, since another machine's IPv6 route may be healthy.
    """
    _getaddrinfo = socket.getaddrinfo

    def _ipv4_only(host, port, family=0, type=0, proto=0, flags=0):
        return _getaddrinfo(host, port, socket.AF_INET, type, proto, flags)

    socket.getaddrinfo = _ipv4_only


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
    ap.add_argument("--ipv4", action="store_true",
                     help="Force IPv4-only DNS resolution (workaround for a blackholed IPv6 route "
                          "to the HF CDN; see module docstring). Off by default.")
    args = ap.parse_args()

    if args.ipv4:
        _force_ipv4()

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
