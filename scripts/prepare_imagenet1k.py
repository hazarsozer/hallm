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
# Rows pulled from parquet at a time. Bounds each worker's Arrow footprint; see _decode_shard.
_BATCH_ROWS = 512


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
    """Worker: decode one shard into `bin_path` at `offset`. Returns (shard_index, labels, n_resized).

    Two properties matter as much as correctness here, both learned the hard way on 2026-09-18 when an
    earlier version of this function put a 30 GB desktop into swap mid-conversion:

    1. **Peak memory must not scale with shard size or worker count.** `ParquetFile.read()` materialises
       an entire 438 MB shard as an Arrow table (~1.4 GB resident once image bytes are decoded); eight
       of those is ~11 GB. `iter_batches` streams row groups instead — these files hold 986 groups of
       ~99 rows, so a worker's Arrow footprint stays in the low megabytes no matter how big the shard.
    2. **The write stream must not evict everyone else's page cache.** Writing 63 GB through the page
       cache pushes out whatever the user is running. After each flush we fsync just-written bytes and
       `POSIX_FADV_DONTNEED` them: the pages are on disk and nothing here will read them again, so
       dropping them costs this script nothing and leaves the cache to its owner.
    """
    shard_index, path, bin_path, offset, limit = args
    import pyarrow.parquet as pq
    from PIL import Image

    labels, n_resized, written, done = [], 0, 0, 0
    pf = pq.ParquetFile(path)
    with open(bin_path, "r+b") as f:
        f.seek(offset)
        fd = f.fileno()
        for batch in pf.iter_batches(batch_size=_BATCH_ROWS, columns=["image", "label"]):
            if limit is not None and done >= limit:
                break
            img_col, lab_col = batch.column("image"), batch.column("label")
            for i in range(batch.num_rows):
                if limit is not None and done >= limit:
                    break
                cell = img_col[i].as_py()
                raw = cell["bytes"] if isinstance(cell, dict) else cell
                img = Image.open(io.BytesIO(raw)) if isinstance(raw, (bytes, bytearray)) else raw
                img = img.convert("RGB")
                if img.size != (STORED, STORED):
                    n_resized += 1
                f.write(decode_row(img).tobytes())
                labels.append(int(lab_col[i].as_py()))
                done += 1
            f.flush()
            start = offset + written
            written = done * BYTES_PER_IMAGE
            os.fsync(fd)
            os.posix_fadvise(fd, start, written - (start - offset), os.POSIX_FADV_DONTNEED)
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
    ap.add_argument("--workers", type=int, default=2,
                    help="Parallel shard decoders. Memory is now flat per worker (~200 MB), but each "
                         "still drives CPU and write bandwidth; 2 keeps the machine usable while the "
                         "conversion runs in the background.")
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
