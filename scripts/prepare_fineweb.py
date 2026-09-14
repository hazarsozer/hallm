"""Download FineWeb-Edu sample-10BT shards and write data/fineweb/{train,val,test}.bin.

Train: shards in name order (all but the last) until --train-tokens. Val and test: the LAST shard,
never used for training — val is its first --heldout-tokens, test the next --heldout-tokens.
Writes data/fineweb/SOURCE.json (shards used, token counts, sha256 per bin) for provenance.

Usage: uv run --group data python scripts/prepare_fineweb.py --out data/fineweb --train-tokens 2600000000
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from hallm.data.fineweb import write_until
from hallm.manifest import file_sha256

REPO = "HuggingFaceFW/fineweb-edu"
PREFIX = "sample/10BT/"


def main() -> None:
    import pyarrow.parquet as pq
    import tiktoken
    from huggingface_hub import hf_hub_download, list_repo_files

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/fineweb")
    ap.add_argument("--train-tokens", type=int, default=2_600_000_000)
    ap.add_argument("--heldout-tokens", type=int, default=5_000_000)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    enc = tiktoken.get_encoding("gpt2")
    threads = os.cpu_count() or 4

    def encode_batch(texts: list[str]) -> list[list[int]]:
        return enc.encode_ordinary_batch(texts, num_threads=threads)

    shards = sorted(f for f in list_repo_files(REPO, repo_type="dataset")
                    if f.startswith(PREFIX) and f.endswith(".parquet"))
    train_shards, heldout = shards[:-1], shards[-1]
    used: list[str] = []

    def batches(files: list[str]):
        for name in files:
            used.append(name)
            path = hf_hub_download(REPO, name, repo_type="dataset")
            for rb in pq.ParquetFile(path).iter_batches(batch_size=1024, columns=["text"]):
                yield rb.column(0).to_pylist()

    counts = {"train.bin": write_until(batches(train_shards), encode_batch, out / "train.bin", args.train_tokens)}
    held = batches([heldout])
    counts["val.bin"] = write_until(held, encode_batch, out / "val.bin", args.heldout_tokens)
    counts["test.bin"] = write_until(held, encode_batch, out / "test.bin", args.heldout_tokens)

    source = {"repo": REPO, "shards_read": sorted(set(used)), "heldout_shard": heldout,
              "tokens": counts, "sha256": {k: file_sha256(out / k) for k in counts},
              "tokenizer": "gpt2", "separator": "EOT 50256 after every document"}
    (out / "SOURCE.json").write_text(json.dumps(source, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
