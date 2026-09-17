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


def _make_shard(path: Path, labels: list[int]) -> None:
    """Write a tiny parquet shard whose image bytes encode the row's own label (so a test can
    verify, after decode, which image ended up at which output offset) alongside a distinct
    per-row label."""
    import io

    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = []
    for lab in labels:
        arr = np.full((128, 128, 3), lab % 256, dtype=np.uint8)
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        rows.append({"image": {"bytes": buf.getvalue(), "path": None}, "label": lab})
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)


def test_convert_reassembles_labels_in_shard_order_despite_parallel_out_of_order_completion(tmp_path):
    """Regression guard for the single most damaging possible bug here (spec 2026-09-18 §4):
    shards are decoded by separate worker processes that may finish in any order, each writing
    image bytes at its own byte offset and returning its own labels — the parent must reassemble
    labels by shard index, not by completion order, or every label in the corpus is misaligned
    against its image."""
    # Shard 0 is the largest so, with a worker pool smaller than the shard count, it is very
    # unlikely to be the first job to *finish* even though it is submitted first — this exercises
    # the out-of-order-completion path, not just simultaneous dispatch.
    shard_labels = [[100, 101, 102, 103, 104], [10, 11], [55]]
    files = []
    for i, labs in enumerate(shard_labels):
        p = tmp_path / f"shard-{i}.parquet"
        _make_shard(p, labs)
        files.append(p)

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    # `convert` hands work to a plain `ProcessPoolExecutor()`, which — on this Python (3.14) —
    # defaults to the "forkserver" start method. That server process is a fresh interpreter that
    # only knows how to resurrect a real `__main__` script by re-running it from `sys.argv[0]`; it
    # has no way to import this test's dynamically-aliased "prep1k" module (see module-level
    # importlib dance above), so pickling `_decode_shard` (whose `__module__` is "prep1k") into a
    # forkserver worker fails with `ModuleNotFoundError`. That failure is purely an artifact of
    # this test harness's dynamic import, not of the script: run normally as
    # `python scripts/prepare_imagenet1k.py`, the module *is* `__main__`, which multiprocessing's
    # spawn/forkserver bootstrap always knows how to re-import. Forcing "fork" here for the
    # duration of this one test sidesteps the harness artifact: a forked worker inherits this
    # process's already-populated `sys.modules["prep1k"]` directly via copy-on-write, no import
    # needed.
    import multiprocessing

    multiprocessing.set_start_method("fork", force=True)
    total, _resized = prep1k.convert("train", files, out_dir, limit=None, workers=2)

    expected_labels = [lab for labs in shard_labels for lab in labs]
    assert total == len(expected_labels)

    got_labels = np.load(out_dir / "train_labels.npy")
    assert got_labels.tolist() == expected_labels

    per = 3 * 128 * 128
    images = np.memmap(out_dir / "train.bin", dtype=np.uint8, mode="r")
    assert images.size == total * per
    images = images.reshape(total, 3, 128, 128)
    # Each image's pixel value was set to its own label (mod 256) at fixture-build time, so the
    # image at output row i must carry expected_labels[i] — this is what actually catches a
    # shard-order/label-order mismatch, as opposed to merely checking the label array's length.
    for i, lab in enumerate(expected_labels):
        assert int(images[i, 0, 0, 0]) == lab % 256
