from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from hallm.model.config import VSHAPES


def test_in1k_shapes_differ_from_base_only_in_class_count():
    for base, derived in (("v4", "v4-in1k"), ("v8", "v8-in1k")):
        assert VSHAPES[derived].n_classes == 1000
        assert VSHAPES[base].n_classes == 100
        # every other field identical — this is the whole point of deriving them
        assert replace(VSHAPES[derived], n_classes=100) == VSHAPES[base]


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
    from hallm.model.vit import ViT
    from hallm.vision_runqueue import _result_row
    from hallm.vision_train import VisionTrainConfig

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
    from hallm.model.vit import ViT
    from hallm.vision_runqueue import _result_row
    from hallm.vision_train import VisionTrainConfig

    cfg = VSHAPES["v4"]
    model = ViT(cfg)
    tcfg = VisionTrainConfig(max_steps=50000, batch_size=256)
    row = _result_row(model, cfg, tcfg, "V4-A0-s1337", 0.4814, 2.5436)
    assert row["top1_best"] is None and row["best_step"] is None and row["passes"] is None
    assert row["corpus"] == "imagenet-100"
    assert row["top1"] == 0.4814


def test_stratified_subsample_is_balanced_deterministic_and_sorted():
    from hallm.data.imagenet import stratified_subsample

    labels = np.repeat(np.arange(100), 50)          # 5000 images, 50 per class
    idx = stratified_subsample(labels, per_class=10, seed=1337)
    assert len(idx) == 1000
    assert np.array_equal(idx, np.sort(idx))        # sorted => contiguous memmap reads
    counts = np.bincount(labels[idx], minlength=100)
    assert set(counts.tolist()) == {10}
    assert np.array_equal(idx, stratified_subsample(labels, per_class=10, seed=1337))
    assert not np.array_equal(idx, stratified_subsample(labels, per_class=10, seed=1338))


def test_stratified_subsample_takes_all_when_class_is_smaller_than_quota():
    from hallm.data.imagenet import stratified_subsample

    labels = np.array([0, 0, 1, 1, 1, 2])
    idx = stratified_subsample(labels, per_class=2, seed=1337)
    counts = np.bincount(labels[idx], minlength=3)
    assert counts.tolist() == [2, 2, 1]


def test_eval_subsample_defaults_to_whole_split():
    from hallm.vision_train import VisionTrainConfig
    assert VisionTrainConfig().eval_subsample == 0


def test_run_one_vision_uses_subsample_for_periodic_and_full_split_for_final():
    """The periodic eval sees 2 images per class; the final number sees all 6."""
    import hallm.vision_runqueue as vrq

    labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    n = vrq._split_for_periodic_eval(np.zeros((6, 3, 128, 128), np.uint8), labels, 2, 1337)
    assert len(n[0]) == 4 and len(n[1]) == 4
    assert np.bincount(n[1]).tolist() == [2, 2]


IN100 = Path("data/in100")

# the recorded first-batch labels for seed 1337, produced on the pre-change commit — a canary on
# the ImageNet-100 data order itself (spec 2026-09-18 §7 — this pass must not move it)
RECORDED_IN100_SEED1337_LABELS = [6, 89, 65, 7, 95, 33, 60, 68]


@pytest.mark.skipif(not (IN100 / "train.bin").exists(), reason="ImageNet-100 memmaps not present")
def test_in100_batches_unchanged_by_this_pass():
    """Same seed, same loader, same bytes — the first pass's data order must not move."""
    import torch

    from hallm.data.imagenet import get_image_batch, load_images, load_labels

    images, labels = load_images(IN100 / "train.bin"), load_labels(IN100 / "train_labels.npy")
    g1 = torch.Generator().manual_seed(1337)
    x1, y1 = get_image_batch(images, labels, 8, 112, "cpu", g1, train=True)
    g2 = torch.Generator().manual_seed(1337)
    x2, y2 = get_image_batch(images, labels, 8, 112, "cpu", g2, train=True)
    assert torch.equal(x1, x2) and torch.equal(y1, y2)
    assert x1.shape == (8, 3, 112, 112)
    assert y1.tolist() == RECORDED_IN100_SEED1337_LABELS


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


def _write_toy_memmap(tmp_path, seed: int, n: int):
    from hallm.data.imagenet import STORED, load_images

    path = tmp_path / f"toy_{seed}.bin"
    data = np.random.default_rng(seed).integers(0, 256, size=(n, 3, STORED, STORED), dtype=np.uint8)
    data.tofile(path)
    return data, load_images(path)


def test_prefetch_reads_identical_bytes_to_plain_read(tmp_path):
    """madvise(WILLNEED) is a caching hint only (see the WHY comment on `_prefetch_random_reads`
    for the 2.9 -> 26.8 batches/s measurement it's there for). Build a real on-disk memmap (a
    genuine mmap.mmap underneath, not an in-RAM array), issue the same prefetch the training path
    uses, then fancy-index it — the bytes must match indexing a plain in-RAM copy with no
    prefetch at all."""
    from hallm.data.imagenet import _prefetch_random_reads

    data, images = _write_toy_memmap(tmp_path, seed=0, n=40)
    assert isinstance(images._mmap, __import__("mmap").mmap)   # exercising the real branch

    idx = np.array([37, 1, 20, 5, 5, 39, 0, 18])
    _prefetch_random_reads(images, idx)          # must not raise, must not touch the bytes
    got = np.ascontiguousarray(images[idx])
    assert np.array_equal(got, data[idx])


def test_prefetch_is_a_noop_when_nothing_to_madvise(tmp_path):
    """Non-mmap-backed arrays must never raise: a plain np.ndarray (most callers/tests), the
    in-RAM eval subsample `_split_for_periodic_eval` produces, and a memmap slice (still
    mmap-backed, included here to confirm the slice path is also handled without error)."""
    from hallm.data.imagenet import _prefetch_random_reads

    plain = np.zeros((10, 3, 128, 128), dtype=np.uint8)
    _prefetch_random_reads(plain, np.array([0, 1, 2]))              # no _mmap attribute at all

    _, images = _write_toy_memmap(tmp_path, seed=1, n=20)
    sl = images[2:8]
    _prefetch_random_reads(sl, np.array([0, 1]))                    # slice: has _mmap, must not raise

    copy = np.ascontiguousarray(images[[0, 1]])
    _prefetch_random_reads(copy, np.array([0]))                     # fancy-index copy: no _mmap


def test_get_image_batch_train_path_unaffected_by_prefetch(tmp_path):
    """Wiring `_prefetch_random_reads` into the training path must not change what comes out:
    same seed -> same batch, deterministically, same as the in100-corpus canary above checks on
    real data."""
    import torch

    from hallm.data.imagenet import get_image_batch

    _, images = _write_toy_memmap(tmp_path, seed=2, n=40)
    labels = np.arange(40, dtype=np.int64)

    g1 = torch.Generator().manual_seed(42)
    x1, y1 = get_image_batch(images, labels, 8, 112, "cpu", g1, train=True)
    g2 = torch.Generator().manual_seed(42)
    x2, y2 = get_image_batch(images, labels, 8, 112, "cpu", g2, train=True)
    assert torch.equal(x1, x2) and torch.equal(y1, y2)


def _load_bvr():
    """Load scripts/build_vision_report.py as a module (mirrors the dance the overfit_flag test
    above already uses — the script is not a package member, so it has to be loaded by path)."""
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        "bvr", Path("scripts/build_vision_report.py"))
    bvr = importlib.util.module_from_spec(spec)
    sys.modules["bvr"] = bvr
    spec.loader.exec_module(bvr)
    return bvr


def test_two_corpus_passes_for_same_arm_both_survive_into_their_own_blocks():
    """Critical regression test (review finding, 2026-09-18): commit 7f560c3 added per-corpus
    blocks to vision_report(), but `_by_arm` — untouched by that commit — (1) filters rows with
    `run.endswith(f"-s{seed}")`, which drops every `-in1k` row outright since real in1k run ids
    end in `-in1k`, not `-s1337`, and (2) even once that filter is relaxed, keys `seen` by
    `run.rsplit("-s", 1)[0]`, which maps BOTH `V4-A0-s1337` and `V4-A0-s1337-in1k` to the same key
    `"V4-A0"` — so the dict structurally cannot hold both corpora's row for one arm at once.

    This pushes realistically-shaped rows for both corpora, same arm (V4-A0), through the real
    `vision_report()` path and asserts both land in their own block rather than one clobbering the
    other."""
    bvr = _load_bvr()

    in100_row = {
        "run": "V4-A0-s1337", "arm": "A0", "dataset": "imagenet-100", "n_classes": 100,
        "n_layer": 4, "non_embedding_params_M": 12.5875, "nonemb_weight_bytes_bf16": 25175040,
        "top1": 0.4814, "total_params_M": 13.0587, "val_loss": 2.5436,
    }
    in1k_row = {
        "run": "V4-A0-s1337-in1k", "arm": "A0", "dataset": "imagenet-1k", "corpus": "imagenet-1k",
        "n_classes": 1000, "n_layer": 4, "non_embedding_params_M": 12.5875,
        "nonemb_weight_bytes_bf16": 25175040, "top1": 0.41, "top1_best": 0.42, "best_step": 48000,
        "passes": 9.99, "total_params_M": 13.0587, "val_loss": 2.9,
    }
    text = bvr.vision_report([in100_row, in1k_row])

    assert "### imagenet-100" in text
    assert "### imagenet-1k" in text
    i100_idx = text.index("### imagenet-100")
    i1k_idx = text.index("### imagenet-1k")
    block_100 = text[i100_idx:i1k_idx]
    block_1k = text[i1k_idx:]

    assert "V4-A0-s1337 " in block_100 and "48.1" in block_100
    assert "V4-A0-s1337-in1k" in block_1k and "41.0" in block_1k
    # exactly what a `seen` keying collision would break: each block must carry only its own row
    assert "V4-A0-s1337-in1k" not in block_100
    assert "V4-A0-s1337 " not in block_1k


def test_p5k_probe_rows_are_excluded_from_the_report():
    """`-p5k` rows are diagnostic probes (spec 2026-09-18 §7), excluded by spec regardless of
    which directory they happen to live in. Real ones live under results/probes/ rather than
    results/runs/, but the parse itself — not directory layout — must be what keeps them out."""
    bvr = _load_bvr()

    p5k_row = {
        "run": "V8-A1u4-s1337-p5k", "arm": "A1u4", "dataset": "imagenet-100", "n_classes": 100,
        "n_layer": 8, "non_embedding_params_M": 12.5875, "nonemb_weight_bytes_bf16": 25175040,
        "top1": 0.99, "total_params_M": 13.0587, "val_loss": 0.01,
    }
    text = bvr.vision_report([p5k_row])
    assert "V8-A1u4-s1337-p5k" not in text
    assert "99.0" not in text


def test_block_header_reports_mixed_when_n_classes_disagrees_within_a_block():
    """Minor finding (review, 2026-09-18): the per-block header used to read n_classes/passes off
    one arbitrary sample row. Two rows landing in the same corpus block with disagreeing
    n_classes must render "mixed" in the header, not silently print one arm's value as if it
    applied to the whole block."""
    bvr = _load_bvr()

    row_a = {
        "run": "V4-A0-s1337", "arm": "A0", "dataset": "imagenet-100", "n_classes": 100,
        "n_layer": 4, "non_embedding_params_M": 12.5875, "nonemb_weight_bytes_bf16": 25175040,
        "top1": 0.4814, "total_params_M": 13.0587, "val_loss": 2.5436,
    }
    row_b = {
        "run": "V8-A0-s1337", "arm": "A0", "dataset": "imagenet-100", "n_classes": 57,
        "n_layer": 8, "non_embedding_params_M": 25.17, "nonemb_weight_bytes_bf16": 50340000,
        "top1": 0.47, "total_params_M": 26.0, "val_loss": 2.77,
    }
    text = bvr.vision_report([row_a, row_b])
    assert "mixed" in text


# --- follow-up: the depth gate / pairs / follow-up-seed trigger, generalised per corpus -------
#
# Before this follow-up, `vision_report` computed the depth gate, pairs and follow-up-seed
# trigger exactly once, through a helper (`im100()`) hardcoded to read ImageNet-100 rows only —
# so no amount of ImageNet-1k data landing could ever produce the ImageNet-1k depth gate, the
# single decision the ImageNet-1k pass exists to make. These tests push both corpora through the
# real `vision_report()` path and assert each corpus gets its own, independently-computed and
# independently-labelled verdict.

def _row(run: str, arm: str, corpus: str, top1: float, *, n_layer: int) -> dict:
    return {
        "run": run, "arm": arm, "dataset": corpus, "n_classes": 100 if corpus == "imagenet-100" else 1000,
        "n_layer": n_layer, "non_embedding_params_M": 12.5875, "nonemb_weight_bytes_bf16": 25175040,
        "top1": top1, "total_params_M": 13.0587, "val_loss": 2.5,
    }


def _block(text: str, corpus: str) -> str:
    """Slice out just one corpus's block (up to the next `### ` heading or end of text)."""
    start = text.index(f"### {corpus}")
    rest = text[start + len(f"### {corpus}"):]
    nxt = rest.find("\n### ")
    return rest if nxt == -1 else rest[:nxt]


def _line_containing(block: str, needle: str) -> str:
    lines = [l for l in block.splitlines() if needle in l]
    assert lines, f"no line containing {needle!r} in block:\n{block}"
    return lines[0]


def test_per_corpus_depth_gate_renders_independently_with_opposite_outcomes():
    """Two corpora with OPPOSITE depth-gate outcomes must each render their own correct verdict —
    not one corpus's numbers leaking into the other's line, and not a single global verdict."""
    bvr = _load_bvr()

    rows = [
        # imagenet-100: V8-A0 (deeper) BEATS V4-A0 -> gate passes
        _row("V4-A0-s1337", "A0", "imagenet-100", 0.40, n_layer=4),
        _row("V8-A0-s1337", "A0", "imagenet-100", 0.45, n_layer=8),
        # imagenet-1k: V8-A0 (deeper) LOSES to V4-A0 -> gate fails
        _row("V4-A0-s1337-in1k", "A0", "imagenet-1k", 0.30, n_layer=4),
        _row("V8-A0-s1337-in1k", "A0", "imagenet-1k", 0.25, n_layer=8),
    ]
    text = bvr.vision_report(rows)

    block_100 = _block(text, "imagenet-100")
    block_1k = _block(text, "imagenet-1k")

    gate_100 = _line_containing(block_100, "Depth gate")
    gate_1k = _line_containing(block_1k, "Depth gate")

    assert "45.0" in gate_100 and "40.0" in gate_100 and "FAIL" not in gate_100
    assert "imagenet-100" in gate_100
    assert "30.0" in gate_1k and "25.0" in gate_1k and "FAIL" in gate_1k
    assert "imagenet-1k" in gate_1k


def test_corpus_with_no_rows_renders_pending_gate_without_crashing():
    """imagenet-1k before its first result lands: zero in1k rows must still produce a labelled,
    non-crashing 'pending' gate and follow-up-seed line for that corpus rather than the block
    vanishing or the call raising."""
    bvr = _load_bvr()

    rows = [_row("V4-A0-s1337", "A0", "imagenet-100", 0.4814, n_layer=4)]   # no in1k rows at all
    text = bvr.vision_report(rows)   # must not raise

    assert "### imagenet-1k" in text
    block_1k = _block(text, "imagenet-1k")
    gate_1k = _line_containing(block_1k, "Depth gate")
    follow_1k = _line_containing(block_1k, "Follow-up seed rule")
    assert "pending" in gate_1k and "imagenet-1k" in gate_1k
    assert "pending" in follow_1k and "imagenet-1k" in follow_1k


def test_imagenet100_gate_and_values_unchanged_by_percorpus_refactor():
    """The five pre-existing ImageNet-100 rows must keep rendering their original values and
    their original FAIL verdict (V8-A0 46.96 < V4-A0 48.14) once the gate becomes per-corpus."""
    bvr = _load_bvr()

    rows = [
        _row("V4-A0-s1337", "A0", "imagenet-100", 0.4814, n_layer=4),
        _row("V8-A0-s1337", "A0", "imagenet-100", 0.4696, n_layer=8),
        _row("V8-A2-s1337", "A2", "imagenet-100", 0.4732, n_layer=8),
        _row("V8-A1u4-s1337", "A1u4", "imagenet-100", 0.4736, n_layer=8),
        _row("V8-A1u4t-s1337", "A1u4t", "imagenet-100", 0.4728, n_layer=8),
    ]
    text = bvr.vision_report(rows)

    block_100 = _block(text, "imagenet-100")
    gate_100 = _line_containing(block_100, "Depth gate")
    assert "47.0" in gate_100 and "48.1" in gate_100
    assert "FAIL" in gate_100 and "imagenet-100" in gate_100
