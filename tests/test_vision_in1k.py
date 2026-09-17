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
