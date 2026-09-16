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


# --- controller decision 2: evaluate_top1 must use unsmoothed CE and restore label_smoothing ---


def test_evaluate_top1_ignores_the_configured_label_smoothing():
    """The reported val loss is plain cross-entropy, independent of model.label_smoothing —
    comparable across runs regardless of the training smoothing constant."""
    model = ViT(tiny_cfg())
    images, labels = fake_data(8)
    cfg = VisionTrainConfig(crop_size=32, eval_batch=8, dtype="float32")

    model.label_smoothing = 0.0
    torch.manual_seed(0)
    _, loss_unsmoothed = evaluate_top1(model, images, labels, cfg, "cpu")

    model.label_smoothing = 0.9
    torch.manual_seed(0)
    _, loss_with_high_smoothing_configured = evaluate_top1(model, images, labels, cfg, "cpu")

    assert loss_unsmoothed == loss_with_high_smoothing_configured


def test_evaluate_top1_restores_label_smoothing_even_on_error():
    """A save/restore around eval must not leak model.label_smoothing to callers — including when
    the evaluation body raises."""
    model = ViT(tiny_cfg())
    model.label_smoothing = 0.42
    images, labels = fake_data(8)
    cfg = VisionTrainConfig(crop_size=32, eval_batch=4, dtype="float32")

    evaluate_top1(model, images, labels, cfg, "cpu")
    assert model.label_smoothing == 0.42

    bad_labels = labels.copy()
    bad_labels[0] = 999  # out-of-range class index -> F.cross_entropy raises
    raised = False
    try:
        evaluate_top1(model, images, bad_labels, cfg, "cpu")
    except (IndexError, RuntimeError):
        raised = True
    assert raised
    assert model.label_smoothing == 0.42


def test_training_after_evaluation_keeps_the_configured_smoothing():
    """A mid-run eval must not leave the model at 0.0 smoothing for the training steps after it."""
    model = ViT(tiny_cfg())
    images, labels = fake_data(8)
    cfg = VisionTrainConfig(max_steps=2, warmup_steps=1, batch_size=4, crop_size=32,
                            log_interval=1, eval_interval=1, eval_batch=4,
                            label_smoothing=0.33, dtype="float32", deterministic=False)
    train_vision(model, cfg, images, labels, device="cpu", val_images=images, val_labels=labels)
    assert model.label_smoothing == 0.33
