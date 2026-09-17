from dataclasses import replace

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
