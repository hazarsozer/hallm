"""P0 item 1: per-run metrics logging must observe training without perturbing it."""

import numpy as np

from hallm.model.config import SHAPES
from hallm.model.gpt import GPT
from hallm.train import TrainConfig, set_seed, train


def _tiny(seed: int = 1337):
    # Seed before construction: the two models in the data-order test must start identical,
    # otherwise the test compares init noise rather than data order.
    set_seed(seed)
    cfg = SHAPES["smoke"]
    return GPT(cfg), cfg


def _cfg(**kw):
    base = dict(max_steps=6, batch_size=2, block_size=64, log_interval=1, eval_interval=3,
                eval_iters=2, checkpoint_interval=0, dtype="float32")
    base.update(kw)
    return TrainConfig(**base)


def test_val_probe_does_not_change_training_data_order():
    """The val probe must draw from its own generator, so train losses are bit-identical."""
    data = np.random.randint(0, 256, size=5000, dtype=np.uint16)
    val = np.random.randint(0, 256, size=2000, dtype=np.uint16)
    tc = _cfg()

    m1, _ = _tiny()
    h_without = train(m1, tc, data, device="cpu")

    m2, _ = _tiny()
    h_with = train(m2, tc, data, device="cpu", val_data=val)

    assert [h["loss"] for h in h_without] == [h["loss"] for h in h_with], \
        "val probe perturbed the training data order"


def test_val_probe_records_val_loss_at_eval_interval():
    data = np.random.randint(0, 256, size=5000, dtype=np.uint16)
    val = np.random.randint(0, 256, size=2000, dtype=np.uint16)
    m, _ = _tiny()
    hist = train(m, _cfg(), data, device="cpu", val_data=val)
    with_val = [h for h in hist if "val_loss" in h]
    assert with_val, "no val_loss recorded"
    assert all(isinstance(h["val_loss"], float) for h in with_val)


def test_metrics_jsonl_written_when_path_given(tmp_path):
    import json
    data = np.random.randint(0, 256, size=5000, dtype=np.uint16)
    val = np.random.randint(0, 256, size=2000, dtype=np.uint16)
    mp = tmp_path / "metrics.jsonl"
    m, _ = _tiny()
    train(m, _cfg(), data, device="cpu", val_data=val, metrics_path=str(mp))
    lines = [json.loads(x) for x in mp.read_text().splitlines() if x.strip()]
    assert lines and all("step" in r and "loss" in r for r in lines)
    assert any("val_loss" in r for r in lines)
