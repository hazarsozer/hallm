"""Load an experiment (model + training) config from YAML (NFR-1: file-based, not hard-coded).

A config names a `shape` (one of `SHAPES`) or an explicit `model:` block, an `arm` (A0–A3 or an
ablation), and an optional `train:` override block. The four arm configs differ ONLY in `arm`, so the
matched-budget control is structurally guaranteed.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from hallm.model.config import SHAPES, ModelConfig, arm_config
from hallm.train import TrainConfig


def load_experiment(path: str | Path) -> tuple[ModelConfig, TrainConfig]:
    """Return (model_config, train_config) from a YAML experiment file."""
    spec = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if "shape" in spec:
        base = SHAPES[spec["shape"]]
    elif "model" in spec:
        base = ModelConfig(**spec["model"])
    else:
        raise ValueError(f"{path}: config must define either `shape` or `model`")

    model_cfg = arm_config(base, spec.get("arm", "A0"))
    train_cfg = TrainConfig(**spec.get("train", {}))
    train_cfg.block_size = model_cfg.block_size  # keep loader/model block_size in sync
    return model_cfg, train_cfg
