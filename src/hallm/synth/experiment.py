"""Load a Track 2 synth experiment (model + task + training) from YAML, mirroring
`hallm.experiment.load_experiment`'s pattern for the WikiText pipeline. Kept separate from that
loader because the schema genuinely differs (a task name + difficulty knob instead of a dataset,
`SynthTrainConfig` instead of `TrainConfig`) -- not because the two pipelines share nothing.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from hallm.model.config import ModelConfig, arm_config
from hallm.synth.data import make_task
from hallm.synth.harness import SynthTrainConfig
from hallm.synth.tasks import SynthTask


def load_synth_experiment(path: str | Path) -> tuple[ModelConfig, SynthTrainConfig, SynthTask, int, str]:
    """Return (model_cfg, train_cfg, task, difficulty, run_id) from a YAML file.

    Expected shape:
        task: p_hop_induction     # key into hallm.synth.data.TASK_CLASSES
        task_kwargs: {vocab_size: 256, seq_len_margin: 4}   # optional, passed to the task's __init__
        difficulty: 4
        arm: A0                   # or A1u<k>[t|n|a], per hallm.model.config.arm_config
        model: {n_embd, n_layer, n_head, ffn_mult, block_size}  # vocab_size comes from the task
        train: {...}              # SynthTrainConfig field overrides
        run_id: synth-p_hop_induction-L4-A0-s0
    """
    spec = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    task = make_task(spec["task"], **spec.get("task_kwargs", {}))
    difficulty = int(spec["difficulty"])

    model_spec = dict(spec.get("model", {}))
    model_spec["vocab_size"] = task.vocab_size
    base = ModelConfig(**model_spec)
    model_cfg = arm_config(base, spec.get("arm", "A0"))

    train_cfg = SynthTrainConfig(**spec.get("train", {}))
    default_run_id = f"synth-{task.name}-L{model_cfg.n_layer}-{model_cfg.arm}-s{train_cfg.seed}"
    run_id = spec.get("run_id", default_run_id)
    return model_cfg, train_cfg, task, difficulty, run_id
