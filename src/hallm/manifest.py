"""Frozen run manifests (spec 06 §8.2).

A manifest is written ONCE at launch next to a run's checkpoints and never touched again. Two runs
form a valid controlled pair iff `manifest_diff` reports only the declared variables (sharing flags,
seed, out_dir). Environment keys (timestamp, GPU, platform, config path) are excluded from the diff:
they describe where/when a run happened, not what was trained."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch

from hallm.model.config import ModelConfig
from hallm.train import TrainConfig

# legit differences within a pair; `determinism` is an OBSERVATION of the host, not a variable
_ENV_KEYS = {"created_utc", "gpu", "platform", "config_path", "determinism"}


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit(repo_dir: str | Path = ".") -> str:
    # The GPU box is not a git checkout, which is why every pre-P0 manifest recorded "unknown".
    # The launcher exports the commit it deployed instead of turning the box into a checkout.
    env = os.environ.get("HALLM_GIT_COMMIT", "").strip()
    if env:
        return env
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_dir, capture_output=True, text=True, timeout=10
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def build_manifest(
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    config_path: str | Path | None = None,
    data_files: tuple | list = (),
    repo_dir: str | Path = ".",
) -> dict:
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_path": str(config_path) if config_path else None,
        "model_cfg": asdict(model_cfg),
        "train_cfg": asdict(train_cfg),
        "data_sha256": {Path(p).name: file_sha256(p) for p in data_files},
        "tokenizer": "gpt2",
        "git_commit": _git_commit(repo_dir),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "platform": platform.platform(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "determinism": {
            # `deterministic: true` in train_cfg is a REQUEST. Flash SDP's backward is
            # non-deterministic and warns so at runtime, so record what is actually true rather
            # than asserting a reproducibility property the run does not have (spec P0 item 5).
            "requested": train_cfg.deterministic,
            "flash_sdp_enabled": bool(torch.backends.cuda.flash_sdp_enabled())
            if torch.cuda.is_available()
            else False,
            "torch_deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        },
    }


def write_manifest(manifest: dict, path: str | Path) -> None:
    """Write-once: a manifest is frozen at launch and must never be overwritten."""
    p = Path(path)
    if p.exists():
        raise FileExistsError(f"manifest already frozen: {p}")
    p.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _flatten(d: dict, prefix: str = "") -> dict:
    out: dict = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


def manifest_diff(a: dict, b: dict) -> dict:
    """Dotted keys whose values differ, environment keys excluded. Empty dict ⇒ identical runs."""
    fa, fb = _flatten(a), _flatten(b)
    return {
        k: (fa.get(k), fb.get(k))
        for k in sorted(set(fa) | set(fb))
        if k.split(".", 1)[0] not in _ENV_KEYS and fa.get(k) != fb.get(k)
    }
