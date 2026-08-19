"""Manifests are frozen at launch and mechanically diffable (spec 06 §8.2): two runs form a valid
controlled pair iff their manifests differ only in the declared variables."""
from __future__ import annotations

import json

import pytest

from hallm.manifest import build_manifest, file_sha256, manifest_diff, write_manifest
from hallm.model import SHAPES, arm_config
from hallm.train import TrainConfig

BASE = SHAPES["smoke"]
TC = TrainConfig(max_steps=4, out_dir="runs/x")


def test_file_sha256(tmp_path):
    p = tmp_path / "data.bin"
    p.write_bytes(b"hallm")
    import hashlib
    assert file_sha256(p) == hashlib.sha256(b"hallm").hexdigest()


def test_manifest_contents(tmp_path):
    m = build_manifest(arm_config(BASE, "A0"), TC, config_path="c.yaml", data_files=())
    for key in ("created_utc", "model_cfg", "train_cfg", "git_commit", "torch", "python", "gpu"):
        assert key in m
    assert m["model_cfg"]["share_intra_ffn"] is False
    assert m["train_cfg"]["seed"] == TC.seed


def test_write_is_frozen(tmp_path):
    m = build_manifest(arm_config(BASE, "A0"), TC)
    path = tmp_path / "manifest.json"
    write_manifest(m, path)
    assert json.loads(path.read_text())["train_cfg"]["max_steps"] == 4
    with pytest.raises(FileExistsError):
        write_manifest(m, path)  # frozen at launch — never overwritten


def test_controlled_pair_diff():
    a = build_manifest(arm_config(BASE, "A0"), TC)
    b = build_manifest(arm_config(BASE, "A2"), TC)
    diff = manifest_diff(a, b)
    assert set(diff) == {"model_cfg.share_intra_ffn", "model_cfg.share_intra_attn"}
    assert manifest_diff(a, a) == {}
