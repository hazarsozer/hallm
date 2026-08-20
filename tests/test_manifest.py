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


# --- P0 items 2 and 5: provenance the manifest previously got wrong -------------------------

def test_git_commit_env_override(monkeypatch):
    """The GPU box is not a git checkout, so the launcher exports the deployed commit."""
    from hallm.manifest import _git_commit

    monkeypatch.setenv("HALLM_GIT_COMMIT", "deadbeef1234")
    assert _git_commit(".") == "deadbeef1234"


def test_git_commit_env_override_ignores_blank(monkeypatch):
    from hallm.manifest import _git_commit

    monkeypatch.setenv("HALLM_GIT_COMMIT", "   ")
    assert _git_commit("/nonexistent-path-xyz") == "unknown"


def test_manifest_records_determinism_truth():
    """`deterministic: true` was asserted while Flash SDP's backward is non-deterministic."""
    from hallm.manifest import build_manifest
    from hallm.model.config import SHAPES
    from hallm.train import TrainConfig

    m = build_manifest(SHAPES["smoke"], TrainConfig(deterministic=True))
    assert "determinism" in m
    assert m["determinism"]["requested"] is True
    assert "flash_sdp_enabled" in m["determinism"]
    assert "torch_deterministic_algorithms" in m["determinism"]


def test_determinism_block_is_not_a_pair_invalidating_difference():
    """Two runs differing only in observed determinism state are still a valid pair."""
    from hallm.manifest import manifest_diff

    a = {"model_cfg": {"n_layer": 8}, "determinism": {"flash_sdp_enabled": True}}
    b = {"model_cfg": {"n_layer": 8}, "determinism": {"flash_sdp_enabled": False}}
    assert manifest_diff(a, b) == {}
