"""Queue semantics (spec 06 §8.3): drain in order, freeze manifest at first launch, skip finished
runs, resume interrupted ones. GPU-free: smoke shape, synthetic bins, 4 steps, CPU float32."""
from __future__ import annotations

import json

import yaml

from hallm.data import make_synthetic_data
from hallm.model import SHAPES
from hallm.runqueue import drain, run_one

SMOKE_TRAIN = dict(
    max_steps=4, warmup_steps=1, batch_size=2, grad_accum=1, dtype="float32",
    eval_iters=2, log_interval=1, checkpoint_interval=2, seed=7,
)


def _setup(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for split in ("train", "val"):
        make_synthetic_data(SHAPES["smoke"].vocab_size, 2048, seed=0).tofile(data_dir / f"{split}.bin")
    cfgs = []
    for arm in ("A0", "A2"):
        name = f"smoke-{arm}-s7"
        spec = {"shape": "smoke", "arm": arm,
                "train": {**SMOKE_TRAIN, "out_dir": str(tmp_path / "runs" / name)}}
        p = tmp_path / f"{name}.yaml"
        p.write_text(yaml.safe_dump(spec))
        cfgs.append(p)
    queue = tmp_path / "queue.txt"
    queue.write_text("\n".join(str(c) for c in cfgs) + "\n")
    return data_dir, cfgs, queue


def test_run_one_trains_freezes_and_skips(tmp_path):
    data_dir, cfgs, _ = _setup(tmp_path)
    row = run_one(cfgs[0], data_dir, device="cpu")
    out = tmp_path / "runs" / "smoke-A0-s7"
    assert (out / "smoke-A0-s7.pt").exists()
    assert (out / "manifest.json").exists()
    assert row["run"] == "smoke-A0-s7" and "test_ppl" in row
    manifest = json.loads((out / "manifest.json").read_text())
    assert set(manifest["data_sha256"]) == {"train.bin", "val.bin"}
    assert run_one(cfgs[0], data_dir, device="cpu") is None  # done ⇒ skip


def test_run_one_resumes_after_interrupt(tmp_path):
    data_dir, cfgs, _ = _setup(tmp_path)
    assert run_one(cfgs[0], data_dir, device="cpu", stop_step=2) is None  # interrupted ⇒ no row yet
    out = tmp_path / "runs" / "smoke-A0-s7"
    assert (out / "resume.pt").exists() and not (out / "smoke-A0-s7.pt").exists()
    row = run_one(cfgs[0], data_dir, device="cpu")  # second session finishes it
    assert row is not None and (out / "smoke-A0-s7.pt").exists()


def test_drain_appends_results(tmp_path):
    data_dir, _, queue = _setup(tmp_path)
    results = tmp_path / "results.jsonl"
    rows = drain(queue, data_dir, results, device="cpu")
    assert [r["run"] for r in rows] == ["smoke-A0-s7", "smoke-A2-s7"]
    lines = [json.loads(l) for l in results.read_text().splitlines()]
    assert len(lines) == 2 and lines[1]["arm"] == "A2"
    assert drain(queue, data_dir, results, device="cpu") == []  # everything done ⇒ no-op
