# Term-2 Campaign Infrastructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the infrastructure the scaling campaign needs before any new training run: optimizer-state checkpoint–resume, frozen run manifests, a drain-the-queue runner, the L4/L32 ladder shapes + configs, and the Tier-1.5 capability eval harness (LAMBADA / BLiMP / sliced PPL).

**Architecture:** Extend the existing single-config training harness (`src/hallm/`) in place: `train.py` gains resumable checkpoints, new modules `manifest.py`, `runqueue.py`, `capeval.py` are added beside it, and thin CLI scripts wrap them. All GPU-free logic is TDD'd against the existing `smoke` shape and synthetic data — tests never touch the network or a GPU.

**Tech Stack:** Python 3.12 via `uv`, PyTorch, numpy, PyYAML, tiktoken (lazy import only), pytest.

**Spec:** `wiki/roadmap/06-scaling-campaign.md` (approved 2026-08-19). Protocol constants come from `RESULTS.md` and `configs/arm2_halvit.yaml`.

## Global Constraints

- Run everything through `uv`: `uv run pytest`, `uv run python …`. Never system python.
- Tests are network-free and GPU-free (repo rule; see `data/wikitext.py` docstring — tiktoken is lazy-imported so tests never need it). Real training stays manual/GPU-only, never launched by tests or automation (`00-master.md` §5).
- Checkpoints must stay loadable with `torch.load(weights_only=True)` — only tensors + primitive containers in saved dicts (existing rule in `train.py:save_checkpoint`).
- The fixed protocol never varies inside a task: lr 6.0e-4, min_lr 6.0e-5, warmup 200, max_steps 50000, weight_decay 0.1, grad_clip 1.0, batch 12 × accum 2, bf16, dropout 0.0 (spec §3).
- Ladder run naming: `<rung>-<arm>-s<seed>`, e.g. `L4-A2-s1338` (spec §9). Rung→shape: L4→`s30h`, L8→`s30`, L16→`s30x2`, L32→`s30x4`.
- Match existing code style: module docstrings tie code to roadmap/spec sections; `from __future__ import annotations`; dataclasses over dicts.

## File Structure

| File | Responsibility |
|---|---|
| `src/hallm/train.py` (modify) | + `checkpoint_interval` in `TrainConfig`; `save_resume_checkpoint` / `load_resume_checkpoint`; `train(..., resume_path=, stop_step=)` |
| `src/hallm/manifest.py` (create) | Frozen run manifests: build, write-once, diff |
| `src/hallm/runqueue.py` (create) | `run_one()` / `drain()` — train-with-resume + manifest + eval per queue entry |
| `src/hallm/capeval.py` (create) | `sequence_nll`, `blimp_accuracy`, `lambada_accuracy`, `greedy_continuation`, `sliced_perplexity`, jsonl loaders |
| `src/hallm/model/config.py` (modify) | + `s30h` (L=4), `s30x4` (L=32) shapes |
| `scripts/gen_ladder_configs.py` (create) | Deterministic generator for the 18 ladder configs + 14-entry `queue.txt` |
| `scripts/run_queue.py` (create) | CLI wrapper over `runqueue.drain` |
| `scripts/capability_eval.py` (create) | CLI: run capability evals over checkpoint globs, write `results/capability.json` + markdown |
| `tests/test_resume.py`, `tests/test_manifest.py`, `tests/test_runqueue.py`, `tests/test_ladder_configs.py`, `tests/test_capeval.py` (create) | One test module per unit |

---

### Task 1: Resumable checkpoints in the training loop

**Files:**
- Modify: `src/hallm/train.py`
- Test: `tests/test_resume.py`

**Interfaces:**
- Produces: `TrainConfig.checkpoint_interval: int = 1000`;
  `save_resume_checkpoint(path, model: GPT, train_cfg: TrainConfig, opt, gen: torch.Generator, step: int) -> None` (atomic: tmp + `os.replace`; model config read from `model.cfg`);
  `load_resume_checkpoint(path, map_location="cpu") -> dict` with keys `model, opt, gen_state, torch_rng, step, model_cfg, train_cfg`;
  `train(model, train_cfg, train_data, device=None, progress=False, resume_path: str | None = None, stop_step: int | None = None) -> list[dict]` — if `resume_path` exists it restores and continues from `step`; while running it rewrites `resume_path` every `checkpoint_interval` steps; `stop_step` saves-and-breaks after that step (bounded GPU sessions + tests).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_resume.py
"""Resume-equivalence: interrupted-and-resumed training must equal an unbroken run (spec §8.1)."""
from __future__ import annotations

from dataclasses import replace

import torch

from hallm.data import make_synthetic_data
from hallm.model import GPT, SHAPES
from hallm.train import (
    TrainConfig,
    load_resume_checkpoint,
    save_resume_checkpoint,
    set_seed,
    train,
)

CFG = SHAPES["smoke"]
TC = TrainConfig(
    max_steps=8, warmup_steps=2, batch_size=4, grad_accum=1, block_size=CFG.block_size,
    dtype="float32", seed=7, deterministic=True, checkpoint_interval=4, log_interval=1,
)
DATA = make_synthetic_data(CFG.vocab_size, 4096, seed=0)


def _fresh_model() -> GPT:
    set_seed(TC.seed, TC.deterministic)
    return GPT(CFG)


def test_resume_checkpoint_roundtrip(tmp_path):
    model = _fresh_model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(7)
    path = tmp_path / "resume.pt"
    save_resume_checkpoint(path, model, TC, opt, gen, step=3)
    ckpt = load_resume_checkpoint(path)  # must survive weights_only=True
    assert ckpt["step"] == 3
    assert ckpt["model_cfg"]["n_layer"] == CFG.n_layer
    assert ckpt["train_cfg"]["max_steps"] == TC.max_steps
    assert torch.equal(ckpt["gen_state"], gen.get_state())


def test_stop_step_writes_checkpoint(tmp_path):
    path = tmp_path / "resume.pt"
    model = _fresh_model()
    history = train(model, TC, DATA, device="cpu", resume_path=str(path), stop_step=4)
    assert path.exists()
    assert load_resume_checkpoint(path)["step"] == 4
    assert max(h["step"] for h in history) == 3  # steps 0..3 ran


def test_resume_equals_unbroken_run(tmp_path):
    # unbroken 8-step run
    m_full = _fresh_model()
    train(m_full, TC, DATA, device="cpu")
    # same run interrupted at step 4, then resumed by a FRESH process (fresh model object)
    path = tmp_path / "resume.pt"
    m_a = _fresh_model()
    train(m_a, TC, DATA, device="cpu", resume_path=str(path), stop_step=4)
    m_b = GPT(CFG)  # arbitrary init — resume must overwrite it entirely
    train(m_b, TC, DATA, device="cpu", resume_path=str(path))
    sd_full, sd_res = m_full.state_dict(), m_b.state_dict()
    assert sd_full.keys() == sd_res.keys()
    for k in sd_full:
        assert torch.equal(sd_full[k], sd_res[k]), f"param {k} diverged after resume"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_resume.py -v`
Expected: FAIL — `ImportError: cannot import name 'save_resume_checkpoint'` (and `TrainConfig` has no `checkpoint_interval`).

- [ ] **Step 3: Implement in `src/hallm/train.py`**

Add to `TrainConfig` (after `out_dir`):

```python
    checkpoint_interval: int = 1000  # steps between resume-checkpoint writes (0 = never)
```

Add after `load_checkpoint` / `build_model_from_checkpoint`:

```python
def save_resume_checkpoint(
    path: str | os.PathLike,
    model: GPT,
    train_cfg: TrainConfig,
    opt: torch.optim.Optimizer,
    gen: torch.Generator,
    step: int,
) -> None:
    """Full training state for exact resume (spec 06 §8.1): weights + AdamW moments + data-order
    generator + global RNG + step. Atomic (tmp + replace) so an interrupt never corrupts the file.
    Stays `weights_only=True`-loadable: tensors and primitive containers only."""
    tmp = str(path) + ".tmp"
    torch.save(
        {
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "gen_state": gen.get_state(),
            "torch_rng": torch.get_rng_state(),
            "step": step,
            "model_cfg": asdict(model.cfg),
            "train_cfg": asdict(train_cfg),
        },
        tmp,
    )
    os.replace(tmp, path)


def load_resume_checkpoint(path: str | os.PathLike, map_location="cpu") -> dict:
    """Load a resume checkpoint written by `save_resume_checkpoint` (safe: weights_only)."""
    return torch.load(path, map_location=map_location, weights_only=True)
```

Modify `train()` — new signature and resume/periodic-save logic (the body between optimizer creation and the loop, and inside the loop):

```python
def train(
    model: GPT,
    train_cfg: TrainConfig,
    train_data: np.ndarray,
    device: str | torch.device | None = None,
    progress: bool = False,
    resume_path: str | None = None,
    stop_step: int | None = None,
) -> list[dict]:
    """Run the matched-budget loop. Returns a history of {step, loss, lr} log dicts.

    If `resume_path` is given, training state is periodically saved there (every
    `checkpoint_interval` steps) and, when the file already exists, restored from it — so an
    interrupted run continues exactly where it stopped (spec 06 §8.1). `stop_step` ends the
    session after that step (checkpoint saved), for bounded GPU windows."""
    from hallm.data import get_batch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    opt = configure_optimizer(model, train_cfg.weight_decay, train_cfg.lr, (train_cfg.beta1, train_cfg.beta2))
    gen = torch.Generator().manual_seed(train_cfg.seed)  # data order — identical across arms

    start_step = 0
    if resume_path and os.path.exists(resume_path):
        ckpt = load_resume_checkpoint(resume_path)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        gen.set_state(ckpt["gen_state"])
        torch.set_rng_state(ckpt["torch_rng"])
        start_step = int(ckpt["step"])

    use_amp = device != "cpu" and train_cfg.dtype in ("bfloat16", "float16")
    amp_dtype = torch.bfloat16 if train_cfg.dtype == "bfloat16" else torch.float16

    history: list[dict] = []
    for step in range(start_step, train_cfg.max_steps):
        # ... existing loop body unchanged (lr set, grad accum, clip, opt.step) ...

        if step % train_cfg.log_interval == 0 or step == train_cfg.max_steps - 1:
            history.append({"step": step, "loss": loss_accum, "lr": lr})
            if progress:
                print(f"step {step:6d} | loss {loss_accum:.4f} | lr {lr:.2e}")

        done = step + 1
        at_interval = train_cfg.checkpoint_interval > 0 and done % train_cfg.checkpoint_interval == 0
        stopping = stop_step is not None and done >= stop_step
        if resume_path and (at_interval or stopping):
            save_resume_checkpoint(resume_path, model, train_cfg, opt, gen, done)
        if stopping:
            break
    return history
```

(The existing loop body — LR assignment, grad-accum inner loop, clipping, `opt.step()` — is untouched; only the range start, the trailing checkpoint block, and the docstring change.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_resume.py -v`
Expected: 3 PASS. The equivalence test is the load-bearing one: it proves LR schedule (driven by `train_cfg.max_steps`, unchanged), data order (generator state), and AdamW moments all survive the round-trip.

- [ ] **Step 5: Run the full suite (no regressions)**

Run: `uv run pytest`
Expected: all pass (26 existing + 3 new).

- [ ] **Step 6: Commit**

```bash
git add src/hallm/train.py tests/test_resume.py
git commit -m "feat: optimizer-state checkpoint-resume in the training loop (spec 06 §8.1)"
```

---

### Task 2: Frozen run manifests

**Files:**
- Create: `src/hallm/manifest.py`
- Test: `tests/test_manifest.py`

**Interfaces:**
- Consumes: `ModelConfig`, `TrainConfig` (dataclasses, via `asdict`).
- Produces: `build_manifest(model_cfg, train_cfg, config_path=None, data_files=(), repo_dir=".") -> dict`;
  `write_manifest(manifest: dict, path) -> None` (raises `FileExistsError` if the file exists — frozen-at-launch);
  `manifest_diff(a: dict, b: dict) -> dict[str, tuple]` (dotted flattened keys → `(a_val, b_val)`, environment keys ignored);
  `file_sha256(path) -> str`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_manifest.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_manifest.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'hallm.manifest'`.

- [ ] **Step 3: Implement `src/hallm/manifest.py`**

```python
"""Frozen run manifests (spec 06 §8.2).

A manifest is written ONCE at launch next to a run's checkpoints and never touched again. Two runs
form a valid controlled pair iff `manifest_diff` reports only the declared variables (sharing flags,
seed, out_dir). Environment keys (timestamp, GPU, platform, config path) are excluded from the diff:
they describe where/when a run happened, not what was trained."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch

from hallm.model.config import ModelConfig
from hallm.train import TrainConfig

_ENV_KEYS = {"created_utc", "gpu", "platform", "config_path"}  # legit differences within a pair


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit(repo_dir: str | Path = ".") -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_dir, capture_output=True, text=True, timeout=10
        )
        return out.stdout.strip() or "unknown"
    except OSError:
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
        "git_commit": _git_commit(repo_dir),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "platform": platform.platform(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_manifest.py -v`
Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hallm/manifest.py tests/test_manifest.py
git commit -m "feat: frozen run manifests with controlled-pair diff (spec 06 §8.2)"
```

---

### Task 3: Ladder shapes and config generator

**Files:**
- Modify: `src/hallm/model/config.py` (SHAPES dict)
- Create: `scripts/gen_ladder_configs.py`
- Test: `tests/test_ladder_configs.py`

**Interfaces:**
- Consumes: `SHAPES`, `load_experiment` (Task 1's `checkpoint_interval` must exist in `TrainConfig`).
- Produces: `SHAPES["s30h"]` (d=512, L=4), `SHAPES["s30x4"]` (d=512, L=32);
  `gen_ladder_configs.generate(out_dir: str | Path) -> list[str]` returning the queue entries it wrote; writes `<out_dir>/<rung>-<arm>-s<seed>.yaml` × 18 and `<out_dir>/queue.txt` (14 pending entries, L4 runs first).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_ladder_configs.py
"""Ladder shapes hit the exact non-embedding counts from spec 06 §3, and the generator emits
protocol-identical configs that differ only in {shape, arm, seed, out_dir}."""
from __future__ import annotations

from hallm.experiment import load_experiment
from hallm.model import GPT, SHAPES, arm_config

from scripts.gen_ladder_configs import EXISTING, generate


def test_ladder_shape_param_counts():
    # non-embedding params for unshared arms: 12·d²·L exactly (roadmap 00-master §3)
    expected = {"s30h": 12 * 512**2 * 4, "s30": 12 * 512**2 * 8, "s30x2": 12 * 512**2 * 16, "s30x4": 12 * 512**2 * 32}
    for shape, count in expected.items():
        model = GPT(arm_config(SHAPES[shape], "A0"))
        assert model.num_parameters(non_embedding=True) == count, shape


def test_generate_ladder(tmp_path):
    queue = generate(tmp_path)
    yamls = sorted(p.name for p in tmp_path.glob("*.yaml"))
    assert len(yamls) == 18  # 3 rungs × 2 arms × 3 seeds
    assert len(queue) == 18 - len(EXISTING) == 14
    assert all("s1337" not in q or "L4" in q for q in queue)  # only L4 keeps seed 1337 (others exist)
    assert [q for q in queue if "L4" in q] == queue[:6]  # L4 runs drain first (spec 06 §4)

    # every generated config loads, and a pair differs ONLY in sharing flags
    a0, _ = load_experiment(tmp_path / "L4-A0-s1338.yaml")
    a2, tc = load_experiment(tmp_path / "L4-A2-s1338.yaml")
    assert (a0.n_layer, a0.arm, a2.arm) == (4, "A0", "A2")
    assert tc.seed == 1338 and tc.max_steps == 50_000 and tc.batch_size == 12 and tc.grad_accum == 2

    qfile = (tmp_path / "queue.txt").read_text().splitlines()
    assert qfile == queue
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_ladder_configs.py -v`
Expected: FAIL — `KeyError: 's30h'` / `ModuleNotFoundError: scripts.gen_ladder_configs`.
Note: `scripts/` has no `__init__.py`; if the import fails for that reason, add an empty `scripts/__init__.py`, and if pytest still can't resolve it, an empty `conftest.py` at the repo root puts the root on `sys.path`.

- [ ] **Step 3: Implement**

In `src/hallm/model/config.py`, add to `SHAPES` (after `"s30x2"`):

```python
    # scaling-campaign ladder (wiki/roadmap/06-scaling-campaign.md §3): width fixed at d=512,
    # depth-scaled. s30h = L4 rung (~12.6M non-emb unshared); s30x4 = L32 stretch rung (~100.7M).
    "s30h": ModelConfig(vocab_size=50257, block_size=512, n_embd=512, n_layer=4, n_head=8),
    "s30x4": ModelConfig(vocab_size=50257, block_size=512, n_embd=512, n_layer=32, n_head=8),
```

Create `scripts/gen_ladder_configs.py`:

```python
"""Generate the Tier-1 ladder configs (wiki/roadmap/06-scaling-campaign.md §4).

18 configs = 3 rungs (L4/L8/L16) × 2 arms (A0/A2) × 3 seeds; the 4 pairs already trained in
Experiment 1 / the iso 2×2 (seed 1337 at L8 and L16) are generated for the record but excluded
from queue.txt. Protocol constants are the Experiment-1 recipe verbatim — never edit them here.

Usage: uv run python scripts/gen_ladder_configs.py [--out configs/ladder]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

RUNGS = {"L4": "s30h", "L8": "s30", "L16": "s30x2"}  # order = drain order (cheapest evidence first)
ARMS = ["A0", "A2"]
SEEDS = [1337, 1338, 1339]
EXISTING = {"L8-A0-s1337", "L8-A2-s1337", "L16-A0-s1337", "L16-A2-s1337"}  # runs/{A0,A2}.pt, runs_iso/

TRAIN = dict(
    lr=6.0e-4, min_lr=6.0e-5, warmup_steps=200, max_steps=50_000, weight_decay=0.1,
    grad_clip=1.0, batch_size=12, grad_accum=2, dtype="bfloat16", deterministic=True,
    eval_interval=1000, checkpoint_interval=1000,
)


def generate(out_dir: str | Path) -> list[str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    queue: list[str] = []
    for rung, shape in RUNGS.items():
        for arm in ARMS:
            for seed in SEEDS:
                name = f"{rung}-{arm}-s{seed}"
                spec = {
                    "shape": shape,
                    "arm": arm,
                    "train": {**TRAIN, "seed": seed, "out_dir": f"runs/ladder/{name}"},
                }
                (out / f"{name}.yaml").write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
                if name not in EXISTING:
                    queue.append(str(out / f"{name}.yaml"))
    (out / "queue.txt").write_text("\n".join(queue) + "\n", encoding="utf-8")
    return queue


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="configs/ladder")
    args = ap.parse_args()
    queue = generate(args.out)
    print(f"wrote 18 configs to {args.out}/, queue.txt has {len(queue)} pending runs")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_ladder_configs.py -v`
Expected: 2 PASS.

- [ ] **Step 5: Generate the real configs and commit them**

```bash
uv run python scripts/gen_ladder_configs.py --out configs/ladder
git add src/hallm/model/config.py scripts/gen_ladder_configs.py tests/test_ladder_configs.py configs/ladder/
git commit -m "feat: ladder shapes s30h/s30x4 + generated Tier-1 configs and queue (spec 06 §4)"
```

---

### Task 4: Run queue

**Files:**
- Create: `src/hallm/runqueue.py`
- Create: `scripts/run_queue.py`
- Test: `tests/test_runqueue.py`

**Interfaces:**
- Consumes: `load_experiment`, `train(..., resume_path=, stop_step=)` (Task 1), `build_manifest`/`write_manifest` (Task 2), `save_checkpoint`, `evaluate_arm`, `load_bin`, `set_seed`, `GPT`.
- Produces: `run_one(cfg_path, data_dir, device, stop_step=None) -> dict | None` — trains one config to completion (or to `stop_step`), freezes `manifest.json` on first launch, writes final `<out_dir>/<name>.pt`, returns the eval row (or `None` if already done or stopped early);
  `drain(queue_file, data_dir, results_path, device, max_runs=None, stop_step=None) -> list[dict]` — processes queue entries in order, appends each eval row as a JSON line to `results_path`.

Run states are inferred from the filesystem — no state file to corrupt: final checkpoint exists ⇒ done (skip); `resume.pt` exists ⇒ resume; otherwise fresh.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_runqueue.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_runqueue.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'hallm.runqueue'`.

- [ ] **Step 3: Implement `src/hallm/runqueue.py`**

```python
"""Drain-the-queue runner (spec 06 §8.3): one ssh command starts "whatever is next", and
interruption is always safe. Run state is inferred from the filesystem — final checkpoint exists ⇒
done; resume.pt exists ⇒ continue; else fresh. The manifest is frozen at FIRST launch and never
rewritten, so it describes what the whole (possibly multi-session) run trained."""

from __future__ import annotations

import json
from pathlib import Path

from hallm.data import load_bin
from hallm.eval import evaluate_arm
from hallm.experiment import load_experiment
from hallm.manifest import build_manifest, write_manifest
from hallm.model import GPT
from hallm.train import save_checkpoint, set_seed, train


def run_one(cfg_path: str | Path, data_dir: str | Path, device: str, stop_step: int | None = None) -> dict | None:
    """Train one queue entry. Returns the eval row, or None if already done / stopped early."""
    cfg_path, data_dir = Path(cfg_path), Path(data_dir)
    model_cfg, train_cfg = load_experiment(cfg_path)
    name = cfg_path.stem
    out = Path(train_cfg.out_dir)
    final = out / f"{name}.pt"
    if final.exists():
        print(f"[skip] {name}: final checkpoint exists")
        return None
    out.mkdir(parents=True, exist_ok=True)

    manifest_path = out / "manifest.json"
    if not manifest_path.exists():
        write_manifest(
            build_manifest(model_cfg, train_cfg, config_path=cfg_path,
                           data_files=[data_dir / "train.bin", data_dir / "val.bin"]),
            manifest_path,
        )

    resume = out / "resume.pt"
    set_seed(train_cfg.seed, train_cfg.deterministic)  # seeds init; resume overwrites weights if present
    model = GPT(model_cfg)
    print(f"[run ] {name}: {'resuming' if resume.exists() else 'fresh'} on {device}")
    train(model, train_cfg, load_bin(data_dir / "train.bin"), device=device, progress=True,
          resume_path=str(resume), stop_step=stop_step)
    if stop_step is not None and stop_step < train_cfg.max_steps:
        print(f"[stop] {name}: paused at step {stop_step} (resume.pt saved)")
        return None

    save_checkpoint(model, model_cfg, train_cfg, final)
    row = evaluate_arm(model, model_cfg, load_bin(data_dir / "val.bin"), batch_size=8, device=device)
    row["run"] = name
    return row


def drain(queue_file: str | Path, data_dir: str | Path, results_path: str | Path, device: str,
          max_runs: int | None = None, stop_step: int | None = None) -> list[dict]:
    """Process queue entries in order; append each finished run's eval row to `results_path`."""
    entries = [l.strip() for l in Path(queue_file).read_text().splitlines()
               if l.strip() and not l.startswith("#")]
    results_path = Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for entry in entries:
        row = run_one(entry, data_dir, device, stop_step=stop_step)
        if row is None:
            continue
        with results_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        rows.append(row)
        if max_runs is not None and len(rows) >= max_runs:
            break
    return rows
```

Create `scripts/run_queue.py`:

```python
"""CLI for the campaign run queue (spec 06 §8.3). GPU, manual — never launched by automation.

Typical remote session (from the Mac, ssh'd into the Linux box, ideally under systemd-inhibit):

    uv run python scripts/run_queue.py --queue configs/ladder/queue.txt \
        --data data/ --results results/ladder.jsonl

Interrupt any time (Ctrl-C, reboot, Windows switch): the next invocation resumes mid-run from
runs/ladder/<name>/resume.pt. Use --max-runs 1 for a single-run session, --stop-step N to bound it.
"""

from __future__ import annotations

import argparse

import torch

from hallm.runqueue import drain


def main() -> None:
    ap = argparse.ArgumentParser(description="hallm campaign run queue")
    ap.add_argument("--queue", default="configs/ladder/queue.txt")
    ap.add_argument("--data", default="data")
    ap.add_argument("--results", default="results/ladder.jsonl")
    ap.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    ap.add_argument("--max-runs", type=int, default=None, help="finish at most N runs this session")
    ap.add_argument("--stop-step", type=int, default=None, help="pause the current run after step N")
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    rows = drain(args.queue, args.data, args.results, device, max_runs=args.max_runs, stop_step=args.stop_step)
    print(f"\nsession complete: {len(rows)} run(s) finished → {args.results}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_runqueue.py -v`
Expected: 3 PASS.

- [ ] **Step 5: Run the full suite and commit**

Run: `uv run pytest` — all pass.

```bash
git add src/hallm/runqueue.py scripts/run_queue.py tests/test_runqueue.py
git commit -m "feat: filesystem-state run queue with resume and frozen manifests (spec 06 §8.3)"
```

---

### Task 5: Capability eval core (Tier 1.5)

**Files:**
- Create: `src/hallm/capeval.py`
- Test: `tests/test_capeval.py`

**Interfaces:**
- Consumes: `GPT` (forward with targets ⇒ full logits + mean CE loss; without ⇒ last-position logits only), `evaluate_perplexity`.
- Produces: `sequence_nll(model, ids, device="cpu") -> float` (sum NLL in nats of `ids[1:]` given prefixes; left-truncates to `block_size + 1`);
  `blimp_accuracy(model, pairs, device="cpu") -> float` (fraction where NLL(good) < NLL(bad), strict — ties count as wrong);
  `greedy_continuation(model, context_ids, n_tokens, device="cpu") -> list[int]`;
  `lambada_accuracy(model, examples, device="cpu") -> float` (exact-match greedy prediction of the target ids);
  `sliced_perplexity(model, data, block_size, n_slices=10, batch_size=8, device="cpu") -> list[float]`;
  `load_lambada(path, encode) -> list[tuple[list[int], list[int]]]`, `load_blimp_file(path, encode) -> list[tuple[list[int], list[int]]]` (encode is injected so tests never need tiktoken).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_capeval.py
"""Capability metrics (spec 06 §5) — verified against hand computations on the smoke shape.
Network-free: fake byte-level encoder, synthetic data, no tiktoken."""
from __future__ import annotations

import json

import torch
import torch.nn.functional as F

from hallm.capeval import (
    blimp_accuracy,
    greedy_continuation,
    lambada_accuracy,
    load_blimp_file,
    load_lambada,
    sequence_nll,
    sliced_perplexity,
)
from hallm.data import make_synthetic_data
from hallm.model import GPT, SHAPES

CFG = SHAPES["smoke"]


def _model() -> GPT:
    torch.manual_seed(0)
    return GPT(CFG)


def test_sequence_nll_matches_manual():
    model = _model()
    ids = list(range(10))
    x = torch.tensor([ids[:-1]])
    with torch.no_grad():
        logits, _ = model(x, torch.tensor([ids[1:]]))
    manual = F.cross_entropy(
        logits[0], torch.tensor(ids[1:]), reduction="sum"
    ).item()
    assert abs(sequence_nll(model, ids) - manual) < 1e-3


def test_blimp_tie_counts_as_wrong():
    model = _model()
    s = list(range(8))
    assert blimp_accuracy(model, [(s, s)]) == 0.0


def test_blimp_prefers_learned_sequence():
    model = _model()
    good = [3] * 16  # trivially learnable
    torch.manual_seed(0)
    bad = torch.randint(0, CFG.vocab_size, (16,)).tolist()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    x, y = torch.tensor([good[:-1]]), torch.tensor([good[1:]])
    for _ in range(50):
        opt.zero_grad()
        loss = model(x, y)[1]
        loss.backward()
        opt.step()
    assert blimp_accuracy(model, [(good, bad)]) == 1.0


def test_greedy_matches_argmax():
    model = _model()
    ctx = list(range(12))
    with torch.no_grad():
        logits, _ = model(torch.tensor([ctx]))
    assert greedy_continuation(model, ctx, 1) == [int(logits[0, -1].argmax())]


def test_lambada_exact_match_semantics():
    model = _model()
    ctx = list(range(12))
    target = greedy_continuation(model, ctx, 2)
    assert lambada_accuracy(model, [(ctx, target)]) == 1.0
    wrong = [(target[0] + 1) % CFG.vocab_size, target[1]]
    assert lambada_accuracy(model, [(ctx, wrong)]) == 0.0


def test_sliced_perplexity():
    model = _model()
    data = make_synthetic_data(CFG.vocab_size, 4096, seed=0)
    slices = sliced_perplexity(model, data, block_size=CFG.block_size, n_slices=4, batch_size=2)
    assert len(slices) == 4
    assert all(s > 0 and s == s for s in slices)  # positive, not NaN


def test_jsonl_loaders(tmp_path):
    encode = lambda s: [ord(c) % 256 for c in s]  # fake byte encoder — no tiktoken in tests
    lam = tmp_path / "lambada.jsonl"
    lam.write_text(json.dumps({"text": "the quick brown fox"}) + "\n")
    (ctx, tgt), = load_lambada(lam, encode)
    assert ctx == encode("the quick brown") and tgt == encode(" fox")

    bl = tmp_path / "anaphor.jsonl"
    bl.write_text(json.dumps({"sentence_good": "he ran", "sentence_bad": "he run"}) + "\n")
    (good, bad), = load_blimp_file(bl, encode)
    assert good == encode("he ran") and bad == encode("he run")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_capeval.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'hallm.capeval'`.

- [ ] **Step 3: Implement `src/hallm/capeval.py`**

```python
"""Capability evals beyond perplexity (spec 06 §5, Tier 1.5).

Inference-only over existing checkpoints — no training, no GPU lockdown. Chosen to discriminate at
12–100M scale: LAMBADA (long-range final-word prediction), BLiMP (grammatical minimal pairs), and
per-slice PPL over the eval stream (a coarse per-domain proxy; contiguous slices ≈ article groups).
The question: is the sharing tax uniform, or does the PPL average hide a lopsided deficit?

Loaders take an `encode` callable (e.g. a tiktoken encoder's) so tests inject a fake encoder and
never touch the network."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from hallm.eval import evaluate_perplexity
from hallm.model.gpt import GPT


@torch.no_grad()
def sequence_nll(model: GPT, ids: list[int], device: str = "cpu") -> float:
    """Sum NLL (nats) of ids[1:] given their prefixes. Left-truncates to block_size + 1 tokens."""
    model.eval()
    ids = list(ids)[-(model.cfg.block_size + 1):]
    x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
    y = torch.tensor([ids[1:]], dtype=torch.long, device=device)
    _, loss = model(x, y)  # mean CE over the sequence
    return loss.item() * (len(ids) - 1)


@torch.no_grad()
def blimp_accuracy(model: GPT, pairs, device: str = "cpu") -> float:
    """Fraction of (good_ids, bad_ids) pairs with NLL(good) < NLL(bad). Strict: a tie is wrong."""
    correct = sum(
        1 for good, bad in pairs
        if sequence_nll(model, good, device) < sequence_nll(model, bad, device)
    )
    return correct / len(pairs)


@torch.no_grad()
def greedy_continuation(model: GPT, context_ids, n_tokens: int, device: str = "cpu") -> list[int]:
    """Argmax-decode n_tokens after the context (sliding window at block_size)."""
    model.eval()
    ids = list(context_ids)
    for _ in range(n_tokens):
        x = torch.tensor([ids[-model.cfg.block_size:]], dtype=torch.long, device=device)
        logits, _ = model(x)  # inference path: logits at the last position only
        ids.append(int(logits[0, -1].argmax()))
    return ids[len(context_ids):]


@torch.no_grad()
def lambada_accuracy(model: GPT, examples, device: str = "cpu") -> float:
    """examples: (context_ids, target_ids). Correct iff greedy continuation matches target exactly."""
    correct = sum(
        1 for ctx, tgt in examples
        if greedy_continuation(model, ctx, len(tgt), device) == list(tgt)
    )
    return correct / len(examples)


@torch.no_grad()
def sliced_perplexity(
    model: GPT, data: np.ndarray, block_size: int, n_slices: int = 10,
    batch_size: int = 8, device: str = "cpu",
) -> list[float]:
    """PPL per contiguous slice of the eval stream. Slices too short for one window are skipped."""
    bounds = np.linspace(0, len(data), n_slices + 1, dtype=int)
    return [
        evaluate_perplexity(model, data[a:b], block_size, batch_size, device)
        for a, b in zip(bounds[:-1], bounds[1:])
        if b - a > block_size
    ]


# --- jsonl loaders (encode injected; real callers pass tiktoken's encode_ordinary) ---

def load_lambada(path: str | Path, encode) -> list[tuple[list[int], list[int]]]:
    """LAMBADA jsonl ({"text": ...}): context = all but the last word, target = " " + last word."""
    examples = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        text = json.loads(line)["text"].strip()
        context, _, last = text.rpartition(" ")
        if not context:
            continue
        examples.append((list(encode(context)), list(encode(" " + last))))
    return examples


def load_blimp_file(path: str | Path, encode) -> list[tuple[list[int], list[int]]]:
    """One BLiMP paradigm jsonl → (sentence_good_ids, sentence_bad_ids) pairs."""
    pairs = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        pairs.append((list(encode(d["sentence_good"])), list(encode(d["sentence_bad"]))))
    return pairs
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_capeval.py -v`
Expected: 7 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hallm/capeval.py tests/test_capeval.py
git commit -m "feat: capability eval core — LAMBADA, BLiMP, sliced PPL (spec 06 §5)"
```

---

### Task 6: Capability eval driver script

**Files:**
- Create: `scripts/capability_eval.py`

**Interfaces:**
- Consumes: `build_model_from_checkpoint` (`train.py`), everything from `capeval.py`, `load_bin`.
- Produces: CLI writing `results/capability.json` (`{run_name: {lambada_acc, blimp_macro, blimp_per_task, sliced_ppl}}`) and `results/capability.md` (one markdown table row per checkpoint). No new library code — all logic already tested in Task 5, so this task's verification is a smoke invocation, not new unit tests.

- [ ] **Step 1: Implement `scripts/capability_eval.py`**

```python
"""Tier-1.5 capability evals over trained checkpoints (spec 06 §5). Inference-only; runs on CPU or
GPU without locking the box for long.

One-time data fetch (manual, like the WikiText prepare step):
  LAMBADA (OpenAI variant, jsonl):
    curl -L -o data/lambada_test.jsonl \
      https://openaipublic.blob.core.windows.net/gpt-2/data/lambada_test.jsonl
  BLiMP (67 paradigm jsonl files):
    git clone --depth 1 https://github.com/alexwarstadt/blimp /tmp/blimp && \
      mkdir -p data/blimp && cp /tmp/blimp/data/*.jsonl data/blimp/

Usage:
  uv run python scripts/capability_eval.py --checkpoints 'runs/*.pt' 'runs/ladder/*/*.pt' \
      --lambada data/lambada_test.jsonl --blimp data/blimp --data data/ --out results/
Use --limit N to subsample each benchmark for a quick pass.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import torch

from hallm.capeval import (
    blimp_accuracy,
    lambada_accuracy,
    load_blimp_file,
    load_lambada,
    sliced_perplexity,
)
from hallm.data import load_bin
from hallm.train import build_model_from_checkpoint


def main() -> None:
    ap = argparse.ArgumentParser(description="hallm capability evals (LAMBADA / BLiMP / sliced PPL)")
    ap.add_argument("--checkpoints", nargs="+", required=True, help="glob(s) of .pt checkpoints")
    ap.add_argument("--lambada", default=None, help="lambada_test.jsonl (skip if omitted)")
    ap.add_argument("--blimp", default=None, help="dir of BLiMP paradigm .jsonl files (skip if omitted)")
    ap.add_argument("--data", default=None, help="dir with val.bin for sliced PPL (skip if omitted)")
    ap.add_argument("--n-slices", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None, help="subsample each benchmark to N examples")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    lam, blimp = None, {}
    if args.lambada or args.blimp:
        import tiktoken  # lazy: only needed when text benchmarks are requested (repo rule)

        encode = tiktoken.get_encoding("gpt2").encode_ordinary
        if args.lambada:
            lam = load_lambada(args.lambada, encode)[: args.limit]
        if args.blimp:
            blimp_files = sorted(glob.glob(str(Path(args.blimp) / "*.jsonl")))
            blimp = {Path(f).stem: load_blimp_file(f, encode)[: args.limit] for f in blimp_files}
    val = load_bin(Path(args.data) / "val.bin") if args.data else None

    paths = sorted(p for pattern in args.checkpoints for p in glob.glob(pattern))
    results: dict[str, dict] = {}
    for path in paths:
        name = Path(path).stem
        model, cfg = build_model_from_checkpoint(path, map_location=device)
        model.to(device).eval()
        row: dict = {"arm": cfg.arm, "n_layer": cfg.n_layer}
        if lam:
            row["lambada_acc"] = round(lambada_accuracy(model, lam, device), 4)
        if blimp:
            per_task = {k: round(blimp_accuracy(model, v, device), 4) for k, v in blimp.items()}
            row["blimp_per_task"] = per_task
            row["blimp_macro"] = round(sum(per_task.values()) / len(per_task), 4)
        if val is not None:
            row["sliced_ppl"] = [round(p, 3) for p in
                                 sliced_perplexity(model, val, cfg.block_size, args.n_slices, device=device)]
        results[name] = row
        print(f"{name}: " + json.dumps({k: v for k, v in row.items() if k != 'blimp_per_task'}))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "capability.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    lines = ["| run | arm | L | lambada | blimp (macro) | sliced PPL min–max |",
             "|-----|-----|---|---------|---------------|--------------------|"]
    for name, r in sorted(results.items()):
        sl = r.get("sliced_ppl")
        sl_txt = f"{min(sl)}–{max(sl)}" if sl else "—"
        lines.append(f"| {name} | {r['arm']} | {r['n_layer']} | {r.get('lambada_acc', '—')} | "
                     f"{r.get('blimp_macro', '—')} | {sl_txt} |")
    (out / "capability.md").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {out / 'capability.json'} and {out / 'capability.md'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-verify the driver end-to-end (no network: sliced-PPL only)**

Build a throwaway checkpoint + synthetic val.bin, then run the script with only `--data` (LAMBADA/BLiMP paths omitted ⇒ skipped):

```bash
uv run python - <<'EOF'
from pathlib import Path
from hallm.data import make_synthetic_data
from hallm.model import GPT, SHAPES
from hallm.train import TrainConfig, save_checkpoint
Path("/tmp/capsmoke/data").mkdir(parents=True, exist_ok=True)
cfg = SHAPES["smoke"]
make_synthetic_data(cfg.vocab_size, 4096, seed=0).tofile("/tmp/capsmoke/data/val.bin")
save_checkpoint(GPT(cfg), cfg, TrainConfig(), "/tmp/capsmoke/smoke-A0.pt")
EOF
uv run python scripts/capability_eval.py --checkpoints '/tmp/capsmoke/*.pt' \
    --data /tmp/capsmoke/data --n-slices 4 --out /tmp/capsmoke/results --device cpu
cat /tmp/capsmoke/results/capability.md
```

Expected: the table prints one row for `smoke-A0` with `—` for lambada/blimp and a real `sliced PPL min–max` range. This exercises checkpoint loading, the skip logic, and both output files without any network access (the `tiktoken` import is guarded behind `--lambada`/`--blimp`).

- [ ] **Step 3: Full suite + commit**

Run: `uv run pytest` — all pass.

```bash
git add scripts/capability_eval.py
git commit -m "feat: capability eval driver over checkpoint globs (spec 06 §5)"
```

---

### Task 7: Documentation close-out

**Files:**
- Modify: `README.md` (add a "Campaign workflow" subsection if a run-commands section exists; otherwise append one)
- Modify: `wiki/log.md` (append entry)

**Interfaces:** none — documentation only.

- [ ] **Step 1: Document the campaign workflow in README.md**

Append (or merge into the existing run instructions) — exact text:

```markdown
## Term-2 campaign workflow (spec: wiki/roadmap/06-scaling-campaign.md)

One-time: `uv run python scripts/gen_ladder_configs.py` (configs + queue already committed).

Each GPU session (on the Linux box, under systemd-inhibit):

    uv run python scripts/run_queue.py --queue configs/ladder/queue.txt \
        --data data/ --results results/ladder.jsonl

Interrupt freely — the next invocation resumes from `runs/ladder/<name>/resume.pt`.
Bound a session with `--max-runs 1` or `--stop-step N`.

Capability evals (inference-only, any machine; data fetch documented in the script docstring):

    uv run python scripts/capability_eval.py --checkpoints 'runs/*.pt' 'runs/ladder/*/*.pt' \
        --lambada data/lambada_test.jsonl --blimp data/blimp --data data/ --out results/
```

- [ ] **Step 2: Append a wiki log entry**

Append to `wiki/log.md`:

```markdown
## [<today's date>] build | Campaign infrastructure landed (spec 06 §8 + Tier 1.5): optimizer-state checkpoint-resume with exact-equivalence test, frozen run manifests + controlled-pair diff, filesystem-state run queue (scripts/run_queue.py), ladder shapes s30h/s30x4 + 18 generated configs (14 queued), capability eval harness (LAMBADA/BLiMP/sliced-PPL, network-free tests). Test suite grown from 26 tests; all green.
```

- [ ] **Step 3: Final verification and commit**

Run: `uv run pytest` — all pass. Then:

```bash
git add README.md wiki/log.md
git commit -m "docs: campaign workflow + wiki log entry for infra build"
```
