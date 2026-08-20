# P0 Instrumentation + P1 Mechanism-Decomposition Configs — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make run artifacts trustworthy (P0 items 1–5) without altering training mathematics, then generate the P1 ablation configs and chain them onto the GPU so no overnight time is lost.

**Architecture:** All P0 changes are **pure instrumentation**: they add observation, never change what is computed. `train()` gains an optional validation-loss probe that draws from its **own** generator so the training data order is bit-identical to every run already completed. Provenance gaps (git commit, determinism claim, memory) are filled by recording the truth rather than by changing behaviour. P1 configs reuse the existing `ARMS["A2-ffn"]` / `ARMS["A2-attn"]` entries, which are implemented and tested but have never been run.

**Tech Stack:** Python 3.11, PyTorch 2.13, uv, pytest, YAML configs.

**Spec:** `docs/superpowers/specs/2026-08-20-research-program-design.md`

## Global Constraints

- **No change to training mathematics.** P1's runs are compared against the already-completed `L8-A0-s1337/1338` and `L8-A2-s1337/1338`. Any change to init, kernels, data order, or optimizer invalidates that comparison. Instrumentation only.
- **P0 item 7 (residual `1/√(2L)` init) is DEFERRED, not skipped.** It changes the model, so it must not land before P1. It becomes its own controlled pair later. Recorded in the spec as a known deviation.
- **P0 item 5 resolves to recording the truth, not enabling deterministic SDP.** Switching kernels changes the computation and would break comparability.
- **`TrainConfig` fields may be ADDED but never renamed or removed.** `runqueue.run_one` validates a resume checkpoint by iterating the *checkpoint's* keys, so new fields are safe; renames would break the in-flight `L16-A2-s1339` if it ever resumes.
- Protocol constants are never edited: lr 6e-4→6e-5, warmup 200, wd 0.1, clip 1.0, bf16, batch 12 × accum 2, 50k steps.
- Run-ID grammar `L<depth>-A<arm>-s<seed>`; arm tags in run IDs carry no hyphen (`A2ffn`, `A2attn`).
- Test command: `uv run pytest`. Must stay green (26 tests) throughout.

---

### Task 1: Validation-loss probe with an isolated generator

**Files:**
- Modify: `src/hallm/train.py` (`train()` signature and loop)
- Test: `tests/test_metrics_logging.py`

**Interfaces:**
- Consumes: `estimate_loss(model, data, cfg, device, generator)` (existing)
- Produces: `train(..., val_data=None, metrics_path=None)` — history dicts gain `val_loss` on eval-interval steps; other steps carry `train`-only keys.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
from hallm.model.config import SHAPES
from hallm.model.gpt import GPT
from hallm.train import TrainConfig, train


def _tiny():
    cfg = SHAPES["smoke"]
    return GPT(cfg), cfg


def test_val_probe_does_not_change_training_data_order():
    """The val probe must draw from its own generator, so train losses are bit-identical."""
    data = np.random.randint(0, 256, size=5000, dtype=np.uint16)
    val = np.random.randint(0, 256, size=2000, dtype=np.uint16)
    tc = TrainConfig(max_steps=6, batch_size=2, block_size=64, log_interval=1,
                     eval_interval=3, eval_iters=2, checkpoint_interval=0, dtype="float32")

    m1, _ = _tiny()
    h_without = train(m1, tc, data, device="cpu")

    m2, _ = _tiny()
    h_with = train(m2, tc, data, device="cpu", val_data=val)

    losses_without = [h["loss"] for h in h_without]
    losses_with = [h["loss"] for h in h_with]
    assert losses_without == losses_with, "val probe perturbed the training data order"


def test_val_probe_records_val_loss_at_eval_interval():
    data = np.random.randint(0, 256, size=5000, dtype=np.uint16)
    val = np.random.randint(0, 256, size=2000, dtype=np.uint16)
    tc = TrainConfig(max_steps=6, batch_size=2, block_size=64, log_interval=1,
                     eval_interval=3, eval_iters=2, checkpoint_interval=0, dtype="float32")
    m, _ = _tiny()
    hist = train(m, tc, data, device="cpu", val_data=val)
    with_val = [h for h in hist if "val_loss" in h]
    assert with_val, "no val_loss recorded"
    assert all(isinstance(h["val_loss"], float) for h in with_val)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_metrics_logging.py -v`
Expected: FAIL — `train() got an unexpected keyword argument 'val_data'`

- [ ] **Step 3: Write minimal implementation**

In `src/hallm/train.py`, extend the signature:

```python
def train(
    model: GPT,
    train_cfg: TrainConfig,
    train_data: np.ndarray,
    device: str | torch.device | None = None,
    progress: bool = False,
    resume_path: str | None = None,
    stop_step: int | None = None,
    val_data: np.ndarray | None = None,
    metrics_path: str | None = None,
) -> list[dict]:
```

After `gen = torch.Generator().manual_seed(train_cfg.seed)`, add an isolated eval generator:

```python
    # SEPARATE generator: the val probe must never advance the training data order, or every
    # completed run becomes incomparable. Offset the seed so the two streams differ.
    eval_gen = torch.Generator().manual_seed(train_cfg.seed + 10_000)
```

Inside the loop, replace the logging block with:

```python
        if step % train_cfg.log_interval == 0 or step == train_cfg.max_steps - 1:
            rec = {"step": step, "loss": loss_accum, "lr": lr}
            if (
                val_data is not None
                and train_cfg.eval_interval > 0
                and (step % train_cfg.eval_interval == 0 or step == train_cfg.max_steps - 1)
            ):
                rec["val_loss"] = estimate_loss(model, val_data, train_cfg, device, eval_gen)
            history.append(rec)
            if metrics_path:
                with open(metrics_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
            if progress:
                extra = f" | val {rec['val_loss']:.4f}" if "val_loss" in rec else ""
                print(f"step {step:6d} | loss {loss_accum:.4f} | lr {lr:.2e}{extra}")
```

Add `import json` at the top of the module.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_metrics_logging.py -v && uv run pytest`
Expected: new tests PASS, full suite still green.

- [ ] **Step 5: Commit**

```bash
git add src/hallm/train.py tests/test_metrics_logging.py
git commit -m "feat(train): val-loss probe on an isolated generator + metrics jsonl"
```

---

### Task 2: Git-commit provenance via environment override

**Files:**
- Modify: `src/hallm/manifest.py` (`_git_commit`)
- Test: `tests/test_manifest.py`

**Interfaces:**
- Produces: `_git_commit(repo_dir)` honours `HALLM_GIT_COMMIT` when set.

The GPU box is not a git checkout, which is why every manifest records `"unknown"`. Rather than turn the box into one mid-campaign, the launcher exports the commit it deployed.

- [ ] **Step 1: Write the failing test**

```python
def test_git_commit_env_override(monkeypatch):
    from hallm.manifest import _git_commit
    monkeypatch.setenv("HALLM_GIT_COMMIT", "deadbeef1234")
    assert _git_commit(".") == "deadbeef1234"


def test_git_commit_env_override_ignores_blank(monkeypatch):
    from hallm.manifest import _git_commit
    monkeypatch.setenv("HALLM_GIT_COMMIT", "   ")
    assert _git_commit("/nonexistent-path-xyz") == "unknown"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_manifest.py -k git_commit_env -v`
Expected: FAIL — env var ignored, returns real commit or "unknown".

- [ ] **Step 3: Write minimal implementation**

```python
def _git_commit(repo_dir: str | Path = ".") -> str:
    # The GPU box is not a git checkout, so the launcher exports the deployed commit instead.
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
```

Add `import os` at the top of `manifest.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_manifest.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/hallm/manifest.py tests/test_manifest.py
git commit -m "feat(manifest): HALLM_GIT_COMMIT override for the non-checkout GPU box"
```

---

### Task 3: Memory accounting as a measured quantity

**Files:**
- Modify: `src/hallm/metrics.py`
- Test: `tests/test_memory_metrics.py`

**Interfaces:**
- Produces: `kv_cache_bytes(cfg, ctx, batch, bytes_per_elem=2) -> int` and
  `memory_row(model, cfg) -> dict` with keys
  `weight_bytes_bf16`, `kv_bytes_ctx512_b1`, `kv_bytes_ctx2048_b8`, `weight_frac_of_total_ctx512_b1`.

Spec §1 is the thesis's central axis and is currently only ever computed by hand from parameter counts.

- [ ] **Step 1: Write the failing test**

```python
from hallm.model.config import SHAPES
from hallm.model.gpt import GPT
from hallm.metrics import kv_cache_bytes, memory_row


def test_kv_cache_bytes_is_two_d_l_per_token():
    cfg = SHAPES["s30"]  # d=512, L=8
    assert kv_cache_bytes(cfg, ctx=1, batch=1, bytes_per_elem=2) == 2 * 512 * 8 * 2


def test_kv_cache_is_independent_of_sharing():
    from hallm.model.config import arm_config
    base = SHAPES["s30"]
    a0, a2 = arm_config(base, "A0"), arm_config(base, "A2")
    assert kv_cache_bytes(a0, 512, 1) == kv_cache_bytes(a2, 512, 1)


def test_memory_row_reports_weight_fraction_below_one():
    cfg = SHAPES["smoke"]
    row = memory_row(GPT(cfg), cfg)
    assert 0.0 < row["weight_frac_of_total_ctx512_b1"] <= 1.0
    assert row["weight_bytes_bf16"] > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_memory_metrics.py -v`
Expected: FAIL — `ImportError: cannot import name 'kv_cache_bytes'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/hallm/metrics.py`:

```python
def kv_cache_bytes(cfg: ModelConfig, ctx: int, batch: int, bytes_per_elem: int = 2) -> int:
    """KV cache size. 2 (K and V) · d · L per token — INDEPENDENT of every sharing flag, which is
    why re-investing shared weights into depth pays the saving back in cache (spec §1)."""
    return 2 * cfg.n_embd * cfg.n_layer * ctx * batch * bytes_per_elem


def memory_row(model: nn.Module, cfg: ModelConfig) -> dict[str, float | int]:
    """Measured memory footprint, so the thesis's memory claim stops being inferred from params."""
    pc = count_parameters(model)
    weight_bytes = pc["total"] * 2  # bf16
    kv_512_1 = kv_cache_bytes(cfg, 512, 1)
    kv_2048_8 = kv_cache_bytes(cfg, 2048, 8)
    return {
        "weight_bytes_bf16": weight_bytes,
        "nonemb_weight_bytes_bf16": pc["non_embedding"] * 2,
        "kv_bytes_ctx512_b1": kv_512_1,
        "kv_bytes_ctx2048_b8": kv_2048_8,
        "weight_frac_of_total_ctx512_b1": round(weight_bytes / (weight_bytes + kv_512_1), 4),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_memory_metrics.py -v && uv run pytest`
Expected: PASS, suite green.

- [ ] **Step 5: Commit**

```bash
git add src/hallm/metrics.py tests/test_memory_metrics.py
git commit -m "feat(metrics): measured KV-cache and weight-memory accounting"
```

---

### Task 4: Record the determinism truth in the manifest

**Files:**
- Modify: `src/hallm/manifest.py` (`build_manifest`, `_ENV_KEYS`)
- Test: `tests/test_manifest.py`

**Interfaces:**
- Produces: manifest key `determinism` = `{"requested": bool, "flash_sdp_enabled": bool, "torch_deterministic_algorithms": bool}`.

`deterministic: true` is asserted in every manifest, but Flash Attention's backward is non-deterministic and warns so at runtime. Record what is actually true; do not switch kernels (that would change the computation).

- [ ] **Step 1: Write the failing test**

```python
def test_manifest_records_determinism_truth():
    from hallm.manifest import build_manifest
    from hallm.model.config import SHAPES
    from hallm.train import TrainConfig
    m = build_manifest(SHAPES["smoke"], TrainConfig(deterministic=True))
    assert "determinism" in m
    assert m["determinism"]["requested"] is True
    assert "flash_sdp_enabled" in m["determinism"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_manifest.py -k determinism -v`
Expected: FAIL — `KeyError: 'determinism'`

- [ ] **Step 3: Write minimal implementation**

In `build_manifest`'s returned dict add:

```python
        "determinism": {
            "requested": train_cfg.deterministic,
            # Flash SDP's backward is non-deterministic; recording this stops the manifest from
            # asserting a reproducibility property the run does not actually have.
            "flash_sdp_enabled": bool(torch.backends.cuda.flash_sdp_enabled())
            if torch.cuda.is_available()
            else False,
            "torch_deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        },
```

Add `"determinism"` to `_ENV_KEYS` so it never counts as a pair-invalidating difference:

```python
_ENV_KEYS = {"created_utc", "gpu", "platform", "config_path", "determinism"}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_manifest.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/hallm/manifest.py tests/test_manifest.py
git commit -m "feat(manifest): record actual determinism state instead of asserting it"
```

---

### Task 5: Wire instrumentation into the run queue

**Files:**
- Modify: `src/hallm/runqueue.py` (`run_one`)
- Test: `tests/test_runqueue.py`

**Interfaces:**
- Consumes: Tasks 1 and 3.
- Produces: each run directory gains `metrics.jsonl`; each `ladder.jsonl` row gains
  `final_train_loss`, `final_val_loss`, `peak_vram_bytes`, plus all `memory_row` keys.

- [ ] **Step 1: Write the failing test**

```python
def test_run_one_writes_metrics_and_enriched_row(tmp_path, monkeypatch):
    import numpy as np, json, yaml
    from hallm.runqueue import run_one

    data_dir = tmp_path / "data"; data_dir.mkdir()
    for name in ("train.bin", "val.bin"):
        np.random.randint(0, 256, size=4000, dtype=np.uint16).tofile(data_dir / name)

    cfg = {"shape": "smoke", "arm": "A0",
           "train": {"max_steps": 4, "batch_size": 2, "log_interval": 1, "eval_interval": 2,
                     "eval_iters": 2, "checkpoint_interval": 0, "dtype": "float32",
                     "out_dir": str(tmp_path / "runs" / "smoke-run")}}
    p = tmp_path / "smoke-run.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    row = run_one(p, data_dir, "cpu")
    assert row is not None
    assert (tmp_path / "runs" / "smoke-run" / "metrics.jsonl").exists()
    assert "final_train_loss" in row and "final_val_loss" in row
    assert "kv_bytes_ctx512_b1" in row
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_runqueue.py -k metrics_and_enriched -v`
Expected: FAIL — `metrics.jsonl` missing / `KeyError: 'final_train_loss'`

- [ ] **Step 3: Write minimal implementation**

In `run_one`, load val data before training and pass the new arguments:

```python
    val_data = load_bin(data_dir / "val.bin")
    metrics_path = out / "metrics.jsonl"
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    set_seed(train_cfg.seed, train_cfg.deterministic)
    model = GPT(model_cfg)
    print(f"[run ] {name}: {'resuming' if resume.exists() else 'fresh'} on {device}")
    history = train(model, train_cfg, load_bin(data_dir / "train.bin"), device=device, progress=True,
                    resume_path=str(resume), stop_step=stop_step,
                    val_data=val_data, metrics_path=str(metrics_path))
```

Then after `row["run"] = name`, enrich it:

```python
    row.update(memory_row(model, model_cfg))
    if history:
        row["final_train_loss"] = round(history[-1]["loss"], 4)
        vals = [h["val_loss"] for h in history if "val_loss" in h]
        if vals:
            row["final_val_loss"] = round(vals[-1], 4)
    if torch.cuda.is_available():
        row["peak_vram_bytes"] = int(torch.cuda.max_memory_allocated())
```

Add imports at the top of `runqueue.py`:

```python
import torch
from hallm.metrics import memory_row
from hallm.train import train
```

Note: `train` is already imported via `hallm.train`; keep one import line and reuse it.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_runqueue.py -v && uv run pytest`
Expected: PASS, suite green.

- [ ] **Step 5: Commit**

```bash
git add src/hallm/runqueue.py tests/test_runqueue.py
git commit -m "feat(runqueue): per-run metrics.jsonl and enriched results rows"
```

---

### Task 6: Generate the P1 ablation configs

**Files:**
- Modify: `scripts/gen_ladder_configs.py`
- Create: `configs/ladder/L8-A2ffn-s1337.yaml`, `L8-A2ffn-s1338.yaml`, `L8-A2attn-s1337.yaml`, `L8-A2attn-s1338.yaml`, `configs/ladder/queue-p1.txt`
- Test: `tests/test_ladder_configs.py`

**Interfaces:**
- Consumes: `ARMS["A2-ffn"]`, `ARMS["A2-attn"]` (already implemented in `model/config.py`).
- Produces: `generate_p1(out_dir) -> list[str]` returning queue entries in drain order.

- [ ] **Step 1: Write the failing test**

```python
def test_generate_p1_creates_four_configs_with_correct_flags(tmp_path):
    from hallm.experiment import load_experiment
    from scripts.gen_ladder_configs import generate_p1

    queue = generate_p1(tmp_path)
    assert len(queue) == 4

    names = {p.split("/")[-1] for p in queue}
    assert names == {"L8-A2ffn-s1337.yaml", "L8-A2ffn-s1338.yaml",
                     "L8-A2attn-s1337.yaml", "L8-A2attn-s1338.yaml"}

    mc, tc = load_experiment(tmp_path / "L8-A2ffn-s1337.yaml")
    assert mc.share_intra_ffn is True and mc.share_intra_attn is False
    assert mc.n_layer == 8 and tc.seed == 1337 and tc.max_steps == 50_000

    mc, _ = load_experiment(tmp_path / "L8-A2attn-s1338.yaml")
    assert mc.share_intra_ffn is False and mc.share_intra_attn is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ladder_configs.py -k p1 -v`
Expected: FAIL — `ImportError: cannot import name 'generate_p1'`

- [ ] **Step 3: Write minimal implementation**

Append to `scripts/gen_ladder_configs.py`:

```python
# P1 mechanism decomposition (program spec P1): which sublayer's sharing causes the tax.
# Run-ID arm tags carry no hyphen; they map onto ARMS entries that already exist and are tested.
P1_ARMS = {"A2ffn": "A2-ffn", "A2attn": "A2-attn"}
P1_SEEDS = [1337, 1338]


def generate_p1(out_dir: str | Path) -> list[str]:
    """Generate the P1 ablation configs at the L8 rung and return queue entries in drain order."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    queue: list[str] = []
    for tag, arm in P1_ARMS.items():
        for seed in P1_SEEDS:
            name = f"L8-{tag}-s{seed}"
            spec = {
                "shape": "s30",
                "arm": arm,
                "train": {**TRAIN, "seed": seed, "out_dir": f"runs/ladder/{name}"},
            }
            (out / f"{name}.yaml").write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
            queue.append(str(out / f"{name}.yaml"))
    (out / "queue-p1.txt").write_text("\n".join(queue) + "\n", encoding="utf-8")
    return queue
```

Extend `main()` with a `--p1` flag that calls `generate_p1` instead of `generate`.

- [ ] **Step 4: Run tests and generate the real configs**

Run:
```bash
uv run pytest tests/test_ladder_configs.py -v
uv run python scripts/gen_ladder_configs.py --p1 --out configs/ladder
cat configs/ladder/queue-p1.txt
```
Expected: tests PASS; four YAML files plus `queue-p1.txt` written.

- [ ] **Step 5: Commit**

```bash
git add scripts/gen_ladder_configs.py tests/test_ladder_configs.py configs/ladder/
git commit -m "feat(configs): P1 mechanism-decomposition ablation configs at the L8 rung"
```

---

### Task 7: Chain P1 onto the GPU behind the in-flight run

**Files:**
- Create: `scripts/chain_next.sh` (deployed to the box, not run locally)

**Interfaces:**
- Consumes: Task 6's `configs/ladder/queue-p1.txt`.
- Produces: an unattended overnight sequence — wait → verify → deploy → launch.

`L16-A2-s1339` was launched with `--max-runs 1`, so the queue process exits when it finishes and the GPU would otherwise idle until morning. The chain must **verify the row landed** before deploying new code, so a failed run never silently rolls into P1 under changed instrumentation.

- [ ] **Step 1: Write the chain script**

```bash
#!/usr/bin/env bash
# Wait for the in-flight ladder run to finish, verify it landed, deploy P0 code, launch P1.
set -uo pipefail
cd "$HOME/Dev/hallm" || exit 1
LOG="$HOME/Dev/hallm/chain.log"
say() { echo "[$(date -Is)] $*" >>"$LOG"; }

say "chain armed; waiting for run_queue to exit"
while pgrep -f "scripts/run_queue.py" >/dev/null 2>&1; do sleep 60; done
say "run_queue exited"

if ! grep -q '"run": "L16-A2-s1339"' results/ladder.jsonl; then
  say "ABORT: L16-A2-s1339 row not found in results/ladder.jsonl — not launching P1"
  exit 1
fi
say "L16-A2-s1339 row present"

if [ -d "$HOME/hallm-staging/src" ]; then
  cp -r "$HOME/hallm-staging/src/." src/ && \
  cp -r "$HOME/hallm-staging/scripts/." scripts/ && \
  cp -r "$HOME/hallm-staging/configs/." configs/ && say "P0 code deployed" || { say "ABORT: deploy failed"; exit 1; }
else
  say "ABORT: staging dir missing"; exit 1
fi

export HALLM_GIT_COMMIT="$(cat "$HOME/hallm-staging/COMMIT" 2>/dev/null || echo unknown)"
say "launching P1 (4 runs) with commit $HALLM_GIT_COMMIT"
"$HOME/.local/bin/uv" run python scripts/run_queue.py \
  --queue configs/ladder/queue-p1.txt \
  --data data/ --results results/ladder.jsonl --max-runs 4 >>queue.log 2>&1
say "P1 session complete (exit $?)"
```

- [ ] **Step 2: Stage code and configs to the box**

```bash
ssh hsozer@100.77.131.53 'mkdir -p ~/hallm-staging'
git rev-parse HEAD > /tmp/COMMIT
rsync -a src scripts configs hsozer@100.77.131.53:~/hallm-staging/
scp /tmp/COMMIT hsozer@100.77.131.53:~/hallm-staging/COMMIT
scp scripts/chain_next.sh hsozer@100.77.131.53:~/Dev/hallm/scripts/chain_next.sh
```

- [ ] **Step 3: Verify the guard works before arming**

Run: `ssh hsozer@100.77.131.53 'grep -c "L16-A2-s1339" ~/Dev/hallm/results/ladder.jsonl'`
Expected: `0` while the run is in flight — confirming the chain will block, not fire early.

- [ ] **Step 4: Arm the chain**

```bash
ssh hsozer@100.77.131.53 'cd ~/Dev/hallm && chmod +x scripts/chain_next.sh && \
  setsid nohup bash scripts/chain_next.sh >/dev/null 2>&1 < /dev/null & echo armed'
```

- [ ] **Step 5: Confirm armed and waiting**

Run: `ssh hsozer@100.77.131.53 'pgrep -af chain_next; tail -3 ~/Dev/hallm/chain.log'`
Expected: process listed; log shows "chain armed; waiting".

---

## Self-Review

**Spec coverage.** P0 items 1–5 map to Tasks 1, 2, 3, 4 and 5. Item 6 (RESULTS.md correction) is documentation and is handled outside this plan. Item 7 (residual init) is deliberately deferred by a Global Constraint, because landing it would invalidate P1's comparison against completed L8 runs. P1's instrument (4 runs at L8, seeds 1337/1338) is Task 6; its GPU scheduling is Task 7. P1's decision rule needs no code — it is applied at analysis time.

**Placeholder scan.** No TBDs; every code step carries runnable code and every test step an exact command with an expected outcome.

**Type consistency.** `memory_row` is defined in Task 3 and consumed by that exact name in Task 5. `generate_p1` is defined and consumed as `generate_p1`. `train(..., val_data=, metrics_path=)` is defined in Task 1 and called with those exact keywords in Task 5. `kv_cache_bytes(cfg, ctx, batch, bytes_per_elem)` keeps one signature throughout.

**Known gap, accepted:** Tasks 1–5 land *after* `L16-A2-s1339` starts, so that run keeps the old instrumentation — no `metrics.jsonl`, `git_commit: "unknown"`. This is correct: changing code under a live run is the larger risk, and the run is the last of a ladder whose other rows share exactly those gaps. P1 is the first cohort with full provenance.
