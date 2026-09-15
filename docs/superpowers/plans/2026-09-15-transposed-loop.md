# Transposed Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the three transposed-loop arms (`A1u<k>t|n|a`) and the FFN-Jacobian symmetry measurement to hallm, then queue the three seed-1337 runs and run the measurement.

**Architecture:** A new `ModelConfig.loop_pass2` field (default `None`) turns on a transposed second pass inside the existing looped arm: the sublayers gain a `transposed=` flag that reuses the stored `nn.Linear` weights through `W.t()`, `Block` scales pass-2 updates (+1, −1 or learned α), and `GPT.forward` routes odd passes to it. With the field at `None`, every existing arm executes exactly the operations it does today; a golden-logits fixture recorded before any change pins that. The symmetry measurement is a separate pure module (`hallm.symmetry`) plus a script.

**Tech Stack:** Python 3.14, PyTorch 2.13, pytest, uv. `transformers` + `pillow` in a new `analysis` dependency group, for DeiT-small only.

**Spec:** `docs/superpowers/specs/2026-09-15-transposed-loop-design.md`

## Global Constraints

- Default path unchanged: `loop_pass2=None` must give bit-identical logits for A0, A2, A2-attn and A1u2 (spec §7 invariant). No change to training mathematics for any existing arm.
- Old checkpoints and resume files must still load (a missing `loop_pass2` key defaults to `None`).
- Run IDs: `L8-A1u4t-s1337`, `L8-A1u4n-s1337`, `L8-A1u4a-s1337`; arm tags `A1u<k>t` (transpose), `A1u<k>n` (negate), `A1u<k>a` (scaled).
- Recipe: the `TRAIN` dict in `scripts/gen_ladder_configs.py`, verbatim; seed 1337.
- α: one scalar per sublayer per block, `nn.Parameter(torch.ones(()))`, never weight-decayed (already true: `configure_optimizer` decays only `ndim >= 2`).
- Follow-up-seed threshold (used in Task 9 only): seed-1337 `val_ppl` ≤ 28.4431.
- Work happens in the worktree `~/Dev/hallm-tloop` on branch `transposed-loop`. Never launch GPU jobs from the worktree. `main` is not touched until Task 9.
- Run tests on CPU while the GPU trains: `CUDA_VISIBLE_DEVICES= uv run pytest ...`.
- Python via `uv` only (no pip). Every commit message ends with:
  ```
  Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01JLzx4KhEpsAuZ2ivU8fKso
  ```
- `tests/fixtures/prepatch_logits.pt` is written once in Task 1 and never regenerated.

---

## File map

| file | change | responsibility |
|---|---|---|
| `tests/prepatch_fixture.py` | create | regenerates the golden logits (run once, Task 1) |
| `tests/fixtures/prepatch_logits.pt` | create | golden logits of the pre-amendment code |
| `tests/test_transposed_loop.py` | create | all model-side tests for the amendment |
| `src/hallm/model/config.py` | modify | `loop_pass2` field, validation, tags |
| `src/hallm/model/sharing.py` | modify | `transposed=` path in `MLP` and `CausalSelfAttention` |
| `src/hallm/model/gpt.py` | modify | `Block` pass-2 scaling + α, `GPT.forward` routing, `loop_scales()` |
| `src/hallm/train.py` | modify | α values in each metrics record |
| `src/hallm/runqueue.py` | modify | final α values in the result row |
| `scripts/gen_ladder_configs.py` | modify | `--transposed` → 3 configs + `queue-transposed.txt` |
| `scripts/build_reports.py` | modify | 9 new comparison rows |
| `src/hallm/symmetry.py` | create | rotation share + FFN input collectors (hallm and HF ViT) |
| `scripts/ffn_symmetry.py` | create | runs the measurement, writes JSON + report |
| `tests/test_symmetry.py` | create | symmetry tests |
| `README.md`, `.gitignore`, `pyproject.toml`, `uv.lock` | modify | arm docs, image data ignore, analysis group |

---

### Task 1: Worktree and golden-logits fixture

**Files:**
- Create: `tests/prepatch_fixture.py`, `tests/fixtures/prepatch_logits.pt`, `tests/test_transposed_loop.py`

**Interfaces:**
- Produces: `prepatch_fixture.make() -> dict[str, torch.Tensor]` (arm tag → logits of shape (2, 16, 256)); `prepatch_fixture.CASES`.

- [ ] **Step 1: Create the worktree** (superpowers:using-git-worktrees)

```bash
cd ~/Dev/hallm && git worktree add ../hallm-tloop -b transposed-loop && cd ../hallm-tloop && uv sync --group dev
```
Expected: a new directory `~/Dev/hallm-tloop` on branch `transposed-loop`; `uv sync` finishes (torch comes from the uv cache).

- [ ] **Step 2: Write the fixture generator** — `tests/prepatch_fixture.py`

```python
"""Golden logits of the model code as it was BEFORE the transposed-loop amendment (spec 2026-09-15
§7 invariant: with loop_pass2=None every existing arm runs exactly the same operations as before).

Written ONCE by `uv run python tests/prepatch_fixture.py` before the model is touched. Never
regenerate it afterwards, or test_default_path_is_bit_identical loses its meaning."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import torch

from hallm.model import GPT, SHAPES, arm_config

SMOKE = SHAPES["smoke"]
CASES = {"A0": SMOKE, "A2": SMOKE, "A2-attn": SMOKE, "A1u2": dataclasses.replace(SMOKE, n_layer=4)}
FIXTURE = Path(__file__).parent / "fixtures" / "prepatch_logits.pt"


def make() -> dict[str, torch.Tensor]:
    out = {}
    for arm, base in CASES.items():
        torch.manual_seed(0)
        model = GPT(arm_config(base, arm)).eval()
        x = torch.randint(0, base.vocab_size, (2, 16), generator=torch.Generator().manual_seed(1))
        with torch.no_grad():
            logits, _ = model(x, x)   # with targets → logits for every position
        out[arm] = logits
    return out


if __name__ == "__main__":
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    torch.save(make(), FIXTURE)
    print(f"wrote {FIXTURE}")
```

- [ ] **Step 3: Generate the fixture with the unmodified code**

Run: `CUDA_VISIBLE_DEVICES= uv run python tests/prepatch_fixture.py`
Expected: `wrote .../tests/fixtures/prepatch_logits.pt`

- [ ] **Step 4: Write the regression test** — create `tests/test_transposed_loop.py`

```python
"""Transposed loop (spec 2026-09-15): config, sublayer transposition, pass routing, α, and the
invariant that every pre-existing arm is untouched."""

from __future__ import annotations

import dataclasses

import pytest
import torch
import torch.nn.functional as F

import prepatch_fixture
from hallm.model import GPT, SHAPES, arm_config
from hallm.model.config import ModelConfig

SMOKE = SHAPES["smoke"]
D = SMOKE.n_embd


def _deep(L: int) -> ModelConfig:
    return dataclasses.replace(SMOKE, n_layer=L)


def test_default_path_is_bit_identical():
    """loop_pass2=None ⇒ the exact logits the pre-amendment code produced (CPU, float32)."""
    golden = torch.load(prepatch_fixture.FIXTURE, weights_only=True)
    now = prepatch_fixture.make()
    assert set(golden) == set(now) == set(prepatch_fixture.CASES)
    for arm in golden:
        assert torch.equal(golden[arm], now[arm]), arm
```

- [ ] **Step 5: Run it**

Run: `CUDA_VISIBLE_DEVICES= uv run pytest tests/test_transposed_loop.py -v`
Expected: 1 passed.

- [ ] **Step 6: Commit**

```bash
git add tests/prepatch_fixture.py tests/fixtures/prepatch_logits.pt tests/test_transposed_loop.py
git commit -m "test: golden logits of the pre-amendment model (transposed-loop invariant)"
```

---

### Task 2: Config field, validation and arm tags

**Files:**
- Modify: `src/hallm/model/config.py` (dataclass fields ~l.42–48, `__post_init__` ~l.50–60, `arm` ~l.70–84, `_LOOPED`/`arm_config` ~l.98–112)
- Modify: `README.md` (arms list, ~l.150–152)
- Test: `tests/test_transposed_loop.py`

**Interfaces:**
- Produces: `ModelConfig.loop_pass2: str | None` (`None`, `"transpose"`, `"negate"`, `"scaled"`); `config.LOOP_PASS2 = {"transpose": "t", "negate": "n", "scaled": "a"}`; `arm_config(base, "A1u<k>[t|n|a]")`; `ModelConfig.arm` returns `"A1u<k>t"` etc. whenever `loop_pass2` is set (also for k = 1).

- [ ] **Step 1: Write the failing tests** — append to `tests/test_transposed_loop.py`

```python
@pytest.mark.parametrize("suffix,mode", [("t", "transpose"), ("n", "negate"), ("a", "scaled")])
def test_tag_round_trip(suffix, mode):
    cfg = arm_config(_deep(4), f"A1u2{suffix}")
    assert cfg.loop_pass2 == mode and cfg.n_unique_blocks == 2 and cfg.share_cross_layer
    assert cfg.arm == f"A1u2{suffix}"


def test_k1_transposed_tag_keeps_its_suffix():
    assert arm_config(SMOKE, "A1u1t").arm == "A1u1t"
    assert arm_config(SMOKE, "A1u1").arm == "A1"   # unchanged plain behaviour


def test_plain_loop_has_no_pass2():
    cfg = arm_config(_deep(4), "A1u2")
    assert cfg.loop_pass2 is None and cfg.arm == "A1u2"


def test_loop_pass2_requires_a_looped_arm():
    with pytest.raises(ValueError):
        dataclasses.replace(SMOKE, loop_pass2="transpose")


def test_loop_pass2_rejects_intra_layer_sharing():
    looped = arm_config(_deep(4), "A1u2")
    with pytest.raises(ValueError):
        dataclasses.replace(looped, share_intra_ffn=True, loop_pass2="transpose")


def test_loop_pass2_rejects_biases():
    looped = arm_config(dataclasses.replace(_deep(4), bias=True), "A1u2")
    with pytest.raises(ValueError):
        dataclasses.replace(looped, loop_pass2="transpose")


def test_loop_pass2_rejects_unknown_mode():
    looped = arm_config(_deep(4), "A1u2")
    with pytest.raises(ValueError):
        dataclasses.replace(looped, loop_pass2="sideways")


def test_non_looped_arm_resets_loop_pass2():
    assert arm_config(arm_config(_deep(4), "A1u2t"), "A0").loop_pass2 is None
    assert arm_config(arm_config(_deep(4), "A1u2t"), "A1u2").loop_pass2 is None


def test_pre_amendment_config_still_loads():
    d = dataclasses.asdict(SMOKE)
    d.pop("loop_pass2")
    assert ModelConfig(**d).loop_pass2 is None
```

- [ ] **Step 2: Run them to verify they fail**

Run: `CUDA_VISIBLE_DEVICES= uv run pytest tests/test_transposed_loop.py -v`
Expected: FAIL (`unknown arm 'A1u2t'`, `unexpected keyword argument 'loop_pass2'`, `KeyError: 'loop_pass2'`).

- [ ] **Step 3: Implement** — in `src/hallm/model/config.py`

Above `@dataclass(frozen=True)` add:
```python
# Transposed loop (spec 2026-09-15): how a looped arm's odd passes add their transposed update.
# Value → run-ID suffix: A1u<k>t / A1u<k>n / A1u<k>a.
LOOP_PASS2 = {"transpose": "t", "negate": "n", "scaled": "a"}
```
After the `sharing_warmup_steps` field add:
```python
    loop_pass2: str | None = None     # transposed loop: odd passes use Wᵀ ("transpose"|"negate"|"scaled")
```
At the end of `__post_init__` add:
```python
        if self.loop_pass2 is not None:
            if self.loop_pass2 not in LOOP_PASS2:
                raise ValueError(f"loop_pass2 must be one of {sorted(LOOP_PASS2)}, got {self.loop_pass2!r}")
            if self.n_unique_blocks is None:
                raise ValueError("loop_pass2 requires a looped arm (n_unique_blocks set)")
            if self.share_intra_ffn or self.share_intra_attn:
                raise ValueError("loop_pass2 cannot be combined with intra-layer (W+Wᵀ) sharing")
            if self.bias:
                raise ValueError("loop_pass2 assumes bias=False (the transposed pass has no bias)")
```
At the top of the `arm` property body add:
```python
        if self.loop_pass2 is not None:
            return f"A1u{self.n_unique_blocks}{LOOP_PASS2[self.loop_pass2]}"
```
Replace `_LOOPED` and `arm_config`:
```python
_LOOPED = re.compile(r"A1u(\d+)([tna]?)")
_PASS2_FOR_SUFFIX = {suffix: mode for mode, suffix in LOOP_PASS2.items()}


def arm_config(base: ModelConfig, arm: str) -> ModelConfig:
    """Return a copy of ``base`` with the sharing flags set for the named ``arm``.

    ``A1u<k>`` is the looped arm (spec 2026-09-14 §2): A1's cross-layer flag with k distinct blocks
    cycled to depth L. ``A1u<k>t|n|a`` is the transposed loop (spec 2026-09-15 §2): odd passes use the
    blocks' transposed weights, added, subtracted or scaled by a learned α. The shape
    (vocab/block/embd/layer/head/ffn) is preserved exactly.
    """
    m = _LOOPED.fullmatch(arm)
    if m:
        return replace(base, **ARMS["A1"], n_unique_blocks=int(m.group(1)),
                       loop_pass2=_PASS2_FOR_SUFFIX.get(m.group(2)))
    if arm not in ARMS:
        raise KeyError(f"unknown arm {arm!r}; choose from {sorted(ARMS)} or A1u<k>[t|n|a]")
    return replace(base, **ARMS[arm], n_unique_blocks=None, loop_pass2=None)
```

- [ ] **Step 4: README arms list** — after the line `` `A1u<k>` looped: k blocks cycled to depth L. `` add:
```
`A1u<k>t|n|a` transposed loop: odd passes reuse the blocks with Wᵀ — added, subtracted, or
scaled by a learned α (spec 2026-09-15).
```

- [ ] **Step 5: Run the tests**

Run: `CUDA_VISIBLE_DEVICES= uv run pytest tests/test_transposed_loop.py tests/test_sharing.py -v`
Expected: all pass (including `test_default_path_is_bit_identical`).

- [ ] **Step 6: Commit**

```bash
git add src/hallm/model/config.py README.md tests/test_transposed_loop.py
git commit -m "feat(config): loop_pass2 field and A1u<k>t|n|a arm tags"
```

---

### Task 3: Transposed path in the sublayers

**Files:**
- Modify: `src/hallm/model/sharing.py` (`MLP.forward`, `CausalSelfAttention.forward`)
- Test: `tests/test_transposed_loop.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `MLP.forward(x, transposed: bool = False)`: up = W_downᵀ, down = W_upᵀ. `CausalSelfAttention.forward(x, transposed: bool = False)`: Q, K, V, O each use their own weight transposed. Both raise `ValueError` when `transposed=True` on a W+Wᵀ (shared) sublayer.

- [ ] **Step 1: Write the failing tests** — append

```python
from hallm.model.sharing import MLP, CausalSelfAttention


def test_transposed_mlp_matches_manual():
    torch.manual_seed(0)
    mlp = MLP(SMOKE)
    x = torch.randn(2, 5, D)
    expected = F.gelu(x @ mlp.proj.weight) @ mlp.fc.weight   # up = W_downᵀ, down = W_upᵀ
    assert torch.allclose(mlp(x, transposed=True), expected, atol=1e-6)


def test_untransposed_mlp_unchanged():
    torch.manual_seed(0)
    mlp = MLP(SMOKE)
    x = torch.randn(2, 5, D)
    assert torch.equal(mlp(x), mlp(x, transposed=False))


def test_transposed_attention_matches_manual():
    torch.manual_seed(0)
    attn = CausalSelfAttention(SMOKE)
    x = torch.randn(2, 5, D)
    B, T, C, H = 2, 5, D, SMOKE.n_head

    def heads(t):
        return t.view(B, T, H, C // H).transpose(1, 2)

    q, k, v = (heads(x @ lin.weight) for lin in (attn.q, attn.k, attn.v))   # x @ W = F.linear(x, Wᵀ)
    y = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, C)
    expected = y @ attn.proj.weight
    assert torch.allclose(attn(x, transposed=True), expected, atol=1e-6)


def test_transposed_path_adds_no_parameters():
    n_mlp = sum(p.numel() for p in MLP(SMOKE).parameters())
    assert n_mlp == 2 * D * SMOKE.ffn_hidden


def test_transposed_path_rejects_halvit_sublayers():
    x = torch.randn(1, 3, D)
    with pytest.raises(ValueError):
        MLP(dataclasses.replace(SMOKE, share_intra_ffn=True))(x, transposed=True)
    with pytest.raises(ValueError):
        CausalSelfAttention(dataclasses.replace(SMOKE, share_intra_attn=True))(x, transposed=True)
```

- [ ] **Step 2: Run to verify failure**

Run: `CUDA_VISIBLE_DEVICES= uv run pytest tests/test_transposed_loop.py -k "transposed_mlp or transposed_attention or transposed_path" -v`
Expected: FAIL with `forward() got an unexpected keyword argument 'transposed'`.

- [ ] **Step 3: Implement `MLP.forward`**

```python
    def forward(self, x: torch.Tensor, transposed: bool = False) -> torch.Tensor:
        if transposed:
            # Transposed loop (spec 2026-09-15 §2): the SAME two tensors in swapped, transposed roles —
            # up = W_downᵀ (d→h), down = W_upᵀ (h→d). No copy, no new parameter.
            if self.shared:
                raise ValueError("transposed pass is defined for unshared FFNs only")
            a = F.gelu(F.linear(x, self.proj.weight.t()))   # x @ W_down: (..., d) → (..., h)
            y = F.linear(a, self.fc.weight.t())             # a @ W_up:   (..., h) → (..., d)
            return self.dropout(y)
        if self.shared:
            a = F.gelu(F.linear(x, self.w, self.up_bias))      # up:   x @ Wᵀ  → (..., h)
            y = F.linear(a, self.w.t(), self.down_bias)        # down: a @ W   → (..., d)  (Wᵀ applied)
        else:
            a = F.gelu(self.fc(x))
            y = self.proj(a)
        return self.dropout(y)
```

- [ ] **Step 4: Implement `CausalSelfAttention.forward`**

Change the signature to `def forward(self, x: torch.Tensor, transposed: bool = False) -> torch.Tensor:` and add as its first line:
```python
        if transposed and self.shared:
            raise ValueError("transposed pass is defined for unshared attention only")
```
Replace `        else:\n            q, k, v = self.q(x), self.k(x), self.v(x)` with:
```python
        elif transposed:
            # Transposed loop (spec 2026-09-15 §2): each projection uses its own weight transposed.
            q = F.linear(x, self.q.weight.t())
            k = F.linear(x, self.k.weight.t())
            v = F.linear(x, self.v.weight.t())
        else:
            q, k, v = self.q(x), self.k(x), self.v(x)
```
Replace the output projection block with:
```python
        if self.shared:
            y = F.linear(y, self.w_q.t(), self.out_bias)   # Out = W_qᵀ · ŷ
        elif transposed:
            y = F.linear(y, self.proj.weight.t())
        else:
            y = self.proj(y)
```

- [ ] **Step 5: Run the tests**

Run: `CUDA_VISIBLE_DEVICES= uv run pytest tests/test_transposed_loop.py tests/test_sharing.py tests/test_smoke.py -v`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/hallm/model/sharing.py tests/test_transposed_loop.py
git commit -m "feat(sharing): transposed path for unshared MLP and attention"
```

---

### Task 4: Block scaling, α and pass routing

**Files:**
- Modify: `src/hallm/model/gpt.py` (`Block`, `GPT.forward`, new `GPT.loop_scales`)
- Modify: `tests/test_smoke.py` (`ARMS` list)
- Test: `tests/test_transposed_loop.py`

**Interfaces:**
- Consumes: `MLP/CausalSelfAttention.forward(x, transposed=...)` (Task 3); `cfg.loop_pass2` (Task 2).
- Produces: `Block.forward(x, transposed: bool = False)`; `Block.alpha_attn`, `Block.alpha_mlp` (0-dim `nn.Parameter`, only when `loop_pass2 == "scaled"`); `GPT.loop_scales() -> dict[str, float]` with keys `alpha_attn_<j>`, `alpha_mlp_<j>` (empty dict unless `"scaled"`). Layer i uses block `i % k`, transposed iff `loop_pass2` is set and `(i // k) % 2 == 1`.

- [ ] **Step 1: Write the failing tests** — append

```python
from hallm.model.gpt import Block


def _params(model):
    return model.num_parameters()


@pytest.mark.parametrize("suffix", ["t", "n"])
def test_zero_param_variants_match_the_plain_loop(suffix):
    assert _params(GPT(arm_config(_deep(4), f"A1u2{suffix}"))) == _params(GPT(arm_config(_deep(4), "A1u2")))


def test_scaled_variant_adds_two_scalars_per_block():
    plain = _params(GPT(arm_config(_deep(4), "A1u2")))
    assert _params(GPT(arm_config(_deep(4), "A1u2a"))) == plain + 2 * 2


def test_odd_passes_are_transposed():
    model = GPT(arm_config(_deep(4), "A1u2t"))
    calls = []
    for j, block in enumerate(model.blocks):
        block.register_forward_pre_hook(
            lambda m, args, kwargs, j=j: calls.append((j, kwargs.get("transposed", False))),
            with_kwargs=True)
    model(torch.zeros(1, 8, dtype=torch.long))
    assert calls == [(0, False), (1, False), (0, True), (1, True)]


def test_plain_loop_never_transposes():
    model = GPT(arm_config(_deep(4), "A1u2"))
    calls = []
    for block in model.blocks:
        block.register_forward_pre_hook(
            lambda m, args, kwargs: calls.append(kwargs.get("transposed", False)), with_kwargs=True)
    model(torch.zeros(1, 8, dtype=torch.long))
    assert calls == [False] * 4


def _pair(a: str, b: str):
    """Two models with identical stored weights (b gets a's tensors; α stays at init)."""
    torch.manual_seed(0)
    ma = GPT(arm_config(_deep(4), a))
    mb = GPT(arm_config(_deep(4), b))
    mb.load_state_dict(ma.state_dict(), strict=False)
    return ma, mb


def _logits(model):
    x = torch.randint(0, SMOKE.vocab_size, (2, 12), generator=torch.Generator().manual_seed(3))
    return model(x, x)[0]


def test_transposed_pass_changes_the_function():
    plain, t = _pair("A1u2", "A1u2t")
    assert not torch.allclose(_logits(plain), _logits(t))


def test_scaled_equals_transpose_at_init():
    t, a = _pair("A1u2t", "A1u2a")
    assert torch.equal(_logits(t), _logits(a))


def test_negate_subtracts_each_pass2_update():
    torch.manual_seed(0)
    blk = Block(arm_config(_deep(4), "A1u2n"))
    x = torch.randn(2, 6, D)
    x1 = x - blk.attn(blk.ln1(x), transposed=True)
    expected = x1 - blk.mlp(blk.ln2(x1), transposed=True)
    assert torch.allclose(blk(x, transposed=True), expected, atol=1e-6)


@pytest.mark.parametrize("suffix", ["t", "n", "a"])
def test_gradients_reach_every_stored_matrix(suffix):
    torch.manual_seed(0)
    model = GPT(arm_config(_deep(4), f"A1u2{suffix}"))
    x = torch.randint(0, SMOKE.vocab_size, (2, 12))
    model(x, x)[1].backward()
    for name, p in model.blocks.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, name


def test_loop_scales_reports_alpha_only_for_scaled():
    scales = GPT(arm_config(_deep(4), "A1u2a")).loop_scales()
    assert scales == {"alpha_attn_0": 1.0, "alpha_mlp_0": 1.0, "alpha_attn_1": 1.0, "alpha_mlp_1": 1.0}
    assert GPT(arm_config(_deep(4), "A1u2t")).loop_scales() == {}
    assert GPT(arm_config(SMOKE, "A0")).loop_scales() == {}
```

- [ ] **Step 2: Run to verify failure**

Run: `CUDA_VISIBLE_DEVICES= uv run pytest tests/test_transposed_loop.py -v`
Expected: the new tests FAIL (e.g. `loop_scales` missing, `forward() got an unexpected keyword argument 'transposed'`, call list all `False`).

- [ ] **Step 3: Implement `Block`** — replace the class in `src/hallm/model/gpt.py`

```python
class Block(nn.Module):
    """Pre-LN transformer block: x + attn(LN(x)); x + mlp(LN(x)).

    Transposed loop (spec 2026-09-15): on a transposed pass the sublayers use their weights
    transposed and each update is multiplied by +1 ("transpose"), −1 ("negate") or a learned
    per-sublayer α initialised to 1 ("scaled"). LayerNorm gains are shared by both passes."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)
        self.pass2 = cfg.loop_pass2
        if cfg.loop_pass2 == "scaled":
            self.alpha_attn = nn.Parameter(torch.ones(()))
            self.alpha_mlp = nn.Parameter(torch.ones(()))

    def _pass2_scales(self):
        if self.pass2 == "negate":
            return -1.0, -1.0
        if self.pass2 == "scaled":
            return self.alpha_attn, self.alpha_mlp
        return 1.0, 1.0

    def forward(self, x: torch.Tensor, transposed: bool = False) -> torch.Tensor:
        if not transposed:
            x = x + self.attn(self.ln1(x))
            x = x + self.mlp(self.ln2(x))
            return x
        s_attn, s_mlp = self._pass2_scales()
        x = x + s_attn * self.attn(self.ln1(x), transposed=True)
        x = x + s_mlp * self.mlp(self.ln2(x), transposed=True)
        return x
```
(The untransposed path is pinned by `test_default_path_is_bit_identical`.)

- [ ] **Step 4: Implement routing and `loop_scales`** — in `GPT.forward` replace the layer loop with:

```python
        n_unique = len(self.blocks)
        for i in range(self.cfg.n_layer):
            block = self.blocks[i % n_unique]   # cross-layer: n_unique==1 ⇒ block 0 reused L times
            if self.cfg.loop_pass2 is not None and (i // n_unique) % 2 == 1:
                x = block(x, transposed=True)   # transposed loop: odd passes use Wᵀ (spec 2026-09-15)
            else:
                x = block(x)
```
and add the method after `forward`:
```python
    def loop_scales(self) -> dict[str, float]:
        """Learned pass-2 multipliers of the transposed loop's `a` variant; empty for every other arm."""
        if self.cfg.loop_pass2 != "scaled":
            return {}
        out: dict[str, float] = {}
        for j, block in enumerate(self.blocks):
            out[f"alpha_attn_{j}"] = round(float(block.alpha_attn), 6)
            out[f"alpha_mlp_{j}"] = round(float(block.alpha_mlp), 6)
        return out
```

- [ ] **Step 5: Extend the smoke tests** — in `tests/test_smoke.py` change `ARMS = ["A0", "A1", "A2", "A3"]` to:
```python
ARMS = ["A0", "A1", "A2", "A3", "A1u1t", "A1u1n", "A1u1a"]   # A1u1<x>: layer 0 W, layer 1 Wᵀ (L=2)
```

- [ ] **Step 6: Run the whole suite**

Run: `CUDA_VISIBLE_DEVICES= uv run pytest -q`
Expected: all pass, including `test_default_path_is_bit_identical` and the overfit-a-batch smoke test for the three new arms.

- [ ] **Step 7: Commit**

```bash
git add src/hallm/model/gpt.py tests/test_transposed_loop.py tests/test_smoke.py
git commit -m "feat(model): transposed-loop pass routing, negate/scaled updates, loop_scales()"
```

---

### Task 5: Log α during training and in the result row

**Files:**
- Modify: `src/hallm/train.py` (logging block, after `rec = {"step": ...}`)
- Modify: `src/hallm/runqueue.py` (`run_one`, after `row["dataset"] = ...`)
- Test: `tests/test_runqueue.py`

**Interfaces:**
- Consumes: `GPT.loop_scales()` (Task 4).
- Produces: `metrics.jsonl` records and `results/runs/<id>.json` rows carry `alpha_attn_<j>` / `alpha_mlp_<j>` for `A1u<k>a` runs only.

- [ ] **Step 1: Write the failing test** — append to `tests/test_runqueue.py`

```python
def test_scaled_transposed_loop_logs_alpha(tmp_path):
    data_dir, _, _ = _setup(tmp_path)
    name = "smoke-A1u1a-s7"
    cfg = tmp_path / f"{name}.yaml"
    cfg.write_text(yaml.safe_dump({"shape": "smoke", "arm": "A1u1a",
                                   "train": {**SMOKE_TRAIN, "out_dir": str(tmp_path / "runs" / name)}}))
    row = run_one(cfg, data_dir, device="cpu")
    recs = [json.loads(x) for x in (tmp_path / "runs" / name / "metrics.jsonl").read_text().splitlines()]
    assert all("alpha_attn_0" in r and "alpha_mlp_0" in r for r in recs)
    assert "alpha_attn_0" in row and "alpha_mlp_0" in row


def test_other_arms_log_no_alpha(tmp_path):
    data_dir, cfgs, _ = _setup(tmp_path)
    row = run_one(cfgs[0], data_dir, device="cpu")
    recs = [json.loads(x) for x in (tmp_path / "runs" / "smoke-A0-s7" / "metrics.jsonl").read_text().splitlines()]
    assert not any(k.startswith("alpha_") for r in recs for k in r)
    assert not any(k.startswith("alpha_") for k in row)
```

- [ ] **Step 2: Run to verify failure**

Run: `CUDA_VISIBLE_DEVICES= uv run pytest tests/test_runqueue.py -k alpha -v`
Expected: `test_scaled_transposed_loop_logs_alpha` FAILS (`alpha_attn_0` missing); `test_other_arms_log_no_alpha` passes.

- [ ] **Step 3: Implement** — in `src/hallm/train.py`, directly after `rec = {"step": step, "loss": loss_accum, "lr": lr}` add:
```python
            rec.update(model.loop_scales() if hasattr(model, "loop_scales") else {})  # transposed-loop α
```
In `src/hallm/runqueue.py`, directly after `row["dataset"] = train_cfg.dataset` add:
```python
    row.update(model.loop_scales())   # transposed loop `a` variant: final α per sublayer (else empty)
```

- [ ] **Step 4: Run the tests**

Run: `CUDA_VISIBLE_DEVICES= uv run pytest tests/test_runqueue.py tests/test_metrics_logging.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/hallm/train.py src/hallm/runqueue.py tests/test_runqueue.py
git commit -m "feat(train): log transposed-loop α in metrics.jsonl and the result row"
```

---

### Task 6: Configs, queue and report comparisons

**Files:**
- Modify: `scripts/gen_ladder_configs.py` (new section after the pilot section; `main()`)
- Modify: `scripts/build_reports.py` (`COMPARISONS`)
- Test: `tests/test_ladder_configs.py`, `tests/test_reports.py`

**Interfaces:**
- Consumes: arm tags from Task 2 (via `load_experiment` → `arm_config`).
- Produces: `generate_transposed(out_dir) -> list[str]`; `configs/runs/L8-A1u4{t,n,a}-s1337.yaml`; `configs/runs/queue-transposed.txt`; report rows `| L8-A1u4t | L8-A1u4 | matched |` etc.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_ladder_configs.py`

```python
# --- Transposed loop (spec 2026-09-15 §2) ----------------------------------------------------

def test_generate_transposed_order_and_flags(tmp_path):
    from scripts.gen_ladder_configs import generate_transposed

    queue = generate_transposed(tmp_path)
    assert [p.split("/")[-1] for p in queue] == [
        "L8-A1u4t-s1337.yaml", "L8-A1u4n-s1337.yaml", "L8-A1u4a-s1337.yaml"]
    assert (tmp_path / "queue-transposed.txt").read_text().splitlines() == queue
    for suffix, mode in (("t", "transpose"), ("n", "negate"), ("a", "scaled")):
        mc, tc = load_experiment(tmp_path / f"L8-A1u4{suffix}-s1337.yaml")
        assert (mc.n_layer, mc.n_unique_blocks, mc.loop_pass2, mc.arm, tc.seed) == \
            (8, 4, mode, f"A1u4{suffix}", 1337)


def test_transposed_recipe_matches_the_plain_loop_run(tmp_path):
    """Everything but the run directory equals the completed L8-A1u4-s1337 run's config."""
    import dataclasses as dc

    from scripts.gen_ladder_configs import generate_transposed

    generate_transposed(tmp_path)
    _, loop = load_experiment("configs/runs/L8-A1u4-s1337.yaml")
    for suffix in "tna":
        _, tc = load_experiment(tmp_path / f"L8-A1u4{suffix}-s1337.yaml")
        assert dc.replace(tc, out_dir=loop.out_dir) == loop, suffix
```
Append to `tests/test_reports.py`:
```python
def test_transposed_loop_rows_use_the_hl_rule():
    rows = []
    for s, (t, loop) in zip((1337, 1338, 1339), ((27.8, 28.2), (27.9, 28.2), (27.8, 28.2))):
        rows += [{"run": f"L8-A1u4t-s{s}", "val_ppl": t}, {"run": f"L8-A1u4-s{s}", "val_ppl": loop}]
    lines = _build_reports().build_iso_storage(rows).splitlines()
    ht = next(l for l in lines if l.startswith("| L8-A1u4t | L8-A1u4 | matched |"))
    assert "L8-A1u4t better" in ht and "| 3 |" in ht
    for a in ("L8-A1u4t", "L8-A1u4n", "L8-A1u4a"):
        for b in ("L8-A1u4", "L4-A0", "L8-A2"):
            assert any(l.startswith(f"| {a} | {b} |") for l in lines), (a, b)
```

- [ ] **Step 2: Run to verify failure**

Run: `CUDA_VISIBLE_DEVICES= uv run pytest tests/test_ladder_configs.py tests/test_reports.py -k "transposed" -v`
Expected: FAIL (`cannot import name 'generate_transposed'`; `StopIteration` for the missing report row).

- [ ] **Step 3: Implement the generator** — in `scripts/gen_ladder_configs.py`, after `generate_pilot`:

```python
# --- Transposed loop (spec 2026-09-15 §2, exploratory) ----------------------------------------
# Three pass-2 variants of the plain loop L8-A1u4 at identical storage and FLOPs, the pilot recipe
# verbatim. Seed 1337 only: follow-up seeds are added by a later commit if spec §3's rule fires.
TRANSPOSED = [("L8", "s30", "A1u4t"), ("L8", "s30", "A1u4n"), ("L8", "s30", "A1u4a")]


def generate_transposed(out_dir: str | Path) -> list[str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    queue: list[str] = []
    for rung, shape, arm in TRANSPOSED:
        name = f"{rung}-{arm}-s1337"
        spec = {"shape": shape, "arm": arm,
                "train": {**TRAIN, "seed": 1337, "out_dir": f"runs/ladder/{name}"}}
        (out / f"{name}.yaml").write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
        queue.append(str(out / f"{name}.yaml"))
    (out / "queue-transposed.txt").write_text("\n".join(queue) + "\n", encoding="utf-8")
    return queue
```
In `main()` add the flag and branch (before `if args.bridge:`):
```python
    ap.add_argument("--transposed", action="store_true",
                    help="generate the transposed-loop configs + queue-transposed.txt")
```
```python
    if args.transposed:
        queue = generate_transposed(args.out)
        print(f"wrote {len(queue)} transposed-loop configs to {args.out}/, queue-transposed.txt lists them")
        return
```
(`ap.add_argument` goes with the other `add_argument` calls, before `args = ap.parse_args()`.)

- [ ] **Step 4: Implement the report rows** — in `scripts/build_reports.py` append to `COMPARISONS`:

```python
    # transposed loop (spec 2026-09-15 §3, exploratory): H-T vs the plain loop, then the usual pairs
    ("L8-A1u4t", "L8-A1u4", "matched"),
    ("L8-A1u4n", "L8-A1u4", "matched"),
    ("L8-A1u4a", "L8-A1u4", "matched"),
    ("L8-A1u4t", "L4-A0", "iso-storage"),
    ("L8-A1u4n", "L4-A0", "iso-storage"),
    ("L8-A1u4a", "L4-A0", "iso-storage"),
    ("L8-A1u4t", "L8-A2", "matched"),
    ("L8-A1u4n", "L8-A2", "matched"),
    ("L8-A1u4a", "L8-A2", "matched"),
```

- [ ] **Step 5: Generate the real configs and run the tests**

Run: `uv run python scripts/gen_ladder_configs.py --transposed && CUDA_VISIBLE_DEVICES= uv run pytest -q`
Expected: `wrote 3 transposed-loop configs ...`; all tests pass. `git status` shows only the 3 new yaml files and `queue-transposed.txt` under `configs/runs/` (no existing config modified).

- [ ] **Step 6: Commit**

```bash
git add scripts/gen_ladder_configs.py scripts/build_reports.py tests/test_ladder_configs.py tests/test_reports.py \
        configs/runs/L8-A1u4t-s1337.yaml configs/runs/L8-A1u4n-s1337.yaml configs/runs/L8-A1u4a-s1337.yaml \
        configs/runs/queue-transposed.txt
git commit -m "feat(pilot): transposed-loop configs, queue and report comparisons"
```

---

### Task 7: Symmetry core (`hallm.symmetry`)

**Files:**
- Create: `src/hallm/symmetry.py`
- Test: `tests/test_symmetry.py`

**Interfaces:**
- Consumes: `MLP` internals (`shared`, `w`, `fc`, `proj`) and the `transposed` kwarg (Tasks 3–4).
- Produces:
  - `gelu_grad(z: Tensor) -> Tensor` (exact GELU derivative)
  - `antisymmetric_share(J: Tensor) -> Tensor`, J (n, d, d) → (n,)
  - `rotation_share(w_up: Tensor, w_down: Tensor, u: Tensor, b_up: Tensor | None = None, chunk: int = 32) -> float`, where w_up is (h, d), w_down is (d, h) and u is (N, d)
  - `effective_ffn_weights(mlp: MLP, transposed: bool) -> tuple[Tensor, Tensor]` → (w_up (h, d), w_down (d, h))
  - `hallm_rotation_shares(model: GPT, idx: LongTensor, n_positions: int = 256, seed: int = 0) -> list[float]`, one value per layer (FFN call) in depth order

- [ ] **Step 1: Write the failing tests** — `tests/test_symmetry.py`

```python
"""FFN Jacobian symmetry (spec 2026-09-15 §5)."""

from __future__ import annotations

import dataclasses

import torch
import torch.nn.functional as F

from hallm.model import GPT, SHAPES, arm_config
from hallm.symmetry import (antisymmetric_share, effective_ffn_weights, gelu_grad,
                            hallm_rotation_shares, rotation_share)

SMOKE = SHAPES["smoke"]


def test_gelu_grad_matches_autograd():
    z = torch.linspace(-6, 6, 101, dtype=torch.float64, requires_grad=True)
    (g,) = torch.autograd.grad(F.gelu(z).sum(), z)
    assert torch.allclose(gelu_grad(z.detach()), g, atol=1e-10)


def test_antisymmetric_share_extremes():
    a = torch.randn(3, 8, 8, dtype=torch.float64)
    assert torch.allclose(antisymmetric_share(a - a.transpose(1, 2)), torch.ones(3, dtype=torch.float64))
    assert torch.allclose(antisymmetric_share(a + a.transpose(1, 2)), torch.zeros(3, dtype=torch.float64))


def test_halvit_ffn_is_exactly_symmetric():
    torch.manual_seed(0)
    w = torch.randn(64, 16)            # (h, d): up = W, down = Wᵀ
    u = torch.randn(40, 16)
    assert rotation_share(w, w.t(), u) < 1e-20


def test_random_ffn_is_about_half_rotation():
    torch.manual_seed(0)
    share = rotation_share(torch.randn(256, 64), torch.randn(64, 256), torch.randn(64, 64))
    assert 0.4 < share < 0.6


def test_bias_shifts_the_gate():
    torch.manual_seed(0)
    w_up, w_down, u = torch.randn(64, 16), torch.randn(16, 64), torch.randn(20, 16)
    assert rotation_share(w_up, w_down, u) != rotation_share(w_up, w_down, u, b_up=torch.full((64,), 3.0))


def test_collector_matches_a_direct_computation():
    torch.manual_seed(0)
    model = GPT(arm_config(SMOKE, "A0")).eval()
    idx = torch.randint(0, SMOKE.vocab_size, (2, 10))
    shares = hallm_rotation_shares(model, idx, n_positions=20)
    with torch.no_grad():
        x = model.tok_emb(idx) + model.pos_emb(torch.arange(10))
        for layer, block in enumerate(model.blocks):
            x = x + block.attn(block.ln1(x))
            u = block.ln2(x).reshape(-1, SMOKE.n_embd)
            direct = rotation_share(block.mlp.fc.weight, block.mlp.proj.weight, u)
            assert abs(shares[layer] - direct) < 1e-9
            x = x + block.mlp(block.ln2(x))


def test_collector_gives_zero_for_a_w_plus_wt_model():
    model = GPT(arm_config(SMOKE, "A2")).eval()
    shares = hallm_rotation_shares(model, torch.randint(0, SMOKE.vocab_size, (2, 10)), n_positions=20)
    assert len(shares) == SMOKE.n_layer and max(shares) < 1e-20


def test_collector_uses_transposed_weights_on_pass_two():
    model = GPT(arm_config(SMOKE, "A1u1t")).eval()          # L=2: layer 0 W, layer 1 Wᵀ
    mlp = model.blocks[0].mlp
    up, down = effective_ffn_weights(mlp, transposed=True)
    assert torch.equal(up, mlp.proj.weight.t()) and torch.equal(down, mlp.fc.weight.t())
    shares = hallm_rotation_shares(model, torch.randint(0, SMOKE.vocab_size, (2, 10)), n_positions=20)
    assert len(shares) == 2
```

- [ ] **Step 2: Run to verify failure**

Run: `CUDA_VISIBLE_DEVICES= uv run pytest tests/test_symmetry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'hallm.symmetry'`.

- [ ] **Step 3: Implement** — `src/hallm/symmetry.py`

```python
"""FFN Jacobian symmetry (spec 2026-09-15 §5): how much of an FFN's input Jacobian is antisymmetric.

A W+Wᵀ FFN y = Wᵀ·GELU(W·u) has J = Wᵀ·D·W, exactly symmetric for any diagonal D, so it cannot
produce the antisymmetric ("rotation-like") part of an update. `rotation_share` measures how much of
that part a trained FFN actually uses: ‖½(J − Jᵀ)‖²_F / ‖J‖²_F at real inputs u, where
J(u) = W_down · diag(GELU′(W_up·u + b_up)) · W_up. Reference points: 0 for W+Wᵀ, ~0.5 random, 1 pure
rotation. Computed in float64: for a symmetric J the numerator is a difference of near-equal terms.
"""

from __future__ import annotations

import math

import torch

from hallm.model.sharing import MLP


def gelu_grad(z: torch.Tensor) -> torch.Tensor:
    """Exact GELU derivative Φ(z) + z·φ(z) (matches F.gelu with approximate='none')."""
    return 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0))) + z * torch.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def antisymmetric_share(J: torch.Tensor) -> torch.Tensor:
    """(n, d, d) → (n,): ‖½(J − Jᵀ)‖²_F / ‖J‖²_F per matrix."""
    A = 0.5 * (J - J.transpose(1, 2))
    return A.pow(2).sum((1, 2)) / J.pow(2).sum((1, 2))


@torch.no_grad()
def rotation_share(w_up: torch.Tensor, w_down: torch.Tensor, u: torch.Tensor,
                   b_up: torch.Tensor | None = None, chunk: int = 32) -> float:
    """Mean rotation share over the N inputs. w_up (h, d) maps d→h; w_down (d, h) maps h→d; u (N, d)."""
    w_up, w_down, u = w_up.double(), w_down.double(), u.double()
    pre = u @ w_up.T
    if b_up is not None:
        pre = pre + b_up.double()
    g = gelu_grad(pre)                                                   # (N, h)
    shares = [antisymmetric_share(torch.einsum("ih,nh,hj->nij", w_down, g[i:i + chunk], w_up))
              for i in range(0, u.shape[0], chunk)]
    return float(torch.cat(shares).mean())


def effective_ffn_weights(mlp: MLP, transposed: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """(w_up (h, d), w_down (d, h)) as the FFN actually applies them on this call."""
    if mlp.shared:                 # HaLViT: up = W, down = Wᵀ
        return mlp.w, mlp.w.t()
    if transposed:                 # transposed loop pass 2: up = W_downᵀ, down = W_upᵀ
        return mlp.proj.weight.t(), mlp.fc.weight.t()
    return mlp.fc.weight, mlp.proj.weight


@torch.no_grad()
def hallm_rotation_shares(model, idx: torch.Tensor, n_positions: int = 256, seed: int = 0) -> list[float]:
    """One rotation share per layer (FFN call, depth order), each over n_positions of the FFN's real
    inputs (the LN2 outputs), subsampled with a fixed seed from all B·T positions of `idx`."""
    calls: list[tuple[MLP, torch.Tensor, bool]] = []
    hooks = [block.mlp.register_forward_pre_hook(
                 lambda m, args, kwargs: calls.append((m, args[0].detach(), kwargs.get("transposed", False))),
                 with_kwargs=True)
             for block in model.blocks]
    try:
        model(idx)
    finally:
        for h in hooks:
            h.remove()
    gen = torch.Generator().manual_seed(seed)
    out = []
    for mlp, x, transposed in calls:
        u = x.reshape(-1, x.shape[-1])
        pick = torch.randperm(u.shape[0], generator=gen)[:n_positions].to(u.device)
        w_up, w_down = effective_ffn_weights(mlp, transposed)
        out.append(rotation_share(w_up, w_down, u[pick]))
    return out
```
Note: `model(idx)` without targets only computes logits for the last position, but every layer still runs on all positions, so the hooks see all B·T inputs.

- [ ] **Step 4: Run the tests**

Run: `CUDA_VISIBLE_DEVICES= uv run pytest tests/test_symmetry.py -v`
Expected: all 9 pass.

- [ ] **Step 5: Commit**

```bash
git add src/hallm/symmetry.py tests/test_symmetry.py
git commit -m "feat(symmetry): FFN Jacobian rotation share and hallm input collector"
```

---

### Task 8: DeiT collector and the measurement script

**Files:**
- Modify: `src/hallm/symmetry.py` (add `vit_rotation_shares`)
- Create: `scripts/ffn_symmetry.py`
- Modify: `pyproject.toml` + `uv.lock` (group `analysis`), `.gitignore`
- Test: `tests/test_symmetry.py`

**Interfaces:**
- Consumes: `rotation_share` (Task 7); `hallm.train.build_model_from_checkpoint`; `hallm.data.load_bin`.
- Produces: `vit_rotation_shares(vit_model, pixel_values: Tensor, n_positions: int = 256, seed: int = 0) -> list[float]`, where `vit_model` is an HF `ViTModel` (has `.encoder.layer[i].intermediate.dense` / `.output.dense` and `.config.hidden_act`). Also `scripts/ffn_symmetry.py main(argv: list[str] | None = None)`, which writes `results/analysis/ffn-symmetry.json` (list of rows `{"model", "kind", "arm", "per_layer", "mean"}`) and `results/reports/ffn-symmetry.md`.

- [ ] **Step 1: Add the dependency group and ignore the image data**

```bash
uv add --group analysis transformers pillow
```
Append to `.gitignore` after `data/blimp/`:
```
data/imagenette2-160/
data/imagenette2-160.tgz
```

- [ ] **Step 2: Write the failing tests** — append to `tests/test_symmetry.py`

```python
import json

import numpy as np
import pytest


def test_vit_collector_on_a_tiny_random_vit():
    transformers = pytest.importorskip("transformers")
    from hallm.symmetry import vit_rotation_shares

    cfg = transformers.ViTConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                                 intermediate_size=64, image_size=32, patch_size=8, hidden_act="gelu")
    torch.manual_seed(0)
    vit = transformers.ViTModel(cfg).eval()
    shares = vit_rotation_shares(vit, torch.randn(3, 3, 32, 32), n_positions=30)
    assert len(shares) == 2 and all(0.0 < s < 1.0 for s in shares)


def test_vit_collector_rejects_a_non_gelu_model():
    transformers = pytest.importorskip("transformers")
    from hallm.symmetry import vit_rotation_shares

    cfg = transformers.ViTConfig(hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                                 intermediate_size=64, image_size=32, patch_size=8, hidden_act="relu")
    with pytest.raises(ValueError):
        vit_rotation_shares(transformers.ViTModel(cfg).eval(), torch.randn(1, 3, 32, 32))


def test_script_writes_json_and_report_for_an_lm_checkpoint(tmp_path):
    from hallm.data import make_synthetic_data
    from hallm.train import TrainConfig, save_checkpoint
    from scripts.ffn_symmetry import main

    cfg = arm_config(SMOKE, "A0")
    save_checkpoint(GPT(cfg), cfg, TrainConfig(), tmp_path / "smoke-A0-s7.pt")
    data = tmp_path / "data"
    data.mkdir()
    make_synthetic_data(SMOKE.vocab_size, 4096, seed=0).tofile(data / "val.bin")
    out, rep = tmp_path / "sym.json", tmp_path / "sym.md"
    main(["--checkpoints", str(tmp_path / "*.pt"), "--data", str(data), "--n-positions", "32",
          "--windows", "2", "--out", str(out), "--report", str(rep), "--device", "cpu"])
    rows = json.loads(out.read_text())
    lm = next(r for r in rows if r["model"] == "smoke-A0-s7")
    assert lm["kind"] == "lm" and len(lm["per_layer"]) == SMOKE.n_layer
    assert any(r["model"] == "random" for r in rows)
    assert "smoke-A0-s7" in rep.read_text()
```

- [ ] **Step 3: Run to verify failure**

Run: `CUDA_VISIBLE_DEVICES= uv run --group analysis pytest tests/test_symmetry.py -v`
Expected: the 3 new tests FAIL (`cannot import name 'vit_rotation_shares'`, `No module named 'scripts.ffn_symmetry'`).

- [ ] **Step 4: Implement `vit_rotation_shares`** — append to `src/hallm/symmetry.py`

```python
@torch.no_grad()
def vit_rotation_shares(vit_model, pixel_values: torch.Tensor, n_positions: int = 256,
                        seed: int = 0) -> list[float]:
    """Same measure for a Hugging Face ViTModel (DeiT-small loads as one): the FFN input is the
    `layernorm_after` output, i.e. the input of `intermediate.dense` (W_up, biased); the down
    projection is `output.dense`. Requires exact GELU so gelu_grad is the right derivative."""
    if vit_model.config.hidden_act != "gelu":
        raise ValueError(f"expected hidden_act='gelu', got {vit_model.config.hidden_act!r}")
    layers = vit_model.encoder.layer
    inputs: dict[int, list[torch.Tensor]] = {i: [] for i in range(len(layers))}
    hooks = [layer.intermediate.dense.register_forward_pre_hook(
                 lambda m, args, i=i: inputs[i].append(args[0].detach().reshape(-1, args[0].shape[-1])))
             for i, layer in enumerate(layers)]
    try:
        vit_model(pixel_values=pixel_values)
    finally:
        for h in hooks:
            h.remove()
    gen = torch.Generator().manual_seed(seed)
    out = []
    for i, layer in enumerate(layers):
        u = torch.cat(inputs[i])
        pick = torch.randperm(u.shape[0], generator=gen)[:n_positions].to(u.device)
        up, down = layer.intermediate.dense, layer.output.dense
        out.append(rotation_share(up.weight, down.weight, u[pick], b_up=up.bias))
    return out
```

If the installed `transformers` names the ViT submodules differently (`intermediate.dense`, `output.dense`, `encoder.layer`), adapt the attribute paths: the tiny-ViT test catches a mismatch.

- [ ] **Step 5: Implement the script** — `scripts/ffn_symmetry.py`

```python
"""FFN Jacobian symmetry over trained checkpoints (spec 2026-09-15 §5). Inference-only.

LM checkpoints: inputs are `--windows` random WikiText-103 validation windows. DeiT-small (--deit):
~256 Imagenette validation images, fetched once:
    curl -L -o data/imagenette2-160.tgz https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz
    tar -xzf data/imagenette2-160.tgz -C data/
Needs the analysis group:  uv run --group analysis python scripts/ffn_symmetry.py ...

Usage:
  uv run --group analysis python scripts/ffn_symmetry.py \
      --checkpoints 'runs/ladder/L*-A0-s13??/*-s13??.pt' 'runs/ladder/L8-A2-s1337/*.pt' \
      --data data --deit --images data/imagenette2-160/val
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch

from hallm.data import load_bin
from hallm.symmetry import hallm_rotation_shares, rotation_share
from hallm.train import build_model_from_checkpoint


def _row(model: str, kind: str, arm: str, per_layer: list[float]) -> dict:
    return {"model": model, "kind": kind, "arm": arm, "per_layer": [round(s, 6) for s in per_layer],
            "mean": round(float(np.mean(per_layer)), 6)}


def lm_rows(patterns: list[str], data_dir: str, n_positions: int, windows: int, device: str) -> list[dict]:
    val = load_bin(Path(data_dir) / "val.bin")
    rows = []
    for path in sorted(p for pat in patterns for p in glob.glob(pat)):
        if Path(path).stem == "resume":
            continue
        model, cfg = build_model_from_checkpoint(path, map_location=device)
        model.to(device).eval()
        T = cfg.block_size
        starts = np.random.default_rng(0).integers(0, len(val) - T, size=windows)
        idx = torch.from_numpy(np.stack([np.asarray(val[s:s + T], dtype=np.int64) for s in starts])).to(device)
        rows.append(_row(Path(path).stem, "lm", cfg.arm, hallm_rotation_shares(model, idx, n_positions)))
        print(rows[-1]["model"], rows[-1]["mean"])
    return rows


def deit_row(images_dir: str, n_images: int, n_positions: int, device: str) -> dict:
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    from hallm.symmetry import vit_rotation_shares

    name = "facebook/deit-small-patch16-224"
    proc, vit = AutoImageProcessor.from_pretrained(name), AutoModel.from_pretrained(name).to(device).eval()
    files = sorted(Path(images_dir).rglob("*.JPEG"))
    pick = np.random.default_rng(0).choice(len(files), size=min(n_images, len(files)), replace=False)
    images = [Image.open(files[i]).convert("RGB") for i in sorted(pick)]
    pixels = proc(images=images, return_tensors="pt")["pixel_values"].to(device)
    return _row("deit-small-patch16-224", "vision", "A0", vit_rotation_shares(vit, pixels, n_positions))


def random_row(d: int = 256, h: int = 1024, n: int = 64) -> dict:
    g = torch.Generator().manual_seed(0)
    share = rotation_share(torch.randn(h, d, generator=g), torch.randn(d, h, generator=g),
                           torch.randn(n, d, generator=g))
    return _row("random", "reference", "—", [share])


def report(rows: list[dict]) -> str:
    L = ["<!-- GENERATED by scripts/ffn_symmetry.py — do not edit -->",
         "# FFN Jacobian rotation share (spec 2026-09-15 §5)", "",
         "‖½(J − Jᵀ)‖² / ‖J‖² of each FFN's input Jacobian at real inputs. 0 = W+Wᵀ (exact), "
         "~0.5 = random matrices, 1 = pure rotation. Cross-model comparisons are uncontrolled "
         "(different recipe, scale, data): mechanism evidence, not a test.", "",
         "| model | kind | arm | layers | mean | per layer |", "|---|---|---|---|---|---|"]
    for r in rows:
        per = ", ".join(f"{s:.3f}" for s in r["per_layer"])
        L.append(f"| {r['model']} | {r['kind']} | {r['arm']} | {len(r['per_layer'])} | {r['mean']:.3f} | {per} |")
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="FFN Jacobian rotation share")
    ap.add_argument("--checkpoints", nargs="*", default=[])
    ap.add_argument("--data", default="data")
    ap.add_argument("--windows", type=int, default=8)
    ap.add_argument("--n-positions", type=int, default=256)
    ap.add_argument("--deit", action="store_true")
    ap.add_argument("--images", default="data/imagenette2-160/val")
    ap.add_argument("--n-images", type=int, default=256)
    ap.add_argument("--out", default="results/analysis/ffn-symmetry.json")
    ap.add_argument("--report", default="results/reports/ffn-symmetry.md")
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    rows = lm_rows(args.checkpoints, args.data, args.n_positions, args.windows, device)
    if args.deit:
        rows.append(deit_row(args.images, args.n_images, args.n_positions, device))
    rows.append(random_row())
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(report(rows), encoding="utf-8")
    print(f"wrote {args.out} and {args.report}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run the tests**

Run: `CUDA_VISIBLE_DEVICES= uv run --group analysis pytest tests/test_symmetry.py -v && CUDA_VISIBLE_DEVICES= uv run pytest -q`
Expected: all symmetry tests pass with the analysis group. The full suite passes without it (the two ViT tests are skipped only if `transformers` is absent).

- [ ] **Step 7: Commit**

```bash
git add src/hallm/symmetry.py scripts/ffn_symmetry.py tests/test_symmetry.py pyproject.toml uv.lock .gitignore
git commit -m "feat(symmetry): DeiT collector and ffn_symmetry.py measurement script"
```

---

### Task 9: Merge, launch, measure (operational, main checkout)

**Files:**
- Modify: `main` (merge), `results/analysis/ffn-symmetry.json`, `results/reports/ffn-symmetry.md` (generated)

**Interfaces:**
- Consumes: everything above; the pilot queue must have finished.

- [ ] **Step 1: Wait for the A2attn seeds.** Proceed only when both hold:
  - `ls ~/Dev/hallm/results/runs/L10-A2attn-s1339.json` exists
  - `pgrep -f '\.venv/bin/python3 scripts/run_queue.py'` prints nothing

  Commit those results first (manifest copy + `build_reports.py`), as for every pilot run.

- [ ] **Step 2: Final review, then merge into main** (superpowers:finishing-a-development-branch)

```bash
cd ~/Dev/hallm-tloop && CUDA_VISIBLE_DEVICES= uv run --group analysis pytest -q
cd ~/Dev/hallm && git merge --no-ff transposed-loop -m "Merge transposed-loop: arms t/n/a + FFN symmetry measurement"
CUDA_VISIBLE_DEVICES= uv run pytest -q
```
Expected: all tests pass in the worktree and again on `main` after the merge.

- [ ] **Step 3: Launch the three runs with the watchdog** (from `~/Dev/hallm`, in the background)

```bash
env -u HALLM_GIT_COMMIT systemd-inhibit --what=sleep:idle --who=hallm --why=transposed \
  uv run python scripts/run_queue.py --queue configs/runs/queue-transposed.txt --data data \
  --results-dir results/runs >> runs-transposed.log 2>&1
```
Arm the same watchdog as the pilot, with `LOG=runs-transposed.log`. Check after the first 1000 steps: `runs/ladder/L8-A1u4t-s1337/metrics.jsonl` exists and its loss is finite.

- [ ] **Step 4: Fetch the images and run the symmetry measurement** (GPU is shared but this is inference-only; run it on CPU if VRAM is tight: `--device cpu`)

```bash
cd ~/Dev/hallm
curl -L -o data/imagenette2-160.tgz https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz
tar -xzf data/imagenette2-160.tgz -C data/
uv run --group analysis python scripts/ffn_symmetry.py \
  --checkpoints 'runs/ladder/L4-A0-s13??/L4-A0-s13??.pt' 'runs/ladder/L8-A0-s13??/L8-A0-s13??.pt' \
                'runs/ladder/L16-A0-s13??/L16-A0-s13??.pt' 'runs/ladder/L8-A2-s1337/L8-A2-s1337.pt' \
  --data data --deit --images data/imagenette2-160/val --device cpu
```
Expected: rows for 8 LM checkpoints, `L8-A2-s1337` with mean < 1e-12, DeiT-small, and `random` near 0.5. If the Imagenette URL fails, use any public set of ≥256 natural photographs, and record the source in the report.

- [ ] **Step 5: Commit the measurement**

```bash
git add results/analysis/ffn-symmetry.json results/reports/ffn-symmetry.md
git commit -m "results: FFN Jacobian rotation share — LM unshared checkpoints vs DeiT-small"
```

- [ ] **Step 6: As each transposed run lands** — copy its manifest, run `build_reports.py`, commit (the pilot pattern), then run the probe suite on the three checkpoints once all have landed:

```bash
uv run python scripts/capability_eval.py --checkpoints runs/ladder/L8-A1u4?-s1337/L8-A1u4?-s1337.pt \
  --lambada data/lambada_test.jsonl --blimp data/blimp --data data --probes --out results/capability
```
Apply spec §3's follow-up rule (seed-1337 `val_ppl` ≤ 28.4431 → add seeds 1338/1339 to `TRANSPOSED` in a separate commit), and record the outcome in the wiki (`~/Dev/wiki/projects/hallm.md`, `tasks.md`).

- [ ] **Step 7: Remove the worktree**

```bash
cd ~/Dev/hallm && git worktree remove ../hallm-tloop && git branch -d transposed-loop
```
