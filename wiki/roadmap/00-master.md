# ROADMAP — Does W+Wᵀ Weight Sharing Generalize to Language Models?

> **Master plan for Term 2 (Fall 2026).** Single source of truth that turns the accepted Part-1
> Interim Report into an executable research + engineering program. Self-contained; the `roadmap/`
> folder holds the deep-dive companions. Built from the frozen report (`report/00`–`04`), the thesis
> proposal (`proposals/option-2-halvit-language.md`), and the external correctness audit
> (`codex-report.md`). Where this document and the proposal disagree, **the codex corrections win**
> (see §11).

**Project:** Advanced Model Compression Techniques for Resource-Constrained AI Architectures
**Team:** Alper Düzgün, Hazar Utku Sözer · **Advisor:** Prof. Dr. Behçet Uğur Töreyin · **Institution:** ITU AI & Data Engineering

---

## 1. North Star

Answer one question with a clean, controlled experiment:

> **Does HaLViT's intra-layer W+Wᵀ weight-sharing mechanism — which halved Vision Transformer
> parameters — transfer to autoregressive language models, and how does it compare to (and stack
> with) ALBERT-style cross-layer sharing?**

The deliverable is **the first controlled four-way comparison** of intra-layer (width-axis) and
cross-layer (depth-axis) sharing on a language model, all arms trained **from scratch under a matched
compute budget** so any perplexity difference is attributable to the sharing scheme alone.

### What success looks like
- **Primary success:** a complete, reproducible four-arm table (perplexity · parameter count · size ·
  GFLOPs) on WikiText-103, with the per-sublayer ablation, produced on a single RTX 4070.
- **"Breakthrough" outcome:** HaLViT-style intra-layer sharing matches or beats ALBERT-style sharing
  at equal *non-embedding* parameter budget, **and** the combined arm stays within the <2% perplexity
  gate — i.e. the two axes stack. That is a workshop-submittable positive result (EMNLP/ACL/ICML
  Efficient-NLP tracks).
- **A clean negative result is a valid contribution** and an explicit success mode: if the mechanism
  breaks under causal attention, the per-sublayer ablation localizes *where* and the column-space
  argument explains *why*. The thesis stands either way.

**Non-goals (locked):** no product, no UI, no edge/on-device deployment, no real-time serving, no
quantization/pruning experiments (composability is future work only). The "users" are the two
researchers and, via released configs, the community.

---

## 2. The Mechanism (state it correctly)

HaLViT ties a layer's two weight matrices as **W and Wᵀ**: one matrix is stored, the second
transformation is its transpose. The justification is a **column-space** argument: for a stored
W and a nonlinearity F, the activated output F(Wx) **generically** leaves the column space of W, so
WᵀF(Wx) acts as a genuinely independent transformation rather than collapsing into a linear reuse of
W. Two matrices become one; per-pair parameters halve.

- **FFN path (strong argument).** Standard `FFN(x) = W₂·GELU(W₁·x)` becomes `Wᵀ·GELU(W·x)` with
  `W₂ = W₁ᵀ`. GELU supplies a clean nonlinearity between the two applications, so the column-space
  argument transfers directly from the ViT case.
- **Attention path (weaker argument — say so).** With `K = W_kv·x`, `V = W_kvᵀ·x` (and `Q`/`Out`
  similarly), Keys and Values are both *linear* projections of the same `x` with **no explicit
  nonlinearity between them**; the only mixing is the downstream softmax, which under **causal
  masking** operates over a strictly lower-triangular window. Whether that suffices to keep Wᵀ
  independent is exactly the open empirical question (Risk **R1**). The roadmap treats FFN sharing as
  the robust hypothesis and attention sharing as the fragile one.

> **Correction applied (codex §High):** the proposal's claim that "HaLViT explicitly names language
> models as future work" is **not supported** by the HaLViT PDF. Correct framing: HaLViT validates
> W+Wᵀ on vision architectures and frames the method as potentially applicable beyond them; testing
> autoregressive LMs is a **natural extension, not a demonstrated result.** Use "can" / "generically"
> for the column-space claim, never "always" / "does."

See `roadmap/01-mechanism.md` for the formal per-component analysis (GELU FFN, causal attention,
RoPE/GQA/SwiGLU interactions).

---

## 3. The Four Arms — sharing scheme + parameter math

Notation: vocab `V`, context `T`, model dim `d`, layers `L`, FFN hidden `h = 4d`, multi-head
attention (no GQA in the core model). Biases/LayerNorms omitted (negligible). **Per standard block:**
attention `Q,K,V,O` = `4d²`; FFN `W₁,W₂` = `8d²`; **block = 12d²**. Embeddings: token `V·d`,
learned position `T·d`; **LM head weight-tied to the token embedding** (no extra params).

`Baseline params ≈ V·d + T·d + L·12d²`.

| Arm | Sharing scheme | Exactly which matrices are shared | Block params | Non-embedding ratio |
|-----|----------------|-----------------------------------|--------------|---------------------|
| **A0 — Baseline** | none | — | `12d²` | `1.00` (reference) |
| **A1 — ALBERT (cross-layer, depth)** | one block's weights reused across **all L layers** | every projection in every block ties to layer-0's copy | `12d²` total (not ×L) | `1/L` (≈ 8% at L=12) |
| **A2 — HaLViT (intra-layer, width)** | within each layer: `W₂ = W₁ᵀ` (FFN) and `V = K`-path / `Out = Q`-path transposes (attn) | FFN: 2→1 matrix. Attn: 4→2 matrices | `6d²` per layer | `1/2` (50%, depth-independent) |
| **A3 — Combined** | both: one shared block that is **itself** W+Wᵀ-halved | A1 ⊗ A2 | `6d²` total | `1/2L` (≈ 4% at L=12) |

### Derivations (the scientific crux — these become the `test_sharing.py` invariants)
- **A2 FFN:** `W₁ ∈ ℝ^{h×d}`, `W₂ = W₁ᵀ ∈ ℝ^{d×h}` → FFN `8d² → 4d²` (**50% FFN reduction, 2× per
  layer, independent of L**). Attention analog: store `W_kv, W_q`, derive `V, Out` as transposes →
  `4d² → 2d²`. Both sublayers shared → block `12d² → 6d²`.
- **A1:** L blocks collapse to one shared block → block params `L·12d² → 12d²` (**→ 1/L**).
- **A3:** one shared, halved block → `L·12d² → 6d²` (**→ 1/2L**; idealized **2L = 24× reduction** of
  affected block params at L=12).

### Whole-model reality (codex §High — embeddings dominate at small scale; do NOT overclaim)
The 8% / 4% figures are **non-embedding (block-only)**. The token-embedding table `V·d` is huge at
small scale and is **not shared**, so whole-model reduction is far milder. Worked example at GPT-2-124M
shape (`V=50257, d=768, T=1024, L=12`): embeddings ≈ 39.4M, blocks ≈ 84.9M, total ≈ 124M.

| Arm | Block-only ratio | **Whole-model params** | **Whole-model ratio** |
|-----|------------------|------------------------|-----------------------|
| A0 | 1.00 | ≈ 124.3M | 1.00 |
| A1 (ALBERT) | 1/L ≈ 8% | ≈ 46.5M | **≈ 37%** |
| A2 (HaLViT) | 1/2 = 50% | ≈ 81.9M | **≈ 66%** |
| A3 (Combined) | 1/2L ≈ 4% | ≈ 42.9M | **≈ 35%** |

> **Therefore: report _non-embedding_ parameter count as the headline sharing metric** (it is the axis
> the mechanism actually acts on), with whole-model params as a secondary column. Never present a
> single dramatic whole-model "Nx" figure (this is the origin of the rejected 52× claim). At L=12 the
> honest idealized affected-subblock reduction for A3 is **≈24× (2L), before embeddings/norms/biases**.

**Open design decision (→ WAKEUP / advisor):** ALBERT also *factorizes* the embedding table; replicating
only cross-layer sharing (not embedding factorization) isolates the depth-sharing variable but leaves
the embedding floor in place. Recommended: **cross-layer sharing only** for A1, and additionally
**report at a deeper/narrower shape** (larger L, smaller V via a 16k–32k BPE) in a secondary
configuration so blocks dominate and the sharing effect is visible whole-model. Logged in §11.

See `roadmap/02-arms-and-paramcount.md` for the full symbolic + numeric tables and the embedding-floor
analysis.

---

## 4. Architecture Decisions

Decoder-only GPT, trained from scratch. Each decision is justified and isolates the sharing variable.

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Family | **Decoder-only causal Transformer (nanoGPT-class)** | Smallest clean substrate to isolate sharing; standard FFN keeps the strong-argument FFN path central (`report/00-brief`, C4). |
| Scale | **`d/L/heads` range for ≈10M–124M params**; size **locked in Term 2** after a tokens/sec pilot | Matched-budget control matters more than absolute size (advisor "go smaller", R2). Candidate shapes in `roadmap/03-architecture.md`. |
| FFN activation | **GELU** for the core (standard 2-matrix FFN) | Cleanest column-space nonlinearity; the 50% FFN result is exact. SwiGLU (3→2 matrices, 33%) is **stretch only** (FR-9). |
| Positional | **Learned absolute** for the core; RoPE noted for the stretch | Keeps the core minimal; RoPE touches Q/K *after* the W+Wᵀ projection, so it is orthogonal to sharing (no conflict) — relevant only for TinyLlama. |
| Attention | **Plain MHA (no GQA)** in the core | GQA multiplies the roles a shared `W_kv` must play (most aggressive, most uncertain); keep it out of the controlled core, study in the stretch. |
| LM head | **Weight-tied to token embedding** | Standard; keeps embedding floor identical across all four arms so the comparison stays clean. |
| Norm | **Pre-LN, LayerNorm; LN params never shared** | Pre-LN trains stably at small scale; LN parameters kept independent per layer (SPIN warns normalization params are sharing-sensitive — codex §Medium). |

---

## 5. Training Protocol (matched budget = the core control)

All four arms train under an **identical** budget so the only independent variable is the sharing
scheme (FR-3, NFR-1, C3):

- **Data:** WikiText-103, **GPT-2 BPE** tokenizer (`tiktoken`); deterministic loader; fixed data order
  per seed (FR-4). Optional OpenWebText subset only if budget permits. A **tiny in-repo sample** ships
  for tests/smoke (no full-dataset dependency to run the harness).
- **Identical across arms:** depth/width, optimizer = **AdamW** (same β/wd), **cosine schedule + linear
  warmup**, max LR, batch/seq length, **total gradient steps / tokens-seen**, seed policy, dropout.
- **Reproducibility:** every run fully specified by one config file (model · sharing · seed · split ·
  hyperparams · software versions); seeds fixed; configs released (NFR-1, NFR-7).
- **Single-GPU envelope:** each arm trains to convergence on one **RTX 4070 (12 GB)** with peak
  activation **< 11 GB**; any memory trick (e.g. grad checkpointing) applied **identically** to all
  arms (NFR-3). Indicative **≤ 48 h/arm** wall-clock, confirmed after the pilot (NFR-4).
- **Stability hooks for R3:** optional **sharing warm-up** (enforce W+Wᵀ only after N steps) and
  reduced LR on shared layers, applied identically and reported if used.

> **Overnight constraint (this build):** the harness is implemented and **smoke-tested on CPU/tiny
> configs only**. The real WikiText-103 runs are a Term-2 activity, launched manually via the
> `run_real_training` script — never by the loop.

See `roadmap/04-training-protocol.md` for the config schema and candidate hyperparameters.

---

## 6. Evaluation Protocol

Report each metric **separately** — no opaque composite (codex §Medium):

- **Perplexity** (token-level, WikiText-103 test) — primary quality metric (lower = better).
- **Parameter count** — **non-embedding (headline)** and whole-model (secondary), in millions.
- **Model size (MB)** — FP32 and BF16.
- **GFLOPs / forward pass** — via `ptflops`/`fvcore` or an analytic counter (ITU-T F.748.11).
  **Note (important framing):** weight sharing here is *storage/memory* compression, **not** *compute*
  compression — all four arms run the identical matmul graph, so GFLOPs are expected to be ~equal
  across arms. The savings live in stored parameters, model size (MB), and (for A1/A3) weight +
  optimizer-state memory — which is what helps fit the 12 GB GPU (NFR-3) — **not** in forward FLOPs or
  inference latency. State this so the GFLOPs column is not misread as a speed-up.
- **Efficiency comparison:** plot **perplexity vs. non-embedding parameter count** and read off the
  **Pareto frontier** (G1). If a scalar "parameter efficiency" index is used it must be **explicitly
  defined** (e.g. `Δperplexity per million non-embedding params vs A0`) — never an unexplained
  "perplexity-per-parameter" number.
- **Degradation gate (G3, C2, NFR-2):** `(PPL_arm − PPL_A0)/PPL_A0`; an arm is "viable" only if
  ≤ 2%. Arms over the gate are still reported as compression-limit findings.

**Output artifact:** one structured table (JSON/CSV → human-readable markdown), one row per arm:
`scheme · non-emb params · whole-model params · size MB · GFLOPs · test PPL · Δ% vs A0` (FR-6).

See `roadmap/05-eval-protocol.md`.

---

## 7. Ablation Plan (FR-8, G4)

W+Wᵀ sharing is independently togglable for **attention** and **FFN** sublayers, giving:

| Config | Attn shared | FFN shared | Tests |
|--------|-------------|------------|-------|
| FFN-only | ✗ | ✓ | the **strong** column-space hypothesis |
| Attn-only | ✓ | ✗ | the **weak/fragile** path (causal-softmax mixing) |
| Both (= A2) | ✓ | ✓ | full intra-layer sharing |

**Hypothesis (stated as hypothesis, not fact — codex §Medium):** FFN sharing is less costly than
attention sharing. If A2-combined fails G3 but FFN-only passes, the recommended practical config is
**FFN-only sharing**. This ablation is the instrument that converts a negative full-A2 result into a
precise, publishable boundary finding (R1).

---

## 8. Risk Register (carried from `report/04`, R1–R5)

| ID | Risk | Mitigation (one-liner) |
|----|------|------------------------|
| **R1** | W+Wᵀ may not transfer under causal attention | Per-sublayer ablation localizes it; a clean negative is a valid contribution; fall back to FFN-only. |
| **R2** | Compute limits at the upper size range | Lock size after a tokens/sec pilot; shrink within range (conclusions are scale-invariant under matched budget). |
| **R3** | Training instability from sharing (Adam moment interaction) | Sharing warm-up; reduced LR on shared layers; LoRA-style enforcement; report instability as a finding. |
| **R4** | Cluster access denied → no TinyLlama stretch | Core four-way comparison is cluster-independent and self-sufficient; keep the SwiGLU *theory* as a written contribution. |
| **R5** | Concurrent work (e.g. Basis Sharing) erodes novelty | Novelty = the controlled matched-budget four-way *comparison* + causal isolation; literature-monitor at Term-2 start. |

---

## 9. Engineering Standards

- **ITU-T F.748.11** — DNN processor/accelerator metrics → all compression results reported as
  parameter count (M), size (MB), GFLOPs (FR-5).
- **ISO/IEC 22989:2022** — AI terminology → consistent use of "model compression / parameter sharing
  / parameter efficiency"; the **intra-layer (HaLViT) vs cross-layer (ALBERT)** distinction stated
  unambiguously.
- **NeurIPS/ML reproducibility checklist** (best-practice norm) — fixed seeds, public data, released
  configs, compute disclosed.

---

## 10. Timeline (Term 2, Fall 2026) → mapped to code phases

Week numbers relative to Term 2 start (mirrors `report/04` Time Plan & Gantt). **Code phase**
references this repo's build (§12 / Phase 2 of the harness).

| Weeks | Activity | Linked Req. | Code phase |
|-------|----------|-------------|------------|
| T1 | Literature survey & wiki consolidation (done) | — | — |
| 1–2 | Math-validity analysis (GELU/causal attn/RoPE/SwiGLU/GQA) | FR-7 | `roadmap/01-mechanism.md` formalized |
| 2 | System architecture & config-schema design | NFR-6 | `model/config.py` schema |
| 3 | Data pipeline (WikiText-103) | FR-4 | `data/wikitext.py` |
| 3–4 | W+Wᵀ sharing module (FFN) + ALBERT cross-layer module | FR-1, FR-2 | `model/sharing.py` |
| 4–5 | Four-arm training harness (matched budget, logging) | FR-3 | `train.py` |
| 5–6 | Four-way training runs (all arms) | FR-3/4, NFR-3/4 | `run_real_training` (Term 2, GPU) |
| 6 | Evaluation & metrics (PPL, params, GFLOPs, table) | FR-5/6 | `eval.py` |
| 7 | Reproducibility audit & seed/config release | NFR-1 | configs/ + audit |
| 7–10 | *Stretch:* TinyLlama SwiGLU + LoRA (gated on cluster) | FR-9 | stretch module |
| 8–9 | Analysis, interpretation & ablation write-up | G1–G4 | — |
| 9–11 | Final Report drafting & revision | — | — |
| 11–12 | Presentation prep; **final submission (wk 12)** | — | — |

---

## 11. Codex Corrections Baked In (correctness ledger)

Every High/Medium factual fix from `codex-report.md` that affects the plan, and where it lives here:

- **No 52× whole-model claim** → §3 reports block-only *and* whole-model ratios; idealized affected-
  subblock reduction is **2L ≈ 24×** at L=12. ✓
- **HaLViT does not name LMs as future work** → §2 reframed as "natural extension, not demonstrated." ✓
- **No "INT8 PTQ is lossless below 6.7B"** → quantization is out of scope; no such claim made. ✓
- **Column-space is generic, not universal** → §2 uses "generically / can." ✓
- **Pruning-mask `M_ij=M_ji` is wrong for rectangular W** → not asserted; pruning is future-work only. ✓
- **SPIN = ConvMixer BatchNorm caution, not a LayerNorm law** → §4 keeps LN unshared as a *precaution*. ✓
- **"ALBERT obsolete" too strong** → A1 framed as a legitimate controlled arm, not obsolete. ✓
- **No unexplained "perplexity-per-parameter" scalar** → §6 reports metrics separately + defined index. ✓
- **FFN-cheaper-than-attention is a hypothesis** → §7 labels it a hypothesis to test. ✓
- **SPIN bib + AWQ title fixes** → tracked for the Final Report bibliography (not load-bearing here). ✓

---

## 12. Codebase Architecture (bridge to the harness build)

Target layout (uv-managed; package name fixed in Phase 2). Each module maps to requirements:

```
pyproject.toml            # uv; torch, numpy, tiktoken, pyyaml|pydantic, tqdm, pytest, ptflops
src/<pkg>/
  model/config.py         # ModelConfig + 4 arm presets            → NFR-6, FR-3
  model/gpt.py            # GPT / Block / CausalSelfAttention / MLP → FR-3
  model/sharing.py        # W/Wᵀ intra-layer + ALBERT cross-layer   → FR-1, FR-2, FR-8  (THE CRUX)
  data/wikitext.py        # tokenize/splits/loaders (tiny-sample)   → FR-4
  train.py               # matched-budget loop, AdamW, cosine      → FR-3, NFR-3/4
  eval.py                # PPL, params(non-emb+total), MB, GFLOPs   → FR-5/6
  metrics.py             # param-count + FLOPs counters            → FR-5
configs/                  # arm0_none … arm3_both + smoke.yaml      → NFR-1
tests/test_sharing.py     # param-count invariants (§3 formulas)    → FR-7, NFR-5
tests/test_smoke.py       # 1–5 step CPU train, loss decreases      → FR-7
scripts/run_real_training # READY, not executed overnight           → FR-3
```

### Requirements-Traceability Matrix
| Req | What | Roadmap § | Code module | Verified by |
|-----|------|-----------|-------------|-------------|
| C1 | single-GPU trainable | §5 | train.py (envelope) | NFR-3 check |
| C2 | <2% PPL gate | §6 | eval.py | G3 |
| C3 | PyTorch + reproducible | §5 | configs/, seed util | NFR-1 |
| C4 | 10M–124M scale | §4 | model/config.py | pilot |
| C5 | stretch scope (TinyLlama, gated) | §4 | stretch module (deferred) | cluster access |
| C6 | late-start (Term-2 impl; this is planning) | §10 | — (timeline) | schedule |
| FR-1 | W+Wᵀ FFN module | §2,§3 | model/sharing.py | test_sharing |
| FR-2 | ALBERT cross-layer | §3 | model/sharing.py | test_sharing |
| FR-3 | four-arm harness | §5 | train.py + config presets | test_smoke |
| FR-4 | WikiText-103 pipeline | §5 | data/wikitext.py | test (tiny sample) |
| FR-5 | compression metrics | §6 | metrics.py / eval.py | eval output |
| FR-6 | four-way table | §6 | eval.py | artifact |
| FR-7 | gradient/param checks | §3 | tests/ | pytest |
| FR-8 | per-sublayer ablation | §7 | sharing.py flags | test_sharing |
| FR-9 | (stretch) SwiGLU/RoPE/GQA | §4 | stretch module | deferred |
| NFR-1 | reproducibility | §5 | configs/, seeds | audit |
| NFR-2 | <2% viability | §6 | eval.py | G3 |
| NFR-3 | single-GPU mem | §5 | train.py | runtime |
| NFR-4 | bounded train time | §5,§10 | train.py | pilot |
| NFR-5 | numerical correctness | §3 | sharing.py | test_sharing |
| NFR-6 | modularity | §12 | module split | review |
| NFR-7 | experiment logging | §5 | train.py logger | logs |
| G1–G4 | eval goals | §6,§7 | eval.py | artifacts |
| G5 | (stretch) SwiGLU/RoPE/GQA compat | §4 | stretch module (deferred) | cluster access |

---

## 13. Sources
Frozen report: `report/00-brief.md`, `report/01`–`04`*.md, `report/citation-pool.md`. Proposal:
`proposals/option-2-halvit-language.md`. Audit: `codex-report.md`. Wiki: `wiki/index.md`,
`wiki/analyses/halvit-vs-albert-cross-layer-sharing.md`, `wiki/sources/halvit.md`,
`wiki/concepts/weight-sharing.md`. Detail companions: `roadmap/01`–`05`.
