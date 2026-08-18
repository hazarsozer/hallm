# 05 — Evaluation Protocol & Comparison Table

> Companion to `ROADMAP.md` §6. Exact metric definitions, the FLOPs method, the <2% gate, and the
> four-way table schema `eval.py` emits (FR-5/6, G1–G3, ITU-T F.748.11). Metrics reported
> **separately** — no opaque composite (codex §Medium).

## 1. Metrics (exact definitions)

- **Perplexity** (primary, lower=better):
  `PPL = exp( (1/N) · Σ_t NLL(token_t) )` over the WikiText-103 **test** set, summed token-level
  cross-entropy with the model's own tokenizer, evaluated with a **non-overlapping** sliding window of
  length `block_size` (stride = block_size; document if strided differently). Same eval code for all
  arms.
- **Parameter count** (two columns, ITU-T F.748.11):
  - **non-embedding** (headline) = total − token-emb − pos-emb,
  - **whole-model** (secondary) = all unique stored params.
  Count **unique** params (a shared/tied tensor counted once).
- **Model size (MB):** `unique_params × bytes/param`, reported for FP32 (4 B) and BF16 (2 B).
- **GFLOPs / forward pass:** analytic counter (preferred, deterministic) cross-checked with
  `ptflops`/`fvcore`. Per-token matmul FLOPs ≈ `2 · (params actually applied in the forward pass)` +
  attention `2·L·T·d`; report the formula used. **Crucial:** *every* arm executes the same compute
  graph — A2 still does two FFN matmuls (`W` then `Wᵀ`); A1 still applies its shared block L times — so
  **all four arms have ~identical GFLOPs.** Weight sharing compresses *storage/memory*, not *compute*;
  the FLOPs column is reported for ITU-T F.748.11 completeness and is expected to be flat across arms.
  The real savings: stored params, size (MB), and — for A1/A3 — weight + Adam-state memory (helps fit
  the 12 GB GPU, NFR-3). Activation memory is unchanged. Do not present sharing as an inference speed-up.

## 2. Derived efficiency comparison (define it, don't ship a mystery scalar)
- Primary: **plot perplexity vs non-embedding parameter count**; the arm nearest the low-PPL/low-param
  **Pareto frontier** is strongest (G1).
- If a scalar index is quoted, define it: **`ΔPPL-per-Mparam = (PPL_arm − PPL_A0) / (nonemb_M_A0 − nonemb_M_arm)`**
  (perplexity cost per million non-embedding params saved). Never an unexplained "perplexity-per-parameter".

## 3. Degradation gate (G3, C2, NFR-2)
`deg_arm = (PPL_arm − PPL_A0) / PPL_A0`. An arm is **"viable"** iff `deg_arm ≤ 0.02`. Arms over the gate
are still reported (as compression-limit findings), just not recommended.

## 4. Comparison table schema (FR-6, the primary deliverable)

One row per arm, written to JSON/CSV and rendered to markdown:

| arm | scheme | nonemb_params(M) | total_params(M) | size_fp32(MB) | size_bf16(MB) | GFLOPs/fwd | test_PPL | Δ% vs A0 | viable |
|-----|--------|------------------|-----------------|---------------|---------------|------------|----------|---------|--------|
| A0 | none | … | … | … | … | … | … | 0.0% | — |
| A1 | ALBERT cross-layer | … | … | … | … | … | … | … | ✓/✗ |
| A2 | HaLViT intra-layer | … | … | … | … | … | … | … | ✓/✗ |
| A3 | combined | … | … | … | … | … | … | … | ✓/✗ |

Plus the **ablation rows** (FFN-only, attn-only) for G4 and a `PPL-vs-params` Pareto plot.

## 5. What `eval.py` does (use-case scenario 3)
1. Load a trained checkpoint + its frozen config.
2. Compute PPL on the held-out test set (deterministic).
3. Compute param counts (non-emb + total), sizes, GFLOPs via `metrics.py`.
4. Append a row to the structured results table; never require manual post-processing.
5. Emit the consolidated human-readable table + Pareto plot as artifacts.
