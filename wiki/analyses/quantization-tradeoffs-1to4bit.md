---
query: "What are the accuracy-compression trade-offs across quantization approaches at 1–4 bit, and how does HaLViT's W+Wᵀ weight sharing interact with each method?"
date: 2026-04-30
sources_consulted:
  - "wiki/sources/2210.17323.md"
  - "wiki/sources/2306.00978.md"
  - "wiki/sources/2306.03078.md"
  - "wiki/sources/2306.07629.md"
  - "wiki/sources/2307.13304.md"
  - "wiki/sources/2308.13137.md"
  - "wiki/concepts/quantization.md"
---

## Executive Summary

Six methods define the 1–4 bit quantization landscape for Transformer architectures: GPTQ, AWQ, SpQR, SqueezeLLM, QuIP, and OmniQuant (for post-training), plus BitNet b1.58 (for training-time quantization). The progression is: INT8 is essentially free for large models; INT4 is near-lossless with GPTQ or AWQ; INT3 is the first hard regime where method choice matters significantly (AWQ/SpQR/SqueezeLLM hold, GPTQ degrades at small scales); INT2 requires QuIP's incoherence principle; and 1.58-bit (ternary) requires training from scratch. Every post-training method faces a structurally identical challenge when applied to HaLViT's shared W+Wᵀ: their calibration signal was designed for single-use weight matrices, and must be extended to account for both the W→x and Wᵀ→x' pathways simultaneously.

---

## 1. The Shared-Weight Quantization Problem

### 1.1 Why Sharing Complicates Quantization

In standard PTQ, each weight matrix W is quantized to minimize its own layer's reconstruction error. The quantization error e = Q(W) − W affects exactly one forward pass: `y = Wx → y_q = Q(W)x`.

In HaLViT [arxiv:halvit], each shared tensor W is involved in two forward passes:
- `y = Wx` (key or forward-FFN path)
- `z = Wᵀx'` (value or output-FFN path)

Quantization error in W therefore propagates to **both** outputs simultaneously:
```
y_q = Q(W)x       error: ε_y = (Q(W) − W)x  
z_q = Q(W)ᵀx'     error: ε_z = (Q(W) − W)ᵀx'
```

The total quantization damage to a block is `‖ε_y‖ + ‖ε_z‖` — sum of errors across both uses. A standard PTQ method calibrated only on the W→x path will minimize `‖ε_y‖` while leaving `‖ε_z‖` uncontrolled. Whether the resulting quantization quality degrades significantly depends on the correlation between the two activation distributions — an empirical question specific to HaLViT's architecture.

### 1.2 The Three Joint Calibration Sub-problems

Every major PTQ method faces the same structural challenge in a different form:

| Method | Standard calibration | Shared-weight problem | Required extension |
|--------|---------------------|----------------------|-------------------|
| GPTQ | H = 2XX^T (forward path X) | H undercounts Wᵀ sensitivity | H_joint = 2(XX^T + X'(X')^T) |
| AWQ | s_X from forward path activations | Ignores Wᵀ→x' channel importance | s_{joint} from both X and X' |
| SmoothQuant | diag(s)·W for W→x path | diag(s)·W scales rows; (diag(s)·W)ᵀ = Wᵀ·diag(s) scales columns | Same s must satisfy both — analytically constrained |
| OmniQuant/LET | Block-level SGD on single path | Block objective integrates both paths if W+Wᵀ are in same block | Handled naturally if block contains both uses |
| QuIP | Random orthogonal pre/post-multiply | Same rotation used for both W and Wᵀ contexts | Rotation must simultaneously incoherence both — open question |

These sub-problems share the same mathematical structure: a calibration optimization designed for a single-path objective must be extended to a joint two-path objective. This is the defining technical challenge for quantizing HaLViT.

---

## 2. Bit-Width Progression: Accuracy vs Compression

### 2.1 INT8 — The Free Regime

At 8-bit precision, all standard PTQ methods achieve essentially lossless compression for Transformer models above a few hundred million parameters. LLM.int8() [arxiv:2208.07339] proves this with zero degradation on OPT-175B and BLOOM-176B. For HaLViT at 11M–22M parameters — far below the 6.7B outlier threshold — standard INT8 PTQ (layerwise or channelwise) is expected to be lossless without special treatment.

**Memory saving**: ~4× vs FP32; 2× vs FP16.
**Practical starting point for HaLViT**: INT8 PTQ with standard GPTQ or Jacob et al.'s QAT [arxiv:1712.05877] is the zero-risk baseline.

### 2.2 INT4 — The Near-Lossless Regime

At 4 bits, two methods dominate: **GPTQ** [arxiv:2210.17323] and **AWQ** [arxiv:2306.00978]. They are orthogonal and complementary.

**GPTQ at INT4** (LLaMA family, WikiText2 PPL, group-size 128):

| Model | FP16 | GPTQ | AWQ | Δ(AWQ−FP16) |
|-------|------|------|-----|-------------|
| LLaMA-7B | 5.68 | 6.22 | **5.78** | +0.10 |
| LLaMA-13B | 5.09 | 5.23 | **5.19** | +0.10 |
| LLaMA-30B | 4.10 | 4.22 | **4.21** | +0.11 |
| LLaMA-65B | 3.53 | 3.66 | **3.62** | +0.09 |
| Llama-2-70B | 3.32 | 3.42 | **3.41** | +0.09 |

AWQ is best at INT4 across all sizes. The gap to FP16 is minimal (≈0.10 PPL). GPTQ is close but slightly worse due to GPTQ's calibration domain sensitivity (+2.3–4.9 PPL shift under domain mismatch vs AWQ's +0.5–0.6).

**Memory saving**: ~8× vs FP32; 4× vs FP16. At INT4, OPT-175B fits on 1 GPU (vs 5 GPUs for FP16). Throughput: GPTQ 3.24× on A100, AWQ TinyChat 3.9× on RTX 4090.

**Composability verdict**: AWQ's activation-aware scaling is more robust and domain-stable. Both require joint calibration extensions for HaLViT (see §3). **AWQ is the recommended starting point for INT4 quantization of HaLViT.**

### 2.3 INT3 — The First Hard Regime

At 3 bits, the field splits sharply. RTN collapses completely (OPT-175B: 7,300 PPL). GPTQ holds for very large models but degrades noticeably at 7B scale. AWQ, SpQR, and SqueezeLLM hold reliably across all scales.

**GPTQ at INT3** (LLaMA-7B WikiText2):

| Model | FP16 | GPTQ g128 | AWQ g128 | SpQR | SqueezeLLM |
|-------|------|-----------|----------|------|------------|
| LLaMA-7B | 5.68 | 8.81 | **6.35** | <1% loss | **7.75** |
| LLaMA-65B | 3.53 | 4.17 | **3.95** | — | 7.56 |
| OPT-175B | 8.34 | **8.64** | — | — | — |

Key observation: **GPTQ at INT3 degrades severely at 7B scale (8.81 vs 5.68) but holds at 175B scale (8.64 vs 8.34)**. Overparameterization provides redundancy that compensates for quantization rounding errors. AWQ closes most of the gap at 7B (6.35) via its activation-aware channel scaling.

**SpQR** [arxiv:2306.03078]: near-lossless at 3–4 bit across all scales, including the 1–10B range where GPTQ struggles. Achieves this by identifying outlier weights during calibration and storing them in FP16, compressing all remaining weights to 3–4 bit. The outlier mask is sparse (<<1% of weights) so storage overhead is minimal. Custom GPU decoder gives 4× compression + 15% speedup vs FP16.

**SqueezeLLM** [arxiv:2306.07629]: matches SpQR's near-lossless result at 3-bit via a different approach — Hessian-guided non-uniform quantization bin placement + Dense-and-Sparse (D&S) decomposition. LLaMA-7B 3-bit: **7.75 PPL** vs GPTQ 9.85 — **2.1× better PPL gap** at the same memory budget. 2.3× throughput speedup on A6000. Key frame: LLM inference is memory-bandwidth-bound, not compute-bound, for single-token generation — reducing weight size reduces the bottleneck directly.

**Memory saving at INT3**: ~10.7× vs FP32; ~5.3× vs FP16.

### 2.4 INT2 — The Principled Regime

At 2 bits, all preceding methods fail except QuIP [arxiv:2307.13304].

**Why prior methods fail at INT2**: Weight and Hessian matrices are *coherent* — weights have outliers aligned with coordinate axes, and the quantization error directions are also coordinate-aligned. This means rounding errors are large and correlated. Simple scaling (SmoothQuant, AWQ) can reduce individual channel errors but cannot break the global coherence structure.

**QuIP's solution — Incoherence Processing**:
1. Pre-multiply W and H by random orthogonal matrices: `W̃ = U^T W V`, `H̃ = V^T H V`
2. Apply adaptive rounding (LDLQ = reformulated GPTQ) in the incoherent space
3. Post-multiply back: recover `W_q = U W̃_q V^T`

Incoherence processing scrambles the coordinate system so weights become approximately even in magnitude (no outliers) and rounding error directions are no longer aligned with any axis. The mathematical insight: a random Kronecker product of orthogonal matrices produces a uniformly distributed rotation — weights become incoherent with probability 1 in the limit.

**Results**: First viable 2-bit LLM quantization. At >2B parameters, the 2-bit vs 4-bit quality gap is small and shrinks with model size. LDLQ without incoherence = GPTQ (QuIP provides GPTQ's first theoretical guarantee). QuIP also unifies LLM.int8(), SmoothQuant, SpQR, and SqueezeLLM as heuristic approximations of the same incoherence principle.

**Memory saving at INT2**: ~16× vs FP32; ~8× vs FP16. A 22M-parameter HaLViT-M at 2-bit would occupy ~5.5MB — viable for microcontrollers.

### 2.5 OmniQuant — The Learnable Bridge (W2A16 through W4A4)

OmniQuant [arxiv:2308.13137] occupies a distinct position: it is not purely PTQ (it runs 1.6h of SGD) nor QAT (it uses only 128 calibration samples and does not retrain base weights). It learns two sets of quantization parameters while freezing the original weights:

- **LWC (Learnable Weight Clipping)**: Per-channel weight clipping thresholds optimized via gradient descent. Standard PTQ hand-crafts these thresholds; OmniQuant finds optimal values automatically.
- **LET (Learnable Equivalent Transformation)**: Learns per-channel scale factors s that migrate quantization difficulty from activations to weights (a generalization of SmoothQuant). After training, s is fused into the quantized weights — no inference overhead.

**W4A4 results** (LLaMA-7B, zero-shot accuracy):

| Method | W4A4 Accuracy | Data | Training time |
|--------|--------------|------|---------------|
| SmoothQuant | 38.41% | 128 samples | 10 min |
| LLM-QAT | 46.43% | 100K samples | 90h |
| **OmniQuant** | **52.65%** | **128 samples** | **1.6h** |

OmniQuant achieves QAT-level accuracy at PTQ-level cost — a 6.22 percentage-point improvement over LLM-QAT at 1/800th the data and 1/56th the training time. At W3A16: best-in-class; at W2A16: practical for first time without QuIP's orthogonal rotation.

---

## 3. Method Comparison Table

| Method | Precision | Key Mechanism | PPL Gap to FP16 (7B) | Calibration | Retraining | HW-friendly |
|--------|-----------|--------------|----------------------|-------------|------------|-------------|
| GPTQ | 2–4 bit | OBS + Cholesky (col-sequential) | INT4: +0.54; INT3: +3.13 | 128 seqs, 4h | No | Custom CUDA |
| AWQ | 3–4 bit | Activation-aware channel scaling | INT4: +0.10; INT3: +0.67 | 16 seqs, fast | No | TinyChat (ARM/GPU) |
| SpQR | 3–4 bit | Outlier FP16 + bulk 3-4 bit | <1% | Calibration | No | Custom decoder |
| SqueezeLLM | 3 bit | Hessian non-uniform bins + D&S | 7B: 7.75 (−2.1× gap vs GPTQ) | Calibration | No | GPU |
| QuIP | 2 bit | Incoherence + LDLQ | First viable 2-bit | Calibration | No | GPU |
| OmniQuant | 2–4 bit | Learnable LWC + LET | W4A4: best-in-class | 128 samples, 1.6h SGD | Params only | Fused |
| BitNet b1.58 | 1.58 bit | Ternary training from scratch | ≈0 at 3B+ (if trained) | N/A | Full retraining | Add-only hardware |

---

## 4. Shared-Weight Composability — Method by Method

### 4.1 GPTQ [arxiv:2210.17323] — Requires joint Hessian

**Problem**: H = 2XX^T uses only X, the inputs to the W→x forward path. For shared W, the Wᵀ→x' path contributes equally to quantization damage but is invisible to this Hessian.

**Fix**: H_joint = 2(XX^T + X'(X')^T), where X' are inputs to the Wᵀ→x' path collected during the same calibration forward passes. The Cholesky reformulation and lazy batch update algorithm apply unchanged to H_joint — only the calibration step changes. Implementation cost: collect and sum two activation covariance matrices instead of one.

**Impact on efficiency**: Identical to standard GPTQ; only calibration forward-pass statistics change. Runtime unchanged.

### 4.2 AWQ [arxiv:2306.00978] — Requires joint activation calibration

**Problem**: Per-channel scale `s_X = mean activation magnitude of input channel` is measured on the W→x forward path. Input channels of W become output channels of Wᵀ — their magnitudes in the Wᵀ→x' path differ from their magnitudes in the W→x path.

**Fix**: Joint activation statistics. For each channel j, collect:
- `s_j^W = mean |x_j|` from the W→x path
- `s_j^{Wᵀ} = mean |x'_j|` from the Wᵀ→x' path

Then solve for a per-channel scale `s_j` that jointly minimizes quantization error across both uses. The simplest joint scale: `s_j = (s_j^W + s_j^{Wᵀ}) / 2` or `s_j = max(s_j^W, s_j^{Wᵀ})` (max protects the more sensitive channel). The optimal joint scale requires a small grid search over the combined criterion — the same structure as standard AWQ but with a two-path loss function.

**Composability with standard AWQ**: Joint AWQ is orthogonal to joint GPTQ. Using both together (analogous to AWQ+GPTQ for INT2 in the standard setting) would give the best achievable result for shared-weight quantization.

### 4.3 OmniQuant [arxiv:2308.13137] — **Most naturally compatible**

**Why it's natural**: OmniQuant's block-level SGD objective already integrates over all operations within the Transformer block. In HaLViT, both the W→x and Wᵀ→x' paths are computed within the same attention or FFN block. The block reconstruction error that LWC and LET optimize therefore automatically sees both paths:

```
min_{θ_LWC, θ_LET} ‖Block(X; Q(W|θ_LWC, θ_LET)) − Block(X; W)‖_F^2
```

When the block contains both `y = Wx` and `z = Wᵀx'`, the reconstruction loss penalizes errors on both simultaneously. No modification is needed — the same mathematical structure that makes BESA natural for pruning makes OmniQuant natural for quantization.

**For W4A4**: OmniQuant already achieves 52.65% accuracy (vs 38.41% SmoothQuant) on LLaMA-7B; its learnable LET transformation adapts to whatever distribution both paths exhibit without analytical coupling constraints.

### 4.4 SpQR [arxiv:2306.03078] and SqueezeLLM [arxiv:2306.07629] — Partial compatibility

Both methods protect outlier weights in FP16. The outlier mask for shared W is computed once and applies to all uses of that weight — both W→x and Wᵀ→x' paths see the same sparse FP16 overlay, which is correct: an outlier weight at (i,j) in W is also an outlier at (j,i) in Wᵀ, since the same weight value is involved.

**Limitation at HaLViT scale**: At 11M–22M parameters, HaLViT is far below the 6.7B outlier threshold where the systematic outlier phenomenon (LLM.int8()) emerges. Standard uniform 3–4 bit PTQ should suffice without SpQR/SqueezeLLM's outlier handling for a ViT at this scale. These methods become relevant if HaLViT is scaled up or applied to larger backbone architectures.

### 4.5 QuIP [arxiv:2307.13304] — Open theoretical question

**Problem**: QuIP pre-multiplies W by U and V (random orthogonal matrices) and post-multiplies by V and U respectively. For W+Wᵀ sharing, the rotation applied to W implicitly applies a transposed rotation to Wᵀ:

```
W_q = U W̃_q V^T   →   W_q^T = V W̃_q^T U^T
```

Whether the orthogonal rotation simultaneously incoherences both the W→x context (where W_q is applied) and the Wᵀ→x' context (where V W̃_q^T U^T is applied) is a theoretical question. The answer depends on whether the activation distributions X and X' are both made approximately uniform by the rotation — which is guaranteed for independent random X, X', but not necessarily for the correlated activations of a trained ViT.

**Practical implication**: QuIP's 2-bit regime is aspirational for HaLViT at its current scale (11M–22M). If quantization targets of 2-bit are pursued, the theoretical validity of incoherence processing for shared weights requires study before application.

### 4.6 BitNet b1.58 [arxiv:2402.17764] — Training-time, most radical

BitNet b1.58 uses ternary weights {-1, 0, +1} via absmean quantization: `W̃ = RoundClip(W/γ + ε, -1, 1)`, where γ = mean(|W_ij|). This requires training from scratch via STE — PTQ cannot be applied post-hoc to FP16 weights.

For HaLViT: both W+Wᵀ sharing (HaLViT's mechanism) and ternary weights (BitNet's mechanism) are training-time modifications. They can co-exist in the same architecture by implementing a BitLinear module that also enforces the W+Wᵀ constraint. The STE gradient accumulates from both the W→x and Wᵀ→x' paths (natural shared-weight gradient flow), and absmean quantization applies to the single physical tensor W.

**Key subtlety**: absmean γ = mean(|W_ij|) is computed over the flat weight matrix. When W is used as both W and Wᵀ, γ is the same for both uses (correct — it is a property of the shared tensor). However, the quantization bin spacing is determined by a single γ — if the two pathways have different effective weight magnitudes at different positions, the single γ may not be optimal for both. In practice, ternary weights with |W_ij| ≤ 1 may make this moot (the clipping boundary dominates).

---

## 5. Accuracy-Compression Frontier: Summary

The following represents the achievable accuracy-compression frontier on LLaMA-family models, extrapolated to HaLViT scale:

| Bit-width | Best method | Quality (7B LLaMA PPL) | Compression vs FP16 | HaLViT applicability |
|-----------|-------------|----------------------|----------------------|---------------------|
| FP16 (baseline) | — | 5.68 | 1× | Full baseline |
| INT8 | Any PTQ | ≈5.68 (free) | 2× | Direct, no modification |
| INT4 | AWQ | 5.78 (+0.10) | 4× | Joint activation calibration needed |
| INT3 | AWQ / SpQR | 6.35 (+0.67) / <1% | 5.3× | Joint calibration or outlier mask |
| INT2 | QuIP | First viable | 8× | Theoretical open question |
| 1.58-bit | BitNet b1.58 | ≈FP16 at 3B+ if trained | ~11× | Training-time co-design |

**The practical frontier for HaLViT** is INT4 with AWQ (joint activation calibration extension), yielding approximately 4× quantization compression on top of HaLViT's inherent 2× weight-sharing compression — an 8× total compression relative to a standard FP16 ViT with separate weight matrices.

---

## 6. Composition Pipeline for HaLViT

Combining weight sharing, pruning (from the pruning analysis), and quantization yields a three-axis compression pipeline:

```
Stage 1: Train HaLViT with W+Wᵀ sharing (2× weight compression baked in)
Stage 2: Optional — Basis Sharing cross-layer (additional ~2× via SVD across blocks)
Stage 3: Prune  — BESA or Wanda (30–50% sparsity; 1.5–2× compression)
Stage 4: Quantize — Joint AWQ calibration + GPTQ reconstruction (INT4; 4× compression)
```

Sequential stages are tractable because each operates on the shared tensor W:
- BESA reduces which weights exist (sparsity mask)
- AWQ determines per-channel scales (absorbed into adjacent operators)
- GPTQ refines the quantization within each channel (Cholesky-based rounding)

All three calibration steps can in principle be unified into a single calibration pass (analogous to SparseGPT+GPTQ joint compression [arxiv:2301.00774]) — the most ambitious formulation.

**Estimated total compression** (multiplicative, rough):
- Weight sharing: 2×
- Cross-layer Basis Sharing: 1.5–2×
- 50% unstructured pruning: 2× (at same memory, before quantization)
- INT4 quantization: 4×
- Combined (weight × quantization, after pruning): **~16×** vs a standard FP16 ViT

A 22M-parameter DeiT-Small at 16× compression → 1.375M effective parameters (quantized, sparse, shared). This is in the range of microcontroller-deployable vision models.

---

## 7. Open Research Questions and Thesis Experiments

### Experiment 1: Baseline INT8 PTQ on HaLViT-T (Lowest cost)

**Setup**: Apply standard GPTQ INT8 to trained HaLViT-T; 128 calibration images from ImageNet; measure top-1 accuracy. Compare to: (a) full-precision HaLViT-T; (b) standard DeiT-Small at INT8.

**Expected**: Zero accuracy loss. This is the baseline to confirm before exploring lower bit-widths.

### Experiment 2: Joint AWQ at INT4 on HaLViT-T (Primary experiment)

**Setup**: Implement joint activation calibration for AWQ — measure `s_j^W` from Wkv→x (key path) and `s_j^{Wᵀ}` from Wkvᵀ→x' (value path); compute joint scale `s_j`; apply AWQ-INT4.

**Hypothesis**: Joint AWQ (measuring both paths) achieves higher top-1 at INT4 than standard single-path AWQ applied to HaLViT. The gap quantifies the cost of ignoring the transposed path.

**Baseline**: Standard AWQ applied naively (single path), as an ablation. Difference = the value of the joint calibration extension.

**Expected contribution**: First empirical demonstration of joint activation calibration for shared-weight attention.

### Experiment 3: Joint-Hessian GPTQ at INT3 on HaLViT-T

**Setup**: Implement H_joint = 2(XX^T + X'(X')^T) during GPTQ calibration; apply at INT3; compare to standard single-path GPTQ-INT3.

**Hypothesis**: Joint-Hessian GPTQ degrades less than standard GPTQ at INT3, because the weight reconstruction error is minimized over both uses of W simultaneously.

**Expected contribution**: First formulation and evaluation of joint-Hessian GPTQ for W+Wᵀ shared weights.

### Experiment 4: OmniQuant on HaLViT-T at W4A4

**Setup**: Apply OmniQuant as-is to HaLViT-T (no modification needed, block objective covers both paths). Optimize LWC thresholds and LET scales for 1.6h with 128 calibration samples; measure W4A4 top-1 accuracy.

**Hypothesis**: OmniQuant achieves better W4A4 accuracy than standard AWQ or GPTQ on HaLViT, because its learnable block-level objective naturally integrates the joint path constraint.

**Expected cost**: 1.6h on one A100 (same as OmniQuant on LLaMA-7B, smaller model so likely faster).

### Experiment 5: Three-axis compression (weight sharing + pruning + quantization)

**Setup**: Train HaLViT-T → apply BESA at 30% sparsity → apply joint AWQ-INT4 → measure top-1 accuracy and total compression ratio vs DeiT-Small FP16.

**Target**: ≥70% of DeiT-Small accuracy at ≤10% of DeiT-Small parameter storage (accounting for sparsity mask storage and quantization codebook overhead).

**Expected**: 11.1M (HaLViT-T) × 70% (BESA) × 4-bit = approximately 0.93MB of quantized weights — well within embedded device constraints.

---

## 8. Relationship to Prior Analyses

This analysis is the third in a thesis-critical series:

1. [[analyses/halvit-vs-albert-cross-layer-sharing]] — establishes the two orthogonal axes of weight sharing (intra-layer vs cross-layer); shows stacking is feasible in principle
2. [[analyses/pruning-composability-with-weight-sharing]] — maps six pruning methods to a compatibility tier; identifies symmetric mask as the unifying concept; defines five pruning experiments
3. **This analysis** — maps quantization methods to a bit-width progression; identifies joint calibration as the unifying concept; defines five quantization experiments

The three-axis compression pipeline (sharing + pruning + quantization) is the central contribution the thesis is building toward. The joint calibration problem — whether for pruning (symmetric mask), quantization (joint Hessian/AWQ), or weight-sharing across layers (Basis Sharing) — is the same mathematical structure appearing in three different guises.

---

## Sources Consulted

- [[sources/2210.17323]] — GPTQ; second-order PTQ; OPT-175B INT4; joint-Hessian open question for shared weights
- [[sources/2306.00978]] — AWQ; activation-aware channel scaling; LLaMA-7B INT4 5.78; joint AWQ open question
- [[sources/2306.03078]] — SpQR; outlier FP16 + bulk 3-4 bit; near-lossless across all scales
- [[sources/2306.07629]] — SqueezeLLM; Hessian non-uniform bins + D&S; LLaMA-7B 3-bit 7.75; bandwidth-bound frame
- [[sources/2307.13304]] — QuIP; incoherence processing; first viable 2-bit; LDLQ theory subsumes GPTQ
- [[sources/2308.13137]] — OmniQuant; learnable LWC + LET; W4A4 52.65%; most naturally compatible with shared weights
- [[concepts/quantization]] — unified taxonomy; open questions; all method descriptions
