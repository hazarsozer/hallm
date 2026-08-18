---
query: "How does HaLViT's W+Wᵀ intra-layer sharing compare to ALBERT-style cross-layer sharing — mechanisms, compression ratios, accuracy costs, composability, and thesis implications?"
date: 2026-04-30
sources_consulted:
  - "wiki/sources/halvit.md"
  - "wiki/sources/1909.11942.md"
  - "wiki/sources/2101.00234.md"
  - "wiki/sources/2207.10237.md"
  - "wiki/sources/2410.03765.md"
  - "wiki/concepts/weight-sharing.md"
---

## Executive Summary

HaLViT and ALBERT represent two orthogonal axes of weight sharing in Transformer architectures. ALBERT shares the *same* weight matrix across all **L layers** (sharing along the depth axis). HaLViT uses a single weight matrix **W** and its transpose **Wᵀ** as two independent transformations **within one layer** (sharing along the width axis). The two mechanisms are not competing approaches to the same problem — they compress different dimensions of the parameter space and can, in principle, be stacked. The thesis opportunity lies precisely in this composability.

---

## 1. Mechanism Comparison

### 1.1 HaLViT: Intra-Layer Sharing (Width Axis)

HaLViT's core mathematical insight: for a weight matrix **W** (m×n) and input **x**,

```
y = Wx          → projects x into the column space of W
F(Wx)           → nonlinear activation exits the column space
Wᵀ · F(Wx)      → genuinely independent transformation; NOT redundant
```

Because the activation function F(·) is nonlinear, the output of **Wx** no longer resides in the column space of **W**. Multiplying by **Wᵀ** therefore produces a projection that cannot be expressed as a rescaling of **Wx**. This is the theoretical justification for treating **W** and **Wᵀ** as two distinct, independently expressive linear transforms within a single layer [arxiv:halvit].

Applied in practice:
- **Multi-Head Attention**: Wkv generates keys (Wkv·x) and values (Wkvᵀ·x); Wq generates queries (Wq·x) and the output projection (Wqᵀ·x̂). Four matrices collapse to two.
- **FFN**: `FFN(x) = Wᵀ·F(W·x + b₁) + b₂` — one matrix W replaces the standard two-matrix sequence W₁, W₂.
- **ResNet Bottleneck**: `Bottleneck(x) = Wᵀ·G(W·x)` — applied to stages 3 and 4 (stages 1–2 showed diminishing returns in ablation).

### 1.2 ALBERT: Cross-Layer Sharing (Depth Axis)

ALBERT ties the same weight tensor W across all L Transformer layers:

```
Layer 1 output: h₁ = TF(x;  W_attn, W_FFN)
Layer 2 output: h₂ = TF(h₁; W_attn, W_FFN)   ← same W_attn, W_FFN
...
Layer L output: hₗ = TF(hₗ₋₁; W_attn, W_FFN) ← same W_attn, W_FFN
```

No mathematical proof exists that this converges to an optimal solution. Figure 1 of [arxiv:1909.11942] shows that ALBERT's layer representations exhibit *oscillating* L2 distances and cosine similarities rather than converging — a stable but fundamentally different solution space than BERT's monotonically converging layers. The approach is justified empirically rather than theoretically.

Companion technique required: **Factorized Embedding Parameterization** — decoupling vocabulary embedding size E from hidden size H (E ≪ H). Without this, the embedding matrix V×H would dominate parameter count and negate the savings from weight sharing. HaLViT requires no such companion technique.

### 1.3 Mechanism Summary Table

| Dimension | HaLViT [arxiv:halvit] | ALBERT [arxiv:1909.11942] |
|-----------|----------------------|--------------------------|
| Sharing axis | Width (within one layer) | Depth (across all L layers) |
| What is shared | W used as W forward, Wᵀ backward | Identical W_attn + W_FFN across L layers |
| Shared units per layer | 2 transforms from 1 matrix | 1 copy of 1 matrix reused L times |
| Mathematical justification | Nonlinearity breaks column-space; Wᵀ·F(Wx) is independent | Empirical; oscillating layer dynamics, not convergence |
| Companion technique needed | None | Factorized embedding parameterization (E ≪ H) |
| Domain | Vision (ViT, ResNet) | NLP (BERT-style encoder) |
| Training regime | From scratch | From scratch |

---

## 2. Compression Ratios

### 2.1 HaLViT

The sharing mechanism halves the parameter count of every attention block and every FFN block in the shared layers. For a ViT applied to ImageNet-1K:

| Model | Params | Top-1 |
|-------|--------|-------|
| DeiT-Small | 22M | 79.9% |
| PVTv2-B1 | 14M | 78.7% |
| HaLViT-T² | **11.1M** | **78.8%** |
| HaLViT-M | **43M** | **81.3%** |

HaLViT-T² (11.1M) outperforms PVTv2-B1 (14M) with 2.9M fewer parameters. On ResNet50: 25.6M → 13.4M (~2×) with only 1.0 pp Top-1 loss.

Compression ratio per-layer: ~2×. This is modest by ALBERT standards, but it is **lossless in the parameter-count-to-capacity ratio** because nonlinearity makes the two transforms genuinely independent.

### 2.2 ALBERT

ALBERT's cross-layer sharing achieves much larger compression ratios because the same parameters serve L layers simultaneously:

| Model | Params | GLUE Avg |
|-------|--------|----------|
| BERT-large | 334M | 85.2 |
| ALBERT-base (L=12, H=768, E=128) | **12M** | 80.1 |
| ALBERT-large (L=24, H=1024, E=128) | **18M** | 82.4 |
| ALBERT-xxlarge (L=12, H=4096, E=128) | **235M** | **88.7** |

ALBERT-large achieves **18× fewer parameters than BERT-large** (18M vs 334M). ALBERT-base: **9× fewer than BERT-base** (12M vs 108M). The enabling mechanism is that all L layers run on the same physical weight copy — compression scales with depth.

### 2.3 Comparison

ALBERT achieves far higher compression ratios because the savings multiply with depth (L layers → 1 set of weights). HaLViT achieves constant 2× per layer regardless of depth. However:
- ALBERT at high compression (9–18×) incurs ~1.5 GLUE Avg loss.
- HaLViT at 2× incurs ~1–2% ImageNet Top-1 loss.
- The two ratios are not directly comparable (cross-domain, different metrics).

The key structural difference: ALBERT's compression ratio grows with depth (L×); HaLViT's is fixed at ~2× per layer. For very deep models, ALBERT's approach has an exponential advantage in parameter reduction. For moderately deep models (ViT-Small: L=12, ViT-Base: L=12), cross-layer sharing reduces 12 parameter sets to 1 — a 12× gain *on top of* HaLViT's 2× intra-layer gain if the two were stacked.

---

## 3. Accuracy Costs

### 3.1 HaLViT

ImageNet-1K (224×224):
- HaLViT-T² vs DeiT-Small: 78.8% vs 79.9% — **−1.1% Top-1** at 50% parameter reduction
- HaLViT-M vs comparable models: 81.3%, competitive at 43M params

The accuracy cost is recoverable, especially with longer training. HaLViT-T¹ (standard training, 77.3%) vs HaLViT-T² (600 epochs, 78.8%) — +1.5 pp from extended training, essentially closing the gap to PVTv2-B1.

Critical ablation: **extreme cross-layer sharing** on HaLViT* (9M, all layers except Wq shared both intra- and cross-layer): Top-1 drops to 67.6%. The W+Wᵀ intra-layer scheme is what makes HaLViT competitive — naive cross-layer sharing on top destroys accuracy. This suggests the intra-layer axis is the "safe" one; naive cross-layer stacking without care (e.g., Sandwich constraints) causes collapse.

### 3.2 ALBERT

Critical ablation from Table 4 of [arxiv:1909.11942] (E=128, L=12, H=768):

| Strategy | Params | GLUE Avg | Cost vs not-shared |
|----------|--------|----------|--------------------|
| not-shared (baseline) | 89M | 81.6 | — |
| shared-attention only | 64M | **81.7** | **+0.1** (free) |
| shared-FFN only | 38M | 80.2 | −1.4 |
| all-shared (ALBERT default) | 12M | 80.1 | −1.5 |

**Sharing attention costs essentially nothing (+0.1 Avg)**. Sharing FFN is where the accuracy loss materializes. All-sharing (89M → 12M) incurs only −1.5 Avg points. This is arguably the most important finding in the paper: the attention mechanism's weights are dramatically redundant across layers; FFN weights are less so.

This result has a direct parallel in HaLViT: the paper shares both attention (Wkv, Wq) and FFN (W) intra-layer. The question whether HaLViT's FFN sharing W+Wᵀ incurs more cost than its attention sharing (analogous to ALBERT's FFN > attention cost profile) was not ablated independently in [arxiv:halvit]. This is an open experimental question.

### 3.3 SPIN's Cross-Layer Baseline on ViT

SPIN [arxiv:2207.10237] provides the closest apples-to-apples cross-layer sharing baseline on DeiT-S (architecturally the same as HaLViT's target):

| Share rate | Params | DeiT-S Top-1 | Loss |
|------------|--------|--------------|------|
| 1 (baseline) | 22.05M | 80.52% | — |
| 2 + CWM fusion | 11.41M | **79.44%** | **−1.08%** |
| 3 + CWM fusion | 7.87M | 77.11% | −3.41% |
| 4 + CWM fusion | 5.91M | 75.12% | −5.40% |

Cross-layer sharing at 2× on DeiT-S: −1.08%. HaLViT intra-layer at 2×: −1.1%. The two mechanisms incur *strikingly similar accuracy costs* at the same compression ratio, despite being entirely different mechanisms on different axes. This suggests the accuracy cost of the 2× compression may be a floor set by ImageNet task difficulty rather than by the sharing mechanism itself.

---

## 4. Theoretical Grounding

### 4.1 HaLViT's Argument

The W + Wᵀ approach rests on a provable property: after a nonlinear activation, the output F(Wx) lies outside the column space of W. Wᵀ·F(Wx) therefore cannot be written as AW·x for any matrix A — the two transformations are linearly independent. This is a clean algebraic argument that holds for any nonlinear F(·), making the method applicable across architectures without additional assumptions.

Limitation: The argument shows W and Wᵀ are *linearly independent as operators*, not that they are *information-theoretically independent* as feature extractors. Whether the column space of W and the row space of W (which Wᵀ projects into) are *semantically complementary* in a trained network is an empirical question. The results confirm they are functionally complementary, but the theory alone cannot guarantee this.

### 4.2 ALBERT's Argument

No formal theorem underlies ALBERT's cross-layer sharing. The justification is: (a) empirically, deep networks exhibit layer redundancy (supported by CKA analysis in [arxiv:2207.10237]); (b) universal Transformers [Dehghani et al. 2019] already showed shared-weight recurrent Transformers converge; (c) ALBERT's oscillating layer representations suggest the shared weights learn to cycle through transformations rather than learn L separate ones — an emergent iterative refinement rather than a feed-forward stack.

The CKA analysis in SPIN [arxiv:2207.10237] independently validates the empirical basis: **middle layers of ViT have the highest cross-layer representational similarity** (CKA peak at the center), meaning cross-layer sharing imposes minimal information loss where it matters most. The Subformer [arxiv:2101.00234] confirms this from another direction: the first and last layers cannot be shared (they encode task-specific representations), but the middle layers can be shared without quality loss in generative tasks.

---

## 5. Structural Constraints and Failure Modes

### 5.1 HaLViT Constraints

- **Early-stage sharing degrades**: In ResNet, applying HaLViT sharing to stages 1–2 costs −0.9 pp (ablation). Early layers are more sensitive to parameter sharing — they extract low-level features with fewer neurons, so halving parameters reduces representational capacity meaningfully.
- **Extreme cross-layer stacking collapses**: HaLViT* (all layers shared, 9M params): 67.6% Top-1 — 11 pp below the standard HaLViT-T². The intra-layer W+Wᵀ mechanism is robust; naive cross-layer extension is not.
- **Non-square weight matrices**: HaLViT's shared W is typically non-square (m×n, m≠n). This means Wᵀ is n×m, a different-shaped operator. In MHA, Wkv projects input x (d_model-dimensional) to keys and values — Wkv and Wkvᵀ are different-shaped operators, which must be applied to compatible-dimensional inputs. The architecture is designed around this, but it constrains how the sharing can be extended or modified.

### 5.2 ALBERT Constraints

- **Requires factorized embedding**: Without V×E + E×H decomposition, the embedding matrix V×H at large H negates the savings from weight sharing. HaLViT has no such coupling constraint.
- **Oscillating dynamics**: Layer representations don't converge — they oscillate. Fine-tuning may behave differently than BERT (which converges). Empirical GLUE results are strong, but the non-converging dynamics suggest the learned solution is a fundamentally different computational structure.
- **Fails for generation (all-layer variant)**: Subformer [arxiv:2101.00234] proved conclusively that ALBERT's all-layer sharing collapses generative quality (WMT'14 BLEU: 14.3 vs 27.3 baseline with all-layer sharing). **Fix**: Sandwich sharing — share only middle layers; keep layers 1 and L independent. This is structurally consistent with the CKA redundancy finding.
- **Normalization layers must be independent**: SPIN [arxiv:2207.10237] found that sharing BatchNorm across layers causes training divergence. Any cross-layer sharing scheme — ALBERT or otherwise — must keep LayerNorm parameters (γ, β) layer-specific.

### 5.3 Basis Sharing: Strict Improvement Over Identical-Weight Sharing

Basis Sharing [arxiv:2410.03765] exposes a critical weakness of ALBERT's approach: setting W^(i) = W for all layers sacrifices per-layer expressiveness. The shared basis + unique coefficients approach W^(i) ≈ B·C^(i) strictly dominates:

| Method | Model | PPL (20% compression) |
|--------|-------|-----------------------|
| ALBERT-style (Dynamic Tying) | GPT2-XL (264M) | 49.37 |
| Basis Sharing | GPT2 (94M) | **43.15** |

A 3× smaller model with unique coefficients beats the 3× larger identical-weight model — at 26 seconds calibration vs 13.75 hours of training. ALBERT's identical-weight sharing is obsolete as a compression technique (though it remains valid as an architecture choice for other reasons, e.g., enabling scaling to larger H without parameter explosion).

---

## 6. Composability Analysis

The central thesis question: can these mechanisms stack?

### 6.1 Orthogonality Confirmed

The two sharing axes are structurally orthogonal:
- **HaLViT** (intra-layer): replaces [W₁, W₂] with [W, Wᵀ] within a single layer's computation
- **ALBERT** (cross-layer): replaces [W^(1), W^(2), …, W^(L)] with [W, W, …, W] across L layers

These operate on different indices. Applying both simultaneously would produce: a single W shared across layers *and* used as W and Wᵀ within each layer — yielding W used 2L times instead of the original L separate matrices for each sub-operation. The parameter reduction would be ~2× (intra-layer) × L× (cross-layer) if all layers share, or a combination for partial cross-layer sharing.

### 6.2 Sandwich Constraint in a Composed System

If extending HaLViT's intra-layer sharing to also operate cross-layer (i.e., the shared W+Wᵀ pair is itself shared across ViT blocks), the Subformer's Sandwich rule [arxiv:2101.00234] suggests:
- Middle ViT blocks can safely share the W+Wᵀ pair across depth
- The first block (processes patch embeddings + positional encoding — task-specific input representation) should remain unshared across layers
- The last block (feeds directly into classification head) should remain unshared across layers

However, this is the Subformer finding for *generative* seq2seq transformers. ViT is a discriminative architecture more analogous to BERT (encoder-only, full attention, CLS token). ALBERT's all-layer sharing works for BERT, suggesting ViT's first/last blocks may also tolerate cross-layer sharing. The HaLViT ablation (HaLViT* at 9M: 67.6%) does not isolate whether the collapse is caused by cross-layer sharing or by the extreme compression ratio — this is an unresolved experimental question.

SPIN's CKA analysis on DeiT-S [arxiv:2207.10237] is the most relevant data point: middle layers have highest representational similarity, but the first and last layers are also more similar to each other than they are to the middle layers. For ViT (discriminative), Sandwich-style protection of first/last may not be *required*, but is likely beneficial as a soft constraint.

### 6.3 Weight Fusion Requirement

SPIN establishes that cross-layer sharing from pretrained weights requires **weight fusion** (Channel Weighted Mean, CWM) to recover 0.5–1.3 pp of accuracy. For a pretrained HaLViT model, extending to cross-layer sharing adds a fusion challenge: each layer's "weight" is not a single matrix W but a pair (W, Wᵀ). Standard CWM applied independently to the W-path weights across layers would produce a fused W̄, from which W̄ᵀ would naturally be the cross-layer-fused transposed counterpart. This is self-consistent — CWM applied to the W matrices directly preserves the W+Wᵀ pairing. Whether this fusion is optimal requires empirical validation.

### 6.4 Basis Sharing as the Modern Cross-Layer Option

Rather than ALBERT's identical-weight sharing, **Basis Sharing** [arxiv:2410.03765] is the superior post-training cross-layer compression method. Applied to a trained HaLViT model:
- Step 1: Train HaLViT with W+Wᵀ intra-layer sharing (intra-layer compression done)
- Step 2: Apply Basis Sharing to compress HaLViT's already-shared W matrices across adjacent ViT blocks (cross-layer compression on top)
- Result: W^(block i) ≈ B·C^(i), where B is shared across adjacent blocks and C^(i) is per-block

Critically: W and Wᵀ both depend on the same B. Compressing B (via SVD) would automatically reduce storage for both the forward and transposed paths. This is a multiplied compression: SVD rank reduction on the basis that is *already* reused as a transpose. The interaction must be studied carefully — low-rank approximation of W changes Wᵀ correspondingly, which may affect the accuracy of the transposed path differently than the forward path.

---

## 7. Parameter Counting: A Unified View

For a ViT with L layers, each with attention parameters of size d×d and FFN parameters of size d×4d (standard expansion factor 4):

| Scheme | Attention params | FFN params | Total (relative) |
|--------|-----------------|------------|------------------|
| Baseline | L × 4×(d²) | L × 2×(4d²) | L × 12d² |
| HaLViT only | L × 2×(d²) | L × 1×(d²) | L × 3d² — **4× fewer** |
| Cross-layer only (ALBERT) | 1 × 4×(d²) | 1 × 2×(4d²) | 12d² — **L× fewer** |
| HaLViT + Cross-layer | 1 × 2×(d²) | 1 × 1×(d²) | 3d² — **4L× fewer** |

For ViT-Small with L=12: baseline ~156d², HaLViT alone ~36d² (4×), ALBERT alone ~12d² (13×), combined ~3d² (~52×). This theoretical maximum assumes perfect accuracy recovery — in practice, accuracy floors prevent achieving such extreme ratios without significant loss.

---

## 8. Implications for the Thesis

### 8.1 Confirmed Findings

1. **HaLViT and ALBERT solve different compression problems** — they are not competitors. HaLViT halves width (within-layer parameter count); ALBERT halves depth (layer count translated to shared copies).

2. **Accuracy cost at 2× is ~1% for both mechanisms on ViT**, and this may be a task-difficulty floor rather than a mechanism-specific penalty. SPIN's cross-layer 2× compression on DeiT-S achieves −1.08%; HaLViT's intra-layer 2× achieves −1.1%.

3. **Attention sharing is free; FFN sharing is where accuracy is spent** — ALBERT's ablation shows attention-only sharing costs +0.1 Avg (essentially free) while FFN sharing costs −1.4 Avg. Whether this cost structure applies to HaLViT's within-layer FFN sharing is not yet ablated.

4. **Identical-weight cross-layer sharing (ALBERT) is dominated by Basis Sharing** for post-training compression — unique coefficients recover the individuality that forces ALBERT's accuracy tax.

5. **Normalization layers (LayerNorm in ViT) must never be shared cross-layer** — SPIN confirms divergence.

### 8.2 Open Experiments

1. **HaLViT ablation: attention-only vs FFN-only sharing** — Does HaLViT's FFN sharing (W+Wᵀ) incur disproportionate accuracy loss compared to attention sharing (Wkv, Wq)? If yes, the architecture should preferentially apply full sharing to attention and lighter sharing to FFN, analogous to ALBERT's finding.

2. **Cross-layer sharing of HaLViT's W+Wᵀ pair using Sandwich strategy** — Share middle ViT blocks' W matrices across depth; leave blocks 1 and L unshared. Combine with weight fusion (CWM on the shared W matrices). Expected: ~2× additional compression on top of HaLViT, with −1–2% accuracy cost.

3. **Basis Sharing applied post-HaLViT-training** — Calibrate with 256 WikiText-2 (or 256 ImageNet) samples; apply SVD-based cross-layer basis sharing to HaLViT's W matrices across adjacent blocks. Unique C^(i) matrices per block. Measure accuracy vs compression compared to HaLViT baseline.

4. **Layer dynamics analysis of HaLViT** — Compute L2 distance and cosine similarity between HaLViT's input and output representations across ViT blocks (analogous to ALBERT's Figure 1). If oscillating, HaLViT's dynamics match ALBERT's — suggesting the two mechanisms converge to similar solution-space structure despite different sharing axes. If converging, the intra-layer sharing imposes a different inductive bias.

5. **Composability with pruning and quantization** — HaLViT's Discussion section explicitly names this as future work. Given that cross-layer sharing multiplies the impact of any pruning/quantization error on shared weights, the composition order matters: prune/quantize first, then apply weight sharing? Or share first, then prune/quantize with awareness of the shared-weight gradient? Basis Sharing's post-training approach suggests the latter is more tractable.

---

## 9. Related Analyses (To Be Written)

- `wiki/analyses/pruning-composability-with-weight-sharing.md` — SparseGPT, Wanda, LLM-Pruner, BESA, SNIP, Lottery Ticket compatibility with shared weights
- `wiki/analyses/quantization-tradeoffs-1to4bit.md` — GPTQ, AWQ, SpQR, SqueezeLLM, QuIP, OmniQuant accuracy vs compression at 1–4 bit

---

## Sources Consulted

- [[sources/halvit]] — HaLViT; W+Wᵀ intra-layer sharing; anchor paper
- [[sources/1909.11942]] — ALBERT; cross-layer sharing; factorized embeddings; canonical cross-layer reference
- [[sources/2101.00234]] — Subformer; Sandwich-style selective cross-layer sharing; SAFE embeddings
- [[sources/2207.10237]] — SPIN; systematic 2D topology evaluation; weight fusion; CKA analysis on DeiT-S
- [[sources/2410.03765]] — Basis Sharing; cross-layer SVD with unique coefficients; dominates ALBERT for post-training compression
- [[concepts/weight-sharing]] — synthesis of all sharing variants; open questions
