# Overview: Model Compression for Resource-Constrained AI

*Top-level synthesis. Updated as new sources are ingested. Last updated: 2026-04-30 (5 analyses written; 6 source pages re-ingested to full quality).*

---

## Project Scope

This wiki supports an ITU graduation project titled **"Advanced Model Compression Techniques for Resource-Constrained AI Architectures"** (Alper Düzgün & Hazar Utku Sözer). The central focus paper is **HaLViT** — a Vision Transformer using row/column-space W+Wᵀ weight sharing within attention and FFN blocks. The goal is to situate HaLViT within the broader compression literature, map which techniques compose with its sharing scheme, and identify a viable original thesis contribution.

**Wiki state**: 56 source pages, 19 entity pages, 8 concept pages, 5 analysis pages, 4 wrong-file skips (2002.07905 RL, 2105.10140 physics, 2411.16744 combinatorics, 2501.18511 WildChat dataset). DeltaLLM (2501.18511 placeholder) still missing its actual PDF.

---

## The Compression Landscape (as of April 2026)

The field has converged on five major pillars, which are increasingly composed rather than applied in isolation.

### 1. Pruning

**Evolution**: magnitude-based iterative pruning (OBD 1990, Han et al. 2015) → lottery ticket theory (Frankle & Carlin 2019) → training-free LLM pruning without retraining (SparseGPT 2023, Wanda 2024) → differentiable blockwise pruning (BESA 2024).

**Current frontier**:
- **SparseGPT** [2301.00774]: one-shot, second-order, 50% sparsity on OPT-175B with zero accuracy loss
- **Wanda** [2306.11695]: |w|×||x|| criterion, no Hessian, no retraining — competitive with SparseGPT at 50% sparsity; activation norms for shared W are naturally joint across W→x and Wᵀ→x' paths
- **LLM-Pruner** [2305.11627]: structural pruning via dependency graphs + LoRA recovery in 3h/1GPU
- **BESA** [2402.16880]: first differentiable LLM pruning — blockwise reconstruction + learned per-layer sparsity rates via STE; strictly better than layer-wise methods

**Insight for HaLViT**: every LLM pruning method needs extension for shared weights. The W+Wᵀ coupling means pruning row i of W simultaneously affects the Wᵀ pathway — a symmetric mask constraint. Wanda and SNIP naturally produce joint saliency signals for shared W (activation norms accumulate over both paths); SparseGPT and BESA require symmetric mask enforcement.

**Open tension**: structured pruning (whole heads/layers) is hardware-friendly but loses more accuracy; unstructured (individual weights) is more accurate but requires sparse hardware support.

### 2. Quantization

**Evolution**: k-means codebook quantization (Deep Compression 2016) → INT8 QAT (Jacob et al. 2018) → one-shot LLM PTQ (GPTQ 2022) → activation-aware PTQ (AWQ, SmoothQuant 2023) → 2-bit viable (QuIP 2023) → learnable PTQ (OmniQuant 2024) → ternary trained-from-scratch (BitNet b1.58 2024).

**Current frontier** (organized by bit-width):
- **INT8**: SmoothQuant (W8A8, offline scale migration), LLM.int8() (mixed-precision for outlier features at 6.7B+) — both effectively solved
- **INT4**: GPTQ (second-order PTQ), AWQ (activation-aware scaling), OmniQuant (learnable LWC+LET) — near-lossless
- **3-bit**: SpQR and SqueezeLLM independently validate the Dense-and-Sparse paradigm (outlier FP16 + bulk low-bit); SqueezeLLM adds Hessian non-uniform bins — 2.1× better PPL gap vs GPTQ at same memory
- **2-bit**: QuIP (incoherence processing via random orthogonal matrices) is the first viable 2-bit LLM quantization; provides theoretical unification of all outlier-suppression methods (SmoothQuant, SpQR, LLM.int8()) as heuristic forms of the same incoherence principle
- **1-bit**: BitNet b1.58 (ternary {-1,0,+1}, trained from scratch) — addition-only arithmetic, 71.4× less energy; requires training from scratch (cannot PTQ existing models)

**W4A4 serving**: ATOM [2310.19102] combines all techniques for INT4 weight-activation quantization — 7.73× throughput vs FP16, 2.53× vs INT8.

**Insight for HaLViT**: HaLViT at 11M–22M params is below all outlier thresholds — standard INT8 QAT (Jacob et al.) applies directly. QLoRA-style fine-tuning of a 4-bit HaLViT is practical. SmoothQuant's joint calibration constraint (shared W must satisfy both W→x and Wᵀ→x' scales simultaneously) is the key challenge for W4A4; OmniQuant's LET enforces this differentiably.

**Open tension**: PTQ methods (GPTQ, AWQ) cannot be applied to BitNet — ternary weights require training-from-scratch. Choosing between PTQ of an existing model vs training a new BitNet-style HaLViT is a binary architectural decision.

### 3. Weight Sharing

**Taxonomy** (two axes):
- **Cross-layer**: same matrix reused across transformer layers — ALBERT (18× params vs BERT-large), Subformer (Sandwich pattern preserves first/last layer independence), SPIN (2D topology framework, normalization exclusion rule), Basis Sharing (SVD shared basis + per-layer coefficients)
- **Intra-layer**: parameters tied within one layer — HashedNets (random hash buckets), Soft Weight-Sharing (Bayesian GMM), Probabilistic Weight Fixing (BNN per-weight uncertainty), ArbNet (unifying framework: balance + determinism are the two quality axes), **HaLViT** (W+Wᵀ row/column-space sharing)

**HaLViT's unique position**: HaLViT is the only method using the mathematical non-commutativity of F(Wx) vs Wᵀ as a *principled* justification for intra-layer sharing. After nonlinear activation, the output of Wx does not live in the column space of W — making Wᵀx' an independently expressive transformation. This justification does not apply to cross-layer sharing (ALBERT) where the same transformation is applied in the same context.

**Insight**: SPIN's normalization exclusion rule and weight fusion from pretrained initialization are both directly applicable to HaLViT. ArbNet's balance+determinism framework predicts HaLViT's sharing scheme should work (it is deterministic and the W+Wᵀ split is balanced). Probabilistic Weight Fixing is the state-of-the-art for weight-sharing quantization on ViT-Tiny — relevant for compressing HaLViT's shared parameters to a codebook.

### 4. Knowledge Distillation

**Canonical formulation**: Hinton 2015 (temperature-scaled soft targets) → FitNets/Attention Transfer (intermediate feature matching) → federated KD (DiReDi, Wu 2024 survey) → CoT distillation for LLMs (black-box teacher).

**Edge AI KD sub-field**:
- **Speech/audio**: Kerpicci et al. (ICASSP 2023) — layer-wise KD + joint end-to-end fine-tuning critical for speech transformers; frozen backbone yields significantly worse results
- **Vision captioning**: Kwok et al. (2025) — multi-modal joint distillation (encoder+decoder together) > component-wise; directly applicable to HaLViT-based captioning tasks
- **AIoT**: DiReDi (Sony 2024) — reverse distillation as privacy-preserving model update mechanism (ΔKnowledge upload); FCOS-Lite + KD + INT8 on Sony IMX500 (8MB) confirms INT8+KD as viable ultra-constrained pipeline
- **Federated**: Wu et al. (2024) — four KD roles in federated edge learning; heterogeneous student sizes per device; privacy-preserving compression without raw data transfer

**Insight for HaLViT**: KD is the cleanest post-compression step — train HaLViT with weight sharing, then KD-slim it for a specific edge target. Joint fine-tuning through shared W (not freezing it) is critical. A teacher→HaLViT pipeline where the teacher is a standard ViT distilling into HaLViT student is the natural experiment.

### 5. Tensor Decomposition / Low-Rank

**Core methods**: SVD per-layer (baseline) → Basis Sharing [2410.03765] (shared SVD basis across adjacent layers + per-layer coefficients; beats all per-layer baselines at 20–50% compression) → PEFT (LoRA = low-rank adapter, Diff Pruning = sparse delta, IncreLoRA = incremental rank growth).

**Composition with sharing**: LoRA's rank-r adapter on shared W serves both W→x and Wᵀ→x' paths. QLoRA (4-bit base + LoRA) is directly applicable to HaLViT. LoRAPrune's LoRA-guided structural pruning produces joint importance signals for shared W. IncreLoRA's shared-W allocation must normalize budget by sharing count.

---

## Composition: The Central Thesis Question

Every paper in the wiki was evaluated against the question: *does this technique compose with HaLViT's W+Wᵀ sharing?*

Summary of composition compatibility:

| Technique | Composability with HaLViT sharing | Key constraint |
|-----------|----------------------------------|----------------|
| INT8 QAT (Jacob et al.) | ✅ Direct | Dual-path STE must accumulate from W→x and Wᵀ→x' |
| SmoothQuant (W8A8) | ⚠️ Constrained | Joint calibration: s must satisfy both paths simultaneously |
| GPTQ / AWQ (INT4 PTQ) | ⚠️ Constrained | Joint Hessian / joint activation scaling across both paths |
| OmniQuant (learnable PTQ) | ✅ Natural | LET's differentiable optimization finds joint solution automatically |
| SpQR / SqueezeLLM (3-4 bit) | ✅ Direct | Outlier mask computed once for shared W, applies to both paths |
| QuIP (2-bit) | ⚠️ Open question | Orthogonal rotation must simultaneously incoherence both paths |
| BitNet b1.58 (1-bit) | ⚠️ Train-from-scratch | Ternary shared W; training-time STE is naturally joint |
| Wanda (activation pruning) | ✅ Natural | Activation norms accumulate from both paths; no modification needed |
| SparseGPT (one-shot pruning) | ⚠️ Constrained | Symmetric mask required; W+Wᵀ coupling not in current algorithm |
| SNIP (gradient-sensitivity) | ✅ Natural | Backward pass accumulates from both paths simultaneously |
| Movement Pruning (fine-tuning) | ✅ Natural | Score gradient accumulates from both paths per step |
| BESA (differentiable pruning) | ✅ Natural | Blockwise reconstruction integrates both paths in block objective |
| LLM-Pruner (structural) | ⚠️ Extension needed | Dependency graph must model W+Wᵀ coupling constraint |
| QLoRA (4-bit + LoRA) | ⚠️ Open question | Single adapter BAᵀ serves W path; Wᵀ path needs transposed correction |
| Basis Sharing (cross-layer SVD) | ✅ Compatible | SVD of shared W produces one basis serving both paths |
| KD (teacher→student) | ✅ Direct | Standard KD; joint fine-tuning through shared W critical |

**Thesis opportunity**: the symmetric mask constraint for SparseGPT + shared W is a concrete, tractable experiment. Constrained SparseGPT (restrict solutions to symmetric mask space) vs unconstrained SparseGPT at same sparsity ratio on a trained HaLViT. How much accuracy is lost by enforcing symmetry? This quantifies the cost of the W+Wᵀ coupling for pruning.

---

## Emerging Themes Across the Wiki

1. **Outlier protection is universal**: LLM.int8(), SmoothQuant, SpQR, SqueezeLLM, ATOM — all converge on the same insight. QuIP formalizes this as "incoherence" and proves it's the optimal principle. The practical methods are heuristic implementations of the same mathematical idea.

2. **Differentiable over hand-crafted**: OmniQuant (quantization params), BESA (sparsity rates), IncreLoRA (rank allocation) — a consistent 2024 trend of making previously hand-crafted hyperparameters learnable via SGD. This is the "learnable PTQ" movement.

3. **Composition is the frontier**: QLoRA (quant+PEFT), LoRAPrune (PEFT+pruning), Deep Compression (pruning+quant+Huffman), BESA+OmniQuant (pruning+quant jointly). No paper has composed all three with weight sharing. HaLViT + quant + pruning is the next composition step.

4. **Two schools for LLM-scale**: PTQ (GPTQ, AWQ — works on existing models, no retraining) vs training-from-scratch (BitNet — requires new training run). These are diverging paradigms; a thesis must choose one or bridge them.

5. **NAS over compression pipeline**: HW-EvRSNAS, MO-HDNAS, HPC2Edge — hardware-aware NAS is now a credible alternative to post-hoc compression. Applied to HaLViT, NAS over sharing topologies could automate the configuration search.

6. **Deployment compilers are the last mile**: TVM, TensorRT, OpenVINO — flagged by the Edge AI survey (2501.15014) as the gap between compressed models and actual hardware speedups. Not covered in detail in the wiki; worth one source page on TVM auto-tuning for ViT models.

---

## Thesis Direction Assessment (Updated)

> **Decision LOCKED (2026-06-12):** the advisor selected **Option 2 — W+Wᵀ → language models** (the final row of the table below), framed as **pure research, no product** — see [[analyses/thesis-options-2026-05]]. **Option 1 (HaLViT-Edge)** and its edge-deployment structure are **dropped**; rows 2–6 below remain as compression-background reference only, not the thesis. Model scale may go smaller than GPT-2 (report keeps a range ≈10M–124M); the small-LM four-way comparison (none/ALBERT/HaLViT/both, from scratch under matched budget) runs entirely on the RTX 4070, with TinyLlama as an optional cluster-gated stretch.

| Direction | Technique mix | Novelty | Tractability | Verdict |
|-----------|--------------|---------|--------------|---------|
| HaLViT + INT8 QAT | weight sharing + QAT | Low | High | Baseline experiment; confirms composability |
| HaLViT + SmoothQuant/OmniQuant (W4A4) | weight sharing + learnable PTQ | Medium | Medium | Joint calibration constraint is the novel contribution |
| HaLViT + Wanda/SNIP pruning | weight sharing + activation pruning | Medium | High | Joint saliency signal property is well-motivated |
| HaLViT + Symmetric SparseGPT | weight sharing + constrained one-shot pruning | High | Medium | First study of symmetric mask constraint cost |
| HaLViT + QLoRA fine-tuning | weight sharing + 4-bit PEFT | Medium | High | Transposed-path LoRA adapter is the open question |
| HaLViT → KD → Edge deployment | weight sharing + KD + INT8 + TVM | Low | High | Engineering clarity; strong for benchmarking chapter |
| HaLViT adaptation to LLM FFN (Idea D) | weight sharing expanded to language | High | Low | Research risk; no trained model exists |

**Locked thesis structure (Option 2)**: (1) mathematical validation of W+Wᵀ for LM attention/FFN under causal masking, RoPE, SwiGLU; (2) small from-scratch GPT-style LM, four-way comparison (none / ALBERT cross-layer / HaLViT W+Wᵀ / both) on WikiText-103 under a matched budget, measuring perplexity-per-parameter; (3) optional TinyLlama stretch (SwiGLU/RoPE/GQA), gated on SP4CING/UHEM cluster access. No edge-deployment/product chapter — research-only per the advisor.
