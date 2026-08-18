---
query: "Which pruning methods are compatible with weight sharing — specifically HaLViT's W+Wᵀ intra-layer sharing — and what modifications are needed?"
date: 2026-04-30
sources_consulted:
  - "wiki/sources/halvit.md"
  - "wiki/sources/1803.03635.md"
  - "wiki/sources/1810.02340.md"
  - "wiki/sources/2301.00774.md"
  - "wiki/sources/2306.11695.md"
  - "wiki/sources/2305.11627.md"
  - "wiki/sources/2402.16880.md"
  - "wiki/concepts/pruning.md"
---

## Executive Summary

HaLViT's W+Wᵀ sharing creates a structural constraint that most pruning methods silently violate: zeroing position (i,j) in W simultaneously zeroes position (j,i) in the Wᵀ pathway, because they share the same physical tensor. A pruning method that ignores this coupling produces inconsistent masks — the pruned W is no longer a valid transpose pair with the masked Wᵀ. The good news is that three of the six methods studied handle this naturally without modification; the other three require extensions that are well-defined and tractable. This analysis classifies every method, identifies the exact required modifications, and maps five concrete thesis experiments.

---

## 1. The Core Composability Problem

### 1.1 What W+Wᵀ Sharing Means for Pruning

In standard pruning, each weight matrix W has one role. The pruning mask M selects which elements of W to zero:

```
W_pruned = M ⊙ W
```

In HaLViT [arxiv:halvit], each shared weight tensor W is used in two roles simultaneously:
- **Forward path**: `y = Wx` (W as a projection matrix, m×n)
- **Transposed path**: `z = Wᵀx'` (Wᵀ as a projection matrix, n×m)

When pruning is applied, the naive approach would produce separate masks M_forward and M_backward. But because W and Wᵀ are the same physical tensor, these cannot be independent. Zeroing element w_{ij} in the forward path simultaneously zeroes w_{ji} in the transposed path, regardless of what M_backward specifies:

```
Forward:   y_i = Σ_j M_ij · w_ij · x_j       (zero if M_ij = 0)
Transposed: z_j = Σ_i M_ji · w_ij · x'_i      (same w_ij — zero if M_ij = 0 regardless of M_ji)
```

**The symmetric mask constraint** resolves this: define a single mask M where M_ij = M_ji (symmetric). Then zeroing (i,j) automatically zeroes (j,i), making the mask simultaneously valid for both uses. Any pruning method that computes a single mask for the shared W matrix and enforces symmetry is compositionally correct.

### 1.2 Specific HaLViT Configurations

HaLViT's MHA uses two shared matrices:
- **Wkv** (d_model × d_k): keys = Wkv·x, values = Wkvᵀ·x — Wkv is typically non-square (rectangular)
- **Wq** (d_model × d_model): queries = Wq·x, projection = Wqᵀ·x̂ — Wq is square

For Wq (square): the symmetric mask constraint is well-defined (M ∈ {0,1}^{n×n}, M = Mᵀ).

For Wkv (non-square, m×n, m≠n): the "transpose" swaps dimensions, so the constraint becomes cross-dimensional. Zeroing element (i,j) in Wkv (used in key projection) zeroes element (j,i) in Wkvᵀ (used in value projection). This is valid — the constraint still holds — but the mask must be explicitly represented as an m×n binary matrix where M_{ij} = 1 implies the (j,i) element of the transposed path is also active. For structured pruning (removing entire heads or rows/columns), this maps cleanly to removing head h from both key and value simultaneously — which is actually *simpler* than the unstructured case.

HaLViT's FFN uses one shared matrix:
- **W** (d_model × d_ffn): forward = W·x, output = Wᵀ·F(W·x + b₁) + b₂

Here W is the same matrix used for both input expansion and output contraction. The symmetric mask applies directly.

---

## 2. Method-by-Method Compatibility Analysis

### Tier 1 — Naturally Compatible: No Modification Required

These methods implicitly compute a joint signal over both the W→x and Wᵀ→x' pathways, making them correct for shared-weight pruning without any algorithmic change.

---

#### 2.1 Wanda [arxiv:2306.11695] — **Best practical choice for post-training pruning**

**Mechanism**: `score(i,j) = |w_ij| × ||x_j||₂` — weight magnitude × L2 norm of the corresponding input activation column, estimated over 128 calibration sequences.

**Why it's naturally compatible**: The activation norm ||x_j||₂ is accumulated over all forward passes during calibration. For shared W in HaLViT, calibration runs the full model — meaning both the W→x and Wᵀ→x' paths execute, and both contribute to the activation statistics seen by column j. The resulting ||x_j||₂ already reflects the importance of column j across *both* uses of W. Wanda thus produces a joint saliency score without any modification.

**Additionally**: The pruning mask derived from Wanda is applied once to the shared tensor — automatically making the same sparsity pattern active for both forward and transposed paths. No symmetric enforcement is needed because Wanda computes one score per element of W directly.

**Limitation**: Wanda's score does not account for the asymmetric use of rows vs. columns in non-square Wkv. The score for element (i,j) reflects the importance of column j (from ||x_j||₂) but not the importance of row i in the transposed path (where row i of Wkv becomes column i of Wkvᵀ). A more complete Wanda extension would accumulate ||x'_i||₂ from the transposed-path inputs as well, producing: `score(i,j) = |w_ij| × (||x_j||₂ + ||x'_i||₂)` — a joint activation-weighted score. This extension is straightforward and would be an original thesis contribution.

**Recommended experiment**: Apply standard Wanda to trained HaLViT; measure accuracy vs sparsity at 10–50%; compare to extended joint-activation Wanda. This is the simplest and lowest-cost starting experiment.

---

#### 2.2 SNIP [arxiv:1810.02340] — **Best choice for pruning at initialization**

**Mechanism**: `s_j = |∂L/∂c_j · w_j| / Σ|∂L/∂c_k · w_k|` — gradient of loss w.r.t. a binary indicator gating each connection, evaluated at initialization.

**Why it's naturally compatible**: PyTorch/JAX autograd accumulates the gradient of L w.r.t. each parameter across *all* uses of that parameter in the forward graph. For HaLViT's shared W, autograd accumulates gradient contributions from the W→x path AND the Wᵀ→x' path in a single backward pass. The resulting ∂L/∂w_ij already contains the joint importance signal — it reflects how much zeroing w_ij would affect both uses of W simultaneously.

No modification to SNIP is needed. The shared-weight structure is actually an *advantage*: each element of W receives gradient signal from two pathways rather than one, making the sensitivity score more informative per parameter.

**Important constraint**: SNIP's joint backward pass must be run on the full HaLViT model — not on the W→x path in isolation. If the Wᵀ path is bypassed during the sensitivity evaluation, the signal degrades to single-path importance, losing the joint benefit. Standard PyTorch training with shared weights handles this automatically.

**Recommended experiment**: SNIP at HaLViT initialization with 30%, 50%, 70% connection removal; train the resulting sparse HaLViT from scratch; compare to unpruned HaLViT accuracy. This is a low-cost training-time experiment.

---

#### 2.3 BESA [arxiv:2402.16880] — **Most principled for post-training blockwise pruning**

**Mechanism**: Blockwise reconstruction error minimization with learned per-layer sparsity rates via differentiable binary masks and STE.

**Why it's naturally compatible**: BESA minimizes the error of the entire transformer block — attention + FFN together — not individual layers. In HaLViT, the W→x path (key/value projection) and Wᵀ→x' path (value/output projection) both fall within the same transformer block. The block-level reconstruction loss accounts for both paths simultaneously: any pruning decision for shared W that degrades the W→x accuracy or the Wᵀ→x' accuracy is penalized directly by the block objective.

Furthermore, BESA's differentiable pruning probability vector is learned for the shared W as a single tensor — the learned mask naturally adapts to the joint block-level error. No coupling constraint needs to be explicitly enforced; the gradient signal from both uses already flows back to the same set of pruning probability parameters.

BESA also learns per-layer sparsity rates automatically — it will naturally assign higher sparsity to shared W matrices that are more compressible across both uses, and lower sparsity where one of the two paths is sensitive.

**This is the most principled pruning approach for HaLViT and is recommended for the primary thesis experiment.**

---

### Tier 2 — Compatible with Modification: Well-Defined Extensions Required

---

#### 2.4 SparseGPT [arxiv:2301.00774] — Requires symmetric mask constraint

**Mechanism**: Hessian Synchronization — shared recursive sequence of inverse Hessians H^{-1}_{U_j} to minimize per-layer reconstruction error ‖W X - (M ⊙ Ŵ) X‖₂² in one shot, no retraining.

**Composability issue**: SparseGPT's Hessian H = XX^T is derived from the W→x forward path only. The reconstruction objective minimizes the error of the output Wx for the forward path. For the Wᵀ pathway, the relevant Hessian would be H' = X'(X')^T where X' are the inputs to the transposed path. These two Hessians are not the same — they capture different activation statistics.

Consequence: an unconstrained SparseGPT mask is optimal for the W→x path but likely suboptimal for the Wᵀ→x' path. The columns of Wkv selected for pruning may not be the rows of Wkvᵀ that are least important.

**Modification — Joint Hessian**: Replace H = XX^T with H_joint = XX^T + X'(X')^T, where X' is the input activation matrix for the transposed path. This joint Hessian accounts for the importance of each weight position across both uses. The Cholesky-based recursive inverse still applies; only the calibration data accumulation changes (collect both X and X' during forward passes).

**Modification — Symmetric mask**: As a simpler alternative, constrain the OBS mask search to the symmetric subspace: at each pruning step, only consider symmetric (i,j)+(j,i) pairs for removal. This approximately halves the effective number of free pruning decisions but guarantees correctness. For non-square Wkv, the symmetric constraint crosses the two transposed paths naturally.

**Joint SparseGPT + GPTQ**: SparseGPT natively supports merging with GPTQ into a single compression pass [Section 3.5, arxiv:2301.00774]. For HaLViT, applying joint-Hessian SparseGPT + joint-Hessian GPTQ in one pass would yield three axes of compression simultaneously (weight sharing + pruning + quantization). This is the most ambitious thesis experiment.

---

#### 2.5 Lottery Ticket Hypothesis / IMP [arxiv:1803.03635] — Requires symmetric mask + accounts for entangled initialization

**Mechanism**: Iterative Magnitude Pruning (IMP) with weight rewind — train → prune by magnitude → reset to θ₀ → repeat. Winning ticket subnetwork f(x; m⊙θ₀) matches full-network accuracy at ≤10-20% of parameters.

**Composability issue — mask**: Standard IMP applies magnitude pruning to each weight independently. For shared W, independent per-element magnitude pruning violates the symmetric constraint: the mask is not guaranteed to satisfy M_ij = M_ji.

**Fix**: After each IMP pruning step, symmetrize the mask: M ← (M ∨ Mᵀ) (keep a position if either M_ij or M_ji recommends keeping it — union), or M ← (M ∧ Mᵀ) (keep only if both agree — intersection). The union strategy preserves more connections; the intersection strategy achieves higher sparsity. Which is better for HaLViT is an empirical question.

**Composability issue — initialization**: HaLViT's single θ₀ must simultaneously serve as a good initial condition for the W→x path AND the Wᵀ→x' path. In the LTH framework, winning tickets are distinguished by their initialization, not just their structure. For shared weights, θ₀ must "win the lottery" for both uses at once. Two hypotheses:
- **Easier lottery** (optimistic): shared weights receive gradient from both paths during training — more gradient signal per parameter → stronger optimization signal → better initialization at convergence → easier to find winning tickets.
- **Harder lottery** (pessimistic): the two paths place conflicting requirements on θ₀ — a good initialization for the W→x path may be a poor initialization for the Wᵀ→x' path — entangled lottery reduces the chance of winning.

Empirically testing this on ViT-scale HaLViT models would be a concrete original contribution.

**Feasibility**: LTH with full IMP cycles is feasible for HaLViT on ViT-Small/Base (ViT runs in hours, not weeks). This is not feasible for LLM-scale models (SNIP or SparseGPT are the practical alternatives there).

---

#### 2.6 LLM-Pruner [arxiv:2305.11627] — Requires extended dependency graph

**Mechanism**: Three stages — dependency graph construction → gradient+Hessian importance scoring → LoRA recovery. The dependency graph enumerates all coupled structures (neurons/heads that must be pruned together due to graph topology).

**Composability issue**: LLM-Pruner's dependency detection assumes each weight matrix has a single role in the computation graph. For standard LLMs, this is correct. For HaLViT's Wkv, the same node W appears in two distinct edges of the computation graph: W→Key and Wᵀ→Value. A standard topological traversal treats these as separate dependencies, potentially grouping them inconsistently.

**Fix**: Add a W+Wᵀ coupling edge to the dependency graph: whenever W is used in both W→x and Wᵀ→x' roles, declare them a coupled pair. Pruning group i from the W→Key path must simultaneously prune the corresponding group from the Wᵀ→Value path. In practice for structured pruning (attention heads): removing head h from HaLViT's attention block removes rows (h-1)·d_h to h·d_h from Wkv for the key projection, AND implicitly removes columns (h-1)·d_h to h·d_h from Wkvᵀ for the value projection (same physical rows in W). This coupling is automatically satisfied for structured head pruning — making LLM-Pruner actually simpler for HaLViT than it appears.

**LoRA recovery advantage**: LLM-Pruner's recovery stage uses a LoRA adapter added to each weight matrix. For shared W in HaLViT, a single LoRA adapter W + AB simultaneously updates both the forward path (W + AB)x and the transposed path ((W + AB)ᵀ x' = (Wᵀ + BᵀAᵀ)x'). One adapter covers both uses — recovery is thus potentially *more* parameter-efficient under weight sharing than in unshared models. This is a concrete strength of combining LLM-Pruner-style recovery with HaLViT.

---

### Summary Compatibility Table

| Method | When Applied | Compatibility | Key Issue | Fix Required |
|--------|-------------|---------------|-----------|--------------|
| Wanda | Post-training | **Natural** | Ignores transposed-path activation | Optional: joint ||x_j||₂ + ||x'_i||₂ score |
| SNIP | At initialization | **Natural** | None — autograd handles joint signal | None |
| BESA | Post-training | **Natural** | None — block objective covers both paths | None |
| SparseGPT | Post-training | Requires mod | H = XX^T is forward-path only | Symmetric mask OR joint Hessian H + H' |
| LTH / IMP | During training | Requires mod | Independent per-element mask violates W=Wᵀ symmetry | Symmetrize mask (union or intersection) after each round |
| LLM-Pruner | Post-training | Requires mod | Dependency graph misses W+Wᵀ coupling | Add coupling edge; LoRA recovery covers both paths automatically |

---

## 3. Structured vs. Unstructured Pruning Under Weight Sharing

### 3.1 Unstructured Pruning

Zeroes individual weight elements. The symmetric mask constraint applies directly. All six methods above operate in this regime (SparseGPT, Wanda, SNIP natively; LTH with symmetrized mask; BESA via its differentiable element-wise masks; LLM-Pruner for its unstructured variant).

Hardware note: unstructured sparsity requires sparse matrix hardware (NVIDIA Ampere's 2:4 pattern, or CPUs with DeepSparse). SparseGPT's 2:4 semi-structured sparsity achieves 1.54–1.79× GPU speedup on Ampere. For HaLViT on ViT, 2:4 semi-structured is applicable but requires the symmetric mask to be compatible with the 2:4 pattern — a constrained optimization not yet studied.

### 3.2 Structured Pruning (Head/Neuron Removal)

Removes entire attention heads or FFN neurons, producing dense smaller weight matrices.

For **attention head pruning** in HaLViT: removing head h from Wkv removes rows (h-1)·d_h through h·d_h from W. In the transposed path (Wkvᵀ), those rows become columns — also removed. Head removal is **automatically symmetric**: a whole head is a contiguous block in both W and Wᵀ, so structured pruning of attention heads does not require any special symmetric-mask machinery. This is an advantage of structured pruning for weight-sharing architectures.

For **FFN neuron pruning**: HaLViT's FFN uses W for expansion and Wᵀ for contraction. Removing neuron k from the FFN removes row k of W (the expansion) AND column k of Wᵀ (the contraction). Again automatically symmetric.

**LLM-Pruner's dependency graph** captures exactly these structured removal patterns — once the W+Wᵀ coupling edge is added, structured pruning under HaLViT's sharing is clean and correct.

### 3.3 Accuracy Cost Comparison

Structured pruning typically incurs higher accuracy cost than unstructured at the same compression ratio (fewer degrees of freedom in mask selection), but produces dense matrices with immediate hardware speedup. For HaLViT deployed on edge hardware without sparse tensor support, structured pruning may be the only path to real-world speedup.

LLM-Pruner achieves 20% structural pruning of LLaMA-7B with 94.97% performance retained [arxiv:2305.11627]. An analogous 20% head pruning of HaLViT-T (11.1M) would target ~8.9M parameters. At that scale, the attention blocks at 2× (from W+Wᵀ sharing) would contribute ~3× overall compression when combined — matching small ViT variants like DeiT-Ti (5.7M) with potentially much higher accuracy.

---

## 4. Composition Order and Pipeline Design

### 4.1 When to Prune Relative to Weight Sharing

HaLViT's weight sharing is **baked in at training time** — it is an architectural constraint, not a post-hoc compression step. This means the composition pipeline is:

```
[Train HaLViT with W+Wᵀ sharing] → [Prune W] → [Quantize W]
```

Not:
```
[Train dense] → [Share weights] → [Prune]
```

This ordering matters because:
1. Post-training methods (Wanda, SparseGPT, BESA) see the already-shared W — the calibration data captures the joint activation statistics naturally.
2. SNIP is applied *before* training — it pruning HaLViT at initialization with the shared-weight architecture already in place.
3. LTH runs *during* training — the shared structure is maintained throughout, and IMP operates on the shared tensor directly.

### 4.2 Joint Pruning + Quantization

SparseGPT's Section 3.5 [arxiv:2301.00774] proves that joint sparsification+quantization in a single pass outperforms sequential application:

```
50% sparse + 4-bit (joint): OPT-175B PPL 8.55
3-bit GPTQ (quantization only): OPT-175B PPL 8.68
```

For HaLViT, the ideal compression pipeline would be:
1. **Train** HaLViT with W+Wᵀ sharing (intra-layer weight sharing baked in)
2. **Optional cross-layer sharing** (Basis Sharing [arxiv:2410.03765]) — post-training SVD across adjacent ViT blocks
3. **Joint prune + quantize** in one pass — modified SparseGPT+GPTQ with joint Hessian H + H' to account for both paths

This three-axis compression (sharing + pruning + quantization) in a unified calibration pass is the most ambitious but also the most principled endpoint of the thesis trajectory.

### 4.3 LoRA Recovery as a Universal Bridge

Both LLM-Pruner [arxiv:2305.11627] and BESA [arxiv:2402.16880] incorporate LoRA-based recovery after pruning. For HaLViT's shared W:

A LoRA adapter `W_adapted = W + AB` (where A ∈ ℝ^{m×r}, B ∈ ℝ^{r×n}, r ≪ min(m,n)) modifies both paths simultaneously:
- Forward: `(W + AB)x`
- Transposed: `(W + AB)ᵀ x' = (Wᵀ + BᵀAᵀ)x'`

One adapter with 2r rank matrices covers both the W and Wᵀ uses — the recovery is at most as costly as standard LoRA and potentially more efficient (one adapter serves two roles). The LoRA adapter does not break the W+Wᵀ pairing; it maintains it by modifying the shared tensor, which both paths then use via their respective access patterns.

---

## 5. Open Research Questions and Thesis Experiments

### Experiment 1: Wanda on HaLViT (Lowest-cost baseline)

**Setup**: Train HaLViT-T on ImageNet-1K; apply Wanda with 128 calibration images; measure accuracy vs sparsity at 10%, 20%, 30%, 50% unstructured sparsity.

**Hypothesis**: Wanda's joint activation signal (accumulated from both W→x and Wᵀ→x' forward passes) produces a pruning mask that degrades accuracy more gracefully than magnitude pruning alone.

**Control**: Magnitude pruning applied at the same sparsity levels, as a baseline.

**Expected cost**: ~2 hours calibration on a single GPU (same as Wanda on LLaMA-7B, but smaller model).

---

### Experiment 2: BESA on HaLViT (Primary post-training experiment)

**Setup**: Apply BESA to trained HaLViT-T; let it learn per-layer sparsity rates; compare to uniform-sparsity Wanda at the same overall parameter budget.

**Hypothesis**: BESA's block-level objective with automatically learned sparsity rates achieves lower accuracy loss than Wanda at the same compression ratio, because it can assign lower sparsity to the sensitive early ViT blocks and higher sparsity to the redundant middle blocks.

**Expected contribution**: First application of differentiable blockwise pruning to a weight-sharing vision architecture.

---

### Experiment 3: Symmetric-Mask SparseGPT (Most novel modification)

**Setup**: Implement joint-Hessian SparseGPT for HaLViT: collect both X (W→x inputs) and X' (Wᵀ→x' inputs) during calibration; compute H_joint = XX^T + X'(X')^T; run OBS with this joint Hessian; enforce symmetric mask M_ij = M_ji.

**Hypothesis**: The joint Hessian produces a pruning mask more accurate for the shared-weight operator than the single-path Hessian, measurable as lower PPL/higher top-1 at the same sparsity.

**Expected contribution**: First formulation of a symmetric-mask constraint for OBS-style pruning under W+Wᵀ sharing.

---

### Experiment 4: SNIP at HaLViT Initialization

**Setup**: Apply SNIP to HaLViT-T at Glorot initialization with a single ImageNet mini-batch; prune 50%, 70%, 90% of connections; train the sparse architecture to full convergence.

**Hypothesis**: SNIP's gradient signal, accumulated from both W→x and Wᵀ→x' in the same backward pass, produces a sparse HaLViT that is competitive with post-training pruning at equivalent sparsity — with near-zero additional compute overhead.

**Expected cost**: One forward+backward pass at initialization + standard training time.

---

### Experiment 5: Symmetric IMP + Weight Rewind (LTH for HaLViT)

**Setup**: Run IMP with 20% pruning per round, symmetric mask (union strategy), weight rewind to θ₀; repeat 3–5 rounds. Compare winning ticket top-1 accuracy to unpruned HaLViT at the same parameter count.

**Hypothesis (shared-weight lottery)**: Shared parameters receive gradient from two paths → stronger gradient signal → θ₀ is better "adapted" by SGD → higher probability of winning the initialization lottery. Counter-hypothesis: two conflicting path requirements entangle the lottery, reducing winning ticket quality.

**Expected cost**: ~5 IMP rounds × 600-epoch training = feasible with 2–4 GPUs over several days.

---

## 6. Synthesis: Which Methods Should HaLViT Use?

| Use Case | Recommended Method | Rationale |
|----------|--------------------|-----------|
| Training-time pruning (sparsest, most accurate subnetwork) | SNIP at init → standard training | Zero overhead; joint sensitivity signal handles sharing naturally |
| Post-training pruning (fastest, no retraining) | Wanda | Simpler than SparseGPT, competitive accuracy, zero modification needed |
| Post-training pruning (best accuracy) | BESA | Block-level objective naturally integrates both paths; best PPL at 50% |
| Post-training pruning + quantization (most compression) | Modified SparseGPT + GPTQ (joint Hessian) | Single-pass; best total compression |
| Structural head removal (edge hardware, no sparse support) | LLM-Pruner with W+Wᵀ coupling extension | Head pruning automatically symmetric; LoRA recovery covers both paths |
| Theoretical best subnetwork (with training budget) | IMP/LTH with symmetric mask | Best accuracy-sparsity tradeoff; directly tests shared-weight initialization lottery hypothesis |

---

## Sources Consulted

- [[sources/halvit]] — HaLViT; W+Wᵀ sharing; pruning composability as explicit open future work
- [[sources/1803.03635]] — Lottery Ticket Hypothesis; IMP + weight rewind; initialization lottery
- [[sources/1810.02340]] — SNIP; single-shot sensitivity at initialization; joint gradient for shared weights
- [[sources/2301.00774]] — SparseGPT; Hessian synchronization; joint SparseGPT+GPTQ single-pass pipeline
- [[sources/2306.11695]] — Wanda; |w|×||x||₂ criterion; joint activation saliency for shared weights
- [[sources/2305.11627]] — LLM-Pruner; dependency graph; LoRA recovery; W+Wᵀ coupling extension
- [[sources/2402.16880]] — BESA; blockwise differentiable pruning; block objective naturally handles sharing
- [[concepts/pruning]] — unified taxonomy; symmetric mask open question; full open-questions list
