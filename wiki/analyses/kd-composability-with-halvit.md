---
query: "How does Knowledge Distillation compose with HaLViT's W+Wᵀ weight sharing — as a student, as a teacher, and as a post-compression recovery tool?"
date: 2026-04-30
sources_consulted:
  - "wiki/concepts/knowledge-distillation.md"
  - "wiki/sources/halvit.md"
  - "wiki/sources/2210.16611v2.md"
  - "wiki/sources/2603.26145v1.md"
  - "wiki/sources/2407.09562v3.md"
  - "wiki/sources/2602.14301v1.md"
  - "wiki/sources/2305.11627.md"
  - "wiki/sources/2402.16880.md"
---

## Executive Summary

Knowledge Distillation is the one compression technique in the thesis corpus that creates **no structural conflict** with HaLViT's W+Wᵀ weight sharing — and may actively benefit from it. KD operates by adjusting the student's loss function, not its weight structure. The shared weight W receives gradient signals from the KD loss just as it does from the task loss; the sharing constraint is automatically maintained. The more interesting questions are positional: whether HaLViT is more useful as a *student* (learning from a large teacher) or as a *teacher* (guiding a smaller model), and how KD composes with the pruning and quantization pipelines established in the prior analyses.

---

## 1. HaLViT as a Student: Receiving Knowledge from a Teacher

### 1.1 Output (Logit) Distillation

In standard Hinton-style KD, the student minimizes a combined loss:

```
L = (1 − α)·L_CE(student_output, hard_labels) + α·L_KL(student_output, teacher_output)
```

For HaLViT as student and a standard DeiT-Base or ViT-Base as teacher, the logit distillation operates entirely on the final classification outputs. The teacher never sees HaLViT's internals; HaLViT's W+Wᵀ constraint is fully encapsulated within the student and is transparent to the distillation objective.

**Gradient flow**: During backpropagation, ∂L/∂W accumulates contributions from both L_CE (task) and L_KL (distillation). For HaLViT's shared W, the gradient also accumulates from both the W→x and Wᵀ→x' pathways within the student's own forward pass. These accumulations add correctly — output KD simply adds more gradient signal to W without conflicting with the sharing structure.

**Practical implication**: A larger DeiT-Base/ViT-Base teacher (86M parameters, 81.8% ImageNet) can guide HaLViT-T (11.1M parameters) during training. The teacher's richer inter-class similarity information (encoded in soft targets) could compensate for the representational constraint that HaLViT's weight sharing imposes — the sharing reduces capacity, but KD compensates by providing better training signal. This is the DeiT insight [Touvron et al., 2021] applied specifically to HaLViT's compressed architecture.

**Expected benefit**: 1–2% Top-1 accuracy improvement on ImageNet over vanilla training, based on DeiT's results (DeiT-S with distillation token: 80.0% vs 79.8% without, at the same parameter count). The gain may be larger for HaLViT because the weight-sharing constraint makes standard training harder.

### 1.2 Feature (Intermediate) Distillation

Feature distillation matches intermediate representations between teacher and student layers. This is more information-rich but requires careful layer alignment.

**Key challenge for HaLViT**: In standard ViT-Base (teacher), each attention block has independent Wq, Wk, Wv, Wproj matrices. The intermediate features after attention are determined by all four independently. In HaLViT (student), keys and values are produced by a shared Wkv and its transpose — the key-feature space and value-feature space are constrained to be related by transposition (after the nonlinearity separates them). A teacher's key-feature map and value-feature map, trained without this constraint, will not satisfy this relationship.

**Implication**: Naively matching teacher key features to student key features AND teacher value features to student value features creates an inconsistent optimization target — the student cannot simultaneously match the teacher's unconstrained key features AND unconstrained value features while maintaining the W+Wᵀ structural coupling. The student must "choose" which path to prioritize.

**Practical resolution**: Match features at the block level, not the K/V level. Feature distillation at the attention output (after all K/V computations) avoids the W+Wᵀ inconsistency. The student's attention output is compared to the teacher's attention output — both are the result of the full attention computation and don't expose the individual K/V matrices. This is the layer-wise KD pattern used in Kerpicci et al. [arxiv:2210.16611] for speech transformers: match transformer block outputs, not internal projection features.

**From Kerpicci et al.** [arxiv:2210.16611]: Joint fine-tuning of the distilled backbone is critical — freezing the student's shared weights during KD recovery and only training task heads yields significantly worse results. Applied to HaLViT: any post-distillation fine-tuning must update W (not freeze it), allowing the shared tensor to adapt to the distilled representation.

### 1.3 HaLViT's Sharing as an Inductive Bias for KD

The W+Wᵀ constraint forces HaLViT to learn representations where the key-space and value-space of attention are related by transposition (after nonlinearity). This is a regularization. From a KD perspective:

**Positive**: Stronger inductive bias → fewer parameters needed to express the same representation quality → the student can be guided more efficiently by the teacher's soft targets. A highly regularized student is more "willing" to learn from soft targets because it has fewer degrees of freedom to overfit to hard labels.

**Potential negative**: If the teacher's representation fundamentally violates the W+Wᵀ coupling (which it does — teacher has independent Wk and Wv), then feature distillation is asking the student to approximate something structurally incompatible. This is the asymptotic limit of the constraint cost. In practice, because the nonlinearity makes the W and Wᵀ paths genuinely independent (the HaLViT paper's main mathematical argument), they can approximate teacher K and V features independently at the block output level — the coupling is internal to the layer, not visible at the block output.

---

## 2. HaLViT as a Teacher: Guiding a Smaller Student

### 2.1 HaLViT-M → HaLViT-T Distillation

HaLViT-M (43M params, 81.3% Top-1) can serve as teacher for HaLViT-T (11.1M params, 78.8%). This is a natural cascaded compression: both models use the W+Wᵀ sharing scheme, so the teacher's internal representations **are** structurally consistent with the student's architecture.

In this configuration, feature distillation at the K, V level is valid — teacher and student both have the same W+Wᵀ coupling, so their key-space and value-space features are structurally comparable. The teacher's KV features lie in the same algebraic space as the student's KV features (both constrained by the W+Wᵀ relation), making layer-wise feature matching well-defined.

**Expected result**: HaLViT-T distilled from HaLViT-M should outperform HaLViT-T trained from scratch (currently 78.8% at 600 epochs). A rough estimate based on the DeiT T→S distillation: +1–2% Top-1.

### 2.2 HaLViT as a Compressed Teacher in the Pipeline

HaLViT's primary role in the thesis is as the target architecture — the model being compressed. However, after training, a compressed HaLViT can serve as a "specialized teacher" in a two-stage pipeline:

```
Stage 1: Large ViT-Base → distill → HaLViT (student, with W+Wᵀ sharing)
Stage 2: HaLViT → prune/quantize (BESA + OmniQuant from prior analyses)
Stage 3: Compressed HaLViT → distill → ultra-small target (e.g., 3M params for IMX500)
```

Stage 3 applies to the IMX500 deployment scenario: the Sony IMX500 has 8MB memory, which requires ~3M parameters at INT8. A compressed HaLViT-T at INT8 (approximately 3MB at 4-bit or 6MB at 8-bit quantization) may fit; a further distilled micro-model certainly would. This is the pipeline established empirically in [arxiv:2407.09562] (FCOS-Lite + KD + INT8 on IMX500).

---

## 3. KD as Post-Compression Recovery

### 3.1 LoRA Recovery in Pruning (LLM-Pruner, BESA)

From the pruning analysis [analyses/pruning-composability-with-weight-sharing.md], LLM-Pruner and BESA both use LoRA fine-tuning for post-pruning recovery. This is a form of parameter-efficient self-distillation: the pruned model is guided back toward the unpruned model's representations using a small LoRA adapter.

For HaLViT, the LoRA adapter (PQ where P ∈ ℝ^{m×r}, Q ∈ ℝ^{r×n}) covers both the W→x and Wᵀ→x' paths simultaneously — one adapter serves both pathways. This is more parameter-efficient than standard LoRA on unshared weights.

**Extension**: Instead of LoRA recovery guided only by task loss, use the original full-precision HaLViT as a teacher. After pruning, the pruned HaLViT (student) is fine-tuned with combined task loss + KD loss from the original HaLViT (teacher). This is "self-distillation after pruning" — a well-studied pattern that consistently outperforms task-loss-only recovery.

### 3.2 KD After Joint SparseGPT + GPTQ Compression

From the quantization analysis [analyses/quantization-tradeoffs-1to4bit.md], the three-axis compression pipeline ends with joint pruning + quantization. After this compression, a KD recovery step using the full-precision HaLViT as teacher would be the highest-quality recovery option — better than LoRA alone, at the cost of requiring the full-precision model in memory during recovery.

**Practical concern**: Requiring both compressed HaLViT (student) and full-precision HaLViT-M (teacher) in GPU memory simultaneously. HaLViT-M at FP16 is ~86MB; compressed HaLViT-T at INT4 is ~6MB. On a single GPU with 40GB+ memory (A100-40G), this is trivial. On a 16GB consumer GPU (RTX 4080), it is also feasible.

---

## 4. Federated KD for Heterogeneous Edge Deployment

### 4.1 DeepFusion Pattern Applied to HaLViT

DeepFusion [arxiv:2602.14301v1] demonstrates that KD can aggregate knowledge from heterogeneous distributed models into a single stronger model. The key technique: **VAA (View-Aligned Attention)** resolves cross-architecture feature mismatches by projecting features from different architectural families into a common representation before distillation.

Applied to HaLViT in a federated scenario: multiple edge devices might run different compression levels of HaLViT (HaLViT-T on mobile, HaLViT-M on Jetson). A central server could use VAA-style distillation to aggregate device-specific knowledge back into a common HaLViT checkpoint, improving generalization without centralizing raw data.

### 4.2 DiReDi's Bidirectional KD

DiReDi [arxiv:2409.08308v1] shows that reverse KD (edge → cloud) can transport user-specific knowledge without privacy violations. Applied to HaLViT: a personalized edge adaptation (e.g., user's specific gesture classes) can be uploaded as ΔKnowledge = (adapted HaLViT − base HaLViT), not raw data. This is directly relevant if the thesis demonstrates HaLViT on a personalized classification task.

---

## 5. Compatibility Summary

| KD Variant | HaLViT Compatibility | Notes |
|-----------|---------------------|-------|
| Output KD (logit distillation) | **Full** — no modification needed | Gradient flows naturally through shared W from both paths |
| Block-output feature KD | **Full** — match at block output, not K/V level | Avoids W+Wᵀ inconsistency; uses same pattern as Kerpicci et al. |
| K/V-level feature KD (teacher same arch) | **Full** — HaLViT-M → HaLViT-T | Structurally compatible when both teacher and student share weights |
| K/V-level feature KD (teacher standard ViT) | **Partial** — mismatch at K/V level | Match at block output instead; K/V matching is overdetermined |
| LoRA + KD recovery after pruning | **Full** — LoRA covers both paths | Most parameter-efficient recovery option for shared weights |
| Federated KD (VAA-style) | **Supported** — standard FL protocol | VAA handles cross-architecture mismatches |
| Joint fine-tuning (not frozen W) | **Required** | Never freeze shared W during KD fine-tuning (Kerpicci finding) |

---

## 6. Thesis Experiments

### Experiment 1: HaLViT-T trained with DeiT-B teacher (baseline KD)

Apply standard output KD with a DeiT-Base teacher (86M) during HaLViT-T training. Add a distillation token (DeiT pattern). Measure Top-1 on ImageNet-1K.

**Expected**: +1–2% over vanilla HaLViT-T (78.8% → 79.8–80.8%), at zero parameter cost.

### Experiment 2: HaLViT-M → HaLViT-T cascade distillation

Train HaLViT-M (43M) normally, then distill HaLViT-T from HaLViT-M using block-level feature KD. Compare to: (a) HaLViT-T from scratch, (b) HaLViT-T with DeiT-B output KD.

**Hypothesis**: Same-architecture distillation (HaLViT-M → HaLViT-T) outperforms cross-architecture feature KD (DeiT-B → HaLViT-T) because feature spaces are structurally compatible.

### Experiment 3: KD recovery after three-axis compression

Train HaLViT-T → apply BESA 30% pruning → apply OmniQuant INT4 → KD recovery from full-precision HaLViT-T teacher (128 calibration samples, 3h). Compare to: LoRA-only recovery.

**Expected**: KD recovery +0.5–1% Top-1 over LoRA-only recovery at equivalent compute cost.

---

## Sources Consulted

- [[sources/halvit]] — HaLViT; W+Wᵀ sharing; Discussion section names KD composability as future work
- [[concepts/knowledge-distillation]] — full taxonomy; open questions for HaLViT
- [[sources/2210.16611v2]] — Kerpicci: joint fine-tuning critical; block-level KD pattern; 75% size reduction on speech transformers
- [[sources/2603.26145v1]] — MobileViT KD on Jetson Orin Nano; +14% 1-shot accuracy at −69% params
- [[sources/2407.09562v3]] — FCOS-Lite + KD + INT8 on Sony IMX500; confirms KD + quantization pipeline
- [[sources/2602.14301v1]] — DeepFusion; VAA for cross-architecture KD; federated learning pattern
- [[sources/2305.11627]] — LLM-Pruner; LoRA recovery as self-distillation proxy
- [[sources/2402.16880]] — BESA; joint pruning + quantization; composable with OmniQuant
