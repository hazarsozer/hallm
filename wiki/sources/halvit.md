---
title: "HaLViT: Half of the Weights are Enough"
arxiv: null
venue: "CVPR Workshop (Computer Vision Foundation, Open Access)"
authors: ["Onur Can Koyun", "Behçet Uğur Töreyin"]
affiliation: "SP4CING, Dept. of AI and Data Engineering, Istanbul Technical University"
year: 2023
category: weight-sharing
status: ingested
source_file: raw/papers/halvit.pdf
funding: "TUBITAK 1515, BAP MGA-2024-45372 (HIZDEP), UHEM"
---

## Problem

Vision Transformers (ViTs) and ResNets require a large number of parameters due to the use of separate weight matrices for each linear transformation within attention and bottleneck layers. This makes them unsuitable for resource-constrained deployment without an explicit compression step.

## Method

HaLViT exploits the **row and column spaces of a single weight matrix W** to replace two separate weight matrices in both Transformer and Bottleneck layers.

### Mathematical Foundation

For a weight matrix **W** (m×n) applied to input **x**:
- `y = Wx` projects **x** into the **column space** of **W**
- After a nonlinear activation F(·), the output `F(Wx)` no longer resides in the column space
- Therefore `Wᵀ·F(Wx)` is a genuinely **independent transformation** — not a redundant one
- This justifies using **W** and **Wᵀ** as two distinct transformations within the same layer

This is the theoretical core: nonlinearity breaks the column-space constraint, making W and Wᵀ independently expressive.

### Multi-Head Attention (MHA)

Standard MHA uses four separate matrices: Wq, Wk, Wv, Wproj.

HaLViT collapses this to two:
- **Shared Wkv**: keys = Wkv·x, values = Wkvᵀ·x (row and column space of same matrix)
- **Shared Wq**: queries = Wq·x, final projection = Wqᵀ·x̂

```
x̂ = MHA(Wq·x, Wkv·x, Wkvᵀ·x)
Proj(x̂) = Wqᵀ·x̂
```

This **halves the parameter count** of each attention block.

### Feed Forward Network (FFN)

Standard: `FFN(x) = W₂·F(W₁·x + b₁) + b₂`  
HaLViT:  `FFN(x) = Wᵀ·F(W·x + b₁) + b₂`

Single W serves dual purpose: forward projection (W) and output projection (Wᵀ). Parameter count halved.

### Bottleneck Layer (ResNet)

Standard bottleneck uses separate W₁, W₂ for 1×1 convolutions.  
HaLViT: `Bottleneck(x) = Wᵀ·G(W·x)` — same W reused via transpose.

Applied to stages 3 and 4 of ResNet (sharing in stages 1–2 shows diminishing returns per ablation).

## Results

### ImageNet-1K Classification (224×224)

| Model | Params | FLOPs | Top-1 |
|-------|--------|-------|-------|
| DeiT-Small | 22M | 4.6G | 79.9% |
| PVTv2-B1 | 14M | 2.1G | 78.7% |
| **HaLViT-T¹** | **11.1M** | **4.6G** | **77.3%** |
| **HaLViT-T²** | **11.1M** | **4.6G** | **78.8%** |
| **HaLViT-M** | **43M** | **16.8G** | **81.3%** |

HaLViT-T² (600 epochs) **outperforms PVTv2-B1 with 2.9M fewer parameters**.

### COCO Object Detection + Instance Segmentation (Mask R-CNN)

| Model | Params | APᵇ | APᵐ |
|-------|--------|-----|-----|
| ResNet101 | 63.2M | 40.4 | 36.4 |
| PVT-M | 63.9M | 42.0 | 39.0 |
| **HaLViT-T** | **30.8M** | **35.3** | **33.3** |
| **HaLViT-M** | **63.0M** | **42.3** | **39.2** |

HaLViT-M exceeds PVT-M with 0.9M fewer parameters.

### Transfer Learning

| Model | CIFAR-10 | CIFAR-100 | Flowers-102 |
|-------|----------|-----------|-------------|
| HaLViT-T | 98.7% | 90.3% | 96.5% |
| HaLViT-M | 99.2% | 91.0% | 98.3% |

Competitive with ViT-B/16 (98.1%, 87.1%, 89.5%) at fraction of parameters.

### ResNet50 with HaLViT (Table 4)

| Model | Params | Top-1 | Top-5 |
|-------|--------|-------|-------|
| ResNet50 | 25.6M | 76.1% | 92.8% |
| **ResNet50*** | **13.4M** | **75.1%** | **92.8%** |

~2× parameter reduction with only 1.0 point Top-1 loss.

## Ablation Findings

- **Extreme cross-layer sharing** (HaLViT*, 9M params, all layers except Wq): Top-1 = 67.6% — convergence achieved but accuracy drops sharply. The within-layer (W + Wᵀ) scheme is what makes the approach competitive, not aggressive cross-layer sharing.
- **ResNet early-stage sharing** (stages 1–2): −0.9 points accuracy. Early features are more sensitive to parameter sharing.

## Significance

**This is the anchor paper of the graduation thesis.** Three reasons this matters:

1. The W + Wᵀ mechanism is **architecturally general** — applicable to any model with attention or bottleneck layers (ViT, ResNet, and explicitly noted: language models).
2. The Discussion section **explicitly names pruning + quantization composition** as future work — this is the open research direction the thesis should target.
3. The paper comes from **SP4CING at ITU** — direct institutional context. The thesis builds on this group's prior work.

**Official code** (inspected 2026-08-17): [[sources/halvit-official-code]] — CNN half only (no ViT/attention code released); the shared `convp` is reused across **all blocks in a stage** (intra + cross-block sharing composed). Follow-up work: [[sources/halsp]].

## Open Questions Raised

- Can HaLViT's W + Wᵀ sharing compose with post-training quantization (GPTQ/AWQ) without amplifying shared-weight quantization errors?
- Can structured pruning of attention heads interact with shared Wkv/Wq without ambiguous gradient signals?
- Can the row/column-space argument be extended to the FFN of a language model (as proposed in Thesis Idea D)?
- Is there an optimal sharing assignment (which layers share, which don't) discoverable via NAS rather than hand-design?

## Related

[[concepts/weight-sharing]] — core mechanism  
[[concepts/quantization]] — composability target  
[[concepts/pruning]] — composability target  
tensor-decomposition — mathematical adjacency (W + Wᵀ vs. SVD)  
[[concepts/knowledge-distillation]] — possible post-training compression step  
[[entities/halvit-model]] — model-level reference card  
vit — baseline architecture  
deit — training recipe used  
