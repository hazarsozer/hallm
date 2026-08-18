---
type: model
title: "HaLViT"
full_name: "Half a Vision Transformer"
authors: ["Onur Can Koyun", "Behçet Uğur Töreyin"]
institution: "SP4CING, Istanbul Technical University"
source_page: "[[sources/halvit]]"
---

## What It Is

HaLViT is a parameter-efficient Vision Transformer that replaces two separate weight matrices in each attention and FFN block with a single matrix **W** and its transpose **Wᵀ**, halving the parameter count while maintaining competitive accuracy. Applied to both ViT (Transformer layers) and ResNet (Bottleneck layers).

## Model Variants

| Variant | Params | FLOPs | ImageNet Top-1 |
|---------|--------|-------|----------------|
| HaLViT-Tiny¹ | 11.1M | 4.6G | 77.3% (300 ep) |
| HaLViT-Tiny² | 11.1M | 4.6G | 78.8% (600 ep) |
| HaLViT-Small | 43M | — | — |
| HaLViT-Medium | 43M | 16.8G | 81.3% |
| ResNet50* | 13.4M | 4.09G | 75.1% Top-1 |
| HaLViT* (extreme) | 9M | 16.8G | 67.6% |

## Core Equations

**MHA**: `x̂ = MHA(Wq·x, Wkv·x, Wkvᵀ·x)`, projection via `Wqᵀ·x̂`  
**FFN**: `FFN(x) = Wᵀ·F(W·x + b₁) + b₂`  
**Bottleneck**: `B(x) = Wᵀ·G(W·x)`

## Key Benchmark Numbers

- Beats PVTv2-B1 (14M → 11.1M, same accuracy 78.8%)
- Matches ResNet50 (25.6M → 13.4M, −1.0 pt Top-1)
- Transfer: CIFAR-10 99.2%, CIFAR-100 91.0% (HaLViT-M)

## Papers That Should Reference This

All papers in the literature review that discuss weight sharing, parameter-efficient ViTs, or edge deployment of vision models should cross-reference this entity.

Referenced by: [[concepts/weight-sharing]], [[sources/halvit]], [[sources/halvit-official-code]], [[sources/halsp]], [[analyses/wplusw-lm-review-2026-08]]
