---
title: "HaLViT Official Code Release (sp4cing-itu/halvit)"
arxiv: "n/a (code release for CVPR 2024 Workshop paper)"
authors: ["SPACING Lab, ITU (Koyun et al.)"]
year: 2024
category: weight-sharing
status: ingested
source_file: https://github.com/sp4cing-itu/halvit (inspected 2026-08-17, commit bd971f3)
---

## Problem
The official code accompanying [[sources/halvit]]. Not a full training release — the repo contains **only the CNN half** of the paper: `halvit_cnn/resnet50_halvit.py` and `halvit_cnn/resnext101_halvit.py` (~520 lines, no training scripts, no data pipeline). **The Vision Transformer / attention code is not published.**

## Method (as released)
- One shared 1×1-conv weight `convp` of shape `(mid, out, 1, 1)` per stage. Entry projection is `F.conv2d(x, convp)` (out→mid channels); exit projection reuses the transpose `convp.permute(1, 0, 2, 3)` (mid→out). This is the W/Wᵀ mechanism exactly as described in the paper.
- The 3×3 spatial convs and all BatchNorms remain **independent per block**.
- Init: `xavier_normal_` (ResNet-50) / `kaiming_normal_` (ResNeXt-101) — explicit init of the shared parameter, scheme not held constant across files.
- Sharing applied only to the **later stages** (ResNet-50 layer3/layer4); early stages use standard bottlenecks.

## Results
No training code or logs in the repo; results live in the paper ([[sources/halvit]]). The ResNeXt file prints a parameter-reduction accounting (`shared_params` via the `convp` name filter).

## Significance
Three findings that matter for the thesis:
1. **The official code composes both sharing axes.** `convp` is defined once *per stage* and reused across **all blocks in that stage** — intra-layer W/Wᵀ *and* cross-block (depth) sharing simultaneously, with only normalization kept per-block. This is direct precedent from the authors' own code for our combined arm (A3) and for the per-layer LayerNorm choice ([[entities/spin-method]]).
2. **Partial-depth sharing precedent.** The authors apply sharing only to later stages — supports a possible layers-subset ablation.
3. **The attention formulation is unverifiable against reference code.** Our (and [[analyses/wplusw-lm-review-2026-08|Alper's]]) Wkv/Wq attention-sharing scheme is a reconstruction from the paper text — state this honestly in the thesis; confirm with Prof. Töreyin.

## Related
[[sources/halvit]], [[entities/halvit-model]], [[concepts/weight-sharing]], [[sources/halsp]], [[analyses/halvit-vs-albert-cross-layer-sharing]], [[analyses/wplusw-lm-review-2026-08]]

## Contradicts / Supports
- Supports: [[sources/2207.10237]] (SPIN) on keeping normalization parameters unshared.
- Supports: composability of intra-layer + cross-layer sharing assumed by our A3 arm ([[analyses/halvit-vs-albert-cross-layer-sharing]]).
