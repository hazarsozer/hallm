---
title: "HALSP-Net: A Shared Projection Architecture with Dynamic Channel Selection"
arxiv: "n/a (code+paper release, sp4cing-itu/HALSP)"
authors: ["SPACING Lab, ITU"]
year: 2026
category: weight-sharing
status: ingested
source_file: https://github.com/sp4cing-itu/HALSP (inspected 2026-08-17)
---

## Problem
Follow-up to [[sources/halvit]] from the same lab (repo last updated 2026-07). Targets efficient inference *and on-device training* under tight parameter/compute/memory budgets — pushes "one matrix, many roles" beyond two roles.

## Method
- A **single learnable matrix W per stage carries three roles**: entry projection, inner channel mixing (via cyclic column shifts per block), and exit projection. Paired with a per-block depthwise spatial filter (channel and spatial paths explicitly decoupled).
- **Dynamic channel selection**: during training only a subset of latent channels (active fraction r_a) is live; topology refreshed every T steps via a Focus/Reserve pool split scored by mean |w| or momentum, plus an EMA "opportunity map" Q = v/(K+ε) steering exploration toward uncovered input variance. Inference is fully dense — sparsity is train-side only.

## Results
CIFAR-100, single A100: ~78.7–79.4% top-1 at **1.71–1.75M params** (comparable to much larger ResNet/WRN baselines); highest single-image throughput in their panel (~308 img/s at bs=1); r_a<1 lets training fit batch sizes at which baselines OOM, with negligible accuracy loss.

## Significance
- Confirms the lab treats shared-projection role-multiplexing as its live research thread — our "does W+Wᵀ generalize to LMs" question extends the same agenda (README credits [[sources/halvit]] as the origin).
- The three-role idea (entry/mix/exit from one matrix) is a potential Term-2 stretch analogue for LMs (e.g., one matrix serving both attention and FFN) — out of current proposal scope, note only.
- Train-side dynamic sparsity with dense inference is an interesting memory lever for our single-12GB-GPU constraint, but adopting it would break the matched-budget C3 control — do not adopt for the core four arms.

## Related
[[sources/halvit]], [[sources/halvit-official-code]], [[entities/halvit-model]], [[concepts/weight-sharing]], [[concepts/pruning]]

## Contradicts / Supports
- Supports: [[sources/halvit]] W/Wᵀ mechanism generality (extended to three roles).
