---
query: "Given the full compression literature and the signed charter, what are the two strongest thesis directions to bring to the advisor, and why were the others set aside?"
date: 2026-05-28
sources_consulted:
  - "wiki/overview.md"
  - "wiki/sources/halvit.md"
  - "wiki/analyses/halvit-vs-albert-cross-layer-sharing.md"
  - "wiki/analyses/pruning-composability-with-weight-sharing.md"
  - "wiki/analyses/quantization-tradeoffs-1to4bit.md"
  - "wiki/analyses/edge-hardware-targets.md"
  - "raw/papers/project-charter.pdf"
---

## Decision

After a full pass over the wiki and the signed charter, the thesis was narrowed to **two options**, both anchored on [[sources/halvit]], both charter-compliant, both runnable primarily on a single RTX 4070. Full proposals live in `proposals/option-1-halvit-edge.md` and `proposals/option-2-halvit-language.md`; the wider brainstorm is archived in `proposals/brainstorm-log.md`.

> ### ✅ RESOLVED — 2026-06-12
> The advisor (Prof. Töreyin) **selected Option 2** (W+Wᵀ → language models) and steered it to **pure research, no product**. Two consequences:
> - **Option 1 (HaLViT-Edge) is dropped** — it was the deployment/product option; the advisor wants research, not an artifact. Its structure-aware-compression analyses stay in the wiki as background only.
> - **Model scale may go smaller than GPT-2** — the report carries size as a range (≈10M–124M, locked Term 2); smaller-from-scratch is favored because it lets all four arms train under a matched budget. **This dissolves Option 2's only real blocker (compute/cluster):** the core thesis now runs entirely on the 4070. TinyLlama survives as an optional stretch gated on SP4CING/UHEM access.
>
> (Conveyed secondhand via a phone call relayed by a teammate — wording approximate; re-confirm size floor + cluster access at the next direct meeting.)

## Context that frames the decision

- **Charter signed 2026-02-02** — the *domain* (model compression, HaLViT-anchored) is already approved. What remains is choosing the *specific contribution*, which is charter task §3.2 ("analyze alternative solutions + assess risks") plus the Requirement Analysis. **This page is that artifact.**
- **The Part-1 deliverable (due 2026-06-15) is a planning/requirements report, not experiments** — implementation is Term 2. The literature survey (56 sources) is effectively done, so most of the report can be assembled from the wiki.
- **Constraints (charter §5)**: mathematically verifiable reduction in params / MB / GFLOPs; PyTorch-compatible/reproducible; <2% Top-1 (explicitly "*or perplexity, if evaluating language models*") loss vs the uncompressed baseline.
- **Standards (charter §6)**: ITU-T F748.11 (DNN processor benchmark metrics), ISO/IEC 22989:2022 (terminology) — apply to either option.
- **Compute**: RTX 4070 12GB local; SP4CING/UHEM cluster only if the advisor grants it.

## Option 1 — HaLViT-Edge *(recommended core)*

Structure-aware extreme compression of HaLViT-T + real edge deployment. **Novel kernel**: every off-the-shelf compressor (Wanda/AWQ/SparseGPT) silently violates HaLViT's W=Wᵀ pairing — pruning weight (i,j) also kills (j,i) on the transposed path. We formulate the sharing-aware fix (symmetric mask + joint saliency for pruning; joint calibration for quantization), compose pruning + low-bit quant (+ optional KD), and deploy to a Pi/Jetson with measured latency + energy.

- **Impressive on all axes**: novel (HaLViT's own named future work), tangible (live edge demo), composition (the field's frontier per [[overview]]), rigorous (accuracy–compression–energy frontier).
- **Feasibility**: low risk, ~1 week compute on the 4070 (HaLViT-T = 11.1M params); robust even if the structure-aware gain is small — the analysis, the composed pipeline, the first edge deployment, and the energy benchmark are all contributions regardless.
- **Backing**: [[analyses/pruning-composability-with-weight-sharing]], [[analyses/quantization-tradeoffs-1to4bit]], [[analyses/edge-hardware-targets]].

## Option 2 — Does W+Wᵀ generalize to language? *(higher ceiling)*

Implement HaLViT's W+Wᵀ intra-layer sharing in a GPT-2-scale language model; run a four-way comparison (none / ALBERT-style / HaLViT-style / both) on WikiText-103; optionally extend to TinyLlama.

- **Impressive by novelty/prestige**: answers an open question the advisor is personally curious about; potentially workshop-publishable; a clean negative result is still a contribution.
- **Feasibility / risk**: GPT-2 feasible-but-slow on the 4070; **TinyLlama needs the cluster**; medium risk (causal attention / SwiGLU / GQA may break the mechanism).
- **Backing**: [[analyses/halvit-vs-albert-cross-layer-sharing]] (establishes the two sharing axes are orthogonal and may stack — this experiment is its empirical validation).
- Can also ride along as the optional "Chapter 5: does it generalize?" stretch of Option 1.

## Why the wider brainstorm was set aside

A 16-candidate brainstorm (C–R, archived in `proposals/brainstorm-log.md`) was considered. For a late-starting two-person team on a consumer GPU, the non-finalists were dropped because:

- **NAS over HaLViT (E)** — heaviest compute of any option; no dedicated wiki analysis; least write-ready.
- **Local-LLM-on-edge stack (G)** — low novelty, engineering not research, not HaLViT-anchored (survives only as the edge-deployment chapter of Option 1).
- **On-device personalization (I)** — not HaLViT-anchored, systems-heavy.
- **KV-cache (M), speculative decoding (J), adaptive inference (L)** — crowded or reportedly solved.
- **Others (C, D, F, H, N, O, P, R)** — compute risk, weak novelty, or off-anchor.

The two finalists win because they are charter-anchored, advisor-aligned, write-ready from the wiki, and feasible on the team's hardware.

## Advisor decisions — resolved 2026-06-12

1. Option 1, Option 2, or stretch? → **✅ Option 2**, framed as pure research (no product/deployment).
2. Pretrained HaLViT-T checkpoint from SP4CING? → moot — Option 1 dropped, so HaLViT-T is no longer the working model.
3. SP4CING/UHEM GPU access for language-model work? → **still open**, but now only gates the *optional TinyLlama stretch*; the small-LM core needs no cluster. Re-confirm at the next direct meeting.

## Related

[[overview]] — Thesis Direction Assessment table (this decision operationalizes it) · [[sources/halvit]] — anchor paper · [[analyses/halvit-vs-albert-cross-layer-sharing]] · [[analyses/pruning-composability-with-weight-sharing]] · [[analyses/quantization-tradeoffs-1to4bit]] · [[analyses/edge-hardware-targets]]
