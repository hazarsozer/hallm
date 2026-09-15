---
query: "Is a looped transformer whose second pass uses the transposed weights (the 'transposed loop') novel?"
date: 2026-09-15
sources_consulted: ["wiki/sources/halvit", "web search + opened papers (listed below)"]
---

# Transposed loop — literature check (2026-09-15)

**Idea.** A looped LM where each stored block runs twice and the second pass uses the transposed
matrices: FFN up = W_downᵀ (4d×d), down = W_upᵀ (d×4d); attention W_Qᵀ, W_Kᵀ, W_Vᵀ, W_Oᵀ. Same
storage and FLOPs as plain looping; the two passes compute different functions. Motivation: plain
looped passes are identical; Relaxed Recursive Transformers untie them with LoRA (extra params).
Prompted by the pilot result that looping beats W+Wᵀ at matched storage and compute
(`results/reports/iso-storage.md`, 3 seeds).

**Verdict: novel as specified, from known ingredients — moderate confidence (~70%).** No
paper found that loops a transformer with transposed second-pass weights on the same residual
stream, or uses transposition as parameter-free iteration differentiation. Transposed reuse in
general is old, so the contribution is "transposition as iteration differentiation in looped
LMs", not "transposed reuse". Coverage caveat: arXiv API, Semantic Scholar and OpenAlex were
rate-limited and OpenReview blocked; most coverage came from web search, and the idea is simple
enough to have been tried quietly.

Every paper below was opened and checked by the search agent.

## HaLViT and the lab

- Koyun & Töreyin, "HaLViT: Half of the Weights are Enough", CVPR 2024 Workshops (ELVM)
  ([CVF PDF](https://openaccess.thecvf.com/content/CVPR2024W/ELVM/papers/Koyun_HaLViT_Half_of_the_Weights_are_Enough_CVPRW_2024_paper.pdf)).
  Already combines W+Wᵀ with cross-layer sharing (HaLViT* shares all but the query weights
  across 12 layers; the ResNet variant shares one W/Wᵀ across a stage), but the orientation
  never flips with depth. No follow-up does; the lab's HALSP repo (2026) gives one matrix three
  roles in a CNN.

## Closest work

1. Haber & Ruthotto, "Stable Architectures for Deep Neural Networks", Inverse Problems 2017
   ([1705.03341](https://arxiv.org/abs/1705.03341)). The Verlet network applies K_jᵀ then K_j in
   consecutive half-steps (antisymmetric Jacobian). It acts on two separate state halves, the
   Kᵀ step has a minus sign, and the model is a CNN.
2. Chang et al., "Reversible Architectures for Arbitrarily Deep Residual Neural Networks",
   AAAI 2018 ([1709.03698](https://arxiv.org/abs/1709.03698)). The Hamiltonian block
   Y + h·Kᵀσ(KZ) is HaLViT's bottleneck form, six years earlier; HaLViT does not cite it. Same
   orientation at every depth. **Cite in related work.**
3. Wen et al., "Deep Predictive Coding Network for Object Recognition", ICML 2018
   ([PMLR](https://proceedings.mlr.press/v80/wen18a.html)). Feedback = transposed feedforward
   weights with recurrent up/down cycles, but between adjacent layers, not stream to stream.
4. Amit, Levy & Mirsky, "Transpose Attack", NDSS 2024 ([2311.07389](https://arxiv.org/abs/2311.07389)).
   Transposed weights in reversed layer order compute a second task; transformer blocks are not
   transposed internally.
5. Bae et al., "Relaxed Recursive Transformers", ICLR 2025 ([2410.20672](https://arxiv.org/abs/2410.20672)).
   Unties loop iterations with depth-wise LoRA (extra parameters).
6. Xu & Sato, ICML 2025 ([2410.01405](https://arxiv.org/abs/2410.01405)). A proven limitation
   of looped transformers, fixed with timestep-conditioned per-loop scaling.

Non-transposing neighbours: Takase & Kiyono CYCLE(REV) layer-order reversal
([2104.06022](https://arxiv.org/abs/2104.06022)); Subformer sandwich sharing
([2101.00234](https://arxiv.org/abs/2101.00234)); MobileLLM immediate block-wise sharing
([2402.14905](https://arxiv.org/abs/2402.14905)); CRATE ties Q=K=V=U_kᵀ and uses D/Dᵀ inside a
layer ([2306.01129](https://arxiv.org/abs/2306.01129)); Universal Transformers' timestep
embeddings ([1807.03819](https://arxiv.org/abs/1807.03819)); SpiralFormer's per-loop resolution
([2602.11698](https://arxiv.org/abs/2602.11698)). Classic transpose-at-later-depth: tied
autoencoders (Vincent et al., JMLR 2010), tied embeddings (Press & Wolf, EACL 2017,
[1608.05859](https://arxiv.org/abs/1608.05859)).

**What stays new:** transposed second passes in a decoder-only looped LM, at matched storage
and FLOPs, against plain looping (and, if time, LoRA-relaxed looping and timestep encoding).

## Why it might hurt or help (derivation, not from a paper)

Ignoring the change in which hidden units fire, pass 2's FFN Jacobian is the transpose of pass 1's:
two residual steps compose to roughly (I + Mᵀ)(I + M) = I + (M + Mᵀ) + MᵀM, so the
antisymmetric (rotation-like) part of M cancels to first order, while plain looping keeps
(I + M)² = I + 2M + M². That is a concrete reason it could **lose** to plain looping. Attention
partly escapes: pass 2 scores with a different bilinear form and transposition regroups head
dimensions. Two cheap ablations isolate this:

- **negated second pass**, which keeps the antisymmetric part: (I − Mᵀ)(I + M) = I + (M − Mᵀ) − MᵀM
- **a learned scalar per pass** (1 parameter): I + M + αMᵀ

In its favour: Blayney et al. 2026 ([2604.11791](https://arxiv.org/abs/2604.11791)) find looped LMs
converge to the same attention behaviour on every recurrence — the redundancy this breaks.
Implementation note: the GPT-style scaled init of the down-projection lands on the wrong role
after transposing.

## Not checked

"Depth as Modulation in Weight-Sharing Transformers" (OpenReview, blocked) and Kosko's 1988
bidirectional associative memory paper (IEEE, blocked) — worth a manual look.
