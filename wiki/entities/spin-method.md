---
type: paper
title: "SPIN — Sharing Parameters of Isotropic Networks"
arxiv: "2207.10237"
---

Systematic empirical evaluation of cross-layer weight sharing topologies in isotropic architectures (ViT/DeiT, ConvMixer, ConvNeXt). University of Washington / Apple, Jul 2022.

**Two-dimensional topology framework**:
1. **Sharing mapping**: Sequential, Strided, Pyramid, Random — how layers are grouped
2. **Sharing distribution**: Uniform, Front, Middle, Back — which groups get unique weights

**Share rate** = L / P (total layers / unique tensors). Share rate 2 = 2× parameter reduction.

**Key finding — weight fusion**: Merging pretrained weights (Channel Weighted Mean or Scalar Weighted Mean) into shared initialization recovers 0.5–1.3 pp of accuracy vs from-scratch initialization. Must use fusion when adapting a pretrained model.

**CKA insight**: Middle layers have highest cross-layer representational similarity → most amenable to sharing. Supports keeping first/last layers independent (consistent with Subformer Sandwich).

**BatchNorm constraint**: Sharing normalization layers causes training divergence. Keep all normalization layers independent.

**DeiT-S numbers (most relevant to HaLViT)**:
| Setup | Params | Top-1 |
|-------|--------|-------|
| Baseline | 22.05M | 80.52% |
| Share rate 2 | 11.41M | 78.61% |
| Share rate 2 + Mean fusion | 11.41M | **79.44%** |
| Share rate 3 + Mean fusion | 7.87M | 77.11% |

**HaLViT composition**: SPIN cross-layer sharing ⊥ HaLViT intra-layer W+Wᵀ sharing — orthogonal axes. Applying both simultaneously could achieve compound compression. Joint fusion strategy for W+Wᵀ pairs is unstudied.

See [[sources/2207.10237]] for full topology ablations, ConvMixer/ConvNeXt results, and all fusion strategies.

Related: [[concepts/weight-sharing]], [[sources/2101.00234]], subformer-model, [[entities/halvit-model]]
