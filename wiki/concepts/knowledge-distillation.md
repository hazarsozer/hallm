---
title: "Knowledge Distillation"
aliases: ["KD", "teacher-student", "model distillation"]
papers: ["2210.16611v2", "2301.05849", "2409.08308v1", "2602.14301v1", "2603.26145v1"]
tags: [#kd, #edge-ai]
last_updated: 2026-04-29
---

## Definition
Knowledge Distillation (KD) transfers knowledge from a large, pretrained teacher model to a smaller student model. The student is trained on a combination of hard labels (ground truth) and soft targets (teacher's output probability distribution), which encodes inter-class relationships the student could not learn from labels alone.

## Major Variants

- **Output (logit) distillation**: Student matches the teacher's softmax probabilities. **Hinton et al. (2015)** [abs/1503.02531] is the landmark formulation: temperature-scaled softmax p_i = exp(z_i/T)/Σexp(z_j/T) with T>1 to soften teacher output; student trained on soft targets + hard labels jointly. Soft targets encode inter-class similarity not present in one-hot labels. Origins: Caruana et al. (KDD 2006) compressed ensembles of strong classifiers into single shallow networks using pseudo-labels.
- **Feature/intermediate distillation**: Student matches activations or attention maps from intermediate teacher layers — more information-rich. **FitNets** (Romero et al., 2014): hint layers — student mimics full feature maps of teacher's intermediate layers, allowing student to be thinner AND deeper. **Attention Transfer** (Zagoruyko & Komodakis, 2016): student matches attention map summaries of teacher activations — relaxes FitNets' strict feature map matching.
- **Relation distillation**: Student learns pairwise or group-wise relationships between samples in the teacher's representation space.
- **Reverse Distillation** (DiReDi): **DiReDi** [arxiv:2409.08308] (Sun, Tong, Yang, Zhang — Sony China/NTU, 2024) uses KD bidirectionally for privacy-preserving AIoT model adaptation. Phase 1: standard forward KD (cloud teacher → edge student); edge model deployed for inference only. Phase 2: reverse distillation — on user device, two tutor models (larger than student) capture current behavior (Tutor A mimics edge model) and desired behavior (Tutor B fine-tuned on user data); only ΔKnowledge = Tutor B − Tutor A is uploaded (no raw data); cloud updates teacher using Δ; updated teacher re-distills new edge model. Key insight: KD used not for compression but for knowledge transport / domain adaptation without privacy violation. Related to federated KD taxonomy in 2301.05849. See 2409.08308v1.
- **Layer-wise KD + multi-task joint fine-tuning**: **Kerpicci et al. (ICASSP 2023)** [arxiv:2210.16611] distills speech transformer models (wav2vec 2.0, HuBERT) to 28% size via layer-wise intermediate distillation, then jointly fine-tunes the entire student (SRL module + task heads) end-to-end for multiple voice tasks. Results: 75% size reduction, 0.1% KWS accuracy drop, 0.9% SV EER increase. **Key finding**: fine-tuning the distilled backbone (not just task heads) is critical — frozen SRL yields significantly worse performance. Domain: speech, but the pattern (layer-wise distillation + joint end-to-end fine-tuning) applies to ViT compression. For HaLViT: any post-distillation fine-tuning should propagate gradients through the shared weight W, not freeze it. See [[sources/2210.16611v2]].
- **Federated KD**: Distillation across distributed edge devices without sharing raw data. **Wu et al. (2024)** [arxiv:2301.05849] (CAS survey) maps the full space: four KD roles in FEL — knowledge transfer, model representation exchange, backbone component, dataset distillation — deployed at edge, end, or edge-end collaboratively. KD enables heterogeneous student sizes per device (different hardware budgets) where FedAvg forces a uniform model. Public unlabeled datasets serve as shared distillation medium for privacy-preserving compression. See 2301.05849. **DeepFusion** [arxiv:2602.14301] (Li, Hu, Abdelmoniem et al. — QMUL/Exeter, 2026) is the state-of-the-art in federated MoE training via heterogeneous KD. Three-phase pipeline: (1) K-means clustering of N on-device LLMs by cosine similarity of low-rank feature embeddings → K proxy models (weight-averaged within clusters); (2) Cross-architecture KD via **VAA (View-Aligned Attention)** module — student's J-stage features are patched, projected, self-attended to blend representation perspectives, then stage-divided for feature matching; total loss L_KD = L_CE + α·L_FM + β·L_KL; (3) K base models merged into global MoE (FFN layers → experts, shared layers averaged), gate fine-tuned on public data. One-shot FL: each device uploads once → −71% communication vs FedJETS. FedJETS requires 3.3–9.3× more RAM; DeepFusion enables TinyLLaMA/OLMo/BLOOM/GPT-2 class edge devices. DeepSeek-MoE financial open-ended QA at N=128: 3.9723 PPL (best), 29.22% accuracy (best). Approaches centralized DeepSpeed-MoE upper bound. VAA's cross-architecture perspective alignment is a general principle for any heterogeneous teacher-student pair. See [[sources/2602.14301v1]].
- **Environment-Aware KD**: Adapts the distillation signal based on edge hardware constraints (e.g., available memory, latency budget).
- **Weight-Inherited Distillation**: Combines KD with cross-layer weight sharing; the student inherits teacher weights and distills from them.

## Key Papers

- [[sources/2210.16611v2]] — Kerpicci et al. (ICASSP 2023): layer-wise KD + joint fine-tuning; 75% size reduction on speech transformers; joint fine-tuning critical; pattern generalizes to ViT
- 1710.09282 — Cheng et al. survey: KD section covers Hinton 2015, FitNets, AT; foundational pre-LLM taxonomy
- 2301.05849 — Wu et al. (CAS, 2024 survey): KD in FEL; four roles (transfer/exchange/backbone/dataset distillation); heterogeneous student sizes; privacy-preserving compression; edge deployment taxonomy
- 2409.08308v1 — DiReDi (Sun, Tong et al. — Sony China/NTU, 2024): bidirectional KD for AIoT; ΔKnowledge upload (no raw data); privacy-preserving cloud model update via reverse distillation
- [[sources/2602.14301v1]] — DeepFusion (Li, Hu, Abdelmoniem et al. — QMUL/Exeter, 2026): federated KD for MoE LLM training via heterogeneous on-device teachers; VAA module resolves cross-architecture view-mismatch; −71% communication, one-shot FL; approaches centralized upper bound
- [[sources/2603.26145v1]] — MobileViT KD for few-shot edge AI (Tsuyuki et al., 2026): +14% 1-shot on MiniImageNet vs ResNet12 at −69% params; Jetson Orin Nano deployment with direct power measurement

## Trade-offs

| KD Type | Compression | Accuracy | Requires Teacher at Inference |
|---------|------------|----------|-------------------------------|
| Output KD | High | Good | No — student deployed alone |
| Feature KD | High | Better | No |
| Federated KD | Moderate | Good | No (distributed) |
| Reverse KD | Moderate | High | Both teacher and student |

## Open Questions

- When the teacher itself needs compression, does distilling from a compressed teacher cascade errors?
- Can KD be applied after HaLViT training to further compress the student without breaking the weight sharing?
- How do federated KD methods handle non-IID data distributions on heterogeneous edge devices?

## Relevance to Project

KD is the most deployment-oriented compression technique in the literature. For the thesis, it is most relevant as a post-compression step: train HaLViT (with weight sharing), then use KD to further slim it for a specific edge target. Alternatively, use HaLViT as the student in a teacher→HaLViT distillation pipeline.
