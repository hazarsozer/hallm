---
title: "Edge AI"
aliases: ["edge inference", "on-device AI", "TinyML", "resource-constrained deployment"]
papers: ["1712.05877", "2407.09562v3", "2501.15014v2", "2505.19995v1", "2506.03607v1", "2602.14301v1", "2603.26145v1"]
tags: [#edge-ai]
last_updated: 2026-04-29
---

## Definition
Edge AI refers to running neural network inference (and sometimes training) directly on resource-constrained devices at or near the data source — rather than in the cloud. Key constraints: limited RAM (often <1 GB), limited compute (no GPU or mobile NPU only), battery power budgets, and real-time latency requirements.

## Hardware Targets in the Literature

| Platform | RAM | Compute | Common in |
|----------|-----|---------|-----------|
| Raspberry Pi 4 | 4–8 GB | ARM Cortex-A72 | Research benchmarks |
| Jetson Nano | 4 GB | 128-core Maxwell GPU | Robotics, vision |
| Mobile SoCs (Snapdragon, A-series) | 4–16 GB | Mobile NPU | Phone-class KD papers |
| Microcontrollers (STM32, Arduino) | <1 MB | ARM Cortex-M | TinyML / ultra-constrained |
| Sony IMX500 CMOS | 8 MB | Embedded AI core | AIoT / smart sensors [arxiv:2407.09562] |

## Key Deployment Strategies

- **Export formats**: ONNX, TensorRT, TFLite, Core ML — each targets a hardware/runtime combination.
- **Quantization for edge**: INT8 and INT4 are the standard targets; 1-bit requires specialized kernels.
- **Knowledge Distillation to edge**: Train a small student specifically sized to fit the target hardware budget.
- **White-box vs. black-box deployment**: White-box (exposing model structure) allows hardware-specific optimization but raises IP concerns.
- **Federated learning**: Training across edge devices without centralizing data (relevant for DiReDi, DeepFusion).

## Key Papers

- [[sources/1712.05877]] — Jacob et al. 2018 (TFLite): INT8 QAT with hardware-verified 2× latency reduction on Qualcomm Snapdragon 835 for MobileNets; canonical ARM CPU INT8 benchmark
- [[sources/2501.15014v2]] — Survey: On Accelerating Edge AI (2025)
- [[sources/2603.26145v1]] — MobileViT KD for few-shot edge AI (Tsuyuki et al., Tohoku/IMT Atlantique, 2026): +14% 1-shot accuracy, −69% params, −37% energy on Jetson Orin Nano
- [[sources/2603.26145v1]] — Efficient Few-Shot Learning for Edge AI via KD on MobileViT
- 2505.19995v1 — Optimizing edge AI models on HPC systems with edge-in-the-loop

## Open Questions

- Which hardware target is most relevant for the ITU graduation project's benchmarking goal (Idea B)?
- What is the minimum viable model size for HaLViT to fit within the RAM budget of a Raspberry Pi 4?
- How do ONNX export → TensorRT optimization pipelines interact with weight-sharing schemes?

## Relevance to Project

Edge AI is the deployment target for the entire project. Thesis Idea B (Edge Deployment Benchmarking) directly operationalizes this: export HaLViT to ONNX/TensorRT, run on Raspberry Pi and Jetson Nano, measure latency (ms), memory footprint (RAM), and power consumption (Watts). The compression techniques (sharing, quantization) are justified precisely by these edge constraints.
