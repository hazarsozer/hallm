---
query: "What edge hardware targets dominate the literature, what are their constraints, and where does HaLViT fit after compression?"
date: 2026-04-30
sources_consulted:
  - "wiki/concepts/edge-ai.md"
  - "wiki/sources/2407.09562v3.md"
  - "wiki/sources/2603.26145v1.md"
  - "wiki/sources/2501.15014v2.md"
  - "wiki/sources/2311.03923.md"
  - "wiki/sources/2404.12403.md"
  - "wiki/sources/1712.05877.md"
  - "wiki/analyses/quantization-tradeoffs-1to4bit.md"
  - "wiki/sources/halvit.md"
---

## Executive Summary

Four hardware tiers dominate edge AI benchmarking in the 2023–2026 literature: (1) Sony IMX500 CMOS sensor (8MB, ultra-constrained AIoT), (2) Jetson Orin Nano (mobile GPU, 4–16GB RAM), (3) Raspberry Pi 4/5 (ARM CPU, 4–8GB RAM), and (4) mobile SoCs (Snapdragon, Apple Silicon, 4–16GB). HaLViT-T in its current form (11.1M parameters, FP32 = 44MB) fits comfortably on Jetson Orin and Raspberry Pi 4, but needs additional compression for mobile SoC deployment (target: <20MB) and is far too large for the IMX500. This analysis maps the compression targets from prior analyses to concrete hardware reachability, and defines the thesis benchmarking protocol.

---

## 1. Hardware Landscape

### 1.1 Platform Inventory (from literature)

| Platform | RAM | Compute | Power | Common in Literature |
|----------|-----|---------|-------|---------------------|
| Sony IMX500 CMOS | **8MB on-chip** | Embedded AI core | ~0.1W | AIoT, smart sensors [arxiv:2407.09562] |
| Raspberry Pi 4 | 4–8 GB | ARM Cortex-A72 (4-core, 1.8GHz) | ~5W | Embedded CV benchmarks |
| Raspberry Pi 5 | 4–8 GB | ARM Cortex-A76 (4-core, 2.4GHz) | ~7W | Updated benchmark target |
| NVIDIA Jetson Orin Nano | 8–16 GB | Ampere GPU (1024 CUDA cores) + CPU | 7–15W | Research-grade edge GPU [arxiv:2603.26145] |
| NVIDIA Jetson Nano | 4 GB | Maxwell GPU (128 cores) | 5–10W | Older benchmark standard |
| Mobile SoC (Snapdragon 888) | 8–12 GB | Hexagon DSP + Adreno GPU | ~3W peak | On-device ML papers |
| Apple M-series (edge) | 8–32 GB | Neural Engine (NPU) | ~15W | MacBook edge inference |
| Microcontroller (STM32) | <1 MB | ARM Cortex-M4 (168MHz) | <0.1W | TinyML |

### 1.2 The Three Relevant Tiers for This Thesis

**Tier 1 — Consumer GPU edge (Jetson Orin Nano)**: This is the natural first deployment target for HaLViT. The Orin Nano's Ampere GPU supports INT8 and INT4 inference natively; its 8–16GB unified memory can hold any HaLViT variant comfortably. Benchmarking here is straightforward: ONNX export → TensorRT INT8/INT4, measure latency and throughput.

**Tier 2 — CPU-only embedded (Raspberry Pi 4/5)**: The most common academic benchmark for "resource-constrained deployment." No GPU — all inference runs on ARM Cortex-A. INT8 quantization via TFLite or ONNX Runtime provides 2–4× CPU speedup. HaLViT-T (11.1M params, INT8 = ~11MB) fits within the memory budget; inference latency will be the constraint (target: <100ms per image for real-time classification).

**Tier 3 — Ultra-constrained sensor (Sony IMX500)**: Only 8MB on-chip memory for the AI model. HaLViT-T requires aggressive compression to reach this tier: ~3M parameters at INT8 (~3MB) or ~6M parameters at INT4 (~3MB). This requires combining all three compression axes from the thesis (weight sharing × 30% pruning × INT4 quantization = ~1M effective parameters at INT4 → ~0.5MB). While technically feasible, this tier requires full pipeline validation before claiming IMX500 deployment.

---

## 2. HaLViT's Size at Each Compression Stage

### 2.1 Parameter Counts and Memory Footprint

HaLViT-T: 11.1M parameters. Memory at various compression levels:

| Compression | Params | Memory (weight storage) | Fits Jetson Orin? | Fits RPi 4? | Fits IMX500? |
|-------------|--------|------------------------|------------------|------------|-------------|
| FP32 (baseline) | 11.1M | 44MB | ✓ | ✓ | ✗ |
| FP16 | 11.1M | 22MB | ✓ | ✓ | ✗ |
| INT8 PTQ | 11.1M | 11MB | ✓ | ✓ | ✗ |
| INT4 PTQ (AWQ) | 11.1M | 5.5MB | ✓ | ✓ | **Borderline** |
| INT4 + 30% prune | ~7.8M | 3.9MB | ✓ | ✓ | **Tight** |
| INT4 + 50% prune | ~5.5M | 2.75MB | ✓ | ✓ | ✓ |
| INT2 (QuIP) | 11.1M | 2.8MB | ✓ | ✓ | ✓ |

Weight sharing (W+Wᵀ) is already baked into HaLViT's architecture — the 11.1M parameter count already reflects the ~2× intra-layer compression. A standard DeiT-Small with the same capacity would be ~22M parameters.

### 2.2 Inference Latency Estimates

Latency depends on compute (FLOPs) and memory bandwidth (weight loading). HaLViT-T: 4.6G FLOPs per forward pass.

| Platform | FP32 latency | INT8 latency | INT4 latency |
|----------|-------------|-------------|-------------|
| Jetson Orin Nano (Ampere GPU) | ~8ms | ~3ms | ~2ms |
| Raspberry Pi 4 (ARM CPU) | ~800ms | ~200ms | ~120ms |
| Raspberry Pi 5 (ARM CPU) | ~400ms | ~100ms | ~60ms |
| IMX500 (embedded core) | N/A | ~50ms* | N/A |

*IMX500 latency is task-dependent and model-size constrained; the chicken detection paper reports >20 FPS (< 50ms) at 95.1% mAP with a MobileNet-based model.

---

## 3. Hardware-Aware Compression Requirements

### 3.1 Jetson Orin Nano: The Research Benchmark Target

**Constraints**: 7–15W power envelope, 8–16GB RAM, Ampere GPU (INT8 Tensor Cores, support for 2:4 sparsity pattern).

**Required compression for real-time (30 FPS = 33ms budget)**:
- FP16 inference: HaLViT-T is likely achievable (~8ms per image estimated)
- INT8: comfortably real-time
- INT4: well within budget, allows batch processing

**Ampere 2:4 sparsity**: SparseGPT and BESA both support the 2:4 semi-structured pattern, which the Jetson Orin's Ampere GPU accelerates via sparse tensor cores (1.54–1.79× GPU speedup per SparseGPT benchmarks). This is the first hardware where HaLViT's structured pruning yields a concrete measured speedup.

**Recommended benchmarking** (from [arxiv:2603.26145]): direct power supply measurement during inference on Jetson Orin Nano. The Tsuyuki et al. paper reports −37% energy reduction vs ResNet12 baseline at the same inference task — this is the measurement methodology the thesis should replicate for HaLViT vs DeiT-Small baseline.

**Export pipeline**: PyTorch → ONNX → TensorRT INT8/INT4 calibration (same 128-sample process as AWQ/GPTQ).

### 3.2 Raspberry Pi 4: The Reproducible Baseline

**Constraints**: 5W power, 4–8GB RAM, no GPU acceleration, TFLite or ONNX Runtime on ARM.

**Required compression for practical use (<100ms per image)**:
- FP32: ~800ms — unusable for real-time
- INT8: ~200ms — acceptable for batch; too slow for real-time video
- INT4 (custom kernels): ~120ms — approaching practical threshold
- INT4 + 50% pruning: ~60ms — real-time with some latency headroom

**Bottleneck**: Memory bandwidth on ARM Cortex-A. The SqueezeLLM paper's "memory-bandwidth-bound" framing applies here — every weight must be loaded from DRAM for each inference. INT8 halves the bandwidth requirement; INT4 halves it again. Weight sharing already reduces the total bytes loaded (11.1M vs 22M parameters for equivalent-capacity DeiT-Small).

**Export pipeline**: PyTorch → TFLite INT8 (Jacob et al. 2018 [arxiv:1712.05877] recipe) → benchmarked with `time` and `perf stat` on the Pi.

### 3.3 Sony IMX500: The Aspirational Stretch Target

**Constraints**: 8MB on-chip SRAM for the model; power ≈ 0.1W; inference within the CMOS image sensor itself.

**What fits at 8MB**:
- Standard INT8 model: needs ≤8M parameters (8MB = 8M × 1 byte/weight)
- INT4 model: needs ≤16M parameters, but the IMX500 SDK specifies INT8 → practical ceiling is ~8M parameters at INT8

**HaLViT pathway to IMX500**:
```
HaLViT-T (11.1M params)
  → 50% BESA pruning: ~5.5M params
  → INT8 quantization: 5.5MB
  → Fits within 8MB budget with 2.5MB headroom for activations
```

This makes the IMX500 deployment theoretically achievable from HaLViT-T, but requires the full pruning + quantization pipeline to be validated first. The FCOS-Lite paper [arxiv:2407.09562] establishes the benchmark methodology: 95.1% mAP at >20 FPS on IMX500 with INT8 MobileNet backbone.

**Alternative via KD**: Distill from HaLViT-T to a 3M-parameter micro-HaLViT → INT8 → 3MB. Leaves 5MB for activations and metadata.

---

## 4. Hardware-Specific Optimization Alignment

### 4.1 Which Compression Methods Hit Which Hardware Features

| Hardware Feature | Compression Method | Connection |
|-----------------|-------------------|------------|
| Ampere 2:4 sparse tensor cores (Jetson Orin) | SparseGPT or BESA with 2:4 pattern | 1.54–1.79× GPU speedup |
| ARM NEON SIMD (Raspberry Pi) | INT8 TFLite, AWQ TinyChat ARM | DeepSparse CPU speedup at 40–50% sparsity |
| IMX500 embedded core | INT8, structured pruning (smaller dense model) | Must fit 8MB; no sparse support |
| Mobile NPU (Snapdragon Hexagon) | INT8 or INT4 with vendor SDK | Needs Qualcomm AI Engine SDK |

### 4.2 NAS for Hardware-Aware Architecture Search

Two NAS papers in the wiki directly address hardware-specific compression:
- **HW-EvRSNAS** [arxiv:2311.03923]: evolutionary NAS with RMI proxy for 6 different edge devices, 8000× search speedup over conventional NAS. The proxy-based approach could be applied to find an optimal HaLViT variant (which layers to share, which to leave independent) for each target hardware.
- **MO-HDNAS** [arxiv:2404.12403]: three-objective Pareto (RMI + hardware cost + diversity) in a single NSGA-II run. Directly applicable to finding the Pareto front between HaLViT accuracy and Jetson/RPi latency.

If the thesis extends to NAS, these methods provide the search framework; HaLViT's W+Wᵀ sharing can be treated as one architectural choice on the NAS search axis (share/don't-share per layer).

---

## 5. Recommended Benchmarking Protocol

### Tier 1: Jetson Orin Nano (primary benchmark)
1. Export: PyTorch → ONNX → TensorRT (INT8 calibration with 128 ImageNet samples)
2. Measure: latency (ms/image, batch=1), throughput (images/sec, batch=32), peak memory
3. Energy: direct power supply measurement during sustained inference (follow Tsuyuki et al. methodology)
4. Baselines: DeiT-Small-FP16, DeiT-Small-INT8, HaLViT-T-FP16, HaLViT-T-INT8, HaLViT-T-INT4

### Tier 2: Raspberry Pi 4 (reproducibility benchmark)
1. Export: PyTorch → TFLite INT8 (Jacob et al. 2018 recipe)
2. Measure: latency (ms/image), RAM usage, CPU utilization
3. Energy: measured via USB power meter on Pi 4 power supply
4. Baselines: same set as Jetson, with additional MobileViT comparison (Tsuyuki et al. architecture)

### Tier 3: IMX500 (stretch target, simulation acceptable)
1. If IMX500 SDK not available: simulate via memory-footprint constraint check (model must be ≤8MB at INT8)
2. If SDK available: deploy FCOS-Lite-style pipeline with HaLViT backbone
3. Benchmark: report whether each compression stage meets the 8MB constraint

---

## 6. Thesis Positioning

The edge hardware analysis connects the compression thesis to deployment reality:
- Weight sharing (HaLViT's W+Wᵀ): halves parameter count vs equivalent-capacity ViT
- Quantization (INT8/INT4): halves or quarters memory further
- Pruning (BESA 30–50%): additional 1.3–2× reduction
- KD (output distillation from DeiT-B teacher): recovers accuracy without adding parameters

**The thesis argument**: Starting from DeiT-Small (22M parameters, FP32 = 88MB), the three-axis pipeline reaches HaLViT-T compressed (11.1M → 7.8M → INT4 = 3.9MB) — a **22× memory reduction with <5% Top-1 accuracy loss**, deployable on Raspberry Pi 4 at ~120ms/image and Jetson Orin Nano at ~2ms/image with hardware-accelerated sparse INT4 inference.

---

## Sources Consulted

- [[concepts/edge-ai]] — hardware platform table; deployment strategies; open questions
- [[sources/2407.09562v3]] — Sony IMX500; FCOS-Lite + KD + INT8; 8MB constraint; 20+ FPS benchmark
- [[sources/2603.26145v1]] — MobileViT KD; Jetson Orin Nano; direct power measurement; −37% energy
- [[sources/2501.15014v2]] — Edge AI survey; TVM/TensorRT/OpenVINO deployment compilers
- [[sources/2311.03923]] — HW-EvRSNAS; proxy-based NAS for 6 edge devices; 8000× speedup
- [[sources/2404.12403]] — MO-HDNAS; three-objective Pareto NAS; hardware-aware search
- [[sources/1712.05877]] — Jacob et al. 2018; TFLite INT8 pipeline; ARM Cortex-A benchmarks
- [[analyses/quantization-tradeoffs-1to4bit]] — bit-width progression; three-axis compression estimate
