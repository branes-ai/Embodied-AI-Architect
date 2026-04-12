---
title: "NVIDIA Jetson Orin Platform Analysis"
domain: accelerator
tags: [nvidia, jetson, orin, edge, gpu, inference]
relevance:
  - mission_decomposition
  - hardware_selection
  - design_tradeoffs
peak_tops: 275
compute_density_tops_w: 4.6
process_nm: 8
target_workloads: [cnn, transformer, detection, segmentation, slam]
last_updated: 2026-03-16
---

## Architecture Overview

The Jetson Orin family uses NVIDIA's Ampere GPU architecture combined with Arm
Cortex-A78AE CPU cores. Three SKUs span a wide power/performance range:

| SKU | GPU Cores | CPU Cores | TOPS (INT8) | TDP (W) | TOPS/W |
|-----|-----------|-----------|-------------|---------|--------|
| AGX Orin 64GB | 2048 | 12× A78AE | 275 | 60 | 4.6 |
| Orin NX 16GB | 1024 | 8× A78AE | 100 | 25 | 4.0 |
| Orin Nano 8GB | 512 | 6× A78AE | 40 | 15 | 2.7 |

Memory: LPDDR5 at 204.8 GB/s (AGX), 102.4 GB/s (NX), 68 GB/s (Nano).

## Key Design Decisions & Tradeoffs

- **Unified memory** between CPU and GPU eliminates copy overhead but creates
  contention under mixed workloads (perception + control).
- **DLA (Deep Learning Accelerator)** provides a second inference path at higher
  efficiency (~6 TOPS/W for supported layers) but limited layer coverage.
- **Power modes** allow dynamic TDP scaling (e.g., AGX: 15W/30W/50W/60W),
  trading throughput for battery life.
- **TensorRT** compiler is essential — raw PyTorch inference wastes 60-80% of
  available throughput.

## Performance Characteristics

**Detection (YOLOv8s, 640×640, INT8 TensorRT):**
- AGX 60W: ~3.2ms (312 FPS)
- AGX 30W: ~5.8ms (172 FPS)
- NX 25W: ~8.1ms (123 FPS)
- Nano 15W: ~18ms (55 FPS)

**Segmentation (DeepLabV3-MobileNetV3, 520×520, FP16):**
- AGX 60W: ~6.5ms (153 FPS)
- NX 25W: ~14ms (71 FPS)

**Multi-model pipeline (detection + tracking + depth):**
- AGX 60W: ~12ms total at 640×640 — leaves headroom for control loop
- Nano 15W: ~35ms total — marginal for 30 FPS real-time

## Lessons for Design Space Exploration

1. **DLA vs GPU tradeoff**: DLA is 30-50% more efficient but only accelerates
   standard conv/pool/activation layers. Transformer attention stays on GPU.
   Design space should include a `use_dla` boolean.
2. **Power mode is a first-class variable**: 4× power range with ~2.5× perf
   range. The sweet spot for drones is often 15-30W (Nano/NX).
3. **Memory bandwidth is the bottleneck** for large models: AGX at 204.8 GB/s
   saturates at ~4 concurrent models. Pipeline design must account for this.
4. **Batch size 1 is standard** for real-time — batching helps throughput but
   hurts latency. Design space should fix batch=1 for real-time use cases.
