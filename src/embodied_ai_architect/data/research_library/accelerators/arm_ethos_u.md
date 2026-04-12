---
title: "Arm Ethos-U NPU for Microcontrollers"
domain: accelerator
tags: [arm, ethos, npu, microcontroller, ultra-low-power]
relevance:
  - hardware_selection
  - design_tradeoffs
peak_tops: 0.5
compute_density_tops_w: 5.0
process_nm: 7
target_workloads: [cnn, keyword_spotting, anomaly_detection, classification]
last_updated: 2026-03-16
---

## Architecture Overview

Ethos-U55 and U85 are microNPUs designed to sit alongside Cortex-M and Cortex-A
processors. They target always-on, ultra-low-power inference.

| Metric | U55 (128 MAC) | U55 (256 MAC) | U85 |
|--------|---------------|---------------|-----|
| Peak TOPS (INT8) | 0.128 | 0.256 | 0.5 |
| Area | 0.1 mm² | 0.14 mm² | 0.3 mm² |
| Power | 25 mW | 50 mW | 100 mW |
| TOPS/W | 5.0 | 5.0 | 5.0 |
| SRAM | Shared with MCU | Shared with MCU | 512 KB dedicated |

## Key Design Decisions & Tradeoffs

- **Tiny footprint** (0.1 mm²) enables integration into any SoC with negligible
  area/power cost. Ideal for always-on wake-word or anomaly detection.
- **TFLite Micro only**: Supports TFLite quantized models via Vela compiler.
  No ONNX or PyTorch direct path.
- **No off-chip memory access for weights**: Model must fit entirely in on-chip
  SRAM or flash-backed TCM. Maximum model size ~300KB-2MB.
- **Deterministic latency**: Fixed pipeline, no caches, no dynamic scheduling.

## Performance Characteristics

**Keyword Spotting (DS-CNN, 256 KB model):**
- U55-128: 5ms, 200 inferences/sec at 25 mW
- U85: 1.5ms, 660 inferences/sec at 100 mW

**Person Detection (MobileNetV2-0.35, INT8):**
- U55-256: 30ms at 96×96 input
- U85: 10ms at 96×96 input

## Lessons for Design Space Exploration

1. **Different design regime**: Sub-watt inference with model sizes under 2 MB.
   Design space variables like array_rows=64 are irrelevant here.
2. **Model size is the primary constraint**, not FLOPS. Optimization should
   target parameter count and activation memory, not just TOPS.
3. **Useful as a co-processor** in larger SoCs for always-on tasks (wake-word,
   gesture detection) while the main NPU/GPU handles heavy inference.
4. **5 TOPS/W efficiency** matches or beats much larger accelerators — proves
   that specialization beats scale for narrow workloads.
