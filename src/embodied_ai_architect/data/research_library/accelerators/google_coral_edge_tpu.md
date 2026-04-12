---
title: "Google Coral Edge TPU Architecture"
domain: accelerator
tags: [google, coral, tpu, edge, inference, microcontroller]
relevance:
  - hardware_selection
  - design_tradeoffs
peak_tops: 4
compute_density_tops_w: 2.0
process_nm: 28
target_workloads: [cnn, classification, detection]
last_updated: 2026-03-16
---

## Architecture Overview

The Edge TPU is a fixed-function ASIC optimized for 8-bit quantized CNN inference.
Two form factors: USB Accelerator (2 TOPS @ 2W) and Dev Board (4 TOPS @ 2W TDP).

| Metric | Value |
|--------|-------|
| Peak throughput | 4 TOPS (INT8) |
| TDP | 2W |
| TOPS/W | 2.0 |
| On-chip SRAM | 8 MB |
| Process | 28nm |
| Supported ops | Conv2D, DepthwiseConv, Pool, FC, Add, Concat |

## Key Design Decisions & Tradeoffs

- **INT8-only**: No FP16/FP32 path. Models must be fully quantized via
  TFLite + post-training quantization or QAT.
- **Layer partitioning**: Unsupported ops (custom activations, attention, LSTM)
  fall back to CPU, causing 10-100× slowdown for those layers.
- **Single model at a time**: No concurrent model execution. Pipeline must
  serialize model calls.
- **No on-device training**: Inference only.

## Performance Characteristics

**Classification (MobileNetV2, 224×224, INT8):**
- 3.0ms latency, 333 FPS

**Detection (SSD-MobileNetV2, 300×300, INT8):**
- 8.0ms latency, 125 FPS

**Detection (EfficientDet-D0, 512×512, INT8):**
- 25ms latency, 40 FPS (some layers on CPU)

## Lessons for Design Space Exploration

1. **Best for simple, single-model pipelines** where the full model fits on TPU.
   Not suitable for multi-model perception stacks.
2. **Model architecture matters enormously**: MobileNet/EfficientNet families
   map well; anything with attention or custom ops does not.
3. **2W TDP is ideal for battery-powered** devices (small drones, wearables).
4. **Quantization is mandatory** — design space must treat INT8 as the only
   option when targeting Coral, and accuracy tables should reflect this.
