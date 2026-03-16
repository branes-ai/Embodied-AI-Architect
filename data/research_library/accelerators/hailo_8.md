---
title: "Hailo-8 Dataflow Architecture"
domain: accelerator
tags: [hailo, dataflow, edge, inference, npu]
relevance:
  - hardware_selection
  - design_tradeoffs
  - design_space_definition
peak_tops: 26
compute_density_tops_w: 5.2
process_nm: 16
target_workloads: [cnn, detection, segmentation, pose]
last_updated: 2026-03-16
---

## Architecture Overview

Hailo-8 uses a novel "Structure-Defined Dataflow" architecture where the
compute fabric is dynamically reconfigured per layer to match the dataflow
pattern, minimizing data movement.

| Metric | Value |
|--------|-------|
| Peak throughput | 26 TOPS (INT8) |
| TDP | 5W (typical 2.5W) |
| TOPS/W | 5.2 (up to 10.4 at typical power) |
| On-chip memory | 32 MB |
| Process | 16nm |
| Interface | M.2, mini PCIe |

## Key Design Decisions & Tradeoffs

- **Dataflow reconfiguration per layer** eliminates fixed-function limitations
  of traditional NPUs — supports a wider range of layer types.
- **Compiler-driven mapping**: The Hailo Dataflow Compiler translates ONNX
  models to per-layer dataflow configurations. Compilation is offline.
- **Multi-context scheduling**: Can run 2-4 models concurrently with
  time-division multiplexing.
- **INT8/INT16 mixed precision**: Sensitive layers can use INT16 with ~2× slower
  execution but better accuracy preservation.

## Performance Characteristics

**Detection (YOLOv8s, 640×640, INT8):**
- 10ms latency at 5W, 100 FPS

**Detection (YOLOv8m, 640×640, INT8):**
- 28ms latency at 5W, 35 FPS

**Multi-model (YOLOv8n + pose estimation, concurrent):**
- 18ms combined at 5W

## Lessons for Design Space Exploration

1. **Best-in-class TOPS/W at the edge** — 5-10× better than Jetson Nano.
   Design space should model the efficiency gap between dataflow and GPU.
2. **Multi-model concurrency** is a differentiator — pipeline designs with
   2-3 models can run without serialization overhead.
3. **Compiler quality matters**: Real-world utilization is 40-70% of peak TOPS
   depending on model regularity. Irregular models (transformers) see lower
   utilization.
4. **No GPU fallback**: Unlike Jetson, unsupported layers fail compilation
   rather than falling back. Model compatibility must be validated upfront.
