---
title: "Stillwater KPU Architecture"
domain: accelerator
tags: [stillwater, kpu, custom, posit, edge, inference]
relevance:
  - mission_decomposition
  - hardware_selection
  - design_space_definition
  - design_tradeoffs
peak_tops: 16
compute_density_tops_w: 8.0
process_nm: 16
target_workloads: [cnn, transformer, signal_processing, state_estimation, control]
last_updated: 2026-03-16
---

## Architecture Overview

The Stillwater Knowledge Processing Unit (KPU) is a custom accelerator designed
for embodied AI workloads. It uses a tiled architecture with configurable
systolic arrays and posit number system support for higher dynamic range at
lower bit widths.

| Metric | KPU-T256 | KPU-T64 |
|--------|----------|---------|
| Compute tiles | 16 | 4 |
| Systolic array | 16×16 per tile | 16×16 per tile |
| Peak TOPS (INT8) | 16 | 4 |
| Peak TOPS (Posit8) | 16 | 4 |
| TDP | 2W | 0.5W |
| TOPS/W | 8.0 | 8.0 |
| On-chip SRAM | 2 MB | 512 KB |
| Process | 16nm | 16nm |

## Key Design Decisions & Tradeoffs

- **Posit arithmetic**: Posit8 provides ~2× the dynamic range of INT8 with
  equivalent hardware cost, reducing quantization accuracy loss for sensitive
  layers (batch norm, softmax, attention scaling).
- **Tiled architecture**: Each tile is independently clockable and power-gatable,
  enabling fine-grained DVFS. A 4-tile config at 0.5W suits nano-drones.
- **Heterogeneous scheduling**: Tiles can be assigned to different models or
  pipeline stages simultaneously — native multi-model concurrency.
- **Programmable interconnect**: Mesh NoC with configurable link widths enables
  dataflow optimization per workload pattern.

## Performance Characteristics

**Detection (YOLOv8s, 640×640, Posit8):**
- KPU-T256: ~6ms latency, 166 FPS at 2W
- KPU-T64: ~22ms latency, 45 FPS at 0.5W

**Perception pipeline (detect + track + scene graph):**
- KPU-T256: ~10ms total pipeline at 2W

**State estimation (EKF, 12-state):**
- KPU-T64: <0.1ms per update — can run at 10 kHz on a single tile

## Lessons for Design Space Exploration

1. **Design space variables map directly to KPU config**: array_rows, array_cols,
   num_compute_tiles, sram_kb, noc_link_width are all KPU parameters.
2. **Posit arithmetic** should be a design variable (dtype: posit8 vs int8 vs fp16)
   with its own accuracy impact curve (~0.5% drop vs FP32, better than INT8).
3. **Tile-level power gating** means the design space should explore partial-tile
   configurations (e.g., 8 of 16 tiles active) for power-constrained missions.
4. **Multi-model concurrency** enables pipeline parallelism without the
   serialization overhead seen on Coral or basic NPUs.
