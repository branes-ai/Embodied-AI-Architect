---
title: "Systolic Array Design Space"
domain: architecture
tags: [systolic, array, dataflow, gemm, convolution]
relevance:
  - design_space_definition
  - design_tradeoffs
last_updated: 2026-03-16
---

## Architecture Overview

Systolic arrays are the dominant compute primitive for DNN accelerators (TPU,
Jetson DLA, Intel Gaudi, Apple ANE). An N×M array performs N×M MACs per cycle,
streaming data through a regular grid of processing elements (PEs).

## Design Variables

| Variable | Range | Impact |
|----------|-------|--------|
| Array rows (N) | 4-256 | Throughput ∝ N×M, area ∝ N×M |
| Array cols (M) | 4-256 | Throughput ∝ N×M, area ∝ N×M |
| Data width | 4-32 bits | Area per PE ∝ width², power ∝ width |
| Dataflow | OS/WS/RS | Utilization varies by workload |

## Key Tradeoffs

**Size vs utilization**: A 256×256 array delivers peak TOPS on large GEMMs but
drops to <10% utilization on depthwise convolutions (1×1 effective array use).
A 16×16 array has lower peak but higher average utilization across diverse
workloads.

**Rule of thumb**: For edge inference with mixed Conv2D + depthwise + FC layers,
16×16 to 32×32 arrays hit the sweet spot. Larger arrays (64+) only pay off for
large batch sizes or very regular workloads.

**Dataflow choices**:
- **Output-stationary (OS)**: Accumulates partial sums in place. Good for large
  output channels. Used by TPU v1.
- **Weight-stationary (WS)**: Keeps weights fixed. Good for small batch. Used
  by most edge accelerators.
- **Row-stationary (RS)**: Maximizes data reuse across rows. Best overall for
  CNN inference. Used by Eyeriss.

## Lessons for Design Space Exploration

1. **array_rows × array_cols** is the single most impactful design variable —
   it determines peak throughput, area, and power simultaneously.
2. **Utilization is workload-dependent**: The evaluator should model utilization
   per layer type, not assume a fixed 30%.
3. **Multiple small arrays** can outperform one large array for multi-model
   pipelines due to independent scheduling.
4. **Data width directly impacts efficiency**: INT8 arrays are 4× the density
   of FP32 at equivalent area. Posit8 achieves INT8 density with FP16-like
   accuracy.
