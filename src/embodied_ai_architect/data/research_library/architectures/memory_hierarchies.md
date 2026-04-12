---
title: "Memory Hierarchy Design for DNN Accelerators"
domain: architecture
tags: [memory, sram, hbm, lpddr, bandwidth, hierarchy]
relevance:
  - design_space_definition
  - design_tradeoffs
last_updated: 2026-03-16
---

## Architecture Overview

DNN inference is fundamentally memory-bound for most layers. The memory hierarchy
determines achievable throughput, energy efficiency, and model capacity.

## Memory Technologies

| Technology | Bandwidth | Capacity | Energy/access | Area density |
|-----------|-----------|----------|---------------|-------------|
| Register file | 10+ TB/s | <1 KB | 0.1 pJ | Lowest |
| SRAM (on-chip) | 1-10 TB/s | 32 KB - 32 MB | 1-10 pJ | Low |
| LPDDR4X | 25-50 GB/s | 2-8 GB | 100-200 pJ | Medium (off-chip) |
| LPDDR5 | 50-200 GB/s | 4-32 GB | 80-150 pJ | Medium (off-chip) |
| HBM2e | 400-900 GB/s | 8-80 GB | 3-5 pJ | High (in-package) |

## Design Rules of Thumb

**SRAM budget**: Allocate SRAM to hold at least one layer's activations plus
weights. For YOLOv8s at 640×640, the largest activation is ~2.5 MB. A 4 MB
SRAM budget avoids DRAM spills for all but the first layers.

**Bandwidth requirement**: For real-time detection at 30 FPS:
- Model weights: ~11 MB (YOLOv8s INT8) × 30 = 330 MB/s
- Activations: ~5 MB × 30 = 150 MB/s
- Minimum bandwidth: ~500 MB/s (0.5 GB/s) — easily met by LPDDR4X

**Bandwidth bottleneck threshold**: When peak_compute_tops × bytes_per_op >
memory_bandwidth, the design is memory-bound. This is the roofline knee point.

## Lessons for Design Space Exploration

1. **sram_kb is the variable that determines the memory-bound/compute-bound
   boundary**. Current evaluator under-weights this — larger SRAM reduces DRAM
   accesses and thus energy/latency.
2. **Memory bandwidth should be a constraint**, not just an implicit consequence
   of process node. Adding `ddr_bandwidth_gbps` as a design variable would
   improve model fidelity.
3. **For edge devices (1-15W)**, LPDDR4X/5 is the only viable off-chip option.
   HBM is datacenter/automotive only due to cost and integration complexity.
4. **Double buffering** SRAM (load next layer while computing current) hides
   latency but requires 2× the SRAM budget. Design space should model this.
