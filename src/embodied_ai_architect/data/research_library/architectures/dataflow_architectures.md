---
title: "Spatial Dataflow Architectures"
domain: architecture
tags: [dataflow, spatial, temporal, row-stationary, weight-stationary]
relevance:
  - design_space_definition
  - design_tradeoffs
last_updated: 2026-03-16
---

## Architecture Overview

Dataflow architectures organize computation around data movement patterns rather
than instruction streams. Three primary paradigms:

1. **Temporal (SIMD/GPU)**: All PEs execute the same instruction on different
   data. Flexible but high control overhead.
2. **Spatial (Systolic)**: Data flows through a fixed PE topology. Low control
   overhead but inflexible.
3. **Reconfigurable (CGRA/Hailo)**: PE connections are reprogrammed per layer.
   Balances flexibility and efficiency.

## Dataflow Taxonomy for DNN Layers

| Dataflow | Best For | Weakness | Examples |
|----------|----------|----------|----------|
| Weight-stationary | Small batch inference | Underutilized on depthwise conv | Most edge NPUs |
| Output-stationary | Large output channels | High input bandwidth | TPU v1 |
| Row-stationary | CNN inference (mixed) | Complex control | Eyeriss, Eyeriss v2 |
| No local reuse | Transformers (attention) | High bandwidth demand | GPUs |

## Energy Cost Hierarchy

Data movement dominates energy in DNN inference:

| Operation | Energy (pJ) | Relative |
|-----------|-------------|----------|
| INT8 MAC | 0.2 | 1× |
| SRAM read (256 KB) | 5 | 25× |
| DRAM read | 200 | 1000× |

**Implication**: A design that keeps data on-chip saves 40× energy per access.
SRAM sizing is critical — too small forces DRAM spills.

## Lessons for Design Space Exploration

1. **SRAM size is the second most important variable** after array size. The
   evaluator should model working set vs SRAM capacity and penalize DRAM spills.
2. **Dataflow should be a categorical variable** in the design space — different
   workloads prefer different dataflows.
3. **Energy per inference** = compute energy + data movement energy. Current
   evaluator only models compute. Adding memory traffic modeling would improve
   accuracy 2-3×.
4. **Reconfigurable dataflow** (like Hailo) adds ~20% area overhead but achieves
   1.5-2× better utilization across diverse workloads.
