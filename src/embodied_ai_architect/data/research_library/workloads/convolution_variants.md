---
title: "Convolution Variants and HW Mapping"
domain: workload
tags: [convolution, depthwise, grouped, pointwise, hw-mapping]
relevance:
  - design_space_definition
  - design_tradeoffs
last_updated: 2026-03-16
---

## Convolution Types in Modern DNNs

| Type | FLOPS | Params | HW Utilization | Used By |
|------|-------|--------|----------------|---------|
| Standard 3×3 | C_in × C_out × K² × H × W | C_in × C_out × K² | 80-95% | ResNet, VGG |
| Depthwise 3×3 | C × K² × H × W | C × K² | 5-30% | MobileNet, EfficientNet |
| Pointwise 1×1 | C_in × C_out × H × W | C_in × C_out | 70-90% | MobileNet, ShuffleNet |
| Grouped | (C_in/G × C_out/G × K² × H × W) × G | (C_in/G × C_out/G × K²) × G | 40-70% | ResNeXt, ShuffleNet |

## The Depthwise Conv Problem

Depthwise convolutions are 8-10× fewer FLOPS than standard convolutions but
achieve only 5-30% utilization on systolic arrays. This is because:

1. **No output-channel reuse**: Each filter operates on one channel only,
   yielding an effective 1×1 systolic array usage per channel.
2. **Low arithmetic intensity**: Ratio of compute to memory access is ~K² ≈ 9,
   far below the ~100 needed to saturate most arrays.

**Impact**: MobileNet-style models that are "efficient" in FLOPS can be slower
than "heavier" models on NPUs. MobileNetV2 (0.3 GFLOPS) can be slower than
ResNet-18 (1.8 GFLOPS) on systolic arrays.

## Hardware-Specific Efficiency

| Conv Type | GPU | Systolic NPU | DSP | CGRA |
|-----------|-----|-------------|-----|------|
| Standard 3×3 | High | High | Medium | High |
| Depthwise 3×3 | Medium | Low | High | High |
| Pointwise 1×1 | High | High | Low | Medium |
| Dilated 3×3 | Medium | Medium | Medium | High |

**Key insight**: DSPs (like Hexagon) handle depthwise convolutions 3-5× more
efficiently than systolic arrays because they use vector processing without
the systolic data flow constraint.

## Lessons for Design Space Exploration

1. **Model choice interacts with hardware choice**: "Efficient" MobileNet may
   perform worse than "heavy" ResNet on a systolic-array-based NPU. The
   evaluator must model per-layer utilization.
2. **Heterogeneous compute helps**: Offloading depthwise convolutions to a DSP
   or vector unit while running pointwise on the systolic array can recover
   50% of the utilization gap.
3. **For the KPU design space**: The tile architecture should consider adding a
   vector processing mode for depthwise layers alongside the systolic array.
