---
title: "Hardware-Aware Neural Architecture Search"
domain: efficiency
tags: [nas, codesign, efficientnet, ofa, once-for-all, hw-aware]
relevance:
  - design_space_definition
  - design_tradeoffs
last_updated: 2026-03-16
---

## Overview

Hardware-aware NAS jointly optimizes model architecture and hardware target,
finding models that maximize accuracy under hardware constraints (latency,
power, memory). Key approaches:

## Approaches

### 1. EfficientNet (Tan & Le, 2019)
- **Method**: Compound scaling of width, depth, resolution via grid search.
- **Result**: EfficientNet-B0 achieves 77.1% top-1 at 0.4 GFLOPS — 8.4×
  smaller than ResNet-50 at similar accuracy.
- **Lesson**: Width/depth/resolution scaling laws are predictable. A lookup
  table indexed by GFLOPS budget can estimate accuracy without running NAS.

### 2. Once-for-All (OFA) (Cai et al., 2020)
- **Method**: Train one supernet; extract sub-networks per hardware target
  via evolutionary search. No retraining needed.
- **Result**: Achieves near-SOTA accuracy across diverse hardware (GPU, CPU,
  mobile) with <1% accuracy gap from dedicated models.
- **Lesson**: A single trained supernet can serve the entire design space.
  Sub-network extraction is nearly free (milliseconds).

### 3. FBNet (Wu et al., 2019)
- **Method**: Differentiable NAS with latency lookup tables per target hardware.
- **Result**: FBNet-A achieves 73.0% top-1 with 249ms latency on Samsung S8
  vs MobileNetV2's 75ms at 71.8%.
- **Lesson**: Hardware-specific latency tables (not just FLOPS) are critical.
  The same model can be 3× faster on one chip vs another.

## Key Findings

1. **FLOPS is a poor proxy for latency**: Memory-bound layers (depthwise conv,
   attention) have low FLOPS but high latency. Hardware-aware NAS uses measured
   or modeled latency instead.

2. **Scaling laws enable accuracy estimation without NAS**: For a model family,
   accuracy ≈ a × log(FLOPS) + b. EfficientNet scaling achieves R² > 0.98
   within a family.

3. **Architecture + quantization co-search** is more impactful than either
   alone. MobileNetV3 + INT8 achieves higher accuracy per TOPS than
   EfficientNet-B3 + FP32 on mobile hardware.

## Lessons for Design Space Exploration

1. **Phase 3 NAS variables** should include `model_family`, `width_scale`,
   `depth_variant` — these are the knobs that EfficientNet/OFA proved impactful.
2. **Phase 1 accuracy tables already implement** the "lookup table" approach
   from EfficientNet scaling — extend with per-hardware latency tables in Phase 3.
3. **Co-design is multiplicative**: Optimizing model + hardware + quantization
   jointly achieves 5-10× better capability/watt vs optimizing any one dimension.
