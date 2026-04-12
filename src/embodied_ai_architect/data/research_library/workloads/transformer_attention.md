---
title: "Transformer Attention Compute Patterns"
domain: workload
tags: [transformer, attention, memory-bound, kv-cache, inference]
relevance:
  - mission_decomposition
  - design_space_definition
last_updated: 2026-03-16
---

## Compute Characteristics

Self-attention has fundamentally different compute patterns from convolutions:

| Aspect | Convolution | Self-Attention |
|--------|-------------|----------------|
| Compute | O(C² × H × W × K²) | O(N² × D) |
| Memory | Weight-dominated | Activation-dominated |
| Reuse pattern | High weight reuse | Low reuse (each Q×K unique) |
| Bottleneck | Compute-bound (large K) | Memory-bound (large N) |
| Systolic utilization | 70-90% | 20-50% |

Where N = sequence length, D = model dimension, K = kernel size.

## Hardware Implications

1. **Memory bandwidth is critical**: Attention reads Q, K, V matrices for every
   token pair. At sequence length 1024, this is 3× more memory traffic per FLOP
   than convolution.

2. **Systolic arrays underperform**: The irregular access patterns of Q×K^T
   softmax yield 20-50% utilization on weight-stationary arrays. GPUs with
   their flexible memory hierarchy handle this better.

3. **KV-cache for autoregressive**: In generative models, KV-cache grows
   linearly with sequence length. A 1024-token context at 768D needs ~6 MB
   in FP16. This must fit in SRAM for real-time.

## Relevance to Embodied AI

Most embodied AI perception pipelines use CNNs (YOLO, EfficientNet) rather
than transformers. However, emerging trends:

- **DETR** (Detection Transformer): Competitive with YOLO at higher compute
  cost. 41 GFLOPS vs 28.6 for YOLOv8s.
- **Vision Transformers (ViT)**: Strong for classification but expensive.
  ViT-B requires 17.6 GFLOPS vs 4.1 for ResNet-50.
- **Scene understanding**: Transformer-based scene graphs and reasoning
  modules are increasingly used for high-level decision making.

## Lessons for Design Space Exploration

1. **Transformer workloads need different hardware than CNNs**: High memory
   bandwidth, large on-chip buffers, flexible compute. Don't assume a CNN-
   optimized design will run transformers efficiently.
2. **Pipeline design should identify transformer stages** and route them to
   appropriate compute (GPU or flexible NPU, not fixed-function DLA).
3. **Hybrid architectures** (CNN backbone + transformer head) are common —
   the design space should support heterogeneous compute allocation.
