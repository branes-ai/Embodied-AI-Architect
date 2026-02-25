---
title: Roofline Analysis for Model Optimization
description: Step-by-step guide to using roofline modeling to diagnose bottlenecks and choose the right optimization strategy.
---

This tutorial walks you through using roofline analysis to understand whether your model is compute-bound or memory-bound on specific hardware, and how to use that knowledge to pick the right optimization.

## Overview

You'll learn how to:
- Run roofline analysis on a model-hardware pair
- Read the roofline diagram to identify bottleneck type
- Compare hardware using their roofline characteristics
- Choose optimization strategies based on bottleneck type
- Use batch size and precision to shift the operating point

## Prerequisites

- **Embodied AI Architect** installed (`pip install -e ".[dev]"`)
- Basic understanding of model architectures (ResNet, YOLO, etc.)

## 1. Your First Roofline Analysis

Let's analyze ResNet-50 on a Jetson Orin Nano:

```bash
embodied-ai analyze resnet50 --hardware jetson-orin-nano --roofline
```

**Expected Output:**

```
Roofline Analysis: resnet50 on Jetson Orin Nano
────────────────────────────────────────────────
Arithmetic Intensity: 23.4 FLOPs/byte
Ridge Point:          89.6 FLOPs/byte
Bottleneck:           MEMORY-BOUND

Performance:
  Peak Compute:  100 TOPS (INT8)
  Peak Bandwidth: 68 GB/s
  Achieved:       14.2 TOPS (14.2% utilization)

Breakdown:
  Compute Utilization:  14.2%
  Memory Utilization:   89.1%

  ┌─ Performance (TOPS)
  │
  │         ┌─────────────────── Peak Compute (100 TOPS)
  │        /
  │       /
  │      /
  │     /  ← ridge point (89.6)
  │    /
  │   / ★ ResNet-50 (23.4, 14.2 TOPS)
  │  /
  │ /
  └──────────────────────────────────
     Arithmetic Intensity (FLOPs/byte)
```

**How to read this:**
- The **★** marker shows where your model operates
- It sits on the **memory bandwidth slope** (left of the ridge point)
- This means the model is **memory-bound** — the hardware can compute faster than it can feed data
- **14.2% compute utilization** confirms most of the silicon is idle, waiting for data

## 2. Understanding Bottleneck Types

### Memory-Bound (most common at batch=1)

```bash
embodied-ai analyze resnet50 --hardware a100 --roofline
```

```
Arithmetic Intensity: 23.4 FLOPs/byte
Ridge Point:          156.0 FLOPs/byte
Bottleneck:           MEMORY-BOUND
Compute Utilization:  15.0%
Memory Utilization:   89.2%
```

The model's arithmetic intensity (23.4) is well below the hardware ridge point (156.0). The GPU spends most of its time moving data, not computing.

**What causes memory-bound behavior:**
- Small batch sizes (batch=1 is worst)
- Depthwise convolutions (low arithmetic intensity per layer)
- Large activations with small kernels
- Lots of element-wise operations (ReLU, batch norm)

### Compute-Bound

```bash
embodied-ai analyze resnet50 --hardware jetson-orin-nano \
  --batch-size 32 --roofline
```

```
Arithmetic Intensity: 412.8 FLOPs/byte
Ridge Point:          89.6 FLOPs/byte
Bottleneck:           COMPUTE-BOUND
Compute Utilization:  82.3%
Memory Utilization:   24.1%
```

At batch=32, arithmetic intensity jumps to 412.8 — well above the ridge point. Now the GPU is fully utilized and data supply isn't the bottleneck.

**What causes compute-bound behavior:**
- Large batch sizes
- Dense matrix multiplications (transformers, large convolutions)
- High arithmetic intensity per layer

## 3. Compare Across Hardware

Different hardware has very different roofline shapes. Analyze the same model on several targets:

```bash
embodied-ai analyze resnet50 --hardware h100 --roofline
embodied-ai analyze resnet50 --hardware a100 --roofline
embodied-ai analyze resnet50 --hardware jetson-orin-nano --roofline
embodied-ai analyze resnet50 --hardware coral-edge-tpu --roofline
```

**Comparison:**

| Hardware | Peak TFLOPS | Memory BW | Ridge Point | Bottleneck |
|----------|-------------|-----------|-------------|------------|
| H100 SXM | 1,979 (FP16) | 3.35 TB/s | 590 | Memory-bound |
| A100 SXM | 312 (FP16) | 2.0 TB/s | 156 | Memory-bound |
| Orin Nano | 100 (INT8) | 68 GB/s | 89.6 | Memory-bound |
| Coral TPU | 4 (INT8) | 8 GB/s | 500 | Memory-bound |

**Key insight:** ResNet-50 at batch=1 is memory-bound on *all* of these. The H100 has 20x the compute of the A100, but at batch=1 that extra compute is entirely wasted. You'd get better cost-efficiency from a cheaper card with high memory bandwidth.

## 4. Shift the Operating Point

The roofline shows you two levers to change your model's operating point.

### Lever 1: Increase Batch Size

Batching amortizes weight loading across multiple inputs, increasing arithmetic intensity:

```bash
embodied-ai analyze resnet50 --hardware a100 --batch-size 1 --roofline
embodied-ai analyze resnet50 --hardware a100 --batch-size 8 --roofline
embodied-ai analyze resnet50 --hardware a100 --batch-size 32 --roofline
```

| Batch | Arith. Intensity | Utilization | Bottleneck |
|-------|-----------------|-------------|------------|
| 1 | 23.4 | 15.0% | Memory |
| 8 | 142.1 | 68.4% | Memory (near ridge) |
| 32 | 412.8 | 92.1% | Compute |

At batch=8 you're near the ridge point — the sweet spot where both compute and memory are well utilized.

### Lever 2: Reduce Precision

Lower precision reduces bytes per element, increasing arithmetic intensity and reducing memory traffic:

```bash
embodied-ai analyze resnet50 --hardware a100 --precision fp32 --roofline
embodied-ai analyze resnet50 --hardware a100 --precision fp16 --roofline
embodied-ai analyze resnet50 --hardware a100 --precision int8 --roofline
```

| Precision | Arith. Intensity | Speedup | Bottleneck |
|-----------|-----------------|---------|------------|
| FP32 | 11.7 | 1.0x | Memory |
| FP16 | 23.4 | 1.9x | Memory |
| INT8 | 46.8 | 3.6x | Memory |

Each precision step roughly doubles the arithmetic intensity. FP16 and INT8 also unlock dedicated hardware units (tensor cores) with higher peak throughput.

## 5. Analyze Individual Layers

Not all layers have the same bottleneck. Break down by layer type:

```bash
embodied-ai analyze resnet50 --hardware a100 --roofline --per-layer
```

**Expected Output (selected layers):**

```
Layer Roofline Breakdown
────────────────────────
 Layer              | Type     | AI (F/B) | Bottleneck | Utilization
 conv1 (7x7, s2)   | Conv2d   | 42.3     | Memory     | 27.1%
 layer1.0.conv1     | Conv2d   | 18.2     | Memory     | 11.7%
 layer1.0.conv2     | Conv2d   | 36.4     | Memory     | 23.4%
 layer4.2.conv2     | Conv2d   | 128.5    | Memory     | 82.4%
 fc                 | Linear   | 8.1      | Memory     | 5.2%

Worst layers (lowest utilization):
  1. fc (5.2%) — small matrix, dominated by weight loading
  2. layer1.0.conv1 (11.7%) — 1x1 conv, low arithmetic intensity
```

**Key insight:** The final FC layer is extremely memory-bound (5.2% utilization). If this layer is your bottleneck, operator fusion or replacing it with a global average pool + small head can help.

## 6. Interactive Chat

```
You: Is YOLOv8n compute-bound or memory-bound on a Jetson Orin Nano?

Architect: YOLOv8n is memory-bound on the Jetson Orin Nano at batch=1.

  Arithmetic Intensity: 18.7 FLOPs/byte
  Ridge Point: 89.6 FLOPs/byte
  Compute Utilization: 20.9%
  Memory Utilization: 93.4%

  The model is far left of the ridge point — memory bandwidth is the
  bottleneck, not compute.

You: How do I make it faster?

Architect: For a memory-bound model, focus on reducing data movement:
  1. INT8 quantization: 2x less memory traffic, ~1.8x speedup
  2. Operator fusion: eliminate intermediate activations between conv+bn+relu
  3. TensorRT: does both automatically — expect 25ms → 14ms

  Increasing batch size won't help for real-time (adds latency per frame).

You: What if I switch to the A100?

Architect: Still memory-bound on the A100 — arithmetic intensity doesn't
  change with hardware. But the A100 has 29x more memory bandwidth
  (2.0 TB/s vs 68 GB/s), so latency drops from 28ms to ~2ms.

  For a drone at 30fps, the Orin Nano is sufficient (28ms < 33ms).
  The A100 is overkill for this workload.
```

## 7. Python API

```python
from graphs.analysis import UnifiedAnalyzer

analyzer = UnifiedAnalyzer()

# Run roofline analysis
result = analyzer.analyze_model(
    model_name="resnet50",
    hardware_name="A100-SXM4-80GB",
    batch_size=1,
)

roofline = result.roofline_report
print(f"Bottleneck: {roofline.bottleneck_type}")
print(f"Arithmetic Intensity: {roofline.arithmetic_intensity:.1f} FLOPs/byte")
print(f"Ridge Point: {roofline.ridge_point:.1f} FLOPs/byte")
print(f"Compute Utilization: {roofline.compute_utilization:.1%}")
print(f"Memory Utilization: {roofline.memory_utilization:.1%}")

# Check if increasing batch size would help
result_b8 = analyzer.analyze_model(
    model_name="resnet50",
    hardware_name="A100-SXM4-80GB",
    batch_size=8,
)
print(f"Batch=8 utilization: {result_b8.roofline_report.compute_utilization:.1%}")
```

## Optimization Decision Tree

Use the roofline result to pick your strategy:

```
Is the model memory-bound?
├── YES (AI < ridge point)
│   ├── Can you increase batch size?
│   │   ├── YES → Batch until you hit ridge point
│   │   └── NO (real-time) → Reduce precision (FP16/INT8)
│   ├── Can you fuse operators?
│   │   └── YES → Use TensorRT/ONNX Runtime graph optimization
│   └── Still too slow?
│       └── Choose hardware with higher memory bandwidth
│
└── NO — compute-bound (AI > ridge point)
    ├── Can you reduce precision?
    │   └── YES → INT8/INT4 quantization
    ├── Can you use a smaller model?
    │   └── YES → Distillation, pruning, NAS
    └── Still too slow?
        └── Choose hardware with higher peak compute
```

## Tips

- **Batch=1 is almost always memory-bound** on modern GPUs — don't blame compute
- **Ridge point is hardware-specific** — the same model is memory-bound on an A100 (ridge=156) but compute-bound on a Coral TPU (ridge=500)
- **Arithmetic intensity is model-specific** — it doesn't change with hardware, only with batch size and precision
- **Utilization < 30%** means most of your hardware investment is wasted — consider cheaper hardware or larger batches
- **Transformer models** tend to be more compute-bound than CNNs due to dense attention matrices

## Next Steps

- [Check deployment constraints](/tutorials/constraint-checking/) with concrete PASS/FAIL verdicts
- [Analyze your full codebase](/tutorials/codebase-analysis/) to find which kernels are bottlenecks
- [Explore custom hardware designs](/tutorials/design-space-optimization/) optimized for your workload's roofline position
- See the [hardware catalog](/catalog/hardware/) for roofline parameters of all supported targets
