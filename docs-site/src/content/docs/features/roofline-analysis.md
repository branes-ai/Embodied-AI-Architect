---
title: Roofline Analysis
description: Understand model-hardware fit using roofline modeling.
---

Roofline analysis helps you understand whether your model is **compute-bound** or **memory-bound** on specific hardware. This determines your optimization strategy.

## What is Roofline Modeling?

The roofline model plots achievable performance against arithmetic intensity:

```
Performance
(FLOPS)
    │
    │     ┌─────────────── Compute Ceiling
    │    /
    │   /
    │  /  Memory
    │ /   Bandwidth
    │/    Ceiling
    └────────────────────────
         Arithmetic Intensity (FLOPS/byte)
```

- **Memory-bound**: Operations below the diagonal (limited by memory bandwidth)
- **Compute-bound**: Operations at the ceiling (limited by compute capacity)

## Using Roofline Analysis

### CLI

```bash
branes analyze yolov8n.pt --hardware h100 --roofline
```

### Interactive Chat

```
> Is ResNet-50 compute-bound or memory-bound on an A100?

ResNet-50 is **memory-bound** on the A100 at batch size 1.

Breakdown:
- Arithmetic Intensity: 23.4 FLOPs/byte
- A100 Ridge Point: 156 FLOPs/byte
- Compute Utilization: 15.0%
- Memory Utilization: 89.2%

The model spends most time waiting for data, not computing.

**Recommendations:**
1. Increase batch size to improve arithmetic intensity
2. Use FP16 to reduce memory traffic
3. Consider operator fusion to reduce intermediate activations
```

## Bottleneck Types

### Memory-Bound

**Symptoms:**
- Low compute utilization (<50%)
- High memory bandwidth utilization (>80%)
- Arithmetic intensity below hardware ridge point

**Solutions:**
- Reduce precision (FP32 → FP16 → INT8)
- Increase batch size
- Operator fusion
- Model pruning

### Compute-Bound

**Symptoms:**
- High compute utilization (>80%)
- Moderate memory utilization
- Arithmetic intensity above hardware ridge point

**Solutions:**
- Use faster hardware
- Quantization (INT8, INT4)
- Model distillation
- Architecture changes (MobileNet, EfficientNet)

## Hardware-Specific Analysis

Different hardware has different characteristics:

| Hardware | Peak TFLOPS | Memory BW | Ridge Point |
|----------|-------------|-----------|-------------|
| H100 | 1,979 (FP16) | 3.35 TB/s | 590 |
| A100 | 312 (FP16) | 2.0 TB/s | 156 |
| Jetson Orin AGX | 275 (INT8) | 204 GB/s | 1,348 |
| Coral Edge TPU | 4 (INT8) | 8 GB/s | 500 |

## API Usage

```python
from graphs.analysis import UnifiedAnalyzer

analyzer = UnifiedAnalyzer()
result = analyzer.analyze_model(
    model_name="resnet50",
    hardware_name="A100-SXM4-80GB",
    batch_size=1,
)

print(f"Bottleneck: {result.roofline_report.bottleneck_type}")
print(f"Compute Utilization: {result.roofline_report.compute_utilization:.1%}")
print(f"Memory Utilization: {result.roofline_report.memory_utilization:.1%}")
```

## Next Steps

- [Verify constraints](/features/constraint-checking/)
- [Compare hardware options](/features/hardware-selection/)
