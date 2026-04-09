---
title: Model Analysis
description: Analyze neural network models for edge deployment.
---

Model analysis is the foundation of Embodied AI Architect. Before you can optimize or deploy a model, you need to understand its characteristics.

## What Gets Analyzed

When you analyze a model, we extract:

### Architecture Information
- Layer types and counts (Conv2D, Linear, Attention, etc.)
- Network depth and width
- Skip connections and branching

### Computational Requirements
- Total FLOPs (floating-point operations)
- Per-layer FLOP breakdown
- Arithmetic intensity (FLOPs per byte)

### Memory Requirements
- Parameter count and size
- Activation memory (per batch size)
- Peak memory during inference

## Using the CLI

### Basic Analysis

```bash
branes analyze model.pt
```

### With Hardware Target

```bash
branes analyze model.pt --hardware jetson-orin-nano
```

### With Input Shape

```bash
branes analyze model.pt --input-shape 1,3,640,640
```

## Understanding the Output

```
Model Analysis: yolov8n
========================

Architecture:
  - Type: Object Detection (YOLO)
  - Backbone: CSPDarknet
  - Layers: 225
  - Parameters: 3.2M

Compute:
  - FLOPs: 8.7 GFLOPs
  - Arithmetic Intensity: 45.2 FLOPs/byte

Memory:
  - Weights: 12.4 MB (FP32)
  - Activations: 24.8 MB @ batch=1
  - Peak: 37.2 MB

Hardware Fit (Jetson Orin Nano):
  - Predicted Latency: 28.5 ms
  - Bottleneck: Memory-bound (67% BW utilization)
  - Recommendation: INT8 quantization for 1.8x speedup
```

## Supported Model Formats

| Format | Extension | Support |
|--------|-----------|---------|
| PyTorch | `.pt`, `.pth` | Full |
| ONNX | `.onnx` | Full |
| TorchScript | `.pt` | Full |
| SavedModel | `saved_model/` | Partial |

## API Usage

```python
from embodied_ai_architect.agents import ModelAnalyzerAgent

analyzer = ModelAnalyzerAgent()
result = analyzer.execute({"model": "yolov8n.pt"})

print(f"Parameters: {result.data['parameters']:,}")
print(f"FLOPs: {result.data['flops'] / 1e9:.1f} GFLOPs")
```

## Next Steps

- [Check hardware fit with roofline analysis](/features/roofline-analysis/)
- [Verify deployment constraints](/features/constraint-checking/)
