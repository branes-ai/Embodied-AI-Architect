---
title: YOLOv8 on Jetson Orin
description: Step-by-step guide to deploying YOLOv8 on NVIDIA Jetson with TensorRT optimization.
---

This tutorial walks you through deploying YOLOv8 on a Jetson Orin device with TensorRT optimization.

## Overview

You'll learn how to:
- Analyze YOLOv8 variants for your Jetson
- Choose the right model size and precision
- Export and optimize with TensorRT
- Benchmark real-world performance

## Prerequisites

- **Hardware**: Jetson Orin Nano, NX, or AGX
- **Software**: JetPack 5.1+ installed
- **Model**: YOLOv8 (we'll use nano and small variants)

## 1. Analyze Model-Hardware Fit

First, let's see how different YOLOv8 variants perform on your Jetson:

```bash
branes compare yolov8n,yolov8s,yolov8m \
  --hardware jetson-orin-nano \
  --metric latency,fps,power
```

**Expected Output:**

| Model | Latency | FPS | Power | Verdict |
|-------|---------|-----|-------|---------|
| YOLOv8n | 28ms | 35 | 12W | PASS |
| YOLOv8s | 45ms | 22 | 14W | MARGINAL |
| YOLOv8m | 92ms | 11 | 15W | FAIL |

For 30 FPS real-time detection, YOLOv8n is the clear choice.

## 2. Check Constraints

Verify YOLOv8n meets your requirements:

```bash
# Check 30fps target (33ms)
branes check-latency yolov8n \
  --hardware jetson-orin-nano \
  --target 33ms

# Check power budget
branes check-power yolov8n \
  --hardware jetson-orin-nano \
  --budget 15W
```

## 3. Export to ONNX

Export your PyTorch model to ONNX:

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.export(format="onnx", imgsz=640, simplify=True)
```

Or with the CLI:

```bash
yolo export model=yolov8n.pt format=onnx imgsz=640
```

## 4. Optimize with TensorRT

### FP16 (Recommended Start)

```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=yolov8n.onnx \
  --saveEngine=yolov8n_fp16.engine \
  --fp16 \
  --workspace=4096
```

### INT8 (Maximum Performance)

For INT8, you need calibration images:

```bash
# Prepare calibration images (100-500 representative images)
mkdir calibration_images
# Copy images to calibration_images/

# Export with INT8
/usr/src/tensorrt/bin/trtexec \
  --onnx=yolov8n.onnx \
  --saveEngine=yolov8n_int8.engine \
  --int8 \
  --calib=calibration.cache \
  --workspace=4096
```

## 5. Benchmark Performance

Test your optimized engine:

```bash
/usr/src/tensorrt/bin/trtexec \
  --loadEngine=yolov8n_fp16.engine \
  --warmUp=500 \
  --iterations=1000
```

**Expected Results (Orin Nano, FP16):**

| Metric | Value |
|--------|-------|
| Latency (median) | 25.1ms |
| Latency (99th) | 27.3ms |
| Throughput | 39.8 FPS |

## 6. Run Inference

```python
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2

# Load engine
with open("yolov8n_fp16.engine", "rb") as f:
    engine = trt.Runtime(trt.Logger()).deserialize_cuda_engine(f.read())

# Create execution context
context = engine.create_execution_context()

# Allocate buffers and run inference
# (See full example in repository)
```

## Expected Results

| Configuration | Latency | FPS | Accuracy |
|---------------|---------|-----|----------|
| FP32 (PyTorch) | 85ms | 12 | 37.3 mAP |
| FP16 (TensorRT) | 25ms | 40 | 37.2 mAP |
| INT8 (TensorRT) | 14ms | 71 | 36.8 mAP |

## Troubleshooting

### "Unsupported operator" error

Some ONNX operators may not be supported. Try:

```bash
yolo export model=yolov8n.pt format=onnx simplify=True opset=11
```

### Out of memory during conversion

Reduce workspace size:

```bash
trtexec --onnx=model.onnx --workspace=2048  # 2GB instead of 4GB
```

### Low FPS in practice

1. Check power mode: `sudo nvpmodel -q`
2. Set maximum performance: `sudo nvpmodel -m 0`
3. Enable max clocks: `sudo jetson_clocks`

## Next Steps

- Add tracking with [ByteTrack](/tutorials/adding-tracking/)
- Try [INT8 quantization](/tutorials/int8-quantization/)
- Deploy [multiple models](/tutorials/multi-model-pipeline/)
