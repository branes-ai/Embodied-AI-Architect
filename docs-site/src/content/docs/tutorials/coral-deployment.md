---
title: Deploying to Coral Edge TPU
description: Step-by-step guide to deploying models on Google Coral devices for ultra-low-power inference.
---

This tutorial walks you through deploying a detection model to a Google Coral Edge TPU — from model export through INT8 quantization to running inference at sub-watt power levels.

## Overview

You'll learn how to:
- Understand Coral Edge TPU constraints (INT8 only, supported ops)
- Export and quantize a model for Edge TPU
- Compile with the Edge TPU compiler
- Validate performance against a TFLite baseline
- Run inference on Coral hardware

## Prerequisites

- **Embodied AI Architect** installed with Coral extras:
  ```bash
  pip install -e ".[coral]"
  ```
- **Hardware** (any one):
  - Coral USB Accelerator
  - Coral Dev Board
  - Coral M.2 or Mini PCIe Accelerator
- **Edge TPU compiler** installed on your system ([installation guide](https://coral.ai/docs/edgetpu/compiler/))
- **Calibration images**: 100-500 representative images for INT8 quantization

## 1. Check Hardware Fit

Before deploying, verify the model can meet your constraints on the Edge TPU:

```bash
branes check-latency mobilenetv2 \
  --hardware coral-edge-tpu \
  --target 10ms
```

**Expected Output:**

```
PASS - Latency: 3.0ms (target: 10ms)
       Headroom: 70.0%
       Confidence: HIGH
```

The Coral Edge TPU delivers 4 TOPS at ~0.5W per TOPS — ideal for always-on and battery-powered applications. But it only supports INT8, so model selection matters.

Check what works and what doesn't:

```bash
# Small models: excellent fit
branes check-latency mobilenetv2 --hardware coral-edge-tpu --target 10ms
branes check-latency yolov8n --hardware coral-edge-tpu --target 33ms

# Large models: likely FAIL
branes check-latency resnet152 --hardware coral-edge-tpu --target 33ms
```

| Model | Latency | Power | Verdict |
|-------|---------|-------|---------|
| MobileNetV2 | 3ms | ~0.5W | PASS |
| EfficientDet-Lite0 | 6ms | ~0.8W | PASS |
| YOLOv8n | 28ms | ~1.2W | PASS (at 30fps) |
| ResNet-152 | 180ms+ | ~2W | FAIL |

## 2. Export to ONNX

Start with a PyTorch model and export to ONNX:

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.export(format="onnx", imgsz=320, simplify=True)
```

We use 320x320 input (not 640) because the Edge TPU has limited on-chip SRAM — smaller inputs keep more of the model on-chip and avoid costly off-chip data movement.

## 3. Deploy with Branes

The deployment command handles the full pipeline — ONNX to TFLite conversion, INT8 quantization with calibration, and Edge TPU compilation:

```bash
branes deploy yolov8n.onnx \
  --target coral \
  --precision int8 \
  --calibration-data ./calibration_images/ \
  --input-shape 1,3,320,320 \
  --output-dir ./coral_deployment
```

**Expected Output:**

```
Deployment: Coral Edge TPU
──────────────────────────
Step 1/4: Converting ONNX → TFLite (FP32)... done
Step 2/4: INT8 quantization with calibration (237 images)... done
Step 3/4: Edge TPU compilation... done
Step 4/4: Validation... done

Artifacts:
  ./coral_deployment/
  ├── model.tflite              # Quantized TFLite model
  ├── model_edgetpu.tflite      # Edge TPU compiled model
  ├── model.onnx                # Original ONNX
  └── config.json               # Deployment config

Validation:
  Input shape:  [1, 320, 320, 3]
  Output shape: [1, 52, 2100]
  Latency (CPU TFLite): 45.2ms
  Latency (Edge TPU):   12.8ms  (3.5x speedup)
  Power estimate:        ~1.0W
```

## 4. Understanding Edge TPU Compilation

The Edge TPU compiler partitions the model into operations that run on the TPU and operations that fall back to CPU:

```
Edge TPU Compilation Report
  Operations mapped to Edge TPU: 127/134 (94.8%)
  Operations on CPU fallback:    7/134 (5.2%)

  CPU fallback ops:
    - CUSTOM_OP (SiLU activation) x4
    - RESIZE_BILINEAR x2
    - CONCATENATION x1
```

**Key insight:** Operations not supported by the Edge TPU fall back to CPU, creating a round-trip penalty. Common fallback ops include:
- Custom activations (SiLU, Mish — use ReLU instead)
- Some resize modes
- Dynamic shapes

For maximum performance, choose architectures designed for Edge TPU (MobileNet, EfficientDet-Lite).

## 5. Run Inference

### With pycoral (on Coral device)

```python
from pycoral.adapters import common, detect
from pycoral.utils.edgetpu import make_interpreter
import numpy as np
from PIL import Image

# Load Edge TPU model
interpreter = make_interpreter("coral_deployment/model_edgetpu.tflite")
interpreter.allocate_tensors()

# Prepare input
image = Image.open("test.jpg").resize((320, 320))
common.set_input(interpreter, image)

# Run inference
interpreter.invoke()

# Get results
detections = detect.get_objects(interpreter, score_threshold=0.5)
for d in detections:
    print(f"  {d.id}: {d.score:.2f} at ({d.bbox.xmin}, {d.bbox.ymin})")
```

### With TFLite Runtime (without pycoral)

```python
import tflite_runtime.interpreter as tflite
import numpy as np

# Load with Edge TPU delegate
interpreter = tflite.Interpreter(
    model_path="coral_deployment/model_edgetpu.tflite",
    experimental_delegates=[
        tflite.load_delegate("libedgetpu.so.1")
    ],
)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Run inference
input_data = np.random.rand(1, 320, 320, 3).astype(np.uint8)
interpreter.set_tensor(input_details[0]["index"], input_data)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]["index"])
```

## 6. Benchmark on Device

Once deployed, benchmark on the actual Coral hardware:

```bash
# On the Coral device / host with USB Accelerator
branes benchmark coral_deployment/model_edgetpu.tflite \
  --backend local \
  --iterations 1000 \
  --warmup 50
```

**Expected Results (YOLOv8n, 320x320, INT8):**

| Metric | USB Accelerator | Dev Board |
|--------|----------------|-----------|
| Latency (median) | 14.2ms | 12.8ms |
| Latency (99th) | 16.1ms | 13.9ms |
| Throughput | 70 FPS | 78 FPS |
| Power | ~1.0W | ~1.2W |
| Efficiency | 70 inf/W | 65 inf/W |

## 7. Power Efficiency Comparison

The Coral's strength is inferences-per-watt:

| Platform | YOLOv8n Latency | Power | Inf/Watt |
|----------|----------------|-------|----------|
| Coral USB | 14ms | 1.0W | 70 |
| Jetson Orin Nano | 28ms | 12W | 3 |
| Ryzen AI NPU | 8ms | 15W | 8 |

The Coral delivers 10-20x better power efficiency, making it the right choice for battery-powered robots, drones with limited power, and always-on monitoring.

## Tips

- **Use 320x320 inputs** instead of 640x640 — the Edge TPU has limited SRAM; smaller inputs keep more ops on-chip
- **Prefer MobileNet/EfficientDet-Lite** architectures — they're designed for Edge TPU and achieve near-100% on-chip mapping
- **Avoid SiLU/Mish activations** — they fall back to CPU; use ReLU or ReLU6 instead
- **Check the partition report** — if >10% of ops fall back to CPU, consider a different architecture
- **Stack multiple USB Accelerators** for pipeline parallelism on multi-camera systems
- **Calibration matters** — use representative images from your actual deployment environment

## Next Steps

- [Deploy to OpenVINO](/tutorials/openvino-deployment/) for Intel/AMD x86 targets
- [Deploy to Ryzen AI NUC](/tutorials/ryzen-ai-deployment/) for AMD NPU acceleration
- [Check constraints](/tutorials/constraint-checking/) before committing to a platform
- See the [hardware catalog](/catalog/hardware/) for Coral Edge TPU specifications
