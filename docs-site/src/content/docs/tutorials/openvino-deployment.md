---
title: Deploying with OpenVINO
description: Step-by-step guide to deploying models with OpenVINO on Intel CPUs, GPUs, and NPUs.
---

This tutorial walks you through deploying models with OpenVINO for high-performance inference on Intel and compatible x86 hardware — from ONNX export through NNCF quantization to multi-device deployment.

## Overview

You'll learn how to:
- Deploy a model with OpenVINO in FP16 and INT8 precision
- Use NNCF for INT8 calibration with minimal accuracy loss
- Target different devices (CPU, GPU, NPU, AUTO)
- Validate and benchmark the deployed model
- Choose the right device for your workload

## Prerequisites

- **Embodied AI Architect** installed with OpenVINO extras:
  ```bash
  pip install -e ".[openvino]"
  ```
  This installs: `openvino>=2024.0.0`, `nncf>=2.7.0`, `onnxruntime>=1.16.0`
- **Hardware**: Any Intel CPU (Haswell or newer), Intel GPU, or Intel NPU (Meteor Lake+)
- **Calibration images**: 100-500 representative images for INT8 quantization

## 1. Check Hardware Fit

Start by verifying your model meets constraints on the target:

```bash
# Intel Core Ultra (Meteor Lake) with NPU
branes check-latency yolov8n --hardware intel-core-ultra --target 33ms

# Server with Xeon
branes check-latency yolov8s --hardware xeon-w9-3495x --target 10ms
```

OpenVINO supports a range of Intel hardware:

| Device | Best For | Key Advantage |
|--------|----------|---------------|
| CPU | General inference, compatibility | Available everywhere, AVX-512/AMX acceleration |
| GPU | Parallel workloads, batched inference | Higher throughput than CPU for vision models |
| NPU | Always-on, low-power inference | Ultra-low power, dedicated AI accelerator |
| AUTO | Let OpenVINO decide | Automatically picks best available device |

## 2. Deploy with FP16

FP16 is the easiest starting point — no calibration needed, negligible accuracy loss:

```bash
branes deploy yolov8n.pt \
  --target openvino \
  --precision fp16 \
  --input-shape 1,3,640,640 \
  --output-dir ./openvino_deployment
```

**Expected Output:**

```
Deployment: OpenVINO
────────────────────
Step 1/3: Converting PyTorch → ONNX... done
Step 2/3: ONNX → OpenVINO IR (FP16)... done
Step 3/3: Validation... done

Artifacts:
  ./openvino_deployment/
  ├── model.xml            # OpenVINO IR (graph)
  ├── model.bin            # OpenVINO IR (weights, FP16)
  ├── model.onnx           # Intermediate ONNX
  └── config.json          # Deployment config

Validation (CPU):
  Input shape:  [1, 3, 640, 640]
  Latency:      18.4ms
  Throughput:   54.3 FPS
  Memory:       14.2MB
```

## 3. Deploy with INT8

For maximum performance, use INT8 with NNCF calibration:

```bash
branes deploy yolov8n.pt \
  --target openvino \
  --precision int8 \
  --calibration-data ./calibration_images/ \
  --input-shape 1,3,640,640 \
  --output-dir ./openvino_int8
```

**Expected Output:**

```
Step 1/4: Converting PyTorch → ONNX... done
Step 2/4: ONNX → OpenVINO IR (FP32)... done
Step 3/4: INT8 quantization with NNCF (calibrating on 200 images)... done
Step 4/4: Validation... done

Validation (CPU):
  Latency:      9.2ms  (2.0x faster than FP16)
  Throughput:   108.7 FPS
  Memory:       8.1MB  (43% smaller)
```

NNCF performs post-training quantization using your calibration dataset. It typically loses <0.5% accuracy on detection models.

## 4. Target Different Devices

### CPU (default)

Best compatibility. Uses AVX-512 and AMX instructions on supported Xeons:

```python
import openvino as ov

core = ov.Core()
model = core.read_model("openvino_int8/model.xml")
compiled = core.compile_model(model, "CPU")

result = compiled([input_data])
```

### GPU

Higher throughput for batched or parallel workloads:

```python
compiled = core.compile_model(model, "GPU")
```

Requires Intel GPU with compute support (integrated or Arc discrete).

### NPU

Dedicated AI accelerator on Intel Core Ultra (Meteor Lake, Lunar Lake):

```python
compiled = core.compile_model(model, "NPU")
```

Ultra-low power (~3-5W) but supports a subset of operations. Best for small-to-medium models.

### AUTO

Let OpenVINO pick the best available device:

```python
compiled = core.compile_model(model, "AUTO")
# OpenVINO profiles available devices and selects optimal target
```

## 5. Benchmark

Benchmark the deployed model:

```bash
branes benchmark openvino_int8/model.xml \
  --backend local \
  --iterations 1000 \
  --warmup 100
```

You can also use OpenVINO's built-in benchmark tool for detailed profiling:

```bash
benchmark_app -m openvino_int8/model.xml \
  -d CPU \
  -niter 1000 \
  -hint throughput
```

**Expected Results (YOLOv8n, Core i7-13700, INT8):**

| Mode | Latency | Throughput | Power |
|------|---------|------------|-------|
| Latency-optimized | 9.2ms | 108 FPS | 28W |
| Throughput-optimized | 12.1ms | 245 FPS | 65W |

## 6. Run Inference

Complete inference example with pre/post-processing:

```python
import openvino as ov
import numpy as np
import cv2

# Load model
core = ov.Core()
model = core.read_model("openvino_int8/model.xml")
compiled = core.compile_model(model, "CPU")

# Get I/O info
input_layer = compiled.input(0)
output_layer = compiled.output(0)

# Preprocess
image = cv2.imread("test.jpg")
resized = cv2.resize(image, (640, 640))
input_data = resized.transpose(2, 0, 1)[np.newaxis].astype(np.float32) / 255.0

# Infer
result = compiled([input_data])
output = result[output_layer]

# Postprocess (model-specific)
print(f"Output shape: {output.shape}")
```

## 7. Precision Comparison

Run the same model at all three precisions to understand the tradeoffs:

| Precision | Latency | Throughput | Memory | Accuracy |
|-----------|---------|------------|--------|----------|
| FP32 | 32.1ms | 31 FPS | 25.6MB | 37.3 mAP |
| FP16 | 18.4ms | 54 FPS | 14.2MB | 37.3 mAP |
| INT8 (NNCF) | 9.2ms | 108 FPS | 8.1MB | 36.9 mAP |

FP16 is essentially free (same accuracy, half the memory). INT8 doubles throughput again with <0.5% accuracy loss.

## 8. Interactive Chat

```
You: Deploy YOLOv8n with OpenVINO at INT8 precision

Architect: Deploying YOLOv8n to OpenVINO IR with INT8 quantization...
  Conversion: PyTorch → ONNX → OpenVINO IR
  Quantization: NNCF with 200 calibration images
  Result: 9.2ms latency, 108 FPS on CPU

You: How does it compare to TensorRT on Jetson?

Architect: On equivalent workloads:
  OpenVINO INT8 (Core i7): 9.2ms, 28W → 3.7 inf/W
  TensorRT FP16 (Orin Nano): 25.1ms, 12W → 3.3 inf/W

  OpenVINO on x86 is faster in absolute terms but the Jetson
  is competitive on power efficiency. Choose based on your
  deployment environment.
```

## Tips

- **Start with FP16** — zero configuration, no calibration needed, half the memory
- **INT8 with NNCF** is the production choice — 2x speedup over FP16 with calibration
- **Use AUTO device** when deploying to unknown hardware — OpenVINO picks the best target
- **Throughput vs latency** modes matter — throughput mode uses all cores but adds latency
- **Check available devices** with `ov.Core().available_devices` before hardcoding a target
- **AVX-512/AMX** on Xeon gives significant CPU speedups — check with `lscpu | grep avx`

## Next Steps

- [Deploy to Coral Edge TPU](/tutorials/coral-deployment/) for ultra-low-power applications
- [Deploy on Ryzen AI NUC](/tutorials/ryzen-ai-deployment/) for AMD NPU acceleration
- [Run roofline analysis](/tutorials/roofline-analysis/) to understand bottlenecks
- See the [CLI reference](/reference/cli/) for all deploy command options
