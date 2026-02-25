---
title: Deploying on AMD Ryzen AI NUCs
description: Step-by-step guide to deploying models on AMD Ryzen AI NUCs using the XDNA NPU, Radeon iGPU, and CPU.
---

This tutorial walks you through deploying inference workloads on AMD Ryzen AI NUC-class devices — using the XDNA NPU for efficient INT8 inference, the Radeon 780M iGPU for GPU-accelerated workloads, and the Zen 4 CPU for everything else.

## Overview

You'll learn how to:
- Understand the Ryzen AI compute hierarchy (NPU, iGPU, CPU)
- Deploy a detection model to the XDNA NPU via ONNX Runtime
- Quantize models with AMD Quark for NPU execution
- Benchmark across all three compute units
- Build a multi-accelerator pipeline (NPU + CPU)

## Target Hardware

| Spec | Ryzen 7 8845HS | Ryzen 9 8945HS |
|------|----------------|----------------|
| CPU | 8C/16T Zen 4, 3.8-5.1 GHz | 8C/16T Zen 4, 4.0-5.2 GHz |
| iGPU | Radeon 780M (12 CUs, RDNA 3) | Radeon 780M (12 CUs, RDNA 3) |
| NPU | XDNA, 16 TOPS INT8 | XDNA, 16 TOPS INT8 |
| Memory | DDR5-5600, dual-channel, up to 64 GB | DDR5-5600, dual-channel, up to 64 GB |
| TDP | 15-45W (configurable) | 15-54W (configurable) |

NUC examples: Beelink SER8, ASUS NUC 14 Pro AI, Geekom A8, Minisforum UM890 Pro.

## Prerequisites

### Software Stack

```bash
# Ubuntu 22.04 or 24.04
# Python 3.10+

# Install Embodied AI Architect with OpenVINO extras
pip install -e ".[openvino]"

# Install AMD Quark for NPU quantization
pip install quark

# Install ONNX Runtime with Vitis AI EP (from Ryzen AI Software 1.7.0)
# Download the wheel from AMD's Ryzen AI Software page
pip install onnxruntime-vitisai-*.whl
```

### NPU Driver

```bash
# Linux 6.14+ has the amdxdna driver built-in
# For older kernels, install from: https://github.com/amd/xdna-driver

# Verify NPU is detected
ls /dev/accel/accel*

# Verify ONNX Runtime providers
python3 -c "import onnxruntime; print(onnxruntime.get_available_providers())"
# Should include: 'VitisAIExecutionProvider'
```

## 1. Understand the Compute Hierarchy

The Ryzen AI NUC has three compute units, each suited to different workloads:

| Compute | Strength | Use For |
|---------|----------|---------|
| **XDNA NPU** (16 TOPS) | Efficient INT8 inference | Detection, classification, segmentation |
| **Radeon 780M iGPU** (12 CUs) | Parallel FP16/FP32 compute | Image preprocessing, models that don't quantize well |
| **Zen 4 CPU** (8C/16T) | General purpose, low latency | Tracking, control loops, postprocessing |

A typical perception pipeline splits across all three:

```
Camera → [iGPU: preprocess] → [NPU: detect] → [CPU: track + control]
```

## 2. Export and Quantize for NPU

The XDNA NPU requires INT8 quantized ONNX models. The workflow is:

### Export to ONNX

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.export(format="onnx", imgsz=640, simplify=True)
```

### Quantize with AMD Quark

Quark performs post-training quantization with calibration:

```python
from quark.onnx import ModelQuantizer, QuantizationConfig
import numpy as np

# Prepare calibration data (100-500 representative images)
def calibration_reader():
    for img_path in calibration_images[:200]:
        img = preprocess(img_path)  # your preprocessing
        yield {"images": img.astype(np.float32)}

# Configure INT8 quantization for NPU
config = QuantizationConfig(
    quant_format="QDQ",          # Quantize-Dequantize format for Vitis AI
    calibrate_method="MinMax",
    activation_type="uint8",
    weight_type="int8",
)

quantizer = ModelQuantizer(config)
quantizer.quantize_model(
    input_model_path="yolov8n.onnx",
    output_model_path="yolov8n_int8.onnx",
    calibration_data_reader=calibration_reader(),
)
```

## 3. Deploy to the NPU

Run inference using ONNX Runtime with the Vitis AI Execution Provider:

```python
import onnxruntime as ort
import numpy as np

# Create session targeting the NPU
session = ort.InferenceSession(
    "yolov8n_int8.onnx",
    providers=["VitisAIExecutionProvider"],
    provider_options=[{
        "config_file": "vaip_config.json",
        "cacheDir": "./npu_cache",
        "cacheKey": "yolov8n",
    }],
)

# First run compiles the model for NPU (may take a few minutes)
# Subsequent runs load from cache (fast)
input_data = np.random.rand(1, 3, 640, 640).astype(np.float32)
result = session.run(None, {"images": input_data})
```

**What happens under the hood:**
1. The Vitis AI EP **partitions** the ONNX graph — NPU-supported ops run on the NPU, the rest fall back to CPU
2. NPU subgraphs are **compiled** into micro-coded executables for the XDNA array
3. Compiled models are **cached** in `cacheDir` — first run is slow, subsequent runs are fast

## 4. Benchmark All Three Compute Units

Compare the same model across NPU, iGPU, and CPU:

```python
import time

providers_to_test = {
    "NPU": (["VitisAIExecutionProvider"], [{"config_file": "vaip_config.json"}]),
    "CPU": (["CPUExecutionProvider"], [{}]),
}

for name, (providers, options) in providers_to_test.items():
    session = ort.InferenceSession(
        "yolov8n_int8.onnx",
        providers=providers,
        provider_options=options,
    )

    # Warmup
    for _ in range(10):
        session.run(None, {"images": input_data})

    # Benchmark
    timings = []
    for _ in range(100):
        start = time.perf_counter()
        session.run(None, {"images": input_data})
        timings.append((time.perf_counter() - start) * 1000)

    import statistics
    print(f"{name}: {statistics.median(timings):.1f}ms median, "
          f"{statistics.mean(timings):.1f}ms mean")
```

**Expected Results (YOLOv8n, 640x640, INT8, 45W TDP):**

| Compute | Median Latency | Throughput | Power Draw |
|---------|---------------|------------|------------|
| XDNA NPU | 8.2ms | 122 FPS | ~5W (NPU only) |
| CPU (8 threads) | 32.1ms | 31 FPS | ~28W |
| Radeon 780M (FP16) | 12.4ms | 80 FPS | ~15W |

The NPU delivers the best efficiency at 24 inf/W, while the iGPU provides a good middle ground for models that need FP16 precision.

## 5. Build a Multi-Accelerator Pipeline

For a real perception pipeline, split workloads across accelerators:

```python
import onnxruntime as ort
import numpy as np
import time

class RyzenAIPipeline:
    """Drone perception pipeline using NPU + CPU."""

    def __init__(self):
        # Detection on NPU (INT8, highest efficiency)
        self.detector = ort.InferenceSession(
            "yolov8n_int8.onnx",
            providers=["VitisAIExecutionProvider"],
            provider_options=[{"config_file": "vaip_config.json"}],
        )

        # Tracking on CPU (low-latency sequential operations)
        self.tracker = ByteTracker()  # your tracker implementation

    def process_frame(self, frame: np.ndarray) -> dict:
        # Preprocess (CPU — fast for single images)
        input_tensor = self.preprocess(frame)

        # Detect on NPU
        detections = self.detector.run(None, {"images": input_tensor})

        # Track on CPU
        tracks = self.tracker.update(detections)

        return {"detections": detections, "tracks": tracks}

    def preprocess(self, frame):
        resized = cv2.resize(frame, (640, 640))
        return resized.transpose(2, 0, 1)[np.newaxis].astype(np.float32) / 255.0
```

**Pipeline Latency Breakdown:**

| Stage | Accelerator | Latency |
|-------|------------|---------|
| Preprocess | CPU | 0.8ms |
| Detection | NPU | 8.2ms |
| Postprocess (NMS) | CPU | 0.4ms |
| Tracking | CPU | 2.1ms |
| **Total** | | **11.5ms (87 FPS)** |

## 6. TDP Mode Configuration

The NUC's TDP mode affects all three compute units. Choose based on your power budget:

```bash
# Check current TDP mode
cat /sys/class/powercap/intel-rapl:0/constraint_0_power_limit_uw

# Common TDP configurations:
# 15W — quiet/battery mode, NPU still runs at full speed
# 28W — balanced mode
# 45W — performance mode, full CPU boost
```

| TDP Mode | NPU Latency | CPU Latency | iGPU Latency | System Power |
|----------|-------------|-------------|--------------|-------------|
| 15W | 8.5ms | 52ms | 18ms | 12-15W |
| 28W | 8.3ms | 38ms | 14ms | 22-28W |
| 45W | 8.2ms | 32ms | 12ms | 35-45W |

**Key insight:** The NPU latency barely changes across TDP modes — it has its own power domain. This makes it ideal for power-constrained deployments where you want to throttle the CPU/GPU but keep inference fast.

## 7. Power Monitoring

Use AMD SMU for real power measurement during benchmarks:

```bash
# Install AMD power monitoring
sudo apt install ryzen_monitor  # or build from source

# Monitor during inference
sudo ryzen_monitor --show-power
```

Or via Branes:

```python
from embodied_ai_architect.benchmark.power import AMDSMUMonitor

monitor = AMDSMUMonitor()
monitor.start()

# Run your inference workload
for _ in range(1000):
    session.run(None, {"images": input_data})

power = monitor.stop()
print(f"Average power: {power['mean_watts']:.1f}W")
print(f"Peak power: {power['max_watts']:.1f}W")
```

## 8. Interactive Chat

```
You: Deploy YOLOv8n on my Ryzen 7 8845HS NUC for drone perception

Architect: The 8845HS has three compute units. For YOLOv8n:

  NPU (XDNA, 16 TOPS):  8.2ms, ~5W  ← recommended for detection
  iGPU (780M):           12.4ms, ~15W
  CPU (8C Zen 4):        32.1ms, ~28W

  I'll quantize with AMD Quark for the NPU. Need calibration images.

You: What about the full pipeline with tracking?

Architect: Split across accelerators:
  Preprocess (CPU): 0.8ms
  Detection (NPU):  8.2ms
  NMS (CPU):        0.4ms
  Tracking (CPU):   2.1ms
  Total: 11.5ms (87 FPS) at ~18W system power

  The NPU handles the heavy lifting at 5W while the CPU does
  lightweight tracking. Well under your 33ms target.

You: How does this compare to Jetson Orin?

Architect: Similar latency, different power profiles:
  Ryzen AI NUC: 11.5ms total, 18W system → 4.8 inf/W
  Orin Nano:    29.6ms total, 12W system → 2.8 inf/W

  The NUC is faster but draws more total power. The Orin is
  more power-efficient as a dedicated edge device. The NUC
  also gives you a full x86 desktop for development.
```

## XDNA 2: What's Next

Strix Point processors (Ryzen AI 9 HX 370) bring XDNA 2 with major improvements:

| Spec | XDNA 1 (Hawk Point) | XDNA 2 (Strix Point) |
|------|---------------------|----------------------|
| INT8 TOPS | 16 | 50 |
| Block FP16 | No | Yes |
| AI Engine tiles | 20 | 32 (3.5x wider) |
| Power efficiency | 1x | 2x |

XDNA 2's **Block FP16** support means many models can run at FP16 accuracy without quantization — eliminating the calibration step entirely. Strix Point mini-PCs are starting to appear and will be supported by the same Vitis AI EP workflow.

## Tips

- **NPU first, CPU second** — always offload detection/classification to the NPU
- **Cache the compiled model** — first NPU compilation takes minutes, cached loads are instant
- **15W TDP mode** barely affects NPU — use it for power-constrained deployments
- **INT8 is required** for XDNA 1 NPU — FP16/FP32 falls back to CPU
- **Check partition ratio** — if >20% of ops fall back to CPU, the model may not benefit from NPU
- **Use AMD Quark over vai_q_onnx** — Quark is the modern replacement with better accuracy
- **Monitor power with AMD SMU** — don't rely on TDP spec, measure actual draw

## Next Steps

- [Deploy to Coral Edge TPU](/tutorials/coral-deployment/) for ultra-low-power applications
- [Deploy with OpenVINO](/tutorials/openvino-deployment/) for Intel CPU/GPU/NPU
- [Explore SoC design tradeoffs](/tutorials/design-space-optimization/) for custom hardware
- See the [benchmark procedures](https://github.com/branes-ai/embodied-ai-architect/blob/main/docs/ryzen-ai-nuc-benchmark-procedures.md) for detailed profiling methodology
