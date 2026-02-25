---
title: Analyzing a Drone Perception Codebase
description: Step-by-step guide to scanning, analyzing, and assessing a complete C++ application for hardware deployment.
---

This tutorial walks you through using codebase analysis to take a complete drone perception application — C++ with YOLO inference, tracking, and PID control — and map its full workload to hardware targets.

## Overview

You'll learn how to:
- Scan a project to detect languages, build system, and ML models
- Run multi-pass LLM analysis to extract compute kernels
- Assess the full workload against hardware targets with power and latency constraints
- Interpret the results and iterate on hardware choices

## Prerequisites

- **Embodied AI Architect** installed (`pip install -e ".[dev]"`)
- **ANTHROPIC_API_KEY** set (for `analyze` and `assess` commands; `scan` works without it)
- A project directory to analyze (we'll use a drone perception app as the example)

## 1. Scan the Project

Start with a fast static scan. This needs no API key and completes in seconds:

```bash
branes codebase scan ~/projects/drone_perception
```

**Expected Output:**

```
Codebase Scan Results
─────────────────────
Project: drone_perception
Path:    /home/user/projects/drone_perception

Languages:
  C++    2,840 lines (68.2%)
  Python   890 lines (21.4%)
  CMake    432 lines (10.4%)

Build System: CMake
  CMakeLists.txt found at root
  Dependencies: OpenCV 4.x, TensorRT, Eigen3

ML Models Found:
  yolov8n.onnx     (25.3 MB)
  tracker.onnx     (4.1 MB)

Dependencies:
  opencv, tensorrt, eigen3, yaml-cpp
```

**What to look for:**
- **Languages** tell you what kind of hardware operators to expect (C++ usually means tight loops, SIMD)
- **ML Models** are automatically picked up — these become `ml_inference` kernels
- **Build System** detection means the analyzer knows how to parse the project structure

## 2. Analyze with LLM

Now run the full 4-pass LLM analysis to extract compute kernels:

```bash
branes codebase analyze ~/projects/drone_perception
```

The analyzer makes four focused passes over the source code:

```
Pass 1/4: Build & Config
  Parsing CMakeLists.txt, package.xml...
  Found: OpenCV 4.8, TensorRT 8.6, Eigen 3.4

Pass 2/4: Entry Points
  Scanning main.cpp, pipeline.cpp...
  Found pipeline stages: capture → preprocess → detect → track → control

Pass 3/4: Compute Kernels
  Analyzing detector.cpp, tracker.cpp, pid_controller.cpp...
  Extracted 5 kernels with ops estimates

Pass 4/4: Synthesis
  Building dataflow graph...
  Analysis complete.
```

**Expected Kernel Output:**

```
Compute Kernels
───────────────
 # | Name              | Type             | Ops Est.   | Data Type | Parallelism
 1 | camera_preprocess | image_processing | 12.4M      | uint8     | pixel-parallel
 2 | yolo_inference    | ml_inference     | 3.2G       | fp16      | tensor cores
 3 | nms_postprocess   | signal_processing| 0.8M       | fp32      | sequential
 4 | bytetrack         | sensor_fusion    | 45.2M      | fp32      | per-object
 5 | pid_control       | control_loop     | 0.02M      | fp64      | sequential

Dataflow:
  camera → preprocess → yolo → nms → tracker → pid_control
  Rate: 30 Hz end-to-end
```

**How to read this:**
- **Ops estimates** show where compute is concentrated — `yolo_inference` at 3.2G dominates
- **Data types** tell you what precision the hardware needs (fp16 for inference, fp64 for control)
- **Parallelism** indicates which operators benefit from wide compute (tensor cores vs sequential)

## 3. Assess on Hardware

Run the end-to-end assessment with power and latency constraints:

```bash
branes codebase assess ~/projects/drone_perception \
  --hardware jetson_orin \
  --power-budget 15 \
  --latency-target 33
```

**Expected Output:**

```
Hardware Assessment: Jetson Orin Nano
─────────────────────────────────────
Power Budget: 15W | Latency Target: 33ms

Kernel Mapping:
 Kernel            | Operator          | Latency  | Power | Verdict
 camera_preprocess | GPU (CUDA cores)  | 1.2ms    | 0.8W  | PASS
 yolo_inference    | GPU (tensor cores)| 25.1ms   | 10.2W | PASS
 nms_postprocess   | CPU (ARM A78)     | 0.4ms    | 0.3W  | PASS
 bytetrack         | CPU (ARM A78)     | 2.8ms    | 0.9W  | PASS
 pid_control       | CPU (ARM A78)     | 0.1ms    | 0.1W  | PASS

Pipeline Summary:
  Total Latency: 29.6ms  (target: 33ms)  PASS  headroom: 10.2%
  Total Power:   12.3W   (budget: 15W)   PASS  headroom: 18.0%

  Bottleneck: yolo_inference (84.8% of latency)
```

## 4. Try Multiple Targets

Compare against several hardware targets at once:

```bash
branes codebase assess ~/projects/drone_perception \
  --hardware jetson_orin,jetson_orin_agx,coral_edge_tpu \
  --power-budget 15 \
  --latency-target 33
```

This shows how the same workload maps to different hardware, making it easy to pick the right platform.

## 5. Interactive Chat Workflow

You can also run the full pipeline conversationally:

```
You: Scan the project at ~/projects/drone_perception

Architect: Found C++ CMake project with 2 ML models:
  - 2,840 lines C++, 890 lines Python
  - Dependencies: OpenCV, TensorRT, Eigen3
  - Models: yolov8n.onnx (25.3 MB), tracker.onnx (4.1 MB)

You: Analyze the codebase

Architect: Identified 5 compute kernels:
  1. camera_preprocess (image_processing, 12.4M ops)
  2. yolo_inference (ml_inference, 3.2G ops)
  3. nms_postprocess (signal_processing, 0.8M ops)
  4. bytetrack (sensor_fusion, 45.2M ops)
  5. pid_control (control_loop, 0.02M ops)

  Pipeline: capture → preprocess → yolo → nms → tracker → control @ 30Hz

You: Assess on Jetson Orin with 10W power budget

Architect: MARGINAL — total power is 12.3W, 23% over your 10W budget.
  The YOLO inference alone draws 10.2W.

  Suggestions:
  1. Switch to YOLOv8-nano INT8: ~6.8W for inference
  2. Increase budget to 15W
  3. Use Coral Edge TPU to offload inference: ~2W for YOLO
```

## 6. Python API

For scripting or integration into CI pipelines:

```python
from embodied_ai_architect.codebase import CodebaseScanner, CodebaseConverter
from embodied_ai_architect.codebase.analyzer import CodeAnalyzer
from embodied_ai_architect.llm.client import LLMClient

# Step 1: Scan
scanner = CodebaseScanner()
scan = scanner.scan("/home/user/projects/drone_perception")
print(f"Languages: {scan.languages}")
print(f"Models found: {[m.name for m in scan.ml_models]}")

# Step 2: Analyze (requires API key)
llm = LLMClient()
analyzer = CodeAnalyzer(llm)
analysis = analyzer.analyze(scan, "/home/user/projects/drone_perception")
print(f"Kernels: {[k.name for k in analysis.kernels]}")

# Step 3: Convert to workload profile
converter = CodebaseConverter()
profile = converter.to_workload_profile(analysis)
# profile is now compatible with the hardware explorer pipeline
```

## Kernel Types Reference

The analyzer maps code patterns to these kernel types:

| Kernel Type | Code Patterns | Hardware Mapping |
|------------|---------------|------------------|
| `ml_inference` | ONNX Runtime, TensorRT, PyTorch forward() | Tensor cores, NPU, TPU |
| `image_processing` | OpenCV resize, cvtColor, warpAffine | GPU CUDA cores, ISP |
| `signal_processing` | FFT, filtering, NMS | CPU SIMD, DSP |
| `sensor_fusion` | Kalman filter, tracker state update | CPU, light GPU |
| `control_loop` | PID, MPC, state machines | CPU (low latency) |

## Tips

- **Start with `scan`** — it's free (no API key) and catches basic issues like missing model files
- **Check the dataflow** — sequential bottlenecks can't be parallelized away by faster hardware
- **Watch ops estimates** — they're approximate, but the relative ratios are informative
- **Compare hardware** — always assess on at least 2 targets to see where you have choices
- **Use in CI** — `scan` can run in CI to catch dependency changes; `assess` can gate deployments

## Next Steps

- [Check individual constraints](/features/constraint-checking/) for specific models
- [Run roofline analysis](/tutorials/roofline-analysis/) to understand bottleneck types
- [Optimize the design space](/tutorials/design-space-optimization/) for custom hardware
- See the [CLI reference](/reference/cli/) for all `codebase` command options
