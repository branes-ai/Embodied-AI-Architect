---
title: Codebase Analysis
description: Analyze full C++, Rust, and Python applications for hardware assessment.
---

Codebase analysis extends Branes beyond individual models to analyze complete applications — ML inference, signal processing, control loops, and sensor fusion — and map the full workload to hardware.

## Quick Start

```bash
# Scan a project (no API key needed)
branes codebase scan /path/to/my/app

# Full LLM analysis (needs ANTHROPIC_API_KEY)
branes codebase analyze /path/to/my/app

# End-to-end hardware assessment
branes codebase assess /path/to/my/app --hardware jetson_orin --power-budget 15
```

## Three-Stage Pipeline

### 1. Scan (static, fast)

Walks the directory tree and detects languages, build systems, ML model files, and dependencies. No LLM needed.

```bash
branes codebase scan /path/to/drone_app
```

Supports: CMake, Cargo, pip/poetry, Make. Finds `.onnx`, `.pt`, `.tflite`, `.safetensors` model files.

### 2. Analyze (LLM, 4-pass)

Reads source files through focused LLM passes to extract compute kernels:

| Pass | Purpose |
|------|---------|
| Build & Config | Identify project structure and frameworks |
| Entry Points | Map pipeline stages and execution model |
| Compute Kernels | Extract ops estimates, data types, parallelism |
| Synthesis | Combine into analysis result with dataflow graph |

```bash
branes codebase analyze /path/to/drone_app
```

### 3. Assess (PPA pipeline)

Converts kernels to the workload profile format and runs through the existing hardware explorer:

```bash
branes codebase assess /path/to/drone_app \
    --hardware jetson_orin,custom_kpu \
    --power-budget 15 \
    --latency-target 33
```

## Kernel Types

The analyzer classifies code into kernel types that map to hardware operators:

| Kernel Type | Example | Maps To |
|------------|---------|---------|
| `ml_inference` | YOLOv8 detector | convolution, matrix_multiply |
| `signal_processing` | FFT filter | fft, filtering |
| `control_loop` | PID controller | matrix_multiply, accumulate |
| `sensor_fusion` | Kalman filter | matrix_multiply, accumulate |
| `image_processing` | Camera preprocess | convolution, resize |

## Interactive Chat

```
You: Scan the project at ~/drone_perception
Architect: Found C++ CMake project: 12 files, 3,450 lines...

You: Analyze this codebase
Architect: Identified 4 compute kernels: yolo_inference, preprocess, tracker, pid_control...

You: Assess on Jetson Orin with 10W budget
Architect: [runs full pipeline]...
```

## API Usage

```python
from embodied_ai_architect.codebase import CodebaseScanner, CodebaseConverter
from embodied_ai_architect.codebase.analyzer import CodeAnalyzer
from embodied_ai_architect.llm.client import LLMClient

scanner = CodebaseScanner()
scan = scanner.scan("/path/to/app")

llm = LLMClient()
analyzer = CodeAnalyzer(llm)
analysis = analyzer.analyze(scan, "/path/to/app")

converter = CodebaseConverter()
profile = converter.to_workload_profile(analysis)
# profile is compatible with workload_analyzer() and hw_explorer()
```

## Supported Projects

| Type | Build System | Example |
|------|-------------|---------|
| ML app | pip/poetry, CMake | PyTorch inference pipeline |
| Embedded C++ | CMake, Make | Drone perception + control |
| Embedded Rust | Cargo | Motor controller with sensor fusion |
| Hybrid | CMake + pip | Robot with YOLO + PID |

## Next Steps

- [Understand workload profiles](/features/model-analysis/) for deeper model-level analysis
- [Check hardware fit](/features/hardware-selection/) for specific targets
- [Run roofline analysis](/features/roofline-analysis/) on individual operators
- See the [full guide](/../../docs/codebase-analysis-guide) for methodology details
