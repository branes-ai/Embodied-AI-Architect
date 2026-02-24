# Codebase Analysis: Full Application Hardware Assessment

The Codebase Analysis feature lets you bring complete C++, Rust, or Python applications to Branes and get hardware assessment across different targets. Instead of analyzing individual PyTorch models, you analyze an entire application — including ML inference, signal processing, control loops, and sensor fusion — and map it to hardware.

## Why Codebase Analysis?

Traditional model analysis answers: *"Can this YOLOv8 model run on Jetson Orin?"*

Codebase analysis answers: *"Can this entire drone perception + control application — with its YOLO detector, Kalman tracker, PID controller, and IMU fusion — run on Jetson Orin within my power and latency budget?"*

Real embedded applications are more than a single ML model. They combine:
- **ML inference** (object detection, segmentation, depth estimation)
- **Signal processing** (FFT, filtering, sensor conditioning)
- **Control loops** (PID, MPC, trajectory planning)
- **Sensor fusion** (Kalman filters, complementary filters, EKF)
- **Image processing** (resize, color conversion, undistortion)

All of these compete for compute, memory, and power. Codebase analysis maps the full workload to hardware.

## Methodology

### Three-Stage Pipeline

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  1. SCAN     │ ──→ │  2. ANALYZE  │ ──→ │  3. ASSESS   │
│  (static)    │     │  (LLM)       │     │  (PPA)       │
└──────────────┘     └──────────────┘     └──────────────┘
```

#### Stage 1: Static Scan

Fast, no LLM needed. The scanner walks the project directory and extracts:

| What | How |
|------|-----|
| Languages | File extensions (`.cpp`, `.rs`, `.py`, etc.) |
| Build system | Marker files (`CMakeLists.txt`, `Cargo.toml`, `pyproject.toml`) |
| ML models | Model files (`.onnx`, `.pt`, `.tflite`, `.safetensors`) |
| Dependencies | Parsed from build files (`find_package`, `[dependencies]`, etc.) |
| File roles | Entry point detection (`int main()`, `fn main()`, `if __name__`) |
| Line counts | Per-file for prioritizing LLM analysis |

Output: `ScanResult` — a structured inventory of the project.

#### Stage 2: LLM Analysis (4 passes)

The LLM reads source files in focused passes to stay within context limits:

| Pass | Reads | Extracts |
|------|-------|----------|
| **1. Build & Config** | Build files, manifests | Project type, frameworks, dependency graph |
| **2. Entry Points** | main files, orchestration | Pipeline stages, execution model (sequential/multi-rate/event-driven) |
| **3. Compute Kernels** | Implementation files | `ComputeKernel` instances with ops estimates, data types, parallelism |
| **4. Synthesis** | All pass results | `CodebaseAnalysisResult` with dataflow, summary |

Each pass sends a focused prompt with relevant file contents and asks for structured JSON. Files are priority-sorted (entry points first, then by size) to ensure the most important code is analyzed within context limits.

Output: `CodebaseAnalysisResult` — kernels, dataflow, and application summary.

#### Stage 3: Hardware Assessment

The converter maps each `ComputeKernel` to operator types understood by the existing PPA pipeline:

| Kernel Type | Maps To | Example |
|-------------|---------|---------|
| `ml_inference` | convolution, matrix_multiply, activation | YOLOv8 detector |
| `signal_processing` | fft, filtering, accumulate | Audio DSP, radar processing |
| `image_processing` | convolution, resize, color_convert | Camera preprocessing |
| `control_loop` | matrix_multiply, accumulate | PID controller |
| `sensor_fusion` | matrix_multiply, accumulate | Kalman filter, EKF |
| `io_bound` | memory_copy | DMA transfers, I/O |
| `general_compute` | general_purpose | Business logic |

The resulting `workload_profile` feeds directly into the existing `hw_explorer` and `ppa_assessor` pipeline — no modifications to the SoC design flow needed.

## Usage

### CLI

#### Quick scan (no LLM, no API key needed)

```bash
# Scan a C++ drone project
branes codebase scan /path/to/drone_app

# JSON output for scripting
branes --json codebase scan /path/to/my_project
```

Output:
```
Scan complete: drone_app

╭──────────────────── Project Summary ────────────────────╮
│ Languages: cpp                                          │
│ Build system: cmake                                     │
│ Source files: 12                                        │
│ Total lines: 3,450                                      │
│ ML models: 1                                            │
╰─────────────────────────────────────────────────────────╯

ML Model Files:
  models/yolov8n.onnx (onnx, 6.2 MB)

Dependencies: OpenCV, Eigen3, onnxruntime
```

#### Full LLM analysis (requires ANTHROPIC_API_KEY)

```bash
export ANTHROPIC_API_KEY=your-key-here
branes codebase analyze /path/to/drone_app
```

Output includes compute kernels with their types, ops estimates, data types, and parallelism characteristics.

#### End-to-end hardware assessment

```bash
# Assess against specific hardware targets
branes codebase assess /path/to/drone_app \
    --hardware jetson_orin,custom_kpu

# With power and latency constraints
branes codebase assess /path/to/drone_app \
    --power-budget 15 \
    --latency-target 33
```

### Interactive Chat

Start a chat session and use natural language:

```bash
branes chat
```

```
You: Scan the project at /home/user/drone_perception

Architect: [calls scan_project]
I found a C++ CMake project with 12 source files and 3,450 lines.
Languages: C++. Build system: CMake.
Dependencies: OpenCV, Eigen3, onnxruntime.
1 ML model found: models/yolov8n.onnx (6.2 MB)

You: Analyze this codebase for hardware mapping

Architect: [calls analyze_codebase]
I identified 4 compute kernels in the drone perception pipeline:

1. **yolo_inference** (ml_inference) — 8.7 GFLOPS, float32, data_parallel
2. **image_preprocess** (image_processing) — 0.5 GFLOPS, uint8→float32
3. **kalman_tracker** (sensor_fusion) — 0.1 GFLOPS, float64, sequential
4. **pid_controller** (control_loop) — 0.01 GFLOPS, float32, 100Hz

Dataflow: camera → preprocess → yolo → tracker → controller

You: Assess this app on Jetson Orin vs a custom KPU with 10W power budget

Architect: [calls assess_codebase_on_hardware]
...
```

### Python API

```python
from embodied_ai_architect.codebase import CodebaseScanner, CodebaseConverter
from embodied_ai_architect.codebase.analyzer import CodeAnalyzer
from embodied_ai_architect.llm.client import LLMClient
from pathlib import Path

project = Path("/path/to/my/app")

# Stage 1: Scan
scanner = CodebaseScanner()
scan = scanner.scan(project)
print(f"Found {len(scan.source_files)} files, build={scan.build_system}")

# Stage 2: LLM Analysis
llm = LLMClient()
analyzer = CodeAnalyzer(llm)
analysis = analyzer.analyze(scan, project)
print(f"Kernels: {[k.name for k in analysis.kernels]}")

# Stage 3: Convert to workload profile
converter = CodebaseConverter()
profile = converter.to_workload_profile(analysis)
print(f"Total GFLOPS: {profile['total_estimated_gflops']}")
print(f"Dominant op: {profile['dominant_op']}")

# The profile is now compatible with the existing PPA pipeline:
# - workload_analyzer() reads it
# - hw_explorer() scores hardware against it
# - ppa_assessor() produces final assessment
```

Or use the agent wrapper for one-call operation:

```python
from embodied_ai_architect.agents.codebase_analyzer import CodebaseAnalyzerAgent

agent = CodebaseAnalyzerAgent()
result = agent.execute({
    "project_path": "/path/to/app",
    "skip_llm": False,  # set True for scan-only (no API key needed)
})

if result.success:
    profile = result.data["workload_profile"]
    print(f"Workloads: {profile['workload_count']}")
    print(f"GFLOPS: {profile['total_estimated_gflops']}")
```

## Supported Project Types

| Type | Languages | Build System | Example |
|------|-----------|-------------|---------|
| ML-heavy app | Python, C++ | pip/poetry, CMake | PyTorch inference server |
| Embedded perception | C++, CUDA | CMake | Drone obstacle avoidance |
| Embedded control | Rust, C | Cargo, Make | Motor controller with sensor fusion |
| Hybrid ML + control | C++, Python | CMake, pip | Robot with YOLO + PID |
| Signal processing | C++, Python | CMake, pip | Radar/audio DSP pipeline |

## Kernel Type Reference

| Kernel Type | Description | Typical Ops | Hardware Affinity |
|------------|-------------|-------------|-------------------|
| `ml_inference` | Neural network forward pass | 1-100 GFLOPS | GPU, NPU, TPU |
| `signal_processing` | FFT, filtering, spectral analysis | 0.01-1 GFLOPS | DSP, CPU |
| `image_processing` | Resize, warp, color convert | 0.1-2 GFLOPS | GPU, ISP |
| `control_loop` | PID, MPC, trajectory planning | 0.001-0.1 GFLOPS | CPU |
| `sensor_fusion` | Kalman filter, EKF, complementary | 0.01-0.5 GFLOPS | CPU, NPU |
| `io_bound` | DMA, network I/O, file access | minimal | DMA engine |
| `general_compute` | Business logic, decision making | varies | CPU |

## How It Connects to the Existing Pipeline

```
                  Codebase Analysis (NEW)
                  ═══════════════════════
                         │
    scan → analyze → convert
                         │
                    workload_profile
                         │
         ┌───────────────┼───────────────┐
         ↓               ↓               ↓
   workload_analyzer  hw_explorer   ppa_assessor
         │               │               │
         └───────────────┼───────────────┘
                         ↓
                  SoC Design Decision
                  (EXISTING PIPELINE)
```

The converter is the bridge: it maps rich `CodebaseAnalysisResult` data (with kernel types, ops estimates, data types) into the `workload_profile` format that the existing specialists already consume. No changes to the SoC design pipeline were needed.

## Limitations (Phase 1)

- **Static analysis only** — the LLM reads source code but doesn't execute it. Runtime hotspots may differ from static estimates.
- **Ops estimates are heuristic** — without profiling, GFLOPS per kernel are estimated by the LLM based on code structure. Phase 2 will add optional dynamic profiling.
- **Context limits** — very large codebases (>100K LOC) may require multiple analysis sessions. Files are priority-sorted to ensure the most important code is analyzed first.
- **LLM accuracy** — kernel type classification depends on LLM interpretation. Complex or obfuscated code may be misclassified.

## Phase 2 Roadmap

- **Dynamic profiling** (`profiler.py`) — instrument and run the application to capture actual runtime hotspots, feeding measured data back into the workload profile
- **Build system integration** — parse CMake/Cargo build graphs for more accurate dependency tracking
- **Multi-file kernel tracking** — follow function calls across files to build more accurate kernel boundaries
- **Calibration** — compare LLM estimates against profiling results to improve future estimates
