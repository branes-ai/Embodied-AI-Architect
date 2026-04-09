# Benchmark Harness

The benchmark harness provides infrastructure for measuring the performance of complete embodied AI software architectures. It executes dataflow pipelines with per-operator timing, power measurement, and requirement validation.

## Purpose

The benchmark harness addresses three core needs in embodied AI system development:

1. **End-to-End Performance Measurement** - Measure total pipeline latency and throughput for complete perception-to-action pipelines, not just individual operators.

2. **Per-Operator Profiling** - Identify bottlenecks by measuring each operator's contribution to total latency with statistical rigor (mean, std, min, max, p95).

3. **Requirement Validation** - Automatically check whether an architecture meets its specified latency, throughput, and power constraints.

### When to Use

- **Architecture Comparison**: Compare different operator configurations (e.g., YOLOv8n vs YOLOv8s)
- **Hardware Evaluation**: Benchmark the same architecture on different execution targets (CPU, GPU, NPU)
- **Regression Testing**: Verify performance hasn't degraded after code changes
- **System Sizing**: Determine if target hardware meets application requirements

## Components

### ArchitectureRunner

Loads and executes complete software architectures from `embodied-schemas`:

```python
from embodied_ai_architect.benchmark import ArchitectureRunner

# Load from registry
runner = ArchitectureRunner.from_registry(
    "drone_perception_v1",
    variant_id="edge",  # Optional variant
    hardware_id="jetson-orin-nano"
)

# Or from an architecture object
from embodied_schemas import Registry
registry = Registry.load()
arch = registry.architectures["pick_and_place_v1"]
runner = ArchitectureRunner(arch)
```

### PowerMonitor

Measures power consumption during benchmarks:

```python
from embodied_ai_architect.benchmark import get_power_monitor

# Auto-detect available backend (RAPL, AMD SMU, or stub)
power_monitor = get_power_monitor()

# Or specify backend
power_monitor = get_power_monitor(backend="rapl", interval_ms=100)
```

Supported backends:
- **RAPL**: Intel Running Average Power Limit (Linux, requires read access to `/sys/class/powercap/intel-rapl`)
- **AMD SMU**: AMD System Management Unit (requires `ryzen_monitor` tool)
- **External**: Placeholder for external power meters
- **Stub**: Returns no data when no backend is available

## Usage

### Python API

#### Basic Benchmark

```python
import numpy as np
from embodied_ai_architect.benchmark import ArchitectureRunner, get_power_monitor

# Load architecture
runner = ArchitectureRunner.from_registry("pick_and_place_v1")
runner.load_operators()

# Create sample inputs (image for first operator)
sample_inputs = {
    "detector": {"image": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)}
}

# Run benchmark
result = runner.benchmark(
    sample_inputs,
    iterations=100,
    warmup=10,
    power_monitor=get_power_monitor()
)

# Print summary
runner.print_summary(result)

# Access detailed results
print(f"Mean latency: {result.timing.mean_total_ms:.2f} ms")
print(f"Throughput: {result.timing.throughput_fps:.1f} fps")

for op_id, timing in result.timing.operator_timings.items():
    print(f"  {op_id}: {timing.mean_ms:.2f} ms (p95: {timing.p95_ms:.2f} ms)")

# Clean up
runner.teardown()
```

#### Single Pipeline Pass

For debugging or single-shot execution:

```python
# Run one pass and get outputs
outputs, timings = runner.run_pipeline(sample_inputs, collect_timing=True)

print("Operator outputs:")
for op_id, output in outputs.items():
    print(f"  {op_id}: {list(output.keys())}")

print("\nTimings:")
for op_id, ms in timings.items():
    print(f"  {op_id}: {ms:.2f} ms")
```

#### Export Results

```python
# Convert to dictionary for serialization
result_dict = result.to_dict()

# Save to JSON
import json
with open("benchmark_results.json", "w") as f:
    json.dump(result_dict, f, indent=2)
```

### CLI

#### List Available Architectures

```bash
branes benchmark arch-list
```

Output:
```
                            Available Architectures
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┓
┃ ID              ┃ Name            ┃ Platform    ┃ Operators ┃ Variants       ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━┩
│ drone_percepti… │ Drone Obstacle  │ drone       │ 6         │ edge, accuracy │
│                 │ Avoidance       │             │           │                │
│ simple_adas_v1  │ Simple ADAS     │ vehicle     │ 4         │ edge, highway  │
│ pick_and_place… │ Pick and Place  │ manipulator │ 3         │ fast, accurate │
└─────────────────┴─────────────────┴─────────────┴───────────┴────────────────┘
```

#### Run Architecture Benchmark

```bash
# Basic benchmark
branes benchmark arch drone_perception_v1

# With options
branes benchmark arch pick_and_place_v1 \
    --iterations 100 \
    --warmup 10 \
    --output results.json

# Use a variant
branes benchmark arch drone_perception_v1 --variant edge

# Disable power monitoring
branes benchmark arch pick_and_place_v1 --no-power
```

#### JSON Output

```bash
# Get results as JSON
embodied-ai --json benchmark arch pick_and_place_v1

# List architectures as JSON
embodied-ai --json benchmark arch-list
```

## Examples

### Example 1: Compare YOLO Variants

```python
import numpy as np
from embodied_schemas import Registry
from embodied_ai_architect.benchmark import ArchitectureRunner

registry = Registry.load()
base_arch = registry.architectures["drone_perception_v1"]

results = {}

for variant_id in ["drone_perception_v1_edge", "drone_perception_v1_accuracy"]:
    runner = ArchitectureRunner.from_registry("drone_perception_v1", variant_id=variant_id)
    runner.load_operators()

    sample_inputs = {
        "preprocess": {"image": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)}
    }

    result = runner.benchmark(sample_inputs, iterations=50, warmup=10)
    results[variant_id] = {
        "latency_ms": result.timing.mean_total_ms,
        "throughput_fps": result.timing.throughput_fps,
    }
    runner.teardown()

print("Variant Comparison:")
for variant, metrics in results.items():
    print(f"  {variant}: {metrics['latency_ms']:.1f}ms ({metrics['throughput_fps']:.1f} fps)")
```

### Example 2: Profile Operator Bottlenecks

```python
from embodied_ai_architect.benchmark import ArchitectureRunner
import numpy as np

runner = ArchitectureRunner.from_registry("drone_perception_v1")
runner.load_operators()

sample_inputs = {
    "preprocess": {"image": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)}
}

result = runner.benchmark(sample_inputs, iterations=100, warmup=10)

# Find bottleneck
total_ms = result.timing.mean_total_ms
print(f"Total: {total_ms:.2f} ms\n")
print("Operator breakdown:")

for op_id, timing in sorted(
    result.timing.operator_timings.items(),
    key=lambda x: x[1].mean_ms,
    reverse=True
):
    pct = (timing.mean_ms / total_ms) * 100
    print(f"  {op_id:25} {timing.mean_ms:8.2f} ms ({pct:5.1f}%)")

runner.teardown()
```

### Example 3: Requirement Validation

```python
from embodied_ai_architect.benchmark import ArchitectureRunner
import numpy as np

runner = ArchitectureRunner.from_registry("drone_perception_v1")
runner.load_operators()

sample_inputs = {
    "preprocess": {"image": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)}
}

result = runner.benchmark(sample_inputs, iterations=100, warmup=10)

# Check requirements
arch = runner.architecture
checks = []

if arch.end_to_end_latency_ms:
    meets = result.timing.mean_total_ms <= arch.end_to_end_latency_ms
    checks.append(("Latency", meets, result.timing.mean_total_ms, arch.end_to_end_latency_ms))

if arch.min_throughput_fps:
    meets = result.timing.throughput_fps >= arch.min_throughput_fps
    checks.append(("Throughput", meets, result.timing.throughput_fps, arch.min_throughput_fps))

if arch.power_budget_w and result.mean_power_w:
    meets = result.mean_power_w <= arch.power_budget_w
    checks.append(("Power", meets, result.mean_power_w, arch.power_budget_w))

print("Requirement Validation:")
all_pass = True
for name, meets, actual, target in checks:
    status = "PASS" if meets else "FAIL"
    all_pass = all_pass and meets
    print(f"  [{status}] {name}: {actual:.1f} (target: {target})")

print(f"\nOverall: {'PASS' if all_pass else 'FAIL'}")
runner.teardown()
```

## Available Operators

The benchmark harness includes these runnable operators:

| Operator ID | Class | Category | Description |
|-------------|-------|----------|-------------|
| `yolo_detector_[n/s/m/l/x]` | YOLOv8ONNX | Perception | Object detection with ONNX Runtime |
| `image_preprocessor` | ImagePreprocessor | Perception | Resize, normalize, letterbox |
| `bytetrack` | ByteTrack | Perception | Multi-object tracking |
| `scene_graph_manager` | SceneGraphManager | Perception | 3D scene graph from tracks |
| `ekf_6dof` | EKF6DOF | State Estimation | 6-DOF pose estimation |
| `trajectory_predictor` | TrajectoryPredictor | State Estimation | Predict object trajectories |
| `collision_detector` | CollisionDetector | State Estimation | Detect collision risks |
| `pid_controller` | PIDController | Control | Multi-axis PID control |
| `trajectory_follower` | PathFollower | Control | Pure pursuit path following |
| `path_planner_astar` | PathPlannerAStar | Control | A* path planning |

## Result Schema

The `ArchitectureBenchmarkResult` contains:

```python
@dataclass
class ArchitectureBenchmarkResult:
    architecture_id: str
    hardware_id: str
    variant_id: str | None

    # Timing
    timing: PipelineTiming  # Contains total and per-operator timing

    # Power (optional)
    mean_power_w: float | None
    peak_power_w: float | None

    # Memory
    peak_memory_mb: float | None

    # Configuration
    iterations: int
    warmup_iterations: int

    # Metadata
    timestamp: str | None
    hardware_fingerprint: str | None
    software_fingerprint: str | None
```

## See Also

- [Runnable Operators](runnable-operators.md) - Detailed operator documentation
- [Ryzen AI NUC Benchmark Procedures](ryzen-ai-nuc-benchmark-procedures.md) - Hardware-specific benchmarking
- [Software Architecture Catalog](../embodied-schemas/docs/architectures.md) - Architecture definitions
