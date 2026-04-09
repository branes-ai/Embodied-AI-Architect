# Session: Architecture Benchmark Harness and Runnable Operators

**Date**: 2026-01-03
**Focus**: Phase 5 of Ryzen AI NUC Demo Plan - Benchmark Harness Implementation

## Summary

Completed Phase 5 of the Software Architecture Catalog roadmap by implementing a comprehensive benchmark harness for embodied AI architectures. This enables end-to-end performance measurement of complete perception-to-action pipelines with per-operator timing, power monitoring, and requirement validation.

## Work Completed

### 1. Runnable Operators Package

Created `src/embodied_ai_architect/operators/` with composable operators that can be configured for different hardware targets (CPU, GPU, NPU):

**Base Infrastructure:**
- `Operator` ABC with `setup()`, `process()`, `teardown()`, `benchmark()` interface
- `OperatorResult` and `OperatorConfig` dataclasses
- Dynamic operator registry mapping embodied-schemas catalog IDs to implementations

**Perception Operators:**
- `YOLOv8ONNX` - Object detection with ONNX Runtime (supports CPU, GPU, NPU execution providers)
- `ImagePreprocessor` - Resize, normalize, letterbox with optional CUDA acceleration
- `ByteTrack` - Multi-object tracking with Kalman filtering
- `SceneGraphManager` - 3D scene graph from 2D tracks with depth estimation

**State Estimation Operators:**
- `EKF6DOF` - Extended Kalman Filter for 6-DOF pose estimation
- `TrajectoryPredictor` - Predict future object positions using constant velocity model
- `CollisionDetector` - Detect collision risks from trajectory predictions

**Control Operators:**
- `PIDController` - Multi-axis PID control with anti-windup
- `PathFollower` - Pure pursuit path following
- `PathPlannerAStar` - A* path planning on occupancy grids

### 2. Benchmark Harness

Created `src/embodied_ai_architect/benchmark/` package:

**ArchitectureRunner (`runner.py`):**
- Loads architectures from embodied-schemas Registry
- Instantiates and configures operators based on architecture definition
- Executes dataflow graph with per-operator timing
- Supports architecture variants for configuration overrides
- Collects comprehensive statistics: mean, std, min, max, p95

**PowerMonitor (`power.py`):**
- Intel RAPL backend via sysfs (`/sys/class/powercap/intel-rapl`)
- AMD SMU backend via `ryzen_monitor` tool
- Background thread sampling at configurable interval
- Computes mean, peak, min power and total energy

**ArchitectureBenchmarkResult:**
- Per-operator timing with `OperatorTiming` dataclass
- Total pipeline timing with throughput calculation
- Power metrics (optional)
- Automatic requirement validation against architecture specs
- Serialization to dict/JSON for storage

### 3. CLI Commands

Extended `cli/commands/benchmark.py`:

```bash
# List available architectures
branes benchmark arch-list

# Run architecture benchmark
branes benchmark arch drone_perception_v1 --iterations 100 --warmup 10

# Use variant and save results
branes benchmark arch drone_perception_v1 --variant edge -o results.json

# JSON output
embodied-ai --json benchmark arch pick_and_place_v1
```

### 4. Documentation

Created comprehensive documentation in `docs/benchmark-harness.md`:
- Purpose and use cases
- Python API examples
- CLI usage
- Example code for common workflows
- Operator reference table
- Result schema documentation

## Test Results

All 3 reference architectures benchmark successfully:

| Architecture | Operators | Measured FPS | Target FPS | Status |
|-------------|-----------|--------------|------------|--------|
| pick_and_place_v1 | 3 | 24.4 | 10 | ✓ PASS |
| drone_perception_v1 | 6 | 15.6 | 30 | Below target |
| simple_adas_v1 | 4 | - | 20 | Functional |

**Per-Operator Breakdown (pick_and_place_v1):**
```
detector (yolo_detector_s):  42.79ms (98.5%)
planner (path_planner_astar):  0.60ms (1.4%)
controller (pid_controller):   0.06ms (0.1%)
```

## Files Created/Modified

### New Files
- `src/embodied_ai_architect/operators/base.py`
- `src/embodied_ai_architect/operators/__init__.py`
- `src/embodied_ai_architect/operators/perception/yolo_onnx.py`
- `src/embodied_ai_architect/operators/perception/preprocessor.py`
- `src/embodied_ai_architect/operators/perception/bytetrack.py`
- `src/embodied_ai_architect/operators/perception/scene_graph.py`
- `src/embodied_ai_architect/operators/perception/__init__.py`
- `src/embodied_ai_architect/operators/state_estimation/ekf_6dof.py`
- `src/embodied_ai_architect/operators/state_estimation/trajectory_predictor.py`
- `src/embodied_ai_architect/operators/state_estimation/collision_detector.py`
- `src/embodied_ai_architect/operators/state_estimation/__init__.py`
- `src/embodied_ai_architect/operators/control/pid_controller.py`
- `src/embodied_ai_architect/operators/control/path_follower.py`
- `src/embodied_ai_architect/operators/control/path_planner.py`
- `src/embodied_ai_architect/operators/control/__init__.py`
- `src/embodied_ai_architect/benchmark/__init__.py`
- `src/embodied_ai_architect/benchmark/runner.py`
- `src/embodied_ai_architect/benchmark/power.py`
- `docs/benchmark-harness.md`

### Modified Files
- `src/embodied_ai_architect/cli/commands/benchmark.py` - Added `arch` and `arch-list` commands

## Operator Registry

14 operators mapped to embodied-schemas catalog IDs:

```python
OPERATOR_REGISTRY = {
    # Perception
    "yolo_detector_n": YOLOv8ONNX,
    "yolo_detector_s": YOLOv8ONNX,
    "yolo_detector_m": YOLOv8ONNX,
    "yolo_detector_l": YOLOv8ONNX,
    "yolo_detector_x": YOLOv8ONNX,
    "image_preprocessor": ImagePreprocessor,
    "bytetrack": ByteTrack,
    "scene_graph_manager": SceneGraphManager,
    # State estimation
    "ekf_6dof": EKF6DOF,
    "trajectory_predictor": TrajectoryPredictor,
    "collision_detector": CollisionDetector,
    # Control
    "pid_controller": PIDController,
    "trajectory_follower": PathFollower,
    "path_planner_astar": PathPlannerAStar,
}
```

## Architecture Integration

The benchmark harness integrates with embodied-schemas architectures:

```yaml
# Example: drone_perception_v1
operators:
  - id: preprocess
    operator_id: image_preprocessor
    rate_hz: 30
    execution_target: gpu
  - id: detector
    operator_id: yolo_detector_n
    rate_hz: 30
    execution_target: gpu
  - id: tracker
    operator_id: bytetrack
  # ...

dataflow:
  - source_op: preprocess
    source_port: processed
    target_op: detector
    target_port: image
  # ...

end_to_end_latency_ms: 33.3
min_throughput_fps: 30
power_budget_w: 15
```

## Next Steps

1. **GPU/NPU Execution**: Test operators on actual GPU and NPU hardware (Ryzen AI NUC)
2. **Power Monitoring**: Verify RAPL/AMD SMU backends on target hardware
3. **Optimization**: Profile and optimize bottleneck operators
4. **Additional Operators**: Add depth estimation, stereo matching operators
5. **Benchmark Database**: Store results for regression tracking

## Technical Notes

### ONNX Runtime Providers
The YOLOv8ONNX operator supports multiple execution providers:
```python
PROVIDER_MAP = {
    "cpu": ["CPUExecutionProvider"],
    "gpu": ["CUDAExecutionProvider", "ROCMExecutionProvider", "CPUExecutionProvider"],
    "npu": ["RyzenAIExecutionProvider", "QNNExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider"],
}
```

### Flexible Input Handling
Operators handle flexible inputs for dataflow compatibility:
- PathPlannerAStar accepts detections as goal input
- PIDController extracts setpoints from path waypoints
- CollisionDetector works with trajectory predictions or object lists

### Power Monitoring Backends
- **RAPL**: Reads `/sys/class/powercap/intel-rapl/*/energy_uj`, computes power from energy delta
- **AMD SMU**: Calls `ryzen_monitor -p` and parses package power
- **Stub**: Returns None when no backend available (graceful degradation)
