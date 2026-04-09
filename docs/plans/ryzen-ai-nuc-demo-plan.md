# Ryzen AI NUC Embodied AI Demonstration Plan

## Executive Summary

Demonstrate end-to-end agentic design and optimization of an embodied AI system targeting AMD Ryzen AI NUC platforms. The demo showcases the complete workflow: architecture design → analysis → optimization → real execution → performance characterization.

## Target Hardware

### AMD Ryzen 7 AI NUC
- **CPU**: AMD Ryzen 7 (8 cores, Zen 4)
- **GPU**: AMD Radeon integrated (RDNA 3, ~12 CUs)
- **NPU**: AMD XDNA (~10 TOPS INT8)
- **Memory**: 32GB DDR5
- **TDP**: 15-28W configurable
- **Form Factor**: NUC (robot-friendly)

### AMD Ryzen 9 AI NUC
- **CPU**: AMD Ryzen 9 (8 cores, Zen 4, higher clocks)
- **GPU**: AMD Radeon 780M (RDNA 3, ~12 CUs)
- **NPU**: AMD XDNA (~16 TOPS INT8)
- **Memory**: 32GB DDR5
- **TDP**: 35-54W configurable
- **Form Factor**: NUC (robot-friendly)

### Why These Platforms?

1. **Heterogeneous compute**: CPU + GPU + NPU mirrors real robot compute modules
2. **Power envelope**: 15-54W matches autonomous system budgets
3. **Form factor**: NUC size fits robot chassis
4. **Software stack**: ROCm for GPU, ONNX Runtime for NPU, standard Linux

## Target Application: Robot PGN&C Pipeline

### Perception, Guidance, Navigation & Control

```
┌─────────────────────────────────────────────────────────────────┐
│                    PGN&C Architecture                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Camera ──► Preprocess ──► Detect ──► Track ──► Scene Graph    │
│    30Hz        GPU          NPU       CPU          CPU          │
│                                          │                      │
│                                          ▼                      │
│  IMU ────────────────────────────► State Estimator              │
│  100Hz                                  CPU                     │
│                                          │                      │
│                                          ▼                      │
│                                     Path Planner                │
│                                        CPU                      │
│                                          │                      │
│                                          ▼                      │
│                                   Motion Controller             │
│                                      CPU 100Hz                  │
│                                          │                      │
│                                          ▼                      │
│                                      Actuators                  │
└─────────────────────────────────────────────────────────────────┘
```

### Multi-Rate Requirements

| Subsystem | Rate | Latency Budget | Execution Target |
|-----------|------|----------------|------------------|
| Perception | 30 Hz | 33ms | GPU/NPU |
| State Estimation | 100 Hz | 10ms | CPU |
| Planning | 10 Hz | 100ms | CPU |
| Control | 100 Hz | 10ms | CPU |

### End-to-End Requirements

- **Perception-to-Control latency**: < 50ms
- **Power budget**: 25W (Ryzen 7), 45W (Ryzen 9)
- **Memory budget**: 8GB (leaving headroom for OS)

## MVP Scope

### Phase 1: Platform Characterization (Week 1-2)

**Goal**: Create accurate hardware profiles for both NUCs

#### 1.1 Hardware Catalog Entries

Create `ryzen-7-ai-nuc.yaml` and `ryzen-9-ai-nuc.yaml` in embodied-schemas:

```yaml
id: Ryzen-7-AI-NUC
name: AMD Ryzen 7 AI NUC
vendor: AMD
type: edge_soc

compute:
  cpu:
    cores: 8
    threads: 16
    arch: zen4
    base_clock_ghz: 3.8
    boost_clock_ghz: 5.1
    l3_cache_mb: 16
  gpu:
    name: Radeon 780M
    arch: rdna3
    compute_units: 12
    clock_mhz: 2800
    tflops_fp16: 8.1
    memory_shared: true
  npu:
    name: XDNA
    tops_int8: 10
    supported_ops: [conv2d, matmul, attention]

memory:
  type: DDR5
  capacity_gb: 32
  bandwidth_gbps: 76.8

power:
  tdp_w: 28
  configurable_tdp: [15, 20, 28]
  idle_w: 5

interfaces:
  usb: [USB4, USB-C]
  display: [HDMI, DP]
  network: [2.5GbE, WiFi6E]

software:
  os: [Ubuntu 22.04, Ubuntu 24.04]
  gpu_runtime: ROCm
  npu_runtime: [ONNX Runtime, Ryzen AI SDK]
```

#### 1.2 Baseline Benchmarks

Measure actual performance on both platforms:

1. **CPU benchmarks**:
   - Single-thread latency for typical operators
   - Multi-thread scaling
   - Memory bandwidth (STREAM)

2. **GPU benchmarks**:
   - ROCm/HIP inference latency
   - ResNet-50, MobileNet, YOLO variants
   - Batch size sensitivity

3. **NPU benchmarks**:
   - ONNX Runtime + Ryzen AI EP
   - INT8 quantized model performance
   - Supported vs unsupported ops fallback

#### 1.3 Power Measurement Setup

- Use `ryzen_smu` or RAPL for CPU/GPU power
- NPU power via system-level measurement
- Per-component breakdown if possible

### Phase 2: Operator Profiling (Week 2-3)

**Goal**: Profile key operators on both platforms

#### 2.1 Perception Operators

| Operator | Ryzen 7 Target | Ryzen 9 Target | Notes |
|----------|----------------|----------------|-------|
| YOLOv8n | NPU | NPU | INT8 quantized |
| YOLOv8s | GPU | NPU | May need GPU fallback |
| MobileNet-SSD | NPU | NPU | Lightweight option |
| ByteTrack | CPU | CPU | Association logic |
| Preprocessing | GPU | GPU | Resize, normalize |

#### 2.2 State Estimation Operators

| Operator | Target | Notes |
|----------|--------|-------|
| EKF 6-DOF | CPU | Real-time constraints |
| Complementary Filter | CPU | Lightweight |
| Visual-Inertial Odometry | CPU+GPU | If available |

#### 2.3 Control Operators

| Operator | Target | Notes |
|----------|--------|-------|
| PID Controller | CPU | 100Hz requirement |
| MPC (simple) | CPU | 10Hz planning rate |
| Path Follower | CPU | Trajectory tracking |

#### 2.4 Profile Data Format

```yaml
# operators/perception/yolo_detector_n.yaml
perf_profiles:
  - hardware_id: Ryzen-7-AI-NUC
    execution_target: npu
    latency_ms: 12.5
    memory_mb: 180
    power_w: 3.2
    throughput_fps: 80
    conditions: "INT8, batch_size=1, 640x640, ONNX Runtime"

  - hardware_id: Ryzen-9-AI-NUC
    execution_target: npu
    latency_ms: 8.3
    memory_mb: 180
    power_w: 4.1
    throughput_fps: 120
    conditions: "INT8, batch_size=1, 640x640, ONNX Runtime"
```

### Phase 3: Reference Architecture (Week 3-4)

**Goal**: Create optimized PGN&C architecture with variants

#### 3.1 Base Architecture: `pgnc_robot_v1`

```yaml
id: pgnc_robot_v1
name: Robot PGN&C Pipeline
description: Perception-Guidance-Navigation-Control for mobile robots
platform_type: robot

sensors:
  - rgb_camera
  - imu_6dof

operators:
  # Perception pipeline (30 Hz)
  - id: preprocess
    operator_id: image_preprocessor
    rate_hz: 30
    execution_target: gpu

  - id: detector
    operator_id: yolo_detector_n
    rate_hz: 30
    execution_target: npu
    config:
      conf_threshold: 0.3

  - id: tracker
    operator_id: bytetrack
    rate_hz: 30
    execution_target: cpu

  - id: scene
    operator_id: scene_graph_manager
    rate_hz: 30
    execution_target: cpu

  # State estimation (100 Hz)
  - id: state_est
    operator_id: ekf_6dof
    rate_hz: 100
    execution_target: cpu

  # Planning (10 Hz)
  - id: planner
    operator_id: path_planner_astar
    rate_hz: 10
    execution_target: cpu

  # Control (100 Hz)
  - id: controller
    operator_id: pid_controller
    rate_hz: 100
    execution_target: cpu

dataflow:
  - source_op: preprocess
    source_port: processed
    target_op: detector
    target_port: image

  - source_op: detector
    source_port: detections
    target_op: tracker
    target_port: detections

  - source_op: tracker
    source_port: tracks
    target_op: scene
    target_port: tracks

  - source_op: scene
    source_port: obstacles
    target_op: planner
    target_port: obstacles

  - source_op: state_est
    source_port: pose
    target_op: planner
    target_port: current_pose

  - source_op: planner
    source_port: path
    target_op: controller
    target_port: trajectory

end_to_end_latency_ms: 50
min_throughput_fps: 30
power_budget_w: 25

variants:
  - id: pgnc_robot_v1_ryzen7
    name: Ryzen 7 AI Optimized
    description: Optimized for 25W power budget
    target_hardware: [Ryzen-7-AI-NUC]
    operator_overrides:
      detector: yolo_detector_n  # Nano for power
    expected_latency_ms: 45
    expected_power_w: 22

  - id: pgnc_robot_v1_ryzen9
    name: Ryzen 9 AI High-Performance
    description: Higher accuracy with more power
    target_hardware: [Ryzen-9-AI-NUC]
    operator_overrides:
      detector: yolo_detector_s  # Small for accuracy
    expected_latency_ms: 38
    expected_power_w: 35
```

#### 3.2 Architecture Variants

| Variant | Detector | Target Platform | Latency | Power |
|---------|----------|-----------------|---------|-------|
| Base (nano) | YOLOv8n | Both | 45ms | 22W |
| Ryzen 7 | YOLOv8n | Ryzen 7 | 48ms | 20W |
| Ryzen 9 | YOLOv8s | Ryzen 9 | 38ms | 35W |
| Low-power | MobileNet-SSD | Ryzen 7 | 55ms | 15W |

### Phase 4: Real Operator Implementations (Week 4-6)

**Goal**: Implement runnable operators for benchmarking

#### 4.1 Operator Implementation Strategy

```
operators/
├── perception/
│   ├── yolo_onnx.py          # ONNX Runtime inference
│   ├── preprocessor_cv2.py   # OpenCV preprocessing
│   └── bytetrack_impl.py     # Python ByteTrack
├── state_estimation/
│   ├── ekf_numpy.py          # NumPy EKF
│   └── complementary.py      # Simple filter
├── control/
│   ├── pid_controller.py     # Standard PID
│   └── path_follower.py      # Trajectory tracking
└── base.py                   # Operator base class
```

#### 4.2 Operator Base Interface

```python
class Operator(ABC):
    """Base class for runnable operators."""

    @abstractmethod
    def setup(self, config: dict) -> None:
        """Initialize operator with configuration."""
        pass

    @abstractmethod
    def process(self, inputs: dict) -> dict:
        """Process inputs and return outputs."""
        pass

    def benchmark(self, iterations: int = 100) -> BenchmarkResult:
        """Run benchmark and return timing statistics."""
        pass
```

#### 4.3 YOLO NPU Implementation

```python
class YOLOv8ONNX(Operator):
    """YOLOv8 inference using ONNX Runtime with Ryzen AI NPU."""

    def setup(self, config: dict) -> None:
        import onnxruntime as ort

        # Configure for NPU execution
        providers = ['RyzenAIExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(
            config['model_path'],
            providers=providers
        )
        self.conf_threshold = config.get('conf_threshold', 0.25)

    def process(self, inputs: dict) -> dict:
        image = inputs['image']
        # Preprocess
        blob = self._preprocess(image)
        # Inference
        outputs = self.session.run(None, {'images': blob})
        # Postprocess
        detections = self._postprocess(outputs)
        return {'detections': detections}
```

### Phase 5: Benchmark Harness (Week 5-6)

**Goal**: Real execution and measurement infrastructure

#### 5.1 Architecture Runner

```python
class ArchitectureRunner:
    """Execute complete architecture pipeline with timing."""

    def __init__(self, architecture: SoftwareArchitecture):
        self.arch = architecture
        self.operators: dict[str, Operator] = {}
        self.timings: dict[str, list[float]] = {}

    def load_operators(self) -> None:
        """Instantiate all operators in the architecture."""
        for op_inst in self.arch.operators:
            operator_class = get_operator_class(op_inst.operator_id)
            self.operators[op_inst.id] = operator_class()
            self.operators[op_inst.id].setup(op_inst.config)

    def run_pipeline(self, inputs: dict) -> tuple[dict, dict]:
        """Run full pipeline, return outputs and timings."""
        data = inputs.copy()
        timings = {}

        for op_inst in self.arch.operators:
            op = self.operators[op_inst.id]

            start = time.perf_counter_ns()
            outputs = op.process(data)
            elapsed_ms = (time.perf_counter_ns() - start) / 1e6

            timings[op_inst.id] = elapsed_ms
            data.update(outputs)

        return data, timings

    def benchmark(self, iterations: int, warmup: int) -> ArchitectureBenchmarkResult:
        """Run benchmark and return results."""
        pass
```

#### 5.2 Power Measurement Integration

```python
class PowerMonitor:
    """Monitor power consumption during benchmark."""

    def __init__(self, platform: str):
        self.platform = platform
        # Use appropriate backend
        if 'Ryzen' in platform:
            self.backend = RyzenSMUMonitor()
        else:
            self.backend = RAPLMonitor()

    def start(self) -> None:
        """Start power monitoring."""
        self.backend.start_sampling(interval_ms=10)

    def stop(self) -> PowerMetrics:
        """Stop monitoring and return metrics."""
        samples = self.backend.stop_sampling()
        return PowerMetrics(
            mean_watts=statistics.mean(samples),
            peak_watts=max(samples),
            samples=samples
        )
```

### Phase 6: Demo Workflow (Week 6-7)

**Goal**: Interactive demonstration of agentic design

#### 6.1 Demo Script

```
User: "I need to deploy a perception-control pipeline on the Ryzen 7 AI NUC.
       It should run at 30fps with under 25W power budget."

Agent: [calls list_architectures with platform_type=robot]
       "I found the pgnc_robot_v1 architecture. Let me analyze it on your target."

Agent: [calls analyze_architecture('pgnc_robot_v1', 'Ryzen-7-AI-NUC')]
       "The base architecture estimates 48ms latency on Ryzen 7 - that's 20fps,
        below your 30fps target."

Agent: [calls identify_bottleneck('pgnc_robot_v1', 'Ryzen-7-AI-NUC')]
       "Bottleneck: The YOLOv8n detector takes 32ms (67% of total) on GPU.
        However, the NPU can run it faster."

Agent: [calls suggest_optimizations('pgnc_robot_v1', 'Ryzen-7-AI-NUC')]
       "Recommendation: Use NPU for detection instead of GPU.
        Expected improvement: 32ms → 12ms, new total: 28ms (35fps)."

User: "Run a benchmark to verify."

Agent: [calls benchmark_architecture('pgnc_robot_v1_ryzen7', 'Ryzen-7-AI-NUC')]
       "Benchmark results:
        - Measured latency: 29.3ms ± 2.1ms (target: 33ms) ✓
        - Throughput: 34.1 fps ✓
        - Power: 21.3W (budget: 25W) ✓

        All constraints met. Ready for deployment."
```

#### 6.2 Comparison Demo

```
User: "Compare performance on Ryzen 7 vs Ryzen 9"

Agent: [calls compare_variants on both platforms]

| Metric | Ryzen 7 AI | Ryzen 9 AI |
|--------|------------|------------|
| Latency | 29.3ms | 22.1ms |
| Throughput | 34 fps | 45 fps |
| Power | 21.3W | 34.2W |
| Efficiency | 1.6 fps/W | 1.3 fps/W |

"Ryzen 7 is more power-efficient (1.6 vs 1.3 fps/W).
 Ryzen 9 has 36% higher throughput but uses 60% more power.

 Recommendation: Ryzen 7 for battery-powered robots,
                 Ryzen 9 for performance-critical applications."
```

### Phase 7: Documentation & Polish (Week 7-8)

#### 7.1 Deliverables

1. **Hardware profiles** for both NUCs in embodied-schemas
2. **Operator profiles** with real measurements
3. **Reference architecture** (pgnc_robot_v1) with variants
4. **Runnable operators** for perception, estimation, control
5. **Benchmark harness** with power measurement
6. **Demo script** showing full workflow
7. **Results report** comparing both platforms

#### 7.2 Demo Video Script

1. Introduction: The problem of embodied AI design
2. Show the agentic workflow in `branes chat`
3. Design a PGN&C architecture interactively
4. Analyze on Ryzen 7 AI NUC
5. Identify bottleneck (GPU detector)
6. Apply optimization (move to NPU)
7. Run real benchmark on hardware
8. Compare with Ryzen 9 AI NUC
9. Conclusion: Power/performance trade-offs

## Implementation Timeline

| Week | Phase | Deliverables |
|------|-------|--------------|
| 1-2 | Platform Characterization | Hardware profiles, baseline benchmarks |
| 2-3 | Operator Profiling | Perf profiles for 15+ operators |
| 3-4 | Reference Architecture | pgnc_robot_v1 with 4 variants |
| 4-6 | Operator Implementations | Runnable perception, control operators |
| 5-6 | Benchmark Harness | Real execution with power measurement |
| 6-7 | Demo Workflow | Interactive demo script |
| 7-8 | Documentation | Report, video, polish |

## Dependencies

### Software Requirements

- Ubuntu 22.04 or 24.04
- ROCm 6.0+ (GPU compute)
- ONNX Runtime with Ryzen AI EP (NPU)
- Python 3.10+
- OpenCV 4.x
- NumPy, SciPy

### Hardware Access

- [ ] Ryzen 7 AI NUC available and configured
- [ ] Ryzen 9 AI NUC available and configured
- [ ] USB camera for real input testing
- [ ] Power measurement capability (external or ryzen_smu)

## Success Criteria

1. **Functional**: Complete pipeline runs on both platforms
2. **Accurate**: Estimates within 15% of measured performance
3. **Useful**: Optimization suggestions improve performance
4. **Demonstrable**: Clear before/after in demo workflow
5. **Documented**: Reproducible by others

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| NPU driver issues | High | Fall back to GPU, document workarounds |
| Power measurement inaccurate | Medium | Use external meter as ground truth |
| Operator implementations slow | Medium | Start with simpler operators |
| Hardware availability | High | Cloud instance backup? |

## Next Steps

1. **Immediate**: Verify Ryzen AI SDK installation on both NUCs
2. **Week 1**: Create hardware catalog entries
3. **Week 1**: Run baseline YOLO benchmarks on NPU
4. **Week 2**: Profile all perception operators
5. **Week 3**: Create pgnc_robot_v1 architecture

## Appendix: Key Commands

```bash
# Check NPU availability
python -c "import onnxruntime; print(onnxruntime.get_available_providers())"

# Run YOLO on NPU
python -m operators.perception.yolo_onnx --model yolov8n.onnx --device npu

# Benchmark architecture
branes chat
> benchmark_architecture pgnc_robot_v1 Ryzen-7-AI-NUC

# Power monitoring
sudo ryzen_smu --power-monitor --interval 100
```
