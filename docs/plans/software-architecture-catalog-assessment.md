# Software Architecture Catalog Assessment

## Executive Summary

The Embodied AI Architect currently excels at **point analysis**: given a DNN model and hardware target, it provides verdict-first assessments of latency, power, memory, and cost. However, real embodied AI systems are **compositions** of multiple operators, sensors, control loops, and planners—not isolated models.

To help architects design state-of-the-art embodied AI, we need to model **software architectures as first-class artifacts**, just as we model hardware platforms and DNN kernels today.

## The Gap

### What We Have

```
┌────────────────────────────────────────────────────────────┐
│                    Current Scope                           │
├────────────────────────────────────────────────────────────┤
│  Hardware Catalog          Model Catalog                   │
│  ├─ H100, A100, TPU       ├─ ResNet, YOLO, ViT             │
│  ├─ Jetson Orin variants  ├─ MobileNet, EfficientNet       │
│  ├─ Coral Edge TPU        └─ (Attention, Enc/Dec planned)  │
│  └─ TDA4VM, Hailo-8                                        │
│                                                            │
│  Analysis Tools                                            │
│  ├─ check_latency(model, hw, target)                       │
│  ├─ check_power(model, hw, budget)                         │
│  ├─ check_memory(model, hw, limit)                         │
│  └─ full_analysis(model, hw)                               │
└────────────────────────────────────────────────────────────┘
```

### What We Need

```
┌────────────────────────────────────────────────────────────┐
│                    Required Scope                          │
├────────────────────────────────────────────────────────────┤
│  Hardware    Model         Operator        Application     │
│  Catalog     Catalog       Catalog         Catalog         │
│  (exists)    (exists)      (NEW)           (NEW)           │
│                                                            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Application = Composition of Operators             │   │
│  │                                                     │   │
│  │  Sensor → Preprocess → Detect → Track → Reason      │   │
│  │    ↓                                      ↓         │   │
│  │  IMU ──────────────→ State Est. ──→ Controller      │   │
│  │                          ↓              ↓           │   │
│  │                      Planner ──────→ Actuator       │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                            │
│  Analysis Tools (Extended)                                 │
│  ├─ check_latency(model | operator | app, hw, target)      │
│  ├─ analyze_dataflow(app, hw) → bottleneck identification  │
│  ├─ analyze_scheduling(app, hw) → rate feasibility         │
│  └─ full_analysis(app, hw) → end-to-end verdict            │
└────────────────────────────────────────────────────────────┘
```

## Conceptual Model

### Three Layers of Abstraction

| Layer | Artifact | Examples | Analysis Questions |
|-------|----------|----------|-------------------|
| **Kernel** | Individual compute kernel | MatMul, Conv2D, Attention | Roofline position, memory vs compute bound |
| **Operator** | Functional unit | YOLO detector, Kalman filter, PID controller | Latency, power, memory footprint |
| **Application** | Complete system | Drone perception, ADAS, robot manipulation | End-to-end latency, power envelope, scheduling feasibility |

### Operator Taxonomy

Based on the drone perception prototype and common embodied AI patterns:

```
Operators
├── Perception
│   ├── Detection (YOLO, SSD, DETR, RT-DETR)
│   ├── Segmentation (Mask R-CNN, SAM)
│   ├── Depth Estimation (MiDaS, DPT)
│   ├── Optical Flow (RAFT, FlowNet)
│   ├── SLAM (ORB-SLAM, RTAB-MAP)
│   └── Tracking (ByteTrack, DeepSORT, SORT)
│
├── State Estimation
│   ├── Kalman Filters (EKF, UKF, Particle)
│   ├── Sensor Fusion (IMU+Vision, Multi-camera)
│   └── Localization (Monte Carlo, Factor Graph)
│
├── Planning
│   ├── Path Planning (RRT*, A*, D*)
│   ├── Motion Planning (MPC, DWA)
│   ├── Trajectory Optimization (CHOMP, TrajOpt)
│   └── Behavior Planning (FSM, BT, HFSM)
│
├── Control
│   ├── PID Controllers
│   ├── LQR/LQG
│   ├── Model Predictive Control
│   └── Impedance/Admittance Control
│
├── Reasoning
│   ├── Scene Understanding
│   ├── Collision Prediction
│   ├── Behavior Classification
│   └── Risk Assessment
│
└── Infrastructure
    ├── Preprocessing (resize, normalize, color convert)
    ├── Postprocessing (NMS, filtering, smoothing)
    ├── Data Fusion (temporal, spatial)
    └── Communication (serialization, pub/sub)
```

### Application Composition Patterns

Real embodied AI systems use these composition patterns:

| Pattern | Description | Example |
|---------|-------------|---------|
| **Pipeline** | Sequential data flow | Camera → Detect → Track → Display |
| **Fan-out** | One source, multiple consumers | Frame → {Detect, Depth, Flow} |
| **Fan-in** | Multiple sources, one consumer | {Camera, LiDAR, IMU} → Fusion |
| **Feedback** | Output influences input | Controller → Actuator → Sensor → Controller |
| **Multi-rate** | Components at different frequencies | Perception@30Hz, Control@200Hz, Planning@10Hz |
| **Conditional** | Data-dependent branching | if obstacle_detected → emergency_stop else → normal_control |

## Proposed Schema Extensions

### Operator Entry (for embodied-schemas)

```python
class OperatorCategory(str, Enum):
    PERCEPTION = "perception"
    STATE_ESTIMATION = "state_estimation"
    PLANNING = "planning"
    CONTROL = "control"
    REASONING = "reasoning"
    INFRASTRUCTURE = "infrastructure"

class OperatorEntry(BaseModel):
    """Catalog entry for a reusable operator."""
    id: str                          # "bytetrack-v1"
    name: str                        # "ByteTrack Multi-Object Tracker"
    category: OperatorCategory

    # Implementation
    impl_type: str                   # "python", "pytorch", "cpp", "cuda"
    reference_impl: str | None       # GitHub URL or package

    # Interface specification
    inputs: dict[str, TensorSpec]    # {"detections": TensorSpec(...)}
    outputs: dict[str, TensorSpec]   # {"tracks": TensorSpec(...)}
    config_schema: dict | None       # JSON schema for configuration

    # Resource characteristics (reference, not hardware-specific)
    typical_latency_ms: float | None
    memory_footprint_mb: float | None
    compute_flops: float | None

    # Dependencies
    requires_gpu: bool = False
    requires_model: str | None       # If wraps a DNN: "yolov8n"

    # Metadata
    tags: list[str] = []             # ["real-time", "multi-object", "occlusion-robust"]
    papers: list[str] = []           # Reference papers
```

### Software Architecture Entry

```python
class DataflowEdge(BaseModel):
    """Connection between operators."""
    source_op: str                   # Operator ID
    source_port: str                 # Output port name
    target_op: str                   # Operator ID
    target_port: str                 # Input port name
    queue_size: int = 1              # Buffer depth

class OperatorInstance(BaseModel):
    """Instantiation of an operator in an architecture."""
    id: str                          # Instance ID within this architecture
    operator_id: str                 # Reference to OperatorEntry
    config: dict = {}                # Operator-specific configuration
    rate_hz: float | None            # Execution rate (None = event-driven)
    priority: int = 0                # Scheduling priority
    cpu_affinity: list[int] | None   # Core pinning

class SoftwareArchitecture(BaseModel):
    """Complete embodied AI application architecture."""
    id: str                          # "drone-perception-v1"
    name: str                        # "Drone Obstacle Avoidance Pipeline"
    description: str

    # Platform binding
    platform_type: PlatformType      # drone, quadruped, vehicle, etc.
    sensors: list[str]               # Sensor IDs from catalog
    actuators: list[str]             # Actuator IDs (future)

    # Composition
    operators: list[OperatorInstance]
    dataflow: list[DataflowEdge]

    # Timing requirements
    end_to_end_latency_ms: float | None
    min_throughput_fps: float | None

    # Resource envelope
    power_budget_w: float | None
    memory_budget_mb: float | None

    # Variants
    variants: list[ArchitectureVariant] = []  # Different configs

    # Metadata
    reference_impl: str | None       # Link to working code
    tags: list[str] = []
```

### Architecture Variant

```python
class ArchitectureVariant(BaseModel):
    """Alternative configuration of an architecture."""
    id: str                          # "drone-perception-v1-edge"
    name: str                        # "Edge-optimized variant"
    description: str

    # What changes from base
    operator_overrides: dict[str, str]  # {"detector": "yolov8n"} instead of "yolov8s"
    config_overrides: dict[str, dict]   # Per-operator config changes

    # Expected characteristics
    target_hardware: list[str]       # ["Jetson-Orin-Nano"]
    expected_latency_ms: float | None
    expected_power_w: float | None
```

## Analysis Capabilities

### New Analysis Tools

| Tool | Input | Output | Purpose |
|------|-------|--------|---------|
| `analyze_architecture` | app_id, hw_id | Full analysis | End-to-end performance |
| `check_scheduling` | app_id, hw_id | PASS/FAIL | Rate feasibility |
| `identify_bottleneck_op` | app_id, hw_id | Operator ID | Which operator limits throughput |
| `analyze_dataflow` | app_id, hw_id | Memory traffic | Data movement costs |
| `compare_variants` | app_id, [variants] | Ranked list | Trade-off analysis |
| `suggest_optimizations` | app_id, hw_id | Suggestions | Operator swaps, fusion opportunities |

### Multi-Rate Scheduling Analysis

For applications with operators at different rates:

```python
class SchedulingAnalysis(BaseModel):
    """Result of multi-rate scheduling analysis."""
    verdict: Verdict                 # PASS if all rates achievable
    confidence: Confidence

    operators: list[OperatorSchedule]

    # Aggregate metrics
    total_cpu_utilization: float     # 0-1 (or >1 if infeasible)
   #  total_gpu_utilization: float   gpu/kpu is too specific, we need a more flexible set
   #  total_kpu_utilization: float
    total_acc_utilization: dict[str, float]     # we could have a set of very different types of accelerators
    total_acc_efficiency: dict[str, float]      # efficiency is key to cost
    # we need to a multi-objective optimization of utilization and efficiency
    worst_case_latency_ms: float

    # Rate feasibility
    rate_analysis: list[RateFeasibility]

    # Suggestions if FAIL
    suggestions: list[str]

class RateFeasibility(BaseModel):
    operator_id: str
    target_rate_hz: float
    achievable: bool
    actual_rate_hz: float | None
    limiting_factor: str | None      # "compute", "memory_bandwidth", "dependency"
```

## Implementation Plan


  | Phase                           | Weeks | Deliverables                      |
  |---------------------------------|-------|-----------------------------------|
  | 1. Operator Catalog             | 4     | Schema, 20 operators, profiling   |
  | 2. Architecture Representation  | 4     | Schema, 3 reference architectures |
  | 3. Architecture Analysis        | 6     | E2E analysis, scheduling checks   |
  | 4. Optimization Recommendations | 4     | Bottleneck ID, suggestions        |
  | 5. End-to-End Benchmarking      | 4     | Real measurement, ROS bags        |


### Phase 1: Operator Catalog Foundation (4 weeks)

**Goal**: Establish operator abstraction and initial catalog

1. **Schema definition** (Week 1)
   - Add `OperatorEntry` to embodied-schemas
   - Add `OperatorCategory` enum
   - Define `TensorSpec` for I/O typing

2. **Initial operator catalog** (Week 2)
   - Perception: YOLO variants, ByteTrack, depth estimators
   - State estimation: Kalman filter variants
   - Control: PID controller
   - Infrastructure: preprocessing operators

3. **Operator profiling agent** (Week 3-4)
   - Profile operators on reference hardware
   - Store latency/memory characteristics
   - Integrate with existing benchmark backends

**Deliverables**:
- `OperatorEntry` schema in embodied-schemas
- 15-20 operator catalog entries
- `profile_operator` tool for the LLM

### Phase 2: Architecture Representation (4 weeks)

**Goal**: Model complete application architectures

1. **Schema definition** (Week 1)
   - Add `SoftwareArchitecture` to embodied-schemas
   - Add `DataflowEdge`, `OperatorInstance`
   - Define composition patterns

2. **Reference architectures** (Week 2-3)
   - Drone perception pipeline (from prototype)
   - Simple ADAS (camera → detect → track → warn)
   - Robot manipulation (perceive → plan → control)

3. **Architecture visualization** (Week 4)
   - DOT/GraphViz export
   - Mermaid diagram generation
   - CLI `branes app show <id>` command

**Deliverables**:
- `SoftwareArchitecture` schema
- 3 reference architectures
- `list_architectures`, `show_architecture` tools

### Phase 3: Architecture Analysis (6 weeks)

**Goal**: Analyze architectures on hardware targets

1. **Per-operator analysis** (Week 1-2)
   - Extend `full_analysis` to accept operator_id
   - Aggregate hardware-specific profiles

2. **Composition analysis** (Week 3-4)
   - End-to-end latency calculation (sum of critical path)
   - Memory aggregation (peak concurrent usage)
   - Power estimation (sum of operator powers)

3. **Scheduling analysis** (Week 5-6)
   - Multi-rate feasibility checking
   - Utilization calculation
   - Rate-monotonic analysis for priority assignment

**Deliverables**:
- `analyze_architecture(app_id, hw_id)` tool
- `check_scheduling(app_id, hw_id)` tool
- Scheduling feasibility verdicts

### Phase 4: Optimization Recommendations (4 weeks)

**Goal**: Suggest architecture optimizations

1. **Bottleneck identification** (Week 1-2)
   - Identify limiting operator
   - Classify bottleneck type (compute, memory, I/O)

2. **Optimization suggestions** (Week 3-4)
   - Operator swaps (YOLO-s → YOLO-n for edge)
   - Quantization recommendations
   - Fusion opportunities
   - Hardware upgrade suggestions

**Deliverables**:
- `identify_bottleneck_op` tool
- `suggest_optimizations` tool
- Architecture variant recommendations

### Phase 5: End-to-End Benchmarking (4 weeks)

**Goal**: Measure real application performance

1. **Application benchmark runner** (Week 1-2)
   - Execute full pipeline on hardware
   - Measure per-operator and end-to-end latency
   - Collect power/thermal metrics

2. **Dataset integration** (Week 3-4)
   - ROS bag replay support
   - Synthetic data generation
   - Benchmark result storage

**Deliverables**:
- `benchmark_architecture` command
- Per-operator breakdown in results
- Integration with existing benchmark backends

## Data Model Integration

### Repository Split

| Repository | New Artifacts |
|------------|---------------|
| **embodied-schemas** | `OperatorEntry`, `SoftwareArchitecture`, `ArchitectureVariant` |
| **graphs** | Operator roofline parameters, scheduling algorithms |
| **embodied-ai-architect** | Analysis agents, CLI commands, benchmarking |

### Catalog Population Strategy

1. **Operators**: Start with operators used in drone_perception prototype
2. **Architectures**: Formalize drone_perception as first catalog entry
3. **Expand**: Add architectures for other PlatformTypes (quadruped, vehicle, manipulator)

## Example Usage

### Architect Workflow

```
User: Can the drone perception pipeline run at 30fps on Jetson Orin Nano?

Agent: [calls analyze_architecture("drone-perception-v1", "Jetson-Orin-Nano")]

Tool returns:
{
  "verdict": "FAIL",
  "confidence": "high",
  "end_to_end_latency_ms": 52.3,
  "target_latency_ms": 33.3,
  "bottleneck_operator": "yolo_detector",
  "bottleneck_latency_ms": 38.1,
  "suggestion": "Replace YOLOv8s with YOLOv8n (est. 18ms) to achieve target"
}

Agent: The drone perception pipeline **cannot** meet 30fps on Jetson Orin Nano.

**Bottleneck**: The YOLOv8s detector takes 38.1ms alone, exceeding the 33.3ms budget.

**Recommendation**: Switch to YOLOv8n, which should reduce detection to ~18ms,
bringing end-to-end latency to ~32ms—just within budget.

Would you like me to analyze the YOLOv8n variant?
```

### Architecture Comparison

```
User: Compare the edge and cloud variants of the ADAS pipeline

Agent: [calls compare_variants("adas-v1", ["edge", "cloud"])]

| Metric | Edge (Orin Nano) | Cloud (A100) |
|--------|------------------|--------------|
| E2E Latency | 45ms | 8ms |
| Power | 15W | 400W |
| Detector | YOLOv8n | YOLOv8x |
| Accuracy (mAP) | 42.3 | 53.9 |
| Verdict | PASS (50ms target) | PASS |

Recommendation: Edge variant meets latency target with 10% headroom.
Cloud offers 28% better accuracy but requires network round-trip.
```

## Success Criteria

1. **Operator Catalog**: 30+ operators with profiling data
2. **Architecture Catalog**: 5+ reference architectures
3. **Analysis Coverage**: End-to-end latency, scheduling, power for any arch+hw pair
4. **Tool Integration**: All new tools work in `branes chat`
5. **User Value**: Architects can evaluate complete systems, not just isolated models

## Open Questions

1. **Heterogeneous execution**: How to model operators split across CPU/GPU/NPU?
2. **Communication overhead**: How to account for data serialization/transport?
3. **Dynamic behavior**: How to handle conditional execution paths?
4. **Accuracy modeling**: How to propagate accuracy through operator chains?

## Conclusion

Adding software architecture modeling transforms the Embodied AI Architect from a **component analyzer** to a **system design tool**. The phased approach builds on existing infrastructure:

- Phase 1-2: Data model and catalog (foundation)
- Phase 3-4: Analysis and optimization (value delivery)
- Phase 5: Real benchmarking (validation)

Total estimated effort: **22 weeks** across schema, analysis, and benchmarking work.
