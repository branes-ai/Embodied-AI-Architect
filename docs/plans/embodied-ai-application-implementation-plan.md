# Embodied AI Application Framework - Implementation Plan

**Date**: 2025-11-02
**Architecture**: Hybrid (Python Framework + Graph IR + DSL)
**Status**: Planning Phase
**Target Timeline**: 12-16 weeks for core functionality

---

## Executive Summary

This plan details the implementation of a comprehensive Embodied AI Application framework that enables:
1. Authoring complete embodied AI applications in Python
2. Capturing computational graphs for analysis
3. Benchmarking end-to-end performance
4. Transpiling to C++/Rust for production deployment

**Key Innovation**: Extend beyond individual DNN benchmarking to complete control loop applications with heterogeneous operators.

---

## Existing Infrastructure (Leverage)

We have significant infrastructure already available:

| Component | Status | Integration Point |
|-----------|--------|-------------------|
| PyTorch FX | Available | Graph capture for DNNs |
| MLIR/IREE | Available | C++ IR analysis & compilation |
| Python Profiling/Tracing | Available | Execution trace capture |
| Simulators (Functional/Perf/Energy) | Available | Benchmark backends |
| Compiler/Runtime Modeler | Available | Hardware mapping analysis |
| Current Agent System | Implemented | Extend for applications |

**Strategy**: Build new framework on top of existing infrastructure rather than reinventing.

---

## Phase 1: Foundation - Core Application Framework

**Duration**: 3-4 weeks
**Goal**: Enable users to write embodied AI applications in Python

### Milestone 1.1: Base Framework Classes (Week 1)

**Tasks:**
1. Design application base class hierarchy
2. Implement `EmbodiedAIApp` base class
3. Create operator interface (`Operator` ABC)
4. Implement sensor/actuator abstractions
5. Design type system for data flow

**Deliverables:**
- `src/embodied_ai_architect/app/base.py`
- `src/embodied_ai_architect/app/operator.py`
- `src/embodied_ai_architect/app/sensors.py`
- `src/embodied_ai_architect/app/actuators.py`
- `src/embodied_ai_architect/app/types.py`
- Unit tests for base classes

**Acceptance Criteria:**
```python
# User can write:
class MyApp(EmbodiedAIApp):
    def __init__(self):
        self.sensor = Sensor("camera", shape=(3, 224, 224))
        self.actuator = Actuator("motors", shape=(4,))

    def step(self):
        data = self.sensor.read()
        output = self.process(data)
        self.actuator.write(output)
```

### Milestone 1.2: Control Loop Framework (Week 2)

**Tasks:**
1. Implement `@control_loop` decorator
2. Create execution scheduler
3. Implement timing and frequency control
4. Add state management
5. Design data bundle classes (SensorBundle, ActuatorBundle)

**Deliverables:**
- `src/embodied_ai_architect/app/decorators.py`
- `src/embodied_ai_architect/app/scheduler.py`
- `src/embodied_ai_architect/app/bundles.py`
- `src/embodied_ai_architect/app/runtime.py`
- Integration tests

**Acceptance Criteria:**
```python
@embai_application(name="test_app")
class TestApp:
    @control_loop(frequency=30)  # 30 Hz
    def step(self, sensors: SensorBundle) -> ActuatorBundle:
        # Runs at specified frequency
        return ActuatorBundle(motors=control)
```

### Milestone 1.3: Graph IR Data Structures (Week 3)

**Tasks:**
1. Design graph IR schema
2. Implement node/edge data structures
3. Create graph builder API
4. Implement graph validation
5. Add graph visualization (GraphViz export)

**Deliverables:**
- `src/embodied_ai_architect/graph/ir.py`
- `src/embodied_ai_architect/graph/nodes.py`
- `src/embodied_ai_architect/graph/edges.py`
- `src/embodied_ai_architect/graph/builder.py`
- `src/embodied_ai_architect/graph/validation.py`
- `src/embodied_ai_architect/graph/visualization.py`

**Acceptance Criteria:**
```python
graph = EmbodiedAIGraph()
graph.add_operator("detector", PyTorchOperator(...))
graph.add_edge("sensors.camera", "detector.input")
graph.validate()  # Check data types, cycles, etc.
graph.visualize("app_graph.png")
```

### Milestone 1.4: Simple Example Application (Week 4)

**Tasks:**
1. Create reference implementation (e.g., object tracking)
2. Write comprehensive documentation
3. Add examples directory
4. Create tutorial notebook

**Deliverables:**
- `examples/simple_tracking_app.py`
- `examples/notebooks/application_tutorial.ipynb`
- `docs/guides/writing-applications.md`
- Architecture documentation updates

**Acceptance Criteria:**
- Example app runs successfully
- Documentation is clear and comprehensive
- Tutorial guides new users through app creation

---

## Phase 2: Operator Library

**Duration**: 2-3 weeks
**Goal**: Provide rich library of operators for common embodied AI tasks

### Milestone 2.1: PyTorch Operator Integration (Week 5)

**Tasks:**
1. Create PyTorchOperator wrapper
2. Integrate with existing ModelAnalyzerAgent
3. Handle input/output tensor shapes
4. Support for common architectures (CNN, ResNet, YOLO, etc.)
5. Add model loading utilities

**Deliverables:**
- `src/embodied_ai_architect/operators/pytorch_operator.py`
- `src/embodied_ai_architect/operators/model_loader.py`
- Unit tests with real models
- Integration with PyTorch FX

**Acceptance Criteria:**
```python
detector = PyTorchOperator(
    model=torch.load("yolov5.pt"),
    input_shape=(1, 3, 640, 640)
)
detections = detector(image_tensor)
```

### Milestone 2.2: Classical Algorithm Operators (Week 6-7)

**Tasks:**
1. Implement KalmanFilter operator
2. Implement PIDController operator
3. Implement PathPlanner operators (A*, RRT*)
4. Add common preprocessing operators (resize, normalize, etc.)
5. Create operator catalog/registry

**Deliverables:**
- `src/embodied_ai_architect/operators/kalman_filter.py`
- `src/embodied_ai_architect/operators/pid_controller.py`
- `src/embodied_ai_architect/operators/path_planners.py`
- `src/embodied_ai_architect/operators/preprocessing.py`
- `src/embodied_ai_architect/operators/registry.py`
- Comprehensive operator tests

**Acceptance Criteria:**
```python
kf = KalmanFilter(state_dim=6, measurement_dim=3)
state = kf.predict()
state = kf.update(measurement)

pid = PIDController(kp=1.0, ki=0.1, kd=0.01)
control = pid.compute(current, setpoint)

planner = PathPlanner(algorithm="rrt_star")
path = planner.plan(start, goal, obstacles)
```

### Milestone 2.3: Operator Type System (Week 7)

**Tasks:**
1. Define operator input/output types
2. Implement type checking system
3. Add type inference
4. Create type validation framework
5. Add runtime type assertions

**Deliverables:**
- `src/embodied_ai_architect/operators/types.py`
- `src/embodied_ai_architect/operators/type_checker.py`
- Type annotation examples
- Documentation on type system

**Acceptance Criteria:**
```python
# Type errors caught at graph construction:
detector.output_type  # → Tensor[float32, (N, 5)]
kalman.input_type     # → Vector[float32, (3,)]
# Mismatched types detected before execution
```

---

## Phase 3: Graph Capture & Tracing

**Duration**: 2 weeks
**Goal**: Automatically capture computational graphs from Python execution

### Milestone 3.1: PyTorch FX Integration (Week 8)

**Tasks:**
1. Create tracer using PyTorch FX
2. Handle DNN graph capture
3. Extract DNN subgraphs
4. Integrate with existing graph IR

**Deliverables:**
- `src/embodied_ai_architect/graph/tracer_fx.py`
- FX-based graph capture
- Integration tests with real models

**Acceptance Criteria:**
```python
app = MyApp()
tracer = FXTracer()
graph = tracer.trace(app.step)
# Graph contains all PyTorch operations
```

### Milestone 3.2: Full Application Tracing (Week 9)

**Tasks:**
1. Extend tracing beyond PyTorch operations
2. Capture calls to custom operators
3. Handle control flow (if/else, loops)
4. Build complete application graph
5. Add debugging/introspection tools

**Deliverables:**
- `src/embodied_ai_architect/graph/tracer.py`
- `src/embodied_ai_architect/graph/control_flow.py`
- Full application graph capture
- Debugging utilities

**Acceptance Criteria:**
```python
app = DroneNav()
graph = app.compile()
# Graph includes: detector, kalman, planner, controller
# All data flow edges captured
# Control flow annotated
```

---

## Phase 4: Benchmarking Framework

**Duration**: 2-3 weeks
**Goal**: Benchmark complete applications end-to-end

### Milestone 4.1: Dataset Infrastructure (Week 10)

**Tasks:**
1. Design dataset interface
2. Implement ROS bag loader
3. Create custom dataset format
4. Add dataset preprocessing
5. Support for ground truth annotations

**Deliverables:**
- `src/embodied_ai_architect/datasets/base.py`
- `src/embodied_ai_architect/datasets/rosbag.py`
- `src/embodied_ai_architect/datasets/custom.py`
- `src/embodied_ai_architect/datasets/preprocessing.py`
- Example datasets

**Acceptance Criteria:**
```python
dataset = RosbagDataset("flight_001.bag")
for sample in dataset:
    camera_image = sample.sensors["camera"]
    imu_reading = sample.sensors["imu"]
    ground_truth = sample.ground_truth
```

### Milestone 4.2: Metrics Collection (Week 11)

**Tasks:**
1. Design metrics framework
2. Implement latency metrics (mean, p50, p95, p99)
3. Implement accuracy metrics (task-specific)
4. Implement energy metrics (via simulators)
5. Add per-operator profiling

**Deliverables:**
- `src/embodied_ai_architect/benchmark/metrics/base.py`
- `src/embodied_ai_architect/benchmark/metrics/latency.py`
- `src/embodied_ai_architect/benchmark/metrics/accuracy.py`
- `src/embodied_ai_architect/benchmark/metrics/energy.py`
- `src/embodied_ai_architect/benchmark/profiler.py`

**Acceptance Criteria:**
```python
metrics = [
    LatencyMetric(percentiles=[50, 95, 99]),
    AccuracyMetric(iou_threshold=0.5),
    EnergyMetric()
]
results = benchmark.run(metrics=metrics)
# Per-operator and end-to-end metrics collected
```

### Milestone 4.3: Application Benchmark Harness (Week 12)

**Tasks:**
1. Create ApplicationBenchmark class
2. Integrate with existing BenchmarkAgent
3. Add simulator backend integration
4. Implement result aggregation
5. Generate comparison reports

**Deliverables:**
- `src/embodied_ai_architect/benchmark/application_benchmark.py`
- Integration with existing infrastructure
- `src/embodied_ai_architect/benchmark/backends/simulator.py`
- Report generation for applications

**Acceptance Criteria:**
```python
benchmark = ApplicationBenchmark(
    application=app.compile(),
    dataset=dataset,
    metrics=metrics
)
results = benchmark.run(backends=[
    LocalCPU(),
    JetsonAGX(),
    Simulator(energy_model="nvidia_orin")
])
report = results.generate_report()
```

---

## Phase 5: MLIR/IREE Integration

**Duration**: 3-4 weeks
**Goal**: Connect to existing MLIR/IREE infrastructure for compilation

### Milestone 5.1: MLIR Export (Week 13-14)

**Tasks:**
1. Design IR → MLIR mapping
2. Implement graph traversal for export
3. Handle operator lowering to MLIR
4. Map PyTorch ops to MLIR dialects
5. Export complete application graphs

**Deliverables:**
- `src/embodied_ai_architect/codegen/mlir_exporter.py`
- `src/embodied_ai_architect/codegen/mlir_dialects.py`
- MLIR generation from graphs
- Integration tests

**Acceptance Criteria:**
```python
graph = app.compile()
mlir_module = graph.to_mlir()
# Valid MLIR module with all operators
```

### Milestone 5.2: IREE Compilation Pipeline (Week 15)

**Tasks:**
1. Create IREE compiler integration
2. Configure compilation targets
3. Add optimization passes
4. Generate executable artifacts
5. Benchmark compiled applications

**Deliverables:**
- `src/embodied_ai_architect/codegen/iree_compiler.py`
- IREE compilation integration
- Hardware-specific optimizations
- Compiled artifact execution

**Acceptance Criteria:**
```python
compiler = IREECompiler(target="nvidia_orin")
binary = compiler.compile(mlir_module)
runtime = IREERuntime(binary)
results = runtime.execute(inputs)
```

### Milestone 5.3: Hardware Mapping Analysis (Week 16)

**Tasks:**
1. Integrate with existing compiler/runtime modeler
2. Analyze operator placement
3. Evaluate different mapping strategies
4. Provide hardware recommendations
5. Feed back to HardwareProfileAgent

**Deliverables:**
- `src/embodied_ai_architect/analysis/hardware_mapping.py`
- Integration with compiler modeler
- Mapping strategy evaluation
- Updated hardware recommendations

**Acceptance Criteria:**
```python
mapper = HardwareMapper(graph, target_hardware)
strategies = mapper.evaluate_mappings()
best_strategy = mapper.recommend()
# Recommendations: place detector on GPU, kalman on CPU
```

---

## Phase 6: DSL Serialization

**Duration**: 1-2 weeks
**Goal**: Enable graph serialization and storage

### Milestone 6.1: DSL Format Definition (Week 17)

**Tasks:**
1. Define JSON schema for applications
2. Create DSL specification document
3. Add schema validation
4. Design versioning strategy

**Deliverables:**
- `docs/specifications/application-dsl-v1.md`
- JSON schema file
- Validation utilities
- Examples in DSL format

**Acceptance Criteria:**
```json
{
  "application": {...},
  "operators": [...],
  "dataflow": [...],
  "schedule": [...]
}
```

### Milestone 6.2: Serialization Implementation (Week 18)

**Tasks:**
1. Implement IR → DSL serializer
2. Implement DSL → IR deserializer
3. Add round-trip testing
4. Handle versioning and migrations

**Deliverables:**
- `src/embodied_ai_architect/dsl/serializer.py`
- `src/embodied_ai_architect/dsl/deserializer.py`
- `src/embodied_ai_architect/dsl/validation.py`
- Round-trip tests

**Acceptance Criteria:**
```python
# Serialize
dsl_dict = graph.to_dsl()
with open("app.json", "w") as f:
    json.dump(dsl_dict, f)

# Deserialize
graph2 = EmbodiedAIGraph.from_dsl(dsl_dict)
assert graph == graph2  # Round-trip successful
```

---

## Phase 7: C++/Rust Transpilation (Future Work)

**Duration**: 4+ weeks
**Goal**: Generate production-ready C++/Rust code

### Milestone 7.1: C++ Code Generation

**Tasks:**
1. Design C++ code generation templates
2. Implement operator code generators
3. Generate runtime library
4. Add build system integration (CMake)
5. Validate generated code correctness

### Milestone 7.2: Runtime Library

**Tasks:**
1. Implement C++ runtime library
2. Port operators to C++
3. Add scheduling and timing
4. Integrate with IREE runtime

### Milestone 7.3: Rust Code Generation (Optional)

**Tasks:**
1. Rust code generation templates
2. Memory safety verification
3. Cargo build integration

---

## Integration Points with Existing System

### With Current Agents

| Existing Component | Integration |
|-------------------|-------------|
| ModelAnalyzerAgent | Used by PyTorchOperator for analysis |
| BenchmarkAgent | Extended for application benchmarking |
| HardwareProfileAgent | Receives application-level recommendations |
| ReportSynthesisAgent | Generates application benchmark reports |

### With Infrastructure

| Infrastructure | Integration |
|----------------|-------------|
| PyTorch FX | Graph capture for DNNs |
| MLIR/IREE | Compilation backend |
| Simulators | Benchmark backends |
| Compiler Modeler | Hardware mapping analysis |
| CLI | New `branes app` commands |

---

## CLI Extensions

Add new command group:

```bash
# Application commands
branes app create <name>          # Scaffold new app
branes app validate <app.py>      # Validate application
branes app compile <app.py>       # Compile to graph IR
branes app visualize <app.py>     # Visualize graph
branes app benchmark <app.py>     # Benchmark application
branes app export <app.py>        # Export to DSL/MLIR/C++

# Examples
branes app benchmark drone_nav.py \
  --dataset flight_001.bag \
  --backend jetson_agx \
  --metrics latency,accuracy,energy
```

---

## Testing Strategy

### Unit Tests
- All operator implementations
- Graph IR operations
- Type checking system
- Serialization round-trips

### Integration Tests
- Complete application workflows
- Benchmarking pipelines
- MLIR/IREE compilation
- Multi-backend execution

### End-to-End Tests
- Reference applications (drone nav, robot manipulation)
- Real datasets (ROS bags)
- Hardware deployment (if available)

### Performance Tests
- Profiling overhead measurement
- Compilation time benchmarks
- Runtime performance validation

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| PyTorch FX tracing complexity | Start with simple operators, gradual expansion |
| MLIR integration difficulties | Early prototype, seek IREE community help |
| Operator library incompleteness | Prioritize common operators, extensibility |
| Performance overhead | Profile early, optimize hot paths |
| Control flow handling | Use symbolic execution, limit dynamic behavior |

---

## Success Metrics

### Phase 1-2 Success:
- ✅ Can write 3+ example applications
- ✅ All operators work correctly
- ✅ Type system catches errors

### Phase 3-4 Success:
- ✅ Graph capture success rate > 95%
- ✅ Benchmark 5+ applications
- ✅ Per-operator latency breakdown

### Phase 5-6 Success:
- ✅ MLIR export works for all operators
- ✅ IREE compilation successful
- ✅ DSL round-trip 100% accurate

### Overall Success:
- ✅ 10+ reference applications
- ✅ Performance within 10% of hand-written C++
- ✅ Deployment to real hardware
- ✅ External users can write applications

---

## Timeline Summary

| Phase | Weeks | Key Deliverable |
|-------|-------|-----------------|
| 1: Foundation | 4 | Application framework |
| 2: Operators | 3 | Operator library |
| 3: Tracing | 2 | Graph capture |
| 4: Benchmarking | 3 | Application benchmarks |
| 5: MLIR/IREE | 4 | Compilation pipeline |
| 6: DSL | 2 | Serialization |
| **Total (Core)** | **18** | **Production-ready framework** |
| 7: Transpilation | 4+ | C++/Rust code generation |

**Total with transpilation: 22+ weeks (~5-6 months)**

---

## Dependencies

**External:**
- PyTorch >= 2.0 (FX support)
- MLIR/IREE (existing)
- ROS (optional, for datasets)
- NumPy/SciPy
- GraphViz (for visualization)

**Internal:**
- Existing agent system
- Existing benchmark infrastructure
- CLI framework
- Secrets management

---

## Next Steps

1. **Immediate (This Week)**:
   - Review and approve this plan
   - Set up project board with milestones
   - Create feature branch: `feature/embodied-ai-applications`
   - Begin Phase 1, Milestone 1.1

2. **Short Term (Next 2 Weeks)**:
   - Complete Phase 1, Milestone 1.1-1.2
   - Create first simple example
   - Document architecture decisions

3. **Medium Term (Next 4 Weeks)**:
   - Complete Phase 1
   - Begin Phase 2
   - Prototype MLIR integration

4. **Long Term (3 Months)**:
   - Complete Phases 1-4
   - Begin MLIR/IREE integration
   - Publish early access to external users

---

## Questions for Discussion

1. **Operator Priority**: Which operators are most critical? (Kalman, PID, path planners?)
2. **Dataset Format**: Prioritize ROS bags or custom format?
3. **MLIR/IREE Timeline**: Can we access IREE team for integration help?
4. **Hardware Access**: Which hardware platforms for real deployment tests?
5. **Resource Allocation**: How many developers on this project?
6. **External Dependencies**: Any concerns about PyTorch FX limitations?

---

## Appendix: Reference Applications

Planned reference applications to validate framework:

1. **Drone Navigation** (Phase 1)
   - Object detection → Kalman filter → Path planning → PID control
   - Dataset: Simulated drone flight

2. **Robot Arm Manipulation** (Phase 2)
   - Object detection → Pose estimation → Inverse kinematics → Trajectory control
   - Dataset: Pick-and-place tasks

3. **Autonomous Vehicle** (Phase 3)
   - Lane detection → Vehicle tracking → Path planning → MPC control
   - Dataset: KITTI or Waymo Open Dataset

4. **Legged Robot Locomotion** (Phase 4)
   - Terrain classification → Gait planning → Joint control
   - Dataset: ANYmal or Spot robot logs

5. **Humanoid Balance** (Phase 5)
   - IMU processing → State estimation → Balance control
   - Dataset: Atlas or Digit robot data
