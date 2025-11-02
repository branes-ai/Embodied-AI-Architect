# Multi-Rate Control Architecture for Embodied AI
## Research Document

**Date**: 2025-11-02
**Status**: Research & Architecture Proposal
**Focus**: Multi-rate control framework using Zenoh pub/sub

---

## Executive Summary

This document researches and proposes an architecture for a multi-rate control framework suitable for humanoid and quadruped robots, using Zenoh as the distributed communication backbone. The framework must support heterogeneous control loops running at different frequencies while maintaining real-time guarantees and enabling distributed execution.

**Key Innovation**: Combine Zenoh's zero-copy, real-time pub/sub with explicit multi-rate scheduling and time-domain isolation.

---

## Problem Analysis

### The Multi-Rate Control Challenge

Embodied AI systems (humanoids, quadrupeds) have fundamentally different temporal requirements:

```
Control Loop Hierarchy:
┌─────────────────────────────────────────────────────────┐
│ Planning Layer              1-10 Hz                     │  ← Slower
│ - Path planning, task planning                          │
│ - High-level decision making                            │
└─────────────────────────────────────────────────────────┘
                        ↓ Goals/waypoints
┌─────────────────────────────────────────────────────────┐
│ Perception Layer           30-60 Hz                     │  ← Medium
│ - Vision processing, object detection                   │
│ - State estimation, sensor fusion                       │
└─────────────────────────────────────────────────────────┘
                        ↓ State estimates
┌─────────────────────────────────────────────────────────┐
│ Control Layer             100-1000 Hz                   │  ← Faster
│ - Joint PID/torque control                              │
│ - Balance control, inverse dynamics                     │
│ - Direct sensor → actuator loops                        │
└─────────────────────────────────────────────────────────┘
```

**Why Multi-Rate?**
1. **Physics**: Joint control needs tight coupling with sensors/actuators (< 1ms)
2. **Computation**: DNNs are expensive, don't need to run at 1kHz
3. **Energy**: Running everything at max rate wastes power
4. **Modularity**: Different teams can develop subsystems independently
5. **Hardware Mapping**: Different loops may run on different processors

### Why Zenoh?

**Zenoh Advantages:**
- ✅ **Zero-copy**: Critical for high-frequency loops
- ✅ **Real-time**: Deterministic latency, no GC pauses
- ✅ **Automotive-grade**: Safety-certified variant available
- ✅ **Open source**: Apache 2.0, not vendor-locked
- ✅ **Protocol agnostic**: UDP, TCP, shared memory, DDS bridge
- ✅ **Routing**: Automatic data routing in distributed systems
- ✅ **QoS**: Reliability, priority, deadline policies
- ✅ **Queryable**: Request/response patterns
- ✅ **Storage**: Time-series data recording

**Zenoh vs Alternatives:**

| Feature | Zenoh | ROS2/DDS | MQTT | gRPC |
|---------|-------|----------|------|------|
| Zero-copy | ✅ | ⚠️ (vendor) | ❌ | ❌ |
| Real-time | ✅ | ⚠️ | ❌ | ❌ |
| Automotive cert | ✅ | ❌ | ❌ | ❌ |
| Lightweight | ✅ | ❌ | ✅ | ⚠️ |
| Multi-transport | ✅ | ⚠️ | ⚠️ | ❌ |
| Open source | ✅ | ✅ | ✅ | ✅ |

---

## Zenoh Architecture Deep Dive

### Core Concepts

**1. Key Expression (Path-based addressing)**
```
robot/joint/knee_left/position        ← Topic-like
robot/joint/*/position                ← Wildcard subscription
robot/joint/[knee_left,knee_right]/*  ← Set expression
```

**2. Publication Modes**
```rust
// Put (latest value)
session.put("robot/joint/knee/angle", angle).wait();

// Publisher (high frequency)
let publisher = session.declare_publisher("robot/imu/data");
loop {
    publisher.put(imu_data);  // Zero-copy if possible
}
```

**3. Subscription Modes**
```rust
// Callback-based
session.declare_subscriber("robot/joint/*/position")
    .callback(|sample| { /* process */ });

// Pull-based (for control loops)
let subscriber = session.declare_subscriber("robot/imu/data")
    .pull_mode();
loop {
    let samples = subscriber.recv();  // Blocking or non-blocking
}
```

**4. Queryable (request/response)**
```rust
// Service provider
session.declare_queryable("robot/planner/path")
    .callback(|query| {
        let path = compute_path(query.parameters());
        query.reply(path);
    });

// Service consumer
let replies = session.get("robot/planner/path?start=A&goal=B");
```

**5. Shared Memory Transport**
```rust
// Zero-copy between processes on same machine
let config = zenoh::config::peer()
    .transport(zenoh::config::TransportConfig::SharedMemory);
```

### QoS Policies

**Reliability:**
- `Reliable`: Guaranteed delivery (TCP-like)
- `BestEffort`: No retransmission (UDP-like)

**History:**
- `KeepLast(n)`: Keep n most recent samples
- `KeepAll`: Unbounded queue

**Liveliness:**
- Automatic detection of dead publishers

**Priority:**
- Control data can be prioritized over logging

---

## Multi-Rate Control Patterns

### Pattern 1: Time-Triggered Multi-Rate Scheduling

**Concept**: Each control loop runs at its own fixed frequency, triggered by a timer.

```python
# Pseudo-code
class ControlLoop:
    def __init__(self, frequency_hz: float):
        self.period = 1.0 / frequency_hz
        self.timer = Timer(self.period, self.tick)

    def tick(self):
        # Read inputs at this rate
        # Execute computation
        # Write outputs at this rate
        pass

# Different loops at different rates
joint_control = ControlLoop(1000)  # 1 kHz
perception = ControlLoop(30)        # 30 Hz
planning = ControlLoop(5)           # 5 Hz
```

**Zenoh Integration:**
```python
class ZenohControlLoop:
    def __init__(self, session, frequency_hz):
        self.session = session
        self.period = 1.0 / frequency_hz
        self.subscribers = {}
        self.publishers = {}

    def subscribe(self, key_expr):
        sub = self.session.declare_subscriber(
            key_expr,
            mode=zenoh.PullMode()  # Pull at our rate
        )
        self.subscribers[key_expr] = sub

    def publish(self, key_expr, value):
        if key_expr not in self.publishers:
            self.publishers[key_expr] = \
                self.session.declare_publisher(key_expr)
        self.publishers[key_expr].put(value)

    def run(self):
        while True:
            start = time.time()

            # Read latest data from all subscriptions
            inputs = {}
            for key, sub in self.subscribers.items():
                sample = sub.recv()  # Get latest
                inputs[key] = sample.payload

            # Execute control logic
            outputs = self.control_step(inputs)

            # Publish outputs
            for key, value in outputs.items():
                self.publish(key, value)

            # Sleep until next period
            elapsed = time.time() - start
            sleep_time = max(0, self.period - elapsed)
            time.sleep(sleep_time)
```

**Pros:**
- ✅ Simple to reason about
- ✅ Deterministic timing
- ✅ Easy to implement

**Cons:**
- ❌ Drift over time (need monotonic clock)
- ❌ Jitter from OS scheduling
- ❌ Fixed rates, not adaptive

### Pattern 2: Data-Triggered Multi-Rate (Event-Driven)

**Concept**: Loops triggered by data availability rather than timers.

```python
class DataDrivenLoop:
    def __init__(self, session, trigger_topic):
        self.session = session
        self.subscriber = session.declare_subscriber(
            trigger_topic,
            callback=self.on_data
        )

    def on_data(self, sample):
        # Triggered whenever new data arrives
        inputs = self.gather_inputs()
        outputs = self.control_step(inputs)
        self.publish_outputs(outputs)
```

**Example: Cascade Control**
```
IMU data (1000 Hz)
    ↓ triggers
Joint Control Loop (1000 Hz)
    ↓ publishes joint states (100 Hz decimation)
Balance Controller (100 Hz)
    ↓ triggers when joint states available
    ↓ publishes target CoM (30 Hz decimation)
Whole-Body Planner (30 Hz)
```

**Pros:**
- ✅ Minimal latency (immediate reaction)
- ✅ Automatically adapts to data rates
- ✅ No polling overhead

**Cons:**
- ❌ Can be overwhelmed by fast producers
- ❌ Timing becomes implicit
- ❌ Harder to analyze (non-deterministic)

### Pattern 3: Hybrid Time/Data Triggered

**Concept**: Combine time-triggered base with data-driven reactions.

```python
class HybridLoop:
    def __init__(self, base_frequency_hz):
        self.period = 1.0 / base_frequency_hz
        self.latest_data = {}

        # Subscribe to inputs (updates async)
        for topic in self.input_topics:
            self.session.declare_subscriber(
                topic,
                callback=lambda s: self.update_cache(s)
            )

    def update_cache(self, sample):
        # Async callback updates cache
        self.latest_data[sample.key_expr] = sample.payload

    def run(self):
        while True:
            # Time-triggered execution
            start = time.time()

            # Use latest cached data
            outputs = self.control_step(self.latest_data)
            self.publish_outputs(outputs)

            # Wait for next period
            self.wait_next_period(start)
```

**Pros:**
- ✅ Deterministic execution rate
- ✅ Uses latest available data
- ✅ Decouples producers from consumers

**Cons:**
- ⚠️ May use stale data
- ⚠️ Need to handle missing data

---

## Proposed Architecture: Multi-Rate Zenoh Framework

### Design Principles

1. **Explicit Rate Declaration**: Each component declares its execution rate
2. **Time Domain Isolation**: Fast loops don't block slow loops
3. **Zero-Copy Fast Path**: High-frequency loops use shared memory
4. **Typed Interfaces**: Strong typing between components
5. **Python for Slow, C++/Rust for Fast**: Language choice by rate
6. **Declarative Wiring**: Data flow specified separately from logic
7. **Observable**: Built-in tracing, profiling, recording

### Architecture Layers

```
┌─────────────────────────────────────────────────────────┐
│ Application Layer (Python)                              │
│ - User writes components with @control_loop decorator   │
│ - Declares rates, inputs, outputs                       │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ Framework Layer (Python + Rust bindings)                │
│ - Multi-rate scheduler                                  │
│ - Component lifecycle management                        │
│ - Data marshalling                                      │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ Zenoh Communication Layer (Rust)                        │
│ - Pub/sub with QoS                                      │
│ - Zero-copy shared memory                               │
│ - Time-series storage                                   │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ Hardware/OS Layer                                       │
│ - Real-time scheduler (PREEMPT_RT, Xenomai)            │
│ - DMA, memory locking                                   │
└─────────────────────────────────────────────────────────┘
```

### Component Model

```python
from embodied_ai_architect.framework import Component, Input, Output
from embodied_ai_architect.framework import control_loop
import zenoh
import numpy as np

@control_loop(
    rate_hz=1000,
    priority=99,  # RT priority
    cpu_affinity=[2, 3],  # Bind to specific cores
    deadline_ms=0.9  # Must complete within 0.9ms
)
class JointController(Component):
    """High-frequency joint control."""

    # Declare inputs (Zenoh subscriptions created automatically)
    joint_angles: Input[np.ndarray] = Input(
        topic="robot/sensors/joint/angles",
        qos=zenoh.QoS.reliable(),
        history=zenoh.KeepLast(1)
    )

    joint_velocities: Input[np.ndarray] = Input(
        topic="robot/sensors/joint/velocities"
    )

    target_position: Input[np.ndarray] = Input(
        topic="robot/control/joint/targets",
        default=np.zeros(12)  # Default if no data
    )

    # Declare outputs (Zenoh publishers created automatically)
    motor_torques: Output[np.ndarray] = Output(
        topic="robot/actuators/joint/torques",
        qos=zenoh.QoS.reliable()
    )

    def __init__(self):
        # PID controller state
        self.kp = 100.0
        self.kd = 10.0
        self.integral = np.zeros(12)

    def step(self, dt: float):
        """Called at 1000 Hz by framework."""

        # Framework automatically populates inputs
        angles = self.joint_angles.value
        velocities = self.joint_velocities.value
        targets = self.target_position.value

        # PID control
        error = targets - angles
        self.integral += error * dt
        control = self.kp * error + self.kd * velocities

        # Framework automatically publishes
        self.motor_torques.value = control

        # Optional: publish diagnostics at lower rate
        if self.tick_count % 100 == 0:  # Every 100ms
            self.publish_diagnostics()


@control_loop(
    rate_hz=30,
    priority=50
)
class PerceptionModule(Component):
    """Medium-frequency perception."""

    camera_image: Input[np.ndarray] = Input(
        topic="robot/sensors/camera/rgb"
    )

    detected_objects: Output[list] = Output(
        topic="robot/perception/objects"
    )

    def __init__(self):
        # Load DNN
        self.detector = torch.load("yolov5.pt")

    def step(self, dt: float):
        """Called at 30 Hz."""
        image = self.camera_image.value
        detections = self.detector(image)
        self.detected_objects.value = detections


@control_loop(
    rate_hz=5,
    priority=10
)
class PathPlanner(Component):
    """Low-frequency planning."""

    current_pose: Input[np.ndarray] = Input(
        topic="robot/state/pose"
    )

    detected_objects: Input[list] = Input(
        topic="robot/perception/objects"
    )

    goal: Input[np.ndarray] = Input(
        topic="robot/goal",
        default=np.array([0, 0, 0])
    )

    waypoints: Output[list] = Output(
        topic="robot/control/waypoints"
    )

    def __init__(self):
        self.planner = RRTStar()

    def step(self, dt: float):
        """Called at 5 Hz."""
        pose = self.current_pose.value
        obstacles = self.detected_objects.value
        goal = self.goal.value

        path = self.planner.plan(pose, goal, obstacles)
        self.waypoints.value = path
```

### Application Definition

```python
from embodied_ai_architect.framework import Application

# Define application
app = Application(
    name="quadruped_locomotion",
    zenoh_config="config/zenoh.json5"
)

# Register components
app.register(JointController())
app.register(PerceptionModule())
app.register(PathPlanner())

# Framework handles:
# - Creating Zenoh session
# - Setting up pub/sub for all inputs/outputs
# - Spawning threads for each control loop
# - Multi-rate scheduling
# - Data marshalling

# Run application
app.run()
```

### Framework Internals: Multi-Rate Scheduler

```python
class MultiRateScheduler:
    """Schedules components at different rates."""

    def __init__(self, zenoh_session: zenoh.Session):
        self.session = zenoh_session
        self.components: List[Component] = []
        self.threads: Dict[Component, Thread] = {}

    def register(self, component: Component):
        """Register a component."""
        self.components.append(component)

        # Create Zenoh publishers/subscribers for component
        self._setup_zenoh_bindings(component)

    def _setup_zenoh_bindings(self, component: Component):
        """Create Zenoh pub/sub for component I/O."""

        # For each Input field, create subscriber
        for field_name, input_spec in component.__inputs__.items():
            subscriber = self.session.declare_subscriber(
                input_spec.topic,
                mode=zenoh.PullMode(),  # Pull at component rate
                qos=input_spec.qos
            )
            component.__subscribers__[field_name] = subscriber

        # For each Output field, create publisher
        for field_name, output_spec in component.__outputs__.items():
            publisher = self.session.declare_publisher(
                output_spec.topic,
                qos=output_spec.qos
            )
            component.__publishers__[field_name] = publisher

    def run(self):
        """Start all components."""

        # Spawn thread for each component
        for component in self.components:
            thread = Thread(
                target=self._run_component,
                args=(component,),
                name=f"{component.__class__.__name__}_thread"
            )

            # Set real-time priority if specified
            if component.__rate_config__.priority:
                self._set_rt_priority(
                    thread,
                    component.__rate_config__.priority
                )

            # Set CPU affinity if specified
            if component.__rate_config__.cpu_affinity:
                self._set_cpu_affinity(
                    thread,
                    component.__rate_config__.cpu_affinity
                )

            thread.start()
            self.threads[component] = thread

        # Wait for all threads
        for thread in self.threads.values():
            thread.join()

    def _run_component(self, component: Component):
        """Run a single component at its specified rate."""

        rate_config = component.__rate_config__
        period_sec = 1.0 / rate_config.rate_hz

        # Use monotonic clock for accurate timing
        next_tick = time.monotonic()

        while not component.should_stop:
            tick_start = time.monotonic()

            # 1. Read inputs from Zenoh
            self._read_inputs(component)

            # 2. Execute component logic
            component.step(dt=period_sec)

            # 3. Write outputs to Zenoh
            self._write_outputs(component)

            # 4. Check deadline
            tick_end = time.monotonic()
            execution_time_ms = (tick_end - tick_start) * 1000

            if rate_config.deadline_ms:
                if execution_time_ms > rate_config.deadline_ms:
                    logging.warning(
                        f"{component} missed deadline: "
                        f"{execution_time_ms:.2f}ms > "
                        f"{rate_config.deadline_ms}ms"
                    )

            # 5. Sleep until next period
            next_tick += period_sec
            sleep_time = next_tick - time.monotonic()

            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # We're behind schedule
                logging.warning(f"{component} overran period")
                next_tick = time.monotonic()  # Reset

    def _read_inputs(self, component: Component):
        """Read latest data from Zenoh for all inputs."""
        for field_name, subscriber in component.__subscribers__.items():
            try:
                # Non-blocking receive, get latest
                sample = subscriber.recv(timeout=0)
                if sample:
                    # Deserialize and update component field
                    value = self._deserialize(sample.payload)
                    setattr(component, field_name, InputValue(value))
            except zenoh.Timeout:
                # No new data, keep previous value (or default)
                pass

    def _write_outputs(self, component: Component):
        """Publish outputs to Zenoh."""
        for field_name, publisher in component.__publishers__.items():
            output_value = getattr(component, field_name)
            if output_value.dirty:  # Only publish if changed
                payload = self._serialize(output_value.value)
                publisher.put(payload)
                output_value.dirty = False
```

---

## Rate-Specific Optimizations

### Ultra-Fast Loops (> 500 Hz): C++/Rust

For loops > 500 Hz, Python GIL and overhead becomes problematic. Use compiled languages:

```rust
// Rust component (compiled, no GC)
use zenoh::prelude::*;
use std::time::{Duration, Instant};

struct JointController {
    session: Arc<Session>,
    angle_sub: Subscriber,
    torque_pub: Publisher,
    kp: f64,
    kd: f64,
}

impl JointController {
    fn run(&mut self) {
        let period = Duration::from_micros(1000); // 1 kHz
        let mut next_tick = Instant::now() + period;

        loop {
            // Read sensor
            let angle = self.angle_sub.recv().unwrap();

            // PID control
            let error = self.target - angle;
            let torque = self.kp * error;

            // Publish (zero-copy if possible)
            self.torque_pub.put(torque).wait();

            // Sleep until next tick
            std::thread::sleep_until(next_tick);
            next_tick += period;
        }
    }
}
```

**Strategy: Hybrid Python/Rust**
- **Python**: Perception (30 Hz), Planning (5 Hz), orchestration
- **Rust**: Joint control (1000 Hz), balance control (200 Hz)
- **Communication**: All via Zenoh (language-agnostic)

```
Python Components          Rust Components
     (slow)                    (fast)
        ↓                         ↓
   Zenoh Python API        Zenoh Rust API
        ↓                         ↓
    ┌────────────────────────────────┐
    │    Zenoh Router (Rust)         │
    │  - Shared memory transport     │
    │  - Zero-copy where possible    │
    └────────────────────────────────┘
```

### Medium Loops (30-100 Hz): Optimized Python

Use:
- **NumPy** for vectorization
- **PyTorch** with CUDA for DNN inference
- **Cython/Numba** for hot paths
- **GIL release** where possible

```python
@control_loop(rate_hz=100)
class BalanceController(Component):
    def step(self, dt: float):
        # NumPy operations release GIL
        with nogil():  # Cython annotation
            state = np.array([...])
            control = self.lqr_gain @ state
```

### Slow Loops (< 30 Hz): Pure Python

Full expressiveness of Python:
- Async/await for I/O
- Rich libraries (path planning, optimization)
- Easy prototyping

---

## Real-Time Considerations

### Operating System Setup

**Linux RT-PREEMPT:**
```bash
# Install RT kernel
sudo apt install linux-image-rt-amd64

# Set thread priority
import os
os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(99))

# Lock memory
import mlock
mlock.mlockall()
```

**CPU Isolation:**
```bash
# Boot with isolated CPUs
# Add to kernel parameters: isolcpus=2,3

# Bind fast loops to isolated CPUs
component.cpu_affinity = [2, 3]
```

**DMA and Zero-Copy:**
```python
# Use Zenoh shared memory for zero-copy
config = {
    "transport": {
        "shared_memory": {
            "enabled": true
        }
    }
}
```

### Timing Guarantees

**Deadline Monitoring:**
```python
@control_loop(rate_hz=1000, deadline_ms=0.9)
class FastController(Component):
    def step(self, dt):
        # Framework monitors execution time
        # Logs/alerts if > 0.9ms
        pass
```

**Watchdog:**
```python
# Automatic restart if component hangs
@control_loop(rate_hz=100, watchdog_timeout_ms=50)
class CriticalComponent(Component):
    def step(self, dt):
        # Must complete within 50ms or watchdog triggers
        pass
```

---

## Example: Quadruped Robot Architecture

### System Decomposition

```
Quadruped Robot (ANYmal/Spot-like)

Rate Hierarchy:
┌──────────────────────────────────────────────────┐
│ Planning (5 Hz)                                  │
│ - Terrain mapping                                │
│ - Path planning (RRT*)                           │
│ - Gait selection                                 │
└──────────────────────────────────────────────────┘
          ↓ waypoints, gait params
┌──────────────────────────────────────────────────┐
│ Perception (30 Hz)                               │
│ - Vision (depth, RGB)                            │
│ - Terrain classification (DNN)                   │
│ - Obstacle detection                             │
└──────────────────────────────────────────────────┘
          ↓ terrain map, obstacles
┌──────────────────────────────────────────────────┐
│ Whole-Body Control (100 Hz)                     │
│ - Foot trajectory planning                       │
│ - Center of mass control                         │
│ - Contact force optimization                     │
└──────────────────────────────────────────────────┘
          ↓ joint targets
┌──────────────────────────────────────────────────┐
│ Joint Control (1000 Hz)                          │
│ - PD control per joint                           │
│ - Torque/position commands                       │
│ - Safety limits                                  │
└──────────────────────────────────────────────────┘
```

### Component Implementation

```python
# File: quadruped_app.py

from embodied_ai import *

# ========== 1 kHz: Joint Control (Rust) ==========
# Implemented in Rust, exposing Zenoh topics
# Source: src/joint_controller.rs (separate crate)

# ========== 100 Hz: Whole-Body Control ==========
@control_loop(rate_hz=100, priority=80)
class WholeBodyController(Component):
    # Inputs
    imu_data: Input[IMUData] = Input("robot/sensors/imu")
    joint_states: Input[JointStates] = Input("robot/sensors/joints")
    foot_contacts: Input[ContactForces] = Input("robot/sensors/contacts")
    target_velocity: Input[Twist] = Input("robot/control/cmd_vel")

    # Outputs
    joint_targets: Output[JointTargets] = Output("robot/control/joint_targets")

    def __init__(self):
        # Whole-body controller (MPC or simplified)
        self.controller = QuadrupedMPC(horizon=10)

    def step(self, dt):
        # State estimation
        state = self.estimate_state(
            self.imu_data.value,
            self.joint_states.value,
            self.foot_contacts.value
        )

        # MPC optimization
        targets = self.controller.solve(
            current_state=state,
            target_velocity=self.target_velocity.value,
            dt=dt
        )

        self.joint_targets.value = targets

# ========== 30 Hz: Perception ==========
@control_loop(rate_hz=30, priority=50)
class TerrainPerception(Component):
    depth_image: Input[np.ndarray] = Input("robot/sensors/camera/depth")
    rgb_image: Input[np.ndarray] = Input("robot/sensors/camera/rgb")

    terrain_map: Output[TerrainMap] = Output("robot/perception/terrain")

    def __init__(self):
        self.terrain_classifier = torch.load("terrain_net.pt")

    def step(self, dt):
        depth = self.depth_image.value
        rgb = self.rgb_image.value

        # DNN inference
        terrain_classes = self.terrain_classifier(rgb, depth)

        # Build elevation map
        terrain = self.build_map(depth, terrain_classes)

        self.terrain_map.value = terrain

# ========== 5 Hz: Planning ==========
@control_loop(rate_hz=5, priority=20)
class GlobalPlanner(Component):
    current_pose: Input[Pose] = Input("robot/state/pose")
    terrain_map: Input[TerrainMap] = Input("robot/perception/terrain")
    goal: Input[Pose] = Input("robot/goal")

    path: Output[Path] = Output("robot/plan/path")
    cmd_vel: Output[Twist] = Output("robot/control/cmd_vel")

    def __init__(self):
        self.planner = RRTStar(max_iterations=1000)
        self.path_follower = PurePursuitController()

    def step(self, dt):
        # Replan periodically
        if self.should_replan():
            path = self.planner.plan(
                start=self.current_pose.value,
                goal=self.goal.value,
                cost_map=self.terrain_map.value
            )
            self.path.value = path

        # Follow current path
        cmd_vel = self.path_follower.compute(
            pose=self.current_pose.value,
            path=self.path.value
        )
        self.cmd_vel.value = cmd_vel

# ========== Application ==========
app = Application(
    name="quadruped_locomotion",
    zenoh_config="config/zenoh.json5"
)

app.register(WholeBodyController())
app.register(TerrainPerception())
app.register(GlobalPlanner())

# Rust joint controller runs separately, communicates via Zenoh

if __name__ == "__main__":
    app.run()
```

### Zenoh Configuration

```json5
// config/zenoh.json5
{
  mode: "peer",  // Peer-to-peer mode

  transport: {
    // Use shared memory for local communication
    shared_memory: {
      enabled: true,
    },

    // Also support UDP for distributed
    udp: {
      enabled: true,
      multicast: {
        enabled: true,
        address: "224.0.0.1:7447"
      }
    }
  },

  // QoS defaults
  qos: {
    reliability: "reliable",
    priority: 5
  },

  // Storage for data recording
  plugins: {
    storage_manager: {
      storages: {
        recording: {
          key_expr: "robot/**",  // Record everything
          volume: "file_system",
          path: "/var/log/robot/recordings"
        }
      }
    }
  }
}
```

---

## Next Steps: Implementation Plan

### Week 1-2: Zenoh Prototype
- Set up Zenoh in Python
- Implement basic pub/sub
- Test zero-copy shared memory
- Benchmark latency at different rates

### Week 3-4: Component Framework
- Implement `@control_loop` decorator
- Create `Input`/`Output` descriptors
- Build multi-rate scheduler
- Add deadline monitoring

### Week 5-6: Rust Integration
- Create Rust joint controller example
- Zenoh bridge Python ↔ Rust
- Benchmark performance

### Week 7-8: Full Application
- Implement quadruped example
- End-to-end testing
- Performance profiling
- Real-time validation

---

## Open Questions

1. **Data Synchronization**: How to handle temporal alignment between different-rate loops?
2. **Fault Tolerance**: What happens if high-frequency loop crashes?
3. **Distributed Execution**: How to split components across machines?
4. **Hot Reloading**: Can we update slow components without restarting fast ones?
5. **Testing**: How to unit test components in isolation?

---

## References

- Zenoh: https://zenoh.io/
- Zenoh Python API: https://github.com/eclipse-zenoh/zenoh-python
- Zenoh Rust API: https://github.com/eclipse-zenoh/zenoh
- Linux RT-PREEMPT: https://wiki.linuxfoundation.org/realtime/start
- Multi-rate control: https://ieeexplore.ieee.org/document/8594277
