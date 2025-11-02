# Phase 1: Multi-Rate Control Framework Architecture
## Concrete Implementation Proposal

**Date**: 2025-11-02
**Status**: Architecture Proposal
**Based On**: Multi-rate control research (docs/research/multi-rate-control-architecture.md)

---

## Executive Summary

This document proposes a concrete architecture for Phase 1 of the Embodied AI Application Framework, based on comprehensive research into multi-rate control systems. The framework enables building embodied AI applications with heterogeneous control loops running at different frequencies, using Zenoh as the communication backbone.

**Core Innovation**: Declarative multi-rate components with automatic Zenoh pub/sub wiring and hybrid Python/Rust execution.

---

## Architecture Decision

### Selected Pattern: Hybrid Time/Data Triggered

**Rationale:**
- ✅ **Deterministic**: Time-triggered ensures predictable execution
- ✅ **Low Latency**: Data cached asynchronously, always available
- ✅ **Simple Mental Model**: Each loop runs at fixed rate
- ✅ **Composable**: Loops don't block each other
- ✅ **Testable**: Can replay with recorded data

**How it works:**
1. Each component runs in its own thread at specified frequency
2. Zenoh subscribers update cached data asynchronously (callbacks)
3. Component reads latest cached data each tick (time-triggered)
4. Component publishes outputs to Zenoh
5. Framework handles all threading, scheduling, marshalling

---

## System Architecture

### Layer Stack

```
┌────────────────────────────────────────────────────────────┐
│ Application Layer                                          │
│ - User writes components with @control_loop                │
│ - Declares inputs/outputs with type annotations            │
│ - Implements step() method                                 │
│ - Pure Python for slow loops, Rust for fast loops          │
└────────────────────────────────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────┐
│ Framework Core (embodied_ai_architect.framework)           │
│ - Component lifecycle management                           │
│ - Multi-rate scheduler (one thread per component)          │
│ - Automatic Zenoh pub/sub creation                         │
│ - Data marshalling (Python ↔ bytes)                        │
│ - Deadline monitoring and diagnostics                      │
└────────────────────────────────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────┐
│ Zenoh Communication Layer (zenoh-python / zenoh-rust)      │
│ - Pub/sub with QoS                                         │
│ - Shared memory for zero-copy local communication          │
│ - UDP multicast for distributed communication              │
│ - Time-series storage for data recording                   │
└────────────────────────────────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────┐
│ Operating System Layer                                     │
│ - Thread scheduling (normal or RT)                         │
│ - Memory management                                        │
│ - Hardware access (sensors, actuators)                     │
└────────────────────────────────────────────────────────────┘
```

---

## Core Framework Components

### 1. Component Base Class

```python
# src/embodied_ai_architect/framework/component.py

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import time

class Component(ABC):
    """Base class for all control loop components."""

    def __init__(self):
        # Populated by framework
        self.__rate_config__: Optional[RateConfig] = None
        self.__subscribers__: Dict[str, Any] = {}
        self.__publishers__: Dict[str, Any] = {}
        self.__input_cache__: Dict[str, Any] = {}

        # Component state
        self.tick_count: int = 0
        self.should_stop: bool = False
        self.last_tick_time: float = 0

    @abstractmethod
    def step(self, dt: float) -> None:
        """Execute one control step.

        Args:
            dt: Time since last step (seconds)
        """
        pass

    def on_start(self) -> None:
        """Called once when component starts (before first step)."""
        pass

    def on_stop(self) -> None:
        """Called once when component stops (after last step)."""
        pass

    def on_deadline_miss(self, execution_time_ms: float) -> None:
        """Called when execution exceeds deadline.

        Args:
            execution_time_ms: Actual execution time
        """
        pass
```

### 2. Rate Configuration

```python
# src/embodied_ai_architect/framework/rate_config.py

from dataclasses import dataclass
from typing import List, Optional

@dataclass
class RateConfig:
    """Configuration for component execution rate."""

    rate_hz: float
    """Execution frequency in Hz."""

    priority: Optional[int] = None
    """Thread priority (0-99). Higher = more important. 99 = RT."""

    cpu_affinity: Optional[List[int]] = None
    """CPU cores to bind thread to (e.g., [2, 3])."""

    deadline_ms: Optional[float] = None
    """Maximum allowed execution time in milliseconds."""

    jitter_tolerance_ms: Optional[float] = None
    """Acceptable timing jitter in milliseconds."""

    @property
    def period_sec(self) -> float:
        """Period in seconds."""
        return 1.0 / self.rate_hz

    @property
    def period_ms(self) -> float:
        """Period in milliseconds."""
        return 1000.0 / self.rate_hz
```

### 3. Input/Output Descriptors

```python
# src/embodied_ai_architect/framework/io.py

from typing import TypeVar, Generic, Optional, Any
from dataclasses import dataclass

T = TypeVar('T')

@dataclass
class InputSpec(Generic[T]):
    """Specification for a component input."""

    topic: str
    """Zenoh topic to subscribe to."""

    default: Optional[T] = None
    """Default value if no data available."""

    required: bool = True
    """Whether input must have data before first step."""

    qos: Optional[str] = "reliable"
    """QoS policy: 'reliable' or 'best_effort'."""

    history: int = 1
    """Number of samples to keep (1 = latest only)."""


@dataclass
class OutputSpec(Generic[T]):
    """Specification for a component output."""

    topic: str
    """Zenoh topic to publish to."""

    qos: Optional[str] = "reliable"
    """QoS policy: 'reliable' or 'best_effort'."""

    publish_on_change: bool = False
    """Only publish if value changed since last tick."""


class Input(Generic[T]):
    """Input descriptor for component fields."""

    def __init__(
        self,
        topic: str,
        default: Optional[T] = None,
        required: bool = True,
        qos: str = "reliable",
        history: int = 1
    ):
        self.spec = InputSpec(
            topic=topic,
            default=default,
            required=required,
            qos=qos,
            history=history
        )

    def __set_name__(self, owner, name):
        self.name = name
        # Register with owner class
        if not hasattr(owner, '__inputs__'):
            owner.__inputs__ = {}
        owner.__inputs__[name] = self.spec

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        # Return cached value
        return obj.__input_cache__.get(self.name, self.spec.default)


class Output(Generic[T]):
    """Output descriptor for component fields."""

    def __init__(
        self,
        topic: str,
        qos: str = "reliable",
        publish_on_change: bool = False
    ):
        self.spec = OutputSpec(
            topic=topic,
            qos=qos,
            publish_on_change=publish_on_change
        )
        self._value = None
        self._dirty = False

    def __set_name__(self, owner, name):
        self.name = name
        # Register with owner class
        if not hasattr(owner, '__outputs__'):
            owner.__outputs__ = {}
        owner.__outputs__[name] = self.spec

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return self._value

    def __set__(self, obj, value):
        self._value = value
        self._dirty = True
```

### 4. Control Loop Decorator

```python
# src/embodied_ai_architect/framework/decorators.py

from typing import Type, Optional, List
from .component import Component
from .rate_config import RateConfig

def control_loop(
    rate_hz: float,
    priority: Optional[int] = None,
    cpu_affinity: Optional[List[int]] = None,
    deadline_ms: Optional[float] = None,
    jitter_tolerance_ms: Optional[float] = None
):
    """Decorator to mark a class as a control loop component.

    Args:
        rate_hz: Execution frequency in Hz
        priority: Thread priority (0-99, higher = more important)
        cpu_affinity: CPU cores to bind to
        deadline_ms: Maximum execution time
        jitter_tolerance_ms: Acceptable timing jitter

    Example:
        @control_loop(rate_hz=1000, priority=99)
        class JointController(Component):
            def step(self, dt):
                pass
    """
    def decorator(cls: Type[Component]) -> Type[Component]:
        # Validate that it's a Component subclass
        if not issubclass(cls, Component):
            raise TypeError(
                f"{cls.__name__} must inherit from Component"
            )

        # Attach rate configuration to class
        cls.__rate_config__ = RateConfig(
            rate_hz=rate_hz,
            priority=priority,
            cpu_affinity=cpu_affinity,
            deadline_ms=deadline_ms,
            jitter_tolerance_ms=jitter_tolerance_ms
        )

        return cls

    return decorator
```

### 5. Multi-Rate Scheduler

```python
# src/embodied_ai_architect/framework/scheduler.py

import threading
import time
import logging
from typing import List, Dict
from .component import Component

logger = logging.getLogger(__name__)

class MultiRateScheduler:
    """Schedules and executes components at different rates."""

    def __init__(self, zenoh_session):
        self.session = zenoh_session
        self.components: List[Component] = []
        self.threads: Dict[Component, threading.Thread] = {}
        self._stop_event = threading.Event()

    def register(self, component: Component):
        """Register a component to be scheduled."""
        self.components.append(component)
        self._setup_zenoh_bindings(component)

    def _setup_zenoh_bindings(self, component: Component):
        """Create Zenoh pub/sub for component I/O."""

        # Create subscribers for inputs
        for field_name, input_spec in component.__inputs__.items():
            subscriber = self.session.declare_subscriber(
                input_spec.topic,
                callback=lambda sample, fn=field_name:
                    self._on_data(component, fn, sample)
            )
            component.__subscribers__[field_name] = subscriber

        # Create publishers for outputs
        for field_name, output_spec in component.__outputs__.items():
            publisher = self.session.declare_publisher(
                output_spec.topic
            )
            component.__publishers__[field_name] = publisher

    def _on_data(self, component: Component, field_name: str, sample):
        """Async callback when new data arrives (updates cache)."""
        # Deserialize payload
        value = self._deserialize(sample.payload)

        # Update cache (thread-safe)
        component.__input_cache__[field_name] = value

    def run(self):
        """Start all components."""

        # Initialize all components
        for component in self.components:
            component.on_start()

        # Spawn thread for each component
        for component in self.components:
            thread = threading.Thread(
                target=self._run_component,
                args=(component,),
                name=f"{component.__class__.__name__}"
            )
            thread.daemon = True

            # Set thread priority if specified
            if component.__rate_config__.priority is not None:
                self._set_thread_priority(
                    thread,
                    component.__rate_config__.priority
                )

            thread.start()
            self.threads[component] = thread

        # Wait for stop signal
        try:
            self._stop_event.wait()
        except KeyboardInterrupt:
            logger.info("Shutdown requested")

        # Stop all components
        for component in self.components:
            component.should_stop = True

        # Wait for threads to finish
        for thread in self.threads.values():
            thread.join(timeout=5.0)

        # Cleanup
        for component in self.components:
            component.on_stop()

    def _run_component(self, component: Component):
        """Run a component at its specified rate."""

        rate_config = component.__rate_config__
        period_sec = rate_config.period_sec

        # Use monotonic clock for stable timing
        next_tick = time.monotonic()

        logger.info(
            f"Starting {component.__class__.__name__} "
            f"at {rate_config.rate_hz} Hz"
        )

        while not component.should_stop:
            tick_start = time.monotonic()

            # Execute component step
            try:
                component.step(dt=period_sec)
            except Exception as e:
                logger.error(
                    f"{component.__class__.__name__} error: {e}",
                    exc_info=True
                )

            # Publish outputs
            self._publish_outputs(component)

            # Timing diagnostics
            tick_end = time.monotonic()
            execution_time_sec = tick_end - tick_start
            execution_time_ms = execution_time_sec * 1000

            # Check deadline
            if rate_config.deadline_ms is not None:
                if execution_time_ms > rate_config.deadline_ms:
                    logger.warning(
                        f"{component.__class__.__name__} "
                        f"deadline miss: {execution_time_ms:.2f}ms > "
                        f"{rate_config.deadline_ms}ms"
                    )
                    component.on_deadline_miss(execution_time_ms)

            # Update tick count and time
            component.tick_count += 1
            component.last_tick_time = tick_start

            # Sleep until next period
            next_tick += period_sec
            sleep_time = next_tick - time.monotonic()

            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # Overran, log and reset
                jitter_ms = -sleep_time * 1000
                logger.warning(
                    f"{component.__class__.__name__} "
                    f"overran by {jitter_ms:.2f}ms"
                )
                next_tick = time.monotonic()

    def _publish_outputs(self, component: Component):
        """Publish component outputs to Zenoh."""
        for field_name, output_spec in component.__outputs__.items():
            output = getattr(component.__class__, field_name)

            if output._dirty or not output_spec.publish_on_change:
                # Serialize value
                payload = self._serialize(output._value)

                # Publish to Zenoh
                publisher = component.__publishers__[field_name]
                publisher.put(payload)

                # Clear dirty flag
                output._dirty = False

    def _serialize(self, value):
        """Serialize Python object to bytes."""
        # TODO: Use efficient serialization (msgpack, protobuf, etc.)
        import pickle
        return pickle.dumps(value)

    def _deserialize(self, payload):
        """Deserialize bytes to Python object."""
        import pickle
        return pickle.loads(bytes(payload))

    def _set_thread_priority(self, thread, priority):
        """Set real-time priority for thread (Linux only)."""
        try:
            import os
            # Set SCHED_FIFO with priority
            os.sched_setscheduler(
                0,
                os.SCHED_FIFO,
                os.sched_param(priority)
            )
        except (AttributeError, OSError) as e:
            logger.warning(
                f"Could not set RT priority: {e}. "
                "Run as root or configure RT limits."
            )

    def stop(self):
        """Stop all components."""
        self._stop_event.set()
```

### 6. Application Container

```python
# src/embodied_ai_architect/framework/application.py

import zenoh
import logging
from typing import Optional
from .component import Component
from .scheduler import MultiRateScheduler

logger = logging.getLogger(__name__)

class Application:
    """Container for a multi-component embodied AI application."""

    def __init__(
        self,
        name: str,
        zenoh_config: Optional[str] = None
    ):
        """Initialize application.

        Args:
            name: Application name
            zenoh_config: Path to Zenoh config file (JSON5)
        """
        self.name = name
        self.zenoh_config = zenoh_config

        # Create Zenoh session
        if zenoh_config:
            config = zenoh.Config.from_file(zenoh_config)
        else:
            config = zenoh.Config()

        self.session = zenoh.open(config)
        self.scheduler = MultiRateScheduler(self.session)

        logger.info(f"Application '{name}' initialized")

    def register(self, component: Component):
        """Register a component.

        Args:
            component: Component instance to register
        """
        self.scheduler.register(component)
        logger.info(
            f"Registered {component.__class__.__name__} "
            f"at {component.__rate_config__.rate_hz} Hz"
        )

    def run(self):
        """Run the application (blocks until stopped)."""
        logger.info(f"Starting application '{self.name}'")
        self.scheduler.run()

    def stop(self):
        """Stop the application."""
        logger.info(f"Stopping application '{self.name}'")
        self.scheduler.stop()
```

---

## Example: Simple Multi-Rate Application

```python
# examples/multi_rate_example.py

import numpy as np
from embodied_ai_architect.framework import (
    Component, Application, Input, Output, control_loop
)

@control_loop(rate_hz=1000)
class FastLoop(Component):
    """Fast control loop at 1kHz."""

    sensor_data: Input[float] = Input("robot/sensor/fast")
    control_output: Output[float] = Output("robot/actuator/fast")

    def __init__(self):
        super().__init__()
        self.state = 0.0

    def step(self, dt):
        # Read sensor (updated async by Zenoh)
        sensor = self.sensor_data or 0.0

        # Simple control law
        self.state += sensor * dt
        control = -self.state  # Negative feedback

        # Output
        self.control_output = control

        # Log every 1000 ticks (once per second)
        if self.tick_count % 1000 == 0:
            print(f"FastLoop: state={self.state:.3f}")


@control_loop(rate_hz=10)
class SlowLoop(Component):
    """Slow planning loop at 10Hz."""

    world_state: Input[dict] = Input("robot/world/state")
    goal: Input[np.ndarray] = Input("robot/goal", default=np.zeros(3))
    waypoint: Output[np.ndarray] = Output("robot/waypoint")

    def step(self, dt):
        # Planning computation (expensive)
        world = self.world_state or {}
        goal = self.goal

        # Dummy planner
        waypoint = goal * 0.1  # Move 10% towards goal

        self.waypoint = waypoint

        print(f"SlowLoop: waypoint={waypoint}")


# Create application
app = Application(
    name="multi_rate_test",
    zenoh_config="config/zenoh.json5"
)

# Register components
app.register(FastLoop())
app.register(SlowLoop())

# Run
if __name__ == "__main__":
    app.run()
```

---

## Implementation Roadmap

### Week 1-2: Core Framework
- ✅ Implement `Component` base class
- ✅ Implement `Input`/`Output` descriptors
- ✅ Implement `@control_loop` decorator
- ✅ Implement `RateConfig`
- ✅ Basic serialization (pickle)

### Week 3-4: Scheduler & Zenoh
- ✅ Implement `MultiRateScheduler`
- ✅ Zenoh session management
- ✅ Automatic pub/sub creation
- ✅ Async data caching
- ✅ Thread management

### Week 5-6: Application Container
- ✅ Implement `Application` class
- ✅ Component registration
- ✅ Lifecycle management
- ✅ Logging and diagnostics

### Week 7-8: Examples & Testing
- ✅ Multi-rate example
- ✅ Quadruped control example
- ✅ Unit tests
- ✅ Integration tests
- ✅ Performance benchmarks

---

## Testing Strategy

### Unit Tests
- Component lifecycle (start, step, stop)
- Input/Output descriptor behavior
- Rate config validation
- Serialization round-trips

### Integration Tests
- Multi-component communication via Zenoh
- Different rate combinations (1 Hz, 10 Hz, 100 Hz, 1000 Hz)
- Deadline monitoring
- Graceful shutdown

### Performance Tests
- Latency measurements (pub → sub)
- Throughput (messages/sec)
- CPU usage per component
- Jitter analysis

### Real-World Tests
- Quadruped locomotion (if hardware available)
- Simulated robot (Gazebo/PyBullet)

---

## Success Criteria

✅ **Phase 1 Complete When:**
1. Can define multi-rate components with decorators
2. Components communicate via Zenoh transparently
3. Scheduler maintains timing within 5% jitter
4. Example quadruped app runs successfully
5. Documentation and examples complete

---

## Next Steps

1. **Immediate**: Begin implementation of core framework classes
2. **Week 1**: Have basic Component + decorators working
3. **Week 2**: Multi-rate scheduler functional
4. **Week 4**: First multi-rate example running
5. **Week 8**: Phase 1 complete, move to Phase 2 (operators)

---

## Open Questions for Discussion

1. **Serialization**: Use pickle (slow but flexible) or msgpack/protobuf (fast but rigid)?
2. **Type Checking**: Use runtime checking or static type analysis (mypy)?
3. **Error Handling**: Restart failed components or shut down entire application?
4. **Configuration**: YAML for component parameters or Python only?
5. **Deployment**: How to package applications for deployment on robots?
