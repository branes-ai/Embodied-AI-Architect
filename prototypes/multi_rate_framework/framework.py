"""Minimal multi-rate control framework prototype.

This is a simplified implementation to validate the architecture.
"""

import threading
import time
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, TypeVar, Generic
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger(__name__)

# Type variable for generic Input/Output
T = TypeVar('T')


# ============================================================================
# Core Data Structures
# ============================================================================

@dataclass
class RateConfig:
    """Configuration for component execution rate."""
    rate_hz: float
    priority: Optional[int] = None
    deadline_ms: Optional[float] = None

    @property
    def period_sec(self) -> float:
        return 1.0 / self.rate_hz


# ============================================================================
# Input/Output Descriptors
# ============================================================================

class Input(Generic[T]):
    """Descriptor for component inputs (Zenoh subscriptions)."""

    def __init__(self, topic: str, default: Optional[T] = None):
        self.topic = topic
        self.default = default
        self.name = None  # Set by __set_name__

    def __set_name__(self, owner, name):
        self.name = name
        # Register input with owner class
        if not hasattr(owner, '__inputs__'):
            owner.__inputs__ = {}
        owner.__inputs__[name] = self

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        # Return cached value from component
        return obj.__input_cache__.get(self.name, self.default)


class Output(Generic[T]):
    """Descriptor for component outputs (Zenoh publishers)."""

    def __init__(self, topic: str):
        self.topic = topic
        self.name = None
        self._value = None
        self._dirty = False

    def __set_name__(self, owner, name):
        self.name = name
        # Register output with owner class
        if not hasattr(owner, '__outputs__'):
            owner.__outputs__ = {}
        owner.__outputs__[name] = self

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return self._value

    def __set__(self, obj, value):
        self._value = value
        self._dirty = True


# ============================================================================
# Component Base Class
# ============================================================================

class Component(ABC):
    """Base class for all control loop components."""

    def __init__(self):
        # Framework-managed attributes
        # Copy class-level __rate_config__ to instance if not already set
        if not hasattr(self, '__rate_config__'):
            self.__rate_config__ = getattr(self.__class__, '__rate_config__', None)

        self.__subscribers__: Dict[str, Any] = {}
        self.__publishers__: Dict[str, Any] = {}
        self.__input_cache__: Dict[str, Any] = {}

        # Component state
        self.tick_count: int = 0
        self.should_stop: bool = False
        self.last_execution_time_ms: float = 0

    @abstractmethod
    def step(self, dt: float) -> None:
        """Execute one control step.

        Args:
            dt: Time since last step (seconds)
        """
        pass

    def on_start(self) -> None:
        """Called once before first step."""
        pass

    def on_stop(self) -> None:
        """Called once after last step."""
        pass


# ============================================================================
# Decorator
# ============================================================================

def control_loop(rate_hz: float, priority: Optional[int] = None,
                deadline_ms: Optional[float] = None):
    """Decorator to mark a class as a control loop component.

    Args:
        rate_hz: Execution frequency in Hz
        priority: Thread priority (optional)
        deadline_ms: Maximum execution time in ms (optional)

    Example:
        @control_loop(rate_hz=100)
        class MyController(Component):
            def step(self, dt):
                pass
    """
    def decorator(cls):
        if not issubclass(cls, Component):
            raise TypeError(f"{cls.__name__} must inherit from Component")

        # Attach rate configuration
        cls.__rate_config__ = RateConfig(
            rate_hz=rate_hz,
            priority=priority,
            deadline_ms=deadline_ms
        )

        return cls

    return decorator


# ============================================================================
# Multi-Rate Scheduler
# ============================================================================

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

        logger.info(
            f"Registered {component.__class__.__name__} "
            f"at {component.__rate_config__.rate_hz} Hz"
        )

    def _setup_zenoh_bindings(self, component: Component):
        """Create Zenoh pub/sub for component I/O."""

        # Create subscribers for inputs
        if hasattr(component.__class__, '__inputs__'):
            for field_name, input_desc in component.__class__.__inputs__.items():
                subscriber = self.session.declare_subscriber(
                    input_desc.topic,
                    lambda sample, comp=component, fn=field_name:
                        self._on_data(comp, fn, sample)
                )
                component.__subscribers__[field_name] = subscriber
                logger.debug(f"  Subscribed to {input_desc.topic}")

        # Create publishers for outputs
        if hasattr(component.__class__, '__outputs__'):
            for field_name, output_desc in component.__class__.__outputs__.items():
                publisher = self.session.declare_publisher(output_desc.topic)
                component.__publishers__[field_name] = publisher
                logger.debug(f"  Publishing to {output_desc.topic}")

    def _on_data(self, component: Component, field_name: str, sample):
        """Callback when new data arrives (updates cache asynchronously)."""
        # Simple deserialization (just use bytes as-is for prototype)
        try:
            # In Zenoh 1.6+, payload is ZBytes, convert to bytes first
            payload_bytes = bytes(sample.payload)
            value = payload_bytes.decode('utf-8')

            # Try to convert to int/float if possible
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass  # Keep as string

            # Update cache (thread-safe with GIL)
            component.__input_cache__[field_name] = value
        except Exception as e:
            logger.error(f"Error deserializing data: {e}")

    def run(self):
        """Start all components."""

        # Call on_start for all components
        for component in self.components:
            component.on_start()

        # Spawn thread for each component
        for component in self.components:
            thread = threading.Thread(
                target=self._run_component,
                args=(component,),
                name=f"{component.__class__.__name__}",
                daemon=True
            )
            thread.start()
            self.threads[component] = thread

        logger.info(f"Started {len(self.components)} components")

        # Wait for stop signal
        try:
            self._stop_event.wait()
        except KeyboardInterrupt:
            logger.info("Shutdown requested (Ctrl+C)")

        # Stop all components
        for component in self.components:
            component.should_stop = True

        # Wait for threads
        for thread in self.threads.values():
            thread.join(timeout=2.0)

        # Call on_stop
        for component in self.components:
            component.on_stop()

        logger.info("All components stopped")

    def _run_component(self, component: Component):
        """Run a component at its specified rate."""

        rate_config = component.__rate_config__
        period_sec = rate_config.period_sec

        # Use monotonic clock for stable timing
        next_tick = time.monotonic()

        logger.info(
            f"Starting {component.__class__.__name__} "
            f"at {rate_config.rate_hz} Hz "
            f"(period={period_sec*1000:.1f}ms)"
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
            execution_time_ms = (tick_end - tick_start) * 1000
            component.last_execution_time_ms = execution_time_ms

            # Check deadline
            if rate_config.deadline_ms and execution_time_ms > rate_config.deadline_ms:
                logger.warning(
                    f"{component.__class__.__name__} "
                    f"deadline miss: {execution_time_ms:.2f}ms > "
                    f"{rate_config.deadline_ms}ms"
                )

            # Update tick count
            component.tick_count += 1

            # Sleep until next period
            next_tick += period_sec
            sleep_time = next_tick - time.monotonic()

            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # Overran
                jitter_ms = -sleep_time * 1000
                if jitter_ms > 1.0:  # Only log if significant
                    logger.warning(
                        f"{component.__class__.__name__} "
                        f"overran by {jitter_ms:.2f}ms"
                    )
                next_tick = time.monotonic()

    def _publish_outputs(self, component: Component):
        """Publish component outputs to Zenoh."""
        if not hasattr(component.__class__, '__outputs__'):
            return

        for field_name, output_desc in component.__class__.__outputs__.items():
            output = getattr(component.__class__, field_name)

            if output._dirty:
                # Simple serialization (convert to string)
                value = str(output._value)
                payload = value.encode('utf-8')

                # Publish
                publisher = component.__publishers__[field_name]
                publisher.put(payload)

                # Clear dirty flag
                output._dirty = False

    def stop(self):
        """Stop all components."""
        self._stop_event.set()


# ============================================================================
# Application Container
# ============================================================================

class Application:
    """Container for a multi-component application."""

    def __init__(self, name: str):
        """Initialize application.

        Args:
            name: Application name
        """
        self.name = name

        # Create Zenoh session (peer mode for simplicity)
        try:
            import zenoh
            config = zenoh.Config()
            self.session = zenoh.open(config)
            self.scheduler = MultiRateScheduler(self.session)
            logger.info(f"Application '{name}' initialized with Zenoh")
        except ImportError:
            logger.error("Zenoh not installed. Run: pip install eclipse-zenoh")
            raise

    def register(self, component: Component):
        """Register a component.

        Args:
            component: Component instance to register
        """
        self.scheduler.register(component)

    def run(self):
        """Run the application (blocks until stopped)."""
        logger.info(f"Starting application '{self.name}'")
        logger.info("Press Ctrl+C to stop")
        self.scheduler.run()

    def stop(self):
        """Stop the application."""
        self.scheduler.stop()
