"""Multi-rate example with three components at different frequencies.

This example demonstrates a simplified control hierarchy:
- Sensor (100 Hz): Fast sensor reading simulation
- Controller (10 Hz): Medium-rate control loop
- Planner (1 Hz): Slow planning loop

Data flow:
  Sensor (100 Hz) → Controller (10 Hz) → Planner (1 Hz)
       ↓                  ↓                   ↓
    sensor_data      control_output         plan
"""

import math
from framework import Component, Application, Input, Output, control_loop


@control_loop(rate_hz=100, deadline_ms=5.0)
class SensorSimulator(Component):
    """Simulates a fast sensor (e.g., IMU) at 100 Hz."""

    sensor_value: Output[float] = Output("robot/sensor/imu")

    def __init__(self):
        super().__init__()
        self.time = 0.0

    def step(self, dt):
        """Generate synthetic sensor data."""
        # Simulate sinusoidal sensor reading
        self.sensor_value = math.sin(self.time * 2 * math.pi * 0.5)  # 0.5 Hz sine wave
        self.time += dt

        # Log every 100 ticks (once per second)
        if self.tick_count % 100 == 0:
            print(
                f"[Sensor @ 100 Hz] tick={self.tick_count} "
                f"value={self.sensor_value:.3f} "
                f"exec_time={self.last_execution_time_ms:.2f}ms"
            )


@control_loop(rate_hz=10, deadline_ms=50.0)
class Controller(Component):
    """Medium-rate controller at 10 Hz."""

    # Input from sensor
    sensor_value: Input[float] = Input("robot/sensor/imu", default=0.0)

    # Input from planner
    setpoint: Input[float] = Input("robot/plan/setpoint", default=0.0)

    # Output control signal
    control: Output[float] = Output("robot/control/output")

    def __init__(self):
        super().__init__()
        self.integral = 0.0
        self.last_error = 0.0
        # Simple PID gains
        self.kp = 1.0
        self.ki = 0.1
        self.kd = 0.05

    def step(self, dt):
        """PID control at 10 Hz."""
        # Read latest sensor value (updated at 100 Hz)
        current = self.sensor_value

        # Read setpoint from planner (updated at 1 Hz)
        target = self.setpoint

        # PID control
        error = target - current
        self.integral += error * dt
        derivative = (error - self.last_error) / dt

        control_output = (
            self.kp * error +
            self.ki * self.integral +
            self.kd * derivative
        )

        self.control = control_output
        self.last_error = error

        # Log every 10 ticks (once per second)
        if self.tick_count % 10 == 0:
            print(
                f"[Controller @ 10 Hz] tick={self.tick_count} "
                f"error={error:.3f} control={control_output:.3f} "
                f"exec_time={self.last_execution_time_ms:.2f}ms"
            )


@control_loop(rate_hz=1)
class Planner(Component):
    """Slow planning loop at 1 Hz."""

    # Input: current control output (for monitoring)
    control: Input[float] = Input("robot/control/output", default=0.0)

    # Output: setpoint for controller
    setpoint: Output[float] = Output("robot/plan/setpoint")

    def __init__(self):
        super().__init__()
        self.current_setpoint = 0.0

    def step(self, dt):
        """Update plan/setpoint at 1 Hz."""

        # Simple planner: alternate between +0.5 and -0.5 every 5 seconds
        if (self.tick_count % 5) == 0:
            self.current_setpoint = -self.current_setpoint if self.current_setpoint != 0 else 0.5

        self.setpoint = self.current_setpoint

        control_value = self.control

        print(
            f"[Planner @ 1 Hz] tick={self.tick_count} "
            f"setpoint={self.current_setpoint:.3f} "
            f"control={control_value:.3f} "
            f"exec_time={self.last_execution_time_ms:.2f}ms"
        )


def main():
    """Run the multi-rate example."""

    print("=" * 70)
    print("Multi-Rate Control Example")
    print("=" * 70)
    print("Sensor:     100 Hz (fast sensor simulation)")
    print("Controller:  10 Hz (PID control)")
    print("Planner:      1 Hz (setpoint planning)")
    print()
    print("Data Flow:")
    print("  Sensor → Controller → Planner")
    print("           ↑______________|")
    print()
    print("Press Ctrl+C to stop")
    print("=" * 70)
    print()

    # Create application
    app = Application(name="multirate_control")

    # Register components (order doesn't matter)
    app.register(SensorSimulator())
    app.register(Controller())
    app.register(Planner())

    # Run
    app.run()


if __name__ == "__main__":
    main()
