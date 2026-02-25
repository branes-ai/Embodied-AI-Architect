"""PID controller operator.

Multi-axis PID control for real-time actuation.
"""

from typing import Any

import numpy as np

from ..base import Operator


class PIDController(Operator):
    """Multi-axis PID controller.

    Supports multiple independent control axes (e.g., 6-DOF robot joints).
    Includes anti-windup and output limiting.
    """

    def __init__(self):
        super().__init__(operator_id="pid_controller")
        self.num_axes = 1
        self.kp = None
        self.ki = None
        self.kd = None
        self.dt = 0.01
        self.integral = None
        self.prev_error = None
        self.output_min = -1.0
        self.output_max = 1.0

    def setup(self, config: dict[str, Any], execution_target: str = "cpu") -> None:
        """Initialize PID controller.

        Args:
            config: Configuration with keys:
                - kp: Proportional gain (scalar or per-axis list)
                - ki: Integral gain (scalar or per-axis list)
                - kd: Derivative gain (scalar or per-axis list)
                - dt: Control timestep in seconds
                - num_axes: Number of control axes (default: inferred from gains)
                - output_min: Minimum output value
                - output_max: Maximum output value
            execution_target: Only cpu supported
        """
        if execution_target != "cpu":
            print("[PIDController] Warning: Only CPU supported")

        self._execution_target = "cpu"
        self._config = config

        # Get gains
        kp = config.get("kp", 1.0)
        ki = config.get("ki", 0.0)
        kd = config.get("kd", 0.0)

        # Determine number of axes
        if isinstance(kp, (list, np.ndarray)):
            self.num_axes = len(kp)
            self.kp = np.array(kp)
            self.ki = (
                np.array(ki) if isinstance(ki, (list, np.ndarray)) else np.full(self.num_axes, ki)
            )
            self.kd = (
                np.array(kd) if isinstance(kd, (list, np.ndarray)) else np.full(self.num_axes, kd)
            )
        else:
            self.num_axes = config.get("num_axes", 1)
            self.kp = np.full(self.num_axes, kp)
            self.ki = np.full(self.num_axes, ki)
            self.kd = np.full(self.num_axes, kd)

        self.dt = config.get("dt", 0.01)
        self.output_min = config.get("output_min", -1.0)
        self.output_max = config.get("output_max", 1.0)

        # Initialize state
        self.integral = np.zeros(self.num_axes)
        self.prev_error = np.zeros(self.num_axes)

        self._is_setup = True
        print(f"[PIDController] Ready ({self.num_axes} axes, dt={self.dt}s)")

    def process(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Compute PID control output.

        Args:
            inputs: Dictionary with:
                - 'setpoint': Target value(s) (scalar or array)
                - 'measurement': Current value(s) (scalar or array)
                - 'path': Alternative to setpoint (uses first waypoint)
                - 'dt': Optional timestep override

        Returns:
            Dictionary with:
                - 'output': Control output(s)
                - 'error': Current error(s)
        """
        # Get setpoint - can come from 'setpoint', 'path', or default
        if "setpoint" in inputs:
            setpoint = np.atleast_1d(inputs["setpoint"]).flatten()
        elif "path" in inputs and inputs["path"]:
            # Use next waypoint from path planner (skip current position)
            path = inputs["path"]
            if isinstance(path, list) and len(path) > 1:
                # Take the second waypoint (first step from current)
                waypoint = np.atleast_1d(path[1]).flatten()
            elif isinstance(path, list) and len(path) == 1:
                waypoint = np.atleast_1d(path[0]).flatten()
            else:
                waypoint = np.zeros(self.num_axes)
            setpoint = waypoint
        else:
            setpoint = np.zeros(self.num_axes)

        # Ensure setpoint matches num_axes (take first N or pad)
        setpoint = np.atleast_1d(setpoint).flatten()
        if len(setpoint) > self.num_axes:
            setpoint = setpoint[: self.num_axes]
        elif len(setpoint) < self.num_axes:
            setpoint = np.pad(setpoint, (0, self.num_axes - len(setpoint)))

        # Get measurement - can come from 'measurement' or default to zeros
        if "measurement" in inputs:
            measurement = np.atleast_1d(inputs["measurement"]).flatten()
            # Ensure measurement matches num_axes
            if len(measurement) > self.num_axes:
                measurement = measurement[: self.num_axes]
            elif len(measurement) < self.num_axes:
                measurement = np.pad(measurement, (0, self.num_axes - len(measurement)))
        else:
            measurement = np.zeros(self.num_axes)

        dt = inputs.get("dt", self.dt)

        # Compute error
        error = setpoint - measurement

        # Proportional term
        p_term = self.kp * error

        # Integral term with anti-windup
        self.integral += error * dt
        # Anti-windup: limit integral
        integral_limit = (self.output_max - self.output_min) / (2 * self.ki + 1e-6)
        self.integral = np.clip(self.integral, -integral_limit, integral_limit)
        i_term = self.ki * self.integral

        # Derivative term
        derivative = (error - self.prev_error) / dt
        d_term = self.kd * derivative
        self.prev_error = error.copy()

        # Total output
        output = p_term + i_term + d_term
        output = np.clip(output, self.output_min, self.output_max)

        return {
            "output": output.tolist() if self.num_axes > 1 else float(output[0]),
            "error": error.tolist() if self.num_axes > 1 else float(error[0]),
        }

    def reset(self):
        """Reset controller state."""
        self.integral = np.zeros(self.num_axes)
        self.prev_error = np.zeros(self.num_axes)

    def teardown(self) -> None:
        """Clean up."""
        self.integral = None
        self.prev_error = None
        self._is_setup = False
