"""Trajectory predictor operator.

Predicts future object positions using linear or Kalman-based prediction.
"""

from typing import Any

import numpy as np

from ..base import Operator


class TrajectoryPredictor(Operator):
    """Predict future trajectories for tracked objects.

    Uses constant velocity model with optional acceleration estimation
    to predict object positions over a time horizon.
    """

    def __init__(self):
        super().__init__(operator_id="trajectory_predictor")
        self.prediction_horizon_s = 2.0
        self.prediction_steps = 10
        self.dt = 0.1

    def setup(self, config: dict[str, Any], execution_target: str = "cpu") -> None:
        """Initialize trajectory predictor.

        Args:
            config: Configuration with optional keys:
                - prediction_horizon_s: How far ahead to predict (seconds)
                - prediction_steps: Number of prediction points
                - use_acceleration: Whether to estimate acceleration
            execution_target: Only cpu supported
        """
        if execution_target != "cpu":
            print("[TrajectoryPredictor] Warning: Only CPU supported")

        self._execution_target = "cpu"
        self._config = config

        self.prediction_horizon_s = config.get("prediction_horizon_s", 2.0)
        self.prediction_steps = config.get("prediction_steps", 10)
        self.dt = self.prediction_horizon_s / self.prediction_steps
        self.use_acceleration = config.get("use_acceleration", False)

        # State history for velocity/acceleration estimation
        self._history: dict[int, list[tuple[float, list[float]]]] = {}
        self._max_history = 10

        self._is_setup = True
        print(f"[TrajectoryPredictor] Ready (horizon={self.prediction_horizon_s}s)")

    def process(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Predict trajectories for tracked objects.

        Args:
            inputs: Dictionary with:
                - 'tracked_object': Single object dict or list of objects
                - 'objects': Alternative key for list of objects
                - 'timestamp': Optional current timestamp

        Returns:
            Dictionary with:
                - 'predictions': List of trajectory predictions
                - 'prediction': Alias for predictions (single object case)
        """
        # Handle both single object and list inputs
        objects = inputs.get("tracked_object")
        if objects is None:
            objects = inputs.get("objects", [])
        if isinstance(objects, dict):
            objects = [objects]

        timestamp = inputs.get("timestamp", 0.0)

        predictions = []
        for obj in objects:
            obj_id = obj.get("id", 0)
            position = obj.get("position", [0.0, 0.0, 0.0])
            velocity = obj.get("velocity", None)

            # Update history
            if obj_id not in self._history:
                self._history[obj_id] = []
            self._history[obj_id].append((timestamp, position))
            if len(self._history[obj_id]) > self._max_history:
                self._history[obj_id].pop(0)

            # Estimate velocity if not provided
            if velocity is None:
                velocity = self._estimate_velocity(obj_id)

            # Generate prediction
            trajectory = self._predict_trajectory(position, velocity)

            predictions.append(
                {
                    "object_id": obj_id,
                    "current_position": position,
                    "velocity": velocity,
                    "predicted_trajectory": trajectory,
                    "prediction_horizon_s": self.prediction_horizon_s,
                    "timestamps": [
                        timestamp + i * self.dt for i in range(self.prediction_steps + 1)
                    ],
                }
            )

        return {
            "predictions": predictions,
            "prediction": predictions[0] if len(predictions) == 1 else predictions,
        }

    def _estimate_velocity(self, obj_id: int) -> list[float]:
        """Estimate velocity from position history."""
        history = self._history.get(obj_id, [])

        if len(history) < 2:
            return [0.0, 0.0, 0.0]

        # Use last two positions
        t1, p1 = history[-2]
        t2, p2 = history[-1]

        dt = t2 - t1 if t2 > t1 else 0.1  # Assume 0.1s if timestamps not available

        velocity = [
            (p2[0] - p1[0]) / dt,
            (p2[1] - p1[1]) / dt,
            (p2[2] - p1[2]) / dt,
        ]

        return velocity

    def _predict_trajectory(
        self,
        position: list[float],
        velocity: list[float],
    ) -> list[list[float]]:
        """Predict trajectory using constant velocity model."""
        trajectory = [position.copy()]

        current_pos = np.array(position)
        vel = np.array(velocity)

        for _ in range(self.prediction_steps):
            current_pos = current_pos + vel * self.dt
            trajectory.append(current_pos.tolist())

        return trajectory

    def reset(self):
        """Clear prediction history."""
        self._history = {}

    def teardown(self) -> None:
        """Clean up."""
        self.reset()
        self._is_setup = False
