"""Collision detection operator.

Detects potential collisions from trajectory predictions.
"""

from typing import Any

import numpy as np

from ..base import Operator


class CollisionDetector(Operator):
    """Detect potential collisions from predicted trajectories.

    Analyzes trajectory predictions to identify collision risks
    and compute time-to-collision estimates.
    """

    def __init__(self):
        super().__init__(operator_id="collision_detector")
        self.critical_distance = 2.0
        self.warning_distance = 5.0
        self.ego_radius = 0.5

    def setup(self, config: dict[str, Any], execution_target: str = "cpu") -> None:
        """Initialize collision detector.

        Args:
            config: Configuration with optional keys:
                - critical_distance: Distance for critical alert (meters)
                - warning_distance: Distance for warning alert (meters)
                - ego_radius: Radius of ego vehicle/robot (meters)
            execution_target: Only cpu supported
        """
        if execution_target != "cpu":
            print(f"[CollisionDetector] Warning: Only CPU supported")

        self._execution_target = "cpu"
        self._config = config

        self.critical_distance = config.get("critical_distance", 2.0)
        self.warning_distance = config.get("warning_distance", 5.0)
        self.ego_radius = config.get("ego_radius", 0.5)

        self._is_setup = True
        print(
            f"[CollisionDetector] Ready (critical={self.critical_distance}m, warning={self.warning_distance}m)"
        )

    def process(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Detect collisions from trajectory predictions.

        Args:
            inputs: Dictionary with:
                - 'predictions': List of trajectory predictions from TrajectoryPredictor
                - 'trajectories': Alternative key for predictions
                - 'ego_position': Optional ego position (default: origin)
                - 'ego_trajectory': Optional ego trajectory for moving ego

        Returns:
            Dictionary with:
                - 'collision_risks': List of collision risk assessments
                - 'has_critical': Whether any critical collision detected
                - 'has_warning': Whether any warning-level collision detected
                - 'closest_distance': Distance to closest object
                - 'time_to_collision': Estimated time to first collision (seconds)
        """
        predictions = inputs.get("predictions") or inputs.get("trajectories", [])
        if isinstance(predictions, dict):
            predictions = [predictions]

        ego_position = inputs.get("ego_position", [0.0, 0.0, 0.0])
        ego_trajectory = inputs.get("ego_trajectory")

        # If no ego trajectory, assume stationary
        if ego_trajectory is None:
            ego_trajectory = [ego_position for _ in range(20)]

        collision_risks = []
        closest_distance = float("inf")
        min_ttc = float("inf")
        has_critical = False
        has_warning = False

        for pred in predictions:
            obj_id = pred.get("object_id", 0)
            trajectory = pred.get("predicted_trajectory", [])
            timestamps = pred.get("timestamps", [])

            if not trajectory:
                continue

            # Analyze trajectory for collision risk
            min_dist = float("inf")
            collision_time = None
            collision_point = None

            for i, (obj_pos, ego_pos) in enumerate(
                zip(trajectory, ego_trajectory[: len(trajectory)])
            ):
                dist = np.sqrt(
                    (obj_pos[0] - ego_pos[0]) ** 2
                    + (obj_pos[1] - ego_pos[1]) ** 2
                    + (obj_pos[2] - ego_pos[2]) ** 2
                )

                if dist < min_dist:
                    min_dist = dist
                    if dist < self.critical_distance + self.ego_radius:
                        if i < len(timestamps):
                            collision_time = timestamps[i] - timestamps[0]
                        collision_point = obj_pos

            # Determine risk level
            if min_dist < self.critical_distance:
                risk_level = "critical"
                has_critical = True
            elif min_dist < self.warning_distance:
                risk_level = "warning"
                has_warning = True
            else:
                risk_level = "safe"

            closest_distance = min(closest_distance, min_dist)
            if collision_time is not None:
                min_ttc = min(min_ttc, collision_time)

            collision_risks.append(
                {
                    "object_id": obj_id,
                    "risk_level": risk_level,
                    "min_distance": min_dist,
                    "collision_time": collision_time,
                    "collision_point": collision_point,
                }
            )

        return {
            "collision_risks": collision_risks,
            "has_critical": has_critical,
            "has_warning": has_warning,
            "closest_distance": closest_distance if closest_distance != float("inf") else None,
            "time_to_collision": min_ttc if min_ttc != float("inf") else None,
        }

    def teardown(self) -> None:
        """Clean up."""
        self._is_setup = False
