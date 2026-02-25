"""Path following operator using Pure Pursuit.

Computes velocity commands to follow a trajectory.
"""

from typing import Any

import numpy as np

from ..base import Operator


class PathFollower(Operator):
    """Pure Pursuit path follower.

    Computes linear and angular velocity commands to follow a waypoint path.
    """

    def __init__(self):
        super().__init__(operator_id="trajectory_follower")
        self.lookahead_distance = 1.0
        self.max_linear_vel = 1.0
        self.max_angular_vel = 2.0
        self.goal_tolerance = 0.1

    def setup(self, config: dict[str, Any], execution_target: str = "cpu") -> None:
        """Initialize path follower.

        Args:
            config: Configuration with optional keys:
                - lookahead_distance: Lookahead distance for pure pursuit
                - max_linear_vel: Maximum linear velocity (m/s)
                - max_angular_vel: Maximum angular velocity (rad/s)
                - goal_tolerance: Distance to consider goal reached
            execution_target: Only cpu supported
        """
        if execution_target != "cpu":
            print(f"[PathFollower] Warning: Only CPU supported")

        self._execution_target = "cpu"
        self._config = config

        self.lookahead_distance = config.get("lookahead_distance", 1.0)
        self.max_linear_vel = config.get("max_linear_vel", 1.0)
        self.max_angular_vel = config.get("max_angular_vel", 2.0)
        self.goal_tolerance = config.get("goal_tolerance", 0.1)

        self._is_setup = True
        print(f"[PathFollower] Ready (lookahead={self.lookahead_distance}m)")

    def process(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Compute velocity command to follow path.

        Args:
            inputs: Dictionary with:
                - 'pose': Current robot pose [x, y, theta]
                - 'path': List of waypoints [[x, y], ...] or [[x, y, theta], ...]

        Returns:
            Dictionary with:
                - 'velocity': Linear velocity (m/s)
                - 'angular_velocity': Angular velocity (rad/s)
                - 'lookahead_point': Target point on path
                - 'at_goal': Whether goal is reached
        """
        pose = np.array(inputs["pose"])
        path = np.array(inputs["path"])

        if len(path) == 0:
            return {
                "velocity": 0.0,
                "angular_velocity": 0.0,
                "lookahead_point": pose[:2].tolist(),
                "at_goal": True,
            }

        x, y, theta = pose

        # Check if at goal
        goal = path[-1]
        dist_to_goal = np.sqrt((goal[0] - x) ** 2 + (goal[1] - y) ** 2)

        if dist_to_goal < self.goal_tolerance:
            return {
                "velocity": 0.0,
                "angular_velocity": 0.0,
                "lookahead_point": goal[:2].tolist(),
                "at_goal": True,
            }

        # Find lookahead point
        lookahead_point = self._find_lookahead_point(pose, path)

        # Pure pursuit control
        v, omega = self._pure_pursuit(pose, lookahead_point)

        return {
            "velocity": float(v),
            "angular_velocity": float(omega),
            "lookahead_point": lookahead_point.tolist(),
            "at_goal": False,
        }

    def _find_lookahead_point(
        self,
        pose: np.ndarray,
        path: np.ndarray,
    ) -> np.ndarray:
        """Find lookahead point on path."""
        x, y = pose[:2]

        # Compute distances to all waypoints
        waypoints = path[:, :2]
        distances = np.sqrt(np.sum((waypoints - [x, y]) ** 2, axis=1))

        # Find closest point
        closest_idx = np.argmin(distances)

        # Search forward for lookahead point
        for i in range(closest_idx, len(path)):
            if distances[i] >= self.lookahead_distance:
                return waypoints[i]

        # Return last point if no point is far enough
        return waypoints[-1]

    def _pure_pursuit(
        self,
        pose: np.ndarray,
        lookahead: np.ndarray,
    ) -> tuple[float, float]:
        """Pure pursuit steering law.

        Args:
            pose: Robot pose [x, y, theta]
            lookahead: Lookahead point [x, y]

        Returns:
            (linear_velocity, angular_velocity)
        """
        x, y, theta = pose
        lx, ly = lookahead[:2]

        # Transform lookahead to robot frame
        dx = lx - x
        dy = ly - y
        local_x = dx * np.cos(theta) + dy * np.sin(theta)
        local_y = -dx * np.sin(theta) + dy * np.cos(theta)

        # Distance to lookahead
        L = np.sqrt(local_x**2 + local_y**2)

        if L < 0.01:
            return 0.0, 0.0

        # Curvature = 2 * y / L^2
        curvature = 2 * local_y / (L**2)

        # Velocity (reduce when turning sharply)
        v = self.max_linear_vel * (1 - abs(curvature) / 2)
        v = max(0.1, min(v, self.max_linear_vel))

        # Angular velocity
        omega = v * curvature
        omega = np.clip(omega, -self.max_angular_vel, self.max_angular_vel)

        return v, omega

    def teardown(self) -> None:
        """Clean up."""
        self._is_setup = False
