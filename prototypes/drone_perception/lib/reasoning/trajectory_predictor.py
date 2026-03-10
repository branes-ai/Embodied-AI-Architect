"""
Trajectory prediction for tracked 3D objects.

Predicts future positions and paths based on:
- Current position and velocity
- Historical trajectory
- Physics-based motion models (constant velocity, constant acceleration)
- Optional: Learning-based prediction (future work)
"""

from typing import List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from ..scene_graph import SceneGraphNode


@dataclass
class PredictedTrajectory:
    """Predicted future trajectory for an object."""

    track_id: int
    class_name: str

    # Current state
    current_position: np.ndarray  # (x, y, z)
    current_velocity: np.ndarray  # (vx, vy, vz)

    # Predicted future positions
    future_positions: np.ndarray  # (N, 3) - N future timesteps
    future_times: np.ndarray  # (N,) - Time deltas in seconds

    # Confidence/uncertainty
    confidence: float  # 0-1
    uncertainty: Optional[np.ndarray] = None  # (N, 3) - Standard deviation per position

    # Metadata
    prediction_method: str = "constant_velocity"

    @property
    def predicted_endpoint(self) -> np.ndarray:
        """Get final predicted position."""
        return self.future_positions[-1] if len(self.future_positions) > 0 else self.current_position

    @property
    def predicted_path_length(self) -> float:
        """Get total predicted path length in meters."""
        if len(self.future_positions) < 2:
            return 0.0

        positions = np.vstack([self.current_position, self.future_positions])
        diffs = np.diff(positions, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        return float(np.sum(distances))


class TrajectoryPredictor:
    """
    Predicts future trajectories for tracked objects.

    Supports multiple prediction methods:
    - Constant velocity: Assumes object continues at current velocity
    - Constant acceleration: Uses acceleration from velocity history
    - Physics-based: Considers physics constraints (gravity, friction)
    """

    def __init__(
        self,
        prediction_horizon: float = 3.0,  # seconds
        dt: float = 0.2,  # timestep (increased from 0.1 for fewer points)
        method: str = "constant_velocity",
        min_history_length: int = 3
    ):
        """
        Initialize trajectory predictor.

        Args:
            prediction_horizon: How far into future to predict (seconds)
            dt: Time step for predictions (seconds)
            method: Prediction method - "constant_velocity", "constant_acceleration", "physics"
            min_history_length: Minimum trajectory history needed for prediction
        """
        self.prediction_horizon = prediction_horizon
        self.dt = dt
        self.method = method
        self.min_history_length = min_history_length

        # Calculate number of prediction steps
        self.num_steps = int(prediction_horizon / dt)

    def predict(self, node: SceneGraphNode) -> Optional[PredictedTrajectory]:
        """
        Predict future trajectory for a scene graph node.

        Args:
            node: Scene graph node with 3D position and velocity

        Returns:
            PredictedTrajectory or None if insufficient data
        """
        # Check if we have enough data
        if node.position_3d is None:
            return None

        if node.velocity_3d is None or np.linalg.norm(node.velocity_3d) < 0.01:
            # Object is stationary - predict it stays in place
            return self._predict_stationary(node)

        # Check trajectory history length
        if len(node.trajectory_3d) < self.min_history_length:
            # Use simple constant velocity
            return self._predict_constant_velocity(node)

        # Use selected method
        if self.method == "constant_velocity":
            return self._predict_constant_velocity(node)
        elif self.method == "constant_acceleration":
            return self._predict_constant_acceleration(node)
        elif self.method == "physics":
            return self._predict_physics_based(node)
        else:
            raise ValueError(f"Unknown prediction method: {self.method}")

    def _predict_stationary(self, node: SceneGraphNode) -> PredictedTrajectory:
        """Predict stationary object (stays in place)."""
        future_positions = np.tile(node.position_3d, (self.num_steps, 1))
        future_times = np.arange(1, self.num_steps + 1) * self.dt

        return PredictedTrajectory(
            track_id=node.track_id,
            class_name=node.class_name,
            current_position=node.position_3d,
            current_velocity=np.zeros(3),
            future_positions=future_positions,
            future_times=future_times,
            confidence=0.9,  # High confidence for stationary objects
            prediction_method="stationary"
        )

    def _predict_constant_velocity(self, node: SceneGraphNode) -> PredictedTrajectory:
        """Predict using constant velocity model: p(t) = p0 + v*t"""
        position = node.position_3d
        velocity = node.velocity_3d

        # Generate future positions
        future_positions = []
        for i in range(1, self.num_steps + 1):
            t = i * self.dt
            future_pos = position + velocity * t
            future_positions.append(future_pos)

        future_positions = np.array(future_positions)
        future_times = np.arange(1, self.num_steps + 1) * self.dt

        # Confidence decreases with prediction horizon
        speed = np.linalg.norm(velocity)
        base_confidence = 0.8
        confidence = base_confidence * np.exp(-0.1 * self.prediction_horizon)

        return PredictedTrajectory(
            track_id=node.track_id,
            class_name=node.class_name,
            current_position=position,
            current_velocity=velocity,
            future_positions=future_positions,
            future_times=future_times,
            confidence=confidence,
            prediction_method="constant_velocity"
        )

    def _predict_constant_acceleration(self, node: SceneGraphNode) -> PredictedTrajectory:
        """Predict using constant acceleration: p(t) = p0 + v0*t + 0.5*a*t^2"""
        position = node.position_3d
        velocity = node.velocity_3d

        # Estimate acceleration from velocity history
        acceleration = self._estimate_acceleration(node)

        # Generate future positions
        future_positions = []
        for i in range(1, self.num_steps + 1):
            t = i * self.dt
            future_pos = position + velocity * t + 0.5 * acceleration * t**2
            future_positions.append(future_pos)

        future_positions = np.array(future_positions)
        future_times = np.arange(1, self.num_steps + 1) * self.dt

        # Lower confidence than constant velocity
        confidence = 0.7 * np.exp(-0.15 * self.prediction_horizon)

        return PredictedTrajectory(
            track_id=node.track_id,
            class_name=node.class_name,
            current_position=position,
            current_velocity=velocity,
            future_positions=future_positions,
            future_times=future_times,
            confidence=confidence,
            prediction_method="constant_acceleration"
        )

    def _predict_physics_based(self, node: SceneGraphNode) -> PredictedTrajectory:
        """
        Predict using physics constraints.

        Considers:
        - Gravity (for z-axis)
        - Ground plane constraint
        - Maximum acceleration limits
        """
        position = node.position_3d.copy()
        velocity = node.velocity_3d.copy()

        # Estimate acceleration
        acceleration = self._estimate_acceleration(node)

        # Physics parameters
        g = 9.81  # gravity (m/s^2)
        max_accel = 5.0  # max horizontal acceleration (m/s^2) for vehicles
        ground_z = 0.0  # ground plane

        # Generate future positions with physics
        future_positions = []
        current_pos = position.copy()
        current_vel = velocity.copy()
        current_acc = acceleration.copy()

        for i in range(1, self.num_steps + 1):
            t = self.dt

            # Apply acceleration limits
            current_acc[:2] = np.clip(current_acc[:2], -max_accel, max_accel)

            # Update velocity and position
            current_vel += current_acc * t
            current_pos += current_vel * t

            # Ground plane constraint (objects don't go underground)
            if current_pos[2] < ground_z:
                current_pos[2] = ground_z
                current_vel[2] = max(0, current_vel[2])  # No downward velocity

            future_positions.append(current_pos.copy())

        future_positions = np.array(future_positions)
        future_times = np.arange(1, self.num_steps + 1) * self.dt

        confidence = 0.75 * np.exp(-0.12 * self.prediction_horizon)

        return PredictedTrajectory(
            track_id=node.track_id,
            class_name=node.class_name,
            current_position=position,
            current_velocity=velocity,
            future_positions=future_positions,
            future_times=future_times,
            confidence=confidence,
            prediction_method="physics"
        )

    def _estimate_acceleration(self, node: SceneGraphNode) -> np.ndarray:
        """
        Estimate acceleration from velocity history.

        Uses linear regression on recent velocity samples.
        """
        trajectory = node.trajectory_3d

        if len(trajectory) < 2:
            return np.zeros(3)

        # Use last few positions to estimate velocity change
        recent_positions = trajectory[-min(5, len(trajectory)):]

        if len(recent_positions) < 2:
            return np.zeros(3)

        # Estimate velocities at each timestep
        velocities = []
        for i in range(1, len(recent_positions)):
            dt = 0.033  # Assume ~30 FPS
            vel = (recent_positions[i] - recent_positions[i-1]) / dt
            velocities.append(vel)

        if len(velocities) < 2:
            return np.zeros(3)

        # Estimate acceleration from velocity change
        velocities = np.array(velocities)
        dv = velocities[-1] - velocities[0]
        dt_total = 0.033 * (len(velocities) - 1)
        acceleration = dv / dt_total

        return acceleration

    def predict_all(self, nodes: List[SceneGraphNode]) -> List[PredictedTrajectory]:
        """
        Predict trajectories for all nodes.

        Args:
            nodes: List of scene graph nodes

        Returns:
            List of predicted trajectories (excluding None)
        """
        predictions = []
        for node in nodes:
            pred = self.predict(node)
            if pred is not None:
                predictions.append(pred)
        return predictions

    def visualize_trajectory(
        self,
        prediction: PredictedTrajectory,
        resolution: int = 50
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get visualization points for trajectory.

        Args:
            prediction: Predicted trajectory
            resolution: Number of points for smooth curve

        Returns:
            (positions, colors) - Arrays for visualization
        """
        # Interpolate for smooth visualization
        if len(prediction.future_positions) < 2:
            return prediction.future_positions, np.array([[0, 255, 0]])

        # Create smooth curve through predicted points
        all_positions = np.vstack([
            prediction.current_position,
            prediction.future_positions
        ])

        # Simple linear interpolation
        t_orig = np.linspace(0, 1, len(all_positions))
        t_smooth = np.linspace(0, 1, resolution)

        smooth_positions = np.column_stack([
            np.interp(t_smooth, t_orig, all_positions[:, 0]),
            np.interp(t_smooth, t_orig, all_positions[:, 1]),
            np.interp(t_smooth, t_orig, all_positions[:, 2])
        ])

        # Color gradient (green -> yellow -> red based on time)
        # OpenCV uses BGR format, so we generate in BGR
        colors = []
        for i in range(resolution):
            t = i / (resolution - 1)
            # Green to red gradient in BGR format
            r = int(255 * t)
            g = int(255 * (1 - t))
            b = 0
            colors.append([b, g, r])  # BGR format for OpenCV

        colors = np.array(colors)

        return smooth_positions, colors
