"""
Collision detection and risk assessment for 3D trajectories.

Detects potential collisions between:
- Drone and tracked objects
- Predicted trajectories and obstacles
- Multiple tracked objects (for scene understanding)
"""

from typing import List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np

from .trajectory_predictor import PredictedTrajectory


class RiskLevel(Enum):
    """Collision risk levels."""
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class CollisionRisk:
    """Detected collision risk between two objects or trajectories."""

    # Objects involved
    object_a_id: int
    object_b_id: int
    object_a_class: str
    object_b_class: str

    # Risk assessment
    risk_level: RiskLevel
    time_to_collision: float  # seconds (inf if no collision)
    closest_distance: float  # meters
    collision_point: Optional[np.ndarray]  # (x, y, z) if collision predicted

    # Details
    relative_velocity: float  # m/s (closing speed)
    confidence: float  # 0-1

    def is_collision_imminent(self, threshold_time: float = 2.0) -> bool:
        """Check if collision is imminent (within threshold time)."""
        return self.time_to_collision < threshold_time and self.time_to_collision > 0


class CollisionDetector:
    """
    Detects potential collisions using 3D trajectories.

    Uses:
    - Predicted trajectories
    - Bounding volumes (spheres, boxes)
    - Time-to-collision estimation
    - Configurable safety margins
    """

    def __init__(
        self,
        safety_margin: float = 2.0,  # meters
        time_horizon: float = 3.0,  # seconds
        critical_distance: float = 1.0,  # meters
        warning_distance: float = 3.0,  # meters
    ):
        """
        Initialize collision detector.

        Args:
            safety_margin: Additional clearance for collision checking
            time_horizon: How far ahead to check for collisions
            critical_distance: Distance threshold for CRITICAL risk
            warning_distance: Distance threshold for HIGH risk
        """
        self.safety_margin = safety_margin
        self.time_horizon = time_horizon
        self.critical_distance = critical_distance
        self.warning_distance = warning_distance

    def check_trajectory_collision(
        self,
        traj_a: PredictedTrajectory,
        traj_b: PredictedTrajectory,
        object_radius_a: float = 0.5,
        object_radius_b: float = 0.5
    ) -> Optional[CollisionRisk]:
        """
        Check if two predicted trajectories will collide.

        Args:
            traj_a: First trajectory
            traj_b: Second trajectory
            object_radius_a: Bounding sphere radius for object A
            object_radius_b: Bounding sphere radius for object B

        Returns:
            CollisionRisk if collision detected, None otherwise
        """
        # Ensure trajectories have same time steps
        if len(traj_a.future_positions) != len(traj_b.future_positions):
            return None

        # Check distance at each timestep
        min_distance = float('inf')
        min_distance_idx = -1
        collision_detected = False

        combined_radius = object_radius_a + object_radius_b + self.safety_margin

        for i in range(len(traj_a.future_positions)):
            pos_a = traj_a.future_positions[i]
            pos_b = traj_b.future_positions[i]

            distance = np.linalg.norm(pos_a - pos_b)

            if distance < min_distance:
                min_distance = distance
                min_distance_idx = i

            # Check collision
            if distance < combined_radius:
                collision_detected = True
                break

        # Calculate time to collision
        if collision_detected:
            time_to_collision = traj_a.future_times[min_distance_idx]
            collision_point = (traj_a.future_positions[min_distance_idx] +
                             traj_b.future_positions[min_distance_idx]) / 2.0
        else:
            time_to_collision = float('inf')
            collision_point = None

        # Calculate relative velocity (closing speed)
        relative_vel = np.linalg.norm(traj_a.current_velocity - traj_b.current_velocity)

        # Determine risk level
        risk_level = self._assess_risk_level(
            min_distance,
            time_to_collision,
            relative_vel
        )

        # Skip if no risk
        if risk_level == RiskLevel.NONE:
            return None

        # Calculate confidence (lower if predictions are uncertain)
        confidence = min(traj_a.confidence, traj_b.confidence)

        return CollisionRisk(
            object_a_id=traj_a.track_id,
            object_b_id=traj_b.track_id,
            object_a_class=traj_a.class_name,
            object_b_class=traj_b.class_name,
            risk_level=risk_level,
            time_to_collision=time_to_collision,
            closest_distance=min_distance,
            collision_point=collision_point,
            relative_velocity=relative_vel,
            confidence=confidence
        )

    def check_point_collision(
        self,
        point: np.ndarray,
        trajectory: PredictedTrajectory,
        point_radius: float = 0.5,
        object_radius: float = 0.5
    ) -> Optional[CollisionRisk]:
        """
        Check if a static point (e.g., drone position) will collide with trajectory.

        Args:
            point: 3D point (e.g., drone position)
            trajectory: Predicted trajectory
            point_radius: Radius of point object
            object_radius: Radius of moving object

        Returns:
            CollisionRisk if collision detected
        """
        # Find closest approach
        min_distance = float('inf')
        min_distance_idx = -1

        combined_radius = point_radius + object_radius + self.safety_margin

        for i in range(len(trajectory.future_positions)):
            pos = trajectory.future_positions[i]
            distance = np.linalg.norm(pos - point)

            if distance < min_distance:
                min_distance = distance
                min_distance_idx = i

        # Check collision
        collision_detected = min_distance < combined_radius

        if collision_detected:
            time_to_collision = trajectory.future_times[min_distance_idx]
            collision_point = trajectory.future_positions[min_distance_idx]
        else:
            time_to_collision = float('inf')
            collision_point = None

        # Relative velocity is just object velocity (point is stationary)
        relative_vel = np.linalg.norm(trajectory.current_velocity)

        # Determine risk level
        risk_level = self._assess_risk_level(
            min_distance,
            time_to_collision,
            relative_vel
        )

        if risk_level == RiskLevel.NONE:
            return None

        return CollisionRisk(
            object_a_id=-1,  # Drone/static point
            object_b_id=trajectory.track_id,
            object_a_class="drone",
            object_b_class=trajectory.class_name,
            risk_level=risk_level,
            time_to_collision=time_to_collision,
            closest_distance=min_distance,
            collision_point=collision_point,
            relative_velocity=relative_vel,
            confidence=trajectory.confidence
        )

    def check_all_collisions(
        self,
        trajectories: List[PredictedTrajectory],
        drone_position: Optional[np.ndarray] = None,
        object_radii: Optional[dict] = None
    ) -> List[CollisionRisk]:
        """
        Check all pairwise collisions and drone collisions.

        Args:
            trajectories: List of predicted trajectories
            drone_position: Optional drone position to check against
            object_radii: Optional dict mapping track_id to radius

        Returns:
            List of all detected collision risks
        """
        risks = []

        # Default object radii by class
        default_radii = {
            "person": 0.3,
            "bicycle": 0.5,
            "car": 1.0,
            "motorcycle": 0.5,
            "bus": 2.0,
            "truck": 2.0,
            "traffic light": 0.2,
            "default": 0.5
        }

        def get_radius(traj: PredictedTrajectory) -> float:
            """Get object radius."""
            if object_radii and traj.track_id in object_radii:
                return object_radii[traj.track_id]
            return default_radii.get(traj.class_name, default_radii["default"])

        # Check drone collisions if position provided
        if drone_position is not None:
            for traj in trajectories:
                risk = self.check_point_collision(
                    drone_position,
                    traj,
                    point_radius=0.3,  # Drone radius
                    object_radius=get_radius(traj)
                )
                if risk is not None:
                    risks.append(risk)

        # Check pairwise collisions
        for i in range(len(trajectories)):
            for j in range(i + 1, len(trajectories)):
                traj_a = trajectories[i]
                traj_b = trajectories[j]

                risk = self.check_trajectory_collision(
                    traj_a,
                    traj_b,
                    object_radius_a=get_radius(traj_a),
                    object_radius_b=get_radius(traj_b)
                )

                if risk is not None:
                    risks.append(risk)

        # Sort by risk level (highest first)
        risks.sort(key=lambda r: r.risk_level.value, reverse=True)

        return risks

    def _assess_risk_level(
        self,
        distance: float,
        time_to_collision: float,
        relative_velocity: float
    ) -> RiskLevel:
        """
        Assess collision risk level based on distance, time, and velocity.

        Args:
            distance: Closest distance (meters)
            time_to_collision: Time until collision (seconds, inf if none)
            relative_velocity: Closing speed (m/s)

        Returns:
            RiskLevel enum
        """
        # Critical: Collision imminent
        if distance < self.critical_distance and time_to_collision < 1.0:
            return RiskLevel.CRITICAL

        # High: Close and approaching fast
        if distance < self.warning_distance and time_to_collision < 2.0:
            return RiskLevel.HIGH

        # Medium: Collision within time horizon
        if time_to_collision < self.time_horizon:
            return RiskLevel.MEDIUM

        # Low: Close but not approaching
        if distance < self.warning_distance * 2 and relative_velocity < 1.0:
            return RiskLevel.LOW

        # None: Safe
        return RiskLevel.NONE

    def get_avoidance_vector(
        self,
        drone_position: np.ndarray,
        risks: List[CollisionRisk]
    ) -> Optional[np.ndarray]:
        """
        Calculate avoidance vector for drone based on collision risks.

        Args:
            drone_position: Current drone position
            risks: List of collision risks

        Returns:
            3D avoidance vector (direction to move) or None if safe
        """
        if not risks:
            return None

        # Filter to critical/high risks involving drone
        critical_risks = [
            r for r in risks
            if r.object_a_id == -1 and r.risk_level.value >= RiskLevel.MEDIUM.value
        ]

        if not critical_risks:
            return None

        # Calculate avoidance direction (away from collision points)
        avoidance = np.zeros(3)

        for risk in critical_risks:
            if risk.collision_point is not None:
                # Direction away from collision point
                direction = drone_position - risk.collision_point
                distance = np.linalg.norm(direction)

                if distance > 0.01:
                    direction = direction / distance

                    # Weight by risk level
                    weight = risk.risk_level.value

                    avoidance += direction * weight

        # Normalize
        if np.linalg.norm(avoidance) > 0.01:
            avoidance = avoidance / np.linalg.norm(avoidance)
            return avoidance

        return None
