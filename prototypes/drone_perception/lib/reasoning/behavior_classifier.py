"""
Object behavior classification from 3D trajectories.

Classifies behaviors such as:
- Stationary vs moving
- Acceleration/deceleration
- Turning/straight movement
- Approaching/receding
- Stopping/starting
"""

from typing import List, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np

from ..scene_graph import SceneGraphNode


class BehaviorType(Enum):
    """Object behavior types."""
    STATIONARY = "stationary"
    MOVING_STRAIGHT = "moving_straight"
    TURNING = "turning"
    ACCELERATING = "accelerating"
    DECELERATING = "decelerating"
    STOPPING = "stopping"
    STARTING = "starting"
    APPROACHING = "approaching"
    RECEDING = "receding"
    HOVERING = "hovering"  # For aerial objects
    UNKNOWN = "unknown"


@dataclass
class ObjectBehavior:
    """Classified behavior for an object."""

    track_id: int
    class_name: str

    # Primary behavior
    behavior_type: BehaviorType
    confidence: float  # 0-1

    # Movement characteristics
    speed: float  # m/s
    acceleration: float  # m/s^2
    angular_velocity: float  # degrees/s

    # Direction
    heading_angle: float  # degrees (0-360)
    is_approaching: bool  # Approaching camera/drone

    # Additional context
    time_in_behavior: float  # seconds
    description: str  # Natural language description

    def is_potential_threat(self) -> bool:
        """Check if behavior indicates potential threat."""
        # Fast approaching objects are potential threats
        if self.is_approaching and self.speed > 5.0:
            return True
        # Rapid acceleration toward camera
        if self.acceleration > 2.0 and self.is_approaching:
            return True
        return False


class BehaviorClassifier:
    """
    Classifies object behaviors from 3D trajectories and velocities.

    Uses:
    - Velocity and acceleration analysis
    - Trajectory pattern recognition
    - Temporal behavior tracking
    """

    def __init__(
        self,
        stationary_threshold: float = 0.1,  # m/s
        straight_threshold: float = 5.0,  # degrees/s
        acceleration_threshold: float = 1.0,  # m/s^2
        min_history_length: int = 5
    ):
        """
        Initialize behavior classifier.

        Args:
            stationary_threshold: Speed below which object is stationary
            straight_threshold: Angular velocity below which movement is straight
            acceleration_threshold: Acceleration magnitude for accel/decel
            min_history_length: Minimum trajectory history for classification
        """
        self.stationary_threshold = stationary_threshold
        self.straight_threshold = straight_threshold
        self.acceleration_threshold = acceleration_threshold
        self.min_history_length = min_history_length

        # Track behavior history
        self.behavior_history = {}  # track_id -> list of (behavior, timestamp)

    def classify(
        self,
        node: SceneGraphNode,
        reference_position: Optional[np.ndarray] = None
    ) -> ObjectBehavior:
        """
        Classify behavior for an object.

        Args:
            node: Scene graph node
            reference_position: Optional reference point (e.g., drone position)

        Returns:
            ObjectBehavior classification
        """
        # Default unknown behavior
        if node.position_3d is None:
            return ObjectBehavior(
                track_id=node.track_id,
                class_name=node.class_name,
                behavior_type=BehaviorType.UNKNOWN,
                confidence=0.0,
                speed=0.0,
                acceleration=0.0,
                angular_velocity=0.0,
                heading_angle=0.0,
                is_approaching=False,
                time_in_behavior=0.0,
                description="Unknown (no position data)"
            )

        # Calculate movement characteristics
        speed = 0.0 if node.velocity_3d is None else np.linalg.norm(node.velocity_3d)
        acceleration_mag = self._estimate_acceleration_magnitude(node)
        angular_vel = self._estimate_angular_velocity(node)
        heading = self._estimate_heading(node)
        is_approaching = self._is_approaching(node, reference_position)

        # Classify primary behavior
        behavior_type, confidence = self._classify_behavior(
            speed,
            acceleration_mag,
            angular_vel,
            node
        )

        # Generate description
        description = self._generate_description(
            behavior_type,
            speed,
            acceleration_mag,
            is_approaching
        )

        # Estimate time in behavior
        time_in_behavior = self._estimate_time_in_behavior(
            node.track_id,
            behavior_type
        )

        return ObjectBehavior(
            track_id=node.track_id,
            class_name=node.class_name,
            behavior_type=behavior_type,
            confidence=confidence,
            speed=speed,
            acceleration=acceleration_mag,
            angular_velocity=angular_vel,
            heading_angle=heading,
            is_approaching=is_approaching,
            time_in_behavior=time_in_behavior,
            description=description
        )

    def _classify_behavior(
        self,
        speed: float,
        acceleration: float,
        angular_velocity: float,
        node: SceneGraphNode
    ) -> tuple[BehaviorType, float]:
        """
        Classify primary behavior type.

        Returns:
            (BehaviorType, confidence)
        """
        # Stationary
        if speed < self.stationary_threshold:
            return BehaviorType.STATIONARY, 0.9

        # Moving behaviors
        confidence = 0.7

        # Check acceleration
        if acceleration > self.acceleration_threshold:
            return BehaviorType.ACCELERATING, confidence

        if acceleration < -self.acceleration_threshold:
            # Could be decelerating or stopping
            if speed < self.stationary_threshold * 3:
                return BehaviorType.STOPPING, confidence
            else:
                return BehaviorType.DECELERATING, confidence

        # Check turning
        if angular_velocity > self.straight_threshold:
            return BehaviorType.TURNING, confidence

        # Default: Moving straight
        return BehaviorType.MOVING_STRAIGHT, confidence

    def _estimate_acceleration_magnitude(self, node: SceneGraphNode) -> float:
        """Estimate acceleration magnitude from velocity history."""
        trajectory = node.trajectory_3d

        if len(trajectory) < 2:
            return 0.0

        # Estimate velocities from positions
        velocities = []
        dt = 0.033  # Assume ~30 FPS

        for i in range(1, len(trajectory)):
            vel = (trajectory[i] - trajectory[i-1]) / dt
            velocities.append(vel)

        if len(velocities) < 2:
            return 0.0

        # Estimate acceleration from velocity change
        velocities = np.array(velocities)
        vel_magnitudes = np.linalg.norm(velocities, axis=1)

        # Linear regression or simple difference
        accel = (vel_magnitudes[-1] - vel_magnitudes[0]) / (len(vel_magnitudes) * dt)

        return float(accel)

    def _estimate_angular_velocity(self, node: SceneGraphNode) -> float:
        """Estimate angular velocity (turning rate) from trajectory."""
        trajectory = node.trajectory_3d

        if len(trajectory) < 3:
            return 0.0

        # Use last 3 points to estimate turning
        recent = trajectory[-3:]

        # Calculate angles between segments
        v1 = recent[1] - recent[0]
        v2 = recent[2] - recent[1]

        # Project to XY plane
        v1_xy = v1[:2]
        v2_xy = v2[:2]

        # Calculate angle change
        if np.linalg.norm(v1_xy) < 0.01 or np.linalg.norm(v2_xy) < 0.01:
            return 0.0

        v1_xy = v1_xy / np.linalg.norm(v1_xy)
        v2_xy = v2_xy / np.linalg.norm(v2_xy)

        # Angle between vectors
        cos_angle = np.clip(np.dot(v1_xy, v2_xy), -1.0, 1.0)
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)

        # Convert to angular velocity (degrees/second)
        dt = 0.033 * 2  # Time span for 3 points
        angular_vel = angle_deg / dt

        return float(angular_vel)

    def _estimate_heading(self, node: SceneGraphNode) -> float:
        """Estimate heading direction from velocity."""
        if node.velocity_3d is None:
            return 0.0

        vel_xy = node.velocity_3d[:2]

        if np.linalg.norm(vel_xy) < 0.01:
            return 0.0

        angle_rad = np.arctan2(vel_xy[1], vel_xy[0])
        angle_deg = np.degrees(angle_rad) % 360

        return float(angle_deg)

    def _is_approaching(
        self,
        node: SceneGraphNode,
        reference_position: Optional[np.ndarray]
    ) -> bool:
        """Check if object is approaching reference point."""
        if reference_position is None or node.velocity_3d is None:
            return False

        if node.position_3d is None:
            return False

        # Vector from object to reference
        to_ref = reference_position - node.position_3d

        # Dot product with velocity
        # Positive if moving toward reference
        dot = np.dot(to_ref, node.velocity_3d)

        return dot > 0

    def _generate_description(
        self,
        behavior: BehaviorType,
        speed: float,
        acceleration: float,
        is_approaching: bool
    ) -> str:
        """Generate natural language description."""
        desc_parts = []

        # Speed description
        if speed < 0.5:
            speed_desc = "stationary"
        elif speed < 2.0:
            speed_desc = "moving slowly"
        elif speed < 5.0:
            speed_desc = "moving"
        else:
            speed_desc = "moving fast"

        desc_parts.append(speed_desc)

        # Behavior description
        if behavior == BehaviorType.TURNING:
            desc_parts.append("and turning")
        elif behavior == BehaviorType.ACCELERATING:
            desc_parts.append("and accelerating")
        elif behavior == BehaviorType.DECELERATING:
            desc_parts.append("and slowing down")
        elif behavior == BehaviorType.STOPPING:
            desc_parts.append("and stopping")

        # Direction
        if is_approaching:
            desc_parts.append("(approaching)")
        else:
            desc_parts.append("(moving away)")

        return " ".join(desc_parts).capitalize()

    def _estimate_time_in_behavior(
        self,
        track_id: int,
        current_behavior: BehaviorType
    ) -> float:
        """Estimate how long object has been in current behavior."""
        # Simple implementation: track behavior changes
        if track_id not in self.behavior_history:
            self.behavior_history[track_id] = []

        history = self.behavior_history[track_id]

        # Add current behavior
        history.append(current_behavior)

        # Keep last 30 frames (~1 second at 30 FPS)
        if len(history) > 30:
            history.pop(0)

        # Count consecutive frames with same behavior
        count = 0
        for b in reversed(history):
            if b == current_behavior:
                count += 1
            else:
                break

        # Convert to time (assuming 30 FPS)
        time_seconds = count / 30.0

        return time_seconds

    def classify_all(
        self,
        nodes: List[SceneGraphNode],
        reference_position: Optional[np.ndarray] = None
    ) -> List[ObjectBehavior]:
        """
        Classify behaviors for all objects.

        Args:
            nodes: List of scene graph nodes
            reference_position: Optional reference point

        Returns:
            List of object behaviors
        """
        behaviors = []

        for node in nodes:
            behavior = self.classify(node, reference_position)
            behaviors.append(behavior)

        return behaviors
