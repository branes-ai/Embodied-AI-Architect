"""Scene graph manager for tracking 3D objects."""

from typing import Dict, List, Optional
import time

import numpy as np
from filterpy.kalman import KalmanFilter

from ..common import Track, TrackedObject, Frame, project_2d_to_3d, estimate_depth_from_bbox_height


class Object3DKalman:
    """Kalman filter for 3D object state (position, velocity, acceleration)."""

    def __init__(self, initial_position: np.ndarray):
        """
        Initialize 3D Kalman filter.

        State: [x, y, z, vx, vy, vz, ax, ay, az] (9D)
        Measurement: [x, y, z] (3D position only)
        """
        self.kf = KalmanFilter(dim_x=9, dim_z=3)

        # State transition matrix (constant acceleration model)
        dt = 1.0 / 30.0  # Assume 30 FPS
        self.kf.F = np.array([
            [1, 0, 0, dt, 0, 0, 0.5*dt**2, 0, 0],
            [0, 1, 0, 0, dt, 0, 0, 0.5*dt**2, 0],
            [0, 0, 1, 0, 0, dt, 0, 0, 0.5*dt**2],
            [0, 0, 0, 1, 0, 0, dt, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, dt, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, dt],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
        ])

        # Measurement function (we observe position only)
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0],
        ])

        # Initial state
        self.kf.x = np.zeros(9)
        self.kf.x[:3] = initial_position

        # Covariance matrices
        self.kf.P *= 1000.0  # Initial uncertainty
        self.kf.R = np.eye(3) * 0.5  # Measurement noise
        self.kf.Q = np.eye(9) * 0.01  # Process noise

    def predict(self):
        """Predict next state."""
        self.kf.predict()

    def update(self, position: np.ndarray):
        """Update with new position measurement."""
        self.kf.update(position)

    def get_state(self) -> tuple:
        """Get current state (position, velocity, acceleration)."""
        x = self.kf.x
        position = x[:3]
        velocity = x[3:6]
        acceleration = x[6:9]
        return position, velocity, acceleration


class SceneGraphManager:
    """
    Manages 3D scene graph of tracked objects.

    Responsibilities:
    - Convert 2D tracks to 3D world positions
    - Estimate velocity and acceleration using Kalman filters
    - Maintain object history for trajectory visualization
    - Prune stale objects
    """

    # Default object heights (meters) for monocular depth estimation.
    # Covers all classes in DRONE_CLASSES from yolo.py.
    OBJECT_HEIGHTS = {
        'person': 1.7,
        'car': 1.5,
        'truck': 2.5,
        'bus': 3.0,
        'bicycle': 1.5,
        'motorcycle': 1.2,
        'bird': 0.3,
        'cat': 0.3,
        'dog': 0.5,
        'default': 1.0,
    }

    def __init__(self, ttl_seconds: float = 5.0):
        """
        Initialize scene graph.

        Args:
            ttl_seconds: Time-to-live for objects (seconds)
        """
        self.objects: Dict[int, TrackedObject] = {}
        self.kalman_filters: Dict[int, Object3DKalman] = {}
        self.ttl_seconds = ttl_seconds

    def update(
        self,
        tracks: List[Track],
        frame: Frame,
        depth_estimation_mode: str = 'heuristic'
    ):
        """
        Update scene graph with new tracks.

        Args:
            tracks: List of 2D tracks from tracker
            frame: Current frame with camera params and optional depth
            depth_estimation_mode: 'heuristic' (bbox height) or 'depth_map'
        """
        current_time = time.time()

        for track in tracks:
            # Get depth
            if depth_estimation_mode == 'depth_map' and frame.depth is not None:
                depth = self._get_depth_from_map(track, frame.depth)
            else:
                # Heuristic: estimate from bbox height
                object_height = self.OBJECT_HEIGHTS.get(
                    track.class_name.lower(),
                    self.OBJECT_HEIGHTS['default']
                )
                depth = estimate_depth_from_bbox_height(
                    track.bbox.h,
                    object_height,
                    frame.camera_params
                )

            # Clamp to reasonable range (0.5m – 200m)
            depth = max(0.5, min(depth, 200.0))

            # Project to 3D
            cx, cy = track.bbox.center
            position_3d = project_2d_to_3d(
                cx, cy, depth, frame.camera_params
            )

            # Update or create tracked object
            if track.id not in self.objects:
                # New object
                obj = TrackedObject(
                    track_id=track.id,
                    class_name=track.class_name,
                    position=position_3d,
                    timestamp=current_time,
                    confidence=track.confidence,
                    last_seen=current_time,
                    age=track.age
                )
                self.objects[track.id] = obj

                # Initialize Kalman filter
                self.kalman_filters[track.id] = Object3DKalman(position_3d)
            else:
                # Existing object - update with Kalman filter
                kf = self.kalman_filters[track.id]
                kf.update(position_3d)

                # Get filtered state
                pos, vel, acc = kf.get_state()

                obj = self.objects[track.id]
                obj.position = pos
                obj.velocity = vel
                obj.acceleration = acc
                obj.timestamp = current_time
                obj.last_seen = current_time
                obj.confidence = track.confidence
                obj.age = track.age

                # Update history
                obj.position_history.append(pos.copy())
                if len(obj.position_history) > obj.max_history:
                    obj.position_history.pop(0)

        # Predict for objects not updated this frame
        updated_ids = {track.id for track in tracks}
        for obj_id, kf in self.kalman_filters.items():
            if obj_id not in updated_ids:
                kf.predict()
                pos, vel, acc = kf.get_state()
                self.objects[obj_id].position = pos
                self.objects[obj_id].velocity = vel
                self.objects[obj_id].acceleration = acc

        # Prune stale objects
        self._prune_stale_objects(current_time)

    def _get_depth_from_map(self, track: Track, depth_map: np.ndarray) -> float:
        """Get depth from depth map at track location."""
        cx, cy = track.bbox.center
        cx, cy = int(cx), int(cy)

        # Clamp to image bounds
        h, w = depth_map.shape
        cx = max(0, min(w - 1, cx))
        cy = max(0, min(h - 1, cy))

        # Get median depth in bbox (more robust than single pixel)
        x1, y1, x2, y2 = track.bbox.to_xyxy()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        bbox_depth = depth_map[y1:y2, x1:x2]
        valid_depth = bbox_depth[bbox_depth > 0]

        if len(valid_depth) > 0:
            return np.median(valid_depth)
        else:
            return 5.0  # Default fallback

    def _prune_stale_objects(self, current_time: float):
        """Remove objects that haven't been seen recently."""
        stale_ids = [
            obj_id for obj_id, obj in self.objects.items()
            if current_time - obj.last_seen > self.ttl_seconds
        ]

        for obj_id in stale_ids:
            del self.objects[obj_id]
            del self.kalman_filters[obj_id]

    def get_objects(self) -> List[TrackedObject]:
        """Get all active tracked objects."""
        return list(self.objects.values())

    def get_object(self, track_id: int) -> Optional[TrackedObject]:
        """Get specific object by track ID."""
        return self.objects.get(track_id)

    def reset(self):
        """Clear all objects."""
        self.objects.clear()
        self.kalman_filters.clear()

    def get_stats(self) -> dict:
        """Get scene statistics."""
        return {
            'total_objects': len(self.objects),
            'class_counts': self._get_class_counts(),
            'avg_velocity': self._get_avg_velocity(),
        }

    def _get_class_counts(self) -> dict:
        """Get count of objects by class."""
        counts = {}
        for obj in self.objects.values():
            counts[obj.class_name] = counts.get(obj.class_name, 0) + 1
        return counts

    def _get_avg_velocity(self) -> float:
        """Get average object velocity magnitude."""
        if not self.objects:
            return 0.0
        velocities = [np.linalg.norm(obj.velocity) for obj in self.objects.values()]
        return np.mean(velocities)

    # Convenience methods for reasoning pipeline
    def update_node(
        self,
        track_id: int,
        class_name: str,
        bbox,
        position_3d: np.ndarray,
        confidence: float
    ):
        """
        Update or create a single node (simplified interface for reasoning pipeline).

        Args:
            track_id: Unique track ID
            class_name: Object class name
            bbox: Bounding box
            position_3d: 3D position (x, y, z)
            confidence: Detection confidence
        """
        current_time = time.time()

        if track_id not in self.objects:
            # New object
            obj = TrackedObject(
                track_id=track_id,
                class_name=class_name,
                position=position_3d,
                timestamp=current_time,
                confidence=confidence,
                last_seen=current_time,
                age=1
            )
            self.objects[track_id] = obj
            self.kalman_filters[track_id] = Object3DKalman(position_3d)
        else:
            # Existing object - update with Kalman filter
            kf = self.kalman_filters[track_id]
            kf.update(position_3d)

            # Get filtered state
            pos, vel, acc = kf.get_state()

            obj = self.objects[track_id]
            obj.position = pos
            obj.velocity = vel
            obj.acceleration = acc
            obj.timestamp = current_time
            obj.last_seen = current_time
            obj.confidence = confidence

            # Update history
            obj.position_history.append(pos.copy())
            if len(obj.position_history) > obj.max_history:
                obj.position_history.pop(0)

    def get_active_nodes(self) -> List[TrackedObject]:
        """Get all active tracked objects (alias for get_objects)."""
        return self.get_objects()

    def get_node(self, track_id: int) -> Optional[TrackedObject]:
        """Get specific node by track ID (alias for get_object)."""
        return self.get_object(track_id)

    def num_nodes(self) -> int:
        """Get number of active nodes."""
        return len(self.objects)

    def clear_trajectories(self):
        """Clear all trajectory histories (keep current positions)."""
        for obj in self.objects.values():
            obj.position_history.clear()
            # Keep current position
            if obj.position is not None:
                obj.position_history.append(obj.position.copy())
        print("[SceneGraph] Cleared all trajectory histories")

    @property
    def trajectory_3d(self) -> List[np.ndarray]:
        """Compatibility property for reasoning modules."""
        # This is a workaround - reasoning modules expect this on nodes
        # In practice, each TrackedObject has position_history
        return []
