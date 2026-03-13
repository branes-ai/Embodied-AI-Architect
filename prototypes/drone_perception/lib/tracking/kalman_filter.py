"""Kalman filter for 2D bounding box tracking (used by ByteTrack)."""

import numpy as np


class KalmanBoxFilter:
    """
    Kalman filter for tracking 2D bounding boxes.

    State vector: [cx, cy, s, r, vx, vy, vs, vr]
    where:
        cx, cy: center coordinates
        s: scale (area)
        r: aspect ratio (width/height)
        vx, vy, vs, vr: velocities of the above
    """

    def __init__(self, bbox_xywh: np.ndarray):
        """
        Initialize Kalman filter with initial bounding box.

        Args:
            bbox_xywh: Initial bbox [x, y, w, h]
        """
        # State dimension: 8 (position + velocity)
        # Measurement dimension: 4 (position only)
        self.ndim = 4

        # State: [cx, cy, s, r, vx, vy, vs, vr]
        self.mean = np.zeros(8)
        self.mean[:4] = self._bbox_to_z(bbox_xywh)

        # Covariance matrix — trust initial measurement, high uncertainty for velocity
        self.covariance = np.eye(8)
        self.covariance[4:, 4:] *= 1000.0  # High uncertainty for velocities
        self.covariance[:4, :4] *= 1.0  # Trust initial position measurement

        # Measurement function (H matrix)
        self.update_mat = np.eye(4, 8)

        # Process noise weights — tuned for drone footage (more erratic motion than
        # ground-level pedestrian tracking)
        self.std_weight_position = 1.0 / 10
        self.std_weight_velocity = 1.0 / 40

    def predict(self, dt: float = None):
        """
        Predict next state.

        Args:
            dt: Time step in seconds. Defaults to 1/30 (~30fps).
        """
        if dt is None:
            dt = 1.0 / 30.0

        # Build motion model for this time step
        motion_mat = np.eye(8)
        for i in range(4):
            motion_mat[i, i + 4] = dt

        # Scale process noise by dt (larger steps → more uncertainty)
        dt_scale = dt * 30.0  # Normalize so dt=1/30 gives scale=1.0

        # Generate process noise covariance Q
        scale = max(self.mean[2], 1.0)  # Use bbox area as scale, floor at 1.0
        std_pos = [
            self.std_weight_position * scale * dt_scale,
            self.std_weight_position * scale * dt_scale,
            self.std_weight_position * scale * dt_scale,
            self.std_weight_position * scale * dt_scale,
        ]
        std_vel = [
            self.std_weight_velocity * scale * dt_scale,
            self.std_weight_velocity * scale * dt_scale,
            self.std_weight_velocity * scale * dt_scale,
            self.std_weight_velocity * scale * dt_scale,
        ]

        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        # Predict
        self.mean = np.dot(motion_mat, self.mean)
        self.covariance = np.linalg.multi_dot((
            motion_mat, self.covariance, motion_mat.T
        )) + motion_cov

    def update(self, bbox_xywh: np.ndarray):
        """
        Update with new measurement.

        Args:
            bbox_xywh: Measured bbox [x, y, w, h]
        """
        measurement = self._bbox_to_z(bbox_xywh)

        # Measurement noise covariance R
        scale = max(self.mean[2], 1.0)
        std = [
            self.std_weight_position * scale,
            self.std_weight_position * scale,
            self.std_weight_position * scale,
            self.std_weight_position * scale,
        ]
        innovation_cov = np.diag(np.square(std))

        # Kalman gain
        projected_cov = np.linalg.multi_dot((
            self.update_mat, self.covariance, self.update_mat.T
        ))
        kalman_gain = np.linalg.multi_dot((
            self.covariance,
            self.update_mat.T,
            np.linalg.inv(projected_cov + innovation_cov)
        ))

        # Update
        innovation = measurement - np.dot(self.update_mat, self.mean)
        self.mean = self.mean + np.dot(kalman_gain, innovation)
        self.covariance = self.covariance - np.linalg.multi_dot((
            kalman_gain, self.update_mat, self.covariance
        ))

    def get_bbox(self) -> np.ndarray:
        """Get current bbox [x, y, w, h]."""
        return self._z_to_bbox(self.mean[:4])

    @staticmethod
    def _bbox_to_z(bbox_xywh: np.ndarray) -> np.ndarray:
        """
        Convert bbox [x, y, w, h] to measurement [cx, cy, s, r].

        where s = area, r = aspect ratio
        """
        x, y, w, h = bbox_xywh
        cx = x + w / 2
        cy = y + h / 2
        s = w * h  # Scale (area)
        r = w / max(h, 1e-6)  # Aspect ratio
        return np.array([cx, cy, s, r])

    @staticmethod
    def _z_to_bbox(z: np.ndarray) -> np.ndarray:
        """
        Convert measurement [cx, cy, s, r] back to bbox [x, y, w, h].
        """
        cx, cy, s, r = z
        w = np.sqrt(s * r)
        h = s / max(w, 1e-6)
        x = cx - w / 2
        y = cy - h / 2
        return np.array([x, y, w, h])
