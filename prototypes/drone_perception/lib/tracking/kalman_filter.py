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
        self.dt = 1.0  # Time step (1 frame)

        # State: [cx, cy, s, r, vx, vy, vs, vr]
        self.mean = np.zeros(8)
        self.mean[:4] = self._bbox_to_z(bbox_xywh)

        # Covariance matrix
        self.covariance = np.eye(8)
        self.covariance[4:, 4:] *= 1000.0  # High uncertainty for velocities
        self.covariance[:4, :4] *= 10.0  # Moderate uncertainty for position

        # Motion model (constant velocity)
        self.motion_mat = np.eye(8)
        for i in range(4):
            self.motion_mat[i, i + 4] = self.dt

        # Measurement function (H matrix)
        self.update_mat = np.eye(4, 8)

        # Process noise
        self.std_weight_position = 1.0 / 20
        self.std_weight_velocity = 1.0 / 160

    def predict(self):
        """Predict next state."""
        # Generate process noise covariance Q
        std_pos = [
            self.std_weight_position * self.mean[2],  # std for cx
            self.std_weight_position * self.mean[2],  # std for cy
            self.std_weight_position * self.mean[2],  # std for s
            self.std_weight_position * self.mean[2],  # std for r
        ]
        std_vel = [
            self.std_weight_velocity * self.mean[2],  # std for vx
            self.std_weight_velocity * self.mean[2],  # std for vy
            self.std_weight_velocity * self.mean[2],  # std for vs
            self.std_weight_velocity * self.mean[2],  # std for vr
        ]

        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        # Predict
        self.mean = np.dot(self.motion_mat, self.mean)
        self.covariance = np.linalg.multi_dot((
            self.motion_mat, self.covariance, self.motion_mat.T
        )) + motion_cov

    def update(self, bbox_xywh: np.ndarray):
        """
        Update with new measurement.

        Args:
            bbox_xywh: Measured bbox [x, y, w, h]
        """
        measurement = self._bbox_to_z(bbox_xywh)

        # Measurement noise covariance R
        std = [
            self.std_weight_position * self.mean[2],
            self.std_weight_position * self.mean[2],
            self.std_weight_position * self.mean[2],
            self.std_weight_position * self.mean[2],
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
