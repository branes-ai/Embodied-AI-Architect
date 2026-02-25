"""Extended Kalman Filter for 6-DOF pose estimation.

Fuses IMU data for position and orientation estimation.
"""

from typing import Any

import numpy as np

from ..base import Operator


class EKF6DOF(Operator):
    """Extended Kalman Filter for 6-DOF pose estimation.

    State: [x, y, z, roll, pitch, yaw, vx, vy, vz, wx, wy, wz]
    - Position (x, y, z)
    - Orientation as Euler angles (roll, pitch, yaw)
    - Linear velocity (vx, vy, vz)
    - Angular velocity (wx, wy, wz)
    """

    def __init__(self):
        super().__init__(operator_id="ekf_6dof")
        self.state = None
        self.covariance = None
        self.dt = 0.01
        self.process_noise = None
        self.measurement_noise = None

    def setup(self, config: dict[str, Any], execution_target: str = "cpu") -> None:
        """Initialize EKF.

        Args:
            config: Configuration with optional keys:
                - dt: Time step in seconds (default: 0.01)
                - process_noise: Process noise covariance diagonal
                - measurement_noise: Measurement noise covariance diagonal
                - initial_pose: Initial [x, y, z, roll, pitch, yaw]
            execution_target: Only cpu supported
        """
        if execution_target != "cpu":
            print(f"[EKF6DOF] Warning: Only CPU supported, ignoring target={execution_target}")

        self._execution_target = "cpu"
        self._config = config

        self.dt = config.get("dt", 0.01)

        # Initialize state [x, y, z, roll, pitch, yaw, vx, vy, vz, wx, wy, wz]
        initial_pose = config.get("initial_pose", [0.0] * 6)
        self.state = np.zeros(12)
        self.state[:6] = initial_pose

        # State covariance
        self.covariance = np.eye(12) * 0.1

        # Process noise (higher for velocities)
        process_noise_diag = config.get(
            "process_noise",
            [0.01] * 6 + [0.1] * 6,  # Low for pose, higher for velocities
        )
        self.process_noise = np.diag(process_noise_diag)

        # Measurement noise
        measurement_noise_diag = config.get(
            "measurement_noise",
            [0.1] * 3 + [0.01] * 3,  # Higher for accel, lower for gyro
        )
        self.measurement_noise = np.diag(measurement_noise_diag)

        self._is_setup = True
        print(f"[EKF6DOF] Ready (dt={self.dt}s)")

    def process(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Process IMU measurement and update state.

        Args:
            inputs: Dictionary with:
                - 'accel': Linear acceleration [ax, ay, az] in body frame
                - 'gyro': Angular velocity [wx, wy, wz] in body frame
                - 'dt': Optional time step override

        Returns:
            Dictionary with:
                - 'pose': [x, y, z, roll, pitch, yaw]
                - 'velocity': [vx, vy, vz, wx, wy, wz]
                - 'covariance': State covariance diagonal
        """
        accel = np.array(inputs["accel"])
        gyro = np.array(inputs["gyro"])
        dt = inputs.get("dt", self.dt)

        # Prediction step
        self._predict(dt)

        # Update step with IMU
        self._update_imu(accel, gyro)

        return {
            "pose": self.state[:6].tolist(),
            "velocity": self.state[6:].tolist(),
            "covariance": np.diag(self.covariance).tolist(),
        }

    def _predict(self, dt: float):
        """Prediction step using constant velocity model."""
        # State transition: integrate velocities
        F = np.eye(12)
        F[0, 6] = dt  # x += vx * dt
        F[1, 7] = dt  # y += vy * dt
        F[2, 8] = dt  # z += vz * dt
        F[3, 9] = dt  # roll += wx * dt
        F[4, 10] = dt  # pitch += wy * dt
        F[5, 11] = dt  # yaw += wz * dt

        # Predict state
        self.state = F @ self.state

        # Normalize angles
        self.state[3:6] = self._normalize_angles(self.state[3:6])

        # Predict covariance
        self.covariance = F @ self.covariance @ F.T + self.process_noise * dt

    def _update_imu(self, accel: np.ndarray, gyro: np.ndarray):
        """Update with IMU measurements."""
        # Get current orientation
        roll, pitch, yaw = self.state[3:6]

        # Rotation matrix from body to world
        R = self._rotation_matrix(roll, pitch, yaw)

        # Transform acceleration to world frame and remove gravity
        gravity = np.array([0, 0, 9.81])
        accel_world = R @ accel - gravity

        # Measurement model:
        # z = [ax_world, ay_world, az_world, wx, wy, wz]
        z = np.concatenate([accel_world, gyro])

        # Measurement matrix (maps state to measurement)
        # Accelerometer measures d(velocity)/dt, gyro measures angular velocity
        H = np.zeros((6, 12))
        # Accelerometer observes velocity change (simplified)
        H[0, 6] = 1 / self.dt  # ax ~ dvx/dt
        H[1, 7] = 1 / self.dt  # ay ~ dvy/dt
        H[2, 8] = 1 / self.dt  # az ~ dvz/dt
        # Gyro directly observes angular velocity
        H[3, 9] = 1  # wx
        H[4, 10] = 1  # wy
        H[5, 11] = 1  # wz

        # Kalman update
        y = z - H @ self.state  # Innovation
        S = H @ self.covariance @ H.T + self.measurement_noise  # Innovation covariance
        K = self.covariance @ H.T @ np.linalg.inv(S)  # Kalman gain

        self.state = self.state + K @ y
        self.covariance = (np.eye(12) - K @ H) @ self.covariance

        # Normalize angles
        self.state[3:6] = self._normalize_angles(self.state[3:6])

    def _rotation_matrix(self, roll: float, pitch: float, yaw: float) -> np.ndarray:
        """Compute rotation matrix from Euler angles (ZYX convention)."""
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)

        R = np.array(
            [
                [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                [-sp, cp * sr, cp * cr],
            ]
        )
        return R

    def _normalize_angles(self, angles: np.ndarray) -> np.ndarray:
        """Normalize angles to [-pi, pi]."""
        return np.arctan2(np.sin(angles), np.cos(angles))

    def reset(self, initial_pose: list[float] | None = None):
        """Reset filter state."""
        self.state = np.zeros(12)
        if initial_pose:
            self.state[:6] = initial_pose
        self.covariance = np.eye(12) * 0.1

    def teardown(self) -> None:
        """Clean up."""
        self.state = None
        self.covariance = None
        self._is_setup = False
