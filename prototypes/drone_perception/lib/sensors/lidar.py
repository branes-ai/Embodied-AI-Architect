"""
LiDAR sensor interface for LiDAR + Camera fusion.

Supports common 3D LiDAR sensors:
- Velodyne (VLP-16, VLP-32, HDL-64E)
- Ouster (OS0, OS1, OS2)
- Livox (Avia, Mid-360)
- RoboSense
- Generic point cloud sources (ROS, files)
"""

from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import cv2

from .base import BaseSensor
from ..common import Frame, CameraParams


class LiDARCameraSensor(BaseSensor):
    """
    Combined LiDAR + Camera sensor.

    Synchronizes LiDAR point clouds with camera frames and projects
    LiDAR depth onto image plane for object detection with metric depth.
    """

    def __init__(
        self,
        camera_id: int = 0,
        lidar_source: str = "velodyne",
        lidar_config: Optional[dict] = None,
        camera_params: Optional[CameraParams] = None,
        extrinsics: Optional[np.ndarray] = None,
        resolution: Tuple[int, int] = (1280, 720)
    ):
        """
        Initialize LiDAR + Camera sensor.

        Args:
            camera_id: Camera device ID
            lidar_source: LiDAR type - "velodyne", "ouster", "livox", "ros", "file"
            lidar_config: LiDAR-specific configuration
            camera_params: Camera intrinsic parameters
            extrinsics: LiDAR-to-camera extrinsic calibration (4x4 transform)
            resolution: Camera resolution (width, height)
        """
        self.camera_id = camera_id
        self.lidar_source = lidar_source
        self.lidar_config = lidar_config or {}
        self.resolution = resolution

        # Camera
        self.camera = None
        self.frame_id = 0

        # Camera intrinsics
        if camera_params is None:
            # Default camera parameters (need proper calibration)
            self.camera_params = CameraParams(
                fx=700.0,
                fy=700.0,
                cx=resolution[0] / 2,
                cy=resolution[1] / 2,
                width=resolution[0],
                height=resolution[1]
            )
        else:
            self.camera_params = camera_params

        # LiDAR-to-camera extrinsics (4x4 transformation matrix)
        if extrinsics is None:
            # Identity transform (need proper calibration)
            # In practice, this should be calibrated using techniques like:
            # - Checkboard calibration
            # - Manual alignment
            # - Automatic calibration (e.g., using apollo-calibration)
            self.extrinsics = np.eye(4)
            print("WARNING: Using identity LiDAR-camera transform. Please calibrate!")
        else:
            self.extrinsics = extrinsics

        # LiDAR interface
        self.lidar = None

        # Initialize
        self._init_camera()
        self._init_lidar()

    def _init_camera(self):
        """Initialize camera."""
        print(f"[LiDARCamera] Opening camera {self.camera_id}...")
        self.camera = cv2.VideoCapture(self.camera_id)

        if not self.camera.isOpened():
            raise RuntimeError(f"Failed to open camera {self.camera_id}")

        # Set resolution
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])

        print(f"[LiDARCamera] Camera ready: {self.resolution[0]}x{self.resolution[1]}")

    def _init_lidar(self):
        """Initialize LiDAR sensor."""
        print(f"[LiDARCamera] Initializing LiDAR: {self.lidar_source}")

        if self.lidar_source == "velodyne":
            self.lidar = VelodyneLiDAR(self.lidar_config)
        elif self.lidar_source == "ouster":
            self.lidar = OusterLiDAR(self.lidar_config)
        elif self.lidar_source == "livox":
            self.lidar = LivoxLiDAR(self.lidar_config)
        elif self.lidar_source == "ros":
            self.lidar = ROSLiDAR(self.lidar_config)
        elif self.lidar_source == "file":
            self.lidar = FileLiDAR(self.lidar_config)
        else:
            raise ValueError(f"Unsupported LiDAR source: {self.lidar_source}")

        print(f"[LiDARCamera] LiDAR ready")

    def get_frame(self) -> Optional[Frame]:
        """
        Get synchronized camera frame with LiDAR depth.

        Returns:
            Frame with RGB image and projected LiDAR depth map
        """
        if not self.is_opened():
            return None

        # Capture camera frame
        ret, image = self.camera.read()
        if not ret:
            return None

        # Get LiDAR point cloud
        point_cloud = self.lidar.get_point_cloud()
        if point_cloud is None:
            return None

        # Project LiDAR points to camera image plane
        depth_map = self._project_lidar_to_image(point_cloud, image.shape[:2])

        # Create frame
        self.frame_id += 1
        frame = Frame(
            image=image,
            depth=depth_map,
            frame_id=self.frame_id,
            camera_params=self.camera_params
        )

        return frame

    def _project_lidar_to_image(
        self,
        point_cloud: np.ndarray,
        image_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        Project 3D LiDAR points to 2D image plane.

        Args:
            point_cloud: Nx4 array (x, y, z, intensity)
            image_shape: (height, width) of image

        Returns:
            depth_map: HxW depth image in meters
        """
        height, width = image_shape

        # Transform points from LiDAR frame to camera frame
        # point_cloud: Nx3 or Nx4 (x, y, z, [intensity])
        points_3d = point_cloud[:, :3]  # Nx3

        # Apply extrinsic transformation
        points_3d_hom = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])  # Nx4
        points_cam = (self.extrinsics @ points_3d_hom.T).T[:, :3]  # Nx3

        # Filter points behind camera
        valid_mask = points_cam[:, 2] > 0.1  # At least 10cm in front
        points_cam = points_cam[valid_mask]

        if len(points_cam) == 0:
            return np.zeros((height, width), dtype=np.float32)

        # Project to image plane
        # [u, v, 1] = K @ [x, y, z]
        fx, fy = self.camera_params.fx, self.camera_params.fy
        cx, cy = self.camera_params.cx, self.camera_params.cy

        u = (points_cam[:, 0] / points_cam[:, 2]) * fx + cx
        v = (points_cam[:, 1] / points_cam[:, 2]) * fy + cy
        depth = points_cam[:, 2]

        # Filter points within image bounds
        valid_mask = (
            (u >= 0) & (u < width) &
            (v >= 0) & (v < height)
        )

        u = u[valid_mask].astype(np.int32)
        v = v[valid_mask].astype(np.int32)
        depth = depth[valid_mask]

        # Create depth map (sparse initially)
        depth_map = np.zeros((height, width), dtype=np.float32)

        # Assign depth values (handle overlapping points by keeping closest)
        for ui, vi, di in zip(u, v, depth):
            if depth_map[vi, ui] == 0 or di < depth_map[vi, ui]:
                depth_map[vi, ui] = di

        # Optional: Interpolate sparse depth (for denser depth map)
        # This can be useful for object detection
        # depth_map = self._interpolate_depth(depth_map)

        return depth_map

    def _interpolate_depth(self, depth_map: np.ndarray, radius: int = 5) -> np.ndarray:
        """
        Interpolate sparse LiDAR depth map for denser coverage.

        Args:
            depth_map: Sparse depth map
            radius: Interpolation radius

        Returns:
            Interpolated depth map
        """
        # Simple dilation-based interpolation
        # For production, consider: bilateral filter, guided filter, or learned inpainting
        mask = (depth_map > 0).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius, radius))

        # Dilate to fill small gaps
        dilated_mask = cv2.dilate(mask, kernel)

        # Inpaint
        depth_inpainted = cv2.inpaint(
            (depth_map * 1000).astype(np.uint16),
            255 - (mask * 255),
            radius,
            cv2.INPAINT_NS
        ).astype(np.float32) / 1000.0

        # Only keep interpolated values where depth was zero
        result = depth_map.copy()
        result[dilated_mask > 0] = depth_inpainted[dilated_mask > 0]

        return result

    def is_opened(self) -> bool:
        """Check if sensor is ready."""
        return (
            self.camera is not None and
            self.camera.isOpened() and
            self.lidar is not None and
            self.lidar.is_opened()
        )

    def release(self):
        """Release resources."""
        if self.camera is not None:
            self.camera.release()
        if self.lidar is not None:
            self.lidar.release()
        print("[LiDARCamera] Released")


# ============================================================================
# LiDAR Driver Interfaces
# ============================================================================

class BaseLiDAR:
    """Base class for LiDAR interfaces."""

    def get_point_cloud(self) -> Optional[np.ndarray]:
        """Get point cloud (Nx4: x, y, z, intensity)."""
        raise NotImplementedError

    def is_opened(self) -> bool:
        """Check if LiDAR is ready."""
        raise NotImplementedError

    def release(self):
        """Release resources."""
        pass


class VelodyneLiDAR(BaseLiDAR):
    """Velodyne LiDAR interface (VLP-16, VLP-32, HDL-64E)."""

    def __init__(self, config: dict):
        """
        Initialize Velodyne LiDAR.

        Config options:
            ip: LiDAR IP address (default: "192.168.1.201")
            port: Data port (default: 2368)
            model: "VLP-16", "VLP-32", "HDL-64E"
        """
        self.config = config
        self.ip = config.get("ip", "192.168.1.201")
        self.port = config.get("port", 2368)
        self.model = config.get("model", "VLP-16")

        # TODO: Initialize Velodyne driver
        # This would use python-pcl, ros-velodyne, or custom UDP receiver
        print(f"[Velodyne] Connecting to {self.ip}:{self.port} ({self.model})")
        print("[Velodyne] TODO: Implement actual driver (use python-pcl or ROS)")

        self._opened = False

    def get_point_cloud(self) -> Optional[np.ndarray]:
        """Get latest point cloud."""
        # TODO: Read from Velodyne and return Nx4 array
        # For now, return None (not implemented)
        return None

    def is_opened(self) -> bool:
        return self._opened

    def release(self):
        print("[Velodyne] Released")


class OusterLiDAR(BaseLiDAR):
    """Ouster LiDAR interface (OS0, OS1, OS2)."""

    def __init__(self, config: dict):
        self.config = config
        # TODO: Implement Ouster driver
        print("[Ouster] TODO: Implement driver (use ouster-sdk)")
        self._opened = False

    def get_point_cloud(self) -> Optional[np.ndarray]:
        return None

    def is_opened(self) -> bool:
        return self._opened

    def release(self):
        print("[Ouster] Released")


class LivoxLiDAR(BaseLiDAR):
    """Livox LiDAR interface (Avia, Mid-360)."""

    def __init__(self, config: dict):
        self.config = config
        # TODO: Implement Livox driver
        print("[Livox] TODO: Implement driver (use livox-sdk)")
        self._opened = False

    def get_point_cloud(self) -> Optional[np.ndarray]:
        return None

    def is_opened(self) -> bool:
        return self._opened

    def release(self):
        print("[Livox] Released")


class ROSLiDAR(BaseLiDAR):
    """ROS-based LiDAR interface (subscribes to sensor_msgs/PointCloud2)."""

    def __init__(self, config: dict):
        """
        Initialize ROS LiDAR subscriber.

        Config options:
            topic: ROS topic (default: "/velodyne_points")
            frame_id: Expected frame_id
        """
        self.config = config
        self.topic = config.get("topic", "/velodyne_points")

        # TODO: Initialize ROS subscriber
        print(f"[ROS] Subscribing to {self.topic}")
        print("[ROS] TODO: Implement ROS bridge (use rospy or rclpy)")

        self._opened = False
        self._latest_cloud = None

    def get_point_cloud(self) -> Optional[np.ndarray]:
        """Get latest point cloud from ROS."""
        return self._latest_cloud

    def is_opened(self) -> bool:
        return self._opened

    def release(self):
        print("[ROS] Released")


class FileLiDAR(BaseLiDAR):
    """
    File-based LiDAR (reads pre-recorded point clouds).

    Supports:
    - .pcd files (Point Cloud Data)
    - .ply files
    - .bin files (KITTI format)
    - .npy files (numpy arrays)
    """

    def __init__(self, config: dict):
        """
        Initialize file-based LiDAR.

        Config options:
            path: Path to point cloud file or directory
            format: "pcd", "ply", "bin", "npy"
            loop: Loop through files (default: True)
        """
        self.config = config
        self.path = Path(config.get("path", ""))
        self.format = config.get("format", "auto")
        self.loop = config.get("loop", True)

        # Find all point cloud files
        if self.path.is_dir():
            self.files = sorted(self.path.glob(f"*.{self.format}")) if self.format != "auto" else \
                         sorted(list(self.path.glob("*.pcd")) +
                                list(self.path.glob("*.ply")) +
                                list(self.path.glob("*.bin")) +
                                list(self.path.glob("*.npy")))
        else:
            self.files = [self.path]

        self.current_idx = 0
        self._opened = len(self.files) > 0

        print(f"[FileLiDAR] Loaded {len(self.files)} files from {self.path}")

    def get_point_cloud(self) -> Optional[np.ndarray]:
        """Get next point cloud from file."""
        if not self.files:
            return None

        # Get current file
        file_path = self.files[self.current_idx]

        # Load based on format
        try:
            if file_path.suffix == ".npy":
                cloud = np.load(file_path)
            elif file_path.suffix == ".bin":
                # KITTI format: flat binary of float32 (x,y,z,intensity)
                cloud = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
            elif file_path.suffix in [".pcd", ".ply"]:
                # TODO: Use python-pcl or open3d
                print(f"[FileLiDAR] {file_path.suffix} files require python-pcl or open3d")
                cloud = None
            else:
                cloud = None

            # Advance to next file
            self.current_idx = (self.current_idx + 1) % len(self.files) if self.loop else \
                              min(self.current_idx + 1, len(self.files) - 1)

            return cloud

        except Exception as e:
            print(f"[FileLiDAR] Error loading {file_path}: {e}")
            return None

    def is_opened(self) -> bool:
        return self._opened

    def release(self):
        print("[FileLiDAR] Released")
