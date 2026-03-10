"""Stereo camera sensor (RealSense D435, OAK-D, etc.)."""

import time
from typing import Optional
import numpy as np

from ..common import Frame, CameraParams
from .base import BaseSensor


class StereoCamera(BaseSensor):
    """
    Stereo camera sensor with depth capability.

    Supports:
    - Intel RealSense D435/D455
    - OAK-D (via depthai)
    - Custom stereo rigs

    Provides real metric depth from stereo matching.
    """

    def __init__(
        self,
        device_id: Optional[str] = None,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        depth_enabled: bool = True,
        backend: str = "realsense"
    ):
        """
        Initialize stereo camera.

        Args:
            device_id: Device serial number (None = auto-detect first device)
            width: Frame width
            height: Frame height
            fps: Target framerate
            depth_enabled: Enable depth stream
            backend: "realsense" or "oakd"
        """
        self.device_id = device_id
        self.width = width
        self.height = height
        self.fps = fps
        self.depth_enabled = depth_enabled
        self.backend = backend
        self.frame_id = 0
        self.start_time = time.time()

        if backend == "realsense":
            self._init_realsense()
        elif backend == "oakd":
            self._init_oakd()
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def _init_realsense(self):
        """Initialize RealSense camera."""
        try:
            import pyrealsense2 as rs
        except ImportError:
            raise ImportError(
                "pyrealsense2 not installed. Install with: pip install pyrealsense2"
            )

        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Enable device by serial if specified
        if self.device_id:
            self.config.enable_device(self.device_id)

        # Configure streams
        self.config.enable_stream(
            rs.stream.color,
            self.width,
            self.height,
            rs.format.rgb8,
            self.fps
        )

        if self.depth_enabled:
            self.config.enable_stream(
                rs.stream.depth,
                self.width,
                self.height,
                rs.format.z16,
                self.fps
            )

        # Start pipeline
        profile = self.pipeline.start(self.config)

        # Get camera intrinsics from color stream
        color_stream = profile.get_stream(rs.stream.color)
        intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

        self.camera_params = CameraParams(
            fx=intrinsics.fx,
            fy=intrinsics.fy,
            cx=intrinsics.ppx,
            cy=intrinsics.ppy,
            width=intrinsics.width,
            height=intrinsics.height
        )

        # Create align object (align depth to color)
        align_to = rs.stream.color
        self.align = rs.align(align_to)

        # Depth scale (convert units to meters)
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        print(f"[StereoCamera] Initialized RealSense")
        print(f"  Resolution: {self.width}x{self.height}")
        print(f"  FPS: {self.fps}")
        print(f"  Depth scale: {self.depth_scale}")
        print(f"  Camera intrinsics: fx={intrinsics.fx:.1f}, fy={intrinsics.fy:.1f}")

    def _init_oakd(self):
        """Initialize OAK-D camera."""
        try:
            import depthai as dai
        except ImportError:
            raise ImportError(
                "depthai not installed. Install with: pip install depthai"
            )

        # Create pipeline
        self.pipeline_oakd = dai.Pipeline()

        # Define sources
        cam_rgb = self.pipeline_oakd.create(dai.node.ColorCamera)
        stereo = self.pipeline_oakd.create(dai.node.StereoDepth)

        # Create outputs
        xout_rgb = self.pipeline_oakd.create(dai.node.XLinkOut)
        xout_depth = self.pipeline_oakd.create(dai.node.XLinkOut)

        xout_rgb.setStreamName("rgb")
        xout_depth.setStreamName("depth")

        # Configure color camera
        cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam_rgb.setInterleaved(False)
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

        # Configure stereo depth
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        stereo.setDepthAlign(dai.CameraBoardSocket.RGB)

        # Link nodes
        cam_rgb.video.link(xout_rgb.input)
        stereo.depth.link(xout_depth.input)

        # Connect to device
        self.device = dai.Device(self.pipeline_oakd)

        # Output queues
        self.q_rgb = self.device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        self.q_depth = self.device.getOutputQueue(name="depth", maxSize=4, blocking=False)

        # Estimate camera params (OAK-D RGB camera)
        self.camera_params = CameraParams.from_fov(
            width=self.width,
            height=self.height,
            fov_deg=69.0  # OAK-D RGB camera FOV
        )

        print(f"[StereoCamera] Initialized OAK-D")
        print(f"  Resolution: {self.width}x{self.height}")

    def get_frame(self) -> Optional[Frame]:
        """
        Capture next frame with aligned depth.

        Returns:
            Frame object with RGB image and depth map, or None on error
        """
        if self.backend == "realsense":
            return self._get_frame_realsense()
        elif self.backend == "oakd":
            return self._get_frame_oakd()

    def _get_frame_realsense(self) -> Optional[Frame]:
        """Get frame from RealSense camera."""
        import pyrealsense2 as rs

        try:
            # Wait for frames
            frames = self.pipeline.wait_for_frames(timeout_ms=5000)

            # Align depth to color
            aligned_frames = self.align.process(frames)

            # Get aligned frames
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame() if self.depth_enabled else None

            if not color_frame:
                return None

            # Convert to numpy arrays
            image = np.asanyarray(color_frame.get_data())

            depth = None
            if depth_frame:
                # Get depth as uint16, convert to meters
                depth_raw = np.asanyarray(depth_frame.get_data())
                depth = depth_raw.astype(np.float32) * self.depth_scale

            # Create frame
            frame = Frame(
                timestamp=time.time() - self.start_time,
                image=image,
                depth=depth,
                camera_params=self.camera_params,
                frame_id=self.frame_id
            )

            self.frame_id += 1
            return frame

        except RuntimeError as e:
            print(f"[StereoCamera] Error getting frame: {e}")
            return None

    def _get_frame_oakd(self) -> Optional[Frame]:
        """Get frame from OAK-D camera."""
        # Get RGB frame
        in_rgb = self.q_rgb.get()
        image = in_rgb.getCvFrame()

        # Get depth frame
        depth = None
        if self.depth_enabled:
            in_depth = self.q_depth.get()
            depth = in_depth.getFrame()
            # OAK-D depth is in millimeters, convert to meters
            depth = depth.astype(np.float32) / 1000.0

        # Create frame
        frame = Frame(
            timestamp=time.time() - self.start_time,
            image=image,
            depth=depth,
            camera_params=self.camera_params,
            frame_id=self.frame_id
        )

        self.frame_id += 1
        return frame

    def is_opened(self) -> bool:
        """Check if camera is ready."""
        if self.backend == "realsense":
            return self.pipeline is not None
        elif self.backend == "oakd":
            return self.device is not None
        return False

    def release(self):
        """Release camera resources."""
        if self.backend == "realsense":
            if hasattr(self, 'pipeline') and self.pipeline is not None:
                self.pipeline.stop()
                print("[StereoCamera] Released RealSense")
        elif self.backend == "oakd":
            if hasattr(self, 'device') and self.device is not None:
                self.device.close()
                print("[StereoCamera] Released OAK-D")

    def get_depth_at_point(self, x: int, y: int, radius: int = 5) -> Optional[float]:
        """
        Get robust depth at a point using median filtering.

        Args:
            x: X coordinate (pixel)
            y: Y coordinate (pixel)
            radius: Median filter radius (for robustness against noise)

        Returns:
            Median depth in meters, or None if unavailable
        """
        frame = self.get_frame()
        if frame is None or frame.depth is None:
            return None

        depth = frame.depth
        h, w = depth.shape

        # Bounds checking
        x = max(radius, min(x, w - radius - 1))
        y = max(radius, min(y, h - radius - 1))

        # Extract window
        window = depth[y-radius:y+radius+1, x-radius:x+radius+1]

        # Filter out invalid depths (0 or too far)
        valid_depths = window[(window > 0.1) & (window < 10.0)]

        if len(valid_depths) == 0:
            return None

        return float(np.median(valid_depths))


class StereoRecordedCamera(BaseSensor):
    """
    Playback recorded stereo camera data from files.

    Useful for testing without hardware.
    """

    def __init__(
        self,
        rgb_video_path: str,
        depth_video_path: str,
        camera_params: Optional[CameraParams] = None,
        depth_scale: float = 0.001  # 1mm per unit (common for saved depth)
    ):
        """
        Initialize from pre-recorded RGB + depth videos.

        Args:
            rgb_video_path: Path to RGB video
            depth_video_path: Path to depth video (grayscale 16-bit)
            camera_params: Camera intrinsics
            depth_scale: Scale factor to convert depth values to meters
        """
        import cv2

        self.rgb_cap = cv2.VideoCapture(rgb_video_path)
        self.depth_cap = cv2.VideoCapture(depth_video_path)
        self.depth_scale = depth_scale
        self.frame_id = 0
        self.start_time = time.time()

        if not self.rgb_cap.isOpened():
            raise RuntimeError(f"Failed to open RGB video: {rgb_video_path}")
        if not self.depth_cap.isOpened():
            raise RuntimeError(f"Failed to open depth video: {depth_video_path}")

        # Get video properties
        width = int(self.rgb_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.rgb_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if camera_params is None:
            self.camera_params = CameraParams.from_fov(width, height, fov_deg=60.0)
        else:
            self.camera_params = camera_params

        print(f"[StereoRecordedCamera] Opened recorded stereo data")
        print(f"  RGB: {rgb_video_path}")
        print(f"  Depth: {depth_video_path}")

    def get_frame(self) -> Optional[Frame]:
        """Get next frame from recorded data."""
        import cv2

        ret_rgb, image_bgr = self.rgb_cap.read()
        ret_depth, depth_raw = self.depth_cap.read()

        if not ret_rgb or not ret_depth:
            return None

        # Convert BGR to RGB
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Convert depth to float and scale to meters
        if len(depth_raw.shape) == 3:
            depth_raw = cv2.cvtColor(depth_raw, cv2.COLOR_BGR2GRAY)
        depth = depth_raw.astype(np.float32) * self.depth_scale

        frame = Frame(
            timestamp=time.time() - self.start_time,
            image=image,
            depth=depth,
            camera_params=self.camera_params,
            frame_id=self.frame_id
        )

        self.frame_id += 1
        return frame

    def is_opened(self) -> bool:
        """Check if files are open."""
        return self.rgb_cap.isOpened() and self.depth_cap.isOpened()

    def release(self):
        """Release video captures."""
        self.rgb_cap.release()
        self.depth_cap.release()
        print("[StereoRecordedCamera] Released")
