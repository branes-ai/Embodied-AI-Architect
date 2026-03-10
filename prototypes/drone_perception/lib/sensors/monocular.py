"""Monocular camera sensor (video file or webcam)."""

import time
from typing import Optional

import cv2
import numpy as np

from ..common import Frame, CameraParams
from .base import BaseSensor


class MonocularCamera(BaseSensor):
    """
    Monocular camera sensor using OpenCV.

    Supports:
    - Video files (.mp4, .avi, etc.)
    - Webcam (device_id=0)
    - Image sequences
    """

    def __init__(
        self,
        source: str | int = 0,
        camera_params: Optional[CameraParams] = None,
        target_fps: Optional[float] = None
    ):
        """
        Initialize monocular camera.

        Args:
            source: Video file path, image directory, or device ID (0 for webcam)
            camera_params: Camera intrinsics (auto-estimated if None)
            target_fps: Target FPS for playback (None = source fps)
        """
        self.source = source
        self.cap = cv2.VideoCapture(source)
        self.frame_id = 0
        self.start_time = time.time()

        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {source}")

        # Get video properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.source_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Set camera params
        if camera_params is None:
            # Auto-estimate from FOV (common for webcams/action cameras)
            self.camera_params = CameraParams.from_fov(self.width, self.height, fov_deg=60.0)
        else:
            self.camera_params = camera_params

        # FPS control
        self.target_fps = target_fps or self.source_fps
        self.frame_delay = 1.0 / self.target_fps if self.target_fps > 0 else 0
        self.last_frame_time = time.time()

        print(f"[MonocularCamera] Opened: {source}")
        print(f"  Resolution: {self.width}x{self.height}")
        print(f"  FPS: {self.source_fps:.1f} → {self.target_fps:.1f}")
        print(f"  Total frames: {self.total_frames}")

    def get_frame(self) -> Optional[Frame]:
        """
        Capture next frame.

        Returns:
            Frame object or None if no more frames
        """
        # Rate limiting for video playback
        if self.frame_delay > 0:
            elapsed = time.time() - self.last_frame_time
            sleep_time = self.frame_delay - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        ret, image = self.cap.read()
        if not ret:
            return None

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Create frame
        frame = Frame(
            timestamp=time.time() - self.start_time,
            image=image_rgb,
            depth=None,  # No depth for monocular
            camera_params=self.camera_params,
            frame_id=self.frame_id
        )

        self.frame_id += 1
        self.last_frame_time = time.time()

        return frame

    def is_opened(self) -> bool:
        """Check if capture is open."""
        return self.cap.isOpened()

    def release(self):
        """Release video capture."""
        if self.cap is not None:
            self.cap.release()
            print(f"[MonocularCamera] Released")

    def seek(self, frame_number: int):
        """Seek to specific frame (for video files)."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        self.frame_id = frame_number

    def get_progress(self) -> float:
        """Get playback progress (0.0 to 1.0)."""
        if self.total_frames > 0:
            return self.frame_id / self.total_frames
        return 0.0
