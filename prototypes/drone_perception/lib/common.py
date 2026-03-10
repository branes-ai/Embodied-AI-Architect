"""Common data structures for the drone perception pipeline."""

from dataclasses import dataclass, field
from typing import Optional, Tuple
import numpy as np


@dataclass
class CameraParams:
    """Camera intrinsic parameters for 2D->3D projection."""
    fx: float  # Focal length x
    fy: float  # Focal length y
    cx: float  # Principal point x
    cy: float  # Principal point y
    width: int  # Image width
    height: int  # Image height

    @classmethod
    def from_fov(cls, width: int, height: int, fov_deg: float = 60.0) -> 'CameraParams':
        """Create camera params from field of view (common for webcams)."""
        fov_rad = np.radians(fov_deg)
        fx = fy = width / (2 * np.tan(fov_rad / 2))
        cx = width / 2
        cy = height / 2
        return cls(fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height)


@dataclass
class Frame:
    """A captured frame with image and optional depth."""
    timestamp: float
    image: np.ndarray  # RGB image (H, W, 3)
    depth: Optional[np.ndarray] = None  # Depth map (H, W) in meters
    camera_params: Optional[CameraParams] = None
    frame_id: int = 0


@dataclass
class BBox:
    """2D bounding box in image coordinates."""
    x: float  # Top-left x
    y: float  # Top-left y
    w: float  # Width
    h: float  # Height

    @property
    def center(self) -> Tuple[float, float]:
        """Return center point (cx, cy)."""
        return (self.x + self.w / 2, self.y + self.h / 2)

    @property
    def bottom_center(self) -> Tuple[float, float]:
        """Return bottom-center point (useful for ground plane projection)."""
        return (self.x + self.w / 2, self.y + self.h)

    @property
    def area(self) -> float:
        """Return bbox area."""
        return self.w * self.h

    def to_xyxy(self) -> Tuple[float, float, float, float]:
        """Convert to (x1, y1, x2, y2) format."""
        return (self.x, self.y, self.x + self.w, self.y + self.h)

    @classmethod
    def from_xyxy(cls, x1: float, y1: float, x2: float, y2: float) -> 'BBox':
        """Create from (x1, y1, x2, y2) format."""
        return cls(x=x1, y=y1, w=x2 - x1, h=y2 - y1)

    def iou(self, other: 'BBox') -> float:
        """Calculate Intersection over Union with another bbox."""
        # Get intersection rectangle
        x1 = max(self.x, other.x)
        y1 = max(self.y, other.y)
        x2 = min(self.x + self.w, other.x + other.w)
        y2 = min(self.y + self.h, other.y + other.h)

        # Calculate areas
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        union = self.area + other.area - intersection

        return intersection / union if union > 0 else 0.0


@dataclass
class Detection:
    """A detected object in a frame."""
    bbox: BBox
    class_id: int
    class_name: str
    confidence: float
    frame_id: int = 0


@dataclass
class Track:
    """A tracked object across multiple frames."""
    id: int  # Unique track ID
    bbox: BBox  # Current bounding box
    class_id: int
    class_name: str
    confidence: float
    age: int = 0  # Number of frames tracked
    hits: int = 0  # Number of successful updates
    time_since_update: int = 0  # Frames since last detection
    state: Optional[np.ndarray] = None  # Kalman filter state


@dataclass
class TrackedObject:
    """A 3D tracked object in world coordinates."""
    # Identity
    track_id: int
    class_name: str

    # 3D State (in world/body frame)
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))  # [x, y, z] meters
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))  # [vx, vy, vz] m/s
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))  # [ax, ay, az] m/s²

    # Metadata
    timestamp: float = 0.0
    confidence: float = 0.0
    last_seen: float = 0.0
    age: int = 0  # Number of frames tracked

    # History for trajectory visualization
    position_history: list = field(default_factory=list)
    max_history: int = 100

    def update_position(self, position: np.ndarray, timestamp: float):
        """Update position and add to history."""
        self.position = position
        self.timestamp = timestamp
        self.last_seen = timestamp

        # Add to history
        self.position_history.append(position.copy())
        if len(self.position_history) > self.max_history:
            self.position_history.pop(0)

    def estimate_velocity(self, dt: float):
        """Estimate velocity from position history (simple finite difference)."""
        if len(self.position_history) >= 2 and dt > 0:
            self.velocity = (self.position_history[-1] - self.position_history[-2]) / dt

    def estimate_acceleration(self, dt: float):
        """Estimate acceleration from velocity changes."""
        if len(self.position_history) >= 3 and dt > 0:
            v_current = (self.position_history[-1] - self.position_history[-2]) / dt
            v_previous = (self.position_history[-2] - self.position_history[-3]) / dt
            self.acceleration = (v_current - v_previous) / dt

    @property
    def trajectory_3d(self) -> list:
        """Get 3D trajectory (alias for position_history for reasoning modules)."""
        return self.position_history

    @property
    def position_3d(self) -> np.ndarray:
        """Get current 3D position (alias for reasoning modules)."""
        return self.position

    @property
    def velocity_3d(self) -> np.ndarray:
        """Get 3D velocity (alias for reasoning modules)."""
        return self.velocity


def project_2d_to_3d(
    pixel_x: float,
    pixel_y: float,
    depth: float,
    camera_params: CameraParams
) -> np.ndarray:
    """
    Project a 2D pixel + depth to 3D point in camera frame.

    Returns:
        np.ndarray: [x, y, z] in meters (camera frame: +X right, +Y down, +Z forward)
    """
    x = (pixel_x - camera_params.cx) * depth / camera_params.fx
    y = (pixel_y - camera_params.cy) * depth / camera_params.fy
    z = depth
    return np.array([x, y, z])


def estimate_depth_from_bbox_height(
    bbox_height_pixels: float,
    object_height_meters: float,
    camera_params: CameraParams
) -> float:
    """
    Estimate depth from bbox height (assuming known object size).

    Formula: depth = (focal_length * real_height) / pixel_height

    Args:
        bbox_height_pixels: Height of bounding box in pixels
        object_height_meters: Assumed real-world height of object (e.g., 1.7m for person)
        camera_params: Camera intrinsics

    Returns:
        Estimated depth in meters
    """
    if bbox_height_pixels <= 0:
        return 10.0  # Default fallback

    depth = (camera_params.fy * object_height_meters) / bbox_height_pixels
    return depth
