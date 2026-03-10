"""
Wide-angle (fisheye) camera sensor with DNN depth estimation.

Supports:
- Fisheye cameras (180°-200° FOV like Skydio)
- Depth Any Camera (DAC) for metric depth estimation
- Single or dual wide-angle camera configurations
- Zero-shot depth on any camera model
"""

import sys
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import cv2

from .base import BaseSensor
from ..common import Frame, CameraParams


class WideAngleCamera(BaseSensor):
    """
    Wide-angle (fisheye) camera with DNN depth estimation.

    Uses Depth Any Camera (DAC) model for metric depth estimation
    from wide-angle/fisheye cameras without special training.
    """

    def __init__(
        self,
        camera_id: int = 0,
        resolution: Tuple[int, int] = (1280, 720),
        fov: float = 200.0,
        dac_model_path: Optional[str] = None,
        dac_config_path: Optional[str] = None,
        camera_params: Optional[CameraParams] = None,
        use_dac: bool = True
    ):
        """
        Initialize wide-angle camera.

        Args:
            camera_id: Camera device ID
            resolution: Camera resolution (width, height)
            fov: Field of view in degrees (e.g., 200 for Skydio-like)
            dac_model_path: Path to DAC model weights (.pt)
            dac_config_path: Path to DAC config (.json)
            camera_params: Camera intrinsic parameters (if known)
            use_dac: Use DAC for depth estimation (requires setup)
        """
        self.camera_id = camera_id
        self.resolution = resolution
        self.fov = fov
        self.use_dac = use_dac

        # Camera
        self.camera = None
        self.frame_id = 0

        # Camera intrinsics
        if camera_params is None:
            # Estimate fisheye camera parameters
            # For fisheye: fx ≈ fy ≈ (width / 2) / tan(fov/2)
            fov_rad = np.deg2rad(fov / 2)
            focal = (resolution[0] / 2) / np.tan(fov_rad)

            self.camera_params = CameraParams(
                fx=focal,
                fy=focal,
                cx=resolution[0] / 2,
                cy=resolution[1] / 2,
                width=resolution[0],
                height=resolution[1]
            )
        else:
            self.camera_params = camera_params

        # DAC model
        self.dac_model = None
        self.dac_config = None

        if use_dac:
            self._init_dac(dac_model_path, dac_config_path)

        # Initialize camera
        self._init_camera()

    def _init_camera(self):
        """Initialize camera."""
        print(f"[WideAngle] Opening camera {self.camera_id}...")
        self.camera = cv2.VideoCapture(self.camera_id)

        if not self.camera.isOpened():
            raise RuntimeError(f"Failed to open camera {self.camera_id}")

        # Set resolution
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])

        # Get actual resolution
        actual_w = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"[WideAngle] Camera ready: {actual_w}x{actual_h}, FOV: {self.fov}°")

    def _init_dac(self, model_path: Optional[str], config_path: Optional[str]):
        """
        Initialize Depth Any Camera model.

        Args:
            model_path: Path to DAC weights
            config_path: Path to DAC config
        """
        try:
            # Import DAC (requires installation)
            import torch
            import json

            # Add DAC to path if installed in subdirectory
            dac_path = Path(__file__).parent.parent / "third_party" / "depth_any_camera"
            if dac_path.exists():
                sys.path.insert(0, str(dac_path))

            from dac.dataloaders import cameras
            from dac.models import DACPredictor

            # Load config
            if config_path is None:
                config_path = "checkpoints/dac_swinl_indoor.json"

            if model_path is None:
                model_path = "checkpoints/dac_swinl_indoor.pt"

            config_path = Path(config_path)
            model_path = Path(model_path)

            if not config_path.exists() or not model_path.exists():
                print(f"[WideAngle] DAC model/config not found. Using placeholder.")
                print(f"  Expected: {config_path}, {model_path}")
                print(f"  Download from: https://huggingface.co/yuliangguo/depth-any-camera")
                self.use_dac = False
                return

            # Load config
            with open(config_path) as f:
                self.dac_config = json.load(f)

            # Initialize model
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.dac_model = DACPredictor(
                config=self.dac_config,
                model_path=str(model_path),
                device=device
            )

            print(f"[WideAngle] DAC model loaded: {model_path.name}")
            print(f"[WideAngle] Device: {device}")

        except ImportError as e:
            print(f"[WideAngle] DAC not available: {e}")
            print("[WideAngle] Install with: cd third_party/depth_any_camera && pip install -r requirements.txt")
            self.use_dac = False
        except Exception as e:
            print(f"[WideAngle] Failed to load DAC: {e}")
            self.use_dac = False

    def get_frame(self) -> Optional[Frame]:
        """
        Get camera frame with depth estimation.

        Returns:
            Frame with RGB image and depth map
        """
        if not self.is_opened():
            return None

        # Capture frame
        ret, image = self.camera.read()
        if not ret:
            return None

        # Estimate depth
        depth_map = None
        if self.use_dac and self.dac_model is not None:
            depth_map = self._estimate_depth_dac(image)

        # Create frame
        self.frame_id += 1
        frame = Frame(
            image=image,
            depth=depth_map,
            frame_id=self.frame_id,
            camera_params=self.camera_params
        )

        return frame

    def _estimate_depth_dac(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Estimate depth using DAC model.

        Args:
            image: RGB image

        Returns:
            Depth map in meters
        """
        try:
            import torch

            # Prepare camera intrinsics for DAC
            # DAC expects: fx, fy, cx, cy
            K = np.array([
                [self.camera_params.fx, 0, self.camera_params.cx],
                [0, self.camera_params.fy, self.camera_params.cy],
                [0, 0, 1]
            ])

            # Run inference
            with torch.no_grad():
                depth = self.dac_model.predict(
                    image=image,
                    intrinsics=K,
                    fisheye=(self.fov > 120)  # Flag fisheye for DAC
                )

            return depth

        except Exception as e:
            print(f"[WideAngle] DAC inference failed: {e}")
            return None

    def is_opened(self) -> bool:
        """Check if sensor is ready."""
        return self.camera is not None and self.camera.isOpened()

    def release(self):
        """Release resources."""
        if self.camera is not None:
            self.camera.release()
        print("[WideAngle] Released")


class DualWideAngleCamera(BaseSensor):
    """
    Dual wide-angle stereo camera (like Skydio).

    Uses two wide-angle cameras for improved depth via:
    - Stereo matching (if calibrated)
    - Dual DAC depth fusion
    - Left/right consistency checking
    """

    def __init__(
        self,
        left_camera_id: int = 0,
        right_camera_id: int = 1,
        resolution: Tuple[int, int] = (1280, 720),
        fov: float = 200.0,
        dac_model_path: Optional[str] = None,
        dac_config_path: Optional[str] = None,
        baseline: float = 0.1,  # meters
        use_dac: bool = True
    ):
        """
        Initialize dual wide-angle camera system.

        Args:
            left_camera_id: Left camera device ID
            right_camera_id: Right camera device ID
            resolution: Camera resolution
            fov: Field of view in degrees
            dac_model_path: Path to DAC model
            dac_config_path: Path to DAC config
            baseline: Stereo baseline in meters
            use_dac: Use DAC for depth
        """
        self.left_cam = WideAngleCamera(
            camera_id=left_camera_id,
            resolution=resolution,
            fov=fov,
            dac_model_path=dac_model_path,
            dac_config_path=dac_config_path,
            use_dac=use_dac
        )

        self.right_cam = WideAngleCamera(
            camera_id=right_camera_id,
            resolution=resolution,
            fov=fov,
            dac_model_path=dac_model_path,
            dac_config_path=dac_config_path,
            use_dac=use_dac
        )

        self.baseline = baseline
        self.frame_id = 0

        print(f"[DualWideAngle] Stereo baseline: {baseline}m")

    def get_frame(self) -> Optional[Frame]:
        """
        Get synchronized stereo frame with fused depth.

        Returns:
            Frame with left image and fused depth map
        """
        # Get frames from both cameras
        left_frame = self.left_cam.get_frame()
        right_frame = self.right_cam.get_frame()

        if left_frame is None or right_frame is None:
            return None

        # Fuse depth maps (simple average for now)
        # TODO: Implement proper stereo fusion or consistency checking
        depth_map = None
        if left_frame.depth is not None and right_frame.depth is not None:
            # Average left and right depth (simple fusion)
            depth_map = (left_frame.depth + right_frame.depth) / 2.0
        elif left_frame.depth is not None:
            depth_map = left_frame.depth
        elif right_frame.depth is not None:
            depth_map = right_frame.depth

        # Create fused frame
        self.frame_id += 1
        frame = Frame(
            image=left_frame.image,  # Use left as primary
            depth=depth_map,
            frame_id=self.frame_id,
            camera_params=left_frame.camera_params
        )

        return frame

    def is_opened(self) -> bool:
        """Check if both cameras are ready."""
        return self.left_cam.is_opened() and self.right_cam.is_opened()

    def release(self):
        """Release both cameras."""
        self.left_cam.release()
        self.right_cam.release()
        print("[DualWideAngle] Released")
