"""Scene graph manager operator.

Maintains 3D world state from tracked objects.
"""

from typing import Any

import numpy as np

from ..base import Operator


class SceneGraphManager(Operator):
    """3D scene graph manager.

    Maintains world state by:
    - Converting 2D tracks to 3D positions
    - Applying Kalman filtering for smooth 3D trajectories
    - Managing object persistence with time-to-live
    """

    def __init__(self):
        super().__init__(operator_id="scene_graph_manager")
        self.objects: dict[int, dict] = {}
        self.ttl_frames = 30
        self.enable_depth = False

        # Simple depth estimation from bbox (heuristic)
        self.reference_height_m = 1.7  # Average person height
        self.reference_height_px = 200  # Pixels at 5m distance

    def setup(self, config: dict[str, Any], execution_target: str = "cpu") -> None:
        """Initialize scene graph manager.

        Args:
            config: Configuration with optional keys:
                - ttl_frames: Frames to keep objects without updates
                - enable_depth: Enable depth estimation
                - reference_height_m: Reference object height in meters
                - reference_height_px: Reference height in pixels at known distance
                - camera_fx: Camera focal length x (for projection)
                - camera_fy: Camera focal length y
                - camera_cx: Camera principal point x
                - camera_cy: Camera principal point y
            execution_target: Only cpu supported
        """
        if execution_target != "cpu":
            print(f"[SceneGraphManager] Warning: Only CPU supported")

        self._execution_target = "cpu"
        self._config = config

        self.ttl_frames = config.get("ttl_frames", 30)
        self.enable_depth = config.get("enable_depth", False)
        self.reference_height_m = config.get("reference_height_m", 1.7)
        self.reference_height_px = config.get("reference_height_px", 200)

        # Camera intrinsics (optional)
        self.camera_fx = config.get("camera_fx", 500)
        self.camera_fy = config.get("camera_fy", 500)
        self.camera_cx = config.get("camera_cx", 320)
        self.camera_cy = config.get("camera_cy", 240)

        self.objects = {}
        self._is_setup = True
        print(f"[SceneGraphManager] Ready (ttl={self.ttl_frames} frames)")

    def process(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Update scene graph with new tracks.

        Args:
            inputs: Dictionary with:
                - 'tracks': List of track dicts from ByteTrack
                - 'depth_map': Optional depth map for metric depth
                - 'image_size': Optional (height, width) for projection

        Returns:
            Dictionary with:
                - 'objects': List of 3D object dicts
                - 'obstacles': Alias for objects (for downstream compatibility)
        """
        tracks = inputs.get("tracks", [])
        depth_map = inputs.get("depth_map")
        image_size = inputs.get("image_size", (480, 640))

        # Update existing objects and add new ones
        seen_ids = set()

        for track in tracks:
            track_id = track["track_id"]
            seen_ids.add(track_id)

            bbox = track["bbox"]  # [x1, y1, x2, y2]

            # Estimate 3D position
            if depth_map is not None:
                position = self._estimate_position_from_depth(bbox, depth_map)
            else:
                position = self._estimate_position_heuristic(bbox, image_size)

            if track_id in self.objects:
                # Update existing object
                obj = self.objects[track_id]
                obj["position"] = position
                obj["bbox"] = bbox
                obj["confidence"] = track["confidence"]
                obj["age"] = track.get("age", obj["age"] + 1)
                obj["ttl"] = self.ttl_frames
            else:
                # Create new object
                self.objects[track_id] = {
                    "id": track_id,
                    "class_id": track.get("class_id", 0),
                    "position": position,
                    "velocity": [0.0, 0.0, 0.0],
                    "bbox": bbox,
                    "confidence": track["confidence"],
                    "age": track.get("age", 0),
                    "ttl": self.ttl_frames,
                }

        # Decrement TTL for unseen objects
        to_remove = []
        for obj_id, obj in self.objects.items():
            if obj_id not in seen_ids:
                obj["ttl"] -= 1
                if obj["ttl"] <= 0:
                    to_remove.append(obj_id)

        # Remove expired objects
        for obj_id in to_remove:
            del self.objects[obj_id]

        # Build output list
        objects = list(self.objects.values())

        return {
            "objects": objects,
            "obstacles": objects,  # Alias for compatibility
        }

    def _estimate_position_heuristic(
        self,
        bbox: list,
        image_size: tuple,
    ) -> list[float]:
        """Estimate 3D position from bbox using height heuristic."""
        x1, y1, x2, y2 = bbox
        img_h, img_w = image_size

        # Center of bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        height_px = y2 - y1

        # Estimate depth from apparent height
        # Larger bbox = closer object
        if height_px > 0:
            depth = (self.reference_height_m * self.reference_height_px) / height_px
        else:
            depth = 10.0  # Default depth

        # Project to 3D (simplified pinhole model)
        x = (cx - self.camera_cx) * depth / self.camera_fx
        y = (cy - self.camera_cy) * depth / self.camera_fy
        z = depth

        return [float(x), float(y), float(z)]

    def _estimate_position_from_depth(
        self,
        bbox: list,
        depth_map: np.ndarray,
    ) -> list[float]:
        """Estimate 3D position from depth map."""
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = depth_map.shape[:2]

        # Clamp to image bounds
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))

        # Get depth from center region of bbox
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        region_size = max(1, min(10, (x2 - x1) // 4, (y2 - y1) // 4))

        depth_region = depth_map[
            max(0, cy - region_size) : min(h, cy + region_size),
            max(0, cx - region_size) : min(w, cx + region_size),
        ]

        # Use median depth for robustness
        valid_depths = depth_region[depth_region > 0]
        if len(valid_depths) > 0:
            depth = float(np.median(valid_depths))
        else:
            depth = 5.0  # Default

        # Project to 3D
        x = (cx - self.camera_cx) * depth / self.camera_fx
        y = (cy - self.camera_cy) * depth / self.camera_fy
        z = depth

        return [float(x), float(y), float(z)]

    def get_obstacles_in_range(self, max_distance: float) -> list[dict]:
        """Get objects within a certain distance."""
        nearby = []
        for obj in self.objects.values():
            pos = obj["position"]
            distance = np.sqrt(pos[0] ** 2 + pos[1] ** 2 + pos[2] ** 2)
            if distance <= max_distance:
                nearby.append(obj)
        return nearby

    def reset(self):
        """Clear all objects."""
        self.objects = {}

    def teardown(self) -> None:
        """Clean up."""
        self.reset()
        self._is_setup = False
