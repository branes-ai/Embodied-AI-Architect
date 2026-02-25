"""ByteTrack multi-object tracker.

Simple, fast tracker using IoU association and Kalman filtering.
"""

from collections import deque
from typing import Any

import numpy as np

from ..base import Operator


class KalmanBoxFilter:
    """Kalman filter for 2D bounding box tracking.

    State: [x, y, w, h, vx, vy, vw, vh]
    Measurement: [x, y, w, h]
    """

    def __init__(self, bbox: np.ndarray):
        """Initialize filter with initial bounding box.

        Args:
            bbox: [x_center, y_center, width, height]
        """
        # State transition matrix
        self.F = np.eye(8)
        self.F[0, 4] = 1  # x += vx
        self.F[1, 5] = 1  # y += vy
        self.F[2, 6] = 1  # w += vw
        self.F[3, 7] = 1  # h += vh

        # Measurement matrix
        self.H = np.eye(4, 8)

        # Process noise
        self.Q = np.eye(8) * 0.01
        self.Q[4:, 4:] *= 10  # Higher noise for velocities

        # Measurement noise
        self.R = np.eye(4) * 1.0

        # State covariance
        self.P = np.eye(8) * 10.0
        self.P[4:, 4:] *= 1000  # High uncertainty in initial velocities

        # State vector
        self.x = np.zeros(8)
        self.x[:4] = bbox

    def predict(self) -> np.ndarray:
        """Predict next state."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:4]

    def update(self, bbox: np.ndarray) -> np.ndarray:
        """Update with measurement."""
        y = bbox - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P
        return self.x[:4]

    def get_bbox(self) -> np.ndarray:
        """Get current bounding box estimate."""
        return self.x[:4].copy()


class STrack:
    """Single track object."""

    _count = 0

    def __init__(self, detection: dict):
        """Initialize track from detection."""
        STrack._count += 1
        self.track_id = STrack._count

        self.class_id = detection.get("class_id", 0)
        self.confidence = detection["confidence"]

        # Convert xyxy to xywh
        bbox = detection["bbox"]
        x1, y1, x2, y2 = bbox
        bbox_xywh = np.array(
            [
                (x1 + x2) / 2,
                (y1 + y2) / 2,
                x2 - x1,
                y2 - y1,
            ]
        )

        self.kalman = KalmanBoxFilter(bbox_xywh)
        self.hits = 1
        self.age = 0
        self.time_since_update = 0
        self.is_activated = False
        self.history = deque(maxlen=50)
        self.history.append(bbox_xywh)

    def predict(self):
        """Predict next state."""
        self.kalman.predict()
        self.age += 1
        self.time_since_update += 1

    def update(self, detection: dict):
        """Update with new detection."""
        bbox = detection["bbox"]
        x1, y1, x2, y2 = bbox
        bbox_xywh = np.array(
            [
                (x1 + x2) / 2,
                (y1 + y2) / 2,
                x2 - x1,
                y2 - y1,
            ]
        )

        self.kalman.update(bbox_xywh)
        self.confidence = detection["confidence"]
        self.hits += 1
        self.time_since_update = 0
        self.is_activated = True
        self.history.append(bbox_xywh)

    def get_bbox_xyxy(self) -> list[float]:
        """Get current bounding box in xyxy format."""
        xywh = self.kalman.get_bbox()
        x, y, w, h = xywh
        return [x - w / 2, y - h / 2, x + w / 2, y + h / 2]

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "track_id": self.track_id,
            "bbox": self.get_bbox_xyxy(),
            "class_id": self.class_id,
            "confidence": self.confidence,
            "age": self.age,
            "hits": self.hits,
        }


class ByteTrack(Operator):
    """ByteTrack multi-object tracker.

    Two-stage association:
    1. High-confidence detections matched to existing tracks
    2. Low-confidence detections matched to remaining tracks
    """

    def __init__(self):
        super().__init__(operator_id="bytetrack")
        self.tracks: list[STrack] = []
        self.track_thresh = 0.5
        self.match_thresh = 0.8
        self.track_buffer = 30
        self.frame_id = 0

    def setup(self, config: dict[str, Any], execution_target: str = "cpu") -> None:
        """Initialize tracker.

        Args:
            config: Configuration with optional keys:
                - track_thresh: Threshold for high-confidence detections
                - match_thresh: IoU threshold for matching
                - track_buffer: Frames to keep lost tracks
            execution_target: Only cpu supported
        """
        if execution_target != "cpu":
            print(f"[ByteTrack] Warning: Only CPU supported, ignoring target={execution_target}")

        self._execution_target = "cpu"
        self._config = config

        self.track_thresh = config.get("track_thresh", 0.5)
        self.match_thresh = config.get("match_thresh", 0.8)
        self.track_buffer = config.get("track_buffer", 30)

        # Reset state
        self.tracks = []
        self.frame_id = 0
        STrack._count = 0

        self._is_setup = True
        print(f"[ByteTrack] Ready (thresh={self.track_thresh}, buffer={self.track_buffer})")

    def process(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Process detections and update tracks.

        Args:
            inputs: Dictionary with 'detections' key containing list of detection dicts

        Returns:
            Dictionary with 'tracks' key containing list of track dicts
        """
        detections = inputs["detections"]
        self.frame_id += 1

        # Split detections by confidence
        high_conf = [d for d in detections if d["confidence"] >= self.track_thresh]
        low_conf = [d for d in detections if d["confidence"] < self.track_thresh]

        # Predict all tracks
        for track in self.tracks:
            track.predict()

        # First association: high-confidence with all tracks
        matched_tracks, unmatched_tracks, unmatched_dets = self._associate(
            self.tracks, high_conf, self.match_thresh
        )

        # Update matched tracks
        for track, det in matched_tracks:
            track.update(det)

        # Second association: low-confidence with remaining tracks
        matched_low, still_unmatched, _ = self._associate(
            unmatched_tracks, low_conf, self.match_thresh
        )

        for track, det in matched_low:
            track.update(det)

        # Create new tracks from unmatched high-confidence detections
        for det in unmatched_dets:
            if det["confidence"] >= self.track_thresh:
                new_track = STrack(det)
                new_track.is_activated = True
                self.tracks.append(new_track)

        # Remove lost tracks
        self.tracks = [t for t in self.tracks if t.time_since_update < self.track_buffer]

        # Output active tracks
        active_tracks = [t.to_dict() for t in self.tracks if t.is_activated]

        return {"tracks": active_tracks}

    def _associate(
        self,
        tracks: list[STrack],
        detections: list[dict],
        iou_thresh: float,
    ) -> tuple[list, list, list]:
        """Associate tracks with detections using IoU.

        Returns:
            (matched_pairs, unmatched_tracks, unmatched_detections)
        """
        if len(tracks) == 0 or len(detections) == 0:
            return [], tracks, detections

        # Compute IoU matrix
        iou_matrix = np.zeros((len(tracks), len(detections)))
        for i, track in enumerate(tracks):
            track_box = track.get_bbox_xyxy()
            for j, det in enumerate(detections):
                det_box = det["bbox"]
                iou_matrix[i, j] = self._iou(track_box, det_box)

        # Greedy matching
        matched_pairs = []
        matched_track_ids = set()
        matched_det_ids = set()

        while True:
            if iou_matrix.size == 0:
                break

            max_iou = iou_matrix.max()
            if max_iou < iou_thresh:
                break

            idx = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
            i, j = idx

            matched_pairs.append((tracks[i], detections[j]))
            matched_track_ids.add(i)
            matched_det_ids.add(j)

            # Remove matched row and column from consideration
            iou_matrix[i, :] = 0
            iou_matrix[:, j] = 0

        unmatched_tracks = [t for i, t in enumerate(tracks) if i not in matched_track_ids]
        unmatched_dets = [d for j, d in enumerate(detections) if j not in matched_det_ids]

        return matched_pairs, unmatched_tracks, unmatched_dets

    def _iou(self, box1: list, box2: list) -> float:
        """Compute IoU between two boxes in xyxy format."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter

        return inter / union if union > 0 else 0

    def reset(self):
        """Reset tracker state."""
        self.tracks = []
        self.frame_id = 0
        STrack._count = 0

    def teardown(self) -> None:
        """Clean up."""
        self.reset()
        self._is_setup = False
