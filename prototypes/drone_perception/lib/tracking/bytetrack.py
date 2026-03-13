"""ByteTrack: Simple, fast multi-object tracking using IOU + Kalman filter."""

from typing import List
from collections import deque

import numpy as np

from ..common import Detection, Track, BBox
from .kalman_filter import KalmanBoxFilter


class TrackState:
    """Track states."""
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class STrack:
    """Single track object."""

    _count = 0  # Global track ID counter

    def __init__(self, detection: Detection):
        """Initialize track from detection."""
        # Identity
        self.track_id = self._next_id()
        self.class_id = detection.class_id
        self.class_name = detection.class_name

        # State
        self.state = TrackState.New
        self.age = 0
        self.hits = 1
        self.time_since_update = 0

        # Bounding box
        bbox_xywh = np.array([
            detection.bbox.x,
            detection.bbox.y,
            detection.bbox.w,
            detection.bbox.h
        ])
        self.kalman = KalmanBoxFilter(bbox_xywh)
        self.confidence = detection.confidence

        # History
        self.history = deque(maxlen=30)
        self.history.append(bbox_xywh)

    @classmethod
    def _next_id(cls):
        """Get next track ID."""
        cls._count += 1
        return cls._count

    def predict(self, dt: float = None):
        """Predict next position."""
        self.kalman.predict(dt=dt)
        self.age += 1
        self.time_since_update += 1

    def update(self, detection: Detection):
        """Update with new detection."""
        bbox_xywh = np.array([
            detection.bbox.x,
            detection.bbox.y,
            detection.bbox.w,
            detection.bbox.h
        ])
        self.kalman.update(bbox_xywh)
        self.confidence = detection.confidence
        self.time_since_update = 0
        self.hits += 1
        self.state = TrackState.Tracked
        self.history.append(bbox_xywh)

    def mark_lost(self):
        """Mark track as lost."""
        self.state = TrackState.Lost

    def mark_removed(self):
        """Mark track as removed."""
        self.state = TrackState.Removed

    def get_bbox(self) -> BBox:
        """Get current bounding box."""
        xywh = self.kalman.get_bbox()
        return BBox(x=xywh[0], y=xywh[1], w=xywh[2], h=xywh[3])

    def to_track(self) -> Track:
        """Convert to Track object."""
        return Track(
            id=self.track_id,
            bbox=self.get_bbox(),
            class_id=self.class_id,
            class_name=self.class_name,
            confidence=self.confidence,
            age=self.age,
            hits=self.hits,
            time_since_update=self.time_since_update,
            state=self.kalman.mean
        )


class ByteTracker:
    """
    ByteTrack: Simple multi-object tracker.

    Uses IOU matching + Kalman filter for motion prediction.
    No deep learning ReID network needed.

    Reference: https://arxiv.org/abs/2110.06864
    """

    def __init__(
        self,
        high_thresh: float = 0.5,
        low_thresh: float = 0.2,
        match_thresh: float = 0.3,
        max_time_lost: int = 30,
        n_init: int = 3
    ):
        """
        Initialize ByteTracker.

        Args:
            high_thresh: High confidence threshold for first association
            low_thresh: Low confidence threshold for second association
            match_thresh: IOU threshold for matching
            max_time_lost: Maximum frames to keep lost tracks
            n_init: Consecutive detections required to confirm a track
        """
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.match_thresh = match_thresh
        self.max_time_lost = max_time_lost
        self.n_init = n_init

        self.tracked_tracks: List[STrack] = []
        self.tentative_tracks: List[STrack] = []
        self.lost_tracks: List[STrack] = []
        self.removed_tracks: List[STrack] = []

        self.frame_id = 0

    def update(self, detections: List[Detection]) -> List[Track]:
        """
        Update tracker with new detections.

        Args:
            detections: List of detections from current frame

        Returns:
            List of active tracks
        """
        self.frame_id += 1

        # Split detections by confidence
        high_dets = [d for d in detections if d.confidence >= self.high_thresh]
        low_dets = [
            d for d in detections
            if self.low_thresh <= d.confidence < self.high_thresh
        ]

        # Predict current locations for all track pools
        for track in self.tracked_tracks:
            track.predict()
        for track in self.tentative_tracks:
            track.predict()

        # ── First association: high-confidence detections with tracked tracks ──
        matched_tracks, unmatched_tracks, unmatched_dets = self._associate(
            self.tracked_tracks, high_dets
        )

        # Update matched tracks
        for track_idx, det_idx in matched_tracks:
            self.tracked_tracks[track_idx].update(high_dets[det_idx])

        # ── Tentative track association: unmatched high dets with tentative tracks ──
        remaining_high = [high_dets[i] for i in unmatched_dets]
        matched_tent, unmatched_tent, unmatched_remaining = self._associate(
            self.tentative_tracks, remaining_high
        )

        # Update matched tentative tracks; promote if enough hits
        promoted_indices = set()
        for track_idx, det_idx in matched_tent:
            tent_track = self.tentative_tracks[track_idx]
            tent_track.update(remaining_high[det_idx])
            if tent_track.hits >= self.n_init:
                tent_track.state = TrackState.Tracked
                self.tracked_tracks.append(tent_track)
                promoted_indices.add(track_idx)

        # Remove promoted and unmatched tentative tracks (no grace period)
        remove_tent = promoted_indices | set(unmatched_tent)
        for idx in sorted(remove_tent, reverse=True):
            del self.tentative_tracks[idx]

        # Update remaining unmatched high detections after tentative consumed some
        unmatched_high_dets = [remaining_high[i] for i in unmatched_remaining]

        # ── Second association: remaining dets + low dets with lost tracks ──
        all_unmatched_dets = unmatched_high_dets + low_dets

        matched_lost, unmatched_lost, unmatched_dets2 = self._associate(
            self.lost_tracks, all_unmatched_dets
        )

        # Reactivate lost tracks
        for track_idx, det_idx in matched_lost:
            self.lost_tracks[track_idx].update(all_unmatched_dets[det_idx])
            self.lost_tracks[track_idx].state = TrackState.Tracked
            self.tracked_tracks.append(self.lost_tracks[track_idx])

        # Remove reactivated from lost
        for track_idx, _ in sorted(matched_lost, reverse=True):
            del self.lost_tracks[track_idx]

        # Mark unmatched tracked as lost
        for track_idx in unmatched_tracks:
            track = self.tracked_tracks[track_idx]
            track.mark_lost()
            self.lost_tracks.append(track)

        # Remove lost from tracked
        for track_idx in sorted(unmatched_tracks, reverse=True):
            del self.tracked_tracks[track_idx]

        # ── Initialize new tentative tracks from remaining high-conf detections ──
        for det_idx in unmatched_dets2:
            det = all_unmatched_dets[det_idx]
            if det.confidence >= self.high_thresh:
                new_track = STrack(det)
                # Stays TrackState.New — must accumulate n_init hits to be confirmed
                self.tentative_tracks.append(new_track)

        # Remove tracks that have been lost too long
        self.lost_tracks = [
            t for t in self.lost_tracks
            if t.time_since_update <= self.max_time_lost
        ]

        # Return only confirmed tracks
        return [t.to_track() for t in self.tracked_tracks]

    def _associate(
        self,
        tracks: List[STrack],
        detections: List[Detection]
    ) -> tuple:
        """
        Associate tracks with detections using IOU.

        Returns:
            (matched_pairs, unmatched_track_indices, unmatched_detection_indices)
        """
        if len(tracks) == 0 or len(detections) == 0:
            return [], list(range(len(tracks))), list(range(len(detections)))

        # Compute IOU matrix
        iou_matrix = np.zeros((len(tracks), len(detections)))
        for i, track in enumerate(tracks):
            track_bbox = track.get_bbox()
            for j, det in enumerate(detections):
                iou_matrix[i, j] = track_bbox.iou(det.bbox)

        # Greedy matching (highest IOU first)
        matched = []
        unmatched_tracks = list(range(len(tracks)))
        unmatched_dets = list(range(len(detections)))

        while len(unmatched_tracks) > 0 and len(unmatched_dets) > 0:
            # Find best match
            max_iou = 0
            best_track = None
            best_det = None

            for i in unmatched_tracks:
                for j in unmatched_dets:
                    if iou_matrix[i, j] > max_iou:
                        max_iou = iou_matrix[i, j]
                        best_track = i
                        best_det = j

            # If best match is above threshold, accept it
            if max_iou >= self.match_thresh:
                matched.append((best_track, best_det))
                unmatched_tracks.remove(best_track)
                unmatched_dets.remove(best_det)
            else:
                break

        return matched, unmatched_tracks, unmatched_dets

    def reset(self):
        """Reset tracker."""
        self.tracked_tracks = []
        self.tentative_tracks = []
        self.lost_tracks = []
        self.removed_tracks = []
        self.frame_id = 0
        STrack._count = 0
