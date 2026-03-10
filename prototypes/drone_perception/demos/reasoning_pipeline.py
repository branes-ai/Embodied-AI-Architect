"""
3D Reasoning Pipeline Demo.

Demonstrates advanced reasoning capabilities on top of perception:
- Trajectory prediction (where will objects move?)
- Collision detection (will anything collide?)
- Spatial analysis (how are objects positioned?)
- Behavior classification (what are objects doing?)

Can be used with any perception backend:
- Monocular (full_pipeline.py)
- Stereo (stereo_pipeline.py)
- LiDAR + Camera (lidar_camera_pipeline.py)
- Wide-angle (wide_angle_pipeline.py)
"""

import sys
from pathlib import Path
import argparse
import time
import time as time_module
import cv2
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.sensors import MonocularCamera
from lib.detection import YOLODetector
from lib.tracking import ByteTracker
from lib.scene_graph import SceneGraphManager
from lib.common import project_2d_to_3d

# Reasoning modules
from lib.reasoning import (
    TrajectoryPredictor,
    CollisionDetector,
    SpatialAnalyzer,
    BehaviorClassifier,
    RiskLevel
)


def draw_trajectory_prediction(
    image: np.ndarray,
    predictor: TrajectoryPredictor,
    predictions: list,
    camera_params
) -> np.ndarray:
    """Draw predicted trajectories on image."""
    for pred in predictions:
        # Get smooth trajectory visualization (reduced resolution for speed)
        positions, colors = predictor.visualize_trajectory(pred, resolution=15)  # Was 50

        # Project 3D points to 2D
        for i, pos_3d in enumerate(positions):
            # Simple projection (assuming camera looking along X axis)
            if pos_3d[2] > 0.1:  # Valid depth
                x_2d = int(camera_params.fx * pos_3d[0] / pos_3d[2] + camera_params.cx)
                y_2d = int(camera_params.fy * pos_3d[1] / pos_3d[2] + camera_params.cy)

                # Check if in image bounds
                if 0 <= x_2d < image.shape[1] and 0 <= y_2d < image.shape[0]:
                    color = tuple(int(c) for c in colors[i])
                    cv2.circle(image, (x_2d, y_2d), 3, color, -1)

        # Draw endpoint (blue circle in BGR format)
        endpoint = pred.predicted_endpoint
        if endpoint[2] > 0.1:
            x_2d = int(camera_params.fx * endpoint[0] / endpoint[2] + camera_params.cx)
            y_2d = int(camera_params.fy * endpoint[1] / endpoint[2] + camera_params.cy)

            if 0 <= x_2d < image.shape[1] and 0 <= y_2d < image.shape[0]:
                # Blue circle (BGR format)
                cv2.circle(image, (x_2d, y_2d), 8, (255, 0, 0), 2)

    return image


def draw_collision_warnings(
    image: np.ndarray,
    collision_risks: list
) -> np.ndarray:
    """Draw collision warnings on image."""
    y_offset = 100

    for i, risk in enumerate(collision_risks[:5]):  # Show top 5 risks
        # Color by risk level
        if risk.risk_level == RiskLevel.CRITICAL:
            color = (0, 0, 255)  # Red
            prefix = "CRITICAL"
        elif risk.risk_level == RiskLevel.HIGH:
            color = (0, 165, 255)  # Orange
            prefix = "HIGH"
        elif risk.risk_level == RiskLevel.MEDIUM:
            color = (0, 255, 255)  # Yellow
            prefix = "MEDIUM"
        else:
            color = (0, 255, 0)  # Green
            prefix = "LOW"

        # Build warning text
        if risk.time_to_collision < 100:
            text = f"{prefix}: {risk.object_a_class} & {risk.object_b_class} - {risk.time_to_collision:.1f}s"
        else:
            text = f"{prefix}: {risk.object_a_class} & {risk.object_b_class} - {risk.closest_distance:.1f}m"

        cv2.putText(image, text, (10, y_offset + i*30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return image


def draw_behavior_info(
    image: np.ndarray,
    behaviors: list
) -> np.ndarray:
    """Draw behavior information on image."""
    y_start = image.shape[0] - 150

    # Show behaviors for up to 3 objects
    for i, behavior in enumerate(behaviors[:3]):
        text = f"ID{behavior.track_id} ({behavior.class_name}): {behavior.description}"

        # Color by threat level
        color = (0, 0, 255) if behavior.is_potential_threat() else (0, 255, 0)

        cv2.putText(image, text, (10, y_start + i*30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return image


def main():
    parser = argparse.ArgumentParser(description='3D Reasoning Pipeline Demo')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera device ID')
    parser.add_argument('--model', type=str, default='s',
                        choices=['n', 's', 'm', 'l', 'x'],
                        help='YOLO model size')
    parser.add_argument('--classes', type=int, nargs='+', default=None,
                        help='Filter detection classes')
    parser.add_argument('--conf-threshold', type=float, default=0.3,
                        help='Detection confidence threshold')
    parser.add_argument('--prediction-horizon', type=float, default=3.0,
                        help='Trajectory prediction time (seconds)')
    parser.add_argument('--prediction-method', type=str, default='constant_velocity',
                        choices=['constant_velocity', 'constant_acceleration', 'physics'],
                        help='Trajectory prediction method')
    parser.add_argument('--drone-x', type=float, default=0.0,
                        help='Simulated drone X position')
    parser.add_argument('--drone-y', type=float, default=0.0,
                        help='Simulated drone Y position')
    parser.add_argument('--drone-z', type=float, default=1.5,
                        help='Simulated drone Z position (height)')
    parser.add_argument('--save-video', type=str, default=None,
                        help='Save annotated video')

    args = parser.parse_args()

    print("="*70)
    print("3D REASONING PIPELINE")
    print("="*70)
    print(f"Prediction horizon: {args.prediction_horizon}s")
    print(f"Prediction method: {args.prediction_method}")
    print(f"Drone position: ({args.drone_x}, {args.drone_y}, {args.drone_z})")
    print()
    print("Capabilities:")
    print("  ✓ Trajectory prediction - Where will objects move?")
    print("  ✓ Collision detection - Risk assessment")
    print("  ✓ Spatial analysis - Object relationships")
    print("  ✓ Behavior classification - What are objects doing?")
    print()
    print("Controls:")
    print("  'q' - Quit")
    print("  'p' - Pause/Resume")
    print("  't' - Toggle trajectory visualization")
    print("  'c' - Toggle collision warnings")
    print("  'b' - Toggle behavior info")
    print("  's' - Print scene description")
    print("  'x' - Clear trajectory histories")
    print("  'd' - Debug: show object velocities")
    print("="*70)

    # Initialize sensor
    print("[1/6] Initializing camera...")
    sensor = MonocularCamera(source=args.camera)

    if not sensor.is_opened():
        print("ERROR: Failed to initialize camera")
        return 1

    # Initialize detector
    print("[2/6] Loading detection model...")
    detector = YOLODetector(
        model_size=args.model,
        conf_threshold=args.conf_threshold,
        classes=args.classes
    )

    # Initialize tracker
    print("[3/6] Initializing tracker...")
    tracker = ByteTracker(
        high_thresh=0.5,     # Detection confidence threshold
        low_thresh=0.1,      # Low confidence detections (catch partial detections)
        match_thresh=0.3,    # Very permissive IOU for re-ID (was 0.5, default 0.8)
        max_time_lost=30     # Keep tracks alive for 30 frames (~1s) for re-ID
    )

    # Initialize scene graph
    print("[4/6] Creating scene graph...")
    scene_graph = SceneGraphManager(ttl_seconds=1.5)  # Match ByteTracker timeout + buffer

    # Initialize reasoning modules
    print("[5/6] Initializing reasoning modules...")
    predictor = TrajectoryPredictor(
        prediction_horizon=args.prediction_horizon,
        method=args.prediction_method
    )
    collision_detector = CollisionDetector()
    spatial_analyzer = SpatialAnalyzer()
    behavior_classifier = BehaviorClassifier()

    print("[6/6] Ready!")

    # Simulated drone position
    drone_position = np.array([args.drone_x, args.drone_y, args.drone_z])

    # Video writer
    video_writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        test_frame = sensor.get_frame()
        if test_frame is not None:
            h, w = test_frame.image.shape[:2]
            video_writer = cv2.VideoWriter(args.save_video, fourcc, 30, (w, h))

    print()
    print("="*70)
    print("STARTING PIPELINE")
    print("="*70)
    print()

    # Main loop
    frame_count = 0
    paused = False
    show_trajectories = True
    show_collisions = True
    show_behaviors = True

    # Prediction optimization: only predict every N frames
    prediction_skip_frames = 3  # Predict every 3rd frame
    predictions = []
    collision_risks = []
    behaviors = []

    try:
        while True:
            if not paused:
                # Get frame
                frame = sensor.get_frame()
                if frame is None:
                    break

                # Detect and track
                detections = detector.detect(frame.image)
                tracks = tracker.update(detections)

                # Update scene graph (use simple depth for monocular)
                for track in tracks:
                    # Estimate depth using bbox size (rough approximation)
                    # Larger bbox = closer object, smaller bbox = farther object
                    # Assuming average person height of 1.7m
                    assumed_height = 1.7  # meters
                    bbox_height_pixels = track.bbox.h
                    if bbox_height_pixels > 10:  # Avoid division by zero
                        estimated_depth = (assumed_height * frame.camera_params.fy) / bbox_height_pixels
                        estimated_depth = max(1.0, min(estimated_depth, 20.0))  # Clamp to 1-20m
                    else:
                        estimated_depth = 5.0

                    cx, cy = track.bbox.center
                    position_3d = project_2d_to_3d(cx, cy, estimated_depth, frame.camera_params)

                    scene_graph.update_node(
                        track_id=track.id,  # Track uses 'id' not 'track_id'
                        class_name=track.class_name,
                        bbox=track.bbox,
                        position_3d=position_3d,
                        confidence=track.confidence
                    )

                # Prune stale objects from scene graph (must be called manually since we use update_node)
                current_time = time_module.time()
                stale_ids = []
                for obj_id, obj in scene_graph.objects.items():
                    if current_time - obj.last_seen > scene_graph.ttl_seconds:
                        stale_ids.append(obj_id)

                # Remove stale objects
                if len(stale_ids) > 0:
                    print(f"[Pruning] Removing {len(stale_ids)} stale objects: {stale_ids}")
                for obj_id in stale_ids:
                    if obj_id in scene_graph.objects:
                        del scene_graph.objects[obj_id]
                    if obj_id in scene_graph.kalman_filters:
                        del scene_graph.kalman_filters[obj_id]

                # Get active nodes (only objects currently being tracked)
                current_track_ids = {track.id for track in tracks}
                nodes = [node for node in scene_graph.get_active_nodes()
                        if node.track_id in current_track_ids]

                # Run reasoning modules (skip frames for performance)
                if frame_count % prediction_skip_frames == 0:
                    predictions = predictor.predict_all(nodes)
                    collision_risks = collision_detector.check_all_collisions(
                        predictions,
                        drone_position=drone_position
                    )
                    behaviors = behavior_classifier.classify_all(nodes, drone_position)

                # Visualize
                # Convert RGB to BGR for OpenCV display
                viz_frame = cv2.cvtColor(frame.image, cv2.COLOR_RGB2BGR)

                # Draw tracks
                for track in tracks:
                    x, y, w, h = int(track.bbox.x), int(track.bbox.y), \
                                int(track.bbox.w), int(track.bbox.h)
                    cv2.rectangle(viz_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    label = f"{track.id}: {track.class_name}"
                    cv2.putText(viz_frame, label, (x, y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Draw trajectories (only for currently visible objects)
                if show_trajectories and len(predictions) > 0:
                    # Filter predictions to only show those for current tracks
                    visible_predictions = [p for p in predictions if p.track_id in current_track_ids]
                    if len(visible_predictions) > 0:
                        viz_frame = draw_trajectory_prediction(
                            viz_frame, predictor, visible_predictions, frame.camera_params
                        )

                # Draw collision warnings
                if show_collisions:
                    viz_frame = draw_collision_warnings(viz_frame, collision_risks)

                # Draw behavior info
                if show_behaviors:
                    viz_frame = draw_behavior_info(viz_frame, behaviors)

                # Info overlay
                high_risk_count = sum(1 for r in collision_risks
                                     if r.risk_level.value >= RiskLevel.HIGH.value)
                total_scene_objects = len(scene_graph.objects)
                info = f"Frame: {frame_count} | Tracks: {len(tracks)} | Scene: {total_scene_objects} | Active: {len(nodes)} | Risks: {high_risk_count}"
                cv2.putText(viz_frame, info, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Show
                cv2.imshow("Reasoning Pipeline", viz_frame)

                # Save video
                if video_writer is not None:
                    video_writer.write(viz_frame)

                frame_count += 1

            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = not paused
                print(f"{'Paused' if paused else 'Resumed'}")
            elif key == ord('t'):
                show_trajectories = not show_trajectories
                print(f"Trajectories: {'ON' if show_trajectories else 'OFF'}")
            elif key == ord('c'):
                show_collisions = not show_collisions
                print(f"Collision warnings: {'ON' if show_collisions else 'OFF'}")
            elif key == ord('b'):
                show_behaviors = not show_behaviors
                print(f"Behavior info: {'ON' if show_behaviors else 'OFF'}")
            elif key == ord('x'):
                # Clear trajectory histories AND predictions
                scene_graph.clear_trajectories()
                old_count = len(predictions)
                predictions = []  # Clear predicted trajectories
                collision_risks = []  # Clear collision warnings
                behaviors = []  # Clear behaviors
                print(f"Cleared {old_count} predictions and all trajectory histories")
            elif key == ord('d'):
                # Debug: show object info
                print("\n" + "="*60)
                print("DEBUG INFO")
                print("="*60)
                print(f"Current tracks from ByteTracker: {len(tracks)}")
                print(f"Objects in scene graph: {len(scene_graph.objects)}")
                print(f"Active nodes (filtered): {len(nodes)}")
                print()
                current_time = time_module.time()
                for obj_id, obj in scene_graph.objects.items():
                    vel_mag = np.linalg.norm(obj.velocity_3d)
                    age = current_time - obj.last_seen
                    in_current_tracks = obj_id in {track.id for track in tracks}
                    print(f"ID {obj_id} ({obj.class_name}):")
                    print(f"  Position: {obj.position_3d}")
                    print(f"  Velocity: {obj.velocity_3d} (mag: {vel_mag:.3f} m/s)")
                    print(f"  History length: {len(obj.trajectory_3d)}")
                    print(f"  Last seen: {age:.2f}s ago")
                    print(f"  In current tracks: {in_current_tracks}")
                    print()
                print(f"Total predictions: {len(predictions)}")
                print("="*60 + "\n")
            elif key == ord('s'):
                # Print scene description
                print("\n" + "="*60)
                print(spatial_analyzer.describe_scene(nodes, drone_position))
                print("="*60 + "\n")

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        print()
        print("="*70)
        print("SHUTTING DOWN")
        print("="*70)

        sensor.release()
        cv2.destroyAllWindows()
        if video_writer is not None:
            video_writer.release()
            print(f"Saved video: {args.save_video}")

        print(f"\nProcessed {frame_count} frames")
        print("Done!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
