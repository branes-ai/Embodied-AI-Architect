"""
LiDAR + Camera fusion perception pipeline demo.

Demonstrates:
- LiDAR + Camera sensor fusion
- Metric depth from 3D LiDAR point clouds
- Projected LiDAR depth for object detection
- Accurate 3D localization and tracking

Supports:
- Velodyne (VLP-16, VLP-32, HDL-64E)
- Ouster (OS0, OS1, OS2)
- Livox (Avia, Mid-360)
- ROS point cloud topics
- Pre-recorded point cloud files
"""

import sys
from pathlib import Path
import argparse
import time
import cv2
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.sensors import LiDARCameraSensor
from lib.detection import YOLODetector
from lib.tracking import ByteTracker
from lib.scene_graph import SceneGraphManager
from lib.visualization import LiveViewer3D
from lib.common import project_2d_to_3d, CameraParams


def get_depth_at_bbox(bbox, depth_map, method='median'):
    """
    Extract robust depth from LiDAR depth map at bbox location.

    Args:
        bbox: Bounding box (x, y, w, h)
        depth_map: Projected LiDAR depth map (H, W) in meters
        method: 'center', 'median', or 'bottom'

    Returns:
        Depth in meters, or None if invalid
    """
    if depth_map is None:
        return None

    x, y, w, h = int(bbox.x), int(bbox.y), int(bbox.w), int(bbox.h)

    if method == 'center':
        # Get depth at center point
        cx, cy = int(x + w/2), int(y + h/2)
        cy = max(0, min(cy, depth_map.shape[0] - 1))
        cx = max(0, min(cx, depth_map.shape[1] - 1))
        depth = depth_map[cy, cx]

    elif method == 'median':
        # Get median depth in bbox (more robust for sparse LiDAR)
        y1, y2 = max(0, y), min(depth_map.shape[0], y + h)
        x1, x2 = max(0, x), min(depth_map.shape[1], x + w)

        roi = depth_map[y1:y2, x1:x2]
        valid_depths = roi[(roi > 0.1) & (roi < 100.0)]  # Extended range for LiDAR

        if len(valid_depths) == 0:
            return None

        depth = np.median(valid_depths)

    elif method == 'bottom':
        # Get depth at bottom center (for ground-based objects)
        cx, cy = int(x + w/2), int(y + h)
        cy = max(0, min(cy, depth_map.shape[0] - 1))
        cx = max(0, min(cx, depth_map.shape[1] - 1))

        # Median filter in small region around bottom
        radius = 10  # Larger radius for sparse LiDAR
        y1, y2 = max(0, cy - radius), min(depth_map.shape[0], cy + radius)
        x1, x2 = max(0, cx - radius), min(depth_map.shape[1], cx + radius)

        roi = depth_map[y1:y2, x1:x2]
        valid_depths = roi[(roi > 0.1) & (roi < 100.0)]

        if len(valid_depths) == 0:
            return None

        depth = np.median(valid_depths)

    else:
        raise ValueError(f"Unknown method: {method}")

    # Validate depth range (LiDAR can see much farther)
    if depth < 0.1 or depth > 100.0:
        return None

    return float(depth)


def visualize_lidar_overlay(image, depth_map):
    """
    Visualize LiDAR depth overlay on RGB image.

    Args:
        image: RGB image
        depth_map: LiDAR depth map

    Returns:
        Image with LiDAR overlay
    """
    # Create depth colormap
    depth_vis = depth_map.copy()
    depth_vis = np.clip(depth_vis, 0, 50)  # Clip to 50m for visualization
    depth_vis = (depth_vis / 50.0 * 255).astype(np.uint8)
    depth_colormap = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

    # Overlay on image (only where depth > 0)
    overlay = image.copy()
    mask = depth_map > 0
    overlay[mask] = cv2.addWeighted(image[mask], 0.6, depth_colormap[mask], 0.4, 0)

    return overlay


def main():
    parser = argparse.ArgumentParser(description='LiDAR + Camera perception pipeline')
    parser.add_argument('--lidar', type=str, default='velodyne',
                        choices=['velodyne', 'ouster', 'livox', 'ros', 'file'],
                        help='LiDAR type')
    parser.add_argument('--lidar-config', type=str, default=None,
                        help='LiDAR config JSON file')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera device ID')
    parser.add_argument('--calibration', type=str, default=None,
                        help='LiDAR-camera calibration file (extrinsics + intrinsics)')
    parser.add_argument('--model', type=str, default='s',
                        choices=['n', 's', 'm', 'l', 'x'],
                        help='YOLO model size')
    parser.add_argument('--classes', type=int, nargs='+', default=None,
                        help='Filter detection classes (e.g., --classes 0 2 7 for person, car, truck)')
    parser.add_argument('--conf-threshold', type=float, default=0.3,
                        help='Detection confidence threshold')
    parser.add_argument('--depth-method', type=str, default='median',
                        choices=['center', 'median', 'bottom'],
                        help='Depth extraction method')
    parser.add_argument('--save-video', type=str, default=None,
                        help='Save annotated video to file')
    parser.add_argument('--no-viz-3d', action='store_true',
                        help='Disable 3D visualization')
    parser.add_argument('--show-lidar-overlay', action='store_true',
                        help='Show LiDAR depth overlay on image')

    args = parser.parse_args()

    # Parse LiDAR config
    lidar_config = {}
    if args.lidar_config:
        import json
        with open(args.lidar_config) as f:
            lidar_config = json.load(f)

    # Parse calibration
    camera_params = None
    extrinsics = None
    if args.calibration:
        import json
        with open(args.calibration) as f:
            calib = json.load(f)
            # Camera intrinsics
            camera_params = CameraParams(
                fx=calib['intrinsics']['fx'],
                fy=calib['intrinsics']['fy'],
                cx=calib['intrinsics']['cx'],
                cy=calib['intrinsics']['cy'],
                width=calib['intrinsics']['width'],
                height=calib['intrinsics']['height']
            )
            # LiDAR-to-camera extrinsics (4x4 matrix)
            extrinsics = np.array(calib['extrinsics'])

    print("="*70)
    print("LIDAR + CAMERA PERCEPTION PIPELINE (Phase 2)")
    print("="*70)
    print(f"LiDAR: {args.lidar}")
    print(f"Camera: {args.camera}")
    print(f"Model: yolov8{args.model} on cpu")
    print(f"Detection threshold: {args.conf_threshold}")
    print(f"Depth method: {args.depth_method}")
    if args.classes:
        print(f"Tracking classes: {args.classes}")
    print()
    print("Pipeline: LiDAR+Camera → YOLO → ByteTrack → Scene Graph → 3D Viz")
    print()
    print("Advantages over monocular:")
    print("  ✓ Real metric depth (no scale ambiguity)")
    print("  ✓ Long-range accurate depth (50m+)")
    print("  ✓ Works in any lighting (active sensor)")
    print("  ✓ Sparse but highly accurate depth")
    print()
    print("Controls:")
    print("  'q' - Quit")
    print("  'p' - Pause/Resume")
    print("  's' - Save 3D view screenshot")
    print("  'd' - Toggle depth visualization")
    print("  'l' - Toggle LiDAR overlay")
    print("="*70)

    # Initialize sensor
    print("[1/5] Initializing LiDAR + Camera...")
    sensor = LiDARCameraSensor(
        camera_id=args.camera,
        lidar_source=args.lidar,
        lidar_config=lidar_config,
        camera_params=camera_params,
        extrinsics=extrinsics
    )

    if not sensor.is_opened():
        print("ERROR: Failed to initialize sensor")
        return 1

    # Initialize detector
    print("[2/5] Loading detection model...")
    detector = YOLODetector(
        model_size=args.model,
        conf_threshold=args.conf_threshold,
        classes_filter=args.classes
    )

    # Initialize tracker
    print("[3/5] Initializing tracker...")
    tracker = ByteTracker()

    # Initialize scene graph
    print("[4/5] Creating scene graph...")
    scene_graph = SceneGraphManager()

    # Initialize visualization
    print("[5/5] Setting up visualization...")
    viewer_3d = None if args.no_viz_3d else LiveViewer3D()

    # Video writer
    video_writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # Get first frame to determine size
        test_frame = sensor.get_frame()
        if test_frame is not None:
            h, w = test_frame.image.shape[:2]
            video_writer = cv2.VideoWriter(args.save_video, fourcc, 30, (w, h))
            print(f"Saving video: {args.save_video}")

    print()
    print("="*70)
    print("STARTING PIPELINE")
    print("="*70)
    print()

    # Main loop
    frame_count = 0
    paused = False
    show_depth = False
    show_lidar = args.show_lidar_overlay

    try:
        while True:
            if not paused:
                # Get frame
                frame = sensor.get_frame()
                if frame is None:
                    print("End of stream")
                    break

                # Detect objects
                detections = detector.detect(frame.image)

                # Track objects (ByteTracker manages frame_id internally)
                tracks = tracker.update(detections)

                # Update scene graph with LiDAR depth
                for track in tracks:
                    # Get depth at bbox
                    depth = get_depth_at_bbox(track.bbox, frame.depth, method=args.depth_method)

                    if depth is not None:
                        # Project to 3D using LiDAR depth
                        cx, cy = track.bbox.center
                        position_3d = project_2d_to_3d(
                            cx, cy, depth, frame.camera_params
                        )

                        # Update scene graph
                        scene_graph.update_node(
                            track_id=track.track_id,
                            class_name=track.class_name,
                            bbox=track.bbox,
                            position_3d=position_3d,
                            confidence=track.confidence
                        )

                # Visualize
                viz_frame = frame.image.copy()

                # Draw tracks
                for track in tracks:
                    # Get node from scene graph
                    node = scene_graph.get_node(track.track_id)
                    if node is None:
                        continue

                    # Draw bbox
                    x, y, w, h = int(track.bbox.x), int(track.bbox.y), \
                                int(track.bbox.w), int(track.bbox.h)
                    cv2.rectangle(viz_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # Draw label with depth
                    depth = get_depth_at_bbox(track.bbox, frame.depth, method=args.depth_method)
                    depth_str = f"{depth:.2f}m" if depth else "N/A"
                    label = f"{track.track_id}: {track.class_name} ({depth_str})"

                    # Add velocity if available
                    if node.velocity_3d is not None:
                        vel_mag = np.linalg.norm(node.velocity_3d)
                        label += f" | {vel_mag:.1f}m/s"

                    cv2.putText(viz_frame, label, (x, y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Show depth or LiDAR overlay
                if show_depth and frame.depth is not None:
                    # Show depth map
                    depth_vis = frame.depth.copy()
                    depth_vis = np.clip(depth_vis, 0, 50)
                    depth_vis = (depth_vis / 50.0 * 255).astype(np.uint8)
                    depth_colormap = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
                    viz_frame = np.hstack([viz_frame, depth_colormap])
                elif show_lidar and frame.depth is not None:
                    # Show LiDAR overlay
                    viz_frame = visualize_lidar_overlay(viz_frame, frame.depth)

                # Info overlay
                info = f"Frame: {frame_count} | Tracks: {len(tracks)} | Objects: {scene_graph.num_nodes()}"
                cv2.putText(viz_frame, info, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Show
                cv2.imshow("LiDAR + Camera Pipeline", viz_frame)

                # Save video
                if video_writer is not None and not show_depth:
                    video_writer.write(viz_frame)

                # Update 3D viewer
                if viewer_3d is not None:
                    viewer_3d.update(scene_graph)

                frame_count += 1

            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = not paused
                print(f"{'Paused' if paused else 'Resumed'}")
            elif key == ord('d'):
                show_depth = not show_depth
                show_lidar = False
                print(f"Depth view: {'ON' if show_depth else 'OFF'}")
            elif key == ord('l'):
                show_lidar = not show_lidar
                show_depth = False
                print(f"LiDAR overlay: {'ON' if show_lidar else 'OFF'}")
            elif key == ord('s') and viewer_3d is not None:
                viewer_3d.save_screenshot()

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        print()
        print("="*70)
        print("SHUTTING DOWN")
        print("="*70)

        # Cleanup
        sensor.release()
        cv2.destroyAllWindows()
        if video_writer is not None:
            video_writer.release()
            print(f"Saved output video: {args.save_video}")
        if viewer_3d is not None:
            viewer_3d.close()

        print(f"\nProcessed {frame_count} frames")
        print("Done!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
