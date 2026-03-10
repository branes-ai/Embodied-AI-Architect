"""
Wide-angle (fisheye) camera perception pipeline with DNN depth.

Demonstrates:
- Wide-angle/fisheye camera support (180°-200° FOV like Skydio)
- Depth Any Camera (DAC) for metric depth estimation
- Zero-shot depth on fisheye cameras
- Single or dual wide-angle camera configurations
- Object detection and tracking with wide FOV

Supports:
- Single wide-angle camera + DAC monocular depth
- Dual wide-angle cameras + fused DAC depth
- Any fisheye camera model (no special training required)
"""

import sys
from pathlib import Path
import argparse
import time
import cv2
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.sensors import WideAngleCamera, DualWideAngleCamera
from lib.detection import YOLODetector
from lib.tracking import ByteTracker
from lib.scene_graph import SceneGraphManager
from lib.visualization import LiveViewer3D
from lib.common import project_2d_to_3d


def get_depth_at_bbox(bbox, depth_map, method='median'):
    """
    Extract robust depth from depth map at bbox location.

    Args:
        bbox: Bounding box (x, y, w, h)
        depth_map: Depth map (H, W) in meters
        method: 'center', 'median', or 'bottom'

    Returns:
        Depth in meters, or None if invalid
    """
    if depth_map is None:
        return None

    x, y, w, h = int(bbox.x), int(bbox.y), int(bbox.w), int(bbox.h)

    if method == 'center':
        cx, cy = int(x + w/2), int(y + h/2)
        cy = max(0, min(cy, depth_map.shape[0] - 1))
        cx = max(0, min(cx, depth_map.shape[1] - 1))
        depth = depth_map[cy, cx]

    elif method == 'median':
        y1, y2 = max(0, y), min(depth_map.shape[0], y + h)
        x1, x2 = max(0, x), min(depth_map.shape[1], x + w)

        roi = depth_map[y1:y2, x1:x2]
        valid_depths = roi[(roi > 0.1) & (roi < 50.0)]

        if len(valid_depths) == 0:
            return None

        depth = np.median(valid_depths)

    elif method == 'bottom':
        cx, cy = int(x + w/2), int(y + h)
        cy = max(0, min(cy, depth_map.shape[0] - 1))
        cx = max(0, min(cx, depth_map.shape[1] - 1))

        radius = 5
        y1, y2 = max(0, cy - radius), min(depth_map.shape[0], cy + radius)
        x1, x2 = max(0, cx - radius), min(depth_map.shape[1], cx + radius)

        roi = depth_map[y1:y2, x1:x2]
        valid_depths = roi[(roi > 0.1) & (roi < 50.0)]

        if len(valid_depths) == 0:
            return None

        depth = np.median(valid_depths)

    else:
        raise ValueError(f"Unknown method: {method}")

    if depth < 0.1 or depth > 50.0:
        return None

    return float(depth)


def main():
    parser = argparse.ArgumentParser(description='Wide-angle camera perception pipeline')
    parser.add_argument('--mode', type=str, default='single',
                        choices=['single', 'dual'],
                        help='Single or dual wide-angle camera')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera device ID (or left camera if dual)')
    parser.add_argument('--camera-right', type=int, default=1,
                        help='Right camera device ID (dual mode only)')
    parser.add_argument('--fov', type=float, default=200.0,
                        help='Field of view in degrees (e.g., 200 for Skydio-like)')
    parser.add_argument('--resolution', type=str, default='1280x720',
                        help='Camera resolution (WIDTHxHEIGHT)')
    parser.add_argument('--dac-model', type=str, default='checkpoints/dac_swinl_indoor.pt',
                        help='DAC model weights path')
    parser.add_argument('--dac-config', type=str, default='checkpoints/dac_swinl_indoor.json',
                        help='DAC config path')
    parser.add_argument('--no-dac', action='store_true',
                        help='Disable DAC depth estimation (image only)')
    parser.add_argument('--model', type=str, default='s',
                        choices=['n', 's', 'm', 'l', 'x'],
                        help='YOLO model size')
    parser.add_argument('--classes', type=int, nargs='+', default=None,
                        help='Filter detection classes')
    parser.add_argument('--conf-threshold', type=float, default=0.3,
                        help='Detection confidence threshold')
    parser.add_argument('--depth-method', type=str, default='median',
                        choices=['center', 'median', 'bottom'],
                        help='Depth extraction method')
    parser.add_argument('--save-video', type=str, default=None,
                        help='Save annotated video to file')
    parser.add_argument('--no-viz-3d', action='store_true',
                        help='Disable 3D visualization')
    parser.add_argument('--baseline', type=float, default=0.1,
                        help='Stereo baseline in meters (dual mode only)')

    args = parser.parse_args()

    # Parse resolution
    width, height = map(int, args.resolution.split('x'))

    print("="*70)
    print("WIDE-ANGLE PERCEPTION PIPELINE (Skydio-style)")
    print("="*70)
    print(f"Mode: {args.mode}")
    print(f"FOV: {args.fov}°")
    print(f"Resolution: {width}x{height}")
    print(f"Model: yolov8{args.model} on cpu")
    print(f"Detection threshold: {args.conf_threshold}")
    print(f"Depth method: {args.depth_method}")
    if args.classes:
        print(f"Tracking classes: {args.classes}")
    print()
    print("Pipeline: Wide-Angle Camera → DAC Depth → YOLO → ByteTrack → Scene Graph")
    print()
    print("Advantages of wide-angle + DAC:")
    print("  ✓ Ultra-wide field of view (200°+ vs 60-90° normal)")
    print("  ✓ Zero-shot metric depth (no camera-specific training)")
    print("  ✓ Works with any fisheye camera model")
    print("  ✓ Dense depth estimation")
    print("  ✓ Better coverage for obstacle avoidance")
    print()
    print("Controls:")
    print("  'q' - Quit")
    print("  'p' - Pause/Resume")
    print("  's' - Save 3D view screenshot")
    print("  'd' - Toggle depth visualization")
    print("="*70)

    # Initialize sensor
    print("[1/5] Initializing camera...")
    if args.mode == 'single':
        sensor = WideAngleCamera(
            camera_id=args.camera,
            resolution=(width, height),
            fov=args.fov,
            dac_model_path=args.dac_model if not args.no_dac else None,
            dac_config_path=args.dac_config if not args.no_dac else None,
            use_dac=not args.no_dac
        )
    else:  # dual
        sensor = DualWideAngleCamera(
            left_camera_id=args.camera,
            right_camera_id=args.camera_right,
            resolution=(width, height),
            fov=args.fov,
            dac_model_path=args.dac_model if not args.no_dac else None,
            dac_config_path=args.dac_config if not args.no_dac else None,
            baseline=args.baseline,
            use_dac=not args.no_dac
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
        video_writer = cv2.VideoWriter(args.save_video, fourcc, 30, (width, height))
        print(f"Saving video: {args.save_video}")

    print()
    print("="*70)
    print("STARTING PIPELINE")
    print("="*70)
    print()

    if args.no_dac:
        print("WARNING: DAC disabled - running without depth estimation")
        print()

    # Main loop
    frame_count = 0
    paused = False
    show_depth = False

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

                # Track objects
                tracks = tracker.update(detections)

                # Update scene graph with depth
                for track in tracks:
                    depth = get_depth_at_bbox(track.bbox, frame.depth, method=args.depth_method)

                    if depth is not None:
                        cx, cy = track.bbox.center
                        position_3d = project_2d_to_3d(
                            cx, cy, depth, frame.camera_params
                        )

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
                    node = scene_graph.get_node(track.track_id)
                    if node is None:
                        continue

                    x, y, w, h = int(track.bbox.x), int(track.bbox.y), \
                                int(track.bbox.w), int(track.bbox.h)
                    cv2.rectangle(viz_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    depth = get_depth_at_bbox(track.bbox, frame.depth, method=args.depth_method)
                    depth_str = f"{depth:.2f}m" if depth else "N/A"
                    label = f"{track.track_id}: {track.class_name} ({depth_str})"

                    if node.velocity_3d is not None:
                        vel_mag = np.linalg.norm(node.velocity_3d)
                        label += f" | {vel_mag:.1f}m/s"

                    cv2.putText(viz_frame, label, (x, y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Show depth
                if show_depth and frame.depth is not None:
                    depth_vis = frame.depth.copy()
                    depth_vis = np.clip(depth_vis, 0, 30)
                    depth_vis = (depth_vis / 30.0 * 255).astype(np.uint8)
                    depth_colormap = cv2.applyColorMap(depth_vis, cv2.COLORMAP_MAGMA)
                    viz_frame = np.hstack([viz_frame, depth_colormap])

                # Info overlay
                dac_status = "DAC ON" if not args.no_dac else "DAC OFF"
                info = f"Frame: {frame_count} | Tracks: {len(tracks)} | {dac_status} | FOV: {args.fov}°"
                cv2.putText(viz_frame, info, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Show
                cv2.imshow("Wide-Angle Pipeline", viz_frame)

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
                print(f"Depth view: {'ON' if show_depth else 'OFF'}")
            elif key == ord('s') and viewer_3d is not None:
                viewer_3d.save_screenshot()

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
            print(f"Saved output video: {args.save_video}")
        if viewer_3d is not None:
            viewer_3d.close()

        print(f"\nProcessed {frame_count} frames")
        print("Done!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
