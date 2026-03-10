"""
Stereo camera perception pipeline demo.

Demonstrates:
- Stereo camera integration (RealSense D435 or OAK-D)
- Real metric depth from stereo matching
- Improved 3D position accuracy
- Velocity estimation with ground truth depth
"""

import sys
from pathlib import Path
import argparse
import time
import cv2
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.sensors import StereoCamera, StereoRecordedCamera
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
        # Get depth at center point
        cx, cy = int(x + w/2), int(y + h/2)
        cy = max(0, min(cy, depth_map.shape[0] - 1))
        cx = max(0, min(cx, depth_map.shape[1] - 1))
        depth = depth_map[cy, cx]

    elif method == 'median':
        # Get median depth in bbox (more robust)
        y1, y2 = max(0, y), min(depth_map.shape[0], y + h)
        x1, x2 = max(0, x), min(depth_map.shape[1], x + w)

        roi = depth_map[y1:y2, x1:x2]
        valid_depths = roi[(roi > 0.1) & (roi < 20.0)]

        if len(valid_depths) == 0:
            return None

        depth = np.median(valid_depths)

    elif method == 'bottom':
        # Get depth at bottom center (ground plane)
        cx, cy = int(x + w/2), int(y + h)
        cy = max(0, min(cy, depth_map.shape[0] - 1))
        cx = max(0, min(cx, depth_map.shape[1] - 1))

        # Median filter in small region around bottom
        radius = 5
        y1, y2 = max(0, cy - radius), min(depth_map.shape[0], cy + radius)
        x1, x2 = max(0, cx - radius), min(depth_map.shape[1], cx + radius)

        roi = depth_map[y1:y2, x1:x2]
        valid_depths = roi[(roi > 0.1) & (roi < 20.0)]

        if len(valid_depths) == 0:
            return None

        depth = np.median(valid_depths)

    else:
        raise ValueError(f"Unknown method: {method}")

    # Validate depth range
    if depth < 0.1 or depth > 20.0:
        return None

    return float(depth)


def main():
    parser = argparse.ArgumentParser(description='Stereo camera perception pipeline')
    parser.add_argument('--backend', type=str, default='realsense',
                        choices=['realsense', 'oakd', 'recorded'],
                        help='Stereo camera backend')
    parser.add_argument('--device-id', type=str, default=None,
                        help='Camera device serial number (auto-detect if not specified)')
    parser.add_argument('--rgb-video', type=str, default=None,
                        help='RGB video path (for recorded backend)')
    parser.add_argument('--depth-video', type=str, default=None,
                        help='Depth video path (for recorded backend)')
    parser.add_argument('--model', type=str, default='n',
                        choices=['n', 's', 'm', 'l', 'x'],
                        help='YOLO model size')
    parser.add_argument('--conf', type=float, default=0.3,
                        help='Detection confidence threshold')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help='Device for inference')
    parser.add_argument('--classes', type=int, nargs='+', default=None,
                        help='Class IDs to track (e.g., 0=person, 2=car)')
    parser.add_argument('--no-viz-3d', action='store_true',
                        help='Disable 3D visualization')
    parser.add_argument('--save-video', type=str, default=None,
                        help='Save 2D output video')
    parser.add_argument('--depth-method', type=str, default='median',
                        choices=['center', 'median', 'bottom'],
                        help='Method for extracting depth from bbox')

    args = parser.parse_args()

    print("\n" + "="*70)
    print("STEREO PERCEPTION PIPELINE (Phase 2)")
    print("="*70)
    print(f"Backend: {args.backend}")
    print(f"Model: yolov8{args.model} on {args.device}")
    print(f"Detection threshold: {args.conf}")
    print(f"Depth method: {args.depth_method}")
    if args.classes:
        print(f"Tracking classes: {args.classes}")
    print("\nPipeline: Stereo → YOLO → ByteTrack → Scene Graph → 3D Viz")
    print("\nAdvantages over monocular:")
    print("  ✓ Real metric depth (no scale ambiguity)")
    print("  ✓ Accurate velocity estimation")
    print("  ✓ Better 3D localization")
    print("\nControls:")
    print("  'q' - Quit")
    print("  'p' - Pause/Resume")
    print("  's' - Save 3D view screenshot")
    print("  'd' - Toggle depth visualization")
    print("="*70 + "\n")

    # Initialize camera
    print("[1/5] Initializing stereo camera...")
    if args.backend == 'recorded':
        if not args.rgb_video or not args.depth_video:
            raise ValueError("--rgb-video and --depth-video required for recorded backend")
        camera = StereoRecordedCamera(
            rgb_video_path=args.rgb_video,
            depth_video_path=args.depth_video
        )
    else:
        camera = StereoCamera(
            device_id=args.device_id,
            backend=args.backend,
            width=640,
            height=480,
            fps=30
        )

    print("[2/5] Loading detection model...")
    detector = YOLODetector(
        model_size=args.model,
        conf_threshold=args.conf,
        device=args.device,
        classes=args.classes
    )

    print("[3/5] Initializing tracker...")
    tracker = ByteTracker(
        high_thresh=args.conf,
        low_thresh=0.1,
        match_thresh=0.7
    )

    print("[4/5] Creating scene graph...")
    scene_graph = SceneGraphManager(ttl_seconds=3.0)

    print("[5/5] Setting up visualization...")
    viewer_3d = None
    if not args.no_viz_3d:
        viewer_3d = LiveViewer3D(
            xlim=(-10, 10),
            ylim=(-5, 5),
            zlim=(0, 20),
            show_velocity=True,
            show_trajectories=True
        )

    # Video writer
    video_writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(args.save_video, fourcc, 30.0, (640, 480))

    print("\n" + "="*70)
    print("STARTING PIPELINE")
    print("="*70 + "\n")

    frame_count = 0
    paused = False
    show_depth = False
    last_frame_time = time.time()

    try:
        while camera.is_opened():
            # Handle pause
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = not paused
                print(f"{'Paused' if paused else 'Resumed'}")
            elif key == ord('d'):
                show_depth = not show_depth
                print(f"Depth visualization: {'ON' if show_depth else 'OFF'}")

            if paused:
                continue

            # Get frame
            frame = camera.get_frame()
            if frame is None:
                print("End of stream")
                break

            # Detect objects
            detections = detector.detect(frame.image)

            # Track objects (ByteTracker manages frame_id internally)
            tracks = tracker.update(detections)

            # Update scene graph with stereo depth
            for track in tracks:
                # Get depth at bbox
                depth = get_depth_at_bbox(track.bbox, frame.depth, method=args.depth_method)

                if depth is not None:
                    # Project to 3D using real depth
                    cx, cy = track.bbox.center
                    position_3d = project_2d_to_3d(
                        cx, cy, depth, frame.camera_params
                    )

                    # Update scene graph
                    scene_graph.update_track(
                        track_id=track.id,
                        position=position_3d,
                        timestamp=frame.timestamp,
                        class_name=track.class_name,
                        confidence=track.confidence
                    )

            # Propagate states (Kalman filtering)
            scene_graph.propagate_states(frame.timestamp)

            # Visualize
            image_display = frame.image.copy()

            # Draw bounding boxes and IDs
            for track in tracks:
                x, y, w, h = int(track.bbox.x), int(track.bbox.y), int(track.bbox.w), int(track.bbox.h)

                # Color by track ID
                color = ((track.id * 50) % 255, (track.id * 100) % 255, (track.id * 150) % 255)

                cv2.rectangle(image_display, (x, y), (x + w, y + h), color, 2)

                # Label with ID and depth
                depth = get_depth_at_bbox(track.bbox, frame.depth, method=args.depth_method)
                depth_str = f"{depth:.2f}m" if depth else "N/A"

                label = f"ID:{track.id} {track.class_name} {depth_str}"
                cv2.putText(image_display, label, (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Display FPS
            current_time = time.time()
            fps = 1.0 / (current_time - last_frame_time) if (current_time - last_frame_time) > 0 else 0
            last_frame_time = current_time

            cv2.putText(image_display, f"FPS: {fps:.1f} | Frame: {frame_count}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Show image
            image_bgr = cv2.cvtColor(image_display, cv2.COLOR_RGB2BGR)
            cv2.imshow("Stereo Pipeline - RGB", image_bgr)

            # Show depth visualization
            if show_depth and frame.depth is not None:
                # Normalize depth for visualization
                depth_vis = frame.depth.copy()
                depth_vis = np.clip(depth_vis, 0, 10.0)
                depth_vis = (depth_vis / 10.0 * 255).astype(np.uint8)
                depth_colormap = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
                cv2.imshow("Stereo Pipeline - Depth", depth_colormap)

            # Save video
            if video_writer:
                video_writer.write(image_bgr)

            # Update 3D viewer
            if viewer_3d:
                objects = scene_graph.get_objects()
                viewer_3d.update(objects, frame.timestamp)

            frame_count += 1

            # Progress indicator
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames | Active tracks: {len(tracks)} | "
                      f"Scene objects: {len(scene_graph.get_objects())} | FPS: {fps:.1f}")

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        # Cleanup
        print("\n" + "="*70)
        print("SHUTTING DOWN")
        print("="*70)

        camera.release()
        cv2.destroyAllWindows()

        if video_writer:
            video_writer.release()
            print(f"Saved output video: {args.save_video}")

        if viewer_3d:
            viewer_3d.close()

        print(f"\nProcessed {frame_count} frames")
        print("Done!")


if __name__ == "__main__":
    main()
