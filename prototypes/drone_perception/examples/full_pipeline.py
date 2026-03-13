"""
Complete drone perception pipeline demo.

Demonstrates:
- Object detection (YOLOv8)
- Multi-object tracking (ByteTrack)
- 3D scene graph with velocity/acceleration
- Real-time 3D visualization
"""

import sys
from pathlib import Path
import argparse
import time
import cv2

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.sensors import MonocularCamera, StereoCamera, StereoRecordedCamera
from lib.detection import YOLODetector
from lib.tracking import ByteTracker
from lib.scene_graph import SceneGraphManager
from lib.visualization import LiveViewer3D


def main():
    parser = argparse.ArgumentParser(description='Full perception pipeline demo')
    parser.add_argument('--video', type=str, default='0',
                        help='Video file path or camera device (0 for webcam)')
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
    parser.add_argument('--stereo', action='store_true',
                        help='Use stereo camera (RealSense/OAK-D) instead of monocular')
    parser.add_argument('--stereo-backend', type=str, default='realsense',
                        choices=['realsense', 'oakd'],
                        help='Stereo camera backend (when --stereo is enabled)')

    args = parser.parse_args()

    # Parse video source
    try:
        video_source = int(args.video)
    except ValueError:
        video_source = args.video

    print("\n" + "="*70)
    print("FULL DRONE PERCEPTION PIPELINE")
    print("="*70)

    # Determine sensor mode
    sensor_mode = "Stereo" if args.stereo else "Monocular"
    depth_mode = "stereo" if args.stereo else "heuristic"

    if not args.stereo:
        print(f"Video: {video_source}")
    else:
        print(f"Stereo camera: {args.stereo_backend}")

    print(f"Sensor mode: {sensor_mode}")
    print(f"Model: yolov8{args.model} on {args.device}")
    print(f"Detection threshold: {args.conf}")
    if args.classes:
        print(f"Tracking classes: {args.classes}")
    print(f"\nPipeline: Camera ({sensor_mode}) → YOLO → ByteTrack → Scene Graph → 3D Viz")
    print("\nControls:")
    print("  'q' - Quit")
    print("  'p' - Pause/Resume")
    print("  's' - Save 3D view screenshot")
    print("="*70 + "\n")

    # Initialize pipeline components
    print("[1/5] Initializing camera...")
    if args.stereo:
        # Use stereo camera
        camera = StereoCamera(
            backend=args.stereo_backend,
            width=640,
            height=480,
            fps=30
        )
    else:
        # Use monocular camera
        camera = MonocularCamera(source=video_source)

    print("[2/5] Loading detection model...")
    detector = YOLODetector(
        model_size=args.model,
        conf_threshold=args.conf,
        device=args.device,
        classes=args.classes
    )

    print("[3/5] Initializing tracker...")
    tracker = ByteTracker(
        high_thresh=0.5,
        low_thresh=0.2,
        match_thresh=0.3
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

    # Setup video writer
    video_writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            args.save_video,
            fourcc,
            camera.target_fps,
            (camera.width, camera.height)
        )

    print("\n[READY] Starting pipeline...\n")

    # Performance tracking
    frame_times = []
    paused = False

    try:
        while True:
            loop_start = time.time()

            if not paused:
                # 1. Capture frame
                frame = camera.get_frame()
                if frame is None:
                    print("\n[INFO] End of video")
                    break

                # 2. Detect objects
                detections = detector.detect(frame.image, frame_id=frame.frame_id)

                # 3. Track objects
                tracks = tracker.update(detections)

                # 4. Update 3D scene graph
                scene_graph.update(tracks, frame, depth_estimation_mode=depth_mode)

                # 5. Get tracked objects
                objects_3d = scene_graph.get_objects()

                # 6. Visualize 2D (detection + tracking)
                img_display = detector.draw_detections(frame.image, detections)

                # Draw track IDs
                for track in tracks:
                    cx, cy = track.bbox.center
                    cv2.circle(img_display, (int(cx), int(cy)), 5, (0, 255, 0), -1)
                    cv2.putText(
                        img_display,
                        f"ID:{track.id}",
                        (int(cx) + 10, int(cy) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2
                    )

                # Add stats
                stats = scene_graph.get_stats()
                loop_time = time.time() - loop_start
                fps = 1.0 / loop_time if loop_time > 0 else 0

                stats_text = [
                    f"Frame: {frame.frame_id}",
                    f"FPS: {fps:.1f}",
                    f"Detections: {len(detections)}",
                    f"Tracks: {len(tracks)}",
                    f"3D Objects: {stats['total_objects']}",
                    f"Avg Speed: {stats['avg_velocity']:.2f} m/s",
                ]

                y_offset = 25
                for i, text in enumerate(stats_text):
                    cv2.putText(
                        img_display,
                        text,
                        (10, y_offset + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )

                # 7. Visualize 3D scene
                if viewer_3d:
                    frame_info = f"Frame {frame.frame_id} | FPS: {fps:.1f}"
                    viewer_3d.render(objects_3d, frame_info)

                # Performance
                frame_times.append(loop_time)
                if len(frame_times) > 100:
                    frame_times.pop(0)

                # Print summary
                if frame.frame_id % 30 == 0:
                    avg_fps = 1.0 / (sum(frame_times) / len(frame_times))
                    print(f"Frame {frame.frame_id:04d} | "
                          f"FPS: {avg_fps:.1f} | "
                          f"Detections: {len(detections)} | "
                          f"Tracks: {len(tracks)} | "
                          f"3D Objects: {stats['total_objects']}")

            # Display 2D view
            img_bgr = cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR)
            cv2.imshow('Detection & Tracking', img_bgr)

            # Save video
            if video_writer and not paused:
                video_writer.write(img_bgr)

            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = not paused
                print(f"[INFO] {'Paused' if paused else 'Resumed'}")
            elif key == ord('s') and viewer_3d:
                filename = f"3d_view_frame_{frame.frame_id:04d}.png"
                viewer_3d.save_frame(filename)
                print(f"[INFO] Saved 3D view: {filename}")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")

    finally:
        # Cleanup
        print("\n[CLEANUP] Shutting down...")
        camera.release()
        if video_writer:
            video_writer.release()
        if viewer_3d:
            viewer_3d.close()
        cv2.destroyAllWindows()

        # Final stats
        if frame_times:
            avg_fps = 1.0 / (sum(frame_times) / len(frame_times))
            print(f"\n[STATS] Average FPS: {avg_fps:.1f}")
        print("[INFO] Done!")


if __name__ == '__main__':
    main()
