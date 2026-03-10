"""Simple detection example - detect objects in video and display results."""

import sys
from pathlib import Path
import argparse
import cv2

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.sensors import MonocularCamera
from lib.detection import YOLODetector


def main():
    parser = argparse.ArgumentParser(description='Simple object detection demo')
    parser.add_argument('--video', type=str, default='0',
                        help='Video file path or camera device (0 for webcam)')
    parser.add_argument('--model', type=str, default='n',
                        choices=['n', 's', 'm', 'l', 'x'],
                        help='YOLO model size (n=nano, s=small, m=medium, l=large, x=xlarge)')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help='Device to run inference on')
    parser.add_argument('--classes', type=int, nargs='+', default=None,
                        help='Class IDs to detect (e.g., 0=person, 2=car)')
    parser.add_argument('--save', type=str, default=None,
                        help='Save output video to file')

    args = parser.parse_args()

    # Parse video source (convert '0' to int for webcam)
    try:
        video_source = int(args.video)
    except ValueError:
        video_source = args.video

    print("\n" + "="*60)
    print("Simple Detection Demo")
    print("="*60)
    print(f"Video: {video_source}")
    print(f"Model: yolov8{args.model}")
    print(f"Device: {args.device}")
    print(f"Confidence: {args.conf}")
    if args.classes:
        print(f"Classes filter: {args.classes}")
    print("\nPress 'q' to quit, 'p' to pause")
    print("="*60 + "\n")

    # Initialize sensor and detector
    camera = MonocularCamera(source=video_source)
    detector = YOLODetector(
        model_size=args.model,
        conf_threshold=args.conf,
        device=args.device,
        classes=args.classes
    )

    # Setup video writer if saving
    video_writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            args.save,
            fourcc,
            camera.target_fps,
            (camera.width, camera.height)
        )
        print(f"Saving output to: {args.save}")

    # Processing loop
    frame_count = 0
    paused = False

    try:
        while True:
            if not paused:
                # Get frame
                frame = camera.get_frame()
                if frame is None:
                    print("\n[INFO] End of video")
                    break

                # Run detection
                detections = detector.detect(frame.image, frame_id=frame.frame_id)

                # Draw results
                img_display = detector.draw_detections(frame.image, detections)

                # Add stats overlay
                stats_text = [
                    f"Frame: {frame.frame_id}",
                    f"Detections: {len(detections)}",
                    f"FPS: {camera.target_fps:.1f}",
                ]

                y_offset = 30
                for i, text in enumerate(stats_text):
                    cv2.putText(
                        img_display,
                        text,
                        (10, y_offset + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )

                # Print detections
                if detections:
                    det_summary = ", ".join([
                        f"{d.class_name}({d.confidence:.2f})" for d in detections
                    ])
                    print(f"Frame {frame.frame_id:04d}: {det_summary}")

                frame_count += 1

            # Display
            img_bgr = cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR)
            cv2.imshow('Detection', img_bgr)

            # Save if enabled
            if video_writer is not None and not paused:
                video_writer.write(img_bgr)

            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = not paused
                print(f"[INFO] {'Paused' if paused else 'Resumed'}")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")

    finally:
        # Cleanup
        camera.release()
        if video_writer is not None:
            video_writer.release()
        cv2.destroyAllWindows()

        print(f"\n[INFO] Processed {frame_count} frames")
        print("[INFO] Done!")


if __name__ == '__main__':
    main()
