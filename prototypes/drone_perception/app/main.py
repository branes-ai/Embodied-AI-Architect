#!/usr/bin/env python3
"""Drone perception pipeline application.

Configurable real-time perception for drone deployment.
Runs headless by default; add --display for visualization.

Usage:
    python -m app.main --sensor mono --model yolov8n
    python -m app.main --sensor stereo --model yolov8s --reasoning --display
    python -m app.main --sensor mono --video traffic.mp4 --output json
"""

import sys
from pathlib import Path
import argparse

import signal
import json
import time
import logging

import cv2
import numpy as np

# Add drone_perception root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.sensors import MonocularCamera, StereoCamera
from lib.sensors.wide_angle import WideAngleCamera
from lib.sensors.lidar import LiDARCameraSensor
from lib.detection import YOLODetector
from lib.tracking import ByteTracker
from lib.scene_graph import SceneGraphManager
from lib.common import project_2d_to_3d

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("drone_perception")

# Graceful shutdown flag
_shutdown = False


def _handle_signal(signum, frame):
    global _shutdown
    log.info("Shutdown signal received")
    _shutdown = True


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


def build_sensor(args):
    """Build sensor from CLI args."""
    if args.sensor == "mono":
        source = int(args.video) if args.video.isdigit() else args.video
        return MonocularCamera(source=source)
    elif args.sensor == "stereo":
        return StereoCamera(backend=args.stereo_backend, width=640, height=480, fps=30)
    elif args.sensor == "wide":
        return WideAngleCamera(camera_id=int(args.video), fov=args.fov)
    elif args.sensor == "lidar":
        return LiDARCameraSensor(
            camera_id=int(args.video),
            lidar_source=args.lidar_source,
        )
    else:
        raise ValueError(f"Unknown sensor type: {args.sensor}")


def build_pipeline(args):
    """Build detection, tracking, and scene graph components."""
    detector = YOLODetector(
        model_size=args.model,
        conf_threshold=args.conf,
        device=args.device,
        classes=args.classes,
    )
    tracker = ByteTracker(
        high_thresh=args.conf,
        low_thresh=0.1,
        match_thresh=0.7,
    )
    scene = SceneGraphManager(ttl_seconds=3.0)
    return detector, tracker, scene


def build_reasoning(args):
    """Build optional reasoning modules."""
    from lib.reasoning import (
        TrajectoryPredictor,
        CollisionDetector,
        SpatialAnalyzer,
        BehaviorClassifier,
    )

    return {
        "predictor": TrajectoryPredictor(
            prediction_horizon=args.prediction_horizon,
            method="constant_velocity",
        ),
        "collision": CollisionDetector(),
        "spatial": SpatialAnalyzer(),
        "behavior": BehaviorClassifier(),
    }


def output_frame_result(objects, frame_id, output_mode):
    """Output frame results in the requested format."""
    if output_mode == "json":
        result = {
            "frame": frame_id,
            "timestamp": time.time(),
            "objects": [
                {
                    "id": obj.track_id,
                    "class": obj.class_name,
                    "position": obj.position.tolist(),
                    "velocity": obj.velocity.tolist(),
                    "confidence": obj.confidence,
                }
                for obj in objects
            ],
        }
        print(json.dumps(result), flush=True)
    elif output_mode == "log":
        log.info(
            "Frame %04d | Objects: %d | Classes: %s",
            frame_id,
            len(objects),
            ", ".join(sorted({o.class_name for o in objects})),
        )


def run(args):
    """Run the perception pipeline."""
    log.info("Building sensor: %s", args.sensor)
    sensor = build_sensor(args)

    log.info("Building pipeline: yolov8%s on %s", args.model, args.device)
    detector, tracker, scene = build_pipeline(args)

    reasoning = None
    if args.reasoning:
        log.info("Building reasoning modules")
        reasoning = build_reasoning(args)

    viewer = None
    if args.display:
        from lib.visualization import LiveViewer3D

        viewer = LiveViewer3D()

    depth_mode = "depth_map" if args.sensor in ("stereo", "lidar") else "heuristic"

    log.info("Pipeline ready — starting main loop")
    frame_count = 0
    frame_times = []

    try:
        while not _shutdown:
            t0 = time.time()

            frame = sensor.get_frame()
            if frame is None:
                log.info("End of stream")
                break

            detections = detector.detect(frame.image, frame_id=frame.frame_id)
            tracks = tracker.update(detections)
            scene.update(tracks, frame, depth_estimation_mode=depth_mode)
            objects = scene.get_objects()

            if reasoning:
                nodes = objects
                predictions = reasoning["predictor"].predict_all(nodes)
                reasoning["collision"].check_all_collisions(predictions)
                reasoning["behavior"].classify_all(nodes)

            if viewer:
                viewer.render(objects)

            output_frame_result(objects, frame.frame_id, args.output)

            dt = time.time() - t0
            frame_times.append(dt)
            if len(frame_times) > 100:
                frame_times.pop(0)

            frame_count += 1
            if frame_count % 100 == 0:
                avg_fps = 1.0 / (sum(frame_times) / len(frame_times)) if frame_times else 0
                log.info(
                    "Frame %d | FPS: %.1f | Tracks: %d | 3D Objects: %d",
                    frame_count,
                    avg_fps,
                    len(tracks),
                    len(objects),
                )

    finally:
        log.info("Shutting down...")
        sensor.release()
        if viewer:
            viewer.close()
        cv2.destroyAllWindows()

        if frame_times:
            avg_fps = 1.0 / (sum(frame_times) / len(frame_times))
            log.info("Processed %d frames | Average FPS: %.1f", frame_count, avg_fps)


def main():
    parser = argparse.ArgumentParser(
        description="Drone perception pipeline application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  %(prog)s --sensor mono --video 0                     # webcam
  %(prog)s --sensor mono --video traffic.mp4 --output json  # file, JSON output
  %(prog)s --sensor stereo --reasoning --display        # stereo + reasoning + viz
""",
    )

    # Sensor
    parser.add_argument(
        "--sensor",
        choices=["mono", "stereo", "wide", "lidar"],
        default="mono",
        help="Sensor type (default: mono)",
    )
    parser.add_argument("--video", default="0", help="Video source (file path or device id)")
    parser.add_argument(
        "--stereo-backend",
        choices=["realsense", "oakd"],
        default="realsense",
        help="Stereo camera backend",
    )
    parser.add_argument("--fov", type=float, default=200.0, help="Wide-angle FOV in degrees")
    parser.add_argument(
        "--lidar-source", default="velodyne", help="LiDAR source type for lidar sensor"
    )

    # Detection
    parser.add_argument(
        "--model",
        choices=["n", "s", "m", "l", "x"],
        default="n",
        help="YOLO model size (default: n)",
    )
    parser.add_argument("--conf", type=float, default=0.3, help="Detection confidence threshold")
    parser.add_argument(
        "--device", choices=["cpu", "cuda"], default="cpu", help="Inference device"
    )
    parser.add_argument(
        "--classes", type=int, nargs="+", default=None, help="Class IDs to detect"
    )

    # Reasoning
    parser.add_argument("--reasoning", action="store_true", help="Enable reasoning modules")
    parser.add_argument(
        "--prediction-horizon",
        type=float,
        default=3.0,
        help="Trajectory prediction horizon (seconds)",
    )

    # Output
    parser.add_argument("--display", action="store_true", help="Enable visualization window")
    parser.add_argument(
        "--output",
        choices=["json", "log", "none"],
        default="log",
        help="Output mode (default: log)",
    )

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
