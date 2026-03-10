#!/usr/bin/env python3
"""
Run perception pipeline on test video suite and save results.

Usage:
    python scripts/run_test_suite.py --suite quick
    python scripts/run_test_suite.py --video test_data/videos/traffic/traffic_001.mp4
    python scripts/run_test_suite.py --suite comprehensive --device cuda
"""

import argparse
import sys
import json
import time
from pathlib import Path
from datetime import datetime
import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.sensors import MonocularCamera
from lib.detection import YOLODetector
from lib.tracking import ByteTracker
from lib.scene_graph import SceneGraphManager


class TestRunner:
    """Run pipeline on test videos and collect metrics."""

    def __init__(
        self,
        model_size: str = 'n',
        device: str = 'cpu',
        conf_threshold: float = 0.3
    ):
        """Initialize test runner."""
        self.model_size = model_size
        self.device = device
        self.conf_threshold = conf_threshold

        # Initialize pipeline components
        print("[TestRunner] Initializing pipeline...")
        self.detector = YOLODetector(
            model_size=model_size,
            conf_threshold=conf_threshold,
            device=device
        )
        self.tracker = ByteTracker(
            high_thresh=conf_threshold,
            low_thresh=0.1,
            match_thresh=0.7
        )
        self.scene_graph = SceneGraphManager(ttl_seconds=3.0)

    def run_on_video(self, video_path: Path) -> dict:
        """
        Run pipeline on single video and collect metrics.

        Returns:
            Dict with results and metrics
        """
        print(f"\n{'='*70}")
        print(f"Processing: {video_path.name}")
        print(f"{'='*70}")

        # Open video
        camera = MonocularCamera(source=str(video_path))

        # Reset pipeline state
        self.tracker.reset()
        self.scene_graph.reset()

        # Metrics
        frame_count = 0
        detection_counts = []
        track_counts = []
        object_counts = []
        frame_times = []

        start_time = time.time()

        # Process video
        while True:
            frame_start = time.time()

            # Get frame
            frame = camera.get_frame()
            if frame is None:
                break

            # Run pipeline
            detections = self.detector.detect(frame.image, frame_id=frame.frame_id)
            tracks = self.tracker.update(detections)
            self.scene_graph.update(tracks, frame, depth_estimation_mode='heuristic')
            objects_3d = self.scene_graph.get_objects()

            # Collect metrics
            frame_time = time.time() - frame_start
            frame_times.append(frame_time)
            detection_counts.append(len(detections))
            track_counts.append(len(tracks))
            object_counts.append(len(objects_3d))

            frame_count += 1

            # Progress
            if frame_count % 30 == 0:
                avg_fps = 1.0 / (sum(frame_times[-30:]) / 30)
                print(f"  Frame {frame_count:04d} | "
                      f"FPS: {avg_fps:.1f} | "
                      f"Det: {len(detections)} | "
                      f"Tracks: {len(tracks)}")

        total_time = time.time() - start_time
        camera.release()

        # Compute statistics
        avg_fps = frame_count / total_time if total_time > 0 else 0
        avg_detections = sum(detection_counts) / len(detection_counts) if detection_counts else 0
        avg_tracks = sum(track_counts) / len(track_counts) if track_counts else 0
        avg_objects = sum(object_counts) / len(object_counts) if object_counts else 0

        results = {
            'video': video_path.name,
            'video_path': str(video_path),
            'timestamp': datetime.now().isoformat(),
            'config': {
                'model_size': self.model_size,
                'device': self.device,
                'conf_threshold': self.conf_threshold,
            },
            'metrics': {
                'frame_count': frame_count,
                'total_time_sec': round(total_time, 2),
                'avg_fps': round(avg_fps, 2),
                'avg_detections_per_frame': round(avg_detections, 2),
                'avg_tracks_per_frame': round(avg_tracks, 2),
                'avg_3d_objects_per_frame': round(avg_objects, 2),
                'max_detections': max(detection_counts) if detection_counts else 0,
                'max_tracks': max(track_counts) if track_counts else 0,
            },
            'per_frame': {
                'detection_counts': detection_counts,
                'track_counts': track_counts,
                'object_counts': object_counts,
                'frame_times_sec': [round(t, 4) for t in frame_times],
            }
        }

        # Print summary
        print(f"\n{'='*70}")
        print("RESULTS")
        print(f"{'='*70}")
        print(f"Frames processed: {frame_count}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average FPS: {avg_fps:.1f}")
        print(f"Avg detections/frame: {avg_detections:.1f}")
        print(f"Avg tracks/frame: {avg_tracks:.1f}")
        print(f"Avg 3D objects/frame: {avg_objects:.1f}")

        return results


def main():
    parser = argparse.ArgumentParser(description='Run test suite on drone perception pipeline')
    parser.add_argument('--suite', type=str, help='Test suite to run (quick, comprehensive, etc.)')
    parser.add_argument('--video', type=str, help='Run on specific video file')
    parser.add_argument('--model', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'],
                        help='YOLO model size')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                        help='Device for inference')
    parser.add_argument('--conf', type=float, default=0.3,
                        help='Confidence threshold')
    parser.add_argument('--output', type=str, default='test_data/results',
                        help='Output directory for results')
    parser.add_argument('--catalog', type=str, default='test_data/video_catalog.yaml',
                        help='Path to video catalog')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("Drone Perception Pipeline - Test Suite Runner")
    print("="*70)
    print(f"Model: yolov8{args.model}")
    print(f"Device: {args.device}")
    print(f"Confidence: {args.conf}")
    print()

    # Initialize runner
    runner = TestRunner(
        model_size=args.model,
        device=args.device,
        conf_threshold=args.conf
    )

    # Determine videos to process
    videos_to_process = []

    if args.video:
        # Single video
        video_path = Path(args.video)
        if not video_path.exists():
            print(f"Error: Video not found: {video_path}")
            return 1
        videos_to_process = [video_path]

    elif args.suite:
        # Load catalog and get suite
        catalog_path = Path(args.catalog)
        if not catalog_path.exists():
            print(f"Error: Catalog not found: {catalog_path}")
            return 1

        with open(catalog_path, 'r') as f:
            catalog = yaml.safe_load(f)

        suite = catalog['test_suites'].get(args.suite)
        if not suite:
            print(f"Error: Suite not found: {args.suite}")
            print(f"Available: {', '.join(catalog['test_suites'].keys())}")
            return 1

        # Find video files
        for video_id in suite['videos']:
            # Try to find in videos directory
            video_meta = next((v for v in catalog['videos'] if v['id'] == video_id), None)
            if not video_meta:
                print(f"Warning: Video metadata not found: {video_id}")
                continue

            category = video_meta['category']
            video_path = Path(f"test_data/videos/{category}/{video_id}.mp4")

            if not video_path.exists():
                print(f"Warning: Video file not found: {video_path}")
                print(f"  Run: python scripts/download_test_videos.py --id {video_id}")
                continue

            videos_to_process.append(video_path)

    else:
        print("Error: Specify --video or --suite")
        return 1

    if not videos_to_process:
        print("Error: No videos to process")
        return 1

    print(f"Processing {len(videos_to_process)} video(s)...\n")

    # Run on each video
    all_results = []
    for video_path in videos_to_process:
        results = runner.run_on_video(video_path)
        all_results.append(results)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"results_{args.suite or 'single'}_{timestamp}.json"

    with open(results_file, 'w') as f:
        json.dump({
            'suite': args.suite or 'single',
            'timestamp': datetime.now().isoformat(),
            'config': {
                'model_size': args.model,
                'device': args.device,
                'conf_threshold': args.conf,
            },
            'results': all_results
        }, f, indent=2)

    print(f"\n{'='*70}")
    print("TEST SUITE COMPLETE")
    print(f"{'='*70}")
    print(f"Results saved to: {results_file}")
    print()

    # Print summary
    total_frames = sum(r['metrics']['frame_count'] for r in all_results)
    avg_fps = sum(r['metrics']['avg_fps'] for r in all_results) / len(all_results)

    print("Summary:")
    print(f"  Videos processed: {len(all_results)}")
    print(f"  Total frames: {total_frames}")
    print(f"  Average FPS: {avg_fps:.1f}")
    print()

    return 0


if __name__ == '__main__':
    sys.exit(main())
