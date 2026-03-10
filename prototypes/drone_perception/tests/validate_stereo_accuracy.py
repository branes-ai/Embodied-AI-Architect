#!/usr/bin/env python3
"""
Stereo camera accuracy validation script.

Tests:
1. Depth measurement accuracy at known distances
2. 3D position estimation accuracy
3. Velocity estimation validation
4. Comparison between monocular and stereo modes

Usage:
    python scripts/validate_stereo_accuracy.py --backend realsense
    python scripts/validate_stereo_accuracy.py --backend oakd --interactive
"""

import sys
from pathlib import Path
import argparse
import time
import numpy as np
import cv2
from typing import List, Tuple, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.sensors import StereoCamera
from lib.common import project_2d_to_3d


class DepthAccuracyTest:
    """Test depth measurement accuracy."""

    def __init__(self, camera: StereoCamera):
        self.camera = camera
        self.measurements = []

    def measure_point(self, x: int, y: int, n_samples: int = 10) -> dict:
        """
        Measure depth at a point with statistics.

        Args:
            x, y: Pixel coordinates
            n_samples: Number of samples to collect

        Returns:
            dict with depth statistics
        """
        depths = []

        print(f"\nMeasuring depth at pixel ({x}, {y})...")
        print(f"Collecting {n_samples} samples...")

        for i in range(n_samples):
            frame = self.camera.get_frame()
            if frame is None or frame.depth is None:
                continue

            h, w = frame.depth.shape
            if 0 <= y < h and 0 <= x < w:
                depth = frame.depth[y, x]
                if 0.1 < depth < 20.0:  # Valid range
                    depths.append(depth)

            # Small delay between samples
            time.sleep(0.1)

        if len(depths) == 0:
            return {"valid": False}

        depths = np.array(depths)

        result = {
            "valid": True,
            "mean": float(np.mean(depths)),
            "std": float(np.std(depths)),
            "min": float(np.min(depths)),
            "max": float(np.max(depths)),
            "median": float(np.median(depths)),
            "n_samples": len(depths)
        }

        print(f"  Mean depth: {result['mean']:.3f} m")
        print(f"  Std dev: {result['std']:.4f} m")
        print(f"  Range: [{result['min']:.3f}, {result['max']:.3f}] m")

        return result

    def test_known_distance(
        self,
        pixel_coords: Tuple[int, int],
        ground_truth_m: float,
        tolerance_m: float = 0.05
    ) -> dict:
        """
        Test depth measurement against known ground truth.

        Args:
            pixel_coords: (x, y) pixel coordinates
            ground_truth_m: Known distance in meters
            tolerance_m: Acceptable error tolerance

        Returns:
            dict with test results
        """
        print(f"\n{'='*60}")
        print(f"Testing known distance: {ground_truth_m:.2f} m")
        print(f"{'='*60}")

        result = self.measure_point(*pixel_coords, n_samples=20)

        if not result["valid"]:
            print("  FAILED: No valid depth measurements")
            return {"passed": False, "error": "No valid measurements"}

        error = abs(result["mean"] - ground_truth_m)
        error_pct = (error / ground_truth_m) * 100

        passed = error <= tolerance_m

        print(f"\nGround truth: {ground_truth_m:.3f} m")
        print(f"Measured:     {result['mean']:.3f} m")
        print(f"Error:        {error:.3f} m ({error_pct:.1f}%)")
        print(f"Status:       {'PASS' if passed else 'FAIL'}")

        return {
            "passed": passed,
            "ground_truth": ground_truth_m,
            "measured": result["mean"],
            "error_m": error,
            "error_pct": error_pct,
            "std": result["std"]
        }

    def depth_consistency_test(self, duration_sec: float = 5.0) -> dict:
        """
        Test depth measurement consistency over time.

        Args:
            duration_sec: Test duration in seconds

        Returns:
            dict with consistency metrics
        """
        print(f"\n{'='*60}")
        print(f"Depth Consistency Test ({duration_sec}s)")
        print(f"{'='*60}")
        print("Point the camera at a static scene...")

        depths_over_time = []
        start_time = time.time()
        frame_count = 0

        while time.time() - start_time < duration_sec:
            frame = self.camera.get_frame()
            if frame is None or frame.depth is None:
                continue

            # Sample center region
            h, w = frame.depth.shape
            center_region = frame.depth[h//2-50:h//2+50, w//2-50:w//2+50]
            valid_depths = center_region[(center_region > 0.1) & (center_region < 20.0)]

            if len(valid_depths) > 0:
                median_depth = np.median(valid_depths)
                depths_over_time.append(median_depth)
                frame_count += 1

            time.sleep(0.03)  # ~30 FPS

        if len(depths_over_time) == 0:
            return {"valid": False}

        depths_over_time = np.array(depths_over_time)

        # Calculate temporal stability metrics
        drift = depths_over_time[-1] - depths_over_time[0]
        std_dev = np.std(depths_over_time)
        max_deviation = np.max(np.abs(depths_over_time - np.mean(depths_over_time)))

        result = {
            "valid": True,
            "frames": frame_count,
            "mean_depth": float(np.mean(depths_over_time)),
            "std_dev": float(std_dev),
            "drift": float(drift),
            "max_deviation": float(max_deviation),
            "fps": frame_count / duration_sec
        }

        print(f"\nFrames captured: {frame_count}")
        print(f"Average FPS: {result['fps']:.1f}")
        print(f"Mean depth: {result['mean_depth']:.3f} m")
        print(f"Std deviation: {result['std_dev']:.4f} m")
        print(f"Drift: {result['drift']:.4f} m")
        print(f"Max deviation: {result['max_deviation']:.4f} m")

        return result


def interactive_measurement_mode(camera: StereoCamera):
    """
    Interactive mode for manual depth measurements.

    User clicks on points in the image to measure depth.
    """
    print("\n" + "="*60)
    print("INTERACTIVE MEASUREMENT MODE")
    print("="*60)
    print("Click on points in the image to measure depth")
    print("Press 'q' to quit, 's' to save screenshot")
    print("="*60 + "\n")

    measurements = []

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Measure depth at clicked point
            frame = param["frame"]
            if frame.depth is not None:
                h, w = frame.depth.shape
                if 0 <= y < h and 0 <= x < w:
                    depth = frame.depth[y, x]

                    # Get median in small window for robustness
                    radius = 5
                    y1, y2 = max(0, y-radius), min(h, y+radius)
                    x1, x2 = max(0, x-radius), min(w, x+radius)
                    window = frame.depth[y1:y2, x1:x2]
                    valid = window[(window > 0.1) & (window < 20.0)]

                    if len(valid) > 0:
                        depth = np.median(valid)
                        pos_3d = project_2d_to_3d(x, y, depth, frame.camera_params)

                        measurements.append({
                            "pixel": (x, y),
                            "depth": depth,
                            "position_3d": pos_3d
                        })

                        print(f"Measurement #{len(measurements)}:")
                        print(f"  Pixel: ({x}, {y})")
                        print(f"  Depth: {depth:.3f} m")
                        print(f"  3D Position: [{pos_3d[0]:.3f}, {pos_3d[1]:.3f}, {pos_3d[2]:.3f}] m")

    cv2.namedWindow("Interactive Measurement")

    param = {"frame": None}

    try:
        while True:
            frame = camera.get_frame()
            if frame is None:
                break

            param["frame"] = frame

            # Display RGB image
            img_display = frame.image.copy()

            # Draw previous measurements
            for i, m in enumerate(measurements):
                px, py = m["pixel"]
                cv2.circle(img_display, (px, py), 5, (0, 255, 0), -1)
                cv2.putText(img_display, f"#{i+1}: {m['depth']:.2f}m",
                           (px + 10, py - 10), cv2.FONT_HERSHEY_SIMPLEX,
                           0.5, (0, 255, 0), 2)

            # Add instructions
            cv2.putText(img_display, "Click to measure depth",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, (0, 255, 255), 2)
            cv2.putText(img_display, f"Measurements: {len(measurements)}",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, (0, 255, 255), 2)

            cv2.setMouseCallback("Interactive Measurement", mouse_callback, param)

            img_bgr = cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR)
            cv2.imshow("Interactive Measurement", img_bgr)

            # Show depth map
            if frame.depth is not None:
                depth_vis = np.clip(frame.depth, 0, 10.0)
                depth_vis = (depth_vis / 10.0 * 255).astype(np.uint8)
                depth_colormap = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
                cv2.imshow("Depth Map", depth_colormap)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite(f"measurement_screenshot_{len(measurements)}.png", img_bgr)
                print(f"Saved screenshot")

    finally:
        cv2.destroyAllWindows()

    return measurements


def main():
    parser = argparse.ArgumentParser(description='Validate stereo camera accuracy')
    parser.add_argument('--backend', type=str, default='realsense',
                        choices=['realsense', 'oakd'],
                        help='Stereo camera backend')
    parser.add_argument('--interactive', action='store_true',
                        help='Run in interactive measurement mode')
    parser.add_argument('--known-distance', type=float, default=None,
                        help='Known distance to test (meters)')
    parser.add_argument('--consistency-test', action='store_true',
                        help='Run temporal consistency test')

    args = parser.parse_args()

    print("\n" + "="*60)
    print("STEREO CAMERA VALIDATION (Phase 2)")
    print("="*60)
    print(f"Backend: {args.backend}")
    print("="*60 + "\n")

    # Initialize camera
    print("Initializing stereo camera...")
    try:
        camera = StereoCamera(
            backend=args.backend,
            width=640,
            height=480,
            fps=30
        )
    except Exception as e:
        print(f"ERROR: Failed to initialize camera: {e}")
        print("\nMake sure:")
        print("  1. Camera is connected")
        print("  2. Required libraries are installed (pyrealsense2 or depthai)")
        print("  3. You have permissions to access the camera")
        return

    print("Camera initialized successfully!\n")

    try:
        # Run tests
        if args.interactive:
            # Interactive measurement mode
            measurements = interactive_measurement_mode(camera)
            print(f"\nCollected {len(measurements)} measurements")

        else:
            tester = DepthAccuracyTest(camera)

            # Consistency test
            if args.consistency_test:
                tester.depth_consistency_test(duration_sec=10.0)

            # Known distance test
            if args.known_distance:
                # Measure center point
                h, w = 480, 640
                center = (w // 2, h // 2)
                tester.test_known_distance(center, args.known_distance, tolerance_m=0.05)

            # If no specific test, run basic measurement
            if not args.consistency_test and not args.known_distance:
                print("No specific test selected. Running basic measurement at center...")
                tester.measure_point(320, 240, n_samples=20)

    finally:
        camera.release()
        print("\nValidation complete!")


if __name__ == "__main__":
    main()
