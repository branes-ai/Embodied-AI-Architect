"""Pipeline runner with batch and streaming execution modes."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Iterator, Optional, Union

import numpy as np

from embodied_ai_architect.graphs.state import (
    EmbodiedPipelineState,
    create_initial_state,
    format_timing_summary,
    get_total_latency_ms,
)


class VideoSource:
    """
    Video source wrapper for streaming pipeline execution.

    Supports:
    - Video files (mp4, avi, etc.)
    - Camera devices (by index)
    - Image directories

    Usage:
        source = VideoSource("video.mp4")
        async for frame in source:
            process(frame)
    """

    def __init__(
        self,
        source: Union[str, int, Path],
        loop: bool = False,
        max_frames: Optional[int] = None,
    ):
        """
        Initialize video source.

        Args:
            source: Video file path, camera index, or image directory
            loop: Whether to loop video (for testing)
            max_frames: Maximum frames to yield (None = unlimited)
        """
        self.source = source
        self.loop = loop
        self.max_frames = max_frames
        self._cap = None
        self._frame_count = 0

    def _open(self):
        """Open video capture."""
        try:
            import cv2
        except ImportError:
            raise ImportError("opencv-python required for video streaming")

        if isinstance(self.source, int):
            self._cap = cv2.VideoCapture(self.source)
        else:
            self._cap = cv2.VideoCapture(str(self.source))

        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {self.source}")

    def _close(self):
        """Close video capture."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __iter__(self) -> Iterator[np.ndarray]:
        """Iterate over frames synchronously."""
        try:
            import cv2
        except ImportError:
            raise ImportError("opencv-python required for video streaming")

        self._open()
        self._frame_count = 0

        try:
            while True:
                if self.max_frames and self._frame_count >= self.max_frames:
                    break

                ret, frame = self._cap.read()

                if not ret:
                    if self.loop:
                        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    break

                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self._frame_count += 1
                yield frame_rgb

        finally:
            self._close()

    async def __aiter__(self) -> AsyncIterator[np.ndarray]:
        """Iterate over frames asynchronously."""
        for frame in self:
            yield frame
            # Yield control to event loop
            await asyncio.sleep(0)


class PipelineRunner:
    """
    Executes LangGraph pipelines in batch or streaming mode.

    Usage:
        from embodied_ai_architect.graphs.pipelines import build_perception_graph

        graph = build_perception_graph()
        runner = PipelineRunner(graph)

        # Batch mode
        result = runner.run_batch(frame)

        # Streaming mode
        async for result in runner.run_stream(VideoSource("video.mp4")):
            print(result["timing"])
    """

    def __init__(self, graph, verbose: bool = False):
        """
        Initialize pipeline runner.

        Args:
            graph: Compiled LangGraph StateGraph
            verbose: Print progress during execution
        """
        self.graph = graph
        self.verbose = verbose

    def run_batch(
        self,
        frame: Optional[np.ndarray] = None,
        imu_data: Optional[dict] = None,
        goal: Optional[dict] = None,
        frame_id: int = 0,
        execution_target: str = "cpu",
        operator_configs: Optional[dict[str, dict]] = None,
        latency_budget_ms: float = 100.0,
    ) -> EmbodiedPipelineState:
        """
        Execute pipeline on a single frame.

        Args:
            frame: Input image as numpy array (H, W, 3) RGB
            imu_data: IMU sensor data for state estimation
            goal: Goal position for path planning
            frame_id: Frame identifier
            execution_target: Hardware target
            operator_configs: Per-operator configuration overrides
            latency_budget_ms: Target latency for optimization

        Returns:
            Final pipeline state with all results
        """
        # Create initial state
        state = create_initial_state(
            frame=(
                {"image": frame, "timestamp": 0.0, "frame_id": frame_id}
                if frame is not None
                else None
            ),
            imu_data=imu_data,
            goal=goal,
            frame_id=frame_id,
            execution_target=execution_target,
            operator_configs=operator_configs,
            latency_budget_ms=latency_budget_ms,
        )

        # Execute graph
        final_state = state
        for step in self.graph.stream(state):
            node_name = list(step.keys())[0]
            node_output = step[node_name]

            # Merge output into state
            final_state = {**final_state, **node_output}

            if self.verbose:
                timing = node_output.get("timing", {})
                if node_name in timing:
                    print(f"  [{node_name}] {timing[node_name]:.2f}ms")

        if self.verbose:
            print(format_timing_summary(final_state))

        return final_state

    async def run_stream(
        self,
        source: VideoSource,
        callback: Optional[Callable[[EmbodiedPipelineState], None]] = None,
        imu_source: Optional[AsyncIterator[dict]] = None,
        goal: Optional[dict] = None,
        execution_target: str = "cpu",
        operator_configs: Optional[dict[str, dict]] = None,
        latency_budget_ms: float = 100.0,
    ) -> AsyncIterator[EmbodiedPipelineState]:
        """
        Execute pipeline on continuous video stream.

        Args:
            source: VideoSource yielding frames
            callback: Optional callback for each result
            imu_source: Optional async iterator of IMU data
            goal: Goal position for path planning
            execution_target: Hardware target
            operator_configs: Per-operator configuration
            latency_budget_ms: Target latency

        Yields:
            Pipeline state for each processed frame
        """
        frame_id = 0

        async for frame in source:
            # Get IMU data if available
            imu_data = None
            if imu_source is not None:
                try:
                    imu_data = await imu_source.__anext__()
                except StopAsyncIteration:
                    pass

            # Run batch on this frame
            result = self.run_batch(
                frame=frame,
                imu_data=imu_data,
                goal=goal,
                frame_id=frame_id,
                execution_target=execution_target,
                operator_configs=operator_configs,
                latency_budget_ms=latency_budget_ms,
            )

            # Call callback if provided
            if callback is not None:
                callback(result)

            frame_id += 1
            yield result

    def benchmark(
        self,
        frame: np.ndarray,
        iterations: int = 100,
        warmup: int = 10,
    ) -> dict[str, Any]:
        """
        Benchmark pipeline performance.

        Args:
            frame: Input frame for benchmarking
            iterations: Number of timed iterations
            warmup: Number of warmup iterations

        Returns:
            Benchmark results with statistics
        """
        import statistics
        import time

        # Warmup
        for _ in range(warmup):
            self.run_batch(frame=frame)

        # Timed runs
        latencies = []
        all_timings = []

        for _ in range(iterations):
            start = time.perf_counter_ns()
            result = self.run_batch(frame=frame)
            elapsed_ms = (time.perf_counter_ns() - start) / 1e6

            latencies.append(elapsed_ms)
            all_timings.append(result.get("timing", {}))

        # Aggregate statistics
        return {
            "total_latency": {
                "mean_ms": statistics.mean(latencies),
                "std_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0,
                "min_ms": min(latencies),
                "max_ms": max(latencies),
                "p50_ms": statistics.median(latencies),
                "p95_ms": sorted(latencies)[int(len(latencies) * 0.95)],
                "p99_ms": sorted(latencies)[int(len(latencies) * 0.99)],
            },
            "per_stage": _aggregate_stage_timings(all_timings),
            "iterations": iterations,
            "warmup": warmup,
        }


def _aggregate_stage_timings(timings: list[dict[str, float]]) -> dict[str, dict]:
    """Aggregate per-stage timing statistics."""
    import statistics

    # Collect all values per stage
    stage_values: dict[str, list[float]] = {}
    for t in timings:
        for stage, ms in t.items():
            if stage not in stage_values:
                stage_values[stage] = []
            stage_values[stage].append(ms)

    # Compute statistics
    result = {}
    for stage, values in stage_values.items():
        result[stage] = {
            "mean_ms": statistics.mean(values),
            "std_ms": statistics.stdev(values) if len(values) > 1 else 0,
            "min_ms": min(values),
            "max_ms": max(values),
        }

    return result


def load_image(path: Union[str, Path]) -> np.ndarray:
    """Load image from file path."""
    try:
        import cv2
    except ImportError:
        raise ImportError("opencv-python required for image loading")

    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")

    # Convert BGR to RGB
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
