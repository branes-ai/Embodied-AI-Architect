"""State schema for LangGraph operator pipelines."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from typing_extensions import TypedDict
import uuid


class PipelineStage(str, Enum):
    """Pipeline execution stages for routing control."""

    # Perception stages
    PREPROCESS = "preprocess"
    DETECT = "detect"
    TRACK = "track"
    SCENE_GRAPH = "scene_graph"

    # State estimation stages
    EKF = "ekf"
    TRAJECTORY = "trajectory"
    COLLISION = "collision"

    # Control stages
    PLANNING = "planning"
    PATH_FOLLOW = "path_follow"
    EMERGENCY_STOP = "emergency_stop"

    # Join/sync stages
    JOIN_STATE = "join_state"

    # Terminal stages
    COMPLETE = "complete"
    ERROR = "error"

    # LLM architect (optional)
    ARCHITECT = "architect"


class EmbodiedPipelineState(TypedDict, total=False):
    """
    State flowing through LangGraph operator pipelines.

    This TypedDict defines all fields that can be read/written by graph nodes.
    Using total=False allows optional fields while maintaining type safety.

    The `next_stage` field is the routing control signal - each node sets this
    to determine which node executes next via conditional edges.
    """

    # === Input Data ===
    frame: dict  # {"image": ndarray, "timestamp": float, "frame_id": int}
    imu_data: dict  # {"accel": [ax, ay, az], "gyro": [wx, wy, wz], "dt": float}
    goal: dict  # {"position": [x, y], "tolerance": float}

    # === Intermediate Results (accumulated by nodes) ===
    preprocessed: dict  # Output of preprocess node
    detections: dict  # Output of detect node (YOLO)
    tracks: dict  # Output of track node (ByteTrack)
    scene_objects: dict  # Output of scene_graph node (3D objects)
    ego_pose: dict  # Output of ekf node (pose, velocity)
    trajectories: dict  # Output of trajectory node (predicted paths)
    collision_risks: dict  # Output of collision node (risk assessment)
    planned_path: dict  # Output of planning node (waypoints)
    control_output: dict  # Output of control nodes (velocity commands)

    # === Routing Control ===
    next_stage: str  # PipelineStage value - determines next node
    enabled_stages: list[str]  # Which stages to run (skip others)

    # === Configuration ===
    operator_configs: dict[str, dict[str, Any]]  # Per-operator config overrides
    execution_target: str  # "cpu", "gpu", "npu"
    latency_budget_ms: float  # Target latency for architect optimization

    # === Tracking & Metrics ===
    iteration: int  # Loop iteration counter
    frame_id: int  # Current frame number
    timing: dict[str, float]  # Per-stage timing in ms
    errors: list[str]  # Accumulated error messages

    # === Parallel Execution State ===
    # Used by join nodes to wait for parallel branches
    parallel_results: dict[str, dict]  # {branch_name: result}
    pending_branches: list[str]  # Branches not yet completed

    # === Metadata ===
    pipeline_id: str  # Unique ID for this pipeline run
    created_at: str  # ISO timestamp


def create_initial_state(
    frame: Optional[dict] = None,
    imu_data: Optional[dict] = None,
    goal: Optional[dict] = None,
    frame_id: int = 0,
    execution_target: str = "cpu",
    enabled_stages: Optional[list[str]] = None,
    operator_configs: Optional[dict[str, dict[str, Any]]] = None,
    latency_budget_ms: float = 100.0,
    pipeline_id: Optional[str] = None,
) -> EmbodiedPipelineState:
    """
    Create initial state for pipeline execution.

    Args:
        frame: Input frame data with image and timestamp
        imu_data: IMU sensor data for state estimation
        goal: Goal position for path planning
        frame_id: Current frame number (for streaming)
        execution_target: Hardware target ("cpu", "gpu", "npu")
        enabled_stages: List of stages to run (None = all)
        operator_configs: Per-operator configuration overrides
        latency_budget_ms: Target latency for LLM architect optimization
        pipeline_id: Unique identifier (auto-generated if None)

    Returns:
        Initial EmbodiedPipelineState ready for graph execution
    """
    # Default to all stages enabled
    if enabled_stages is None:
        enabled_stages = [stage.value for stage in PipelineStage]

    return EmbodiedPipelineState(
        # Input data
        frame=frame or {},
        imu_data=imu_data or {},
        goal=goal or {},
        # Intermediate results (empty initially)
        preprocessed={},
        detections={},
        tracks={},
        scene_objects={},
        ego_pose={},
        trajectories={},
        collision_risks={},
        planned_path={},
        control_output={},
        # Routing - start with preprocess
        next_stage=PipelineStage.PREPROCESS.value,
        enabled_stages=enabled_stages,
        # Configuration
        operator_configs=operator_configs or {},
        execution_target=execution_target,
        latency_budget_ms=latency_budget_ms,
        # Tracking
        iteration=0,
        frame_id=frame_id,
        timing={},
        errors=[],
        # Parallel execution
        parallel_results={},
        pending_branches=[],
        # Metadata
        pipeline_id=pipeline_id or f"pipeline_{uuid.uuid4().hex[:8]}",
        created_at=datetime.now().isoformat(),
    )


def get_total_latency_ms(state: EmbodiedPipelineState) -> float:
    """Calculate total pipeline latency from timing dict."""
    return sum(state.get("timing", {}).values())


def is_over_budget(state: EmbodiedPipelineState) -> bool:
    """Check if pipeline is over latency budget."""
    budget = state.get("latency_budget_ms", float("inf"))
    return get_total_latency_ms(state) > budget


def format_timing_summary(state: EmbodiedPipelineState) -> str:
    """Format timing information for display."""
    timing = state.get("timing", {})
    if not timing:
        return "No timing data"

    lines = []
    for stage, ms in sorted(timing.items(), key=lambda x: x[1], reverse=True):
        lines.append(f"  {stage}: {ms:.2f}ms")

    total = sum(timing.values())
    budget = state.get("latency_budget_ms", float("inf"))
    status = "OK" if total <= budget else "OVER BUDGET"

    lines.append("  --------")
    lines.append(f"  Total: {total:.2f}ms ({status})")

    return "\n".join(lines)
