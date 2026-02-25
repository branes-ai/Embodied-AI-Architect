"""Runnable operators for embodied AI pipelines.

This package provides composable operators that can be:
- Configured and executed on different hardware targets (CPU, GPU, NPU)
- Connected via dataflow edges to form complete architectures
- Benchmarked for latency and throughput

Usage:
    from embodied_ai_architect.operators import get_operator_class, create_operator

    # Create operator by ID
    detector = create_operator("yolo_detector_n", config={"conf_threshold": 0.25})

    # Or get class and instantiate
    cls = get_operator_class("pid_controller")
    controller = cls()
"""

from .base import Operator, OperatorResult, OperatorConfig

# Perception operators
from .perception import YOLOv8ONNX, ImagePreprocessor, ByteTrack, SceneGraphManager

# State estimation operators
from .state_estimation import EKF6DOF, TrajectoryPredictor, CollisionDetector

# Control operators
from .control import PIDController, PathFollower, PathPlannerAStar

# Operator registry mapping operator_id to class
OPERATOR_REGISTRY: dict[str, type[Operator]] = {
    # Perception
    "yolo_detector_n": YOLOv8ONNX,
    "yolo_detector_s": YOLOv8ONNX,
    "yolo_detector_m": YOLOv8ONNX,
    "yolo_detector_l": YOLOv8ONNX,
    "yolo_detector_x": YOLOv8ONNX,
    "image_preprocessor": ImagePreprocessor,
    "bytetrack": ByteTrack,
    "scene_graph_manager": SceneGraphManager,
    # State estimation
    "ekf_6dof": EKF6DOF,
    "trajectory_predictor": TrajectoryPredictor,
    "collision_detector": CollisionDetector,
    # Control
    "pid_controller": PIDController,
    "trajectory_follower": PathFollower,
    "path_planner_astar": PathPlannerAStar,
}

# Variant mapping for YOLO
YOLO_VARIANTS = {
    "yolo_detector_n": "n",
    "yolo_detector_s": "s",
    "yolo_detector_m": "m",
    "yolo_detector_l": "l",
    "yolo_detector_x": "x",
}


def get_operator_class(operator_id: str) -> type[Operator]:
    """Get operator class by ID.

    Args:
        operator_id: Operator ID from embodied-schemas catalog

    Returns:
        Operator class

    Raises:
        KeyError: If operator_id not found
    """
    if operator_id not in OPERATOR_REGISTRY:
        raise KeyError(
            f"Unknown operator: {operator_id}. " f"Available: {list(OPERATOR_REGISTRY.keys())}"
        )
    return OPERATOR_REGISTRY[operator_id]


def create_operator(
    operator_id: str,
    config: dict | None = None,
    execution_target: str = "cpu",
) -> Operator:
    """Create and setup an operator by ID.

    Args:
        operator_id: Operator ID from embodied-schemas catalog
        config: Optional configuration dictionary
        execution_target: Execution target (cpu, gpu, npu)

    Returns:
        Configured operator instance
    """
    config = config or {}

    # Handle YOLO variants
    if operator_id in YOLO_VARIANTS:
        variant = YOLO_VARIANTS[operator_id]
        operator = YOLOv8ONNX(variant=variant)
    else:
        cls = get_operator_class(operator_id)
        operator = cls()

    operator.setup(config, execution_target)
    return operator


def list_operators() -> list[str]:
    """List all available operator IDs."""
    return list(OPERATOR_REGISTRY.keys())


__all__ = [
    # Base classes
    "Operator",
    "OperatorResult",
    "OperatorConfig",
    # Perception
    "YOLOv8ONNX",
    "ImagePreprocessor",
    "ByteTrack",
    "SceneGraphManager",
    # State estimation
    "EKF6DOF",
    "TrajectoryPredictor",
    "CollisionDetector",
    # Control
    "PIDController",
    "PathFollower",
    "PathPlannerAStar",
    # Registry functions
    "get_operator_class",
    "create_operator",
    "list_operators",
    "OPERATOR_REGISTRY",
]
