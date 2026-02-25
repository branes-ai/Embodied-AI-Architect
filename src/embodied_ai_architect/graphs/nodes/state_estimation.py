"""LangGraph nodes for state estimation operators."""

from __future__ import annotations

from typing import Any, Callable, Optional

from embodied_ai_architect.graphs.state import EmbodiedPipelineState, PipelineStage
from embodied_ai_architect.graphs.utils import (
    NodeTimer,
    is_stage_enabled,
    merge_parallel_result,
)
from embodied_ai_architect.operators import create_operator, Operator


def create_ekf_node(
    config: Optional[dict[str, Any]] = None,
) -> Callable[[EmbodiedPipelineState], dict]:
    """
    Create Extended Kalman Filter node for 6DOF state estimation.

    Wraps EKF6DOF operator for LangGraph execution.
    Estimates ego pose and velocity from IMU data.

    Args:
        config: Operator configuration overrides

    Returns:
        Node function for LangGraph
    """
    operator: Optional[Operator] = None

    def ekf_node(state: EmbodiedPipelineState) -> dict:
        nonlocal operator

        # Skip if not enabled
        if not is_stage_enabled(state, PipelineStage.EKF):
            # For parallel execution, still need to signal completion
            return merge_parallel_result(state, "ekf", {"ego_pose": {}})

        # Get IMU data
        imu_data = state.get("imu_data", {})
        accel = imu_data.get("accel")
        gyro = imu_data.get("gyro")

        # If no IMU data, return default pose
        if accel is None or gyro is None:
            default_pose = {
                "pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "velocity": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "covariance": [0.0] * 12,
            }
            result = merge_parallel_result(state, "ekf", {"ego_pose": default_pose})
            result["ego_pose"] = default_pose
            result["timing"] = {**state.get("timing", {}), "ekf": 0.0}
            return result

        # Lazy initialization
        if operator is None:
            merged_config = {**(config or {})}
            merged_config.update(state.get("operator_configs", {}).get("ekf", {}))
            operator = create_operator("ekf_6dof", merged_config, "cpu")

        # Execute with timing
        with NodeTimer(state, "ekf") as timer:
            try:
                result = operator.process(
                    {
                        "accel": accel,
                        "gyro": gyro,
                        "dt": imu_data.get("dt", 0.01),
                    }
                )
            except Exception as e:
                return {
                    "next_stage": PipelineStage.ERROR.value,
                    "errors": state.get("errors", []) + [f"ekf: {e}"],
                    "timing": timer.timing,
                }

        # Store result for parallel execution join
        parallel_update = merge_parallel_result(state, "ekf", {"ego_pose": result})

        return {
            **parallel_update,
            "ego_pose": result,
            "timing": timer.timing,
        }

    return ekf_node


def create_trajectory_node(
    config: Optional[dict[str, Any]] = None,
) -> Callable[[EmbodiedPipelineState], dict]:
    """
    Create trajectory prediction node.

    Wraps TrajectoryPredictor operator for LangGraph execution.
    Predicts future trajectories of tracked objects.

    Args:
        config: Operator configuration overrides

    Returns:
        Node function for LangGraph
    """
    operator: Optional[Operator] = None

    def trajectory_node(state: EmbodiedPipelineState) -> dict:
        nonlocal operator

        # Skip if not enabled
        if not is_stage_enabled(state, PipelineStage.TRAJECTORY):
            result = merge_parallel_result(state, "trajectory", {"trajectories": {}})
            result["next_stage"] = PipelineStage.COLLISION.value
            return result

        # Get scene objects
        scene_objects = state.get("scene_objects", {})
        objects = scene_objects.get("objects", [])

        # Skip if no objects (not an error)
        if not objects:
            result = merge_parallel_result(
                state, "trajectory", {"trajectories": {"predictions": []}}
            )
            result["trajectories"] = {"predictions": []}
            result["next_stage"] = PipelineStage.COLLISION.value
            result["timing"] = {**state.get("timing", {}), "trajectory": 0.0}
            return result

        # Lazy initialization
        if operator is None:
            merged_config = {**(config or {})}
            merged_config.update(state.get("operator_configs", {}).get("trajectory", {}))
            operator = create_operator("trajectory_predictor", merged_config, "cpu")

        # Execute with timing
        with NodeTimer(state, "trajectory") as timer:
            try:
                result = operator.process({"objects": objects})
            except Exception as e:
                return {
                    "next_stage": PipelineStage.ERROR.value,
                    "errors": state.get("errors", []) + [f"trajectory: {e}"],
                    "timing": timer.timing,
                }

        # Store result for parallel execution join
        parallel_update = merge_parallel_result(state, "trajectory", {"trajectories": result})

        return {
            **parallel_update,
            "trajectories": result,
            "next_stage": PipelineStage.COLLISION.value,
            "timing": timer.timing,
        }

    return trajectory_node


def create_collision_node(
    config: Optional[dict[str, Any]] = None,
) -> Callable[[EmbodiedPipelineState], dict]:
    """
    Create collision detection node.

    Wraps CollisionDetector operator for LangGraph execution.
    Assesses collision risk based on predicted trajectories and ego pose.

    Args:
        config: Operator configuration overrides

    Returns:
        Node function for LangGraph
    """
    operator: Optional[Operator] = None

    def collision_node(state: EmbodiedPipelineState) -> dict:
        nonlocal operator

        # Skip if not enabled
        if not is_stage_enabled(state, PipelineStage.COLLISION):
            return {
                "collision_risks": {
                    "collision_risks": [],
                    "has_critical": False,
                    "has_warning": False,
                },
                "next_stage": PipelineStage.PATH_FOLLOW.value,
            }

        # Get trajectories
        trajectories = state.get("trajectories", {})
        predictions = trajectories.get("predictions", [])

        # Get ego pose for collision calculation
        ego_pose = state.get("ego_pose", {})
        ego_position = ego_pose.get("pose", [0.0, 0.0, 0.0])[:3]

        # Skip if no predictions (safe by default)
        if not predictions:
            return {
                "collision_risks": {
                    "collision_risks": [],
                    "has_critical": False,
                    "has_warning": False,
                    "closest_distance": None,
                    "time_to_collision": None,
                },
                "next_stage": PipelineStage.PATH_FOLLOW.value,
                "timing": {**state.get("timing", {}), "collision": 0.0},
            }

        # Lazy initialization
        if operator is None:
            merged_config = {**(config or {})}
            merged_config.update(state.get("operator_configs", {}).get("collision", {}))
            operator = create_operator("collision_detector", merged_config, "cpu")

        # Execute with timing
        with NodeTimer(state, "collision") as timer:
            try:
                result = operator.process(
                    {
                        "predictions": predictions,
                        "ego_position": ego_position,
                    }
                )
            except Exception as e:
                return {
                    "next_stage": PipelineStage.ERROR.value,
                    "errors": state.get("errors", []) + [f"collision: {e}"],
                    "timing": timer.timing,
                }

        # Determine next stage based on collision risk
        if result.get("has_critical", False):
            next_stage = PipelineStage.EMERGENCY_STOP.value
        else:
            next_stage = PipelineStage.PATH_FOLLOW.value

        return {
            "collision_risks": result,
            "next_stage": next_stage,
            "timing": timer.timing,
        }

    return collision_node
