"""LangGraph nodes for control operators."""

from __future__ import annotations

from typing import Any, Callable, Optional

from embodied_ai_architect.graphs.state import EmbodiedPipelineState, PipelineStage
from embodied_ai_architect.graphs.utils import NodeTimer, is_stage_enabled
from embodied_ai_architect.operators import create_operator, Operator


def create_path_planner_node(
    config: Optional[dict[str, Any]] = None,
) -> Callable[[EmbodiedPipelineState], dict]:
    """
    Create A* path planning node.

    Wraps PathPlannerAStar operator for LangGraph execution.
    Plans path from current position to goal avoiding obstacles.

    Args:
        config: Operator configuration overrides

    Returns:
        Node function for LangGraph
    """
    operator: Optional[Operator] = None

    def path_planner_node(state: EmbodiedPipelineState) -> dict:
        nonlocal operator

        # Skip if not enabled
        if not is_stage_enabled(state, PipelineStage.PLANNING):
            return {"next_stage": PipelineStage.PATH_FOLLOW.value}

        # Get current pose and goal
        ego_pose = state.get("ego_pose", {})
        pose = ego_pose.get("pose", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        start = (pose[0], pose[1])

        goal_data = state.get("goal", {})
        goal_position = goal_data.get("position")

        if goal_position is None:
            # No goal specified - skip planning
            return {
                "planned_path": {"path": [], "success": False},
                "next_stage": PipelineStage.PATH_FOLLOW.value,
                "timing": {**state.get("timing", {}), "planning": 0.0},
            }

        goal = tuple(goal_position[:2])

        # Get obstacles from scene graph for grid
        scene_objects = state.get("scene_objects", {})
        obstacles = scene_objects.get("obstacles", [])

        # Lazy initialization
        if operator is None:
            merged_config = {**(config or {})}
            merged_config.update(state.get("operator_configs", {}).get("planning", {}))
            operator = create_operator("path_planner_astar", merged_config, "cpu")

        # Execute with timing
        with NodeTimer(state, "planning") as timer:
            try:
                # Create occupancy grid from obstacles (simplified)
                import numpy as np

                grid_size = (100, 100)
                grid = np.zeros(grid_size, dtype=np.uint8)

                # Mark obstacles in grid (simplified - real impl would use proper transforms)
                for obj in obstacles:
                    pos = obj.get("position", [0, 0, 0])
                    # Convert world coords to grid coords (simplified)
                    gx = int(pos[0] * 10 + 50)
                    gy = int(pos[1] * 10 + 50)
                    if 0 <= gx < grid_size[0] and 0 <= gy < grid_size[1]:
                        grid[gx, gy] = 1

                result = operator.process(
                    {
                        "start": start,
                        "goal": goal,
                        "grid": grid,
                        "use_meters": True,
                    }
                )
            except Exception as e:
                return {
                    "next_stage": PipelineStage.ERROR.value,
                    "errors": state.get("errors", []) + [f"planning: {e}"],
                    "timing": timer.timing,
                }

        return {
            "planned_path": result,
            "next_stage": PipelineStage.PATH_FOLLOW.value,
            "timing": timer.timing,
        }

    return path_planner_node


def create_path_follower_node(
    config: Optional[dict[str, Any]] = None,
) -> Callable[[EmbodiedPipelineState], dict]:
    """
    Create Pure Pursuit path following node.

    Wraps PathFollower operator for LangGraph execution.
    Generates velocity commands to follow a path.

    Args:
        config: Operator configuration overrides

    Returns:
        Node function for LangGraph
    """
    operator: Optional[Operator] = None

    def path_follower_node(state: EmbodiedPipelineState) -> dict:
        nonlocal operator

        # Skip if not enabled
        if not is_stage_enabled(state, PipelineStage.PATH_FOLLOW):
            return {"next_stage": PipelineStage.COMPLETE.value}

        # Get current pose
        ego_pose = state.get("ego_pose", {})
        pose_6dof = ego_pose.get("pose", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        pose = [pose_6dof[0], pose_6dof[1], pose_6dof[5]]  # [x, y, yaw]

        # Get path
        planned_path = state.get("planned_path", {})
        path = planned_path.get("path", [])

        # Apply speed reduction if warning-level collision risk
        collision_risks = state.get("collision_risks", {})
        speed_factor = 0.5 if collision_risks.get("has_warning", False) else 1.0

        # If no path, output zero velocity
        if not path:
            return {
                "control_output": {
                    "velocity": 0.0,
                    "angular_velocity": 0.0,
                    "at_goal": True,
                    "speed_factor": speed_factor,
                },
                "next_stage": PipelineStage.COMPLETE.value,
                "timing": {**state.get("timing", {}), "path_follow": 0.0},
            }

        # Lazy initialization
        if operator is None:
            merged_config = {**(config or {})}
            merged_config.update(state.get("operator_configs", {}).get("path_follow", {}))
            operator = create_operator("trajectory_follower", merged_config, "cpu")

        # Execute with timing
        with NodeTimer(state, "path_follow") as timer:
            try:
                result = operator.process(
                    {
                        "pose": pose,
                        "path": path,
                    }
                )
            except Exception as e:
                return {
                    "next_stage": PipelineStage.ERROR.value,
                    "errors": state.get("errors", []) + [f"path_follow: {e}"],
                    "timing": timer.timing,
                }

        # Apply speed factor for collision avoidance
        output = {
            "velocity": result.get("velocity", 0.0) * speed_factor,
            "angular_velocity": result.get("angular_velocity", 0.0),
            "lookahead_point": result.get("lookahead_point"),
            "at_goal": result.get("at_goal", False),
            "speed_factor": speed_factor,
        }

        return {
            "control_output": output,
            "next_stage": PipelineStage.COMPLETE.value,
            "timing": timer.timing,
        }

    return path_follower_node


def create_emergency_stop_node() -> Callable[[EmbodiedPipelineState], dict]:
    """
    Create emergency stop node.

    Outputs zero velocity and emergency stop flag when triggered
    by critical collision risk.

    Returns:
        Node function for LangGraph
    """

    def emergency_stop_node(state: EmbodiedPipelineState) -> dict:
        collision_risks = state.get("collision_risks", {})
        closest = collision_risks.get("closest_distance")
        ttc = collision_risks.get("time_to_collision")

        return {
            "control_output": {
                "velocity": 0.0,
                "angular_velocity": 0.0,
                "emergency_stop": True,
                "at_goal": False,
                "reason": f"Critical collision risk: distance={closest}, TTC={ttc}",
            },
            "next_stage": PipelineStage.COMPLETE.value,
            "timing": {**state.get("timing", {}), "emergency_stop": 0.0},
        }

    return emergency_stop_node
