"""Utility functions for LangGraph operator pipelines."""

from __future__ import annotations

import time
from typing import Any, Callable, Optional

from embodied_ai_architect.graphs.state import EmbodiedPipelineState, PipelineStage


def route_by_next_stage(state: EmbodiedPipelineState) -> str:
    """
    Standard routing function for conditional edges.

    Extracts the `next_stage` field from state for LangGraph routing.

    Usage:
        workflow.add_conditional_edges(
            "node_name",
            route_by_next_stage,
            {
                PipelineStage.TRACK.value: "track",
                PipelineStage.ERROR.value: "error_handler",
            }
        )
    """
    return state.get("next_stage", PipelineStage.ERROR.value)


def route_on_collision_risk(state: EmbodiedPipelineState) -> str:
    """
    Route based on collision risk assessment.

    Returns routing key based on collision_risks output:
    - "critical" -> emergency_stop
    - "warning" -> path_follower (reduced speed)
    - "safe" -> path_follower (normal speed)
    """
    collision_risks = state.get("collision_risks", {})

    if collision_risks.get("has_critical", False):
        return "critical"
    elif collision_risks.get("has_warning", False):
        return "warning"
    else:
        return "safe"


def is_stage_enabled(state: EmbodiedPipelineState, stage: PipelineStage) -> bool:
    """Check if a pipeline stage is enabled."""
    enabled = state.get("enabled_stages", [])
    return stage.value in enabled


def skip_to_next_stage(
    state: EmbodiedPipelineState,
    current: PipelineStage,
    next_stage: PipelineStage,
) -> dict:
    """
    Create state update to skip current stage and proceed to next.

    Used when a stage is disabled via enabled_stages.
    """
    return {"next_stage": next_stage.value}


def timed_execution(
    func: Callable[[dict[str, Any]], dict[str, Any]],
    inputs: dict[str, Any],
    stage_name: str,
    current_timing: dict[str, float],
) -> tuple[dict[str, Any], dict[str, float]]:
    """
    Execute a function with timing measurement.

    Args:
        func: Function to execute (typically operator.process)
        inputs: Input dictionary for the function
        stage_name: Name of the stage for timing dict
        current_timing: Current timing dict to update

    Returns:
        Tuple of (function result, updated timing dict)
    """
    start = time.perf_counter_ns()
    result = func(inputs)
    elapsed_ms = (time.perf_counter_ns() - start) / 1e6

    updated_timing = {**current_timing, stage_name: elapsed_ms}
    return result, updated_timing


def create_error_handler_node() -> Callable[[EmbodiedPipelineState], dict]:
    """
    Create an error handler node for pipeline failures.

    This node logs errors and terminates execution gracefully.
    """

    def error_handler_node(state: EmbodiedPipelineState) -> dict:
        errors = state.get("errors", [])
        if errors:
            # Log errors (could integrate with logging framework)
            for error in errors:
                print(f"[ERROR] {error}")

        return {
            "next_stage": PipelineStage.ERROR.value,
            "control_output": {
                "velocity": 0.0,
                "angular_velocity": 0.0,
                "emergency_stop": True,
                "reason": "; ".join(errors) if errors else "Unknown error",
            },
        }

    return error_handler_node


def create_join_node(
    expected_branches: list[str],
    output_stage: PipelineStage = PipelineStage.COLLISION,
) -> Callable[[EmbodiedPipelineState], dict]:
    """
    Create a join node that waits for parallel branches to complete.

    In LangGraph, parallel branches execute and their results are merged.
    This node checks if all expected results are present before proceeding.

    Args:
        expected_branches: List of branch names to wait for
        output_stage: Stage to route to when all branches complete

    Returns:
        Node function that synchronizes parallel branches
    """

    def join_node(state: EmbodiedPipelineState) -> dict:
        parallel_results = state.get("parallel_results", {})

        # Check which branches have completed
        [b for b in expected_branches if b in parallel_results]
        pending = [b for b in expected_branches if b not in parallel_results]

        if pending:
            # Still waiting for some branches
            return {
                "pending_branches": pending,
                "next_stage": PipelineStage.JOIN_STATE.value,
            }

        # All branches complete - merge results and proceed
        merged = {}
        for branch_name in expected_branches:
            branch_result = parallel_results.get(branch_name, {})
            merged.update(branch_result)

        return {
            **merged,
            "pending_branches": [],
            "next_stage": output_stage.value,
        }

    return join_node


def merge_parallel_result(
    state: EmbodiedPipelineState,
    branch_name: str,
    result: dict[str, Any],
) -> dict:
    """
    Helper to merge a parallel branch result into state.

    Used by parallel nodes to store their results for the join node.
    """
    current_results = state.get("parallel_results", {})
    return {
        "parallel_results": {
            **current_results,
            branch_name: result,
        }
    }


def create_passthrough_node(
    next_stage: PipelineStage,
) -> Callable[[EmbodiedPipelineState], dict]:
    """
    Create a node that passes through without modification.

    Useful for optional nodes that can be disabled.
    """

    def passthrough_node(state: EmbodiedPipelineState) -> dict:
        return {"next_stage": next_stage.value}

    return passthrough_node


class NodeTimer:
    """
    Context manager for timing node execution.

    Usage:
        with NodeTimer(state, "detect") as timer:
            result = operator.process(inputs)
        return {**result, "timing": timer.timing}
    """

    def __init__(self, state: EmbodiedPipelineState, stage_name: str):
        self.state = state
        self.stage_name = stage_name
        self.start_time: Optional[int] = None
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> "NodeTimer":
        self.start_time = time.perf_counter_ns()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.start_time is not None:
            self.elapsed_ms = (time.perf_counter_ns() - self.start_time) / 1e6

    @property
    def timing(self) -> dict[str, float]:
        """Get updated timing dict with this stage's measurement."""
        current = self.state.get("timing", {})
        return {**current, self.stage_name: self.elapsed_ms}
