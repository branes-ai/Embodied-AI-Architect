"""Optional LLM-powered architect node for runtime optimization."""

from __future__ import annotations

import json
from typing import Any, Callable, Optional

from embodied_ai_architect.graphs.state import (
    EmbodiedPipelineState,
    PipelineStage,
    get_total_latency_ms,
    is_over_budget,
)

# System prompt for the architect LLM
ARCHITECT_SYSTEM_PROMPT = """You are an embedded systems architect optimizing a real-time perception pipeline.

The pipeline consists of these stages:
- preprocess: Image preprocessing (resize, normalize)
- detect: YOLO object detection
- track: ByteTrack multi-object tracking
- scene_graph: 3D scene representation
- trajectory: Trajectory prediction
- collision: Collision risk assessment
- path_follow: Path following control

Your goal is to optimize the pipeline to meet latency constraints while maintaining functionality.

Available optimizations:
1. **Model variant**: Switch YOLO variant (n=nano, s=small, m=medium, l=large, x=extra)
   - Smaller variants are faster but less accurate
   - Current: {yolo_variant}

2. **Stage skipping**: Disable non-essential stages
   - Example: Skip trajectory prediction if no objects detected
   - Skip collision if objects far away

3. **Threshold adjustment**: Modify operator configs
   - Detection confidence threshold (higher = fewer detections, faster tracking)
   - Track buffer (shorter = fewer tracks to manage)

4. **Execution target**: cpu, gpu, npu
   - GPU is faster for detection but has memory overhead
   - NPU is fastest but limited availability

Respond with JSON containing optimization suggestions:
{
    "reasoning": "Brief explanation of why these changes help",
    "changes": {
        "yolo_variant": "n",  // Optional: change model variant
        "skip_stages": ["trajectory"],  // Optional: stages to disable
        "operator_configs": {  // Optional: per-operator config changes
            "detect": {"conf_threshold": 0.5}
        }
    }
}
"""


def format_optimization_request(state: EmbodiedPipelineState) -> str:
    """Format pipeline state into optimization request for LLM."""
    timing = state.get("timing", {})
    total_ms = get_total_latency_ms(state)
    budget_ms = state.get("latency_budget_ms", 100.0)

    # Sort stages by latency
    sorted_timing = sorted(timing.items(), key=lambda x: x[1], reverse=True)

    lines = [
        f"Current pipeline performance:",
        f"- Total latency: {total_ms:.2f}ms",
        f"- Budget: {budget_ms:.2f}ms",
        f"- Over budget by: {total_ms - budget_ms:.2f}ms",
        "",
        "Per-stage breakdown (sorted by latency):",
    ]

    for stage, ms in sorted_timing:
        pct = (ms / total_ms * 100) if total_ms > 0 else 0
        lines.append(f"  - {stage}: {ms:.2f}ms ({pct:.1f}%)")

    # Add context about current state
    detections = state.get("detections", {})
    num_detections = len(detections.get("detections", []))
    tracks = state.get("tracks", {})
    num_tracks = len(tracks.get("tracks", []))

    lines.extend(
        [
            "",
            f"Current workload:",
            f"- Detections: {num_detections}",
            f"- Active tracks: {num_tracks}",
        ]
    )

    return "\n".join(lines)


def apply_suggestions(
    state: EmbodiedPipelineState,
    suggestions: dict[str, Any],
) -> dict:
    """
    Apply LLM optimization suggestions to pipeline state.

    Args:
        state: Current pipeline state
        suggestions: Parsed suggestions from LLM

    Returns:
        State updates to apply
    """
    updates = {}
    changes = suggestions.get("changes", {})

    # Handle stage skipping
    skip_stages = changes.get("skip_stages", [])
    if skip_stages:
        current_enabled = state.get("enabled_stages", [])
        new_enabled = [s for s in current_enabled if s not in skip_stages]
        updates["enabled_stages"] = new_enabled

    # Handle operator config changes
    config_changes = changes.get("operator_configs", {})
    if config_changes:
        current_configs = state.get("operator_configs", {})
        merged_configs = {**current_configs}
        for op_name, op_config in config_changes.items():
            if op_name in merged_configs:
                merged_configs[op_name] = {**merged_configs[op_name], **op_config}
            else:
                merged_configs[op_name] = op_config
        updates["operator_configs"] = merged_configs

    # Note: yolo_variant and execution_target changes require graph rebuild
    # These are logged for the user but not applied dynamically
    if "yolo_variant" in changes:
        updates["_suggested_yolo_variant"] = changes["yolo_variant"]

    return updates


def create_architect_node(
    llm_client: Optional[Any] = None,
    enabled: bool = False,
    auto_optimize: bool = True,
) -> Callable[[EmbodiedPipelineState], dict]:
    """
    Create optional LLM-powered architect node.

    This node analyzes pipeline performance and suggests optimizations
    when the pipeline exceeds its latency budget.

    Args:
        llm_client: LLM client instance (from embodied_ai_architect.llm)
        enabled: Whether the architect is active
        auto_optimize: Automatically apply suggestions (vs just logging)

    Returns:
        Node function for LangGraph
    """

    def architect_node(state: EmbodiedPipelineState) -> dict:
        # Pass-through if disabled or no client
        if not enabled or llm_client is None:
            return {"next_stage": state.get("next_stage", PipelineStage.COMPLETE.value)}

        # Only optimize if over budget
        if not is_over_budget(state):
            return {"next_stage": state.get("next_stage", PipelineStage.COMPLETE.value)}

        # Format request for LLM
        request = format_optimization_request(state)

        try:
            # Call LLM for suggestions
            response = llm_client.chat(
                messages=[
                    {"role": "system", "content": ARCHITECT_SYSTEM_PROMPT},
                    {"role": "user", "content": request},
                ],
                tools=[],  # No tools needed for this
            )

            # Parse response
            content = response.text if hasattr(response, "text") else str(response)

            # Try to extract JSON from response
            try:
                # Look for JSON block in response
                if "```json" in content:
                    json_str = content.split("```json")[1].split("```")[0]
                elif "```" in content:
                    json_str = content.split("```")[1].split("```")[0]
                else:
                    json_str = content

                suggestions = json.loads(json_str.strip())

            except (json.JSONDecodeError, IndexError):
                # Couldn't parse - return without changes
                return {
                    "next_stage": state.get("next_stage", PipelineStage.COMPLETE.value),
                    "_architect_error": f"Could not parse LLM response: {content[:200]}",
                }

            # Apply suggestions if auto_optimize is enabled
            if auto_optimize:
                updates = apply_suggestions(state, suggestions)
                updates["_architect_reasoning"] = suggestions.get("reasoning", "")
                updates["next_stage"] = state.get("next_stage", PipelineStage.COMPLETE.value)
                return updates
            else:
                # Just log suggestions without applying
                return {
                    "next_stage": state.get("next_stage", PipelineStage.COMPLETE.value),
                    "_architect_suggestions": suggestions,
                }

        except Exception as e:
            # LLM call failed - continue without optimization
            return {
                "next_stage": state.get("next_stage", PipelineStage.COMPLETE.value),
                "_architect_error": str(e),
            }

    return architect_node
