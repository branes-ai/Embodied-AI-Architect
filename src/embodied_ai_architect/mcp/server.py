"""MCP server exposing optimization tools for the LLM agent.

Dual-Response pattern: tools return a preview for the LLM context
plus a reference to the full data for detailed analysis.

Requires mcp (optional dependency).

Usage:
    from embodied_ai_architect.mcp.server import create_mcp_server

    server = create_mcp_server()
    # Server exposes tools via MCP protocol
"""

from __future__ import annotations

import json
import logging
from typing import Any

from embodied_ai_architect.mcp.session import SessionManager

logger = logging.getLogger(__name__)

# Global session manager
_session_manager: SessionManager | None = None


def get_session_manager() -> SessionManager:
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager


def get_mcp_tool_definitions() -> list[dict[str, Any]]:
    """Get MCP tool definitions for the optimization engine."""
    return [
        {
            "name": "start_exploration",
            "description": (
                "Start a multi-objective design space exploration. Runs the 3-layer "
                "MOO pipeline (MAP-Elites -> Bayesian BO) in the background. "
                "Returns a session_id for tracking progress and querying results."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "goal": {
                        "type": "string",
                        "description": "Design goal (e.g., 'drone perception SoC')",
                    },
                    "constraints": {
                        "type": "object",
                        "description": "Design constraints dict with keys like max_power_watts, max_latency_ms, max_cost_usd",
                    },
                    "config": {
                        "type": "object",
                        "description": "Optional optimization config overrides",
                    },
                },
                "required": ["goal", "constraints"],
            },
        },
        {
            "name": "get_pareto_front",
            "description": (
                "Get the Pareto-optimal design points from a completed exploration. "
                "Returns top-N designs for LLM context with a link to the full front."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "session_id": {"type": "string"},
                    "top_n": {"type": "integer", "default": 5},
                },
                "required": ["session_id"],
            },
        },
        {
            "name": "get_sensitivity",
            "description": (
                "Get parameter sensitivity analysis from a completed exploration. "
                "Shows which design parameters most affect each objective."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "session_id": {"type": "string"},
                },
                "required": ["session_id"],
            },
        },
        {
            "name": "explain_tradeoff",
            "description": (
                "Explain the tradeoff between two design points from the Pareto front. "
                "Shows what changes in parameters lead to what changes in objectives."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "session_id": {"type": "string"},
                    "point_a_index": {"type": "integer", "description": "Index of first point"},
                    "point_b_index": {"type": "integer", "description": "Index of second point"},
                },
                "required": ["session_id", "point_a_index", "point_b_index"],
            },
        },
        {
            "name": "get_exploration_status",
            "description": "Get the status and progress of an optimization session.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "session_id": {"type": "string"},
                },
                "required": ["session_id"],
            },
        },
    ]


def execute_mcp_tool(tool_name: str, args: dict[str, Any]) -> str:
    """Execute an MCP tool and return JSON result."""
    manager = get_session_manager()

    if tool_name == "start_exploration":
        return _start_exploration(manager, args)
    elif tool_name == "get_pareto_front":
        return _get_pareto_front(manager, args)
    elif tool_name == "get_sensitivity":
        return _get_sensitivity(manager, args)
    elif tool_name == "explain_tradeoff":
        return _explain_tradeoff(manager, args)
    elif tool_name == "get_exploration_status":
        return _get_exploration_status(manager, args)
    else:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})


def _start_exploration(manager: SessionManager, args: dict[str, Any]) -> str:
    """Start a new optimization session."""
    try:
        session_id = manager.create_session(
            goal=args["goal"],
            constraints=args.get("constraints", {}),
            config=args.get("config"),
        )
        return json.dumps(
            {
                "session_id": session_id,
                "status": "running",
                "message": "Optimization started. Use get_exploration_status to track progress.",
            }
        )
    except Exception as e:
        return json.dumps({"error": str(e)})


def _get_pareto_front(manager: SessionManager, args: dict[str, Any]) -> str:
    """Get Pareto front with dual-response pattern."""
    session_id = args["session_id"]
    top_n = args.get("top_n", 5)

    result = manager.get_result(session_id)
    if result is None:
        status = manager.get_status(session_id)
        return json.dumps({"error": "Results not ready", "status": status})

    front = result.get("pareto_front", [])

    # Preview: top-N for LLM context
    preview = front[:top_n]

    return json.dumps(
        {
            "preview": preview,
            "total_points": len(front),
            "hypervolume": result.get("hypervolume", 0),
            "knee_point": result.get("knee_point"),
        },
        indent=2,
    )


def _get_sensitivity(manager: SessionManager, args: dict[str, Any]) -> str:
    """Get parameter sensitivity analysis."""
    session_id = args["session_id"]

    result = manager.get_result(session_id)
    if result is None:
        return json.dumps({"error": "Results not ready"})

    return json.dumps(
        {
            "sensitivity": result.get("sensitivity", {}),
            "layers_used": result.get("layers_used", []),
        },
        indent=2,
    )


def _explain_tradeoff(manager: SessionManager, args: dict[str, Any]) -> str:
    """Explain tradeoff between two Pareto points."""
    session_id = args["session_id"]
    idx_a = args["point_a_index"]
    idx_b = args["point_b_index"]

    result = manager.get_result(session_id)
    if result is None:
        return json.dumps({"error": "Results not ready"})

    front = result.get("pareto_front", [])
    if idx_a >= len(front) or idx_b >= len(front):
        return json.dumps({"error": f"Index out of range (front has {len(front)} points)"})

    from embodied_ai_architect.graphs.moo.design_space import create_soc_design_space

    ds = create_soc_design_space()
    # Use engine's static method for tradeoff explanation
    point_a = front[idx_a]
    point_b = front[idx_b]

    obj_deltas = {}
    for name in ds.objectives:
        va = point_a.get("objectives", {}).get(name, 0)
        vb = point_b.get("objectives", {}).get(name, 0)
        obj_deltas[name] = {
            "point_a": va,
            "point_b": vb,
            "delta": round(vb - va, 4),
            "pct_change": round((vb - va) / va * 100, 1) if va != 0 else 0,
        }

    param_deltas = {}
    pa = point_a.get("design_params", {})
    pb = point_b.get("design_params", {})
    for key in set(list(pa.keys()) + list(pb.keys())):
        if pa.get(key) != pb.get(key):
            param_deltas[key] = {"from": pa.get(key), "to": pb.get(key)}

    return json.dumps(
        {
            "objective_deltas": obj_deltas,
            "parameter_changes": param_deltas,
        },
        indent=2,
    )


def _get_exploration_status(manager: SessionManager, args: dict[str, Any]) -> str:
    """Get exploration status."""
    return json.dumps(manager.get_status(args["session_id"]), indent=2)
