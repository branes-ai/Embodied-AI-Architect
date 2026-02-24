"""LLM tool definitions for multi-objective optimization.

Follows the get_*_tool_definitions() + create_*_tool_executors() pattern
from codebase_tools.py. Provides 5 tools for design space exploration.

When MCP server available: tools proxy to MCP (async, stateful sessions).
When MCP not available: tools run optimization inline (blocking).

Usage:
    from embodied_ai_architect.llm.optimization_tools import (
        get_optimization_tool_definitions,
        create_optimization_tool_executors,
    )
"""

from __future__ import annotations

import json
import traceback
from typing import Any, Callable


def get_optimization_tool_definitions() -> list[dict[str, Any]]:
    """Get tool definitions for optimization in Anthropic format."""
    return [
        {
            "name": "explore_design_space",
            "description": (
                "Explore the SoC design space using multi-objective optimization. "
                "Runs MAP-Elites for fast design space illumination, optionally followed "
                "by Bayesian optimization for refined Pareto front. Returns Pareto-optimal "
                "design points with power, latency, area, and cost tradeoffs."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "goal": {
                        "type": "string",
                        "description": "Design goal (e.g., 'drone perception SoC with real-time detection')",
                    },
                    "max_power_watts": {
                        "type": "number",
                        "description": "Power budget in watts (optional)",
                    },
                    "max_latency_ms": {
                        "type": "number",
                        "description": "Latency target in ms (optional)",
                    },
                    "max_cost_usd": {
                        "type": "number",
                        "description": "Cost budget in USD (optional)",
                    },
                    "fast_mode": {
                        "type": "boolean",
                        "description": "Use reduced evaluation budget for quick results (default: true)",
                    },
                },
                "required": ["goal"],
            },
        },
        {
            "name": "get_pareto_front",
            "description": (
                "Get the Pareto-optimal design points from the most recent design "
                "space exploration. Shows the best tradeoff designs."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "top_n": {
                        "type": "integer",
                        "description": "Number of top designs to return (default: 5)",
                    },
                },
                "required": [],
            },
        },
        {
            "name": "get_design_sensitivity",
            "description": (
                "Get parameter sensitivity analysis showing which design variables "
                "most affect each objective (power, latency, area, cost)."
            ),
            "input_schema": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
        {
            "name": "explain_design_tradeoff",
            "description": (
                "Explain the tradeoff between two design points from the Pareto front. "
                "Shows what parameter changes lead to what objective changes."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "point_a": {
                        "type": "integer",
                        "description": "Index of first design point (0-based)",
                    },
                    "point_b": {
                        "type": "integer",
                        "description": "Index of second design point (0-based)",
                    },
                },
                "required": ["point_a", "point_b"],
            },
        },
        {
            "name": "suggest_optimal_design",
            "description": (
                "Suggest the best design point based on the exploration results "
                "and user priorities. Returns the knee point or the design that "
                "best balances all objectives."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "priority": {
                        "type": "string",
                        "enum": ["balanced", "low_power", "low_latency", "low_cost", "small_area"],
                        "description": "Optimization priority (default: balanced)",
                    },
                },
                "required": [],
            },
        },
    ]


# Module-level state for inline mode
_last_result: dict[str, Any] | None = None


def create_optimization_tool_executors() -> dict[str, Callable]:
    """Create executor functions for optimization tools."""

    def explore_design_space(
        goal: str,
        max_power_watts: float | None = None,
        max_latency_ms: float | None = None,
        max_cost_usd: float | None = None,
        fast_mode: bool = True,
    ) -> str:
        """Execute design space exploration."""
        global _last_result
        try:
            from embodied_ai_architect.graphs.moo.design_space import create_soc_design_space
            from embodied_ai_architect.graphs.moo.evaluator import DesignEvaluator
            from embodied_ai_architect.graphs.moo.engine import (
                OptimizationConfig,
                OptimizationEngine,
            )
            from embodied_ai_architect.graphs.moo.map_elites import MAPElitesConfig

            constraints: dict[str, Any] = {}
            if max_power_watts is not None:
                constraints["max_power_watts"] = max_power_watts
            if max_latency_ms is not None:
                constraints["max_latency_ms"] = max_latency_ms
            if max_cost_usd is not None:
                constraints["max_cost_usd"] = max_cost_usd

            ds = create_soc_design_space(constraints)
            evaluator = DesignEvaluator(
                design_space=ds,
                base_state={"constraints": constraints},
                constraint_bounds=ds.constraint_bounds,
            )

            if fast_mode:
                me_config = MAPElitesConfig(
                    n_iterations=20,
                    batch_size=32,
                    initial_population=64,
                )
                config = OptimizationConfig(
                    layers="map_elites",
                    map_elites=me_config,
                    max_workers=4,
                )
            else:
                config = OptimizationConfig(layers="auto")

            engine = OptimizationEngine(ds, evaluator, config)
            try:
                result = engine.run()
            finally:
                engine.shutdown()

            _last_result = result.model_dump()

            # Return summary for LLM
            front = result.pareto_front[:5]
            output = {
                "total_evaluations": result.total_evaluations,
                "pareto_front_size": len(result.pareto_front),
                "hypervolume": result.hypervolume,
                "layers_used": result.layers_used,
                "top_designs": front,
                "knee_point": result.knee_point,
            }
            return json.dumps(output, indent=2, default=str)

        except Exception as e:
            return f"Error: {e}\n{traceback.format_exc()}"

    def get_pareto_front(top_n: int = 5) -> str:
        """Get Pareto front from last exploration."""
        if _last_result is None:
            return json.dumps({"error": "No exploration results. Run explore_design_space first."})

        front = _last_result.get("pareto_front", [])[:top_n]
        return json.dumps(
            {
                "designs": front,
                "total": len(_last_result.get("pareto_front", [])),
                "hypervolume": _last_result.get("hypervolume", 0),
            },
            indent=2,
            default=str,
        )

    def get_design_sensitivity() -> str:
        """Get parameter sensitivity from last exploration."""
        if _last_result is None:
            return json.dumps({"error": "No exploration results. Run explore_design_space first."})

        return json.dumps(
            {
                "sensitivity": _last_result.get("sensitivity", {}),
                "note": (
                    "Sensitivity requires Bayesian optimization layer (botorch)."
                    if not _last_result.get("sensitivity")
                    else "Higher importance = more impact on objective."
                ),
            },
            indent=2,
            default=str,
        )

    def explain_design_tradeoff(point_a: int, point_b: int) -> str:
        """Explain tradeoff between two designs."""
        if _last_result is None:
            return json.dumps({"error": "No exploration results."})

        front = _last_result.get("pareto_front", [])
        if point_a >= len(front) or point_b >= len(front):
            return json.dumps({"error": f"Index out of range (front has {len(front)} points)"})

        pa = front[point_a]
        pb = front[point_b]

        obj_deltas = {}
        for name in ["power_watts", "latency_ms", "area_mm2", "cost_usd"]:
            va = pa.get("objectives", {}).get(name, 0)
            vb = pb.get("objectives", {}).get(name, 0)
            obj_deltas[name] = {
                "point_a": va,
                "point_b": vb,
                "delta": round(vb - va, 4),
                "pct_change": round((vb - va) / va * 100, 1) if va else 0,
            }

        param_changes = {}
        da = pa.get("design_params", {})
        db = pb.get("design_params", {})
        for k in set(list(da.keys()) + list(db.keys())):
            if da.get(k) != db.get(k):
                param_changes[k] = {"from": da.get(k), "to": db.get(k)}

        return json.dumps(
            {
                "objective_deltas": obj_deltas,
                "parameter_changes": param_changes,
            },
            indent=2,
            default=str,
        )

    def suggest_optimal_design(priority: str = "balanced") -> str:
        """Suggest the best design based on priority."""
        if _last_result is None:
            return json.dumps({"error": "No exploration results."})

        front = _last_result.get("pareto_front", [])
        if not front:
            return json.dumps({"error": "Empty Pareto front."})

        if priority == "balanced":
            design = _last_result.get("knee_point") or front[0]
        elif priority == "low_power":
            design = min(front, key=lambda d: d.get("objectives", {}).get("power_watts", 1e6))
        elif priority == "low_latency":
            design = min(front, key=lambda d: d.get("objectives", {}).get("latency_ms", 1e6))
        elif priority == "low_cost":
            design = min(front, key=lambda d: d.get("objectives", {}).get("cost_usd", 1e6))
        elif priority == "small_area":
            design = min(front, key=lambda d: d.get("objectives", {}).get("area_mm2", 1e6))
        else:
            design = front[0]

        return json.dumps(
            {
                "priority": priority,
                "suggested_design": design,
                "pareto_front_size": len(front),
            },
            indent=2,
            default=str,
        )

    return {
        "explore_design_space": explore_design_space,
        "get_pareto_front": get_pareto_front,
        "get_design_sensitivity": get_design_sensitivity,
        "explain_design_tradeoff": explain_design_tradeoff,
        "suggest_optimal_design": suggest_optimal_design,
    }
