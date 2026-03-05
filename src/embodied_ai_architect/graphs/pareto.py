"""Pareto front exploration for multi-objective SoC design space.

Computes the non-dominated Pareto front across power, latency, and cost
objectives, identifies the knee point, and exposes a specialist agent
for integration into the task graph.

Usage:
    from embodied_ai_architect.graphs.pareto import (
        ParetoPoint, compute_pareto_front, identify_knee_point, design_explorer,
    )
"""

from __future__ import annotations

import logging
import math
from typing import Any

from pydantic import BaseModel, Field

from embodied_ai_architect.graphs.soc_state import SoCDesignState, get_constraints
from embodied_ai_architect.graphs.task_graph import TaskNode

logger = logging.getLogger(__name__)


class ParetoPoint(BaseModel):
    """A single design point in the multi-objective space."""

    hardware_name: str = ""
    power: float = 0.0
    latency: float = 0.0
    cost: float = 0.0
    dominated: bool = False
    knee_point: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


def compute_pareto_front(
    candidates: list[dict[str, Any]],
    objectives: list[str] | None = None,
) -> list[ParetoPoint]:
    """Compute the non-dominated Pareto front from hardware candidates.

    All objectives are minimized. A candidate is dominated if another candidate
    is <= on all objectives and < on at least one.

    Args:
        candidates: List of hardware candidate dicts with numeric fields.
        objectives: Field names to optimize (default: power, latency, cost).

    Returns:
        All candidates as ParetoPoint with dominated flag set.
    """
    if objectives is None:
        objectives = ["power", "latency", "cost"]

    # Map candidates to objective vectors
    field_map = {
        "power": "tdp_watts",
        "latency": "latency_ms",
        "cost": "cost_usd",
    }

    points: list[ParetoPoint] = []
    vectors: list[list[float]] = []

    for c in candidates:
        vals = []
        for obj in objectives:
            field = field_map.get(obj, obj)
            val = c.get(field, c.get(obj, float("inf")))
            if val is None:
                val = float("inf")
            vals.append(float(val))
        vectors.append(vals)
        points.append(
            ParetoPoint(
                hardware_name=c.get("name", "unknown"),
                power=c.get("tdp_watts", c.get("power", 0.0)) or 0.0,
                latency=c.get("latency_ms", c.get("latency", 0.0)) or 0.0,
                cost=c.get("cost_usd", c.get("cost", 0.0)) or 0.0,
                metadata={k: v for k, v in c.items() if k not in ("name",)},
            )
        )

    # Non-dominated sorting
    n = len(vectors)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            vi, vj = vectors[i], vectors[j]
            # Check if j dominates i (j <= i on all, j < i on at least one)
            all_leq = all(vj[k] <= vi[k] for k in range(len(objectives)))
            any_lt = any(vj[k] < vi[k] for k in range(len(objectives)))
            if all_leq and any_lt:
                points[i].dominated = True
                break

    return points


def identify_knee_point(front: list[ParetoPoint]) -> ParetoPoint | None:
    """Identify the knee point of a Pareto front.

    The knee point is the non-dominated point with minimum normalized
    Euclidean distance to the origin (utopia point).

    Args:
        front: List of ParetoPoints (may include dominated points).

    Returns:
        The knee point, or None if no non-dominated points exist.
    """
    non_dominated = [p for p in front if not p.dominated]
    if not non_dominated:
        return None

    # Compute min/max for normalization
    powers = [p.power for p in non_dominated]
    latencies = [p.latency for p in non_dominated]
    costs = [p.cost for p in non_dominated]

    p_range = max(powers) - min(powers) if len(set(powers)) > 1 else 1.0
    l_range = max(latencies) - min(latencies) if len(set(latencies)) > 1 else 1.0
    c_range = max(costs) - min(costs) if len(set(costs)) > 1 else 1.0

    p_min = min(powers)
    l_min = min(latencies)
    c_min = min(costs)

    best_dist = float("inf")
    best_point = None

    for p in non_dominated:
        norm_p = (p.power - p_min) / p_range if p_range > 0 else 0.0
        norm_l = (p.latency - l_min) / l_range if l_range > 0 else 0.0
        norm_c = (p.cost - c_min) / c_range if c_range > 0 else 0.0
        dist = math.sqrt(norm_p**2 + norm_l**2 + norm_c**2)
        if dist < best_dist:
            best_dist = dist
            best_point = p

    if best_point is not None:
        best_point.knee_point = True

    return best_point


def design_explorer(task: TaskNode, state: SoCDesignState) -> dict[str, Any]:
    """Specialist agent: explore the hardware design space via Pareto analysis.

    Reads hardware_candidates from state, computes the Pareto front across
    power/latency/cost, identifies the knee point, and writes results to state.

    Writes to state: pareto_results
    """
    candidates = state.get("hardware_candidates", [])
    get_constraints(state)

    if not candidates:
        return {
            "summary": "No hardware candidates to explore",
            "pareto_results": {"front": [], "knee_point": None, "total": 0},
            "_state_updates": {
                "pareto_results": {"front": [], "knee_point": None, "total": 0},
            },
        }

    # Add latency estimates to candidates if missing
    enriched = _enrich_candidates_with_latency(candidates, state)

    front = compute_pareto_front(enriched)
    knee = identify_knee_point(front)

    non_dominated = [p for p in front if not p.dominated]
    front_dicts = [p.model_dump() for p in front]
    knee_dict = knee.model_dump() if knee else None

    results = {
        "front": front_dicts,
        "knee_point": knee_dict,
        "total": len(front),
        "non_dominated_count": len(non_dominated),
    }

    summary_parts = [
        f"Explored {len(front)} design points",
        f"{len(non_dominated)} non-dominated",
    ]
    if knee:
        summary_parts.append(f"knee={knee.hardware_name}")

    return {
        "summary": ", ".join(summary_parts),
        "pareto_results": results,
        "_state_updates": {"pareto_results": results},
    }


# ---------------------------------------------------------------------------
# Generic N-objective Pareto utilities
# ---------------------------------------------------------------------------


class GenericParetoPoint(BaseModel):
    """A design point in an N-dimensional objective space."""

    name: str = ""
    objectives: dict[str, float] = Field(default_factory=dict)
    dominated: bool = False
    knee_point: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


def compute_pareto_front_generic(
    candidates: list[dict[str, Any]],
    objective_names: list[str],
    directions: list[str] | None = None,
) -> list[GenericParetoPoint]:
    """Compute non-dominated Pareto front for N arbitrary objectives.

    Args:
        candidates: List of dicts with at least 'name' and 'objectives' keys.
            'objectives' should be a dict mapping objective_name -> value.
            Alternatively, objective values can be top-level keys.
        objective_names: Names of objectives to consider.
        directions: Per-objective direction ("minimize" or "maximize").
            Defaults to all minimize.

    Returns:
        All candidates as GenericParetoPoint with dominated flag set.
    """
    if directions is None:
        directions = ["minimize"] * len(objective_names)

    # Extract objective vectors with sign flip for maximize
    points: list[GenericParetoPoint] = []
    vectors: list[list[float]] = []

    for c in candidates:
        objs = c.get("objectives", c)
        vals: list[float] = []
        for i, name in enumerate(objective_names):
            val = float(objs.get(name, float("inf")))
            # Flip sign for maximize objectives so we can use minimize-only domination
            if directions[i] == "maximize":
                val = -val
            vals.append(val)
        vectors.append(vals)
        points.append(
            GenericParetoPoint(
                name=c.get("name", ""),
                objectives={name: float(objs.get(name, 0.0)) for name in objective_names},
                metadata={k: v for k, v in c.items() if k not in ("name", "objectives")},
            )
        )

    # Non-dominated sorting: O(n^2)
    n = len(vectors)
    n_obj = len(objective_names)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            vi, vj = vectors[i], vectors[j]
            all_leq = all(vj[k] <= vi[k] for k in range(n_obj))
            any_lt = any(vj[k] < vi[k] for k in range(n_obj))
            if all_leq and any_lt:
                points[i].dominated = True
                break

    return points


def identify_knee_point_generic(
    front: list[GenericParetoPoint],
    objective_names: list[str],
) -> GenericParetoPoint | None:
    """Identify the knee point of a generic Pareto front.

    The knee point is the non-dominated point with minimum normalized
    Euclidean distance to the utopia point (minimum of each objective).

    Args:
        front: List of GenericParetoPoints.
        objective_names: Objective names to use for distance calculation.

    Returns:
        The knee point, or None if no non-dominated points exist.
    """
    non_dominated = [p for p in front if not p.dominated]
    if not non_dominated:
        return None

    # Compute per-objective min/range for normalization
    obj_min: dict[str, float] = {}
    obj_range: dict[str, float] = {}
    for name in objective_names:
        values = [p.objectives.get(name, 0.0) for p in non_dominated]
        mn = min(values)
        mx = max(values)
        obj_min[name] = mn
        obj_range[name] = (mx - mn) if mx > mn else 1.0

    best_dist = float("inf")
    best_point: GenericParetoPoint | None = None

    for p in non_dominated:
        dist_sq = 0.0
        for name in objective_names:
            norm = (p.objectives.get(name, 0.0) - obj_min[name]) / obj_range[name]
            dist_sq += norm * norm
        dist = math.sqrt(dist_sq)
        if dist < best_dist:
            best_dist = dist
            best_point = p

    if best_point is not None:
        best_point.knee_point = True

    return best_point


def _enrich_candidates_with_latency(
    candidates: list[dict[str, Any]], state: SoCDesignState
) -> list[dict[str, Any]]:
    """Add latency_ms estimate to candidates if not present."""
    workload = state.get("workload_profile", {})
    total_gflops = workload.get("total_estimated_gflops", workload.get("estimated_gflops", 5.0))

    enriched = []
    for c in candidates:
        c = dict(c)  # copy
        if "latency_ms" not in c:
            peak_tops = c.get("peak_tops_int8", 1.0)
            if peak_tops <= 0:
                peak_tops = c.get("peak_tflops_fp16", 1.0)
            utilization = 0.3
            if peak_tops > 0:
                latency_s = (total_gflops / 1000) / (peak_tops * utilization)
                c["latency_ms"] = round(latency_s * 1000 + 2.0, 1)
            else:
                c["latency_ms"] = 100.0
        enriched.append(c)
    return enriched
