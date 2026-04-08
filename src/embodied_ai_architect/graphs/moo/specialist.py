"""Bridge between MOO engine and the existing specialist dispatcher.

Provides `moo_explorer(task, state) -> dict` with the same signature as
all other specialists, reading constraints and workload from state and
writing both backward-compatible `pareto_results` and new `moo_results`.

Usage:
    Registered automatically in create_default_dispatcher() when available.
"""

from __future__ import annotations

import logging
from typing import Any

from embodied_ai_architect.graphs.soc_state import SoCDesignState, get_constraints
from embodied_ai_architect.graphs.task_graph import TaskNode

logger = logging.getLogger(__name__)


def _pareto_front_to_points(pareto_front: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OptimizationResult.pareto_front to ParetoPoint-compatible dicts."""
    points = []
    for p in pareto_front:
        objs = p.get("objectives", {})
        points.append(
            {
                "power": objs.get("power_watts"),
                "latency": objs.get("latency_ms"),
                "cost": objs.get("cost_usd"),
                "area": objs.get("area_mm2"),
                "hardware": f"custom-{objs.get('process_nm', '?')}nm",
                "dominated": False,
                "metadata": p,
            }
        )
    return points


def _find_knee_point_index(
    pareto_front: list[dict[str, Any]],
    knee_point: dict[str, Any] | None,
) -> int | None:
    """Find the index of the knee point in the Pareto front."""
    if not knee_point or not pareto_front:
        return None
    for i, p in enumerate(pareto_front):
        if p == knee_point:
            return i
    return 0 if pareto_front else None


# Default objective dimensions used by moo_explorer (4-objective MOO).
# swap_explorer uses 6 objectives — see _merge_swap_frontiers.
_DEFAULT_OBJECTIVES = ("power", "latency", "cost", "area")


def _merge_pareto_frontiers(
    accumulated: list[dict[str, Any]],
    new_points: list[dict[str, Any]],
    objective_keys: tuple[str, ...] = _DEFAULT_OBJECTIVES,
) -> tuple[list[dict[str, Any]], int, int]:
    """Merge new Pareto points with accumulated frontier.

    Returns (merged_non_dominated, num_new_added, num_accumulated_dominated).

    Each input point is in the ParetoPoint-compatible dict format produced
    by `_pareto_front_to_points()` (top-level `power`, `latency`, `cost`,
    `area`, plus `metadata` with the original objectives dict).
    """
    if not accumulated:
        # First run: just normalize the dominated flag
        merged = []
        for p in new_points:
            cp = dict(p)
            cp["dominated"] = False
            merged.append(cp)
        return merged, len(new_points), 0

    # Combine and re-compute dominance using the simple objective tuple.
    # All four default objectives are minimize (lower is better for power,
    # latency, cost, area).
    combined = list(accumulated) + list(new_points)
    n = len(combined)
    n_obj = len(objective_keys)

    # Skip points where any objective is None (corrupt/missing data)
    vectors: list[list[float] | None] = []
    for p in combined:
        try:
            vec = [float(p[k]) if p.get(k) is not None else float("inf") for k in objective_keys]
        except (TypeError, ValueError):
            vec = None
        vectors.append(vec)

    dominated_flags = [False] * n
    for i in range(n):
        if vectors[i] is None:
            dominated_flags[i] = True
            continue
        for j in range(n):
            if i == j or vectors[j] is None:
                continue
            vi, vj = vectors[i], vectors[j]
            all_leq = all(vj[k] <= vi[k] for k in range(n_obj))
            any_lt = any(vj[k] < vi[k] for k in range(n_obj))
            if all_leq and any_lt:
                dominated_flags[i] = True
                break

    # Build the merged non-dominated set
    merged: list[dict[str, Any]] = []
    accumulated_dominated = 0
    new_added = 0
    for idx, p in enumerate(combined):
        if dominated_flags[idx]:
            if idx < len(accumulated):
                accumulated_dominated += 1
            continue
        cp = dict(p)
        cp["dominated"] = False
        merged.append(cp)
        if idx >= len(accumulated):
            new_added += 1

    return merged, new_added, accumulated_dominated


def _build_history_entry(
    iteration: int,
    merged: list[dict[str, Any]],
    new_added: int,
    dominated_removed: int,
    hypervolume: float,
) -> dict[str, Any]:
    """Build a per-iteration frontier history snapshot for issue #23."""
    return {
        "iteration": iteration,
        "num_points": len(merged),
        "hypervolume": float(hypervolume),
        "new_points_added": new_added,
        "dominated_removed": dominated_removed,
    }


def moo_explorer(task: TaskNode, state: SoCDesignState) -> dict[str, Any]:
    """Specialist agent: multi-objective design space exploration.

    Reads constraints and workload_profile from state. Runs the 3-layer
    MOO pipeline (MAP-Elites -> BO/NSGA-III) and writes results.

    Supports task.metadata["fast_mode"] for reduced evaluation budgets.

    Writes to state: pareto_results (backward compat), moo_results (new)
    """
    from embodied_ai_architect.graphs.moo.design_space import create_soc_design_space
    from embodied_ai_architect.graphs.moo.evaluator import DesignEvaluator
    from embodied_ai_architect.graphs.moo.engine import OptimizationConfig, OptimizationEngine
    from embodied_ai_architect.graphs.moo.map_elites import MAPElitesConfig

    constraints = get_constraints(state)
    constraint_dict = constraints.model_dump(exclude_none=True)

    # Create design space
    ds = create_soc_design_space(constraint_dict)

    # Create evaluator with state context
    evaluator = DesignEvaluator(
        design_space=ds,
        base_state=state,
        constraint_bounds=ds.constraint_bounds,
    )

    # Configure based on fast_mode. fast_mode skips the expensive Bayesian
    # layer (~minutes per run on CI) and uses a moderate MAP-Elites budget.
    # The original 704-eval budget was too sparse for the 17-variable SoC
    # design space — tests would find 0 feasible designs. ~3000 evals is the
    # sweet spot: still ~5x faster than full mode, but reliably finds the
    # feasible region for AMR-class constraints.
    fast_mode = (task.metadata or {}).get("fast_mode", False)
    if fast_mode:
        me_config = MAPElitesConfig(
            n_iterations=50,
            batch_size=48,
            initial_population=128,
        )
        config = OptimizationConfig(
            layers="map_elites",
            map_elites=me_config,
            max_workers=4,
        )
    else:
        me_config = MAPElitesConfig(
            n_iterations=100,
            batch_size=64,
            initial_population=256,
        )
        config = OptimizationConfig(
            layers="auto",
            map_elites=me_config,
            max_workers=8,
        )

    # Run optimization
    engine = OptimizationEngine(ds, evaluator, config)

    try:
        result = engine.run()
    finally:
        engine.shutdown()

    # Build backward-compatible pareto_results
    pareto_results = result.to_pareto_results()

    # Build rich moo_results
    moo_results = result.model_dump()

    # Convert Pareto front to ParetoPoint-compatible dicts for API/snapshot
    new_points = _pareto_front_to_points(result.pareto_front)
    knee_index = _find_knee_point_index(result.pareto_front, result.knee_point)
    pareto_results["knee_point_index"] = knee_index

    # Issue #23: merge new points with accumulated frontier from prior runs
    accumulated = state.get("pareto_points", []) or []
    merged_points, new_added, dominated_removed = _merge_pareto_frontiers(accumulated, new_points)

    # Append to frontier history (per-iteration evolution snapshot)
    history = list(state.get("pareto_frontier_history", []) or [])
    iteration = state.get("iteration", 0)
    history.append(
        _build_history_entry(
            iteration=iteration,
            merged=merged_points,
            new_added=new_added,
            dominated_removed=dominated_removed,
            hypervolume=result.hypervolume,
        )
    )

    # Summary
    n_front = len(result.pareto_front)
    knee_info = ""
    if result.knee_point:
        objs = result.knee_point.get("objectives", {})
        knee_info = (
            f", knee: {objs.get('power_watts', '?')}W / "
            f"{objs.get('latency_ms', '?')}ms / "
            f"${objs.get('cost_usd', '?')}"
        )

    summary = (
        f"MOO exploration: {result.total_evaluations} evals, "
        f"{n_front} new Pareto-optimal designs "
        f"(merged: {len(merged_points)} total, +{new_added} new, "
        f"-{dominated_removed} dominated), "
        f"HV={result.hypervolume:.2f}"
        f"{knee_info}"
    )

    return {
        "summary": summary,
        "pareto_results": pareto_results,
        "moo_results": moo_results,
        "_state_updates": {
            "pareto_points": merged_points,
            "pareto_results": pareto_results,
            "pareto_frontier_history": history,
            "moo_results": moo_results,
        },
    }


def swap_explorer(task: TaskNode, state: SoCDesignState) -> dict[str, Any]:
    """Specialist agent: 6-objective SWaP-C design space exploration.

    Extends moo_explorer with system-level weight, volume, and thermal
    constraints via create_swap_design_space() + SWaPCEvaluator.

    Supports task.metadata["fast_mode"] for reduced evaluation budgets.

    Writes to state: pareto_results (backward compat), moo_results, swap_results, system_bom
    """
    from embodied_ai_architect.graphs.moo.design_space import create_swap_design_space
    from embodied_ai_architect.graphs.moo.evaluator import SWaPCEvaluator
    from embodied_ai_architect.graphs.moo.engine import OptimizationConfig, OptimizationEngine
    from embodied_ai_architect.graphs.moo.map_elites import MAPElitesConfig

    constraints = get_constraints(state)
    constraint_dict = constraints.model_dump(exclude_none=True)

    # Create 6-objective design space
    ds = create_swap_design_space(constraint_dict)

    # Thermal config from constraints
    thermal_config = {}
    ambient = constraints.operating_temp_max_c
    if ambient is not None:
        thermal_config["ambient_temp_c"] = ambient

    # Create SWaP-C evaluator
    evaluator = SWaPCEvaluator(
        design_space=ds,
        base_state=state,
        constraint_bounds=ds.constraint_bounds,
        thermal_config=thermal_config or None,
    )

    # Configure based on fast_mode. fast_mode skips the expensive Bayesian
    # layer (~minutes per run on CI) and uses a moderate MAP-Elites budget.
    # The original 704-eval budget was too sparse for the 17-variable SoC
    # design space — tests would find 0 feasible designs. ~3000 evals is the
    # sweet spot: still ~5x faster than full mode, but reliably finds the
    # feasible region for AMR-class constraints.
    fast_mode = (task.metadata or {}).get("fast_mode", False)
    if fast_mode:
        me_config = MAPElitesConfig(
            n_iterations=50,
            batch_size=48,
            initial_population=128,
        )
        config = OptimizationConfig(
            layers="map_elites",
            map_elites=me_config,
            max_workers=4,
        )
    else:
        me_config = MAPElitesConfig(
            n_iterations=100,
            batch_size=64,
            initial_population=256,
        )
        config = OptimizationConfig(
            layers="auto",
            map_elites=me_config,
            max_workers=8,
        )

    # Run optimization
    engine = OptimizationEngine(ds, evaluator, config)

    try:
        result = engine.run()
    finally:
        engine.shutdown()

    # Build results
    pareto_results = result.to_pareto_results()
    moo_results = result.model_dump()

    # Convert Pareto front to ParetoPoint-compatible dicts for API/snapshot
    new_points = _pareto_front_to_points(result.pareto_front)
    knee_index = _find_knee_point_index(result.pareto_front, result.knee_point)
    pareto_results["knee_point_index"] = knee_index

    # Issue #23: merge with accumulated frontier from prior runs
    accumulated = state.get("pareto_points", []) or []
    merged_points, new_added, dominated_removed = _merge_pareto_frontiers(accumulated, new_points)

    history = list(state.get("pareto_frontier_history", []) or [])
    iteration = state.get("iteration", 0)
    history.append(
        _build_history_entry(
            iteration=iteration,
            merged=merged_points,
            new_added=new_added,
            dominated_removed=dominated_removed,
            hypervolume=result.hypervolume,
        )
    )

    # Extract BOM from knee point
    system_bom = {}
    if result.knee_point:
        bom_summary = result.knee_point.get("metadata", {}).get("bom_summary", [])
        system_bom = {"knee_point_bom": bom_summary}

    # Summary
    n_front = len(result.pareto_front)
    knee_info = ""
    if result.knee_point:
        objs = result.knee_point.get("objectives", {})
        knee_info = (
            f", knee: {objs.get('power_watts', '?')}W / "
            f"{objs.get('weight_grams', '?')}g / "
            f"{objs.get('volume_cm3', '?')}cm³"
        )

    summary = (
        f"SWaP-C exploration: {result.total_evaluations} evals, "
        f"{n_front} new Pareto-optimal designs "
        f"(merged: {len(merged_points)} total, +{new_added} new, "
        f"-{dominated_removed} dominated), "
        f"HV={result.hypervolume:.2f}"
        f"{knee_info}"
    )

    return {
        "summary": summary,
        "pareto_results": pareto_results,
        "moo_results": moo_results,
        "swap_results": moo_results,
        "system_bom": system_bom,
        "_state_updates": {
            "pareto_points": merged_points,
            "pareto_results": pareto_results,
            "pareto_frontier_history": history,
            "moo_results": moo_results,
            "swap_assessment": moo_results,
            "system_bom": system_bom,
        },
    }
