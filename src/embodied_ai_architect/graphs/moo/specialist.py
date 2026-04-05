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

    # Configure based on fast_mode
    fast_mode = (task.metadata or {}).get("fast_mode", False)
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
    pareto_points = _pareto_front_to_points(result.pareto_front)
    knee_index = _find_knee_point_index(result.pareto_front, result.knee_point)
    pareto_results["knee_point_index"] = knee_index

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
        f"{n_front} Pareto-optimal designs, "
        f"HV={result.hypervolume:.2f}"
        f"{knee_info}"
    )

    return {
        "summary": summary,
        "pareto_results": pareto_results,
        "moo_results": moo_results,
        "_state_updates": {
            "pareto_points": pareto_points,
            "pareto_results": pareto_results,
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

    # Configure based on fast_mode
    fast_mode = (task.metadata or {}).get("fast_mode", False)
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
    pareto_points = _pareto_front_to_points(result.pareto_front)
    knee_index = _find_knee_point_index(result.pareto_front, result.knee_point)
    pareto_results["knee_point_index"] = knee_index

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
        f"{n_front} Pareto-optimal designs, "
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
            "pareto_points": pareto_points,
            "pareto_results": pareto_results,
            "moo_results": moo_results,
            "swap_assessment": moo_results,
            "system_bom": system_bom,
        },
    }
