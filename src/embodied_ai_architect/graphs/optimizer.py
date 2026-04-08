"""Design optimizer specialist for iterative SoC refinement.

Uses a deterministic strategy catalog (no LLM) to fix failing PPA constraints.
Each strategy has applicability conditions and estimated reduction factors.
The optimizer reads failing verdicts from ppa_metrics, filters out already-tried
strategies from working memory, selects the best applicable one, and applies it
by modifying workload_profile/architecture/ip_blocks via _state_updates.

Usage:
    from embodied_ai_architect.graphs.optimizer import design_optimizer

    result = design_optimizer(task, state)
    # result["_state_updates"] contains modified workload_profile, etc.
"""

from __future__ import annotations

import logging
from typing import Any

from embodied_ai_architect.graphs.memory import WorkingMemoryStore
from embodied_ai_architect.graphs.soc_state import (
    SoCDesignState,
    get_constraints,
)
from embodied_ai_architect.graphs.task_graph import TaskNode

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Optimization strategy catalog
# ---------------------------------------------------------------------------

OPTIMIZATION_STRATEGIES: list[dict[str, Any]] = [
    {
        "name": "quantize_int8",
        "description": "Quantize model weights and activations from FP16/FP32 to INT8",
        "applicable_when": ["power"],
        "power_reduction_factor": 0.20,  # ~20% power reduction
        "latency_reduction_factor": 0.15,  # ~15% latency reduction
        "accuracy_impact": "minor",
        "applies_to": "workload_profile",
    },
    {
        "name": "reduce_resolution",
        "description": "Reduce input resolution (e.g. 640x640 -> 480x480)",
        "applicable_when": ["power", "latency"],
        "power_reduction_factor": 0.25,  # ~25% power reduction (quadratic in resolution)
        "latency_reduction_factor": 0.30,  # ~30% latency reduction
        "accuracy_impact": "moderate",
        "applies_to": "workload_profile",
    },
    {
        "name": "clock_scaling",
        "description": "Reduce accelerator clock frequency for power savings",
        "applicable_when": ["power"],
        "power_reduction_factor": 0.15,  # ~15% power reduction
        "latency_reduction_factor": -0.10,  # latency increases ~10%
        "accuracy_impact": "none",
        "applies_to": "ip_blocks",
    },
    {
        "name": "model_pruning",
        "description": "Structured pruning to reduce model size by ~30%",
        "applicable_when": ["power", "latency"],
        "power_reduction_factor": 0.18,  # ~18% power reduction
        "latency_reduction_factor": 0.20,  # ~20% latency reduction
        "accuracy_impact": "moderate",
        "applies_to": "workload_profile",
    },
    {
        "name": "smaller_model",
        "description": "Switch to a smaller model variant (e.g. YOLOv8n -> YOLOv8p)",
        "applicable_when": ["power", "latency"],
        "power_reduction_factor": 0.35,  # ~35% power reduction
        "latency_reduction_factor": 0.40,  # ~40% latency reduction
        "accuracy_impact": "significant",
        "applies_to": "workload_profile",
    },
    {
        "name": "shrink_process_node",
        "description": "Shrink to next smaller process node (lower power/area, higher NRE)",
        "applicable_when": ["power", "latency", "area"],
        "power_reduction_factor": 0.0,  # PPA is recomputed from physics
        "latency_reduction_factor": 0.0,
        "accuracy_impact": "none",
        "applies_to": "constraints",
    },
    {
        "name": "grow_process_node",
        "description": "Grow to next larger process node (lower cost via cheaper wafers/NRE)",
        "applicable_when": ["cost"],
        "power_reduction_factor": 0.0,
        "latency_reduction_factor": 0.0,
        "accuracy_impact": "none",
        "applies_to": "constraints",
    },
]


# Maps each optimization strategy to the MOO design space variable it directly
# turns. Used by the MOO-aware selector (issue #25) to consult per-variable
# sensitivity from the BO layer when picking which knob to turn.
#
# Strategies that modify the workload (quantize, prune, smaller model, lower
# resolution) don't map to a single MOO variable — they get a neutral boost
# of 1.0 in the scoring so they remain candidates without skewing the choice.
STRATEGY_VARIABLE_MAP: dict[str, str | None] = {
    "quantize_int8": None,  # workload-targeting
    "reduce_resolution": None,  # workload-targeting
    "model_pruning": None,  # workload-targeting
    "smaller_model": None,  # workload-targeting
    "clock_scaling": "clock_mhz",  # directly targets clock_mhz in MOO space
    "shrink_process_node": "process_nm",  # targets process_nm
    "grow_process_node": "process_nm",  # targets process_nm
}

# Maps a failing constraint name (PPA verdict key) to the MOO objective key
# used in the sensitivity dict. (e.g. "power" → "power_watts")
_FAILING_TO_OBJECTIVE: dict[str, str] = {
    "power": "power_watts",
    "latency": "latency_ms",
    "area": "area_mm2",
    "cost": "cost_usd",
}


def _moo_aware_score(
    strategy: dict[str, Any],
    failing: list[str],
    sensitivity: dict[str, dict[str, float]],
) -> tuple[float, str]:
    """Score a strategy using MOO sensitivity data when available.

    Returns (score, rationale_fragment). Higher score = better choice.

    The score combines:
      base = static reduction factor for the failing constraint
      boost = 1.0 + sensitivity[strategy_var][failing_obj] when the
              strategy targets a MOO variable, else 1.0 (neutral)

    This way, two strategies with similar static factors get differentiated
    by which one turns the highest-impact MOO variable.
    """
    # Pick the primary failing constraint to score against
    primary = failing[0] if failing else "power"

    # Static factor for the primary failing constraint
    if primary == "power":
        base = strategy.get("power_reduction_factor", 0.0)
    elif primary == "latency":
        base = strategy.get("latency_reduction_factor", 0.0)
    else:
        # area/cost: use the higher of power/latency factor as a proxy
        base = max(
            strategy.get("power_reduction_factor", 0.0),
            strategy.get("latency_reduction_factor", 0.0),
        )

    # Compute the sensitivity boost
    boost = 1.0
    rationale_frag = "static reduction factor only"

    var = STRATEGY_VARIABLE_MAP.get(strategy["name"])
    obj = _FAILING_TO_OBJECTIVE.get(primary)

    if var and obj and sensitivity:
        var_impacts = sensitivity.get(var, {})
        impact = var_impacts.get(obj)
        if impact is not None:
            boost = 1.0 + float(impact)
            rationale_frag = f"sensitivity[{var}][{obj}]={impact:.2f}, boost={boost:.2f}x"

    return base * boost, rationale_frag


def _select_strategy_with_moo(
    applicable: list[dict[str, Any]],
    failing: list[str],
    state: SoCDesignState,
) -> tuple[dict[str, Any], str]:
    """Pick the best strategy, consulting MOO sensitivity when available.

    Returns (selected_strategy, structured_rationale).

    Backward-compatible: when sensitivity data is empty (MOO didn't run, or
    only MAP-Elites ran without BO), this falls back to the original greedy
    logic — sort by reduction factor for the primary failing constraint.
    """
    moo_results = state.get("moo_results", {}) or {}
    raw_sensitivity = moo_results.get("sensitivity", {})

    # Normalize via the shared helper from optimization_review (issue #24)
    sensitivity: dict[str, dict[str, float]] = {}
    if raw_sensitivity:
        try:
            from embodied_ai_architect.graphs.optimization_review import (
                normalize_sensitivity,
            )

            sensitivity = normalize_sensitivity(raw_sensitivity)
        except Exception:
            sensitivity = {}

    if not sensitivity:
        # Greedy fallback (original behavior)
        if "power" in failing:
            applicable.sort(key=lambda s: s["power_reduction_factor"], reverse=True)
        elif "latency" in failing:
            applicable.sort(key=lambda s: s["latency_reduction_factor"], reverse=True)
        else:
            applicable.sort(key=lambda s: s["power_reduction_factor"], reverse=True)
        selected = applicable[0]
        rationale = (
            f"Selected '{selected['name']}' by greedy reduction factor "
            f"(MOO sensitivity unavailable). Failing: {', '.join(failing)}."
        )
        return selected, rationale

    # MOO-aware: score each strategy and pick the highest
    scored: list[tuple[float, str, dict[str, Any]]] = []
    for strat in applicable:
        score, frag = _moo_aware_score(strat, failing, sensitivity)
        scored.append((score, frag, strat))
    scored.sort(key=lambda x: x[0], reverse=True)

    best_score, best_frag, selected = scored[0]
    rationale = (
        f"Selected '{selected['name']}' (score={best_score:.3f}) using MOO "
        f"sensitivity. Reason: {best_frag}. Failing: {', '.join(failing)}."
    )
    return selected, rationale


def design_optimizer(task: TaskNode, state: SoCDesignState) -> dict[str, Any]:
    """Apply an optimization strategy to fix failing PPA constraints.

    Reads failing verdicts from ppa_metrics, filters applicable strategies
    excluding already-tried ones (from working memory), selects best,
    and applies by modifying state artifacts via _state_updates.

    Args:
        task: Current task node.
        state: Current SoC design state with ppa_metrics containing verdicts.

    Returns:
        Result dict with _state_updates for modified artifacts.
    """
    ppa = state.get("ppa_metrics", {})
    verdicts = ppa.get("verdicts", {})
    failing = [k for k, v in verdicts.items() if v == "FAIL"]

    if not failing:
        return {
            "summary": "No failing constraints — no optimization needed",
            "strategy": None,
            "applied": False,
        }

    # Load working memory to see what was already tried
    wm_data = state.get("working_memory", {})
    store = WorkingMemoryStore(**wm_data) if wm_data else WorkingMemoryStore()
    already_tried = store.get_tried_descriptions("design_optimizer")

    # Filter applicable strategies
    applicable = []
    for strat in OPTIMIZATION_STRATEGIES:
        if strat["name"] in already_tried:
            continue
        if any(f in strat["applicable_when"] for f in failing):
            applicable.append(strat)

    if not applicable:
        return {
            "summary": f"No untried strategies for failing constraints: {failing}",
            "strategy": None,
            "applied": False,
            "failing_constraints": failing,
            "already_tried": already_tried,
        }

    # Select best strategy. When MOO has run and produced sensitivity data,
    # use it to pick the strategy that targets the highest-impact variable
    # for the failing constraint (issue #25). Otherwise fall back to the
    # greedy reduction-factor heuristic (backward compatible).
    selected, selection_rationale = _select_strategy_with_moo(applicable, failing, state)
    logger.info(
        "Optimizer selected strategy '%s' for failing constraints %s — %s",
        selected["name"],
        failing,
        selection_rationale,
    )

    # Apply strategy
    state_updates = _apply_strategy(selected, state)

    # Record attempt in working memory — use dynamic name for process strategies
    # so the optimizer can apply them multiple times (e.g. 28nm -> 22nm -> 16nm)
    iteration = state.get("iteration", 0)
    strategy_key = selected["name"]
    if selected["applies_to"] == "constraints":
        current_nm = get_constraints(state).target_process_nm or 28
        strategy_key = f"{selected['name']}_{current_nm}"
    store.record_attempt(
        agent_name="design_optimizer",
        description=strategy_key,
        outcome=f"Applied {selected['description']} at iteration {iteration}",
        iteration=iteration,
    )
    store.record_decision(
        "design_optimizer", f"Applied {selected['name']}: {selected['description']}"
    )
    state_updates["working_memory"] = store.model_dump()

    # Surface the selection rationale on the state so the optimization
    # review snapshot picks it up (issue #25). This replaces the generic
    # rationale built in build_optimization_review_snapshot when MOO data
    # is the actual driver of the choice.
    state_updates["last_strategy_rationale"] = selection_rationale

    return {
        "summary": f"Applied optimization: {selected['description']}",
        "strategy": selected["name"],
        "applied": True,
        "failing_constraints": failing,
        "power_reduction_factor": selected["power_reduction_factor"],
        "latency_reduction_factor": selected["latency_reduction_factor"],
        "selection_rationale": selection_rationale,
        "_state_updates": state_updates,
    }


def _apply_strategy(strategy: dict[str, Any], state: SoCDesignState) -> dict[str, Any]:
    """Apply a strategy by modifying copies of state artifacts.

    Returns a dict of state keys to update.
    """
    updates: dict[str, Any] = {}
    ppa = dict(state.get("ppa_metrics", {}))

    power_factor = strategy["power_reduction_factor"]
    latency_factor = strategy["latency_reduction_factor"]

    if strategy["applies_to"] == "workload_profile":
        workload = dict(state.get("workload_profile", {}))

        # Reduce estimated compute requirements
        if "total_estimated_gflops" in workload:
            workload["total_estimated_gflops"] = round(
                workload["total_estimated_gflops"] * (1 - power_factor), 2
            )
        if "estimated_gflops" in workload:
            workload["estimated_gflops"] = round(
                workload["estimated_gflops"] * (1 - power_factor), 2
            )

        # Scale sub-workloads
        for w in workload.get("workloads", []):
            if "estimated_gflops" in w:
                w["estimated_gflops"] = round(w["estimated_gflops"] * (1 - power_factor), 2)

        # Record optimization applied
        optimizations = workload.get("optimizations_applied", [])
        optimizations.append(strategy["name"])
        workload["optimizations_applied"] = optimizations

        updates["workload_profile"] = workload

    elif strategy["applies_to"] == "ip_blocks":
        ip_blocks = [dict(b) for b in state.get("ip_blocks", [])]

        for block in ip_blocks:
            if block.get("type") in ("kpu", "gpu", "npu", "tpu", "accelerator"):
                config = dict(block.get("config", {}))
                if "frequency_mhz" in config:
                    config["frequency_mhz"] = int(config["frequency_mhz"] * (1 - power_factor))
                block["config"] = config

        updates["ip_blocks"] = ip_blocks

    elif strategy["applies_to"] == "constraints":
        constraints = get_constraints(state)
        current_nm = constraints.target_process_nm or 28
        from embodied_ai_architect.graphs.technology import get_adjacent_nodes

        adjacent = get_adjacent_nodes(current_nm)

        if strategy["name"] == "shrink_process_node":
            new_nm = adjacent["smaller"]
        else:  # grow_process_node
            new_nm = adjacent["larger"]

        if new_nm is not None:
            new_constraints = constraints.model_dump()
            new_constraints["target_process_nm"] = new_nm
            updates["constraints"] = new_constraints

    # Adjust PPA estimates based on reduction factors
    if ppa.get("power_watts") is not None:
        ppa["power_watts"] = round(ppa["power_watts"] * (1 - power_factor), 2)
    if ppa.get("latency_ms") is not None:
        ppa["latency_ms"] = round(ppa["latency_ms"] * (1 - latency_factor), 2)

    # Clear verdicts — they'll be recomputed by ppa_assessor
    ppa["verdicts"] = {}
    ppa["bottlenecks"] = []
    ppa["suggestions"] = []
    updates["ppa_metrics"] = ppa

    return updates
