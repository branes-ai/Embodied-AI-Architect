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
    # ----------------------------------------------------------------
    # KPU micro-architecture strategies (issue #32). Only applicable
    # when state["kpu_config"] is populated, gated in design_optimizer.
    # ----------------------------------------------------------------
    {
        "name": "reduce_systolic_array",
        "description": "Reduce systolic array dimensions (e.g. 16x16 → 12x12)",
        "applicable_when": ["area", "power"],
        "power_reduction_factor": 0.25,
        "latency_reduction_factor": -0.15,  # latency increases
        "accuracy_impact": "none",
        "applies_to": "kpu_config",
    },
    {
        "name": "upgrade_dram_technology",
        "description": "Upgrade DRAM tech (LPDDR4X → LPDDR5 → HBM2E) for higher bandwidth",
        "applicable_when": ["latency"],
        "power_reduction_factor": -0.05,  # higher BW → slightly more power
        "latency_reduction_factor": 0.20,
        "accuracy_impact": "none",
        "applies_to": "kpu_config",
    },
    {
        "name": "add_sram_banks",
        "description": "Add L2/L3 SRAM banks (more bandwidth, more area)",
        "applicable_when": ["latency"],
        "power_reduction_factor": -0.03,
        "latency_reduction_factor": 0.15,
        "accuracy_impact": "none",
        "applies_to": "kpu_config",
    },
    {
        "name": "widen_noc",
        "description": "Double NoC link width (more bandwidth, more area)",
        "applicable_when": ["latency"],
        "power_reduction_factor": -0.05,
        "latency_reduction_factor": 0.18,
        "accuracy_impact": "none",
        "applies_to": "kpu_config",
    },
    {
        "name": "reduce_compute_tiles",
        "description": "Drop one row/column from the compute-tile grid",
        "applicable_when": ["area", "power"],
        "power_reduction_factor": 0.20,
        "latency_reduction_factor": -0.20,
        "accuracy_impact": "none",
        "applies_to": "kpu_config",
    },
    {
        "name": "clock_scale_kpu",
        "description": "Reduce KPU compute-tile frequency for lower power",
        "applicable_when": ["power"],
        "power_reduction_factor": 0.18,
        "latency_reduction_factor": -0.12,  # latency increases
        "accuracy_impact": "none",
        "applies_to": "kpu_config",
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
    # KPU micro-architecture strategies (issue #32). Tie each to the
    # closest MOO design space variable when one exists, so the MOO-aware
    # selector can boost it via per-variable sensitivity.
    "reduce_systolic_array": "systolic_array_size",
    "upgrade_dram_technology": "dram_bandwidth_gbps",
    "add_sram_banks": "sram_size_kb",
    "widen_noc": "noc_link_width_bits",
    "reduce_compute_tiles": "num_compute_tiles",
    "clock_scale_kpu": "clock_mhz",
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

    # KPU strategies (issue #32) only make sense when state has a kpu_config
    # to mutate. On non-RTL pipelines they'd silently no-op, so filter them
    # out at the source instead of letting them be selected.
    has_kpu_config = bool(state.get("kpu_config"))

    # Filter applicable strategies
    applicable = []
    for strat in OPTIMIZATION_STRATEGIES:
        if strat["name"] in already_tried:
            continue
        if strat.get("applies_to") == "kpu_config" and not has_kpu_config:
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


# ---------------------------------------------------------------------------
# KPU strategy application (issue #32)
# ---------------------------------------------------------------------------

# DRAM technology upgrade chain. Each step bumps both technology label and
# the per-channel bandwidth (apply_kpu_overrides forwards both fields).
_DRAM_UPGRADE_CHAIN: list[tuple[str, float]] = [
    ("LPDDR4X", 6.4),
    ("LPDDR5", 12.8),
    ("HBM2E", 25.6),
]


def _next_dram_step(current_tech: str) -> tuple[str, float] | None:
    """Return the next DRAM technology in the upgrade chain, or None at top."""
    for i, (tech, _bw) in enumerate(_DRAM_UPGRADE_CHAIN):
        if tech == current_tech and i + 1 < len(_DRAM_UPGRADE_CHAIN):
            return _DRAM_UPGRADE_CHAIN[i + 1]
    return None


def _apply_kpu_strategy(
    strategy: dict[str, Any],
    state: SoCDesignState,
) -> dict[str, Any]:
    """Apply a KPU-targeted strategy by mutating state['kpu_config'].

    Each strategy reduces or grows specific KPU micro-architecture parameters
    via the dotted-path override helper from issue #29. After mutation, the
    stale floorplan_estimate / bandwidth_match are cleared so the next
    dispatch iteration re-validates against the new config.

    Returns a state-update dict (always includes 'kpu_config' on success).
    """
    from embodied_ai_architect.graphs.kpu_config import (
        KPUMicroArchConfig,
        apply_kpu_overrides,
    )

    kpu_dict = state.get("kpu_config") or {}
    if not kpu_dict:
        return {}

    config = KPUMicroArchConfig(**kpu_dict)
    name = strategy["name"]
    overrides: dict[str, Any] = {}

    if name == "reduce_systolic_array":
        new_rows = max(4, config.compute_tile.array_rows - 4)
        new_cols = max(4, config.compute_tile.array_cols - 4)
        overrides["compute_tile.array_rows"] = new_rows
        overrides["compute_tile.array_cols"] = new_cols

    elif name == "upgrade_dram_technology":
        step = _next_dram_step(config.dram.technology)
        if step is not None:
            new_tech, new_bw = step
            overrides["dram.technology"] = new_tech
            overrides["dram.bandwidth_per_channel_gbps"] = new_bw

    elif name == "add_sram_banks":
        overrides["compute_tile.l2_num_banks"] = config.compute_tile.l2_num_banks + 2
        overrides["memory_tile.l3_num_banks"] = config.memory_tile.l3_num_banks + 1

    elif name == "widen_noc":
        new_width = min(1024, config.noc.link_width_bits * 2)
        overrides["noc.link_width_bits"] = new_width

    elif name == "reduce_compute_tiles":
        # Drop a row first, fall back to a column when rows hit the floor.
        if config.array_rows > 2:
            overrides["array_rows"] = config.array_rows - 1
        elif config.array_cols > 2:
            overrides["array_cols"] = config.array_cols - 1

    elif name == "clock_scale_kpu":
        new_freq = max(100.0, config.compute_tile.frequency_mhz * 0.8)
        overrides["compute_tile.frequency_mhz"] = new_freq

    if not overrides:
        # Strategy was applicable in name but couldn't change anything
        # (already at the floor) — return empty so the caller skips it.
        return {}

    new_config = apply_kpu_overrides(config, overrides)
    return {
        "kpu_config": new_config.model_dump(),
        # Clear stale validator output — next dispatch iteration regenerates them.
        "floorplan_estimate": {},
        "bandwidth_match": {},
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

    elif strategy["applies_to"] == "kpu_config":
        # Issue #32: KPU micro-architecture strategies. Mutate kpu_config
        # via the dotted-path overrides helper from issue #29 and clear the
        # stale floorplan/bandwidth so the next dispatch iteration re-runs
        # the validators against the new config.
        kpu_updates = _apply_kpu_strategy(strategy, state)
        updates.update(kpu_updates)

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
