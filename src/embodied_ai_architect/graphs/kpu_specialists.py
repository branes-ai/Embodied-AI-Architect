"""Specialist agents for KPU micro-architecture configuration and validation.

Four specialists that form the KPU inner loop:
- kpu_configurator: Size KPU components for workload
- floorplan_validator: Check 2D die area feasibility
- bandwidth_validator: Verify bandwidth matching through hierarchy
- kpu_optimizer: Adjust config to fix violations

Usage:
    from embodied_ai_architect.graphs.kpu_specialists import kpu_configurator
    result = kpu_configurator(task, state)
"""

from __future__ import annotations

import logging
from typing import Any

from embodied_ai_architect.graphs.soc_state import (
    SoCDesignState,
    get_constraints,
)
from embodied_ai_architect.graphs.task_graph import TaskNode

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# KPU Configurator
# ---------------------------------------------------------------------------


def kpu_configurator(task: TaskNode, state: SoCDesignState) -> dict[str, Any]:
    """Size KPU micro-architecture components for workload.

    Reads workload profile and constraints, generates initial KPU config
    using heuristic sizing. If the architect provided overrides at plan
    review time (issue #29), they are applied on top of the heuristic
    output — architect's choices win over the auto-sizer.

    Writes to state: kpu_config
    """
    from embodied_ai_architect.graphs.kpu_config import (
        apply_kpu_overrides,
        create_kpu_config,
    )

    constraints = get_constraints(state)
    workload = state.get("workload_profile", {})
    use_case = state.get("use_case", "")

    constraints_dict = constraints.model_dump(exclude_none=True)
    config = create_kpu_config(use_case, constraints_dict, workload)

    # Issue #29: respect architect overrides from plan review.
    overrides = state.get("kpu_config_overrides", {}) or {}
    if overrides:
        config = apply_kpu_overrides(config, overrides)

    config_dict = config.model_dump()

    summary = (
        f"Configured KPU '{config.name}' at {config.process_nm}nm: "
        f"{config.array_rows}x{config.array_cols} checkerboard "
        f"({config.num_compute_tiles} compute + {config.num_memory_tiles} memory tiles), "
        f"{config.peak_tops_int8:.2f} TOPS INT8, "
        f"{config.total_sram_bytes / (1024 * 1024):.1f}MB SRAM"
    )
    if overrides:
        summary += f" (with {len(overrides)} architect override(s))"

    return {
        "summary": summary,
        "kpu_config": config_dict,
        "_state_updates": {"kpu_config": config_dict},
    }


# ---------------------------------------------------------------------------
# Floorplan Validator
# ---------------------------------------------------------------------------


def floorplan_validator(task: TaskNode, state: SoCDesignState) -> dict[str, Any]:
    """Check 2D die area feasibility at target process node.

    Reads kpu_config from state, runs floorplan estimation,
    reports pitch matching and area verdict.

    Writes to state: floorplan_estimate
    """
    from embodied_ai_architect.graphs.floorplan import estimate_floorplan
    from embodied_ai_architect.graphs.kpu_config import KPUMicroArchConfig

    kpu_dict = state.get("kpu_config", {})
    if not kpu_dict:
        return {
            "summary": "No KPU config available — skipping floorplan validation",
            "verdict": "SKIP",
        }

    config = KPUMicroArchConfig(**kpu_dict)
    constraints = get_constraints(state)
    max_area = constraints.max_area_mm2 or 100.0

    fp = estimate_floorplan(config, max_die_area_mm2=max_area)
    fp_dict = fp.model_dump()

    verdict = "PASS" if fp.feasible else "FAIL"

    return {
        "summary": (
            f"Floorplan {verdict}: "
            f"compute tile {fp.compute_tile.width_mm:.2f}x{fp.compute_tile.height_mm:.2f}mm, "
            f"memory tile {fp.memory_tile.width_mm:.2f}x{fp.memory_tile.height_mm:.2f}mm, "
            f"pitch ratio W={fp.pitch_ratio_width:.2f} H={fp.pitch_ratio_height:.2f}, "
            f"die {fp.total_area_mm2:.1f}mm² "
            f"({'< ' + str(int(max_area)) + 'mm² budget' if fp.feasible else 'EXCEEDS budget'})"
        ),
        "verdict": verdict,
        "floorplan_estimate": fp_dict,
        "_state_updates": {"floorplan_estimate": fp_dict},
    }


# ---------------------------------------------------------------------------
# Bandwidth Validator
# ---------------------------------------------------------------------------


def bandwidth_validator(task: TaskNode, state: SoCDesignState) -> dict[str, Any]:
    """Verify ingress/egress bandwidth matching through memory hierarchy.

    Reads kpu_config and workload_profile from state.

    Writes to state: bandwidth_match
    """
    from embodied_ai_architect.graphs.bandwidth import check_bandwidth_match
    from embodied_ai_architect.graphs.kpu_config import KPUMicroArchConfig

    kpu_dict = state.get("kpu_config", {})
    if not kpu_dict:
        return {
            "summary": "No KPU config available — skipping bandwidth validation",
            "verdict": "SKIP",
        }

    config = KPUMicroArchConfig(**kpu_dict)
    workload = state.get("workload_profile", {})

    result = check_bandwidth_match(config, workload)
    result_dict = result.model_dump()

    verdict = "PASS" if result.balanced else "FAIL"

    link_summary = ", ".join(f"{link.name}: {link.utilization:.0%}" for link in result.links)

    return {
        "summary": (
            f"Bandwidth {verdict}: peak utilization {result.peak_utilization:.0%} "
            f"({link_summary})"
        ),
        "verdict": verdict,
        "bandwidth_match": result_dict,
        "_state_updates": {"bandwidth_match": result_dict},
    }


# ---------------------------------------------------------------------------
# KPU Optimizer
# ---------------------------------------------------------------------------


def kpu_optimizer(task: TaskNode, state: SoCDesignState) -> dict[str, Any]:
    """Adjust KPU config to fix floorplan/bandwidth violations.

    Reads floorplan_estimate and bandwidth_match from state,
    applies targeted adjustments to the kpu_config.

    Writes to state: kpu_config (updated)
    """
    from embodied_ai_architect.graphs.kpu_config import KPUMicroArchConfig

    kpu_dict = state.get("kpu_config", {})
    if not kpu_dict:
        return {"summary": "No KPU config to optimize", "changes": []}

    config = KPUMicroArchConfig(**kpu_dict)
    fp = state.get("floorplan_estimate", {})
    bw = state.get("bandwidth_match", {})

    changes = []

    # --- Fix pitch mismatch ---
    if fp and not fp.get("pitch_matched", True):
        ratio_w = fp.get("pitch_ratio_width", 1.0)
        ratio_h = fp.get("pitch_ratio_height", 1.0)

        if ratio_w > 1.15:
            # Compute tile too wide — reduce array cols or L2
            new_cols = max(4, config.compute_tile.array_cols - 4)
            config.compute_tile.array_cols = new_cols
            changes.append(f"Reduced systolic array cols to {new_cols} (pitch match)")
        elif ratio_w < 0.85:
            # Memory tile too wide — increase L3 banks
            new_banks = config.memory_tile.l3_num_banks + 2
            config.memory_tile.l3_num_banks = new_banks
            changes.append(f"Increased L3 banks to {new_banks} (pitch match)")

        if ratio_h > 1.15:
            # Compute tile too tall — reduce vector lanes
            new_lanes = max(4, config.compute_tile.vector_lanes - 4)
            config.compute_tile.vector_lanes = new_lanes
            changes.append(f"Reduced vector lanes to {new_lanes} (pitch match)")
        elif ratio_h < 0.85:
            # Memory tile too tall — add block movers
            config.memory_tile.num_block_movers += 1
            changes.append(f"Added block mover (now {config.memory_tile.num_block_movers})")

    # --- Fix die area ---
    if (
        fp
        and not fp.get("feasible", True)
        and fp.get("total_area_mm2", 0) > fp.get("max_die_area_mm2", 100.0)
    ):
        if config.array_rows > 2:
            config.array_rows -= 1
            changes.append(f"Reduced array_rows to {config.array_rows} (area)")
        if config.array_cols > 2:
            config.array_cols -= 1
            changes.append(f"Reduced array_cols to {config.array_cols} (area)")
        # Also shrink SRAM
        config.compute_tile.l2_size_bytes = max(64 * 1024, config.compute_tile.l2_size_bytes // 2)
        changes.append(f"Halved L2 to {config.compute_tile.l2_size_bytes // 1024}KB (area)")

    # --- Fix bandwidth ---
    if bw and not bw.get("balanced", True):
        bottleneck = bw.get("bottleneck_link", "")

        if "dram" in bottleneck:
            if config.dram.technology == "LPDDR4X":
                config.dram.num_controllers += 1
                changes.append(f"Added DRAM controller (now {config.dram.num_controllers})")
            elif config.dram.technology != "HBM2E":
                config.dram.technology = "LPDDR5"
                config.dram.bandwidth_per_channel_gbps = 12.8
                changes.append("Upgraded DRAM to LPDDR5")

        elif "l3" in bottleneck or "noc" in bottleneck:
            config.noc.link_width_bits = min(512, config.noc.link_width_bits * 2)
            changes.append(f"Widened NoC links to {config.noc.link_width_bits} bits")

        elif "l2" in bottleneck:
            config.compute_tile.l2_num_banks += 2
            changes.append(f"Added L2 banks (now {config.compute_tile.l2_num_banks})")

        elif "l1" in bottleneck:
            config.compute_tile.l1_num_banks += 2
            changes.append(f"Added L1 banks (now {config.compute_tile.l1_num_banks})")

    config_dict = config.model_dump()

    if not changes:
        changes.append("No adjustments needed")

    return {
        "summary": f"KPU optimizer applied {len(changes)} changes: {'; '.join(changes)}",
        "changes": changes,
        "kpu_config": config_dict,
        "_state_updates": {"kpu_config": config_dict},
    }


# ---------------------------------------------------------------------------
# RTL → KPU area feedback (issue #31)
# ---------------------------------------------------------------------------


def rtl_area_feedback(task: TaskNode, state: SoCDesignState) -> dict[str, Any]:
    """Feed RTL synthesis area back into the KPU sizing loop.

    Schedule this task after `rtl_ppa_assessor`. When `state["rtl_area_feedback"]`
    is True and synthesis area exceeds the floorplan estimate by more than 10%,
    re-runs the KPU optimizer with the synthesis area as a tightened constraint
    and re-validates floorplan + bandwidth (bounded at 3 iterations).

    The actual re-sizing logic lives in `kpu_loop.apply_rtl_area_feedback()` —
    this specialist is just the dispatcher-facing wrapper.

    Writes to state: kpu_config, floorplan_estimate, bandwidth_match,
    kpu_optimization_history.
    """
    from embodied_ai_architect.graphs.kpu_loop import apply_rtl_area_feedback

    if not state.get("rtl_area_feedback", False):
        return {
            "summary": "RTL area feedback skipped (rtl_area_feedback=False)",
            "verdict": "SKIP",
        }

    if not state.get("rtl_enabled", False):
        return {
            "summary": "RTL area feedback skipped (rtl_enabled=False)",
            "verdict": "SKIP",
        }

    metadata = task.metadata or {}
    area_tolerance = float(metadata.get("area_tolerance", 1.1))
    max_iterations = int(metadata.get("max_iterations", 3))

    updates = apply_rtl_area_feedback(
        state,
        area_tolerance=area_tolerance,
        max_iterations=max_iterations,
    )

    if not updates:
        return {
            "summary": "RTL area feedback: insufficient state (need synthesis + floorplan)",
            "verdict": "SKIP",
        }

    summary = updates.pop("rtl_area_feedback_summary", "RTL area feedback applied")

    return {
        "summary": summary,
        "verdict": "PASS",
        "_state_updates": updates,
    }
