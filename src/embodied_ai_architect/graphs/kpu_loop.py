"""KPU validation loop: configure → floorplan check → bandwidth check → adjust.

Iterates on KPU micro-architecture configuration until both floorplan
(pitch matching + area) and bandwidth checks pass.

Usage:
    from embodied_ai_architect.graphs.kpu_loop import run_kpu_loop

    result = run_kpu_loop(workload, constraints, use_case="delivery_drone")
    assert result.success
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class KPULoopConfig:
    """Configuration for the KPU validation loop."""

    max_iterations: int = 10
    max_die_area_mm2: float = 100.0
    bandwidth_threshold: float = 0.85
    pitch_tolerance: float = 0.15


@dataclass
class KPULoopResult:
    """Result of KPU validation loop."""

    success: bool
    config: dict = field(default_factory=dict)
    floorplan: dict = field(default_factory=dict)
    bandwidth: dict = field(default_factory=dict)
    iterations_used: int = 0
    history: list[dict] = field(default_factory=list)


def run_kpu_loop(
    workload: dict[str, Any],
    constraints: dict[str, Any],
    use_case: str = "",
    loop_config: Optional[KPULoopConfig] = None,
) -> KPULoopResult:
    """Configure → floorplan check → bandwidth check → [adjust|accept].

    Args:
        workload: Workload profile dict.
        constraints: Design constraints dict.
        use_case: Application type.
        loop_config: Loop parameters.

    Returns:
        KPULoopResult with final config, floorplan, and bandwidth results.
    """
    from embodied_ai_architect.graphs.bandwidth import check_bandwidth_match
    from embodied_ai_architect.graphs.floorplan import estimate_floorplan
    from embodied_ai_architect.graphs.kpu_config import create_kpu_config

    if loop_config is None:
        loop_config = KPULoopConfig()

    # Use area constraint if available
    max_area = constraints.get("max_area_mm2", loop_config.max_die_area_mm2)

    # Step 1: Generate initial config
    config = create_kpu_config(use_case, constraints, workload)
    history: list[dict] = []

    for iteration in range(loop_config.max_iterations):
        logger.info("KPU loop iteration %d", iteration)

        # Step 2: Floorplan check
        fp = estimate_floorplan(
            config,
            max_die_area_mm2=max_area,
            pitch_tolerance=loop_config.pitch_tolerance,
        )

        # Step 3: Bandwidth check
        bw = check_bandwidth_match(
            config,
            workload,
            bottleneck_threshold=loop_config.bandwidth_threshold,
        )

        # Record history
        history.append(
            {
                "iteration": iteration,
                "config_name": config.name,
                "floorplan_feasible": fp.feasible,
                "pitch_matched": fp.pitch_matched,
                "total_area_mm2": fp.total_area_mm2,
                "bandwidth_balanced": bw.balanced,
                "peak_utilization": bw.peak_utilization,
            }
        )

        # Step 4: Check if both pass
        if fp.feasible and bw.balanced:
            logger.info("KPU loop converged in %d iterations", iteration + 1)
            return KPULoopResult(
                success=True,
                config=config.model_dump(),
                floorplan=fp.model_dump(),
                bandwidth=bw.model_dump(),
                iterations_used=iteration + 1,
                history=history,
            )

        # Step 5: Apply adjustments
        config = _apply_adjustments(config, fp, bw, max_area)

    # Did not converge
    logger.warning("KPU loop did not converge in %d iterations", loop_config.max_iterations)
    fp = estimate_floorplan(config, max_die_area_mm2=max_area)
    bw = check_bandwidth_match(config, workload)

    return KPULoopResult(
        success=False,
        config=config.model_dump(),
        floorplan=fp.model_dump(),
        bandwidth=bw.model_dump(),
        iterations_used=loop_config.max_iterations,
        history=history,
    )


# ---------------------------------------------------------------------------
# RTL → KPU area feedback (issue #31)
# ---------------------------------------------------------------------------


def _aggregate_synthesis_area_mm2(state: dict[str, Any]) -> float:
    """Sum the post-synthesis area across all RTL modules.

    Each entry in state['rtl_synthesis_results'] is the per-module result
    written by the rtl_generator specialist; `area_um2` is the synthesis
    cell-area total. We sum all successful modules and convert to mm².
    """
    synth_results = state.get("rtl_synthesis_results", {}) or {}
    total_um2 = 0.0
    for result in synth_results.values():
        if isinstance(result, dict) and result.get("success"):
            total_um2 += float(result.get("area_um2", 0.0) or 0.0)
    return total_um2 / 1e6


def apply_rtl_area_feedback(
    state: dict[str, Any],
    *,
    area_tolerance: float = 1.1,
    max_iterations: int = 3,
) -> dict[str, Any]:
    """Issue #31: feed real synthesis area back into the KPU sizing loop.

    After RTL synthesis runs, the cell-area total may exceed the floorplan
    estimate (which only counts SRAM macros + periphery + estimated logic
    pitch). When that happens, the KPU was sized too aggressively and the
    floorplan estimate is wrong — re-run the optimizer with the synthesis
    area as a tightened area constraint, then re-validate floorplan and
    bandwidth. Bounded at `max_iterations` to prevent infinite loops.

    Returns a state-update dict with possibly:
      - kpu_config (re-sized)
      - floorplan_estimate (re-validated)
      - bandwidth_match (re-validated)
      - kpu_optimization_history (per-iteration record)
      - rtl_area_feedback_summary (str describing what happened)

    No-op (returns empty dict) when:
      - rtl_synthesis_results is empty / missing
      - kpu_config is missing
      - floorplan_estimate is missing
      - synthesis area is within `area_tolerance × floorplan area`
    """
    from embodied_ai_architect.graphs.bandwidth import check_bandwidth_match
    from embodied_ai_architect.graphs.floorplan import estimate_floorplan
    from embodied_ai_architect.graphs.kpu_config import KPUMicroArchConfig

    kpu_dict = state.get("kpu_config") or {}
    fp_dict = state.get("floorplan_estimate") or {}
    if not kpu_dict or not fp_dict:
        return {}

    floorplan_area = float(fp_dict.get("total_area_mm2", 0.0) or 0.0)
    if floorplan_area <= 0:
        return {}

    synthesis_area = _aggregate_synthesis_area_mm2(state)
    if synthesis_area <= 0:
        return {}

    threshold = floorplan_area * area_tolerance
    if synthesis_area <= threshold:
        # Within tolerance — nothing to do.
        return {
            "rtl_area_feedback_summary": (
                f"Synthesis area {synthesis_area:.3f}mm² within "
                f"{area_tolerance:.0%} of floorplan {floorplan_area:.3f}mm² — "
                f"no re-sizing needed"
            ),
        }

    # Re-size: drop the area budget to the synthesis area so the optimizer
    # has a tighter target than the floorplan currently shows.
    workload = state.get("workload_profile", {}) or {}
    config = KPUMicroArchConfig(**kpu_dict)
    history: list[dict[str, Any]] = list(state.get("kpu_optimization_history", []) or [])

    # Each iteration: re-validate against the synthesis-derived budget, then
    # apply adjustments if still over budget. We re-use the existing
    # _apply_adjustments helper to avoid drift between this path and the
    # standalone kpu_loop.
    iterations_used = 0
    final_fp = None
    final_bw = None

    for i in range(max_iterations):
        iterations_used = i + 1

        fp = estimate_floorplan(config, max_die_area_mm2=synthesis_area)
        bw = check_bandwidth_match(config, workload)
        final_fp = fp
        final_bw = bw

        history.append(
            {
                "source": "rtl_area_feedback",
                "iteration": i,
                "config_name": config.name,
                "synthesis_area_mm2": round(synthesis_area, 3),
                "floorplan_total_area_mm2": round(fp.total_area_mm2, 3),
                "feasible": fp.feasible,
                "pitch_matched": fp.pitch_matched,
                "bandwidth_balanced": bw.balanced,
            }
        )

        if fp.feasible and bw.balanced:
            break

        config = _apply_adjustments(config, fp, bw, synthesis_area)

    summary = (
        f"RTL area feedback: synthesis {synthesis_area:.3f}mm² > "
        f"floorplan {floorplan_area:.3f}mm² × {area_tolerance:.2f}; "
        f"re-sized in {iterations_used} iteration(s); "
        f"converged={final_fp.feasible if final_fp else False}"
    )

    updates: dict[str, Any] = {
        "kpu_config": config.model_dump(),
        "kpu_optimization_history": history,
        "rtl_area_feedback_summary": summary,
    }
    if final_fp is not None:
        updates["floorplan_estimate"] = final_fp.model_dump()
    if final_bw is not None:
        updates["bandwidth_match"] = final_bw.model_dump()
    return updates


def _apply_adjustments(config, fp, bw, max_area: float):
    """Apply targeted adjustments to fix violations.

    Returns a new KPUMicroArchConfig with adjustments applied.
    """
    from embodied_ai_architect.graphs.kpu_config import KPUMicroArchConfig

    # Work with a copy
    d = config.model_dump()

    # Fix area first (most impactful)
    if fp.total_area_mm2 > max_area:
        if d["array_rows"] > 2:
            d["array_rows"] -= 1
        elif d["array_cols"] > 2:
            d["array_cols"] -= 1
        else:
            # Shrink SRAM
            d["compute_tile"]["l2_size_bytes"] = max(
                64 * 1024, d["compute_tile"]["l2_size_bytes"] // 2
            )
            d["memory_tile"]["l3_tile_size_bytes"] = max(
                64 * 1024, d["memory_tile"]["l3_tile_size_bytes"] // 2
            )

    # Fix pitch mismatch
    if not fp.pitch_matched:
        if fp.pitch_ratio_width > 1.0 + fp.pitch_tolerance:
            d["compute_tile"]["array_cols"] = max(4, d["compute_tile"]["array_cols"] - 2)
        elif fp.pitch_ratio_width < 1.0 - fp.pitch_tolerance:
            d["memory_tile"]["l3_num_banks"] += 1

        if fp.pitch_ratio_height > 1.0 + fp.pitch_tolerance:
            d["compute_tile"]["vector_lanes"] = max(4, d["compute_tile"]["vector_lanes"] - 2)
        elif fp.pitch_ratio_height < 1.0 - fp.pitch_tolerance:
            d["memory_tile"]["num_block_movers"] += 1

    # Fix bandwidth
    if not bw.balanced and bw.bottleneck_link:
        if "dram" in bw.bottleneck_link:
            d["dram"]["num_controllers"] = min(8, d["dram"]["num_controllers"] + 1)
        elif "l3" in bw.bottleneck_link or "noc" in bw.bottleneck_link:
            d["noc"]["link_width_bits"] = min(1024, d["noc"]["link_width_bits"] * 2)
        elif "l2" in bw.bottleneck_link:
            d["compute_tile"]["l2_num_banks"] += 2
        elif "l1" in bw.bottleneck_link:
            d["compute_tile"]["l1_num_banks"] += 2

    return KPUMicroArchConfig(**d)
