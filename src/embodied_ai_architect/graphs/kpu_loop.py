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
