"""Floorplan estimator for KPU checkerboard layout.

Estimates 2D tile dimensions and checks pitch matching between
compute tiles and memory tiles in the checkerboard array.

Primary check: compute tile width/height ≈ memory tile width/height
(within tolerance) for efficient 2D tiling.

Usage:
    from embodied_ai_architect.graphs.floorplan import estimate_floorplan

    result = estimate_floorplan(kpu_config, max_die_area_mm2=100.0)
    assert result.feasible
"""

from __future__ import annotations

import math
from typing import Any, Optional

from pydantic import BaseModel, Field

from embodied_ai_architect.graphs.technology import estimate_sram_area_mm2, get_technology

# ---------------------------------------------------------------------------
# Area estimation constants
# ---------------------------------------------------------------------------

# ALU area at 5nm reference (mm²) — scales with (nm/5)²
ALU_AREA_5NM = 0.0001  # ~100 um² for a single MAC at 5nm

# Multipliers for more complex structures relative to a single ALU
SYSTOLIC_MULT = 1.5  # Systolic cell overhead (control, pipeline regs)
VECTOR_MULT = 2.0  # Vector unit per-lane overhead


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class SubBlockArea(BaseModel):
    """Area breakdown for a single sub-block."""

    name: str
    area_mm2: float
    width_mm: float = 0.0
    height_mm: float = 0.0


class TileDimensions(BaseModel):
    """2D dimensions of a single tile."""

    width_mm: float
    height_mm: float
    area_mm2: float
    sub_blocks: list[SubBlockArea] = Field(default_factory=list)


class FloorplanEstimate(BaseModel):
    """Checkerboard floorplan estimate."""

    compute_tile: TileDimensions
    memory_tile: TileDimensions
    pitch_matched: bool
    pitch_ratio_width: float
    pitch_ratio_height: float
    pitch_tolerance: float
    array_width_mm: float
    array_height_mm: float
    core_area_mm2: float
    periphery_area_mm2: float
    total_area_mm2: float
    die_edge_mm: float
    feasible: bool
    max_die_area_mm2: float
    issues: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Sub-component area estimation
# ---------------------------------------------------------------------------


def _estimate_systolic_area(rows: int, cols: int, process_nm: int) -> float:
    """Estimate systolic array area in mm²."""
    mac_area = ALU_AREA_5NM * (process_nm / 5) ** 2 * SYSTOLIC_MULT
    return rows * cols * mac_area


def _estimate_vector_unit_area(lanes: int, process_nm: int) -> float:
    """Estimate vector unit area in mm²."""
    lane_area = ALU_AREA_5NM * (process_nm / 5) ** 2 * VECTOR_MULT
    return lanes * lane_area


def _estimate_streamer_area(count: int, process_nm: int) -> float:
    """Estimate streamer control logic area."""
    return count * 0.01 * (process_nm / 5) ** 2


def _estimate_tile_control_area(process_nm: int) -> float:
    """Estimate tile control/scheduler area."""
    return 0.05 * (process_nm / 5) ** 2


def _estimate_block_mover_area(count: int, process_nm: int) -> float:
    """Estimate block mover area."""
    return count * 0.02 * (process_nm / 5) ** 2


def _estimate_dma_area(count: int, process_nm: int) -> float:
    """Estimate DMA engine area."""
    return count * 0.03 * (process_nm / 5) ** 2


def _estimate_mem_control_area(process_nm: int) -> float:
    """Estimate memory tile control logic area."""
    return 0.02 * (process_nm / 5) ** 2


# ---------------------------------------------------------------------------
# Tile dimensioning
# ---------------------------------------------------------------------------


def _area_to_dimensions(area_mm2: float, aspect_ratio: float = 1.0) -> tuple[float, float]:
    """Convert area to width x height given aspect ratio."""
    if area_mm2 <= 0:
        return 0.0, 0.0
    # width/height = aspect_ratio, width * height = area
    # height = sqrt(area / aspect_ratio)
    height = math.sqrt(area_mm2 / aspect_ratio)
    width = area_mm2 / height
    return width, height


def estimate_compute_tile_dimensions(
    compute_tile,
    process_nm: int,
    aspect_ratio: float = 1.0,
) -> TileDimensions:
    """Estimate compute tile dimensions from its sub-components.

    Components: systolic array + vector unit + L2 SRAM + L1 SRAM + streamers + control
    """
    sub_blocks = []

    # Systolic array
    array_area = _estimate_systolic_area(
        compute_tile.array_rows, compute_tile.array_cols, process_nm
    )
    sub_blocks.append(SubBlockArea(name="systolic_array", area_mm2=array_area))

    # Vector unit
    vector_area = _estimate_vector_unit_area(compute_tile.vector_lanes, process_nm)
    sub_blocks.append(SubBlockArea(name="vector_unit", area_mm2=vector_area))

    # L2 SRAM
    l2_area = estimate_sram_area_mm2(compute_tile.l2_size_bytes, process_nm)
    sub_blocks.append(SubBlockArea(name="l2_sram", area_mm2=l2_area))

    # L1 SRAM
    l1_area = estimate_sram_area_mm2(compute_tile.l1_size_bytes, process_nm)
    sub_blocks.append(SubBlockArea(name="l1_sram", area_mm2=l1_area))

    # Streamers
    streamer_area = _estimate_streamer_area(compute_tile.num_streamers, process_nm)
    sub_blocks.append(SubBlockArea(name="streamers", area_mm2=streamer_area))

    # Tile control
    control_area = _estimate_tile_control_area(process_nm)
    sub_blocks.append(SubBlockArea(name="tile_control", area_mm2=control_area))

    total_area = sum(sb.area_mm2 for sb in sub_blocks)
    width, height = _area_to_dimensions(total_area, aspect_ratio)

    return TileDimensions(
        width_mm=round(width, 3),
        height_mm=round(height, 3),
        area_mm2=round(total_area, 4),
        sub_blocks=sub_blocks,
    )


def estimate_memory_tile_dimensions(
    memory_tile,
    process_nm: int,
    aspect_ratio: float = 1.0,
) -> TileDimensions:
    """Estimate memory tile dimensions from its sub-components.

    Components: L3 SRAM + block movers + DMA engines + control
    """
    sub_blocks = []

    # L3 SRAM
    l3_area = estimate_sram_area_mm2(memory_tile.l3_tile_size_bytes, process_nm)
    sub_blocks.append(SubBlockArea(name="l3_sram", area_mm2=l3_area))

    # Block movers
    mover_area = _estimate_block_mover_area(memory_tile.num_block_movers, process_nm)
    sub_blocks.append(SubBlockArea(name="block_movers", area_mm2=mover_area))

    # DMA engines
    dma_area = _estimate_dma_area(memory_tile.num_dma_engines, process_nm)
    sub_blocks.append(SubBlockArea(name="dma_engines", area_mm2=dma_area))

    # Memory tile control
    control_area = _estimate_mem_control_area(process_nm)
    sub_blocks.append(SubBlockArea(name="mem_tile_control", area_mm2=control_area))

    total_area = sum(sb.area_mm2 for sb in sub_blocks)
    width, height = _area_to_dimensions(total_area, aspect_ratio)

    return TileDimensions(
        width_mm=round(width, 3),
        height_mm=round(height, 3),
        area_mm2=round(total_area, 4),
        sub_blocks=sub_blocks,
    )


# ---------------------------------------------------------------------------
# Periphery estimation
# ---------------------------------------------------------------------------


def _estimate_periphery_area(config, process_nm: int) -> float:
    """Estimate periphery area: DRAM controllers, I/O pads, NoC edge routers."""
    area = 0.0
    # DRAM controllers
    area += config.dram.num_controllers * 0.5 * (process_nm / 5) ** 2
    # I/O ring (rough estimate)
    area += 1.0 * (process_nm / 5) ** 2
    # NoC edge routers
    edge_routers = 2 * (config.array_rows + config.array_cols)
    area += edge_routers * 0.01 * (process_nm / 5) ** 2
    return area


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def estimate_floorplan(
    config,
    max_die_area_mm2: float = 100.0,
    pitch_tolerance: float = 0.15,
    routing_overhead: float = 0.20,
) -> FloorplanEstimate:
    """Estimate checkerboard floorplan with pitch matching.

    Steps:
    1. Estimate compute tile dimensions
    2. Estimate memory tile dimensions
    3. Check pitch matching (width and height ratios)
    4. Compute die dimensions
    5. Add periphery
    6. Check total area against budget

    Args:
        config: KPUMicroArchConfig instance.
        max_die_area_mm2: Maximum allowed die area.
        pitch_tolerance: Acceptable pitch mismatch (0.15 = 15%).
        routing_overhead: Fraction of core area added for routing.

    Returns:
        FloorplanEstimate with feasibility verdict.
    """
    process_nm = config.process_nm
    issues = []

    # 1. Compute tile dimensions
    compute_dims = estimate_compute_tile_dimensions(config.compute_tile, process_nm)

    # 2. Memory tile dimensions
    memory_dims = estimate_memory_tile_dimensions(config.memory_tile, process_nm)

    # 3. Pitch matching
    if memory_dims.width_mm > 0 and memory_dims.height_mm > 0:
        pitch_ratio_w = compute_dims.width_mm / memory_dims.width_mm
        pitch_ratio_h = compute_dims.height_mm / memory_dims.height_mm
    else:
        pitch_ratio_w = 1.0
        pitch_ratio_h = 1.0

    width_ok = abs(pitch_ratio_w - 1.0) <= pitch_tolerance
    height_ok = abs(pitch_ratio_h - 1.0) <= pitch_tolerance
    pitch_matched = width_ok and height_ok

    if not width_ok:
        if pitch_ratio_w > 1.0 + pitch_tolerance:
            issues.append(
                f"Compute tile too wide vs memory tile (ratio {pitch_ratio_w:.2f}). "
                "Consider: reduce array_cols, reduce L2 banks, or increase L3 banks."
            )
        else:
            issues.append(
                f"Memory tile too wide vs compute tile (ratio {pitch_ratio_w:.2f}). "
                "Consider: increase L2 size or reduce L3 tile size."
            )

    if not height_ok:
        if pitch_ratio_h > 1.0 + pitch_tolerance:
            issues.append(
                f"Compute tile too tall vs memory tile (ratio {pitch_ratio_h:.2f}). "
                "Consider: reduce vector_lanes or add block movers to memory tile."
            )
        else:
            issues.append(
                f"Memory tile too tall vs compute tile (ratio {pitch_ratio_h:.2f}). "
                "Consider: increase streamer buffers or reduce DMA engines."
            )

    # 4. Die dimensions (use max of tile widths/heights for grid)
    tile_width = max(compute_dims.width_mm, memory_dims.width_mm)
    tile_height = max(compute_dims.height_mm, memory_dims.height_mm)
    array_width = config.array_cols * tile_width
    array_height = config.array_rows * tile_height
    core_area = array_width * array_height

    # 5. Periphery
    periphery_area = _estimate_periphery_area(config, process_nm)

    # 6. Total with routing overhead
    total_area = (core_area + periphery_area) * (1.0 + routing_overhead)
    die_edge = math.sqrt(total_area) if total_area > 0 else 0.0

    # Feasibility
    area_ok = total_area <= max_die_area_mm2
    if not area_ok:
        issues.append(
            f"Die area {total_area:.1f} mm² exceeds budget {max_die_area_mm2:.0f} mm². "
            "Consider: reduce array_rows/cols or SRAM sizes."
        )

    # Reticle limit check (~33mm edge)
    if die_edge > 33.0:
        issues.append(f"Die edge {die_edge:.1f}mm exceeds reticle limit (~33mm).")

    feasible = pitch_matched and area_ok and die_edge <= 33.0

    return FloorplanEstimate(
        compute_tile=compute_dims,
        memory_tile=memory_dims,
        pitch_matched=pitch_matched,
        pitch_ratio_width=round(pitch_ratio_w, 3),
        pitch_ratio_height=round(pitch_ratio_h, 3),
        pitch_tolerance=pitch_tolerance,
        array_width_mm=round(array_width, 3),
        array_height_mm=round(array_height, 3),
        core_area_mm2=round(core_area, 2),
        periphery_area_mm2=round(periphery_area, 2),
        total_area_mm2=round(total_area, 2),
        die_edge_mm=round(die_edge, 2),
        feasible=feasible,
        max_die_area_mm2=max_die_area_mm2,
        issues=issues,
    )
