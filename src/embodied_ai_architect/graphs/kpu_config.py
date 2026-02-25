"""KPU micro-architecture configuration model.

Defines the configuration space for a Knowledge Processing Unit (KPU)
organized as a 2D checkerboard of alternating compute tiles and memory tiles.

Compute tiles contain: systolic array + vector unit + L2 cache + L1 skew buffer + streamers
Memory tiles contain: L3 SRAM + block movers + DMA engines

Usage:
    from embodied_ai_architect.graphs.kpu_config import (
        KPUMicroArchConfig, KPU_PRESETS, create_kpu_config,
    )

    config = KPU_PRESETS["edge_balanced"]
    config = create_kpu_config("delivery_drone", {"max_power_watts": 5.0}, {"gflops": 8.7})
"""

from __future__ import annotations

import math
from typing import Any, Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Sub-component configs
# ---------------------------------------------------------------------------


class DRAMConfig(BaseModel):
    """External memory configuration."""

    technology: str = "LPDDR4X"
    num_controllers: int = 2
    channels_per_controller: int = 2
    bandwidth_per_channel_gbps: float = 6.4
    capacity_gb: float = 4.0

    @property
    def total_bandwidth_gbps(self) -> float:
        return self.num_controllers * self.channels_per_controller * self.bandwidth_per_channel_gbps


class NoCConfig(BaseModel):
    """Network on Chip configuration."""

    topology: str = "mesh_2d"
    link_width_bits: int = 256
    frequency_mhz: float = 1000.0
    num_routers: int = 16

    @property
    def link_bandwidth_gbps(self) -> float:
        return self.link_width_bits * self.frequency_mhz / 8 / 1000


class ComputeTileConfig(BaseModel):
    """Compute tile configuration.

    A compute tile contains:
    - Systolic array (MAC array for matrix ops)
    - Vector unit (element-wise ops)
    - L2 cache banks (inside tile to minimize wire loading)
    - L1 skew buffer (scratchpad closest to compute)
    - Streamers (prefetch engines feeding L1 from L2)
    - Local control/scheduler
    """

    num_tiles: int = 4
    # Systolic array
    array_rows: int = 16
    array_cols: int = 16
    vector_lanes: int = 16
    frequency_mhz: float = 500.0
    supported_precisions: list[str] = Field(default_factory=lambda: ["int8", "fp16", "bf16"])
    # L2 (inside compute tile)
    l2_size_bytes: int = 256 * 1024
    l2_num_banks: int = 8
    l2_read_ports: int = 2
    l2_write_ports: int = 1
    # L1 skew buffer (inside compute tile)
    l1_size_bytes: int = 32 * 1024
    l1_num_banks: int = 4
    # Streamers (inside compute tile)
    num_streamers: int = 2
    streamer_prefetch_depth: int = 4
    streamer_buffer_bytes: int = 16 * 1024

    @property
    def peak_tops_int8(self) -> float:
        ops_per_cycle = self.array_rows * self.array_cols * 2  # multiply + accumulate
        total_ops = ops_per_cycle * self.num_tiles * self.frequency_mhz * 1e6
        return total_ops / 1e12


class MemoryTileConfig(BaseModel):
    """Memory tile configuration.

    Memory tiles alternate with compute tiles in a checkerboard.
    Each contains L3 SRAM + block movers + DMA engines.
    Width/height must pitch-match compute tiles.
    """

    l3_tile_size_bytes: int = 512 * 1024
    l3_num_banks: int = 4
    num_block_movers: int = 2
    block_mover_bw_gbps: float = 32.0
    num_dma_engines: int = 1
    dma_max_transfer_bytes: int = 1024 * 1024
    dma_queue_depth: int = 8


# ---------------------------------------------------------------------------
# Top-level KPU config
# ---------------------------------------------------------------------------


class KPUMicroArchConfig(BaseModel):
    """Complete KPU micro-architecture configuration.

    Organized as a 2D checkerboard of alternating compute and memory tiles:

        C  M  C  M
        M  C  M  C
        C  M  C  M

    Compute tiles and memory tiles must be pitch-matched for efficient layout.
    """

    name: str = "swkpu-v1"
    process_nm: int = 28
    dram: DRAMConfig = Field(default_factory=DRAMConfig)
    noc: NoCConfig = Field(default_factory=NoCConfig)
    compute_tile: ComputeTileConfig = Field(default_factory=ComputeTileConfig)
    memory_tile: MemoryTileConfig = Field(default_factory=MemoryTileConfig)
    array_rows: int = 3
    array_cols: int = 3

    @property
    def num_compute_tiles(self) -> int:
        total = self.array_rows * self.array_cols
        return (total + 1) // 2

    @property
    def num_memory_tiles(self) -> int:
        total = self.array_rows * self.array_cols
        return total // 2

    @property
    def total_l3_bytes(self) -> int:
        return self.num_memory_tiles * self.memory_tile.l3_tile_size_bytes

    @property
    def total_l2_bytes(self) -> int:
        return self.num_compute_tiles * self.compute_tile.l2_size_bytes

    @property
    def total_l1_bytes(self) -> int:
        return self.num_compute_tiles * self.compute_tile.l1_size_bytes

    @property
    def total_sram_bytes(self) -> int:
        return self.total_l1_bytes + self.total_l2_bytes + self.total_l3_bytes

    @property
    def total_dram_bandwidth_gbps(self) -> float:
        return self.dram.total_bandwidth_gbps

    @property
    def peak_tops_int8(self) -> float:
        ct = self.compute_tile
        ops_per_cycle = ct.array_rows * ct.array_cols * 2
        total_ops = ops_per_cycle * self.num_compute_tiles * ct.frequency_mhz * 1e6
        return total_ops / 1e12


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------


KPU_PRESETS: dict[str, KPUMicroArchConfig] = {
    "drone_minimal": KPUMicroArchConfig(
        name="swkpu-drone-min",
        process_nm=28,
        dram=DRAMConfig(
            technology="LPDDR4X",
            num_controllers=1,
            channels_per_controller=2,
            bandwidth_per_channel_gbps=6.4,
            capacity_gb=2.0,
        ),
        noc=NoCConfig(topology="mesh_2d", link_width_bits=128, frequency_mhz=500.0, num_routers=4),
        compute_tile=ComputeTileConfig(
            num_tiles=2,
            array_rows=8,
            array_cols=8,
            vector_lanes=8,
            frequency_mhz=400.0,
            l2_size_bytes=128 * 1024,
            l2_num_banks=4,
            l1_size_bytes=16 * 1024,
            l1_num_banks=2,
            num_streamers=1,
            streamer_buffer_bytes=8 * 1024,
        ),
        memory_tile=MemoryTileConfig(
            l3_tile_size_bytes=256 * 1024,
            l3_num_banks=2,
            num_block_movers=1,
            block_mover_bw_gbps=16.0,
            num_dma_engines=1,
        ),
        array_rows=2,
        array_cols=2,
    ),
    "edge_balanced": KPUMicroArchConfig(
        name="swkpu-edge-bal",
        process_nm=28,
        dram=DRAMConfig(
            technology="LPDDR4X",
            num_controllers=2,
            channels_per_controller=2,
            bandwidth_per_channel_gbps=6.4,
            capacity_gb=4.0,
        ),
        noc=NoCConfig(
            topology="mesh_2d", link_width_bits=256, frequency_mhz=1000.0, num_routers=16
        ),
        compute_tile=ComputeTileConfig(
            num_tiles=4,
            array_rows=16,
            array_cols=16,
            vector_lanes=16,
            frequency_mhz=500.0,
            l2_size_bytes=256 * 1024,
            l2_num_banks=8,
            l1_size_bytes=32 * 1024,
            l1_num_banks=4,
            num_streamers=2,
            streamer_buffer_bytes=16 * 1024,
        ),
        memory_tile=MemoryTileConfig(
            l3_tile_size_bytes=512 * 1024,
            l3_num_banks=4,
            num_block_movers=2,
            block_mover_bw_gbps=32.0,
            num_dma_engines=1,
        ),
        array_rows=3,
        array_cols=3,
    ),
    "server_max": KPUMicroArchConfig(
        name="swkpu-server-max",
        process_nm=7,
        dram=DRAMConfig(
            technology="HBM2E",
            num_controllers=4,
            channels_per_controller=4,
            bandwidth_per_channel_gbps=25.6,
            capacity_gb=16.0,
        ),
        noc=NoCConfig(
            topology="mesh_2d", link_width_bits=512, frequency_mhz=2000.0, num_routers=64
        ),
        compute_tile=ComputeTileConfig(
            num_tiles=16,
            array_rows=32,
            array_cols=32,
            vector_lanes=32,
            frequency_mhz=1000.0,
            l2_size_bytes=1024 * 1024,
            l2_num_banks=16,
            l2_read_ports=4,
            l2_write_ports=2,
            l1_size_bytes=128 * 1024,
            l1_num_banks=8,
            num_streamers=4,
            streamer_buffer_bytes=32 * 1024,
        ),
        memory_tile=MemoryTileConfig(
            l3_tile_size_bytes=2 * 1024 * 1024,
            l3_num_banks=8,
            num_block_movers=4,
            block_mover_bw_gbps=64.0,
            num_dma_engines=2,
            dma_queue_depth=16,
        ),
        array_rows=5,
        array_cols=5,
    ),
}


# ---------------------------------------------------------------------------
# Heuristic sizing
# ---------------------------------------------------------------------------


def create_kpu_config(
    use_case: str,
    constraints: dict[str, Any],
    workload: dict[str, Any],
) -> KPUMicroArchConfig:
    """Generate a KPU config from workload requirements and constraints.

    Heuristic sizing based on:
    - Workload GFLOPS and memory requirements
    - Power/area/cost constraints
    - Use case characteristics

    Args:
        use_case: Application type (e.g. "delivery_drone").
        constraints: Design constraints dict.
        workload: Workload profile dict with gflops, memory_mb, etc.

    Returns:
        A sized KPUMicroArchConfig.
    """
    gflops = workload.get("total_estimated_gflops", workload.get("estimated_gflops", 5.0))
    memory_mb = workload.get("total_estimated_memory_mb", workload.get("estimated_memory_mb", 20.0))
    max_power = constraints.get("max_power_watts", 10.0)
    max_area = constraints.get("max_area_mm2", 100.0)
    process_nm = constraints.get("target_process_nm", 28)

    # Size systolic array based on GFLOPS target
    # At 500MHz, a 16x16 array does 16*16*2*500e6 = 256 GOPS = 0.256 TOPS
    # Need gflops/1000 TOPS -> num_tiles = ceil(tops_needed / 0.256)
    tops_needed = gflops / 1000
    single_tile_tops = 16 * 16 * 2 * 500e6 / 1e12  # 0.256 TOPS
    min_compute_tiles = max(2, math.ceil(tops_needed / single_tile_tops))

    # Power scaling: limit compute tiles based on power
    if max_power and max_power < 3.0:
        min_compute_tiles = min(min_compute_tiles, 2)
        array_size = 8
        freq = 400.0
    elif max_power and max_power < 8.0:
        min_compute_tiles = min(min_compute_tiles, 5)
        array_size = 16
        freq = 500.0
    else:
        array_size = min(32, 16 + int(gflops / 20))
        freq = min(1000.0, 500.0 + gflops * 10)

    # Determine checkerboard grid from compute tile count
    # Compute tiles = ceil(grid_total / 2), so grid_total = compute_tiles * 2
    grid_total = min_compute_tiles * 2
    grid_side = max(2, math.ceil(math.sqrt(grid_total)))

    # Size memory based on workload
    l2_size = 256 * 1024  # 256KB default
    l1_size = 32 * 1024
    l3_tile_size = 512 * 1024

    if memory_mb > 50:
        l2_size = 512 * 1024
        l3_tile_size = 1024 * 1024
    elif memory_mb < 10:
        l2_size = 128 * 1024
        l3_tile_size = 256 * 1024

    # DRAM sizing
    if max_power and max_power < 5.0:
        dram_tech = "LPDDR4X"
        dram_controllers = 1
        dram_channels = 2
        dram_bw = 6.4
        dram_cap = 2.0
    elif gflops > 50:
        dram_tech = "HBM2E"
        dram_controllers = 4
        dram_channels = 4
        dram_bw = 25.6
        dram_cap = 16.0
    else:
        dram_tech = "LPDDR4X"
        dram_controllers = 2
        dram_channels = 2
        dram_bw = 6.4
        dram_cap = 4.0

    # Build initial tile configs
    compute_tile_cfg = ComputeTileConfig(
        num_tiles=min_compute_tiles,
        array_rows=array_size,
        array_cols=array_size,
        vector_lanes=array_size,
        frequency_mhz=freq,
        l2_size_bytes=l2_size,
        l2_num_banks=max(4, l2_size // (32 * 1024)),
        l1_size_bytes=l1_size,
        l1_num_banks=max(2, l1_size // (8 * 1024)),
        num_streamers=2 if array_size >= 16 else 1,
        streamer_buffer_bytes=16 * 1024,
    )
    memory_tile_cfg = MemoryTileConfig(
        l3_tile_size_bytes=l3_tile_size,
        l3_num_banks=max(2, l3_tile_size // (128 * 1024)),
        num_block_movers=2 if grid_side >= 3 else 1,
        block_mover_bw_gbps=32.0,
        num_dma_engines=1,
    )

    # Pitch-matching: increase L3 until memory tile pitch matches compute tile.
    # Import here to avoid circular dependency.
    from embodied_ai_architect.graphs.floorplan import (
        estimate_compute_tile_dimensions,
        estimate_memory_tile_dimensions,
    )

    ct_dims = estimate_compute_tile_dimensions(compute_tile_cfg, process_nm)
    mt_dims = estimate_memory_tile_dimensions(memory_tile_cfg, process_nm)
    pitch_tolerance = 0.15
    max_l3 = l3_tile_size * 16  # upper bound to avoid infinite loop

    while mt_dims.width_mm > 0 and ct_dims.width_mm / mt_dims.width_mm > 1.0 + pitch_tolerance:
        l3_tile_size = int(l3_tile_size * 1.1)  # 10% steps for fine convergence
        if l3_tile_size > max_l3:
            break
        memory_tile_cfg = MemoryTileConfig(
            l3_tile_size_bytes=l3_tile_size,
            l3_num_banks=max(2, l3_tile_size // (128 * 1024)),
            num_block_movers=memory_tile_cfg.num_block_movers,
            block_mover_bw_gbps=memory_tile_cfg.block_mover_bw_gbps,
            num_dma_engines=memory_tile_cfg.num_dma_engines,
        )
        mt_dims = estimate_memory_tile_dimensions(memory_tile_cfg, process_nm)

    return KPUMicroArchConfig(
        name=f"swkpu-{use_case.replace('_', '-')}",
        process_nm=process_nm,
        dram=DRAMConfig(
            technology=dram_tech,
            num_controllers=dram_controllers,
            channels_per_controller=dram_channels,
            bandwidth_per_channel_gbps=dram_bw,
            capacity_gb=dram_cap,
        ),
        noc=NoCConfig(
            topology="mesh_2d",
            link_width_bits=256 if grid_side <= 4 else 512,
            frequency_mhz=freq * 2,
            num_routers=grid_side * grid_side,
        ),
        compute_tile=compute_tile_cfg,
        memory_tile=memory_tile_cfg,
        array_rows=grid_side,
        array_cols=grid_side,
    )
