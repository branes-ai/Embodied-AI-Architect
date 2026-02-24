"""Composable IP block models with process-aware area/power estimation.

Provides Pydantic models for heterogeneous SoC composition: CPU cores,
GPU clusters, KPU tiles, memory controllers, NoC, I/O, and ISP blocks.
Each block implements physics-based PPA estimation using technology.py data.

Usage:
    from embodied_ai_architect.graphs.ip_blocks import (
        CPUCoreConfig, GPUClusterConfig, KPUBlockConfig,
        SoCComposition, ARM_A78_PRESET, SMALL_GPU_PRESET,
    )

    soc = SoCComposition(
        name="drone-soc",
        process_nm=28,
        blocks=[ARM_A78_PRESET, SMALL_GPU_PRESET],
    )
    print(soc.total_area_mm2())
    print(soc.total_power_watts())
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from embodied_ai_architect.graphs.technology import (
    estimate_sram_area_mm2,
    get_technology,
)

# ---------------------------------------------------------------------------
# ISA complexity factors
# ---------------------------------------------------------------------------
# Multiplier on base ALU/FPU area to account for pipeline depth, reorder
# buffers, branch predictors, register files, and control logic.

ISA_COMPLEXITY: dict[str, float] = {
    "ARM_A78": 150.0,  # big OoO core, 4-wide decode, deep pipeline
    "ARM_A55": 60.0,  # little in-order core, 2-wide
    "RISCV_RV64": 40.0,  # minimal 64-bit RISC-V core
}

# GPU shader core complexity relative to a single FPU_32bit
GPU_CORE_COMPLEXITY: float = 2.5  # warp scheduler + register file + SFU overhead

# Systolic/tensor unit density relative to FPU_32bit
TENSOR_UNIT_COMPLEXITY: float = 1.2  # simpler than full FPU, packed tighter


# ---------------------------------------------------------------------------
# Base IP Block
# ---------------------------------------------------------------------------


class IPBlockConfig(BaseModel):
    """Base for all composable IP blocks on a die."""

    block_type: str
    name: str
    count: int = 1

    def estimate_area_mm2(self, process_nm: int) -> float:
        """Estimate total area in mm² for all instances at given process node."""
        return 0.0

    def estimate_power_watts(self, process_nm: int, utilization: float = 0.8) -> float:
        """Estimate total power in watts for all instances."""
        return 0.0

    def peak_gops(self) -> float:
        """Peak throughput in giga-operations per second."""
        return 0.0


# ---------------------------------------------------------------------------
# CPU Core IP
# ---------------------------------------------------------------------------


class CPUCoreConfig(IPBlockConfig):
    """CPU core IP block with process-aware area/power estimation.

    Area model: (issue_width * ALU_area + FPU_area) * ISA_complexity + SRAM
    Power model: frequency * Vdd^2 * area_factor * utilization
    """

    block_type: Literal["cpu_core"] = "cpu_core"
    isa: str = "ARM_A78"
    frequency_mhz: float = 2000.0
    pipeline_stages: int = 13
    issue_width: int = 4
    l1i_bytes: int = 64 * 1024
    l1d_bytes: int = 64 * 1024
    l2_bytes: int = 256 * 1024
    has_fpu: bool = True
    has_simd: bool = True

    def estimate_area_mm2(self, process_nm: int) -> float:
        tech = get_technology(process_nm)
        ref = tech.reference_areas

        alu_area_um2 = ref.get("ALU_32bit", 400.0)
        fpu_area_um2 = ref.get("FPU_32bit", 8000.0)
        complexity = ISA_COMPLEXITY.get(self.isa, 60.0)

        # Base logic area per core
        logic_um2 = self.issue_width * alu_area_um2 * complexity
        if self.has_fpu:
            logic_um2 += fpu_area_um2 * complexity * 0.5
        if self.has_simd:
            logic_um2 += alu_area_um2 * complexity * 0.3 * self.issue_width

        logic_mm2 = logic_um2 / 1e6

        # Cache area
        sram_mm2 = (
            estimate_sram_area_mm2(self.l1i_bytes, process_nm)
            + estimate_sram_area_mm2(self.l1d_bytes, process_nm)
            + estimate_sram_area_mm2(self.l2_bytes, process_nm)
        )

        return (logic_mm2 + sram_mm2) * self.count

    def estimate_power_watts(self, process_nm: int, utilization: float = 0.8) -> float:
        tech = get_technology(process_nm)
        area_mm2 = self.estimate_area_mm2(process_nm)

        # Dynamic power ~ f * V^2 * C (area proxy for capacitance)
        # Normalize: 1W per mm² at 1GHz, 0.9V, 28nm reference
        freq_ghz = self.frequency_mhz / 1000.0
        voltage_factor = (tech.vdd_v / 0.90) ** 2
        power_density = 1.0  # W/mm² at reference

        return area_mm2 * freq_ghz * voltage_factor * power_density * utilization

    def peak_gops(self) -> float:
        # OPS per cycle: issue_width * 2 (SIMD doubles it if present)
        ops_per_cycle = self.issue_width * (2 if self.has_simd else 1)
        return ops_per_cycle * self.frequency_mhz / 1000.0 * self.count


# ---------------------------------------------------------------------------
# GPU Cluster IP
# ---------------------------------------------------------------------------


class GPUClusterConfig(IPBlockConfig):
    """GPU shader cluster IP block.

    Area model: shader_cores * (fp32_units * FPU_area * complexity + L1 SRAM)
                + tensor_units * FPU_area * tensor_density + shared L2 SRAM
    Power model: frequency * Vdd^2 * area_factor * utilization
    """

    block_type: Literal["gpu_cluster"] = "gpu_cluster"
    shader_cores: int = 4
    frequency_mhz: float = 1000.0
    fp32_units_per_core: int = 128
    tensor_units: int = 0
    l1_bytes_per_core: int = 64 * 1024
    shared_l2_bytes: int = 512 * 1024
    memory_bus_width_bits: int = 128

    def estimate_area_mm2(self, process_nm: int) -> float:
        tech = get_technology(process_nm)
        ref = tech.reference_areas

        fpu_area_um2 = ref.get("FPU_32bit", 8000.0)

        # Per-core: FP ALUs + L1 SRAM
        fp_area_um2 = self.fp32_units_per_core * fpu_area_um2 * GPU_CORE_COMPLEXITY
        fp_area_mm2 = fp_area_um2 / 1e6
        l1_mm2 = estimate_sram_area_mm2(self.l1_bytes_per_core, process_nm)
        per_core_mm2 = fp_area_mm2 + l1_mm2

        # Tensor units (systolic density)
        tensor_mm2 = 0.0
        if self.tensor_units > 0:
            tensor_um2 = self.tensor_units * fpu_area_um2 * TENSOR_UNIT_COMPLEXITY
            tensor_mm2 = tensor_um2 / 1e6

        # Shared L2
        shared_l2_mm2 = estimate_sram_area_mm2(self.shared_l2_bytes, process_nm)

        total = per_core_mm2 * self.shader_cores + tensor_mm2 + shared_l2_mm2
        return total * self.count

    def estimate_power_watts(self, process_nm: int, utilization: float = 0.8) -> float:
        tech = get_technology(process_nm)
        area_mm2 = self.estimate_area_mm2(process_nm)

        freq_ghz = self.frequency_mhz / 1000.0
        voltage_factor = (tech.vdd_v / 0.90) ** 2
        # GPUs have higher power density than CPUs due to more active transistors
        power_density = 1.5  # W/mm² at reference

        return area_mm2 * freq_ghz * voltage_factor * power_density * utilization

    def peak_gops(self) -> float:
        # Each FP32 unit does 2 ops (FMA) per cycle
        fp_gops = self.shader_cores * self.fp32_units_per_core * 2 * self.frequency_mhz / 1000.0
        # Tensor units contribute additional throughput
        tensor_gops = self.tensor_units * 2 * self.frequency_mhz / 1000.0
        return (fp_gops + tensor_gops) * self.count


# ---------------------------------------------------------------------------
# KPU Block IP
# ---------------------------------------------------------------------------


class KPUBlockConfig(IPBlockConfig):
    """KPU tile array IP block — wraps existing KPUMicroArchConfig.

    Area/power delegates to floorplan.py estimator and technology data.
    """

    block_type: Literal["kpu_tile"] = "kpu_tile"
    kpu_preset: Optional[str] = None
    # Allow inline micro-arch overrides (optional)
    array_rows: int = 3
    array_cols: int = 3
    compute_frequency_mhz: float = 500.0
    systolic_rows: int = 16
    systolic_cols: int = 16

    def _get_kpu_config(self):
        """Resolve to a KPUMicroArchConfig."""
        from embodied_ai_architect.graphs.kpu_config import (
            KPU_PRESETS,
            KPUMicroArchConfig,
        )

        if self.kpu_preset and self.kpu_preset in KPU_PRESETS:
            return KPU_PRESETS[self.kpu_preset]
        return KPUMicroArchConfig(
            name=self.name,
            array_rows=self.array_rows,
            array_cols=self.array_cols,
            compute_tile=dict(
                num_tiles=(self.array_rows * self.array_cols + 1) // 2,
                array_rows=self.systolic_rows,
                array_cols=self.systolic_cols,
                frequency_mhz=self.compute_frequency_mhz,
            ),
        )

    def estimate_area_mm2(self, process_nm: int) -> float:
        from embodied_ai_architect.graphs.floorplan import (
            estimate_compute_tile_dimensions,
            estimate_memory_tile_dimensions,
        )

        kpu = self._get_kpu_config()
        ct = estimate_compute_tile_dimensions(kpu.compute_tile, process_nm)
        mt = estimate_memory_tile_dimensions(kpu.memory_tile, process_nm)

        # Die area = grid of tiles + routing overhead
        tile_width = max(ct.width_mm, mt.width_mm)
        tile_height = max(ct.height_mm, mt.height_mm)
        core_area = kpu.array_cols * tile_width * kpu.array_rows * tile_height
        routing_overhead = 1.20
        return core_area * routing_overhead * self.count

    def estimate_power_watts(self, process_nm: int, utilization: float = 0.8) -> float:
        tech = get_technology(process_nm)
        kpu = self._get_kpu_config()

        # Compute power from systolic array activity
        ct = kpu.compute_tile
        num_compute = kpu.num_compute_tiles
        ops_per_cycle = ct.array_rows * ct.array_cols * 2
        total_ops_per_sec = ops_per_cycle * num_compute * ct.frequency_mhz * 1e6

        # Energy per op scales with technology
        energy_per_op_pj = tech.energy_per_cell_pj * 10  # ~10 cells per MAC
        compute_power = total_ops_per_sec * energy_per_op_pj * 1e-12 * utilization

        # SRAM leakage + dynamic
        sram_area = estimate_sram_area_mm2(kpu.total_sram_bytes, process_nm)
        sram_power = sram_area * 0.3 * (tech.vdd_v / 0.90) ** 2  # 0.3 W/mm²

        # NoC + control overhead
        noc_power = kpu.noc.num_routers * 0.01 * (tech.vdd_v / 0.90) ** 2

        return (compute_power + sram_power + noc_power) * self.count

    def peak_gops(self) -> float:
        kpu = self._get_kpu_config()
        # TOPS * 1000 = GOPS
        return kpu.peak_tops_int8 * 1000.0 * self.count


# ---------------------------------------------------------------------------
# Infrastructure IPs
# ---------------------------------------------------------------------------


class MemoryControllerConfig(IPBlockConfig):
    """DRAM memory controller IP block."""

    block_type: Literal["memory_ctrl"] = "memory_ctrl"
    dram_type: str = "LPDDR4X"
    channels: int = 2
    bandwidth_gbps_per_channel: float = 6.4
    capacity_gb: float = 4.0

    def estimate_area_mm2(self, process_nm: int) -> float:
        # Memory controller area: PHY + digital logic, scales with process
        # ~0.5mm² per channel at 28nm reference
        ref_area_per_channel = 0.5  # mm² at 28nm
        scale = (process_nm / 28) ** 2
        return ref_area_per_channel * self.channels * scale * self.count

    def estimate_power_watts(self, process_nm: int, utilization: float = 0.8) -> float:
        tech = get_technology(process_nm)
        # ~0.3W per channel at 28nm, scales with voltage
        ref_power = 0.3
        voltage_factor = (tech.vdd_v / 0.90) ** 2
        return ref_power * self.channels * voltage_factor * utilization * self.count

    @property
    def total_bandwidth_gbps(self) -> float:
        return self.channels * self.bandwidth_gbps_per_channel

    def peak_gops(self) -> float:
        return 0.0  # memory controller doesn't do compute


class NoCIPConfig(IPBlockConfig):
    """Network-on-Chip IP block."""

    block_type: Literal["noc"] = "noc"
    topology: str = "mesh_2d"
    link_width_bits: int = 256
    frequency_mhz: float = 1000.0
    num_routers: int = 16

    def estimate_area_mm2(self, process_nm: int) -> float:
        # ~0.02mm² per router at 28nm
        ref_router_area = 0.02  # mm² at 28nm
        scale = (process_nm / 28) ** 2
        # Link width also adds area for wire drivers
        width_factor = self.link_width_bits / 256.0
        return ref_router_area * self.num_routers * scale * width_factor * self.count

    def estimate_power_watts(self, process_nm: int, utilization: float = 0.8) -> float:
        tech = get_technology(process_nm)
        # ~0.01W per router at 28nm reference
        ref_power = 0.01
        voltage_factor = (tech.vdd_v / 0.90) ** 2
        freq_factor = self.frequency_mhz / 1000.0
        return (
            ref_power * self.num_routers * voltage_factor * freq_factor * utilization * self.count
        )

    def peak_gops(self) -> float:
        return 0.0


class IOSubsystemConfig(IPBlockConfig):
    """I/O subsystem: MIPI CSI, SPI, I2C, UART, GPIO."""

    block_type: Literal["io"] = "io"
    mipi_csi_lanes: int = 2
    spi_ports: int = 2
    i2c_ports: int = 4
    uart_ports: int = 2
    gpio_count: int = 32

    def estimate_area_mm2(self, process_nm: int) -> float:
        scale = (process_nm / 28) ** 2
        # Fixed area estimates at 28nm
        mipi_area = self.mipi_csi_lanes * 0.3  # mm² per lane
        serial_area = (self.spi_ports + self.i2c_ports + self.uart_ports) * 0.05
        gpio_area = self.gpio_count * 0.005
        return (mipi_area + serial_area + gpio_area) * scale * self.count

    def estimate_power_watts(self, process_nm: int, utilization: float = 0.8) -> float:
        tech = get_technology(process_nm)
        voltage_factor = (tech.vdd_v / 0.90) ** 2
        # I/O power is mostly PHY-driven, relatively constant
        base_power = 0.3 + self.mipi_csi_lanes * 0.1
        return base_power * voltage_factor * utilization * self.count

    def peak_gops(self) -> float:
        return 0.0


class ISPConfig(IPBlockConfig):
    """Image Signal Processor IP block."""

    block_type: Literal["isp"] = "isp"
    pipeline_stages: int = 8
    max_resolution: str = "4K"

    def estimate_area_mm2(self, process_nm: int) -> float:
        scale = (process_nm / 28) ** 2

        # ISP area depends on pipeline depth and resolution
        res_factor = {"720p": 0.5, "1080p": 0.8, "4K": 1.5, "8K": 3.0}.get(self.max_resolution, 1.0)
        # ~0.3mm² per pipeline stage at 28nm, 4K reference
        base_area = 0.3 * self.pipeline_stages * res_factor
        return base_area * scale * self.count

    def estimate_power_watts(self, process_nm: int, utilization: float = 0.8) -> float:
        tech = get_technology(process_nm)
        voltage_factor = (tech.vdd_v / 0.90) ** 2
        res_factor = {"720p": 0.3, "1080p": 0.5, "4K": 1.0, "8K": 2.0}.get(self.max_resolution, 0.5)
        base_power = 0.5 * res_factor
        return base_power * voltage_factor * utilization * self.count

    def peak_gops(self) -> float:
        return 0.0  # ISP is fixed-function, not measured in GOPS


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------


ARM_A78_PRESET = CPUCoreConfig(
    name="ARM Cortex-A78",
    isa="ARM_A78",
    frequency_mhz=2000.0,
    pipeline_stages=13,
    issue_width=4,
    l1i_bytes=64 * 1024,
    l1d_bytes=64 * 1024,
    l2_bytes=256 * 1024,
    has_fpu=True,
    has_simd=True,
)

ARM_A55_PRESET = CPUCoreConfig(
    name="ARM Cortex-A55",
    isa="ARM_A55",
    frequency_mhz=1400.0,
    pipeline_stages=8,
    issue_width=2,
    l1i_bytes=32 * 1024,
    l1d_bytes=32 * 1024,
    l2_bytes=128 * 1024,
    has_fpu=True,
    has_simd=True,
)

RISCV_SMALL_PRESET = CPUCoreConfig(
    name="RISC-V RV64 Minimal",
    isa="RISCV_RV64",
    frequency_mhz=1000.0,
    pipeline_stages=5,
    issue_width=2,
    l1i_bytes=16 * 1024,
    l1d_bytes=16 * 1024,
    l2_bytes=64 * 1024,
    has_fpu=True,
    has_simd=False,
)

SMALL_GPU_PRESET = GPUClusterConfig(
    name="Small GPU (4 SM)",
    shader_cores=4,
    frequency_mhz=800.0,
    fp32_units_per_core=128,
    tensor_units=0,
    l1_bytes_per_core=64 * 1024,
    shared_l2_bytes=512 * 1024,
    memory_bus_width_bits=128,
)

MEDIUM_GPU_PRESET = GPUClusterConfig(
    name="Medium GPU (8 SM)",
    shader_cores=8,
    frequency_mhz=1200.0,
    fp32_units_per_core=128,
    tensor_units=16,
    l1_bytes_per_core=96 * 1024,
    shared_l2_bytes=1024 * 1024,
    memory_bus_width_bits=256,
)


def kpu_preset_block(preset_name: str, count: int = 1) -> KPUBlockConfig:
    """Create a KPUBlockConfig from an existing KPU_PRESETS name."""
    from embodied_ai_architect.graphs.kpu_config import KPU_PRESETS

    if preset_name not in KPU_PRESETS:
        raise ValueError(
            f"Unknown KPU preset '{preset_name}'. " f"Available: {list(KPU_PRESETS.keys())}"
        )
    kpu = KPU_PRESETS[preset_name]
    return KPUBlockConfig(
        name=f"KPU ({preset_name})",
        kpu_preset=preset_name,
        array_rows=kpu.array_rows,
        array_cols=kpu.array_cols,
        compute_frequency_mhz=kpu.compute_tile.frequency_mhz,
        systolic_rows=kpu.compute_tile.array_rows,
        systolic_cols=kpu.compute_tile.array_cols,
        count=count,
    )


# Convenient infrastructure defaults

LPDDR4X_CONTROLLER_PRESET = MemoryControllerConfig(
    name="LPDDR4X Controller",
    channels=4,
    bandwidth_gbps_per_channel=6.4,
    capacity_gb=4.0,
)

MESH_NOC_PRESET = NoCIPConfig(
    name="Mesh NoC",
    topology="mesh_2d",
    link_width_bits=256,
    frequency_mhz=1000.0,
    num_routers=16,
)

DEFAULT_IO_PRESET = IOSubsystemConfig(
    name="Edge I/O",
    mipi_csi_lanes=2,
    spi_ports=2,
    i2c_ports=4,
    uart_ports=2,
    gpio_count=32,
)

DEFAULT_ISP_PRESET = ISPConfig(
    name="Vision ISP",
    pipeline_stages=8,
    max_resolution="4K",
)


# ---------------------------------------------------------------------------
# SoC Composition
# ---------------------------------------------------------------------------


class SoCComposition(BaseModel):
    """Complete die composition from IP blocks.

    Aggregates area, power, and peak GOPS across all blocks,
    including interconnect overhead.
    """

    name: str
    process_nm: int = 28
    blocks: list[IPBlockConfig] = Field(default_factory=list)
    interconnect_overhead: float = Field(
        default=0.20,
        description="Fraction of core area added for routing/interconnect",
    )

    model_config = {"arbitrary_types_allowed": True}

    def total_area_mm2(self) -> float:
        """Total die area including interconnect overhead."""
        core_area = sum(b.estimate_area_mm2(self.process_nm) for b in self.blocks)
        return core_area * (1.0 + self.interconnect_overhead)

    def total_power_watts(self, utilization: float = 0.8) -> float:
        """Total power across all blocks."""
        return sum(b.estimate_power_watts(self.process_nm, utilization) for b in self.blocks)

    def total_peak_gops(self) -> float:
        """Sum of peak GOPS across all compute blocks."""
        return sum(b.peak_gops() for b in self.blocks)

    def block_summary(self) -> dict[str, Any]:
        """Summary dict of blocks by type with area/power breakdown."""
        summary: dict[str, Any] = {"blocks": [], "totals": {}}
        for b in self.blocks:
            area = b.estimate_area_mm2(self.process_nm)
            power = b.estimate_power_watts(self.process_nm)
            summary["blocks"].append(
                {
                    "name": b.name,
                    "block_type": b.block_type,
                    "count": b.count,
                    "area_mm2": round(area, 4),
                    "power_watts": round(power, 3),
                    "peak_gops": round(b.peak_gops(), 2),
                }
            )
        summary["totals"] = {
            "area_mm2": round(self.total_area_mm2(), 2),
            "power_watts": round(self.total_power_watts(), 2),
            "peak_gops": round(self.total_peak_gops(), 2),
        }
        return summary

    def to_ip_block_dicts(self) -> list[dict[str, Any]]:
        """Serialize blocks as plain dicts for state storage.

        Each dict has a 'block_type' discriminator for reconstruction.
        """
        result = []
        for b in self.blocks:
            d = b.model_dump()
            d["_area_mm2"] = round(b.estimate_area_mm2(self.process_nm), 4)
            d["_power_watts"] = round(b.estimate_power_watts(self.process_nm), 3)
            d["_peak_gops"] = round(b.peak_gops(), 2)
            result.append(d)
        return result
