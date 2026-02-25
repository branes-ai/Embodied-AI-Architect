"""Technology node database for process-aware area/timing estimation.

Ports TECHNOLOGY_NODES from experiments/langgraph/soc_optimizer/workflow.py
and adds SRAM density data from chip_area_estimates.md.

Usage:
    from embodied_ai_architect.graphs.technology import (
        get_technology, estimate_area_um2, estimate_sram_area_mm2,
    )

    tech = get_technology(28)
    area = estimate_sram_area_mm2(256 * 1024, 28)  # 256KB at 28nm
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# SRAM density by process node (Mb/mm²)
# Source: chip_area_estimates.md and industry references
# ---------------------------------------------------------------------------

SRAM_DENSITY: dict[int, float] = {
    2: 38.1,
    3: 42.0,
    5: 48.0,
    7: 45.0,
    10: 35.0,
    12: 32.0,
    14: 25.0,
    16: 18.0,
    22: 12.0,
    28: 8.0,
    40: 5.0,
    65: 2.5,
    90: 1.5,
    130: 0.8,
    180: 0.5,
}


# ---------------------------------------------------------------------------
# Technology nodes — full 2nm–180nm process database
# ---------------------------------------------------------------------------

TECHNOLOGY_NODES: dict[str, dict[str, Any]] = {
    "2nm": {
        "name": "2nm GAA (N2)",
        "vendor": "TSMC/Intel/Samsung",
        "generation": "GAA Nanosheet",
        "year": 2025,
        "cell_area_um2": 0.020,
        "gate_delay_ps": 5,
        "energy_per_cell_pj": 0.003,
        "vdd_v": 0.55,
        "reference_areas": {
            "INV_X1": 0.008,
            "NAND2_X1": 0.012,
            "NOR2_X1": 0.014,
            "DFF_X1": 0.080,
            "REG_32bit": 2.6,
            "ADDER_32bit": 6.0,
            "ADDER_32bit_CLA": 10.0,
            "MUL_32bit": 320.0,
            "MUL_32bit_BOOTH": 200.0,
            "SRAM_1KB": 100.0,
            "SRAM_32KB": 3200.0,
            "ALU_32bit": 40.0,
            "FPU_32bit": 800.0,
        },
    },
    "3nm": {
        "name": "3nm GAA (N3)",
        "vendor": "TSMC/Samsung",
        "generation": "GAA Nanosheet",
        "year": 2022,
        "cell_area_um2": 0.025,
        "gate_delay_ps": 6,
        "energy_per_cell_pj": 0.004,
        "vdd_v": 0.60,
        "reference_areas": {
            "INV_X1": 0.010,
            "NAND2_X1": 0.015,
            "NOR2_X1": 0.018,
            "DFF_X1": 0.100,
            "REG_32bit": 3.2,
            "ADDER_32bit": 7.5,
            "ADDER_32bit_CLA": 12.5,
            "MUL_32bit": 400.0,
            "MUL_32bit_BOOTH": 250.0,
            "SRAM_1KB": 125.0,
            "SRAM_32KB": 4000.0,
            "ALU_32bit": 50.0,
            "FPU_32bit": 1000.0,
        },
    },
    "4nm": {
        "name": "4nm EUV FinFET (N4)",
        "vendor": "TSMC/Samsung",
        "generation": "EUV FinFET",
        "year": 2021,
        "cell_area_um2": 0.030,
        "gate_delay_ps": 7,
        "energy_per_cell_pj": 0.005,
        "vdd_v": 0.65,
        "reference_areas": {
            "INV_X1": 0.012,
            "NAND2_X1": 0.018,
            "NOR2_X1": 0.021,
            "DFF_X1": 0.120,
            "REG_32bit": 3.8,
            "ADDER_32bit": 9.0,
            "ADDER_32bit_CLA": 15.0,
            "MUL_32bit": 480.0,
            "MUL_32bit_BOOTH": 300.0,
            "SRAM_1KB": 150.0,
            "SRAM_32KB": 4800.0,
            "ALU_32bit": 60.0,
            "FPU_32bit": 1200.0,
        },
    },
    "5nm": {
        "name": "5nm EUV FinFET (N5)",
        "vendor": "TSMC/Samsung",
        "generation": "EUV FinFET",
        "year": 2020,
        "cell_area_um2": 0.040,
        "gate_delay_ps": 8,
        "energy_per_cell_pj": 0.007,
        "vdd_v": 0.65,
        "reference_areas": {
            "INV_X1": 0.016,
            "NAND2_X1": 0.024,
            "NOR2_X1": 0.028,
            "DFF_X1": 0.160,
            "REG_32bit": 5.1,
            "ADDER_32bit": 12.0,
            "ADDER_32bit_CLA": 20.0,
            "MUL_32bit": 640.0,
            "MUL_32bit_BOOTH": 400.0,
            "SRAM_1KB": 200.0,
            "SRAM_32KB": 6400.0,
            "ALU_32bit": 80.0,
            "FPU_32bit": 1600.0,
        },
    },
    "7nm": {
        "name": "7nm FinFET (N7)",
        "vendor": "TSMC/Samsung",
        "generation": "DUV/EUV FinFET",
        "year": 2018,
        "cell_area_um2": 0.050,
        "gate_delay_ps": 10,
        "energy_per_cell_pj": 0.010,
        "vdd_v": 0.70,
        "reference_areas": {
            "INV_X1": 0.020,
            "NAND2_X1": 0.030,
            "NOR2_X1": 0.035,
            "DFF_X1": 0.200,
            "REG_32bit": 6.4,
            "ADDER_32bit": 15.0,
            "ADDER_32bit_CLA": 25.0,
            "MUL_32bit": 800.0,
            "MUL_32bit_BOOTH": 500.0,
            "SRAM_1KB": 250.0,
            "SRAM_32KB": 8000.0,
            "ALU_32bit": 100.0,
            "FPU_32bit": 2000.0,
        },
    },
    "8nm": {
        "name": "8nm FinFET (N8)",
        "vendor": "Samsung",
        "generation": "DUV FinFET",
        "year": 2018,
        "cell_area_um2": 0.060,
        "gate_delay_ps": 11,
        "energy_per_cell_pj": 0.012,
        "vdd_v": 0.70,
        "reference_areas": {
            "INV_X1": 0.024,
            "NAND2_X1": 0.036,
            "NOR2_X1": 0.042,
            "DFF_X1": 0.240,
            "REG_32bit": 7.7,
            "ADDER_32bit": 18.0,
            "ADDER_32bit_CLA": 30.0,
            "MUL_32bit": 960.0,
            "MUL_32bit_BOOTH": 600.0,
            "SRAM_1KB": 300.0,
            "SRAM_32KB": 9600.0,
            "ALU_32bit": 120.0,
            "FPU_32bit": 2400.0,
        },
    },
    "10nm": {
        "name": "10nm FinFET (N10)",
        "vendor": "TSMC/Samsung/Intel",
        "generation": "FinFET",
        "year": 2016,
        "cell_area_um2": 0.070,
        "gate_delay_ps": 12,
        "energy_per_cell_pj": 0.015,
        "vdd_v": 0.75,
        "reference_areas": {
            "INV_X1": 0.028,
            "NAND2_X1": 0.042,
            "NOR2_X1": 0.049,
            "DFF_X1": 0.280,
            "REG_32bit": 9.0,
            "ADDER_32bit": 21.0,
            "ADDER_32bit_CLA": 35.0,
            "MUL_32bit": 1120.0,
            "MUL_32bit_BOOTH": 700.0,
            "SRAM_1KB": 350.0,
            "SRAM_32KB": 11200.0,
            "ALU_32bit": 140.0,
            "FPU_32bit": 2800.0,
        },
    },
    "12nm": {
        "name": "12nm FinFET (N12)",
        "vendor": "TSMC/GlobalFoundries",
        "generation": "FinFET",
        "year": 2017,
        "cell_area_um2": 0.085,
        "gate_delay_ps": 13,
        "energy_per_cell_pj": 0.018,
        "vdd_v": 0.75,
        "reference_areas": {
            "INV_X1": 0.034,
            "NAND2_X1": 0.051,
            "NOR2_X1": 0.060,
            "DFF_X1": 0.340,
            "REG_32bit": 10.9,
            "ADDER_32bit": 25.5,
            "ADDER_32bit_CLA": 42.5,
            "MUL_32bit": 1360.0,
            "MUL_32bit_BOOTH": 850.0,
            "SRAM_1KB": 425.0,
            "SRAM_32KB": 13600.0,
            "ALU_32bit": 170.0,
            "FPU_32bit": 3400.0,
        },
    },
    "14nm": {
        "name": "14nm FinFET (N14)",
        "vendor": "Intel/TSMC/Samsung",
        "generation": "FinFET",
        "year": 2014,
        "cell_area_um2": 0.100,
        "gate_delay_ps": 15,
        "energy_per_cell_pj": 0.020,
        "vdd_v": 0.80,
        "reference_areas": {
            "INV_X1": 0.040,
            "NAND2_X1": 0.060,
            "NOR2_X1": 0.070,
            "DFF_X1": 0.400,
            "REG_32bit": 12.8,
            "ADDER_32bit": 30.0,
            "ADDER_32bit_CLA": 50.0,
            "MUL_32bit": 1600.0,
            "MUL_32bit_BOOTH": 1000.0,
            "SRAM_1KB": 500.0,
            "SRAM_32KB": 16000.0,
            "ALU_32bit": 200.0,
            "FPU_32bit": 4000.0,
        },
    },
    "16nm": {
        "name": "16nm FinFET (N16)",
        "vendor": "TSMC",
        "generation": "FinFET",
        "year": 2013,
        "cell_area_um2": 0.120,
        "gate_delay_ps": 17,
        "energy_per_cell_pj": 0.025,
        "vdd_v": 0.80,
        "reference_areas": {
            "INV_X1": 0.048,
            "NAND2_X1": 0.072,
            "NOR2_X1": 0.084,
            "DFF_X1": 0.480,
            "REG_32bit": 15.4,
            "ADDER_32bit": 36.0,
            "ADDER_32bit_CLA": 60.0,
            "MUL_32bit": 1920.0,
            "MUL_32bit_BOOTH": 1200.0,
            "SRAM_1KB": 600.0,
            "SRAM_32KB": 19200.0,
            "ALU_32bit": 240.0,
            "FPU_32bit": 4800.0,
        },
    },
    "22nm": {
        "name": "22nm Planar/TriGate",
        "vendor": "Intel/GlobalFoundries",
        "generation": "Planar/TriGate",
        "year": 2012,
        "cell_area_um2": 0.160,
        "gate_delay_ps": 20,
        "energy_per_cell_pj": 0.035,
        "vdd_v": 0.85,
        "reference_areas": {
            "INV_X1": 0.064,
            "NAND2_X1": 0.096,
            "NOR2_X1": 0.112,
            "DFF_X1": 0.640,
            "REG_32bit": 20.5,
            "ADDER_32bit": 48.0,
            "ADDER_32bit_CLA": 80.0,
            "MUL_32bit": 2560.0,
            "MUL_32bit_BOOTH": 1600.0,
            "SRAM_1KB": 800.0,
            "SRAM_32KB": 25600.0,
            "ALU_32bit": 320.0,
            "FPU_32bit": 6400.0,
        },
    },
    "28nm": {
        "name": "28nm Planar CMOS",
        "vendor": "TSMC/GlobalFoundries/Samsung",
        "generation": "Planar CMOS",
        "year": 2011,
        "cell_area_um2": 0.200,
        "gate_delay_ps": 25,
        "energy_per_cell_pj": 0.050,
        "vdd_v": 0.90,
        "reference_areas": {
            "INV_X1": 0.080,
            "NAND2_X1": 0.120,
            "NOR2_X1": 0.140,
            "DFF_X1": 0.800,
            "REG_32bit": 25.6,
            "ADDER_32bit": 60.0,
            "ADDER_32bit_CLA": 100.0,
            "MUL_32bit": 3200.0,
            "MUL_32bit_BOOTH": 2000.0,
            "SRAM_1KB": 1000.0,
            "SRAM_32KB": 32000.0,
            "ALU_32bit": 400.0,
            "FPU_32bit": 8000.0,
        },
    },
    "40nm": {
        "name": "40nm Planar CMOS",
        "vendor": "TSMC/GlobalFoundries",
        "generation": "Planar CMOS",
        "year": 2008,
        "cell_area_um2": 0.350,
        "gate_delay_ps": 35,
        "energy_per_cell_pj": 0.080,
        "vdd_v": 1.00,
        "reference_areas": {
            "INV_X1": 0.140,
            "NAND2_X1": 0.210,
            "NOR2_X1": 0.245,
            "DFF_X1": 1.400,
            "REG_32bit": 44.8,
            "ADDER_32bit": 105.0,
            "ADDER_32bit_CLA": 175.0,
            "MUL_32bit": 5600.0,
            "MUL_32bit_BOOTH": 3500.0,
            "SRAM_1KB": 1750.0,
            "SRAM_32KB": 56000.0,
            "ALU_32bit": 700.0,
            "FPU_32bit": 14000.0,
        },
    },
    "65nm": {
        "name": "65nm Planar CMOS",
        "vendor": "TSMC/GlobalFoundries",
        "generation": "Planar CMOS",
        "year": 2006,
        "cell_area_um2": 0.600,
        "gate_delay_ps": 50,
        "energy_per_cell_pj": 0.120,
        "vdd_v": 1.10,
        "reference_areas": {
            "INV_X1": 0.240,
            "NAND2_X1": 0.360,
            "NOR2_X1": 0.420,
            "DFF_X1": 2.400,
            "REG_32bit": 76.8,
            "ADDER_32bit": 180.0,
            "ADDER_32bit_CLA": 300.0,
            "MUL_32bit": 9600.0,
            "MUL_32bit_BOOTH": 6000.0,
            "SRAM_1KB": 3000.0,
            "SRAM_32KB": 96000.0,
            "ALU_32bit": 1200.0,
            "FPU_32bit": 24000.0,
        },
    },
    "90nm": {
        "name": "90nm Planar CMOS",
        "vendor": "TSMC/IBM",
        "generation": "Planar CMOS",
        "year": 2004,
        "cell_area_um2": 1.000,
        "gate_delay_ps": 70,
        "energy_per_cell_pj": 0.200,
        "vdd_v": 1.20,
        "reference_areas": {
            "INV_X1": 0.400,
            "NAND2_X1": 0.600,
            "NOR2_X1": 0.700,
            "DFF_X1": 4.000,
            "REG_32bit": 128.0,
            "ADDER_32bit": 300.0,
            "ADDER_32bit_CLA": 500.0,
            "MUL_32bit": 16000.0,
            "MUL_32bit_BOOTH": 10000.0,
            "SRAM_1KB": 5000.0,
            "SRAM_32KB": 160000.0,
            "ALU_32bit": 2000.0,
            "FPU_32bit": 40000.0,
        },
    },
    "130nm": {
        "name": "130nm Planar CMOS (SKY130)",
        "vendor": "SkyWater/Google (Open PDK)",
        "generation": "Planar CMOS",
        "year": 2001,
        "open_pdk": True,
        "cell_area_um2": 1.600,
        "gate_delay_ps": 100,
        "energy_per_cell_pj": 0.350,
        "vdd_v": 1.80,
        "reference_areas": {
            "INV_X1": 0.640,
            "NAND2_X1": 0.960,
            "NOR2_X1": 1.120,
            "DFF_X1": 6.400,
            "REG_32bit": 204.8,
            "ADDER_32bit": 480.0,
            "ADDER_32bit_CLA": 800.0,
            "MUL_32bit": 25600.0,
            "MUL_32bit_BOOTH": 16000.0,
            "SRAM_1KB": 8000.0,
            "SRAM_32KB": 256000.0,
            "ALU_32bit": 3200.0,
            "FPU_32bit": 64000.0,
        },
    },
    "180nm": {
        "name": "180nm Planar CMOS (GF180)",
        "vendor": "GlobalFoundries (Open PDK)",
        "generation": "Planar CMOS",
        "year": 1999,
        "open_pdk": True,
        "cell_area_um2": 3.000,
        "gate_delay_ps": 150,
        "energy_per_cell_pj": 0.600,
        "vdd_v": 1.80,
        "reference_areas": {
            "INV_X1": 1.200,
            "NAND2_X1": 1.800,
            "NOR2_X1": 2.100,
            "DFF_X1": 12.000,
            "REG_32bit": 384.0,
            "ADDER_32bit": 900.0,
            "ADDER_32bit_CLA": 1500.0,
            "MUL_32bit": 48000.0,
            "MUL_32bit_BOOTH": 30000.0,
            "SRAM_1KB": 15000.0,
            "SRAM_32KB": 480000.0,
            "ALU_32bit": 6000.0,
            "FPU_32bit": 120000.0,
        },
    },
}


# Sorted list of available process nodes (nm)
_AVAILABLE_NODES = sorted([int(k.replace("nm", "")) for k in TECHNOLOGY_NODES.keys()])


# ---------------------------------------------------------------------------
# TechnologyNode model
# ---------------------------------------------------------------------------


class TechnologyNode(BaseModel):
    """Typed wrapper around a technology node entry."""

    name: str
    vendor: str
    generation: str
    year: int
    process_nm: int
    cell_area_um2: float
    gate_delay_ps: float
    energy_per_cell_pj: float
    vdd_v: float
    reference_areas: dict[str, float] = Field(default_factory=dict)
    open_pdk: bool = False
    sram_density_mb_per_mm2: float = 0.0


# ---------------------------------------------------------------------------
# Lookup functions
# ---------------------------------------------------------------------------


def get_technology(process_nm: int) -> TechnologyNode:
    """Get technology node parameters, using nearest-match lookup.

    Args:
        process_nm: Target process node in nm.

    Returns:
        TechnologyNode with parameters for the nearest available node.
    """
    key = f"{process_nm}nm"
    if key in TECHNOLOGY_NODES:
        data = TECHNOLOGY_NODES[key]
    else:
        # Find nearest node
        nearest = min(_AVAILABLE_NODES, key=lambda n: abs(n - process_nm))
        key = f"{nearest}nm"
        data = TECHNOLOGY_NODES[key]
        process_nm = nearest

    # Find SRAM density
    sram_density = _get_nearest_sram_density(process_nm)

    return TechnologyNode(
        name=data["name"],
        vendor=data["vendor"],
        generation=data["generation"],
        year=data["year"],
        process_nm=process_nm,
        cell_area_um2=data["cell_area_um2"],
        gate_delay_ps=data["gate_delay_ps"],
        energy_per_cell_pj=data["energy_per_cell_pj"],
        vdd_v=data["vdd_v"],
        reference_areas=data.get("reference_areas", {}),
        open_pdk=data.get("open_pdk", False),
        sram_density_mb_per_mm2=sram_density,
    )


def _get_nearest_sram_density(process_nm: int) -> float:
    """Get SRAM density for the nearest process node."""
    if process_nm in SRAM_DENSITY:
        return SRAM_DENSITY[process_nm]
    nodes = sorted(SRAM_DENSITY.keys())
    nearest = min(nodes, key=lambda n: abs(n - process_nm))
    return SRAM_DENSITY[nearest]


# ---------------------------------------------------------------------------
# Estimation functions
# ---------------------------------------------------------------------------


def estimate_area_um2(cell_count: int, process_nm: int) -> float:
    """Estimate area in um² from cell count and process node.

    Args:
        cell_count: Number of standard cells.
        process_nm: Target process node in nm.

    Returns:
        Estimated area in um².
    """
    tech = get_technology(process_nm)
    return cell_count * tech.cell_area_um2


def estimate_sram_area_mm2(size_bytes: int, process_nm: int) -> float:
    """Estimate SRAM area in mm² for a given capacity and process node.

    Args:
        size_bytes: SRAM capacity in bytes.
        process_nm: Target process node in nm.

    Returns:
        Estimated SRAM area in mm².
    """
    size_mb = size_bytes / (1024 * 1024)
    density = _get_nearest_sram_density(process_nm)
    if density <= 0:
        return 0.0
    return size_mb / density


def get_adjacent_nodes(process_nm: int) -> dict[str, int | None]:
    """Return the next smaller and larger available process nodes.

    Args:
        process_nm: Current process node in nm.

    Returns:
        Dict with "smaller" (next smaller nm value = more advanced) and
        "larger" (next larger nm value = more mature) process nodes,
        or None if at the boundary.
    """
    if process_nm in _AVAILABLE_NODES:
        idx = _AVAILABLE_NODES.index(process_nm)
    else:
        # Snap to nearest node
        nearest = min(_AVAILABLE_NODES, key=lambda n: abs(n - process_nm))
        idx = _AVAILABLE_NODES.index(nearest)

    smaller = _AVAILABLE_NODES[idx - 1] if idx > 0 else None
    larger = _AVAILABLE_NODES[idx + 1] if idx < len(_AVAILABLE_NODES) - 1 else None
    return {"smaller": smaller, "larger": larger}


def estimate_timing_ps(logic_levels: int, process_nm: int) -> float:
    """Estimate timing in picoseconds for a logic path.

    Args:
        logic_levels: Number of logic levels in the path.
        process_nm: Target process node in nm.

    Returns:
        Estimated path delay in picoseconds.
    """
    tech = get_technology(process_nm)
    return logic_levels * tech.gate_delay_ps
