"""Manufacturing cost model for custom SoC design.

Replaces static BOM cost lookup with a physics-based model that accounts for
wafer cost, die yield, packaging, test, and NRE amortization.  This makes
cost a *derived* quantity from die area, process node, and production volume —
all quantities the optimizer can change.

Key formulas:
    dies_per_wafer: circular wafer packing (300mm, 3mm edge exclusion)
    yield:          Murphy's model — Y = ((1 - e^(-D·A)) / (D·A))²
    unit_cost:      die_cost + package + test + NRE/volume

Usage:
    from embodied_ai_architect.graphs.manufacturing import estimate_manufacturing_cost

    mfg = estimate_manufacturing_cost(die_area_mm2=30, process_nm=28, volume=100_000)
    print(mfg.total_unit_cost_usd)  # ~$77 (NRE-dominated at 100K volume)
"""

from __future__ import annotations

import math

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Process economics database
# ---------------------------------------------------------------------------

PROCESS_ECONOMICS: dict[int, dict[str, float]] = {
    # process_nm: {wafer_cost_usd, defect_density_per_cm2, mask_set_cost_usd,
    #              design_nre_usd, test_nre_usd}
    #
    # Defect density *increases* for leading-edge nodes (more complex process,
    # less mature).  Mature nodes (180nm) have very low defect density.
    #
    # NRE has three components:
    #   mask_set_cost — photomask fabrication (scales steeply with EUV layers)
    #   design_nre    — EDA licenses, engineering, IP licensing, verification
    #                   (typically 2-5× mask cost; dominates total NRE)
    #   test_nre      — probe card fabrication, test program development,
    #                   reliability qualification ($100K–$1M+)
    180: {
        "wafer_cost_usd": 1_500,
        "defect_density": 0.02,
        "mask_set_cost_usd": 50_000,
        "design_nre_usd": 100_000,
        "test_nre_usd": 50_000,
    },
    130: {
        "wafer_cost_usd": 2_000,
        "defect_density": 0.03,
        "mask_set_cost_usd": 100_000,
        "design_nre_usd": 250_000,
        "test_nre_usd": 75_000,
    },
    90: {
        "wafer_cost_usd": 2_500,
        "defect_density": 0.04,
        "mask_set_cost_usd": 200_000,
        "design_nre_usd": 500_000,
        "test_nre_usd": 100_000,
    },
    65: {
        "wafer_cost_usd": 3_000,
        "defect_density": 0.05,
        "mask_set_cost_usd": 500_000,
        "design_nre_usd": 1_500_000,
        "test_nre_usd": 200_000,
    },
    40: {
        "wafer_cost_usd": 3_500,
        "defect_density": 0.06,
        "mask_set_cost_usd": 1_000_000,
        "design_nre_usd": 3_000_000,
        "test_nre_usd": 300_000,
    },
    28: {
        "wafer_cost_usd": 4_000,
        "defect_density": 0.08,
        "mask_set_cost_usd": 2_000_000,
        "design_nre_usd": 5_000_000,
        "test_nre_usd": 500_000,
    },
    22: {
        "wafer_cost_usd": 5_000,
        "defect_density": 0.09,
        "mask_set_cost_usd": 3_000_000,
        "design_nre_usd": 8_000_000,
        "test_nre_usd": 600_000,
    },
    16: {
        "wafer_cost_usd": 6_000,
        "defect_density": 0.10,
        "mask_set_cost_usd": 5_000_000,
        "design_nre_usd": 15_000_000,
        "test_nre_usd": 800_000,
    },
    14: {
        "wafer_cost_usd": 6_500,
        "defect_density": 0.10,
        "mask_set_cost_usd": 6_000_000,
        "design_nre_usd": 18_000_000,
        "test_nre_usd": 800_000,
    },
    12: {
        "wafer_cost_usd": 7_000,
        "defect_density": 0.11,
        "mask_set_cost_usd": 8_000_000,
        "design_nre_usd": 25_000_000,
        "test_nre_usd": 1_000_000,
    },
    10: {
        "wafer_cost_usd": 8_000,
        "defect_density": 0.12,
        "mask_set_cost_usd": 10_000_000,
        "design_nre_usd": 30_000_000,
        "test_nre_usd": 1_000_000,
    },
    7: {
        "wafer_cost_usd": 10_000,
        "defect_density": 0.14,
        "mask_set_cost_usd": 15_000_000,
        "design_nre_usd": 50_000_000,
        "test_nre_usd": 1_500_000,
    },
    5: {
        "wafer_cost_usd": 13_000,
        "defect_density": 0.16,
        "mask_set_cost_usd": 25_000_000,
        "design_nre_usd": 100_000_000,
        "test_nre_usd": 2_000_000,
    },
    4: {
        "wafer_cost_usd": 15_000,
        "defect_density": 0.18,
        "mask_set_cost_usd": 30_000_000,
        "design_nre_usd": 120_000_000,
        "test_nre_usd": 2_500_000,
    },
    3: {
        "wafer_cost_usd": 18_000,
        "defect_density": 0.20,
        "mask_set_cost_usd": 40_000_000,
        "design_nre_usd": 150_000_000,
        "test_nre_usd": 3_000_000,
    },
    2: {
        "wafer_cost_usd": 22_000,
        "defect_density": 0.22,
        "mask_set_cost_usd": 50_000_000,
        "design_nre_usd": 200_000_000,
        "test_nre_usd": 4_000_000,
    },
}

# Package costs per unit
PACKAGE_COSTS: dict[str, float] = {
    "QFN": 0.30,  # small, cheap, limited I/O
    "BGA": 1.50,  # standard for mid-range SoCs
    "FCBGA": 3.00,  # flip-chip, high-performance
    "WLCSP": 0.80,  # wafer-level, compact
}

# Test cost per die (simplified)
TEST_COST_USD = 0.10


# ---------------------------------------------------------------------------
# Manufacturing cost breakdown
# ---------------------------------------------------------------------------


class ManufacturingCostBreakdown(BaseModel):
    """Itemized manufacturing cost for a single die.

    Costs split into variable (per-unit) and fixed (NRE amortized over volume):
      Variable: die_cost + package_cost + test_cost
      Fixed:    nre_per_unit = (mask_set + design_nre + test_nre) / volume
    """

    die_area_mm2: float = Field(..., description="Die area in mm²")
    process_nm: int = Field(..., description="Process node in nm")
    volume: int = Field(..., description="Production volume")
    dies_per_wafer: int = Field(..., description="Good die sites per wafer")
    yield_percent: float = Field(..., description="Manufacturing yield (0-100)")
    die_cost_usd: float = Field(..., description="Wafer cost / (dies × yield)")
    package_cost_usd: float = Field(..., description="Package cost per unit")
    test_cost_usd: float = Field(..., description="Test cost per die")
    nre_per_unit_usd: float = Field(..., description="Total NRE (masks + design + test) / volume")
    total_unit_cost_usd: float = Field(..., description="die + package + test + NRE per unit")


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

# 300mm wafer: radius 150mm, 3mm edge exclusion → effective R = 147mm
_WAFER_RADIUS_MM = 147.0


def dies_per_wafer(die_area_mm2: float) -> int:
    """Estimate die count on a 300mm wafer using circular packing formula.

    N = (pi * R²) / A  -  (pi * 2R) / sqrt(2A)

    where R = 147mm (300mm wafer with 3mm edge exclusion), A = die area.
    """
    if die_area_mm2 <= 0:
        return 0
    r = _WAFER_RADIUS_MM
    a = die_area_mm2
    n = (math.pi * r * r) / a - (math.pi * 2 * r) / math.sqrt(2 * a)
    return max(int(n), 1)


def murphy_yield(defect_density: float, die_area_cm2: float) -> float:
    """Murphy's yield model — industry standard, more conservative than Poisson.

    Y = ((1 - e^(-D·A)) / (D·A))²

    Args:
        defect_density: Defects per cm² for the process node.
        die_area_cm2: Die area in cm².

    Returns:
        Yield as a fraction (0.0 – 1.0).
    """
    da = defect_density * die_area_cm2
    if da <= 0:
        return 1.0
    y = (1.0 - math.exp(-da)) / da
    return y * y


def _get_process_economics(process_nm: int) -> dict[str, float]:
    """Look up process economics, falling back to nearest available node."""
    if process_nm in PROCESS_ECONOMICS:
        return PROCESS_ECONOMICS[process_nm]
    nearest = min(PROCESS_ECONOMICS.keys(), key=lambda n: abs(n - process_nm))
    return PROCESS_ECONOMICS[nearest]


def estimate_manufacturing_cost(
    die_area_mm2: float,
    process_nm: int,
    volume: int,
    package_type: str = "BGA",
) -> ManufacturingCostBreakdown:
    """Estimate per-unit manufacturing cost from die area, process, and volume.

    Args:
        die_area_mm2: Total die area in mm².
        process_nm: Target process node in nm.
        volume: Production volume for NRE amortization.
        package_type: Package type (QFN, BGA, FCBGA, WLCSP).

    Returns:
        ManufacturingCostBreakdown with itemized costs.
    """
    econ = _get_process_economics(process_nm)
    wafer_cost = econ["wafer_cost_usd"]
    defect_density = econ["defect_density"]
    mask_set_cost = econ["mask_set_cost_usd"]
    design_nre = econ.get("design_nre_usd", 0)
    test_nre = econ.get("test_nre_usd", 0)

    n_dies = dies_per_wafer(die_area_mm2)
    die_area_cm2 = die_area_mm2 / 100.0  # 1 cm² = 100 mm²
    y = murphy_yield(defect_density, die_area_cm2)

    good_dies = n_dies * y
    die_cost = wafer_cost / good_dies if good_dies > 0 else wafer_cost

    package_cost = PACKAGE_COSTS.get(package_type, PACKAGE_COSTS["BGA"])
    test_cost = TEST_COST_USD
    total_nre = mask_set_cost + design_nre + test_nre
    nre_per_unit = total_nre / max(volume, 1)

    total = die_cost + package_cost + test_cost + nre_per_unit

    return ManufacturingCostBreakdown(
        die_area_mm2=round(die_area_mm2, 2),
        process_nm=process_nm,
        volume=volume,
        dies_per_wafer=n_dies,
        yield_percent=round(y * 100, 1),
        die_cost_usd=round(die_cost, 2),
        package_cost_usd=round(package_cost, 2),
        test_cost_usd=round(test_cost, 2),
        nre_per_unit_usd=round(nre_per_unit, 2),
        total_unit_cost_usd=round(total, 2),
    )
