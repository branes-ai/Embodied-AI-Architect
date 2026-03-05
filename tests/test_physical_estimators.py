"""Tests for parametric physical estimators (Phase 2)."""

from embodied_ai_architect.graphs.bom import BOMLevel
from embodied_ai_architect.graphs.physical_estimators import (
    COOLING_PARAMS,
    PACKAGE_PHYSICAL,
    compute_thermal_feasibility,
    estimate_die_physical,
    estimate_enclosure_physical,
    estimate_heatsink_physical,
    estimate_package_physical,
    estimate_pcb_physical,
    estimate_system_bom,
)

# ---------------------------------------------------------------------------
# Die estimator
# ---------------------------------------------------------------------------


def test_die_weight_100mm2():
    """100 mm^2 die at 775um thickness should weigh ~0.18g."""
    die = estimate_die_physical(100.0, 28)
    # area_cm2 = 1.0, thickness_cm = 0.0775, density = 2.33
    expected = 1.0 * 0.0775 * 2.33
    assert abs(die.weight_grams - expected) < 0.01
    assert die.die_area_mm2 == 100.0
    assert die.process_nm == 28
    assert die.dimensions is not None
    assert die.level == BOMLevel.DIE


def test_die_weight_monotonic():
    """Larger die -> heavier."""
    small = estimate_die_physical(25.0, 28)
    large = estimate_die_physical(100.0, 28)
    assert large.weight_grams > small.weight_grams


def test_die_dimensions_square():
    """Die dimensions should be sqrt(area) x sqrt(area)."""
    die = estimate_die_physical(100.0, 28)
    assert die.dimensions is not None
    assert abs(die.dimensions.length_mm - 10.0) < 0.01
    assert abs(die.dimensions.width_mm - 10.0) < 0.01


# ---------------------------------------------------------------------------
# Package estimator
# ---------------------------------------------------------------------------


def test_package_wraps_die():
    die = estimate_die_physical(50.0, 28)
    pkg = estimate_package_physical(die, "BGA")
    assert pkg.level == BOMLevel.PACKAGE
    assert pkg.package_type == "BGA"
    assert len(pkg.children) == 1
    assert pkg.children[0] is die


def test_bga_heavier_than_qfn():
    """BGA package overhead > QFN for same die."""
    die = estimate_die_physical(50.0, 28)
    qfn = estimate_package_physical(die, "QFN")
    bga = estimate_package_physical(die, "BGA")
    assert bga.total_weight_grams() > qfn.total_weight_grams()


def test_fcbga_larger_area_overhead():
    """FCBGA should have larger area overhead than WLCSP."""
    die = estimate_die_physical(50.0, 28)
    wlcsp = estimate_package_physical(die, "WLCSP")
    fcbga = estimate_package_physical(die, "FCBGA")
    assert fcbga.dimensions is not None
    assert wlcsp.dimensions is not None
    assert fcbga.dimensions.length_mm > wlcsp.dimensions.length_mm


def test_package_thermal_resistance():
    """Package should have non-zero theta_jc."""
    die = estimate_die_physical(50.0, 28)
    pkg = estimate_package_physical(die, "BGA")
    assert pkg.thermal_resistance_c_per_w == PACKAGE_PHYSICAL["BGA"]["theta_jc"]


# ---------------------------------------------------------------------------
# PCB estimator
# ---------------------------------------------------------------------------


def test_pcb_with_components():
    die = estimate_die_physical(50.0, 28)
    pkg = estimate_package_physical(die, "BGA")
    pcb = estimate_pcb_physical([pkg], layer_count=4)
    assert pcb.level == BOMLevel.PCB
    assert pcb.board_area_cm2 > 0
    assert pcb.weight_grams > 0
    assert len(pcb.children) == 1


def test_pcb_more_layers_heavier():
    die = estimate_die_physical(50.0, 28)
    pkg = estimate_package_physical(die, "BGA")
    pcb_4 = estimate_pcb_physical([pkg], layer_count=4)
    pcb_8 = estimate_pcb_physical([pkg], layer_count=8)
    assert pcb_8.weight_grams > pcb_4.weight_grams


def test_pcb_connectors_add_weight():
    die = estimate_die_physical(50.0, 28)
    pkg = estimate_package_physical(die, "BGA")
    pcb_0 = estimate_pcb_physical([pkg], connector_count=0)
    pcb_4 = estimate_pcb_physical([pkg], connector_count=4)
    assert pcb_4.weight_grams > pcb_0.weight_grams


# ---------------------------------------------------------------------------
# Heatsink estimator
# ---------------------------------------------------------------------------


def test_passive_heatsink_heavier_per_watt():
    """Passive cooling uses more mass per watt than active."""
    passive = estimate_heatsink_physical(10.0, "passive")
    active = estimate_heatsink_physical(10.0, "active_fan")
    # passive g/W > active g/W
    assert passive.weight_grams > active.weight_grams


def test_heatsink_scales_with_tdp():
    low = estimate_heatsink_physical(5.0, "passive")
    high = estimate_heatsink_physical(15.0, "passive")
    assert high.weight_grams > low.weight_grams
    assert high.total_volume_cm3() > low.total_volume_cm3()


def test_heatsink_thermal_resistance():
    hs = estimate_heatsink_physical(10.0, "passive")
    assert hs.thermal_resistance_c_per_w == COOLING_PARAMS["passive"]["theta_sa"]


# ---------------------------------------------------------------------------
# Enclosure estimator
# ---------------------------------------------------------------------------


def test_enclosure_aluminum():
    die = estimate_die_physical(50.0, 28)
    pkg = estimate_package_physical(die, "BGA")
    pcb = estimate_pcb_physical([pkg])
    enc = estimate_enclosure_physical(pcb, "aluminum")
    assert enc.weight_grams > 0
    assert enc.dimensions is not None
    assert enc.metadata["material"] == "aluminum"


def test_plastic_lighter_than_aluminum():
    die = estimate_die_physical(50.0, 28)
    pkg = estimate_package_physical(die, "BGA")
    pcb = estimate_pcb_physical([pkg])
    al = estimate_enclosure_physical(pcb, "aluminum")
    plastic = estimate_enclosure_physical(pcb, "abs_plastic")
    assert plastic.weight_grams < al.weight_grams


# ---------------------------------------------------------------------------
# Thermal feasibility
# ---------------------------------------------------------------------------


def test_thermal_feasible():
    bom = estimate_system_bom(50.0, 5.0, 28, "BGA", "passive")
    result = compute_thermal_feasibility(bom, 5.0, ambient_temp_c=25.0)
    assert result["feasible"] is True
    assert result["junction_temp_c"] > 25.0
    assert result["margin_c"] > 0


def test_thermal_infeasible_high_power():
    bom = estimate_system_bom(200.0, 50.0, 7, "FCBGA", "passive")
    result = compute_thermal_feasibility(bom, 50.0, ambient_temp_c=60.0)
    # 50W * ~13 C/W + 60C >> 125C
    assert result["junction_temp_c"] > 125.0
    assert result["feasible"] is False
    assert result["margin_c"] < 0


# ---------------------------------------------------------------------------
# System BOM (top-level)
# ---------------------------------------------------------------------------


def test_system_bom_structure():
    bom = estimate_system_bom(50.0, 5.0, 28, "BGA", "passive")
    assert bom.level == BOMLevel.SYSTEM
    assert bom.cooling_type == "passive"
    assert len(bom.children) == 3  # pcb, heatsink, enclosure
    assert bom.total_weight_grams() > 0
    assert bom.total_cost_usd() > 0
    assert bom.total_volume_cm3() > 0


def test_system_bom_monotonicity():
    """More power -> heavier system (heatsink dominates)."""
    light = estimate_system_bom(50.0, 3.0, 28, "BGA", "passive")
    heavy = estimate_system_bom(50.0, 10.0, 28, "BGA", "passive")
    assert heavy.total_weight_grams() > light.total_weight_grams()


def test_system_bom_package_affects_weight():
    """Different package types produce different weights."""
    qfn = estimate_system_bom(50.0, 5.0, 28, "QFN", "passive")
    fcbga = estimate_system_bom(50.0, 5.0, 28, "FCBGA", "passive")
    assert fcbga.total_weight_grams() != qfn.total_weight_grams()


def test_system_bom_summary_table():
    bom = estimate_system_bom(50.0, 5.0, 28, "BGA", "passive")
    table = bom.summary_table()
    assert len(table) > 0
    names = [row["name"] for row in table]
    assert "system" in names
    assert "die" in names


def test_system_bom_flat_list():
    bom = estimate_system_bom(50.0, 5.0, 28, "BGA", "passive")
    flat = bom.flatten()
    levels = {item.level for item in flat}
    assert BOMLevel.DIE in levels
    assert BOMLevel.PACKAGE in levels
    assert BOMLevel.PCB in levels
    assert BOMLevel.SYSTEM in levels
