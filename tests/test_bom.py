"""Tests for BOM data model (Phase 1)."""

from embodied_ai_architect.graphs.bom import (
    BOMComponent,
    BOMLevel,
    DieBOM,
    Dimensions3D,
    MCMBOM,
    PackageBOM,
    PCBBOM,
    SystemBOM,
)

# ---------------------------------------------------------------------------
# Dimensions3D
# ---------------------------------------------------------------------------


def test_dimensions_volume():
    d = Dimensions3D(length_mm=10.0, width_mm=10.0, height_mm=10.0)
    assert d.volume_cm3 == 1.0  # 1000 mm^3 = 1 cm^3


def test_dimensions_volume_non_cube():
    d = Dimensions3D(length_mm=20.0, width_mm=10.0, height_mm=5.0)
    assert d.volume_cm3 == 1.0  # 1000 mm^3 = 1 cm^3


# ---------------------------------------------------------------------------
# BOMComponent recursive aggregation
# ---------------------------------------------------------------------------


def test_total_weight_single():
    comp = BOMComponent(name="a", level=BOMLevel.DIE, weight_grams=5.0)
    assert comp.total_weight_grams() == 5.0


def test_total_weight_recursive():
    child1 = BOMComponent(name="c1", level=BOMLevel.DIE, weight_grams=2.0)
    child2 = BOMComponent(name="c2", level=BOMLevel.DIE, weight_grams=3.0)
    parent = BOMComponent(
        name="p", level=BOMLevel.PACKAGE, weight_grams=1.0, children=[child1, child2]
    )
    assert parent.total_weight_grams() == 6.0


def test_total_cost_recursive():
    child = BOMComponent(name="c", level=BOMLevel.DIE, cost_usd=10.0)
    parent = BOMComponent(name="p", level=BOMLevel.PACKAGE, cost_usd=5.0, children=[child])
    assert parent.total_cost_usd() == 15.0


def test_total_volume_from_dimensions():
    comp = BOMComponent(
        name="a",
        level=BOMLevel.DIE,
        dimensions=Dimensions3D(length_mm=10.0, width_mm=10.0, height_mm=10.0),
    )
    assert comp.total_volume_cm3() == 1.0


def test_total_volume_sum_children():
    c1 = BOMComponent(
        name="c1",
        level=BOMLevel.DIE,
        dimensions=Dimensions3D(length_mm=10.0, width_mm=10.0, height_mm=10.0),
    )
    c2 = BOMComponent(
        name="c2",
        level=BOMLevel.DIE,
        dimensions=Dimensions3D(length_mm=10.0, width_mm=10.0, height_mm=10.0),
    )
    parent = BOMComponent(name="p", level=BOMLevel.PACKAGE, children=[c1, c2])
    assert parent.total_volume_cm3() == 2.0


def test_bounding_dimensions():
    c1 = BOMComponent(
        name="c1",
        level=BOMLevel.DIE,
        dimensions=Dimensions3D(length_mm=20.0, width_mm=10.0, height_mm=5.0),
    )
    c2 = BOMComponent(
        name="c2",
        level=BOMLevel.DIE,
        dimensions=Dimensions3D(length_mm=15.0, width_mm=15.0, height_mm=3.0),
    )
    parent = BOMComponent(name="p", level=BOMLevel.PACKAGE, children=[c1, c2])
    bd = parent.bounding_dimensions_mm()
    assert bd.length_mm == 20.0
    assert bd.width_mm == 15.0
    assert bd.height_mm == 8.0  # sum of heights


def test_flatten():
    leaf = BOMComponent(name="leaf", level=BOMLevel.DIE)
    mid = BOMComponent(name="mid", level=BOMLevel.PACKAGE, children=[leaf])
    root = BOMComponent(name="root", level=BOMLevel.SYSTEM, children=[mid])
    flat = root.flatten()
    assert len(flat) == 3
    assert [f.name for f in flat] == ["root", "mid", "leaf"]


def test_summary_table():
    child = BOMComponent(
        name="die",
        level=BOMLevel.DIE,
        weight_grams=0.5,
        cost_usd=2.0,
        dimensions=Dimensions3D(length_mm=5.0, width_mm=5.0, height_mm=0.8),
    )
    parent = BOMComponent(
        name="pkg",
        level=BOMLevel.PACKAGE,
        weight_grams=1.0,
        cost_usd=1.5,
        children=[child],
    )
    table = parent.summary_table()
    assert len(table) == 2
    assert table[0]["name"] == "pkg"
    assert table[1]["name"] == "die"
    assert table[1]["weight_grams"] == 0.5


# ---------------------------------------------------------------------------
# Specialized subclasses
# ---------------------------------------------------------------------------


def test_die_bom_defaults():
    die = DieBOM(name="test-die", die_area_mm2=50.0, process_nm=28)
    assert die.level == BOMLevel.DIE
    assert die.silicon_thickness_um == 775.0


def test_package_bom_defaults():
    pkg = PackageBOM(name="test-pkg", package_type="BGA")
    assert pkg.level == BOMLevel.PACKAGE
    assert pkg.pin_count == 256
    assert pkg.substrate_layers == 4


def test_mcm_bom():
    mcm = MCMBOM(name="test-mcm", interposer_type="silicon", interposer_area_mm2=400.0)
    assert mcm.level == BOMLevel.MCM


def test_pcb_bom_defaults():
    pcb = PCBBOM(name="test-pcb", board_area_cm2=25.0)
    assert pcb.level == BOMLevel.PCB
    assert pcb.layer_count == 4
    assert pcb.connector_count == 0


def test_system_bom_defaults():
    sys = SystemBOM(name="test-sys")
    assert sys.level == BOMLevel.SYSTEM
    assert sys.cooling_type == "passive"
    assert sys.enclosure_material == "aluminum"


def test_composability():
    """Full hierarchy: die -> package -> PCB -> system."""
    die = DieBOM(
        name="die",
        die_area_mm2=50.0,
        process_nm=28,
        weight_grams=0.1,
        cost_usd=5.0,
    )
    pkg = PackageBOM(
        name="pkg",
        package_type="BGA",
        weight_grams=0.5,
        cost_usd=1.5,
        children=[die],
    )
    pcb = PCBBOM(
        name="pcb",
        board_area_cm2=20.0,
        weight_grams=10.0,
        cost_usd=3.0,
        children=[pkg],
    )
    system = SystemBOM(
        name="sys",
        weight_grams=50.0,
        cost_usd=2.0,
        children=[pcb],
    )

    # Recursive totals
    assert system.total_weight_grams() == 50.0 + 10.0 + 0.5 + 0.1
    assert system.total_cost_usd() == 2.0 + 3.0 + 1.5 + 5.0
    assert len(system.flatten()) == 4
