"""Parametric physical estimators for system-level SWaP-C.

Pure functions returning BOM instances, following the same pattern as
``estimate_manufacturing_cost()`` in manufacturing.py. Material properties
stored as constant lookup tables.

Usage:
    from embodied_ai_architect.graphs.physical_estimators import estimate_system_bom

    bom = estimate_system_bom(soc, package_type="BGA", cooling_type="passive")
    print(bom.total_weight_grams(), bom.total_cost_usd())
"""

from __future__ import annotations

import math
from typing import Any

from embodied_ai_architect.graphs.bom import (
    BOMComponent,
    BOMLevel,
    DieBOM,
    Dimensions3D,
    PackageBOM,
    PCBBOM,
    SystemBOM,
)

# ---------------------------------------------------------------------------
# Material property tables
# ---------------------------------------------------------------------------

MATERIAL_DENSITY: dict[str, float] = {
    # g/cm^3
    "silicon": 2.33,
    "fr4": 1.85,
    "aluminum": 2.70,
    "copper": 8.96,
    "abs_plastic": 1.05,
    "magnesium": 1.74,
}

# ---------------------------------------------------------------------------
# Package physical parameters
# ---------------------------------------------------------------------------

PACKAGE_PHYSICAL: dict[str, dict[str, float]] = {
    # body_thickness_mm: package body height excluding balls/leads
    # substrate_g_per_cm2: substrate weight per cm^2
    # theta_ja: junction-to-ambient thermal resistance (C/W)
    # theta_jc: junction-to-case thermal resistance (C/W)
    # area_overhead: package-to-die area ratio
    "QFN": {
        "body_thickness_mm": 0.85,
        "substrate_g_per_cm2": 0.15,
        "theta_ja": 35.0,
        "theta_jc": 3.0,
        "area_overhead": 1.5,
    },
    "BGA": {
        "body_thickness_mm": 1.70,
        "substrate_g_per_cm2": 0.25,
        "theta_ja": 25.0,
        "theta_jc": 2.0,
        "area_overhead": 2.0,
    },
    "FCBGA": {
        "body_thickness_mm": 2.50,
        "substrate_g_per_cm2": 0.35,
        "theta_ja": 15.0,
        "theta_jc": 1.0,
        "area_overhead": 2.5,
    },
    "WLCSP": {
        "body_thickness_mm": 0.50,
        "substrate_g_per_cm2": 0.10,
        "theta_ja": 40.0,
        "theta_jc": 5.0,
        "area_overhead": 1.05,
    },
}

# ---------------------------------------------------------------------------
# Cooling solution parameters
# ---------------------------------------------------------------------------

COOLING_PARAMS: dict[str, dict[str, float]] = {
    # g_per_w:  heatsink weight per watt of TDP
    # cm3_per_w: heatsink volume per watt of TDP
    # theta_sa: sink-to-ambient thermal resistance (C/W)
    # max_tdp:  maximum TDP this cooling type can handle (W)
    # parasitic_power_w: fan/pump power consumption (W)
    "passive": {
        "g_per_w": 8.0,
        "cm3_per_w": 4.0,
        "theta_sa": 10.0,
        "max_tdp": 15.0,
        "parasitic_power_w": 0.0,
    },
    "active_fan": {
        "g_per_w": 3.0,
        "cm3_per_w": 2.0,
        "theta_sa": 3.0,
        "max_tdp": 100.0,
        "parasitic_power_w": 1.5,
    },
    "liquid": {
        "g_per_w": 2.0,
        "cm3_per_w": 1.5,
        "theta_sa": 1.0,
        "max_tdp": 500.0,
        "parasitic_power_w": 5.0,
    },
}


# ---------------------------------------------------------------------------
# Estimator functions
# ---------------------------------------------------------------------------


def estimate_die_physical(
    die_area_mm2: float,
    process_nm: int,
    silicon_thickness_um: float = 775.0,
) -> DieBOM:
    """Estimate physical properties of a silicon die.

    Weight = area_cm2 * thickness_cm * silicon_density.

    Args:
        die_area_mm2: Die area in mm^2.
        process_nm: Process technology node in nm.
        silicon_thickness_um: Silicon substrate thickness (default 775 um).

    Returns:
        DieBOM with weight and dimensions populated.
    """
    area_cm2 = die_area_mm2 / 100.0
    thickness_cm = silicon_thickness_um / 10_000.0
    weight_g = area_cm2 * thickness_cm * MATERIAL_DENSITY["silicon"]

    side_mm = math.sqrt(die_area_mm2)
    thickness_mm = silicon_thickness_um / 1000.0

    return DieBOM(
        name="die",
        die_area_mm2=die_area_mm2,
        process_nm=process_nm,
        silicon_thickness_um=silicon_thickness_um,
        weight_grams=round(weight_g, 4),
        dimensions=Dimensions3D(
            length_mm=round(side_mm, 2),
            width_mm=round(side_mm, 2),
            height_mm=round(thickness_mm, 3),
        ),
        thermal_resistance_c_per_w=0.0,  # negligible for bulk silicon
    )


def estimate_package_physical(
    die: DieBOM,
    package_type: str = "BGA",
    pin_count: int = 256,
) -> PackageBOM:
    """Estimate physical properties of an IC package.

    Package area = die_area * area_overhead. Weight = substrate + die.

    Args:
        die: DieBOM from estimate_die_physical().
        package_type: One of QFN, BGA, FCBGA, WLCSP.
        pin_count: Pin/ball count.

    Returns:
        PackageBOM wrapping the die with physical estimates.
    """
    props = PACKAGE_PHYSICAL.get(package_type, PACKAGE_PHYSICAL["BGA"])

    pkg_area_mm2 = die.die_area_mm2 * props["area_overhead"]
    pkg_area_cm2 = pkg_area_mm2 / 100.0
    substrate_weight_g = pkg_area_cm2 * props["substrate_g_per_cm2"]
    total_weight_g = substrate_weight_g  # die weight added via children

    side_mm = math.sqrt(pkg_area_mm2)
    body_h_mm = props["body_thickness_mm"]

    from embodied_ai_architect.graphs.manufacturing import PACKAGE_COSTS

    pkg_cost = PACKAGE_COSTS.get(package_type, PACKAGE_COSTS["BGA"])

    return PackageBOM(
        name=f"{package_type}-{pin_count}",
        package_type=package_type,
        pin_count=pin_count,
        weight_grams=round(total_weight_g, 4),
        dimensions=Dimensions3D(
            length_mm=round(side_mm, 2),
            width_mm=round(side_mm, 2),
            height_mm=round(body_h_mm, 2),
        ),
        thermal_resistance_c_per_w=props["theta_jc"],
        cost_usd=round(pkg_cost, 2),
        children=[die],
    )


def estimate_pcb_physical(
    components: list[BOMComponent],
    layer_count: int = 4,
    connector_count: int = 0,
) -> PCBBOM:
    """Estimate PCB physical properties.

    Board area = sum(component footprints) * 2.5 (routing overhead).
    Weight ~ area * (0.07 * layers + 0.05) g/cm^2. Connectors ~5g each.

    Args:
        components: Mounted components (packages, passives).
        layer_count: PCB layer count.
        connector_count: Number of board connectors.

    Returns:
        PCBBOM containing the components.
    """
    # Sum component footprint areas
    total_footprint_cm2 = 0.0
    for comp in components:
        dims = comp.bounding_dimensions_mm()
        footprint_mm2 = dims.length_mm * dims.width_mm
        total_footprint_cm2 += footprint_mm2 / 100.0

    # Routing overhead factor
    board_area_cm2 = max(total_footprint_cm2 * 2.5, 1.0)

    # PCB weight: FR4 layers + copper
    weight_per_cm2 = 0.07 * layer_count + 0.05  # g/cm^2
    board_weight_g = board_area_cm2 * weight_per_cm2
    connector_weight_g = connector_count * 5.0
    total_weight_g = board_weight_g + connector_weight_g

    # PCB dimensions
    side_mm = math.sqrt(board_area_cm2) * 10.0  # cm -> mm
    pcb_thickness_mm = 0.2 * layer_count + 0.4  # typical FR4 stackup

    # PCB cost: rough estimate
    pcb_cost = board_area_cm2 * (0.5 + 0.15 * layer_count) + connector_count * 0.50

    return PCBBOM(
        name="pcb",
        board_area_cm2=round(board_area_cm2, 2),
        layer_count=layer_count,
        connector_count=connector_count,
        weight_grams=round(total_weight_g, 2),
        dimensions=Dimensions3D(
            length_mm=round(side_mm, 2),
            width_mm=round(side_mm, 2),
            height_mm=round(pcb_thickness_mm, 2),
        ),
        cost_usd=round(pcb_cost, 2),
        children=components,
    )


def estimate_heatsink_physical(
    tdp_watts: float,
    cooling_type: str = "passive",
) -> BOMComponent:
    """Estimate heatsink/cooling solution physical properties.

    Weight and volume scale linearly with TDP.

    Args:
        tdp_watts: Thermal design power in watts.
        cooling_type: One of passive, active_fan, liquid.

    Returns:
        BOMComponent representing the cooling solution.
    """
    params = COOLING_PARAMS.get(cooling_type, COOLING_PARAMS["passive"])

    weight_g = tdp_watts * params["g_per_w"]
    volume_cm3 = tdp_watts * params["cm3_per_w"]

    # Approximate cubic dimensions
    side_cm = volume_cm3 ** (1.0 / 3.0) if volume_cm3 > 0 else 0.0
    side_mm = side_cm * 10.0

    return BOMComponent(
        name=f"heatsink-{cooling_type}",
        level=BOMLevel.SYSTEM,
        weight_grams=round(weight_g, 2),
        dimensions=Dimensions3D(
            length_mm=round(side_mm, 1),
            width_mm=round(side_mm, 1),
            height_mm=round(side_mm, 1),
        ),
        thermal_resistance_c_per_w=params["theta_sa"],
        cost_usd=0.0,  # bundled into system cost
        metadata={
            "cooling_type": cooling_type,
            "parasitic_power_w": params["parasitic_power_w"],
            "max_tdp_w": params["max_tdp"],
        },
    )


def estimate_enclosure_physical(
    pcb: PCBBOM,
    material: str = "aluminum",
    wall_thickness_mm: float = 2.0,
) -> BOMComponent:
    """Estimate enclosure physical properties.

    Enclosure dimensions = PCB dims + 2*wall on each side.
    Weight = surface_area * wall_thickness * material_density.

    Args:
        pcb: PCBBOM providing internal dimensions.
        material: Enclosure material (aluminum, abs_plastic, magnesium).
        wall_thickness_mm: Wall thickness in mm.

    Returns:
        BOMComponent representing the enclosure shell.
    """
    density = MATERIAL_DENSITY.get(material, MATERIAL_DENSITY["aluminum"])

    pcb_dims = pcb.bounding_dimensions_mm()
    inner_l = pcb_dims.length_mm
    inner_w = pcb_dims.width_mm
    inner_h = pcb_dims.height_mm + 10.0  # clearance for components

    outer_l = inner_l + 2 * wall_thickness_mm
    outer_w = inner_w + 2 * wall_thickness_mm
    outer_h = inner_h + 2 * wall_thickness_mm

    # Surface area of 6 faces (outer box) in cm^2
    surface_area_cm2 = (2 * (outer_l * outer_w + outer_l * outer_h + outer_w * outer_h)) / 100.0

    wall_cm = wall_thickness_mm / 10.0
    weight_g = surface_area_cm2 * wall_cm * density

    # Enclosure cost: rough material + machining estimate
    enclosure_cost = weight_g * 0.05  # ~$0.05/g for formed metal

    return BOMComponent(
        name=f"enclosure-{material}",
        level=BOMLevel.SYSTEM,
        weight_grams=round(weight_g, 2),
        dimensions=Dimensions3D(
            length_mm=round(outer_l, 2),
            width_mm=round(outer_w, 2),
            height_mm=round(outer_h, 2),
        ),
        cost_usd=round(enclosure_cost, 2),
        metadata={"material": material, "wall_thickness_mm": wall_thickness_mm},
    )


def compute_thermal_feasibility(
    system_bom: SystemBOM,
    tdp_watts: float,
    ambient_temp_c: float = 40.0,
    max_junction_temp_c: float = 125.0,
) -> dict[str, Any]:
    """Compute junction temperature and thermal feasibility.

    T_junction = T_ambient + TDP * (theta_jc + theta_cs + theta_sa).
    Theta values aggregated from BOM components in the thermal path.

    Args:
        system_bom: Complete system BOM tree.
        tdp_watts: Total thermal design power in watts.
        ambient_temp_c: Ambient temperature in Celsius.
        max_junction_temp_c: Maximum allowed junction temperature.

    Returns:
        Dict with feasible, junction_temp_c, margin_c, total_theta.
    """
    # Walk the BOM tree and sum thermal resistances
    total_theta = 0.0
    for item in system_bom.flatten():
        total_theta += item.thermal_resistance_c_per_w

    junction_temp_c = ambient_temp_c + tdp_watts * total_theta
    margin_c = max_junction_temp_c - junction_temp_c
    feasible = junction_temp_c <= max_junction_temp_c

    return {
        "feasible": feasible,
        "junction_temp_c": round(junction_temp_c, 1),
        "ambient_temp_c": ambient_temp_c,
        "max_junction_temp_c": max_junction_temp_c,
        "margin_c": round(margin_c, 1),
        "total_theta_c_per_w": round(total_theta, 2),
        "tdp_watts": tdp_watts,
    }


def estimate_system_bom(
    soc_area_mm2: float,
    soc_power_watts: float,
    process_nm: int,
    package_type: str = "BGA",
    cooling_type: str = "passive",
    enclosure_material: str = "aluminum",
    volume: int = 10_000,
    layer_count: int = 4,
    connector_count: int = 0,
    extra_components: list[BOMComponent] | None = None,
) -> SystemBOM:
    """Top-level entry point: build a complete system BOM from SoC parameters.

    Calls all estimators in sequence and assembles the hierarchical tree.
    Integrates manufacturing cost into the die-level cost roll-up.

    Args:
        soc_area_mm2: Total SoC die area in mm^2.
        soc_power_watts: SoC TDP in watts.
        process_nm: Process technology node.
        package_type: IC package type.
        cooling_type: Cooling solution type.
        enclosure_material: Enclosure material.
        volume: Production volume for cost estimation.
        layer_count: PCB layer count.
        connector_count: Board connector count.
        extra_components: Additional BOM items (sensors, connectors, etc.).

    Returns:
        SystemBOM with fully populated hierarchy and costs.
    """
    from embodied_ai_architect.graphs.manufacturing import estimate_manufacturing_cost

    # 1. Die
    die = estimate_die_physical(soc_area_mm2, process_nm)

    # Integrate manufacturing cost at die level
    mfg = estimate_manufacturing_cost(soc_area_mm2, process_nm, volume, package_type)
    die.cost_usd = round(mfg.die_cost_usd + mfg.nre_per_unit_usd + mfg.test_cost_usd, 2)

    # 2. Package
    package = estimate_package_physical(die, package_type)

    # 3. PCB
    pcb_components: list[BOMComponent] = [package]
    if extra_components:
        pcb_components.extend(extra_components)
    pcb = estimate_pcb_physical(pcb_components, layer_count, connector_count)

    # 4. Heatsink
    heatsink = estimate_heatsink_physical(soc_power_watts, cooling_type)

    # 5. Enclosure
    enclosure = estimate_enclosure_physical(pcb, enclosure_material)

    # 6. Assemble system
    system = SystemBOM(
        name="system",
        cooling_type=cooling_type,
        enclosure_material=enclosure_material,
        children=[pcb, heatsink, enclosure],
        metadata={
            "process_nm": process_nm,
            "package_type": package_type,
            "volume": volume,
            "soc_area_mm2": soc_area_mm2,
            "soc_power_watts": soc_power_watts,
        },
    )

    return system
