"""Hierarchical Bill of Materials (BOM) for system-level SWaP-C estimation.

Models the physical hierarchy from die through package, PCB, and system
enclosure. Each level is a composable BOMComponent with recursive weight,
volume, and cost aggregation.

Usage:
    from embodied_ai_architect.graphs.bom import (
        DieBOM, PackageBOM, PCBBOM, SystemBOM, Dimensions3D,
    )

    die = DieBOM(name="KPU", die_area_mm2=50.0, process_nm=28)
    pkg = PackageBOM(name="BGA-256", package_type="BGA", children=[die])
    print(pkg.total_weight_grams())
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class BOMLevel(str, Enum):
    """Hierarchy level in the physical BOM."""

    DIE = "die"
    PACKAGE = "package"
    MCM = "mcm"
    PCB = "pcb"
    SYSTEM = "system"


# ---------------------------------------------------------------------------
# Geometry helper
# ---------------------------------------------------------------------------


class Dimensions3D(BaseModel):
    """Physical dimensions in millimeters."""

    length_mm: float = Field(..., description="Length in mm")
    width_mm: float = Field(..., description="Width in mm")
    height_mm: float = Field(..., description="Height in mm")

    @property
    def volume_cm3(self) -> float:
        """Volume in cubic centimeters."""
        return (self.length_mm * self.width_mm * self.height_mm) / 1000.0


# ---------------------------------------------------------------------------
# Base BOM component
# ---------------------------------------------------------------------------


class BOMComponent(BaseModel):
    """Base class for every item in the physical BOM hierarchy.

    Supports recursive aggregation: total_weight/cost/volume walk the
    ``children`` tree. Each component stores its own direct contribution
    plus a thermal resistance for thermal-path analysis.
    """

    name: str = Field(..., description="Component identifier")
    level: BOMLevel = Field(..., description="Hierarchy level")
    weight_grams: float = Field(default=0.0, description="Direct weight in grams")
    dimensions: Dimensions3D | None = Field(
        default=None, description="Physical dimensions (LxWxH mm)"
    )
    thermal_resistance_c_per_w: float = Field(
        default=0.0, description="Thermal resistance junction-to-case or case-to-sink (C/W)"
    )
    cost_usd: float = Field(default=0.0, description="Direct cost in USD")
    children: list[BOMComponent] = Field(default_factory=list, description="Nested sub-components")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Extra annotations")

    def total_weight_grams(self) -> float:
        """Recursive weight: self + all children."""
        return self.weight_grams + sum(c.total_weight_grams() for c in self.children)

    def total_cost_usd(self) -> float:
        """Recursive cost: self + all children."""
        return self.cost_usd + sum(c.total_cost_usd() for c in self.children)

    def total_volume_cm3(self) -> float:
        """Volume from own dimensions, or sum of children if no dimensions."""
        if self.dimensions is not None:
            return self.dimensions.volume_cm3
        return sum(c.total_volume_cm3() for c in self.children)

    def bounding_dimensions_mm(self) -> Dimensions3D:
        """Enclosing bounding box across self and children."""
        if self.dimensions is not None:
            return self.dimensions
        if not self.children:
            return Dimensions3D(length_mm=0.0, width_mm=0.0, height_mm=0.0)
        lengths = [c.bounding_dimensions_mm().length_mm for c in self.children]
        widths = [c.bounding_dimensions_mm().width_mm for c in self.children]
        heights = [c.bounding_dimensions_mm().height_mm for c in self.children]
        return Dimensions3D(
            length_mm=max(lengths),
            width_mm=max(widths),
            height_mm=sum(heights),
        )

    def flatten(self) -> list[BOMComponent]:
        """Return a flat list of this component and all descendants."""
        result: list[BOMComponent] = [self]
        for child in self.children:
            result.extend(child.flatten())
        return result

    def summary_table(self) -> list[dict[str, Any]]:
        """Flat summary suitable for tabular display."""
        rows: list[dict[str, Any]] = []
        for item in self.flatten():
            rows.append(
                {
                    "name": item.name,
                    "level": item.level.value,
                    "weight_grams": round(item.weight_grams, 3),
                    "volume_cm3": round(item.total_volume_cm3(), 3) if item.dimensions else 0.0,
                    "cost_usd": round(item.cost_usd, 2),
                }
            )
        return rows


# ---------------------------------------------------------------------------
# Specialized subclasses
# ---------------------------------------------------------------------------


class DieBOM(BOMComponent):
    """Silicon die with process and geometry info."""

    level: BOMLevel = BOMLevel.DIE
    die_area_mm2: float = Field(..., description="Die area in mm^2")
    process_nm: int = Field(..., description="Process technology node in nm")
    silicon_thickness_um: float = Field(
        default=775.0, description="Silicon substrate thickness in um"
    )


class PackageBOM(BOMComponent):
    """IC package containing one or more dies."""

    level: BOMLevel = BOMLevel.PACKAGE
    package_type: str = Field(..., description="Package type (QFN, BGA, FCBGA, WLCSP)")
    pin_count: int = Field(default=256, description="Total pin/ball count")
    substrate_layers: int = Field(default=4, description="Package substrate layers")


class MCMBOM(BOMComponent):
    """Multi-chip module (defined, estimators deferred)."""

    level: BOMLevel = BOMLevel.MCM
    interposer_type: str = Field(
        ..., description="Interposer type (silicon, organic, embedded_bridge)"
    )
    interposer_area_mm2: float = Field(..., description="Interposer area in mm^2")


class PCBBOM(BOMComponent):
    """Printed circuit board."""

    level: BOMLevel = BOMLevel.PCB
    board_area_cm2: float = Field(..., description="Board area in cm^2")
    layer_count: int = Field(default=4, description="PCB layer count")
    connector_count: int = Field(default=0, description="Number of board connectors")


class SystemBOM(BOMComponent):
    """Top-level system enclosure with all sub-assemblies."""

    level: BOMLevel = BOMLevel.SYSTEM
    cooling_type: str = Field(default="passive", description="Cooling solution type")
    enclosure_material: str = Field(
        default="aluminum", description="Enclosure material (aluminum, abs_plastic, magnesium)"
    )
