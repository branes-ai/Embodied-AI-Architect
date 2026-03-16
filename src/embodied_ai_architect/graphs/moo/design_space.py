"""Formal design space definition with mixed discrete/continuous variables.

Provides encoding/decoding between human-readable parameter dicts and
numeric arrays for optimizers, plus export to pymoo and BoTorch formats.

Usage:
    from embodied_ai_architect.graphs.moo.design_space import (
        DesignSpace, DesignVariable, create_soc_design_space,
    )

    ds = create_soc_design_space()
    sample = ds.sample_random(100)
    x = ds.encode(sample[0])
    params = ds.decode(x)
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

import numpy as np
from pydantic import BaseModel, Field


class VarType(str, Enum):
    """Variable type in the design space."""

    INTEGER = "integer"
    CONTINUOUS = "continuous"
    CATEGORICAL = "categorical"


class DesignVariable(BaseModel):
    """A single variable in the design space."""

    name: str
    var_type: VarType
    lower: Optional[float] = None  # for integer/continuous
    upper: Optional[float] = None  # for integer/continuous
    choices: Optional[list[Any]] = None  # for categorical

    def n_levels(self) -> int:
        """Number of discrete levels for this variable."""
        if self.var_type == VarType.CATEGORICAL:
            return len(self.choices) if self.choices else 1
        if self.var_type == VarType.INTEGER:
            return int(self.upper - self.lower) + 1 if self.upper is not None else 1
        return 0  # continuous

    def validate_value(self, value: Any) -> bool:
        """Check if a value is valid for this variable."""
        if self.var_type == VarType.CATEGORICAL:
            return value in (self.choices or [])
        if self.var_type == VarType.INTEGER:
            return self.lower <= int(value) <= self.upper
        if self.var_type == VarType.CONTINUOUS:
            return self.lower <= float(value) <= self.upper
        return False


class ObjectiveDirection(str, Enum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class DesignSpace(BaseModel):
    """Formal definition of a multi-objective design space.

    Variables can be integer, continuous, or categorical.
    Objectives have names and directions (minimize/maximize).
    Constraint bounds define feasibility region.
    """

    variables: list[DesignVariable] = Field(default_factory=list)
    objectives: list[str] = Field(default_factory=list)
    directions: list[ObjectiveDirection] = Field(default_factory=list)
    constraint_bounds: dict[str, tuple[float, float]] = Field(
        default_factory=dict,
        description="Constraint name -> (lower, upper) bounds",
    )

    @property
    def n_var(self) -> int:
        return len(self.variables)

    @property
    def n_obj(self) -> int:
        return len(self.objectives)

    def sample_random(self, n: int, seed: int | None = None) -> list[dict[str, Any]]:
        """Generate n random samples using Latin Hypercube Sampling.

        Returns list of parameter dicts.
        """
        rng = np.random.default_rng(seed)
        d = self.n_var

        # LHS: create stratified samples in [0, 1]^d
        result = np.zeros((n, d))
        for j in range(d):
            perm = rng.permutation(n)
            for i in range(n):
                result[i, j] = (perm[i] + rng.random()) / n

        # Map from [0,1] to actual variable ranges
        samples = []
        for i in range(n):
            params: dict[str, Any] = {}
            for j, var in enumerate(self.variables):
                u = result[i, j]
                if var.var_type == VarType.CONTINUOUS:
                    params[var.name] = var.lower + u * (var.upper - var.lower)
                elif var.var_type == VarType.INTEGER:
                    val = int(round(var.lower + u * (var.upper - var.lower)))
                    params[var.name] = max(int(var.lower), min(int(var.upper), val))
                elif var.var_type == VarType.CATEGORICAL:
                    idx = int(u * len(var.choices))
                    idx = min(idx, len(var.choices) - 1)
                    params[var.name] = var.choices[idx]
            samples.append(params)
        return samples

    def encode(self, params: dict[str, Any]) -> np.ndarray:
        """Encode a parameter dict to a numpy array for optimizers.

        Categorical variables are encoded as integer indices.
        """
        x = np.zeros(self.n_var)
        for j, var in enumerate(self.variables):
            val = params.get(var.name)
            if var.var_type == VarType.CATEGORICAL:
                if val in var.choices:
                    x[j] = var.choices.index(val)
                else:
                    x[j] = 0
            elif var.var_type == VarType.INTEGER:
                x[j] = float(int(val))
            else:
                x[j] = float(val)
        return x

    def decode(self, x: np.ndarray) -> dict[str, Any]:
        """Decode a numpy array back to a parameter dict."""
        params: dict[str, Any] = {}
        for j, var in enumerate(self.variables):
            if var.var_type == VarType.CATEGORICAL:
                idx = int(round(x[j]))
                idx = max(0, min(idx, len(var.choices) - 1))
                params[var.name] = var.choices[idx]
            elif var.var_type == VarType.INTEGER:
                val = int(round(x[j]))
                params[var.name] = max(int(var.lower), min(int(var.upper), val))
            else:
                val = float(x[j])
                params[var.name] = max(var.lower, min(var.upper, val))
        return params

    def to_pymoo_problem_kwargs(self) -> dict[str, Any]:
        """Export bounds/types for pymoo Problem construction.

        Returns dict with n_var, n_obj, xl, xu suitable for pymoo.Problem.__init__.
        """
        xl = np.zeros(self.n_var)
        xu = np.zeros(self.n_var)
        for j, var in enumerate(self.variables):
            if var.var_type == VarType.CATEGORICAL:
                xl[j] = 0
                xu[j] = len(var.choices) - 1
            elif var.var_type in (VarType.INTEGER, VarType.CONTINUOUS):
                xl[j] = var.lower
                xu[j] = var.upper
        return {
            "n_var": self.n_var,
            "n_obj": self.n_obj,
            "xl": xl,
            "xu": xu,
        }

    def to_botorch_bounds(self) -> Any:
        """Export bounds as a (2, d) torch tensor for BoTorch.

        Returns torch.Tensor or raises ImportError if torch not available.
        """
        import torch

        lower = []
        upper = []
        for var in self.variables:
            if var.var_type == VarType.CATEGORICAL:
                lower.append(0.0)
                upper.append(float(len(var.choices) - 1))
            else:
                lower.append(float(var.lower))
                upper.append(float(var.upper))
        return torch.tensor([lower, upper], dtype=torch.float64)

    def get_variable(self, name: str) -> DesignVariable | None:
        """Look up a variable by name."""
        for v in self.variables:
            if v.name == name:
                return v
        return None


def create_soc_design_space(
    constraints: dict[str, Any] | None = None,
) -> DesignSpace:
    """Factory: create a design space for SoC design exploration.

    Default variables cover the key SoC design dimensions:
    - process_nm: Technology node
    - clock_mhz: Operating frequency
    - array_rows/cols: Systolic array dimensions
    - sram_kb: On-chip SRAM
    - num_compute_tiles: Number of compute tiles
    - noc_link_width_bits: NoC interconnect width

    Args:
        constraints: Optional dict with constraint bounds to apply.

    Returns:
        DesignSpace configured for SoC exploration.
    """
    from embodied_ai_architect.graphs.technology import TECHNOLOGY_NODES

    # Extract available process nodes from technology database
    process_nodes = sorted([int(k.replace("nm", "")) for k in TECHNOLOGY_NODES.keys()])

    variables = [
        DesignVariable(
            name="process_nm",
            var_type=VarType.CATEGORICAL,
            choices=process_nodes,
        ),
        DesignVariable(
            name="clock_mhz",
            var_type=VarType.CONTINUOUS,
            lower=100.0,
            upper=3000.0,
        ),
        DesignVariable(
            name="array_rows",
            var_type=VarType.INTEGER,
            lower=2,
            upper=64,
        ),
        DesignVariable(
            name="array_cols",
            var_type=VarType.INTEGER,
            lower=2,
            upper=64,
        ),
        DesignVariable(
            name="sram_kb",
            var_type=VarType.INTEGER,
            lower=32,
            upper=2048,
        ),
        DesignVariable(
            name="num_compute_tiles",
            var_type=VarType.INTEGER,
            lower=1,
            upper=16,
        ),
        DesignVariable(
            name="noc_link_width_bits",
            var_type=VarType.CATEGORICAL,
            choices=[64, 128, 256, 512],
        ),
    ]

    objectives = ["power_watts", "latency_ms", "area_mm2", "cost_usd"]
    directions = [
        ObjectiveDirection.MINIMIZE,
        ObjectiveDirection.MINIMIZE,
        ObjectiveDirection.MINIMIZE,
        ObjectiveDirection.MINIMIZE,
    ]

    # Add capability objectives if accuracy-aware optimization is requested
    include_accuracy = constraints.get("include_accuracy", False) if constraints else False
    if include_accuracy:
        objectives.append("accuracy_percent")
        directions.append(ObjectiveDirection.MAXIMIZE)
        objectives.append("capability_per_watt")
        directions.append(ObjectiveDirection.MAXIMIZE)

    constraint_bounds: dict[str, tuple[float, float]] = {}
    if constraints:
        if constraints.get("max_power_watts"):
            constraint_bounds["power_watts"] = (0.0, constraints["max_power_watts"])
        if constraints.get("max_latency_ms"):
            constraint_bounds["latency_ms"] = (0.0, constraints["max_latency_ms"])
        if constraints.get("max_area_mm2"):
            constraint_bounds["area_mm2"] = (0.0, constraints["max_area_mm2"])
        if constraints.get("max_cost_usd"):
            constraint_bounds["cost_usd"] = (0.0, constraints["max_cost_usd"])
        if constraints.get("min_accuracy_percent"):
            constraint_bounds["accuracy_percent"] = (
                constraints["min_accuracy_percent"],
                100.0,
            )

    return DesignSpace(
        variables=variables,
        objectives=objectives,
        directions=directions,
        constraint_bounds=constraint_bounds,
    )


def create_swap_design_space(
    constraints: dict[str, Any] | None = None,
) -> DesignSpace:
    """Factory: create a 6-objective SWaP-C design space.

    Extends the SoC design space with package_type and cooling_type variables,
    and adds weight_grams and volume_cm3 objectives.

    Variables: 7 SoC variables + package_type (categorical) + cooling_type (categorical)
    Objectives: power_watts, latency_ms, area_mm2, cost_usd, weight_grams, volume_cm3
    Directions: all MINIMIZE

    Args:
        constraints: Optional dict with constraint bounds including
            max_weight_grams and max_volume_cm3.

    Returns:
        DesignSpace configured for 6-objective SWaP-C exploration.
    """
    from embodied_ai_architect.graphs.technology import TECHNOLOGY_NODES

    process_nodes = sorted([int(k.replace("nm", "")) for k in TECHNOLOGY_NODES.keys()])

    variables = [
        DesignVariable(
            name="process_nm",
            var_type=VarType.CATEGORICAL,
            choices=process_nodes,
        ),
        DesignVariable(
            name="clock_mhz",
            var_type=VarType.CONTINUOUS,
            lower=100.0,
            upper=3000.0,
        ),
        DesignVariable(
            name="array_rows",
            var_type=VarType.INTEGER,
            lower=2,
            upper=64,
        ),
        DesignVariable(
            name="array_cols",
            var_type=VarType.INTEGER,
            lower=2,
            upper=64,
        ),
        DesignVariable(
            name="sram_kb",
            var_type=VarType.INTEGER,
            lower=32,
            upper=2048,
        ),
        DesignVariable(
            name="num_compute_tiles",
            var_type=VarType.INTEGER,
            lower=1,
            upper=16,
        ),
        DesignVariable(
            name="noc_link_width_bits",
            var_type=VarType.CATEGORICAL,
            choices=[64, 128, 256, 512],
        ),
        # SWaP-C specific variables
        DesignVariable(
            name="package_type",
            var_type=VarType.CATEGORICAL,
            choices=["QFN", "BGA", "FCBGA", "WLCSP"],
        ),
        DesignVariable(
            name="cooling_type",
            var_type=VarType.CATEGORICAL,
            choices=["passive", "active_fan", "liquid"],
        ),
    ]

    objectives = [
        "power_watts",
        "latency_ms",
        "area_mm2",
        "cost_usd",
        "weight_grams",
        "volume_cm3",
    ]
    directions = [ObjectiveDirection.MINIMIZE] * 6

    # Add capability objectives if accuracy-aware optimization is requested
    include_accuracy = constraints.get("include_accuracy", False) if constraints else False
    if include_accuracy:
        objectives.append("accuracy_percent")
        directions.append(ObjectiveDirection.MAXIMIZE)
        objectives.append("capability_per_watt")
        directions.append(ObjectiveDirection.MAXIMIZE)

    constraint_bounds: dict[str, tuple[float, float]] = {}
    if constraints:
        if constraints.get("max_power_watts"):
            constraint_bounds["power_watts"] = (0.0, constraints["max_power_watts"])
        if constraints.get("max_latency_ms"):
            constraint_bounds["latency_ms"] = (0.0, constraints["max_latency_ms"])
        if constraints.get("max_area_mm2"):
            constraint_bounds["area_mm2"] = (0.0, constraints["max_area_mm2"])
        if constraints.get("max_cost_usd"):
            constraint_bounds["cost_usd"] = (0.0, constraints["max_cost_usd"])
        if constraints.get("max_weight_grams"):
            constraint_bounds["weight_grams"] = (0.0, constraints["max_weight_grams"])
        if constraints.get("max_volume_cm3"):
            constraint_bounds["volume_cm3"] = (0.0, constraints["max_volume_cm3"])
        if constraints.get("min_accuracy_percent"):
            constraint_bounds["accuracy_percent"] = (
                constraints["min_accuracy_percent"],
                100.0,
            )

    return DesignSpace(
        variables=variables,
        objectives=objectives,
        directions=directions,
        constraint_bounds=constraint_bounds,
    )
