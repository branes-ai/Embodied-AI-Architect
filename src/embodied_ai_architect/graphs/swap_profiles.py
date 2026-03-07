"""Mission profiles for SWaP-C optimization.

Defines weighted objective profiles for different deployment scenarios and
provides AHP (Analytic Hierarchy Process) helper for deriving custom weights.

Usage:
    from embodied_ai_architect.graphs.swap_profiles import (
        MissionProfile, get_profile, list_profiles, ahp_weights,
    )

    profile = get_profile("drone")
    score = sum(w * v for w, v in zip(profile.weights.values(), scores))
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from pydantic import BaseModel, Field, model_validator

# The 6 SWaP-C objectives (all minimized)
OBJECTIVES = [
    "power_watts",
    "latency_ms",
    "area_mm2",
    "cost_usd",
    "weight_grams",
    "volume_cm3",
]


class MissionProfile(BaseModel):
    """Weighted objective profile for a deployment scenario.

    Weights must sum to 1.0 and cover all 6 SWaP-C objectives.
    Budgets are optional per-metric upper bounds.
    """

    name: str
    description: str = ""
    weights: dict[str, float] = Field(..., description="Objective name -> weight (must sum to 1.0)")
    budgets: dict[str, float] = Field(
        default_factory=dict, description="Optional metric -> max value"
    )

    @model_validator(mode="after")
    def _validate_weights(self) -> "MissionProfile":
        total = sum(self.weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {total:.6f}")
        missing = set(OBJECTIVES) - set(self.weights.keys())
        if missing:
            raise ValueError(f"Missing weights for objectives: {missing}")
        return self


# ---------------------------------------------------------------------------
# Preset profiles
# ---------------------------------------------------------------------------

_PRESETS: dict[str, MissionProfile] = {
    "drone": MissionProfile(
        name="drone",
        description="Aerial platform: payload weight and battery life dominate",
        weights={
            "power_watts": 0.25,
            "latency_ms": 0.05,
            "area_mm2": 0.05,
            "cost_usd": 0.05,
            "weight_grams": 0.35,
            "volume_cm3": 0.25,
        },
    ),
    "rack": MissionProfile(
        name="rack",
        description="Data-center rack: TCO and cooling capacity dominate",
        weights={
            "power_watts": 0.30,
            "latency_ms": 0.10,
            "area_mm2": 0.10,
            "cost_usd": 0.30,
            "weight_grams": 0.05,
            "volume_cm3": 0.15,
        },
    ),
    "wearable": MissionProfile(
        name="wearable",
        description="Body-worn device: every physical dimension is tight",
        weights={
            "power_watts": 0.25,
            "latency_ms": 0.05,
            "area_mm2": 0.10,
            "cost_usd": 0.10,
            "weight_grams": 0.25,
            "volume_cm3": 0.25,
        },
    ),
    "vehicle": MissionProfile(
        name="vehicle",
        description="Automotive: unit economics and real-time response dominate",
        weights={
            "power_watts": 0.10,
            "latency_ms": 0.30,
            "area_mm2": 0.05,
            "cost_usd": 0.35,
            "weight_grams": 0.05,
            "volume_cm3": 0.15,
        },
    ),
}


def get_profile(name: str) -> MissionProfile:
    """Look up a preset mission profile by name.

    Args:
        name: Profile name (drone, rack, wearable, vehicle).

    Returns:
        MissionProfile with preset weights.

    Raises:
        KeyError: If the profile name is not found.
    """
    if name not in _PRESETS:
        raise KeyError(f"Unknown profile '{name}'. Available: {list(_PRESETS.keys())}")
    return _PRESETS[name]


def list_profiles() -> list[str]:
    """Return available preset profile names."""
    return list(_PRESETS.keys())


def ahp_weights(
    pairwise_matrix: list[list[float]] | Any,
    objectives: list[str] | None = None,
) -> dict[str, float]:
    """Derive objective weights from an AHP pairwise comparison matrix.

    Uses the geometric-mean method:
      1. Compute the geometric mean of each row.
      2. Normalize to sum to 1.0.
      3. Check consistency ratio (CR); warn if CR > 0.10.

    Args:
        pairwise_matrix: n x n matrix where entry (i, j) indicates how much
            objective i is preferred over objective j (1 = equal, 9 = extreme).
        objectives: Objective names. Defaults to the 6 SWaP-C objectives.

    Returns:
        dict mapping objective name -> weight.

    Raises:
        ValueError: If the matrix is not square or doesn't match objective count.
    """
    objectives = objectives or OBJECTIVES
    A = np.array(pairwise_matrix, dtype=float)
    n = A.shape[0]

    if A.shape != (n, n):
        raise ValueError(f"Pairwise matrix must be square, got shape {A.shape}")
    if n != len(objectives):
        raise ValueError(f"Matrix size {n} doesn't match {len(objectives)} objectives")

    # Geometric mean of each row
    geo_means = np.prod(A, axis=1) ** (1.0 / n)
    weights = geo_means / geo_means.sum()

    # Consistency check
    Aw = A @ weights
    lambdas = Aw / weights
    lambda_max = np.mean(lambdas)
    ci = (lambda_max - n) / (n - 1) if n > 1 else 0.0

    # Random consistency index (Saaty) for n = 1..10
    ri_table = [0.0, 0.0, 0.58, 0.90, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49]
    ri = ri_table[n - 1] if n <= len(ri_table) else 1.49
    cr = ci / ri if ri > 0 else 0.0

    if cr > 0.10:
        warnings.warn(
            f"AHP consistency ratio {cr:.3f} exceeds 0.10 threshold. "
            "Consider revising pairwise comparisons.",
            stacklevel=2,
        )

    return {obj: float(w) for obj, w in zip(objectives, weights)}
