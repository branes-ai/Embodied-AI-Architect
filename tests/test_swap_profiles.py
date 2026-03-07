"""Tests for SWaP-C mission profiles."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from embodied_ai_architect.graphs.swap_profiles import (
    OBJECTIVES,
    MissionProfile,
    ahp_weights,
    get_profile,
    list_profiles,
)


def test_preset_weights_sum_to_one():
    """All preset profiles have weights summing to 1.0."""
    for name in list_profiles():
        profile = get_profile(name)
        total = sum(profile.weights.values())
        assert abs(total - 1.0) < 1e-6, f"{name}: weights sum to {total}"


def test_get_profile_known():
    """Known profile returns a MissionProfile."""
    profile = get_profile("drone")
    assert isinstance(profile, MissionProfile)
    assert profile.name == "drone"


def test_get_profile_unknown():
    """Unknown profile raises KeyError."""
    with pytest.raises(KeyError, match="Unknown profile"):
        get_profile("submarine")


def test_list_profiles():
    """Returns all 4 preset names."""
    names = list_profiles()
    assert len(names) == 4
    assert "drone" in names
    assert "rack" in names
    assert "wearable" in names
    assert "vehicle" in names


def test_profile_all_objectives_present():
    """All profiles have weights for all 6 SWaP-C objectives."""
    for name in list_profiles():
        profile = get_profile(name)
        for obj in OBJECTIVES:
            assert obj in profile.weights, f"{name} missing weight for {obj}"


def test_ahp_equal_preference_equal_weights():
    """An all-ones pairwise matrix (equal preference) yields equal weights."""
    n = len(OBJECTIVES)
    ones = np.ones((n, n)).tolist()
    weights = ahp_weights(ones)
    expected = 1.0 / n
    for obj, w in weights.items():
        assert abs(w - expected) < 1e-6, f"{obj}: expected {expected}, got {w}"


def test_ahp_known_ranking():
    """A matrix with clear dominance produces the expected weight ordering."""
    # 3 objectives: A >> B >> C
    objs = ["a", "b", "c"]
    matrix = [
        [1, 3, 5],
        [1 / 3, 1, 3],
        [1 / 5, 1 / 3, 1],
    ]
    weights = ahp_weights(matrix, objectives=objs)
    assert weights["a"] > weights["b"] > weights["c"]


def test_ahp_consistency_check():
    """Highly inconsistent matrix triggers a warning."""
    objs = ["a", "b", "c"]
    # Deliberately inconsistent: A>>B, B>>C, but C>>A
    matrix = [
        [1, 9, 1 / 9],
        [1 / 9, 1, 9],
        [9, 1 / 9, 1],
    ]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ahp_weights(matrix, objectives=objs)
        assert any("consistency" in str(warning.message).lower() for warning in w)
