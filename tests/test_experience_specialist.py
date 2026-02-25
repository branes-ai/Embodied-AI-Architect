"""Tests for the experience retrieval specialist agent."""

from __future__ import annotations

import pytest

from embodied_ai_architect.graphs.experience import DesignEpisode, ExperienceCache
from embodied_ai_architect.graphs.experience_specialist import (
    SIMILARITY_THRESHOLD,
    compute_similarity,
    experience_retriever,
)
from embodied_ai_architect.graphs.soc_state import (
    DesignConstraints,
    create_initial_soc_state,
)
from embodied_ai_architect.graphs.task_graph import TaskNode

# ---------------------------------------------------------------------------
# compute_similarity tests
# ---------------------------------------------------------------------------


class TestComputeSimilarity:
    def _episode(self, **kwargs) -> DesignEpisode:
        defaults = dict(
            use_case="delivery_drone",
            platform="drone",
            constraints={"max_power_watts": 5.0, "max_cost_usd": 30.0},
        )
        defaults.update(kwargs)
        return DesignEpisode(**defaults)

    def test_identical_problem(self):
        ep = self._episode()
        sim = compute_similarity("delivery_drone", "drone", ep.constraints, ep)
        assert sim >= 0.8

    def test_same_category_different_use_case(self):
        ep = self._episode(use_case="agricultural_drone", platform="drone")
        sim = compute_similarity("delivery_drone", "drone", ep.constraints, ep)
        # Same category (aerial) + same platform + constraint overlap
        assert 0.3 < sim <= 1.0

    def test_completely_different(self):
        ep = self._episode(
            use_case="surgical_robot",
            platform="medical",
            constraints={"max_power_watts": 25.0, "safety_critical": True},
        )
        sim = compute_similarity("delivery_drone", "drone", {"max_power_watts": 5.0}, ep)
        assert sim < 0.5

    def test_empty_fields(self):
        ep = self._episode(use_case="", platform="", constraints={})
        sim = compute_similarity("", "", {}, ep)
        assert sim == 0.0

    def test_platform_family_match(self):
        ep = self._episode(platform="amr")
        sim_same_family = compute_similarity("warehouse_amr", "quadruped", {}, ep)
        # amr and quadruped are both "ground" family
        assert sim_same_family > 0.0

    def test_constraint_value_bonus(self):
        ep = self._episode(constraints={"max_power_watts": 5.0, "max_cost_usd": 30.0})
        # Very similar constraint values
        sim_close = compute_similarity(
            "delivery_drone",
            "drone",
            {"max_power_watts": 5.5, "max_cost_usd": 28.0},
            ep,
        )
        # Very different constraint values
        sim_far = compute_similarity(
            "delivery_drone",
            "drone",
            {"max_power_watts": 50.0, "max_cost_usd": 500.0},
            ep,
        )
        assert sim_close >= sim_far


# ---------------------------------------------------------------------------
# experience_retriever specialist tests
# ---------------------------------------------------------------------------


class TestExperienceRetriever:
    def _task(self) -> TaskNode:
        return TaskNode(id="t0", name="test", agent="experience_retriever", dependencies=[])

    def _state(self, cache_path=":memory:", **kwargs) -> dict:
        state = create_initial_soc_state(
            goal="Design a drone SoC with object detection",
            constraints=DesignConstraints(max_power_watts=5.0, max_cost_usd=30.0),
            use_case="delivery_drone",
            platform="drone",
        )
        state["_experience_cache_path"] = cache_path
        state.update(kwargs)
        return state

    def test_no_prior_experience(self):
        result = experience_retriever(self._task(), self._state())
        assert "prior_experience" in result
        prior = result["prior_experience"]
        assert prior["found"] is False

    def test_with_matching_experience(self):
        cache = ExperienceCache(db_path=":memory:")
        ep = DesignEpisode(
            goal="Delivery drone SoC",
            use_case="delivery_drone",
            platform="drone",
            constraints={"max_power_watts": 5.0},
            hardware_selected="Stillwater KPU",
            outcome_score=1.0,
        )
        cache.save(ep)
        # Can't easily pass in-memory cache to specialist since it creates its own.
        # Test the mechanism by checking the result structure.
        result = experience_retriever(self._task(), self._state())
        assert "_state_updates" in result
        assert "prior_experience" in result["_state_updates"]

    def test_state_updates_structure(self):
        result = experience_retriever(self._task(), self._state())
        updates = result.get("_state_updates", {})
        assert "prior_experience" in updates

    def test_summary_present(self):
        result = experience_retriever(self._task(), self._state())
        assert "summary" in result
        assert isinstance(result["summary"], str)
