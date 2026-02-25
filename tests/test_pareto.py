"""Tests for Pareto front exploration and design space analysis.

Tests ParetoPoint creation, compute_pareto_front, identify_knee_point,
and the design_explorer specialist agent.
"""

import pytest

from embodied_ai_architect.graphs.pareto import (
    ParetoPoint,
    compute_pareto_front,
    design_explorer,
    identify_knee_point,
)
from embodied_ai_architect.graphs.soc_state import (
    DesignConstraints,
    create_initial_soc_state,
)
from embodied_ai_architect.graphs.task_graph import TaskNode

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def test_state():
    return create_initial_soc_state(
        goal="test goal with object detection",
        constraints=DesignConstraints(max_power_watts=15.0, max_cost_usd=100.0),
        use_case="test",
    )


@pytest.fixture
def sample_candidates():
    """Hardware candidates for Pareto front testing."""
    return [
        {"name": "A", "tdp_watts": 5.0, "latency_ms": 10.0, "cost_usd": 25.0},
        {"name": "B", "tdp_watts": 15.0, "latency_ms": 5.0, "cost_usd": 199.0},
        {"name": "C", "tdp_watts": 25.0, "latency_ms": 10.0, "cost_usd": 150.0},
    ]


@pytest.fixture
def dominated_candidates():
    """Candidates where one clearly dominates another."""
    return [
        {"name": "A", "tdp_watts": 5.0, "latency_ms": 10.0, "cost_usd": 25.0},
        {"name": "B", "tdp_watts": 10.0, "latency_ms": 20.0, "cost_usd": 50.0},
    ]


def make_task(task_id: str, agent: str, deps: list[str] | None = None) -> TaskNode:
    return TaskNode(id=task_id, name=f"Test {agent}", agent=agent, dependencies=deps or [])


# ============================================================================
# ParetoPoint Tests
# ============================================================================


class TestParetoPoint:
    def test_creation_with_defaults(self):
        """Test creating a ParetoPoint with default values."""
        point = ParetoPoint()
        assert point.hardware_name == ""
        assert point.power == 0.0
        assert point.latency == 0.0
        assert point.cost == 0.0
        assert point.dominated is False
        assert point.knee_point is False
        assert point.metadata == {}

    def test_creation_with_values(self):
        """Test creating a ParetoPoint with explicit values."""
        point = ParetoPoint(
            hardware_name="TestChip",
            power=5.0,
            latency=10.0,
            cost=25.0,
            metadata={"arch": "arm"},
        )
        assert point.hardware_name == "TestChip"
        assert point.power == 5.0
        assert point.latency == 10.0
        assert point.cost == 25.0
        assert point.metadata == {"arch": "arm"}


# ============================================================================
# compute_pareto_front Tests
# ============================================================================


class TestComputeParetoFront:
    def test_three_candidates_domination(self, sample_candidates):
        """Test Pareto front with 3 candidates where domination may occur."""
        front = compute_pareto_front(sample_candidates)

        assert len(front) == 3
        assert all(isinstance(p, ParetoPoint) for p in front)

        # A should not be dominated (best on power and cost)
        point_a = next(p for p in front if p.hardware_name == "A")
        assert point_a.dominated is False
        assert point_a.power == 5.0
        assert point_a.latency == 10.0
        assert point_a.cost == 25.0

        # B should not be dominated (best on latency)
        point_b = next(p for p in front if p.hardware_name == "B")
        assert point_b.dominated is False

        # C should be dominated (worse than A in power and cost, worse than B in latency)
        point_c = next(p for p in front if p.hardware_name == "C")
        assert point_c.dominated is True

    def test_single_candidate(self):
        """Test Pareto front with single candidate (should be non-dominated)."""
        candidates = [{"name": "Solo", "tdp_watts": 10.0, "latency_ms": 20.0, "cost_usd": 50.0}]
        front = compute_pareto_front(candidates)

        assert len(front) == 1
        assert front[0].dominated is False
        assert front[0].hardware_name == "Solo"

    def test_empty_list(self):
        """Test Pareto front with empty candidate list."""
        front = compute_pareto_front([])
        assert front == []

    def test_clear_domination(self, dominated_candidates):
        """Test case where one candidate strictly dominates another."""
        front = compute_pareto_front(dominated_candidates)

        assert len(front) == 2

        # A dominates B (better on all objectives)
        point_a = next(p for p in front if p.hardware_name == "A")
        point_b = next(p for p in front if p.hardware_name == "B")

        assert point_a.dominated is False
        assert point_b.dominated is True


# ============================================================================
# identify_knee_point Tests
# ============================================================================


class TestIdentifyKneePoint:
    def test_knee_point_with_non_dominated(self, sample_candidates):
        """Test knee point identification with non-dominated points."""
        front = compute_pareto_front(sample_candidates)
        knee = identify_knee_point(front)

        assert knee is not None
        assert isinstance(knee, ParetoPoint)
        assert knee.knee_point is True
        assert knee.dominated is False

    def test_knee_point_with_no_non_dominated(self):
        """Test knee point identification when no non-dominated points exist."""
        # Create all dominated points (artificial scenario)
        front = [
            ParetoPoint(hardware_name="A", power=5.0, latency=10.0, cost=25.0, dominated=True),
            ParetoPoint(hardware_name="B", power=10.0, latency=20.0, cost=50.0, dominated=True),
        ]
        knee = identify_knee_point(front)

        assert knee is None

    def test_knee_point_marking(self, sample_candidates):
        """Test that knee_point flag is set to True on result."""
        front = compute_pareto_front(sample_candidates)
        knee = identify_knee_point(front)

        assert knee is not None
        assert knee.knee_point is True

        # Check that exactly one point in the front has knee_point=True
        knee_count = sum(1 for p in front if p.knee_point)
        assert knee_count == 1

    def test_knee_point_single_non_dominated(self):
        """Test knee point with single non-dominated point."""
        candidates = [{"name": "Solo", "tdp_watts": 10.0, "latency_ms": 20.0, "cost_usd": 50.0}]
        front = compute_pareto_front(candidates)
        knee = identify_knee_point(front)

        assert knee is not None
        assert knee.hardware_name == "Solo"
        assert knee.knee_point is True


# ============================================================================
# design_explorer Specialist Tests
# ============================================================================


class TestDesignExplorer:
    def test_explore_with_candidates(self, test_state, sample_candidates):
        """Test design_explorer specialist with hardware candidates."""
        task = make_task("t1", "design_explorer")

        # Add candidates and workload profile to state
        test_state["hardware_candidates"] = sample_candidates
        test_state["workload_profile"] = {"total_estimated_gflops": 5.0}

        result = design_explorer(task, test_state)

        assert "summary" in result
        assert "pareto_results" in result
        assert "_state_updates" in result

        # Check pareto_results structure
        pareto_results = result["pareto_results"]
        assert "front" in pareto_results
        assert "knee_point" in pareto_results
        assert "total" in pareto_results
        assert "non_dominated_count" in pareto_results

        # Should have 3 points in the front
        assert pareto_results["total"] == 3
        assert len(pareto_results["front"]) == 3

        # Should have state updates
        assert result["_state_updates"]["pareto_results"] == pareto_results

    def test_explore_with_empty_candidates(self, test_state):
        """Test design_explorer with empty hardware candidates list."""
        task = make_task("t1", "design_explorer")

        result = design_explorer(task, test_state)

        assert "summary" in result
        assert "No hardware candidates" in result["summary"]

        pareto_results = result["pareto_results"]
        assert pareto_results["front"] == []
        assert pareto_results["knee_point"] is None
        assert pareto_results["total"] == 0

        # Check state updates
        assert "_state_updates" in result
        assert result["_state_updates"]["pareto_results"] == pareto_results

    def test_state_updates_written(self, test_state, sample_candidates):
        """Test that design_explorer writes pareto_results to _state_updates."""
        task = make_task("t1", "design_explorer")
        test_state["hardware_candidates"] = sample_candidates

        result = design_explorer(task, test_state)

        assert "_state_updates" in result
        assert "pareto_results" in result["_state_updates"]

        # Verify the structure matches the returned pareto_results
        state_results = result["_state_updates"]["pareto_results"]
        assert state_results == result["pareto_results"]

    def test_knee_point_in_summary(self, test_state, sample_candidates):
        """Test that knee point is mentioned in summary when found."""
        task = make_task("t1", "design_explorer")
        test_state["hardware_candidates"] = sample_candidates

        result = design_explorer(task, test_state)

        assert "summary" in result
        assert "knee=" in result["summary"]
        assert "non-dominated" in result["summary"]

    def test_non_dominated_count(self, test_state, sample_candidates):
        """Test that non-dominated count is correct in results."""
        task = make_task("t1", "design_explorer")
        test_state["hardware_candidates"] = sample_candidates

        result = design_explorer(task, test_state)

        pareto_results = result["pareto_results"]
        # From sample_candidates, A and B should be non-dominated, C dominated
        assert pareto_results["non_dominated_count"] == 2
