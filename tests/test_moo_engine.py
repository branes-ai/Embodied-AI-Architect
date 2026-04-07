"""Tests for the OptimizationEngine orchestrator."""

import pytest

from embodied_ai_architect.graphs.moo.design_space import create_soc_design_space
from embodied_ai_architect.graphs.moo.evaluator import DesignEvaluator
from embodied_ai_architect.graphs.moo.engine import (
    OptimizationConfig,
    OptimizationEngine,
    OptimizationResult,
)
from embodied_ai_architect.graphs.moo.map_elites import MAPElitesConfig


@pytest.fixture
def design_space():
    return create_soc_design_space({"max_power_watts": 10.0, "max_latency_ms": 100.0})


@pytest.fixture
def evaluator(design_space):
    return DesignEvaluator(
        design_space=design_space,
        base_state={
            "workload_profile": {"total_estimated_gflops": 5.0},
            "constraints": {"target_volume": 10000},
        },
    )


class TestOptimizationConfig:
    def test_default_config(self):
        cfg = OptimizationConfig()
        assert cfg.layers == "auto"
        assert cfg.max_workers == 8

    def test_fast_config(self):
        cfg = OptimizationConfig(
            layers="map_elites",
            map_elites=MAPElitesConfig(n_iterations=5, batch_size=8),
        )
        assert cfg.layers == "map_elites"


class TestOptimizationEngine:
    def test_auto_layer_selection(self, design_space, evaluator):
        """Auto mode should at least select MAP-Elites."""
        engine = OptimizationEngine(design_space, evaluator)
        layers = engine._select_layers()
        assert "map_elites" in layers
        engine.shutdown()

    def test_explicit_map_elites_only(self, design_space, evaluator):
        config = OptimizationConfig(layers="map_elites")
        engine = OptimizationEngine(design_space, evaluator, config)
        layers = engine._select_layers()
        assert layers == ["map_elites"]
        engine.shutdown()

    def test_full_pipeline_map_elites(self, design_space, evaluator):
        """Run the full pipeline with MAP-Elites only."""
        config = OptimizationConfig(
            layers="map_elites",
            map_elites=MAPElitesConfig(
                n_iterations=5,
                batch_size=8,
                initial_population=16,
                resolution=5,
            ),
            max_workers=4,
        )
        engine = OptimizationEngine(design_space, evaluator, config)
        try:
            result = engine.run()
        finally:
            engine.shutdown()

        assert isinstance(result, OptimizationResult)
        assert result.total_evaluations > 0
        assert "map_elites" in result.layers_used
        assert len(result.pareto_front) > 0
        assert result.hypervolume >= 0

    def test_callback_invoked(self, design_space, evaluator):
        config = OptimizationConfig(
            layers="map_elites",
            map_elites=MAPElitesConfig(
                n_iterations=3, batch_size=4, initial_population=8, resolution=3
            ),
            max_workers=2,
        )
        engine = OptimizationEngine(design_space, evaluator, config)

        calls = []
        try:
            engine.run(callback=lambda layer, i, e, m: calls.append((layer, i, e, m)))
        finally:
            engine.shutdown()

        assert len(calls) > 0
        assert calls[0][0] == "map_elites"

    def test_result_has_atlas(self, design_space, evaluator):
        config = OptimizationConfig(
            layers="map_elites",
            map_elites=MAPElitesConfig(
                n_iterations=3, batch_size=4, initial_population=8, resolution=3
            ),
        )
        engine = OptimizationEngine(design_space, evaluator, config)
        try:
            result = engine.run()
        finally:
            engine.shutdown()

        assert result.atlas.get("coverage", 0) > 0
        assert result.atlas.get("filled_cells", 0) > 0

    def test_knee_point_found(self, design_space, evaluator):
        config = OptimizationConfig(
            layers="map_elites",
            map_elites=MAPElitesConfig(
                n_iterations=10, batch_size=16, initial_population=32, resolution=5
            ),
        )
        engine = OptimizationEngine(design_space, evaluator, config)
        try:
            result = engine.run()
        finally:
            engine.shutdown()

        # With enough evaluations, knee point should be found
        if len(result.pareto_front) > 1:
            assert result.knee_point is not None
            assert "objectives" in result.knee_point

    def test_explain_tradeoff(self, design_space, evaluator):
        config = OptimizationConfig(
            layers="map_elites",
            map_elites=MAPElitesConfig(
                n_iterations=5, batch_size=8, initial_population=16, resolution=5
            ),
        )
        engine = OptimizationEngine(design_space, evaluator, config)
        try:
            result = engine.run()
        finally:
            pass

        if len(result.pareto_front) >= 2:
            explanation = engine.explain_tradeoff(result.pareto_front[0], result.pareto_front[1])
            assert "objective_deltas" in explanation
            assert "parameter_deltas" in explanation
            assert "summary" in explanation
        engine.shutdown()

    def test_backward_compatible_pareto_results(self, design_space, evaluator):
        """Result should convert to existing pareto_results format."""
        config = OptimizationConfig(
            layers="map_elites",
            map_elites=MAPElitesConfig(
                n_iterations=3, batch_size=4, initial_population=8, resolution=3
            ),
        )
        engine = OptimizationEngine(design_space, evaluator, config)
        try:
            result = engine.run()
        finally:
            engine.shutdown()

        pareto_compat = result.to_pareto_results()
        assert "front" in pareto_compat
        assert "knee_point" in pareto_compat
        assert "total" in pareto_compat
        assert "non_dominated_count" in pareto_compat


class TestSpecialistIntegration:
    def test_moo_specialist(self):
        """Test the moo_explorer specialist function."""
        from embodied_ai_architect.graphs.moo.specialist import moo_explorer
        from embodied_ai_architect.graphs.task_graph import TaskNode

        task = TaskNode(
            id="moo_test",
            name="MOO Test",
            agent="moo_explorer",
            dependencies=[],
            metadata={"fast_mode": True},
        )

        state = {
            "goal": "drone perception SoC",
            "constraints": {"max_power_watts": 10.0, "max_latency_ms": 100.0},
            "workload_profile": {"total_estimated_gflops": 5.0},
        }

        result = moo_explorer(task, state)

        assert "summary" in result
        assert "pareto_results" in result
        assert "moo_results" in result
        assert "_state_updates" in result
        assert "pareto_results" in result["_state_updates"]
        assert "moo_results" in result["_state_updates"]


class TestParetoFrontierMerge:
    """Issue #23: Pareto frontier accumulation across MOO iterations."""

    def test_merge_first_run_initializes_frontier(self):
        from embodied_ai_architect.graphs.moo.specialist import _merge_pareto_frontiers

        new_points = [
            {"power": 2.0, "latency": 10.0, "cost": 50.0, "area": 5.0, "metadata": {}},
            {"power": 3.0, "latency": 8.0, "cost": 60.0, "area": 6.0, "metadata": {}},
        ]
        merged, added, removed = _merge_pareto_frontiers([], new_points)
        assert len(merged) == 2
        assert added == 2
        assert removed == 0
        assert all(p["dominated"] is False for p in merged)

    def test_merge_dominated_points_removed(self):
        """A new point that dominates an old one should remove the old."""
        from embodied_ai_architect.graphs.moo.specialist import _merge_pareto_frontiers

        # Old frontier: a single weak point
        accumulated = [
            {"power": 5.0, "latency": 20.0, "cost": 100.0, "area": 10.0, "metadata": {}},
        ]
        # New point dominates the old (better in every objective)
        new_points = [
            {"power": 2.0, "latency": 10.0, "cost": 50.0, "area": 5.0, "metadata": {}},
        ]
        merged, added, removed = _merge_pareto_frontiers(accumulated, new_points)
        assert len(merged) == 1
        assert merged[0]["power"] == 2.0
        assert added == 1
        assert removed == 1  # the old point was dominated and removed

    def test_merge_preserves_non_dominated_old_points(self):
        """Old points not dominated by new points should survive the merge."""
        from embodied_ai_architect.graphs.moo.specialist import _merge_pareto_frontiers

        # Two non-comparable old points (trade off power vs latency)
        accumulated = [
            {"power": 2.0, "latency": 50.0, "cost": 50.0, "area": 5.0, "metadata": {}},
            {"power": 8.0, "latency": 5.0, "cost": 50.0, "area": 5.0, "metadata": {}},
        ]
        # New point on a third trade-off
        new_points = [
            {"power": 5.0, "latency": 20.0, "cost": 30.0, "area": 5.0, "metadata": {}},
        ]
        merged, added, removed = _merge_pareto_frontiers(accumulated, new_points)
        assert len(merged) == 3
        assert added == 1
        assert removed == 0

    def test_three_iterations_grow_monotonically(self):
        """Issue #23 acceptance criteria: 3 MOO iterations should accumulate
        the frontier (point count grows or stays equal across iterations,
        depending on dominance)."""
        from embodied_ai_architect.graphs.moo.specialist import (
            _build_history_entry,
            _merge_pareto_frontiers,
        )

        # Iteration 0: discover the first 3 design points
        accumulated = []
        history = []
        run0 = [
            {"power": 5.0, "latency": 30.0, "cost": 80.0, "area": 8.0, "metadata": {}},
            {"power": 8.0, "latency": 15.0, "cost": 100.0, "area": 10.0, "metadata": {}},
            {"power": 3.0, "latency": 50.0, "cost": 60.0, "area": 6.0, "metadata": {}},
        ]
        merged, added, removed = _merge_pareto_frontiers(accumulated, run0)
        history.append(_build_history_entry(0, merged, added, removed, 100.0))
        assert len(merged) == 3
        assert history[-1]["num_points"] == 3
        assert history[-1]["new_points_added"] == 3

        # Iteration 1: discover 2 new non-comparable points + 1 dominator
        accumulated = merged
        run1 = [
            {"power": 4.0, "latency": 25.0, "cost": 70.0, "area": 7.0, "metadata": {}},
            {"power": 6.0, "latency": 12.0, "cost": 90.0, "area": 9.0, "metadata": {}},
            # This dominates run0[0] (5/30/80/8)
            {"power": 4.5, "latency": 28.0, "cost": 75.0, "area": 7.5, "metadata": {}},
        ]
        merged, added, removed = _merge_pareto_frontiers(accumulated, run1)
        history.append(_build_history_entry(1, merged, added, removed, 130.0))
        # Frontier evolves: some new points added, possibly some old removed
        assert history[-1]["new_points_added"] > 0
        assert history[-1]["num_points"] >= 1

        # Iteration 2: another exploration round
        accumulated = merged
        run2 = [
            {"power": 2.0, "latency": 60.0, "cost": 50.0, "area": 5.0, "metadata": {}},
        ]
        merged, added, removed = _merge_pareto_frontiers(accumulated, run2)
        history.append(_build_history_entry(2, merged, added, removed, 145.0))

        # Verify the history grew
        assert len(history) == 3
        assert history[0]["iteration"] == 0
        assert history[1]["iteration"] == 1
        assert history[2]["iteration"] == 2
        # Hypervolume should increase as we accumulate (better designs found)
        assert history[2]["hypervolume"] > history[0]["hypervolume"]

    def test_moo_explorer_writes_frontier_history(self):
        """End-to-end: running moo_explorer twice should produce a 2-entry
        frontier_history in the state updates."""
        from embodied_ai_architect.graphs.moo.specialist import moo_explorer
        from embodied_ai_architect.graphs.task_graph import TaskNode

        task = TaskNode(
            id="moo_test",
            name="MOO Test",
            agent="moo_explorer",
            dependencies=[],
            metadata={"fast_mode": True},
        )

        # First run — empty state
        state = {
            "goal": "drone perception SoC",
            "constraints": {"max_power_watts": 15.0, "max_latency_ms": 100.0},
            "workload_profile": {"total_estimated_gflops": 5.0},
            "iteration": 0,
        }
        result1 = moo_explorer(task, state)
        updates1 = result1["_state_updates"]
        assert "pareto_frontier_history" in updates1
        assert len(updates1["pareto_frontier_history"]) == 1
        assert updates1["pareto_frontier_history"][0]["iteration"] == 0

        # Second run — feed prior state back in (simulating optimizer loop)
        state2 = dict(state)
        state2["pareto_points"] = updates1["pareto_points"]
        state2["pareto_frontier_history"] = updates1["pareto_frontier_history"]
        state2["iteration"] = 1
        result2 = moo_explorer(task, state2)
        updates2 = result2["_state_updates"]

        # History should now have 2 entries
        assert len(updates2["pareto_frontier_history"]) == 2
        assert updates2["pareto_frontier_history"][0]["iteration"] == 0
        assert updates2["pareto_frontier_history"][1]["iteration"] == 1
