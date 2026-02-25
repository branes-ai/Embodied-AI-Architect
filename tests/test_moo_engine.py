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
