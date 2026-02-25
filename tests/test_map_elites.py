"""Tests for the MAP-Elites algorithm."""

import pytest

from embodied_ai_architect.graphs.moo.design_space import create_soc_design_space
from embodied_ai_architect.graphs.moo.evaluator import DesignEvaluator
from embodied_ai_architect.graphs.moo.executor import LocalThreadExecutor
from embodied_ai_architect.graphs.moo.map_elites import (
    MAPElites,
    MAPElitesConfig,
    MAPElitesResult,
)


@pytest.fixture
def design_space():
    return create_soc_design_space({"max_power_watts": 10.0})


@pytest.fixture
def evaluator(design_space):
    return DesignEvaluator(
        design_space=design_space,
        base_state={"workload_profile": {"total_estimated_gflops": 5.0}},
    )


@pytest.fixture
def executor(evaluator):
    return LocalThreadExecutor(evaluator, max_workers=4)


class TestMAPElitesConfig:
    def test_default_config(self):
        cfg = MAPElitesConfig()
        assert cfg.n_iterations == 100
        assert cfg.batch_size == 64
        assert len(cfg.feature_dimensions) == 2

    def test_custom_config(self):
        cfg = MAPElitesConfig(n_iterations=10, batch_size=8, resolution=10)
        assert cfg.n_iterations == 10
        assert cfg.resolution == 10


class TestMAPElites:
    def test_initialization(self, design_space, evaluator, executor):
        me = MAPElites(design_space, evaluator, executor)
        assert me.ds == design_space

    def test_small_run(self, design_space, evaluator, executor):
        """Run a small MAP-Elites to verify basic functionality."""
        cfg = MAPElitesConfig(
            n_iterations=5,
            batch_size=8,
            initial_population=16,
            resolution=5,
        )
        me = MAPElites(design_space, evaluator, executor, cfg)
        result = me.run()

        assert isinstance(result, MAPElitesResult)
        assert result.total_evaluations > 0
        assert result.filled_cells > 0
        assert 0 <= result.coverage <= 1.0
        assert result.total_cells == 5**2  # 2D grid, resolution=5

    def test_callback_called(self, design_space, evaluator, executor):
        """Verify callback is invoked during run."""
        cfg = MAPElitesConfig(n_iterations=3, batch_size=4, initial_population=8, resolution=3)
        me = MAPElites(design_space, evaluator, executor, cfg)

        callback_calls = []
        me.run(callback=lambda i, e, c: callback_calls.append((i, e, c)))

        # Should be called: once for initial + once per iteration
        assert len(callback_calls) >= 4  # 1 init + 3 iterations

    def test_result_structure(self, design_space, evaluator, executor):
        cfg = MAPElitesConfig(n_iterations=3, batch_size=4, initial_population=8, resolution=4)
        me = MAPElites(design_space, evaluator, executor, cfg)
        result = me.run()

        # Grid cells should be populated
        assert isinstance(result.grid_cells, dict)
        assert len(result.grid_cells) > 0

        # Best per objective should exist
        assert isinstance(result.best_per_objective, dict)

        # Best designs should be non-empty
        assert len(result.best_designs) > 0
        assert "design_params" in result.best_designs[0]
        assert "objectives" in result.best_designs[0]

    def test_mutation_stays_in_bounds(self, design_space, evaluator, executor):
        """Verify that mutations produce valid parameters."""
        cfg = MAPElitesConfig(n_iterations=10, batch_size=16, initial_population=16, resolution=5)
        me = MAPElites(design_space, evaluator, executor, cfg)
        result = me.run()

        # All designs should have valid parameters
        for cell_data in result.grid_cells.values():
            params = cell_data["design_params"]
            for var in design_space.variables:
                val = params.get(var.name)
                assert val is not None, f"Missing {var.name}"
                assert var.validate_value(val), f"{var.name}={val} out of bounds"

    def test_coverage_increases(self, design_space, evaluator, executor):
        """More iterations should generally lead to better coverage."""
        cfg_small = MAPElitesConfig(
            n_iterations=2, batch_size=4, initial_population=8, resolution=5
        )
        cfg_large = MAPElitesConfig(
            n_iterations=10, batch_size=16, initial_population=32, resolution=5
        )

        me_small = MAPElites(design_space, evaluator, executor, cfg_small)
        me_large = MAPElites(design_space, evaluator, executor, cfg_large)

        result_small = me_small.run()
        result_large = me_large.run()

        assert result_large.filled_cells >= result_small.filled_cells
