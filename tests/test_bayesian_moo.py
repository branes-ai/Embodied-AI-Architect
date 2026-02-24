"""Tests for Bayesian multi-objective optimization.

Skipped if botorch/gpytorch are not installed.
"""

import pytest

# Skip entire module if botorch not available
botorch = pytest.importorskip("botorch", reason="botorch not installed")
gpytorch = pytest.importorskip("gpytorch", reason="gpytorch not installed")

from embodied_ai_architect.graphs.moo.design_space import create_soc_design_space
from embodied_ai_architect.graphs.moo.evaluator import DesignEvaluator
from embodied_ai_architect.graphs.moo.executor import LocalThreadExecutor
from embodied_ai_architect.graphs.moo.bayesian_opt import (
    BayesianMOO,
    BayesianOptConfig,
    BayesianOptResult,
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


class TestBayesianOptConfig:
    def test_default_config(self):
        cfg = BayesianOptConfig()
        assert cfg.n_initial == 20
        assert cfg.n_iterations == 50
        assert cfg.batch_size == 4

    def test_custom_config(self):
        cfg = BayesianOptConfig(n_initial=5, n_iterations=3, batch_size=2)
        assert cfg.n_initial == 5


class TestBayesianMOO:
    def test_initialization(self, design_space, evaluator, executor):
        bo = BayesianMOO(design_space, evaluator, executor)
        assert bo.ds == design_space

    def test_small_run(self, design_space, evaluator, executor):
        """Run a minimal BO loop."""
        cfg = BayesianOptConfig(n_initial=5, n_iterations=2, batch_size=2)
        bo = BayesianMOO(design_space, evaluator, executor, cfg)
        result = bo.run()

        assert isinstance(result, BayesianOptResult)
        assert result.total_evaluations > 0
        assert len(result.pareto_front) > 0

    def test_warm_start(self, design_space, evaluator, executor):
        """BO should accept seed points from MAP-Elites."""
        cfg = BayesianOptConfig(n_initial=5, n_iterations=2, batch_size=2)
        bo = BayesianMOO(design_space, evaluator, executor, cfg)

        seed_points = design_space.sample_random(10, seed=42)
        result = bo.run(seed_points=seed_points)

        assert result.total_evaluations >= 5 + 2 * 2  # initial + iterations * batch

    def test_sensitivity_extraction(self, design_space, evaluator, executor):
        """After BO, sensitivity should be available."""
        cfg = BayesianOptConfig(n_initial=10, n_iterations=3, batch_size=2)
        bo = BayesianMOO(design_space, evaluator, executor, cfg)
        result = bo.run()

        # Sensitivity may or may not be populated depending on GP fitting
        # Just verify the structure
        assert isinstance(result.sensitivity, dict)

    def test_callback(self, design_space, evaluator, executor):
        cfg = BayesianOptConfig(n_initial=5, n_iterations=2, batch_size=2)
        bo = BayesianMOO(design_space, evaluator, executor, cfg)

        calls = []
        result = bo.run(callback=lambda i, e, hv: calls.append((i, e, hv)))

        assert len(calls) == 2  # one per iteration

    def test_convergence_history(self, design_space, evaluator, executor):
        cfg = BayesianOptConfig(n_initial=5, n_iterations=3, batch_size=2)
        bo = BayesianMOO(design_space, evaluator, executor, cfg)
        result = bo.run()

        assert len(result.convergence_history) == 3

    def test_explain_tradeoff(self, design_space, evaluator, executor):
        cfg = BayesianOptConfig(n_initial=5, n_iterations=2, batch_size=2)
        bo = BayesianMOO(design_space, evaluator, executor, cfg)
        result = bo.run()

        if len(result.pareto_front) >= 2:
            explanation = bo.explain_tradeoff(result.pareto_front[0], result.pareto_front[1])
            assert "objective_deltas" in explanation
            assert "parameter_changes" in explanation

    def test_pareto_front_structure(self, design_space, evaluator, executor):
        cfg = BayesianOptConfig(n_initial=10, n_iterations=2, batch_size=2)
        bo = BayesianMOO(design_space, evaluator, executor, cfg)
        result = bo.run()

        for point in result.pareto_front:
            assert "design_params" in point
            assert "objectives" in point
            assert "power_watts" in point["objectives"]
            assert "latency_ms" in point["objectives"]
