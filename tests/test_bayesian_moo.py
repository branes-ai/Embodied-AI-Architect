"""Tests for Bayesian multi-objective optimization.

Skipped if botorch/gpytorch are not installed.

Performance note: each fresh BayesianMOO.run() pays the full GP-fit +
acquisition-optimization cost. With production acquisition defaults
(num_restarts=5, raw_samples=256) and a per-test fixture, the suite
was taking >2 min locally and hanging in CI.

Two speedups applied:
  1. A `fast_cfg` fixture sets aggressive acquisition knobs (num_restarts=2,
     raw_samples=32) — fine for tests that just verify structure/contracts.
  2. A module-scoped `shared_bo_result` fixture runs the BO loop ONCE and
     all read-only tests inspect the same result.
"""

import pytest

# Skip entire module if botorch not available
botorch = pytest.importorskip("botorch", reason="botorch not installed")
gpytorch = pytest.importorskip("gpytorch", reason="gpytorch not installed")

from embodied_ai_architect.graphs.moo.design_space import create_soc_design_space  # noqa: E402
from embodied_ai_architect.graphs.moo.evaluator import DesignEvaluator  # noqa: E402
from embodied_ai_architect.graphs.moo.executor import LocalThreadExecutor  # noqa: E402
from embodied_ai_architect.graphs.moo.bayesian_opt import (  # noqa: E402
    BayesianMOO,
    BayesianOptConfig,
    BayesianOptResult,
)

# ---------------------------------------------------------------------------
# Fixtures — module-scoped to amortize GP fitting cost across the suite
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def design_space():
    return create_soc_design_space({"max_power_watts": 10.0})


@pytest.fixture(scope="module")
def evaluator(design_space):
    return DesignEvaluator(
        design_space=design_space,
        base_state={"workload_profile": {"total_estimated_gflops": 5.0}},
    )


@pytest.fixture(scope="module")
def executor(evaluator):
    return LocalThreadExecutor(evaluator, max_workers=4)


@pytest.fixture(scope="module")
def fast_cfg():
    """Minimal-cost BO config for tests.

    Reduces both the BO budget (n_initial/n_iterations/batch_size) AND the
    underlying acquisition optimizer work (num_restarts/raw_samples). For
    contract tests that just verify result shape, the lower-quality solver
    is fine — and the wall time drops by an order of magnitude.
    """
    return BayesianOptConfig(
        n_initial=5,
        n_iterations=2,
        batch_size=2,
        acq_num_restarts=2,
        acq_raw_samples=32,
    )


@pytest.fixture(scope="module")
def shared_bo_result(design_space, evaluator, executor, fast_cfg):
    """Run a single fast BO loop and share its result across read-only tests.

    Tests that need a custom config or invoke .run() with a special argument
    (warm_start, callback, etc.) build their own — but tests that just inspect
    a generic result reuse this one. Cuts the suite from ~120s → ~15s.
    """
    bo = BayesianMOO(design_space, evaluator, executor, fast_cfg)
    return bo.run()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBayesianOptConfig:
    def test_default_config(self):
        cfg = BayesianOptConfig()
        assert cfg.n_initial == 20
        assert cfg.n_iterations == 50
        assert cfg.batch_size == 4
        # Issue #27 follow-up: acquisition knobs are now configurable
        assert cfg.acq_num_restarts == 5
        assert cfg.acq_raw_samples == 256

    def test_custom_config(self):
        cfg = BayesianOptConfig(
            n_initial=5,
            n_iterations=3,
            batch_size=2,
            acq_num_restarts=2,
            acq_raw_samples=32,
        )
        assert cfg.n_initial == 5
        assert cfg.acq_num_restarts == 2
        assert cfg.acq_raw_samples == 32


class TestBayesianMOO:
    def test_initialization(self, design_space, evaluator, executor):
        bo = BayesianMOO(design_space, evaluator, executor)
        assert bo.ds == design_space

    def test_small_run(self, shared_bo_result):
        """The shared BO result must be a valid populated BayesianOptResult."""
        assert isinstance(shared_bo_result, BayesianOptResult)
        assert shared_bo_result.total_evaluations > 0
        assert len(shared_bo_result.pareto_front) > 0

    def test_warm_start(self, design_space, evaluator, executor, fast_cfg):
        """BO should accept seed points from MAP-Elites."""
        bo = BayesianMOO(design_space, evaluator, executor, fast_cfg)
        seed_points = design_space.sample_random(10, seed=42)
        result = bo.run(seed_points=seed_points)
        # initial + iterations * batch = 5 + 2*2
        assert result.total_evaluations >= 5 + 2 * 2

    def test_sensitivity_extraction(self, shared_bo_result):
        """Sensitivity field is always present (may be empty if GP fit failed)."""
        assert isinstance(shared_bo_result.sensitivity, dict)

    def test_callback(self, design_space, evaluator, executor, fast_cfg):
        bo = BayesianMOO(design_space, evaluator, executor, fast_cfg)
        calls = []
        bo.run(callback=lambda i, e, hv: calls.append((i, e, hv)))
        assert len(calls) == fast_cfg.n_iterations

    def test_convergence_history(self, shared_bo_result, fast_cfg):
        assert len(shared_bo_result.convergence_history) == fast_cfg.n_iterations

    def test_explain_tradeoff(self, design_space, evaluator, executor, shared_bo_result):
        if len(shared_bo_result.pareto_front) >= 2:
            bo = BayesianMOO(design_space, evaluator, executor)
            explanation = bo.explain_tradeoff(
                shared_bo_result.pareto_front[0],
                shared_bo_result.pareto_front[1],
            )
            assert "objective_deltas" in explanation
            assert "parameter_changes" in explanation

    def test_pareto_front_structure(self, shared_bo_result):
        for point in shared_bo_result.pareto_front:
            assert "design_params" in point
            assert "objectives" in point
            assert "power_watts" in point["objectives"]
            assert "latency_ms" in point["objectives"]
