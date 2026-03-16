"""Tests for capability-per-watt integration in the MOO pipeline.

Validates that accuracy and capability_per_watt flow correctly through:
- DesignSpace (as objectives)
- DesignEvaluator (computation)
- OptimizationEngine (Pareto front with mixed directions)
"""

import pytest

from embodied_ai_architect.graphs.moo.design_space import (
    ObjectiveDirection,
    create_soc_design_space,
    create_swap_design_space,
)
from embodied_ai_architect.graphs.moo.evaluator import DesignEvaluator
from embodied_ai_architect.graphs.moo.engine import (
    OptimizationConfig,
    OptimizationEngine,
)
from embodied_ai_architect.graphs.moo.map_elites import MAPElitesConfig

DEFAULT_PARAMS = {
    "process_nm": 28,
    "clock_mhz": 500.0,
    "array_rows": 8,
    "array_cols": 8,
    "sram_kb": 256,
    "num_compute_tiles": 4,
    "noc_link_width_bits": 256,
}


class TestDesignSpaceWithAccuracy:
    def test_soc_space_without_accuracy(self):
        ds = create_soc_design_space()
        assert "accuracy_percent" not in ds.objectives
        assert "capability_per_watt" not in ds.objectives
        assert ds.n_obj == 4

    def test_soc_space_with_accuracy(self):
        ds = create_soc_design_space({"include_accuracy": True})
        assert "accuracy_percent" in ds.objectives
        assert "capability_per_watt" in ds.objectives
        assert ds.n_obj == 6

        # Check directions
        acc_idx = ds.objectives.index("accuracy_percent")
        cpw_idx = ds.objectives.index("capability_per_watt")
        assert ds.directions[acc_idx] == ObjectiveDirection.MAXIMIZE
        assert ds.directions[cpw_idx] == ObjectiveDirection.MAXIMIZE

    def test_soc_space_with_min_accuracy_constraint(self):
        ds = create_soc_design_space(
            {
                "include_accuracy": True,
                "min_accuracy_percent": 40.0,
            }
        )
        assert "accuracy_percent" in ds.constraint_bounds
        lo, hi = ds.constraint_bounds["accuracy_percent"]
        assert lo == 40.0
        assert hi == 100.0

    def test_swap_space_with_accuracy(self):
        ds = create_swap_design_space({"include_accuracy": True})
        assert "accuracy_percent" in ds.objectives
        assert "capability_per_watt" in ds.objectives
        assert ds.n_obj == 8  # 6 SWaP-C + 2 capability


class TestEvaluatorWithAccuracy:
    @pytest.fixture
    def accuracy_evaluator(self):
        ds = create_soc_design_space(
            {
                "include_accuracy": True,
                "max_power_watts": 10.0,
            }
        )
        return DesignEvaluator(
            design_space=ds,
            base_state={
                "workload_profile": {"total_estimated_gflops": 8.0},
                "constraints": {"workload_type": "detection"},
            },
        )

    @pytest.fixture
    def ppa_only_evaluator(self):
        ds = create_soc_design_space({"max_power_watts": 10.0})
        return DesignEvaluator(
            design_space=ds,
            base_state={"workload_profile": {"total_estimated_gflops": 8.0}},
        )

    def test_accuracy_in_objectives(self, accuracy_evaluator):
        result = accuracy_evaluator.evaluate(DEFAULT_PARAMS)
        assert "accuracy_percent" in result.objectives
        assert "capability_per_watt" in result.objectives
        assert result.objectives["accuracy_percent"] > 0
        assert result.objectives["capability_per_watt"] > 0

    def test_accuracy_not_in_ppa_objectives(self, ppa_only_evaluator):
        result = ppa_only_evaluator.evaluate(DEFAULT_PARAMS)
        assert "accuracy_percent" not in result.objectives
        assert "capability_per_watt" not in result.objectives

    def test_accuracy_always_in_metadata(self, ppa_only_evaluator):
        result = ppa_only_evaluator.evaluate(DEFAULT_PARAMS)
        assert "accuracy_percent" in result.metadata
        assert "capability_per_watt" in result.metadata
        assert "gops_per_watt" in result.metadata
        assert "model_family" in result.metadata

    def test_more_compute_enables_better_accuracy(self):
        ds = create_soc_design_space({"include_accuracy": True})
        evaluator = DesignEvaluator(
            design_space=ds,
            base_state={
                "workload_profile": {"total_estimated_gflops": 8.0},
                "constraints": {"workload_type": "detection"},
            },
        )
        small = evaluator.evaluate(
            {
                **DEFAULT_PARAMS,
                "array_rows": 2,
                "array_cols": 2,
                "num_compute_tiles": 1,
            }
        )
        large = evaluator.evaluate(
            {
                **DEFAULT_PARAMS,
                "array_rows": 32,
                "array_cols": 32,
                "num_compute_tiles": 8,
            }
        )
        # Larger compute should support at least as good accuracy
        assert large.objectives["accuracy_percent"] >= small.objectives["accuracy_percent"]

    def test_explicit_model_in_workload(self):
        ds = create_soc_design_space({"include_accuracy": True})
        evaluator = DesignEvaluator(
            design_space=ds,
            base_state={
                "workload_profile": {
                    "total_estimated_gflops": 28.6,
                    "model_accuracy": 44.9,
                    "model_family": "yolov8",
                    "model_variant": "s",
                },
                "constraints": {},
            },
        )
        result = evaluator.evaluate(DEFAULT_PARAMS)
        # Should use the explicit model accuracy (possibly adjusted for dtype)
        assert result.objectives["accuracy_percent"] == pytest.approx(44.9, abs=0.5)

    def test_error_returns_zero_accuracy(self):
        ds = create_soc_design_space({"include_accuracy": True})
        evaluator = DesignEvaluator(
            design_space=ds,
            base_state={},
            constraint_bounds={},
        )
        # Pass completely invalid params that cause evaluation to fail
        result = evaluator.evaluate({"process_nm": -999, "clock_mhz": -1})
        if not result.feasible:
            assert result.objectives.get("accuracy_percent", 0.0) == 0.0

    def test_feasibility_includes_accuracy_constraint(self):
        ds = create_soc_design_space(
            {
                "include_accuracy": True,
                "min_accuracy_percent": 99.0,  # impossibly high
            }
        )
        evaluator = DesignEvaluator(
            design_space=ds,
            base_state={
                "workload_profile": {"total_estimated_gflops": 8.0},
                "constraints": {"workload_type": "detection"},
            },
        )
        result = evaluator.evaluate(DEFAULT_PARAMS)
        # No detection model achieves 99% mAP@50
        assert not result.feasible
        assert result.constraints_satisfied.get("accuracy_percent") is False


class TestOptimizationEngineWithAccuracy:
    def test_engine_with_accuracy_objectives(self):
        ds = create_soc_design_space({"include_accuracy": True})
        evaluator = DesignEvaluator(
            design_space=ds,
            base_state={
                "workload_profile": {"total_estimated_gflops": 8.0},
                "constraints": {"workload_type": "detection"},
            },
        )
        config = OptimizationConfig(
            layers="map_elites",
            map_elites=MAPElitesConfig(
                n_iterations=5,
                batch_size=16,
                initial_population=32,
            ),
        )
        engine = OptimizationEngine(ds, evaluator, config)
        try:
            result = engine.run()
            assert result.total_evaluations > 0
            assert len(result.pareto_front) > 0

            # Each Pareto point should have accuracy objectives
            for point in result.pareto_front:
                assert "accuracy_percent" in point["objectives"]
                assert "capability_per_watt" in point["objectives"]
                assert point["objectives"]["accuracy_percent"] > 0
        finally:
            engine.shutdown()

    def test_pareto_front_handles_mixed_directions(self):
        """Verify that MAXIMIZE objectives are handled correctly in dominance."""
        ds = create_soc_design_space({"include_accuracy": True})
        evaluator = DesignEvaluator(
            design_space=ds,
            base_state={
                "workload_profile": {"total_estimated_gflops": 8.0},
                "constraints": {"workload_type": "detection"},
            },
        )
        config = OptimizationConfig(
            layers="map_elites",
            map_elites=MAPElitesConfig(
                n_iterations=10,
                batch_size=32,
                initial_population=64,
            ),
        )
        engine = OptimizationEngine(ds, evaluator, config)
        try:
            result = engine.run()
            front = result.pareto_front

            # Verify no point on the front is dominated
            for i, pi in enumerate(front):
                for j, pj in enumerate(front):
                    if i == j:
                        continue
                    oi = pi["objectives"]
                    oj = pj["objectives"]
                    # j should NOT dominate i
                    # (domination with mixed directions)
                    j_better_or_equal = True
                    j_strictly_better = False
                    for k, name in enumerate(ds.objectives):
                        if ds.directions[k] == ObjectiveDirection.MINIMIZE:
                            if oj[name] > oi[name]:
                                j_better_or_equal = False
                            if oj[name] < oi[name]:
                                j_strictly_better = True
                        else:  # MAXIMIZE
                            if oj[name] < oi[name]:
                                j_better_or_equal = False
                            if oj[name] > oi[name]:
                                j_strictly_better = True
                    assert not (
                        j_better_or_equal and j_strictly_better
                    ), f"Point {j} dominates point {i} on the Pareto front"
        finally:
            engine.shutdown()
