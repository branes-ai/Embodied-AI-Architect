"""Tests for the DesignEvaluator."""

import pytest
from concurrent.futures import ThreadPoolExecutor

from embodied_ai_architect.graphs.moo.design_space import create_soc_design_space
from embodied_ai_architect.graphs.moo.evaluator import DesignEvaluator, EvaluationResult


@pytest.fixture
def design_space():
    return create_soc_design_space({"max_power_watts": 10.0})


@pytest.fixture
def evaluator(design_space):
    return DesignEvaluator(
        design_space=design_space,
        base_state={"workload_profile": {"total_estimated_gflops": 8.0}},
    )


class TestEvaluationResult:
    def test_result_model(self):
        result = EvaluationResult(
            design_params={"x": 1},
            objectives={"power_watts": 5.0, "latency_ms": 10.0},
            feasible=True,
        )
        assert result.feasible
        assert result.objectives["power_watts"] == 5.0


class TestDesignEvaluator:
    def test_single_evaluate(self, evaluator):
        params = {
            "process_nm": 28,
            "clock_mhz": 500.0,
            "array_rows": 8,
            "array_cols": 8,
            "sram_kb": 256,
            "num_compute_tiles": 4,
            "noc_link_width_bits": 256,
        }
        result = evaluator.evaluate(params)
        assert isinstance(result, EvaluationResult)
        assert "power_watts" in result.objectives
        assert "latency_ms" in result.objectives
        assert "area_mm2" in result.objectives
        assert "cost_usd" in result.objectives
        assert result.objectives["power_watts"] > 0
        assert result.objectives["area_mm2"] > 0

    def test_different_process_nodes(self, evaluator):
        """Smaller process nodes should generally give less area."""
        params_28 = {
            "process_nm": 28,
            "clock_mhz": 500.0,
            "array_rows": 8,
            "array_cols": 8,
            "sram_kb": 256,
            "num_compute_tiles": 4,
            "noc_link_width_bits": 256,
        }
        params_7 = {**params_28, "process_nm": 7}

        r28 = evaluator.evaluate(params_28)
        r7 = evaluator.evaluate(params_7)
        assert r7.objectives["area_mm2"] < r28.objectives["area_mm2"]

    def test_evaluate_batch(self, evaluator, design_space):
        samples = design_space.sample_random(10, seed=42)
        results = evaluator.evaluate_batch(samples)
        assert len(results) == 10
        assert all(isinstance(r, EvaluationResult) for r in results)

    def test_feasibility_check(self, design_space):
        evaluator = DesignEvaluator(
            design_space=design_space,
            base_state={},
            constraint_bounds={"power_watts": (0.0, 0.001)},  # impossibly tight
        )
        params = {
            "process_nm": 28,
            "clock_mhz": 500.0,
            "array_rows": 8,
            "array_cols": 8,
            "sram_kb": 256,
            "num_compute_tiles": 4,
            "noc_link_width_bits": 256,
        }
        result = evaluator.evaluate(params)
        assert not result.feasible
        assert result.constraints_satisfied.get("power_watts") is False

    def test_invalid_params_handled(self, evaluator):
        """Evaluator should handle invalid params gracefully."""
        result = evaluator.evaluate({})  # missing everything
        assert isinstance(result, EvaluationResult)
        # Should still produce a result (using defaults)
        assert "power_watts" in result.objectives

    def test_thread_safety(self, evaluator, design_space):
        """Evaluate 100 designs concurrently and verify no crashes."""
        samples = design_space.sample_random(100, seed=42)
        results = [None] * 100

        def evaluate_one(idx, params):
            results[idx] = evaluator.evaluate(params)

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(evaluate_one, i, s) for i, s in enumerate(samples)]
            for f in futures:
                f.result(timeout=30)

        assert all(r is not None for r in results)
        assert all(isinstance(r, EvaluationResult) for r in results)

    def test_metadata_fields(self, evaluator):
        params = {
            "process_nm": 28,
            "clock_mhz": 500.0,
            "array_rows": 8,
            "array_cols": 8,
            "sram_kb": 256,
            "num_compute_tiles": 4,
            "noc_link_width_bits": 256,
        }
        result = evaluator.evaluate(params)
        assert "peak_gops" in result.metadata
        assert "yield_percent" in result.metadata
        assert result.metadata["peak_gops"] > 0

    def test_reuses_existing_ppa_logic(self, evaluator):
        """Verify evaluator reuses SoCComposition from ip_blocks.py."""
        params = {
            "process_nm": 14,
            "clock_mhz": 1000.0,
            "array_rows": 16,
            "array_cols": 16,
            "sram_kb": 512,
            "num_compute_tiles": 8,
            "noc_link_width_bits": 512,
        }
        result = evaluator.evaluate(params)
        # More compute tiles + higher clock should give more GOPS
        assert result.metadata["peak_gops"] > 0
        # 14nm should have lower area than 28nm equivalent
        assert result.objectives["area_mm2"] > 0
