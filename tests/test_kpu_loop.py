"""Tests for KPU validation loop."""

from embodied_ai_architect.graphs.kpu_loop import KPULoopConfig, KPULoopResult, run_kpu_loop


class TestKPULoop:
    def test_loop_converges_default(self):
        """Default workload and constraints should converge successfully."""
        workload = {"gflops": 4.0}
        constraints = {"max_area_mm2": 100.0}

        result = run_kpu_loop(workload, constraints)

        assert isinstance(result, KPULoopResult)
        assert result.success is True
        assert result.iterations_used >= 1

    def test_loop_result_has_config(self):
        """Result config dict should contain expected top-level keys."""
        workload = {"gflops": 4.0}
        constraints = {"max_area_mm2": 100.0}

        result = run_kpu_loop(workload, constraints)

        expected_keys = {
            "name",
            "process_nm",
            "compute_tile",
            "memory_tile",
            "dram",
            "noc",
            "array_rows",
            "array_cols",
        }
        assert expected_keys.issubset(result.config.keys())
        assert isinstance(result.config["process_nm"], int)
        assert isinstance(result.config["compute_tile"], dict)
        assert isinstance(result.config["memory_tile"], dict)

    def test_loop_records_history(self):
        """History list should be non-empty with required keys per entry."""
        workload = {"gflops": 4.0}
        constraints = {"max_area_mm2": 100.0}

        result = run_kpu_loop(workload, constraints)

        assert len(result.history) > 0
        for entry in result.history:
            assert "iteration" in entry
            assert "floorplan_feasible" in entry
            assert "bandwidth_balanced" in entry

    def test_loop_with_tight_area(self):
        """Tight area constraint should still converge or exhaust iterations."""
        workload = {"gflops": 4.0}
        constraints = {"max_area_mm2": 20.0}
        loop_config = KPULoopConfig(max_die_area_mm2=20.0)

        result = run_kpu_loop(workload, constraints, loop_config=loop_config)

        assert isinstance(result, KPULoopResult)
        # Either it converged under the tight budget or it exhausted iterations
        if result.success:
            assert result.floorplan["total_area_mm2"] <= 20.0
        else:
            assert result.iterations_used == loop_config.max_iterations

    def test_loop_iteration_limit(self):
        """Impossible constraints with max_iterations=1 should fail with iterations_used=1."""
        workload = {"gflops": 4.0}
        constraints = {"max_area_mm2": 1.0}
        loop_config = KPULoopConfig(max_iterations=1, max_die_area_mm2=1.0)

        result = run_kpu_loop(workload, constraints, loop_config=loop_config)

        assert result.success is False
        assert result.iterations_used == 1
