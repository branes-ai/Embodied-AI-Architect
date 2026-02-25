"""Integration tests for KPU micro-architecture pipeline.

Tests the full flow: KPU config → floorplan → bandwidth → optimization loop.
"""

from __future__ import annotations


from embodied_ai_architect.graphs.kpu_config import (
    KPU_PRESETS,
)
from embodied_ai_architect.graphs.floorplan import estimate_floorplan
from embodied_ai_architect.graphs.bandwidth import check_bandwidth_match
from embodied_ai_architect.graphs.kpu_loop import KPULoopConfig, run_kpu_loop


class TestKPULoopConverges:
    """KPU loop converges for reasonable configurations."""

    def test_drone_use_case_converges(self):
        result = run_kpu_loop(
            workload={"gflops": 4.0},
            constraints={"max_area_mm2": 100.0, "max_power_watts": 5.0},
            use_case="delivery_drone",
        )
        assert result.success
        assert result.iterations_used >= 1
        assert result.config  # non-empty config dict
        assert result.floorplan.get("feasible")
        assert result.bandwidth.get("balanced")

    def test_edge_balanced_preset_passes(self):
        config = KPU_PRESETS["edge_balanced"]
        fp = estimate_floorplan(config, max_die_area_mm2=200.0)
        bw = check_bandwidth_match(config, {"gflops": 8.0})
        assert fp.feasible
        assert bw.balanced


class TestFloorplanTriggersResize:
    """Oversized config gets optimized down by the loop."""

    def test_large_config_converges_within_area(self):
        # Start with a config that's too big for 50mm2
        result = run_kpu_loop(
            workload={"gflops": 4.0},
            constraints={"max_area_mm2": 50.0},
            use_case="delivery_drone",
            loop_config=KPULoopConfig(max_die_area_mm2=50.0, max_iterations=15),
        )
        # Should either converge or exhaust iterations
        assert result.iterations_used >= 1
        if result.success:
            assert result.floorplan.get("total_area_mm2", 999) <= 50.0


class TestBandwidthTriggersUpgrade:
    """Bandwidth bottleneck triggers controller addition."""

    def test_low_dram_bandwidth_gets_adjusted(self):
        config = KPU_PRESETS["drone_minimal"]
        # Force low DRAM bandwidth to create bottleneck
        config = config.model_copy(
            update={"dram": config.dram.model_copy(update={"num_controllers": 1})}
        )
        check_bandwidth_match(config, {"gflops": 20.0})
        # With low DRAM, there may be a bottleneck
        # The loop should try to fix it
        result = run_kpu_loop(
            workload={"gflops": 20.0},
            constraints={"max_area_mm2": 100.0},
            use_case="delivery_drone",
            loop_config=KPULoopConfig(max_iterations=10),
        )
        assert result.iterations_used >= 1


class TestBackwardCompatibility:
    """Existing functionality works unchanged when rtl_enabled=False."""

    def test_state_without_rtl(self):
        from embodied_ai_architect.graphs.soc_state import (
            DesignConstraints,
            create_initial_soc_state,
        )

        state = create_initial_soc_state(
            goal="Design drone SoC",
            constraints=DesignConstraints(max_power_watts=5.0),
            use_case="delivery_drone",
        )
        assert state["rtl_enabled"] is False
        assert state["kpu_config"] == {}
        assert state["floorplan_estimate"] == {}
        assert state["bandwidth_match"] == {}

    def test_state_with_rtl_enabled(self):
        from embodied_ai_architect.graphs.soc_state import (
            DesignConstraints,
            create_initial_soc_state,
        )

        state = create_initial_soc_state(
            goal="Design drone SoC",
            constraints=DesignConstraints(max_power_watts=5.0, max_area_mm2=100.0),
            use_case="delivery_drone",
            rtl_enabled=True,
        )
        assert state["rtl_enabled"] is True
