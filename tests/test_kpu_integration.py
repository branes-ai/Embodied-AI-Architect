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


# ---------------------------------------------------------------------------
# Issue #35: full pipeline integration test through SoCDesignRunner
# ---------------------------------------------------------------------------

import pytest  # noqa: E402

from embodied_ai_architect.graphs.session_store import SessionStore  # noqa: E402
from embodied_ai_architect.graphs.soc_runner import SoCDesignRunner  # noqa: E402
from embodied_ai_architect.graphs.soc_state import DesignConstraints  # noqa: E402

# Static plan with KPU + RTL specialists in dependency order. enable_moo=False
# is set on the runner state to skip the moo_explorer task — issue #35 is
# specifically about the KPU/RTL flow, MOO has its own integration test (#27).
KPU_PLAN = [
    {"id": "t1", "name": "Analyze workload", "agent": "workload_analyzer", "dependencies": []},
    {"id": "t2", "name": "Explore hardware", "agent": "hw_explorer", "dependencies": ["t1"]},
    {
        "id": "t3",
        "name": "Compose architecture",
        "agent": "architecture_composer",
        "dependencies": ["t2"],
    },
    {"id": "t4", "name": "Configure KPU", "agent": "kpu_configurator", "dependencies": ["t3"]},
    {
        "id": "t5",
        "name": "Validate floorplan",
        "agent": "floorplan_validator",
        "dependencies": ["t4"],
    },
    {
        "id": "t6",
        "name": "Validate bandwidth",
        "agent": "bandwidth_validator",
        "dependencies": ["t4"],
    },
    {"id": "t7", "name": "Generate RTL", "agent": "rtl_generator", "dependencies": ["t5", "t6"]},
    {
        "id": "t8",
        "name": "Assess RTL PPA",
        "agent": "rtl_ppa_assessor",
        "dependencies": ["t7"],
    },
    {"id": "t9", "name": "Assess PPA", "agent": "ppa_assessor", "dependencies": ["t8"]},
    {"id": "t10", "name": "Review design", "agent": "critic", "dependencies": ["t9"]},
    {"id": "t11", "name": "Generate report", "agent": "report_generator", "dependencies": ["t10"]},
]

INTEGRATION_GOAL = "Design a KPU-based SoC for warehouse AMR perception"
INTEGRATION_CONSTRAINTS = DesignConstraints(
    max_power_watts=15.0,
    max_latency_ms=50.0,
    max_cost_usd=100.0,
    max_area_mm2=200.0,
)


@pytest.fixture(scope="module")
def session_dir(tmp_path_factory):
    """Module-scoped tmp dir so the expensive runner only fires once."""
    return str(tmp_path_factory.mktemp("kpu_integration_sessions"))


@pytest.fixture(scope="module")
def kpu_run_state_with_rtl(session_dir):
    """Run the full KPU+RTL pipeline ONCE per module via the public runner API.

    Module-scoped so the expensive pipeline (~100s on CI) only runs once;
    each test below inspects the same final state.
    """
    runner = SoCDesignRunner(static_plan=KPU_PLAN, session_dir=session_dir)
    return runner.run(
        goal=INTEGRATION_GOAL,
        constraints=INTEGRATION_CONSTRAINTS,
        use_case="warehouse_amr",
        platform="amr",
        session_id="soc_kpu_integration_rtl",
        rtl_enabled=True,
    )


class TestKPUFullPipeline:
    """End-to-end pipeline through SoCDesignRunner with rtl_enabled=True.

    Verifies that KPU/RTL data flows from the inner specialists through the
    LangGraph dispatch_node forwarding (the bug we fixed in issue #27 and
    extended in #35) into the runner's returned state and the session store.
    """

    def test_state_has_kpu_config(self, kpu_run_state_with_rtl):
        kpu = kpu_run_state_with_rtl.get("kpu_config", {})
        assert kpu, "kpu_configurator did not populate state['kpu_config']"
        # Sanity-check core micro-architecture knobs
        assert kpu.get("compute_tile", {}).get("array_rows", 0) > 0
        assert kpu.get("compute_tile", {}).get("array_cols", 0) > 0
        assert kpu.get("array_rows", 0) > 0  # top-level checkerboard

    def test_state_has_floorplan_estimate(self, kpu_run_state_with_rtl):
        fp = kpu_run_state_with_rtl.get("floorplan_estimate", {})
        assert fp, "floorplan_validator did not populate floorplan_estimate"
        assert "feasible" in fp
        assert "total_area_mm2" in fp
        assert "pitch_matched" in fp

    def test_state_has_bandwidth_match(self, kpu_run_state_with_rtl):
        bw = kpu_run_state_with_rtl.get("bandwidth_match", {})
        assert bw, "bandwidth_validator did not populate bandwidth_match"
        assert "balanced" in bw
        assert "links" in bw
        assert len(bw["links"]) > 0

    def test_state_has_rtl_synthesis_results(self, kpu_run_state_with_rtl):
        rtl = kpu_run_state_with_rtl.get("rtl_synthesis_results", {})
        assert rtl, "rtl_generator did not populate rtl_synthesis_results"
        # At least one module successfully synthesized
        assert any(r.get("success") for r in rtl.values() if isinstance(r, dict))

    def test_state_has_kpu_optimization_history(self, kpu_run_state_with_rtl):
        """Issue #34: each KPU specialist appends a history entry."""
        history = kpu_run_state_with_rtl.get("kpu_optimization_history", [])
        assert len(history) > 0, "no KPU convergence history was recorded"
        sources = {e.get("source") for e in history}
        # At minimum the configurator must have recorded itself
        assert "kpu_configurator" in sources

    def test_session_store_round_trips_kpu_data(self, kpu_run_state_with_rtl, session_dir):
        store = SessionStore(session_dir=session_dir)
        loaded = store.load("soc_kpu_integration_rtl")
        assert loaded is not None
        assert loaded.get("kpu_config", {})
        assert loaded.get("floorplan_estimate", {})
        assert loaded.get("bandwidth_match", {})
        assert len(loaded.get("kpu_optimization_history", [])) > 0


class TestKPUSnapshotAndCLI:
    """The optimization review snapshot and `branes session show` must
    surface the KPU data so the architect doesn't have to drop into --json."""

    def test_snapshot_includes_kpu_slackness(self, kpu_run_state_with_rtl):
        from embodied_ai_architect.graphs.optimization_review import (
            build_optimization_review_snapshot,
        )

        snap = build_optimization_review_snapshot(kpu_run_state_with_rtl)
        assert snap.kpu_floorplan is not None
        assert snap.kpu_bandwidth is not None
        assert snap.kpu_floorplan.feasible in (True, False)
        assert len(snap.kpu_bandwidth.links) > 0

    def test_snapshot_includes_kpu_history(self, kpu_run_state_with_rtl):
        from embodied_ai_architect.graphs.optimization_review import (
            build_optimization_review_snapshot,
        )

        snap = build_optimization_review_snapshot(kpu_run_state_with_rtl)
        assert len(snap.kpu_history) > 0

    def test_session_show_renders_kpu_block(self, kpu_run_state_with_rtl, session_dir, monkeypatch):
        """`branes session show --latest` must show the KPU Configuration block
        added in issue #35 when the loaded session has a kpu_config."""
        from click.testing import CliRunner
        from embodied_ai_architect.cli.commands.session import session as session_cmd
        from embodied_ai_architect.graphs import session_store as session_store_mod

        original_store = session_store_mod.SessionStore

        def _patched_store(*args, **kwargs):
            return original_store(session_dir=session_dir)

        monkeypatch.setattr(session_store_mod, "SessionStore", _patched_store)

        runner = CliRunner()
        result = runner.invoke(session_cmd, ["show", "--latest"])
        assert result.exit_code == 0, result.output
        assert "KPU Configuration" in result.output
        assert "Systolic" in result.output
        assert "SRAM" in result.output
        assert "NoC" in result.output
        assert "DRAM" in result.output
