"""Tests for KPU validation loop."""

from embodied_ai_architect.graphs.kpu_loop import (
    KPULoopConfig,
    KPULoopResult,
    apply_rtl_area_feedback,
    run_kpu_loop,
)
from embodied_ai_architect.graphs.kpu_config import KPU_PRESETS
from embodied_ai_architect.graphs.kpu_specialists import rtl_area_feedback
from embodied_ai_architect.graphs.task_graph import TaskNode


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


# ---------------------------------------------------------------------------
# Issue #31: RTL → KPU area feedback
# ---------------------------------------------------------------------------


def _make_state_with_rtl(synthesis_area_mm2: float, floorplan_area_mm2: float):
    """Build a minimal state with kpu_config + floorplan + synthesis results."""
    config = KPU_PRESETS["edge_balanced"]
    state = {
        "kpu_config": config.model_dump(),
        "floorplan_estimate": {
            "compute_tile": {"width_mm": 2.1, "height_mm": 2.3, "area_mm2": 4.83},
            "memory_tile": {"width_mm": 2.0, "height_mm": 2.4, "area_mm2": 4.80},
            "pitch_matched": True,
            "pitch_ratio_width": 1.05,
            "pitch_ratio_height": 0.96,
            "pitch_tolerance": 0.15,
            "array_width_mm": 6.3,
            "array_height_mm": 6.9,
            "core_area_mm2": 43.5,
            "periphery_area_mm2": 4.7,
            "total_area_mm2": floorplan_area_mm2,
            "die_edge_mm": 7.0,
            "feasible": True,
            "max_die_area_mm2": 100.0,
            "issues": [],
        },
        "rtl_synthesis_results": {
            "compute_tile": {
                "success": True,
                "area_um2": synthesis_area_mm2 * 1e6,
                "area_cells": int(synthesis_area_mm2 * 1e6 / 10),
            },
        },
        "workload_profile": {"total_estimated_gflops": 5.0},
        "rtl_enabled": True,
        "rtl_area_feedback": True,
    }
    return state


class TestApplyRTLAreaFeedback:
    """Pure-function tests for apply_rtl_area_feedback."""

    def test_no_op_when_no_synthesis_results(self):
        state = _make_state_with_rtl(synthesis_area_mm2=0.001, floorplan_area_mm2=50.0)
        state["rtl_synthesis_results"] = {}
        updates = apply_rtl_area_feedback(state)
        assert updates == {}

    def test_no_op_when_no_floorplan(self):
        state = _make_state_with_rtl(synthesis_area_mm2=10.0, floorplan_area_mm2=50.0)
        state["floorplan_estimate"] = {}
        updates = apply_rtl_area_feedback(state)
        assert updates == {}

    def test_within_tolerance_no_resize(self):
        """Synthesis 51mm² vs floorplan 50mm² × 1.1 = 55mm² → no re-size."""
        state = _make_state_with_rtl(synthesis_area_mm2=51.0, floorplan_area_mm2=50.0)
        updates = apply_rtl_area_feedback(state)
        # Within tolerance — only the summary is set, no kpu_config change
        assert "kpu_config" not in updates
        assert "rtl_area_feedback_summary" in updates
        assert "no re-sizing needed" in updates["rtl_area_feedback_summary"]

    def test_above_tolerance_triggers_resize(self):
        """Synthesis 80mm² vs floorplan 50mm² × 1.1 = 55mm² → re-size."""
        state = _make_state_with_rtl(synthesis_area_mm2=80.0, floorplan_area_mm2=50.0)
        updates = apply_rtl_area_feedback(state)
        assert "kpu_config" in updates
        assert "floorplan_estimate" in updates
        assert "bandwidth_match" in updates
        assert "kpu_optimization_history" in updates
        # The summary mentions the trigger condition and how many iterations ran
        assert "RTL area feedback" in updates["rtl_area_feedback_summary"]

    def test_history_records_each_iteration(self):
        state = _make_state_with_rtl(synthesis_area_mm2=200.0, floorplan_area_mm2=50.0)
        updates = apply_rtl_area_feedback(state, max_iterations=3)
        history = updates["kpu_optimization_history"]
        # Up to 3 entries — may converge sooner
        assert 1 <= len(history) <= 3
        for entry in history:
            assert entry["source"] == "rtl_area_feedback"
            assert "synthesis_area_mm2" in entry
            assert "floorplan_total_area_mm2" in entry
            assert "feasible" in entry

    def test_max_iterations_bound(self):
        """An impossibly tight synthesis-derived budget hits the iteration cap.

        Use synthesis_area=1.0mm² > floorplan=0.5mm² × 1.1 = 0.55mm² to
        TRIGGER the feedback, then the new budget (1.0mm²) is impossibly
        tight for the edge_balanced preset — verify iteration count is bounded.
        """
        state = _make_state_with_rtl(synthesis_area_mm2=1.0, floorplan_area_mm2=0.5)
        updates = apply_rtl_area_feedback(state, max_iterations=3)
        # Re-size was triggered (1.0 > 0.55) → history populated
        assert "kpu_optimization_history" in updates
        history = updates["kpu_optimization_history"]
        # Bounded by max_iterations
        assert 1 <= len(history) <= 3

    def test_history_appends_to_existing(self):
        """Existing kpu_optimization_history entries must be preserved."""
        state = _make_state_with_rtl(synthesis_area_mm2=80.0, floorplan_area_mm2=50.0)
        state["kpu_optimization_history"] = [
            {"source": "kpu_loop", "iteration": 0, "config_name": "swkpu-prior"}
        ]
        updates = apply_rtl_area_feedback(state)
        history = updates["kpu_optimization_history"]
        assert history[0]["source"] == "kpu_loop"  # prior preserved
        assert any(e["source"] == "rtl_area_feedback" for e in history)


class TestRTLAreaFeedbackSpecialist:
    """Dispatcher-facing wrapper tests."""

    def _task(self, **metadata):
        return TaskNode(
            id="t_feedback",
            name="RTL area feedback",
            agent="rtl_area_feedback",
            metadata=metadata,
        )

    def test_skipped_when_flag_disabled(self):
        state = _make_state_with_rtl(synthesis_area_mm2=80.0, floorplan_area_mm2=50.0)
        state["rtl_area_feedback"] = False
        result = rtl_area_feedback(self._task(), state)
        assert result["verdict"] == "SKIP"
        assert "_state_updates" not in result

    def test_skipped_when_rtl_disabled(self):
        state = _make_state_with_rtl(synthesis_area_mm2=80.0, floorplan_area_mm2=50.0)
        state["rtl_enabled"] = False
        result = rtl_area_feedback(self._task(), state)
        assert result["verdict"] == "SKIP"

    def test_triggers_resize_when_above_threshold(self):
        state = _make_state_with_rtl(synthesis_area_mm2=80.0, floorplan_area_mm2=50.0)
        result = rtl_area_feedback(self._task(), state)
        assert result["verdict"] == "PASS"
        assert "_state_updates" in result
        assert "kpu_config" in result["_state_updates"]
        assert "kpu_optimization_history" in result["_state_updates"]

    def test_metadata_overrides_thresholds(self):
        """task.metadata.area_tolerance and max_iterations are honored."""
        state = _make_state_with_rtl(synthesis_area_mm2=51.0, floorplan_area_mm2=50.0)
        # Default tolerance 1.1 → 55mm² threshold → no trigger
        # Tighter tolerance 1.0 → trigger
        result = rtl_area_feedback(self._task(area_tolerance=1.0), state)
        assert result["verdict"] == "PASS"
        assert "kpu_config" in result["_state_updates"]

    def test_no_synthesis_returns_skip(self):
        state = _make_state_with_rtl(synthesis_area_mm2=80.0, floorplan_area_mm2=50.0)
        state["rtl_synthesis_results"] = {}
        result = rtl_area_feedback(self._task(), state)
        assert result["verdict"] == "SKIP"
        assert "insufficient state" in result["summary"]
