"""Tests for the agentic optimization loop (Phase 4).

Tests cover:
- Individual node functions: decompose, formulate, optimize, evaluate, reason
- Graph topology: correct edges and routing
- Convergence detection: hypervolume threshold, max iterations
- End-to-end run_optimization_loop
- CLI integration
"""

from __future__ import annotations

from embodied_ai_architect.graphs.optimization_loop import (
    OptimizationLoopState,
    build_optimization_loop,
    decompose_node,
    evaluate_node,
    formulate_node,
    iterate_node,
    optimize_node,
    reason_node,
    route_after_reason,
    run_optimization_loop,
)

# ---------------------------------------------------------------------------
# Node unit tests
# ---------------------------------------------------------------------------


class TestDecomposeNode:
    def test_decomposes_mission(self):
        state: OptimizationLoopState = {
            "mission_description": "Drone perception at 5 m/s",
            "errors": [],
        }
        result = decompose_node(state)
        assert result.get("platform") == "drone"
        assert "mission_plan" in result
        assert len(result.get("sub_capabilities", [])) > 0

    def test_empty_mission_returns_error(self):
        state: OptimizationLoopState = {"mission_description": "", "errors": []}
        result = decompose_node(state)
        assert len(result.get("errors", [])) > 0


class TestFormulateNode:
    def _make_state_with_plan(self) -> OptimizationLoopState:
        state: OptimizationLoopState = {
            "mission_description": "Drone perception at 5 m/s",
            "errors": [],
        }
        decomposed = decompose_node(state)
        state.update(decomposed)
        return state

    def test_formulates_design_space(self):
        state = self._make_state_with_plan()
        result = formulate_node(state)
        assert result.get("num_variables", 0) >= 17
        assert "constraints" in result

    def test_no_plan_returns_error(self):
        state: OptimizationLoopState = {"mission_plan": {}, "errors": []}
        result = formulate_node(state)
        assert len(result.get("errors", [])) > 0


class TestOptimizeNode:
    def test_runs_optimization(self):
        # Build state through decompose → formulate
        state: OptimizationLoopState = {
            "mission_description": "Drone perception at 5 m/s",
            "errors": [],
            "iteration": 0,
            "hypervolume_history": [],
            "convergence_history": [],
            "total_evaluations": 0,
        }
        state.update(decompose_node(state))
        state.update(formulate_node(state))
        result = optimize_node(state)
        assert len(result.get("pareto_front", [])) > 0
        assert result.get("total_evaluations", 0) > 0
        assert result.get("hypervolume", 0) > 0


class TestEvaluateNode:
    def test_converges_at_max_iterations(self):
        state: OptimizationLoopState = {
            "pareto_front": [{"objectives": {"capability_per_watt": 0.3}}],
            "hypervolume_history": [1.0, 1.01],
            "iteration": 2,
            "max_iterations": 3,
            "hypervolume": 1.01,
        }
        result = evaluate_node(state)
        assert result["converged"] is True

    def test_converges_on_small_hypervolume_improvement(self):
        state: OptimizationLoopState = {
            "pareto_front": [{"objectives": {"capability_per_watt": 0.3}}],
            "hypervolume_history": [1.0, 1.005],
            "iteration": 0,
            "max_iterations": 5,
            "hypervolume": 1.005,
        }
        result = evaluate_node(state)
        assert result["converged"] is True

    def test_does_not_converge_early(self):
        state: OptimizationLoopState = {
            "pareto_front": [{"objectives": {"capability_per_watt": 0.3}}],
            "hypervolume_history": [1.0, 1.5],
            "iteration": 0,
            "max_iterations": 5,
            "hypervolume": 1.5,
        }
        result = evaluate_node(state)
        assert result["converged"] is False

    def test_empty_pareto_converges(self):
        state: OptimizationLoopState = {
            "pareto_front": [],
            "hypervolume_history": [],
            "iteration": 0,
            "max_iterations": 3,
        }
        result = evaluate_node(state)
        assert result["converged"] is True


class TestReasonNode:
    def test_recommends_when_converged(self):
        state: OptimizationLoopState = {
            "mission_description": "Drone perception",
            "pareto_front": [
                {
                    "objectives": {"capability_per_watt": 0.3, "power_watts": 5.0},
                    "design_params": {
                        "process_nm": 16,
                        "clock_mhz": 1000.0,
                        "array_rows": 8,
                        "array_cols": 8,
                    },
                    "metadata": {"model_family": "yolov8", "model_variant": "s"},
                }
            ],
            "converged": True,
            "iteration": 1,
            "knee_point": None,
            "llm_available": False,
            "total_evaluations": 500,
            "sub_capabilities": [],
            "research_docs_used": [],
            "analysis": "Test analysis",
        }
        result = reason_node(state)
        assert result["should_iterate"] is False
        assert "final_report" in result
        assert len(result["final_report"]) > 0

    def test_iterates_when_low_capability(self):
        state: OptimizationLoopState = {
            "mission_description": "Drone perception",
            "pareto_front": [
                {
                    "objectives": {"capability_per_watt": 0.01},
                    "metadata": {"capability_per_watt": 0.01},
                }
            ],
            "converged": False,
            "iteration": 0,
            "llm_available": False,
            "research_docs_used": [],
        }
        result = reason_node(state)
        assert result["should_iterate"] is True


class TestRouting:
    def test_route_to_iterate(self):
        state: OptimizationLoopState = {"should_iterate": True, "converged": False}
        assert route_after_reason(state) == "iterate"

    def test_route_to_recommend(self):
        state: OptimizationLoopState = {"should_iterate": False}
        assert route_after_reason(state) == "recommend"

    def test_route_converged_overrides_iterate(self):
        state: OptimizationLoopState = {"should_iterate": True, "converged": True}
        assert route_after_reason(state) == "recommend"


class TestIterateNode:
    def test_increments_iteration(self):
        state: OptimizationLoopState = {"iteration": 0}
        result = iterate_node(state)
        assert result["iteration"] == 1


# ---------------------------------------------------------------------------
# Graph topology test
# ---------------------------------------------------------------------------


class TestGraphTopology:
    def test_graph_builds(self):
        graph = build_optimization_loop()
        assert graph is not None


# ---------------------------------------------------------------------------
# End-to-end test
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_run_optimization_loop(self):
        """Full end-to-end test with 1 iteration."""
        result = run_optimization_loop(
            mission_description="Drone perception at 5 m/s within 10W",
            max_iterations=1,
            llm_available=False,
        )
        assert "final_report" in result
        assert len(result.get("final_report", "")) > 0
        assert result.get("total_evaluations", 0) > 0
        assert result.get("platform") == "drone"
        assert "recommendation" in result

    def test_report_contains_design_details(self):
        """Report should include hardware, pipeline, compiler details."""
        result = run_optimization_loop(
            mission_description="Drone perception at 5 m/s",
            max_iterations=1,
            llm_available=False,
        )
        report = result.get("final_report", "")
        assert "Hardware" in report
        assert "Pipeline" in report
        assert "Compiler" in report
        assert "Capability/watt" in report or "capability_per_watt" in report.lower()

    def test_report_contains_comparison_table(self):
        """Report should include a multi-design comparison table."""
        result = run_optimization_loop(
            mission_description="Drone perception at 5 m/s",
            max_iterations=1,
            llm_available=False,
        )
        report = result.get("final_report", "")
        assert "Design Comparison" in report
        assert "★" in report
        # Table should have structured rows for key attributes
        assert "Process" in report
        assert "Detector" in report
        assert "Runtime" in report
        assert "Cap/watt" in report


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------


class TestCLI:
    def test_mission_command_exists(self):
        from embodied_ai_architect.cli.commands.optimize import optimize

        # Check the 'mission' subcommand is registered
        assert "mission" in [cmd.name for cmd in optimize.commands.values()]
