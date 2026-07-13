"""Tests for the agentic optimization loop (Phase 4).

Tests cover:
- Individual node functions: decompose, formulate, optimize, evaluate, reason
- Graph topology: correct edges and routing
- Convergence detection: hypervolume threshold, max iterations
- End-to-end run_optimization_loop
- CLI integration
"""

from __future__ import annotations

import pytest

from embodied_ai_architect.graphs.design_state import DesignState, undeclared_keys
from embodied_ai_architect.graphs.optimization_loop import (
    _build_moo_context_block,
    _rank_design_variables,
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
        state: DesignState = {
            "mission_description": "Drone perception at 5 m/s",
            "errors": [],
        }
        result = decompose_node(state)
        assert result.get("platform") == "drone"
        assert "mission_plan" in result
        assert len(result.get("sub_capabilities", [])) > 0

    def test_empty_mission_returns_error(self):
        state: DesignState = {"mission_description": "", "errors": []}
        result = decompose_node(state)
        assert len(result.get("errors", [])) > 0


class TestFormulateNode:
    def _make_state_with_plan(self) -> DesignState:
        state: DesignState = {
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
        state: DesignState = {"mission_plan": {}, "errors": []}
        result = formulate_node(state)
        assert len(result.get("errors", [])) > 0


class TestOptimizeNode:
    def test_runs_optimization(self):
        # Build state through decompose → formulate
        state: DesignState = {
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
        state: DesignState = {
            "pareto_front": [{"objectives": {"capability_per_watt": 0.3}}],
            "hypervolume_history": [1.0, 1.01],
            "iteration": 2,
            "max_iterations": 3,
            "hypervolume": 1.01,
        }
        result = evaluate_node(state)
        assert result["converged"] is True

    def test_converges_on_small_hypervolume_improvement(self):
        state: DesignState = {
            "pareto_front": [{"objectives": {"capability_per_watt": 0.3}}],
            "hypervolume_history": [1.0, 1.005],
            "iteration": 0,
            "max_iterations": 5,
            "hypervolume": 1.005,
        }
        result = evaluate_node(state)
        assert result["converged"] is True

    def test_does_not_converge_early(self):
        state: DesignState = {
            "pareto_front": [{"objectives": {"capability_per_watt": 0.3}}],
            "hypervolume_history": [1.0, 1.5],
            "iteration": 0,
            "max_iterations": 5,
            "hypervolume": 1.5,
        }
        result = evaluate_node(state)
        assert result["converged"] is False

    def test_empty_pareto_converges(self):
        state: DesignState = {
            "pareto_front": [],
            "hypervolume_history": [],
            "iteration": 0,
            "max_iterations": 3,
        }
        result = evaluate_node(state)
        assert result["converged"] is True


class TestReasonNode:
    def test_recommends_when_converged(self):
        state: DesignState = {
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
        state: DesignState = {
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
        state: DesignState = {"should_iterate": True, "converged": False}
        assert route_after_reason(state) == "iterate"

    def test_route_to_recommend(self):
        state: DesignState = {"should_iterate": False}
        assert route_after_reason(state) == "recommend"

    def test_route_converged_overrides_iterate(self):
        state: DesignState = {"should_iterate": True, "converged": True}
        assert route_after_reason(state) == "recommend"


class TestIterateNode:
    def test_increments_iteration(self):
        state: DesignState = {"iteration": 0}
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
        assert "Cap/watt" in report or "capability_per_watt" in report.lower()

    def test_report_contains_narrative_sections(self):
        """Report should include educational narrative sections."""
        result = run_optimization_loop(
            mission_description="Drone perception at 5 m/s",
            max_iterations=1,
            llm_available=False,
        )
        report = result.get("final_report", "")
        # Executive summary
        assert "Executive Summary" in report
        # Tradeoff analysis
        assert "Key Tradeoffs" in report
        # Design archetypes with labels
        assert "Design Alternatives" in report
        assert "★" in report
        # Per-design commentary
        assert "Design Analysis" in report
        assert "Pareto-optimal" in report
        # Recommendation section
        assert "Recommendation" in report
        assert "Next steps" in report
        # Glossary
        assert "Glossary" in report
        assert "Pareto front" in report


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------


class TestChannelCompliance:
    """S2a (#205) migration guard: every optimization-loop node now writes to the
    unified DesignState, so each node's return dict must contain only declared
    DesignState channels — a key that isn't a channel is silently dropped by
    LangGraph (Seam S1), which the other tests would not catch."""

    def test_all_nodes_write_only_declared_channels(self):
        state: DesignState = {
            "mission_description": "Drone perception at 5 m/s within 10W",
            "errors": [],
            "iteration": 0,
            "hypervolume_history": [],
            "convergence_history": [],
            "total_evaluations": 0,
            "llm_available": False,
        }
        for node in (decompose_node, formulate_node, optimize_node, evaluate_node):
            result = node(state)
            assert undeclared_keys(result) == set(), (
                f"{node.__name__} writes undeclared DesignState channels: "
                f"{sorted(undeclared_keys(result))}"
            )
            state.update(result)

        for node in (reason_node, iterate_node):
            result = node(state)
            assert undeclared_keys(result) == set(), (
                f"{node.__name__} writes undeclared DesignState channels: "
                f"{sorted(undeclared_keys(result))}"
            )

    @pytest.mark.parametrize("decision", ["recommend", "iterate"])
    def test_llm_reason_branch_writes_only_declared_channels(self, monkeypatch, decision):
        """The LLM reasoning path (_reason_with_llm) must also write only declared
        channels — the heuristic case above never exercises it."""
        payload = {
            "recommend": '{"decision": "recommend", "selected_design": null, "analysis": "ok"}',
            "iterate": (
                '{"decision": "iterate", "analysis": "narrow it",'
                ' "refinements": {"tighten_constraints": {"max_power_watts": 4.0}},'
                ' "research_citations": []}'
            ),
        }[decision]

        class _StubResponse:
            text = payload

        class _StubClient:
            def __init__(self, *a, **kw):
                pass

            def chat(self, messages, system):
                return _StubResponse()

        import embodied_ai_architect.llm.client as llm_client_mod

        monkeypatch.setattr(llm_client_mod, "LLMClient", _StubClient)

        state: DesignState = {
            "mission_description": "Drone perception",
            "platform": "drone",
            "iteration": 0,
            "converged": False,
            "hypervolume_history": [1.0, 1.5],
            "pareto_front": [
                {
                    "objectives": {"capability_per_watt": 0.3, "power_watts": 5.0},
                    "design_params": {"process_nm": 16, "clock_mhz": 1000},
                    "metadata": {"model_family": "yolov8", "model_variant": "s"},
                }
            ],
            "knee_point": None,
            "llm_available": True,
            "research_docs_used": [],
        }
        result = reason_node(state)
        assert undeclared_keys(result) == set(), (
            f"LLM reason branch ({decision}) writes undeclared DesignState channels: "
            f"{sorted(undeclared_keys(result))}"
        )


class TestCLI:
    def test_mission_command_exists(self):
        from embodied_ai_architect.cli.commands.optimize import optimize

        # Check the 'mission' subcommand is registered
        assert "mission" in [cmd.name for cmd in optimize.commands.values()]


# ---------------------------------------------------------------------------
# Issue #26: enhanced reasoning context
# ---------------------------------------------------------------------------


class TestRankDesignVariables:
    """The ranking helper should consume the BO producer format and produce
    a sorted list of variables by total impact across all objectives."""

    def test_returns_empty_for_no_sensitivity(self):
        assert _rank_design_variables(None) == []
        assert _rank_design_variables({}) == []

    def test_ranks_by_total_impact_producer_format(self):
        # Real producer format from bayesian_opt._extract_sensitivity:
        # {objective: {variable: {lengthscale, importance}}}
        sensitivity = {
            "power_watts": {
                "clock_mhz": {"lengthscale": 0.3, "importance": 0.90},
                "process_nm": {"lengthscale": 0.5, "importance": 0.40},
                "sram_kb": {"lengthscale": 2.0, "importance": 0.10},
            },
            "latency_ms": {
                "clock_mhz": {"lengthscale": 0.4, "importance": 0.80},
                "process_nm": {"lengthscale": 0.6, "importance": 0.30},
                "sram_kb": {"lengthscale": 1.5, "importance": 0.20},
            },
        }
        ranked = _rank_design_variables(sensitivity)
        # clock_mhz: 0.90 + 0.80 = 1.70 → first
        # process_nm: 0.40 + 0.30 = 0.70 → second
        # sram_kb: 0.10 + 0.20 = 0.30 → third
        assert [r["variable"] for r in ranked] == ["clock_mhz", "process_nm", "sram_kb"]
        assert ranked[0]["total_impact"] == 1.70
        assert ranked[0]["per_objective"]["power_watts"] == 0.90


class TestMOOContextBlock:
    """The reasoning prompt block must surface MOO evidence to Claude."""

    def test_includes_layers_atlas_convergence_and_sensitivity(self):
        state: DesignState = {
            "layers_used": ["map_elites", "bayesian"],
            "atlas": {"filled_cells": 72, "total_cells": 100, "coverage": 0.72},
            "atlas_coverage_pct": 72.0,
            "convergence_history": [
                {"iteration": 0, "hypervolume": 1.2, "pareto_size": 8, "total_evals": 200},
                {"iteration": 1, "hypervolume": 1.5, "pareto_size": 12, "total_evals": 450},
            ],
            "design_variables_ranked": [
                {
                    "variable": "clock_mhz",
                    "total_impact": 1.70,
                    "per_objective": {"power_watts": 0.90, "latency_ms": 0.80},
                },
            ],
        }
        block = _build_moo_context_block(state)
        assert "MOO Search Evidence" in block
        assert "map_elites, bayesian" in block
        assert "72.0%" in block
        assert "iter 1" in block
        assert "HV=1.500" in block
        assert "clock_mhz" in block
        assert "Top design variables by sensitivity" in block

    def test_handles_missing_optional_fields_gracefully(self):
        state: DesignState = {}
        block = _build_moo_context_block(state)
        assert "MOO Search Evidence" in block
        assert "(none)" in block
        assert "not available" in block


class TestOptimizeNodeEnrichedContext:
    """optimize_node must forward sensitivity/atlas/layers_used from the engine."""

    def test_forwards_rich_moo_fields(self):
        state: DesignState = {
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
        # These keys must always be present, even if MAP-Elites alone ran
        # (sensitivity may be empty, layers_used should at least contain map_elites)
        assert "sensitivity" in result
        assert "layers_used" in result
        assert "atlas" in result
        assert "atlas_coverage_pct" in result
        assert "design_variables_ranked" in result
        assert "map_elites" in result["layers_used"]


class TestReasonPromptIncludesMOOContext:
    """The LLM reasoning prompt must include the MOO Search Evidence block."""

    def test_prompt_contains_sensitivity_and_layers(self, monkeypatch):
        # Stub the LLM client so we can capture the prompt without a network call.
        captured = {}

        class _StubResponse:
            text = '{"decision": "recommend", "selected_design": null, "analysis": "ok"}'

        class _StubClient:
            def __init__(self, *a, **kw):
                pass

            def chat(self, messages, system):
                captured["prompt"] = messages[0]["content"]
                captured["system"] = system
                return _StubResponse()

        import embodied_ai_architect.llm.client as llm_client_mod

        monkeypatch.setattr(llm_client_mod, "LLMClient", _StubClient)

        state: DesignState = {
            "mission_description": "Drone perception",
            "platform": "drone",
            "iteration": 0,
            "hypervolume_history": [1.0, 1.5],
            "pareto_front": [
                {
                    "objectives": {"capability_per_watt": 0.3, "power_watts": 5.0},
                    "design_params": {"process_nm": 16, "clock_mhz": 1000},
                    "metadata": {"model_family": "yolov8", "model_variant": "s"},
                }
            ],
            "knee_point": None,
            "layers_used": ["map_elites", "bayesian"],
            "atlas": {"filled_cells": 72, "total_cells": 100, "coverage": 0.72},
            "atlas_coverage_pct": 72.0,
            "convergence_history": [
                {"iteration": 0, "hypervolume": 1.5, "pareto_size": 8, "total_evals": 200},
            ],
            "design_variables_ranked": [
                {
                    "variable": "clock_mhz",
                    "total_impact": 1.70,
                    "per_objective": {"power_watts": 0.90, "latency_ms": 0.80},
                },
            ],
            "sensitivity": {
                "power_watts": {"clock_mhz": {"importance": 0.90, "lengthscale": 0.3}},
            },
            "llm_available": True,
            "converged": False,
            "research_docs_used": [],
        }

        from embodied_ai_architect.graphs.optimization_loop import _reason_with_llm

        _reason_with_llm(state)

        prompt = captured["prompt"]
        assert "MOO Search Evidence" in prompt
        assert "map_elites, bayesian" in prompt
        assert "72.0%" in prompt
        assert "clock_mhz" in prompt
        # The system prompt must have been updated to guide on this evidence
        assert "design_variables_ranked" in captured["system"]
        assert "atlas_coverage_pct" in captured["system"]
