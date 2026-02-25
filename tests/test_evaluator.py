"""Tests for the AgenticEvaluator class."""

from __future__ import annotations

import pytest

from embodied_ai_architect.graphs.evaluation import (
    DimensionScore,
    GoldStandard,
    RunTrace,
    Scorecard,
)
from embodied_ai_architect.graphs.evaluator import (
    DEFAULT_WEIGHTS,
    AgenticEvaluator,
)
from embodied_ai_architect.graphs.gold_standards import ALL_GOLD_STANDARDS


def _make_trace(demo_name: str = "test", **kwargs) -> RunTrace:
    defaults = dict(
        demo_name=demo_name,
        task_graph={
            "nodes": {
                "t1": {"agent": "workload_analyzer", "dependencies": [], "status": "completed"},
                "t2": {"agent": "hw_explorer", "dependencies": ["t1"], "status": "completed"},
                "t3": {
                    "agent": "architecture_composer",
                    "dependencies": ["t2"],
                    "status": "completed",
                },
                "t4": {"agent": "ppa_assessor", "dependencies": ["t3"], "status": "completed"},
                "t5": {"agent": "critic", "dependencies": ["t4"], "status": "completed"},
                "t6": {"agent": "report_generator", "dependencies": ["t5"], "status": "completed"},
            },
        },
        ppa_metrics={
            "power_watts": 4.5,
            "latency_ms": 30.0,
            "cost_usd": 25.0,
            "verdicts": {"power": "PASS", "latency": "PASS", "cost": "PASS"},
        },
        tool_calls=[
            "workload_analyzer",
            "hw_explorer",
            "architecture_composer",
            "ppa_assessor",
            "critic",
            "report_generator",
        ],
        design_rationale=[
            "[workload_analyzer] Analyzed drone perception workload",
            "[hw_explorer] Explored 6 hardware candidates for power-constrained drone",
            "[architecture_composer] Composed SoC architecture around Stillwater KPU",
            "[ppa_assessor] Assessed PPA: all constraints PASS",
            "[critic] Design review: ADEQUATE",
            "[report_generator] Generated design report",
        ],
        duration_seconds=5.0,
    )
    defaults.update(kwargs)
    return RunTrace(**defaults)


def _make_gold(demo_name: str = "test", **kwargs) -> GoldStandard:
    defaults = dict(
        demo_name=demo_name,
        expected_task_graph={
            "nodes": {
                "t1": {"agent": "workload_analyzer", "dependencies": []},
                "t2": {"agent": "hw_explorer", "dependencies": ["t1"]},
                "t3": {"agent": "architecture_composer", "dependencies": ["t2"]},
                "t4": {"agent": "ppa_assessor", "dependencies": ["t3"]},
                "t5": {"agent": "critic", "dependencies": ["t4"]},
                "t6": {"agent": "report_generator", "dependencies": ["t5"]},
            },
        },
        expected_ppa={"power_watts": 5.0, "latency_ms": 33.3, "cost_usd": 30.0},
        expected_tool_calls=[
            "workload_analyzer",
            "hw_explorer",
            "architecture_composer",
            "ppa_assessor",
            "critic",
            "report_generator",
        ],
        rationale_keywords=["drone", "perception", "power", "workload"],
        max_duration_seconds=30.0,
    )
    defaults.update(kwargs)
    return GoldStandard(**defaults)


class TestAgenticEvaluator:
    def test_default_weights(self):
        assert abs(sum(DEFAULT_WEIGHTS.values()) - 1.0) < 0.01

    def test_evaluate_run_basic(self):
        gold = _make_gold()
        evaluator = AgenticEvaluator(gold_standards={"test": gold})
        trace = _make_trace()
        scorecard = evaluator.evaluate_run(trace)
        assert isinstance(scorecard, Scorecard)
        assert scorecard.demo_name == "test"
        assert len(scorecard.dimensions) == 9
        assert 0.0 <= scorecard.composite_score <= 1.0

    def test_evaluate_run_all_pass(self):
        gold = _make_gold()
        evaluator = AgenticEvaluator(gold_standards={"test": gold})
        trace = _make_trace()
        scorecard = evaluator.evaluate_run(trace)
        # Good trace should score reasonably well
        assert scorecard.composite_score > 0.5

    def test_evaluate_run_missing_gold(self):
        evaluator = AgenticEvaluator(gold_standards={})
        trace = _make_trace()
        with pytest.raises(KeyError, match="No gold standard"):
            evaluator.evaluate_run(trace)

    def test_evaluate_all(self):
        gold = _make_gold()
        evaluator = AgenticEvaluator(gold_standards={"test": gold})
        traces = {"test": _make_trace()}
        results = evaluator.evaluate_all(traces)
        assert "test" in results
        assert isinstance(results["test"], Scorecard)

    def test_evaluate_all_skips_missing_golds(self):
        gold = _make_gold()
        evaluator = AgenticEvaluator(gold_standards={"test": gold})
        traces = {"test": _make_trace(), "unknown": _make_trace(demo_name="unknown")}
        results = evaluator.evaluate_all(traces)
        assert "test" in results
        assert "unknown" not in results

    def test_custom_weights(self):
        gold = _make_gold()
        # Weight everything to ppa_accuracy
        weights = {k: 0.0 for k in DEFAULT_WEIGHTS}
        weights["ppa_accuracy"] = 1.0
        evaluator = AgenticEvaluator(gold_standards={"test": gold}, weights=weights)
        trace = _make_trace()
        scorecard = evaluator.evaluate_run(trace)
        # Composite should be close to ppa_accuracy score
        ppa_dim = next(d for d in scorecard.dimensions if d.dimension == "ppa_accuracy")
        assert abs(scorecard.composite_score - ppa_dim.score) < 0.01

    def test_passing_threshold(self):
        gold = _make_gold()
        evaluator = AgenticEvaluator(gold_standards={"test": gold}, passing_threshold=0.99)
        trace = _make_trace()
        scorecard = evaluator.evaluate_run(trace)
        # With very high threshold, likely fails
        assert scorecard.passed == (scorecard.composite_score >= 0.99)

    def test_capture_run_trace(self):
        evaluator = AgenticEvaluator()
        initial = {"status": "planning", "history": []}
        final = {
            "status": "complete",
            "task_graph": {
                "nodes": {
                    "t1": {"agent": "workload_analyzer", "status": "completed", "dependencies": []}
                }
            },
            "ppa_metrics": {"power_watts": 5.0},
            "history": [{"agent": "workload_analyzer", "action": "Completed task 'analyze'"}],
            "design_rationale": ["[workload_analyzer] done"],
            "audit_log": [{"agent": "test", "action": "test", "cost_tokens": 100}],
            "optimization_history": [],
            "pareto_results": {},
        }
        trace = evaluator.capture_run_trace(initial, final, "test_demo", duration_seconds=1.5)
        assert isinstance(trace, RunTrace)
        assert trace.demo_name == "test_demo"
        assert trace.duration_seconds == 1.5
        assert trace.cost_tokens == 100
        assert "workload_analyzer" in trace.tool_calls

    def test_dimensions_have_weights(self):
        gold = _make_gold()
        evaluator = AgenticEvaluator(gold_standards={"test": gold})
        trace = _make_trace()
        scorecard = evaluator.evaluate_run(trace)
        for dim in scorecard.dimensions:
            assert dim.weight >= 0.0

    def test_scorecard_summary(self):
        gold = _make_gold()
        evaluator = AgenticEvaluator(gold_standards={"test": gold})
        trace = _make_trace()
        scorecard = evaluator.evaluate_run(trace)
        summary = scorecard.summary()
        assert "test" in summary
        assert "PASS" in summary or "FAIL" in summary


class TestGoldStandards:
    """Verify the hand-crafted gold standards are well-formed."""

    def test_all_gold_standards_exist(self):
        assert len(ALL_GOLD_STANDARDS) == 7

    @pytest.mark.parametrize("name", list(ALL_GOLD_STANDARDS.keys()))
    def test_gold_standard_has_expected_fields(self, name):
        gold = ALL_GOLD_STANDARDS[name]
        assert gold.demo_name == name
        assert gold.expected_task_graph
        assert gold.expected_tool_calls
        assert gold.max_iterations > 0

    @pytest.mark.parametrize("name", list(ALL_GOLD_STANDARDS.keys()))
    def test_gold_standard_task_graph_has_nodes(self, name):
        gold = ALL_GOLD_STANDARDS[name]
        nodes = gold.expected_task_graph.get("nodes", {})
        assert len(nodes) > 0
