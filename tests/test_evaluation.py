"""Tests for evaluation data models.

Tests the core evaluation types: RunTrace, GoldStandard, DimensionScore, and Scorecard.
"""

import pytest
from pydantic import ValidationError

from embodied_ai_architect.graphs.evaluation import (
    DimensionScore,
    GoldStandard,
    RunTrace,
    Scorecard,
)

# ---------------------------------------------------------------------------
# RunTrace tests
# ---------------------------------------------------------------------------


def test_runtrace_creation_with_defaults():
    """Test RunTrace can be created with minimal fields (all defaults)."""
    trace = RunTrace()

    assert trace.demo_name == ""
    assert trace.task_graph == {}
    assert trace.ppa_metrics == {}
    assert trace.iteration_history == []
    assert trace.tool_calls == []
    assert trace.audit_log == []
    assert trace.failures == 0
    assert trace.recoveries == 0
    assert trace.duration_seconds == 0.0
    assert trace.cost_tokens == 0
    assert trace.human_interventions == 0
    assert trace.design_rationale == []
    assert trace.pareto_points == []
    assert trace.history == []


def test_runtrace_creation_with_all_fields():
    """Test RunTrace can be created with all fields populated."""
    trace = RunTrace(
        demo_name="test_demo",
        task_graph={"nodes": {"n1": {"agent": "ModelAnalyzer"}}},
        ppa_metrics={"power_watts": 5.0, "latency_ms": 10.0},
        iteration_history=[{"iter": 1, "ppa": {"power_watts": 6.0}}],
        tool_calls=["ModelAnalyzer", "HardwareProfile"],
        audit_log=[{"action": "power_budget_exceeded"}],
        failures=2,
        recoveries=1,
        duration_seconds=45.5,
        cost_tokens=12345,
        human_interventions=1,
        design_rationale=["[Planner] Starting decomposition", "[Critic] PPA check"],
        pareto_points=[{"power": 5.0, "latency": 10.0, "dominated": False}],
        history=[{"decision": "use_kpu"}],
    )

    assert trace.demo_name == "test_demo"
    assert "n1" in trace.task_graph["nodes"]
    assert trace.ppa_metrics["power_watts"] == 5.0
    assert len(trace.iteration_history) == 1
    assert "ModelAnalyzer" in trace.tool_calls
    assert len(trace.audit_log) == 1
    assert trace.failures == 2
    assert trace.recoveries == 1
    assert trace.duration_seconds == 45.5
    assert trace.cost_tokens == 12345
    assert trace.human_interventions == 1
    assert len(trace.design_rationale) == 2
    assert len(trace.pareto_points) == 1
    assert len(trace.history) == 1


def test_runtrace_model_dump_roundtrip():
    """Test RunTrace can be serialized and deserialized via model_dump."""
    original = RunTrace(
        demo_name="roundtrip_test",
        task_graph={"nodes": {"a": {"agent": "Planner"}}},
        ppa_metrics={"power_watts": 3.5},
        tool_calls=["Planner", "Critic"],
        failures=1,
        recoveries=1,
        duration_seconds=30.0,
        cost_tokens=5000,
    )

    # Serialize
    data = original.model_dump()

    # Deserialize
    restored = RunTrace(**data)

    assert restored.demo_name == original.demo_name
    assert restored.task_graph == original.task_graph
    assert restored.ppa_metrics == original.ppa_metrics
    assert restored.tool_calls == original.tool_calls
    assert restored.failures == original.failures
    assert restored.recoveries == original.recoveries
    assert restored.duration_seconds == original.duration_seconds
    assert restored.cost_tokens == original.cost_tokens


# ---------------------------------------------------------------------------
# GoldStandard tests
# ---------------------------------------------------------------------------


def test_goldstandard_creation_with_defaults():
    """Test GoldStandard can be created with minimal fields (all defaults)."""
    gold = GoldStandard()

    assert gold.demo_name == ""
    assert gold.expected_task_graph == {}
    assert gold.expected_ppa == {}
    assert gold.governance_triggers == []
    assert gold.expected_tool_calls == []
    assert gold.max_iterations == 10
    assert gold.max_duration_seconds == 60.0
    assert gold.max_cost_tokens == 100000
    assert gold.max_human_interventions == 3
    assert gold.expected_pareto_points == 0
    assert gold.rationale_keywords == []


def test_goldstandard_with_all_fields_set():
    """Test GoldStandard can be created with all fields populated."""
    gold = GoldStandard(
        demo_name="demo_1",
        expected_task_graph={"nodes": {"n1": {"agent": "ModelAnalyzer"}}},
        expected_ppa={"power_watts": 5.0, "latency_ms": 15.0, "area_mm2": 10.0, "cost_usd": 50.0},
        governance_triggers=["power_budget_exceeded", "latency_violation"],
        expected_tool_calls=["ModelAnalyzer", "HardwareProfile", "Benchmark"],
        max_iterations=15,
        max_duration_seconds=120.0,
        max_cost_tokens=200000,
        max_human_interventions=2,
        expected_pareto_points=5,
        rationale_keywords=["kpu", "npu", "power", "latency"],
    )

    assert gold.demo_name == "demo_1"
    assert "n1" in gold.expected_task_graph["nodes"]
    assert gold.expected_ppa["power_watts"] == 5.0
    assert "power_budget_exceeded" in gold.governance_triggers
    assert "ModelAnalyzer" in gold.expected_tool_calls
    assert gold.max_iterations == 15
    assert gold.max_duration_seconds == 120.0
    assert gold.max_cost_tokens == 200000
    assert gold.max_human_interventions == 2
    assert gold.expected_pareto_points == 5
    assert "kpu" in gold.rationale_keywords


# ---------------------------------------------------------------------------
# DimensionScore tests
# ---------------------------------------------------------------------------


def test_dimensionscore_validation_score_bounds():
    """Test DimensionScore enforces score bounds [0.0, 1.0]."""
    # Valid scores
    dim1 = DimensionScore(dimension="test", score=0.0)
    assert dim1.score == 0.0

    dim2 = DimensionScore(dimension="test", score=1.0)
    assert dim2.score == 1.0

    dim3 = DimensionScore(dimension="test", score=0.5)
    assert dim3.score == 0.5

    # Invalid: below 0
    with pytest.raises(ValidationError):
        DimensionScore(dimension="test", score=-0.1)

    # Invalid: above 1
    with pytest.raises(ValidationError):
        DimensionScore(dimension="test", score=1.1)


def test_dimensionscore_with_weight_and_details():
    """Test DimensionScore with optional weight and details fields."""
    dim = DimensionScore(
        dimension="reasoning",
        score=0.85,
        weight=0.15,
        details="Keywords: 0.90, Structure: 0.80, Length: 0.85",
    )

    assert dim.dimension == "reasoning"
    assert dim.score == 0.85
    assert dim.weight == 0.15
    assert "Keywords" in dim.details


# ---------------------------------------------------------------------------
# Scorecard tests
# ---------------------------------------------------------------------------


def test_scorecard_creation_and_summary():
    """Test Scorecard creation and summary() output formatting."""
    dimensions = [
        DimensionScore(dimension="decomposition", score=0.9, weight=0.2),
        DimensionScore(dimension="ppa_accuracy", score=0.8, weight=0.2),
        DimensionScore(dimension="reasoning", score=0.75, weight=0.1),
    ]

    scorecard = Scorecard(
        demo_name="test_demo",
        dimensions=dimensions,
        composite_score=0.82,
        passed=True,
    )

    assert scorecard.demo_name == "test_demo"
    assert len(scorecard.dimensions) == 3
    assert scorecard.composite_score == 0.82
    assert scorecard.passed is True

    # Check summary output
    summary = scorecard.summary()
    assert "test_demo" in summary
    assert "PASS" in summary
    assert "0.82" in summary
    assert "decomposition" in summary
    assert "ppa_accuracy" in summary
    assert "reasoning" in summary

    # Check that bar chart exists (10 characters)
    assert "[" in summary
    assert "]" in summary


def test_scorecard_passed_flag_logic():
    """Test Scorecard passed flag reflects composite_score threshold."""
    # High score: passed
    scorecard_pass = Scorecard(
        demo_name="pass_test",
        composite_score=0.85,
        passed=True,
    )
    assert scorecard_pass.passed is True
    assert "PASS" in scorecard_pass.summary()

    # Low score: failed
    scorecard_fail = Scorecard(
        demo_name="fail_test",
        composite_score=0.45,
        passed=False,
    )
    assert scorecard_fail.passed is False
    assert "FAIL" in scorecard_fail.summary()


def test_scorecard_empty_dimensions():
    """Test Scorecard with no dimensions (edge case)."""
    scorecard = Scorecard(
        demo_name="empty_test",
        dimensions=[],
        composite_score=0.0,
        passed=False,
    )

    assert len(scorecard.dimensions) == 0
    summary = scorecard.summary()
    assert "empty_test" in summary
    assert "FAIL" in summary


def test_scorecard_summary_bar_chart_format():
    """Test that summary bar chart renders correctly for various scores."""
    dimensions = [
        DimensionScore(dimension="perfect", score=1.0, weight=0.1),
        DimensionScore(dimension="half", score=0.5, weight=0.1),
        DimensionScore(dimension="zero", score=0.0, weight=0.1),
    ]

    scorecard = Scorecard(
        demo_name="bar_test",
        dimensions=dimensions,
        composite_score=0.5,
        passed=True,
    )

    summary = scorecard.summary()

    # Perfect score: 10 '#' marks
    assert "##########" in summary

    # Half score: 5 '#' marks followed by dots
    # (exact format depends on implementation but should have mix)
    lines = summary.split("\n")
    half_line = [l for l in lines if "half" in l][0]
    assert "#" in half_line
    assert "." in half_line

    # Zero score: 10 '.' marks
    zero_line = [l for l in lines if "zero" in l][0]
    assert ".........." in zero_line
