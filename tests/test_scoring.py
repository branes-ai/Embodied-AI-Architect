"""Tests for the 9-dimension scoring functions.

Tests all scoring functions with known deterministic inputs to verify
correct score calculation and bounds.
"""

from embodied_ai_architect.graphs.evaluation import GoldStandard, RunTrace
from embodied_ai_architect.graphs.scoring import (
    score_adaptability,
    score_convergence,
    score_decomposition,
    score_efficiency,
    score_exploration_efficiency,
    score_governance,
    score_ppa_accuracy,
    score_reasoning,
    score_tool_use,
)

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_trace(**kwargs) -> RunTrace:
    """Create a RunTrace with defaults + overrides."""
    defaults = {"demo_name": "test"}
    defaults.update(kwargs)
    return RunTrace(**defaults)


def _make_gold(**kwargs) -> GoldStandard:
    """Create a GoldStandard with defaults + overrides."""
    defaults = {"demo_name": "test"}
    defaults.update(kwargs)
    return GoldStandard(**defaults)


# ---------------------------------------------------------------------------
# score_decomposition tests
# ---------------------------------------------------------------------------


def test_score_decomposition_matching_task_graph():
    """Test decomposition score with exact matching task graph."""
    task_graph = {
        "nodes": {
            "n1": {"agent": "ModelAnalyzer", "dependencies": []},
            "n2": {"agent": "HardwareProfile", "dependencies": ["n1"]},
        }
    }

    trace = _make_trace(task_graph=task_graph)
    gold = _make_gold(expected_task_graph=task_graph)

    result = score_decomposition(trace, gold)

    assert result.dimension == "decomposition"
    assert 0.0 <= result.score <= 1.0
    assert result.score == 1.0  # perfect match


def test_score_decomposition_partial_match():
    """Test decomposition score with partial node/edge overlap."""
    trace_graph = {
        "nodes": {
            "n1": {"agent": "ModelAnalyzer", "dependencies": []},
            "n2": {"agent": "HardwareProfile", "dependencies": ["n1"]},
            "n3": {"agent": "ExtraAgent", "dependencies": []},
        }
    }

    gold_graph = {
        "nodes": {
            "n1": {"agent": "ModelAnalyzer", "dependencies": []},
            "n2": {"agent": "HardwareProfile", "dependencies": ["n1"]},
        }
    }

    trace = _make_trace(task_graph=trace_graph)
    gold = _make_gold(expected_task_graph=gold_graph)

    result = score_decomposition(trace, gold)

    assert result.dimension == "decomposition"
    assert 0.0 <= result.score <= 1.0
    assert result.score < 1.0  # partial match


def test_score_decomposition_empty_graphs():
    """Test decomposition score with empty task graphs."""
    trace = _make_trace(task_graph={})
    gold = _make_gold(expected_task_graph={})

    result = score_decomposition(trace, gold)

    assert result.dimension == "decomposition"
    assert 0.0 <= result.score <= 1.0
    # Should be high score since no expectation exists


# ---------------------------------------------------------------------------
# score_ppa_accuracy tests
# ---------------------------------------------------------------------------


def test_score_ppa_accuracy_exact_match():
    """Test PPA accuracy score with exact matching metrics."""
    ppa = {"power_watts": 5.0, "latency_ms": 10.0, "area_mm2": 8.0, "cost_usd": 50.0}

    trace = _make_trace(ppa_metrics=ppa)
    gold = _make_gold(expected_ppa=ppa)

    result = score_ppa_accuracy(trace, gold)

    assert result.dimension == "ppa_accuracy"
    assert 0.0 <= result.score <= 1.0
    assert result.score == 1.0  # <10% error = 1.0


def test_score_ppa_accuracy_10_percent_error():
    """Test PPA accuracy score with 10% error (should get 0.75)."""
    expected_ppa = {"power_watts": 10.0, "latency_ms": 20.0}
    actual_ppa = {"power_watts": 11.0, "latency_ms": 22.0}  # 10% error

    trace = _make_trace(ppa_metrics=actual_ppa)
    gold = _make_gold(expected_ppa=expected_ppa)

    result = score_ppa_accuracy(trace, gold)

    assert result.dimension == "ppa_accuracy"
    assert 0.0 <= result.score <= 1.0
    assert result.score == 0.75  # 10-25% error range


def test_score_ppa_accuracy_30_percent_error():
    """Test PPA accuracy score with 30% error (should get 0.5)."""
    expected_ppa = {"power_watts": 10.0, "latency_ms": 20.0}
    actual_ppa = {"power_watts": 13.0, "latency_ms": 26.0}  # 30% error

    trace = _make_trace(ppa_metrics=actual_ppa)
    gold = _make_gold(expected_ppa=expected_ppa)

    result = score_ppa_accuracy(trace, gold)

    assert result.dimension == "ppa_accuracy"
    assert 0.0 <= result.score <= 1.0
    assert result.score == 0.5  # 25-50% error range


def test_score_ppa_accuracy_60_percent_error():
    """Test PPA accuracy score with 60% error (should get 0.25)."""
    expected_ppa = {"power_watts": 10.0, "latency_ms": 20.0}
    actual_ppa = {"power_watts": 16.0, "latency_ms": 32.0}  # 60% error

    trace = _make_trace(ppa_metrics=actual_ppa)
    gold = _make_gold(expected_ppa=expected_ppa)

    result = score_ppa_accuracy(trace, gold)

    assert result.dimension == "ppa_accuracy"
    assert 0.0 <= result.score <= 1.0
    assert result.score == 0.25  # >50% error


# ---------------------------------------------------------------------------
# score_exploration_efficiency tests
# ---------------------------------------------------------------------------


def test_score_exploration_efficiency_with_pareto_points():
    """Test exploration efficiency with some non-dominated points."""
    pareto_points = [
        {"power": 5.0, "latency": 10.0, "dominated": False},
        {"power": 6.0, "latency": 12.0, "dominated": True},
        {"power": 4.5, "latency": 9.0, "dominated": False},
    ]

    trace = _make_trace(pareto_points=pareto_points)
    gold = _make_gold(expected_pareto_points=2)

    result = score_exploration_efficiency(trace, gold)

    assert result.dimension == "exploration_efficiency"
    assert 0.0 <= result.score <= 1.0
    assert result.score > 0.0  # should get credit for non-dominated points


def test_score_exploration_efficiency_all_dominated():
    """Test exploration efficiency when all points are dominated."""
    pareto_points = [
        {"power": 5.0, "latency": 10.0, "dominated": True},
        {"power": 6.0, "latency": 12.0, "dominated": True},
    ]

    trace = _make_trace(pareto_points=pareto_points)
    gold = _make_gold(expected_pareto_points=2)

    result = score_exploration_efficiency(trace, gold)

    assert result.dimension == "exploration_efficiency"
    assert 0.0 <= result.score <= 1.0
    # Score should be low since no non-dominated points


def test_score_exploration_efficiency_no_exploration_expected():
    """Test exploration efficiency when no Pareto exploration is expected."""
    trace = _make_trace(pareto_points=[])
    gold = _make_gold(expected_pareto_points=0)

    result = score_exploration_efficiency(trace, gold)

    assert result.dimension == "exploration_efficiency"
    assert 0.0 <= result.score <= 1.0
    assert result.score == 0.75  # default when no exploration expected


# ---------------------------------------------------------------------------
# score_reasoning tests
# ---------------------------------------------------------------------------


def test_score_reasoning_with_keywords():
    """Test reasoning score with rationale containing expected keywords."""
    rationale = [
        "[Planner] Using KPU accelerator for power efficiency",
        "[Critic] Latency constraint requires NPU acceleration",
    ]
    keywords = ["kpu", "power", "latency", "npu"]

    trace = _make_trace(design_rationale=rationale)
    gold = _make_gold(rationale_keywords=keywords)

    result = score_reasoning(trace, gold)

    assert result.dimension == "reasoning"
    assert 0.0 <= result.score <= 1.0
    assert result.score > 0.5  # should get high score for keyword match


def test_score_reasoning_missing_keywords():
    """Test reasoning score with rationale missing expected keywords."""
    rationale = [
        "[Planner] Starting design",
        "[Critic] Checking constraints",
    ]
    keywords = ["kpu", "power", "latency", "optimization"]

    trace = _make_trace(design_rationale=rationale)
    gold = _make_gold(rationale_keywords=keywords)

    result = score_reasoning(trace, gold)

    assert result.dimension == "reasoning"
    assert 0.0 <= result.score <= 1.0
    # Should get lower score due to missing keywords


def test_score_reasoning_no_rationale():
    """Test reasoning score with no rationale recorded."""
    trace = _make_trace(design_rationale=[])
    gold = _make_gold(rationale_keywords=["kpu", "power"])

    result = score_reasoning(trace, gold)

    assert result.dimension == "reasoning"
    assert result.score == 0.25  # specific score for no rationale


# ---------------------------------------------------------------------------
# score_convergence tests
# ---------------------------------------------------------------------------


def test_score_convergence_monotonic_improvement():
    """Test convergence score with monotonically improving iterations."""
    history = [
        {"verdicts": {"power": "FAIL", "latency": "PASS"}},
        {"verdicts": {"power": "FAIL", "latency": "PASS"}},
        {"verdicts": {"power": "PASS", "latency": "PASS"}},
    ]
    ppa_metrics = {"verdicts": {"power": "PASS", "latency": "PASS"}}

    trace = _make_trace(iteration_history=history, ppa_metrics=ppa_metrics)
    gold = _make_gold()

    result = score_convergence(trace, gold)

    assert result.dimension == "convergence"
    assert 0.0 <= result.score <= 1.0
    assert result.score >= 0.8  # monotonic improvement + all PASS bonus


def test_score_convergence_single_pass():
    """Test convergence score with single iteration that passes."""
    ppa_metrics = {"verdicts": {"power": "PASS", "latency": "PASS"}}

    trace = _make_trace(iteration_history=[], ppa_metrics=ppa_metrics)
    gold = _make_gold()

    result = score_convergence(trace, gold)

    assert result.dimension == "convergence"
    assert 0.0 <= result.score <= 1.0
    assert result.score == 1.0  # single pass with all PASS


def test_score_convergence_non_improving():
    """Test convergence score with non-improving iterations."""
    history = [
        {"verdicts": {"power": "PASS", "latency": "PASS"}},
        {"verdicts": {"power": "FAIL", "latency": "PASS"}},
        {"verdicts": {"power": "FAIL", "latency": "FAIL"}},
    ]
    ppa_metrics = {"verdicts": {"power": "FAIL", "latency": "FAIL"}}

    trace = _make_trace(iteration_history=history, ppa_metrics=ppa_metrics)
    gold = _make_gold()

    result = score_convergence(trace, gold)

    assert result.dimension == "convergence"
    assert 0.0 <= result.score <= 1.0
    assert result.score < 0.5  # non-improving should get low score


# ---------------------------------------------------------------------------
# score_governance tests
# ---------------------------------------------------------------------------


def test_score_governance_matching_triggers():
    """Test governance score with all expected triggers present."""
    audit_log = [
        {"action": "power_budget_exceeded"},
        {"action": "latency_violation"},
        {"action": "cost_check"},
    ]
    triggers = ["power_budget_exceeded", "latency_violation"]

    trace = _make_trace(audit_log=audit_log)
    gold = _make_gold(governance_triggers=triggers)

    result = score_governance(trace, gold)

    assert result.dimension == "governance"
    assert 0.0 <= result.score <= 1.0
    assert result.score == 1.0  # all triggers matched


def test_score_governance_partial_match():
    """Test governance score with partial trigger match."""
    audit_log = [
        {"action": "power_budget_exceeded"},
    ]
    triggers = ["power_budget_exceeded", "latency_violation", "cost_check"]

    trace = _make_trace(audit_log=audit_log)
    gold = _make_gold(governance_triggers=triggers)

    result = score_governance(trace, gold)

    assert result.dimension == "governance"
    assert 0.0 <= result.score <= 1.0
    assert 0.3 <= result.score <= 0.4  # 1/3 matched


def test_score_governance_no_audit_log():
    """Test governance score with no audit log but triggers expected."""
    trace = _make_trace(audit_log=[])
    gold = _make_gold(governance_triggers=["power_budget_exceeded"])

    result = score_governance(trace, gold)

    assert result.dimension == "governance"
    assert result.score == 0.0  # no audit log


# ---------------------------------------------------------------------------
# score_tool_use tests
# ---------------------------------------------------------------------------


def test_score_tool_use_exact_match():
    """Test tool use score with exact matching tool calls."""
    tools = ["ModelAnalyzer", "HardwareProfile", "Benchmark"]

    trace = _make_trace(tool_calls=tools)
    gold = _make_gold(expected_tool_calls=tools)

    result = score_tool_use(trace, gold)

    assert result.dimension == "tool_use"
    assert 0.0 <= result.score <= 1.0
    assert result.score == 1.0  # perfect F1


def test_score_tool_use_partial_match():
    """Test tool use score with partial tool overlap."""
    actual_tools = ["ModelAnalyzer", "HardwareProfile", "ExtraTool"]
    expected_tools = ["ModelAnalyzer", "HardwareProfile", "Benchmark"]

    trace = _make_trace(tool_calls=actual_tools)
    gold = _make_gold(expected_tool_calls=expected_tools)

    result = score_tool_use(trace, gold)

    assert result.dimension == "tool_use"
    assert 0.0 <= result.score <= 1.0
    assert 0.5 <= result.score < 1.0  # partial F1


def test_score_tool_use_no_tools():
    """Test tool use score with no tools called."""
    trace = _make_trace(tool_calls=[])
    gold = _make_gold(expected_tool_calls=["ModelAnalyzer", "Benchmark"])

    result = score_tool_use(trace, gold)

    assert result.dimension == "tool_use"
    assert result.score == 0.0  # no tools called


# ---------------------------------------------------------------------------
# score_adaptability tests
# ---------------------------------------------------------------------------


def test_score_adaptability_no_failures():
    """Test adaptability score with zero failures."""
    trace = _make_trace(failures=0, recoveries=0)
    gold = _make_gold()

    result = score_adaptability(trace, gold)

    assert result.dimension == "adaptability"
    assert result.score == 1.0  # perfect score


def test_score_adaptability_some_recoveries():
    """Test adaptability score with partial recovery from failures."""
    trace = _make_trace(failures=4, recoveries=3)
    gold = _make_gold()

    result = score_adaptability(trace, gold)

    assert result.dimension == "adaptability"
    assert 0.0 <= result.score <= 1.0
    assert result.score == 0.75  # 3/4 recovery rate


def test_score_adaptability_no_recoveries():
    """Test adaptability score with failures but no recoveries."""
    trace = _make_trace(failures=5, recoveries=0)
    gold = _make_gold()

    result = score_adaptability(trace, gold)

    assert result.dimension == "adaptability"
    assert result.score == 0.0  # no recovery


# ---------------------------------------------------------------------------
# score_efficiency tests
# ---------------------------------------------------------------------------


def test_score_efficiency_within_budget():
    """Test efficiency score when within all budgets."""
    trace = _make_trace(
        duration_seconds=30.0,
        cost_tokens=50000,
        human_interventions=1,
    )
    gold = _make_gold(
        max_duration_seconds=60.0,
        max_cost_tokens=100000,
        max_human_interventions=3,
    )

    result = score_efficiency(trace, gold)

    assert result.dimension == "efficiency"
    assert 0.0 <= result.score <= 1.0
    assert result.score >= 0.75  # within budgets


def test_score_efficiency_over_budget():
    """Test efficiency score when over time and cost budget."""
    trace = _make_trace(
        duration_seconds=120.0,
        cost_tokens=150000,
        human_interventions=2,
    )
    gold = _make_gold(
        max_duration_seconds=60.0,
        max_cost_tokens=100000,
        max_human_interventions=3,
    )

    result = score_efficiency(trace, gold)

    assert result.dimension == "efficiency"
    assert 0.0 <= result.score <= 1.0
    assert result.score < 0.75  # over budget


def test_score_efficiency_way_over_budget():
    """Test efficiency score when way over all budgets."""
    trace = _make_trace(
        duration_seconds=300.0,
        cost_tokens=500000,
        human_interventions=10,
    )
    gold = _make_gold(
        max_duration_seconds=60.0,
        max_cost_tokens=100000,
        max_human_interventions=3,
    )

    result = score_efficiency(trace, gold)

    assert result.dimension == "efficiency"
    assert 0.0 <= result.score <= 1.0
    assert result.score < 0.5  # way over budget
