"""Tests for CostTracker and governance extensions."""

from __future__ import annotations

import pytest

from embodied_ai_architect.graphs.governance import (
    CostTracker,
    GovernanceGuard,
    GovernancePolicy,
)


class TestCostTracker:
    def test_initial_state(self):
        tracker = CostTracker()
        assert tracker.total_tokens == 0
        assert tracker.cost_by_agent == {}

    def test_add_cost(self):
        tracker = CostTracker()
        tracker.add_cost("workload_analyzer", 1000)
        assert tracker.total_tokens == 1000
        assert tracker.cost_by_agent == {"workload_analyzer": 1000}

    def test_add_cost_multiple_agents(self):
        tracker = CostTracker()
        tracker.add_cost("workload_analyzer", 1000)
        tracker.add_cost("hw_explorer", 500)
        tracker.add_cost("workload_analyzer", 200)
        assert tracker.total_tokens == 1700
        assert tracker.cost_by_agent["workload_analyzer"] == 1200
        assert tracker.cost_by_agent["hw_explorer"] == 500

    def test_estimated_cost_usd_default_rate(self):
        tracker = CostTracker()
        tracker.add_cost("test", 10000)
        cost = tracker.estimated_cost_usd()
        assert cost == pytest.approx(0.03, abs=0.001)

    def test_estimated_cost_usd_custom_rate(self):
        tracker = CostTracker()
        tracker.add_cost("test", 10000)
        cost = tracker.estimated_cost_usd(rate_per_1k=0.01)
        assert cost == pytest.approx(0.1, abs=0.001)

    def test_estimated_cost_usd_zero_tokens(self):
        tracker = CostTracker()
        assert tracker.estimated_cost_usd() == 0.0

    def test_format_cost_report(self):
        tracker = CostTracker()
        tracker.add_cost("workload_analyzer", 5000)
        tracker.add_cost("hw_explorer", 3000)
        report = tracker.format_cost_report()
        assert "8,000 tokens" in report
        assert "workload_analyzer" in report
        assert "hw_explorer" in report

    def test_format_cost_report_empty(self):
        tracker = CostTracker()
        report = tracker.format_cost_report()
        assert "0 tokens" in report


class TestGovernancePolicySafetyExtensions:
    def test_safety_critical_actions_default(self):
        policy = GovernancePolicy()
        assert policy.safety_critical_actions == []

    def test_safety_critical_actions_set(self):
        policy = GovernancePolicy(
            safety_critical_actions=["change_safety_architecture", "remove_redundancy"]
        )
        assert len(policy.safety_critical_actions) == 2


class TestGovernanceGuardSafetyMethods:
    def test_auto_detect_safety_critical(self):
        policy = GovernancePolicy(safety_critical_actions=["change_safety_architecture"])
        guard = GovernanceGuard(policy)
        assert guard.auto_detect_safety_critical("change_safety_architecture") is True
        assert guard.auto_detect_safety_critical("normal_action") is False

    def test_flag_safety_decision(self):
        policy = GovernancePolicy(safety_critical_actions=["remove_redundancy"])
        guard = GovernanceGuard(policy)
        entry = guard.flag_safety_decision("critic", "remove_redundancy", iteration=3)
        assert entry.agent == "critic"
        assert "SAFETY" in entry.action
        assert entry.human_approved is True
        assert entry.iteration == 3
        # Should be in audit entries
        assert len(guard.audit_entries) == 1

    def test_flag_safety_decision_accumulates(self):
        guard = GovernanceGuard()
        guard.flag_safety_decision("a", "action1")
        guard.flag_safety_decision("b", "action2")
        assert len(guard.audit_entries) == 2
