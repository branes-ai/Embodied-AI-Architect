"""Tests for the governance layer."""

import pytest

from embodied_ai_architect.graphs.governance import (
    AuditEntry,
    GovernanceGuard,
    GovernancePolicy,
)


class TestGovernancePolicy:
    def test_defaults(self):
        policy = GovernancePolicy()
        assert policy.iteration_limit == 10
        assert policy.cost_budget_tokens == 0  # unlimited
        assert policy.approval_required_actions == []
        assert policy.require_human_approval_on_fail is False
        assert policy.fail_iteration_threshold == 3

    def test_custom_policy(self):
        policy = GovernancePolicy(
            iteration_limit=5,
            cost_budget_tokens=50000,
            approval_required_actions=["deploy", "tape_out"],
            require_human_approval_on_fail=True,
        )
        assert policy.iteration_limit == 5
        assert policy.cost_budget_tokens == 50000
        assert "deploy" in policy.approval_required_actions

    def test_serialization_round_trip(self):
        policy = GovernancePolicy(iteration_limit=5)
        data = policy.model_dump()
        restored = GovernancePolicy(**data)
        assert restored.iteration_limit == 5


class TestAuditEntry:
    def test_defaults(self):
        entry = AuditEntry(agent="optimizer", action="apply INT8")
        assert entry.agent == "optimizer"
        assert entry.action == "apply INT8"
        assert entry.cost_tokens == 0
        assert entry.human_approved is False
        assert entry.iteration == 0
        assert entry.timestamp  # auto-generated

    def test_full_entry(self):
        entry = AuditEntry(
            agent="ppa_assessor",
            action="Assessed PPA",
            input_summary="arch with KPU",
            output_summary="power=6.3W FAIL",
            cost_tokens=1500,
            human_approved=True,
            iteration=2,
        )
        assert entry.cost_tokens == 1500
        assert entry.iteration == 2


class TestGovernanceGuard:
    def test_default_guard(self):
        guard = GovernanceGuard()
        assert guard.check_budget() is True  # unlimited by default
        assert guard.check_iteration_limit(0) is True
        assert guard.total_cost_tokens == 0

    def test_from_dict_empty(self):
        guard = GovernanceGuard.from_dict({})
        assert guard.policy.iteration_limit == 10

    def test_from_dict(self):
        guard = GovernanceGuard.from_dict({"iteration_limit": 3})
        assert guard.policy.iteration_limit == 3

    def test_check_iteration_limit(self):
        guard = GovernanceGuard(GovernancePolicy(iteration_limit=5))
        assert guard.check_iteration_limit(0) is True
        assert guard.check_iteration_limit(4) is True
        assert guard.check_iteration_limit(5) is False
        assert guard.check_iteration_limit(10) is False

    def test_check_budget_unlimited(self):
        guard = GovernanceGuard(GovernancePolicy(cost_budget_tokens=0))
        assert guard.check_budget(999999) is True

    def test_check_budget_limited(self):
        guard = GovernanceGuard(GovernancePolicy(cost_budget_tokens=1000))
        assert guard.check_budget(500) is True
        assert guard.check_budget(1000) is True
        assert guard.check_budget(1001) is False

    def test_check_budget_cumulative(self):
        guard = GovernanceGuard(GovernancePolicy(cost_budget_tokens=1000))
        guard.record(agent="a", action="test", cost_tokens=800)
        assert guard.check_budget(100) is True
        assert guard.check_budget(201) is False

    def test_requires_approval(self):
        guard = GovernanceGuard(GovernancePolicy(approval_required_actions=["deploy", "tape_out"]))
        assert guard.requires_approval("deploy") is True
        assert guard.requires_approval("tape_out") is True
        assert guard.requires_approval("optimize") is False

    def test_should_escalate_to_human(self):
        guard = GovernanceGuard(
            GovernancePolicy(
                require_human_approval_on_fail=True,
                fail_iteration_threshold=3,
            )
        )
        assert guard.should_escalate_to_human(0) is False
        assert guard.should_escalate_to_human(2) is False
        assert guard.should_escalate_to_human(3) is True
        assert guard.should_escalate_to_human(5) is True

    def test_should_not_escalate_when_disabled(self):
        guard = GovernanceGuard(GovernancePolicy(require_human_approval_on_fail=False))
        assert guard.should_escalate_to_human(100) is False

    def test_record(self):
        guard = GovernanceGuard()
        entry = guard.record(
            agent="optimizer",
            action="apply INT8",
            iteration=1,
            cost_tokens=500,
        )
        assert entry.agent == "optimizer"
        assert guard.total_cost_tokens == 500
        assert len(guard.audit_entries) == 1

    def test_record_multiple(self):
        guard = GovernanceGuard()
        guard.record(agent="a", action="first", cost_tokens=100)
        guard.record(agent="b", action="second", cost_tokens=200)
        assert guard.total_cost_tokens == 300
        assert len(guard.audit_entries) == 2
