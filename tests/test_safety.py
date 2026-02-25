"""Tests for safety-critical detection and governance integration.

Tests detect_safety_critical, safety_detector specialist, and safety
standards/redundancy requirements.
"""

import pytest

from embodied_ai_architect.graphs.safety import (
    REDUNDANCY_REQUIREMENTS,
    SAFETY_STANDARDS,
    detect_safety_critical,
    safety_detector,
)
from embodied_ai_architect.graphs.soc_state import (
    DesignConstraints,
    create_initial_soc_state,
)
from embodied_ai_architect.graphs.task_graph import TaskNode

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def safety_critical_state():
    """State with explicit safety_critical flag."""
    return create_initial_soc_state(
        goal="Design an SoC for medical monitoring device",
        constraints=DesignConstraints(
            max_power_watts=10.0,
            safety_critical=True,
        ),
        use_case="medical",
    )


@pytest.fixture
def safety_standard_state():
    """State with safety standard specified."""
    return create_initial_soc_state(
        goal="Design an SoC for medical device",
        constraints=DesignConstraints(
            max_power_watts=10.0,
            safety_standard="IEC 62304 Class C",
        ),
        use_case="medical",
    )


@pytest.fixture
def surgical_state():
    """State with surgical keyword in goal."""
    return create_initial_soc_state(
        goal="Design an SoC for surgical robot with object detection",
        constraints=DesignConstraints(max_power_watts=15.0),
        use_case="surgical_robot",
    )


@pytest.fixture
def non_safety_state():
    """State without safety-critical indicators."""
    return create_initial_soc_state(
        goal="Design an SoC for entertainment drone",
        constraints=DesignConstraints(max_power_watts=5.0),
        use_case="entertainment",
    )


def make_task(task_id: str, agent: str, deps: list[str] | None = None) -> TaskNode:
    return TaskNode(id=task_id, name=f"Test {agent}", agent=agent, dependencies=deps or [])


# ============================================================================
# detect_safety_critical Tests
# ============================================================================


class TestDetectSafetyCritical:
    def test_explicit_safety_critical_flag(self, safety_critical_state):
        """Test detection with explicit safety_critical=True flag."""
        analysis = detect_safety_critical(safety_critical_state)

        assert analysis["is_safety_critical"] is True
        assert "constraints.safety_critical=True" in analysis["reasons"]
        assert len(analysis["required_actions"]) > 0
        assert "add_redundancy" in analysis["required_actions"]

    def test_safety_standard_detection(self, safety_standard_state):
        """Test detection with safety_standard field set."""
        analysis = detect_safety_critical(safety_standard_state)

        assert analysis["is_safety_critical"] is True
        assert analysis["detected_standard"] == "IEC 62304 Class C"
        assert any("safety_standard" in reason for reason in analysis["reasons"])

    def test_keyword_detection(self, surgical_state):
        """Test detection with safety keyword 'surgical' in goal."""
        analysis = detect_safety_critical(surgical_state)

        assert analysis["is_safety_critical"] is True
        assert any("keyword 'surgical'" in reason for reason in analysis["reasons"])
        assert len(analysis["required_actions"]) > 0

    def test_non_safety_goal(self, non_safety_state):
        """Test detection with non-safety goal (should be False)."""
        analysis = detect_safety_critical(non_safety_state)

        assert analysis["is_safety_critical"] is False
        assert analysis["reasons"] == []
        assert analysis["required_actions"] == []

    def test_required_actions(self, safety_critical_state):
        """Test that required actions are present for safety-critical designs."""
        analysis = detect_safety_critical(safety_critical_state)

        expected_actions = [
            "add_redundancy",
            "add_ecc_memory",
            "add_watchdog",
            "require_approval_for_safety_decisions",
        ]

        for action in expected_actions:
            assert action in analysis["required_actions"]

    def test_matched_standard_info(self, safety_standard_state):
        """Test that matched standard info is populated."""
        analysis = detect_safety_critical(safety_standard_state)

        assert analysis["matched_standard_info"] is not None
        assert analysis["matched_standard_info"]["domain"] == "medical"
        assert analysis["matched_standard_info"]["requires_redundancy"] is True


# ============================================================================
# safety_detector Specialist Tests
# ============================================================================


class TestSafetyDetector:
    def test_safety_critical_injects_governance(self, safety_critical_state):
        """Test that safety_detector injects governance approval gates."""
        task = make_task("t1", "safety_detector")

        result = safety_detector(task, safety_critical_state)

        assert "summary" in result
        assert "safety_analysis" in result
        assert "_state_updates" in result

        # Check state updates contain governance
        assert "governance" in result["_state_updates"]
        governance = result["_state_updates"]["governance"]
        assert "approval_required_actions" in governance
        assert "safety_critical_actions" in governance

        # Verify safety actions are present
        safety_actions = governance["safety_critical_actions"]
        assert "change_safety_architecture" in safety_actions
        assert "remove_redundancy" in safety_actions
        assert "override_safety_constraint" in safety_actions
        assert "modify_watchdog_config" in safety_actions

    def test_non_safety_no_redundancy(self, non_safety_state):
        """Test that non-safety case returns no redundancy requirements."""
        task = make_task("t1", "safety_detector")

        result = safety_detector(task, non_safety_state)

        assert "summary" in result
        assert "No safety-critical requirements detected" in result["summary"]

        # Should still have safety_analysis in result
        assert "safety_analysis" in result
        assert result["safety_analysis"]["is_safety_critical"] is False
        assert "redundancy_requirements" not in result["safety_analysis"]

    def test_governance_policy_additions(self, surgical_state):
        """Test that governance policy gets safety_critical_actions added."""
        task = make_task("t1", "safety_detector")

        # Pre-populate governance with some existing actions
        surgical_state["governance"] = {"approval_required_actions": ["deploy_to_production"]}

        result = safety_detector(task, surgical_state)

        governance = result["_state_updates"]["governance"]
        approval_actions = governance["approval_required_actions"]

        # Should preserve existing action
        assert "deploy_to_production" in approval_actions

        # Should add safety actions
        assert "change_safety_architecture" in approval_actions
        assert "remove_redundancy" in approval_actions

    def test_redundancy_requirements_added(self, safety_standard_state):
        """Test that redundancy requirements are added to analysis."""
        task = make_task("t1", "safety_detector")

        result = safety_detector(task, safety_standard_state)

        analysis = result["safety_analysis"]
        assert "redundancy_requirements" in analysis
        assert len(analysis["redundancy_requirements"]) == 3

        # Check that the redundancy requirements contain expected entries
        req_types = [req["type"] for req in analysis["redundancy_requirements"]]
        assert "cpu" in req_types
        assert "memory" in req_types
        assert "watchdog" in req_types

    def test_summary_includes_details(self, safety_standard_state):
        """Test that summary includes detailed information."""
        task = make_task("t1", "safety_detector")

        result = safety_detector(task, safety_standard_state)

        summary = result["summary"]
        assert "Safety-critical design detected" in summary
        assert "IEC 62304 Class C" in summary
        assert "redundancy requirements" in summary
        assert "governance approval gates" in summary


# ============================================================================
# Constants Tests
# ============================================================================


class TestSafetyConstants:
    def test_redundancy_requirements_keys(self):
        """Test that REDUNDANCY_REQUIREMENTS has expected keys."""
        expected_keys = ["dual_lockstep_cpu", "ecc_memory", "watchdog_timer"]

        for key in expected_keys:
            assert key in REDUNDANCY_REQUIREMENTS

    def test_redundancy_requirements_structure(self):
        """Test structure of redundancy requirements."""
        for key, req in REDUNDANCY_REQUIREMENTS.items():
            assert "type" in req
            assert "description" in req
            assert "config" in req
            assert isinstance(req["config"], dict)

    def test_safety_standards_present(self):
        """Test that SAFETY_STANDARDS has expected standards."""
        expected_standards = ["IEC 62304", "ISO 26262", "DO-178C", "IEC 61508"]

        for standard in expected_standards:
            assert standard in SAFETY_STANDARDS

    def test_safety_standards_structure(self):
        """Test structure of safety standards."""
        for standard, info in SAFETY_STANDARDS.items():
            assert "domain" in info
            assert "classes" in info
            assert isinstance(info["classes"], list)
            assert "requires_redundancy" in info
            assert "requires_ecc" in info
            assert "requires_watchdog" in info

    def test_safety_standard_domains(self):
        """Test that safety standards cover expected domains."""
        domains = [info["domain"] for info in SAFETY_STANDARDS.values()]
        assert "medical" in domains
        assert "automotive" in domains
        assert "aerospace" in domains
        assert "industrial" in domains
