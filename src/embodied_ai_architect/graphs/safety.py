"""Safety-critical detection and governance integration.

Detects when a design is safety-critical, injects approval gates into
governance policy, and adds redundancy requirements to the architecture.

Usage:
    from embodied_ai_architect.graphs.safety import safety_detector
"""

from __future__ import annotations

import logging
from typing import Any

from embodied_ai_architect.graphs.soc_state import SoCDesignState, get_constraints
from embodied_ai_architect.graphs.task_graph import TaskNode

logger = logging.getLogger(__name__)

# Safety standards and their implications
SAFETY_STANDARDS = {
    "IEC 62304": {
        "domain": "medical",
        "classes": ["A", "B", "C"],
        "requires_redundancy": True,
        "requires_ecc": True,
        "requires_watchdog": True,
    },
    "ISO 26262": {
        "domain": "automotive",
        "classes": ["ASIL-A", "ASIL-B", "ASIL-C", "ASIL-D"],
        "requires_redundancy": True,
        "requires_ecc": True,
        "requires_watchdog": True,
    },
    "DO-178C": {
        "domain": "aerospace",
        "classes": ["DAL-A", "DAL-B", "DAL-C", "DAL-D", "DAL-E"],
        "requires_redundancy": True,
        "requires_ecc": True,
        "requires_watchdog": True,
    },
    "IEC 61508": {
        "domain": "industrial",
        "classes": ["SIL-1", "SIL-2", "SIL-3", "SIL-4"],
        "requires_redundancy": True,
        "requires_ecc": True,
        "requires_watchdog": True,
    },
}

# Keywords that indicate safety-critical applications
SAFETY_KEYWORDS = [
    "surgical",
    "medical",
    "patient",
    "life-support",
    "automotive",
    "self-driving",
    "adas",
    "aerospace",
    "flight",
    "avionics",
    "nuclear",
    "reactor",
    "safety-critical",
    "safety critical",
    "fail-safe",
    "failsafe",
]

# Redundancy requirements injected into the architecture
REDUNDANCY_REQUIREMENTS = {
    "dual_lockstep_cpu": {
        "type": "cpu",
        "description": "Dual-lockstep CPU for fault detection",
        "config": {"mode": "lockstep", "cores": 2, "fault_detection": True},
    },
    "ecc_memory": {
        "type": "memory",
        "description": "ECC-protected memory controller",
        "config": {"ecc": True, "scrub_interval_ms": 100},
    },
    "watchdog_timer": {
        "type": "watchdog",
        "description": "Hardware watchdog timer for system reset",
        "config": {"timeout_ms": 1000, "window_mode": True},
    },
}


def detect_safety_critical(state: SoCDesignState) -> dict[str, Any]:
    """Detect if the current design is safety-critical.

    Checks constraints.safety_critical flag, safety_standard field,
    and goal/use_case keywords.

    Returns:
        Dict with is_safety_critical, detected_standard, required_actions.
    """
    constraints = get_constraints(state)
    goal = state.get("goal", "").lower()
    use_case = state.get("use_case", "").lower()

    is_critical = constraints.safety_critical
    detected_standard = constraints.safety_standard or ""
    reasons: list[str] = []

    # Check explicit flag
    if is_critical:
        reasons.append("constraints.safety_critical=True")

    # Check safety standard
    if detected_standard:
        is_critical = True
        reasons.append(f"safety_standard={detected_standard}")

    # Check keywords in goal/use_case
    text = f"{goal} {use_case}"
    for kw in SAFETY_KEYWORDS:
        if kw in text:
            is_critical = True
            reasons.append(f"keyword '{kw}' found in goal/use_case")
            break

    # Determine required actions
    required_actions: list[str] = []
    if is_critical:
        required_actions = [
            "add_redundancy",
            "add_ecc_memory",
            "add_watchdog",
            "require_approval_for_safety_decisions",
        ]

    # Match to known standard
    matched_standard = None
    for std_name, std_info in SAFETY_STANDARDS.items():
        if std_name.lower() in detected_standard.lower():
            matched_standard = std_info
            break

    return {
        "is_safety_critical": is_critical,
        "detected_standard": detected_standard,
        "matched_standard_info": matched_standard,
        "reasons": reasons,
        "required_actions": required_actions,
    }


def safety_detector(task: TaskNode, state: SoCDesignState) -> dict[str, Any]:
    """Specialist agent: detect safety-critical constraints and inject requirements.

    Analyzes the design goal and constraints for safety implications,
    adds redundancy requirements, and updates governance policy with
    approval gates for safety-related decisions.

    Writes to state: safety_analysis, governance (updated policy)
    """
    analysis = detect_safety_critical(state)

    state_updates: dict[str, Any] = {"safety_analysis": analysis}

    if analysis["is_safety_critical"]:
        # Inject governance safety actions
        governance = dict(state.get("governance", {}))
        approval_actions = list(governance.get("approval_required_actions", []))
        safety_actions = [
            "change_safety_architecture",
            "remove_redundancy",
            "override_safety_constraint",
            "modify_watchdog_config",
        ]
        for action in safety_actions:
            if action not in approval_actions:
                approval_actions.append(action)
        governance["approval_required_actions"] = approval_actions
        governance["safety_critical_actions"] = safety_actions
        state_updates["governance"] = governance

        # Record redundancy requirements for architecture composer
        analysis["redundancy_requirements"] = [
            REDUNDANCY_REQUIREMENTS["dual_lockstep_cpu"],
            REDUNDANCY_REQUIREMENTS["ecc_memory"],
            REDUNDANCY_REQUIREMENTS["watchdog_timer"],
        ]

        summary = (
            f"Safety-critical design detected: {analysis['detected_standard'] or 'implicit'}. "
            f"Added {len(analysis['redundancy_requirements'])} redundancy requirements "
            f"and {len(safety_actions)} governance approval gates."
        )
    else:
        summary = "No safety-critical requirements detected"

    return {
        "summary": summary,
        "safety_analysis": analysis,
        "_state_updates": state_updates,
    }
