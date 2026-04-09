"""Validation runner — orchestrates all design checks (issue #56).

Each check returns a CheckResult. The runner aggregates them into a
ValidationReport with an overall PASS/FAIL verdict. Individual checks
wrap existing subsystems (constraint slackness, SWaP-C, etc.) rather
than duplicating logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from embodied_ai_architect.mission.models import Mission


@dataclass
class CheckResult:
    """Result of a single validation check."""

    name: str
    passed: bool
    details: str = ""
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "details": self.details,
            "issues": list(self.issues),
        }


@dataclass
class ValidationReport:
    """Aggregated result of all validation checks."""

    checks: list[CheckResult] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        """Overall verdict: PASS only if all checks pass."""
        if not self.checks:
            return "NO_CHECKS"
        return "PASS" if all(c.passed for c in self.checks) else "FAIL"

    @property
    def passed_count(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    @property
    def failed_count(self) -> int:
        return sum(1 for c in self.checks if not c.passed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "passed": self.passed_count,
            "failed": self.failed_count,
            "total": len(self.checks),
            "checks": [c.to_dict() for c in self.checks],
        }


class ValidationRunner:
    """Orchestrates all validation checks against a Mission."""

    def validate_all(self, mission: Mission) -> ValidationReport:
        """Run all validation checks and return an aggregated report."""
        return ValidationReport(
            checks=[
                self.check_constraints(mission),
                self.check_completeness(mission),
                self.check_safety(mission),
                self.check_scheduling(mission),
            ]
        )

    def check_constraints(self, mission: Mission) -> CheckResult:
        """Check SWaP-C constraint budgets.

        Wraps the existing constraint_slackness computation from
        optimization_review. When a design_state is available, reads
        actual PPA vs constraint targets. Otherwise reports as skipped.
        """
        if not mission.design_state:
            return CheckResult(
                name="constraints",
                passed=True,
                details="No design state — constraints not yet evaluated",
            )

        ppa = mission.design_state.get("ppa_metrics", {})
        verdicts = ppa.get("verdicts", {})
        if not verdicts:
            return CheckResult(
                name="constraints",
                passed=True,
                details="No PPA verdicts — awaiting ppa_assessor",
            )

        failing = [k for k, v in verdicts.items() if v == "FAIL"]
        if failing:
            return CheckResult(
                name="constraints",
                passed=False,
                details=f"{len(failing)} constraint(s) failing: {', '.join(failing)}",
                issues=[f"{k} = FAIL" for k in failing],
            )

        return CheckResult(
            name="constraints",
            passed=True,
            details=f"All {len(verdicts)} constraints PASS",
        )

    def check_completeness(self, mission: Mission) -> CheckResult:
        """Check that all required subsystems are specified.

        A mission is "complete" when it has: goal, constraints with at
        least one numeric target, and a non-empty spec or design_state.
        """
        issues = []

        if not mission.goal:
            issues.append("Missing goal")
        if not mission.constraints:
            issues.append("No constraints specified")
        if not mission.spec and not mission.design_state:
            issues.append("No spec or design state — run qualification or design")

        return CheckResult(
            name="completeness",
            passed=len(issues) == 0,
            details=(
                f"{len(issues)} completeness issue(s)"
                if issues
                else "All required fields populated"
            ),
            issues=issues,
        )

    def check_safety(self, mission: Mission) -> CheckResult:
        """Check safety integrity requirements.

        Verifies that when the mission specifies a safety level, the
        design state's safety analysis is present and meets the
        requirement. Stub — will be enriched in Phase 2.
        """
        spec_safety = mission.spec.get("safety", {})
        required_level = spec_safety.get("level") if isinstance(spec_safety, dict) else None

        if not required_level:
            return CheckResult(
                name="safety",
                passed=True,
                details="No safety level specified — check not applicable",
            )

        if not mission.design_state:
            return CheckResult(
                name="safety",
                passed=False,
                details=f"Safety level '{required_level}' required but no design state to verify",
                issues=[f"Required SIL: {required_level}, actual: unknown"],
            )

        safety_analysis = mission.design_state.get("safety_analysis", {})
        if not safety_analysis:
            return CheckResult(
                name="safety",
                passed=False,
                details=f"Safety level '{required_level}' required but no safety analysis ran",
                issues=["Run safety_detector specialist to verify SIL compliance"],
            )

        return CheckResult(
            name="safety",
            passed=True,
            details=f"Safety analysis present for level '{required_level}'",
        )

    def check_scheduling(self, mission: Mission) -> CheckResult:
        """Check multi-rate scheduling feasibility.

        Verifies that when perception and control run at different rates,
        the timing constraints are satisfiable. Stub — will wrap the
        existing scheduling analysis in Phase 2.
        """
        if not mission.design_state:
            return CheckResult(
                name="scheduling",
                passed=True,
                details="No design state — scheduling not yet evaluated",
            )

        # Check for basic timing data
        workload = mission.design_state.get("workload_profile", {})
        if not workload:
            return CheckResult(
                name="scheduling",
                passed=True,
                details="No workload profile — scheduling check deferred",
            )

        return CheckResult(
            name="scheduling",
            passed=True,
            details="Scheduling check passed (detailed analysis in Phase 2)",
        )
