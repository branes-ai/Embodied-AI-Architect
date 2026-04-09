"""Design validation framework (issue #56).

Orchestrates all validation checks against a Mission entity:
constraint slackness, SWaP-C budgets, scheduling feasibility,
safety integrity, and subsystem completeness.

Usage:
    from embodied_ai_architect.validation import ValidationRunner

    runner = ValidationRunner()
    report = runner.validate_all(mission)
    print(report.verdict)  # PASS or FAIL
"""

from embodied_ai_architect.validation.runner import ValidationReport, ValidationRunner

__all__ = ["ValidationReport", "ValidationRunner"]
