"""Tests for the validation runner and CLI commands (issue #56)."""

import pytest

from click.testing import CliRunner

from embodied_ai_architect.cli.commands.validate import validate
from embodied_ai_architect.mission import Mission, MissionStore
from embodied_ai_architect.validation import ValidationReport, ValidationRunner
import embodied_ai_architect.mission.store as store_mod


pytestmark = pytest.mark.cli


@pytest.fixture(autouse=True)
def _use_tmp_store(tmp_path, monkeypatch):
    """Point MissionStore default dir to tmp_path."""
    monkeypatch.setattr(store_mod, "DEFAULT_MISSIONS_DIR", tmp_path)


# ---------------------------------------------------------------------------
# ValidationRunner unit tests
# ---------------------------------------------------------------------------


class TestCheckConstraints:
    def test_no_design_state_passes(self):
        m = Mission(goal="Test")
        check = ValidationRunner().check_constraints(m)
        assert check.passed is True
        assert "No design state" in check.details

    def test_no_verdicts_passes(self):
        m = Mission(goal="Test", design_state={"ppa_metrics": {}})
        check = ValidationRunner().check_constraints(m)
        assert check.passed is True

    def test_all_pass(self):
        m = Mission(
            goal="Test",
            design_state={
                "ppa_metrics": {"verdicts": {"power": "PASS", "latency": "PASS", "cost": "PASS"}}
            },
        )
        check = ValidationRunner().check_constraints(m)
        assert check.passed is True
        assert "3 constraints PASS" in check.details

    def test_failing_constraints(self):
        m = Mission(
            goal="Test",
            design_state={
                "ppa_metrics": {"verdicts": {"power": "FAIL", "latency": "PASS", "cost": "FAIL"}}
            },
        )
        check = ValidationRunner().check_constraints(m)
        assert check.passed is False
        assert "power" in check.details
        assert "cost" in check.details
        assert len(check.issues) == 2


class TestCheckCompleteness:
    def test_empty_mission_fails(self):
        m = Mission()
        check = ValidationRunner().check_completeness(m)
        assert check.passed is False
        assert "Missing goal" in check.issues[0]

    def test_complete_mission_passes(self):
        m = Mission(
            goal="Test goal",
            constraints={"max_power_watts": 5.0},
            spec={"platform_type": "drone"},
        )
        check = ValidationRunner().check_completeness(m)
        assert check.passed is True

    def test_no_constraints_fails(self):
        m = Mission(goal="Test", spec={"platform_type": "drone"})
        check = ValidationRunner().check_completeness(m)
        assert check.passed is False
        assert any("constraints" in i.lower() for i in check.issues)


class TestCheckSafety:
    def test_no_safety_level_passes(self):
        m = Mission(goal="Test")
        check = ValidationRunner().check_safety(m)
        assert check.passed is True
        assert "not applicable" in check.details

    def test_safety_level_without_design_fails(self):
        m = Mission(goal="Test", spec={"safety": {"level": "sil_2"}})
        check = ValidationRunner().check_safety(m)
        assert check.passed is False

    def test_safety_with_analysis_passes(self):
        m = Mission(
            goal="Test",
            spec={"safety": {"level": "sil_1"}},
            design_state={"safety_analysis": {"level": "sil_1"}},
        )
        check = ValidationRunner().check_safety(m)
        assert check.passed is True


class TestCheckScheduling:
    def test_no_design_state_passes(self):
        check = ValidationRunner().check_scheduling(Mission(goal="Test"))
        assert check.passed is True

    def test_with_workload_passes(self):
        m = Mission(
            goal="Test",
            design_state={"workload_profile": {"total_estimated_gflops": 5.0}},
        )
        check = ValidationRunner().check_scheduling(m)
        assert check.passed is True


class TestValidateAll:
    def test_returns_report(self):
        m = Mission(
            goal="Test",
            constraints={"max_power_watts": 5.0},
            spec={"platform_type": "drone"},
        )
        report = ValidationRunner().validate_all(m)
        assert isinstance(report, ValidationReport)
        assert len(report.checks) == 4
        assert report.verdict == "PASS"

    def test_failing_report(self):
        m = Mission()  # empty — completeness will fail
        report = ValidationRunner().validate_all(m)
        assert report.verdict == "FAIL"
        assert report.failed_count >= 1

    def test_report_to_dict(self):
        report = ValidationRunner().validate_all(Mission(goal="Test", constraints={"p": 1}))
        d = report.to_dict()
        assert "verdict" in d
        assert "checks" in d
        assert isinstance(d["checks"], list)


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


class TestValidateMissionCLI:
    def test_all_checks(self):
        store = MissionStore()
        m = Mission(
            name="Test",
            goal="Test goal",
            constraints={"max_power_watts": 5.0},
            spec={"platform_type": "drone"},
        )
        store.save(m)

        result = CliRunner().invoke(validate, ["mission", m.id], obj={})
        assert result.exit_code == 0
        assert "PASS" in result.output

    def test_not_found(self):
        result = CliRunner().invoke(validate, ["mission", "nonexistent"], obj={})
        assert result.exit_code != 0

    def test_json_output(self):
        import json as _json

        store = MissionStore()
        m = Mission(name="J", goal="G", constraints={"p": 1}, spec={"x": 1})
        store.save(m)

        result = CliRunner().invoke(validate, ["mission", m.id], obj={"json": True})
        assert result.exit_code == 0
        data = _json.loads(result.output)
        assert data["verdict"] in ("PASS", "FAIL")
        assert "checks" in data


class TestValidateSubcommands:
    def _make_mission(self):
        store = MissionStore()
        m = Mission(name="Sub", goal="G", constraints={"p": 1}, spec={"x": 1})
        store.save(m)
        return m

    def test_constraints_subcommand(self):
        m = self._make_mission()
        result = CliRunner().invoke(validate, ["constraints", m.id], obj={})
        assert result.exit_code == 0

    def test_completeness_subcommand(self):
        m = self._make_mission()
        result = CliRunner().invoke(validate, ["completeness", m.id], obj={})
        assert result.exit_code == 0

    def test_safety_subcommand(self):
        m = self._make_mission()
        result = CliRunner().invoke(validate, ["safety", m.id], obj={})
        assert result.exit_code == 0

    def test_scheduling_subcommand(self):
        m = self._make_mission()
        result = CliRunner().invoke(validate, ["scheduling", m.id], obj={})
        assert result.exit_code == 0
