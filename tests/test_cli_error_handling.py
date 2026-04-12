"""Negative CLI tests -- bad inputs produce friendly errors, not tracebacks (issue #147).

Every error path must:
1. Return a non-zero exit code
2. Contain a user-friendly message (e.g. "not found")
3. NEVER contain a raw Python traceback ("Traceback")
"""

import json

import pytest
from click.testing import CliRunner

from embodied_ai_architect.cli.commands.actuator import actuator
from embodied_ai_architect.cli.commands.mission import mission
from embodied_ai_architect.cli.commands.sensor import sensor
from embodied_ai_architect.cli.commands.synthesize import synthesize
from embodied_ai_architect.cli.commands.validate import validate
import embodied_ai_architect.mission.store as store_mod


@pytest.fixture(autouse=True)
def _use_tmp_store(tmp_path, monkeypatch):
    """Redirect MissionStore to a temp directory."""
    monkeypatch.setattr(store_mod, "DEFAULT_MISSIONS_DIR", tmp_path)


def _runner():
    return CliRunner()


def _assert_no_traceback(result):
    """Ensure no raw Python traceback leaked into output."""
    assert "Traceback" not in (result.output or ""), f"Traceback found in output:\n{result.output}"


# ── Mission errors ────────────────────────────────────────────────────


class TestMissionShowNotFound:
    def test_nonexistent(self):
        runner = _runner()
        res = runner.invoke(mission, ["show", "nonexistent"], obj={})
        assert res.exit_code != 0
        assert "not found" in res.output.lower()
        _assert_no_traceback(res)


class TestMissionDeleteNotFound:
    def test_nonexistent(self):
        runner = _runner()
        res = runner.invoke(mission, ["delete", "nonexistent", "--yes"], obj={})
        assert res.exit_code != 0
        assert "not found" in res.output.lower()
        _assert_no_traceback(res)


# ── Sensor errors ─────────────────────────────────────────────────────


class TestSensorShowNotFound:
    def test_nonexistent(self):
        runner = _runner()
        res = runner.invoke(sensor, ["show", "nonexistent.sensor"], obj={})
        assert res.exit_code != 0
        assert "not found" in res.output.lower()
        _assert_no_traceback(res)


class TestSensorCompareNeedsTwo:
    def test_only_one_id(self):
        runner = _runner()
        res = runner.invoke(sensor, ["compare", "visual.rgb_camera"], obj={})
        assert res.exit_code != 0
        assert "at least 2" in res.output.lower()
        _assert_no_traceback(res)


class TestSensorSelectMissionNotFound:
    def test_nonexistent_mission(self):
        runner = _runner()
        res = runner.invoke(sensor, ["select", "nonexistent-mission", "visual.rgb_camera"], obj={})
        assert res.exit_code != 0
        assert "not found" in res.output.lower()
        _assert_no_traceback(res)


class TestSensorBudgetMissionNotFound:
    def test_nonexistent_mission(self):
        runner = _runner()
        res = runner.invoke(sensor, ["budget", "nonexistent-mission"], obj={})
        assert res.exit_code != 0
        assert "not found" in res.output.lower()
        _assert_no_traceback(res)


class TestSensorFusionMissionNotFound:
    def test_nonexistent_mission(self):
        runner = _runner()
        res = runner.invoke(sensor, ["fusion", "nonexistent-mission"], obj={})
        assert res.exit_code != 0
        assert "not found" in res.output.lower()
        _assert_no_traceback(res)


# ── Actuator errors ───────────────────────────────────────────────────


class TestActuatorShowNotFound:
    def test_nonexistent(self):
        runner = _runner()
        res = runner.invoke(actuator, ["show", "nonexistent.actuator"], obj={})
        assert res.exit_code != 0
        assert "not found" in res.output.lower()
        _assert_no_traceback(res)


class TestActuatorControlRateNotFound:
    def test_nonexistent(self):
        runner = _runner()
        res = runner.invoke(actuator, ["control-rate", "nonexistent.id"], obj={})
        assert res.exit_code != 0
        assert "not found" in res.output.lower()
        _assert_no_traceback(res)


# ── Synthesize errors ─────────────────────────────────────────────────


class TestSynthesizeSystemNotFound:
    def test_nonexistent_mission(self):
        runner = _runner()
        res = runner.invoke(synthesize, ["system", "nonexistent"], obj={})
        assert res.exit_code != 0
        assert "not found" in res.output.lower()
        _assert_no_traceback(res)


# ── Validate errors ───────────────────────────────────────────────────


class TestValidateMissionNotFound:
    def test_nonexistent(self):
        runner = _runner()
        res = runner.invoke(validate, ["mission", "nonexistent"], obj={})
        assert res.exit_code != 0
        assert "not found" in res.output.lower()
        _assert_no_traceback(res)


# ── JSON error responses ─────────────────────────────────────────────


class TestSensorShowNotFoundJson:
    def test_json_error(self):
        runner = _runner()
        res = runner.invoke(sensor, ["show", "nonexistent"], obj={"json": True})
        assert res.exit_code != 0
        data = json.loads(res.output)
        assert "error" in data
        _assert_no_traceback(res)


class TestActuatorShowNotFoundJson:
    def test_json_error(self):
        runner = _runner()
        res = runner.invoke(actuator, ["show", "nonexistent"], obj={"json": True})
        assert res.exit_code != 0
        data = json.loads(res.output)
        assert "error" in data
        _assert_no_traceback(res)
