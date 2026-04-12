"""Validate --json output for all mission-facing CLI commands (issue #146).

Every command that supports JSON output must produce valid JSON with the
expected top-level keys.
"""

import json

import pytest

from click.testing import CliRunner

from embodied_ai_architect.cli.commands.actuator import actuator
from embodied_ai_architect.cli.commands.mission import mission
from embodied_ai_architect.cli.commands.platform import platform
from embodied_ai_architect.cli.commands.sensor import sensor
from embodied_ai_architect.cli.commands.validate import validate
from embodied_ai_architect.mission import Mission, MissionStore
import embodied_ai_architect.mission.store as store_mod


pytestmark = pytest.mark.cli


@pytest.fixture(autouse=True)
def _use_tmp_store(tmp_path, monkeypatch):
    """Redirect MissionStore to a temp directory."""
    monkeypatch.setattr(store_mod, "DEFAULT_MISSIONS_DIR", tmp_path)


def _runner():
    return CliRunner()


@pytest.fixture()
def sample_mission():
    """Create a mission for commands that need one."""
    store = MissionStore()
    m = Mission(
        name="json-test",
        goal="JSON output testing",
        selected_sensors=["visual.rgb_camera"],
        selected_actuators=["motor.brushless_dc"],
    )
    store.save(m)
    return m


# ── Mission commands ──────────────────────────────────────────────────


class TestMissionListJson:
    def test_empty_list(self):
        runner = _runner()
        res = runner.invoke(mission, ["list"], obj={"json": True})
        assert res.exit_code == 0
        data = json.loads(res.output)
        assert isinstance(data, list)

    def test_with_missions(self, sample_mission):
        runner = _runner()
        res = runner.invoke(mission, ["list"], obj={"json": True})
        assert res.exit_code == 0
        data = json.loads(res.output)
        assert isinstance(data, list)
        assert len(data) >= 1


class TestMissionShowJson:
    def test_show_json(self, sample_mission):
        runner = _runner()
        res = runner.invoke(mission, ["show", sample_mission.id], obj={"json": True})
        assert res.exit_code == 0
        data = json.loads(res.output)
        assert isinstance(data, dict)
        assert "id" in data
        assert "name" in data
        assert "status" in data


# ── Sensor commands ───────────────────────────────────────────────────


class TestSensorListJson:
    def test_list_json(self):
        runner = _runner()
        res = runner.invoke(sensor, ["list"], obj={"json": True})
        assert res.exit_code == 0
        data = json.loads(res.output)
        assert isinstance(data, list)
        if data:
            assert "id" in data[0]
            assert "name" in data[0]


class TestSensorShowJson:
    def test_show_json(self):
        runner = _runner()
        res = runner.invoke(sensor, ["show", "visual.rgb_camera"], obj={"json": True})
        assert res.exit_code == 0
        data = json.loads(res.output)
        assert isinstance(data, dict)
        assert "id" in data


class TestSensorSearchJson:
    def test_search_json(self):
        runner = _runner()
        res = runner.invoke(sensor, ["search", "stereo camera"], obj={"json": True})
        assert res.exit_code == 0
        data = json.loads(res.output)
        assert isinstance(data, list)
        if data:
            assert "id" in data[0]
            assert "score" in data[0]


class TestSensorCategoriesJson:
    def test_categories_json(self):
        runner = _runner()
        res = runner.invoke(sensor, ["categories"], obj={"json": True})
        assert res.exit_code == 0
        data = json.loads(res.output)
        assert isinstance(data, list)
        # Should contain string category names
        if data:
            assert isinstance(data[0], str)


# ── Actuator commands ─────────────────────────────────────────────────


class TestActuatorListJson:
    def test_list_json(self):
        runner = _runner()
        res = runner.invoke(actuator, ["list"], obj={"json": True})
        assert res.exit_code == 0
        data = json.loads(res.output)
        assert isinstance(data, list)
        if data:
            assert "id" in data[0]
            assert "name" in data[0]


class TestActuatorShowJson:
    def test_show_json(self):
        runner = _runner()
        res = runner.invoke(actuator, ["show", "motor.brushless_dc"], obj={"json": True})
        assert res.exit_code == 0
        data = json.loads(res.output)
        assert isinstance(data, dict)
        assert "id" in data


class TestActuatorSearchJson:
    def test_search_json(self):
        runner = _runner()
        res = runner.invoke(actuator, ["search", "servo"], obj={"json": True})
        assert res.exit_code == 0
        data = json.loads(res.output)
        assert isinstance(data, list)
        if data:
            assert "id" in data[0]
            assert "score" in data[0]


class TestActuatorCategoriesJson:
    def test_categories_json(self):
        runner = _runner()
        res = runner.invoke(actuator, ["categories"], obj={"json": True})
        assert res.exit_code == 0
        data = json.loads(res.output)
        assert isinstance(data, list)
        if data:
            assert isinstance(data[0], str)


# ── Platform commands ─────────────────────────────────────────────────


class TestPlatformListJson:
    def test_list_json(self):
        runner = _runner()
        res = runner.invoke(platform, ["list"], obj={"json": True})
        assert res.exit_code == 0
        data = json.loads(res.output)
        assert isinstance(data, list)


class TestPlatformSearchJson:
    def test_search_json(self):
        runner = _runner()
        res = runner.invoke(platform, ["search", "drone"], obj={"json": True})
        assert res.exit_code == 0
        data = json.loads(res.output)
        assert isinstance(data, list)


# ── Validate commands ─────────────────────────────────────────────────


class TestValidateMissionJson:
    def test_validate_json(self, sample_mission):
        runner = _runner()
        res = runner.invoke(validate, ["mission", sample_mission.id], obj={"json": True})
        assert res.exit_code == 0
        data = json.loads(res.output)
        assert isinstance(data, dict)
