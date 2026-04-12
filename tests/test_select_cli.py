"""Tests for the branes select CLI command group."""

import pytest
from click.testing import CliRunner

from embodied_ai_architect.cli.commands.select import select
from embodied_ai_architect.mission import Mission, MissionStore
import embodied_ai_architect.mission.store as store_mod

pytestmark = pytest.mark.cli


@pytest.fixture(autouse=True)
def _use_tmp_store(tmp_path, monkeypatch):
    """Isolate MissionStore to a temp directory."""
    monkeypatch.setattr(store_mod, "DEFAULT_MISSIONS_DIR", tmp_path)


@pytest.fixture()
def mission_id():
    """Create a mission and return its ID."""
    store = MissionStore()
    m = Mission(name="Test Mission", goal="Test goal")
    store.save(m)
    return m.id


def _make_runner():
    return CliRunner()


class TestSelectSensor:
    def test_adds_sensor(self, mission_id):
        runner = _make_runner()
        result = runner.invoke(
            select,
            ["sensor", mission_id, "visual.rgb_camera"],
            obj={},
        )
        assert result.exit_code == 0, result.output

        # Verify sensor was added to mission
        store = MissionStore()
        m = store.load(mission_id)
        assert "visual.rgb_camera" in m.selected_sensors


class TestSelectActuator:
    def test_adds_actuator(self, mission_id):
        runner = _make_runner()
        result = runner.invoke(
            select,
            ["actuator", mission_id, "motor.servo"],
            obj={},
        )
        assert result.exit_code == 0, result.output

        store = MissionStore()
        m = store.load(mission_id)
        assert "motor.servo" in m.selected_actuators


class TestSelectCompute:
    def test_shows_coming_soon(self, mission_id):
        runner = _make_runner()
        result = runner.invoke(
            select,
            ["compute", mission_id],
            obj={},
        )
        assert result.exit_code == 0
        assert "Coming in a future release" in result.output


class TestSelectModel:
    def test_shows_coming_soon(self, mission_id):
        runner = _make_runner()
        result = runner.invoke(
            select,
            ["model", mission_id],
            obj={},
        )
        assert result.exit_code == 0
        assert "Coming in a future release" in result.output
