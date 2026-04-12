"""Tests for the branes synthesize CLI command group."""

import json

import pytest
from click.testing import CliRunner

from embodied_ai_architect.cli.commands.synthesize import synthesize
from embodied_ai_architect.mission import Mission, MissionStore
import embodied_ai_architect.mission.store as store_mod

pytestmark = pytest.mark.cli


@pytest.fixture(autouse=True)
def _use_tmp_store(tmp_path, monkeypatch):
    """Isolate MissionStore to a temp directory."""
    monkeypatch.setattr(store_mod, "DEFAULT_MISSIONS_DIR", tmp_path)


@pytest.fixture()
def mission_id():
    """Create a mission with sensors and return its ID."""
    store = MissionStore()
    m = Mission(
        name="Test Drone",
        goal="30fps at <5W",
        selected_sensors=["visual.rgb_camera"],
        selected_actuators=["motor.servo"],
    )
    store.save(m)
    return m.id


@pytest.fixture()
def bare_mission_id():
    """Create a mission with no selections."""
    store = MissionStore()
    m = Mission(name="Bare Mission", goal="Test")
    store.save(m)
    return m.id


def _make_runner():
    return CliRunner()


class TestSynthesizeSystem:
    def test_shows_summary(self, mission_id):
        runner = _make_runner()
        result = runner.invoke(synthesize, ["system", mission_id], obj={})
        assert result.exit_code == 0, result.output
        assert "Test Drone" in result.output
        assert "visual.rgb_camera" in result.output

    def test_json_output(self, mission_id):
        runner = _make_runner()
        result = runner.invoke(synthesize, ["system", mission_id], obj={"json": True})
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert data["mission"] == "Test Drone"
        assert "visual.rgb_camera" in data["sensors"]

    def test_not_found(self):
        runner = _make_runner()
        result = runner.invoke(synthesize, ["system", "nonexistent"], obj={})
        assert result.exit_code != 0

    def test_missing_selections_warning(self, bare_mission_id):
        runner = _make_runner()
        result = runner.invoke(synthesize, ["system", bare_mission_id], obj={})
        assert result.exit_code == 0
        assert "Missing selections" in result.output


class TestSynthesizeArchitecture:
    def test_shows_mermaid(self, mission_id):
        runner = _make_runner()
        result = runner.invoke(synthesize, ["architecture", mission_id], obj={})
        assert result.exit_code == 0
        assert "graph TD" in result.output or "mermaid" in result.output

    def test_not_found(self):
        runner = _make_runner()
        result = runner.invoke(synthesize, ["architecture", "nonexistent"], obj={})
        assert result.exit_code != 0


class TestSynthesizeBom:
    def test_shows_coming_soon(self, mission_id):
        runner = _make_runner()
        result = runner.invoke(synthesize, ["bom", mission_id], obj={})
        assert result.exit_code == 0
        assert "Coming in a future release" in result.output
