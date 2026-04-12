"""Tests for sensor/actuator select, compare, budget, fusion, control-rate commands (#63)."""

import json

import pytest

from click.testing import CliRunner

from embodied_ai_architect.cli.commands.actuator import actuator
from embodied_ai_architect.cli.commands.sensor import sensor
from embodied_ai_architect.mission.models import Mission
from embodied_ai_architect.mission.store import MissionStore

pytestmark = pytest.mark.cli


@pytest.fixture()
def mission_with_sensors(tmp_path, monkeypatch):
    """Create a mission with selected sensors in a temp dir."""
    monkeypatch.chdir(tmp_path)
    store = MissionStore()
    m = Mission(
        id="test-mission",
        name="Test Mission",
        goal="Test sensor selection",
        selected_sensors=["visual.rgb_camera", "inertial.imu_6dof", "position.gps_l1"],
    )
    store.save(m)
    return m, store


@pytest.fixture()
def mission_with_actuators(tmp_path, monkeypatch):
    """Create a mission with selected actuators in a temp dir."""
    monkeypatch.chdir(tmp_path)
    store = MissionStore()
    m = Mission(
        id="test-mission",
        name="Test Mission",
        goal="Test actuator selection",
        selected_actuators=["motor.brushless_dc", "motor.servo"],
    )
    store.save(m)
    return m, store


@pytest.fixture()
def empty_mission(tmp_path, monkeypatch):
    """Create a mission with no sensors or actuators in a temp dir."""
    monkeypatch.chdir(tmp_path)
    store = MissionStore()
    m = Mission(id="empty-mission", name="Empty Mission", goal="Empty")
    store.save(m)
    return m, store


class TestSensorCompare:
    def test_compare_two_sensors(self):
        runner = CliRunner()
        result = runner.invoke(
            sensor, ["compare", "visual.rgb_camera", "visual.stereo_camera"], obj={}
        )
        assert result.exit_code == 0
        assert "Sensor Comparison" in result.output
        assert "RGB Camera" in result.output

    def test_compare_json(self):
        runner = CliRunner()
        result = runner.invoke(
            sensor,
            ["compare", "visual.rgb_camera", "visual.stereo_camera"],
            obj={"json": True},
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data) == 2
        assert data[0]["id"] == "visual.rgb_camera"

    def test_compare_not_found(self):
        runner = CliRunner()
        result = runner.invoke(sensor, ["compare", "visual.rgb_camera", "nonexistent"], obj={})
        assert result.exit_code != 0
        assert "not found" in result.output

    def test_compare_needs_two(self):
        runner = CliRunner()
        result = runner.invoke(sensor, ["compare", "visual.rgb_camera"], obj={})
        assert result.exit_code != 0
        assert "at least 2" in result.output


class TestSensorSelect:
    def test_select_sensor(self, empty_mission):
        mission, store = empty_mission
        runner = CliRunner()
        result = runner.invoke(sensor, ["select", mission.id, "visual.rgb_camera"], obj={})
        assert result.exit_code == 0
        assert "Added 1 sensor" in result.output
        reloaded = store.load(mission.id)
        assert "visual.rgb_camera" in reloaded.selected_sensors

    def test_select_nonexistent_mission(self, empty_mission):
        # monkeypatch.chdir is active from empty_mission fixture
        runner = CliRunner()
        result = runner.invoke(sensor, ["select", "nonexistent", "visual.rgb_camera"], obj={})
        assert result.exit_code != 0
        assert "not found" in result.output

    def test_select_nonexistent_sensor(self, empty_mission):
        mission, _ = empty_mission
        runner = CliRunner()
        result = runner.invoke(sensor, ["select", mission.id, "nonexistent.sensor"], obj={})
        assert result.exit_code == 0
        assert "not found in registry" in result.output


class TestSensorBudget:
    def test_budget_with_sensors(self, mission_with_sensors):
        mission, _ = mission_with_sensors
        runner = CliRunner()
        result = runner.invoke(sensor, ["budget", mission.id], obj={})
        assert result.exit_code == 0
        assert "TOTAL" in result.output

    def test_budget_json(self, mission_with_sensors):
        mission, _ = mission_with_sensors
        runner = CliRunner()
        result = runner.invoke(sensor, ["budget", mission.id], obj={"json": True})
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "total_power_watts" in data
        assert "total_weight_grams" in data

    def test_budget_no_sensors(self, empty_mission):
        mission, _ = empty_mission
        runner = CliRunner()
        result = runner.invoke(sensor, ["budget", mission.id], obj={})
        assert result.exit_code == 0
        assert "No sensors selected" in result.output


class TestSensorFusion:
    def test_fusion_vio(self, mission_with_sensors):
        mission, _ = mission_with_sensors
        runner = CliRunner()
        result = runner.invoke(sensor, ["fusion", mission.id], obj={})
        assert result.exit_code == 0
        assert "Visual-Inertial" in result.output or "INS/GNSS" in result.output

    def test_fusion_json(self, mission_with_sensors):
        mission, _ = mission_with_sensors
        runner = CliRunner()
        result = runner.invoke(sensor, ["fusion", mission.id], obj={"json": True})
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "recommendations" in data
        assert len(data["recommendations"]) >= 1

    def test_fusion_no_sensors(self, empty_mission):
        mission, _ = empty_mission
        runner = CliRunner()
        result = runner.invoke(sensor, ["fusion", mission.id], obj={})
        assert result.exit_code == 0
        assert "No sensors selected" in result.output


class TestActuatorCompare:
    def test_compare_two_actuators(self):
        runner = CliRunner()
        result = runner.invoke(actuator, ["compare", "motor.brushless_dc", "motor.servo"], obj={})
        assert result.exit_code == 0
        assert "Actuator Comparison" in result.output

    def test_compare_json(self):
        runner = CliRunner()
        result = runner.invoke(
            actuator,
            ["compare", "motor.brushless_dc", "motor.servo"],
            obj={"json": True},
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data) == 2


class TestActuatorSelect:
    def test_select_actuator(self, empty_mission):
        mission, store = empty_mission
        runner = CliRunner()
        result = runner.invoke(actuator, ["select", mission.id, "motor.brushless_dc"], obj={})
        assert result.exit_code == 0
        assert "Added 1 actuator" in result.output
        reloaded = store.load(mission.id)
        assert "motor.brushless_dc" in reloaded.selected_actuators

    def test_select_nonexistent_mission(self, empty_mission):
        runner = CliRunner()
        result = runner.invoke(actuator, ["select", "nonexistent", "motor.brushless_dc"], obj={})
        assert result.exit_code != 0
        assert "not found" in result.output

    def test_select_nonexistent_actuator(self, empty_mission):
        mission, _ = empty_mission
        runner = CliRunner()
        result = runner.invoke(actuator, ["select", mission.id, "nonexistent.actuator"], obj={})
        assert result.exit_code == 0
        assert "not found in registry" in result.output


class TestActuatorBudget:
    def test_budget_with_actuators(self, mission_with_actuators):
        mission, _ = mission_with_actuators
        runner = CliRunner()
        result = runner.invoke(actuator, ["budget", mission.id], obj={})
        assert result.exit_code == 0
        assert "TOTAL" in result.output

    def test_budget_json(self, mission_with_actuators):
        mission, _ = mission_with_actuators
        runner = CliRunner()
        result = runner.invoke(actuator, ["budget", mission.id], obj={"json": True})
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "total_power_watts" in data
        assert "total_weight_grams" in data
        assert "total_cost_usd" in data

    def test_budget_no_actuators(self, empty_mission):
        mission, _ = empty_mission
        runner = CliRunner()
        result = runner.invoke(actuator, ["budget", mission.id], obj={})
        assert result.exit_code == 0
        assert "No actuators selected" in result.output


class TestActuatorControlRate:
    def test_control_rate(self):
        runner = CliRunner()
        result = runner.invoke(actuator, ["control-rate", "motor.servo"], obj={})
        assert result.exit_code == 0
        assert "Hz" in result.output or "not specified" in result.output

    def test_control_rate_json(self):
        runner = CliRunner()
        result = runner.invoke(actuator, ["control-rate", "motor.servo"], obj={"json": True})
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "control_rate_hz" in data

    def test_control_rate_not_found(self):
        runner = CliRunner()
        result = runner.invoke(actuator, ["control-rate", "nonexistent"], obj={})
        assert result.exit_code != 0
        assert "not found" in result.output
