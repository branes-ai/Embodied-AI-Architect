"""Tests for lifecycle command groups select/synthesize/analyze-system (#67)."""

import json

import pytest
from click.testing import CliRunner

from embodied_ai_architect.cli.commands.analyze_group import analyze_lifecycle
from embodied_ai_architect.cli.commands.select import select
from embodied_ai_architect.cli.commands.synthesize import synthesize
from embodied_ai_architect.mission.models import Mission
from embodied_ai_architect.mission.store import MissionStore


@pytest.fixture()
def mission(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = MissionStore()
    m = Mission(
        id="test-mission",
        name="Test Mission",
        goal="Drone perception",
        selected_sensors=["visual.rgb_camera", "inertial.imu_6dof"],
        selected_actuators=["motor.brushless_dc"],
        selected_compute="jetson_orin_nano",
        selected_models=["yolov8n"],
        constraints={"max_power_watts": 5.0},
    )
    store.save(m)
    return m


class TestSelectGroup:
    def test_select_sensor(self, mission):
        runner = CliRunner()
        result = runner.invoke(select, ["sensor", mission.id, "visual.stereo_camera"], obj={})
        assert result.exit_code == 0

    def test_select_actuator(self, mission):
        runner = CliRunner()
        result = runner.invoke(select, ["actuator", mission.id, "motor.servo"], obj={})
        assert result.exit_code == 0

    def test_select_compute_stub(self):
        runner = CliRunner()
        result = runner.invoke(select, ["compute", "any-mission"], obj={})
        assert result.exit_code == 0
        assert "Coming in a future release" in result.output

    def test_select_model_stub(self):
        runner = CliRunner()
        result = runner.invoke(select, ["model", "any-mission"], obj={})
        assert result.exit_code == 0
        assert "Coming in a future release" in result.output


class TestSynthesizeGroup:
    def test_synthesize_system(self, mission):
        runner = CliRunner()
        result = runner.invoke(synthesize, ["system", mission.id], obj={})
        assert result.exit_code == 0
        assert "System Synthesis" in result.output
        assert "visual.rgb_camera" in result.output

    def test_synthesize_system_json(self, mission):
        runner = CliRunner()
        result = runner.invoke(synthesize, ["system", mission.id], obj={"json": True})
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "sensors" in data
        assert "visual.rgb_camera" in data["sensors"]

    def test_synthesize_architecture(self, mission):
        runner = CliRunner()
        result = runner.invoke(synthesize, ["architecture", mission.id], obj={})
        assert result.exit_code == 0
        assert "mermaid" in result.output.lower() or "graph TD" in result.output

    def test_synthesize_bom_stub(self):
        runner = CliRunner()
        result = runner.invoke(synthesize, ["bom", "any"], obj={})
        assert "Coming in a future release" in result.output

    def test_synthesize_nonexistent_mission(self, mission):
        runner = CliRunner()
        result = runner.invoke(synthesize, ["system", "nonexistent"], obj={})
        assert result.exit_code != 0
        assert "not found" in result.output


class TestAnalyzeSystemGroup:
    def test_power_stub(self):
        runner = CliRunner()
        result = runner.invoke(analyze_lifecycle, ["power", "any"], obj={})
        assert "Coming in a future release" in result.output

    def test_latency_stub(self):
        runner = CliRunner()
        result = runner.invoke(analyze_lifecycle, ["latency", "any"], obj={})
        assert "Coming in a future release" in result.output

    def test_safety_stub(self):
        runner = CliRunner()
        result = runner.invoke(analyze_lifecycle, ["safety", "any"], obj={})
        assert "Coming in a future release" in result.output

    def test_help_shown_without_subcommand(self):
        runner = CliRunner()
        result = runner.invoke(analyze_lifecycle, [], obj={})
        assert result.exit_code == 0
        assert "power" in result.output
        assert "latency" in result.output
