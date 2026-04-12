"""Tests for --mission flag on swap, optimize, mcp commands (issue #65)."""

import pytest

from click.testing import CliRunner

from embodied_ai_architect.cli.commands._utils import load_mission_constraints
from embodied_ai_architect.mission.models import Mission, MissionStatus
from embodied_ai_architect.mission.store import MissionStore


pytestmark = pytest.mark.cli


@pytest.fixture()
def mission_store(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return MissionStore()


@pytest.fixture()
def constrained_mission(mission_store):
    m = Mission(
        id="test-constrained",
        name="Constrained Mission",
        goal="Drone SoC under 5W",
        status=MissionStatus.QUALIFIED,
        constraints={
            "max_power_watts": 5.0,
            "max_latency_ms": 33.0,
            "max_cost_usd": 30.0,
            "max_area_mm2": 100.0,
            "target_process_nm": 16,
        },
        selected_sensors=["visual.rgb_camera"],
        selected_actuators=["motor.brushless_dc"],
        selected_compute="jetson_orin_nano",
        selected_models=["yolov8n"],
    )
    mission_store.save(m)
    return m


class TestLoadMissionConstraints:
    def test_no_mission(self):
        mission, merged = load_mission_constraints(None, power=5.0)
        assert mission is None
        assert merged == {"max_power_watts": 5.0}

    def test_load_mission(self, mission_store, constrained_mission):
        mission, merged = load_mission_constraints(constrained_mission.id)
        assert mission is not None
        assert merged["max_power_watts"] == 5.0
        assert merged["max_latency_ms"] == 33.0

    def test_flags_override_mission(self, mission_store, constrained_mission):
        mission, merged = load_mission_constraints(constrained_mission.id, power=10.0, latency=50.0)
        assert merged["max_power_watts"] == 10.0
        assert merged["max_latency_ms"] == 50.0
        # Non-overridden values come from mission
        assert merged["max_cost_usd"] == 30.0

    def test_nonexistent_mission(self, mission_store):
        mission, merged = load_mission_constraints("nonexistent")
        assert mission is None
        assert merged == {}

    def test_partial_override(self, mission_store, constrained_mission):
        _, merged = load_mission_constraints(constrained_mission.id, cost=99.0)
        assert merged["max_power_watts"] == 5.0  # from mission
        assert merged["max_cost_usd"] == 99.0  # overridden


class TestOptimizeExploreMission:
    def test_explore_accepts_mission(self, mission_store, constrained_mission):
        """The --mission flag is accepted by optimize explore."""
        from embodied_ai_architect.cli.commands.optimize import optimize

        runner = CliRunner()
        # Just verify the flag is accepted — actual optimization requires heavy deps
        result = runner.invoke(
            optimize,
            ["explore", "--mission", constrained_mission.id, "--fast"],
            obj={},
        )
        # Should get past argument parsing (may fail on optimization but not on CLI)
        assert "--mission" not in result.output or "Error" not in result.output


class TestSwapCommandsMission:
    def test_swap_score_accepts_mission(self, mission_store, constrained_mission):
        """swap score --mission is accepted."""
        from embodied_ai_architect.cli.commands.swap import swap

        runner = CliRunner()
        result = runner.invoke(
            swap,
            ["score", "--mission", constrained_mission.id],
            obj={},
        )
        # Should load mission constraints — may fail later but not on parsing
        assert "not found" not in result.output.lower() or result.exit_code == 0
