"""Tests for the branes design CLI command group."""

import pytest
from click.testing import CliRunner

from embodied_ai_architect.cli.commands.design import design
from embodied_ai_architect.mission import Mission, MissionStore
import embodied_ai_architect.mission.store as store_mod

pytestmark = pytest.mark.cli


@pytest.fixture(autouse=True)
def _use_tmp_store(tmp_path, monkeypatch):
    """Isolate MissionStore to a temp directory."""
    monkeypatch.setattr(store_mod, "DEFAULT_MISSIONS_DIR", tmp_path)
    monkeypatch.chdir(tmp_path)


def _make_runner():
    return CliRunner()


class TestDesignQualify:
    def test_qualify_with_goal_auto(self):
        runner = _make_runner()
        result = runner.invoke(
            design,
            ["qualify", "drone SoC", "--auto"],
            obj={},
            input="\n" * 50,
        )
        assert result.exit_code == 0, result.output

    def test_qualify_with_mission_auto(self):
        store = MissionStore()
        m = Mission(name="Test Drone", goal="30fps at <5W")
        store.save(m)

        runner = _make_runner()
        result = runner.invoke(
            design,
            ["qualify", "--mission", m.id, "--auto"],
            obj={},
            input="\n" * 50,
        )
        assert result.exit_code == 0, result.output


class TestDesignPlan:
    def test_plan_with_goal_static(self):
        runner = _make_runner()
        result = runner.invoke(
            design,
            ["plan", "drone SoC", "--static"],
            obj={},
        )
        assert result.exit_code == 0, result.output

    def test_plan_with_mission_static(self):
        store = MissionStore()
        m = Mission(name="Test Drone", goal="30fps at <5W")
        store.save(m)

        runner = _make_runner()
        result = runner.invoke(
            design,
            ["plan", "--mission", m.id, "--static"],
            obj={},
        )
        assert result.exit_code == 0, result.output
        assert "Loaded mission" in result.output

    def test_plan_no_goal_no_mission(self):
        runner = _make_runner()
        result = runner.invoke(
            design,
            ["plan"],
            obj={},
        )
        assert result.exit_code != 0

    def test_plan_static_shows_task_graph(self):
        runner = _make_runner()
        result = runner.invoke(
            design,
            ["plan", "drone SoC", "--static"],
            obj={},
        )
        assert result.exit_code == 0
        # Static plan includes known agent names
        assert "workload" in result.output.lower() or "plan" in result.output.lower()
