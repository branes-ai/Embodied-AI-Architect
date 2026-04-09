"""Tests for the branes mission CLI command group (issue #53)."""

import pytest
from click.testing import CliRunner

from embodied_ai_architect.cli.commands.mission import mission
from embodied_ai_architect.mission import Mission, MissionStore
import embodied_ai_architect.mission.store as store_mod


@pytest.fixture(autouse=True)
def _use_tmp_store(tmp_path, monkeypatch):
    """Point MissionStore's default dir to a temp path so CLI commands
    that instantiate MissionStore() without arguments use the test dir."""
    monkeypatch.setattr(store_mod, "DEFAULT_MISSIONS_DIR", tmp_path)


def _make_runner():
    return CliRunner()


class TestMissionNew:
    def test_creates_mission(self):

        runner = _make_runner()
        result = runner.invoke(mission, ["new", "Test Mission", "--goal", "Test goal"], obj={})
        assert result.exit_code == 0
        assert "Mission created" in result.output
        assert "Test Mission" in result.output

        store = MissionStore()
        missions = store.list_missions()
        assert len(missions) == 1
        assert missions[0]["name"] == "Test Mission"

    def test_new_with_platform(self):

        runner = _make_runner()
        result = runner.invoke(
            mission,
            ["new", "Drone", "--goal", "30fps", "--platform", "aerial.delivery"],
            obj={},
        )
        assert result.exit_code == 0

        store = MissionStore()
        m = store.load_latest()
        assert m.platform_id == "aerial.delivery"


class TestMissionList:
    def test_empty(self):

        runner = _make_runner()
        result = runner.invoke(mission, ["list"], obj={})
        assert result.exit_code == 0
        assert "No missions found" in result.output

    def test_with_missions(self):
        store = MissionStore()
        store.save(Mission(name="Alpha", goal="Goal A"))
        store.save(Mission(name="Beta", goal="Goal B"))

        runner = _make_runner()
        result = runner.invoke(mission, ["list"], obj={})
        assert result.exit_code == 0
        assert "Alpha" in result.output
        assert "Beta" in result.output


class TestMissionShow:
    def test_shows_details(self):
        store = MissionStore()
        m = Mission(
            name="Drone SoC",
            goal="30fps at <5W",
            platform_id="aerial.delivery",
            constraints={"max_power_watts": 5.0},
        )
        store.save(m)

        runner = _make_runner()
        result = runner.invoke(mission, ["show", m.id], obj={})
        assert result.exit_code == 0
        assert "Drone SoC" in result.output
        assert "30fps at <5W" in result.output
        assert "aerial.delivery" in result.output
        assert "max_power_watts" in result.output

    def test_not_found(self):

        runner = _make_runner()
        result = runner.invoke(mission, ["show", "nonexistent"], obj={})
        assert result.exit_code != 0


class TestMissionEdit:
    def test_updates_fields(self):
        store = MissionStore()
        m = Mission(name="Original")
        store.save(m)

        runner = _make_runner()
        result = runner.invoke(
            mission,
            ["edit", m.id, "--name", "Renamed", "--status", "qualified"],
            obj={},
        )
        assert result.exit_code == 0
        assert "Renamed" in result.output

        updated = MissionStore().load(m.id)
        assert updated.name == "Renamed"
        assert updated.status.value == "qualified"


class TestMissionDelete:
    def test_deletes(self):
        store = MissionStore()
        m = Mission(name="To Delete")
        store.save(m)

        runner = _make_runner()
        result = runner.invoke(mission, ["delete", m.id, "--yes"], obj={})
        assert result.exit_code == 0
        assert "Deleted" in result.output
        assert not MissionStore().exists(m.id)


class TestMissionFork:
    def test_forks_mission(self):
        store = MissionStore()
        original = Mission(
            name="Original",
            goal="Test goal",
            constraints={"max_power_watts": 5.0},
        )
        store.save(original)

        runner = _make_runner()
        result = runner.invoke(mission, ["fork", original.id, "What-If Variant"], obj={})
        assert result.exit_code == 0
        assert "Forked" in result.output
        assert "What-If Variant" in result.output

        # Two missions in the store
        missions = MissionStore().list_missions()
        assert len(missions) == 2

        # The fork has a different ID but same constraints
        forked = [m for m in missions if m["name"] == "What-If Variant"][0]
        loaded = MissionStore().load(forked["id"])
        assert loaded.id != original.id
        assert loaded.constraints == original.constraints
        assert loaded.goal == original.goal
