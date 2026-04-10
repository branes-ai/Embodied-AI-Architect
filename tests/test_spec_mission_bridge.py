"""Tests for spec-to-mission bridge (issue #66)."""

import pytest
from click.testing import CliRunner

from embodied_ai_architect.cli.commands.spec import spec
from embodied_ai_architect.mission.store import MissionStore
from embodied_ai_architect.specs.mission_bridge import (
    load_spec_from_mission,
    sync_spec_to_mission,
)


@pytest.fixture()
def working_dir(tmp_path, monkeypatch):
    """Set working directory to tmp_path for both SpecStore and MissionStore."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


class TestSyncSpecToMission:
    def test_creates_mission(self, working_dir):
        sync_spec_to_mission("test-spec", {"name": "test-spec", "description": "A test"})
        store = MissionStore()
        mission = store.load("test-spec")
        assert mission is not None
        assert mission.spec["name"] == "test-spec"

    def test_updates_existing_mission(self, working_dir):
        sync_spec_to_mission("test-spec", {"name": "test-spec", "description": "v1"})
        sync_spec_to_mission("test-spec", {"name": "test-spec", "description": "v2"})
        store = MissionStore()
        mission = store.load("test-spec")
        assert mission.spec["description"] == "v2"

    def test_extracts_power_constraint(self, working_dir):
        sync_spec_to_mission(
            "power-spec",
            {"name": "power-spec", "power": {"power_budget_watts": 5.0}},
        )
        store = MissionStore()
        mission = store.load("power-spec")
        assert mission.constraints.get("max_power_watts") == 5.0

    def test_extracts_latency_constraint(self, working_dir):
        sync_spec_to_mission(
            "latency-spec",
            {"name": "latency-spec", "perception": {"max_latency_ms": 33.0}},
        )
        store = MissionStore()
        mission = store.load("latency-spec")
        assert mission.constraints.get("max_latency_ms") == 33.0


class TestLoadSpecFromMission:
    def test_load_existing(self, working_dir):
        sync_spec_to_mission("my-spec", {"name": "my-spec", "foo": "bar"})
        data = load_spec_from_mission("my-spec")
        assert data is not None
        assert data["name"] == "my-spec"

    def test_load_nonexistent(self, working_dir):
        data = load_spec_from_mission("nonexistent")
        assert data is None


class TestSpecNewSyncsToMission:
    def test_spec_new_creates_mission(self, working_dir):
        runner = CliRunner()
        result = runner.invoke(spec, ["new", "my-drone"], obj={})
        assert result.exit_code == 0

        store = MissionStore()
        mission = store.load("my-drone")
        assert mission is not None
        assert mission.spec.get("name") == "my-drone"

    def test_spec_new_with_template_syncs(self, working_dir):
        runner = CliRunner()
        result = runner.invoke(
            spec, ["new", "my-drone-tmpl", "--template", "drone-perception"], obj={}
        )
        assert result.exit_code == 0

        store = MissionStore()
        mission = store.load("my-drone-tmpl")
        assert mission is not None
        # Template should populate spec fields
        assert mission.spec.get("name") == "my-drone-tmpl"


class TestSpecSetSyncsToMission:
    def test_set_syncs(self, working_dir):
        runner = CliRunner()
        # First create the spec
        runner.invoke(spec, ["new", "my-spec"], obj={})
        # Then set a field
        result = runner.invoke(spec, ["set", "my-spec", "/perception/min_fps", "60"], obj={})
        assert result.exit_code == 0

        store = MissionStore()
        mission = store.load("my-spec")
        assert mission is not None
        perception = mission.spec.get("perception") or {}
        assert perception.get("min_fps") == 60
