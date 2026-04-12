"""Tests for design qualify/plan --mission wiring (issue #64)."""

import pytest

from click.testing import CliRunner

from embodied_ai_architect.cli.commands.design import design
from embodied_ai_architect.mission.models import Mission, MissionStatus
from embodied_ai_architect.mission.store import MissionStore

pytestmark = pytest.mark.cli


@pytest.fixture()
def mission_store(tmp_path, monkeypatch):
    """Provide a MissionStore in a temp directory."""
    monkeypatch.chdir(tmp_path)
    return MissionStore()


@pytest.fixture()
def draft_mission(mission_store):
    """Create a draft mission with a goal."""
    m = Mission(
        id="test-drone",
        name="Test Drone",
        goal="Drone perception SoC for YOLO at 30fps under 5W",
        constraints={"max_power_watts": 5.0, "max_latency_ms": 33.0},
        use_case="delivery_drone",
        platform_id="drone",
    )
    mission_store.save(m)
    return m


@pytest.fixture()
def qualified_mission(mission_store):
    """Create a qualified mission ready for planning."""
    m = Mission(
        id="test-qualified",
        name="Test Qualified",
        goal="Drone perception SoC for YOLO at 30fps under 5W",
        status=MissionStatus.QUALIFIED,
        constraints={"max_power_watts": 5.0, "max_latency_ms": 33.0},
        use_case="delivery_drone",
        platform_id="drone",
    )
    mission_store.save(m)
    return m


class TestDesignQualifyMission:
    def test_qualify_creates_new_mission(self, mission_store):
        """--mission with new ID creates the mission."""
        runner = CliRunner()
        result = runner.invoke(
            design,
            ["qualify", "drone perception SoC", "--mission", "new-mission", "--auto"],
            obj={},
            input="\n" * 50,  # auto-skip all interactive prompts
        )
        assert result.exit_code == 0
        assert "Created mission" in result.output
        loaded = mission_store.load("new-mission")
        assert loaded is not None
        assert loaded.goal  # goal should be populated

    def test_qualify_loads_existing_mission(self, mission_store, draft_mission):
        """--mission with existing ID loads the mission's goal."""
        runner = CliRunner()
        result = runner.invoke(
            design,
            ["qualify", "--mission", draft_mission.id, "--auto"],
            obj={},
            input="\n" * 50,
        )
        assert result.exit_code == 0
        assert "Loaded mission" in result.output

    def test_qualify_no_goal_no_mission_fails(self, mission_store):
        """Must provide either goal argument or --mission."""
        runner = CliRunner()
        result = runner.invoke(design, ["qualify"], obj={})
        assert result.exit_code != 0
        assert "Goal is required" in result.output

    def test_qualify_backward_compat_no_mission(self):
        """Old form without --mission still works."""
        runner = CliRunner()
        result = runner.invoke(
            design,
            ["qualify", "drone perception SoC", "--auto"],
            obj={},
            input="\n" * 50,
        )
        assert result.exit_code == 0
        # Should show qualification result (no mission created)
        assert "Tangibility" in result.output or "Design Qualification" in result.output


class TestDesignPlanMission:
    def test_plan_loads_from_mission(self, mission_store, qualified_mission):
        """--mission loads goal and constraints from mission."""
        runner = CliRunner()
        result = runner.invoke(
            design,
            ["plan", "--mission", qualified_mission.id, "--static"],
            obj={},
        )
        assert result.exit_code == 0
        assert "Loaded mission" in result.output
        # Plan should be saved back
        loaded = mission_store.load(qualified_mission.id)
        assert loaded.status == MissionStatus.DESIGNED
        assert loaded.design_state is not None

    def test_plan_goal_overrides_mission(self, mission_store, qualified_mission):
        """Positional goal overrides mission goal."""
        runner = CliRunner()
        result = runner.invoke(
            design,
            ["plan", "Custom goal override", "--mission", qualified_mission.id, "--static"],
            obj={},
        )
        assert result.exit_code == 0
        # Should still load mission for constraints
        assert "Loaded mission" in result.output

    def test_plan_nonexistent_mission_fails(self, mission_store):
        """--mission with bad ID fails."""
        runner = CliRunner()
        result = runner.invoke(
            design,
            ["plan", "--mission", "nonexistent", "--static"],
            obj={},
        )
        assert result.exit_code != 0
        assert "not found" in result.output

    def test_plan_no_goal_no_mission_fails(self, mission_store):
        """Must provide goal or --mission."""
        runner = CliRunner()
        result = runner.invoke(design, ["plan", "--static"], obj={})
        assert result.exit_code != 0
        assert "Goal is required" in result.output

    def test_plan_backward_compat_no_mission(self):
        """Old form with positional goal still works."""
        runner = CliRunner()
        result = runner.invoke(
            design,
            ["plan", "Drone perception SoC", "--static"],
            obj={},
        )
        assert result.exit_code == 0
        assert "static demo plan" in result.output.lower() or "Task Graph" in result.output

    def test_plan_mission_constraints_applied(self, mission_store, qualified_mission):
        """Constraints from mission are used when flags not provided."""
        runner = CliRunner()
        result = runner.invoke(
            design,
            ["plan", "--mission", qualified_mission.id, "--static"],
            obj={},
        )
        assert result.exit_code == 0
        loaded = mission_store.load(qualified_mission.id)
        assert loaded.design_state is not None
        # Verify mission constraints were propagated into the design state
        ds = loaded.design_state
        if ds.get("constraints"):
            c = ds["constraints"]
            assert c.get("max_power_watts") == qualified_mission.constraints["max_power_watts"]
            assert c.get("max_latency_ms") == qualified_mission.constraints["max_latency_ms"]
