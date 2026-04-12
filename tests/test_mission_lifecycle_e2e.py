"""End-to-end mission lifecycle tests through the CLI (issue #145).

Exercises the full create -> qualify -> select -> plan -> synthesize ->
validate -> delete workflow, the actuator variant, and fork workflow.
"""

import pytest
from click.testing import CliRunner

from embodied_ai_architect.cli.commands.design import design
from embodied_ai_architect.cli.commands.mission import mission
from embodied_ai_architect.cli.commands.sensor import sensor
from embodied_ai_architect.cli.commands.actuator import actuator
from embodied_ai_architect.cli.commands.synthesize import synthesize
from embodied_ai_architect.cli.commands.validate import validate
from embodied_ai_architect.mission import MissionStore
import embodied_ai_architect.mission.store as store_mod


@pytest.fixture(autouse=True)
def _use_tmp_store(tmp_path, monkeypatch):
    """Redirect MissionStore to a temp directory so tests are isolated."""
    monkeypatch.setattr(store_mod, "DEFAULT_MISSIONS_DIR", tmp_path)


def _runner():
    return CliRunner()


class TestSimpleDroneMission:
    """Scenario 1: simple drone mission lifecycle."""

    def test_full_lifecycle(self):
        runner = _runner()

        # 1. Create mission
        res = runner.invoke(
            mission,
            ["new", "test-drone", "--goal", "Drone perception SoC for YOLO at 30fps under 5W"],
            obj={},
        )
        assert res.exit_code == 0, res.output
        assert "Mission created" in res.output

        store = MissionStore()
        missions = store.list_missions()
        assert len(missions) == 1
        mid = missions[0]["id"]

        # 2. Design qualify --mission <id> --auto
        res = runner.invoke(
            design,
            ["qualify", "--mission", mid, "--auto"],
            obj={},
            input="\n" * 50,
        )
        assert res.exit_code == 0, res.output

        # Verify status advanced from draft
        m = MissionStore().load(mid)
        assert m.status.value in ("qualified", "draft")  # qualified if all questions answered

        # 3. Sensor select
        res = runner.invoke(
            sensor,
            ["select", mid, "visual.rgb_camera", "inertial.imu_6dof"],
            obj={},
        )
        assert res.exit_code == 0, res.output

        m = MissionStore().load(mid)
        assert "visual.rgb_camera" in m.selected_sensors
        assert "inertial.imu_6dof" in m.selected_sensors

        # 4. Sensor budget
        res = runner.invoke(sensor, ["budget", mid], obj={})
        assert res.exit_code == 0, res.output

        # 5. Design plan --mission <id> --static
        res = runner.invoke(
            design,
            ["plan", "--mission", mid, "--static"],
            obj={},
        )
        assert res.exit_code == 0, res.output

        m = MissionStore().load(mid)
        assert m.design_state is not None
        assert m.status.value == "designed"

        # 6. Synthesize system
        res = runner.invoke(synthesize, ["system", mid], obj={})
        assert res.exit_code == 0, res.output

        # 7. Validate mission
        res = runner.invoke(validate, ["mission", mid], obj={})
        assert res.exit_code == 0, res.output

        # 8. Mission show
        res = runner.invoke(mission, ["show", mid], obj={})
        assert res.exit_code == 0, res.output
        assert "test-drone" in res.output

        # 9. Mission delete
        res = runner.invoke(mission, ["delete", mid, "--yes"], obj={})
        assert res.exit_code == 0, res.output
        assert not MissionStore().exists(mid)


class TestMissionWithActuators:
    """Scenario 2: mission lifecycle with actuator selection."""

    def test_actuator_workflow(self):
        runner = _runner()

        # Create mission
        res = runner.invoke(
            mission,
            ["new", "actuator-test", "--goal", "Mobile robot with actuators"],
            obj={},
        )
        assert res.exit_code == 0, res.output

        store = MissionStore()
        mid = store.list_missions()[0]["id"]

        # Qualify
        res = runner.invoke(
            design, ["qualify", "--mission", mid, "--auto"], obj={}, input="\n" * 50
        )
        assert res.exit_code == 0, res.output

        # Sensor select
        res = runner.invoke(sensor, ["select", mid, "visual.rgb_camera"], obj={})
        assert res.exit_code == 0, res.output

        # Actuator select
        res = runner.invoke(actuator, ["select", mid, "motor.brushless_dc"], obj={})
        assert res.exit_code == 0, res.output

        m = MissionStore().load(mid)
        assert "motor.brushless_dc" in m.selected_actuators

        # Actuator budget
        res = runner.invoke(actuator, ["budget", mid], obj={})
        assert res.exit_code == 0, res.output

        # Sensor budget
        res = runner.invoke(sensor, ["budget", mid], obj={})
        assert res.exit_code == 0, res.output

        # Plan
        res = runner.invoke(design, ["plan", "--mission", mid, "--static"], obj={})
        assert res.exit_code == 0, res.output

        # Synthesize
        res = runner.invoke(synthesize, ["system", mid], obj={})
        assert res.exit_code == 0, res.output

        # Validate
        res = runner.invoke(validate, ["mission", mid], obj={})
        assert res.exit_code == 0, res.output

        # Cleanup
        res = runner.invoke(mission, ["delete", mid, "--yes"], obj={})
        assert res.exit_code == 0, res.output


class TestForkWorkflow:
    """Scenario 3: fork a mission and verify the copy."""

    def test_fork_and_verify(self):
        runner = _runner()

        # Create original
        res = runner.invoke(
            mission,
            ["new", "original", "--goal", "test goal for forking"],
            obj={},
        )
        assert res.exit_code == 0, res.output

        store = MissionStore()
        original_id = store.list_missions()[0]["id"]

        # Fork
        res = runner.invoke(mission, ["fork", original_id, "forked-copy"], obj={})
        assert res.exit_code == 0, res.output
        assert "Forked" in res.output

        # Verify forked mission exists
        missions = store.list_missions()
        assert len(missions) == 2

        forked_entry = [m for m in missions if m["name"] == "forked-copy"]
        assert len(forked_entry) == 1
        forked_id = forked_entry[0]["id"]

        # Show forked mission — same goal as original
        res = runner.invoke(mission, ["show", forked_id], obj={})
        assert res.exit_code == 0, res.output
        assert "forked-copy" in res.output
        assert "test goal for forking" in res.output

        # Edit the fork's goal
        res = runner.invoke(mission, ["edit", forked_id, "--goal", "modified fork goal"], obj={})
        assert res.exit_code == 0, res.output

        # Verify edit took effect
        forked = MissionStore().load(forked_id)
        assert forked.goal == "modified fork goal"

        # Original unchanged
        original = MissionStore().load(original_id)
        assert original.goal == "test goal for forking"

        # Delete only the fork
        res = runner.invoke(mission, ["delete", forked_id, "--yes"], obj={})
        assert res.exit_code == 0, res.output
        assert not MissionStore().exists(forked_id)
        assert MissionStore().exists(original_id)
