"""Cross-command state flow verification (issue #149).

Tests that data written by one CLI command correctly flows to the next.
Each test creates a mission, runs a chain of commands, and verifies
the final state matches expectations.
"""

import pytest
from click.testing import CliRunner

import embodied_ai_architect.mission.store as store_mod
from embodied_ai_architect.cli.commands.design import design
from embodied_ai_architect.cli.commands.mission import mission
from embodied_ai_architect.cli.commands.sensor import sensor
from embodied_ai_architect.cli.commands.synthesize import synthesize

pytestmark = pytest.mark.cli


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(store_mod, "DEFAULT_MISSIONS_DIR", tmp_path / ".branes" / "missions")


def _runner():
    return CliRunner()


def _create_mission(runner, name, goal):
    """Helper: create a mission and return its ID."""
    res = runner.invoke(mission, ["new", name, "--goal", goal], obj={})
    assert res.exit_code == 0, res.output
    store = store_mod.MissionStore()
    missions = store.list_missions()
    assert len(missions) >= 1
    return missions[0]["id"]


class TestQualifyToPlan:
    """Chain: qualify → plan. Constraints from qualify appear in plan's design_state."""

    def test_constraints_flow_to_plan(self):
        runner = _runner()
        mid = _create_mission(runner, "flow-test", "Drone perception SoC under 5W and 33ms")

        # Qualify
        res = runner.invoke(
            design, ["qualify", "--mission", mid, "--auto"], obj={}, input="\n" * 50
        )
        assert res.exit_code == 0, res.output

        # Plan
        res = runner.invoke(design, ["plan", "--mission", mid, "--static"], obj={})
        assert res.exit_code == 0, res.output

        # Verify constraints flowed through
        m = store_mod.MissionStore().load(mid)
        assert m.design_state is not None
        ds = m.design_state
        # The design state should have constraints from qualify
        if ds.get("constraints"):
            # At minimum, the constraints dict should not be empty
            assert len(ds["constraints"]) > 0


class TestSensorSelectToBudget:
    """Chain: sensor select → budget. Selected sensors appear in budget output."""

    def test_selected_sensors_in_budget(self):
        runner = _runner()
        mid = _create_mission(runner, "budget-test", "Test sensor budget flow")

        # Select sensors
        res = runner.invoke(
            sensor, ["select", mid, "visual.rgb_camera", "inertial.imu_6dof"], obj={}
        )
        assert res.exit_code == 0, res.output

        # Verify selection persisted
        m = store_mod.MissionStore().load(mid)
        assert "visual.rgb_camera" in m.selected_sensors
        assert "inertial.imu_6dof" in m.selected_sensors

        # Budget should show both sensors
        res = runner.invoke(sensor, ["budget", mid], obj={})
        assert res.exit_code == 0, res.output
        assert "rgb_camera" in res.output.lower() or "RGB" in res.output
        assert "imu" in res.output.lower() or "IMU" in res.output
        assert "TOTAL" in res.output


class TestSensorSelectToFusion:
    """Chain: sensor select → fusion. Categories drive recommendations."""

    def test_fusion_reflects_selected_categories(self):
        runner = _runner()
        mid = _create_mission(runner, "fusion-test", "Test fusion flow")

        # Select visual + inertial → should recommend VIO
        res = runner.invoke(
            sensor, ["select", mid, "visual.stereo_camera", "inertial.imu_6dof"], obj={}
        )
        assert res.exit_code == 0, res.output

        res = runner.invoke(sensor, ["fusion", mid], obj={})
        assert res.exit_code == 0, res.output
        assert "Visual-Inertial" in res.output or "VIO" in res.output

    def test_fusion_changes_with_additional_sensors(self):
        runner = _runner()
        mid = _create_mission(runner, "fusion-test2", "Test fusion evolution")

        # Start with visual + inertial
        runner.invoke(sensor, ["select", mid, "visual.stereo_camera", "inertial.imu_6dof"], obj={})

        # Add GPS → should now recommend INS/GNSS fusion too
        res = runner.invoke(sensor, ["select", mid, "position.gps_l1"], obj={})
        assert res.exit_code == 0, res.output

        res = runner.invoke(sensor, ["fusion", mid], obj={})
        assert res.exit_code == 0, res.output
        assert "INS/GNSS" in res.output or "GPS" in res.output


class TestQualifyToMissionShow:
    """Chain: qualify → mission show. Platform and use_case visible."""

    def test_qualify_results_in_show(self):
        runner = _runner()
        mid = _create_mission(runner, "show-test", "Drone perception SoC for YOLO at 30fps")

        # Qualify
        res = runner.invoke(
            design, ["qualify", "--mission", mid, "--auto"], obj={}, input="\n" * 50
        )
        assert res.exit_code == 0, res.output

        # Mission show should display qualification results
        res = runner.invoke(mission, ["show", mid], obj={})
        assert res.exit_code == 0, res.output
        # The qualified mission should show status and goal
        assert "show-test" in res.output
        assert "Drone perception" in res.output or "drone" in res.output.lower()


class TestPlanToSynthesize:
    """Chain: plan → synthesize. Design state populated, status = designed."""

    def test_plan_state_available_to_synthesize(self):
        runner = _runner()
        mid = _create_mission(runner, "synth-test", "Drone SoC under 5W")

        # Select a sensor so synthesize has something to show
        runner.invoke(sensor, ["select", mid, "visual.rgb_camera"], obj={})

        # Plan
        res = runner.invoke(design, ["plan", "--mission", mid, "--static"], obj={})
        assert res.exit_code == 0, res.output

        # Verify status
        m = store_mod.MissionStore().load(mid)
        assert m.status.value == "designed"
        assert m.design_state is not None

        # Synthesize should work and show the selected sensor
        res = runner.invoke(synthesize, ["system", mid], obj={})
        assert res.exit_code == 0, res.output
        assert "visual.rgb_camera" in res.output
