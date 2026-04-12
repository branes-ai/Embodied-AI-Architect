"""Tests for the branes actuator CLI command group (issues #55, #62)."""

from click.testing import CliRunner

from embodied_ai_architect.cli.commands.actuator import actuator
import pytest


pytestmark = pytest.mark.cli


class TestActuatorList:
    def test_lists_actuators(self):
        runner = CliRunner()
        result = runner.invoke(actuator, ["list"], obj={})
        assert result.exit_code == 0
        assert "Actuators" in result.output

    def test_with_category_filter(self):
        runner = CliRunner()
        result = runner.invoke(actuator, ["list", "--category", "motor"], obj={})
        assert result.exit_code == 0
        assert "motor" in result.output

    def test_nonexistent_category_empty(self):
        runner = CliRunner()
        result = runner.invoke(actuator, ["list", "--category", "nonexistent"], obj={})
        assert result.exit_code == 0
        assert "No actuators found" in result.output

    def test_json_output(self):
        runner = CliRunner()
        result = runner.invoke(actuator, ["list"], obj={"json": True})
        assert result.exit_code == 0
        assert '"id"' in result.output
        assert '"name"' in result.output


class TestActuatorShow:
    def test_not_found(self):
        runner = CliRunner()
        result = runner.invoke(actuator, ["show", "nonexistent_id"], obj={})
        assert result.exit_code != 0
        assert "not found" in result.output

    def test_show_known_actuator(self):
        runner = CliRunner()
        result = runner.invoke(actuator, ["show", "motor.brushless_dc"], obj={})
        assert result.exit_code == 0
        assert "Brushless DC Motor" in result.output

    def test_show_json(self):
        runner = CliRunner()
        result = runner.invoke(actuator, ["show", "motor.brushless_dc"], obj={"json": True})
        assert result.exit_code == 0
        assert '"id"' in result.output
        assert "motor.brushless_dc" in result.output

    def test_not_found_json(self):
        runner = CliRunner()
        result = runner.invoke(actuator, ["show", "nonexistent_id"], obj={"json": True})
        assert result.exit_code != 0
        assert '"error"' in result.output


class TestActuatorSearch:
    def test_search_finds_results(self):
        runner = CliRunner()
        result = runner.invoke(actuator, ["search", "brushless motor"], obj={})
        assert result.exit_code == 0
        assert "brushless" in result.output.lower()

    def test_search_no_results(self):
        runner = CliRunner()
        result = runner.invoke(actuator, ["search", "qqzzxx"], obj={})
        assert result.exit_code == 0
        assert "No actuators matching" in result.output

    def test_search_json(self):
        runner = CliRunner()
        result = runner.invoke(actuator, ["search", "gripper"], obj={"json": True})
        assert result.exit_code == 0
        assert '"id"' in result.output
        assert '"score"' in result.output


class TestActuatorCategories:
    def test_lists_categories(self):
        runner = CliRunner()
        result = runner.invoke(actuator, ["categories"], obj={})
        assert result.exit_code == 0
        assert "motor" in result.output
        assert "gripper" in result.output
        assert "locomotion" in result.output

    def test_categories_json(self):
        runner = CliRunner()
        result = runner.invoke(actuator, ["categories"], obj={"json": True})
        assert result.exit_code == 0
        assert '"motor"' in result.output
