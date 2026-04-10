"""Tests for the branes actuator CLI command group (issue #55)."""

from click.testing import CliRunner

from embodied_ai_architect.cli.commands.actuator import actuator
from embodied_ai_architect.actuators.registry import ActuatorRegistry


class TestActuatorList:
    def test_empty_registry(self):
        result = CliRunner().invoke(actuator, ["list"], obj={})
        assert result.exit_code == 0
        assert "not yet populated" in result.output

    def test_with_type_filter(self):
        result = CliRunner().invoke(actuator, ["list", "--type", "servo"], obj={})
        assert result.exit_code == 0
        assert "not yet populated" in result.output
        assert "servo" in result.output


class TestActuatorShow:
    def test_not_found(self):
        result = CliRunner().invoke(actuator, ["show", "nonexistent"], obj={})
        assert result.exit_code != 0
        assert "not found" in result.output


class TestActuatorSearch:
    def test_empty_results(self):
        result = CliRunner().invoke(actuator, ["search", "brushless motor"], obj={})
        assert result.exit_code == 0
        assert "No actuators matching" in result.output


class TestActuatorCategories:
    def test_lists_types(self):
        result = CliRunner().invoke(actuator, ["categories"], obj={})
        assert result.exit_code == 0
        assert "motor" in result.output
        assert "gripper" in result.output
        assert "locomotion" in result.output


class TestActuatorRegistry:
    def test_empty_by_default(self):
        assert ActuatorRegistry().list_actuators() == []

    def test_categories_not_empty(self):
        cats = ActuatorRegistry().categories()
        assert len(cats) == 8  # matches taxonomy.yaml top-level categories
        assert "motor" in cats
        assert "gripper" in cats

    def test_search_empty(self):
        assert ActuatorRegistry().search("anything") == []

    def test_get_returns_none(self):
        assert ActuatorRegistry().get("nonexistent") is None

    def test_search_top_k_zero(self):
        assert ActuatorRegistry().search("test", top_k=0) == []
