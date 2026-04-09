"""Tests for the branes sensor CLI command group (issue #54)."""

from click.testing import CliRunner

from embodied_ai_architect.cli.commands.sensor import sensor
from embodied_ai_architect.sensors.registry import SensorRegistry


class TestSensorList:
    def test_empty_registry(self):
        runner = CliRunner()
        result = runner.invoke(sensor, ["list"], obj={})
        assert result.exit_code == 0
        assert "not yet populated" in result.output

    def test_with_modality_filter(self):
        runner = CliRunner()
        result = runner.invoke(sensor, ["list", "--modality", "ranging"], obj={})
        assert result.exit_code == 0
        assert "not yet populated" in result.output
        assert "ranging" in result.output


class TestSensorShow:
    def test_not_found(self):
        runner = CliRunner()
        result = runner.invoke(sensor, ["show", "nonexistent_id"], obj={})
        assert result.exit_code != 0
        assert "not found" in result.output


class TestSensorSearch:
    def test_empty_results(self):
        runner = CliRunner()
        result = runner.invoke(sensor, ["search", "stereo camera"], obj={})
        assert result.exit_code == 0
        assert "No sensors matching" in result.output


class TestSensorCategories:
    def test_lists_modalities(self):
        runner = CliRunner()
        result = runner.invoke(sensor, ["categories"], obj={})
        assert result.exit_code == 0
        assert "visual" in result.output
        assert "ranging" in result.output
        assert "inertial" in result.output
        assert "audio" in result.output


class TestSensorRegistry:
    def test_empty_by_default(self):
        registry = SensorRegistry()
        assert registry.list_sensors() == []

    def test_categories_not_empty(self):
        registry = SensorRegistry()
        cats = registry.categories()
        assert len(cats) == 8  # matches taxonomy.yaml top-level categories
        assert "visual" in cats
        assert "ranging" in cats

    def test_search_empty(self):
        registry = SensorRegistry()
        assert registry.search("anything") == []

    def test_get_returns_none(self):
        registry = SensorRegistry()
        assert registry.get("nonexistent") is None
