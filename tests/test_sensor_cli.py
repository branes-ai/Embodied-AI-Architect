"""Tests for the branes sensor CLI command group (issues #54, #59)."""

from click.testing import CliRunner

from embodied_ai_architect.cli.commands.sensor import sensor


class TestSensorList:
    def test_lists_sensors(self):
        runner = CliRunner()
        result = runner.invoke(sensor, ["list"], obj={})
        assert result.exit_code == 0
        assert "Sensors" in result.output

    def test_with_category_filter(self):
        runner = CliRunner()
        result = runner.invoke(sensor, ["list", "--category", "ranging"], obj={})
        assert result.exit_code == 0
        assert "ranging" in result.output

    def test_nonexistent_category_empty(self):
        runner = CliRunner()
        result = runner.invoke(sensor, ["list", "--category", "nonexistent"], obj={})
        assert result.exit_code == 0
        assert "No sensors found" in result.output

    def test_json_output(self):
        runner = CliRunner()
        result = runner.invoke(sensor, ["list"], obj={"json": True})
        assert result.exit_code == 0
        assert '"id"' in result.output
        assert '"name"' in result.output


class TestSensorShow:
    def test_not_found(self):
        runner = CliRunner()
        result = runner.invoke(sensor, ["show", "nonexistent_id"], obj={})
        assert result.exit_code != 0
        assert "not found" in result.output

    def test_show_known_sensor(self):
        runner = CliRunner()
        result = runner.invoke(sensor, ["show", "visual.rgb_camera"], obj={})
        assert result.exit_code == 0
        assert "RGB Camera" in result.output

    def test_show_json(self):
        runner = CliRunner()
        result = runner.invoke(sensor, ["show", "visual.rgb_camera"], obj={"json": True})
        assert result.exit_code == 0
        assert '"id"' in result.output
        assert "visual.rgb_camera" in result.output

    def test_not_found_json(self):
        runner = CliRunner()
        result = runner.invoke(sensor, ["show", "nonexistent_id"], obj={"json": True})
        assert result.exit_code != 0
        assert '"error"' in result.output


class TestSensorSearch:
    def test_search_finds_results(self):
        runner = CliRunner()
        result = runner.invoke(sensor, ["search", "stereo camera"], obj={})
        assert result.exit_code == 0
        assert "stereo" in result.output.lower()

    def test_search_no_results(self):
        runner = CliRunner()
        result = runner.invoke(sensor, ["search", "qqzzxx"], obj={})
        assert result.exit_code == 0
        assert "No sensors matching" in result.output

    def test_search_json(self):
        runner = CliRunner()
        result = runner.invoke(sensor, ["search", "lidar"], obj={"json": True})
        assert result.exit_code == 0
        assert '"id"' in result.output
        assert '"score"' in result.output


class TestSensorCategories:
    def test_lists_modalities(self):
        runner = CliRunner()
        result = runner.invoke(sensor, ["categories"], obj={})
        assert result.exit_code == 0
        assert "visual" in result.output
        assert "ranging" in result.output
        assert "inertial" in result.output
        assert "audio" in result.output

    def test_categories_json(self):
        runner = CliRunner()
        result = runner.invoke(sensor, ["categories"], obj={"json": True})
        assert result.exit_code == 0
        assert '"visual"' in result.output
