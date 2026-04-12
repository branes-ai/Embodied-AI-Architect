"""Tests for grouped CLI help output (issue #68)."""

from click.testing import CliRunner
import pytest

pytestmark = pytest.mark.cli


def _get_cli():
    """Get the CLI with all commands registered."""
    from embodied_ai_architect.cli import cli
    from embodied_ai_architect.cli.commands import (
        analyze,
        analyze_group,
        backends,
        benchmark,
        chat,
        config,
        demo,
        deploy,
        design,
        model,
        optimize,
        pipeline,
        report,
        secrets,
        session,
        spec,
        swap,
        workflow,
        zoo,
    )
    from embodied_ai_architect.cli.commands import actuator as actuator_cmd
    from embodied_ai_architect.cli.commands import api as api_cmd
    from embodied_ai_architect.cli.commands import codebase
    from embodied_ai_architect.cli.commands import mcp as mcp_cmd
    from embodied_ai_architect.cli.commands import mission as mission_cmd
    from embodied_ai_architect.cli.commands import platform as platform_cmd
    from embodied_ai_architect.cli.commands import select as select_cmd
    from embodied_ai_architect.cli.commands import sensor as sensor_cmd
    from embodied_ai_architect.cli.commands import synthesize as synthesize_cmd
    from embodied_ai_architect.cli.commands import testbench
    from embodied_ai_architect.cli.commands import validate as validate_cmd

    # Always register — add_command overwrites if already present
    cli.add_command(workflow.workflow)
    cli.add_command(analyze.analyze)
    cli.add_command(benchmark.benchmark)
    cli.add_command(report.report)
    cli.add_command(config.config)
    cli.add_command(backends.backends)
    cli.add_command(secrets.secrets)
    cli.add_command(chat.chat)
    cli.add_command(pipeline.pipeline)
    cli.add_command(model.model)
    cli.add_command(zoo.zoo)
    cli.add_command(design.design)
    cli.add_command(testbench.testbench)
    cli.add_command(deploy.deploy)
    cli.add_command(demo.demo)
    cli.add_command(codebase.codebase)
    cli.add_command(optimize.optimize)
    cli.add_command(spec.spec)
    cli.add_command(swap.swap)
    cli.add_command(mcp_cmd.mcp)
    cli.add_command(session.session)
    cli.add_command(api_cmd.api)
    cli.add_command(platform_cmd.platform)
    cli.add_command(mission_cmd.mission)
    cli.add_command(sensor_cmd.sensor)
    cli.add_command(actuator_cmd.actuator)
    cli.add_command(validate_cmd.validate)
    cli.add_command(select_cmd.select)
    cli.add_command(synthesize_cmd.synthesize)
    cli.add_command(analyze_group.analyze_lifecycle, "analyze-system")

    return cli


class TestGroupedHelp:
    def test_help_shows_lifecycle_section(self):
        runner = CliRunner()
        result = runner.invoke(_get_cli(), ["--help"])
        assert result.exit_code == 0
        assert "Lifecycle:" in result.output
        assert "mission" in result.output
        assert "design" in result.output

    def test_help_shows_catalog_section(self):
        runner = CliRunner()
        result = runner.invoke(_get_cli(), ["--help"])
        assert "Catalog:" in result.output
        assert "sensor" in result.output
        assert "actuator" in result.output

    def test_help_shows_infrastructure_section(self):
        runner = CliRunner()
        result = runner.invoke(_get_cli(), ["--help"])
        assert "Infrastructure:" in result.output

    def test_help_shows_analysis_section(self):
        runner = CliRunner()
        result = runner.invoke(_get_cli(), ["--help"])
        assert "Analysis & Benchmarking:" in result.output

    def test_help_shows_deployment_section(self):
        runner = CliRunner()
        result = runner.invoke(_get_cli(), ["--help"])
        assert "Deployment:" in result.output

    def test_all_commands_appear(self):
        runner = CliRunner()
        result = runner.invoke(_get_cli(), ["--help"])
        for cmd in ["mission", "sensor", "actuator", "optimize", "api", "deploy"]:
            assert cmd in result.output, f"Command '{cmd}' missing from help"
