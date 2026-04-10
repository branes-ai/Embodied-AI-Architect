"""Embodied AI Architect CLI.

Human-friendly command-line interface for the Embodied AI Architect system.
"""

from __future__ import annotations

from collections import OrderedDict

import click
from rich.console import Console
from rich.panel import Panel

from embodied_ai_architect import __version__

# Initialize rich console for beautiful output
console = Console()

# ── Command groups for help display ──────────────────────────────────
# Each key is a section header; each value is a list of command names.
# Commands not listed here appear under "Other".

COMMAND_SECTIONS: OrderedDict[str, list[str]] = OrderedDict(
    [
        (
            "Lifecycle",
            [
                "mission",
                "design",
                "select",
                "synthesize",
                "analyze-system",
                "optimize",
                "validate",
                "report",
            ],
        ),
        (
            "Catalog",
            ["platform", "sensor", "actuator", "model", "zoo"],
        ),
        (
            "Analysis & Benchmarking",
            ["analyze", "benchmark", "swap", "mcp", "codebase", "testbench"],
        ),
        (
            "Infrastructure",
            ["api", "session", "spec", "config", "secrets", "backends", "chat"],
        ),
        (
            "Deployment",
            ["deploy", "pipeline", "workflow", "demo"],
        ),
    ]
)


class GroupedGroup(click.Group):
    """Click Group that displays commands organized by section."""

    def format_commands(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        commands = []
        for subcommand in self.list_commands(ctx):
            cmd = self.commands.get(subcommand)
            if cmd is None or cmd.hidden:
                continue
            help_text = cmd.get_short_help_str(limit=formatter.width)
            commands.append((subcommand, help_text))

        if not commands:
            return

        # Build a lookup: command_name → (name, help_text)
        cmd_lookup: dict[str, tuple[str, str]] = {name: (name, help_) for name, help_ in commands}
        shown: set[str] = set()

        for section, cmd_names in COMMAND_SECTIONS.items():
            section_cmds = []
            for name in cmd_names:
                if name in cmd_lookup:
                    section_cmds.append(cmd_lookup[name])
                    shown.add(name)

            if section_cmds:
                with formatter.section(section):
                    formatter.write_dl(section_cmds)

        # Any commands not in a section go under "Other"
        other = [(n, h) for n, h in commands if n not in shown]
        if other:
            with formatter.section("Other"):
                formatter.write_dl(other)


@click.group(cls=GroupedGroup)
@click.version_option(version=__version__, prog_name="branes")
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Increase verbosity (-v, -vv, or --debug for max verbosity)",
)
@click.option("--json", is_flag=True, help="Output in JSON format")
@click.option("--quiet", is_flag=True, help="Minimal output")
@click.pass_context
def cli(ctx: click.Context, verbose: int, json: bool, quiet: bool) -> None:
    """Branes Embodied AI Platform - Design environment for Embodied AI systems.

    \b
    Examples:
      branes mission new "my-drone" --goal "Drone perception SoC"
      branes design qualify --mission my-drone --auto
      branes design plan --mission my-drone --static
      branes select sensor my-drone visual.rgb_camera
      branes sensor search "stereo camera for VIO"
    """
    # Store settings in context for subcommands
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["json"] = json
    ctx.obj["quiet"] = quiet

    # Show welcome banner (unless quiet)
    if not quiet and not json and ctx.invoked_subcommand:
        console.print(
            Panel.fit(
                "[bold cyan]Branes Embodied AI Platform[/bold cyan]\n" f"Version {__version__}",
                border_style="cyan",
            )
        )


def main() -> None:
    """Entry point for the CLI."""
    # Import subcommands
    from embodied_ai_architect.cli.commands import workflow
    from embodied_ai_architect.cli.commands import analyze
    from embodied_ai_architect.cli.commands import benchmark
    from embodied_ai_architect.cli.commands import report
    from embodied_ai_architect.cli.commands import config
    from embodied_ai_architect.cli.commands import backends
    from embodied_ai_architect.cli.commands import secrets
    from embodied_ai_architect.cli.commands import chat
    from embodied_ai_architect.cli.commands import pipeline
    from embodied_ai_architect.cli.commands import model
    from embodied_ai_architect.cli.commands import zoo
    from embodied_ai_architect.cli.commands import design
    from embodied_ai_architect.cli.commands import testbench
    from embodied_ai_architect.cli.commands import deploy
    from embodied_ai_architect.cli.commands import demo
    from embodied_ai_architect.cli.commands import codebase
    from embodied_ai_architect.cli.commands import optimize
    from embodied_ai_architect.cli.commands import spec
    from embodied_ai_architect.cli.commands import swap
    from embodied_ai_architect.cli.commands import mcp as mcp_cmd
    from embodied_ai_architect.cli.commands import session
    from embodied_ai_architect.cli.commands import api as api_cmd
    from embodied_ai_architect.cli.commands import platform as platform_cmd
    from embodied_ai_architect.cli.commands import mission as mission_cmd
    from embodied_ai_architect.cli.commands import sensor as sensor_cmd
    from embodied_ai_architect.cli.commands import actuator as actuator_cmd
    from embodied_ai_architect.cli.commands import validate as validate_cmd
    from embodied_ai_architect.cli.commands import select as select_cmd
    from embodied_ai_architect.cli.commands import synthesize as synthesize_cmd
    from embodied_ai_architect.cli.commands import analyze_group

    # Register command groups
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

    # Run CLI
    cli(obj={})


if __name__ == "__main__":
    main()
