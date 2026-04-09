"""Embodied AI Architect CLI.

Human-friendly command-line interface for the Embodied AI Architect system.
"""

import click
from rich.console import Console
from rich.panel import Panel

from embodied_ai_architect import __version__

# Initialize rich console for beautiful output
console = Console()


@click.group()
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
def cli(ctx, verbose, json, quiet):
    """Branes Embodied AI Platform - Design environment for Embodied AI systems.

    \b
    Examples:
      branes workflow run my_model.pt
      branes analyze my_model.pt
      branes benchmark my_model.pt --backend kubernetes
      branes report view --latest
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


def main():
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

    # Run CLI
    cli(obj={})


if __name__ == "__main__":
    main()
