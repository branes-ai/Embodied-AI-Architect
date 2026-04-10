"""Lifecycle command group: select components for a mission (issue #67).

Delegates to existing sensor/actuator/mcp/zoo commands with mission context.
"""

import click
from rich.console import Console

console = Console()


@click.group()
def select():
    """Select components for a mission.

    \\b
    Examples:
      branes select sensor <mission_id> <sensor_id>
      branes select actuator <mission_id> <actuator_id>
      branes select compute <mission_id>
      branes select model <mission_id>
    """
    pass


@select.command("sensor")
@click.argument("mission_id")
@click.argument("sensor_ids", nargs=-1, required=True)
@click.pass_context
def select_sensor(ctx, mission_id, sensor_ids):
    """Select sensors for a mission."""
    from embodied_ai_architect.cli.commands.sensor import sensor_select

    ctx.invoke(sensor_select, mission_id=mission_id, sensor_ids=sensor_ids)


@select.command("actuator")
@click.argument("mission_id")
@click.argument("actuator_ids", nargs=-1, required=True)
@click.pass_context
def select_actuator(ctx, mission_id, actuator_ids):
    """Select actuators for a mission."""
    from embodied_ai_architect.cli.commands.actuator import actuator_select

    ctx.invoke(actuator_select, mission_id=mission_id, actuator_ids=actuator_ids)


@select.command("compute")
@click.argument("mission_id")
@click.pass_context
def select_compute(ctx, mission_id):
    """Select compute hardware for a mission.

    Lists available hardware and lets you pick one for the mission.
    """
    console.print("[yellow]Coming in a future release.[/yellow]")
    console.print("[dim]For now, use: branes mcp hardware --query <query>[/dim]")


@select.command("model")
@click.argument("mission_id")
@click.pass_context
def select_model(ctx, mission_id):
    """Select ML models for a mission.

    Recommends models based on mission perception tasks.
    """
    console.print("[yellow]Coming in a future release.[/yellow]")
    console.print("[dim]For now, use: branes zoo list[/dim]")
