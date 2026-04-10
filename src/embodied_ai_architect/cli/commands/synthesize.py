"""Lifecycle command group: synthesize system from mission (issue #67).

Commands for composing a full system design from selected components.
"""

import json

import click
from rich.console import Console

console = Console()


@click.group()
def synthesize():
    """Synthesize a system design from mission components.

    \\b
    Examples:
      branes synthesize system <mission_id>
      branes synthesize architecture <mission_id>
      branes synthesize bom <mission_id>
    """
    pass


@synthesize.command("system")
@click.argument("mission_id")
@click.pass_context
def synthesize_system(ctx, mission_id):
    """Compose a full system from mission's selected components.

    Reads selected sensors, actuators, compute, and models from the
    mission and generates a coherent system specification.
    """
    from embodied_ai_architect.mission.store import MissionStore

    json_output = ctx.obj.get("json", False)
    store = MissionStore()
    mission = store.load(mission_id)
    if not mission:
        if json_output:
            click.echo(json.dumps({"error": f"Mission '{mission_id}' not found"}))
        else:
            console.print(f"[red]Mission '{mission_id}' not found.[/red]")
        ctx.exit(1)
        return
    summary = {
        "mission": mission.name,
        "sensors": mission.selected_sensors,
        "actuators": mission.selected_actuators,
        "compute": mission.selected_compute,
        "models": mission.selected_models,
        "constraints": mission.constraints,
    }

    if json_output:
        click.echo(json.dumps(summary, indent=2))
        return

    console.print(f"\n[bold]System Synthesis — {mission.name}[/bold]\n")
    console.print(f"  Sensors:    {', '.join(mission.selected_sensors) or 'none'}")
    console.print(f"  Actuators:  {', '.join(mission.selected_actuators) or 'none'}")
    console.print(f"  Compute:    {mission.selected_compute or 'none'}")
    console.print(f"  Models:     {', '.join(mission.selected_models) or 'none'}")
    if mission.constraints:
        console.print(f"  Constraints: {mission.constraints}")
    console.print()

    missing = []
    if not mission.selected_sensors:
        missing.append("sensors")
    if not mission.selected_actuators:
        missing.append("actuators")
    if not mission.selected_compute:
        missing.append("compute")

    if missing:
        console.print(f"[yellow]Missing selections: {', '.join(missing)}[/yellow]")
        console.print("[dim]Use 'branes select <type> <mission>' to add components.[/dim]")
    else:
        console.print("[green]All component types selected. Ready for analysis.[/green]")


@synthesize.command("architecture")
@click.argument("mission_id")
@click.pass_context
def synthesize_architecture(ctx, mission_id):
    """Generate architecture diagram for a mission's system.

    Produces a Mermaid-compatible diagram showing component relationships.
    """
    from embodied_ai_architect.mission.store import MissionStore

    json_output = ctx.obj.get("json", False)
    store = MissionStore()
    mission = store.load(mission_id)
    if not mission:
        if json_output:
            click.echo(json.dumps({"error": f"Mission '{mission_id}' not found"}))
        else:
            console.print(f"[red]Mission '{mission_id}' not found.[/red]")
        ctx.exit(1)
        return

    # Generate a simple Mermaid diagram from selected components
    lines = ["graph TD"]
    lines.append("    SoC[SoC Design]")

    for sid in mission.selected_sensors:
        safe = sid.replace(".", "_")
        lines.append(f"    {safe}[{sid}] --> SoC")

    if mission.selected_compute:
        lines.append(f"    SoC --> Compute[{mission.selected_compute}]")

    for mid in mission.selected_models:
        safe = mid.replace(".", "_").replace("-", "_")
        lines.append(f"    Compute --> {safe}[{mid}]")

    for aid in mission.selected_actuators:
        safe = aid.replace(".", "_")
        lines.append(f"    SoC --> {safe}[{aid}]")

    mermaid = "\n".join(lines)

    if json_output:
        click.echo(json.dumps({"mermaid": mermaid}))
        return

    console.print(f"\n[bold]Architecture — {mission.name}[/bold]\n")
    console.print("```mermaid")
    console.print(mermaid)
    console.print("```")
    console.print()


@synthesize.command("bom")
@click.argument("mission_id")
@click.pass_context
def synthesize_bom(ctx, mission_id):
    """Generate bill of materials for a mission."""
    console.print("[yellow]Coming in a future release.[/yellow]")
    console.print("[dim]For now, use: branes swap bom --mission <mission>[/dim]")


@synthesize.command("cabling")
@click.argument("mission_id")
@click.pass_context
def synthesize_cabling(ctx, mission_id):
    """Generate cabling and interconnect plan."""
    console.print("[yellow]Coming in a future release.[/yellow]")


@synthesize.command("thermal")
@click.argument("mission_id")
@click.pass_context
def synthesize_thermal(ctx, mission_id):
    """Generate thermal management plan."""
    console.print("[yellow]Coming in a future release.[/yellow]")
