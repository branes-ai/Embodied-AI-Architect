"""Sensor browsing CLI commands (issue #54).

Read-only commands for browsing the sensor registry. Initially backed
by an empty registry; Phase 2 will populate it.
"""

import json

import click
from rich.console import Console
from rich.table import Table

console = Console()


@click.group()
def sensor():
    """Browse the sensor registry.

    \\b
    Examples:
      branes sensor list
      branes sensor list --modality lidar
      branes sensor show <sensor_id>
      branes sensor search "stereo camera 30fps"
      branes sensor categories
    """
    pass


@sensor.command("list")
@click.option("--modality", type=str, default=None, help="Filter by modality (camera, lidar, ...)")
@click.pass_context
def sensor_list(ctx, modality):
    """List all sensors in the registry."""
    from embodied_ai_architect.sensors import SensorRegistry

    registry = SensorRegistry()
    sensors = registry.list_sensors(modality=modality)

    json_output = ctx.obj.get("json", False)
    if json_output:
        click.echo(
            json.dumps(
                [{"id": s.id, "name": s.name, "modality": s.modality} for s in sensors], indent=2
            )
        )
        return

    if not sensors:
        msg = "Sensor registry not yet populated."
        if modality:
            msg += f" (filtered by modality={modality})"
        console.print(f"[yellow]{msg}[/yellow]")
        console.print("[dim]Sensor definitions will be added in Phase 2.[/dim]")
        return

    table = Table(title="Sensors", show_header=True)
    table.add_column("ID", style="cyan")
    table.add_column("Name")
    table.add_column("Modality")
    table.add_column("Vendor", style="dim")

    for s in sensors:
        table.add_row(s.id, s.name, s.modality, s.vendor)

    console.print(table)


@sensor.command("show")
@click.argument("sensor_id")
@click.pass_context
def sensor_show(ctx, sensor_id):
    """Show details for a specific sensor."""
    from embodied_ai_architect.sensors import SensorRegistry

    registry = SensorRegistry()
    s = registry.get(sensor_id)

    if not s:
        console.print(f"[red]Sensor '{sensor_id}' not found.[/red]")
        console.print("[dim]Sensor registry not yet populated (Phase 2).[/dim]")
        ctx.exit(1)
        return

    json_output = ctx.obj.get("json", False)
    if json_output:
        click.echo(
            json.dumps(
                {
                    "id": s.id,
                    "name": s.name,
                    "modality": s.modality,
                    "vendor": s.vendor,
                    "description": s.description,
                    "attributes": s.attributes,
                },
                indent=2,
            )
        )
        return

    console.print(f"\n[bold cyan]{s.name}[/bold cyan]  ({s.id})")
    console.print(f"  Modality: {s.modality}")
    if s.vendor:
        console.print(f"  Vendor:   {s.vendor}")
    if s.description:
        console.print(f"  {s.description}")
    if s.attributes:
        console.print("\n  [bold]Attributes[/bold]")
        for k, v in s.attributes.items():
            console.print(f"    {k}: {v}")


@sensor.command("search")
@click.argument("query")
@click.pass_context
def sensor_search(ctx, query):
    """Search sensors by keyword."""
    from embodied_ai_architect.sensors import SensorRegistry

    registry = SensorRegistry()
    results = registry.search(query)

    json_output = ctx.obj.get("json", False)
    if json_output:
        click.echo(
            json.dumps(
                [{"id": s.id, "name": s.name, "modality": s.modality} for s in results], indent=2
            )
        )
        return

    if not results:
        console.print(f"[yellow]No sensors matching '{query}'.[/yellow]")
        console.print("[dim]Sensor registry not yet populated (Phase 2).[/dim]")
        return

    table = Table(title=f"Search: {query}", show_header=True)
    table.add_column("ID", style="cyan")
    table.add_column("Name")
    table.add_column("Modality")

    for s in results:
        table.add_row(s.id, s.name, s.modality)

    console.print(table)


@sensor.command("categories")
@click.pass_context
def sensor_categories(ctx):
    """List sensor modality categories."""
    from embodied_ai_architect.sensors import SensorRegistry

    registry = SensorRegistry()
    cats = registry.categories()

    json_output = ctx.obj.get("json", False)
    if json_output:
        click.echo(json.dumps(cats, indent=2))
        return

    console.print("\n[bold]Sensor Modality Categories[/bold]\n")
    for cat in cats:
        console.print(f"  {cat}")
    console.print(f"\n[dim]{len(cats)} categories available[/dim]")
