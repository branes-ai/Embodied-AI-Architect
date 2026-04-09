"""Actuator browsing CLI commands (issue #55).

Read-only commands for browsing the actuator registry. Mirrors the
sensor CLI from issue #54. Initially backed by an empty registry.
"""

import json

import click
from rich.console import Console
from rich.table import Table

console = Console()


@click.group()
def actuator():
    """Browse the actuator registry.

    \\b
    Examples:
      branes actuator list
      branes actuator list --type servo
      branes actuator show <actuator_id>
      branes actuator search "brushless motor 100W"
      branes actuator categories
    """
    pass


@actuator.command("list")
@click.option("--type", "actuator_type", type=str, default=None, help="Filter by type")
@click.pass_context
def actuator_list(ctx, actuator_type):
    """List all actuators in the registry."""
    from embodied_ai_architect.actuators import ActuatorRegistry

    registry = ActuatorRegistry()
    actuators = registry.list_actuators(actuator_type=actuator_type)

    json_output = ctx.obj.get("json", False)
    if json_output:
        click.echo(
            json.dumps(
                [{"id": a.id, "name": a.name, "type": a.actuator_type} for a in actuators],
                indent=2,
            )
        )
        return

    if not actuators:
        msg = "Actuator registry not yet populated."
        if actuator_type:
            msg += f" (filtered by type={actuator_type})"
        console.print(f"[yellow]{msg}[/yellow]")
        console.print("[dim]Actuator definitions will be added in Phase 2.[/dim]")
        return

    table = Table(title="Actuators", show_header=True)
    table.add_column("ID", style="cyan")
    table.add_column("Name")
    table.add_column("Type")
    table.add_column("Vendor", style="dim")

    for a in actuators:
        table.add_row(a.id, a.name, a.actuator_type, a.vendor)

    console.print(table)


@actuator.command("show")
@click.argument("actuator_id")
@click.pass_context
def actuator_show(ctx, actuator_id):
    """Show details for a specific actuator."""
    from embodied_ai_architect.actuators import ActuatorRegistry

    registry = ActuatorRegistry()
    a = registry.get(actuator_id)

    json_output = ctx.obj.get("json", False)
    if not a:
        if json_output:
            click.echo(json.dumps({"error": f"Actuator '{actuator_id}' not found"}))
        else:
            console.print(f"[red]Actuator '{actuator_id}' not found.[/red]")
            console.print("[dim]Actuator registry not yet populated (Phase 2).[/dim]")
        ctx.exit(1)
        return

    if json_output:
        click.echo(json.dumps(a.model_dump(), indent=2))
        return

    console.print(f"\n[bold cyan]{a.name}[/bold cyan]  ({a.id})")
    console.print(f"  Type:   {a.actuator_type}")
    if a.vendor:
        console.print(f"  Vendor: {a.vendor}")
    if a.description:
        console.print(f"  {a.description}")
    if a.attributes:
        console.print("\n  [bold]Attributes[/bold]")
        for k, v in a.attributes.items():
            console.print(f"    {k}: {v}")


@actuator.command("search")
@click.argument("query")
@click.pass_context
def actuator_search(ctx, query):
    """Search actuators by keyword."""
    from embodied_ai_architect.actuators import ActuatorRegistry

    registry = ActuatorRegistry()
    results = registry.search(query)

    json_output = ctx.obj.get("json", False)
    if json_output:
        click.echo(
            json.dumps(
                [{"id": a.id, "name": a.name, "type": a.actuator_type} for a in results],
                indent=2,
            )
        )
        return

    if not results:
        console.print(f"[yellow]No actuators matching '{query}'.[/yellow]")
        console.print("[dim]Actuator registry not yet populated (Phase 2).[/dim]")
        return

    table = Table(title=f"Search: {query}", show_header=True)
    table.add_column("ID", style="cyan")
    table.add_column("Name")
    table.add_column("Type")

    for a in results:
        table.add_row(a.id, a.name, a.actuator_type)

    console.print(table)


@actuator.command("categories")
@click.pass_context
def actuator_categories(ctx):
    """List actuator type categories."""
    from embodied_ai_architect.actuators import ActuatorRegistry

    registry = ActuatorRegistry()
    cats = registry.categories()

    json_output = ctx.obj.get("json", False)
    if json_output:
        click.echo(json.dumps(cats, indent=2))
        return

    console.print("\n[bold]Actuator Type Categories[/bold]\n")
    for cat in cats:
        console.print(f"  {cat}")
    console.print(f"\n[dim]{len(cats)} categories available[/dim]")
