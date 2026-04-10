"""Actuator browsing CLI commands (issues #55, #62).

Read-only commands for browsing the actuator registry backed by 80+
actuator YAML definitions with TF-IDF keyword search.
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
      branes actuator list --category motor
      branes actuator show <actuator_id>
      branes actuator search "brushless motor 100W"
      branes actuator categories
    """
    pass


@actuator.command("list")
@click.option(
    "--category",
    type=str,
    default=None,
    help="Filter by category (motor, gripper, locomotion, ...)",
)
@click.pass_context
def actuator_list(ctx, category):
    """List all actuators in the registry."""
    from embodied_ai_architect.actuators import ActuatorRegistry

    registry = ActuatorRegistry()
    actuators = registry.list_actuators(category=category)

    json_output = ctx.obj.get("json", False)
    if json_output:
        click.echo(
            json.dumps(
                [{"id": a.id, "name": a.name, "category": a.category} for a in actuators],
                indent=2,
            )
        )
        return

    if not actuators:
        msg = "No actuators found."
        if category:
            msg += f" (filtered by category={category})"
        console.print(f"[yellow]{msg}[/yellow]")
        return

    table = Table(title="Actuators", show_header=True)
    table.add_column("ID", style="cyan")
    table.add_column("Name")
    table.add_column("Category")
    table.add_column("Type", style="dim")

    for a in actuators:
        table.add_row(a.id, a.name, a.category, a.actuator_type)

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
        ctx.exit(1)
        return

    if json_output:
        click.echo(
            json.dumps(
                {
                    "id": a.id,
                    "name": a.name,
                    "category": a.category,
                    "actuator_type": a.actuator_type,
                    "description": a.description,
                    "attributes": a.attributes,
                },
                indent=2,
            )
        )
        return

    console.print(f"\n[bold cyan]{a.name}[/bold cyan]  ({a.id})")
    console.print(f"  Category:  {a.category}")
    if a.actuator_type:
        console.print(f"  Type:      {a.actuator_type}")
    if a.description:
        console.print(f"  {a.description}")
    if a.aliases:
        console.print(f"  Aliases:   {', '.join(a.aliases)}")
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
                [
                    {
                        "id": r.actuator_id,
                        "name": r.actuator.name,
                        "category": r.actuator.category,
                        "score": r.score,
                    }
                    for r in results
                ],
                indent=2,
            )
        )
        return

    if not results:
        console.print(f"[yellow]No actuators matching '{query}'.[/yellow]")
        return

    table = Table(title=f"Search: {query}", show_header=True)
    table.add_column("ID", style="cyan")
    table.add_column("Name")
    table.add_column("Category")
    table.add_column("Score", style="green")

    for r in results:
        table.add_row(r.actuator_id, r.actuator.name, r.actuator.category, f"{r.score:.3f}")

    console.print(table)


@actuator.command("categories")
@click.pass_context
def actuator_categories(ctx):
    """List actuator categories."""
    from embodied_ai_architect.actuators import ActuatorRegistry

    registry = ActuatorRegistry()
    cats = registry.categories()

    json_output = ctx.obj.get("json", False)
    if json_output:
        click.echo(json.dumps(cats, indent=2))
        return

    console.print("\n[bold]Actuator Categories[/bold]\n")
    for cat in cats:
        console.print(f"  {cat}")
    console.print(f"\n[dim]{len(cats)} categories available[/dim]")
