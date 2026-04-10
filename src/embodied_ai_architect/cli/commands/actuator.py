"""Actuator browsing and selection CLI commands (issues #55, #62, #63).

Commands for browsing the actuator registry, selecting actuators for
missions, comparing actuator specs, and computing budgets.
"""

import json

import click
from rich.console import Console
from rich.table import Table

console = Console()


def _get_attr_typical(attributes: dict, key: str) -> float | None:
    """Extract the typical value from a min/max/typical attribute dict."""
    val = attributes.get(key)
    if isinstance(val, dict):
        return val.get("typical")
    if isinstance(val, (int, float)):
        return float(val)
    return None


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
      branes actuator select <mission_id> <actuator_id>
      branes actuator compare <id1> <id2> ...
      branes actuator budget <mission_id>
      branes actuator control-rate <actuator_id>
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


# ── New commands from issue #63 ──────────────────────────────────────


@actuator.command("select")
@click.argument("mission_id")
@click.argument("actuator_ids", nargs=-1, required=True)
@click.pass_context
def actuator_select(ctx, mission_id, actuator_ids):
    """Add actuators to a mission's selected actuators list."""
    from embodied_ai_architect.actuators import ActuatorRegistry
    from embodied_ai_architect.mission.store import MissionStore

    store = MissionStore()
    mission = store.load(mission_id)
    if not mission:
        console.print(f"[red]Mission '{mission_id}' not found.[/red]")
        ctx.exit(1)
        return

    registry = ActuatorRegistry()
    added = []
    for aid in actuator_ids:
        a = registry.get(aid)
        if not a:
            console.print(f"[yellow]Actuator '{aid}' not found in registry, skipping.[/yellow]")
            continue
        if aid not in mission.selected_actuators:
            mission.selected_actuators.append(aid)
            added.append(aid)
        else:
            console.print(f"[dim]Actuator '{aid}' already selected.[/dim]")

    if added:
        store.save(mission)
        json_output = ctx.obj.get("json", False)
        if json_output:
            click.echo(json.dumps({"added": added, "total": mission.selected_actuators}))
        else:
            console.print(
                f"[green]Added {len(added)} actuator(s) to mission '{mission.name}':[/green]"
            )
            for aid in added:
                a = registry.get(aid)
                console.print(f"  + {aid} ({a.name if a else '?'})")
            console.print(f"\n[dim]Total selected: {len(mission.selected_actuators)}[/dim]")


@actuator.command("compare")
@click.argument("actuator_ids", nargs=-1, required=True)
@click.pass_context
def actuator_compare(ctx, actuator_ids):
    """Compare actuators side by side."""
    from embodied_ai_architect.actuators import ActuatorRegistry

    if len(actuator_ids) < 2:
        console.print("[red]Need at least 2 actuator IDs to compare.[/red]")
        ctx.exit(1)
        return

    registry = ActuatorRegistry()
    actuators = []
    for aid in actuator_ids:
        a = registry.get(aid)
        if not a:
            console.print(f"[red]Actuator '{aid}' not found.[/red]")
            ctx.exit(1)
            return
        actuators.append(a)

    json_output = ctx.obj.get("json", False)
    if json_output:
        click.echo(
            json.dumps(
                [
                    {
                        "id": a.id,
                        "name": a.name,
                        "category": a.category,
                        "attributes": a.attributes,
                    }
                    for a in actuators
                ],
                indent=2,
            )
        )
        return

    # Collect all attribute keys across all actuators
    all_keys = []
    for a in actuators:
        for k in a.attributes:
            if k not in all_keys:
                all_keys.append(k)

    table = Table(title="Actuator Comparison", show_header=True)
    table.add_column("Attribute", style="bold")
    for a in actuators:
        table.add_column(a.name, style="cyan")

    table.add_row("ID", *[a.id for a in actuators])
    table.add_row("Category", *[a.category for a in actuators])
    table.add_row("Type", *[a.actuator_type for a in actuators])

    for key in all_keys:
        vals = []
        for a in actuators:
            v = a.attributes.get(key)
            if isinstance(v, dict):
                typical = v.get("typical", "—")
                unit = v.get("unit", "")
                vals.append(f"{typical} {unit}".strip())
            elif v is not None:
                vals.append(str(v))
            else:
                vals.append("—")
        table.add_row(key, *vals)

    console.print(table)


@actuator.command("budget")
@click.argument("mission_id")
@click.pass_context
def actuator_budget(ctx, mission_id):
    """Show power, weight, and cost budget for mission actuators."""
    from embodied_ai_architect.actuators import ActuatorRegistry
    from embodied_ai_architect.mission.store import MissionStore

    store = MissionStore()
    mission = store.load(mission_id)
    if not mission:
        console.print(f"[red]Mission '{mission_id}' not found.[/red]")
        ctx.exit(1)
        return

    if not mission.selected_actuators:
        console.print("[yellow]No actuators selected for this mission.[/yellow]")
        return

    registry = ActuatorRegistry()
    total_power = 0.0
    total_weight = 0.0
    total_cost = 0.0
    rows = []

    for aid in mission.selected_actuators:
        a = registry.get(aid)
        if not a:
            rows.append((aid, "NOT FOUND", "—", "—", "—"))
            continue
        pw = _get_attr_typical(a.attributes, "continuous_power_watts") or 0.0
        wt = _get_attr_typical(a.attributes, "weight_grams") or 0.0
        cost = _get_attr_typical(a.attributes, "cost_usd") or 0.0
        total_power += pw
        total_weight += wt
        total_cost += cost
        rows.append((aid, a.name, f"{pw:.1f}W", f"{wt:.0f}g", f"${cost:.0f}"))

    json_output = ctx.obj.get("json", False)
    if json_output:
        click.echo(
            json.dumps(
                {
                    "actuators": [{"id": r[0], "name": r[1]} for r in rows],
                    "total_power_watts": round(total_power, 2),
                    "total_weight_grams": round(total_weight, 1),
                    "total_cost_usd": round(total_cost, 2),
                }
            )
        )
        return

    table = Table(title=f"Actuator Budget — {mission.name}", show_header=True)
    table.add_column("ID", style="cyan")
    table.add_column("Name")
    table.add_column("Power", justify="right")
    table.add_column("Weight", justify="right")
    table.add_column("Cost", justify="right")

    for row in rows:
        table.add_row(*row)

    table.add_section()
    table.add_row(
        "[bold]TOTAL[/bold]",
        "",
        f"[bold]{total_power:.1f}W[/bold]",
        f"[bold]{total_weight:.0f}g[/bold]",
        f"[bold]${total_cost:.0f}[/bold]",
    )

    console.print(table)


@actuator.command("control-rate")
@click.argument("actuator_id")
@click.pass_context
def actuator_control_rate(ctx, actuator_id):
    """Show control rate requirements for an actuator."""
    from embodied_ai_architect.actuators import ActuatorRegistry

    registry = ActuatorRegistry()
    a = registry.get(actuator_id)

    if not a:
        console.print(f"[red]Actuator '{actuator_id}' not found.[/red]")
        ctx.exit(1)
        return

    rate = _get_attr_typical(a.attributes, "control_rate_hz")
    response = _get_attr_typical(a.attributes, "response_time_ms")

    json_output = ctx.obj.get("json", False)
    if json_output:
        click.echo(
            json.dumps(
                {
                    "id": a.id,
                    "name": a.name,
                    "control_rate_hz": rate,
                    "response_time_ms": response,
                }
            )
        )
        return

    console.print(f"\n[bold cyan]{a.name}[/bold cyan]  ({a.id})")
    if rate:
        console.print(f"  Control rate:   {rate:.0f} Hz ({1000/rate:.1f} ms loop period)")
    else:
        console.print("  Control rate:   [dim]not specified[/dim]")
    if response:
        console.print(f"  Response time:  {response:.1f} ms")
    else:
        console.print("  Response time:  [dim]not specified[/dim]")
    console.print()
