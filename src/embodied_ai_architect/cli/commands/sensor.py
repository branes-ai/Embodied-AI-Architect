"""Sensor browsing and selection CLI commands (issues #54, #59, #63).

Commands for browsing the sensor registry, selecting sensors for
missions, comparing sensor specs, and computing budgets.
"""

import json

import click
from rich.console import Console
from rich.table import Table

from embodied_ai_architect.cli.commands._utils import get_attr_typical as _get_attr_typical

console = Console()


@click.group()
def sensor():
    """Browse the sensor registry.

    \\b
    Examples:
      branes sensor list
      branes sensor list --category visual
      branes sensor show <sensor_id>
      branes sensor search "stereo camera 30fps"
      branes sensor categories
      branes sensor select <mission_id> <sensor_id>
      branes sensor compare <id1> <id2> ...
      branes sensor budget <mission_id>
      branes sensor fusion <mission_id>
    """
    pass


@sensor.command("list")
@click.option(
    "--category",
    type=str,
    default=None,
    help="Filter by category (visual, ranging, inertial, ...)",
)
@click.pass_context
def sensor_list(ctx, category):
    """List all sensors in the registry."""
    from embodied_ai_architect.sensors import SensorRegistry

    registry = SensorRegistry()
    sensors = registry.list_sensors(modality=category)

    json_output = ctx.obj.get("json", False)
    if json_output:
        click.echo(
            json.dumps(
                [{"id": s.id, "name": s.name, "category": s.category} for s in sensors],
                indent=2,
            )
        )
        return

    if not sensors:
        msg = "No sensors found."
        if category:
            msg += f" (filtered by category={category})"
        console.print(f"[yellow]{msg}[/yellow]")
        return

    table = Table(title="Sensors", show_header=True)
    table.add_column("ID", style="cyan")
    table.add_column("Name")
    table.add_column("Category")
    table.add_column("Type", style="dim")

    for s in sensors:
        table.add_row(s.id, s.name, s.category, s.sensor_type)

    console.print(table)


@sensor.command("show")
@click.argument("sensor_id")
@click.pass_context
def sensor_show(ctx, sensor_id):
    """Show details for a specific sensor."""
    from embodied_ai_architect.sensors import SensorRegistry

    registry = SensorRegistry()
    s = registry.get(sensor_id)

    json_output = ctx.obj.get("json", False)
    if not s:
        if json_output:
            click.echo(json.dumps({"error": f"Sensor '{sensor_id}' not found"}))
        else:
            console.print(f"[red]Sensor '{sensor_id}' not found.[/red]")
        ctx.exit(1)
        return
    if json_output:
        click.echo(
            json.dumps(
                {
                    "id": s.id,
                    "name": s.name,
                    "category": s.category,
                    "sensor_type": s.sensor_type,
                    "description": s.description,
                    "attributes": s.attributes,
                },
                indent=2,
            )
        )
        return

    console.print(f"\n[bold cyan]{s.name}[/bold cyan]  ({s.id})")
    console.print(f"  Category:  {s.category}")
    if s.sensor_type:
        console.print(f"  Type:      {s.sensor_type}")
    if s.description:
        console.print(f"  {s.description}")
    if s.aliases:
        console.print(f"  Aliases:   {', '.join(s.aliases)}")
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
                [
                    {
                        "id": r.sensor_id,
                        "name": r.sensor.name,
                        "category": r.sensor.category,
                        "score": r.score,
                    }
                    for r in results
                ],
                indent=2,
            )
        )
        return

    if not results:
        console.print(f"[yellow]No sensors matching '{query}'.[/yellow]")
        return

    table = Table(title=f"Search: {query}", show_header=True)
    table.add_column("ID", style="cyan")
    table.add_column("Name")
    table.add_column("Category")
    table.add_column("Score", style="green")

    for r in results:
        table.add_row(r.sensor_id, r.sensor.name, r.sensor.category, f"{r.score:.3f}")

    console.print(table)


@sensor.command("categories")
@click.pass_context
def sensor_categories(ctx):
    """List sensor categories."""
    from embodied_ai_architect.sensors import SensorRegistry

    registry = SensorRegistry()
    cats = registry.categories()

    json_output = ctx.obj.get("json", False)
    if json_output:
        click.echo(json.dumps(cats, indent=2))
        return

    console.print("\n[bold]Sensor Categories[/bold]\n")
    for cat in cats:
        console.print(f"  {cat}")
    console.print(f"\n[dim]{len(cats)} categories available[/dim]")


# ── New commands from issue #63 ──────────────────────────────────────


@sensor.command("select")
@click.argument("mission_id")
@click.argument("sensor_ids", nargs=-1, required=True)
@click.pass_context
def sensor_select(ctx, mission_id, sensor_ids):
    """Add sensors to a mission's selected sensors list."""
    from embodied_ai_architect.mission.store import MissionStore
    from embodied_ai_architect.sensors import SensorRegistry

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

    registry = SensorRegistry()
    added = []
    skipped = []
    for sid in sensor_ids:
        s = registry.get(sid)
        if not s:
            skipped.append(sid)
            if not json_output:
                console.print(f"[yellow]Sensor '{sid}' not found in registry, skipping.[/yellow]")
            continue
        if sid not in mission.selected_sensors:
            mission.selected_sensors.append(sid)
            added.append(sid)
        elif not json_output:
            console.print(f"[dim]Sensor '{sid}' already selected.[/dim]")

    if added:
        store.save(mission)

    # BUG-008 / #166: exit 1 when all requested sensors were skipped
    if not added and skipped:
        if json_output:
            click.echo(
                json.dumps({"added": added, "skipped": skipped, "total": mission.selected_sensors})
            )
        else:
            console.print(f"[red]No sensors added — {len(skipped)} not found in registry.[/red]")
        ctx.exit(1)
        return

    if json_output:
        click.echo(
            json.dumps({"added": added, "skipped": skipped, "total": mission.selected_sensors})
        )
    elif added:
        console.print(f"[green]Added {len(added)} sensor(s) to mission '{mission.name}':[/green]")
        for sid in added:
            s = registry.get(sid)
            console.print(f"  + {sid} ({s.name if s else '?'})")
        console.print(f"\n[dim]Total selected: {len(mission.selected_sensors)}[/dim]")


@sensor.command("compare")
@click.argument("sensor_ids", nargs=-1, required=True)
@click.pass_context
def sensor_compare(ctx, sensor_ids):
    """Compare sensors side by side."""
    from embodied_ai_architect.sensors import SensorRegistry

    if len(sensor_ids) < 2:
        console.print("[red]Need at least 2 sensor IDs to compare.[/red]")
        ctx.exit(1)
        return

    registry = SensorRegistry()
    sensors = []
    for sid in sensor_ids:
        s = registry.get(sid)
        if not s:
            console.print(f"[red]Sensor '{sid}' not found.[/red]")
            ctx.exit(1)
            return
        sensors.append(s)

    json_output = ctx.obj.get("json", False)
    if json_output:
        click.echo(
            json.dumps(
                [
                    {"id": s.id, "name": s.name, "category": s.category, "attributes": s.attributes}
                    for s in sensors
                ],
                indent=2,
            )
        )
        return

    # Collect all attribute keys across all sensors
    all_keys = []
    for s in sensors:
        for k in s.attributes:
            if k not in all_keys:
                all_keys.append(k)

    table = Table(title="Sensor Comparison", show_header=True)
    table.add_column("Attribute", style="bold")
    for s in sensors:
        table.add_column(s.name, style="cyan")

    # Name and category rows
    table.add_row("ID", *[s.id for s in sensors])
    table.add_row("Category", *[s.category for s in sensors])
    table.add_row("Type", *[s.sensor_type for s in sensors])

    for key in all_keys:
        vals = []
        for s in sensors:
            v = s.attributes.get(key)
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


@sensor.command("budget")
@click.argument("mission_id")
@click.pass_context
def sensor_budget(ctx, mission_id):
    """Show power, weight, and data rate budget for mission sensors."""
    from embodied_ai_architect.mission.store import MissionStore
    from embodied_ai_architect.sensors import SensorRegistry

    store = MissionStore()
    mission = store.load(mission_id)
    if not mission:
        console.print(f"[red]Mission '{mission_id}' not found.[/red]")
        ctx.exit(1)
        return

    if not mission.selected_sensors:
        console.print("[yellow]No sensors selected for this mission.[/yellow]")
        return

    registry = SensorRegistry()
    total_power = 0.0
    total_weight = 0.0
    total_cost = 0.0
    rows = []

    for sid in mission.selected_sensors:
        s = registry.get(sid)
        if not s:
            rows.append((sid, "NOT FOUND", "—", "—", "—"))
            continue
        pw = _get_attr_typical(s.attributes, "power_watts") or 0.0
        wt = _get_attr_typical(s.attributes, "weight_grams") or 0.0
        cost = _get_attr_typical(s.attributes, "cost_usd") or 0.0
        total_power += pw
        total_weight += wt
        total_cost += cost
        rows.append((sid, s.name, f"{pw:.1f}W", f"{wt:.0f}g", f"${cost:.0f}"))

    json_output = ctx.obj.get("json", False)
    if json_output:
        click.echo(
            json.dumps(
                {
                    "sensors": [{"id": r[0], "name": r[1]} for r in rows],
                    "total_power_watts": round(total_power, 2),
                    "total_weight_grams": round(total_weight, 1),
                    "total_cost_usd": round(total_cost, 2),
                }
            )
        )
        return

    table = Table(title=f"Sensor Budget — {mission.name}", show_header=True)
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


@sensor.command("fusion")
@click.argument("mission_id")
@click.pass_context
def sensor_fusion(ctx, mission_id):
    """Recommend sensor fusion strategy for mission sensors."""
    from embodied_ai_architect.mission.store import MissionStore
    from embodied_ai_architect.sensors import SensorRegistry

    store = MissionStore()
    mission = store.load(mission_id)
    if not mission:
        console.print(f"[red]Mission '{mission_id}' not found.[/red]")
        ctx.exit(1)
        return

    if not mission.selected_sensors:
        console.print("[yellow]No sensors selected for this mission.[/yellow]")
        return

    registry = SensorRegistry()
    categories = set()
    for sid in mission.selected_sensors:
        s = registry.get(sid)
        if s:
            categories.add(s.category)

    # Simple heuristic-based fusion recommendations
    recommendations = []
    if "visual" in categories and "inertial" in categories:
        recommendations.append("Visual-Inertial Odometry (VIO) — fuse camera + IMU for ego-motion")
    if "visual" in categories and "ranging" in categories:
        recommendations.append(
            "Depth-enhanced perception — fuse camera + LiDAR/radar for 3D detection"
        )
    if "inertial" in categories and "position" in categories:
        recommendations.append("INS/GNSS fusion — fuse IMU + GPS for robust localization")
    if "visual" in categories and "inertial" in categories and "position" in categories:
        recommendations.append("Full SLAM stack — VIO + GPS for global + local mapping")
    if "ranging" in categories and "inertial" in categories:
        recommendations.append("LiDAR-inertial odometry (LIO) — fuse LiDAR + IMU for mapping")
    if "force" in categories:
        recommendations.append(
            "Force-guided control — fuse force/torque with motion for compliant manipulation"
        )
    if "environmental" in categories:
        recommendations.append(
            "Environment monitoring — aggregate environmental sensors for condition awareness"
        )

    if not recommendations:
        recommendations.append("Single-modality pipeline — no cross-modal fusion recommended")

    json_output = ctx.obj.get("json", False)
    if json_output:
        click.echo(
            json.dumps(
                {
                    "categories": sorted(categories),
                    "recommendations": recommendations,
                }
            )
        )
        return

    console.print(f"\n[bold]Sensor Fusion — {mission.name}[/bold]\n")
    console.print(f"  Selected categories: {', '.join(sorted(categories))}\n")
    console.print("  [bold]Recommendations:[/bold]")
    for rec in recommendations:
        console.print(f"    • {rec}")
    console.print()
