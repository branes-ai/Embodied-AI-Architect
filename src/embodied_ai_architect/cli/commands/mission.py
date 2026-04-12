"""Mission lifecycle CLI commands (issue #53).

CRUD operations on Mission entities — the persistent backbone that owns
the full design state across qualification, design, optimization, and
validation.

Usage:
    branes mission new "Drone Perception SoC" --goal "30fps at <5W"
    branes mission list
    branes mission show <id>
    branes mission edit <id> --name "Updated Name" --status qualified
    branes mission delete <id>
    branes mission refine <id>
    branes mission fork <src> <dst_name>
"""

import json

import click
from rich.console import Console
from rich.table import Table

console = Console()


@click.group(invoke_without_command=True)
@click.pass_context
def mission(ctx):
    """Manage design missions.

    \\b
    Examples:
      branes mission new "Drone Perception" --goal "30fps at <5W"
      branes mission list
      branes mission show <mission_id>
      branes mission edit <mission_id> --status qualified
      branes mission delete <mission_id>
      branes mission refine <mission_id>
      branes mission fork <source_id> "What-If Variant"
    """
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@mission.command("new")
@click.argument("name")
@click.option("--goal", type=str, default="", help="Design objective")
@click.option("--platform", "platform_id", type=str, default=None, help="Platform registry ID")
@click.option("--use-case", type=str, default=None, help="Use case label")
@click.pass_context
def mission_new(ctx, name, goal, platform_id, use_case):
    """Create a new design mission.

    \\b
    Examples:
      branes mission new "Drone Perception SoC" --goal "30fps detection at <5W"
      branes mission new "Warehouse AMR" --platform ground_wheeled.warehouse_amr
    """
    from embodied_ai_architect.mission import Mission, MissionStore

    m = Mission(
        name=name,
        goal=goal,
        platform_id=platform_id,
        use_case=use_case,
    )

    store = MissionStore()
    mission_id = store.save(m)

    json_output = ctx.obj.get("json", False)
    if json_output:
        click.echo(json.dumps(m.model_dump(), indent=2, default=str))
    else:
        console.print(f"[bold green]Mission created:[/bold green] {mission_id}")
        console.print(f"  Name:     {m.name}")
        console.print(f"  Goal:     {m.goal or '(not set)'}")
        console.print(f"  Status:   {m.status.value}")
        if m.platform_id:
            console.print(f"  Platform: {m.platform_id}")
        console.print(f"\n[dim]Next: branes mission show {mission_id}[/dim]")


@mission.command("list")
@click.pass_context
def mission_list(ctx):
    """List all design missions."""
    from embodied_ai_architect.mission import MissionStore

    store = MissionStore()
    missions = store.list_missions()

    json_output = ctx.obj.get("json", False)
    if json_output:
        click.echo(json.dumps(missions, indent=2, default=str))
        return

    if not missions:
        console.print("[yellow]No missions found.[/yellow]")
        console.print("[dim]Create one: branes mission new 'My Mission' --goal '...'[/dim]")
        return

    table = Table(title="Design Missions", show_header=True)
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Name")
    table.add_column("Status")
    table.add_column("Goal", max_width=40)
    table.add_column("Updated", style="dim")

    status_color = {
        "draft": "dim",
        "qualified": "yellow",
        "designed": "blue",
        "optimized": "green",
        "validated": "bold green",
    }

    for m in missions:
        color = status_color.get(m.get("status", ""), "white")
        table.add_row(
            m.get("id", "?"),
            m.get("name", ""),
            f"[{color}]{m.get('status', '?')}[/{color}]",
            (m.get("goal", "") or "")[:40],
            (m.get("updated_at", "") or "")[:19],
        )

    console.print(table)
    console.print("[dim]Tip: use mission name instead of ID for all commands[/dim]")


@mission.command("show")
@click.argument("mission_id")
@click.option("--json-output", "--json", "json_out", is_flag=True, help="Output as JSON")
@click.pass_context
def mission_show(ctx, mission_id, json_out):
    """Inspect a mission in detail.

    \\b
    Examples:
      branes mission show mission_abc123
    """
    from embodied_ai_architect.mission import MissionStore

    store = MissionStore()
    m = store.load(mission_id)
    if not m:
        console.print(f"[red]Mission '{mission_id}' not found.[/red]")
        ctx.exit(1)
        return

    json_output = json_out or ctx.obj.get("json", False)
    if json_output:
        click.echo(json.dumps(m.model_dump(), indent=2, default=str))
        return

    console.print()
    console.print(f"[bold cyan]Mission: {m.name}[/bold cyan]  ({m.id})")
    console.print(f"  Status:      {m.status.value}")
    console.print(f"  Goal:        {m.goal or '(not set)'}")
    if m.platform_id:
        console.print(f"  Platform:    {m.platform_id}")
    if m.use_case:
        console.print(f"  Use case:    {m.use_case}")
    console.print(f"  Created:     {m.created_at[:19]}")
    console.print(f"  Updated:     {m.updated_at[:19]}")
    console.print()

    # Constraints
    if m.constraints:
        console.print("[bold]Constraints[/bold]")
        for k, v in m.constraints.items():
            console.print(f"  {k}: {v}")
        console.print()

    # Components
    if m.selected_sensors or m.selected_actuators or m.selected_compute or m.selected_models:
        console.print("[bold]Selected Components[/bold]")
        if m.selected_sensors:
            console.print(f"  Sensors:    {', '.join(m.selected_sensors)}")
        if m.selected_actuators:
            console.print(f"  Actuators:  {', '.join(m.selected_actuators)}")
        if m.selected_compute:
            console.print(f"  Compute:    {m.selected_compute}")
        if m.selected_models:
            console.print(f"  Models:     {', '.join(m.selected_models)}")
        console.print()

    # Optimization history
    if m.optimization_history:
        console.print(f"[bold]Optimization History[/bold] ({len(m.optimization_history)} entries)")
        for entry in m.optimization_history[-5:]:
            console.print(f"  {entry}")
        console.print()

    # Artifacts
    if m.bom or m.architecture or m.reports:
        console.print("[bold]Artifacts[/bold]")
        if m.bom:
            console.print("  BOM: present")
        if m.architecture:
            console.print("  Architecture: present")
        if m.reports:
            console.print(f"  Reports: {len(m.reports)}")
        console.print()


@mission.command("edit")
@click.argument("mission_id")
@click.option("--name", type=str, default=None, help="New name")
@click.option("--goal", type=str, default=None, help="New goal")
@click.option("--status", type=str, default=None, help="New status (draft/qualified/designed/...)")
@click.option("--platform", "platform_id", type=str, default=None, help="Platform ID")
@click.option("--use-case", type=str, default=None, help="Use case label")
@click.pass_context
def mission_edit(ctx, mission_id, name, goal, status, platform_id, use_case):
    """Edit mission fields.

    \\b
    Examples:
      branes mission edit mission_abc123 --name "Renamed" --status qualified
      branes mission edit mission_abc123 --goal "Updated goal"
    """
    from embodied_ai_architect.mission import MissionStatus, MissionStore

    store = MissionStore()
    m = store.load(mission_id)
    if not m:
        console.print(f"[red]Mission '{mission_id}' not found.[/red]")
        ctx.exit(1)
        return

    changes = []
    if name is not None:
        m.name = name
        changes.append(f"name → {name}")
    if goal is not None:
        m.goal = goal
        changes.append(f"goal → {goal[:50]}")
    if status is not None:
        try:
            m.status = MissionStatus(status)
            changes.append(f"status → {status}")
        except ValueError:
            valid = ", ".join(s.value for s in MissionStatus)
            console.print(f"[red]Invalid status '{status}'. Valid: {valid}[/red]")
            ctx.exit(1)
            return
    if platform_id is not None:
        m.platform_id = platform_id
        changes.append(f"platform → {platform_id}")
    if use_case is not None:
        m.use_case = use_case
        changes.append(f"use_case → {use_case}")

    if not changes:
        console.print("[yellow]No changes specified.[/yellow]")
        return

    store.save(m)

    json_output = ctx.obj.get("json", False)
    if json_output:
        click.echo(json.dumps(m.model_dump(), indent=2, default=str))
    else:
        console.print(f"[green]Updated mission {mission_id}:[/green]")
        for c in changes:
            console.print(f"  {c}")


@mission.command("delete")
@click.argument("mission_id")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation")
@click.pass_context
def mission_delete(ctx, mission_id, yes):
    """Delete a mission."""
    from embodied_ai_architect.mission import MissionStore

    store = MissionStore()
    if not store.exists(mission_id):
        console.print(f"[red]Mission '{mission_id}' not found.[/red]")
        ctx.exit(1)
        return

    if not yes:
        if not click.confirm(f"Delete mission {mission_id}?"):
            return

    store.delete(mission_id)
    console.print(f"[green]Deleted mission {mission_id}[/green]")


@mission.command("refine")
@click.argument("mission_id")
@click.pass_context
def mission_refine(ctx, mission_id):
    """Re-enter qualification with the mission's current state.

    Loads the mission's spec and constraints, then runs the qualifier
    interactively to refine or extend the qualification.
    """
    from embodied_ai_architect.mission import MissionStore

    store = MissionStore()
    m = store.load(mission_id)
    if not m:
        console.print(f"[red]Mission '{mission_id}' not found.[/red]")
        ctx.exit(1)
        return

    console.print(f"[cyan]Refining mission:[/cyan] {m.name}")
    console.print(f"  Current status: {m.status.value}")
    console.print(f"  Goal: {m.goal}")
    console.print()
    console.print(
        "[dim]To qualify this mission interactively, run:[/dim]\n"
        f'  [cyan]branes design qualify "{m.goal}"[/cyan]'
    )
    console.print(
        "[dim]Then update the mission with the qualification results:[/dim]\n"
        f"  [cyan]branes mission edit {m.id} --status qualified[/cyan]"
    )


@mission.command("fork")
@click.argument("source_id")
@click.argument("new_name")
@click.pass_context
def mission_fork(ctx, source_id, new_name):
    """Fork a mission for what-if analysis.

    Creates a copy of the source mission with a new ID and name,
    preserving all state. The original is untouched.

    \\b
    Examples:
      branes mission fork mission_abc123 "What-If: 7nm Process"
    """
    from embodied_ai_architect.mission import MissionStore

    store = MissionStore()
    source = store.load(source_id)
    if not source:
        console.print(f"[red]Mission '{source_id}' not found.[/red]")
        ctx.exit(1)
        return

    # Deep copy via model_dump → new Mission (gets a fresh ID)
    data = source.model_dump()
    data.pop("id", None)
    data.pop("created_at", None)
    data.pop("updated_at", None)

    from embodied_ai_architect.mission import Mission

    forked = Mission(**data)
    forked.name = new_name
    forked.description = f"Forked from {source.name} ({source.id})"

    new_id = store.save(forked)

    json_output = ctx.obj.get("json", False)
    if json_output:
        click.echo(json.dumps(forked.model_dump(), indent=2, default=str))
    else:
        console.print(f"[bold green]Forked:[/bold green] {source.id} → {new_id}")
        console.print(f"  Name: {forked.name}")
        console.print(f"  Source: {source.name} ({source.id})")
        console.print(f"  Status: {forked.status.value} (preserved from source)")
