"""Spec-of-specs CLI commands.

Commands for creating, managing, and querying hierarchical system specifications
with full provenance tracking.
"""

import json

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

console = Console()


@click.group()
def spec():
    """Manage system specifications with versioning and provenance.

    \b
    Examples:
      branes spec new my-drone --template drone-perception
      branes spec show my-drone
      branes spec set my-drone /perception/min_fps 60 -m "need 60fps"
      branes spec commit my-drone -m "initial requirements"
      branes spec history my-drone
      branes spec why my-drone /perception/min_fps
    """
    pass


@spec.command("new")
@click.argument("name")
@click.option(
    "--template",
    "-t",
    type=str,
    default=None,
    help="Template archetype (drone-perception, quadruped-nav, etc.)",
)
@click.option("--description", "-d", type=str, default=None, help="Spec description")
@click.pass_context
def spec_new(ctx, name: str, template: str | None, description: str | None):
    """Create a new spec, optionally from a template.

    \b
    Examples:
      branes spec new my-drone --template drone-perception
      branes spec new my-robot -d "Warehouse picking robot"
    """
    json_output = ctx.obj.get("json", False)

    try:
        from embodied_ai_architect.specs.store import SpecStore
        from embodied_ai_architect.specs.templates import get_templates

        store = SpecStore()

        if template and template == "list":
            templates = get_templates()
            if json_output:
                click.echo(json.dumps(templates, indent=2))
            else:
                table = Table(title="Available Templates", show_header=True)
                table.add_column("Name", style="cyan")
                table.add_column("Description")
                for tname, desc in templates.items():
                    table.add_row(tname, desc)
                console.print(table)
            return

        spec_obj = store.create(name, template=template, description=description)

        if json_output:
            click.echo(json.dumps(spec_obj.model_dump(exclude_none=True, mode="json"), indent=2))
        else:
            console.print(f"\n[bold green]Created spec:[/bold green] {name}")
            if template:
                console.print(f"  Template: {template}")
            if description:
                console.print(f"  Description: {description}")
            console.print(f"  Subsystems: {', '.join(spec_obj.subsystem_names()) or 'none'}")
            console.print(f"\n  Use [cyan]branes spec show {name}[/cyan] to view")

    except Exception as e:
        if json_output:
            click.echo(json.dumps({"status": "error", "error": str(e)}))
        else:
            console.print(f"\n[bold red]Error:[/bold red] {e}")
        ctx.exit(1)


@spec.command("list")
@click.pass_context
def spec_list(ctx):
    """List all specs.

    \b
    Examples:
      branes spec list
      branes --json spec list
    """
    json_output = ctx.obj.get("json", False)

    try:
        from embodied_ai_architect.specs.store import SpecStore

        store = SpecStore()
        specs = store.list_specs()

        if json_output:
            click.echo(json.dumps(specs, indent=2))
        else:
            if not specs:
                console.print("[dim]No specs found. Create one with:[/dim]")
                console.print("  [cyan]branes spec new <name> --template <template>[/cyan]")
                return

            table = Table(title="Specs", show_header=True)
            table.add_column("Name", style="cyan")
            table.add_column("Platform")
            table.add_column("Template")
            table.add_column("Description")
            table.add_column("Created")

            for name, meta in specs.items():
                table.add_row(
                    name,
                    meta.get("platform_type", "-") or "-",
                    meta.get("template", "-") or "-",
                    (meta.get("description", "-") or "-")[:50],
                    (meta.get("created_at", "-") or "-")[:19],
                )
            console.print(table)

    except Exception as e:
        if json_output:
            click.echo(json.dumps({"status": "error", "error": str(e)}))
        else:
            console.print(f"\n[bold red]Error:[/bold red] {e}")
        ctx.exit(1)


@spec.command("show")
@click.argument("name")
@click.option("--version", "-v", type=str, default=None, help="Version hash or tag")
@click.pass_context
def spec_show(ctx, name: str, version: str | None):
    """Display a spec as a tree.

    \b
    Examples:
      branes spec show my-drone
      branes spec show my-drone --version v1.0
      branes --json spec show my-drone
    """
    json_output = ctx.obj.get("json", False)

    try:
        from embodied_ai_architect.specs.store import SpecStore

        store = SpecStore()

        if version:
            spec_obj = store.get_version(name, version)
        else:
            spec_obj = store.get(name)

        if json_output:
            click.echo(
                json.dumps(spec_obj.model_dump(exclude_none=True, mode="json"), indent=2)
            )
        else:
            _display_spec_tree(spec_obj)

    except Exception as e:
        if json_output:
            click.echo(json.dumps({"status": "error", "error": str(e)}))
        else:
            console.print(f"\n[bold red]Error:[/bold red] {e}")
        ctx.exit(1)


@spec.command("set")
@click.argument("name")
@click.argument("path")
@click.argument("value")
@click.option("--message", "-m", type=str, default=None, help="Reason for the change")
@click.option("--author", type=str, default="user", help="Author of the change")
@click.pass_context
def spec_set(ctx, name: str, path: str, value: str, message: str | None, author: str):
    """Set a field on a spec.

    \b
    Examples:
      branes spec set my-drone /perception/min_fps 60 -m "need 60fps"
      branes spec set my-drone /compute/soc "Jetson Orin NX"
      branes spec set my-drone /tags '["drone","outdoor"]'
    """
    json_output = ctx.obj.get("json", False)

    try:
        from embodied_ai_architect.specs.store import SpecStore

        store = SpecStore()

        # Try to parse JSON values (for lists/dicts)
        try:
            parsed_value = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            parsed_value = value

        spec_obj = store.set_field(name, path, parsed_value, author=author, reason=message)

        if json_output:
            click.echo(
                json.dumps(spec_obj.model_dump(exclude_none=True, mode="json"), indent=2)
            )
        else:
            console.print(f"[green]Set[/green] {path} = {parsed_value}")
            if message:
                console.print(f"  Reason: {message}")

    except Exception as e:
        if json_output:
            click.echo(json.dumps({"status": "error", "error": str(e)}))
        else:
            console.print(f"\n[bold red]Error:[/bold red] {e}")
        ctx.exit(1)


@spec.command("delete")
@click.argument("name")
@click.argument("path")
@click.option("--message", "-m", type=str, default=None, help="Reason for deletion")
@click.option("--author", type=str, default="user", help="Author of the change")
@click.pass_context
def spec_delete(ctx, name: str, path: str, message: str | None, author: str):
    """Remove a field from a spec.

    \b
    Examples:
      branes spec delete my-drone /perception/model_family -m "no longer needed"
    """
    json_output = ctx.obj.get("json", False)

    try:
        from embodied_ai_architect.specs.store import SpecStore

        store = SpecStore()
        store.delete_field(name, path, author=author, reason=message)

        if json_output:
            click.echo(json.dumps({"status": "ok", "path": path}))
        else:
            console.print(f"[red]Deleted[/red] {path}")

    except Exception as e:
        if json_output:
            click.echo(json.dumps({"status": "error", "error": str(e)}))
        else:
            console.print(f"\n[bold red]Error:[/bold red] {e}")
        ctx.exit(1)


@spec.command("commit")
@click.argument("name")
@click.option("--message", "-m", type=str, required=True, help="Commit message")
@click.option("--author", type=str, default="user", help="Author")
@click.pass_context
def spec_commit(ctx, name: str, message: str, author: str):
    """Snapshot the current spec state.

    \b
    Examples:
      branes spec commit my-drone -m "initial requirements"
    """
    json_output = ctx.obj.get("json", False)

    try:
        from embodied_ai_architect.specs.store import SpecStore

        store = SpecStore()
        blob_hash = store.commit(name, message=message, author=author)

        if json_output:
            click.echo(json.dumps({"status": "ok", "hash": blob_hash}))
        else:
            console.print(f"[green]Committed[/green] {blob_hash[:12]}")
            console.print(f"  Message: {message}")

    except Exception as e:
        if json_output:
            click.echo(json.dumps({"status": "error", "error": str(e)}))
        else:
            console.print(f"\n[bold red]Error:[/bold red] {e}")
        ctx.exit(1)


@spec.command("history")
@click.argument("name")
@click.pass_context
def spec_history(ctx, name: str):
    """Show version history for a spec.

    \b
    Examples:
      branes spec history my-drone
    """
    json_output = ctx.obj.get("json", False)

    try:
        from embodied_ai_architect.specs.store import SpecStore

        store = SpecStore()
        versions = store.history(name)

        if json_output:
            click.echo(json.dumps(versions, indent=2))
        else:
            if not versions:
                console.print("[dim]No committed versions. Use:[/dim]")
                console.print(f'  [cyan]branes spec commit {name} -m "message"[/cyan]')
                return

            table = Table(title=f"Version History: {name}", show_header=True)
            table.add_column("Hash", style="cyan", max_width=12)
            table.add_column("Message")
            table.add_column("Author")
            table.add_column("Tags", style="yellow")
            table.add_column("Timestamp")

            for v in reversed(versions):
                table.add_row(
                    v["hash"][:12],
                    v.get("message", ""),
                    v.get("author", ""),
                    ", ".join(v.get("tags", [])) or "-",
                    v.get("timestamp", "")[:19],
                )
            console.print(table)

    except Exception as e:
        if json_output:
            click.echo(json.dumps({"status": "error", "error": str(e)}))
        else:
            console.print(f"\n[bold red]Error:[/bold red] {e}")
        ctx.exit(1)


@spec.command("diff")
@click.argument("name")
@click.argument("v1")
@click.argument("v2")
@click.pass_context
def spec_diff(ctx, name: str, v1: str, v2: str):
    """Show differences between two versions.

    \b
    Examples:
      branes spec diff my-drone v1.0 v2.0
      branes spec diff my-drone abc123 def456
    """
    json_output = ctx.obj.get("json", False)

    try:
        from embodied_ai_architect.specs.diff import format_diff
        from embodied_ai_architect.specs.store import SpecStore

        store = SpecStore()
        changes = store.diff(name, v1, v2)

        if json_output:
            click.echo(
                json.dumps(
                    [c.model_dump(mode="json") for c in changes],
                    indent=2,
                )
            )
        else:
            if not changes:
                console.print("[dim]No differences.[/dim]")
            else:
                console.print(Panel(format_diff(changes), title=f"Diff: {v1[:12]} → {v2[:12]}"))

    except Exception as e:
        if json_output:
            click.echo(json.dumps({"status": "error", "error": str(e)}))
        else:
            console.print(f"\n[bold red]Error:[/bold red] {e}")
        ctx.exit(1)


@spec.command("tag")
@click.argument("name")
@click.argument("tag_name")
@click.option("--message", "-m", type=str, default=None, help="Tag message")
@click.option("--version", type=str, default=None, help="Version hash to tag (default: latest)")
@click.option("--author", type=str, default="user", help="Author")
@click.pass_context
def spec_tag(ctx, name: str, tag_name: str, message: str | None, version: str | None, author: str):
    """Tag a version with a human-readable name.

    \b
    Examples:
      branes spec tag my-drone v1.0 -m "first baseline"
      branes spec tag my-drone review-ready --version abc123
    """
    json_output = ctx.obj.get("json", False)

    try:
        from embodied_ai_architect.specs.store import SpecStore

        store = SpecStore()
        store.tag(name, tag_name, version=version, message=message, author=author)

        if json_output:
            click.echo(json.dumps({"status": "ok", "tag": tag_name}))
        else:
            console.print(f"[green]Tagged[/green] {tag_name}")

    except Exception as e:
        if json_output:
            click.echo(json.dumps({"status": "error", "error": str(e)}))
        else:
            console.print(f"\n[bold red]Error:[/bold red] {e}")
        ctx.exit(1)


@spec.command("export")
@click.argument("name")
@click.option(
    "--format",
    "-f",
    "fmt",
    type=click.Choice(["json", "yaml"]),
    default="json",
    help="Output format",
)
@click.pass_context
def spec_export(ctx, name: str, fmt: str):
    """Export a spec to JSON or YAML.

    \b
    Examples:
      branes spec export my-drone
      branes spec export my-drone --format yaml > spec.yaml
    """
    try:
        from embodied_ai_architect.specs.store import SpecStore

        store = SpecStore()
        output = store.export_spec(name, fmt=fmt)
        click.echo(output)

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        ctx.exit(1)


@spec.command("import")
@click.argument("name")
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--author", type=str, default="user", help="Author")
@click.pass_context
def spec_import(ctx, name: str, file_path: str, author: str):
    """Import a spec from a JSON or YAML file.

    \b
    Examples:
      branes spec import my-drone-copy spec.json
      branes spec import my-drone-copy spec.yaml
    """
    json_output = ctx.obj.get("json", False)

    try:
        from pathlib import Path

        from embodied_ai_architect.specs.store import SpecStore

        store = SpecStore()
        path = Path(file_path)

        if path.suffix in (".yaml", ".yml"):
            try:
                import yaml

                data = yaml.safe_load(path.read_text())
            except ImportError:
                raise ImportError("PyYAML required for YAML import: pip install pyyaml")
        else:
            data = json.loads(path.read_text())

        spec_obj = store.import_spec(name, data, author=author, reason=f"Imported from {path.name}")

        if json_output:
            click.echo(
                json.dumps(spec_obj.model_dump(exclude_none=True, mode="json"), indent=2)
            )
        else:
            console.print(f"[green]Imported[/green] {name} from {path.name}")

    except Exception as e:
        if json_output:
            click.echo(json.dumps({"status": "error", "error": str(e)}))
        else:
            console.print(f"\n[bold red]Error:[/bold red] {e}")
        ctx.exit(1)


@spec.command("why")
@click.argument("name")
@click.argument("path")
@click.pass_context
def spec_why(ctx, name: str, path: str):
    """Show provenance of a field — who changed it, when, and why.

    \b
    Examples:
      branes spec why my-drone /perception/min_fps
    """
    json_output = ctx.obj.get("json", False)

    try:
        from embodied_ai_architect.specs.store import SpecStore

        store = SpecStore()
        events = store.why(name, path)

        if json_output:
            click.echo(
                json.dumps(
                    [e.model_dump(mode="json") for e in events],
                    indent=2,
                )
            )
        else:
            if not events:
                console.print(f"[dim]No history for {path}[/dim]")
                return

            console.print(f"\n[bold]Provenance:[/bold] {path}\n")
            table = Table(show_header=True)
            table.add_column("Seq", style="dim", justify="right")
            table.add_column("Op", style="cyan")
            table.add_column("Value")
            table.add_column("Author")
            table.add_column("Reason")
            table.add_column("Timestamp")

            for e in events:
                table.add_row(
                    str(e.sequence),
                    e.op.value,
                    str(e.value) if e.value is not None else "-",
                    e.author,
                    e.reason or "-",
                    e.timestamp[:19],
                )
            console.print(table)

    except Exception as e:
        if json_output:
            click.echo(json.dumps({"status": "error", "error": str(e)}))
        else:
            console.print(f"\n[bold red]Error:[/bold red] {e}")
        ctx.exit(1)


@spec.command("validate")
@click.argument("name")
@click.pass_context
def spec_validate(ctx, name: str):
    """Run cross-subsystem consistency checks.

    \b
    Examples:
      branes spec validate my-drone
    """
    json_output = ctx.obj.get("json", False)

    try:
        from embodied_ai_architect.specs.store import SpecStore

        store = SpecStore()
        issues = store.validate(name)

        if json_output:
            click.echo(
                json.dumps(
                    [i.model_dump(mode="json") for i in issues],
                    indent=2,
                )
            )
        else:
            if not issues:
                console.print("[bold green]No issues found.[/bold green]")
                return

            for issue in issues:
                if issue.severity.value == "error":
                    icon = "[bold red]ERROR[/bold red]"
                elif issue.severity.value == "warning":
                    icon = "[bold yellow]WARN[/bold yellow]"
                else:
                    icon = "[bold blue]INFO[/bold blue]"

                console.print(f"  {icon}  {issue.message}")
                console.print(f"         Path: [dim]{issue.path}[/dim]")
                if issue.suggestion:
                    console.print(f"         Fix:  [green]{issue.suggestion}[/green]")
                console.print()

    except Exception as e:
        if json_output:
            click.echo(json.dumps({"status": "error", "error": str(e)}))
        else:
            console.print(f"\n[bold red]Error:[/bold red] {e}")
        ctx.exit(1)


@spec.command("resolve")
@click.argument("name")
@click.pass_context
def spec_resolve(ctx, name: str):
    """Flatten a spec to dot-notation for agent consumption.

    \b
    Examples:
      branes spec resolve my-drone
    """
    try:
        from embodied_ai_architect.specs.store import SpecStore

        store = SpecStore()
        flat = store.resolve(name)
        click.echo(json.dumps(flat, indent=2, default=str))

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        ctx.exit(1)


# ── Display Helpers ────────────────────────────────────────────────────────


def _display_spec_tree(spec_obj) -> None:
    """Display a spec as a Rich tree."""
    title = f"[bold]{spec_obj.name}[/bold]"
    if spec_obj.platform_type:
        title += f" [{spec_obj.platform_type.value}]"
    tree = Tree(title)

    if spec_obj.description:
        tree.add(f"[dim]{spec_obj.description}[/dim]")

    if spec_obj.tags:
        tree.add(f"Tags: {', '.join(spec_obj.tags)}")

    subsystem_order = [
        "perception",
        "compute",
        "power",
        "sensors",
        "actuators",
        "comms",
        "autonomy",
        "safety",
    ]

    for sub_name in subsystem_order:
        sub = getattr(spec_obj, sub_name, None)
        if sub is None:
            continue
        branch = tree.add(f"[cyan]{sub_name}[/cyan]")
        data = sub.model_dump(exclude_none=True)
        for key, val in data.items():
            if isinstance(val, list):
                val_str = ", ".join(str(v) for v in val) if val else "[]"
            else:
                val_str = str(val)
            branch.add(f"{key}: [green]{val_str}[/green]")

    if spec_obj.constraints:
        branch = tree.add("[cyan]constraints[/cyan]")
        for c in spec_obj.constraints:
            branch.add(str(c))

    if spec_obj.custom:
        branch = tree.add("[cyan]custom[/cyan]")
        for key, val in spec_obj.custom.items():
            branch.add(f"{key}: {val}")

    console.print(tree)
