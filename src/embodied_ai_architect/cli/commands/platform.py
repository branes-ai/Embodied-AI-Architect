"""CLI commands for the Platform Registry.

branes platform list            — list all platforms or filter by category
branes platform show <id>       — show full platform definition
branes platform search <query>  — search platforms by free-text query
branes platform categories      — list available categories
"""

from __future__ import annotations

import json as json_mod

import click
from rich.console import Console
from rich.table import Table

console = Console()


def _get_registry():
    from embodied_ai_architect.platforms import PlatformRegistry

    return PlatformRegistry()


@click.group()
def platform():
    """Manage the platform registry — browse, search, and inspect platform definitions."""
    pass


@platform.command("list")
@click.option("--category", "-c", default=None, help="Filter by category")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def platform_list(ctx, category, as_json):
    """List all platforms in the registry."""
    registry = _get_registry()

    if category:
        platforms = registry.list_by_category(category)
    else:
        platforms = registry.list_platforms()

    if as_json or ctx.obj.get("json"):
        data = [
            {"id": p.id, "name": p.name, "category": p.category, "description": p.description}
            for p in platforms
        ]
        click.echo(json_mod.dumps(data, indent=2))
        return

    table = Table(title=f"Platform Registry ({len(platforms)} platforms)")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Name", style="bold")
    table.add_column("Category", style="dim")
    table.add_column("Description", max_width=50)

    for p in platforms:
        desc = p.description[:50] + "..." if len(p.description) > 50 else p.description
        table.add_row(p.id, p.name, p.category, desc)

    console.print(table)


@platform.command("show")
@click.argument("platform_id")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def platform_show(ctx, platform_id, as_json):
    """Show full details of a platform definition."""
    registry = _get_registry()
    p = registry.get(platform_id)

    if not p:
        console.print(f"[red]Platform not found: {platform_id}[/red]")
        # Suggest close matches
        matches = registry.search(platform_id, top_k=3)
        if matches:
            console.print("\nDid you mean:")
            for m in matches:
                console.print(f"  [cyan]{m.platform_id}[/cyan] — {m.platform.name}")
        raise SystemExit(1)

    if as_json or ctx.obj.get("json"):
        click.echo(json_mod.dumps(p.raw, indent=2))
        return

    console.print()
    console.print(f"[bold cyan]{p.id}[/bold cyan] — {p.name}")
    console.print(f"[dim]{p.description}[/dim]")
    console.print()

    # Classification
    console.print("[bold]Classification[/bold]")
    cls = p.classification
    for key in ["locomotion", "manipulation", "load_class", "environment", "human_proximity"]:
        if key in cls:
            val = cls[key]
            if isinstance(val, list):
                val = ", ".join(val)
            console.print(f"  {key}: {val}")
    console.print()

    # Attributes
    if p.attributes:
        console.print("[bold]Attribute Ranges[/bold]")
        for key, val in p.attributes.items():
            if isinstance(val, dict) and "typical" in val:
                console.print(
                    f"  {key}: {val.get('min', '?')} — {val.get('max', '?')} "
                    f"(typical: {val['typical']})"
                )
            else:
                console.print(f"  {key}: {val}")
        console.print()

    # Context
    ctx_data = p.context
    if ctx_data:
        console.print("[bold]Domain Context[/bold]")
        if ctx_data.get("typical_architecture"):
            console.print(f"  Architecture: {ctx_data['typical_architecture']}")
        if ctx_data.get("design_considerations"):
            console.print(f"  Design: {ctx_data['design_considerations']}")
        if ctx_data.get("reference_designs"):
            console.print(f"  References: {', '.join(ctx_data['reference_designs'])}")
        if ctx_data.get("common_pitfalls"):
            console.print("  Pitfalls:")
            for pit in ctx_data["common_pitfalls"]:
                console.print(f"    - {pit}")
        if ctx_data.get("regulatory"):
            console.print(f"  Regulatory: {', '.join(ctx_data['regulatory'])}")
        console.print()

    # Keywords
    all_kws = p.all_keywords()
    if all_kws:
        console.print(f"[bold]Keywords[/bold] ({len(all_kws)} total)")
        console.print(f"  {', '.join(all_kws[:20])}")
        if len(all_kws) > 20:
            console.print(f"  ... and {len(all_kws) - 20} more")


@platform.command("search")
@click.argument("query")
@click.option("--top", "-n", default=10, help="Number of results")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def platform_search(ctx, query, top, as_json):
    """Search platforms by free-text query."""
    registry = _get_registry()
    matches = registry.search(query, top_k=top)

    if as_json or ctx.obj.get("json"):
        data = [
            {
                "id": m.platform_id,
                "score": m.score,
                "name": m.platform.name,
                "matched_keywords": m.matched_keywords,
            }
            for m in matches
        ]
        click.echo(json_mod.dumps(data, indent=2))
        return

    if not matches:
        console.print(f"[yellow]No platforms matched: {query!r}[/yellow]")
        console.print("Try broader terms or check: branes platform categories")
        return

    console.print(f"\nSearch: [cyan]{query}[/cyan]")
    console.print(f"Found {len(matches)} matches:\n")

    table = Table()
    table.add_column("Score", style="bold", width=6)
    table.add_column("Platform ID", style="cyan")
    table.add_column("Name")
    table.add_column("Matched Keywords", style="dim", max_width=40)

    for m in matches:
        score_str = f"{m.score:.2f}"
        kws = ", ".join(m.matched_keywords[:5])
        if len(m.matched_keywords) > 5:
            kws += f" +{len(m.matched_keywords) - 5}"
        table.add_row(score_str, m.platform_id, m.platform.name, kws)

    console.print(table)


@platform.command("categories")
@click.pass_context
def platform_categories(ctx):
    """List available platform categories."""
    registry = _get_registry()
    categories = registry.list_categories()

    console.print(f"\n[bold]Platform Categories[/bold] ({len(categories)} categories)\n")
    for cat in categories:
        count = len(registry.list_by_category(cat))
        console.print(f"  [cyan]{cat:<25}[/cyan] {count} platform(s)")
    console.print(f"\n  Total: {registry.platform_count} platforms")
