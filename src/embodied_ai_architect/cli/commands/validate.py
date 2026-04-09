"""Design validation CLI commands (issue #56).

Orchestrates all validation checks against a Mission entity. Wraps
existing subsystem checks into a unified pass/fail report.
"""

import json

import click
from rich.console import Console

console = Console()


def _load_mission(ctx, mission_id: str):
    """Load a mission or exit with error."""
    from embodied_ai_architect.mission import MissionStore

    store = MissionStore()
    m = store.load(mission_id)
    if not m:
        json_output = ctx.obj.get("json", False)
        if json_output:
            click.echo(json.dumps({"error": f"Mission '{mission_id}' not found"}))
        else:
            console.print(f"[red]Mission '{mission_id}' not found.[/red]")
        ctx.exit(1)
        return None
    return m


def _render_check(check, json_output: bool) -> None:
    """Render a single check result."""
    if json_output:
        return  # caller handles JSON
    tag = "[green]PASS[/green]" if check.passed else "[red]FAIL[/red]"
    console.print(f"  {tag}  {check.name}: {check.details}")
    for issue in check.issues:
        console.print(f"       ! {issue}")


@click.group()
def validate():
    """Run design validation checks.

    \\b
    Examples:
      branes validate mission <mission_id>
      branes validate constraints <mission_id>
      branes validate safety <mission_id>
      branes validate scheduling <mission_id>
      branes validate completeness <mission_id>
    """
    pass


@validate.command("mission")
@click.argument("mission_id")
@click.pass_context
def validate_mission(ctx, mission_id):
    """Run ALL validation checks on a mission."""
    from embodied_ai_architect.validation import ValidationRunner

    m = _load_mission(ctx, mission_id)
    if not m:
        return

    runner = ValidationRunner()
    report = runner.validate_all(m)

    json_output = ctx.obj.get("json", False)
    if json_output:
        click.echo(json.dumps(report.to_dict(), indent=2))
        return

    console.print(f"\n[bold]Validation Report:[/bold] {m.name}")
    console.print()
    for check in report.checks:
        _render_check(check, json_output)
    console.print()

    if report.verdict == "PASS":
        console.print(
            f"[bold green]PASS[/bold green] — "
            f"{report.passed_count}/{len(report.checks)} checks passed"
        )
    else:
        console.print(
            f"[bold red]FAIL[/bold red] — "
            f"{report.failed_count}/{len(report.checks)} checks failed"
        )


@validate.command("constraints")
@click.argument("mission_id")
@click.pass_context
def validate_constraints(ctx, mission_id):
    """Check SWaP-C constraint budgets."""
    from embodied_ai_architect.validation import ValidationRunner

    m = _load_mission(ctx, mission_id)
    if not m:
        return

    check = ValidationRunner().check_constraints(m)
    json_output = ctx.obj.get("json", False)
    if json_output:
        click.echo(json.dumps(check.to_dict(), indent=2))
    else:
        _render_check(check, json_output)


@validate.command("completeness")
@click.argument("mission_id")
@click.pass_context
def validate_completeness(ctx, mission_id):
    """Check that all required subsystems are specified."""
    from embodied_ai_architect.validation import ValidationRunner

    m = _load_mission(ctx, mission_id)
    if not m:
        return

    check = ValidationRunner().check_completeness(m)
    json_output = ctx.obj.get("json", False)
    if json_output:
        click.echo(json.dumps(check.to_dict(), indent=2))
    else:
        _render_check(check, json_output)


@validate.command("safety")
@click.argument("mission_id")
@click.pass_context
def validate_safety(ctx, mission_id):
    """Check safety integrity requirements."""
    from embodied_ai_architect.validation import ValidationRunner

    m = _load_mission(ctx, mission_id)
    if not m:
        return

    check = ValidationRunner().check_safety(m)
    json_output = ctx.obj.get("json", False)
    if json_output:
        click.echo(json.dumps(check.to_dict(), indent=2))
    else:
        _render_check(check, json_output)


@validate.command("scheduling")
@click.argument("mission_id")
@click.pass_context
def validate_scheduling(ctx, mission_id):
    """Check multi-rate scheduling feasibility."""
    from embodied_ai_architect.validation import ValidationRunner

    m = _load_mission(ctx, mission_id)
    if not m:
        return

    check = ValidationRunner().check_scheduling(m)
    json_output = ctx.obj.get("json", False)
    if json_output:
        click.echo(json.dumps(check.to_dict(), indent=2))
    else:
        _render_check(check, json_output)
