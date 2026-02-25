"""Report viewing and management commands."""

import json
import webbrowser
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

console = Console()


@click.group()
def report():
    """View and manage reports.

    \b
    Examples:
      # View latest report in browser
      branes report view --latest

      # View specific report
      branes report view abc123

      # List all reports
      branes report list
    """
    pass


@report.command()
@click.argument("workflow_id", required=False)
@click.option("--latest", is_flag=True, help="View the latest report")
@click.pass_context
def view(ctx, workflow_id, latest):
    """View a report in the browser."""
    json_output = ctx.obj.get("json", False)

    reports_dir = Path("./reports")

    if not reports_dir.exists():
        console.print("[bold red]❌ Error:[/bold red] No reports directory found")
        ctx.exit(1)

    # Determine which report to view
    if latest:
        # Find latest report
        workflow_dirs = sorted(reports_dir.iterdir(), reverse=True)
        if not workflow_dirs:
            console.print("[yellow]No reports found.[/yellow]")
            ctx.exit(1)
        report_dir = workflow_dirs[0]
    elif workflow_id:
        report_dir = reports_dir / workflow_id
        if not report_dir.exists():
            console.print(f"[bold red]❌ Error:[/bold red] Report not found: {workflow_id}")
            ctx.exit(1)
    else:
        console.print("[bold red]❌ Error:[/bold red] Specify workflow ID or use --latest")
        ctx.exit(1)

    report_html = report_dir / "report.html"

    if not report_html.exists():
        console.print(f"[bold red]❌ Error:[/bold red] Report HTML not found: {report_html}")
        ctx.exit(1)

    if json_output:
        click.echo(json.dumps({"report_path": str(report_html)}))
    else:
        console.print(f"\n[cyan]Opening report:[/cyan] {report_html}")
        webbrowser.open(f"file://{report_html.absolute()}")
        console.print("[green]✓[/green] Report opened in browser")


@report.command()
@click.option("--limit", default=20, help="Maximum number of reports to list")
@click.pass_context
def list(ctx, limit):
    """List all available reports."""
    json_output = ctx.obj.get("json", False)

    reports_dir = Path("./reports")

    if not reports_dir.exists():
        if json_output:
            click.echo(json.dumps({"reports": []}))
        else:
            console.print("[yellow]No reports found.[/yellow]")
        return

    reports = []
    for workflow_dir in sorted(reports_dir.iterdir(), reverse=True)[:limit]:
        if workflow_dir.is_dir():
            report_html = workflow_dir / "report.html"
            report_json = workflow_dir / "report.json"

            if report_json.exists():
                # Read report metadata
                with open(report_json) as f:
                    data = json.load(f)
                    reports.append(
                        {
                            "id": workflow_dir.name,
                            "timestamp": data.get("timestamp", "unknown"),
                            "model": data.get("model", {}).get("architecture", "unknown"),
                            "backend": data.get("benchmark", {}).get("device", "unknown"),
                            "path": str(report_html),
                        }
                    )

    if json_output:
        click.echo(json.dumps({"reports": reports}, indent=2))
    else:
        if not reports:
            console.print("[yellow]No reports found.[/yellow]")
            return

        table = Table(title="Available Reports", show_header=True)
        table.add_column("Workflow ID", style="cyan")
        table.add_column("Timestamp", style="green")
        table.add_column("Model", style="yellow")
        table.add_column("Backend", style="magenta")

        for r in reports:
            table.add_row(r["id"], r["timestamp"], r["model"], r["backend"])

        console.print(table)
        console.print(f"\n[dim]Tip:[/dim] Use 'branes report view <workflow_id>' to open a report")


@report.command()
@click.argument("workflow_id_1")
@click.argument("workflow_id_2")
@click.pass_context
def compare(ctx, workflow_id_1, workflow_id_2):
    """Compare two reports."""
    json_output = ctx.obj.get("json", False)

    reports_dir = Path("./reports")

    # Load both reports
    report1_json = reports_dir / workflow_id_1 / "report.json"
    report2_json = reports_dir / workflow_id_2 / "report.json"

    if not report1_json.exists():
        console.print(f"[bold red]❌ Error:[/bold red] Report not found: {workflow_id_1}")
        ctx.exit(1)

    if not report2_json.exists():
        console.print(f"[bold red]❌ Error:[/bold red] Report not found: {workflow_id_2}")
        ctx.exit(1)

    with open(report1_json) as f:
        data1 = json.load(f)

    with open(report2_json) as f:
        data2 = json.load(f)

    if json_output:
        comparison = {
            "workflow_1": {
                "id": workflow_id_1,
                "model": data1.get("model", {}),
                "benchmark": data1.get("benchmark", {}),
            },
            "workflow_2": {
                "id": workflow_id_2,
                "model": data2.get("model", {}),
                "benchmark": data2.get("benchmark", {}),
            },
        }
        click.echo(json.dumps(comparison, indent=2))
    else:
        console.print("\n[bold]Report Comparison[/bold]\n")

        # Model comparison
        console.print("[cyan]Model:[/cyan]")
        console.print(
            f"  {workflow_id_1}: {data1.get('model', {}).get('architecture', 'Unknown')} "
            f"({data1.get('model', {}).get('total_params', 0):,} params)"
        )
        console.print(
            f"  {workflow_id_2}: {data2.get('model', {}).get('architecture', 'Unknown')} "
            f"({data2.get('model', {}).get('total_params', 0):,} params)"
        )

        # Benchmark comparison
        bench1 = data1.get("benchmark", {})
        bench2 = data2.get("benchmark", {})

        console.print("\n[cyan]Benchmark Results:[/cyan]")
        console.print(
            f"  {workflow_id_1}: {bench1.get('mean_latency_ms', 0):.3f}ms "
            f"({bench1.get('throughput_samples_per_sec', 0):.2f} samples/sec)"
        )
        console.print(
            f"  {workflow_id_2}: {bench2.get('mean_latency_ms', 0):.3f}ms "
            f"({bench2.get('throughput_samples_per_sec', 0):.2f} samples/sec)"
        )

        # Performance delta
        latency_diff = bench2.get("mean_latency_ms", 0) - bench1.get("mean_latency_ms", 0)
        if latency_diff < 0:
            console.print(f"\n[green]✓[/green] {workflow_id_2} is {abs(latency_diff):.3f}ms faster")
        elif latency_diff > 0:
            console.print(f"\n[yellow]⚠[/yellow] {workflow_id_2} is {latency_diff:.3f}ms slower")
        else:
            console.print("\n[blue]ℹ[/blue] Both have the same latency")
