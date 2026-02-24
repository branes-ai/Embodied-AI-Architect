"""Codebase analysis CLI commands.

Commands for scanning, analyzing, and assessing application codebases
against hardware targets.
"""

import json
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


@click.group()
def codebase():
    """Analyze application codebases for hardware assessment.

    \b
    Examples:
      branes codebase scan /path/to/project
      branes codebase analyze /path/to/project
      branes codebase assess /path/to/project --hardware jetson_orin
    """
    pass


@codebase.command("scan")
@click.argument("project_path", type=click.Path(exists=True, file_okay=False))
@click.pass_context
def codebase_scan(ctx, project_path: str):
    """Quick static scan of a project directory.

    Returns file inventory, languages, build system, ML model files,
    and dependencies. No LLM needed.

    \b
    Examples:
      branes codebase scan .
      branes codebase scan /path/to/my/app
    """
    json_output = ctx.obj.get("json", False)

    try:
        from embodied_ai_architect.codebase.scanner import CodebaseScanner

        scanner = CodebaseScanner()
        result = scanner.scan(Path(project_path))

        if json_output:
            click.echo(json.dumps(result.model_dump(), indent=2, default=str))
        else:
            _display_scan_result(result)

    except Exception as e:
        if json_output:
            click.echo(json.dumps({"status": "error", "error": str(e)}))
        else:
            console.print(f"\n[bold red]Error:[/bold red] {e}")
        ctx.exit(1)


@codebase.command("analyze")
@click.argument("project_path", type=click.Path(exists=True, file_okay=False))
@click.pass_context
def codebase_analyze(ctx, project_path: str):
    """Full LLM-powered codebase analysis.

    Scans the project, then runs multi-pass LLM analysis to extract
    compute kernels, dataflow, and pipeline structure.

    Requires ANTHROPIC_API_KEY to be set.

    \b
    Examples:
      branes codebase analyze /path/to/app
    """
    json_output = ctx.obj.get("json", False)

    try:
        from embodied_ai_architect.codebase.scanner import CodebaseScanner
        from embodied_ai_architect.codebase.analyzer import CodeAnalyzer
        from embodied_ai_architect.llm.client import LLMClient

        path = Path(project_path)

        if not json_output:
            console.print(f"[bold]Scanning {path.name}...[/bold]")

        scanner = CodebaseScanner()
        scan_result = scanner.scan(path)

        if not json_output:
            console.print(
                f"  Found {len(scan_result.source_files)} source files, "
                f"{scan_result.total_lines} lines"
            )
            console.print(f"  Build system: {scan_result.build_system}")
            console.print("[bold]Running LLM analysis (4 passes)...[/bold]")

        llm = LLMClient()
        analyzer = CodeAnalyzer(llm)
        analysis = analyzer.analyze(scan_result, path)

        if json_output:
            click.echo(json.dumps(analysis.model_dump(), indent=2, default=str))
        else:
            _display_analysis_result(analysis)

    except ImportError as e:
        if "anthropic" in str(e).lower() or "LLMClient" in str(e):
            msg = (
                "LLM client not available. Set ANTHROPIC_API_KEY and install: pip install anthropic"
            )
        else:
            msg = str(e)
        if json_output:
            click.echo(json.dumps({"status": "error", "error": msg}))
        else:
            console.print(f"\n[bold red]Error:[/bold red] {msg}")
        ctx.exit(1)
    except Exception as e:
        if json_output:
            click.echo(json.dumps({"status": "error", "error": str(e)}))
        else:
            console.print(f"\n[bold red]Error:[/bold red] {e}")
        ctx.exit(1)


@codebase.command("assess")
@click.argument("project_path", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--hardware",
    type=str,
    default=None,
    help="Comma-separated hardware targets (e.g., jetson_orin,custom_kpu)",
)
@click.option(
    "--power-budget",
    type=float,
    default=None,
    help="Maximum power budget in watts",
)
@click.option(
    "--latency-target",
    type=float,
    default=None,
    help="Target end-to-end latency in milliseconds",
)
@click.pass_context
def codebase_assess(
    ctx,
    project_path: str,
    hardware: str | None,
    power_budget: float | None,
    latency_target: float | None,
):
    """End-to-end hardware assessment of a codebase.

    Scans the project, runs LLM analysis, converts to workload profile,
    and runs through the hardware assessment pipeline.

    \b
    Examples:
      branes codebase assess /path/to/app
      branes codebase assess /path/to/app --hardware jetson_orin,custom_kpu
      branes codebase assess /path/to/app --power-budget 15 --latency-target 33
    """
    json_output = ctx.obj.get("json", False)

    try:
        from embodied_ai_architect.agents.codebase_analyzer import CodebaseAnalyzerAgent

        target_hardware = hardware.split(",") if hardware else None

        agent = CodebaseAnalyzerAgent()
        result = agent.execute(
            {
                "project_path": project_path,
                "target_hardware": target_hardware,
            }
        )

        if not result.success:
            raise Exception(result.error)

        if json_output:
            click.echo(json.dumps(result.data, indent=2, default=str))
        else:
            _display_assessment_result(result.data, power_budget, latency_target)

    except Exception as e:
        if json_output:
            click.echo(json.dumps({"status": "error", "error": str(e)}))
        else:
            console.print(f"\n[bold red]Error:[/bold red] {e}")
        ctx.exit(1)


# --- Display helpers ---


def _display_scan_result(result) -> None:
    """Display scan results using Rich formatting."""
    console.print(f"\n[bold green]Scan complete:[/bold green] {result.project_name}\n")

    # Summary panel
    summary = (
        f"Languages: {', '.join(result.languages) or 'none detected'}\n"
        f"Build system: {result.build_system}\n"
        f"Source files: {len(result.source_files)}\n"
        f"Total lines: {result.total_lines:,}\n"
        f"ML models: {len(result.ml_models)}"
    )
    console.print(Panel(summary, title="Project Summary", border_style="cyan"))

    # File table
    if result.source_files:
        table = Table(title="Source Files", show_header=True)
        table.add_column("File", style="cyan")
        table.add_column("Language")
        table.add_column("Lines", justify="right")
        table.add_column("Role")

        for sf in result.source_files[:20]:
            table.add_row(sf.path, sf.language, str(sf.lines), sf.role)

        if len(result.source_files) > 20:
            table.add_row(f"... +{len(result.source_files) - 20} more", "", "", "")

        console.print(table)

    # ML models
    if result.ml_models:
        console.print("\n[bold]ML Model Files:[/bold]")
        for m in result.ml_models:
            size_mb = m.get("size_bytes", 0) / (1024 * 1024)
            console.print(f"  {m['path']} ({m['format']}, {size_mb:.1f} MB)")

    # Dependencies
    if result.dependencies:
        console.print(f"\n[bold]Dependencies:[/bold] {', '.join(result.dependencies[:15])}")
        if len(result.dependencies) > 15:
            console.print(f"  ... +{len(result.dependencies) - 15} more")


def _display_analysis_result(analysis) -> None:
    """Display full analysis results."""
    console.print(f"\n[bold green]Analysis complete:[/bold green] {analysis.project_name}\n")

    if analysis.summary:
        console.print(Panel(analysis.summary, title="Summary", border_style="blue"))

    # Kernels table
    if analysis.kernels:
        table = Table(title="Compute Kernels", show_header=True)
        table.add_column("Kernel", style="cyan")
        table.add_column("Type")
        table.add_column("Ops/Call", justify="right")
        table.add_column("Freq (Hz)", justify="right")
        table.add_column("Data Types")
        table.add_column("Parallelism")

        for k in analysis.kernels:
            ops_str = (
                f"{k.estimated_ops_per_invocation:.1e}"
                if k.estimated_ops_per_invocation > 0
                else "-"
            )
            freq_str = f"{k.invocation_frequency_hz:.0f}" if k.invocation_frequency_hz > 0 else "-"
            table.add_row(
                k.name,
                k.kernel_type,
                ops_str,
                freq_str,
                ", ".join(k.data_types),
                k.parallelism,
            )

        console.print(table)

    # Dataflow
    if analysis.dataflow:
        console.print("\n[bold]Dataflow:[/bold]")
        for df in analysis.dataflow:
            console.print(f"  {df.source_kernel} -> {df.sink_kernel} ({df.transfer_type})")


def _display_assessment_result(
    data: dict, power_budget: float | None, latency_target: float | None
) -> None:
    """Display assessment results."""
    scan = data.get("scan_result", {})
    wp = data.get("workload_profile", {})

    console.print(
        f"\n[bold green]Assessment complete:[/bold green] "
        f"{scan.get('project_name', 'project')}\n"
    )

    # Workload summary
    summary = (
        f"Workloads: {wp.get('workload_count', 0)}\n"
        f"Total GFLOPS: {wp.get('total_estimated_gflops', 0):.2f}\n"
        f"Total Memory: {wp.get('total_estimated_memory_mb', 0):.1f} MB\n"
        f"Dominant Op: {wp.get('dominant_op', 'unknown')}"
    )
    console.print(Panel(summary, title="Workload Profile", border_style="cyan"))

    # Workload details
    for w in wp.get("workloads", []):
        console.print(
            f"  [cyan]{w['name']}[/cyan]: {w.get('estimated_gflops', 0):.2f} GFLOPS, "
            f"{w.get('estimated_memory_mb', 0):.1f} MB, "
            f"type={w.get('kernel_type', 'unknown')}"
        )

    if power_budget:
        console.print(f"\n[bold]Constraints:[/bold] power={power_budget}W", end="")
    if latency_target:
        console.print(f", latency={latency_target}ms", end="")
    if power_budget or latency_target:
        console.print()
