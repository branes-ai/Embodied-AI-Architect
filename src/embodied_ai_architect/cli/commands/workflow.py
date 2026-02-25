"""Workflow commands."""

import json
import time
from pathlib import Path

import click
import torch
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table

from embodied_ai_architect.orchestrator import Orchestrator
from embodied_ai_architect.agents.model_analyzer import ModelAnalyzerAgent
from embodied_ai_architect.agents.hardware_profile import HardwareProfileAgent
from embodied_ai_architect.agents.benchmark import BenchmarkAgent
from embodied_ai_architect.agents.benchmark.backends.local_cpu import LocalCPUBackend
from embodied_ai_architect.agents.report_synthesis import ReportSynthesisAgent

console = Console()


@click.group()
def workflow():
    """Run complete workflows for model evaluation.

    \b
    Examples:
      # Run on local CPU
      branes workflow run my_model.pt

      # Use Kubernetes backend
      branes workflow run my_model.pt --backend kubernetes

      # Custom constraints
      branes workflow run my_model.pt \\
        --max-latency 50 \\
        --max-power 100 \\
        --max-cost 3000
    """
    pass


@workflow.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.option(
    "--backend",
    default="local_cpu",
    help="Benchmark backend (local_cpu, remote_ssh, kubernetes)",
)
@click.option("--input-shape", help="Model input shape (e.g., 1,3,224,224)")
@click.option("--iterations", default=100, help="Number of benchmark iterations")
@click.option("--warmup", default=10, help="Number of warmup iterations")
@click.option("--max-latency", type=float, help="Maximum latency constraint (ms)")
@click.option("--max-power", type=float, help="Maximum power constraint (watts)")
@click.option("--max-cost", type=float, help="Maximum cost constraint (USD)")
@click.pass_context
def run(
    ctx,
    model_path,
    backend,
    input_shape,
    iterations,
    warmup,
    max_latency,
    max_power,
    max_cost,
):
    """Run complete workflow on a model."""
    verbose = ctx.obj.get("verbose", 0)
    json_output = ctx.obj.get("json", False)
    quiet = ctx.obj.get("quiet", False)

    try:
        # Load model
        if not quiet and not json_output:
            console.print(f"\n[cyan]Loading model:[/cyan] {model_path}")

        model = torch.load(model_path, weights_only=False)
        model.eval()

        # Parse input shape if provided
        if input_shape:
            input_shape_tuple = tuple(map(int, input_shape.split(",")))
        else:
            # Try to infer from model (default to 1,3,224,224 for CNNs)
            input_shape_tuple = (1, 3, 224, 224)

        # Build constraints
        constraints = {}
        if max_latency:
            constraints["target_latency_ms"] = max_latency
        if max_power:
            constraints["max_power_watts"] = max_power
        if max_cost:
            constraints["max_cost_usd"] = max_cost

        # Create backend instance
        if backend == "local_cpu":
            backend_instance = LocalCPUBackend()
        else:
            # For other backends, we'd need to load them here
            if not quiet and not json_output:
                console.print(f"[yellow]⚠[/yellow] Backend '{backend}' not yet implemented in CLI")
                console.print("[dim]Using local_cpu backend instead[/dim]")
            backend_instance = LocalCPUBackend()

        # Create orchestrator
        orchestrator = Orchestrator()
        orchestrator.register_agent(ModelAnalyzerAgent())
        orchestrator.register_agent(HardwareProfileAgent())
        orchestrator.register_agent(BenchmarkAgent(backends=[backend_instance]))
        orchestrator.register_agent(ReportSynthesisAgent())

        # Run workflow with progress tracking
        start_time = time.time()

        if not quiet and not json_output:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("[cyan]Running workflow...", total=4)

                # Step 1: Model Analysis
                progress.update(task, description="[cyan]📊 Model Analysis")
                result = orchestrator.process(
                    {
                        "model": model,
                        "input_shape": input_shape_tuple,
                        "benchmark_iterations": iterations,
                        "warmup_iterations": warmup,
                        "constraints": constraints,
                    }
                )
                progress.advance(task)

                progress.update(task, completed=4)
        else:
            # Quiet mode - no progress
            result = orchestrator.process(
                {
                    "model": model,
                    "input_shape": input_shape_tuple,
                    "benchmark_iterations": iterations,
                    "warmup_iterations": warmup,
                    "constraints": constraints,
                }
            )

        duration = time.time() - start_time

        # Output results
        if json_output:
            # JSON output for scripting
            output = {
                "status": result.status,
                "workflow_id": result.summary.get("workflow_id"),
                "duration_seconds": duration,
                "model_analysis": (
                    result.agent_results.get("ModelAnalyzer", {}).data
                    if "ModelAnalyzer" in result.agent_results
                    else {}
                ),
                "hardware_recommendations": (
                    result.agent_results.get("HardwareProfile", {}).data
                    if "HardwareProfile" in result.agent_results
                    else {}
                ),
                "benchmarks": (
                    result.agent_results.get("Benchmark", {}).data
                    if "Benchmark" in result.agent_results
                    else {}
                ),
                "report_path": (
                    result.agent_results.get("ReportSynthesis", {}).data.get("report_html")
                    if "ReportSynthesis" in result.agent_results
                    else None
                ),
            }
            click.echo(json.dumps(output, indent=2))
        elif quiet:
            # Quiet mode - just output report path
            report_path = (
                result.agent_results.get("ReportSynthesis", {}).data.get("report_html")
                if "ReportSynthesis" in result.agent_results
                else None
            )
            if report_path:
                click.echo(report_path)
        else:
            # Rich output
            console.print(f"\n[bold green]✓[/bold green] Workflow completed in {duration:.1f}s\n")

            # Model summary
            if "ModelAnalyzer" in result.agent_results:
                model_data = result.agent_results["ModelAnalyzer"].data
                console.print("[bold]📊 Model Analysis[/bold]")
                console.print(f"  Parameters: {model_data.get('total_parameters', 0):,}")
                console.print(f"  Layers: {model_data.get('total_layers', 0)}\n")

            # Hardware recommendations
            if "HardwareProfile" in result.agent_results:
                hw_data = result.agent_results["HardwareProfile"].data
                recommendations = hw_data.get("recommendations", [])
                if recommendations:
                    console.print("[bold]🖥️  Top Hardware Recommendation[/bold]")
                    top = recommendations[0]
                    console.print(f"  {top['name']} (score: {top['score']:.1f})")
                    # Get first reason if available
                    reasons = top.get("reasons", [])
                    if reasons:
                        console.print(f"  {reasons[0]}\n")
                    else:
                        console.print()

            # Benchmark results
            if "Benchmark" in result.agent_results:
                bench_data = result.agent_results["Benchmark"].data
                benchmarks = bench_data.get("benchmarks", {})
                if benchmarks:
                    # Get first benchmark result
                    first_backend = list(benchmarks.keys())[0]
                    bench_result = benchmarks[first_backend]
                    console.print("[bold]⚡ Benchmark Results[/bold]")
                    console.print(f"  Backend: {bench_result.get('device', first_backend)}")
                    console.print(
                        f"  Mean Latency: {bench_result.get('mean_latency_ms', 0):.3f} ms"
                    )
                    console.print(
                        f"  Throughput: {bench_result.get('throughput_samples_per_sec', 0):.2f} samples/sec\n"
                    )

            # Report location
            if "ReportSynthesis" in result.agent_results:
                report_data = result.agent_results["ReportSynthesis"].data
                report_path = report_data.get("report_html")
                console.print(f"[bold]📄 Report:[/bold] {report_path}")
                console.print(
                    f"\n[dim]View report:[/dim] branes report view {result.summary.get('workflow_id', '')}"
                )

    except FileNotFoundError:
        console.print(f"\n[bold red]❌ Error:[/bold red] Model file not found: {model_path}")
        console.print("\n[yellow]💡 Tip:[/yellow] Check the file path or use --help for examples")
        ctx.exit(1)
    except Exception as e:
        if json_output:
            click.echo(json.dumps({"status": "error", "error": str(e)}))
        else:
            console.print(f"\n[bold red]❌ Error:[/bold red] {e}")
            if verbose > 0:
                console.print_exception()
        ctx.exit(1)


@workflow.command()
@click.option("--limit", default=10, help="Maximum number of workflows to list")
@click.pass_context
def list(ctx, limit):
    """List past workflow executions."""
    json_output = ctx.obj.get("json", False)

    # Find all report directories
    reports_dir = Path("./reports")
    if not reports_dir.exists():
        if json_output:
            click.echo(json.dumps({"workflows": []}))
        else:
            console.print("[yellow]No workflows found.[/yellow]")
        return

    workflows = []
    for workflow_dir in sorted(reports_dir.iterdir(), reverse=True)[:limit]:
        if workflow_dir.is_dir():
            workflow_dir / "report.html"
            report_json = workflow_dir / "report.json"

            if report_json.exists():
                # Read workflow metadata
                with open(report_json) as f:
                    data = json.load(f)
                    workflows.append(
                        {
                            "id": workflow_dir.name,
                            "timestamp": data.get("timestamp", "unknown"),
                            "model": data.get("model", {}).get("architecture", "unknown"),
                            "backend": data.get("benchmark", {}).get("device", "unknown"),
                        }
                    )

    if json_output:
        click.echo(json.dumps({"workflows": workflows}, indent=2))
    else:
        if not workflows:
            console.print("[yellow]No workflows found.[/yellow]")
            return

        table = Table(title="Past Workflows", show_header=True)
        table.add_column("Workflow ID", style="cyan")
        table.add_column("Timestamp", style="green")
        table.add_column("Model", style="yellow")
        table.add_column("Backend", style="magenta")

        for wf in workflows:
            table.add_row(wf["id"], wf["timestamp"], wf["model"], wf["backend"])

        console.print(table)
