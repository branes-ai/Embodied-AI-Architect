"""Benchmarking commands."""

import json

import click
import numpy as np
import torch
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from embodied_ai_architect.agents.benchmark import BenchmarkAgent
from embodied_ai_architect.agents.benchmark.backends.local_cpu import LocalCPUBackend

console = Console()


def _get_architectures():
    """Load architectures from registry."""
    try:
        from embodied_schemas import Registry

        registry = Registry.load()
        return registry.architectures
    except ImportError:
        return {}


def _get_architecture_runner():
    """Get ArchitectureRunner class."""
    from embodied_ai_architect.benchmark import ArchitectureRunner, get_power_monitor

    return ArchitectureRunner, get_power_monitor


@click.group()
def benchmark():
    """Run performance benchmarks.

    \b
    Examples:
      # Benchmark on local CPU
      branes benchmark run my_model.pt

      # Benchmark on specific backend
      branes benchmark run my_model.pt --backend kubernetes

      # List available backends
      branes benchmark list
    """
    pass


@benchmark.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.option(
    "--backend",
    default="local_cpu",
    help="Benchmark backend (local_cpu, remote_ssh, kubernetes)",
)
@click.option("--input-shape", help="Model input shape (e.g., 1,3,224,224)")
@click.option("--iterations", default=100, help="Number of benchmark iterations")
@click.option("--warmup", default=10, help="Number of warmup iterations")
@click.pass_context
def run(ctx, model_path, backend, input_shape, iterations, warmup):
    """Run benchmark on a model."""
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
            # Default shape
            input_shape_tuple = (1, 3, 224, 224)

        # Create backend instance
        if backend == "local_cpu":
            backend_instance = LocalCPUBackend()
        else:
            # For other backends, we'd need to load them here
            console.print(f"[yellow]⚠[/yellow] Backend '{backend}' not yet implemented in CLI")
            console.print("[dim]Using local_cpu backend instead[/dim]")
            backend_instance = LocalCPUBackend()

        # Run benchmark
        agent = BenchmarkAgent(backends=[backend_instance])

        if not quiet and not json_output:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(f"[cyan]Benchmarking on {backend}...", total=None)
                result = agent.execute(
                    {
                        "model": model,
                        "input_shape": input_shape_tuple,
                        "iterations": iterations,
                        "warmup_iterations": warmup,
                    }
                )
                progress.update(task, completed=1)
        else:
            result = agent.execute(
                {
                    "model": model,
                    "input_shape": input_shape_tuple,
                    "iterations": iterations,
                    "warmup_iterations": warmup,
                }
            )

        if not result.success:
            raise Exception(result.error)

        # Output results
        if json_output:
            click.echo(json.dumps(result.data, indent=2))
        else:
            console.print(f"\n[bold green]✓[/bold green] Benchmark complete\n")

            # Get benchmark data for the backend
            benchmarks = result.data.get("benchmarks", {})
            backend_result = benchmarks.get(backend_instance.name, {})

            console.print("[bold]Benchmark Results[/bold]")
            console.print(f"  Backend: {backend_result.get('device', backend)}")
            console.print(f"  Iterations: {backend_result.get('iterations', iterations)}")
            console.print(f"  Mean Latency: {backend_result.get('mean_latency_ms', 0):.3f} ms")
            console.print(f"  Std Latency: {backend_result.get('std_latency_ms', 0):.3f} ms")
            console.print(f"  Min Latency: {backend_result.get('min_latency_ms', 0):.3f} ms")
            console.print(f"  Max Latency: {backend_result.get('max_latency_ms', 0):.3f} ms")
            console.print(
                f"  Throughput: {backend_result.get('throughput_samples_per_sec', 0):.2f} samples/sec"
            )

    except FileNotFoundError:
        console.print(f"\n[bold red]❌ Error:[/bold red] Model file not found: {model_path}")
        ctx.exit(1)
    except Exception as e:
        if json_output:
            click.echo(json.dumps({"status": "error", "error": str(e)}))
        else:
            console.print(f"\n[bold red]❌ Error:[/bold red] {e}")
        ctx.exit(1)


@benchmark.command()
@click.pass_context
def list(ctx):
    """List available benchmark backends."""
    json_output = ctx.obj.get("json", False)

    backends = [
        {
            "name": "local_cpu",
            "description": "Run benchmarks on local CPU",
            "available": True,
            "requirements": "None",
        },
        {
            "name": "remote_ssh",
            "description": "Run benchmarks on remote machine via SSH",
            "available": "paramiko" in dir(),
            "requirements": "pip install branes[remote]",
        },
        {
            "name": "kubernetes",
            "description": "Run benchmarks on Kubernetes cluster",
            "available": "kubernetes" in dir(),
            "requirements": "pip install branes[kubernetes]",
        },
    ]

    if json_output:
        click.echo(json.dumps({"backends": backends}, indent=2))
    else:
        table = Table(title="Available Backends", show_header=True)
        table.add_column("Backend", style="cyan")
        table.add_column("Description", style="white")
        table.add_column("Available", style="green")
        table.add_column("Requirements", style="yellow")

        for backend in backends:
            table.add_row(
                backend["name"],
                backend["description"],
                "✓" if backend["available"] else "✗",
                backend["requirements"],
            )

        console.print(table)


@benchmark.command("arch")
@click.argument("architecture_id")
@click.option("--variant", help="Architecture variant to use")
@click.option("--iterations", default=100, help="Number of benchmark iterations")
@click.option("--warmup", default=10, help="Number of warmup iterations")
@click.option("--power/--no-power", default=True, help="Enable power monitoring")
@click.option("--output", "-o", type=click.Path(), help="Save results to JSON file")
@click.pass_context
def arch(ctx, architecture_id, variant, iterations, warmup, power, output):
    """Benchmark an architecture pipeline.

    \b
    Examples:
      # Benchmark drone perception pipeline
      branes benchmark arch drone_perception_v1

      # Use a specific variant
      branes benchmark arch drone_perception_v1 --variant edge

      # Save results to file
      branes benchmark arch pick_and_place_v1 -o results.json
    """
    json_output = ctx.obj.get("json", False)
    quiet = ctx.obj.get("quiet", False)

    try:
        architectures = _get_architectures()
        if architecture_id not in architectures:
            available = list(architectures.keys())
            raise ValueError(
                f"Architecture not found: {architecture_id}\n" f"Available: {available}"
            )

        arch_def = architectures[architecture_id]
        ArchitectureRunner, get_power_monitor = _get_architecture_runner()

        if not quiet and not json_output:
            console.print(f"\n[cyan]Benchmarking:[/cyan] {arch_def.name}")
            console.print(f"[dim]  {arch_def.description}[/dim]")
            console.print(f"[dim]  Operators: {len(arch_def.operators)}[/dim]")
            console.print()

        # Create runner
        runner = ArchitectureRunner.from_registry(
            architecture_id,
            variant_id=variant,
        )
        runner.load_operators()

        # Create sample inputs
        sample_inputs = _create_sample_inputs(arch_def)

        # Get power monitor
        power_monitor = get_power_monitor() if power else None

        # Run benchmark
        if not quiet and not json_output:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(
                    f"[cyan]Running {iterations} iterations...",
                    total=None,
                )
                result = runner.benchmark(
                    sample_inputs,
                    iterations=iterations,
                    warmup=warmup,
                    power_monitor=power_monitor,
                )
                progress.update(task, completed=1)
        else:
            result = runner.benchmark(
                sample_inputs,
                iterations=iterations,
                warmup=warmup,
                power_monitor=power_monitor,
            )

        # Output results
        if json_output:
            click.echo(json.dumps(result.to_dict(), indent=2))
        else:
            runner.print_summary(result)

        # Save to file if requested
        if output:
            with open(output, "w") as f:
                json.dump(result.to_dict(), f, indent=2)
            if not quiet and not json_output:
                console.print(f"\n[green]Results saved to:[/green] {output}")

        runner.teardown()

    except Exception as e:
        if json_output:
            click.echo(json.dumps({"status": "error", "error": str(e)}))
        else:
            console.print(f"\n[bold red]❌ Error:[/bold red] {e}")
        ctx.exit(1)


@benchmark.command("arch-list")
@click.pass_context
def arch_list(ctx):
    """List available architectures for benchmarking."""
    json_output = ctx.obj.get("json", False)

    architectures = _get_architectures()

    if json_output:
        arch_data = [
            {
                "id": arch_id,
                "name": arch.name,
                "platform": arch.platform_type,
                "operators": len(arch.operators),
                "variants": [v.id for v in arch.variants],
            }
            for arch_id, arch in architectures.items()
        ]
        click.echo(json.dumps({"architectures": arch_data}, indent=2))
    else:
        if not architectures:
            console.print("[yellow]No architectures found.[/yellow]")
            console.print("[dim]Install embodied-schemas: pip install embodied-schemas[/dim]")
            return

        table = Table(title="Available Architectures", show_header=True)
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="white")
        table.add_column("Platform", style="green")
        table.add_column("Operators", style="yellow")
        table.add_column("Variants", style="dim")

        for arch_id, arch in architectures.items():
            variants = ", ".join(v.id for v in arch.variants) if arch.variants else "-"
            table.add_row(
                arch_id,
                arch.name,
                arch.platform_type,
                str(len(arch.operators)),
                variants,
            )

        console.print(table)


def _create_sample_inputs(arch_def) -> dict:
    """Create sample inputs for architecture benchmark."""
    inputs = {}

    # Find the first operator (usually needs image input)
    if arch_def.operators:
        first_op = arch_def.operators[0]
        op_id = first_op.operator_id

        # Create appropriate sample input based on operator type
        if "preprocessor" in op_id or "detector" in op_id:
            inputs[first_op.id] = {
                "image": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            }
        elif "depth" in op_id:
            inputs[first_op.id] = {"depth_map": np.random.rand(480, 640).astype(np.float32)}
        else:
            # Generic empty input
            inputs[first_op.id] = {}

    return inputs
