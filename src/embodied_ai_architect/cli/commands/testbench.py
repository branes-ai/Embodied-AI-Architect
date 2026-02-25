"""Testbench command - Model validation and drift monitoring.

Commands for validating model accuracy and monitoring
performance drift over time.
"""

from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


@click.group()
def testbench():
    """Model validation and drift monitoring.

    \b
    Examples:
      branes testbench validate model.onnx --dataset val.json
      branes testbench drift yolov8n
      branes testbench history yolov8n
      branes testbench benchmark model.onnx
    """
    pass


@testbench.command("validate")
@click.argument("model_path", type=click.Path(exists=True))
@click.option(
    "--dataset",
    type=click.Path(exists=True),
    help="Path to validation dataset (JSON or directory)",
)
@click.option(
    "--task",
    type=click.Choice(["detection", "classification", "segmentation"]),
    default="detection",
    help="Model task type",
)
@click.option(
    "--threshold",
    type=float,
    default=None,
    help="Accuracy threshold for pass/fail",
)
@click.option(
    "--record/--no-record",
    default=True,
    help="Record result for drift monitoring",
)
@click.pass_context
def validate(
    ctx,
    model_path: str,
    dataset: Optional[str],
    task: str,
    threshold: Optional[float],
    record: bool,
):
    """Validate model accuracy against a dataset.

    Runs inference on validation samples and computes
    accuracy metrics.

    \b
    Examples:
      branes testbench validate yolov8n.onnx --dataset coco_val.json
      branes testbench validate model.onnx --task classification --threshold 0.8
    """
    from embodied_ai_architect.testbench import (
        MetricType,
        record_validation,
    )

    model_path = Path(model_path)

    # Set up thresholds
    thresholds = None
    if threshold:
        if task == "detection":
            thresholds = {MetricType.MAP50: threshold}
        elif task == "classification":
            thresholds = {MetricType.TOP1: threshold}
        elif task == "segmentation":
            thresholds = {MetricType.MIOU: threshold}

    if not dataset:
        # Run quick benchmark without ground truth
        console.print("[bold]Running inference benchmark...[/bold]")
        _run_benchmark(model_path, task)
        return

    # Full validation with dataset
    console.print(f"[bold]Validating {model_path.name}[/bold]")
    console.print(f"  Task: {task}")
    console.print(f"  Dataset: {dataset}")
    console.print()

    try:
        from embodied_ai_architect.testbench.validation import validate_model

        result = validate_model(
            model_path=model_path,
            dataset_path=Path(dataset),
            task=task,
            thresholds=thresholds,
        )

        # Show results
        console.print(
            Panel(
                result.summary(),
                title="Validation Result",
                border_style="green" if result.passed else "red",
            )
        )

        # Record for drift monitoring
        if record:
            record_validation(result)

    except Exception as e:
        console.print(f"[red]Validation failed: {e}[/red]")
        raise SystemExit(1)


@testbench.command("benchmark")
@click.argument("model_path", type=click.Path(exists=True))
@click.option(
    "--iterations",
    default=100,
    help="Number of inference iterations",
)
@click.option(
    "--warmup",
    default=10,
    help="Warmup iterations before timing",
)
@click.option(
    "--input-shape",
    default="1,3,640,640",
    help="Input tensor shape (comma-separated)",
)
@click.pass_context
def benchmark(
    ctx,
    model_path: str,
    iterations: int,
    warmup: int,
    input_shape: str,
):
    """Benchmark model inference latency.

    Runs inference repeatedly to measure average latency
    and throughput.

    \b
    Examples:
      branes testbench benchmark model.onnx
      branes testbench benchmark model.onnx --iterations 1000
    """
    _run_benchmark(
        Path(model_path),
        "benchmark",
        iterations=iterations,
        warmup=warmup,
        input_shape=input_shape,
    )


@testbench.command("drift")
@click.argument("model_id")
@click.option(
    "--metric",
    type=click.Choice(["mAP@50", "top1", "mIoU"]),
    default="mAP@50",
    help="Metric to check for drift",
)
@click.pass_context
def drift(ctx, model_id: str, metric: str):
    """Check for performance drift.

    Compares recent validation results against the baseline
    to detect accuracy degradation.

    \b
    Examples:
      branes testbench drift yolov8n
      branes testbench drift resnet50 --metric top1
    """
    from embodied_ai_architect.testbench import MetricType, check_drift

    metric_map = {
        "mAP@50": MetricType.MAP50,
        "top1": MetricType.TOP1,
        "mIoU": MetricType.MIOU,
    }
    metric_type = metric_map.get(metric, MetricType.MAP50)

    console.print(f"[bold]Checking drift for {model_id}[/bold]")
    console.print()

    report = check_drift(model_id, metric_type)

    if report is None:
        console.print("[yellow]Insufficient validation history[/yellow]")
        console.print("[dim]Run more validations to enable drift detection[/dim]")
        return

    # Show report
    border_style = {
        "stable": "green",
        "warning": "yellow",
        "critical": "red",
    }.get(report.status.value, "white")

    console.print(
        Panel(
            report.summary(),
            title="Drift Report",
            border_style=border_style,
        )
    )


@testbench.command("history")
@click.argument("model_id")
@click.option(
    "--limit",
    default=10,
    help="Number of entries to show",
)
@click.option(
    "--clear",
    is_flag=True,
    help="Clear history for this model",
)
@click.pass_context
def history(ctx, model_id: str, limit: int, clear: bool):
    """Show validation history for a model.

    \b
    Examples:
      branes testbench history yolov8n
      branes testbench history yolov8n --limit 20
      branes testbench history yolov8n --clear
    """
    from embodied_ai_architect.testbench import get_monitor

    monitor = get_monitor()

    if clear:
        monitor.clear_history(model_id)
        console.print(f"[green]✓[/green] Cleared history for {model_id}")
        return

    monitor.show_history(model_id, limit)


@testbench.command("list")
@click.pass_context
def list_models(ctx):
    """List models with validation history.

    \b
    Examples:
      branes testbench list
    """
    from embodied_ai_architect.testbench import get_monitor

    monitor = get_monitor()
    models = monitor.list_models()

    if not models:
        console.print("[dim]No models with validation history[/dim]")
        return

    table = Table(title="Models with Validation History")
    table.add_column("Model ID")
    table.add_column("Entries", justify="right")

    for model_id in models:
        history = monitor.get_history(model_id, limit=1000)
        table.add_row(model_id, str(len(history)))

    console.print(table)


def _run_benchmark(
    model_path: Path,
    task: str,
    iterations: int = 100,
    warmup: int = 10,
    input_shape: str = "1,3,640,640",
) -> None:
    """Run inference benchmark."""
    import time

    import numpy as np

    # Parse input shape
    shape = tuple(int(x) for x in input_shape.split(","))

    console.print(f"[bold]Benchmarking {model_path.name}[/bold]")
    console.print(f"  Input shape: {shape}")
    console.print(f"  Iterations: {iterations}")
    console.print()

    # Load model
    suffix = model_path.suffix.lower()

    if suffix == ".onnx":
        try:
            import onnxruntime as ort

            # Check available providers
            providers = ort.get_available_providers()
            console.print(f"  Providers: {', '.join(providers)}")

            session = ort.InferenceSession(str(model_path), providers=providers)
            input_name = session.get_inputs()[0].name

            def infer(x):
                return session.run(None, {input_name: x})

        except ImportError:
            console.print("[red]onnxruntime not installed[/red]")
            return
    else:
        console.print(f"[red]Unsupported format: {suffix}[/red]")
        return

    # Create dummy input
    dummy_input = np.random.randn(*shape).astype(np.float32)

    # Warmup
    console.print("  Warming up...", end=" ")
    for _ in range(warmup):
        infer(dummy_input)
    console.print("[green]done[/green]")

    # Benchmark
    console.print("  Running benchmark...", end=" ")
    latencies = []
    for _ in range(iterations):
        start = time.perf_counter()
        infer(dummy_input)
        latencies.append((time.perf_counter() - start) * 1000)
    console.print("[green]done[/green]")

    # Compute statistics
    latencies = np.array(latencies)
    avg = np.mean(latencies)
    std = np.std(latencies)
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    throughput = 1000.0 / avg

    # Display results
    console.print()
    table = Table(title="Benchmark Results")
    table.add_column("Metric")
    table.add_column("Value", justify="right")

    table.add_row("Average Latency", f"{avg:.2f} ms")
    table.add_row("Std Dev", f"{std:.2f} ms")
    table.add_row("P50 Latency", f"{p50:.2f} ms")
    table.add_row("P95 Latency", f"{p95:.2f} ms")
    table.add_row("P99 Latency", f"{p99:.2f} ms")
    table.add_row("Throughput", f"{throughput:.1f} FPS")

    console.print(table)
