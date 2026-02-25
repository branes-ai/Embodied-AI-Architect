"""Pipeline CLI command for LangGraph operator orchestration."""

import asyncio
from typing import Optional

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


@click.group()
def pipeline():
    """Run operator pipelines with LangGraph orchestration.

    \b
    Examples:
      branes pipeline run perception --input image.jpg
      branes pipeline run autonomy --input video.mp4 --stream
      branes pipeline benchmark perception --input image.jpg --iterations 100
    """
    pass


@pipeline.command()
@click.argument("pipeline_name", type=click.Choice(["perception", "autonomy", "autonomy-ekf"]))
@click.option(
    "--input",
    "-i",
    "input_path",
    type=click.Path(exists=True),
    required=True,
    help="Input image or video file",
)
@click.option(
    "--execution-target",
    "-t",
    type=click.Choice(["cpu", "gpu", "npu"]),
    default="cpu",
    help="Hardware execution target",
)
@click.option(
    "--yolo-variant",
    "-y",
    type=click.Choice(["n", "s", "m", "l", "x"]),
    default="s",
    help="YOLO model variant (n=nano, s=small, m=medium, l=large, x=extra)",
)
@click.option(
    "--stream",
    is_flag=True,
    help="Enable streaming mode for video input",
)
@click.option(
    "--max-frames",
    type=int,
    default=None,
    help="Maximum frames to process in streaming mode",
)
@click.option(
    "--latency-budget",
    type=float,
    default=100.0,
    help="Target latency in milliseconds",
)
@click.option(
    "--checkpoint",
    is_flag=True,
    help="Enable LangGraph checkpointing for persistence",
)
@click.pass_context
def run(
    ctx,
    pipeline_name: str,
    input_path: str,
    execution_target: str,
    yolo_variant: str,
    stream: bool,
    max_frames: Optional[int],
    latency_budget: float,
    checkpoint: bool,
):
    """Run a pipeline on input data.

    PIPELINE_NAME is one of: perception, autonomy, autonomy-ekf
    """
    verbose = ctx.obj.get("verbose", 0)
    json_output = ctx.obj.get("json", False)

    try:
        from embodied_ai_architect.graphs.pipelines import (
            build_perception_graph,
            build_autonomy_graph,
        )
        from embodied_ai_architect.graphs.pipelines.autonomy import (
            build_autonomy_graph_with_ekf,
        )
        from embodied_ai_architect.graphs.runner import (
            PipelineRunner,
            VideoSource,
            load_image,
        )
    except ImportError as e:
        console.print(f"[red]Error:[/red] Missing dependency: {e}")
        console.print("Install with: pip install -e '.[langgraph]'")
        raise SystemExit(1)

    # Build checkpointer if requested
    checkpointer = None
    if checkpoint:
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver

            checkpointer = SqliteSaver.from_conn_string(":memory:")
        except ImportError:
            console.print("[yellow]Warning:[/yellow] Checkpointing requires langgraph[sqlite]")

    # Build graph
    console.print(f"Building [cyan]{pipeline_name}[/cyan] pipeline...")

    if pipeline_name == "perception":
        graph = build_perception_graph(
            execution_target=execution_target,
            yolo_variant=yolo_variant,
            checkpointer=checkpointer,
        )
    elif pipeline_name == "autonomy":
        graph = build_autonomy_graph(
            execution_target=execution_target,
            yolo_variant=yolo_variant,
            checkpointer=checkpointer,
        )
    else:  # autonomy-ekf
        graph = build_autonomy_graph_with_ekf(
            execution_target=execution_target,
            yolo_variant=yolo_variant,
            checkpointer=checkpointer,
        )

    runner = PipelineRunner(graph, verbose=verbose > 0)

    if stream:
        # Streaming mode
        source = VideoSource(input_path, max_frames=max_frames)

        def print_result(result):
            timing = result.get("timing", {})
            total = sum(timing.values())
            status = "[green]OK[/green]" if total <= latency_budget else "[red]OVER[/red]"
            frame_id = result.get("frame_id", 0)
            console.print(f"Frame {frame_id}: {total:.2f}ms {status}")

        console.print(f"Processing video stream: {input_path}")
        asyncio.run(_run_stream(runner, source, print_result, latency_budget))

    else:
        # Batch mode
        console.print(f"Processing image: {input_path}")
        frame = load_image(input_path)

        result = runner.run_batch(
            frame=frame,
            latency_budget_ms=latency_budget,
            execution_target=execution_target,
        )

        _print_result(result, json_output, latency_budget)


async def _run_stream(runner, source, callback, latency_budget):
    """Run streaming pipeline."""
    async for result in runner.run_stream(
        source,
        callback=callback,
        latency_budget_ms=latency_budget,
    ):
        pass


def _print_result(result: dict, json_output: bool, latency_budget: float):
    """Print pipeline result."""
    if json_output:
        import json

        # Convert to JSON-serializable format
        output = {
            "timing": result.get("timing", {}),
            "detections": len(result.get("detections", {}).get("detections", [])),
            "tracks": len(result.get("tracks", {}).get("tracks", [])),
            "scene_objects": len(result.get("scene_objects", {}).get("objects", [])),
            "errors": result.get("errors", []),
            "latency_budget_ms": latency_budget,
        }
        console.print(json.dumps(output, indent=2))
        return

    # Rich table output
    timing = result.get("timing", {})
    total = sum(timing.values())
    over_budget = total > latency_budget

    # Timing table
    table = Table(title="Pipeline Timing")
    table.add_column("Stage", style="cyan")
    table.add_column("Latency (ms)", justify="right")
    table.add_column("% of Total", justify="right")

    for stage, ms in sorted(timing.items(), key=lambda x: x[1], reverse=True):
        pct = (ms / total * 100) if total > 0 else 0
        table.add_row(stage, f"{ms:.2f}", f"{pct:.1f}%")

    status_style = "red" if over_budget else "green"
    table.add_row(
        "[bold]TOTAL[/bold]",
        f"[bold {status_style}]{total:.2f}[/bold {status_style}]",
        "100%",
    )

    console.print(table)

    # Results summary
    detections = result.get("detections", {})
    tracks = result.get("tracks", {})
    scene_objects = result.get("scene_objects", {})
    collision_risks = result.get("collision_risks", {})

    summary = Table(title="Results Summary")
    summary.add_column("Metric", style="cyan")
    summary.add_column("Value", justify="right")

    summary.add_row("Detections", str(len(detections.get("detections", []))))
    summary.add_row("Active Tracks", str(len(tracks.get("tracks", []))))
    summary.add_row("Scene Objects", str(len(scene_objects.get("objects", []))))

    if collision_risks:
        has_critical = collision_risks.get("has_critical", False)
        has_warning = collision_risks.get("has_warning", False)
        if has_critical:
            summary.add_row("Collision Risk", "[red]CRITICAL[/red]")
        elif has_warning:
            summary.add_row("Collision Risk", "[yellow]WARNING[/yellow]")
        else:
            summary.add_row("Collision Risk", "[green]Safe[/green]")

    console.print(summary)

    # Budget status
    if over_budget:
        console.print(
            Panel(
                f"[red]Over budget by {total - latency_budget:.2f}ms[/red]\n"
                f"Budget: {latency_budget:.2f}ms, Actual: {total:.2f}ms",
                title="[red]Latency Budget Exceeded[/red]",
                border_style="red",
            )
        )
    else:
        headroom = latency_budget - total
        console.print(
            Panel(
                f"[green]Under budget by {headroom:.2f}ms[/green]\n"
                f"Budget: {latency_budget:.2f}ms, Actual: {total:.2f}ms",
                title="[green]Latency Budget Met[/green]",
                border_style="green",
            )
        )

    # Errors
    errors = result.get("errors", [])
    if errors:
        console.print("\n[red]Errors:[/red]")
        for error in errors:
            console.print(f"  - {error}")


@pipeline.command()
@click.argument("pipeline_name", type=click.Choice(["perception", "autonomy"]))
@click.option(
    "--input",
    "-i",
    "input_path",
    type=click.Path(exists=True),
    required=True,
    help="Input image for benchmarking",
)
@click.option(
    "--iterations",
    "-n",
    type=int,
    default=100,
    help="Number of timed iterations",
)
@click.option(
    "--warmup",
    "-w",
    type=int,
    default=10,
    help="Number of warmup iterations",
)
@click.option(
    "--execution-target",
    "-t",
    type=click.Choice(["cpu", "gpu", "npu"]),
    default="cpu",
    help="Hardware execution target",
)
@click.option(
    "--yolo-variant",
    "-y",
    type=click.Choice(["n", "s", "m", "l", "x"]),
    default="s",
    help="YOLO model variant",
)
@click.pass_context
def benchmark(
    ctx,
    pipeline_name: str,
    input_path: str,
    iterations: int,
    warmup: int,
    execution_target: str,
    yolo_variant: str,
):
    """Benchmark pipeline performance.

    Runs multiple iterations and reports timing statistics.
    """
    json_output = ctx.obj.get("json", False)

    try:
        from embodied_ai_architect.graphs.pipelines import (
            build_perception_graph,
            build_autonomy_graph,
        )
        from embodied_ai_architect.graphs.runner import PipelineRunner, load_image
    except ImportError as e:
        console.print(f"[red]Error:[/red] Missing dependency: {e}")
        raise SystemExit(1)

    # Build graph
    console.print(f"Building [cyan]{pipeline_name}[/cyan] pipeline for benchmarking...")

    if pipeline_name == "perception":
        graph = build_perception_graph(
            execution_target=execution_target,
            yolo_variant=yolo_variant,
        )
    else:
        graph = build_autonomy_graph(
            execution_target=execution_target,
            yolo_variant=yolo_variant,
        )

    runner = PipelineRunner(graph)

    # Load input
    frame = load_image(input_path)

    # Run benchmark
    console.print(f"Running benchmark: {warmup} warmup + {iterations} timed iterations...")

    with console.status("[bold green]Benchmarking..."):
        results = runner.benchmark(frame, iterations=iterations, warmup=warmup)

    if json_output:
        import json

        console.print(json.dumps(results, indent=2))
        return

    # Total latency table
    total = results["total_latency"]
    table = Table(title=f"Pipeline Benchmark Results ({iterations} iterations)")
    table.add_column("Metric", style="cyan")
    table.add_column("Value (ms)", justify="right")

    table.add_row("Mean", f"{total['mean_ms']:.2f}")
    table.add_row("Std Dev", f"{total['std_ms']:.2f}")
    table.add_row("Min", f"{total['min_ms']:.2f}")
    table.add_row("Max", f"{total['max_ms']:.2f}")
    table.add_row("P50 (Median)", f"{total['p50_ms']:.2f}")
    table.add_row("P95", f"{total['p95_ms']:.2f}")
    table.add_row("P99", f"{total['p99_ms']:.2f}")

    console.print(table)

    # Per-stage breakdown
    stage_table = Table(title="Per-Stage Breakdown")
    stage_table.add_column("Stage", style="cyan")
    stage_table.add_column("Mean (ms)", justify="right")
    stage_table.add_column("Std (ms)", justify="right")
    stage_table.add_column("Min (ms)", justify="right")
    stage_table.add_column("Max (ms)", justify="right")

    for stage, stats in sorted(
        results["per_stage"].items(),
        key=lambda x: x[1]["mean_ms"],
        reverse=True,
    ):
        stage_table.add_row(
            stage,
            f"{stats['mean_ms']:.2f}",
            f"{stats['std_ms']:.2f}",
            f"{stats['min_ms']:.2f}",
            f"{stats['max_ms']:.2f}",
        )

    console.print(stage_table)

    # Throughput
    fps = 1000 / total["mean_ms"] if total["mean_ms"] > 0 else 0
    console.print(f"\n[bold]Throughput:[/bold] {fps:.1f} FPS")


@pipeline.command()
def list():
    """List available pipelines."""
    pipelines = [
        ("perception", "Image → Detect → Track → Scene Graph"),
        ("autonomy", "Full autonomy with collision avoidance"),
        ("autonomy-ekf", "Autonomy with EKF state estimation"),
    ]

    table = Table(title="Available Pipelines")
    table.add_column("Name", style="cyan")
    table.add_column("Description")

    for name, desc in pipelines:
        table.add_row(name, desc)

    console.print(table)
