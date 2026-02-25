"""Deployment commands for edge/embedded targets."""

import json

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

console = Console()


@click.group()
def deploy():
    """Deploy models to edge/embedded targets.

    \b
    Supported targets: jetson, openvino, coral, kpu, nvdla

    \b
    Examples:
      # Deploy to Jetson with INT8 quantization
      branes deploy run model.pt --target jetson --precision int8 \\
        --calibration-data ./calib_images --input-shape 1,3,224,224

      # Deploy with validation and power budget
      branes deploy run model.pt --target jetson --precision int8 \\
        --calibration-data ./calib --test-data ./test --validate \\
        --power-budget 5.0 --measure-power

      # Deploy to Stillwater KPU (FP16)
      branes deploy run model.onnx --target swkpu --precision fp16 \\
        --input-shape 1,3,224,224

      # Deploy to NVDLA Virtual Platform
      branes deploy run model.onnx --target nvdla --precision int8 \\
        --calibration-data ./calib --input-shape 1,3,224,224

      # Deploy with FP16 (no calibration needed)
      branes deploy run model.onnx --target openvino --precision fp16 \\
        --input-shape 1,3,640,640

      # List available targets
      branes deploy list
    """
    pass


@deploy.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.option(
    "--target",
    "-t",
    default="jetson",
    help="Deployment target (jetson, openvino, coral, swkpu, nvdla)",
)
@click.option(
    "--precision",
    "-p",
    type=click.Choice(["fp32", "fp16", "int8"]),
    default="int8",
    help="Target precision (default: int8)",
)
@click.option(
    "--input-shape",
    required=True,
    help="Model input shape (e.g., 1,3,224,224)",
)
@click.option(
    "--calibration-data",
    "-c",
    type=click.Path(exists=True),
    help="Path to calibration dataset (required for INT8)",
)
@click.option(
    "--calibration-samples",
    default=100,
    help="Number of calibration samples (default: 100)",
)
@click.option(
    "--calibration-preprocessing",
    type=click.Choice(["imagenet", "yolo", "coco", "none"]),
    default="imagenet",
    help="Preprocessing for calibration images (default: imagenet)",
)
@click.option(
    "--test-data",
    type=click.Path(exists=True),
    help="Path to test dataset for validation",
)
@click.option(
    "--validate/--no-validate",
    default=True,
    help="Run validation after deployment",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    default="./deployments",
    help="Output directory for artifacts",
)
@click.option(
    "--tolerance",
    default=1.0,
    help="Max accuracy drop tolerance in percent (default: 1.0)",
)
@click.option(
    "--power-budget",
    type=float,
    default=None,
    help="Power budget in watts for validation (e.g., 5.0)",
)
@click.option(
    "--measure-power/--no-measure-power",
    default=False,
    help="Measure actual power during inference (requires platform support)",
)
@click.pass_context
def run(
    ctx,
    model_path,
    target,
    precision,
    input_shape,
    calibration_data,
    calibration_samples,
    calibration_preprocessing,
    test_data,
    validate,
    output_dir,
    tolerance,
    power_budget,
    measure_power,
):
    """Deploy a model to target hardware.

    \b
    Examples:
      # INT8 deployment with calibration
      branes deploy run yolov8n.pt --target jetson --precision int8 \\
        --calibration-data ./images --input-shape 1,3,640,640

      # FP16 deployment (no calibration)
      branes deploy run resnet18.onnx --target openvino --precision fp16 \\
        --input-shape 1,3,224,224

      # Deploy with power budget validation
      branes deploy run model.onnx --target swkpu --precision int8 \\
        --calibration-data ./calib --input-shape 1,3,224,224 \\
        --power-budget 5.0 --measure-power
    """
    json_output = ctx.obj.get("json", False)
    quiet = ctx.obj.get("quiet", False)

    try:
        from embodied_ai_architect.agents.deployment import DeploymentAgent

        # Parse input shape
        input_shape_tuple = tuple(map(int, input_shape.split(",")))

        # Validate INT8 requirements
        if precision == "int8" and not calibration_data:
            raise click.UsageError(
                "INT8 precision requires --calibration-data. "
                "Provide a directory with representative images."
            )

        if not quiet and not json_output:
            console.print(f"\n[cyan]Deploying model:[/cyan] {model_path}")
            console.print(f"  Target: {target}")
            console.print(f"  Precision: {precision}")
            console.print(f"  Input shape: {input_shape_tuple}")
            if calibration_data:
                console.print(f"  Calibration: {calibration_data} ({calibration_samples} samples)")
            if test_data:
                console.print(f"  Validation: {test_data}")
            console.print()

        # Create agent and execute
        agent = DeploymentAgent()

        # Check if target is available
        if target not in agent.list_targets():
            available = agent.list_targets()
            if not available:
                raise click.ClickException(
                    f"No deployment targets available. "
                    f"Install TensorRT: pip install embodied-ai-architect[jetson]"
                )
            raise click.ClickException(f"Target '{target}' not available. Available: {available}")

        input_data = {
            "model": model_path,
            "target": target,
            "precision": precision,
            "input_shape": input_shape_tuple,
            "output_dir": output_dir,
        }

        if calibration_data:
            input_data["calibration_data"] = calibration_data
            input_data["calibration_samples"] = calibration_samples
            input_data["calibration_preprocessing"] = calibration_preprocessing

        if test_data and validate:
            input_data["test_data"] = test_data
            input_data["accuracy_tolerance"] = tolerance

        # Power validation options
        if power_budget is not None:
            input_data["power_budget_watts"] = power_budget
        if measure_power:
            input_data["measure_power"] = True

        # Run with progress
        if not quiet and not json_output:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("[cyan]Deploying...", total=None)
                result = agent.execute(input_data)
                progress.update(task, completed=1)
        else:
            result = agent.execute(input_data)

        if not result.success:
            raise Exception(result.error)

        # Output results
        if json_output:
            click.echo(json.dumps(result.data, indent=2, default=str))
        else:
            data = result.data
            console.print(f"\n[bold green]Deployment completed[/bold green]")

            # Show logs
            for log in data.get("logs", []):
                console.print(f"  {log}")

            if data.get("artifact"):
                artifact = data["artifact"]
                console.print(f"\n[bold]Artifact:[/bold]")
                console.print(f"  Engine: {artifact['engine_path']}")
                console.print(f"  Size: {artifact['size_bytes'] / 1024 / 1024:.2f} MB")
                console.print(f"  Precision: {artifact['precision']}")

            if data.get("validation"):
                val = data["validation"]
                status = "[green]PASSED[/green]" if val["passed"] else "[red]FAILED[/red]"
                console.print(f"\n[bold]Validation:[/bold] {status}")
                if val.get("speedup"):
                    console.print(f"  Speedup: {val['speedup']:.2f}x")
                if val.get("baseline_latency_ms"):
                    console.print(f"  Baseline latency: {val['baseline_latency_ms']:.2f} ms")
                if val.get("deployed_latency_ms"):
                    console.print(f"  Deployed latency: {val['deployed_latency_ms']:.2f} ms")
                if val.get("max_output_diff") is not None:
                    console.print(f"  Max output diff: {val['max_output_diff']:.6f}")
                if val.get("samples_compared"):
                    console.print(f"  Samples compared: {val['samples_compared']}")

                # Power metrics
                if val.get("power_watts") is not None:
                    console.print(f"\n[bold]Power:[/bold]")
                    console.print(f"  Average power: {val['power_watts']:.2f} W")
                    if val.get("power_budget_watts"):
                        budget = val["power_budget_watts"]
                        within = val.get("power_within_budget", True)
                        status = "[green]OK[/green]" if within else "[red]EXCEEDED[/red]"
                        console.print(f"  Power budget: {budget:.2f} W {status}")
                    if val.get("energy_per_inference_mj"):
                        console.print(
                            f"  Energy/inference: {val['energy_per_inference_mj']:.3f} mJ"
                        )

    except ImportError:
        console.print(
            "[red]Deployment dependencies not installed.[/red]\n"
            "Install with: pip install embodied-ai-architect[jetson]"
        )
        ctx.exit(1)
    except click.ClickException:
        raise
    except Exception as e:
        if json_output:
            click.echo(json.dumps({"status": "error", "error": str(e)}))
        else:
            console.print(f"\n[bold red]Error:[/bold red] {e}")
        ctx.exit(1)


@deploy.command("list")
@click.pass_context
def list_targets(ctx):
    """List available deployment targets."""
    json_output = ctx.obj.get("json", False)

    targets = []

    try:
        from embodied_ai_architect.agents.deployment import DeploymentAgent

        agent = DeploymentAgent()

        for name in agent.list_targets():
            caps = agent.get_target_capabilities(name)
            targets.append(
                {
                    "name": name,
                    "precisions": caps.get("supported_precisions", []),
                    "available": True,
                    "output_format": caps.get("output_format", "unknown"),
                }
            )
    except ImportError:
        pass

    # Add unavailable targets for reference
    all_possible = [
        {
            "name": "jetson",
            "requires": "tensorrt, pycuda (pip install embodied-ai-architect[jetson])",
        },
        {
            "name": "openvino",
            "requires": "openvino, nncf (pip install embodied-ai-architect[openvino])",
        },
        {
            "name": "coral",
            "requires": "tensorflow, onnx2tf (pip install embodied-ai-architect[coral])",
        },
        {"name": "swkpu", "requires": "torch>=2.0 (pip install embodied-ai-architect[kpu])"},
        {"name": "nvdla", "requires": "onnx (pip install embodied-ai-architect[nvdla])"},
    ]

    for possible in all_possible:
        if possible["name"] not in [t["name"] for t in targets]:
            targets.append(
                {
                    "name": possible["name"],
                    "precisions": [],
                    "available": False,
                    "requires": possible["requires"],
                }
            )

    if json_output:
        click.echo(json.dumps({"targets": targets}, indent=2))
    else:
        table = Table(title="Deployment Targets", show_header=True)
        table.add_column("Target", style="cyan")
        table.add_column("Available", style="green")
        table.add_column("Precisions", style="yellow")
        table.add_column("Output", style="white")
        table.add_column("Requirements", style="dim")

        for t in targets:
            table.add_row(
                t["name"],
                "[green]Yes[/green]" if t["available"] else "[red]No[/red]",
                ", ".join(t.get("precisions", [])) or "-",
                t.get("output_format", "-"),
                t.get("requires", "installed"),
            )

        console.print(table)


@deploy.command("info")
@click.argument("target_name")
@click.pass_context
def info(ctx, target_name):
    """Show detailed information about a deployment target."""
    json_output = ctx.obj.get("json", False)

    try:
        from embodied_ai_architect.agents.deployment import DeploymentAgent

        agent = DeploymentAgent()
        caps = agent.get_target_capabilities(target_name)

        if caps is None:
            raise click.ClickException(
                f"Target '{target_name}' not found. Use 'branes deploy list' to see available targets."
            )

        if json_output:
            click.echo(json.dumps(caps, indent=2))
        else:
            console.print(f"\n[bold cyan]{target_name}[/bold cyan] Deployment Target\n")
            for key, value in caps.items():
                console.print(f"  {key}: {value}")

    except ImportError:
        console.print(
            "[red]Deployment dependencies not installed.[/red]\n"
            "Install with: pip install embodied-ai-architect[jetson]"
        )
        ctx.exit(1)
