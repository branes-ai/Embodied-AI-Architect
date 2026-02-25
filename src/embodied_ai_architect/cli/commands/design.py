"""Design command - Pipeline requirements wizard and synthesis.

Commands for defining pipeline requirements and synthesizing
perception pipelines from those requirements.
"""

from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


@click.group()
def design():
    """Design perception pipelines from requirements.

    \b
    Examples:
      branes design new                        # Interactive wizard
      branes design new -o requirements.yaml   # Save to file
      branes design from-usecase drone_avoidance
      branes design show requirements.yaml
      branes design synthesize requirements.yaml
    """
    pass


@design.command("new")
@click.option(
    "-o",
    "--output",
    type=click.Path(dir_okay=False),
    help="Output YAML file path",
)
@click.option(
    "--no-interactive",
    is_flag=True,
    help="Skip interactive wizard, use defaults",
)
@click.pass_context
def design_new(ctx, output: Optional[str], no_interactive: bool):
    """Create new pipeline requirements interactively.

    Launches an interactive wizard to define perception tasks,
    accuracy constraints, and hardware requirements.

    \b
    Examples:
      branes design new
      branes design new -o my-pipeline.yaml
    """
    from embodied_ai_architect.requirements import (
        PipelineRequirements,
        run_wizard,
        save_requirements,
    )

    if no_interactive:
        # Create default requirements
        requirements = PipelineRequirements(name="default-pipeline")
        console.print("[dim]Created default requirements[/dim]")
    else:
        # Run interactive wizard
        requirements = run_wizard()

    # Save if output specified
    if output:
        output_path = Path(output)
        save_requirements(requirements, output_path)
        console.print(f"\n[green]✓[/green] Saved to {output_path}")
    else:
        # Show YAML preview
        from embodied_ai_architect.requirements import requirements_to_yaml

        console.print()
        console.print("[bold]Generated YAML:[/bold]")
        console.print(Panel(requirements_to_yaml(requirements), border_style="dim"))
        console.print("[dim]Tip: Use -o requirements.yaml to save to file[/dim]")


@design.command("from-usecase")
@click.argument("usecase_id")
@click.option(
    "-o",
    "--output",
    type=click.Path(dir_okay=False),
    help="Output YAML file path",
)
@click.pass_context
def design_from_usecase(ctx, usecase_id: str, output: Optional[str]):
    """Create requirements from an embodied-schemas use case.

    Loads a predefined use case from the embodied-schemas catalog
    and generates corresponding pipeline requirements.

    \b
    Examples:
      branes design from-usecase drone_obstacle_avoidance
      branes design from-usecase industrial_inspection -o reqs.yaml
    """
    from embodied_ai_architect.requirements import (
        from_usecase,
        requirements_to_yaml,
        save_requirements,
    )

    requirements = from_usecase(usecase_id)

    if requirements is None:
        console.print(f"[red]✗[/red] Could not load use case: {usecase_id}")
        _show_available_usecases()
        raise SystemExit(1)

    console.print(f"[green]✓[/green] Loaded use case: {usecase_id}")
    console.print()
    console.print(Panel(requirements.summary(), title="Requirements", border_style="blue"))

    if output:
        output_path = Path(output)
        save_requirements(requirements, output_path)
        console.print(f"\n[green]✓[/green] Saved to {output_path}")
    else:
        console.print()
        console.print("[bold]Generated YAML:[/bold]")
        console.print(Panel(requirements_to_yaml(requirements), border_style="dim"))


@design.command("show")
@click.argument("requirements_file", type=click.Path(exists=True))
@click.pass_context
def design_show(ctx, requirements_file: str):
    """Display requirements from a YAML file.

    \b
    Examples:
      branes design show requirements.yaml
    """
    from embodied_ai_architect.requirements import load_requirements

    requirements = load_requirements(requirements_file)

    console.print(
        Panel(
            requirements.summary(),
            title=f"[bold]{requirements.name}[/bold]",
            border_style="blue",
        )
    )


@design.command("synthesize")
@click.argument("requirements_file", type=click.Path(exists=True))
@click.option(
    "-o",
    "--output",
    type=click.Path(dir_okay=False),
    default="pipeline.yaml",
    help="Output pipeline YAML file",
)
@click.option(
    "--download/--no-download",
    default=True,
    help="Download required models",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be done without executing",
)
@click.option(
    "--validate/--no-validate",
    default=False,
    help="Run inference benchmark on downloaded models",
)
@click.pass_context
def design_synthesize(
    ctx,
    requirements_file: str,
    output: str,
    download: bool,
    dry_run: bool,
    validate: bool,
):
    """Synthesize a pipeline from requirements.

    Analyzes requirements, selects appropriate models from the model zoo,
    downloads them, and generates a pipeline configuration.

    \b
    Examples:
      branes design synthesize requirements.yaml
      branes design synthesize requirements.yaml -o my-pipeline.yaml
      branes design synthesize requirements.yaml --dry-run
    """
    from embodied_ai_architect.requirements import load_requirements

    requirements = load_requirements(requirements_file)

    console.print(f"[bold]Synthesizing pipeline: {requirements.name}[/bold]")
    console.print()

    # Step 1: Find matching models
    console.print("[bold]Step 1: Finding models...[/bold]")
    models = _find_models_for_requirements(requirements)

    if not models:
        console.print("[yellow]No matching models found[/yellow]")
        raise SystemExit(1)

    # Show selected models
    _show_model_selection(models)

    if dry_run:
        console.print("\n[dim]--dry-run: Stopping before download[/dim]")
        return

    # Step 2: Download models
    if download:
        console.print("\n[bold]Step 2: Acquiring models...[/bold]")
        model_paths = _acquire_models(models)
    else:
        console.print("\n[dim]Skipping model download (--no-download)[/dim]")
        model_paths = {}

    # Step 3: Validate models (optional)
    if validate and model_paths:
        console.print("\n[bold]Step 3: Validating models...[/bold]")
        _validate_models(models, model_paths, requirements)

    # Step 4: Generate pipeline
    step_num = 4 if validate else 3
    console.print(f"\n[bold]Step {step_num}: Generating pipeline...[/bold]")
    pipeline_config = _generate_pipeline(requirements, models, model_paths)

    # Save pipeline
    output_path = Path(output)
    _save_pipeline(pipeline_config, output_path)

    console.print(f"\n[green]✓[/green] Pipeline saved to {output_path}")
    console.print(f"\n[dim]Run with: branes pipeline run {output_path}[/dim]")


def _find_models_for_requirements(requirements) -> list[dict]:
    """Find models matching requirements."""
    from embodied_ai_architect.model_zoo import discover
    from embodied_ai_architect.requirements import TaskType

    models = []

    # Map task types to model zoo queries
    task_map = {
        TaskType.OBJECT_DETECTION: "detection",
        TaskType.CLASSIFICATION: "classification",
        TaskType.SEGMENTATION: "segmentation",
        TaskType.INSTANCE_SEGMENTATION: "segmentation",
        TaskType.POSE_ESTIMATION: "pose",
        TaskType.DEPTH_ESTIMATION: "depth_estimation",
        TaskType.FACE_DETECTION: "face_detection",
        TaskType.HAND_TRACKING: "hand_tracking",
    }

    for task in requirements.perception.tasks:
        task_query = task_map.get(task, task.value)

        # Build constraints
        max_params = None
        if requirements.hardware.max_params_millions:
            max_params = int(requirements.hardware.max_params_millions * 1_000_000)

        min_accuracy = requirements.perception.min_accuracy

        # Search for models
        candidates = discover(
            task=task_query,
            max_params=max_params,
            min_accuracy=min_accuracy,
        )

        if candidates:
            # Select best match (smallest that meets constraints)
            selected = candidates[0]
            models.append(
                {
                    "task": task.value,
                    "id": selected.id,
                    "name": selected.name,
                    "provider": selected.provider,
                    "params": selected.parameters,
                    "accuracy": selected.accuracy,
                }
            )
            console.print(f"  [green]✓[/green] {task.value}: {selected.name}")
        else:
            console.print(f"  [yellow]![/yellow] {task.value}: No matching model found")

    return models


def _show_model_selection(models: list[dict]) -> None:
    """Display selected models table."""
    table = Table(title="Selected Models", show_header=True)
    table.add_column("Task")
    table.add_column("Model")
    table.add_column("Provider")
    table.add_column("Params", justify="right")
    table.add_column("Accuracy", justify="right")

    for model in models:
        params = model.get("params")
        params_str = f"{params/1e6:.1f}M" if params else "N/A"

        accuracy = model.get("accuracy")
        acc_str = f"{accuracy*100:.1f}%" if accuracy else "N/A"

        table.add_row(
            model["task"],
            model["name"],
            model["provider"],
            params_str,
            acc_str,
        )

    console.print()
    console.print(table)


def _acquire_models(models: list[dict]) -> dict[str, Path]:
    """Download required models."""
    from embodied_ai_architect.model_zoo import acquire

    paths = {}

    for model in models:
        model_id = model["id"]
        console.print(f"  Acquiring {model_id}...", end=" ")

        try:
            path = acquire(model_id, format="onnx")
            paths[model_id] = path
            console.print(f"[green]✓[/green]")
        except Exception as e:
            console.print(f"[red]✗[/red] {e}")

    return paths


def _validate_models(models: list[dict], model_paths: dict, requirements) -> None:
    """Run inference benchmark on downloaded models."""
    import time

    import numpy as np

    for model in models:
        model_id = model["id"]
        model_path = model_paths.get(model_id)

        if not model_path or not model_path.exists():
            console.print(f"  [yellow]![/yellow] {model_id}: Model not found")
            continue

        console.print(f"  Benchmarking {model_id}...", end=" ")

        try:
            # Load ONNX model
            import onnxruntime as ort

            session = ort.InferenceSession(str(model_path))
            input_info = session.get_inputs()[0]
            input_name = input_info.name

            # Get input shape, default to standard detection shape
            shape = input_info.shape
            if any(isinstance(d, str) or d is None for d in shape):
                shape = [1, 3, 640, 640]  # Default YOLO shape

            # Create dummy input
            dummy_input = np.random.randn(*shape).astype(np.float32)

            # Warmup
            for _ in range(5):
                session.run(None, {input_name: dummy_input})

            # Benchmark
            latencies = []
            for _ in range(20):
                start = time.perf_counter()
                session.run(None, {input_name: dummy_input})
                latencies.append((time.perf_counter() - start) * 1000)

            avg_latency = np.mean(latencies)
            fps = 1000.0 / avg_latency

            # Check against requirements
            status = "[green]✓[/green]"
            if requirements.perception.max_latency_ms:
                if avg_latency > requirements.perception.max_latency_ms:
                    status = "[yellow]![/yellow]"

            console.print(f"{status} {avg_latency:.1f}ms ({fps:.1f} FPS)")

        except ImportError:
            console.print("[yellow]![/yellow] onnxruntime not installed")
        except Exception as e:
            console.print(f"[red]✗[/red] {e}")


def _generate_pipeline(requirements, models: list[dict], model_paths: dict) -> dict:
    """Generate pipeline configuration from requirements and models."""
    from embodied_ai_architect.requirements import TaskType

    operators = []

    for model in models:
        task = model["task"]
        model_id = model["id"]
        model_path = model_paths.get(model_id)

        # Map to operator type
        operator_type = _task_to_operator(task)

        operator_config = {
            "id": f"{task}_operator",
            "type": operator_type,
            "config": {
                "model_id": model_id,
            },
        }

        if model_path:
            operator_config["config"]["model_path"] = str(model_path)

        operators.append(operator_config)

    # Build pipeline config
    pipeline = {
        "name": requirements.name,
        "description": requirements.description or f"Pipeline for {requirements.name}",
        "operators": operators,
        "execution": {
            "target": requirements.hardware.execution_target.value,
            "runtime": requirements.deployment.runtime or "onnxruntime",
            "batch_size": requirements.deployment.batch_size,
        },
    }

    if requirements.deployment.quantization:
        pipeline["execution"]["quantization"] = requirements.deployment.quantization

    return pipeline


def _task_to_operator(task: str) -> str:
    """Map task type to operator class."""
    operator_map = {
        "object_detection": "YOLOv8ONNX",
        "classification": "ImageClassifier",
        "segmentation": "SemanticSegmenter",
        "instance_segmentation": "InstanceSegmenter",
        "pose_estimation": "PoseEstimator",
        "depth_estimation": "DepthEstimator",
        "face_detection": "FaceDetector",
        "hand_tracking": "HandTracker",
    }
    return operator_map.get(task, "GenericOperator")


def _save_pipeline(config: dict, path: Path) -> None:
    """Save pipeline configuration to YAML."""
    import yaml

    with open(path, "w") as f:
        yaml.dump(
            config,
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )


def _show_available_usecases() -> None:
    """Show available use cases from embodied-schemas."""
    try:
        from embodied_schemas import Registry

        registry = Registry()
        usecases = registry.list_usecases()

        if usecases:
            console.print("\n[bold]Available use cases:[/bold]")
            for uc in usecases[:10]:
                console.print(f"  • {uc}")
    except ImportError:
        console.print("[dim]Install embodied-schemas for use case support[/dim]")
    except Exception:
        pass
