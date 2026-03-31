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
            console.print("[green]✓[/green]")
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


@design.command("plan")
@click.argument("goal")
@click.option("--power", type=float, help="Max power budget in watts")
@click.option("--latency", type=float, help="Max latency in ms")
@click.option("--cost", type=float, help="Max BOM cost in USD")
@click.option("--area", type=float, help="Max die area in mm²")
@click.option("--process", type=int, help="Target process node in nm (e.g., 28, 16, 7)")
@click.option("--use-case", default="", help="Application type (e.g., delivery_drone)")
@click.option("--platform", default="", help="Platform type (e.g., drone, amr, quadruped)")
@click.option("--static", is_flag=True, help="Use static demo plan instead of LLM")
@click.pass_context
def design_plan(ctx, goal, power, latency, cost, area, process, use_case, platform, static):
    """Show the task plan the LLM generates for a design goal.

    Runs the planner to decompose a natural-language goal into a task graph
    (DAG of specialist agents), then displays the plan for review. Does NOT
    execute the plan — use this to see what the architect would do.

    \\b
    Examples:
      branes design plan "Drone perception SoC: YOLO at 30fps, <5W, <\\$30"
      branes design plan "Robot arm vision: <10ms latency" --power 15 --latency 10
      branes design plan "Drone SoC" --static   # no API key needed
    """
    from embodied_ai_architect.graphs.soc_state import (
        DesignConstraints,
        create_initial_soc_state,
    )
    from embodied_ai_architect.graphs.planner import PlannerNode
    from embodied_ai_architect.graphs.review import (
        build_review_snapshot,
        render_plan_review_rich,
    )
    from embodied_ai_architect.graphs.specialists import create_default_dispatcher

    # Build constraints
    constraint_kwargs = {}
    if power is not None:
        constraint_kwargs["max_power_watts"] = power
    if latency is not None:
        constraint_kwargs["max_latency_ms"] = latency
    if cost is not None:
        constraint_kwargs["max_cost_usd"] = cost
    if area is not None:
        constraint_kwargs["max_area_mm2"] = area
    if process is not None:
        constraint_kwargs["target_process_nm"] = process

    constraints = DesignConstraints(**constraint_kwargs) if constraint_kwargs else None

    state = create_initial_soc_state(
        goal=goal,
        constraints=constraints,
        use_case=use_case,
        platform=platform,
    )

    # Create planner
    if static:
        static_plan = [
            {
                "id": "t1",
                "name": "Analyze workload",
                "agent": "workload_analyzer",
                "dependencies": [],
            },
            {
                "id": "t2",
                "name": "Enumerate feasible hardware",
                "agent": "hw_explorer",
                "dependencies": ["t1"],
            },
            {
                "id": "t3",
                "name": "Compose SoC architecture",
                "agent": "architecture_composer",
                "dependencies": ["t2"],
            },
            {
                "id": "t4",
                "name": "Assess PPA metrics",
                "agent": "ppa_assessor",
                "dependencies": ["t3"],
            },
            {"id": "t5", "name": "Review design", "agent": "critic", "dependencies": ["t4"]},
            {
                "id": "t6",
                "name": "Generate report",
                "agent": "report_generator",
                "dependencies": ["t5"],
            },
        ]
        planner = PlannerNode(static_plan=static_plan)
        console.print("[dim]Using static demo plan (no LLM call)[/dim]")
    else:
        import os

        if not os.environ.get("ANTHROPIC_API_KEY"):
            console.print(
                Panel(
                    "[yellow]ANTHROPIC_API_KEY not set[/yellow]\n\n"
                    "Set your API key for LLM planning:\n"
                    "  [cyan]export ANTHROPIC_API_KEY=your-key-here[/cyan]\n\n"
                    "Or use [cyan]--static[/cyan] for a demo plan without an API key.",
                    title="API Key Required",
                    border_style="yellow",
                )
            )
            return

        from embodied_ai_architect.llm.client import LLMClient

        llm = LLMClient()
        planner = PlannerNode(llm=llm)
        console.print("[dim]Calling LLM to decompose goal into task graph...[/dim]")

    try:
        plan_updates = planner(state)
    except Exception as e:
        console.print(f"[red]Planning failed:[/red] {e}")
        ctx.exit(1)
        return

    state = {**state, **plan_updates}

    # Display the plan review
    dispatcher = create_default_dispatcher()
    available_agents = dispatcher.registered_agents
    snapshot = build_review_snapshot(state, available_agents)
    console.print()
    console.print(render_plan_review_rich(snapshot))

    # Workflow guidance — what can the user do next?
    console.print()
    console.print("─" * 72)
    console.print("  NEXT STEPS")
    console.print("─" * 72)
    console.print()
    console.print("  The plan above shows the task graph the system will execute.")
    console.print("  From here you can:")
    console.print()
    console.print("  [cyan]1. Execute this plan[/cyan]")
    console.print("     Run the full pipeline (plan → dispatch → optimize → report)")
    console.print(
        f"     [dim]branes workflow run <model> --power {power or '?'} "
        f"--latency {latency or '?'}[/dim]"
    )
    console.print()
    console.print("  [cyan]2. Refine the goal[/cyan]")
    console.print("     Go back to qualification to add/change requirements")
    console.print(f'     [dim]branes design qualify "{goal}"[/dim]')
    console.print()
    console.print("  [cyan]3. Interactive design session[/cyan]")
    console.print("     Review, edit, and steer execution in real-time via chat")
    console.print("     [dim]branes chat[/dim]")
    console.print('     then: "Design a drone SoC for fixed-wing survey with VIO at <3W, <20ms"')
    console.print()
    console.print("  [cyan]4. Run the demo[/cyan]")
    console.print("     Execute the full interactive review flow programmatically")
    console.print("     [dim]python examples/demo_interactive_review.py --modify[/dim]")
    console.print()


@design.command("qualify")
@click.argument("goal")
@click.option("--domain", "-d", help="Force a domain (drone, ugv, robot_arm)")
@click.option(
    "--auto",
    is_flag=True,
    help="Auto-answer with defaults where possible (non-interactive)",
)
@click.pass_context
def design_qualify(ctx, goal, domain, auto):
    """Qualify a design goal through structured Q&A.

    Checks whether a goal is specific enough to produce meaningful design
    results. If not, walks through domain-specific questions to refine it.

    \\b
    Examples:
      branes design qualify "drone perception SoC"
      branes design qualify "cobot for assembly" --domain robot_arm
      branes design qualify "warehouse AMR" --auto
    """
    from embodied_ai_architect.qualification import GoalQualifier
    from embodied_ai_architect.qualification.qualifier import render_qualification_result

    qualifier = GoalQualifier()
    result = qualifier.assess(goal, domain=domain)
    console.print()
    console.print(render_qualification_result(result))

    # Run the full survey — every question gathers a requirement that matters.
    # Don't short-circuit on early tangibility; complete all questions first.
    while result.next_question:
        q = result.next_question

        if auto and q.default:
            console.print(f"  [dim]Auto: {q.id} = {q.default}[/dim]")
            result = qualifier.skip(q.id)
            continue

        if auto and not q.required:
            console.print(f"  [dim]Auto: skipping optional '{q.id}'[/dim]")
            result = qualifier.skip(q.id)
            continue

        # Interactive prompt
        console.print()
        if q.options:
            console.print(f"[bold]{q.text}[/bold]")
            if q.explanation:
                console.print(f"[dim]{q.explanation}[/dim]")
            console.print()
            for i, opt in enumerate(q.options, 1):
                desc = q.option_descriptions.get(opt, "")
                if desc:
                    console.print(f"  [cyan]{i}[/cyan]. {opt} — {desc}")
                else:
                    console.print(f"  [cyan]{i}[/cyan]. {opt}")
            console.print()

            # Show custom option hint if allowed
            if q.allow_custom:
                console.print(
                    "  [cyan]c[/cyan]. [italic]Add custom answer (not listed above)[/italic]"
                )
                console.print()

            if q.question_type.value == "multi_choice":
                prompt_text = "Select (comma-separated numbers"
                if q.allow_custom:
                    prompt_text += ", 'c' to add custom"
                prompt_text += ", or 's' to skip): "
                raw = console.input(f"[bold]{prompt_text}[/bold]")
                raw = raw.strip()
                if raw.lower() == "s":
                    result = qualifier.skip(q.id)
                else:
                    # Parse selections — may include 'c' for custom
                    parts = [p.strip() for p in raw.split(",")]
                    indices = [int(p) - 1 for p in parts if p.isdigit()]
                    selected = [q.options[i] for i in indices if 0 <= i < len(q.options)]
                    has_custom = any(p.lower() == "c" for p in parts)

                    if has_custom and q.allow_custom:
                        customs = _collect_custom_answers(q, console)
                        from embodied_ai_architect.qualification.models import CustomAnswer

                        result = qualifier.answer_custom(
                            q.id,
                            [CustomAnswer(**c) for c in customs],
                            also_selected=selected or None,
                        )
                    elif selected:
                        result = qualifier.answer(q.id, selected)
                    else:
                        console.print("[yellow]No valid selection, skipping.[/yellow]")
                        result = qualifier.skip(q.id)
            else:
                prompt_text = "Select (number"
                if q.allow_custom:
                    prompt_text += ", 'c' for custom"
                prompt_text += ", or 's' to skip): "
                raw = console.input(f"[bold]{prompt_text}[/bold]")
                raw = raw.strip()
                if raw.lower() == "s":
                    result = qualifier.skip(q.id)
                elif raw.lower() == "c" and q.allow_custom:
                    customs = _collect_custom_answers(q, console)
                    from embodied_ai_architect.qualification.models import CustomAnswer

                    result = qualifier.answer_custom(q.id, [CustomAnswer(**c) for c in customs])
                elif raw.isdigit():
                    idx = int(raw) - 1
                    if 0 <= idx < len(q.options):
                        result = qualifier.answer(q.id, q.options[idx])
                    else:
                        console.print("[yellow]Invalid selection, skipping.[/yellow]")
                        result = qualifier.skip(q.id)
                else:
                    result = qualifier.skip(q.id)
        else:
            # Free text or numeric
            raw = console.input(f"[bold]{q.text} [/bold]")
            if raw.strip():
                if q.question_type.value == "numeric":
                    try:
                        result = qualifier.answer(q.id, float(raw))
                    except ValueError:
                        console.print("[yellow]Invalid number, skipping.[/yellow]")
                        result = qualifier.skip(q.id)
                else:
                    result = qualifier.answer(q.id, raw.strip())
            else:
                result = qualifier.skip(q.id)

        # Show updated scorecard
        console.print()
        console.print(render_qualification_result(result))

    if result.is_tangible:
        _show_design_inputs(qualifier)
    else:
        console.print("[yellow]Goal could not be fully qualified.[/yellow]")
        console.print(
            "[dim]Missing: " + ", ".join(result.qualification.missing_dimensions) + "[/dim]"
        )


def _collect_custom_answers(question, console) -> list[dict]:
    """Interactively collect custom answer(s) from the user."""
    customs = []
    if question.custom_prompt:
        console.print(f"\n[dim]{question.custom_prompt}[/dim]\n")

    while True:
        value = console.input("[bold]Custom answer name (short identifier): [/bold]").strip()
        if not value:
            break
        description = console.input("[bold]Description: [/bold]").strip()

        # Collect implications interactively
        console.print("[dim]Enter implications as key=value pairs (empty line to finish):[/dim]")
        console.print(
            "[dim]  e.g., sensors.modalities=imu  or  perception.max_latency_ms=5.0[/dim]"
        )
        implications = {}
        while True:
            pair = console.input("  ").strip()
            if not pair:
                break
            if "=" in pair:
                k, v = pair.split("=", 1)
                k = k.strip()
                v = v.strip()
                # Try to parse as number or list
                try:
                    v = float(v)
                except ValueError:
                    if "," in v:
                        v = [x.strip() for x in v.split(",")]
                    elif v.lower() in ("true", "false"):
                        v = v.lower() == "true"
                implications[k] = v

        customs.append(
            {
                "value": value,
                "description": description,
                "implications": implications,
            }
        )

        more = console.input("[bold]Add another custom answer? (y/n) [n]: [/bold]").strip()
        if more.lower() != "y":
            break

    return customs


def _show_design_inputs(qualifier) -> None:
    """Show the design inputs that would be passed to the planner."""
    inputs = qualifier.to_design_inputs()
    console.print()
    console.print("[bold green]Goal qualified. Design inputs:[/bold green]")
    console.print(f"  Goal:     {inputs['goal']}")
    console.print(f"  Platform: {inputs['platform']}")
    console.print(f"  Use case: {inputs['use_case']}")
    if inputs.get("constraints"):
        c = inputs["constraints"]
        data = c.model_dump(exclude_none=True, exclude_defaults=True)
        for k, v in data.items():
            console.print(f"  {k}: {v}")

    # Build the exact command the user should run next
    goal_escaped = inputs["goal"].replace('"', '\\"')
    cmd_parts = [f'branes design plan "{goal_escaped}"']
    if inputs.get("constraints"):
        c = inputs["constraints"]
        if c.max_power_watts is not None:
            cmd_parts.append(f"--power {c.max_power_watts}")
        if c.max_latency_ms is not None:
            cmd_parts.append(f"--latency {c.max_latency_ms}")
        if c.max_cost_usd is not None:
            cmd_parts.append(f"--cost {c.max_cost_usd}")
    if inputs.get("use_case"):
        cmd_parts.append(f"--use-case {inputs['use_case']}")
    if inputs.get("platform"):
        cmd_parts.append(f"--platform {inputs['platform']}")

    next_cmd = " \\\n    ".join(cmd_parts)
    console.print()
    console.print("─" * 72)
    console.print("  NEXT STEP")
    console.print("─" * 72)
    console.print()
    console.print("  See what the planner would build from these requirements:")
    console.print(f"  [cyan]{next_cmd}[/cyan]")
    console.print()
    console.print("  [dim]The planner decomposes the goal into a task graph (DAG)")
    console.print("  of specialist agents. You can then review, edit, and execute it.[/dim]")


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
