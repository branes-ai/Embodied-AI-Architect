"""Model registry CLI commands."""

import json

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from embodied_ai_architect.registry import (
    ModelRegistry,
    ModelAnalyzer,
    ModelLoadError,
    ModelNotFoundError,
    ModelAlreadyExistsError,
    AnalysisError,
)

console = Console()


def parse_input_shape(shape_str: str) -> tuple[int, ...]:
    """Parse input shape string like '1,3,640,640' to tuple."""
    try:
        return tuple(int(x.strip()) for x in shape_str.split(","))
    except ValueError as e:
        raise click.BadParameter(f"Invalid input shape: {e}")


@click.group()
def model():
    """Manage model registry.

    \b
    The model registry stores metadata about PyTorch models for quick
    querying and reasoning. Models are analyzed once and cached.

    \b
    Examples:
      branes model register yolov8s.pt --name "YOLOv8-Small"
      branes model list --architecture cnn
      branes model show yolov8-small
      branes model analyze custom_model.pt
    """
    pass


@model.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--name", "-n", help="Model name (default: filename)")
@click.option("--input-shape", "-i", help="Input shape for analysis, e.g., '1,3,640,640'")
@click.option("--tags", "-t", multiple=True, help="Tags for filtering (can be repeated)")
@click.option("--description", "-d", help="Model description")
@click.option("--overwrite", is_flag=True, help="Overwrite if model ID exists")
@click.pass_context
def register(ctx, path, name, input_shape, tags, description, overwrite):
    """Register a model in the registry.

    \b
    Analyzes the model file and stores metadata for later querying.
    Supports PyTorch (.pt, .pth), TorchScript (.pt), and ONNX (.onnx) formats.

    \b
    Examples:
      branes model register yolov8s.pt
      branes model register model.pt --name "My Model" --tags perception detection
      branes model register model.pt --input-shape 1,3,640,640
    """
    json_output = ctx.obj.get("json", False)
    quiet = ctx.obj.get("quiet", False)

    try:
        # Parse input shape
        input_shape_tuple = None
        if input_shape:
            input_shape_tuple = parse_input_shape(input_shape)

        if not quiet and not json_output:
            console.print(f"\n[cyan]Analyzing model:[/cyan] {path}")

        # Analyze model
        analyzer = ModelAnalyzer()
        metadata = analyzer.analyze(
            path=path,
            name=name,
            input_shape=input_shape_tuple,
            description=description,
            tags=list(tags) if tags else None,
        )

        # Register in registry
        registry = ModelRegistry()
        metadata = registry.register(metadata, overwrite=overwrite)

        if json_output:
            click.echo(json.dumps(metadata.to_dict(), indent=2))
        else:
            console.print(
                f"\n[bold green]✓[/bold green] Registered model: [cyan]{metadata.id}[/cyan]"
            )
            console.print(f"  Parameters: {metadata.format_parameters()}")
            console.print(f"  Architecture: {metadata.architecture_display}")
            console.print(f"  Format: {metadata.format}")
            if metadata.tags:
                console.print(f"  Tags: {', '.join(metadata.tags)}")

    except ModelAlreadyExistsError as e:
        if json_output:
            click.echo(json.dumps({"status": "error", "error": str(e)}))
        else:
            console.print(f"\n[bold yellow]⚠[/bold yellow] {e}")
            console.print("  Use --overwrite to replace the existing model.")
        ctx.exit(1)
    except (ModelLoadError, AnalysisError) as e:
        if json_output:
            click.echo(json.dumps({"status": "error", "error": str(e)}))
        else:
            console.print(f"\n[bold red]❌ Error:[/bold red] {e}")
        ctx.exit(1)
    except Exception as e:
        if json_output:
            click.echo(json.dumps({"status": "error", "error": str(e)}))
        else:
            console.print(f"\n[bold red]❌ Error:[/bold red] {e}")
        ctx.exit(1)


@model.command("list")
@click.option("--architecture", "-a", help="Filter by architecture type (cnn, transformer, mlp)")
@click.option("--family", "-f", help="Filter by architecture family (yolo, resnet, vit)")
@click.option("--min-params", type=int, help="Minimum parameters")
@click.option("--max-params", type=int, help="Maximum parameters")
@click.option("--tags", "-t", multiple=True, help="Filter by tags (models must have ALL tags)")
@click.pass_context
def list_models(ctx, architecture, family, min_params, max_params, tags):
    """List registered models.

    \b
    Examples:
      branes model list
      branes model list --architecture cnn
      branes model list --max-params 15000000
      branes model list --tags perception --tags detection
    """
    json_output = ctx.obj.get("json", False)

    registry = ModelRegistry()

    # Query with filters
    arch_filter = architecture or family
    models = registry.query(
        min_params=min_params,
        max_params=max_params,
        architecture=arch_filter,
        tags=list(tags) if tags else None,
    )

    if json_output:
        click.echo(json.dumps([m.to_dict() for m in models], indent=2))
        return

    if not models:
        console.print("\n[yellow]No models registered.[/yellow]")
        console.print("  Use 'branes model register <path>' to add models.")
        return

    # Display table
    table = Table(title="Registered Models", show_header=True)
    table.add_column("ID", style="cyan")
    table.add_column("Architecture", style="green")
    table.add_column("Parameters", justify="right")
    table.add_column("FLOPs", justify="right")
    table.add_column("Memory", justify="right")
    table.add_column("Tags", style="dim")

    for m in models:
        table.add_row(
            m.id,
            m.architecture_display,
            m.format_parameters(),
            m.format_flops(),
            f"{m.estimated_memory_mb:.1f} MB",
            ", ".join(m.tags[:3]) + ("..." if len(m.tags) > 3 else ""),
        )

    console.print()
    console.print(table)
    console.print(f"\n[dim]Total: {len(models)} model(s)[/dim]")


@model.command()
@click.argument("model_id")
@click.pass_context
def show(ctx, model_id):
    """Show details of a registered model.

    \b
    Examples:
      branes model show yolov8-small
      branes model show resnet50 --json
    """
    json_output = ctx.obj.get("json", False)

    try:
        registry = ModelRegistry()
        metadata = registry.get(model_id)

        if json_output:
            click.echo(json.dumps(metadata.to_dict(), indent=2))
            return

        # Build detailed panel
        lines = [
            f"[bold]ID:[/bold] {metadata.id}",
            f"[bold]Path:[/bold] {metadata.path}",
            f"[bold]Format:[/bold] {metadata.format}",
            f"[bold]Registered:[/bold] {metadata.registered_at[:19]}",
            "",
            f"[bold]Architecture:[/bold] {metadata.architecture_display}",
            f"[bold]Parameters:[/bold] {metadata.total_parameters:,} ({metadata.format_parameters()})",
            f"[bold]Trainable:[/bold] {metadata.trainable_parameters:,}",
            f"[bold]Est. FLOPs:[/bold] {metadata.format_flops()}",
            f"[bold]Memory:[/bold] {metadata.estimated_memory_mb:.2f} MB",
        ]

        if metadata.input_shape:
            lines.append(f"[bold]Input Shape:[/bold] {metadata.input_shape}")
        if metadata.output_shape:
            lines.append(f"[bold]Output Shape:[/bold] {metadata.output_shape}")

        if metadata.layer_counts:
            lines.append("")
            lines.append("[bold]Layer Counts:[/bold]")
            for layer_type, count in list(metadata.layer_counts.items())[:10]:
                lines.append(f"  {layer_type}: {count}")
            if len(metadata.layer_counts) > 10:
                lines.append(f"  ... and {len(metadata.layer_counts) - 10} more")

        if metadata.description:
            lines.append("")
            lines.append(f"[bold]Description:[/bold] {metadata.description}")

        if metadata.tags:
            lines.append("")
            lines.append(f"[bold]Tags:[/bold] {', '.join(metadata.tags)}")

        console.print()
        console.print(
            Panel(
                "\n".join(lines),
                title=f"[bold cyan]{metadata.name}[/bold cyan]",
                border_style="cyan",
            )
        )

    except ModelNotFoundError as e:
        if json_output:
            click.echo(json.dumps({"status": "error", "error": str(e)}))
        else:
            console.print(f"\n[bold red]❌ Error:[/bold red] {e}")
        ctx.exit(1)


@model.command()
@click.argument("model_id")
@click.option("--force", "-f", is_flag=True, help="Skip confirmation")
@click.pass_context
def remove(ctx, model_id, force):
    """Remove a model from the registry.

    \b
    This only removes the registry entry, not the model file itself.

    \b
    Examples:
      branes model remove yolov8-small
      branes model remove old-model --force
    """
    json_output = ctx.obj.get("json", False)

    try:
        registry = ModelRegistry()

        # Check it exists
        metadata = registry.get(model_id)

        # Confirm unless forced
        if not force and not json_output:
            if not click.confirm(f"Remove '{metadata.name}' ({model_id}) from registry?"):
                console.print("[yellow]Cancelled.[/yellow]")
                return

        registry.remove(model_id)

        if json_output:
            click.echo(json.dumps({"status": "success", "removed": model_id}))
        else:
            console.print(f"\n[bold green]✓[/bold green] Removed model: [cyan]{model_id}[/cyan]")

    except ModelNotFoundError as e:
        if json_output:
            click.echo(json.dumps({"status": "error", "error": str(e)}))
        else:
            console.print(f"\n[bold red]❌ Error:[/bold red] {e}")
        ctx.exit(1)


@model.command("analyze")
@click.argument("path", type=click.Path(exists=True))
@click.option("--input-shape", "-i", help="Input shape for analysis, e.g., '1,3,640,640'")
@click.pass_context
def analyze_model(ctx, path, input_shape):
    """Analyze a model without registering.

    \b
    Useful for inspecting models before deciding to register them.

    \b
    Examples:
      branes model analyze custom_model.pt
      branes model analyze model.onnx --input-shape 1,3,224,224
    """
    json_output = ctx.obj.get("json", False)
    quiet = ctx.obj.get("quiet", False)

    try:
        # Parse input shape
        input_shape_tuple = None
        if input_shape:
            input_shape_tuple = parse_input_shape(input_shape)

        if not quiet and not json_output:
            console.print(f"\n[cyan]Analyzing model:[/cyan] {path}")

        # Analyze model
        analyzer = ModelAnalyzer()
        metadata = analyzer.analyze(
            path=path,
            input_shape=input_shape_tuple,
        )

        if json_output:
            click.echo(json.dumps(metadata.to_dict(), indent=2))
            return

        # Display results
        console.print("\n[bold green]✓[/bold green] Analysis complete\n")

        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Key", style="bold")
        table.add_column("Value")

        table.add_row("Format", metadata.format)
        table.add_row("Architecture", metadata.architecture_display)
        table.add_row(
            "Parameters", f"{metadata.total_parameters:,} ({metadata.format_parameters()})"
        )
        table.add_row("Trainable", f"{metadata.trainable_parameters:,}")
        table.add_row("Est. FLOPs", metadata.format_flops())
        table.add_row("Memory", f"{metadata.estimated_memory_mb:.2f} MB")

        if metadata.input_shape:
            table.add_row("Input Shape", str(metadata.input_shape))
        if metadata.output_shape:
            table.add_row("Output Shape", str(metadata.output_shape))

        console.print(table)

        # Layer counts
        if metadata.layer_counts:
            console.print("\n[bold]Layer Counts:[/bold]")
            layer_table = Table(show_header=True)
            layer_table.add_column("Layer Type", style="cyan")
            layer_table.add_column("Count", justify="right")

            for layer_type, count in list(metadata.layer_counts.items())[:15]:
                layer_table.add_row(layer_type, str(count))

            console.print(layer_table)

    except (ModelLoadError, AnalysisError) as e:
        if json_output:
            click.echo(json.dumps({"status": "error", "error": str(e)}))
        else:
            console.print(f"\n[bold red]❌ Error:[/bold red] {e}")
        ctx.exit(1)
    except Exception as e:
        if json_output:
            click.echo(json.dumps({"status": "error", "error": str(e)}))
        else:
            console.print(f"\n[bold red]❌ Error:[/bold red] {e}")
        ctx.exit(1)


@model.command()
@click.argument("model_id")
@click.option("--name", "-n", help="Update model name")
@click.option("--description", "-d", help="Update description")
@click.option("--add-tag", "-t", multiple=True, help="Add tag(s)")
@click.option("--remove-tag", "-r", multiple=True, help="Remove tag(s)")
@click.pass_context
def update(ctx, model_id, name, description, add_tag, remove_tag):
    """Update model metadata.

    \b
    Examples:
      branes model update yolov8-small --name "YOLOv8 Small"
      branes model update mymodel --add-tag perception --add-tag detection
      branes model update mymodel --description "Updated model for drone perception"
    """
    json_output = ctx.obj.get("json", False)

    try:
        registry = ModelRegistry()
        metadata = registry.get(model_id)

        # Build update dict
        updates = {}
        if name:
            updates["name"] = name
        if description:
            updates["description"] = description

        # Handle tags
        if add_tag or remove_tag:
            current_tags = set(metadata.tags)
            if add_tag:
                current_tags.update(add_tag)
            if remove_tag:
                current_tags -= set(remove_tag)
            updates["tags"] = list(current_tags)

        if not updates:
            if json_output:
                click.echo(json.dumps({"status": "error", "error": "No updates specified"}))
            else:
                console.print("[yellow]No updates specified.[/yellow]")
            return

        metadata = registry.update(model_id, **updates)

        if json_output:
            click.echo(json.dumps(metadata.to_dict(), indent=2))
        else:
            console.print(f"\n[bold green]✓[/bold green] Updated model: [cyan]{model_id}[/cyan]")

    except ModelNotFoundError as e:
        if json_output:
            click.echo(json.dumps({"status": "error", "error": str(e)}))
        else:
            console.print(f"\n[bold red]❌ Error:[/bold red] {e}")
        ctx.exit(1)
