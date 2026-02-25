"""Model Zoo CLI commands.

Provides commands for searching, downloading, and managing models
from the unified model zoo.
"""

import json

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

console = Console()


@click.group()
def zoo():
    """Manage models from the Model Zoo.

    \\b
    Examples:
      # Search for detection models
      branes zoo search --task detection

      # Download a YOLO model
      branes zoo download yolov8n --format onnx

      # List cached models
      branes zoo list --cached

      # Get model info
      branes zoo info yolov8n
    """
    pass


@zoo.command()
@click.option("--task", help="Filter by task (detection, classification, segmentation, pose)")
@click.option("--max-params", type=int, help="Maximum parameters (e.g., 10000000)")
@click.option("--min-accuracy", type=float, help="Minimum accuracy (0.0-1.0)")
@click.option("--provider", help="Filter by provider (ultralytics, torchvision, huggingface)")
@click.option("--benchmarked", is_flag=True, help="Only show models with benchmark data")
@click.option("--query", "-q", help="Search query")
@click.pass_context
def search(ctx, task, max_params, min_accuracy, provider, benchmarked, query):
    """Search for models in the zoo.

    \\b
    Examples:
      branes zoo search --task detection
      branes zoo search --task detection --max-params 5000000
      branes zoo search -q yolo --benchmarked
    """
    json_output = ctx.obj.get("json", False)

    from embodied_ai_architect.model_zoo.providers.base import ModelQuery

    model_query = ModelQuery(
        task=task,
        max_params=max_params,
        min_accuracy=min_accuracy,
        provider=provider,
        benchmarked=benchmarked,
        query=query,
    )

    with console.status("[bold blue]Searching models...[/bold blue]"):
        from embodied_ai_architect.model_zoo.discovery import ModelDiscoveryService

        service = ModelDiscoveryService()
        providers = [provider] if provider else None
        candidates = service.discover(model_query, providers)

    if json_output:
        output = [
            {
                "id": c.id,
                "name": c.name,
                "provider": c.provider,
                "task": c.task,
                "parameters": c.parameters,
                "accuracy": c.accuracy,
                "benchmarked": c.benchmarked,
            }
            for c in candidates
        ]
        click.echo(json.dumps({"models": output}, indent=2))
    else:
        if not candidates:
            console.print("[yellow]No models found matching criteria.[/yellow]")
            return

        table = Table(title=f"Available Models ({len(candidates)} found)")
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="white")
        table.add_column("Task", style="green")
        table.add_column("Params", style="yellow", justify="right")
        table.add_column("Accuracy", style="magenta", justify="right")
        table.add_column("Provider", style="blue")

        for c in candidates:
            table.add_row(
                c.id,
                c.name,
                c.task,
                c.format_params(),
                c.format_accuracy(),
                c.provider,
            )

        console.print(table)
        console.print("\n[dim]Download with:[/dim] branes zoo download <model-id>")


@zoo.command()
@click.argument("model_id")
@click.option(
    "--format",
    "-f",
    "format_",
    default="onnx",
    type=click.Choice(["onnx", "pytorch", "torchscript", "tensorrt", "openvino", "coreml"]),
    help="Model format",
)
@click.option("--provider", help="Provider name (auto-detected if not specified)")
@click.option("--force", is_flag=True, help="Force re-download even if cached")
@click.pass_context
def download(ctx, model_id, format_, provider, force):
    """Download a model from the zoo.

    \\b
    Examples:
      branes zoo download yolov8n
      branes zoo download yolov8s --format onnx
      branes zoo download yolov8n --force
    """
    json_output = ctx.obj.get("json", False)
    quiet = ctx.obj.get("quiet", False)

    from embodied_ai_architect.model_zoo import acquire

    try:
        if not quiet and not json_output:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task(f"[cyan]Downloading {model_id}...", total=None)
                path = acquire(model_id, format=format_, provider=provider, force_download=force)
                progress.update(task, description=f"[green]Downloaded {model_id}")
        else:
            path = acquire(model_id, format=format_, provider=provider, force_download=force)

        if json_output:
            output = {
                "status": "success",
                "model_id": model_id,
                "format": format_,
                "path": str(path),
            }
            click.echo(json.dumps(output, indent=2))
        elif quiet:
            click.echo(str(path))
        else:
            console.print(f"\n[green]✓[/green] Downloaded [cyan]{model_id}[/cyan]")
            console.print(f"  Format: {format_}")
            console.print(f"  Path: {path}")

            # Show file size
            if path.exists():
                size_mb = path.stat().st_size / (1024 * 1024)
                console.print(f"  Size: {size_mb:.1f} MB")

    except ImportError as e:
        if json_output:
            click.echo(json.dumps({"status": "error", "error": str(e)}))
        else:
            console.print(f"[red]Missing dependency:[/red] {e}")
            console.print("[dim]Install the required package and try again.[/dim]")
        ctx.exit(1)
    except Exception as e:
        if json_output:
            click.echo(json.dumps({"status": "error", "error": str(e)}))
        else:
            console.print(f"[red]Error:[/red] {e}")
        ctx.exit(1)


@zoo.command(name="list")
@click.option("--cached", is_flag=True, help="List cached models only")
@click.option("--provider", help="Filter by provider")
@click.option("--format", "-f", "format_", help="Filter by format")
@click.pass_context
def list_models(ctx, cached, provider, format_):
    """List available or cached models.

    \\b
    Examples:
      branes zoo list              # List all available models
      branes zoo list --cached     # List downloaded models
      branes zoo list --provider ultralytics
    """
    json_output = ctx.obj.get("json", False)

    if cached:
        # List cached models
        from embodied_ai_architect.model_zoo.cache import ModelCache
        from embodied_ai_architect.model_zoo.providers.base import ModelFormat

        cache = ModelCache()
        format_enum = ModelFormat(format_.lower()) if format_ else None
        entries = cache.list(provider=provider, format=format_enum)

        if json_output:
            output = [
                {
                    "model_id": e.model_id,
                    "provider": e.provider,
                    "format": e.format,
                    "path": e.path,
                    "size_mb": e.size_bytes / (1024 * 1024),
                    "cached_at": e.cached_at,
                }
                for e in entries
            ]
            click.echo(json.dumps({"cached_models": output}, indent=2))
        else:
            if not entries:
                console.print("[yellow]No cached models found.[/yellow]")
                console.print("[dim]Download models with: branes zoo download <model-id>[/dim]")
                return

            table = Table(title=f"Cached Models ({len(entries)})")
            table.add_column("Model ID", style="cyan")
            table.add_column("Provider", style="blue")
            table.add_column("Format", style="green")
            table.add_column("Size", style="yellow", justify="right")
            table.add_column("Cached At", style="dim")

            total_size = 0
            for e in entries:
                size_mb = e.size_bytes / (1024 * 1024)
                total_size += e.size_bytes
                table.add_row(
                    e.model_id,
                    e.provider,
                    e.format,
                    f"{size_mb:.1f} MB",
                    e.cached_at[:19] if e.cached_at else "unknown",
                )

            console.print(table)
            console.print(f"\n[dim]Total cache size: {total_size / (1024 * 1024):.1f} MB[/dim]")
    else:
        # List available models
        from embodied_ai_architect.model_zoo.discovery import ModelDiscoveryService
        from embodied_ai_architect.model_zoo.providers.base import ModelQuery

        service = ModelDiscoveryService()
        query = ModelQuery(provider=provider) if provider else None
        providers = [provider] if provider else None
        candidates = service.discover(query, providers)

        if json_output:
            output = [
                {
                    "id": c.id,
                    "name": c.name,
                    "provider": c.provider,
                    "task": c.task,
                    "parameters": c.parameters,
                }
                for c in candidates
            ]
            click.echo(json.dumps({"models": output}, indent=2))
        else:
            if not candidates:
                console.print("[yellow]No models available.[/yellow]")
                return

            table = Table(title=f"Available Models ({len(candidates)})")
            table.add_column("ID", style="cyan")
            table.add_column("Name", style="white")
            table.add_column("Task", style="green")
            table.add_column("Params", style="yellow", justify="right")
            table.add_column("Provider", style="blue")

            for c in candidates:
                table.add_row(
                    c.id,
                    c.name,
                    c.task,
                    c.format_params(),
                    c.provider,
                )

            console.print(table)


@zoo.command()
@click.argument("model_id")
@click.pass_context
def info(ctx, model_id):
    """Show detailed information about a model.

    \\b
    Examples:
      branes zoo info yolov8n
      branes zoo info yolov8s-seg
    """
    json_output = ctx.obj.get("json", False)

    from embodied_ai_architect.model_zoo.discovery import ModelDiscoveryService

    service = ModelDiscoveryService()

    # Try to find the model in providers
    model_info = None
    for provider in service._providers.values():
        try:
            model_info = provider.get_model_info(model_id)
            break
        except Exception:
            continue

    if model_info is None:
        if json_output:
            click.echo(json.dumps({"error": f"Model '{model_id}' not found"}))
        else:
            console.print(f"[red]Model '{model_id}' not found[/red]")
        ctx.exit(1)

    if json_output:
        click.echo(json.dumps(model_info, indent=2))
    else:
        console.print(f"\n[bold cyan]{model_info.get('name', model_id)}[/bold cyan]")
        console.print(f"[dim]ID: {model_id}[/dim]\n")

        # Basic info
        console.print("[bold]Basic Information[/bold]")
        console.print(f"  Provider: {model_info.get('provider', 'unknown')}")
        console.print(f"  Task: {model_info.get('task', 'unknown')}")
        console.print(f"  Family: {model_info.get('family', 'unknown')}")
        console.print(f"  Version: {model_info.get('version', 'unknown')}")

        # Size metrics
        params = model_info.get("parameters")
        flops = model_info.get("flops")
        if params or flops:
            console.print("\n[bold]Model Size[/bold]")
            if params:
                if params >= 1_000_000:
                    console.print(f"  Parameters: {params:,} ({params / 1_000_000:.1f}M)")
                else:
                    console.print(f"  Parameters: {params:,}")
            if flops:
                if flops >= 1_000_000_000:
                    console.print(f"  FLOPs: {flops:,} ({flops / 1_000_000_000:.1f}G)")
                else:
                    console.print(f"  FLOPs: {flops:,}")

        # Accuracy
        accuracy_fields = ["map50", "map50_95", "top1", "top5", "accuracy"]
        accuracy_values = {k: v for k, v in model_info.items() if k in accuracy_fields and v}
        if accuracy_values:
            console.print("\n[bold]Accuracy[/bold]")
            for k, v in accuracy_values.items():
                console.print(f"  {k}: {v * 100:.1f}%")

        # Input shape
        input_shape = model_info.get("input_shape")
        if input_shape:
            console.print(f"\n[bold]Input Shape[/bold]: {input_shape}")

        console.print(f"\n[dim]Download with: branes zoo download {model_id}[/dim]")


@zoo.command()
@click.option("--provider", help="Clear only specific provider cache")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation")
@click.pass_context
def clear(ctx, provider, yes):
    """Clear the model cache.

    \\b
    Examples:
      branes zoo clear
      branes zoo clear --provider ultralytics
      branes zoo clear -y
    """
    json_output = ctx.obj.get("json", False)

    from embodied_ai_architect.model_zoo.cache import ModelCache

    cache = ModelCache()

    # Get current cache info
    entries = cache.list(provider=provider)
    total_size = sum(e.size_bytes for e in entries)

    if not entries:
        if json_output:
            click.echo(json.dumps({"status": "success", "cleared": 0}))
        else:
            console.print("[yellow]Cache is already empty.[/yellow]")
        return

    if not yes and not json_output:
        console.print(f"[yellow]This will remove {len(entries)} cached models[/yellow]")
        console.print(f"[dim]Total size: {total_size / (1024 * 1024):.1f} MB[/dim]")
        if not click.confirm("Continue?"):
            ctx.exit(0)

    count = cache.clear(provider)

    if json_output:
        click.echo(
            json.dumps(
                {
                    "status": "success",
                    "cleared": count,
                    "size_freed_mb": total_size / (1024 * 1024),
                }
            )
        )
    else:
        console.print(
            f"[green]✓[/green] Cleared {count} models ({total_size / (1024 * 1024):.1f} MB)"
        )
