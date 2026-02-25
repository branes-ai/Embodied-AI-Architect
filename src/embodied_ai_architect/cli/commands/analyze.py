"""Model analysis commands."""

import json

import click
import torch
from rich.console import Console
from rich.table import Table

from embodied_ai_architect.agents.model_analyzer import ModelAnalyzerAgent

console = Console()


@click.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.option("--input-shape", help="Model input shape (e.g., 1,3,224,224)")
@click.pass_context
def analyze(ctx, model_path, input_shape):
    """Analyze model architecture and complexity.

    \b
    Examples:
      branes analyze my_model.pt
      branes analyze my_model.pt --input-shape 1,3,224,224
      branes analyze my_model.pt --json
    """
    json_output = ctx.obj.get("json", False)
    quiet = ctx.obj.get("quiet", False)

    try:
        # Load model
        if not quiet and not json_output:
            console.print(f"\n[cyan]Analyzing model:[/cyan] {model_path}")

        model = torch.load(model_path, weights_only=False)
        model.eval()

        # Parse input shape if provided
        if input_shape:
            input_shape_tuple = tuple(map(int, input_shape.split(",")))
        else:
            input_shape_tuple = None

        # Run analysis
        agent = ModelAnalyzerAgent()
        result = agent.execute({"model": model, "input_shape": input_shape_tuple})

        if not result.success:
            raise Exception(result.error)

        # Output results
        if json_output:
            click.echo(json.dumps(result.data, indent=2))
        else:
            console.print(f"\n[bold green]✓[/bold green] Analysis complete\n")

            # Model summary
            console.print("[bold]Model Summary[/bold]")
            console.print(f"  Architecture: {result.data.get('model_type', 'Unknown')}")
            console.print(f"  Total Parameters: {result.data.get('total_parameters', 0):,}")
            console.print(f"  Trainable Parameters: {result.data.get('trainable_parameters', 0):,}")
            console.print(f"  Total Layers: {result.data.get('total_layers', 0)}\n")

            # Layer breakdown
            layer_types = result.data.get("layer_type_counts", {})
            if layer_types:
                console.print("[bold]Layer Breakdown[/bold]")
                table = Table(show_header=True)
                table.add_column("Layer Type", style="cyan")
                table.add_column("Count", style="green", justify="right")

                for layer_type, count in sorted(
                    layer_types.items(), key=lambda x: x[1], reverse=True
                ):
                    table.add_row(layer_type, str(count))

                console.print(table)

            # Operation types
            operation_types = result.data.get("operation_types", [])
            if operation_types:
                console.print(f"\n[bold]Operation Types:[/bold] {', '.join(operation_types)}")

    except FileNotFoundError:
        console.print(f"\n[bold red]❌ Error:[/bold red] Model file not found: {model_path}")
        ctx.exit(1)
    except Exception as e:
        if json_output:
            click.echo(json.dumps({"status": "error", "error": str(e)}))
        else:
            console.print(f"\n[bold red]❌ Error:[/bold red] {e}")
        ctx.exit(1)
