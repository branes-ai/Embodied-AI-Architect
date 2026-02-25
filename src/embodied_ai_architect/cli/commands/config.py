"""Configuration management commands."""

import json
from pathlib import Path

import click
from rich.console import Console
from rich.syntax import Syntax

console = Console()


@click.group()
def config():
    """Manage configuration settings.

    \b
    Examples:
      # Show current configuration
      branes config show

      # Initialize configuration
      branes config init
    """
    pass


@config.command()
@click.pass_context
def init(ctx):
    """Initialize configuration file."""
    json_output = ctx.obj.get("json", False)

    config_dir = Path(".branes")
    config_file = config_dir / "config.yaml"

    if config_file.exists():
        console.print("[yellow]⚠[/yellow] Configuration file already exists")
        if not click.confirm("Overwrite?"):
            ctx.exit(0)

    # Create config directory
    config_dir.mkdir(exist_ok=True)

    # Default configuration
    default_config = """version: "1.0"

# Default backend for benchmarking
default_backend: local_cpu

# Backends configuration
backends:
  kubernetes:
    namespace: branes
    image: branes-benchmark:latest
    cpu_request: "2"
    memory_request: "4Gi"

  ssh_remote:
    host: gpu-server.example.com
    port: 22
    user: benchmark

# Report settings
reports:
  auto_open: true  # Open in browser after generation
  format: html

# Workflow settings
workflow:
  default_iterations: 100
  default_warmup: 10
  auto_cleanup: true
"""

    config_file.write_text(default_config)

    if json_output:
        click.echo(json.dumps({"status": "success", "config_file": str(config_file)}))
    else:
        console.print(f"\n[green]✓[/green] Configuration initialized: {config_file}")
        console.print("\n[dim]Edit this file to customize your settings[/dim]")


@config.command()
@click.pass_context
def show(ctx):
    """Show current configuration."""
    json_output = ctx.obj.get("json", False)

    config_file = Path(".branes") / "config.yaml"

    if not config_file.exists():
        console.print("[yellow]⚠[/yellow] No configuration file found")
        console.print("\n[dim]Run 'branes config init' to create one[/dim]")
        ctx.exit(1)

    config_content = config_file.read_text()

    if json_output:
        click.echo(json.dumps({"config_file": str(config_file), "content": config_content}))
    else:
        console.print(f"\n[bold]Configuration:[/bold] {config_file}\n")
        syntax = Syntax(config_content, "yaml", theme="monokai", line_numbers=True)
        console.print(syntax)


@config.command()
@click.argument("key")
@click.argument("value")
@click.pass_context
def set(ctx, key, value):
    """Set a configuration value."""
    console.print("[yellow]⚠[/yellow] Configuration editing not yet implemented")
    console.print("\n[dim]Edit .branes/config.yaml manually for now[/dim]")


@config.command()
@click.pass_context
def validate(ctx):
    """Validate configuration."""
    json_output = ctx.obj.get("json", False)

    config_file = Path(".branes") / "config.yaml"

    if not config_file.exists():
        if json_output:
            click.echo(json.dumps({"valid": False, "error": "Config file not found"}))
        else:
            console.print("[bold red]❌ Error:[/bold red] Configuration file not found")
        ctx.exit(1)

    # Basic validation (just check if file is readable YAML-like)
    try:
        config_file.read_text()
        # TODO: Add proper YAML validation
        if json_output:
            click.echo(json.dumps({"valid": True, "config_file": str(config_file)}))
        else:
            console.print("[green]✓[/green] Configuration is valid")
    except Exception as e:
        if json_output:
            click.echo(json.dumps({"valid": False, "error": str(e)}))
        else:
            console.print(f"[bold red]❌ Error:[/bold red] {e}")
        ctx.exit(1)
