"""Backend management commands."""

import json

import click
from rich.console import Console
from rich.table import Table

console = Console()


@click.group()
def backends():
    """Manage benchmark backends.

    \b
    Examples:
      # List available backends
      branes backends list

      # Test backend connection
      branes backends test kubernetes
    """
    pass


@backends.command()
@click.pass_context
def list(ctx):
    """List available backends."""
    json_output = ctx.obj.get("json", False)

    # Check which backends are available
    backends_info = [
        {
            "name": "local_cpu",
            "type": "Local",
            "description": "Run benchmarks on local CPU",
            "status": "✓ Available",
            "requirements": "None",
        }
    ]

    # Check for optional backends
    try:
        import paramiko

        backends_info.append(
            {
                "name": "remote_ssh",
                "type": "Remote",
                "description": "Run benchmarks on remote machine via SSH",
                "status": "✓ Available",
                "requirements": "Configured SSH credentials",
            }
        )
    except ImportError:
        backends_info.append(
            {
                "name": "remote_ssh",
                "type": "Remote",
                "description": "Run benchmarks on remote machine via SSH",
                "status": "✗ Not installed",
                "requirements": "pip install 'branes[remote]'",
            }
        )

    try:
        import kubernetes

        backends_info.append(
            {
                "name": "kubernetes",
                "type": "Cloud",
                "description": "Run benchmarks on Kubernetes cluster",
                "status": "✓ Available",
                "requirements": "Configured kubeconfig",
            }
        )
    except ImportError:
        backends_info.append(
            {
                "name": "kubernetes",
                "type": "Cloud",
                "description": "Run benchmarks on Kubernetes cluster",
                "status": "✗ Not installed",
                "requirements": "pip install 'branes[kubernetes]'",
            }
        )

    if json_output:
        click.echo(json.dumps({"backends": backends_info}, indent=2))
    else:
        table = Table(title="Benchmark Backends", show_header=True)
        table.add_column("Backend", style="cyan")
        table.add_column("Type", style="yellow")
        table.add_column("Description", style="white")
        table.add_column("Status", style="green")
        table.add_column("Requirements", style="dim")

        for backend in backends_info:
            status_style = "green" if "✓" in backend["status"] else "red"
            table.add_row(
                backend["name"],
                backend["type"],
                backend["description"],
                f"[{status_style}]{backend['status']}[/{status_style}]",
                backend["requirements"],
            )

        console.print(table)


@backends.command()
@click.argument("backend_name")
@click.pass_context
def test(ctx, backend_name):
    """Test backend connection."""
    json_output = ctx.obj.get("json", False)

    if backend_name == "local_cpu":
        if json_output:
            click.echo(json.dumps({"backend": "local_cpu", "available": True}))
        else:
            console.print("[green]✓[/green] local_cpu backend is available")
        return

    elif backend_name == "remote_ssh":
        try:
            import paramiko

            console.print(
                "[yellow]⚠[/yellow] SSH backend installed, but connection test not yet implemented"
            )
            console.print("\n[dim]Check your SSH credentials configuration[/dim]")
        except ImportError:
            if json_output:
                click.echo(
                    json.dumps(
                        {"backend": "remote_ssh", "available": False, "error": "Not installed"}
                    )
                )
            else:
                console.print("[red]✗[/red] remote_ssh backend not installed")
                console.print("\n[dim]Install with: pip install 'branes[remote]'[/dim]")
            ctx.exit(1)

    elif backend_name == "kubernetes":
        try:
            from kubernetes import client, config
            from embodied_ai_architect.security import SecretsManager, EnvironmentSecretsProvider

            # Try to load kubeconfig
            secrets = SecretsManager([EnvironmentSecretsProvider()])
            kubeconfig = secrets.get_secret("k8s_kubeconfig", required=False)

            if not kubeconfig:
                if json_output:
                    click.echo(
                        json.dumps(
                            {
                                "backend": "kubernetes",
                                "available": False,
                                "error": "Kubeconfig not configured",
                            }
                        )
                    )
                else:
                    console.print(
                        "[yellow]⚠[/yellow] Kubernetes backend installed, but kubeconfig not configured"
                    )
                    console.print(
                        "\n[dim]Set EMBODIED_AI_K8S_KUBECONFIG environment variable[/dim]"
                    )
                ctx.exit(1)

            # TODO: Actually test K8s connection
            console.print("[green]✓[/green] Kubernetes backend is configured")

        except ImportError:
            if json_output:
                click.echo(
                    json.dumps(
                        {"backend": "kubernetes", "available": False, "error": "Not installed"}
                    )
                )
            else:
                console.print("[red]✗[/red] kubernetes backend not installed")
                console.print("\n[dim]Install with: pip install 'branes[kubernetes]'[/dim]")
            ctx.exit(1)

    else:
        console.print(f"[bold red]❌ Error:[/bold red] Unknown backend: {backend_name}")
        console.print("\n[dim]Available backends: local_cpu, remote_ssh, kubernetes[/dim]")
        ctx.exit(1)


@backends.command()
@click.argument("backend_type")
@click.option("--name", required=True, help="Backend name")
@click.option("--host", help="Host address (for SSH)")
@click.option("--user", help="Username (for SSH)")
@click.pass_context
def add(ctx, backend_type, name, host, user):
    """Add a new backend configuration."""
    console.print("[yellow]⚠[/yellow] Adding backends not yet implemented")
    console.print("\n[dim]Edit .branes/config.yaml manually for now[/dim]")
