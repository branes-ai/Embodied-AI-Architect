"""Secrets management commands."""

import json
import os
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

console = Console()


@click.group()
def secrets():
    """Manage secrets and credentials.

    \b
    Examples:
      # List available secrets (keys only, not values!)
      branes secrets list

      # Set a secret interactively
      branes secrets set ssh_key

      # Validate secrets setup
      branes secrets validate
    """
    pass


@secrets.command()
@click.pass_context
def list(ctx):
    """List available secrets (keys only, not values!)."""
    json_output = ctx.obj.get("json", False)

    # Common secret keys
    secret_keys = [
        ("ssh_key", "SSH private key for remote backends", "EMBODIED_AI_SSH_KEY"),
        ("ssh_host", "SSH host for remote backends", "EMBODIED_AI_SSH_HOST"),
        ("ssh_user", "SSH username for remote backends", "EMBODIED_AI_SSH_USER"),
        ("k8s_kubeconfig", "Kubernetes config for K8s backend", "EMBODIED_AI_K8S_KUBECONFIG"),
    ]

    secrets_info = []
    for key, description, env_var in secret_keys:
        # Check if secret is set (without revealing value!)
        is_set = bool(os.environ.get(env_var))

        # Check if file-based secret exists
        file_path = Path("config/credentials") / key
        file_exists = file_path.exists()

        secrets_info.append(
            {
                "key": key,
                "description": description,
                "env_var": env_var,
                "is_set": is_set or file_exists,
                "source": "environment" if is_set else ("file" if file_exists else "not set"),
            }
        )

    if json_output:
        click.echo(json.dumps({"secrets": secrets_info}, indent=2))
    else:
        table = Table(title="Secrets Configuration", show_header=True)
        table.add_column("Secret Key", style="cyan")
        table.add_column("Description", style="white")
        table.add_column("Status", style="green")
        table.add_column("Source", style="yellow")

        for secret in secrets_info:
            status = "✓ Set" if secret["is_set"] else "✗ Not set"
            status_style = "green" if secret["is_set"] else "red"
            table.add_row(
                secret["key"],
                secret["description"],
                f"[{status_style}]{status}[/{status_style}]",
                secret["source"],
            )

        console.print(table)
        console.print("\n[dim]Note: Secret values are never displayed for security[/dim]")


@secrets.command()
@click.argument("secret_key")
@click.option("--env", is_flag=True, help="Set as environment variable")
@click.option("--file", is_flag=True, help="Set as file in config/credentials/")
@click.pass_context
def set(ctx, secret_key, env, file):
    """Set a secret value interactively."""
    if not env and not file:
        console.print("[yellow]⚠[/yellow] Specify --env or --file")
        ctx.exit(1)

    # Get secret value securely
    secret_value = click.prompt(
        f"Enter value for {secret_key}", hide_input=True, confirmation_prompt=True
    )

    if env:
        console.print("\n[yellow]⚠[/yellow] Environment variable setting not persistent")
        console.print(f"\n[dim]Add to your shell profile:[/dim]")
        console.print(f"export EMBODIED_AI_{secret_key.upper()}=<your-secret>")

    if file:
        # Save to file
        credentials_dir = Path("config/credentials")
        credentials_dir.mkdir(parents=True, exist_ok=True)

        secret_file = credentials_dir / secret_key
        secret_file.write_text(secret_value)
        secret_file.chmod(0o600)  # Secure permissions

        console.print(f"\n[green]✓[/green] Secret saved to: {secret_file}")
        console.print("[yellow]⚠[/yellow] Make sure this directory is in .gitignore!")


@secrets.command()
@click.pass_context
def validate(ctx):
    """Validate secrets setup."""
    json_output = ctx.obj.get("json", False)

    from embodied_ai_architect.security import (
        SecretsManager,
        EnvironmentSecretsProvider,
        FileSecretsProvider,
    )

    # Initialize secrets manager
    secrets = SecretsManager(
        [EnvironmentSecretsProvider(), FileSecretsProvider("config/credentials")]
    )

    # Check common secrets
    checks = []

    # SSH secrets
    ssh_key = secrets.get_secret("ssh_key", required=False)
    ssh_host = secrets.get_secret("ssh_host", required=False)
    ssh_user = secrets.get_secret("ssh_user", required=False)

    ssh_complete = ssh_key and ssh_host and ssh_user
    checks.append(
        {
            "name": "SSH Configuration",
            "status": "✓ Complete" if ssh_complete else "⚠ Incomplete",
            "valid": ssh_complete,
        }
    )

    # Kubernetes secrets
    k8s_kubeconfig = secrets.get_secret("k8s_kubeconfig", required=False)
    checks.append(
        {
            "name": "Kubernetes Configuration",
            "status": "✓ Complete" if k8s_kubeconfig else "⚠ Not configured",
            "valid": bool(k8s_kubeconfig),
        }
    )

    if json_output:
        click.echo(json.dumps({"checks": checks}, indent=2))
    else:
        console.print("\n[bold]Secrets Validation[/bold]\n")

        for check in checks:
            status_style = "green" if check["valid"] else "yellow"
            console.print(f"  [{status_style}]{check['status']}[/{status_style}] {check['name']}")

        console.print("\n[dim]Use 'branes secrets list' to see all secrets[/dim]")
