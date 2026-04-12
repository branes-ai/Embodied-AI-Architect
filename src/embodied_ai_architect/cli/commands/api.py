"""API server CLI command."""

import click
from rich.console import Console

console = Console()


@click.group(invoke_without_command=True)
@click.pass_context
def api(ctx) -> None:
    """Manage the REST API server.

    \\b
    Examples:
      branes api serve                    # Start on default port 8000
      branes api serve --port 9000        # Custom port
      branes api serve --host 0.0.0.0     # Listen on all interfaces
    """
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@api.command()
@click.option("--host", default="127.0.0.1", help="Host to bind to")
@click.option("--port", default=8000, type=int, help="Port to listen on")
@click.option(
    "--cors-origin",
    multiple=True,
    help="Allowed CORS origins (can specify multiple)",
)
@click.option("--reload", "do_reload", is_flag=True, help="Enable auto-reload for development")
def serve(host: str, port: int, cors_origin: tuple[str, ...], do_reload: bool) -> None:
    """Start the REST API server.

    Serves design session data for the branes-frontend dashboard.
    The API is read-only — it does not modify sessions.

    \\b
    Examples:
      branes api serve
      branes api serve --port 9000 --cors-origin http://localhost:3000
      branes api serve --host 0.0.0.0 --reload
    """
    try:
        import uvicorn
    except ImportError:
        console.print(
            "[red]Missing dependency:[/red] uvicorn\n"
            "[dim]Install with: .venv/bin/pip install uvicorn fastapi[/dim]"
        )
        raise SystemExit(1)

    try:
        from embodied_ai_architect.api import server as api_server
    except ImportError as e:
        console.print(
            f"[red]Missing dependency:[/red] {e}\n"
            "[dim]Install with: .venv/bin/pip install fastapi uvicorn[/dim]"
        )
        raise SystemExit(1)

    cors_origins = list(cors_origin) if cors_origin else None

    # Create the app with user-specified CORS origins.
    # Replace the module-level `app` so both direct and import-string
    # modes use the same configured instance.
    api_server.app = api_server.create_app(cors_origins=cors_origins)

    console.print("[bold cyan]Branes Architect API[/bold cyan]")
    console.print(f"  Server:  http://{host}:{port}")
    console.print(f"  Docs:    http://{host}:{port}/docs")
    console.print(f"  OpenAPI: http://{host}:{port}/openapi.json")
    if cors_origins:
        console.print(f"  CORS:    {', '.join(cors_origins)}")
    console.print()
    console.print("[dim]Press Ctrl+C to stop[/dim]")

    if do_reload:
        # --reload requires import string; module-level app was replaced above
        uvicorn.run(
            "embodied_ai_architect.api.server:app",
            host=host,
            port=port,
            reload=True,
            log_level="info",
        )
    else:
        # Pass app object directly — guaranteed to be the configured instance
        uvicorn.run(
            api_server.app,
            host=host,
            port=port,
            log_level="info",
        )
