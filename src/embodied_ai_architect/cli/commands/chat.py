"""Interactive chat command for the Embodied AI Architect.

Provides a Claude Code-style interactive session where users can
converse with an AI agent that can analyze models, recommend hardware,
and run benchmarks.
"""

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

console = Console()


@click.command()
@click.option(
    "--model",
    default="claude-sonnet-4-20250514",
    help="Claude model to use",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Show detailed tool execution info",
)
@click.pass_context
def chat(ctx, model: str, verbose: bool):
    """Start an interactive AI architect session.

    \b
    Examples:
      branes chat
      branes chat --model claude-sonnet-4-20250514
      branes chat -v

    \b
    In the session, you can ask questions like:
      "Analyze the model in ./models/yolov8n.pt"
      "What hardware can run this at 30fps under 10W?"
      "Benchmark it on my local CPU"
    """
    try:
        from embodied_ai_architect.llm import LLMClient, ArchitectAgent
    except ImportError as e:
        console.print(
            Panel(
                f"[red]Missing dependency:[/red] {e}\n\n"
                "Install with: [cyan]pip install anthropic[/cyan]",
                title="Import Error",
                border_style="red",
            )
        )
        return

    # Check for API key
    import os

    if not os.environ.get("ANTHROPIC_API_KEY"):
        console.print(
            Panel(
                "[yellow]ANTHROPIC_API_KEY not set[/yellow]\n\n"
                "Set your API key:\n"
                "  [cyan]export ANTHROPIC_API_KEY=your-key-here[/cyan]\n\n"
                "Get a key at: https://console.anthropic.com/",
                title="API Key Required",
                border_style="yellow",
            )
        )
        return

    # Initialize agent
    try:
        llm = LLMClient(model=model)
        agent = ArchitectAgent(llm=llm, verbose=verbose)
    except Exception as e:
        console.print(f"[red]Failed to initialize agent:[/red] {e}")
        return

    # Welcome message
    console.print()
    console.print(
        Panel(
            "[bold cyan]Branes Embodied AI Platform[/bold cyan]\n"
            "Interactive design environment for embodied AI systems\n\n"
            "[dim]Commands:[/dim]\n"
            "  [green]exit[/green] or [green]quit[/green] - End session\n"
            "  [green]reset[/green] - Clear conversation history\n"
            "  [green]help[/green] - Show example queries\n\n"
            '[dim]Try:[/dim] "Design a drone perception SoC for VIO at <3W, <20ms"',
            border_style="cyan",
        )
    )
    console.print()

    # Main loop
    while True:
        try:
            # Get user input
            user_input = console.input("[bold green]You:[/bold green] ").strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.lower() in ("exit", "quit", "q"):
                console.print("[dim]Goodbye![/dim]")
                break

            if user_input.lower() == "reset":
                agent.reset()
                console.print("[dim]Conversation reset.[/dim]")
                continue

            if user_input.lower() == "help":
                _show_help()
                continue

            # Enrich with platform context on design-related prompts
            enriched_input = _maybe_enrich_with_platform_context(user_input, console, verbose)

            # Process with agent
            console.print()
            _run_agent_turn(agent, enriched_input, verbose)
            console.print()

        except KeyboardInterrupt:
            console.print("\n[dim]Use 'exit' to quit[/dim]")
            continue
        except EOFError:
            console.print("\n[dim]Goodbye![/dim]")
            break


def _maybe_enrich_with_platform_context(user_input: str, console, verbose: bool) -> str:
    """If the user input matches a platform, prepend context to the message.

    This gives the LLM domain-specific knowledge (typical architecture,
    common pitfalls, regulatory requirements) without the user having to
    provide it. Only activates when a strong match is found.
    """
    try:
        from embodied_ai_architect.platforms.context import (
            get_platform_context_for_goal,
            build_context_prompt,
        )

        ctx = get_platform_context_for_goal(user_input)
        if not ctx:
            return user_input

        context_text = build_context_prompt(ctx)
        if not context_text:
            return user_input

        pid = ctx.get("platform_id", "")
        if verbose:
            console.print(f"  [dim]Platform context loaded: {pid}[/dim]")

        # Prepend context as a system-like block the LLM can use
        return (
            f"[Platform context from registry — use this domain knowledge "
            f"to inform your response]\n\n{context_text}\n\n"
            f"---\n\nUser request: {user_input}"
        )
    except Exception:
        return user_input


def _run_agent_turn(agent, user_input: str, verbose: bool) -> None:
    """Run one turn of agent interaction with live updates.

    Args:
        agent: The ArchitectAgent instance
        user_input: User's message
        verbose: Whether to show detailed info
    """
    Text()
    current_tool = None

    def on_tool_start(name: str, args: dict) -> None:
        nonlocal current_tool
        current_tool = name

        # Show tool execution
        args_preview = ", ".join(f"{k}={_truncate(str(v), 30)}" for k, v in args.items())
        console.print(f"  [dim]▶ {name}({args_preview})[/dim]")

    def on_tool_end(name: str, result: str) -> None:
        # Show brief result
        result_preview = _truncate(result.replace("\n", " "), 60)
        if "Error" in result:
            console.print(f"  [red]✗ {result_preview}[/red]")
        else:
            console.print(f"  [green]✓ {result_preview}[/green]")

    def on_thinking(text: str) -> None:
        if verbose and text.strip():
            console.print(f"  [dim italic]{_truncate(text, 100)}[/dim italic]")

    # Show thinking spinner
    with console.status("[bold blue]Thinking...", spinner="dots"):
        try:
            response = agent.run(
                user_input,
                on_tool_start=on_tool_start,
                on_tool_end=on_tool_end,
                on_thinking=on_thinking,
            )
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            return

    # Print response
    console.print()
    console.print("[bold blue]Architect:[/bold blue]")
    console.print(Markdown(response))


def _show_help() -> None:
    """Show help message."""
    console.print(
        Panel(
            "[bold]Available Commands[/bold]\n\n"
            "[green]exit[/green], [green]quit[/green] - End the session\n"
            "[green]reset[/green] - Clear conversation history\n"
            "[green]help[/green] - Show this message\n\n"
            "[bold]Constraint Checking (Verdict-First)[/bold]\n\n"
            '• "Can ResNet-18 meet 10ms latency on H100?"\n'
            '• "Does MobileNetV2 fit in 512MB memory?"\n'
            '• "Can YOLOv8n run under 15W on Jetson Orin?"\n\n'
            "[bold]Analysis & Comparison[/bold]\n\n"
            '• "Full analysis of ResNet-50 on A100"\n'
            '• "Compare ResNet-18 on H100 vs Jetson Orin AGX"\n'
            '• "Is ResNet-18 compute-bound or memory-bound on A100?"\n\n'
            "[bold]Discovery[/bold]\n\n"
            '• "List available hardware targets"\n'
            '• "What edge GPUs can you analyze?"',
            title="Help",
            border_style="blue",
        )
    )


def _truncate(text: str, max_len: int) -> str:
    """Truncate text to max length."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."
