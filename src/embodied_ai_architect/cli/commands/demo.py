"""Demo command — discover and run example demos from the CLI.

Wraps the 7 demo scripts in examples/ so they're runnable via
`branes demo list`, `branes demo run <name>`, etc.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import click
from rich.console import Console
from rich.table import Table

console = Console()


# ---------------------------------------------------------------------------
# Demo registry
# ---------------------------------------------------------------------------


@dataclass
class DemoEntry:
    """Metadata for a single demo script."""

    number: int
    cli_name: str
    script: str
    title: str
    description: str
    accepts_args: bool = False
    arg_names: List[str] = field(default_factory=list)


DEMOS: List[DemoEntry] = [
    DemoEntry(
        number=1,
        cli_name="soc-designer",
        script="demo_soc_designer.py",
        title="Agentic SoC Designer",
        description="Goal decomposition + full pipeline for a delivery drone SoC.",
        accepts_args=True,
        arg_names=["goal", "power", "latency", "cost", "max_iterations", "llm"],
    ),
    DemoEntry(
        number=2,
        cli_name="dse-pareto",
        script="demo_dse_pareto.py",
        title="Design Space Exploration (Pareto)",
        description="3-way hardware comparison with Pareto front for a warehouse AMR.",
    ),
    DemoEntry(
        number=3,
        cli_name="soc-optimizer",
        script="demo_soc_optimizer.py",
        title="SoC Design Optimizer",
        description="Iterative power convergence loop using LangGraph.",
        accepts_args=True,
        arg_names=["power", "latency", "cost", "max_iterations"],
    ),
    DemoEntry(
        number=4,
        cli_name="kpu-rtl",
        script="demo_kpu_rtl.py",
        title="KPU Micro-architecture + RTL",
        description="KPU config, floorplan check, bandwidth check, and RTL generation.",
    ),
    DemoEntry(
        number=5,
        cli_name="hitl-safety",
        script="demo_hitl_safety.py",
        title="HITL Safety (Surgical Robot)",
        description="Safety-critical surgical robot with IEC 62304 compliance.",
    ),
    DemoEntry(
        number=6,
        cli_name="experience-cache",
        script="demo_experience_cache.py",
        title="Experience Cache Reuse",
        description="Warm-start a second design using cached experience from the first.",
    ),
    DemoEntry(
        number=7,
        cli_name="full-campaign",
        script="demo_full_campaign.py",
        title="Full Campaign (Quadruped Robot)",
        description="4-workload quadruped robot with DSE and full reporting.",
    ),
]

DEMO_MAP: Dict[str, DemoEntry] = {d.cli_name: d for d in DEMOS}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_examples_dir() -> Path:
    """Resolve the examples/ directory.

    Tries the repo root (relative to this source file), then falls back
    to the current working directory.
    """
    # From src/embodied_ai_architect/cli/commands/demo.py -> repo root
    repo_root = Path(__file__).resolve().parent.parent.parent.parent.parent
    candidate = repo_root / "examples"
    if candidate.is_dir():
        return candidate

    cwd_candidate = Path.cwd() / "examples"
    if cwd_candidate.is_dir():
        return cwd_candidate

    raise click.ClickException(
        "Cannot find examples/ directory. " "Run this command from the repository root."
    )


def _load_demo_module(entry: DemoEntry):
    """Load a demo script as a module via importlib without making examples/ a package.

    Catches SystemExit so that scripts like demo_soc_optimizer.py that call
    sys.exit(1) on missing dependencies don't kill the CLI.
    """
    examples_dir = _get_examples_dir()
    script_path = examples_dir / entry.script

    if not script_path.exists():
        raise click.ClickException(f"Demo script not found: {script_path}")

    module_name = entry.script.removesuffix(".py")
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise click.ClickException(f"Cannot load module from {script_path}")

    module = importlib.util.module_from_spec(spec)

    # Temporarily add src/ to sys.path so demo imports resolve
    src_dir = str(Path(__file__).resolve().parent.parent.parent.parent)
    path_added = False
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
        path_added = True

    try:
        spec.loader.exec_module(module)
    except SystemExit as e:
        code = e.code if e.code is not None else 1
        raise click.ClickException(
            f"Demo '{entry.cli_name}' exited during import (code={code}). "
            f"It may have unmet dependencies — try running the script directly:\n"
            f"  python examples/{entry.script}"
        )
    finally:
        if path_added:
            sys.path.remove(src_dir)

    return module


def _build_kwargs(entry: DemoEntry, options: Dict[str, Any], module: Any) -> Dict[str, Any]:
    """Map shared CLI options to each demo's run_demo() signature."""
    kwargs: Dict[str, Any] = {}

    if entry.cli_name == "soc-designer":
        # run_demo(goal, constraints, use_llm, max_iterations=5)
        from embodied_ai_architect.graphs.soc_state import DesignConstraints

        goal = options.get("goal") or module.DEMO_1_GOAL
        base = module.DEMO_1_CONSTRAINTS
        constraints = DesignConstraints(
            max_power_watts=options.get("power") or base.max_power_watts,
            max_latency_ms=options.get("latency") or base.max_latency_ms,
            max_cost_usd=options.get("cost") or base.max_cost_usd,
            target_volume=base.target_volume,
            operating_temp_min_c=base.operating_temp_min_c,
            operating_temp_max_c=base.operating_temp_max_c,
            ip_rating=base.ip_rating,
        )
        kwargs["goal"] = goal
        kwargs["constraints"] = constraints
        kwargs["use_llm"] = options.get("llm", False)
        max_iter = options.get("max_iterations")
        if max_iter is not None:
            kwargs["max_iterations"] = max_iter
        return kwargs

    if entry.cli_name == "soc-optimizer":
        # run_demo(max_power, max_latency, max_cost, max_iterations)
        kwargs["max_power"] = options.get("power") or 5.0
        kwargs["max_latency"] = options.get("latency") or 33.3
        kwargs["max_cost"] = options.get("cost") or 30.0
        kwargs["max_iterations"] = options.get("max_iterations") or 10
        return kwargs

    # All other demos: run_demo() takes no args
    return kwargs


def _execute_demo(
    entry: DemoEntry,
    options: Dict[str, Any],
    json_mode: bool = False,
    quiet_mode: bool = False,
) -> Dict[str, Any]:
    """Load a demo module, call run_demo(), capture output, and return results."""
    t0 = time.perf_counter()

    module = _load_demo_module(entry)
    kwargs = _build_kwargs(entry, options, module)

    run_demo_fn: Optional[Callable] = getattr(module, "run_demo", None)
    if run_demo_fn is None:
        raise click.ClickException(f"Demo '{entry.cli_name}' has no run_demo() function.")

    output_buffer = io.StringIO()
    error = None

    try:
        if json_mode or quiet_mode:
            with contextlib.redirect_stdout(output_buffer):
                run_demo_fn(**kwargs)
        else:
            run_demo_fn(**kwargs)
    except SystemExit:
        error = "Demo called sys.exit()"
    except Exception as e:
        error = f"{type(e).__name__}: {e}"

    elapsed = time.perf_counter() - t0

    return {
        "demo": entry.cli_name,
        "number": entry.number,
        "title": entry.title,
        "elapsed_s": round(elapsed, 2),
        "success": error is None,
        "error": error,
        "output": output_buffer.getvalue() if (json_mode or quiet_mode) else None,
    }


# ---------------------------------------------------------------------------
# Click commands
# ---------------------------------------------------------------------------


@click.group()
def demo():
    """Discover and run platform demos.

    \b
    Examples:
      branes demo list
      branes demo run dse-pareto
      branes demo run soc-optimizer --power 4.0
      branes demo run all
      branes demo info kpu-rtl
    """
    pass


@demo.command("list")
@click.pass_context
def demo_list(ctx):
    """List all available demos."""
    json_mode = ctx.obj.get("json", False)

    if json_mode:
        rows = [
            {
                "number": d.number,
                "name": d.cli_name,
                "title": d.title,
                "description": d.description,
                "accepts_args": d.accepts_args,
            }
            for d in DEMOS
        ]
        click.echo(json.dumps(rows, indent=2))
        return

    table = Table(title="Available Demos", show_lines=False)
    table.add_column("#", style="bold cyan", width=3)
    table.add_column("Name", style="bold")
    table.add_column("Title")
    table.add_column("Args?", justify="center")
    table.add_column("Description", max_width=50)

    for d in DEMOS:
        table.add_row(
            str(d.number),
            d.cli_name,
            d.title,
            "yes" if d.accepts_args else "",
            d.description,
        )

    console.print(table)


@demo.command("info")
@click.argument("name")
@click.pass_context
def demo_info(ctx, name: str):
    """Show detailed info about a specific demo."""
    entry = DEMO_MAP.get(name)
    if entry is None:
        raise click.ClickException(
            f"Unknown demo '{name}'. Run 'branes demo list' to see available demos."
        )

    json_mode = ctx.obj.get("json", False)

    if json_mode:
        data = {
            "number": entry.number,
            "name": entry.cli_name,
            "script": entry.script,
            "title": entry.title,
            "description": entry.description,
            "accepts_args": entry.accepts_args,
            "arg_names": entry.arg_names,
        }
        click.echo(json.dumps(data, indent=2))
        return

    console.print(f"[bold cyan]Demo {entry.number}:[/bold cyan] {entry.title}")
    console.print(f"  Script: examples/{entry.script}")
    console.print(f"  {entry.description}")
    if entry.accepts_args:
        console.print(f"  Options: {', '.join('--' + a for a in entry.arg_names)}")
    else:
        console.print("  Options: (none)")
    console.print(f"\n  Run with: branes demo run {entry.cli_name}")


@demo.command("run")
@click.argument("name")
@click.option("--goal", type=str, default=None, help="Custom design goal (soc-designer).")
@click.option("--power", type=float, default=None, help="Max power budget in watts.")
@click.option("--latency", type=float, default=None, help="Max latency in ms.")
@click.option("--cost", type=float, default=None, help="Max BOM cost in USD.")
@click.option("--max-iterations", type=int, default=None, help="Max optimization iterations.")
@click.option("--llm", is_flag=True, default=False, help="Use LLM planner (soc-designer).")
@click.pass_context
def demo_run(ctx, name: str, goal, power, latency, cost, max_iterations, llm):
    """Run a demo by name, or 'all' to run every demo sequentially."""
    json_mode = ctx.obj.get("json", False)
    quiet_mode = ctx.obj.get("quiet", False)

    options = {
        "goal": goal,
        "power": power,
        "latency": latency,
        "cost": cost,
        "max_iterations": max_iterations,
        "llm": llm,
    }

    if name == "all":
        _run_all_demos(options, json_mode, quiet_mode)
        return

    entry = DEMO_MAP.get(name)
    if entry is None:
        raise click.ClickException(
            f"Unknown demo '{name}'. Run 'branes demo list' to see available demos."
        )

    if not quiet_mode and not json_mode:
        console.print(f"[bold cyan]Running Demo {entry.number}:[/bold cyan] {entry.title}")

    result = _execute_demo(entry, options, json_mode, quiet_mode)

    if json_mode:
        click.echo(json.dumps(result, indent=2))
    elif quiet_mode:
        status = "[green]OK[/green]" if result["success"] else "[red]FAIL[/red]"
        console.print(f"  {entry.cli_name}: {status} ({result['elapsed_s']:.1f}s)")
        if result["error"]:
            console.print(f"    Error: {result['error']}")
    else:
        # Normal mode: output was already printed to stdout by run_demo()
        if result["error"]:
            console.print(f"\n[red]Error:[/red] {result['error']}")
        else:
            console.print(f"\n[dim]Completed in {result['elapsed_s']:.1f}s[/dim]")


def _run_all_demos(
    options: Dict[str, Any],
    json_mode: bool,
    quiet_mode: bool,
) -> None:
    """Run all demos sequentially and print a summary."""
    results = []
    total_t0 = time.perf_counter()

    for entry in DEMOS:
        if not quiet_mode and not json_mode:
            console.print(f"\n[bold cyan]{'=' * 60}[/bold cyan]")
            console.print(f"[bold cyan]Demo {entry.number}:[/bold cyan] {entry.title}")
            console.print(f"[bold cyan]{'=' * 60}[/bold cyan]")

        result = _execute_demo(entry, options, json_mode, quiet_mode)
        results.append(result)

        if quiet_mode and not json_mode:
            status = "[green]OK[/green]" if result["success"] else "[red]FAIL[/red]"
            console.print(
                f"  {entry.number}. {entry.cli_name}: {status} ({result['elapsed_s']:.1f}s)"
            )

    total_elapsed = time.perf_counter() - total_t0

    if json_mode:
        summary = {
            "total_elapsed_s": round(total_elapsed, 2),
            "passed": sum(1 for r in results if r["success"]),
            "failed": sum(1 for r in results if not r["success"]),
            "demos": results,
        }
        click.echo(json.dumps(summary, indent=2))
        return

    # Summary table
    console.print()
    table = Table(title="Demo Run Summary")
    table.add_column("#", style="bold cyan", width=3)
    table.add_column("Demo", style="bold")
    table.add_column("Status", justify="center")
    table.add_column("Time", justify="right")
    table.add_column("Error", max_width=40)

    for r in results:
        status = "[green]OK[/green]" if r["success"] else "[red]FAIL[/red]"
        table.add_row(
            str(r["number"]),
            r["demo"],
            status,
            f"{r['elapsed_s']:.1f}s",
            r.get("error") or "",
        )

    console.print(table)
    passed = sum(1 for r in results if r["success"])
    console.print(f"\n  {passed}/{len(results)} passed in {total_elapsed:.1f}s total")
