"""CLI commands for multi-objective optimization.

Usage:
    branes optimize explore --goal "drone SoC" --power 5 --latency 33
    branes optimize show-front [--top 10]
    branes optimize sensitivity
    branes optimize explain --points 0,3
"""

from __future__ import annotations

import json

import click
from rich.console import Console
from rich.table import Table

console = Console()


@click.group()
def optimize():
    """Multi-objective design space optimization."""
    pass


@optimize.command()
@click.option("--mission", "mission_id", default=None, help="Load constraints from a mission")
@click.option("--goal", "-g", required=False, default=None, help="Design goal description")
@click.option("--power", "-p", type=float, default=None, help="Power budget (watts)")
@click.option("--latency", "-l", type=float, default=None, help="Latency target (ms)")
@click.option("--cost", "-c", type=float, default=None, help="Cost budget (USD)")
@click.option("--area", "-a", type=float, default=None, help="Area budget (mm²)")
@click.option("--fast", is_flag=True, help="Fast mode (reduced evaluations)")
@click.option(
    "--layers", type=str, default="auto", help="Layer selection: auto, map_elites, bayesian, nsga3"
)
@click.option("--workers", type=int, default=8, help="Thread pool size")
@click.option("--json-output", "json_out", is_flag=True, help="Output raw JSON")
@click.pass_context
def explore(ctx, mission_id, goal, power, latency, cost, area, fast, layers, workers, json_out):
    """Explore the design space with multi-objective optimization."""
    from embodied_ai_architect.cli.commands._utils import load_mission_constraints
    from embodied_ai_architect.graphs.moo.design_space import create_soc_design_space
    from embodied_ai_architect.graphs.moo.evaluator import DesignEvaluator
    from embodied_ai_architect.graphs.moo.engine import OptimizationConfig, OptimizationEngine
    from embodied_ai_architect.graphs.moo.map_elites import MAPElitesConfig

    mission, merged = load_mission_constraints(mission_id, power, latency, cost, area)
    if mission_id and not mission:
        console.print(f"[red]Mission '{mission_id}' not found.[/red]")
        ctx.exit(1)
        return
    if mission and not goal:
        goal = mission.goal or "design exploration"
    if not goal:
        console.print("[red]--goal is required (or use --mission).[/red]")
        ctx.exit(1)
        return

    constraints = merged

    ds = create_soc_design_space(constraints)
    evaluator = DesignEvaluator(
        design_space=ds,
        base_state={"constraints": constraints},
        constraint_bounds=ds.constraint_bounds,
    )

    if fast:
        me_config = MAPElitesConfig(n_iterations=20, batch_size=32, initial_population=64)
        config = OptimizationConfig(layers="map_elites", map_elites=me_config, max_workers=workers)
    else:
        config = OptimizationConfig(layers=layers, max_workers=workers)

    engine = OptimizationEngine(ds, evaluator, config)

    if not json_out:
        console.print("\n[bold cyan]Design Space Exploration[/bold cyan]")
        console.print(f"Goal: {goal}")
        if constraints:
            console.print(f"Constraints: {constraints}")
        console.print(f"Layers: {config.layers}")
        console.print()

    with console.status("[bold green]Running optimization...") as status:

        def callback(layer, iteration, evals, metric):
            status.update(
                f"[bold green]{layer}: iteration {iteration}, {evals} evals, metric={metric:.4f}"
            )

        try:
            result = engine.run(callback=callback)
        finally:
            engine.shutdown()

    if json_out:
        click.echo(json.dumps(result.model_dump(), indent=2, default=str))
        return

    # Display results
    console.print("\n[bold green]Optimization Complete[/bold green]")
    console.print(f"Total evaluations: {result.total_evaluations}")
    console.print(f"Layers used: {', '.join(result.layers_used)}")
    console.print(f"Pareto front size: {len(result.pareto_front)}")
    console.print(f"Hypervolume: {result.hypervolume:.4f}")

    if result.atlas:
        console.print(
            f"Atlas coverage: {result.atlas.get('filled_cells', 0)}/{result.atlas.get('total_cells', 0)} "
            f"({result.atlas.get('coverage', 0) * 100:.1f}%)"
        )

    # Display Pareto front table
    if result.pareto_front:
        table = Table(title="Pareto Front (Top 10)")
        table.add_column("#", style="dim")
        table.add_column("Process", style="cyan")
        table.add_column("Clock", style="cyan")
        table.add_column("Power (W)", style="green")
        table.add_column("Latency (ms)", style="yellow")
        table.add_column("Area (mm²)", style="blue")
        table.add_column("Cost ($)", style="red")

        for i, point in enumerate(result.pareto_front[:10]):
            objs = point.get("objectives", {})
            params = point.get("design_params", {})
            knee = " *" if point == result.knee_point else ""
            table.add_row(
                f"{i}{knee}",
                f"{params.get('process_nm', '?')}nm",
                f"{params.get('clock_mhz', '?'):.0f}MHz",
                f"{objs.get('power_watts', 0):.2f}",
                f"{objs.get('latency_ms', 0):.2f}",
                f"{objs.get('area_mm2', 0):.2f}",
                f"{objs.get('cost_usd', 0):.2f}",
            )

        console.print(table)
        if result.knee_point:
            console.print("[dim]* = knee point (best balance)[/dim]")

    # Save result for other subcommands
    _save_result(result.model_dump())


@optimize.command("show-front")
@click.option("--top", type=int, default=10, help="Number of designs to show")
def show_front(top):
    """Show the Pareto front from the last exploration."""
    result = _load_result()
    if not result:
        console.print("[red]No exploration results found. Run 'optimize explore' first.[/red]")
        return

    front = result.get("pareto_front", [])[:top]
    if not front:
        console.print("[yellow]Empty Pareto front.[/yellow]")
        return

    table = Table(title=f"Pareto Front (Top {top})")
    table.add_column("#", style="dim")
    table.add_column("Process")
    table.add_column("Power (W)")
    table.add_column("Latency (ms)")
    table.add_column("Area (mm²)")
    table.add_column("Cost ($)")

    for i, p in enumerate(front):
        objs = p.get("objectives", {})
        params = p.get("design_params", {})
        table.add_row(
            str(i),
            f"{params.get('process_nm', '?')}nm",
            f"{objs.get('power_watts', 0):.2f}",
            f"{objs.get('latency_ms', 0):.2f}",
            f"{objs.get('area_mm2', 0):.2f}",
            f"{objs.get('cost_usd', 0):.2f}",
        )
    console.print(table)


@optimize.command()
def sensitivity():
    """Show parameter sensitivity from the last exploration."""
    result = _load_result()
    if not result:
        console.print("[red]No exploration results found.[/red]")
        return

    sens = result.get("sensitivity", {})
    if not sens:
        console.print(
            "[yellow]No sensitivity data available (requires Bayesian optimization layer).[/yellow]"
        )
        return

    for obj_name, params in sens.items():
        table = Table(title=f"Sensitivity: {obj_name}")
        table.add_column("Parameter")
        table.add_column("Lengthscale")
        table.add_column("Importance")

        for param_name, data in params.items():
            table.add_row(
                param_name,
                f"{data.get('lengthscale', 0):.4f}",
                f"{data.get('importance', 0):.4f}",
            )
        console.print(table)


@optimize.command()
@click.option(
    "--points", "-p", required=True, help="Two point indices, comma-separated (e.g., 0,3)"
)
def explain(points):
    """Explain tradeoff between two Pareto-front designs."""
    result = _load_result()
    if not result:
        console.print("[red]No exploration results found.[/red]")
        return

    try:
        idx_a, idx_b = [int(x.strip()) for x in points.split(",")]
    except ValueError:
        console.print("[red]Invalid points format. Use: --points 0,3[/red]")
        return

    front = result.get("pareto_front", [])
    if idx_a >= len(front) or idx_b >= len(front):
        console.print(f"[red]Index out of range. Front has {len(front)} points.[/red]")
        return

    pa, pb = front[idx_a], front[idx_b]

    console.print(f"\n[bold]Tradeoff: Design #{idx_a} vs Design #{idx_b}[/bold]\n")

    table = Table(title="Objective Changes")
    table.add_column("Objective")
    table.add_column(f"Design #{idx_a}")
    table.add_column(f"Design #{idx_b}")
    table.add_column("Delta")
    table.add_column("Change %")

    for name in ["power_watts", "latency_ms", "area_mm2", "cost_usd"]:
        va = pa.get("objectives", {}).get(name, 0)
        vb = pb.get("objectives", {}).get(name, 0)
        delta = vb - va
        pct = (delta / va * 100) if va else 0
        style = "green" if delta < 0 else "red" if delta > 0 else ""
        table.add_row(
            name,
            f"{va:.2f}",
            f"{vb:.2f}",
            f"{delta:+.2f}",
            f"{pct:+.1f}%",
            style=style,
        )
    console.print(table)

    # Parameter changes
    da = pa.get("design_params", {})
    db = pb.get("design_params", {})
    changes = {
        k: (da.get(k), db.get(k))
        for k in set(list(da.keys()) + list(db.keys()))
        if da.get(k) != db.get(k)
    }

    if changes:
        ptable = Table(title="Parameter Changes")
        ptable.add_column("Parameter")
        ptable.add_column(f"Design #{idx_a}")
        ptable.add_column(f"Design #{idx_b}")
        for k, (va, vb) in changes.items():
            ptable.add_row(k, str(va), str(vb))
        console.print(ptable)


@optimize.command()
@click.argument("description")
@click.option("--max-iterations", "-n", type=int, default=3, help="Max reasoning iterations")
@click.option("--use-llm", is_flag=True, help="Use Claude for reasoning (requires API key)")
@click.option("--json-output", "json_out", is_flag=True, help="Output raw JSON")
def mission(description, max_iterations, use_llm, json_out):
    """End-to-end: NL mission → optimized design.

    Runs the full agentic optimization loop: decompose → formulate →
    optimize → evaluate → reason → (iterate | recommend).

    Example:
        branes optimize mission "Drone perception at 5 m/s within 10W"
    """
    from embodied_ai_architect.graphs.optimization_loop import run_optimization_loop

    if not json_out:
        console.print("\n[bold cyan]Agentic Optimization Loop[/bold cyan]")
        console.print(f"Mission: {description}")
        console.print(f"Max iterations: {max_iterations}")
        console.print(f"LLM reasoning: {'enabled' if use_llm else 'heuristic'}")
        console.print()

    with console.status("[bold green]Running optimization loop..."):
        result = run_optimization_loop(
            mission_description=description,
            max_iterations=max_iterations,
            llm_available=use_llm,
        )

    if json_out:
        # Filter to serializable fields
        output = {
            "mission": result.get("mission_description", ""),
            "platform": result.get("platform", ""),
            "mission_type": result.get("mission_type", ""),
            "iterations": result.get("iteration", 0) + 1,
            "total_evaluations": result.get("total_evaluations", 0),
            "hypervolume": result.get("hypervolume", 0),
            "recommendation": result.get("recommendation", {}),
            "convergence_history": result.get("convergence_history", []),
            "pareto_front_size": len(result.get("pareto_front", [])),
            "research_citations": result.get("research_citations", []),
        }
        click.echo(json.dumps(output, indent=2, default=str))
        return

    # Display report
    report = result.get("final_report", "")
    if report:
        console.print(report)
    else:
        console.print("[yellow]No report generated.[/yellow]")

    # Show convergence history
    conv = result.get("convergence_history", [])
    if conv:
        console.print()
        table = Table(title="Convergence History")
        table.add_column("Iteration", style="dim")
        table.add_column("Hypervolume", style="cyan")
        table.add_column("Pareto Size", style="green")
        table.add_column("Evaluations", style="yellow")
        for entry in conv:
            table.add_row(
                str(entry.get("iteration", 0)),
                f"{entry.get('hypervolume', 0):.4f}",
                str(entry.get("pareto_size", 0)),
                str(entry.get("total_evals", 0)),
            )
        console.print(table)

    # Show errors if any
    errors = result.get("errors", [])
    if errors:
        console.print("\n[red]Errors:[/red]")
        for err in errors:
            console.print(f"  - {err}")

    # Save the pareto front for other subcommands
    if result.get("pareto_front"):
        _save_result(
            {
                "pareto_front": result["pareto_front"],
                "hypervolume": result.get("hypervolume", 0),
                "knee_point": result.get("knee_point"),
            }
        )


def _save_result(result: dict) -> None:
    """Save result to a temp file for other subcommands."""
    import tempfile
    import os

    path = os.path.join(tempfile.gettempdir(), "branes_moo_result.json")
    with open(path, "w") as f:
        json.dump(result, f, default=str)


def _load_result() -> dict | None:
    """Load result from temp file."""
    import tempfile
    import os

    path = os.path.join(tempfile.gettempdir(), "branes_moo_result.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None
