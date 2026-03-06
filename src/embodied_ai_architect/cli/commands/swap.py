"""CLI commands for system-level SWaP-C analysis.

Usage:
    branes swap estimate --area 50 --power 5 --process 28
    branes swap bom --area 50 --power 5 --process 28 --package FCBGA
    branes swap check --area 50 --power 5 --process 28 --max-weight 500
    branes swap compare --area 50 --power 5 --process 28 \
        --left "QFN,passive,abs_plastic" --right "FCBGA,active_fan,aluminum"
    branes swap explore -g "drone SoC" --power 10 --weight 500 --fast
    branes swap show-front [--top 10]
    branes swap explain --points 0,3
"""

from __future__ import annotations

import json
import sys
from typing import Any

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


# ---------------------------------------------------------------------------
# Shared option decorators
# ---------------------------------------------------------------------------


def _common_soc_options(f):
    """Shared Click options for SoC + packaging parameters."""
    f = click.option("--area", "-a", type=float, required=True, help="Die area (mm²)")(f)
    f = click.option("--power", "-p", type=float, required=True, help="SoC TDP (watts)")(f)
    f = click.option(
        "--process", type=int, default=28, show_default=True, help="Process node (nm)"
    )(f)
    f = click.option(
        "--package",
        type=click.Choice(["QFN", "BGA", "FCBGA", "WLCSP"], case_sensitive=True),
        default="BGA",
        show_default=True,
        help="Package type",
    )(f)
    f = click.option(
        "--cooling",
        type=click.Choice(["passive", "active_fan", "liquid"], case_sensitive=True),
        default="passive",
        show_default=True,
        help="Cooling type",
    )(f)
    f = click.option(
        "--enclosure",
        type=click.Choice(["aluminum", "abs_plastic", "magnesium"], case_sensitive=True),
        default="aluminum",
        show_default=True,
        help="Enclosure material",
    )(f)
    f = click.option(
        "--volume",
        "prod_volume",
        type=int,
        default=10000,
        show_default=True,
        help="Production volume",
    )(f)
    f = click.option(
        "--layers", "layer_count", type=int, default=4, show_default=True, help="PCB layer count"
    )(f)
    f = click.option(
        "--connectors", type=int, default=0, show_default=True, help="Board connectors"
    )(f)
    f = click.option(
        "--ambient-temp",
        type=float,
        default=40.0,
        show_default=True,
        help="Ambient temperature (°C)",
    )(f)
    return f


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _estimate_and_thermal(
    area: float,
    power: float,
    process: int,
    package: str,
    cooling: str,
    enclosure: str,
    prod_volume: int,
    layer_count: int,
    connectors: int,
    ambient_temp: float,
):
    """Run BOM estimation and thermal analysis, returning (bom, thermal)."""
    from embodied_ai_architect.graphs.physical_estimators import (
        compute_thermal_feasibility,
        estimate_system_bom,
    )

    bom = estimate_system_bom(
        soc_area_mm2=area,
        soc_power_watts=power,
        process_nm=process,
        package_type=package,
        cooling_type=cooling,
        enclosure_material=enclosure,
        volume=prod_volume,
        layer_count=layer_count,
        connector_count=connectors,
    )
    thermal = compute_thermal_feasibility(bom, power, ambient_temp)
    return bom, thermal


def _save_swap_result(result: dict) -> None:
    """Save explore result to a temp file for show-front / explain."""
    import os
    import tempfile

    path = os.path.join(tempfile.gettempdir(), "branes_swap_result.json")
    with open(path, "w") as f:
        json.dump(result, f, default=str)


def _load_swap_result() -> dict | None:
    """Load explore result from temp file."""
    import os
    import tempfile

    path = os.path.join(tempfile.gettempdir(), "branes_swap_result.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _parse_config(config_str: str) -> tuple[str, str, str]:
    """Parse 'package,cooling,enclosure' string."""
    parts = [p.strip() for p in config_str.split(",")]
    if len(parts) != 3:
        raise click.BadParameter(f"Expected 'package,cooling,enclosure', got '{config_str}'")
    return parts[0], parts[1], parts[2]


def _read_spec_budgets(spec_name: str) -> dict[str, float]:
    """Extract SWaP-C budget constraints from a named spec."""
    from embodied_ai_architect.specs.store import SpecStore

    store = SpecStore()
    spec = store.get(spec_name)
    budgets: dict[str, float] = {}

    # Power budget from power subsystem
    if spec.power and spec.power.power_budget_watts:
        budgets["max_power_watts"] = spec.power.power_budget_watts
    elif spec.power and spec.power.compute_power_watts:
        budgets["max_power_watts"] = spec.power.compute_power_watts

    # Extract from success criteria constraints
    for criterion in spec.constraints:
        metric = criterion.metric
        target = criterion.target
        op = criterion.operator
        # Map constraint metrics to SWaP-C budget keys
        if op in ("lt", "lte"):
            if metric == "latency_ms":
                budgets["max_latency_ms"] = target
            elif metric == "power_watts":
                budgets["max_power_watts"] = target
            elif metric == "cost_usd":
                budgets["max_cost_usd"] = target
            elif metric == "weight_grams":
                budgets["max_weight_grams"] = target
            elif metric == "volume_cm3":
                budgets["max_volume_cm3"] = target

    return budgets


# ---------------------------------------------------------------------------
# Command group
# ---------------------------------------------------------------------------


@click.group()
def swap():
    """System-level SWaP-C (Size, Weight, Power, Cost) analysis."""
    pass


# ---------------------------------------------------------------------------
# 1. estimate
# ---------------------------------------------------------------------------


@swap.command()
@_common_soc_options
@click.option("--json-output", "json_out", is_flag=True, help="Output JSON")
@click.pass_context
def estimate(
    ctx,
    area,
    power,
    process,
    package,
    cooling,
    enclosure,
    prod_volume,
    layer_count,
    connectors,
    ambient_temp,
    json_out,
):
    """Quick single-point SWaP-C estimation."""
    bom, thermal = _estimate_and_thermal(
        area,
        power,
        process,
        package,
        cooling,
        enclosure,
        prod_volume,
        layer_count,
        connectors,
        ambient_temp,
    )

    weight = bom.total_weight_grams()
    volume = bom.total_volume_cm3()
    cost = bom.total_cost_usd()
    dims = bom.bounding_dimensions_mm()
    tj = thermal["junction_temp_c"]
    margin = thermal["margin_c"]
    feasible = thermal["feasible"]

    if json_out or ctx.obj.get("json", False):
        click.echo(
            json.dumps(
                {
                    "weight_grams": round(weight, 2),
                    "volume_cm3": round(volume, 2),
                    "cost_usd": round(cost, 2),
                    "dimensions_mm": {
                        "length": round(dims.length_mm, 1),
                        "width": round(dims.width_mm, 1),
                        "height": round(dims.height_mm, 1),
                    },
                    "thermal": thermal,
                },
                indent=2,
            )
        )
        return

    thermal_icon = "[green]✓[/green]" if feasible else "[red]✗[/red]"
    content = (
        f"  Weight:    {weight:.1f} g\n"
        f"  Volume:    {volume:.1f} cm³\n"
        f"  Cost:      ${cost:.2f}\n"
        f"  Dims:      {dims.length_mm:.0f}×{dims.width_mm:.0f}×{dims.height_mm:.0f} mm\n"
        f"  Thermal:   Tj={tj:.0f}°C (margin: {margin:.0f}°C) {thermal_icon}"
    )
    console.print(
        Panel(
            content,
            title=f"SWaP-C Estimate: {area:.0f}mm² / {process}nm / {package} / {cooling}",
            border_style="cyan",
        )
    )


# ---------------------------------------------------------------------------
# 2. bom
# ---------------------------------------------------------------------------


@swap.command()
@_common_soc_options
@click.option("--json-output", "json_out", is_flag=True, help="Output JSON")
@click.pass_context
def bom(
    ctx,
    area,
    power,
    process,
    package,
    cooling,
    enclosure,
    prod_volume,
    layer_count,
    connectors,
    ambient_temp,
    json_out,
):
    """Detailed BOM breakdown with per-component weight/volume/cost."""
    sys_bom, _ = _estimate_and_thermal(
        area,
        power,
        process,
        package,
        cooling,
        enclosure,
        prod_volume,
        layer_count,
        connectors,
        ambient_temp,
    )

    rows = sys_bom.summary_table()

    if json_out or ctx.obj.get("json", False):
        click.echo(json.dumps(rows, indent=2))
        return

    table = Table(title="System BOM Breakdown")
    table.add_column("Component", style="cyan")
    table.add_column("Level", style="dim")
    table.add_column("Weight (g)", justify="right", style="green")
    table.add_column("Vol (cm³)", justify="right", style="blue")
    table.add_column("Cost ($)", justify="right", style="red")

    for row in rows:
        table.add_row(
            row["name"],
            row["level"],
            f"{row['weight_grams']:.2f}",
            f"{row['volume_cm3']:.2f}",
            f"{row['cost_usd']:.2f}",
        )

    # Totals
    total_w = sys_bom.total_weight_grams()
    total_v = sys_bom.total_volume_cm3()
    total_c = sys_bom.total_cost_usd()
    table.add_section()
    table.add_row(
        "TOTAL",
        "",
        f"{total_w:.2f}",
        f"{total_v:.2f}",
        f"{total_c:.2f}",
        style="bold",
    )

    console.print(table)


# ---------------------------------------------------------------------------
# 3. check
# ---------------------------------------------------------------------------


@swap.command()
@_common_soc_options
@click.option("--max-weight", type=float, default=None, help="Weight budget (grams)")
@click.option("--max-volume", type=float, default=None, help="Volume budget (cm³)")
@click.option("--max-power", type=float, default=None, help="Power budget (watts)")
@click.option("--max-cost", type=float, default=None, help="Cost budget (USD)")
@click.option("--max-latency", type=float, default=None, help="Latency budget (ms)")
@click.option("--from-spec", "spec_name", type=str, default=None, help="Read budgets from a spec")
@click.option("--json-output", "json_out", is_flag=True, help="Output JSON")
@click.pass_context
def check(
    ctx,
    area,
    power,
    process,
    package,
    cooling,
    enclosure,
    prod_volume,
    layer_count,
    connectors,
    ambient_temp,
    max_weight,
    max_volume,
    max_power,
    max_cost,
    max_latency,
    spec_name,
    json_out,
):
    """Scorecard against SWaP-C budgets. Exit code 1 if FAIL."""
    from embodied_ai_architect.graphs.swap_report import assess_design_point

    # Build constraints from explicit flags
    constraints: dict[str, float] = {}
    if max_weight is not None:
        constraints["max_weight_grams"] = max_weight
    if max_volume is not None:
        constraints["max_volume_cm3"] = max_volume
    if max_power is not None:
        constraints["max_power_watts"] = max_power
    if max_cost is not None:
        constraints["max_cost_usd"] = max_cost
    if max_latency is not None:
        constraints["max_latency_ms"] = max_latency

    # Merge spec budgets (explicit flags take precedence)
    if spec_name:
        spec_budgets = _read_spec_budgets(spec_name)
        for k, v in spec_budgets.items():
            constraints.setdefault(k, v)

    bom_obj, thermal = _estimate_and_thermal(
        area,
        power,
        process,
        package,
        cooling,
        enclosure,
        prod_volume,
        layer_count,
        connectors,
        ambient_temp,
    )

    weight = bom_obj.total_weight_grams()
    volume = bom_obj.total_volume_cm3()
    cost = bom_obj.total_cost_usd()

    design = {
        "objectives": {
            "power_watts": power,
            "latency_ms": 0.0,  # not estimated in single-point mode
            "area_mm2": area,
            "cost_usd": cost,
            "weight_grams": weight,
            "volume_cm3": volume,
        },
        "design_params": {
            "process_nm": process,
            "package_type": package,
            "cooling_type": cooling,
            "enclosure_material": enclosure,
        },
    }

    scorecard = assess_design_point(
        design=design,
        bom_summary=bom_obj.summary_table(),
        constraints=constraints,
        thermal_data=thermal,
    )

    if json_out or ctx.obj.get("json", False):
        click.echo(json.dumps(scorecard.model_dump(), indent=2, default=str))
        if scorecard.overall_verdict == "FAIL":
            sys.exit(1)
        return

    # Rich scorecard table
    table = Table(title="SWaP-C Scorecard")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    table.add_column("Budget", justify="right")
    table.add_column("Util", justify="right")
    table.add_column("Verdict")

    for m in scorecard.metrics:
        if m.budget is None and m.value == 0.0:
            continue  # skip metrics with no data and no budget
        budget_str = f"{m.budget:.2f}" if m.budget is not None else "-"
        util_str = f"{m.utilization_pct:.0f}%" if m.utilization_pct is not None else "-"
        verdict_style = (
            "green" if m.verdict == "PASS" else "red" if m.verdict == "FAIL" else "yellow"
        )
        table.add_row(
            m.name,
            f"{m.value:.2f}",
            budget_str,
            util_str,
            f"[{verdict_style}]{m.verdict}[/{verdict_style}]",
        )

    console.print(table)

    # Thermal line
    tj = scorecard.thermal.junction_temp_c
    max_tj = scorecard.thermal.max_junction_temp_c
    thermal_verdict = "PASS" if scorecard.thermal.feasible else "FAIL"
    thermal_style = "green" if scorecard.thermal.feasible else "red"
    console.print(
        f" Thermal: Tj={tj:.0f}°C (max {max_tj:.0f}°C), "
        f"{cooling} cooling — [{thermal_style}]{thermal_verdict}[/{thermal_style}]"
    )

    overall_style = (
        "green"
        if scorecard.overall_verdict == "PASS"
        else "red" if scorecard.overall_verdict == "FAIL" else "yellow"
    )
    console.print(f" Overall: [{overall_style}]{scorecard.overall_verdict}[/{overall_style}]")

    if scorecard.overall_verdict == "FAIL":
        sys.exit(1)


# ---------------------------------------------------------------------------
# 4. explore
# ---------------------------------------------------------------------------


@swap.command()
@click.option("--goal", "-g", required=True, help="Design goal description")
@click.option("--power", "-p", type=float, default=None, help="Max power (W)")
@click.option("--latency", "-l", type=float, default=None, help="Max latency (ms)")
@click.option("--cost", "-c", type=float, default=None, help="Max cost (USD)")
@click.option("--area", "-a", type=float, default=None, help="Max area (mm²)")
@click.option("--weight", "-w", type=float, default=None, help="Max weight (g)")
@click.option("--volume", type=float, default=None, help="Max volume (cm³)")
@click.option("--fast", is_flag=True, help="Reduced evaluations")
@click.option(
    "--layers",
    type=str,
    default="auto",
    help="Optimizer layer selection: auto, map_elites, bayesian, nsga3",
)
@click.option("--workers", type=int, default=8, help="Thread pool size")
@click.option("--from-spec", "spec_name", type=str, default=None, help="Read constraints from spec")
@click.option("--json-output", "json_out", is_flag=True, help="Output raw JSON")
@click.pass_context
def explore(
    ctx,
    goal,
    power,
    latency,
    cost,
    area,
    weight,
    volume,
    fast,
    layers,
    workers,
    spec_name,
    json_out,
):
    """6-objective design space exploration with SWaP-C."""
    from embodied_ai_architect.graphs.moo.design_space import create_swap_design_space
    from embodied_ai_architect.graphs.moo.engine import OptimizationConfig, OptimizationEngine
    from embodied_ai_architect.graphs.moo.evaluator import SWaPCEvaluator
    from embodied_ai_architect.graphs.moo.map_elites import MAPElitesConfig

    constraints: dict[str, Any] = {}
    if power is not None:
        constraints["max_power_watts"] = power
    if latency is not None:
        constraints["max_latency_ms"] = latency
    if cost is not None:
        constraints["max_cost_usd"] = cost
    if area is not None:
        constraints["max_area_mm2"] = area
    if weight is not None:
        constraints["max_weight_grams"] = weight
    if volume is not None:
        constraints["max_volume_cm3"] = volume

    if spec_name:
        spec_budgets = _read_spec_budgets(spec_name)
        for k, v in spec_budgets.items():
            constraints.setdefault(k, v)

    ds = create_swap_design_space(constraints)
    evaluator = SWaPCEvaluator(
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
        console.print("\n[bold cyan]SWaP-C Design Space Exploration[/bold cyan]")
        console.print(f"Goal: {goal}")
        if constraints:
            console.print(f"Constraints: {constraints}")
        console.print("Objectives: 6 (power, latency, area, cost, weight, volume)")
        console.print()

    with console.status("[bold green]Running 6-objective optimization...") as status:

        def callback(layer, iteration, evals, metric):
            status.update(
                f"[bold green]{layer}: iteration {iteration}, "
                f"{evals} evals, metric={metric:.4f}"
            )

        try:
            result = engine.run(callback=callback)
        finally:
            engine.shutdown()

    if json_out or ctx.obj.get("json", False):
        click.echo(json.dumps(result.model_dump(), indent=2, default=str))
        _save_swap_result(result.model_dump())
        return

    # Display results
    console.print("\n[bold green]Optimization Complete[/bold green]")
    console.print(f"Total evaluations: {result.total_evaluations}")
    console.print(f"Layers used: {', '.join(result.layers_used)}")
    console.print(f"Pareto front size: {len(result.pareto_front)}")
    console.print(f"Hypervolume: {result.hypervolume:.4f}")

    if result.pareto_front:
        _print_swap_front(
            result.pareto_front[:10], result.knee_point, title="Pareto Front (Top 10)"
        )

    _save_swap_result(result.model_dump())


# ---------------------------------------------------------------------------
# 5. show-front
# ---------------------------------------------------------------------------


@swap.command("show-front")
@click.option("--top", type=int, default=10, help="Number of designs to show")
def show_front(top):
    """Show the Pareto front from the last explore run."""
    result = _load_swap_result()
    if not result:
        console.print("[red]No SWaP-C results found. Run 'swap explore' first.[/red]")
        return

    front = result.get("pareto_front", [])[:top]
    if not front:
        console.print("[yellow]Empty Pareto front.[/yellow]")
        return

    knee = result.get("knee_point")
    _print_swap_front(front, knee, title=f"SWaP-C Pareto Front (Top {top})")


# ---------------------------------------------------------------------------
# 6. compare
# ---------------------------------------------------------------------------


@swap.command()
@click.option("--area", "-a", type=float, required=True, help="Die area (mm²)")
@click.option("--power", "-p", type=float, required=True, help="SoC TDP (watts)")
@click.option("--process", type=int, default=28, show_default=True, help="Process node (nm)")
@click.option("--left", required=True, help="Left config: package,cooling,enclosure")
@click.option("--right", required=True, help="Right config: package,cooling,enclosure")
@click.option("--json-output", "json_out", is_flag=True, help="Output JSON")
@click.pass_context
def compare(ctx, area, power, process, left, right, json_out):
    """Side-by-side comparison of two packaging/cooling configurations."""
    from embodied_ai_architect.graphs.physical_estimators import (
        compute_thermal_feasibility,
        estimate_system_bom,
    )

    l_pkg, l_cool, l_enc = _parse_config(left)
    r_pkg, r_cool, r_enc = _parse_config(right)

    l_bom = estimate_system_bom(area, power, process, l_pkg, l_cool, l_enc)
    r_bom = estimate_system_bom(area, power, process, r_pkg, r_cool, r_enc)
    l_thermal = compute_thermal_feasibility(l_bom, power)
    r_thermal = compute_thermal_feasibility(r_bom, power)

    l_data = {
        "label": f"{l_pkg}/{l_cool}/{l_enc}",
        "weight_grams": round(l_bom.total_weight_grams(), 2),
        "volume_cm3": round(l_bom.total_volume_cm3(), 2),
        "cost_usd": round(l_bom.total_cost_usd(), 2),
        "dims": l_bom.bounding_dimensions_mm(),
        "thermal": l_thermal,
    }
    r_data = {
        "label": f"{r_pkg}/{r_cool}/{r_enc}",
        "weight_grams": round(r_bom.total_weight_grams(), 2),
        "volume_cm3": round(r_bom.total_volume_cm3(), 2),
        "cost_usd": round(r_bom.total_cost_usd(), 2),
        "dims": r_bom.bounding_dimensions_mm(),
        "thermal": r_thermal,
    }

    if json_out or ctx.obj.get("json", False):
        click.echo(
            json.dumps(
                {
                    "left": {**l_data, "dims": _dims_dict(l_data["dims"])},
                    "right": {**r_data, "dims": _dims_dict(r_data["dims"])},
                },
                indent=2,
            )
        )
        return

    table = Table(title=f"Configuration Comparison ({area:.0f}mm² / {process}nm / {power}W)")
    table.add_column("Metric", style="cyan")
    table.add_column(l_data["label"], justify="right")
    table.add_column(r_data["label"], justify="right")

    _compare_row(table, "Weight (g)", l_data["weight_grams"], r_data["weight_grams"])
    _compare_row(table, "Volume (cm³)", l_data["volume_cm3"], r_data["volume_cm3"])
    _compare_row(table, "Cost ($)", l_data["cost_usd"], r_data["cost_usd"])

    l_tj = l_thermal["junction_temp_c"]
    r_tj = r_thermal["junction_temp_c"]
    l_margin = l_thermal["margin_c"]
    r_margin = r_thermal["margin_c"]
    table.add_row("Tj (°C)", f"{l_tj:.0f} ({l_margin:.0f}°C)", f"{r_tj:.0f} ({r_margin:.0f}°C)")

    l_tv = "PASS" if l_thermal["feasible"] else "FAIL"
    r_tv = "PASS" if r_thermal["feasible"] else "FAIL"
    l_ts = "green" if l_thermal["feasible"] else "red"
    r_ts = "green" if r_thermal["feasible"] else "red"
    table.add_row("Thermal", f"[{l_ts}]{l_tv}[/{l_ts}]", f"[{r_ts}]{r_tv}[/{r_ts}]")

    l_d = l_data["dims"]
    r_d = r_data["dims"]
    table.add_row(
        "Dims (mm)",
        f"{l_d.length_mm:.0f}×{l_d.width_mm:.0f}×{l_d.height_mm:.0f}",
        f"{r_d.length_mm:.0f}×{r_d.width_mm:.0f}×{r_d.height_mm:.0f}",
    )

    console.print(table)

    # Delta summary
    w_delta = _pct_delta(l_data["weight_grams"], r_data["weight_grams"])
    v_delta = _pct_delta(l_data["volume_cm3"], r_data["volume_cm3"])
    c_delta = _pct_delta(l_data["cost_usd"], r_data["cost_usd"])
    t_diff = l_tj - r_tj
    console.print(
        f" Delta: right is {w_delta} weight, {v_delta} volume, "
        f"{c_delta} cost, {t_diff:+.0f}°C Tj"
    )


# ---------------------------------------------------------------------------
# 7. explain
# ---------------------------------------------------------------------------


@swap.command()
@click.option(
    "--points",
    "-p",
    required=True,
    help="Two point indices, comma-separated (e.g., 0,3)",
)
def explain(points):
    """Explain tradeoff between two Pareto-front designs."""
    result = _load_swap_result()
    if not result:
        console.print("[red]No SWaP-C results found. Run 'swap explore' first.[/red]")
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
    table.add_column(f"Design #{idx_a}", justify="right")
    table.add_column(f"Design #{idx_b}", justify="right")
    table.add_column("Delta", justify="right")
    table.add_column("Change %", justify="right")

    objectives = [
        "power_watts",
        "latency_ms",
        "area_mm2",
        "cost_usd",
        "weight_grams",
        "volume_cm3",
    ]

    for name in objectives:
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
        for k, (va, vb) in sorted(changes.items()):
            ptable.add_row(k, str(va), str(vb))
        console.print(ptable)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def _print_swap_front(
    front: list[dict],
    knee_point: dict | None,
    title: str = "SWaP-C Pareto Front",
):
    """Print a Rich table for the SWaP-C Pareto front."""
    table = Table(title=title)
    table.add_column("#", style="dim")
    table.add_column("Process", style="cyan")
    table.add_column("Package", style="cyan")
    table.add_column("Cooling", style="cyan")
    table.add_column("Power (W)", style="green", justify="right")
    table.add_column("Latency (ms)", style="yellow", justify="right")
    table.add_column("Area (mm²)", style="blue", justify="right")
    table.add_column("Cost ($)", style="red", justify="right")
    table.add_column("Weight (g)", style="magenta", justify="right")
    table.add_column("Volume (cm³)", style="magenta", justify="right")

    for i, point in enumerate(front):
        objs = point.get("objectives", {})
        params = point.get("design_params", {})
        knee = " *" if point == knee_point else ""
        table.add_row(
            f"{i}{knee}",
            f"{params.get('process_nm', '?')}nm",
            str(params.get("package_type", "?")),
            str(params.get("cooling_type", "?")),
            f"{objs.get('power_watts', 0):.2f}",
            f"{objs.get('latency_ms', 0):.2f}",
            f"{objs.get('area_mm2', 0):.2f}",
            f"{objs.get('cost_usd', 0):.2f}",
            f"{objs.get('weight_grams', 0):.1f}",
            f"{objs.get('volume_cm3', 0):.1f}",
        )

    console.print(table)
    if knee_point:
        console.print("[dim]* = knee point (best balance)[/dim]")


def _compare_row(table: Table, label: str, left_val: float, right_val: float):
    """Add a comparison row with values."""
    table.add_row(label, f"{left_val:.2f}", f"{right_val:.2f}")


def _pct_delta(base: float, other: float) -> str:
    """Format percentage change from base to other."""
    if base == 0:
        return "N/A"
    pct = (other - base) / base * 100
    return f"{pct:+.0f}%"


def _dims_dict(dims) -> dict[str, float]:
    """Convert Dimensions3D to a plain dict."""
    return {
        "length_mm": round(dims.length_mm, 1),
        "width_mm": round(dims.width_mm, 1),
        "height_mm": round(dims.height_mm, 1),
    }
