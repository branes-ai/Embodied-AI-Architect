"""Session management CLI commands.

List, inspect, and resume saved design sessions. Sessions are auto-saved
by SoCDesignRunner after every step, enabling multi-invocation workflows
and Claude Code skill access to LangGraph state.
"""

import json

import click
from rich.console import Console
from rich.table import Table

console = Console()


@click.group()
def session():
    """Manage saved design sessions.

    \\b
    Examples:
      branes session list                    # List all sessions
      branes session show soc_abc123         # Inspect a session
      branes session show --latest           # Inspect most recent
      branes session delete soc_abc123       # Delete a session
    """
    pass


@session.command("list")
@click.pass_context
def session_list(ctx):
    """List all saved design sessions."""
    from embodied_ai_architect.graphs.session_store import SessionStore

    store = SessionStore()
    sessions = store.list_sessions()

    if not sessions:
        console.print("[yellow]No saved sessions.[/yellow]")
        console.print("[dim]Sessions are auto-saved when running design pipelines.[/dim]")
        return

    table = Table(title=f"Design Sessions ({len(sessions)})")
    table.add_column("Session ID", style="cyan")
    table.add_column("Goal", style="white")
    table.add_column("Status", style="green")
    table.add_column("Iter", justify="right")
    table.add_column("Platform", style="blue")
    table.add_column("Saved At", style="dim")

    for s in sessions:
        status_style = "green" if s["status"] == "complete" else "yellow"
        table.add_row(
            s["session_id"],
            s["goal"],
            f"[{status_style}]{s['status']}[/{status_style}]",
            f"{s['iteration']}/{s['max_iterations']}",
            s["platform"],
            s["saved_at"][:19] if s["saved_at"] else "",
        )

    console.print(table)
    console.print()
    console.print("[dim]Inspect with: branes session show <session-id>[/dim]")
    console.print("[dim]Skills use the latest session: /architect-assess, /architect-loop[/dim]")


@session.command("show")
@click.argument("session_id", required=False)
@click.option("--latest", is_flag=True, help="Show the most recent session")
@click.option("--json-output", "--json", "json_out", is_flag=True, help="Output as JSON")
@click.pass_context
def session_show(ctx, session_id, latest, json_out):
    """Inspect a saved session.

    \\b
    Examples:
      branes session show soc_abc123
      branes session show --latest
      branes session show --latest --json
    """
    from embodied_ai_architect.graphs.session_store import SessionStore

    store = SessionStore()

    if latest:
        state = store.load_latest()
        if not state:
            console.print("[yellow]No saved sessions.[/yellow]")
            return
    elif session_id:
        state = store.load(session_id)
        if not state:
            console.print(f"[red]Session '{session_id}' not found.[/red]")
            return
    else:
        console.print("[red]Provide a session ID or use --latest.[/red]")
        return

    if json_out:
        click.echo(json.dumps(state, indent=2, default=str))
        return

    # Render session overview
    console.print()
    console.print(f"[bold cyan]Session: {state.get('session_id', '?')}[/bold cyan]")
    console.print(f"  Goal:     {state.get('goal', '?')}")
    console.print(f"  Status:   {state.get('status', '?')}")
    console.print(f"  Platform: {state.get('platform', '?')}")
    console.print(f"  Use case: {state.get('use_case', '?')}")
    console.print(f"  Iteration: {state.get('iteration', 0)}/{state.get('max_iterations', 20)}")
    console.print(f"  Created:  {state.get('created_at', '?')}")
    console.print()

    # PPA metrics
    ppa = state.get("ppa_metrics", {})
    if ppa:
        console.print("[bold]PPA Metrics[/bold]")
        verdicts = ppa.get("verdicts", {})
        for metric, field in [
            ("Power", "power_watts"),
            ("Latency", "latency_ms"),
            ("Area", "area_mm2"),
            ("Cost", "cost_usd"),
            ("Memory", "memory_mb"),
        ]:
            val = ppa.get(field)
            if val is not None:
                verdict = verdicts.get(metric.lower(), "?")
                v_style = "green" if verdict == "PASS" else "red"
                console.print(f"  {metric:<12} {val:>8.1f}  [{v_style}]{verdict}[/{v_style}]")
        console.print()

    # Optimization review snapshot (if available)
    opt_snap = state.get("optimization_review_snapshot", {})
    if opt_snap:
        console.print("[bold]Constraint Slackness[/bold]")
        for cs in opt_snap.get("constraint_slackness", []):
            name = cs.get("name", "?")
            margin = cs.get("margin_pct")
            verdict = cs.get("verdict", "?")
            trend = cs.get("trend", "?")
            margin_str = f"{margin:+.1f}%" if margin is not None else "?"
            console.print(f"  {name:<12} margin: {margin_str:>8}  {verdict:<6}  {trend}")
        console.print()

    # KPU convergence history (issue #34): replay the inner-loop sequence
    # so the architect can see how the micro-architecture evolved.
    kpu_history = state.get("kpu_optimization_history", [])
    if kpu_history:
        console.print("[bold]KPU Convergence History[/bold] " f"({len(kpu_history)} entries)")
        for entry in kpu_history:
            it = entry.get("iteration", "?")
            src = entry.get("source", "?")
            head = f"  iter {it} [{src}]"
            cfg_name = entry.get("config_name")
            if cfg_name:
                head += f" — {cfg_name}"
            console.print(head)
            details = []
            if entry.get("compute_array"):
                details.append(f"systolic={entry['compute_array']}")
            l2 = entry.get("l2_size_bytes")
            if l2 is not None:
                details.append(f"L2={l2 // 1024}KB")
            if entry.get("noc_link_width_bits") is not None:
                details.append(f"NoC={entry['noc_link_width_bits']}b")
            if "pitch_matched" in entry:
                pv = "PASS" if entry["pitch_matched"] else "FAIL"
                area = entry.get("total_area_mm2")
                d = f"pitch={pv}"
                if area is not None:
                    d += f"/area={area:.1f}mm²"
                details.append(d)
            if "bandwidth_balanced" in entry:
                bv = "PASS" if entry["bandwidth_balanced"] else "FAIL"
                details.append(f"BW={bv}")
            if details:
                console.print("    " + "  |  ".join(details))
            for change in entry.get("changes", []) or []:
                console.print(f"    → {change}")
        console.print()

    # MOO summary (issue #27): show that multi-objective optimization ran and
    # what it produced, so the architect can see at a glance whether the Pareto
    # frontier is populated without dropping into the API or --json.
    moo_results = state.get("moo_results", {})
    pareto_points = state.get("pareto_points", [])
    if moo_results or pareto_points:
        console.print("[bold]MOO Summary[/bold]")
        total_evals = moo_results.get("total_evaluations", 0)
        hv = moo_results.get("hypervolume")
        layers = moo_results.get("layers_used", [])
        front_size = (
            opt_snap.get("pareto_front_size")
            if opt_snap
            else len(moo_results.get("pareto_front", []))
        )
        console.print(f"  Pareto front:    {front_size or 0} non-dominated designs")
        console.print(f"  Accumulated:     {len(pareto_points)} points (across iterations)")
        console.print(f"  Total evals:     {total_evals}")
        if hv is not None:
            console.print(f"  Hypervolume:     {hv:.3f}")
        if layers:
            console.print(f"  Layers used:     {', '.join(layers)}")
        console.print()

    # Design rationale
    rationale = state.get("design_rationale", [])
    if rationale:
        console.print("[bold]Design Journey[/bold] (last 10)")
        for r in rationale[-10:]:
            console.print(f"  {r}")
        console.print()

    # Task graph summary
    task_graph = state.get("task_graph", {})
    nodes = task_graph.get("nodes", {})
    if nodes:
        completed = sum(1 for n in nodes.values() if n.get("status") == "completed")
        failed = sum(1 for n in nodes.values() if n.get("status") == "failed")
        console.print(
            f"[bold]Task Graph[/bold]: {len(nodes)} tasks "
            f"({completed} completed, {failed} failed)"
        )
        console.print()


@session.command("delete")
@click.argument("session_id")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation")
@click.pass_context
def session_delete(ctx, session_id, yes):
    """Delete a saved session."""
    from embodied_ai_architect.graphs.session_store import SessionStore

    store = SessionStore()

    if not store.exists(session_id):
        console.print(f"[red]Session '{session_id}' not found.[/red]")
        return

    if not yes:
        if not click.confirm(f"Delete session {session_id}?"):
            return

    store.delete(session_id)
    console.print(f"[green]Deleted session {session_id}[/green]")
