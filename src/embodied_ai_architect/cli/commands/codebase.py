"""Codebase analysis CLI commands.

Commands for scanning, analyzing, and assessing application codebases
against hardware targets.
"""

import json
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


@click.group()
def codebase():
    """Analyze application codebases for hardware assessment.

    \b
    Examples:
      branes codebase scan /path/to/project
      branes codebase analyze /path/to/project
      branes codebase assess /path/to/project --hardware jetson_orin
    """
    pass


@codebase.command("scan")
@click.argument("project_path", type=click.Path(exists=True, file_okay=False))
@click.pass_context
def codebase_scan(ctx, project_path: str):
    """Quick static scan of a project directory.

    Returns file inventory, languages, build system, ML model files,
    and dependencies. No LLM needed.

    \b
    Examples:
      branes codebase scan .
      branes codebase scan /path/to/my/app
    """
    json_output = ctx.obj.get("json", False)

    try:
        from embodied_ai_architect.codebase.scanner import CodebaseScanner

        scanner = CodebaseScanner()
        result = scanner.scan(Path(project_path))

        if json_output:
            click.echo(json.dumps(result.model_dump(), indent=2, default=str))
        else:
            _display_scan_result(result)

    except Exception as e:
        if json_output:
            click.echo(json.dumps({"status": "error", "error": str(e)}))
        else:
            console.print(f"\n[bold red]Error:[/bold red] {e}")
        ctx.exit(1)


@codebase.command("analyze")
@click.argument("project_path", type=click.Path(exists=True, file_okay=False))
@click.pass_context
def codebase_analyze(ctx, project_path: str):
    """Full LLM-powered codebase analysis.

    Scans the project, then runs multi-pass LLM analysis to extract
    compute kernels, dataflow, and pipeline structure.

    Requires ANTHROPIC_API_KEY to be set.

    \b
    Examples:
      branes codebase analyze /path/to/app
    """
    json_output = ctx.obj.get("json", False)

    try:
        from embodied_ai_architect.codebase.scanner import CodebaseScanner
        from embodied_ai_architect.codebase.analyzer import CodeAnalyzer
        from embodied_ai_architect.llm.client import LLMClient

        path = Path(project_path)

        if not json_output:
            console.print(f"[bold]Scanning {path.name}...[/bold]")

        scanner = CodebaseScanner()
        scan_result = scanner.scan(path)

        if not json_output:
            console.print(
                f"  Found {len(scan_result.source_files)} source files, "
                f"{scan_result.total_lines} lines"
            )
            console.print(f"  Build system: {scan_result.build_system}")
            console.print("[bold]Running LLM analysis (4 passes)...[/bold]")

        llm = LLMClient()
        analyzer = CodeAnalyzer(llm)
        analysis = analyzer.analyze(scan_result, path)

        if json_output:
            click.echo(json.dumps(analysis.model_dump(), indent=2, default=str))
        else:
            _display_analysis_result(analysis)

    except ImportError as e:
        if "anthropic" in str(e).lower() or "LLMClient" in str(e):
            msg = (
                "LLM client not available. Set ANTHROPIC_API_KEY and install: pip install anthropic"
            )
        else:
            msg = str(e)
        if json_output:
            click.echo(json.dumps({"status": "error", "error": msg}))
        else:
            console.print(f"\n[bold red]Error:[/bold red] {msg}")
        ctx.exit(1)
    except Exception as e:
        if json_output:
            click.echo(json.dumps({"status": "error", "error": str(e)}))
        else:
            console.print(f"\n[bold red]Error:[/bold red] {e}")
        ctx.exit(1)


@codebase.command("assess")
@click.argument("project_path", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--hardware",
    type=str,
    default=None,
    help="Comma-separated hardware targets (e.g., jetson_orin,custom_kpu)",
)
@click.option(
    "--power-budget",
    type=float,
    default=None,
    help="Maximum power budget in watts",
)
@click.option(
    "--latency-target",
    type=float,
    default=None,
    help="Target end-to-end latency in milliseconds",
)
@click.option(
    "--save-session",
    is_flag=True,
    help="Persist results as a design session for /architect-* skills",
)
@click.pass_context
def codebase_assess(
    ctx,
    project_path: str,
    hardware: str | None,
    power_budget: float | None,
    latency_target: float | None,
    save_session: bool,
):
    """End-to-end hardware assessment of a codebase.

    Scans the project, runs LLM analysis, converts to workload profile,
    and runs through the hardware assessment pipeline.

    \b
    Examples:
      branes codebase assess /path/to/app
      branes codebase assess /path/to/app --hardware jetson_orin,custom_kpu
      branes codebase assess /path/to/app --power-budget 15 --latency-target 33
      branes codebase assess /path/to/app --save-session
    """
    json_output = ctx.obj.get("json", False)

    try:
        from embodied_ai_architect.agents.codebase_analyzer import CodebaseAnalyzerAgent

        target_hardware = hardware.split(",") if hardware else None

        agent = CodebaseAnalyzerAgent()
        result = agent.execute(
            {
                "project_path": project_path,
                "target_hardware": target_hardware,
            }
        )

        if not result.success:
            raise Exception(result.error)

        session_id: str | None = None
        if save_session:
            session_id = _save_codebase_session(
                project_path, result.data, power_budget, latency_target
            )
            if not json_output:
                console.print(f"\n[bold green]Session saved:[/bold green] {session_id}")
                console.print(
                    "[dim]Run [cyan]/architect-assess[/cyan] in chat to see "
                    "source-mapped operator breakdown[/dim]"
                )

        if json_output:
            payload = dict(result.data)
            if session_id:
                payload["session_id"] = session_id
            click.echo(json.dumps(payload, indent=2, default=str))
        else:
            _display_assessment_result(result.data, power_budget, latency_target)

    except Exception as e:
        if json_output:
            click.echo(json.dumps({"status": "error", "error": str(e)}))
        else:
            console.print(f"\n[bold red]Error:[/bold red] {e}")
        ctx.exit(1)


def _save_codebase_session(
    project_path: str,
    data: dict,
    power_budget: float | None,
    latency_target: float | None,
) -> str:
    """Persist a codebase assessment as a SoC design session.

    Stores the workload_profile, scan_result, and analysis under
    codebase_metadata so /architect-assess and /architect-drill can
    surface source-level information.
    """
    from embodied_ai_architect.graphs.session_store import SessionStore
    from embodied_ai_architect.graphs.soc_state import (
        DesignConstraints,
        create_initial_soc_state,
    )

    workload_profile = data.get("workload_profile", {})
    scan_result = data.get("scan_result", {})
    analysis = data.get("analysis", {})

    # Resolve project_path to an absolute path so /architect-drill source:
    # works regardless of the cwd at session-load time.
    resolved_project_path = str(Path(project_path).resolve())

    # Build a goal text from project name
    project_name = scan_result.get("project_name", Path(project_path).name)
    goal = f"Codebase analysis: {project_name}"

    constraint_kwargs = {}
    if power_budget is not None:
        constraint_kwargs["max_power_watts"] = power_budget
    if latency_target is not None:
        constraint_kwargs["max_latency_ms"] = latency_target
    constraints = DesignConstraints(**constraint_kwargs) if constraint_kwargs else None

    state = create_initial_soc_state(
        goal=goal,
        constraints=constraints,
        use_case="codebase_analysis",
        platform="custom",
    )
    state["workload_profile"] = workload_profile
    state["codebase_metadata"] = {
        "project_path": resolved_project_path,
        "project_name": project_name,
        "languages": scan_result.get("languages", []),
        "build_system": scan_result.get("build_system", "unknown"),
        "kernel_count": len(analysis.get("kernels", [])) if analysis else 0,
        "ml_models": scan_result.get("ml_models", []),
        "scan_summary": {
            "total_lines": scan_result.get("total_lines", 0),
            "source_file_count": len(scan_result.get("source_files", [])),
            "dependencies": scan_result.get("dependencies", [])[:50],
        },
    }

    store = SessionStore()
    session_id = store.save(state)
    return session_id


@codebase.command("qualify")
@click.argument("project_path", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--goal",
    type=str,
    default=None,
    help="Override the auto-generated goal text",
)
@click.option(
    "--domain",
    type=str,
    default=None,
    help="Force a domain (drone, ugv, robot_arm) — overrides auto-detection",
)
@click.option(
    "--auto",
    is_flag=True,
    help="Skip interactive Q&A; only use auto-detected answers",
)
@click.pass_context
def codebase_qualify(
    ctx,
    project_path: str,
    goal: str | None,
    domain: str | None,
    auto: bool,
):
    """Auto-qualify a project by scanning the codebase.

    Detects platform (drone/ugv/robot_arm), perception tasks (object detection,
    SLAM, tracking), and control output (flight controller, path planner) from
    the project's dependencies and source files. Pre-fills the qualifier with
    detected answers, then runs interactive Q&A only on the remaining gaps.

    \b
    Examples:
      branes codebase qualify /path/to/drone_app
      branes codebase qualify . --domain drone
      branes codebase qualify . --auto                # non-interactive
      branes codebase qualify . --goal "custom goal"  # override goal text
    """
    json_output = ctx.obj.get("json", False)

    try:
        from embodied_ai_architect.codebase.qualifier_bridge import (
            codebase_to_qualification,
        )
        from embodied_ai_architect.codebase.scanner import CodebaseScanner
        from embodied_ai_architect.qualification.qualifier import (
            GoalQualifier,
            render_qualification_result,
        )

        # Step 1: Scan the project
        if not json_output:
            console.print(f"[dim]Scanning {project_path}...[/dim]")
        scanner = CodebaseScanner()
        scan = scanner.scan(Path(project_path))

        # Step 2: Bridge → detection report + prefilled answers
        bridge = codebase_to_qualification(scan)

        if not json_output:
            _display_detection_report(scan, bridge)

        # Step 3: Build the goal text
        effective_goal = goal or _synthesize_goal(scan, bridge)

        # Step 4: Run the qualifier with the detected domain
        effective_domain = domain or bridge.domain
        qualifier = GoalQualifier()
        result = qualifier.assess(effective_goal, domain=effective_domain)

        # Step 5: Pre-fill the answers — only when a domain template is
        # actually loaded. If assess() returned a domain selection prompt
        # (no template), the qualifier has no questions yet so prefilling
        # would silently drop everything.
        if bridge.prefilled_answers and effective_domain and result.next_question:
            if not result.next_question.id.startswith("_"):
                result = qualifier.prefill_answers(bridge.prefilled_answers)

        # Step 6: JSON output mode — dump everything and exit
        if json_output:
            click.echo(
                json.dumps(
                    {
                        "scan": scan.model_dump(),
                        "detection": {
                            "domain": bridge.domain,
                            "domain_evidence": bridge.report.domain_evidence,
                            "perception_tasks": bridge.report.perception_tasks,
                            "perception_evidence": bridge.report.perception_evidence,
                            "control_output": bridge.report.control_output,
                            "control_evidence": bridge.report.control_evidence,
                            "ml_frameworks": bridge.report.ml_frameworks,
                            "has_ml_models": bridge.report.has_ml_models,
                            "confidence": bridge.report.confidence,
                        },
                        "prefilled_answers": bridge.prefilled_answers,
                        "qualification": {
                            "is_tangible": result.is_tangible,
                            "missing_dimensions": result.qualification.missing_dimensions,
                            "answers": result.answers,
                            "accumulated_spec": result.accumulated_spec,
                        },
                    },
                    indent=2,
                    default=str,
                )
            )
            return

        # Step 7: Auto mode — print the qualification state and exit
        if auto:
            console.print()
            console.print(render_qualification_result(result))
            return

        # Step 8: Interactive — only ask the gaps
        console.print()
        console.print(render_qualification_result(result))

        while result.next_question:
            q = result.next_question
            console.print()
            console.print(f"[bold]{q.text}[/bold]")
            if q.explanation:
                console.print(f"[dim]{q.explanation}[/dim]")
            if q.options:
                for i, opt in enumerate(q.options, 1):
                    desc = q.option_descriptions.get(opt, "")
                    console.print(f"  {i}. [cyan]{opt}[/cyan]" + (f" — {desc}" if desc else ""))

                is_multi = q.question_type.value == "multi_choice"
                if is_multi:
                    prompt = "Select (comma-separated numbers, or 's' to skip): "
                else:
                    prompt = "Select (number, or 's' to skip): "
                raw = console.input(f"[bold]{prompt}[/bold]").strip()

                if raw.lower() == "s" or not raw:
                    result = qualifier.skip(q.id)
                elif is_multi:
                    # Parse comma-separated indices
                    selected: list[str] = []
                    for tok in raw.split(","):
                        tok = tok.strip()
                        if tok.isdigit():
                            idx = int(tok) - 1
                            if 0 <= idx < len(q.options):
                                selected.append(q.options[idx])
                    if selected:
                        result = qualifier.answer(q.id, selected)
                    else:
                        result = qualifier.skip(q.id)
                elif raw.isdigit():
                    idx = int(raw) - 1
                    if 0 <= idx < len(q.options):
                        result = qualifier.answer(q.id, q.options[idx])
                    else:
                        result = qualifier.skip(q.id)
                else:
                    result = qualifier.skip(q.id)
            else:
                raw = console.input(f"[bold]{q.text} [/bold]")
                if raw.strip():
                    if q.question_type.value == "numeric":
                        try:
                            result = qualifier.answer(q.id, float(raw))
                        except ValueError:
                            result = qualifier.skip(q.id)
                    else:
                        result = qualifier.answer(q.id, raw.strip())
                else:
                    result = qualifier.skip(q.id)
            console.print()
            console.print(render_qualification_result(result))

        if result.is_tangible:
            console.print("\n[bold green]✓ Goal qualified![/bold green]")
            console.print("[dim]Next: branes design plan with these constraints[/dim]")
        else:
            console.print("\n[yellow]Goal could not be fully qualified.[/yellow]")
            console.print(
                "[dim]Missing: " + ", ".join(result.qualification.missing_dimensions) + "[/dim]"
            )

    except Exception as e:
        if json_output:
            click.echo(json.dumps({"status": "error", "error": str(e)}))
        else:
            console.print(f"\n[bold red]Error:[/bold red] {e}")
        ctx.exit(1)


def _synthesize_goal(scan, bridge) -> str:
    """Build a natural-language goal from scan + detection."""
    parts = [f"Application '{scan.project_name}'"]
    if bridge.domain:
        parts.append(f"on a {bridge.domain} platform")
    if bridge.report.perception_tasks:
        tasks = ", ".join(bridge.report.perception_tasks)
        parts.append(f"performing {tasks}")
    if scan.languages:
        parts.append(f"({', '.join(scan.languages[:3])})")
    return " ".join(parts)


def _display_detection_report(scan, bridge) -> None:
    """Display the auto-detection report from the codebase qualifier bridge."""
    report = bridge.report

    console.print()
    console.print(
        Panel(
            f"Project: [bold]{scan.project_name}[/bold]\n"
            f"Languages: {', '.join(scan.languages) or 'none'}\n"
            f"Dependencies: {len(scan.dependencies)}\n"
            f"ML models: {len(scan.ml_models)}",
            title="Codebase Scan",
            border_style="cyan",
        )
    )

    detection_lines = []
    if report.domain:
        ev = ", ".join(report.domain_evidence[:5])
        detection_lines.append(f"  [cyan]Domain:[/cyan] [bold]{report.domain}[/bold]  ({ev})")
    else:
        detection_lines.append("  [cyan]Domain:[/cyan] [dim]not detected[/dim]")

    if report.perception_tasks:
        tasks = ", ".join(report.perception_tasks)
        ev = ", ".join(report.perception_evidence[:3])
        detection_lines.append(f"  [cyan]Perception:[/cyan] {tasks}  ({ev})")
    else:
        detection_lines.append("  [cyan]Perception:[/cyan] [dim]not detected[/dim]")

    if report.control_output:
        outs = ", ".join(report.control_output)
        ev = ", ".join(report.control_evidence[:3])
        detection_lines.append(f"  [cyan]Control:[/cyan] {outs}  ({ev})")
    else:
        detection_lines.append("  [cyan]Control:[/cyan] [dim]not detected[/dim]")

    if report.ml_frameworks:
        detection_lines.append(f"  [cyan]ML frameworks:[/cyan] {', '.join(report.ml_frameworks)}")

    confidence_color = {"high": "green", "medium": "yellow", "low": "red"}.get(
        report.confidence, "white"
    )
    detection_lines.append(
        f"  [cyan]Confidence:[/cyan] [{confidence_color}]{report.confidence}[/{confidence_color}]"
    )

    console.print(
        Panel(
            "\n".join(detection_lines),
            title="Auto-Detection",
            border_style="green" if report.has_any_signal() else "yellow",
        )
    )


# --- Display helpers ---


def _display_scan_result(result) -> None:
    """Display scan results using Rich formatting."""
    console.print(f"\n[bold green]Scan complete:[/bold green] {result.project_name}\n")

    # Summary panel
    summary = (
        f"Languages: {', '.join(result.languages) or 'none detected'}\n"
        f"Build system: {result.build_system}\n"
        f"Source files: {len(result.source_files)}\n"
        f"Total lines: {result.total_lines:,}\n"
        f"ML models: {len(result.ml_models)}"
    )
    console.print(Panel(summary, title="Project Summary", border_style="cyan"))

    # File table
    if result.source_files:
        table = Table(title="Source Files", show_header=True)
        table.add_column("File", style="cyan")
        table.add_column("Language")
        table.add_column("Lines", justify="right")
        table.add_column("Role")
        table.add_column("Entry Type", style="yellow")

        for sf in result.source_files[:20]:
            table.add_row(sf.path, sf.language, str(sf.lines), sf.role, sf.entry_type or "")

        if len(result.source_files) > 20:
            table.add_row(f"... +{len(result.source_files) - 20} more", "", "", "", "")

        console.print(table)

    # ML models
    if result.ml_models:
        console.print("\n[bold]ML Model Files:[/bold]")
        for m in result.ml_models:
            size_mb = m.get("size_bytes", 0) / (1024 * 1024)
            console.print(f"  {m['path']} ({m['format']}, {size_mb:.1f} MB)")

    # Dependencies
    if result.dependencies:
        console.print(f"\n[bold]Dependencies:[/bold] {', '.join(result.dependencies[:15])}")
        if len(result.dependencies) > 15:
            console.print(f"  ... +{len(result.dependencies) - 15} more")

    # Recommended entry point
    rec = result.recommended_entry_point
    if rec:
        confidence = rec.get("confidence", "low")
        path = rec.get("path", "")
        alternatives = rec.get("alternatives", [])

        console.print()
        if confidence == "medium" and alternatives:
            console.print(
                f"[bold]Recommended application entry point:[/bold]\n"
                f"  → {path}  (confidence: {confidence}, "
                f"{len(alternatives) + 1} candidates)\n"
                f"  Alternatives: {', '.join(alternatives)}"
            )
        else:
            console.print(
                f"[bold]Recommended application entry point:[/bold]\n"
                f"  → {path}  (confidence: {confidence})"
            )

        # Use relative path if under cwd, otherwise absolute
        try:
            project_path = str(Path(result.project_path).relative_to(Path.cwd()))
        except ValueError:
            project_path = result.project_path
        console.print(
            f"\n[bold]Next step[/bold] — run LLM-powered analysis on this entry point:\n\n"
            f"  branes codebase analyze {project_path}\n"
        )


def _display_analysis_result(analysis) -> None:
    """Display full analysis results."""
    console.print(f"\n[bold green]Analysis complete:[/bold green] {analysis.project_name}\n")

    if analysis.summary:
        console.print(Panel(analysis.summary, title="Summary", border_style="blue"))

    # Kernels table
    if analysis.kernels:
        table = Table(title="Compute Kernels", show_header=True)
        table.add_column("Kernel", style="cyan")
        table.add_column("Type")
        table.add_column("Ops/Call", justify="right")
        table.add_column("Freq (Hz)", justify="right")
        table.add_column("Data Types")
        table.add_column("Parallelism")

        for k in analysis.kernels:
            ops_str = (
                f"{k.estimated_ops_per_invocation:.1e}"
                if k.estimated_ops_per_invocation > 0
                else "-"
            )
            freq_str = f"{k.invocation_frequency_hz:.0f}" if k.invocation_frequency_hz > 0 else "-"
            table.add_row(
                k.name,
                k.kernel_type,
                ops_str,
                freq_str,
                ", ".join(k.data_types),
                k.parallelism,
            )

        console.print(table)

    # Dataflow
    if analysis.dataflow:
        console.print("\n[bold]Dataflow:[/bold]")
        for df in analysis.dataflow:
            console.print(f"  {df.source_kernel} -> {df.sink_kernel} ({df.transfer_type})")


def _display_assessment_result(
    data: dict, power_budget: float | None, latency_target: float | None
) -> None:
    """Display assessment results."""
    scan = data.get("scan_result", {})
    wp = data.get("workload_profile", {})

    console.print(
        f"\n[bold green]Assessment complete:[/bold green] "
        f"{scan.get('project_name', 'project')}\n"
    )

    # Workload summary
    summary = (
        f"Workloads: {wp.get('workload_count', 0)}\n"
        f"Total GFLOPS: {wp.get('total_estimated_gflops', 0):.2f}\n"
        f"Total Memory: {wp.get('total_estimated_memory_mb', 0):.1f} MB\n"
        f"Dominant Op: {wp.get('dominant_op', 'unknown')}"
    )
    console.print(Panel(summary, title="Workload Profile", border_style="cyan"))

    # Workload details
    for w in wp.get("workloads", []):
        console.print(
            f"  [cyan]{w['name']}[/cyan]: {w.get('estimated_gflops', 0):.2f} GFLOPS, "
            f"{w.get('estimated_memory_mb', 0):.1f} MB, "
            f"type={w.get('kernel_type', 'unknown')}"
        )

    if power_budget:
        console.print(f"\n[bold]Constraints:[/bold] power={power_budget}W", end="")
    if latency_target:
        console.print(f", latency={latency_target}ms", end="")
    if power_budget or latency_target:
        console.print()
