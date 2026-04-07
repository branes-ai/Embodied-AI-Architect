#!/usr/bin/env python3
"""Demo 1: Agentic SoC Designer — Goal Decomposition + Full Pipeline.

Runs the delivery drone SoC design prompt end-to-end through the agentic
pipeline: Planner → Dispatcher → 6 Specialists → Design Report.

Two modes:
  --llm     Use Claude to decompose the goal (requires ANTHROPIC_API_KEY)
  default   Use a static plan (no API key needed, deterministic)

Usage:
    python examples/demo_soc_designer.py             # static plan
    python examples/demo_soc_designer.py --llm        # LLM planner
    python examples/demo_soc_designer.py --goal "..." # custom goal
"""

from __future__ import annotations

import argparse
import sys
import textwrap
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from embodied_ai_architect.graphs.optimizer import design_optimizer
from embodied_ai_architect.graphs.soc_state import (
    DesignConstraints,
    create_initial_soc_state,
    get_iteration_summary,
    get_task_graph,
    record_decision,
)
from embodied_ai_architect.graphs.planner import PlannerNode
from embodied_ai_architect.graphs.specialists import (
    create_default_dispatcher,
    critic,
    ppa_assessor,
)
from embodied_ai_architect.graphs.task_graph import TaskNode

# ---------------------------------------------------------------------------
# Demo prompts
# ---------------------------------------------------------------------------

DEMO_1_GOAL = (
    "Design an SoC for a last-mile delivery drone that must: "
    "run a visual perception pipeline with camera-based object detection "
    "(YOLOv8-nano) and multi-object tracking (ByteTrack) at 30fps on 720p input, "
    "consume less than 5 watts for the compute subsystem, "
    "cost less than $30 in BOM at 100K volume, "
    "and operate reliably in outdoor environments with thermal cycling "
    "(-20C to 60C), vibration, rain, and direct sunlight."
)

DEMO_1_CONSTRAINTS = DesignConstraints(
    max_power_watts=5.0,
    max_latency_ms=33.3,  # 30fps
    max_cost_usd=30.0,
    max_area_mm2=100.0,
    target_volume=100_000,
    operating_temp_min_c=-20,
    operating_temp_max_c=60,
    ip_rating="IP54",
)

# Static plan matching Demo 1's expected task graph
DEMO_1_PLAN = [
    {
        "id": "t1",
        "name": "Analyze perception workload (detection + tracking at 30fps)",
        "agent": "workload_analyzer",
        "dependencies": [],
    },
    {
        "id": "t2",
        "name": "Enumerate feasible hardware under 5W/\u200b$30 constraints",
        "agent": "hw_explorer",
        "dependencies": ["t1"],
    },
    {
        "id": "t3",
        "name": "Compose SoC architecture with selected compute engine",
        "agent": "architecture_composer",
        "dependencies": ["t2"],
    },
    {
        "id": "t4",
        "name": "Explore Pareto frontier (multi-objective optimization)",
        "agent": "moo_explorer",
        "dependencies": ["t2"],
        "metadata": {"fast_mode": True},
    },
    {
        "id": "t5",
        "name": "Assess PPA metrics against drone constraints",
        "agent": "ppa_assessor",
        "dependencies": ["t3", "t4"],
    },
    {
        "id": "t6",
        "name": "Review design for risks and improvement opportunities",
        "agent": "critic",
        "dependencies": ["t5"],
    },
    {
        "id": "t7",
        "name": "Generate design report with trade-off analysis",
        "agent": "report_generator",
        "dependencies": ["t6"],
    },
]


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

W = 72  # output width


def banner(text: str) -> None:
    print(f"\n{'=' * W}")
    print(f"  {text}")
    print(f"{'=' * W}")


def section(text: str) -> None:
    print(f"\n--- {text} {'-' * max(0, W - len(text) - 5)}")


def kv(key: str, value: str, indent: int = 2) -> None:
    pad = " " * indent
    print(f"{pad}{key:.<30s} {value}")


def verdict_str(v: str) -> str:
    if v == "PASS":
        return "PASS"
    elif v == "FAIL":
        return "** FAIL **"
    return v


# ---------------------------------------------------------------------------
# Main demo flow
# ---------------------------------------------------------------------------


def run_demo(
    goal: str,
    constraints: DesignConstraints,
    use_llm: bool,
    max_iterations: int = 5,
) -> None:
    banner("Agentic SoC Designer — Demo 1")
    print(f"\n  Goal:")
    print(textwrap.fill(goal, width=W - 4, initial_indent="    ", subsequent_indent="    "))
    print(f"  Mode: {'LLM Planner (Claude)' if use_llm else 'Static Plan'}")

    # ---- Create initial state ----
    state = create_initial_soc_state(
        goal=goal,
        constraints=constraints,
        use_case="delivery_drone",
        platform="drone",
    )

    section("Constraints")
    cd = constraints.model_dump(exclude_none=True)
    for k, v in cd.items():
        if k == "custom" and not v:
            continue
        kv(k.replace("_", " ").title(), str(v))

    # ---- Plan ----
    section("Planning")
    t0 = time.perf_counter()

    if use_llm:
        try:
            from embodied_ai_architect.llm.client import LLMClient

            llm = LLMClient()
            planner = PlannerNode(llm=llm)
            print("  Using Claude to decompose the goal...")
        except (ImportError, ValueError) as e:
            print(f"  LLM not available ({e}), falling back to static plan")
            planner = PlannerNode(static_plan=DEMO_1_PLAN)
    else:
        planner = PlannerNode(static_plan=DEMO_1_PLAN)

    try:
        plan_updates = planner(state)
    except Exception as e:
        print(f"  LLM call failed ({type(e).__name__}), falling back to static plan")
        planner = PlannerNode(static_plan=DEMO_1_PLAN)
        plan_updates = planner(state)
    state = {**state, **plan_updates}
    plan_time = time.perf_counter() - t0

    graph = get_task_graph(state)
    print(f"\n  Task graph ({len(graph.nodes)} tasks, planned in {plan_time:.1f}s):\n")
    order = graph.execution_order()
    for tid in order:
        task = graph.nodes[tid]
        deps = ", ".join(task.dependencies) if task.dependencies else "(none)"
        print(f"    [{tid}] {task.name}")
        print(f"         agent: {task.agent}  deps: {deps}")

    # ---- Dispatch ----
    section("Executing Pipeline")
    dispatcher = create_default_dispatcher()
    t0 = time.perf_counter()

    step = 0
    completed_ids: set[str] = set()
    while True:
        graph = get_task_graph(state)
        if graph.is_complete:
            state = {**state, "status": "complete"}
            break

        ready = graph.ready_tasks()
        if not ready:
            print(f"\n  Pipeline stuck! Status: {state.get('status')}")
            break

        for task in ready:
            print(f"\n  [{task.id}] {task.agent}: {task.name}")

        state = dispatcher.step(state)
        step += 1

        # Show result summary for newly completed tasks only
        graph = get_task_graph(state)
        for task in graph.nodes.values():
            if task.id not in completed_ids and task.status.value == "completed":
                completed_ids.add(task.id)
                summary = (task.result or {}).get("summary", "")
                if summary:
                    print(f"         -> {summary}")

    exec_time = time.perf_counter() - t0
    print(f"\n  Pipeline completed in {step} steps, {exec_time:.2f}s")

    # ---- Results ----
    _print_results(state)

    # ---- Optimization (if any verdicts FAIL) ----
    opt_iterations = 0
    ppa = state.get("ppa_metrics", {})
    verdicts = ppa.get("verdicts", {})
    has_fail = any(v == "FAIL" for v in verdicts.values())

    if has_fail:
        state, opt_iterations = _run_optimization_loop(state, max_iterations)
        _print_final_ppa(state)
        state = _run_final_review(state)
        _print_decision_trail(state)

    banner("Demo Complete")
    total = plan_time + exec_time
    print(f"\n  Total time: {total:.2f}s  |  Status: {state.get('status')}")
    print(f"  Decisions recorded: {len(state.get('history', []))}")
    if opt_iterations > 0:
        print(f"  Optimization iterations: {opt_iterations}")
    print()


# ---------------------------------------------------------------------------
# Optimization loop
# ---------------------------------------------------------------------------

_STRATEGY_LABELS = {
    "quantize_int8": "INT8 Quantization",
    "reduce_resolution": "Resolution Reduction",
    "clock_scaling": "Clock Scaling",
    "model_pruning": "Model Pruning",
    "smaller_model": "Smaller Model Variant",
    "shrink_process_node": "Process Node Shrink",
    "grow_process_node": "Process Node Grow",
}


def _strategy_display_name(name: str, factor: float | None = None) -> str:
    label = _STRATEGY_LABELS.get(name, name)
    if factor is not None:
        label += f" (~{factor * 100:.0f}% power reduction)"
    return label


def _run_optimization_loop(state: dict, max_iterations: int) -> tuple[dict, int]:
    """Iteratively optimize the design until all PPA verdicts PASS."""
    section("Optimization Phase")

    ppa = state.get("ppa_metrics", {})
    verdicts = ppa.get("verdicts", {})
    failing = [k for k, v in verdicts.items() if v == "FAIL"]
    passing = [k for k, v in verdicts.items() if v == "PASS"]

    parts = []
    if ppa.get("power_watts") is not None:
        v = verdicts.get("power", "?")
        parts.append(f"power={ppa['power_watts']:.1f}W ({v})")
    if ppa.get("latency_ms") is not None:
        v = verdicts.get("latency", "?")
        parts.append(f"latency={ppa['latency_ms']:.1f}ms ({v})")
    if ppa.get("cost_usd") is not None:
        v = verdicts.get("cost", "?")
        parts.append(f"cost=${ppa['cost_usd']:.0f} ({v})")
    print(f"\n  Initial: {', '.join(parts)}")

    iteration = 0
    for iteration in range(1, max_iterations + 1):
        pre_ppa = dict(state.get("ppa_metrics", {}))

        # Run optimizer
        task = TaskNode(
            id=f"opt_{iteration}",
            name=f"Optimization iteration {iteration}",
            agent="design_optimizer",
            dependencies=[],
        )
        opt_result = design_optimizer(task, state)

        if not opt_result.get("applied"):
            reason = opt_result.get("summary", "No applicable strategies")
            print(f"\n  Iteration {iteration}: {reason}")
            break

        # Merge optimizer state updates
        for k, v in opt_result.get("_state_updates", {}).items():
            state = {**state, k: v}

        strategy_name = opt_result.get("strategy", "?")
        power_factor = opt_result.get("power_reduction_factor")

        # Re-assess PPA
        assess_task = TaskNode(
            id=f"reassess_{iteration}",
            name=f"Re-assess PPA after {strategy_name}",
            agent="ppa_assessor",
            dependencies=[],
        )
        assess_result = ppa_assessor(assess_task, state)
        for k, v in assess_result.get("_state_updates", {}).items():
            state = {**state, k: v}

        post_ppa = state.get("ppa_metrics", {})
        post_verdicts = post_ppa.get("verdicts", {})

        # Print iteration summary
        display = _strategy_display_name(strategy_name, power_factor)
        print(f"\n  Iteration {iteration}: Applied {display}")
        _print_before_after(pre_ppa, post_ppa)
        vstr = "  ".join(f"{k}:{v}" for k, v in post_verdicts.items())
        print(f"    Verdicts: {vstr}")

        # Record decision
        state = record_decision(
            state,
            agent="design_optimizer",
            action=f"Applied optimization: {strategy_name}",
            rationale=f"Fixing failing constraints via {display}",
        )

        # Check convergence
        if all(v == "PASS" for v in post_verdicts.values()):
            print(
                f"\n  Converged in {iteration} iteration"
                f"{'s' if iteration > 1 else ''}"
                " — all constraints PASS"
            )
            break
    else:
        still_failing = [k for k, v in post_verdicts.items() if v == "FAIL"]
        print(f"\n  Design did not converge after {max_iterations} iterations.")
        print(f"  Failing constraints: {', '.join(still_failing)}")
        ppa = state.get("ppa_metrics", {})
        bd = ppa.get("cost_breakdown", {})
        if "cost" in still_failing and bd:
            nre = bd.get("nre_per_unit_usd", 0)
            actual = ppa.get("cost_usd", 0)
            if nre > 0 and actual > 0 and nre / actual > 0.5:
                print(
                    f"  NRE is {nre / actual * 100:.0f}% of unit cost — "
                    f"increase volume or relax cost target"
                )

    return state, iteration


def _run_final_review(state: dict) -> dict:
    """Re-run critic on the optimized design and print results."""
    section("Final Design Review")
    task = TaskNode(
        id="final_review",
        name="Final design review after optimization",
        agent="critic",
        dependencies=[],
    )
    result = critic(task, state)

    kv("Assessment", result.get("assessment", "N/A"))

    for label, key in [
        ("Strengths", "strengths"),
        ("Issues", "issues"),
        ("Recommendations", "recommendations"),
    ]:
        items = result.get(key, [])
        if items:
            print(f"\n  {label}:")
            for item in items:
                print(f"    - {item}")

    state = record_decision(
        state,
        agent="critic",
        action=f"Final review: {result.get('assessment', 'N/A')}",
        rationale="Post-optimization design review",
    )
    return state


def _print_before_after(pre_ppa: dict, post_ppa: dict) -> None:
    parts = []
    if pre_ppa.get("power_watts") is not None and post_ppa.get("power_watts") is not None:
        parts.append(f"Power: {pre_ppa['power_watts']:.1f}W → {post_ppa['power_watts']:.1f}W")
    if pre_ppa.get("latency_ms") is not None and post_ppa.get("latency_ms") is not None:
        parts.append(f"Latency: {pre_ppa['latency_ms']:.1f}ms → {post_ppa['latency_ms']:.1f}ms")
    if parts:
        print(f"    {' | '.join(parts)}")


def _print_final_ppa(state: dict) -> None:
    ppa = state.get("ppa_metrics", {})
    if not ppa:
        return

    section("Final PPA (post-optimization)")
    kv("Process Node", f"{ppa.get('process_nm', 28)}nm")
    if ppa.get("power_watts") is not None:
        kv("Power", f"{ppa['power_watts']:.1f} W")
    if ppa.get("latency_ms") is not None:
        kv("Latency", f"{ppa['latency_ms']:.1f} ms")
    if ppa.get("area_mm2") is not None:
        kv("Area", f"{ppa['area_mm2']:.1f} mm2")
    if ppa.get("cost_usd") is not None:
        kv("Cost", f"${ppa['cost_usd']:.0f}")
    if ppa.get("memory_mb") is not None:
        kv("Memory", f"{ppa['memory_mb']:.0f} MB")

    verdicts = ppa.get("verdicts", {})
    if verdicts:
        print()
        for k, v in verdicts.items():
            print(f"  {k:<20} {verdict_str(v)}")

    cost_bd = ppa.get("cost_breakdown")
    if cost_bd:
        print()
        kv("Die Cost", f"${cost_bd.get('die_cost_usd', 0):.2f}")
        kv("Package", f"${cost_bd.get('package_cost_usd', 0):.2f}")
        kv("Test", f"${cost_bd.get('test_cost_usd', 0):.2f}")
        kv("NRE/unit", f"${cost_bd.get('nre_per_unit_usd', 0):.2f}")
        kv("Yield", f"{cost_bd.get('yield_percent', 0):.1f}%")
        kv("Dies/wafer", str(cost_bd.get("dies_per_wafer", 0)))


def _print_decision_trail(state: dict) -> None:
    history = state.get("history", [])
    if not history:
        return

    section("Decision Trail (updated)")
    for i, d in enumerate(history, 1):
        agent = d.get("agent", "?")
        action = d.get("action", "?")
        print(f"    {i}. [{agent}] {action}")


# ---------------------------------------------------------------------------
# Result printing
# ---------------------------------------------------------------------------


def _print_results(state: dict) -> None:
    """Print formatted results from the completed pipeline."""

    # Workload
    wp = state.get("workload_profile", {})
    if wp:
        section("Workload Analysis")
        kv("Source", wp.get("source", "unknown"))
        kv("Dominant Op", wp.get("dominant_op", "unknown"))
        kv("Total Compute", f"{wp.get('total_estimated_gflops', '?')} GFLOPS")
        kv("Total Memory", f"{wp.get('total_estimated_memory_mb', '?')} MB")
        workloads = wp.get("workloads", [])
        if workloads:
            print()
            for w in workloads:
                print(
                    f"    {w['name']:.<30s} {w.get('model_class', '')} "
                    f"({w.get('estimated_gflops', '?')} GFLOPS)"
                )

    # Hardware candidates
    candidates = state.get("hardware_candidates", [])
    if candidates:
        section("Hardware Candidates")
        print(
            f"\n  {'Rank':<5} {'Name':<25} {'TDP':>6} {'Cost':>7} "
            f"{'INT8 TOPS':>10} {'Score':>6}  Verdict"
        )
        print(f"  {'-'*5} {'-'*25} {'-'*6} {'-'*7} {'-'*10} {'-'*6}  {'-'*10}")
        for hw in candidates:
            vstr = " ".join(
                f"{k}:{verdict_str(v)}" for k, v in hw.get("constraint_verdicts", {}).items()
            )
            print(
                f"  {hw.get('rank', '?'):<5} {hw['name']:<25} "
                f"{hw.get('tdp_watts', '?'):>5}W {hw.get('cost_usd', '?'):>6}$ "
                f"{hw.get('peak_tops_int8', '?'):>10} {hw.get('score', 0):>6.1f}  "
                f"{vstr}"
            )

    # Architecture
    arch = state.get("selected_architecture", {})
    if arch:
        section("Selected Architecture")
        kv("Name", arch.get("name", "N/A"))
        kv("Primary Compute", arch.get("primary_compute", "N/A"))
        kv("Paradigm", arch.get("compute_paradigm", "N/A"))
        kv("Process Node", f"{arch.get('target_process_nm', '?')}nm")
        kv("Rationale", arch.get("design_rationale", "N/A"))

    # IP Blocks
    ip_blocks = state.get("ip_blocks", [])
    if ip_blocks:
        section("IP Blocks")
        for b in ip_blocks:
            cfg = b.get("config", {})
            detail = ", ".join(f"{k}={v}" for k, v in cfg.items())
            print(f"    {b['name']:.<25s} [{b.get('type', '?')}]  {detail}")

    # Interconnect
    ic = state.get("interconnect", {})
    if ic:
        section("Interconnect")
        kv("Type", ic.get("type", "N/A"))
        kv("Data Width", f"{ic.get('data_width_bits', '?')} bits")
        kv("Frequency", f"{ic.get('frequency_mhz', '?')} MHz")
        kv("Topology", ic.get("topology", "N/A"))

    # PPA Assessment
    ppa = state.get("ppa_metrics", {})
    if ppa:
        section("PPA Assessment")
        kv("Process Node", f"{ppa.get('process_nm', 28)}nm")
        if ppa.get("power_watts") is not None:
            kv("Power", f"{ppa['power_watts']:.1f} W")
        if ppa.get("latency_ms") is not None:
            kv("Latency", f"{ppa['latency_ms']:.1f} ms")
        if ppa.get("area_mm2") is not None:
            kv("Area", f"{ppa['area_mm2']:.1f} mm2")
        if ppa.get("cost_usd") is not None:
            kv("Cost", f"${ppa['cost_usd']:.0f}")
        if ppa.get("memory_mb") is not None:
            kv("Memory", f"{ppa['memory_mb']:.0f} MB")

        verdicts = ppa.get("verdicts", {})
        if verdicts:
            print()
            print(f"  {'Constraint':<20} {'Verdict':<12}")
            print(f"  {'-'*20} {'-'*12}")
            for k, v in verdicts.items():
                print(f"  {k:<20} {verdict_str(v):<12}")

        bottlenecks = ppa.get("bottlenecks", [])
        if bottlenecks:
            print(f"\n  Bottlenecks:")
            for b in bottlenecks:
                print(f"    - {b}")

        suggestions = ppa.get("suggestions", [])
        if suggestions:
            print(f"\n  Suggestions:")
            for s in suggestions:
                print(f"    - {s}")

        cost_bd = ppa.get("cost_breakdown")
        if cost_bd:
            print()
            kv("Die Cost", f"${cost_bd.get('die_cost_usd', 0):.2f}")
            kv("Package", f"${cost_bd.get('package_cost_usd', 0):.2f}")
            kv("Test", f"${cost_bd.get('test_cost_usd', 0):.2f}")
            kv("NRE/unit", f"${cost_bd.get('nre_per_unit_usd', 0):.2f}")
            kv("Yield", f"{cost_bd.get('yield_percent', 0):.1f}%")
            kv("Dies/wafer", str(cost_bd.get("dies_per_wafer", 0)))

    # Critic review
    graph = get_task_graph(state)
    for task in graph.nodes.values():
        if task.agent == "critic" and task.result:
            section("Design Review (Critic)")
            kv("Assessment", task.result.get("assessment", "N/A"))

            for label, key in [
                ("Strengths", "strengths"),
                ("Issues", "issues"),
                ("Recommendations", "recommendations"),
            ]:
                items = task.result.get(key, [])
                if items:
                    print(f"\n  {label}:")
                    for item in items:
                        print(f"    - {item}")

    # Decision trail
    history = state.get("history", [])
    if history:
        section("Decision Trail")
        for i, d in enumerate(history, 1):
            agent = d.get("agent", "?")
            action = d.get("action", "?")
            print(f"    {i}. [{agent}] {action}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Agentic SoC Designer — Demo 1: Delivery Drone",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Use Claude LLM for goal decomposition (requires ANTHROPIC_API_KEY)",
    )
    parser.add_argument(
        "--goal",
        type=str,
        default=None,
        help="Custom design goal (overrides default drone prompt)",
    )
    parser.add_argument(
        "--power",
        type=float,
        default=None,
        help="Max power budget in watts (default: 5.0)",
    )
    parser.add_argument(
        "--latency",
        type=float,
        default=None,
        help="Max latency in ms (default: 33.3)",
    )
    parser.add_argument(
        "--cost",
        type=float,
        default=None,
        help="Max BOM cost in USD (default: 30)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="Max optimization iterations if PPA fails (default: 5)",
    )
    args = parser.parse_args()

    goal = args.goal or DEMO_1_GOAL
    constraints = DesignConstraints(
        max_power_watts=args.power or DEMO_1_CONSTRAINTS.max_power_watts,
        max_latency_ms=args.latency or DEMO_1_CONSTRAINTS.max_latency_ms,
        max_cost_usd=args.cost or DEMO_1_CONSTRAINTS.max_cost_usd,
        max_area_mm2=DEMO_1_CONSTRAINTS.max_area_mm2,
        target_volume=DEMO_1_CONSTRAINTS.target_volume,
        operating_temp_min_c=DEMO_1_CONSTRAINTS.operating_temp_min_c,
        operating_temp_max_c=DEMO_1_CONSTRAINTS.operating_temp_max_c,
        ip_rating=DEMO_1_CONSTRAINTS.ip_rating,
    )

    run_demo(goal, constraints, use_llm=args.llm, max_iterations=args.max_iterations)


if __name__ == "__main__":
    main()
