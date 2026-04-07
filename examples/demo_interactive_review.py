#!/usr/bin/env python3
"""Demo: Interactive SoC Design with Human-in-the-Loop Review.

Demonstrates the two review stages in the SoC design pipeline:

  1. PLAN REVIEW — After the planner decomposes the goal into a task graph,
     the human architect inspects the DAG, adds/removes tasks, reassigns
     agents, and approves execution.

  2. OPTIMIZATION REVIEW — At each optimization iteration, the human sees
     constraint slackness, the optimization trajectory, available strategies,
     and Pareto frontier data. They can steer the optimizer by focusing on
     a specific objective, forcing a strategy, or relaxing constraints.

This demo uses a static plan (no API key needed). Replace with --llm for
LLM-based planning.

Usage:
    python examples/demo_interactive_review.py               # auto-approve
    python examples/demo_interactive_review.py --interactive  # interactive prompts
    python examples/demo_interactive_review.py --modify       # modify the plan
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from embodied_ai_architect.graphs.review import (
    PlanReviewInput,
    ReviewDecision,
    apply_review_edits,
    build_review_snapshot,
    render_plan_review_rich,
)
from embodied_ai_architect.graphs.optimization_review import (
    OptimizationSteeringInput,
    SteeringDecision,
    apply_steering_input,
    build_optimization_review_snapshot,
    render_optimization_review,
)
from embodied_ai_architect.graphs.optimizer import design_optimizer
from embodied_ai_architect.graphs.soc_state import (
    DesignConstraints,
    create_initial_soc_state,
    get_task_graph,
    record_decision,
)
from embodied_ai_architect.graphs.planner import PlannerNode
from embodied_ai_architect.graphs.specialists import (
    create_default_dispatcher,
    ppa_assessor,
)
from embodied_ai_architect.graphs.task_graph import TaskNode

# ---------------------------------------------------------------------------
# Demo plan and constraints (same as demo_soc_designer)
# ---------------------------------------------------------------------------

GOAL = (
    "Design an SoC for a delivery drone: visual perception pipeline with "
    "YOLOv8-nano detection + ByteTrack at 30fps/720p, <5W compute, <$30 BOM, "
    "outdoor environments (-20C to 60C)."
)

CONSTRAINTS = DesignConstraints(
    max_power_watts=5.0,
    max_latency_ms=33.3,
    max_cost_usd=30.0,
    max_area_mm2=100.0,
    target_process_nm=28,
    target_volume=100_000,
)

PLAN = [
    {
        "id": "t1",
        "name": "Analyze perception workload (detection + tracking at 30fps)",
        "agent": "workload_analyzer",
        "dependencies": [],
    },
    {
        "id": "t2",
        "name": "Enumerate feasible hardware under 5W/$30 constraints",
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

AVAILABLE_AGENTS = [
    "workload_analyzer",
    "hw_explorer",
    "architecture_composer",
    "moo_explorer",
    "ppa_assessor",
    "critic",
    "report_generator",
    "design_optimizer",
    "kpu_configurator",
    "floorplan_validator",
    "bandwidth_validator",
    "kpu_optimizer",
    "design_explorer",
    "safety_detector",
    "experience_retriever",
]


# ---------------------------------------------------------------------------
# Stage 1: Plan Review
# ---------------------------------------------------------------------------


def demo_plan_review(state: dict, interactive: bool = False, modify: bool = False) -> dict:
    """Demonstrate plan review with human edits."""

    print("\n" + "=" * 72)
    print("  STAGE 1: PLAN REVIEW")
    print("=" * 72)

    # Build the review snapshot (same data the review node produces)
    snapshot = build_review_snapshot(state, AVAILABLE_AGENTS)

    # Render for the human
    print()
    print(render_plan_review_rich(snapshot))

    # Decide what to do
    if modify:
        # Programmatic modification: add a safety check, remove the report
        print("\n  [Architect] Adding safety review, removing report generator")
        review = PlanReviewInput(
            decision=ReviewDecision.MODIFY,
            tasks_to_remove=["t6"],
            tasks_to_add=[
                {
                    "id": "t7",
                    "name": "Validate safety constraints for flight-critical system",
                    "agent": "safety_detector",
                    "dependencies": ["t4"],
                },
                {
                    "id": "t8",
                    "name": "Generate report after safety validation",
                    "agent": "report_generator",
                    "dependencies": ["t5", "t7"],
                },
            ],
            notes="Added safety validation because this is a flight-critical drone system",
        )
    elif interactive:
        # Ask the human
        print("\n  Options: approve / modify / reject")
        decision = input("  Decision [approve]: ").strip().lower() or "approve"
        notes = input("  Notes (optional): ").strip()
        review = PlanReviewInput(
            decision=ReviewDecision(decision),
            notes=notes,
        )
    else:
        # Auto-approve
        print("\n  [Auto] Approving plan as-is")
        review = PlanReviewInput(decision=ReviewDecision.APPROVE)

    # Apply edits
    if review.decision == ReviewDecision.REJECT:
        print("\n  Plan REJECTED. Session aborted.")
        sys.exit(0)

    updates = apply_review_edits(state, review, AVAILABLE_AGENTS)
    state = {**state, **updates}

    # Show modified plan if edits were made
    if review.decision == ReviewDecision.MODIFY:
        snapshot = build_review_snapshot(state, AVAILABLE_AGENTS)
        print("\n  MODIFIED PLAN:")
        print(render_plan_review_rich(snapshot))

    return state


# ---------------------------------------------------------------------------
# Stage 2: Execute pipeline
# ---------------------------------------------------------------------------


def demo_execute(state: dict) -> dict:
    """Run the dispatcher."""
    print("\n" + "=" * 72)
    print("  STAGE 2: PIPELINE EXECUTION")
    print("=" * 72)

    dispatcher = create_default_dispatcher()
    step = 0
    completed: set[str] = set()

    while True:
        graph = get_task_graph(state)
        if graph.is_complete:
            break

        ready = graph.ready_tasks()
        if not ready:
            print(f"\n  Pipeline stuck at step {step}")
            break

        for task in ready:
            print(f"\n  [{task.id}] {task.agent}: {task.name}")

        state = dispatcher.step(state)
        step += 1

        graph = get_task_graph(state)
        for task in graph.nodes.values():
            if task.id not in completed and task.status.value == "completed":
                completed.add(task.id)
                summary = (task.result or {}).get("summary", "")
                if summary:
                    print(f"         -> {summary}")

    print(f"\n  Pipeline completed in {step} steps")
    return state


# ---------------------------------------------------------------------------
# Stage 3: Optimization with human steering
# ---------------------------------------------------------------------------


def demo_optimization_review(
    state: dict,
    max_iterations: int = 5,
    interactive: bool = False,
) -> dict:
    """Demonstrate optimization loop with human steering at each iteration."""

    ppa = state.get("ppa_metrics", {})
    verdicts = ppa.get("verdicts", {})
    if all(v == "PASS" for v in verdicts.values()):
        print("\n  All constraints PASS — no optimization needed.")
        return state

    print("\n" + "=" * 72)
    print("  STAGE 3: OPTIMIZATION WITH HUMAN STEERING")
    print("=" * 72)

    for iteration in range(1, max_iterations + 1):
        state["iteration"] = iteration

        # Build and display the optimization review snapshot
        snapshot = build_optimization_review_snapshot(state)
        print()
        print(render_optimization_review(snapshot))

        # Get human steering
        if interactive:
            print("\n  Options: continue / accept / redirect / explore_more / stop")
            decision = input("  Decision [continue]: ").strip().lower() or "continue"
            focus = None
            force = None
            if decision == "redirect":
                focus = input("  Focus objective (power/latency/cost): ").strip() or None
                force = input("  Force strategy (optional): ").strip() or None
            steering = OptimizationSteeringInput(
                decision=SteeringDecision(decision),
                focus_objective=focus,
                force_strategy=force,
            )
        else:
            # Auto-continue, but accept if all pass
            if snapshot.all_pass:
                steering = OptimizationSteeringInput(decision=SteeringDecision.ACCEPT)
            else:
                steering = OptimizationSteeringInput(decision=SteeringDecision.CONTINUE)

        # Apply steering
        steering_updates = apply_steering_input(state, steering)
        state = {**state, **steering_updates}

        if steering.decision in (SteeringDecision.ACCEPT, SteeringDecision.STOP):
            print(f"\n  Human accepted design at iteration {iteration}")
            break

        # Run optimizer
        task = TaskNode(
            id=f"opt_{iteration}",
            name=f"Optimization iteration {iteration}",
            agent="design_optimizer",
        )
        opt_result = design_optimizer(task, state)

        if not opt_result.get("applied"):
            print(f"\n  {opt_result.get('summary', 'No strategies available')}")
            break

        # Merge optimizer updates
        for k, v in opt_result.get("_state_updates", {}).items():
            state[k] = v

        strategy = opt_result.get("strategy", "?")
        print(f"\n  Applied: {strategy}")

        # Re-assess PPA
        assess_task = TaskNode(
            id=f"reassess_{iteration}",
            name=f"Re-assess PPA after {strategy}",
            agent="ppa_assessor",
        )
        assess_result = ppa_assessor(assess_task, state)
        for k, v in assess_result.get("_state_updates", {}).items():
            state[k] = v

        # Record in optimization history
        ppa = state.get("ppa_metrics", {})
        history = list(state.get("optimization_history", []))
        history.append(
            {
                "iteration": iteration,
                "ppa_snapshot": {
                    "power_watts": ppa.get("power_watts"),
                    "latency_ms": ppa.get("latency_ms"),
                    "cost_usd": ppa.get("cost_usd"),
                },
                "verdicts": dict(ppa.get("verdicts", {})),
            }
        )
        state["optimization_history"] = history

        state = record_decision(
            state,
            agent="design_optimizer",
            action=f"Applied optimization: {strategy}",
            rationale=opt_result.get("summary", ""),
        )

        # Check convergence
        verdicts = ppa.get("verdicts", {})
        if all(v == "PASS" for v in verdicts.values()):
            print(f"\n  ALL CONSTRAINTS PASS after {iteration} iteration(s)")
            break

    # Show final optimization review
    snapshot = build_optimization_review_snapshot(state)
    print()
    print(render_optimization_review(snapshot))

    return state


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Interactive SoC Design Review Demo")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Enable interactive prompts at review stages",
    )
    parser.add_argument(
        "--modify",
        action="store_true",
        help="Demonstrate programmatic plan modification",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="Max optimization iterations (default: 5)",
    )
    args = parser.parse_args()

    # 1. Create initial state and plan
    state = create_initial_soc_state(
        goal=GOAL,
        constraints=CONSTRAINTS,
        use_case="delivery_drone",
        platform="drone",
    )

    planner = PlannerNode(static_plan=PLAN)
    plan_updates = planner(state)
    state = {**state, **plan_updates}

    # 2. Plan review
    state = demo_plan_review(state, interactive=args.interactive, modify=args.modify)

    # 3. Execute pipeline
    state = demo_execute(state)

    # 4. Optimization with steering
    state = demo_optimization_review(
        state,
        max_iterations=args.max_iterations,
        interactive=args.interactive,
    )

    # Summary
    print("\n" + "=" * 72)
    print("  DEMO COMPLETE")
    print("=" * 72)
    decisions = state.get("history", [])
    print(f"\n  Decisions recorded: {len(decisions)}")
    rationale = state.get("design_rationale", [])
    if rationale:
        print("\n  Design journey:")
        for r in rationale:
            print(f"    {r}")
    print()


if __name__ == "__main__":
    main()
