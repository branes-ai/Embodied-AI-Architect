"""LangGraph outer loop for iterative SoC design optimization.

Wraps the existing Dispatcher in a StateGraph that iterates until all PPA
constraints are met or the iteration limit is reached.

Two-level architecture:
- Outer loop (this module): LangGraph StateGraph with optional human review:
    planner → [plan_review] → dispatch → evaluate → [optimize → dispatch | report → END]
- Inner loop (dispatcher.py): Dispatcher.run() walks the TaskGraph DAG of specialist agents

When human_review=True, the graph includes a plan_review node between planner
and dispatch that interrupts for human inspection and modification of the task graph.

When optimization_review=True, the evaluate node is enhanced with an optimization
transparency snapshot showing Pareto frontier, constraint slackness, trajectory,
and strategy analysis — enabling human steering of the optimization loop.

Usage:
    from embodied_ai_architect.graphs.soc_graph import build_soc_design_graph
    from embodied_ai_architect.graphs.specialists import create_default_dispatcher
    from embodied_ai_architect.graphs.planner import PlannerNode

    dispatcher = create_default_dispatcher()
    planner = PlannerNode(static_plan=PLAN)

    # Non-interactive (original behavior)
    graph = build_soc_design_graph(dispatcher=dispatcher, planner=planner)
    result = graph.invoke(initial_state)

    # Interactive with human review
    graph = build_soc_design_graph(
        dispatcher=dispatcher, planner=planner,
        human_review=True, optimization_review=True,
    )
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Callable, Optional

from embodied_ai_architect.graphs.dispatcher import Dispatcher
from embodied_ai_architect.graphs.governance import GovernanceGuard
from embodied_ai_architect.graphs.optimizer import design_optimizer
from embodied_ai_architect.graphs.planner import PlannerNode
from embodied_ai_architect.graphs.review import _make_plan_review_node
from embodied_ai_architect.graphs.optimization_review import make_enhanced_evaluate_node
from embodied_ai_architect.graphs.soc_state import (
    DesignStatus,
    SoCDesignState,
    record_decision,
)
from embodied_ai_architect.graphs.task_graph import TaskGraph, TaskNode

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------


def _make_planner_node(planner: PlannerNode) -> Callable[[SoCDesignState], dict[str, Any]]:
    """Create the planner node function."""

    def planner_node(state: SoCDesignState) -> dict[str, Any]:
        logger.info("Outer loop: planner node")
        return planner(state)

    return planner_node


def _make_dispatch_node(dispatcher: Dispatcher) -> Callable[[SoCDesignState], dict[str, Any]]:
    """Create the dispatch node function.

    On iteration 0, runs the full inner DAG.
    On iteration > 0, creates a minimal re-eval plan (just ppa_assessor)
    and runs that instead.
    """

    def dispatch_node(state: SoCDesignState) -> dict[str, Any]:
        iteration = state.get("iteration", 0)
        logger.info("Outer loop: dispatch node (iteration %d)", iteration)

        if iteration > 0:
            # Re-evaluation: only run ppa_assessor on the modified state
            graph = TaskGraph()
            graph.add_task(
                TaskNode(id="reeval_ppa", name="Re-evaluate PPA metrics", agent="ppa_assessor")
            )
            reeval_state = {**state, "task_graph": graph.to_dict()}
            result = dispatcher.run(reeval_state)
            # Extract just the updates we care about
            return {
                "ppa_metrics": result.get("ppa_metrics", state.get("ppa_metrics", {})),
                "task_graph": state.get("task_graph"),  # preserve original task graph
                "status": DesignStatus.OPTIMIZING.value,
            }
        else:
            # Full execution on first pass
            result = dispatcher.run(state)
            return {
                "ppa_metrics": result.get("ppa_metrics", {}),
                "workload_profile": result.get(
                    "workload_profile", state.get("workload_profile", {})
                ),
                "hardware_candidates": result.get(
                    "hardware_candidates", state.get("hardware_candidates", [])
                ),
                "selected_architecture": result.get(
                    "selected_architecture", state.get("selected_architecture", {})
                ),
                "ip_blocks": result.get("ip_blocks", state.get("ip_blocks", [])),
                "memory_map": result.get("memory_map", state.get("memory_map", {})),
                "interconnect": result.get("interconnect", state.get("interconnect", {})),
                "task_graph": result.get("task_graph", state.get("task_graph")),
                "history": result.get("history", state.get("history", [])),
                "design_rationale": result.get(
                    "design_rationale", state.get("design_rationale", [])
                ),
                "working_memory": result.get("working_memory", state.get("working_memory", {})),
                # Issue #27: forward MOO outputs (moo_explorer / swap_explorer)
                # so the LangGraph outer state retains them. Without these keys
                # the runner state ends up with empty pareto_points even though
                # moo_explorer ran successfully inside the inner dispatcher.
                "pareto_points": result.get("pareto_points", state.get("pareto_points", [])),
                "pareto_results": result.get("pareto_results", state.get("pareto_results", {})),
                "pareto_frontier_history": result.get(
                    "pareto_frontier_history", state.get("pareto_frontier_history", [])
                ),
                "moo_results": result.get("moo_results", state.get("moo_results", {})),
                "swap_results": result.get("swap_results", state.get("swap_results", {})),
                "system_bom": result.get("system_bom", state.get("system_bom", {})),
                "status": DesignStatus.OPTIMIZING.value,
            }

    return dispatch_node


def _make_evaluate_node(
    governance: Optional[GovernanceGuard] = None,
) -> Callable[[SoCDesignState], dict[str, Any]]:
    """Create the evaluate node function.

    Checks verdicts, governance limits, records optimization_history entry,
    and sets next_action to either 'optimize' or 'report'.
    """

    def evaluate_node(state: SoCDesignState) -> dict[str, Any]:
        iteration = state.get("iteration", 0)
        ppa = state.get("ppa_metrics", {})
        verdicts = ppa.get("verdicts", {})
        max_iterations = state.get("max_iterations", 20)

        logger.info(
            "Outer loop: evaluate node (iteration %d, verdicts=%s)",
            iteration,
            verdicts,
        )

        # Record optimization history snapshot
        opt_history = list(state.get("optimization_history", []))
        opt_history.append(
            {
                "iteration": iteration,
                "timestamp": datetime.now().isoformat(),
                "ppa_snapshot": {
                    "power_watts": ppa.get("power_watts"),
                    "latency_ms": ppa.get("latency_ms"),
                    "area_mm2": ppa.get("area_mm2"),
                    "cost_usd": ppa.get("cost_usd"),
                },
                "verdicts": dict(verdicts),
            }
        )

        all_pass = all(v == "PASS" for v in verdicts.values()) if verdicts else False

        # Check governance limits
        guard = governance or GovernanceGuard.from_dict(state.get("governance", {}))
        over_iteration_limit = not guard.check_iteration_limit(iteration)
        over_max_iterations = iteration >= max_iterations

        if all_pass:
            next_action = "report"
            logger.info("All constraints PASS — routing to report")
        elif over_iteration_limit or over_max_iterations:
            next_action = "report"
            logger.info(
                "Iteration limit reached (%d) — routing to report with current best",
                iteration,
            )
        else:
            next_action = "optimize"
            logger.info("Constraints failing — routing to optimize (iteration %d)", iteration)

        return {
            "optimization_history": opt_history,
            "next_action": next_action,
        }

    return evaluate_node


def _make_optimize_node() -> Callable[[SoCDesignState], dict[str, Any]]:
    """Create the optimize node function.

    Calls design_optimizer to apply a strategy, then increments iteration.
    """

    def optimize_node(state: SoCDesignState) -> dict[str, Any]:
        iteration = state.get("iteration", 0)
        logger.info("Outer loop: optimize node (iteration %d)", iteration)

        task = TaskNode(
            id=f"optimize_{iteration}",
            name=f"Optimize design (iteration {iteration})",
            agent="design_optimizer",
        )

        result = design_optimizer(task, state)

        updates: dict[str, Any] = {
            "iteration": iteration + 1,
            "status": DesignStatus.OPTIMIZING.value,
        }

        # Merge state updates from optimizer
        state_updates = result.get("_state_updates", {})
        if state_updates:
            updates.update(state_updates)

        strategy = result.get("strategy", "none")
        updates_for_decision = record_decision(
            state,
            agent="design_optimizer",
            action=f"Applied optimization: {strategy}",
            rationale=result.get("summary", "Optimization applied"),
            data={"iteration": iteration, "strategy": strategy},
        )
        updates["history"] = updates_for_decision.get("history", state.get("history", []))
        updates["design_rationale"] = updates_for_decision.get(
            "design_rationale", state.get("design_rationale", [])
        )

        return updates

    return optimize_node


def _make_report_node(
    experience_cache: Any = None,
) -> Callable[[SoCDesignState], dict[str, Any]]:
    """Create the report node function.

    Generates final report. Optionally saves an experience episode.
    """

    def report_node(state: SoCDesignState) -> dict[str, Any]:
        logger.info("Outer loop: report node")

        ppa = state.get("ppa_metrics", {})
        verdicts = ppa.get("verdicts", {})
        all_pass = all(v == "PASS" for v in verdicts.values()) if verdicts else False
        iteration = state.get("iteration", 0)

        report = {
            "title": f"SoC Design Report: {state.get('goal', 'Unknown')}",
            "session_id": state.get("session_id", "unknown"),
            "use_case": state.get("use_case", ""),
            "platform": state.get("platform", ""),
            "outcome": "PASS" if all_pass else "FAIL",
            "iterations_used": iteration,
            "final_ppa": ppa,
            "optimization_history": state.get("optimization_history", []),
            "architecture": state.get("selected_architecture", {}),
            "design_rationale": state.get("design_rationale", []),
        }

        # Include KPU and RTL data when rtl_enabled
        if state.get("rtl_enabled", False):
            report["kpu_config"] = state.get("kpu_config", {})
            report["floorplan_estimate"] = state.get("floorplan_estimate", {})
            report["bandwidth_match"] = state.get("bandwidth_match", {})
            rtl_synth = state.get("rtl_synthesis_results", {})
            report["rtl_summary"] = {
                "modules_generated": len(state.get("rtl_modules", {})),
                "total_cells": sum(
                    r.get("area_cells", 0) for r in rtl_synth.values() if r.get("success")
                ),
            }

        # Save experience episode if cache is available
        if experience_cache is not None:
            try:
                _save_experience_episode(state, report, experience_cache)
            except Exception as e:
                logger.warning("Failed to save experience episode: %s", e)

        return {
            "status": DesignStatus.COMPLETE.value,
            "next_action": "end",
        }

    return report_node


def _save_experience_episode(state: SoCDesignState, report: dict, cache: Any) -> None:
    """Save a design episode to the experience cache."""
    from embodied_ai_architect.graphs.experience import DesignEpisode

    ppa = state.get("ppa_metrics", {})
    verdicts = ppa.get("verdicts", {})
    all_pass = all(v == "PASS" for v in verdicts.values()) if verdicts else False

    # KPU fields
    kpu_config = state.get("kpu_config", {})
    fp = state.get("floorplan_estimate", {})
    bw = state.get("bandwidth_match", {})
    rtl_synth = state.get("rtl_synthesis_results", {})

    episode = DesignEpisode(
        goal=state.get("goal", ""),
        use_case=state.get("use_case", ""),
        platform=state.get("platform", ""),
        constraints=state.get("constraints", {}),
        architecture_chosen=state.get("selected_architecture", {}).get("name", ""),
        hardware_selected=state.get("selected_architecture", {}).get("primary_compute", ""),
        ppa_achieved=ppa,
        constraint_verdicts=verdicts,
        outcome_score=1.0 if all_pass else 0.0,
        iterations_used=state.get("iteration", 0),
        key_decisions=[d.get("action", "") for d in state.get("history", [])],
        lessons_learned=[],
        optimization_trace=state.get("optimization_history", []),
        kpu_config_name=kpu_config.get("name"),
        kpu_process_nm=kpu_config.get("process_nm"),
        floorplan_area_mm2=fp.get("total_area_mm2"),
        bandwidth_balanced=bw.get("balanced"),
        rtl_modules_generated=len(state.get("rtl_modules", {})),
        rtl_total_cells=sum(r.get("area_cells", 0) for r in rtl_synth.values() if r.get("success")),
    )
    cache.save(episode)


# ---------------------------------------------------------------------------
# Routing function
# ---------------------------------------------------------------------------


def _route_after_evaluate(state: SoCDesignState) -> str:
    """Route from evaluate to either optimize or report."""
    return state.get("next_action", "report")


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_soc_design_graph(
    dispatcher: Dispatcher,
    planner: PlannerNode,
    governance: Optional[GovernanceGuard] = None,
    experience_cache: Any = None,
    checkpointer: Any = None,
    interrupt_at_evaluate: bool = False,
    human_review: bool = False,
    optimization_review: bool = False,
    available_agents: Optional[list[str]] = None,
) -> Any:
    """Build the LangGraph StateGraph for iterative SoC design.

    Base topology (5 nodes):
        planner → dispatch → evaluate → [optimize → dispatch | report → END]

    With human_review=True (6 nodes):
        planner → plan_review → dispatch → evaluate → [optimize → dispatch | report → END]

    With optimization_review=True, the evaluate node is enhanced with an
    optimization transparency snapshot (constraint slackness, trajectory,
    strategy analysis, Pareto frontier) stored in state for human inspection.

    Args:
        dispatcher: Pre-configured Dispatcher with registered agents.
        planner: PlannerNode (LLM or static).
        governance: Optional GovernanceGuard for budget/iteration limits.
        experience_cache: Optional ExperienceCache for episode storage.
        checkpointer: Optional LangGraph checkpointer for save/resume.
        interrupt_at_evaluate: If True, add interrupt_before on evaluate
            for HITL approval.
        human_review: If True, add plan_review node between planner and
            dispatch with interrupt_before for human inspection/editing.
        optimization_review: If True, enhance evaluate node with rich
            optimization transparency snapshot and steering support.
        available_agents: Agent names for plan review validation. If None,
            uses dispatcher.registered_agents.

    Returns:
        Compiled LangGraph StateGraph.
    """
    from langgraph.graph import END, StateGraph

    if available_agents is None:
        available_agents = dispatcher.registered_agents

    workflow = StateGraph(SoCDesignState)

    # Add nodes
    workflow.add_node("planner", _make_planner_node(planner))
    workflow.add_node("dispatch", _make_dispatch_node(dispatcher))
    workflow.add_node("optimize", _make_optimize_node())
    workflow.add_node("report", _make_report_node(experience_cache))

    # Evaluate node: enhanced or base
    base_evaluate = _make_evaluate_node(governance)
    if optimization_review:
        evaluate_fn = make_enhanced_evaluate_node(base_evaluate)
    else:
        evaluate_fn = base_evaluate
    workflow.add_node("evaluate", evaluate_fn)

    # Optional plan review node
    if human_review:
        workflow.add_node("plan_review", _make_plan_review_node(available_agents))

    # Entry point
    workflow.set_entry_point("planner")

    # Edges: planner → [plan_review →] dispatch
    if human_review:
        workflow.add_edge("planner", "plan_review")
        workflow.add_edge("plan_review", "dispatch")
    else:
        workflow.add_edge("planner", "dispatch")

    workflow.add_edge("dispatch", "evaluate")

    # Conditional: evaluate → optimize or report
    workflow.add_conditional_edges(
        "evaluate",
        _route_after_evaluate,
        {
            "optimize": "optimize",
            "report": "report",
        },
    )

    # optimize → dispatch (loop back for re-evaluation)
    workflow.add_edge("optimize", "dispatch")

    # report → END
    workflow.add_edge("report", END)

    # Compile with interrupts
    compile_kwargs: dict[str, Any] = {}
    if checkpointer is not None:
        compile_kwargs["checkpointer"] = checkpointer

    interrupt_nodes: list[str] = []
    if human_review:
        interrupt_nodes.append("plan_review")
    if interrupt_at_evaluate or optimization_review:
        interrupt_nodes.append("evaluate")
    if interrupt_nodes:
        compile_kwargs["interrupt_before"] = interrupt_nodes

    return workflow.compile(**compile_kwargs)
