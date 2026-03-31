"""LLM tool definitions for interactive SoC design with human review.

Follows the get_*_tool_definitions() + create_*_tool_executors() pattern.
Provides tools for the ArchitectAgent to invoke the planner → review →
dispatch → optimize flow, with human-in-the-loop at plan review and
optimization steering stages.

Usage:
    from embodied_ai_architect.llm.soc_design_tools import (
        get_soc_design_tool_definitions,
        create_soc_design_tool_executors,
    )
"""

from __future__ import annotations

import json
import traceback
from typing import Any, Callable

# Module-level session state for interactive design
_active_runner = None
_active_status: str = ""
_active_state: dict[str, Any] = {}


def get_soc_design_tool_definitions() -> list[dict[str, Any]]:
    """Get tool definitions for SoC design in Anthropic format."""
    return [
        {
            "name": "design_soc",
            "description": (
                "Start a new SoC design session with human-in-the-loop review. "
                "Runs the planner to decompose the goal into a task graph, then "
                "pauses for human review. Returns the plan review snapshot showing "
                "the task DAG, constraints, execution schedule, and available agents. "
                "The human can then approve, modify, or reject the plan."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "goal": {
                        "type": "string",
                        "description": (
                            "Natural language design objective "
                            "(e.g., 'Design a drone perception SoC: <5W, <33ms, <$30')"
                        ),
                    },
                    "use_case": {
                        "type": "string",
                        "description": "Application type (e.g., 'delivery_drone', 'warehouse_amr')",
                    },
                    "platform": {
                        "type": "string",
                        "description": "Platform type (e.g., 'drone', 'quadruped', 'amr')",
                    },
                    "max_power_watts": {
                        "type": "number",
                        "description": "Power budget in watts (optional)",
                    },
                    "max_latency_ms": {
                        "type": "number",
                        "description": "Latency target in ms (optional)",
                    },
                    "max_cost_usd": {
                        "type": "number",
                        "description": "Cost budget in USD (optional)",
                    },
                    "max_area_mm2": {
                        "type": "number",
                        "description": "Die area budget in mm² (optional)",
                    },
                    "target_process_nm": {
                        "type": "integer",
                        "description": "Target process node in nm (optional, e.g. 28, 16, 7)",
                    },
                    "max_iterations": {
                        "type": "integer",
                        "description": "Maximum optimization iterations (default: 20)",
                    },
                },
                "required": ["goal"],
            },
        },
        {
            "name": "review_plan",
            "description": (
                "Review and optionally modify the SoC design plan. Call this after "
                "design_soc returns a plan review snapshot. You can approve the plan "
                "as-is, modify it (add/remove tasks, reassign agents, reorder), or "
                "reject it entirely. Returns the updated plan or advances to execution."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "decision": {
                        "type": "string",
                        "enum": ["approve", "modify", "reject", "skip"],
                        "description": "Review decision",
                    },
                    "tasks_to_add": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "name": {"type": "string"},
                                "agent": {"type": "string"},
                                "dependencies": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                            "required": ["id", "name", "agent"],
                        },
                        "description": "Tasks to add to the graph",
                    },
                    "tasks_to_remove": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Task IDs to remove",
                    },
                    "agent_reassignments": {
                        "type": "object",
                        "description": "task_id -> new agent name",
                    },
                    "constraint_overrides": {
                        "type": "object",
                        "description": "Override constraint fields (e.g. max_power_watts)",
                    },
                    "notes": {
                        "type": "string",
                        "description": "Architect notes for design history",
                    },
                },
                "required": ["decision"],
            },
        },
        {
            "name": "steer_optimization",
            "description": (
                "Direct the SoC optimization loop at an optimization review point. "
                "Shows the current Pareto frontier, constraint slackness (how tight "
                "or slack each constraint is), optimization trajectory, and available "
                "strategies. You can continue, accept the current design, redirect "
                "focus, relax/tighten constraints, or force a specific strategy. "
                "Returns the next optimization review snapshot."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "decision": {
                        "type": "string",
                        "enum": ["continue", "accept", "redirect", "explore_more", "stop"],
                        "description": (
                            "Steering decision: continue (let optimizer choose), "
                            "accept (take current design), redirect (change focus), "
                            "explore_more (broader search), stop (report current best)"
                        ),
                    },
                    "focus_objective": {
                        "type": "string",
                        "description": "Objective to prioritize (e.g. 'power', 'latency', 'cost')",
                    },
                    "force_strategy": {
                        "type": "string",
                        "description": "Force a specific optimization strategy name",
                    },
                    "constraint_relaxation": {
                        "type": "object",
                        "description": "Constraint name -> new relaxed value",
                    },
                    "constraint_tightening": {
                        "type": "object",
                        "description": "Constraint name -> new tighter value",
                    },
                    "notes": {
                        "type": "string",
                        "description": "Architect reasoning for design history",
                    },
                },
                "required": ["decision"],
            },
        },
        {
            "name": "show_optimization_status",
            "description": (
                "Show the current optimization state without steering. Displays "
                "the Pareto frontier, constraint slackness analysis, optimization "
                "trajectory, strategy analysis, and design rationale. Read-only — "
                "does not advance the optimization loop."
            ),
            "input_schema": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
        {
            "name": "show_plan",
            "description": (
                "Show the current task graph plan. Returns the plan review snapshot "
                "with task DAG visualization, execution schedule, and constraint "
                "summary. Read-only."
            ),
            "input_schema": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
        {
            "name": "qualify_goal",
            "description": (
                "IMPORTANT: Call this BEFORE design_soc. Checks whether a design "
                "goal is specific enough to produce meaningful results. Returns a "
                "tangibility scorecard showing which dimensions are specified vs. "
                "missing (platform, perception tasks, control output, power envelope, "
                "environment). If the goal is underspecified, returns the next "
                "question to ask the user. You MUST qualify the goal before calling "
                "design_soc — design_soc will refuse underspecified goals."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "goal": {
                        "type": "string",
                        "description": "Natural language design goal to qualify",
                    },
                    "domain": {
                        "type": "string",
                        "enum": ["drone", "ugv", "robot_arm"],
                        "description": "Platform domain (auto-detected if not specified)",
                    },
                    "question_id": {
                        "type": "string",
                        "description": (
                            "Question being answered (from previous qualify_goal result). "
                            "Omit on first call."
                        ),
                    },
                    "answer": {
                        "description": (
                            "Answer to the question — string for single choice, "
                            "array of strings for multi-choice, number for numeric"
                        ),
                    },
                },
                "required": ["goal"],
            },
        },
    ]


def create_soc_design_tool_executors() -> dict[str, Callable]:
    """Create tool executor functions for SoC design."""

    def design_soc(
        goal: str,
        use_case: str = "",
        platform: str = "",
        max_power_watts: float | None = None,
        max_latency_ms: float | None = None,
        max_cost_usd: float | None = None,
        max_area_mm2: float | None = None,
        target_process_nm: int | None = None,
        max_iterations: int = 20,
    ) -> str:
        """Start a new interactive SoC design session."""
        global _active_runner, _active_status, _active_state

        try:
            from embodied_ai_architect.graphs.soc_runner import SoCDesignRunner
            from embodied_ai_architect.graphs.soc_state import DesignConstraints

            # Build constraints
            constraint_kwargs: dict[str, Any] = {}
            if max_power_watts is not None:
                constraint_kwargs["max_power_watts"] = max_power_watts
            if max_latency_ms is not None:
                constraint_kwargs["max_latency_ms"] = max_latency_ms
            if max_cost_usd is not None:
                constraint_kwargs["max_cost_usd"] = max_cost_usd
            if max_area_mm2 is not None:
                constraint_kwargs["max_area_mm2"] = max_area_mm2
            if target_process_nm is not None:
                constraint_kwargs["target_process_nm"] = target_process_nm

            constraints = DesignConstraints(**constraint_kwargs) if constraint_kwargs else None

            # Create LLM client for dynamic planning
            from embodied_ai_architect.llm.client import LLMClient

            llm = LLMClient()

            runner = SoCDesignRunner(
                llm=llm,
                human_review=True,
                optimization_review=True,
            )

            status, state = runner.start(
                goal=goal,
                constraints=constraints,
                use_case=use_case,
                platform=platform,
                max_iterations=max_iterations,
            )

            _active_runner = runner
            _active_status = status
            _active_state = dict(state)

            return _format_status_response(status, state)

        except Exception as e:
            return f"Error starting design session: {e}\n{traceback.format_exc()}"

    def review_plan(
        decision: str,
        tasks_to_add: list[dict] | None = None,
        tasks_to_remove: list[str] | None = None,
        agent_reassignments: dict[str, str] | None = None,
        constraint_overrides: dict[str, Any] | None = None,
        notes: str = "",
    ) -> str:
        """Review and optionally modify the plan."""
        global _active_runner, _active_status, _active_state

        if _active_runner is None:
            return "Error: No active design session. Call design_soc first."

        if _active_status != "review_plan":
            return f"Error: Session is in '{_active_status}' state, not 'review_plan'."

        try:
            review_input = {
                "decision": decision,
                "tasks_to_add": tasks_to_add or [],
                "tasks_to_remove": tasks_to_remove or [],
                "agent_reassignments": agent_reassignments or {},
                "constraint_overrides": constraint_overrides or {},
                "notes": notes,
            }

            status, state = _active_runner.step(review_input=review_input)
            _active_status = status
            _active_state = dict(state)

            return _format_status_response(status, state)

        except Exception as e:
            return f"Error during plan review: {e}\n{traceback.format_exc()}"

    def steer_optimization(
        decision: str,
        focus_objective: str | None = None,
        force_strategy: str | None = None,
        constraint_relaxation: dict[str, float] | None = None,
        constraint_tightening: dict[str, float] | None = None,
        notes: str = "",
    ) -> str:
        """Direct the optimization loop."""
        global _active_runner, _active_status, _active_state

        if _active_runner is None:
            return "Error: No active design session. Call design_soc first."

        if _active_status != "review_optimization":
            return f"Error: Session is in '{_active_status}' state, " "not 'review_optimization'."

        try:
            steering_input = {
                "decision": decision,
                "focus_objective": focus_objective,
                "force_strategy": force_strategy,
                "constraint_relaxation": constraint_relaxation or {},
                "constraint_tightening": constraint_tightening or {},
                "notes": notes,
            }

            status, state = _active_runner.step(steering_input=steering_input)
            _active_status = status
            _active_state = dict(state)

            return _format_status_response(status, state)

        except Exception as e:
            return f"Error during optimization steering: {e}\n{traceback.format_exc()}"

    def show_optimization_status() -> str:
        """Show current optimization state (read-only)."""
        if not _active_state:
            return "Error: No active design session."

        snapshot = _active_state.get("optimization_review_snapshot", {})
        if not snapshot:
            return "No optimization review data available yet."

        try:
            from embodied_ai_architect.graphs.optimization_review import (
                OptimizationReviewSnapshot,
                render_optimization_review,
            )

            snap = OptimizationReviewSnapshot(**snapshot)
            return render_optimization_review(snap)
        except Exception:
            return json.dumps(snapshot, indent=2, default=str)

    def show_plan() -> str:
        """Show current task graph plan (read-only)."""
        if not _active_state:
            return "Error: No active design session."

        snapshot = _active_state.get("review_snapshot", {})
        if not snapshot:
            return "No plan review data available yet."

        try:
            from embodied_ai_architect.graphs.review import (
                PlanReviewSnapshot,
                render_plan_review_rich,
            )

            snap = PlanReviewSnapshot(**snapshot)
            return render_plan_review_rich(snap)
        except Exception:
            return json.dumps(snapshot, indent=2, default=str)

    def qualify_goal(
        goal: str,
        domain: str | None = None,
        question_id: str | None = None,
        answer: Any = None,
    ) -> str:
        """Qualify a design goal — check if it's specific enough."""
        global _active_qualifier

        try:
            from embodied_ai_architect.qualification import GoalQualifier
            from embodied_ai_architect.qualification.qualifier import render_qualification_result

            # Initialize or reuse qualifier
            if not hasattr(qualify_goal, "_qualifier") or question_id is None:
                qualify_goal._qualifier = GoalQualifier()
                result = qualify_goal._qualifier.assess(goal, domain=domain)
            else:
                # Continuing a session — answer a question
                if answer is not None and question_id is not None:
                    result = qualify_goal._qualifier.answer(question_id, answer)
                else:
                    result = qualify_goal._qualifier.assess(goal, domain=domain)

            rendered = render_qualification_result(result)

            if result.is_tangible:
                inputs = qualify_goal._qualifier.to_design_inputs()
                rendered += (
                    "\n\nGOAL QUALIFIED. You can now call design_soc with:\n"
                    f"  goal: {inputs['goal']}\n"
                    f"  use_case: {inputs['use_case']}\n"
                    f"  platform: {inputs['platform']}\n"
                )
                constraints = inputs.get("constraints")
                if constraints:
                    data = constraints.model_dump(exclude_none=True, exclude_defaults=True)
                    for k, v in data.items():
                        rendered += f"  {k}: {v}\n"

            return rendered

        except Exception as e:
            return f"Error qualifying goal: {e}\n{traceback.format_exc()}"

    return {
        "design_soc": design_soc,
        "review_plan": review_plan,
        "steer_optimization": steer_optimization,
        "show_optimization_status": show_optimization_status,
        "show_plan": show_plan,
        "qualify_goal": qualify_goal,
    }


def _format_status_response(status: str, state: dict[str, Any]) -> str:
    """Format the response based on session status."""
    if status == "review_plan":
        snapshot = state.get("review_snapshot", {})
        try:
            from embodied_ai_architect.graphs.review import (
                PlanReviewSnapshot,
                render_plan_review_rich,
            )

            snap = PlanReviewSnapshot(**snapshot)
            rendered = render_plan_review_rich(snap)
            return (
                f"STATUS: review_plan\n\n{rendered}\n\n"
                "Use review_plan tool to approve, modify, or reject this plan."
            )
        except Exception:
            return f"STATUS: review_plan\n\n{json.dumps(snapshot, indent=2, default=str)}"

    elif status == "review_optimization":
        snapshot = state.get("optimization_review_snapshot", {})
        try:
            from embodied_ai_architect.graphs.optimization_review import (
                OptimizationReviewSnapshot,
                render_optimization_review,
            )

            snap = OptimizationReviewSnapshot(**snapshot)
            rendered = render_optimization_review(snap)
            return (
                f"STATUS: review_optimization\n\n{rendered}\n\n"
                "Use steer_optimization tool to direct the optimization loop."
            )
        except Exception:
            return (
                f"STATUS: review_optimization\n\n" f"{json.dumps(snapshot, indent=2, default=str)}"
            )

    elif status == "complete":
        design_status = state.get("status", "unknown")
        ppa = state.get("ppa_metrics", {})
        verdicts = ppa.get("verdicts", {})
        iteration = state.get("iteration", 0)

        # Build summary
        lines = [f"STATUS: complete ({design_status})"]
        lines.append(f"Iterations: {iteration}")
        if verdicts:
            lines.append(f"Verdicts: {json.dumps(verdicts)}")
        if ppa:
            parts = []
            if ppa.get("power_watts") is not None:
                parts.append(f"Power: {ppa['power_watts']:.1f}W")
            if ppa.get("latency_ms") is not None:
                parts.append(f"Latency: {ppa['latency_ms']:.1f}ms")
            if ppa.get("area_mm2") is not None:
                parts.append(f"Area: {ppa['area_mm2']:.1f}mm²")
            if parts:
                lines.append(f"PPA: {', '.join(parts)}")

        rationale = state.get("design_rationale", [])
        if rationale:
            lines.append("\nDesign Journey:")
            for r in rationale[-10:]:
                lines.append(f"  {r}")

        return "\n".join(lines)

    else:
        return f"STATUS: {status} (session is running, no review point reached)"
