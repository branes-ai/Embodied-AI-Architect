"""Planner agent for SoC design task decomposition.

The Planner is the "brain" of the agentic system. Given a design goal and
constraints, it decomposes the problem into a TaskGraph — a DAG of specialist
tasks with dependencies and success criteria.

The Planner operates as a LangGraph node: it reads goal/constraints from
SoCDesignState and writes back a populated task_graph.

Usage:
    from embodied_ai_architect.graphs.planner import PlannerNode, create_planner_node

    # As a LangGraph node
    node_fn = create_planner_node(llm_client)
    updated_state = node_fn(state)

    # Direct usage
    planner = PlannerNode(llm_client)
    updated_state = planner(state)
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Optional, Protocol

from embodied_ai_architect.graphs.soc_state import (
    DesignStatus,
    SoCDesignState,
    get_constraints,
    record_decision,
    set_task_graph,
)
from embodied_ai_architect.graphs.task_graph import TaskGraph, TaskNode

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM protocol (for dependency injection / testing)
# ---------------------------------------------------------------------------


class LLMProtocol(Protocol):
    """Minimal interface the Planner needs from an LLM client."""

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
    ) -> Any: ...


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

PLANNER_SYSTEM_PROMPT = """\
You are the SoC Design Planner for an agentic hardware design system.

Your job: given a design goal and constraints, decompose it into a task graph
(a DAG of tasks with dependencies). Each task will be executed by a specialist
agent.

## Available Specialist Agents

| Agent Name | Capability |
|---|---|
| workload_analyzer | Analyze AI model/application workload: operator graph, compute/memory requirements, data flow |
| hw_explorer | Enumerate and score hardware candidates against constraints; Pareto analysis |
| architecture_composer | Compose SoC architecture: map operators to accelerators, memory hierarchy, interconnect |
| ppa_assessor | Evaluate a design against PPA constraints (power, performance, area); identify bottlenecks |
| design_explorer | Explore design variations to find Pareto-optimal points |
| moo_explorer | Run multi-objective optimization (MAP-Elites → BO/NSGA-III) over the design space; populates pareto_points and moo_results. Schedule after hw_explorer; can run in parallel with architecture_composer since both consume hw_candidates and workload_profile. |
| swap_explorer | Run 6-objective SWaP-C optimization (power/latency/area/cost/weight/volume); populates swap_assessment and system_bom. Use for system-level (not just SoC-level) design questions. |
| critic | Review overall design quality, challenge assumptions, suggest improvements |
| report_generator | Generate design report with trade-off analysis and recommendations |
| kpu_configurator | Size KPU micro-architecture components for workload |
| floorplan_validator | Check 2D die area feasibility at target process node |
| bandwidth_validator | Verify ingress/egress bandwidth matching through memory hierarchy |
| kpu_optimizer | Adjust KPU config to fix floorplan/bandwidth violations |
| rtl_generator | Generate RTL for KPU sub-components from templates (requires rtl_enabled) |
| rtl_ppa_assessor | Aggregate RTL synthesis metrics and refine PPA area estimate |

## Default Pipeline (when nothing special is requested)

Always include `moo_explorer` after `hw_explorer` (in parallel with `architecture_composer`) so the design session has a populated Pareto frontier for the architect to inspect. Skip it ONLY if the user explicitly says "fast" or "skip optimization" — in that case the session will have no pareto_points and the architect tools will be limited.

The canonical default pipeline is:

```
workload_analyzer → hw_explorer ┬→ architecture_composer ┐
                                └→ moo_explorer ─────────┴→ ppa_assessor → critic → report_generator
```

## Task Graph Rules

1. Each task must specify: id, name, agent, dependencies (list of task IDs)
2. Dependencies must form a DAG (no cycles)
3. Tasks with no dependencies can execute in parallel
4. Use descriptive task names that explain WHAT the task accomplishes
5. Keep the graph minimal — only the tasks needed, no unnecessary steps
6. If constraints are vague, add an early task to refine them

## Output Format

Respond with a JSON object containing a "tasks" array. Each task object has:
- "id": unique string (e.g. "t1", "t2")
- "name": human-readable description of what the task does
- "agent": one of the agent names from the table above
- "dependencies": list of task IDs that must complete first (empty list if none)
- "preconditions": list of what must be true before this task runs (optional)
- "postconditions": list of what must be true after this task completes (optional)

Example:
```json
{
  "tasks": [
    {"id": "t1", "name": "Analyze perception workload", "agent": "workload_analyzer", "dependencies": []},
    {"id": "t2", "name": "Enumerate feasible hardware", "agent": "hw_explorer", "dependencies": ["t1"]},
    {"id": "t3", "name": "Compose SoC architecture", "agent": "architecture_composer", "dependencies": ["t2"]},
    {"id": "t4", "name": "Explore Pareto frontier", "agent": "moo_explorer", "dependencies": ["t2"]},
    {"id": "t5", "name": "Assess PPA metrics", "agent": "ppa_assessor", "dependencies": ["t3", "t4"]},
    {"id": "t6", "name": "Review design", "agent": "critic", "dependencies": ["t5"]},
    {"id": "t7", "name": "Generate design report", "agent": "report_generator", "dependencies": ["t6"]}
  ]
}
```

Respond ONLY with the JSON object. No explanation, no markdown fences, no preamble.
"""


# ---------------------------------------------------------------------------
# Plan parsing
# ---------------------------------------------------------------------------


def parse_plan_json(raw: str) -> list[dict[str, Any]]:
    """Parse the LLM's JSON response into a list of task dicts.

    Handles common LLM formatting quirks:
    - Markdown code fences around JSON
    - Extra whitespace
    - Missing optional fields

    Raises:
        ValueError: If the JSON is invalid or missing required fields.
    """
    text = raw.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json or ```) and last line (```)
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse plan JSON: {e}\nRaw text: {text[:500]}")

    if isinstance(data, dict) and "tasks" in data:
        tasks = data["tasks"]
    elif isinstance(data, list):
        tasks = data
    else:
        raise ValueError(
            f"Expected a JSON object with 'tasks' key or a JSON array, got: {type(data)}"
        )

    # Validate required fields
    for i, task in enumerate(tasks):
        for field in ("id", "name", "agent"):
            if field not in task:
                raise ValueError(f"Task at index {i} missing required field '{field}'")
        if "dependencies" not in task:
            task["dependencies"] = []

    return tasks


def tasks_to_graph(task_dicts: list[dict[str, Any]]) -> TaskGraph:
    """Convert a list of parsed task dicts into a TaskGraph.

    Raises:
        ValueError: If task construction or graph validation fails.
    """
    nodes = []
    for td in task_dicts:
        nodes.append(
            TaskNode(
                id=td["id"],
                name=td["name"],
                agent=td["agent"],
                dependencies=td.get("dependencies", []),
                preconditions=td.get("preconditions", []),
                postconditions=td.get("postconditions", []),
            )
        )

    graph = TaskGraph()
    graph.add_tasks(nodes)
    return graph


# ---------------------------------------------------------------------------
# Planner node
# ---------------------------------------------------------------------------


class PlannerNode:
    """LangGraph-compatible node that decomposes a goal into a task graph.

    Can operate in two modes:
    1. LLM mode: uses an LLM client to generate the plan
    2. Static mode: uses a pre-built plan (for testing or deterministic workflows)
    """

    def __init__(
        self,
        llm: Optional[LLMProtocol] = None,
        system_prompt: str = PLANNER_SYSTEM_PROMPT,
        static_plan: Optional[list[dict[str, Any]]] = None,
    ):
        """Initialize the Planner.

        Args:
            llm: LLM client implementing chat(). Required if static_plan not provided.
            system_prompt: System prompt for the LLM.
            static_plan: Pre-built plan (list of task dicts) for testing.
                         If provided, LLM is not called.
        """
        if llm is None and static_plan is None:
            raise ValueError("Either llm or static_plan must be provided")
        self.llm = llm
        self.system_prompt = system_prompt
        self.static_plan = static_plan

    def __call__(self, state: SoCDesignState) -> dict[str, Any]:
        """Execute the planner node.

        Reads goal and constraints from state, produces a task graph,
        and returns state updates for LangGraph merging.

        Args:
            state: Current SoC design state.

        Returns:
            Dict of state updates (task_graph, status, history, etc.)
        """
        goal = state.get("goal", "")
        constraints = get_constraints(state)
        use_case = state.get("use_case", "")
        platform = state.get("platform", "")

        if self.static_plan is not None:
            task_dicts = self.static_plan
        else:
            task_dicts = self._plan_with_llm(goal, constraints, use_case, platform)

        graph = tasks_to_graph(task_dicts)

        logger.info(
            "Planner created task graph with %d tasks: %s",
            len(graph.nodes),
            [f"{t.id}:{t.agent}" for t in graph.nodes.values()],
        )

        # Build state updates
        updated = set_task_graph(state, graph)
        updated = record_decision(
            updated,
            agent="planner",
            action=f"Created task graph with {len(graph.nodes)} tasks",
            rationale=f"Decomposed goal into specialist tasks for: {goal[:100]}",
            data={"task_ids": list(graph.nodes.keys())},
        )
        updated["status"] = DesignStatus.EXPLORING.value

        return {
            "task_graph": updated["task_graph"],
            "status": updated["status"],
            "history": updated["history"],
            "design_rationale": updated["design_rationale"],
        }

    def _plan_with_llm(
        self,
        goal: str,
        constraints: Any,
        use_case: str,
        platform: str,
    ) -> list[dict[str, Any]]:
        """Call the LLM to generate a plan."""
        user_message = self._build_user_message(goal, constraints, use_case, platform)

        response = self.llm.chat(  # type: ignore[union-attr]
            messages=[{"role": "user", "content": user_message}],
            system=self.system_prompt,
        )

        raw = response.text
        if not raw.strip():
            raise ValueError("LLM returned empty response for planning request")

        return parse_plan_json(raw)

    def _build_user_message(
        self,
        goal: str,
        constraints: Any,
        use_case: str,
        platform: str,
    ) -> str:
        """Build the user message for the LLM."""
        parts = [f"Design Goal: {goal}"]

        if use_case:
            parts.append(f"Use Case: {use_case}")
        if platform:
            parts.append(f"Platform: {platform}")

        # Include non-None constraints
        constraint_lines = []
        constraint_dict = constraints.model_dump(exclude_none=True)
        for key, value in constraint_dict.items():
            if key == "custom" and not value:
                continue
            label = key.replace("_", " ").title()
            constraint_lines.append(f"  - {label}: {value}")

        if constraint_lines:
            parts.append("Constraints:")
            parts.extend(constraint_lines)

        parts.append("\nDecompose this into a task graph.")
        return "\n".join(parts)


def create_planner_node(
    llm: Optional[LLMProtocol] = None,
    static_plan: Optional[list[dict[str, Any]]] = None,
    system_prompt: str = PLANNER_SYSTEM_PROMPT,
) -> Callable[[SoCDesignState], dict[str, Any]]:
    """Factory function to create a planner node for LangGraph.

    Args:
        llm: LLM client. Required if static_plan not provided.
        static_plan: Pre-built plan for testing.
        system_prompt: Custom system prompt override.

    Returns:
        A callable suitable for StateGraph.add_node().
    """
    return PlannerNode(llm=llm, static_plan=static_plan, system_prompt=system_prompt)
