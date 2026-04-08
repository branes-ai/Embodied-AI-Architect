"""Human-in-the-loop plan review for SoC design.

The ReviewNode sits between the Planner and Dispatcher in the LangGraph
outer loop. It renders the planner's task graph for human inspection,
pauses via LangGraph interrupt, accepts modifications, validates them,
and resumes execution with the approved (possibly modified) graph.

The review node is a pure state transform — it does NOT call input() or
block on I/O. The interrupt mechanism is at the graph compilation level
(interrupt_before), and the CLI/chat layer handles the I/O.

Usage:
    from embodied_ai_architect.graphs.review import (
        PlanReviewSnapshot,
        PlanReviewInput,
        ReviewDecision,
        build_review_snapshot,
        apply_review_edits,
        render_task_graph_ascii,
    )

    # In LangGraph:
    graph = build_soc_design_graph(human_review=True)
    # Graph will interrupt before "plan_review" node.
    # Caller reads state["review_snapshot"], displays it, collects input,
    # patches state with review_input, and resumes.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from embodied_ai_architect.graphs.soc_state import (
    DesignConstraints,
    SoCDesignState,
    get_constraints,
    get_task_graph,
    record_decision,
    set_task_graph,
)
from embodied_ai_architect.graphs.task_graph import TaskGraph, TaskNode, TaskStatus

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


class ReviewDecision(str, Enum):
    """Human decision at plan review."""

    APPROVE = "approve"
    MODIFY = "modify"
    REJECT = "reject"
    SKIP = "skip"


class PlanReviewInput(BaseModel):
    """Structured human input when reviewing/modifying the plan."""

    decision: ReviewDecision = ReviewDecision.APPROVE
    tasks_to_add: list[dict[str, Any]] = Field(
        default_factory=list,
        description="New task dicts: {id, name, agent, dependencies, ...}",
    )
    tasks_to_remove: list[str] = Field(
        default_factory=list,
        description="Task IDs to remove from the graph",
    )
    dependency_overrides: dict[str, list[str]] = Field(
        default_factory=dict,
        description="task_id -> new dependency list (reorder)",
    )
    agent_reassignments: dict[str, str] = Field(
        default_factory=dict,
        description="task_id -> new agent name",
    )
    constraint_overrides: dict[str, Any] = Field(
        default_factory=dict,
        description="Override DesignConstraints fields",
    )
    kpu_overrides: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "KPU micro-architecture overrides (issue #29). "
            "Flat dict of dotted-path keys, e.g. "
            '{"compute_tile.array_rows": 8, "noc.link_width_bits": 512}. '
            "Applied on top of the heuristic kpu_configurator output."
        ),
    )
    notes: str = Field(
        default="",
        description="Freeform architect notes recorded in design history",
    )


class PlanReviewSnapshot(BaseModel):
    """Serializable snapshot shown to the human at review time."""

    goal: str
    use_case: str
    platform: str
    constraints: dict[str, Any]
    constraints_summary: str
    task_graph_ascii: str
    task_table: list[dict[str, Any]]
    execution_order: list[str]
    parallel_groups: list[list[str]]
    available_agents: list[str]
    inferred_context: dict[str, Any]
    # Issue #29: when rtl_enabled, preview the KPU sizing the configurator
    # would produce, so the architect can override before dispatch starts.
    kpu_preview: dict[str, Any] | None = None
    kpu_preview_summary: str = ""


# ---------------------------------------------------------------------------
# ASCII rendering
# ---------------------------------------------------------------------------


def render_task_graph_ascii(graph: TaskGraph) -> str:
    """Render the task DAG as ASCII text.

    Produces a dependency-aware layout:
        [t1] Analyze workload (workload_analyzer) [READY]
          ├── [t2] Explore hardware (hw_explorer)
          │     └── [t3] Compose architecture (architecture_composer)
          │           └── [t4] Assess PPA (ppa_assessor)
          └── [t5] Validate safety (critic)
                └── [t4] (already shown)
    """
    if not graph.nodes:
        return "(empty task graph)"

    order = graph.execution_order()
    lines: list[str] = []

    # Find root tasks (no dependencies)
    roots = [tid for tid in order if not graph.nodes[tid].dependencies]

    # Track which tasks we've fully rendered
    rendered: set[str] = set()

    def _render_subtree(task_id: str, prefix: str, is_last: bool, depth: int) -> None:
        task = graph.nodes[task_id]
        connector = "└── " if is_last else "├── "
        if depth == 0:
            connector = ""
            child_prefix = prefix
        else:
            child_prefix = prefix + ("    " if is_last else "│   ")

        status_tag = f" [{task.status.value.upper()}]" if task.status != TaskStatus.PENDING else ""
        line = f"{prefix}{connector}[{task.id}] {task.name} ({task.agent}){status_tag}"
        lines.append(line)

        if task_id in rendered:
            return
        rendered.add(task_id)

        # Find children (tasks that depend on this one)
        children = [
            t.id for t in graph.nodes.values() if task_id in t.dependencies and t.id not in rendered
        ]
        # Sort children by execution order
        order_idx = {tid: i for i, tid in enumerate(order)}
        children.sort(key=lambda c: order_idx.get(c, 999))

        for i, child_id in enumerate(children):
            _render_subtree(child_id, child_prefix, i == len(children) - 1, depth + 1)

    for i, root_id in enumerate(roots):
        if i > 0:
            lines.append("")
        _render_subtree(root_id, "", i == len(roots) - 1, 0)

    return "\n".join(lines)


def render_task_table(graph: TaskGraph) -> list[dict[str, Any]]:
    """Produce a tabular representation for display."""
    order = graph.execution_order()
    table = []
    for tid in order:
        task = graph.nodes[tid]
        table.append(
            {
                "id": task.id,
                "name": task.name,
                "agent": task.agent,
                "status": task.status.value,
                "dependencies": ", ".join(task.dependencies) or "(none)",
                "preconditions": "; ".join(task.preconditions) or "-",
                "postconditions": "; ".join(task.postconditions) or "-",
            }
        )
    return table


def compute_parallel_groups(graph: TaskGraph) -> list[list[str]]:
    """Compute groups of tasks that can execute in parallel.

    Returns a list of groups, where each group is a list of task IDs
    that can run concurrently (all their dependencies are in prior groups).
    """
    if not graph.nodes:
        return []

    remaining = set(graph.nodes.keys())
    completed: set[str] = set()
    groups: list[list[str]] = []

    while remaining:
        # Find all tasks whose dependencies are all in completed
        group = []
        for tid in remaining:
            task = graph.nodes[tid]
            if all(dep in completed for dep in task.dependencies):
                group.append(tid)

        if not group:
            # Deadlock — shouldn't happen with valid DAG
            break

        groups.append(sorted(group))
        completed.update(group)
        remaining -= set(group)

    return groups


def summarize_constraints(constraints: DesignConstraints) -> str:
    """Format constraints as a human-readable summary."""
    parts = []
    data = constraints.model_dump(exclude_none=True, exclude_defaults=True)
    labels = {
        "max_power_watts": "Power",
        "max_latency_ms": "Latency",
        "max_area_mm2": "Area",
        "max_cost_usd": "Cost",
        "max_memory_mb": "Memory",
        "target_process_nm": "Process",
        "min_fps": "Min FPS",
        "min_accuracy_percent": "Min Accuracy",
        "safety_critical": "Safety Critical",
        "max_weight_grams": "Weight",
        "max_volume_cm3": "Volume",
    }
    units = {
        "max_power_watts": "W",
        "max_latency_ms": "ms",
        "max_area_mm2": "mm²",
        "max_cost_usd": "USD",
        "max_memory_mb": "MB",
        "target_process_nm": "nm",
        "min_fps": "FPS",
        "min_accuracy_percent": "%",
        "max_weight_grams": "g",
        "max_volume_cm3": "cm³",
    }
    for key, value in data.items():
        if key == "custom" and not value:
            continue
        label = labels.get(key, key.replace("_", " ").title())
        unit = units.get(key, "")
        if isinstance(value, bool):
            parts.append(f"{label}: {'Yes' if value else 'No'}")
        elif isinstance(value, (int, float)):
            parts.append(f"{label}: {value} {unit}".strip())
        else:
            parts.append(f"{label}: {value}")
    return " | ".join(parts) if parts else "(no constraints specified)"


# ---------------------------------------------------------------------------
# Snapshot builder
# ---------------------------------------------------------------------------


def _build_kpu_preview(
    state: SoCDesignState,
) -> tuple[dict[str, Any] | None, str]:
    """Eagerly preview the KPU config the configurator would produce.

    Used by `build_review_snapshot` when `rtl_enabled=True` so the architect
    can see and override the KPU micro-architecture before dispatch begins.
    Workload may be empty at plan-review time — the heuristic falls back to
    sensible defaults. If architect overrides are already on the state from
    a prior pass, they are applied to the preview.

    Returns (preview_dict, summary_string). Returns (None, "") on any error
    so plan review never breaks because of a preview-side failure.
    """
    try:
        from embodied_ai_architect.graphs.kpu_config import (
            apply_kpu_overrides,
            create_kpu_config,
        )

        constraints = get_constraints(state)
        workload = state.get("workload_profile", {}) or {}
        use_case = state.get("use_case", "")

        config = create_kpu_config(
            use_case,
            constraints.model_dump(exclude_none=True),
            workload,
        )

        overrides = state.get("kpu_config_overrides", {}) or {}
        if overrides:
            config = apply_kpu_overrides(config, overrides)

        ct = config.compute_tile
        mt = config.memory_tile
        summary_lines = [
            f"  Compute tiles: {config.num_compute_tiles} "
            f"({config.array_rows}x{config.array_cols} checkerboard)",
            f"  Systolic array: {ct.array_rows}x{ct.array_cols} INT8 MACs "
            f"({config.peak_tops_int8:.2f} TOPS peak)",
            f"  L2 SRAM: {ct.l2_size_bytes // 1024}KB ({ct.l2_num_banks} banks)  |  "
            f"L1 skew: {ct.l1_size_bytes // 1024}KB ({ct.l1_num_banks} banks)",
            f"  Memory tiles: {config.num_memory_tiles} "
            f"(L3: {mt.l3_tile_size_bytes // 1024}KB each)",
            f"  NoC: {config.noc.topology}, {config.noc.link_width_bits}-bit links, "
            f"{config.noc.frequency_mhz:.0f}MHz",
            f"  DRAM: {config.dram.technology}, {config.dram.num_controllers} controllers, "
            f"{config.dram.num_controllers * config.dram.channels_per_controller} channels",
        ]
        if overrides:
            summary_lines.append(f"  Architect overrides applied: {len(overrides)}")

        return config.model_dump(), "\n".join(summary_lines)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("KPU preview failed: %s", e)
        return None, ""


def build_review_snapshot(
    state: SoCDesignState,
    available_agents: list[str],
) -> PlanReviewSnapshot:
    """Build a complete review snapshot from the current state."""
    graph = get_task_graph(state)
    constraints = get_constraints(state)
    rtl_enabled = state.get("rtl_enabled", False)

    # Issue #29: KPU preview block when rtl_enabled
    kpu_preview: dict[str, Any] | None = None
    kpu_preview_summary = ""
    if rtl_enabled:
        kpu_preview, kpu_preview_summary = _build_kpu_preview(state)

    return PlanReviewSnapshot(
        goal=state.get("goal", ""),
        use_case=state.get("use_case", ""),
        platform=state.get("platform", ""),
        constraints=constraints.model_dump(exclude_none=True),
        constraints_summary=summarize_constraints(constraints),
        task_graph_ascii=render_task_graph_ascii(graph),
        task_table=render_task_table(graph),
        execution_order=graph.execution_order(),
        parallel_groups=compute_parallel_groups(graph),
        available_agents=sorted(available_agents),
        inferred_context={
            "num_tasks": len(graph.nodes),
            "num_parallel_groups": len(compute_parallel_groups(graph)),
            "max_parallelism": max((len(g) for g in compute_parallel_groups(graph)), default=0),
            "rtl_enabled": rtl_enabled,
            "session_id": state.get("session_id", ""),
        },
        kpu_preview=kpu_preview,
        kpu_preview_summary=kpu_preview_summary,
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_modified_graph(
    graph: TaskGraph,
    available_agents: list[str],
) -> list[str]:
    """Validate a modified task graph. Returns list of errors (empty = valid)."""
    errors: list[str] = []

    if not graph.nodes:
        errors.append("Task graph is empty — at least one task is required")
        return errors

    # Check all agents are registered
    for task in graph.nodes.values():
        if task.agent not in available_agents:
            errors.append(
                f"Task '{task.id}' uses unknown agent '{task.agent}'. "
                f"Available: {', '.join(available_agents)}"
            )

    # Check for cycles (should be caught by TaskGraph, but belt-and-suspenders)
    if graph._has_cycle():
        errors.append("Task graph contains a cycle")

    # Check for dangling dependencies
    for task in graph.nodes.values():
        for dep_id in task.dependencies:
            if dep_id not in graph.nodes:
                errors.append(f"Task '{task.id}' depends on '{dep_id}' which does not exist")

    return errors


# ---------------------------------------------------------------------------
# Edit application
# ---------------------------------------------------------------------------


def apply_review_edits(
    state: SoCDesignState,
    review_input: PlanReviewInput,
    available_agents: list[str],
) -> dict[str, Any]:
    """Apply human edits to the task graph and constraints.

    Returns a state update dict for LangGraph merging.

    Raises:
        ValueError: If the edited graph is invalid.
    """
    graph = get_task_graph(state)

    # 1. Remove tasks (in reverse dependency order to avoid dangling refs)
    for task_id in review_input.tasks_to_remove:
        if task_id in graph.nodes:
            # Remove from other tasks' dependencies first
            for other in graph.nodes.values():
                if task_id in other.dependencies:
                    other.dependencies = [d for d in other.dependencies if d != task_id]
            graph.nodes.pop(task_id)

    # 2. Reassign agents
    for task_id, new_agent in review_input.agent_reassignments.items():
        if task_id in graph.nodes:
            graph.nodes[task_id].agent = new_agent

    # 3. Override dependencies (reorder)
    for task_id, new_deps in review_input.dependency_overrides.items():
        if task_id in graph.nodes:
            graph.nodes[task_id].dependencies = new_deps

    # 4. Add new tasks
    for task_dict in review_input.tasks_to_add:
        node = TaskNode(
            id=task_dict["id"],
            name=task_dict["name"],
            agent=task_dict["agent"],
            dependencies=task_dict.get("dependencies", []),
            preconditions=task_dict.get("preconditions", []),
            postconditions=task_dict.get("postconditions", []),
        )
        graph.add_task(node)

    # 5. Validate
    errors = validate_modified_graph(graph, available_agents)
    if errors:
        raise ValueError(
            "Invalid task graph after edits:\n" + "\n".join(f"  - {e}" for e in errors)
        )

    # 6. Apply constraint overrides
    updates: dict[str, Any] = {}
    if review_input.constraint_overrides:
        constraints = get_constraints(state)
        current = constraints.model_dump()
        current.update(review_input.constraint_overrides)
        updates["constraints"] = current

    # 6b. Apply KPU overrides (issue #29) — merged into kpu_config_overrides
    # so kpu_configurator picks them up when it runs during dispatch.
    if review_input.kpu_overrides:
        existing = dict(state.get("kpu_config_overrides", {}) or {})
        existing.update(review_input.kpu_overrides)
        updates["kpu_config_overrides"] = existing

    # 7. Record in history
    state_with_graph = set_task_graph(state, graph)
    if review_input.notes:
        state_with_graph = record_decision(
            state_with_graph,
            agent="human_architect",
            action="Plan review notes",
            rationale=review_input.notes,
        )

    state_with_graph = record_decision(
        state_with_graph,
        agent="human_architect",
        action=f"Plan review: {review_input.decision.value}",
        rationale=_summarize_edits(review_input),
        data={
            "tasks_added": [t.get("id") for t in review_input.tasks_to_add],
            "tasks_removed": review_input.tasks_to_remove,
            "agents_reassigned": review_input.agent_reassignments,
            "dependencies_overridden": list(review_input.dependency_overrides.keys()),
            "constraints_overridden": list(review_input.constraint_overrides.keys()),
            "kpu_overridden": list(review_input.kpu_overrides.keys()),
        },
    )

    updates["task_graph"] = state_with_graph["task_graph"]
    updates["history"] = state_with_graph["history"]
    updates["design_rationale"] = state_with_graph["design_rationale"]

    return updates


def _summarize_edits(review_input: PlanReviewInput) -> str:
    """Summarize edits for the design history."""
    parts = []
    if review_input.tasks_to_add:
        parts.append(f"added {len(review_input.tasks_to_add)} task(s)")
    if review_input.tasks_to_remove:
        parts.append(f"removed {len(review_input.tasks_to_remove)} task(s)")
    if review_input.agent_reassignments:
        parts.append(f"reassigned {len(review_input.agent_reassignments)} agent(s)")
    if review_input.dependency_overrides:
        parts.append(f"reordered {len(review_input.dependency_overrides)} dependency chain(s)")
    if review_input.constraint_overrides:
        parts.append(f"overrode {len(review_input.constraint_overrides)} constraint(s)")
    if review_input.kpu_overrides:
        parts.append(f"overrode {len(review_input.kpu_overrides)} KPU parameter(s)")
    if not parts:
        return "Approved without changes"
    return "Modified plan: " + ", ".join(parts)


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------


def _make_plan_review_node(
    available_agents: list[str],
) -> Any:
    """Create the plan_review LangGraph node.

    This node:
    1. Builds the PlanReviewSnapshot for display
    2. If review_input is present in state, applies edits
    3. Otherwise, stores snapshot and waits for interrupt/resume
    """

    def plan_review_node(state: SoCDesignState) -> dict[str, Any]:
        logger.info("Plan review node entered")

        review_input_raw = state.get("review_input")

        if review_input_raw:
            # Resume path: human has provided input
            review_input = PlanReviewInput(**review_input_raw)

            if review_input.decision == ReviewDecision.REJECT:
                logger.info("Plan rejected by human architect")
                return {
                    "status": "failed",
                    "review_input": {},
                    "review_snapshot": {},
                }

            if review_input.decision == ReviewDecision.MODIFY:
                logger.info("Applying human modifications to plan")
                updates = apply_review_edits(state, review_input, available_agents)
                updates["review_input"] = {}
                # Rebuild snapshot after edits
                updated_state = {**state, **updates}
                snapshot = build_review_snapshot(updated_state, available_agents)
                updates["review_snapshot"] = snapshot.model_dump()
                return updates

            # APPROVE or SKIP
            logger.info("Plan approved by human architect")
            return {
                "review_input": {},
            }

        else:
            # First entry: build snapshot for display
            snapshot = build_review_snapshot(state, available_agents)
            logger.info(
                "Plan review snapshot ready: %d tasks, %d parallel groups",
                snapshot.inferred_context["num_tasks"],
                snapshot.inferred_context["num_parallel_groups"],
            )
            return {
                "review_snapshot": snapshot.model_dump(),
                "status": "reviewing",
            }

    return plan_review_node


# ---------------------------------------------------------------------------
# Rich console rendering (lazy import)
# ---------------------------------------------------------------------------


def render_plan_review_rich(snapshot: PlanReviewSnapshot) -> str:
    """Render the plan review snapshot as rich formatted text.

    Returns a string suitable for console display. Uses box-drawing
    characters but does not require the Rich library.
    """
    lines: list[str] = []

    # Header
    lines.append("=" * 72)
    lines.append("  PLAN REVIEW — SoC Design Task Graph")
    lines.append("=" * 72)
    lines.append("")

    # Goal & context
    lines.append(f"  Goal:        {snapshot.goal}")
    if snapshot.use_case:
        lines.append(f"  Use Case:    {snapshot.use_case}")
    if snapshot.platform:
        lines.append(f"  Platform:    {snapshot.platform}")
    lines.append(f"  Constraints: {snapshot.constraints_summary}")
    lines.append("")

    # Task graph visualization
    lines.append("─" * 72)
    lines.append("  TASK GRAPH (dependency tree)")
    lines.append("─" * 72)
    lines.append("")
    for graph_line in snapshot.task_graph_ascii.split("\n"):
        lines.append(f"  {graph_line}")
    lines.append("")

    # Parallel execution groups
    lines.append("─" * 72)
    lines.append("  EXECUTION SCHEDULE (parallel groups)")
    lines.append("─" * 72)
    lines.append("")
    for i, group in enumerate(snapshot.parallel_groups):
        lines.append(f"  Step {i + 1}: {', '.join(group)}")
    lines.append("")

    # Task detail table
    lines.append("─" * 72)
    lines.append("  TASK DETAILS")
    lines.append("─" * 72)
    lines.append("")
    for task in snapshot.task_table:
        lines.append(f"  [{task['id']}] {task['name']}")
        lines.append(f"         Agent: {task['agent']}  |  Deps: {task['dependencies']}")
        if task["preconditions"] != "-":
            lines.append(f"         Pre:  {task['preconditions']}")
        if task["postconditions"] != "-":
            lines.append(f"         Post: {task['postconditions']}")
        lines.append("")

    # KPU micro-architecture preview (issue #29) — only when rtl_enabled
    if snapshot.kpu_preview and snapshot.kpu_preview_summary:
        lines.append("─" * 72)
        use_case_tag = f" for {snapshot.use_case}" if snapshot.use_case else ""
        lines.append(f"  KPU MICRO-ARCHITECTURE (initial sizing{use_case_tag})")
        lines.append("─" * 72)
        lines.append(snapshot.kpu_preview_summary)
        lines.append("")
        lines.append("  To override at plan review, set kpu_overrides in PlanReviewInput, e.g.:")
        lines.append('    {"compute_tile.array_rows": 8, "noc.link_width_bits": 512}')
        lines.append("")

    # Available agents
    lines.append("─" * 72)
    lines.append("  AVAILABLE AGENTS")
    lines.append("─" * 72)
    lines.append(f"  {', '.join(snapshot.available_agents)}")
    lines.append("")

    # Summary
    ctx = snapshot.inferred_context
    lines.append("─" * 72)
    lines.append(
        f"  {ctx['num_tasks']} tasks | "
        f"{ctx['num_parallel_groups']} execution steps | "
        f"max parallelism: {ctx['max_parallelism']}"
    )
    if ctx.get("rtl_enabled"):
        lines.append("  RTL generation: ENABLED")
    lines.append("=" * 72)

    return "\n".join(lines)
