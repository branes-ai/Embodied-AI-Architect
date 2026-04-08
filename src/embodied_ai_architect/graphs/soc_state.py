"""State schema for the agentic SoC designer.

Defines SoCDesignState — the single TypedDict that flows through the LangGraph
design pipeline. Every specialist agent reads from and writes to this state.

This is the "single most important design decision" (per the implementation plan):
everything flows from the shape of this state.

Usage:
    state = create_initial_soc_state(
        goal="Design an SoC for a delivery drone",
        constraints=DesignConstraints(
            max_power_watts=5.0,
            max_latency_ms=33.3,
            max_cost_usd=30.0,
        ),
        use_case="delivery_drone",
    )
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from typing_extensions import TypedDict
import uuid

from pydantic import BaseModel, Field

from embodied_ai_architect.graphs.task_graph import TaskGraph

# ---------------------------------------------------------------------------
# Supporting types
# ---------------------------------------------------------------------------


class DesignStatus(str, Enum):
    """High-level status of the design session."""

    PLANNING = "planning"  # Planner is decomposing the goal
    EXPLORING = "exploring"  # Agents are exploring the design space
    OPTIMIZING = "optimizing"  # Iterative PPA optimization loop
    REVIEWING = "reviewing"  # Critic / human review
    COMPLETE = "complete"  # Design finished
    FAILED = "failed"  # Unrecoverable failure


class DesignConstraints(BaseModel):
    """Quantified design targets for SoC architecture.

    All fields are optional — the planner infers missing constraints
    from the use case and platform implications (embodied-schemas).
    """

    # Power
    max_power_watts: Optional[float] = Field(
        default=None, description="Total compute power budget in watts"
    )
    power_class: Optional[str] = Field(
        default=None,
        description="Power class name from embodied-schemas (e.g. 'low_power')",
    )

    # Latency
    max_latency_ms: Optional[float] = Field(
        default=None, description="End-to-end perception latency target in ms"
    )
    min_fps: Optional[float] = Field(default=None, description="Minimum frames per second")
    latency_tier: Optional[str] = Field(
        default=None,
        description="Latency tier from embodied-schemas (e.g. 'real_time')",
    )

    # Area / cost
    max_area_mm2: Optional[float] = Field(default=None, description="Maximum die area in mm^2")
    max_cost_usd: Optional[float] = Field(
        default=None, description="Maximum BOM cost at target volume"
    )
    target_volume: Optional[int] = Field(
        default=None, description="Production volume for cost estimation"
    )

    # Memory
    max_memory_mb: Optional[float] = Field(
        default=None, description="Maximum on-chip + off-chip memory in MB"
    )
    memory_class: Optional[str] = Field(
        default=None,
        description="Memory class from embodied-schemas (e.g. 'small')",
    )

    # Process technology
    target_process_nm: Optional[int] = Field(
        default=None, description="Target process node in nm (e.g. 28, 16, 7)"
    )

    # Environmental
    operating_temp_min_c: Optional[float] = Field(
        default=None, description="Minimum operating temperature in Celsius"
    )
    operating_temp_max_c: Optional[float] = Field(
        default=None, description="Maximum operating temperature in Celsius"
    )
    ip_rating: Optional[str] = Field(
        default=None, description="Ingress protection rating (e.g. 'IP67')"
    )

    # Safety
    safety_critical: bool = Field(
        default=False, description="Whether the application is safety-critical"
    )
    safety_standard: Optional[str] = Field(
        default=None, description="Safety standard (e.g. 'IEC 62304 Class C')"
    )

    # Accuracy / capability
    min_accuracy_percent: Optional[float] = Field(
        default=None,
        description="Minimum acceptable accuracy (mAP, top-1, mIoU) on 0-100 scale",
    )
    workload_type: Optional[str] = Field(
        default=None,
        description="Workload type for accuracy lookup (detection, classification, segmentation)",
    )
    quantization_dtype: Optional[str] = Field(
        default=None,
        description="Quantization data type (fp32, fp16, int8, int4, mixed_int8_fp16)",
    )

    # SWaP-C (system-level)
    max_weight_grams: Optional[float] = Field(
        default=None, description="Maximum system weight in grams"
    )
    max_volume_cm3: Optional[float] = Field(
        default=None, description="Maximum system volume in cm³"
    )

    # Custom
    custom: dict[str, Any] = Field(
        default_factory=dict,
        description="Domain-specific constraints not covered above",
    )


class PPAMetrics(BaseModel):
    """Power, Performance, Area measurements for a design point."""

    power_watts: Optional[float] = None
    latency_ms: Optional[float] = None
    throughput_fps: Optional[float] = None
    area_mm2: Optional[float] = None
    process_nm: Optional[int] = Field(
        default=None,
        description="Process technology node in nm (e.g. 28, 7, 5)",
    )
    cost_usd: Optional[float] = None
    cost_breakdown: Optional[dict[str, float]] = Field(
        default=None,
        description="Manufacturing cost breakdown (die, package, test, NRE)",
    )
    memory_mb: Optional[float] = None
    energy_per_inference_mj: Optional[float] = None

    # SWaP-C system-level metrics
    weight_grams: Optional[float] = Field(default=None, description="Total system weight in grams")
    volume_cm3: Optional[float] = Field(default=None, description="Total system volume in cm³")
    thermal_margin_c: Optional[float] = Field(
        default=None, description="Thermal margin (max_junction - actual junction) in C"
    )

    # Capability metrics (Phase 1: accuracy loop)
    accuracy_percent: Optional[float] = Field(
        default=None,
        description="Task-specific accuracy (mAP, top-1, mIoU) on 0-100 scale",
    )
    capability_score: Optional[float] = Field(
        default=None,
        description="Normalized mission success rate [0, 1]",
    )
    capability_per_watt: Optional[float] = Field(
        default=None,
        description="capability_score / power_watts — intelligence per watt",
    )
    gops_per_watt: Optional[float] = Field(
        default=None,
        description="peak_gops / power_watts — hardware efficiency",
    )
    model_family: Optional[str] = Field(
        default=None,
        description="Best-fit model family for the workload (e.g. 'yolov8')",
    )
    model_variant: Optional[str] = Field(
        default=None,
        description="Best-fit model variant (e.g. 's')",
    )

    # Per-constraint verdicts (PASS/FAIL/PARTIAL)
    verdicts: dict[str, str] = Field(
        default_factory=dict,
        description="Constraint name -> verdict (PASS/FAIL/PARTIAL)",
    )
    bottlenecks: list[str] = Field(
        default_factory=list,
        description="Identified performance bottlenecks",
    )
    suggestions: list[str] = Field(
        default_factory=list,
        description="Improvement suggestions from PPA assessor",
    )


class DesignDecision(BaseModel):
    """A recorded design decision with rationale."""

    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    agent: str = Field(..., description="Agent that made the decision")
    action: str = Field(..., description="What was decided")
    rationale: str = Field(..., description="Why this decision was made")
    alternatives_considered: list[str] = Field(
        default_factory=list,
        description="Other options that were evaluated",
    )
    data: dict[str, Any] = Field(
        default_factory=dict,
        description="Supporting data for the decision",
    )


# ---------------------------------------------------------------------------
# Main state TypedDict
# ---------------------------------------------------------------------------


class SoCDesignState(TypedDict, total=False):
    """State flowing through the LangGraph SoC design pipeline.

    This TypedDict defines all fields that specialist agents read/write.
    Using total=False allows optional fields while maintaining type safety
    with LangGraph's state merging.

    Organized into sections:
    - Goal: what the user asked for
    - Task Graph: the execution plan (DAG)
    - Design Artifacts: intermediate and final design outputs
    - Evaluation: PPA metrics and Pareto analysis
    - History & Memory: decisions, rationale, iteration tracking
    - Control: routing signals and session metadata
    """

    # === Goal ===
    goal: str  # Natural language design objective
    constraints: dict  # DesignConstraints serialized (TypedDict requires plain types)
    use_case: str  # "delivery_drone", "warehouse_amr", "surgical_robot", etc.
    platform: str  # "drone", "quadruped", "biped", "amr", "edge"

    # === Task Graph ===
    task_graph: dict  # TaskGraph serialized via to_dict()
    current_task_id: str  # Task being executed now

    # === Design Artifacts ===
    workload_profile: dict  # Operator graph, compute/memory requirements
    codebase_metadata: dict  # Source mapping when workload came from codebase analysis
    hardware_candidates: list[dict]  # Scored hardware options
    selected_architecture: dict  # Chosen SoC composition
    ip_blocks: list[dict]  # CPU, NPU, ISP, memory controller configs
    memory_map: dict  # Address space layout
    interconnect: dict  # Bus/NoC topology
    rtl_modules: dict[str, str]  # Module name -> Verilog source

    # === KPU Micro-architecture ===
    kpu_config: dict  # KPUMicroArchConfig serialized
    floorplan_estimate: dict  # FloorplanEstimate serialized
    bandwidth_match: dict  # BandwidthMatchResult serialized
    kpu_optimization_history: list[dict]  # Config snapshots per KPU loop iteration

    # === RTL Artifacts ===
    rtl_testbenches: dict[str, str]  # Module name -> testbench source
    rtl_synthesis_results: dict[str, dict]  # Module name -> synthesis result
    rtl_lint_results: dict[str, dict]  # Module name -> lint result
    rtl_validation_results: dict[str, dict]  # Module name -> validation result
    rtl_optimization_history: list[dict]  # RTL optimization snapshots
    rtl_process_nm: int  # Target process node for RTL
    rtl_enabled: bool  # Enables KPU config + floorplan + bandwidth + RTL
    enable_moo: bool  # Enables moo_explorer task in default plan

    # === Evaluation ===
    ppa_metrics: dict  # Current PPAMetrics serialized
    baseline_metrics: dict  # Reference point for optimization
    pareto_points: list[dict]  # Accumulated non-dominated points across all MOO runs
    pareto_results: dict  # Pareto front analysis results
    pareto_frontier_history: list[dict]  # Per-iteration frontier evolution snapshots
    moo_results: dict  # Multi-objective optimization results (rich data)
    last_strategy_rationale: str  # Last design_optimizer's selection rationale (issue #25)
    swap_assessment: dict  # SWaP-C optimization results (6-objective)
    system_bom: dict  # Hierarchical system BOM data
    safety_analysis: dict  # Safety-critical detection results
    prior_experience: dict  # Experience retrieval results
    cost_tracking: dict  # CostTracker serialized state
    evaluation_scorecard: dict  # Scorecard from evaluation framework

    # === History & Memory ===
    iteration: int  # Optimization loop counter
    max_iterations: int  # Safety bound to prevent infinite loops
    history: list[dict]  # All DesignDecision entries serialized
    design_rationale: list[str]  # Human-readable rationale chain
    working_memory: dict  # WorkingMemoryStore serialized
    optimization_history: list[dict]  # PPA snapshot per iteration
    governance: dict  # GovernancePolicy serialized
    audit_log: list[dict]  # AuditEntry list

    # === Human Review ===
    review_snapshot: dict  # PlanReviewSnapshot serialized (for display)
    review_input: dict  # PlanReviewInput serialized (human edits)
    human_review_enabled: bool  # Toggle for review nodes
    optimization_review_snapshot: dict  # OptimizationReviewSnapshot for display
    optimization_steering: dict  # OptimizationSteeringInput from human

    # === Control ===
    next_action: str  # Graph routing signal (node name to execute next)
    status: str  # DesignStatus value
    session_id: str  # Unique session identifier
    created_at: str  # ISO timestamp


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def create_initial_soc_state(
    goal: str,
    constraints: Optional[DesignConstraints] = None,
    use_case: str = "",
    platform: str = "",
    max_iterations: int = 20,
    session_id: Optional[str] = None,
    governance: Optional[dict] = None,
    rtl_enabled: bool = False,
    enable_moo: bool = True,
) -> SoCDesignState:
    """Create initial state for a new SoC design session.

    Args:
        goal: Natural language design objective.
        constraints: Quantified design constraints. None = planner infers them.
        use_case: Application type (e.g. "delivery_drone").
        platform: Platform type for constraint implications (e.g. "drone").
        max_iterations: Safety bound for optimization loops.
        session_id: Unique identifier (auto-generated if None).
        governance: GovernancePolicy serialized dict. None = permissive defaults.
        rtl_enabled: Enable KPU config + floorplan + bandwidth + RTL pipeline.
        enable_moo: Schedule moo_explorer in default plan to populate pareto_points.

    Returns:
        Initial SoCDesignState ready for the Planner agent.
    """
    task_graph = TaskGraph()

    return SoCDesignState(
        # Goal
        goal=goal,
        constraints=constraints.model_dump() if constraints else {},
        use_case=use_case,
        platform=platform,
        # Task Graph
        task_graph=task_graph.to_dict(),
        current_task_id="",
        # Design Artifacts (empty initially)
        workload_profile={},
        codebase_metadata={},
        hardware_candidates=[],
        selected_architecture={},
        ip_blocks=[],
        memory_map={},
        interconnect={},
        rtl_modules={},
        # KPU Micro-architecture
        kpu_config={},
        floorplan_estimate={},
        bandwidth_match={},
        kpu_optimization_history=[],
        # RTL Artifacts
        rtl_testbenches={},
        rtl_synthesis_results={},
        rtl_lint_results={},
        rtl_validation_results={},
        rtl_optimization_history=[],
        rtl_process_nm=constraints.target_process_nm or 28 if constraints else 28,
        rtl_enabled=rtl_enabled,
        enable_moo=enable_moo,
        # Evaluation
        ppa_metrics={},
        baseline_metrics={},
        pareto_points=[],
        pareto_results={},
        pareto_frontier_history=[],
        moo_results={},
        last_strategy_rationale="",
        swap_assessment={},
        system_bom={},
        safety_analysis={},
        prior_experience={},
        cost_tracking={},
        evaluation_scorecard={},
        # History
        iteration=0,
        max_iterations=max_iterations,
        history=[],
        design_rationale=[],
        working_memory={},
        optimization_history=[],
        governance=governance or {},
        audit_log=[],
        # Human Review
        review_snapshot={},
        review_input={},
        human_review_enabled=False,
        optimization_review_snapshot={},
        optimization_steering={},
        # Control
        next_action="planner",
        status=DesignStatus.PLANNING.value,
        session_id=session_id or f"soc_{uuid.uuid4().hex[:8]}",
        created_at=datetime.now().isoformat(),
    )


def get_task_graph(state: SoCDesignState) -> TaskGraph:
    """Deserialize the TaskGraph from state."""
    return TaskGraph.from_dict(state.get("task_graph", {"nodes": {}}))


def set_task_graph(state: SoCDesignState, graph: TaskGraph) -> SoCDesignState:
    """Serialize the TaskGraph back into state.

    Returns a new dict with the updated task_graph (LangGraph state merge).
    """
    return {**state, "task_graph": graph.to_dict()}  # type: ignore[return-value]


def get_constraints(state: SoCDesignState) -> DesignConstraints:
    """Deserialize DesignConstraints from state."""
    return DesignConstraints(**state.get("constraints", {}))


def get_ppa_metrics(state: SoCDesignState) -> PPAMetrics:
    """Deserialize PPAMetrics from state."""
    return PPAMetrics(**state.get("ppa_metrics", {}))


def record_decision(
    state: SoCDesignState,
    agent: str,
    action: str,
    rationale: str,
    alternatives: Optional[list[str]] = None,
    data: Optional[dict[str, Any]] = None,
) -> SoCDesignState:
    """Record a design decision in the state history.

    Returns a new dict with the appended decision (LangGraph state merge).
    """
    decision = DesignDecision(
        agent=agent,
        action=action,
        rationale=rationale,
        alternatives_considered=alternatives or [],
        data=data or {},
    )
    history = list(state.get("history", []))
    history.append(decision.model_dump())
    rationale_chain = list(state.get("design_rationale", []))
    rationale_chain.append(f"[{agent}] {action}: {rationale}")
    return {  # type: ignore[return-value]
        **state,
        "history": history,
        "design_rationale": rationale_chain,
    }


def get_task_result(state: SoCDesignState, task_id: str) -> dict[str, Any]:
    """Get the result of a completed task from state.

    Convenience function for specialist agents that need outputs from
    upstream tasks (their dependencies).

    Args:
        state: Current SoC design state.
        task_id: ID of the completed task.

    Returns:
        The task's result dict.

    Raises:
        ValueError: If task doesn't exist or has no result.
    """
    graph = get_task_graph(state)
    return graph.get_result(task_id)


def get_dependency_results(state: SoCDesignState, task: Any) -> dict[str, dict[str, Any]]:
    """Get results from all dependencies of a task.

    Args:
        state: Current SoC design state.
        task: TaskNode with dependencies list.

    Returns:
        Dict mapping dependency task_id -> result dict.
    """
    graph = get_task_graph(state)
    results = {}
    for dep_id in task.dependencies:
        try:
            results[dep_id] = graph.get_result(dep_id)
        except (KeyError, ValueError):
            pass  # Dependency may be skipped or not have a result
    return results


def is_design_complete(state: SoCDesignState) -> bool:
    """Check if the design session has reached a terminal state."""
    return state.get("status", "") in (DesignStatus.COMPLETE.value, DesignStatus.FAILED.value)


def is_over_iteration_limit(state: SoCDesignState) -> bool:
    """Check if the optimization loop has exceeded its safety bound."""
    return state.get("iteration", 0) >= state.get("max_iterations", 20)


def get_working_memory(state: SoCDesignState) -> dict:
    """Get the working memory dict from state."""
    return state.get("working_memory", {})


def update_working_memory(state: SoCDesignState, memory: dict) -> SoCDesignState:
    """Return state update with new working memory.

    Args:
        state: Current SoC design state.
        memory: WorkingMemoryStore serialized via model_dump().

    Returns:
        New dict with updated working_memory (LangGraph state merge).
    """
    return {**state, "working_memory": memory}  # type: ignore[return-value]


def record_audit(
    state: SoCDesignState,
    agent: str,
    action: str,
    input_summary: str = "",
    output_summary: str = "",
    cost_tokens: int = 0,
    human_approved: bool = False,
) -> SoCDesignState:
    """Append an audit entry to the state audit log.

    Returns a new dict with the appended entry (LangGraph state merge).
    """
    entry = {
        "timestamp": datetime.now().isoformat(),
        "agent": agent,
        "action": action,
        "input_summary": input_summary,
        "output_summary": output_summary,
        "cost_tokens": cost_tokens,
        "human_approved": human_approved,
        "iteration": state.get("iteration", 0),
    }
    log = list(state.get("audit_log", []))
    log.append(entry)
    return {**state, "audit_log": log}  # type: ignore[return-value]


def get_optimization_history(state: SoCDesignState) -> list[dict]:
    """Get the optimization history list from state."""
    return list(state.get("optimization_history", []))


def get_kpu_config(state: SoCDesignState) -> dict:
    """Get the KPU micro-architecture config from state."""
    return state.get("kpu_config", {})


def get_floorplan(state: SoCDesignState) -> dict:
    """Get the floorplan estimate from state."""
    return state.get("floorplan_estimate", {})


def get_bandwidth(state: SoCDesignState) -> dict:
    """Get the bandwidth match result from state."""
    return state.get("bandwidth_match", {})


def get_rtl_summary(state: SoCDesignState) -> dict:
    """Get a summary of RTL generation results."""
    synth = state.get("rtl_synthesis_results", {})
    total_cells = sum(r.get("area_cells", 0) for r in synth.values() if r.get("success"))
    pass_count = sum(1 for r in synth.values() if r.get("success"))
    return {
        "modules_generated": len(state.get("rtl_modules", {})),
        "modules_passed": pass_count,
        "total_cells": total_cells,
        "rtl_enabled": state.get("rtl_enabled", False),
    }


def get_iteration_summary(state: SoCDesignState) -> str:
    """Format a human-readable summary of the current design iteration."""
    lines = []
    lines.append(f"Session: {state.get('session_id', 'unknown')}")
    lines.append(f"Status:  {state.get('status', 'unknown')}")
    lines.append(f"Goal:    {state.get('goal', 'none')}")
    lines.append(f"Iteration: {state.get('iteration', 0)}/{state.get('max_iterations', 20)}")

    # Task graph summary
    graph = get_task_graph(state)
    if graph.nodes:
        summary = graph.summary()
        task_str = ", ".join(f"{k}: {v}" for k, v in sorted(summary.items()))
        lines.append(f"Tasks:   {task_str}")

    # PPA snapshot
    ppa = state.get("ppa_metrics", {})
    if ppa:
        ppa_parts = []
        if ppa.get("power_watts") is not None:
            ppa_parts.append(f"Power: {ppa['power_watts']:.1f}W")
        if ppa.get("latency_ms") is not None:
            ppa_parts.append(f"Latency: {ppa['latency_ms']:.1f}ms")
        if ppa.get("area_mm2") is not None:
            ppa_parts.append(f"Area: {ppa['area_mm2']:.1f}mm²")
        if ppa_parts:
            lines.append(f"PPA:     {', '.join(ppa_parts)}")

    # Latest rationale
    rationale = state.get("design_rationale", [])
    if rationale:
        lines.append(f"Latest:  {rationale[-1]}")

    return "\n".join(lines)
