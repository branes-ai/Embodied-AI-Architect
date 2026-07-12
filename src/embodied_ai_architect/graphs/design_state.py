"""Unified DesignState schema — DRAFT sketch for the Loop Convergence milestone.

See `docs/plans/roadmap-loop-convergence.md`.

This module is a **proposal**, not yet wired into any graph. It sketches the single
state schema that subsumes the two existing, disjoint schemas:

    graphs/soc_state.py:SoCDesignState        (single-design dispatcher loop)
    graphs/optimization_loop.py:OptimizationLoopState  (population / Pareto MOO loop)

...plus the two new structured types that become the *shared currency* between a
reasoning Critic and a reasoning Optimizer, replacing today's coarse free-text
"iterate" feedback:

    DesignIssue  — a typed bottleneck record (critic/specialists -> backlog)
    DesignDelta  — a concrete, applyable edit (critic -> optimizer)

Field convention follows the existing code: the TypedDict stores plain serialized
dicts/lists (not Pydantic instances) so LangGraph can checkpoint it, while the
Pydantic models below define/validate those payloads before they are dumped in.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from typing_extensions import TypedDict
import uuid

from pydantic import BaseModel, Field

# Reuse the existing, already-shipped models so the subsumption is real, not a copy.
# These are re-exported (see __all__) so callers can import them from this module.
from embodied_ai_architect.graphs.soc_state import (
    DesignConstraints,
    DesignDecision,
    DesignStatus,
    PPAMetrics,
)

__all__ = [
    # Re-exported from soc_state
    "DesignConstraints",
    "DesignDecision",
    "DesignStatus",
    "PPAMetrics",
    # Enums
    "MetricAxis",
    "AbstractionLevel",
    "Severity",
    "IssueStatus",
    "DeltaKind",
    # Models
    "DesignIssue",
    "DesignDelta",
    "DesignState",
    # Lifecycle helpers
    "create_initial_design_state",
    "add_issue",
    "add_delta",
    "resolve_issue",
    "open_issues",
    "has_converged",
    # Channel audit (Seam S1)
    "declared_channels",
    "undeclared_keys",
    "assert_declared_channels",
]


# ---------------------------------------------------------------------------
# Shared enums — vocabularies the critic, specialists, and optimizer agree on
# ---------------------------------------------------------------------------


class MetricAxis(str, Enum):
    """The metric a DesignIssue is about. Mirrors the architect-workflows.md axes."""

    POWER = "power"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    AREA = "area"
    COST = "cost"
    THERMAL = "thermal"
    ACCURACY = "accuracy"
    CAPABILITY_PER_WATT = "capability_per_watt"
    UTILIZATION = "utilization"
    BANDWIDTH = "bandwidth"
    MEMORY = "memory"
    WEIGHT = "weight"
    VOLUME = "volume"
    RELIABILITY = "reliability"


class AbstractionLevel(str, Enum):
    """Where in the composition hierarchy the issue lives (architect-workflows.md)."""

    SYSTEM = "system"
    SUBSYSTEM = "subsystem"
    OPERATOR = "operator"
    KERNEL = "kernel"
    HARDWARE = "hardware"
    PHYSICAL = "physical"


class Severity(str, Enum):
    CRITICAL = "critical"  # constraint violated
    HIGH = "high"  # dominant bottleneck, little headroom
    MEDIUM = "medium"
    LOW = "low"


class IssueStatus(str, Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"  # a delta targeting it has been proposed
    RESOLVED = "resolved"
    WONTFIX = "wontfix"  # accepted trade-off


class DeltaKind(str, Enum):
    """The concrete action an Optimizer agent can apply to close issues."""

    DESIGN_SPACE_EDIT = "design_space_edit"  # change a variable's value/category
    VARIABLE_BOUND_CHANGE = "variable_bound_change"  # widen/narrow search bounds
    ADD_VARIABLE = "add_variable"  # bring a new degree of freedom into the space
    REMOVE_VARIABLE = "remove_variable"  # freeze a variable
    CONSTRAINT_RELAXATION = "constraint_relaxation"  # negotiate a constraint
    SPECIALIST_RETASK = "specialist_retask"  # re-run/redirect a specialist agent


# ---------------------------------------------------------------------------
# DesignIssue — the shared backlog currency
# ---------------------------------------------------------------------------


class DesignIssue(BaseModel):
    """A structured bottleneck record.

    Replaces the free-text strings in `PPAMetrics.bottlenecks`. Emitted by the
    critic and by specialist agents; consumed by the optimizer and the
    architect-drill / architect-assess skills.
    """

    id: str = Field(default_factory=lambda: f"issue-{uuid.uuid4().hex[:8]}")
    metric: MetricAxis = Field(..., description="Which metric this issue is about")
    level: AbstractionLevel = Field(..., description="Composition level of the bottleneck")
    component: Optional[str] = Field(
        default=None, description="Operator/block/kernel name, e.g. 'yolo_detector' or 'FFT'"
    )
    severity: Severity = Severity.MEDIUM

    summary: str = Field(..., description="One-line statement of the bottleneck")
    observed_value: Optional[float] = None
    target_value: Optional[float] = Field(
        default=None, description="Constraint or budget this metric must meet"
    )
    contribution_pct: Optional[float] = Field(
        default=None, description="Share of the system total this component drives (0-100)"
    )

    status: IssueStatus = IssueStatus.OPEN
    raised_by: str = Field(..., description="Agent that raised it, e.g. 'critic'")
    resolved_by: Optional[str] = None
    iteration_raised: int = 0
    iteration_resolved: Optional[int] = None

    # Links to the deltas proposed to close this issue.
    delta_ids: list[str] = Field(default_factory=list)

    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_open(self) -> bool:
        return self.status in (IssueStatus.OPEN, IssueStatus.IN_PROGRESS)


# ---------------------------------------------------------------------------
# DesignDelta — the concrete, applyable edit (critic -> optimizer)
# ---------------------------------------------------------------------------


class DesignDelta(BaseModel):
    """A single, applyable change to the design or the search.

    This is what the Critic emits *instead of* a free-text "iterate" verdict, and
    what the Optimizer agent consumes to make a concrete edit before re-running the
    MOO engine (now a tool, not the loop body).
    """

    id: str = Field(default_factory=lambda: f"delta-{uuid.uuid4().hex[:8]}")
    kind: DeltaKind

    target: str = Field(
        ...,
        description="What is changed: a design-space variable name, a constraint field, "
        "or a specialist/task id (dotted paths allowed, e.g. 'hardware.array_rows').",
    )
    # Free-form because the payload shape depends on `kind`:
    #   DESIGN_SPACE_EDIT      -> {"value": <new value>}
    #   VARIABLE_BOUND_CHANGE  -> {"bounds": [lo, hi]} or {"categories": [...]}
    #   ADD_VARIABLE           -> {"variable": <DesignVariable spec>}
    #   CONSTRAINT_RELAXATION  -> {"from": <old>, "to": <new>}
    #   SPECIALIST_RETASK      -> {"specialist": "bandwidth_validator", "reason": "..."}
    change: dict[str, Any] = Field(default_factory=dict)

    rationale: str = Field(..., description="Why this edit should help")
    addresses_issue_ids: list[str] = Field(
        default_factory=list, description="Issues this delta is meant to close"
    )
    proposed_by: str = Field(default="critic")

    applied: bool = False
    applied_at_iteration: Optional[int] = None


# ---------------------------------------------------------------------------
# DesignState — the single TypedDict both loops flow through
# ---------------------------------------------------------------------------


class DesignState(TypedDict, total=False):
    """Unified state schema. Superset of SoCDesignState + OptimizationLoopState.

    total=False: every field is optional; nodes populate what they own. Values are
    plain serialized dicts/lists (checkpoint-friendly), validated by the Pydantic
    models above before insertion.
    """

    # --- Identity / lifecycle (SoCDesignState) ---
    session_id: str
    goal: str
    status: str  # DesignStatus value
    created_at: str
    updated_at: str

    # --- Mission input & decomposition (OptimizationLoopState) ---
    mission_description: str
    mission_plan: dict
    platform: str
    mission_type: str
    sub_capabilities: list[dict]
    research_docs_used: list[str]
    research_citations: list[str]

    # --- Requirements & context (SoCDesignState) ---
    constraints: dict  # DesignConstraints serialized
    platform_context: dict
    task_graph: dict  # TaskGraph serialized

    # --- Design point & joint design space (merged) ---
    design_space_config: dict  # joint_design_space serialized
    num_variables: int
    current_design_point: dict  # the single point under evaluation (dispatcher loop)

    # --- Architecture artifacts (SoCDesignState) ---
    workload_profile: dict
    codebase_metadata: dict
    hardware_candidates: list[dict]
    selected_architecture: dict
    ip_blocks: list[dict]
    memory_map: dict
    interconnect: dict
    rtl_modules: dict  # name -> Verilog

    # --- KPU / RTL inner loop (SoCDesignState) ---
    kpu_config: dict
    kpu_config_overrides: dict
    floorplan_estimate: dict
    bandwidth_match: dict
    kpu_optimization_history: list[dict]
    rtl_synthesis_results: dict
    rtl_optimization_history: list[dict]

    # --- PPA & multi-objective results (merged; both loops write here) ---
    ppa_metrics: dict  # PPAMetrics serialized
    baseline_metrics: dict
    pareto_points: list[dict]  # accumulated non-dominated points
    pareto_frontier_history: list[dict]
    moo_results: dict
    hypervolume_history: list[float]
    knee_point: dict
    sensitivity: dict
    atlas: dict
    convergence_history: list[dict]

    # --- MOO search-loop fields (migrated from OptimizationLoopState, S2a #205) ---
    # NOTE: `pareto_front` is the MOO loop's frontier list; canonicalizing it onto
    # the dispatcher-loop `pareto_points` name is deferred to S2c (#207).
    pareto_front: list[dict]
    hypervolume: float  # scalar for the latest run (hypervolume_history holds the series)
    total_evaluations: int
    layers_used: list[str]
    atlas_coverage_pct: float
    design_variables_ranked: list[dict]
    refinements: dict  # search-space refinements the reason node proposes on iterate

    # --- NEW: shared multi-agent currency (the Loop Convergence delta) ---
    open_issues: list[dict]  # DesignIssue serialized — the bottleneck backlog
    pending_deltas: list[dict]  # DesignDelta serialized — proposed, not yet applied
    applied_deltas: list[dict]  # DesignDelta serialized — history of edits
    pending_specialist_tasks: list[dict]  # SPECIALIST_RETASK deltas the dispatcher picks up

    # --- Reasoning output (OptimizationLoopState) ---
    # (research_citations is declared once, above, in the decomposition section)
    analysis: str
    recommendation: dict
    final_report: str

    # --- Loop control (merged) ---
    iteration: int
    max_iterations: int
    converged: bool
    should_iterate: bool
    llm_available: bool  # gates the reasoning agents' LLM path (read by critic_node)

    # --- History / governance / cost (SoCDesignState) ---
    history: list[dict]  # DesignDecision entries
    design_rationale: list[str]
    working_memory: dict
    optimization_history: list[dict]
    governance: dict
    audit_log: list[dict]
    cost_tracking: dict
    evaluation_scorecard: dict

    # --- Human-in-the-loop review/steering (SoCDesignState) ---
    review_snapshot: dict
    optimization_review_snapshot: dict
    optimization_steering: dict

    # --- Error tracking (OptimizationLoopState) ---
    errors: list[str]


# ---------------------------------------------------------------------------
# Channel audit (Seam S1) — the LangGraph allowlist guard
# ---------------------------------------------------------------------------
#
# A LangGraph StateGraph merges each node's return dict into the state ONLY for
# keys declared on the state schema. Any key a node returns that is not a
# declared DesignState channel is silently dropped at runtime — a bug class with
# no error and no stack trace (hit during the skeleton work with `final_report`,
# `research_citations`, and `pending_specialist_tasks`). These helpers make the
# invariant checkable so migrated nodes (S2a/S2b) can self-guard and tests can
# enforce it.

_DECLARED_CHANNELS: frozenset[str] = frozenset(DesignState.__annotations__)


def declared_channels() -> frozenset[str]:
    """Return the set of declared DesignState channels (the LangGraph allowlist)."""
    return _DECLARED_CHANNELS


def undeclared_keys(update: dict) -> set[str]:
    """Return the keys in a node's return dict that are NOT declared channels.

    Empty set == safe to merge. Non-empty == those keys would be silently dropped.
    """
    return set(update) - _DECLARED_CHANNELS


def assert_declared_channels(update: dict, *, node: str = "") -> None:
    """Raise if `update` contains any key that is not a declared DesignState channel.

    Intended for node self-guarding: `assert_declared_channels(result, node="critic")`.
    """
    extra = undeclared_keys(update)
    if extra:
        where = f" from node '{node}'" if node else ""
        raise KeyError(
            f"DesignState update{where} contains keys not declared as channels: "
            f"{sorted(extra)}. Declare them on DesignState or LangGraph will drop them."
        )


# ---------------------------------------------------------------------------
# Minimal helpers (sketch — the real graph nodes will use these)
# ---------------------------------------------------------------------------


def create_initial_design_state(
    goal: str,
    *,
    constraints: Optional[DesignConstraints] = None,
    mission_description: Optional[str] = None,
    session_id: Optional[str] = None,
) -> DesignState:
    """Seed a fresh unified state. Convert-and-adapt entry point for both loops."""
    now = datetime.now().isoformat()
    state: DesignState = {
        "session_id": session_id or f"design-{uuid.uuid4().hex[:8]}",
        "goal": goal,
        "status": DesignStatus.PLANNING.value,
        "created_at": now,
        "updated_at": now,
        "constraints": (constraints or DesignConstraints()).model_dump(),
        "open_issues": [],
        "pending_deltas": [],
        "applied_deltas": [],
        "iteration": 0,
        "converged": False,
        "errors": [],
    }
    if mission_description:
        state["mission_description"] = mission_description
    return state


def add_issue(state: DesignState, issue: DesignIssue) -> DesignState:
    """Append a validated issue to the backlog."""
    state.setdefault("open_issues", []).append(issue.model_dump(mode="json"))
    return state


def add_delta(state: DesignState, delta: DesignDelta) -> DesignState:
    """Record a proposed edit and mark the issues it targets as in-progress."""
    state.setdefault("pending_deltas", []).append(delta.model_dump(mode="json"))
    targeted = set(delta.addresses_issue_ids)
    for raw in state.get("open_issues", []):
        if raw.get("id") in targeted and raw.get("status") == IssueStatus.OPEN.value:
            raw["status"] = IssueStatus.IN_PROGRESS.value
            raw.setdefault("delta_ids", []).append(delta.id)
    return state


def resolve_issue(state: DesignState, issue_id: str, *, by: str, iteration: int) -> DesignState:
    """Mark an issue resolved once its delta lands and re-evaluation confirms it."""
    for raw in state.get("open_issues", []):
        if raw.get("id") == issue_id:
            raw["status"] = IssueStatus.RESOLVED.value
            raw["resolved_by"] = by
            raw["iteration_resolved"] = iteration
    return state


def open_issues(state: DesignState) -> list[DesignIssue]:
    """Rehydrate the still-open issues, most severe first."""
    order = {
        s: i
        for i, s in enumerate([Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW])
    }
    issues = [DesignIssue(**raw) for raw in state.get("open_issues", [])]
    return sorted((i for i in issues if i.is_open), key=lambda i: order[i.severity])


def has_converged(state: DesignState, *, hypervolume_epsilon: float = 1e-3) -> bool:
    """Unified stop condition (replaces the two separate ones).

    Converged when EITHER the issue backlog is empty OR the Pareto hypervolume has
    stopped improving — whichever the loop reaches first.
    """
    if not open_issues(state):
        return True
    hv = state.get("hypervolume_history", [])
    if len(hv) >= 2 and abs(hv[-1] - hv[-2]) < hypervolume_epsilon:
        return True
    return False
