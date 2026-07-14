"""Critic and Optimizer agent interfaces — DRAFT sketch for the Loop Convergence milestone.

See `docs/plans/roadmap-loop-convergence.md` (Phase 2) and `graphs/design_state.py`.

This module is a **proposal**, not yet wired into any graph. It sketches the two
reasoning agents that replace today's shallow feedback:

    Critic     — reviews the current DesignState, files structured DesignIssues, and
                 emits concrete DesignDeltas (instead of a free-text "iterate" verdict).
                 Promotes `optimization_loop.py:reason_node`.

    Optimizer  — consumes DesignDeltas, applies them as concrete edits to the design
                 space / constraints / design point, then re-runs the MOO engine *as a
                 tool* (not as the loop body). Promotes `optimizer.py:design_optimizer`.

Both follow the established house pattern: an LLM path with a deterministic heuristic
fallback (mirrors `reason_node` → `_reason_with_llm` / `_reason_heuristic`), and both
expose LangGraph node wrappers `critic_node` / `optimizer_node` returning dict updates.

The MOO engine is injected as a *tool* callable (`MooTool`) so the optimizer depends on
a boundary, not on the engine internals — the key structural move of this milestone.
"""

from __future__ import annotations

from abc import ABC
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field

from embodied_ai_architect.graphs.design_state import (
    AbstractionLevel,
    DesignDelta,
    DesignIssue,
    DesignState,
    DeltaKind,
    IssueStatus,
    MetricAxis,
    Severity,
    add_delta,
    add_issue,
    has_converged,
)

# A MooTool takes the current state and returns the state fields the MOO run produces
# (pareto_points, pareto_frontier_history, moo_results, hypervolume_history, sensitivity,
# atlas, ...). This is the boundary that turns the MOO engine from "the loop" into "a tool".
MooTool = Callable[[DesignState], dict]


# ---------------------------------------------------------------------------
# Critic output
# ---------------------------------------------------------------------------


class CriticVerdict(BaseModel):
    """What a Critic returns from a single review pass."""

    issues: list[DesignIssue] = Field(
        default_factory=list, description="New or updated bottlenecks for the backlog"
    )
    deltas: list[DesignDelta] = Field(
        default_factory=list, description="Concrete edits proposed to close the issues"
    )
    converged: bool = Field(
        default=False, description="Critic judges diminishing returns — stop the loop"
    )
    analysis: str = Field(default="", description="Human-readable rationale")
    research_citations: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Shared base — LLM path with deterministic fallback (house pattern)
# ---------------------------------------------------------------------------


class ReasoningAgent(ABC):
    """Base for agents that reason with Claude but degrade to a heuristic offline.

    Mirrors `reason_node`'s try-LLM-then-heuristic shape so behaviour (and testability
    without an API key) is identical across the loop's reasoning nodes.
    """

    name: str = "reasoning_agent"

    def __init__(self, *, llm_available: bool = False, llm_client: Optional[Any] = None):
        self.llm_available = llm_available
        self._llm_client = llm_client  # inject for tests; else lazily constructed

    def _client(self) -> Any:
        if self._llm_client is not None:
            return self._llm_client
        from embodied_ai_architect.llm.client import LLMClient  # lazy: optional dep

        self._llm_client = LLMClient()
        return self._llm_client


# ---------------------------------------------------------------------------
# Critic
# ---------------------------------------------------------------------------


class Critic(ReasoningAgent):
    """Reviews a DesignState and produces a structured verdict.

    Replaces free-text `PPAMetrics.bottlenecks` / `reason_node`'s "iterate" string with
    typed `DesignIssue`s and applyable `DesignDelta`s.
    """

    name = "critic"

    def review(self, state: DesignState) -> CriticVerdict:
        """Single review pass. Never raises — falls back to heuristic on any LLM error."""
        if self.llm_available:
            try:
                return self._review_with_llm(state)
            except Exception:  # pragma: no cover - parity with reason_node fallback
                pass
        return self._review_heuristic(state)

    # -- LLM path -----------------------------------------------------------

    def _review_with_llm(self, state: DesignState) -> CriticVerdict:
        """Claude ranks bottlenecks with research context and proposes deltas.

        Implementation sketch: build a prompt from ppa_metrics + Pareto front + the
        current open_issues backlog + retrieved research (as `_reason_with_llm` does),
        require JSON out, and parse into CriticVerdict. Research retrieval and prompt
        assembly are elided here — see optimization_loop._reason_with_llm for the shape.
        """
        raise NotImplementedError("LLM critic path — assemble prompt + parse JSON verdict")

    # -- Heuristic path -----------------------------------------------------

    def _review_heuristic(self, state: DesignState) -> CriticVerdict:
        """Deterministic critic: derive issues from failing PPA verdicts.

        Enough to run the loop end-to-end without an API key, and to seed regression
        tests. The LLM path enriches this with rationale, cross-metric reasoning, and
        research-grounded deltas.
        """
        ppa = state.get("ppa_metrics", {})
        verdicts: dict[str, str] = ppa.get("verdicts", {})
        iteration = int(state.get("iteration", 0))

        issues: list[DesignIssue] = []
        deltas: list[DesignDelta] = []

        for metric_name, verdict in verdicts.items():
            if verdict != "FAIL":
                continue
            metric = _to_metric_axis(metric_name)
            issue = DesignIssue(
                metric=metric,
                level=AbstractionLevel.SYSTEM,
                severity=Severity.CRITICAL,
                summary=f"{metric_name} constraint failing",
                observed_value=ppa.get(f"{metric_name}"),
                raised_by=self.name,
                iteration_raised=iteration,
            )
            delta = _default_delta_for(metric, issue)
            issue.delta_ids.append(delta.id)
            issues.append(issue)
            deltas.append(delta)

        # Converge when the critic sees nothing failing AND the frontier has stopped moving.
        converged = not issues and has_converged(state)
        analysis = (
            "No failing constraints; frontier stable — recommend."
            if converged
            else f"{len(issues)} failing constraint(s); proposed {len(deltas)} edit(s)."
        )
        return CriticVerdict(issues=issues, deltas=deltas, converged=converged, analysis=analysis)


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------


class Optimizer(ReasoningAgent):
    """Applies DesignDeltas as concrete edits, then re-runs the MOO engine as a tool.

    Promotes `optimizer.py:design_optimizer`: instead of picking a coarse strategy from a
    catalog, it executes the critic's specific edits, records provenance, and can re-task
    specialists. The MOO run is a tool call, not the loop body.
    """

    name = "optimizer"

    def __init__(self, *, moo_tool: Optional[MooTool] = None, **kwargs):
        super().__init__(**kwargs)
        self.moo_tool = moo_tool

    def optimize(self, state: DesignState, deltas: list[DesignDelta]) -> DesignState:
        """Apply each delta to `state`, then run the MOO tool over the edited space."""
        for delta in deltas:
            self._apply_delta(state, delta)
        if self.moo_tool is not None:
            moo_updates = self.moo_tool(state)
            state.update(moo_updates)  # pareto_points, hypervolume_history, sensitivity, ...
        return state

    def _apply_delta(self, state: DesignState, delta: DesignDelta) -> None:
        """Dispatch a single delta to a concrete state mutation.

        Each branch is a real, auditable edit — the antithesis of "re-run MAP-Elites
        with more effort". Retasking is expressed as a queued specialist request that
        the dispatcher picks up (issue #35-style re-validation after a config change).
        """
        space = state.setdefault("design_space_config", {})
        payload = delta.typed_change()  # validated per-kind payload model (S3)

        if delta.kind == DeltaKind.DESIGN_SPACE_EDIT:
            _set_path(space, delta.target, payload.value)
        elif delta.kind == DeltaKind.VARIABLE_BOUND_CHANGE:
            # Continuous vars carry `bounds` [lo, hi]; categorical vars `categories`.
            if payload.categories is not None:
                _set_path(space, f"{delta.target}.categories", payload.categories)
            else:
                _set_path(space, f"{delta.target}.bounds", payload.bounds)
        elif delta.kind == DeltaKind.ADD_VARIABLE:
            space.setdefault("variables", {})[delta.target] = payload.variable
        elif delta.kind == DeltaKind.REMOVE_VARIABLE:
            space.get("variables", {}).pop(delta.target, None)
        elif delta.kind == DeltaKind.CONSTRAINT_RELAXATION:
            _set_path(state.setdefault("constraints", {}), delta.target, payload.to)
        elif delta.kind == DeltaKind.SPECIALIST_RETASK:
            # `specialist` comes from delta.target and must win over any payload
            # extra literally named "specialist" (SpecialistRetaskPayload allows
            # extras), so it is spread last.
            state.setdefault("pending_specialist_tasks", []).append(
                {**payload.model_dump(), "specialist": delta.target}
            )

        delta.applied = True
        delta.applied_at_iteration = int(state.get("iteration", 0))
        state.setdefault("applied_deltas", []).append(delta.model_dump(mode="json"))
        # Mark the issues this delta closed as resolved (re-evaluation confirms next pass).
        for issue_id in delta.addresses_issue_ids:
            for raw in state.get("open_issues", []):
                if raw.get("id") == issue_id:
                    raw["status"] = IssueStatus.RESOLVED.value
                    raw["resolved_by"] = self.name
                    raw["iteration_resolved"] = int(state.get("iteration", 0))


# ---------------------------------------------------------------------------
# LangGraph node wrappers + router (how they plug into the unified loop)
# ---------------------------------------------------------------------------


def critic_node(state: DesignState) -> dict:
    """Node: run the critic, fold its verdict into the shared backlog."""
    critic = Critic(llm_available=bool(state.get("llm_available", False)))
    verdict = critic.review(state)
    for issue in verdict.issues:
        add_issue(state, issue)
    for delta in verdict.deltas:
        add_delta(state, delta)
    return {
        "open_issues": state.get("open_issues", []),
        "pending_deltas": state.get("pending_deltas", []),
        "converged": verdict.converged,
        "analysis": verdict.analysis,
        "research_citations": verdict.research_citations,
    }


def optimizer_node(state: DesignState, *, moo_tool: Optional[MooTool] = None) -> dict:
    """Node: apply the pending deltas and re-run MOO as a tool.

    Returns every field the deltas or the MOO tool touched. LangGraph propagates
    state purely through the return value — in-place mutation of the input `state`
    does not reliably reach the merged graph state — so all MOO-tool outputs
    (knee_point, sensitivity, atlas, moo_results, pareto_frontier_history) are
    re-emitted here, not just the two the loop happens to read next.
    """
    optimizer = Optimizer(moo_tool=moo_tool)
    pending = [DesignDelta(**d) for d in state.get("pending_deltas", []) if not d.get("applied")]
    optimizer.optimize(state, pending)
    updates: dict = {
        "design_space_config": state.get("design_space_config", {}),
        "constraints": state.get("constraints", {}),
        "pending_specialist_tasks": state.get("pending_specialist_tasks", []),
        "applied_deltas": state.get("applied_deltas", []),
        "open_issues": state.get("open_issues", []),
        "pending_deltas": [],  # drained
        "iteration": int(state.get("iteration", 0)) + 1,
    }
    # Re-emit whatever the MOO tool produced so it lands in the merged state.
    for key in (
        "pareto_points",
        "pareto_frontier_history",
        "hypervolume_history",
        "knee_point",
        "sensitivity",
        "atlas",
        "moo_results",
    ):
        if key in state:
            updates[key] = state[key]
    return updates


def route_after_critic(state: DesignState) -> str:
    """Conditional edge: recommend when converged, else keep optimizing.

    Single decision point that replaces the two loops' separate stop conditions.
    """
    if state.get("converged") or has_converged(state):
        return "recommend"
    return "optimize"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _to_metric_axis(name: str) -> MetricAxis:
    try:
        return MetricAxis(name.lower())
    except ValueError:
        # Map common PPA verdict keys onto axes; default to POWER as the dominant driver.
        return {
            "power_watts": MetricAxis.POWER,
            "latency_ms": MetricAxis.LATENCY,
            "area_mm2": MetricAxis.AREA,
            "cost_usd": MetricAxis.COST,
            "accuracy_percent": MetricAxis.ACCURACY,
        }.get(name, MetricAxis.POWER)


def _default_delta_for(metric: MetricAxis, issue: DesignIssue) -> DesignDelta:
    """Heuristic first-guess edit per metric (the LLM path proposes better-targeted ones)."""
    presets: dict[MetricAxis, dict[str, Any]] = {
        MetricAxis.POWER: {"target": "quantization_dtype", "change": {"value": "int8"}},
        MetricAxis.LATENCY: {"target": "hardware.array_rows", "change": {"value": 32}},
        MetricAxis.AREA: {"target": "hardware.sram_kb", "change": {"value": 256}},
        MetricAxis.COST: {"target": "hardware.process_nm", "change": {"value": 28}},
    }
    preset = presets.get(metric, {"target": "quantization_dtype", "change": {"value": "fp16"}})
    return DesignDelta(
        kind=DeltaKind.DESIGN_SPACE_EDIT,
        target=preset["target"],
        change=preset["change"],
        rationale=f"Heuristic edit to relieve {metric.value} bottleneck",
        addresses_issue_ids=[issue.id],
        proposed_by="critic",
    )


def _set_path(root: dict, dotted: str, value: Any) -> None:
    """Set a dotted path inside a nested dict, creating intermediate dicts."""
    keys = dotted.split(".")
    node = root
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = value
