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
        default=False, description="No failing constraints and the frontier is stable"
    )
    diminishing_returns: bool = Field(
        default=False,
        description="Critic judges further iteration won't help even if constraints fail (S12)",
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


CRITIC_SYSTEM_PROMPT = """\
You are an embodied-AI SoC design critic. You review a design state (PPA verdicts,
Pareto front, open issues, research context) and return a STRUCTURED verdict that a
downstream Optimizer agent applies mechanically. Be concrete: every bottleneck is an
issue, every fix is a delta that names a design-space variable or constraint.

Respond with JSON only, in exactly this shape:
{
  "converged": false,
  "diminishing_returns": false,
  "analysis": "one short paragraph",
  "research_citations": ["doc/path.md", ...],
  "issues": [
    {
      "metric": "power|latency|throughput|area|cost|thermal|accuracy|capability_per_watt|utilization|bandwidth|memory|weight|volume|reliability",
      "level": "system|subsystem|operator|kernel|hardware|physical",
      "component": "optional operator/block name",
      "severity": "critical|high|medium|low",
      "summary": "one line",
      "observed_value": 6.0,
      "target_value": 5.0,
      "contribution_pct": 40.0
    }
  ],
  "deltas": [
    {
      "kind": "design_space_edit|variable_bound_change|add_variable|remove_variable|constraint_relaxation|specialist_retask",
      "target": "design-space variable, constraint field, or specialist id",
      "change": { ... per-kind payload ... },
      "rationale": "why this helps",
      "addresses_issue": 0,
      "research_refs": ["doc/path.md that motivates this edit", ...]
    }
  ]
}

Per-kind `change` payloads:
- design_space_edit      -> {"value": <new value>}
- variable_bound_change  -> {"bounds": [lo, hi]}  OR  {"categories": [...]}   (exactly one)
- add_variable           -> {"variable": {<spec>}}
- remove_variable        -> {}
- constraint_relaxation  -> {"to": <new>, "from": <old>}
- specialist_retask      -> {"reason": "..."}     (plus any extra keys)

Rules: set "converged": true only when no constraint is failing AND further search
would not help. Set "diminishing_returns": true when further iteration is unlikely
to help even though constraints still fail (the levers are exhausted) — this stops
the loop and asks the operator to steer. Each delta's "addresses_issue" is the
0-based index into "issues". Prefer edits to high-contribution / high-severity
issues first."""


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
        """Claude ranks bottlenecks with research context and proposes typed deltas.

        Assembles a prompt from ppa_metrics + Pareto front + the open_issues backlog +
        retrieved research (mirrors optimization_loop._reason_with_llm), requires JSON,
        and parses it into a CriticVerdict. Raises on any client/parse error; review()
        catches that and falls back to the heuristic path.
        """
        import json
        import re

        client = self._client()
        ppa = state.get("ppa_metrics", {})
        verdicts = ppa.get("verdicts", {})
        pareto = state.get("pareto_points") or state.get("pareto_front") or []
        open_issues_summary = [
            {"metric": i.get("metric"), "summary": i.get("summary"), "status": i.get("status")}
            for i in state.get("open_issues", [])
        ]
        metrics_now = {
            k: ppa.get(k)
            for k in ("power_watts", "latency_ms", "area_mm2", "cost_usd", "accuracy_percent")
            if ppa.get(k) is not None
        }

        prompt = f"""Review this embodied-AI SoC design state and produce a critic verdict.

Mission: {state.get('mission_description', state.get('goal', 'unknown'))}
Platform: {state.get('platform', 'unknown')}
Iteration: {state.get('iteration', 0)}
Constraints: {json.dumps(state.get('constraints', {}), default=str)[:1500]}

PPA verdicts: {json.dumps(verdicts, default=str)}
Current metrics: {json.dumps(metrics_now, default=str)}

Top Pareto designs:
{json.dumps(pareto[:5], default=str, indent=2)[:2000]}

Already-open issues: {json.dumps(open_issues_summary, default=str)[:1500]}

{self._retrieve_research(state)}

Identify the top bottlenecks as structured issues, propose concrete deltas to
resolve them (each addressing an issue by index), and decide whether the design
has converged. Respond with JSON only."""

        response = client.chat(
            messages=[{"role": "user", "content": prompt}], system=CRITIC_SYSTEM_PROMPT
        )
        text = re.sub(r"^```json\s*|\s*```$", "", response.text.strip())
        data = json.loads(text)
        return self._verdict_from_data(data, state)

    def _retrieve_research(self, state: DesignState) -> str:
        """Best-effort research context block, targeted at the current bottlenecks.

        Tags are derived from the failing metrics / open issues (S5), so a
        power-bound drone and a latency-bound AMR retrieve different research —
        the context (and thus the deltas the LLM grounds in it) varies with the
        mission and its bottlenecks, not just a fixed 'efficiency' query.
        """
        try:
            from embodied_ai_architect.research.library import ResearchLibrary

            library = ResearchLibrary()
            docs = library.retrieve(
                tags=_research_tags_for_state(state),
                relevance="design_tradeoffs",
                mission_type=state.get("mission_type"),
                max_results=4,
            )
            return library.build_context_block(docs, max_tokens=3000)
        except Exception:
            return ""

    def _verdict_from_data(self, data: dict, state: DesignState) -> CriticVerdict:
        """Parse the LLM's JSON into a validated CriticVerdict (skips malformed items)."""
        iteration = int(state.get("iteration", 0))

        # Keep the ORIGINAL LLM array index -> parsed issue, so that a delta's
        # `addresses_issue` still resolves correctly after malformed issues are
        # skipped (skipping would otherwise compact the list and misalign indices).
        issues_by_orig_idx: dict[int, DesignIssue] = {}
        for orig_idx, raw in enumerate(data.get("issues", []) or []):
            try:
                issues_by_orig_idx[orig_idx] = DesignIssue(
                    metric=_to_metric_axis(str(raw.get("metric", ""))),
                    level=_parse_level(raw.get("level")),
                    severity=_parse_severity(raw.get("severity")),
                    component=raw.get("component"),
                    summary=raw.get("summary", "(no summary)"),
                    observed_value=raw.get("observed_value"),
                    target_value=raw.get("target_value"),
                    contribution_pct=raw.get("contribution_pct"),
                    raised_by=self.name,
                    iteration_raised=iteration,
                )
            except Exception:
                continue
        issues = list(issues_by_orig_idx.values())

        deltas: list[DesignDelta] = []
        for raw in data.get("deltas", []) or []:
            try:
                delta = DesignDelta(
                    kind=DeltaKind(str(raw.get("kind", ""))),
                    target=str(raw.get("target", "")),
                    change=raw.get("change", {}) or {},
                    rationale=raw.get("rationale", "(no rationale)"),
                    research_refs=[str(r) for r in (raw.get("research_refs") or [])],
                    proposed_by=self.name,
                )
            except Exception:
                # Bad kind or a payload that fails per-kind validation (S3) — skip it.
                continue
            target_issue = issues_by_orig_idx.get(raw.get("addresses_issue"))
            if target_issue is not None:
                delta.addresses_issue_ids.append(target_issue.id)
                target_issue.delta_ids.append(delta.id)
            deltas.append(delta)

        # Convergence is a real boolean, not Python truthiness ("false"/"0" are False),
        # and can never be claimed while any constraint verdict is still FAIL.
        converged = _coerce_bool(data.get("converged", False))
        verdicts = state.get("ppa_metrics", {}).get("verdicts", {})
        if converged and any(v == "FAIL" for v in verdicts.values()):
            converged = False

        return CriticVerdict(
            issues=issues,
            deltas=deltas,
            converged=converged,
            diminishing_returns=_coerce_bool(data.get("diminishing_returns", False)),
            analysis=str(data.get("analysis", "")),
            research_citations=list(data.get("research_citations", []) or []),
        )

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
        "critic_diminishing_returns": verdict.diminishing_returns,
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
    _run_pending_retasks(state)  # S8: consume SPECIALIST_RETASK deltas
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


def _run_pending_retasks(state: DesignState) -> int:
    """S8: consume `pending_specialist_tasks` — re-run each named specialist agent
    (from the S9 registry), file its fresh DesignIssues, and drain the queue.

    A SPECIALIST_RETASK delta (applied by the Optimizer) enqueues a task; this
    re-runs that specialist so its issues reflect the just-applied design change
    (the loop analog of the dispatcher's #35 re-validation after a config edit).
    Returns the number of specialists re-run.
    """
    tasks = state.get("pending_specialist_tasks", [])
    if not tasks:
        return 0
    from embodied_ai_architect.graphs.specialist_agents import specialist_registry

    registry = specialist_registry()
    ran = 0
    for task in tasks:
        agent = registry.get(task.get("specialist"))
        if agent is None:
            continue  # unknown specialist — leave it out, don't crash the loop
        for issue in agent.assess(state):
            add_issue(state, issue)
        ran += 1
    state["pending_specialist_tasks"] = []  # drain
    return ran


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


# Which research-library tags are relevant to relieving each metric bottleneck (S5).
_METRIC_RESEARCH_TAGS: dict[MetricAxis, list[str]] = {
    MetricAxis.POWER: ["efficiency", "quantization", "sparsity"],
    MetricAxis.CAPABILITY_PER_WATT: ["efficiency", "quantization"],
    MetricAxis.LATENCY: ["dataflow", "systolic"],
    MetricAxis.THROUGHPUT: ["dataflow", "systolic"],
    MetricAxis.AREA: ["memory", "sparsity"],
    MetricAxis.ACCURACY: ["quantization", "nas"],
    MetricAxis.BANDWIDTH: ["memory", "noc"],
    MetricAxis.MEMORY: ["memory"],
    MetricAxis.THERMAL: ["efficiency", "packaging"],
    MetricAxis.COST: ["cost", "manufacturing", "packaging"],
    MetricAxis.UTILIZATION: ["dataflow", "systolic", "tiling"],
    MetricAxis.WEIGHT: ["swap", "packaging"],
    MetricAxis.VOLUME: ["swap", "packaging"],
    MetricAxis.RELIABILITY: ["reliability", "safety"],
}
# Every MetricAxis must map to a tag set, so a bottleneck on any metric yields
# metric-specific research (not the generic fallback). Guarded by a test.
assert set(_METRIC_RESEARCH_TAGS) == set(MetricAxis), "research tag map missing a MetricAxis"


def _research_tags_for_state(state: DesignState) -> list[str]:
    """Derive research-retrieval tags from the state's bottlenecks (failing PPA
    verdicts + open issues), so the retrieved research varies with the mission,
    not just a fixed query. Always includes the platform; falls back to
    'efficiency' when there is no specific bottleneck."""
    tags: list[str] = [str(state.get("platform", "edge"))]
    axes: list[MetricAxis] = []
    for name, verdict in state.get("ppa_metrics", {}).get("verdicts", {}).items():
        if verdict == "FAIL":
            axes.append(_to_metric_axis(name))
    for issue in state.get("open_issues", []):
        try:
            axes.append(MetricAxis(str(issue.get("metric", ""))))
        except ValueError:
            continue
    for axis in axes:
        tags.extend(_METRIC_RESEARCH_TAGS.get(axis, []))
    if len(tags) == 1:  # platform only — no specific bottleneck
        tags.append("efficiency")
    return list(dict.fromkeys(tags))  # dedup, preserve order


def _coerce_bool(value: Any) -> bool:
    """Coerce an LLM-supplied value to bool without Python truthiness surprises.

    A JSON string "false"/"0"/"no" is falsey here (plain bool("false") would be True).
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    return False


def _parse_level(value: Any) -> AbstractionLevel:
    """Map an LLM-supplied level string onto AbstractionLevel (default SYSTEM)."""
    try:
        return AbstractionLevel(str(value).lower())
    except ValueError:
        return AbstractionLevel.SYSTEM


def _parse_severity(value: Any) -> Severity:
    """Map an LLM-supplied severity string onto Severity (default MEDIUM)."""
    try:
        return Severity(str(value).lower())
    except ValueError:
        return Severity.MEDIUM


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
