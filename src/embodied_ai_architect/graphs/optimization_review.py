"""Optimization transparency and human steering for the SoC design loop.

Shows the Pareto frontier, constraint slackness, optimization trajectory,
and strategy rationale at each iteration — enabling a human architect to
build intuition and direct the optimization process.

This module enhances (wraps) the existing evaluate node with rich output.
The optimization review snapshot is stored in state for the CLI/chat layer
to render, and human steering commands are applied to redirect the optimizer.

Key principle: the worst output is just the final answer. We show the
JOURNEY — the Pareto frontier, the trade-offs, the exploration trajectory —
so a human architect can build intuition and direct the process.

Usage:
    from embodied_ai_architect.graphs.optimization_review import (
        OptimizationReviewSnapshot,
        OptimizationSteeringInput,
        build_optimization_review_snapshot,
        render_optimization_review,
    )
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

from embodied_ai_architect.graphs.optimizer import OPTIMIZATION_STRATEGIES
from embodied_ai_architect.graphs.memory import WorkingMemoryStore
from embodied_ai_architect.graphs.soc_state import (
    SoCDesignState,
    get_constraints,
    record_decision,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


class SteeringDecision(str, Enum):
    """Human decision at optimization review."""

    CONTINUE = "continue"  # Let optimizer choose next strategy
    ACCEPT = "accept"  # Accept current design, go to report
    REDIRECT = "redirect"  # Change focus objective or strategy
    EXPLORE_MORE = "explore_more"  # Request broader exploration
    STOP = "stop"  # Stop optimization, go to report with current best


class ConstraintSlackness(BaseModel):
    """Per-constraint analysis: how tight or slack is it?"""

    name: str = Field(description="Constraint name (e.g. 'power', 'latency')")
    target: Optional[float] = Field(default=None, description="Target value from constraints")
    actual: Optional[float] = Field(default=None, description="Current achieved value")
    unit: str = Field(default="", description="Unit of measurement")
    verdict: str = Field(default="UNKNOWN", description="PASS / FAIL / PARTIAL / UNKNOWN")
    margin_pct: Optional[float] = Field(
        default=None,
        description="Margin as %. Positive = slack (within budget), negative = violated",
    )
    binding: bool = Field(
        default=False,
        description="True if margin < 5% (nearly binding constraint)",
    )
    trend: str = Field(
        default="stable",
        description="improving / worsening / stable over recent iterations",
    )


class StrategyAnalysis(BaseModel):
    """Analysis of available and tried optimization strategies."""

    name: str
    description: str
    applicable_when: list[str]
    power_reduction_pct: float
    latency_reduction_pct: float
    accuracy_impact: str
    status: str = Field(description="available / tried / inapplicable")


class OptimizationReviewSnapshot(BaseModel):
    """Everything the human needs to see about the optimization state."""

    # Progress
    iteration: int
    max_iterations: int
    status: str

    # Current PPA
    current_ppa: dict[str, Any]
    verdicts: dict[str, str]
    all_pass: bool

    # Constraint analysis
    constraint_slackness: list[ConstraintSlackness]
    most_violated: Optional[str] = None
    tightest_passing: Optional[str] = None

    # Optimization trajectory
    trajectory: list[dict[str, Any]]
    trajectory_summary: str

    # Strategy analysis
    strategies: list[StrategyAnalysis]
    strategies_tried: list[str]
    strategies_available: list[str]
    recommended_strategy: Optional[str] = None
    strategy_rationale: str = ""

    # Pareto frontier (when available)
    pareto_points: list[dict[str, Any]] = Field(default_factory=list)
    pareto_front_size: int = 0
    hypervolume: Optional[float] = None

    # Frontier evolution across MOO iterations (issue #23)
    frontier_history: list[dict[str, Any]] = Field(default_factory=list)

    # Per-variable sensitivity from Bayesian optimization (issue #24)
    # Shape: {variable_name: {objective_name: impact_score_0_to_1}}
    sensitivity: dict[str, dict[str, float]] = Field(default_factory=dict)

    # MOO convergence (when available)
    moo_summary: dict[str, Any] = Field(default_factory=dict)

    # Design context
    goal: str = ""
    design_rationale: list[str] = Field(default_factory=list)


class OptimizationSteeringInput(BaseModel):
    """Human directives for the optimizer."""

    decision: SteeringDecision = SteeringDecision.CONTINUE
    focus_objective: Optional[str] = Field(
        default=None,
        description="Objective to prioritize (e.g. 'power', 'latency', 'cost')",
    )
    constraint_relaxation: dict[str, float] = Field(
        default_factory=dict,
        description="Constraint name -> new relaxed value (e.g. {'max_power_watts': 6.0})",
    )
    constraint_tightening: dict[str, float] = Field(
        default_factory=dict,
        description="Constraint name -> new tighter value",
    )
    force_strategy: Optional[str] = Field(
        default=None,
        description="Force a specific optimization strategy name",
    )
    accept_point_index: Optional[int] = Field(
        default=None,
        description="Accept a specific Pareto point by index",
    )
    notes: str = ""


# ---------------------------------------------------------------------------
# Constraint slackness computation
# ---------------------------------------------------------------------------

# Map constraint fields to PPA metric fields
_CONSTRAINT_TO_PPA = {
    "max_power_watts": ("power_watts", "W", "power"),
    "max_latency_ms": ("latency_ms", "ms", "latency"),
    "max_area_mm2": ("area_mm2", "mm²", "area"),
    "max_cost_usd": ("cost_usd", "USD", "cost"),
    "max_memory_mb": ("memory_mb", "MB", "memory"),
}

# Min constraints (where actual must be >= target)
_MIN_CONSTRAINTS = {
    "min_fps": ("throughput_fps", "FPS", "throughput"),
    "min_accuracy_percent": ("accuracy_percent", "%", "accuracy"),
}


def compute_constraint_slackness(
    state: SoCDesignState,
) -> list[ConstraintSlackness]:
    """Compute per-constraint slackness analysis."""
    constraints = get_constraints(state)
    ppa = state.get("ppa_metrics", {})
    verdicts = ppa.get("verdicts", {})
    trajectory = state.get("optimization_history", [])

    results: list[ConstraintSlackness] = []

    constraint_data = constraints.model_dump(exclude_none=True)

    # Max constraints (actual must be <= target)
    for field, (ppa_field, unit, short_name) in _CONSTRAINT_TO_PPA.items():
        if field not in constraint_data:
            continue
        target = constraint_data[field]
        actual = ppa.get(ppa_field)

        margin_pct = None
        if actual is not None and target > 0:
            margin_pct = round((target - actual) / target * 100, 1)

        verdict = verdicts.get(short_name, "UNKNOWN")
        binding = margin_pct is not None and 0 <= margin_pct < 5

        trend = _compute_trend(trajectory, ppa_field)

        results.append(
            ConstraintSlackness(
                name=short_name,
                target=target,
                actual=actual,
                unit=unit,
                verdict=verdict,
                margin_pct=margin_pct,
                binding=binding,
                trend=trend,
            )
        )

    # Min constraints (actual must be >= target)
    for field, (ppa_field, unit, short_name) in _MIN_CONSTRAINTS.items():
        if field not in constraint_data:
            continue
        target = constraint_data[field]
        actual = ppa.get(ppa_field)

        margin_pct = None
        if actual is not None and target > 0:
            margin_pct = round((actual - target) / target * 100, 1)

        verdict = verdicts.get(short_name, "UNKNOWN")
        binding = margin_pct is not None and 0 <= margin_pct < 5

        trend = _compute_trend(trajectory, ppa_field, higher_is_better=True)

        results.append(
            ConstraintSlackness(
                name=short_name,
                target=target,
                actual=actual,
                unit=unit,
                verdict=verdict,
                margin_pct=margin_pct,
                binding=binding,
                trend=trend,
            )
        )

    return results


def _compute_trend(
    trajectory: list[dict[str, Any]],
    ppa_field: str,
    higher_is_better: bool = False,
    window: int = 3,
) -> str:
    """Determine trend of a metric over recent iterations."""
    if len(trajectory) < 2:
        return "stable"

    recent = trajectory[-window:]
    values = [
        entry.get("ppa_snapshot", {}).get(ppa_field)
        for entry in recent
        if entry.get("ppa_snapshot", {}).get(ppa_field) is not None
    ]
    if len(values) < 2:
        return "stable"

    delta = values[-1] - values[0]
    threshold = abs(values[0]) * 0.02 if values[0] != 0 else 0.01

    if abs(delta) < threshold:
        return "stable"
    elif (delta < 0 and not higher_is_better) or (delta > 0 and higher_is_better):
        return "improving"
    else:
        return "worsening"


# ---------------------------------------------------------------------------
# Strategy analysis
# ---------------------------------------------------------------------------


def analyze_strategies(state: SoCDesignState) -> list[StrategyAnalysis]:
    """Analyze available and tried optimization strategies."""
    ppa = state.get("ppa_metrics", {})
    verdicts = ppa.get("verdicts", {})
    failing = [k for k, v in verdicts.items() if v == "FAIL"]

    wm_data = state.get("working_memory", {})
    store = WorkingMemoryStore(**wm_data) if wm_data else WorkingMemoryStore()
    already_tried = set(store.get_tried_descriptions("design_optimizer"))

    results = []
    for strat in OPTIMIZATION_STRATEGIES:
        if strat["name"] in already_tried:
            status = "tried"
        elif any(f in strat["applicable_when"] for f in failing):
            status = "available"
        else:
            status = "inapplicable"

        results.append(
            StrategyAnalysis(
                name=strat["name"],
                description=strat["description"],
                applicable_when=strat["applicable_when"],
                power_reduction_pct=round(strat["power_reduction_factor"] * 100, 1),
                latency_reduction_pct=round(strat["latency_reduction_factor"] * 100, 1),
                accuracy_impact=strat["accuracy_impact"],
                status=status,
            )
        )

    return results


# ---------------------------------------------------------------------------
# Trajectory summary
# ---------------------------------------------------------------------------


def summarize_trajectory(trajectory: list[dict[str, Any]]) -> str:
    """Create a human-readable summary of the optimization trajectory."""
    if not trajectory:
        return "No optimization history yet."

    lines = []
    for entry in trajectory:
        it = entry.get("iteration", "?")
        snap = entry.get("ppa_snapshot", {})
        verdicts = entry.get("verdicts", {})

        metrics = []
        if snap.get("power_watts") is not None:
            metrics.append(f"P={snap['power_watts']:.1f}W")
        if snap.get("latency_ms") is not None:
            metrics.append(f"L={snap['latency_ms']:.1f}ms")
        if snap.get("area_mm2") is not None:
            metrics.append(f"A={snap['area_mm2']:.1f}mm²")
        if snap.get("cost_usd") is not None:
            metrics.append(f"C=${snap['cost_usd']:.0f}")

        n_pass = sum(1 for v in verdicts.values() if v == "PASS")
        n_total = len(verdicts)
        verdict_str = f"{n_pass}/{n_total} PASS" if n_total else "no verdicts"

        lines.append(f"  iter {it}: {', '.join(metrics)}  [{verdict_str}]")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Sensitivity normalization (issue #24)
# ---------------------------------------------------------------------------


def normalize_sensitivity(
    raw: dict[str, Any] | None,
) -> dict[str, dict[str, float]]:
    """Normalize the BO sensitivity payload into {variable: {objective: float}}.

    The Bayesian optimizer (`graphs/moo/bayesian_opt.py:_extract_sensitivity`)
    emits a nested dict where each variable carries both `lengthscale` and
    `importance` per objective:

        {
            "power_watts": {
                "quantization_dtype": {"lengthscale": 0.5, "importance": 0.82},
                "npu_frequency_mhz": {"lengthscale": 0.7, "importance": 0.71},
            },
            "latency_ms": {...},
        }

    Consumers (architect skills, API endpoint, snapshot renderer) want a
    transposed view keyed by variable so they can rank knobs by impact:

        {
            "quantization_dtype": {"power_watts": 0.82, "latency_ms": 0.65},
            "npu_frequency_mhz": {"power_watts": 0.71, "latency_ms": 0.58},
        }

    This helper handles both the raw producer format and the already-normalized
    format gracefully (so future producers can short-circuit).
    """
    if not raw:
        return {}

    # Detect format: probe a leaf value
    try:
        first_outer = next(iter(raw.values()))
        if not isinstance(first_outer, dict) or not first_outer:
            return {}
        first_inner = next(iter(first_outer.values()))
    except StopIteration:
        return {}

    normalized: dict[str, dict[str, float]] = {}

    if isinstance(first_inner, dict):
        # Producer format: {objective: {variable: {importance, lengthscale}}}
        for obj_name, var_map in raw.items():
            if not isinstance(var_map, dict):
                continue
            for var_name, metrics in var_map.items():
                if not isinstance(metrics, dict):
                    continue
                importance = metrics.get("importance")
                if importance is None:
                    continue
                normalized.setdefault(var_name, {})[obj_name] = float(importance)
    elif isinstance(first_inner, (int, float)):
        # Already normalized: figure out which axis is which.
        # If the outer keys look like variables (multi-word, snake_case
        # parameter names) we assume {variable: {objective: float}}.
        # Otherwise treat outer as objectives and transpose.
        # Heuristic: known objective names contain "_watts", "_ms", "_usd",
        # "_mm2", "_grams", "_cm3" — if any outer key matches, transpose.
        _OBJ_HINTS = ("_watts", "_ms", "_usd", "_mm2", "_grams", "_cm3", "accuracy")
        outer_keys_look_like_objectives = any(
            any(hint in k for hint in _OBJ_HINTS) for k in raw.keys()
        )
        if outer_keys_look_like_objectives:
            # Transpose objective→variable to variable→objective
            for obj_name, var_map in raw.items():
                if not isinstance(var_map, dict):
                    continue
                for var_name, score in var_map.items():
                    if isinstance(score, (int, float)):
                        normalized.setdefault(var_name, {})[obj_name] = float(score)
        else:
            # Already in {variable: {objective: float}} form
            for var_name, obj_map in raw.items():
                if not isinstance(obj_map, dict):
                    continue
                normalized[var_name] = {
                    obj: float(score)
                    for obj, score in obj_map.items()
                    if isinstance(score, (int, float))
                }

    return normalized


# ---------------------------------------------------------------------------
# Snapshot builder
# ---------------------------------------------------------------------------


def build_optimization_review_snapshot(
    state: SoCDesignState,
) -> OptimizationReviewSnapshot:
    """Aggregate all optimization state into a displayable snapshot."""
    ppa = state.get("ppa_metrics", {})
    verdicts = ppa.get("verdicts", {})
    all_pass = all(v == "PASS" for v in verdicts.values()) if verdicts else False
    trajectory = state.get("optimization_history", [])

    slackness = compute_constraint_slackness(state)
    strategies = analyze_strategies(state)

    # Find most violated and tightest passing
    most_violated = None
    tightest_passing = None
    min_fail_margin = float("inf")
    min_pass_margin = float("inf")

    for cs in slackness:
        if cs.margin_pct is not None:
            if cs.verdict == "FAIL" and cs.margin_pct < min_fail_margin:
                min_fail_margin = cs.margin_pct
                most_violated = cs.name
            elif cs.verdict == "PASS" and cs.margin_pct < min_pass_margin:
                min_pass_margin = cs.margin_pct
                tightest_passing = cs.name

    # Strategy recommendation
    available = [s for s in strategies if s.status == "available"]
    tried = [s.name for s in strategies if s.status == "tried"]
    available_names = [s.name for s in available]

    recommended = None
    rationale = ""
    if available:
        # Pick the most impactful available strategy
        if most_violated:
            key = "power_reduction_pct" if most_violated == "power" else "latency_reduction_pct"
            available.sort(key=lambda s: getattr(s, key, 0), reverse=True)
        recommended = available[0].name
        rationale = (
            f"Recommended '{recommended}' ({available[0].description}) "
            f"targeting '{most_violated or 'general'}' constraint. "
            f"{len(tried)} strategies already tried."
        )
    elif tried:
        rationale = f"All applicable strategies exhausted ({len(tried)} tried). Consider relaxing constraints."

    # If the design_optimizer ran with MOO sensitivity, prefer its
    # structured rationale (issue #25) over the generic one above.
    live_rationale = state.get("last_strategy_rationale", "")
    if live_rationale:
        rationale = live_rationale

    # Pareto data
    pareto_points = state.get("pareto_points", [])
    pareto_results = state.get("pareto_results", {})
    frontier_history = state.get("pareto_frontier_history", [])

    # MOO data (moo_results is OptimizationResult.model_dump())
    moo_results = state.get("moo_results", {})
    moo_summary = {}
    sensitivity: dict[str, dict[str, float]] = {}
    if moo_results:
        # Derive convergence metric from convergence_history (last entry's hypervolume)
        conv_history = moo_results.get("convergence_history", [])
        convergence_metric = conv_history[-1].get("hypervolume") if conv_history else None
        moo_summary = {
            "total_evaluations": moo_results.get("total_evaluations", 0),
            "pareto_size": len(moo_results.get("pareto_front", [])),
            "hypervolume": moo_results.get("hypervolume"),
            "convergence_metric": convergence_metric,
            "layers_used": moo_results.get("layers_used", []),
        }
        # Per-variable sensitivity from BO layer (issue #24).
        # The producer (bayesian_opt._extract_sensitivity) emits
        # {objective: {variable: {importance, lengthscale}}} — normalize
        # to {variable: {objective: float}} for the architect skills.
        sensitivity = normalize_sensitivity(moo_results.get("sensitivity"))

    return OptimizationReviewSnapshot(
        iteration=state.get("iteration", 0),
        max_iterations=state.get("max_iterations", 20),
        status=state.get("status", "unknown"),
        current_ppa=ppa,
        verdicts=verdicts,
        all_pass=all_pass,
        constraint_slackness=[cs.model_dump() for cs in slackness],
        most_violated=most_violated,
        tightest_passing=tightest_passing,
        trajectory=trajectory,
        trajectory_summary=summarize_trajectory(trajectory),
        strategies=[s.model_dump() for s in strategies],
        strategies_tried=tried,
        strategies_available=available_names,
        recommended_strategy=recommended,
        strategy_rationale=rationale,
        pareto_points=pareto_points,
        pareto_front_size=len(pareto_results.get("front", [])) if pareto_results else 0,
        hypervolume=moo_results.get("hypervolume"),
        frontier_history=frontier_history,
        sensitivity=sensitivity,
        moo_summary=moo_summary,
        goal=state.get("goal", ""),
        design_rationale=state.get("design_rationale", [])[-5:],
    )


# ---------------------------------------------------------------------------
# Steering application
# ---------------------------------------------------------------------------


def apply_steering_input(
    state: SoCDesignState,
    steering: OptimizationSteeringInput,
) -> dict[str, Any]:
    """Apply human steering directives to the optimization state.

    Returns a state update dict for LangGraph merging.
    """
    updates: dict[str, Any] = {}

    # Record the decision
    state = record_decision(
        state,
        agent="human_architect",
        action=f"Optimization steering: {steering.decision.value}",
        rationale=steering.notes or f"Human directed: {steering.decision.value}",
        data={
            "focus_objective": steering.focus_objective,
            "force_strategy": steering.force_strategy,
            "constraint_relaxation": steering.constraint_relaxation,
            "constraint_tightening": steering.constraint_tightening,
        },
    )
    updates["history"] = state["history"]
    updates["design_rationale"] = state["design_rationale"]

    if steering.decision == SteeringDecision.ACCEPT:
        updates["next_action"] = "report"

    elif steering.decision == SteeringDecision.STOP:
        updates["next_action"] = "report"

    elif steering.decision == SteeringDecision.REDIRECT:
        # Modify working memory to bias strategy selection
        wm_data = state.get("working_memory", {})
        store = WorkingMemoryStore(**wm_data) if wm_data else WorkingMemoryStore()

        if steering.focus_objective:
            store.record_decision(
                "human_architect",
                f"Focus optimization on: {steering.focus_objective}",
            )

        if steering.force_strategy:
            store.record_decision(
                "human_architect",
                f"Force strategy: {steering.force_strategy}",
            )

        updates["working_memory"] = store.model_dump()

        # Store steering directives for the optimizer to read
        updates["optimization_steering"] = steering.model_dump()
        updates["next_action"] = "optimize"

    elif steering.decision == SteeringDecision.EXPLORE_MORE:
        updates["next_action"] = "optimize"

    else:  # CONTINUE
        pass  # Let existing routing logic decide

    # Apply constraint modifications
    if steering.constraint_relaxation or steering.constraint_tightening:
        constraints = get_constraints(state).model_dump()
        constraints.update(steering.constraint_relaxation)
        constraints.update(steering.constraint_tightening)
        updates["constraints"] = constraints

    return updates


# ---------------------------------------------------------------------------
# Enhanced evaluate node
# ---------------------------------------------------------------------------


def make_enhanced_evaluate_node(
    base_evaluate_fn: Any,
) -> Any:
    """Wrap the base evaluate node with optimization review snapshot.

    The enhanced node:
    1. Runs the base evaluate logic (verdict checking, history recording)
    2. Builds an OptimizationReviewSnapshot
    3. If optimization_steering is in state, applies steering
    4. Stores snapshot for display
    """

    def enhanced_evaluate_node(state: SoCDesignState) -> dict[str, Any]:
        # Run base evaluate
        base_updates = base_evaluate_fn(state)
        merged_state = {**state, **base_updates}

        # Build snapshot
        snapshot = build_optimization_review_snapshot(merged_state)
        base_updates["optimization_review_snapshot"] = snapshot.model_dump()

        # Apply steering if present
        steering_raw = state.get("optimization_steering")
        if steering_raw:
            steering = OptimizationSteeringInput(**steering_raw)
            steering_updates = apply_steering_input(merged_state, steering)
            base_updates.update(steering_updates)
            # Clear steering after applying
            base_updates["optimization_steering"] = {}

        return base_updates

    return enhanced_evaluate_node


# ---------------------------------------------------------------------------
# Rich rendering
# ---------------------------------------------------------------------------


def render_optimization_review(snapshot: OptimizationReviewSnapshot) -> str:
    """Render the optimization review snapshot as formatted text."""
    lines: list[str] = []

    # Header
    lines.append("=" * 72)
    lines.append(
        "  OPTIMIZATION REVIEW — Iteration " f"{snapshot.iteration}/{snapshot.max_iterations}"
    )
    lines.append("=" * 72)
    lines.append("")

    if snapshot.goal:
        lines.append(f"  Goal: {snapshot.goal}")
        lines.append("")

    # Verdict banner
    if snapshot.all_pass:
        lines.append("  *** ALL CONSTRAINTS PASS ***")
    else:
        failing = [k for k, v in snapshot.verdicts.items() if v == "FAIL"]
        lines.append(f"  Failing: {', '.join(failing)}")
    lines.append("")

    # Constraint slackness table
    lines.append("─" * 72)
    lines.append("  CONSTRAINT ANALYSIS")
    lines.append("─" * 72)
    lines.append("")
    lines.append(
        f"  {'Constraint':<12} {'Target':>10} {'Actual':>10} "
        f"{'Margin':>8} {'Verdict':>8} {'Trend':>10}"
    )
    lines.append(f"  {'─' * 12} {'─' * 10} {'─' * 10} {'─' * 8} {'─' * 8} {'─' * 10}")

    for cs_data in snapshot.constraint_slackness:
        cs = ConstraintSlackness(**cs_data) if isinstance(cs_data, dict) else cs_data
        target_str = f"{cs.target:.1f}{cs.unit}" if cs.target is not None else "-"
        actual_str = f"{cs.actual:.1f}{cs.unit}" if cs.actual is not None else "-"
        margin_str = f"{cs.margin_pct:+.1f}%" if cs.margin_pct is not None else "-"

        # Indicators
        verdict_indicator = cs.verdict
        if cs.binding:
            verdict_indicator += "!"

        trend_indicator = {"improving": "^", "worsening": "v", "stable": "="}
        trend_str = f"{cs.trend} {trend_indicator.get(cs.trend, '')}"

        lines.append(
            f"  {cs.name:<12} {target_str:>10} {actual_str:>10} "
            f"{margin_str:>8} {verdict_indicator:>8} {trend_str:>10}"
        )

    if snapshot.most_violated:
        lines.append("")
        lines.append(f"  Most violated: {snapshot.most_violated}")
    if snapshot.tightest_passing:
        lines.append(f"  Tightest passing: {snapshot.tightest_passing}")
    lines.append("")

    # Optimization trajectory
    lines.append("─" * 72)
    lines.append("  OPTIMIZATION TRAJECTORY")
    lines.append("─" * 72)
    lines.append("")
    lines.append(snapshot.trajectory_summary or "  (no trajectory yet)")
    lines.append("")

    # Strategy analysis
    lines.append("─" * 72)
    lines.append("  STRATEGY ANALYSIS")
    lines.append("─" * 72)
    lines.append("")

    for s_data in snapshot.strategies:
        s = StrategyAnalysis(**s_data) if isinstance(s_data, dict) else s_data
        status_mark = {"available": "[avail]", "tried": "[tried]", "inapplicable": "[n/a]"}
        mark = status_mark.get(s.status, "[?]")
        impact = f"P-{s.power_reduction_pct}% L-{s.latency_reduction_pct}%"
        lines.append(f"  {mark:>9} {s.name:<25} {impact:<20} acc: {s.accuracy_impact}")

    lines.append("")
    if snapshot.strategy_rationale:
        lines.append(f"  Rationale: {snapshot.strategy_rationale}")
        lines.append("")

    # Pareto / MOO summary
    if snapshot.pareto_front_size > 0 or snapshot.moo_summary:
        lines.append("─" * 72)
        lines.append("  PARETO FRONTIER & MOO")
        lines.append("─" * 72)
        lines.append("")
        if snapshot.pareto_front_size:
            lines.append(f"  Pareto front size: {snapshot.pareto_front_size} points")
        if snapshot.hypervolume is not None:
            lines.append(f"  Hypervolume: {snapshot.hypervolume:.4f}")
        if snapshot.moo_summary:
            moo = snapshot.moo_summary
            if moo.get("total_evaluations"):
                lines.append(f"  Total evaluations: {moo['total_evaluations']}")
        lines.append("")

    # Frontier evolution across iterations (issue #23)
    if snapshot.frontier_history:
        lines.append("─" * 72)
        lines.append("  FRONTIER EVOLUTION")
        lines.append("─" * 72)
        lines.append("")
        for entry in snapshot.frontier_history:
            it = entry.get("iteration", "?")
            n = entry.get("num_points", 0)
            new_added = entry.get("new_points_added", 0)
            removed = entry.get("dominated_removed", 0)
            hv = entry.get("hypervolume", 0.0)
            delta = ""
            if new_added or removed:
                delta = f" (+{new_added} new, -{removed} dominated)"
            lines.append(f"  Iteration {it}: {n} Pareto points{delta}, HV={hv:.4f}")
        lines.append("")

    # Design variable sensitivity from BO layer (issue #24)
    if snapshot.sensitivity:
        lines.append("─" * 72)
        lines.append("  DESIGN VARIABLE SENSITIVITY (most impactful first)")
        lines.append("─" * 72)
        lines.append("")
        # Discover the objective columns (union across all variables)
        objectives: list[str] = []
        for obj_map in snapshot.sensitivity.values():
            for obj_name in obj_map:
                if obj_name not in objectives:
                    objectives.append(obj_name)
        # Stable column order: power → latency → cost → area → others
        _OBJ_ORDER = ["power_watts", "latency_ms", "cost_usd", "area_mm2"]
        ordered = [o for o in _OBJ_ORDER if o in objectives] + [
            o for o in objectives if o not in _OBJ_ORDER
        ]

        # Rank variables by max impact across all objectives (most impactful first)
        ranked_vars = sorted(
            snapshot.sensitivity.items(),
            key=lambda kv: max(kv[1].values(), default=0.0),
            reverse=True,
        )

        # Header row
        header = f"  {'Variable':<22}"
        for obj in ordered:
            header += f"  {obj:>10}"
        lines.append(header)
        lines.append(f"  {'─' * 22}" + ("  " + "─" * 10) * len(ordered))

        # Data rows
        for var_name, obj_map in ranked_vars:
            row = f"  {var_name:<22}"
            for obj in ordered:
                val = obj_map.get(obj)
                row += f"  {val:>10.2f}" if val is not None else f"  {'—':>10}"
            lines.append(row)
        lines.append("")
        lines.append(
            "  [hint] Highest-impact variables drive the design — focus optimization on these"
        )
        lines.append("")

    # Recent design rationale
    if snapshot.design_rationale:
        lines.append("─" * 72)
        lines.append("  RECENT DECISIONS (last 5)")
        lines.append("─" * 72)
        lines.append("")
        for r in snapshot.design_rationale:
            lines.append(f"  {r}")
        lines.append("")

    # Steering options
    lines.append("─" * 72)
    lines.append("  STEERING OPTIONS")
    lines.append("─" * 72)
    lines.append("")
    lines.append("  continue       — Let optimizer choose next strategy")
    lines.append("  accept         — Accept current design, generate report")
    lines.append("  redirect       — Change focus objective or force strategy")
    lines.append("  explore_more   — Request broader exploration")
    lines.append("  stop           — Stop and report current best")
    lines.append("")
    lines.append("=" * 72)

    return "\n".join(lines)
