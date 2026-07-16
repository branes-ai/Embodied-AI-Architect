"""Specialist agents + estimator tools (Seam S9, #213).

Promotes the judgment-bearing specialists to agents that reason over the shared
`DesignState` via **verdict-first estimator tools** and file `DesignIssue`s — the
same currency the Critic and Optimizer already exchange. The numeric estimation
stays deterministic (the tools); the judgment (which violations matter, how
severe) lives in the agents.

Two specialist agents ship here:

- `PPASpecialist` — power/latency/area/cost tools → DesignIssues, with severity
  scaled by how far over budget each metric is (the dominant bottleneck first).
- `ThermalSpecialist` — a junction-temperature tool (from `physical_estimators`)
  → a THERMAL DesignIssue when the design would overheat — a dimension the
  top-level PPA verdicts don't cover.

`run_specialists(state)` runs them and returns the issues; `file_specialist_issues`
folds them into the state's `open_issues` backlog.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from pydantic import BaseModel

from embodied_ai_architect.graphs.design_state import (
    AbstractionLevel,
    DesignIssue,
    DesignState,
    MetricAxis,
    Severity,
    add_issue,
)
from embodied_ai_architect.graphs.physical_estimators import estimate_junction_temperature
from embodied_ai_architect.graphs.soc_state import get_constraints

# ---------------------------------------------------------------------------
# Verdict-first estimator tools
# ---------------------------------------------------------------------------


class EstimatorResult(BaseModel):
    """A verdict-first estimate: the value, the budget, and whether it fits."""

    metric: MetricAxis
    value: Optional[float] = None
    limit: Optional[float] = None
    detail: str = ""

    @property
    def verdict(self) -> str:
        if self.value is None or self.limit is None:
            return "UNKNOWN"
        return "PASS" if self.value <= self.limit else "FAIL"

    @property
    def margin(self) -> Optional[float]:
        """limit - value; negative means over budget."""
        if self.value is None or self.limit is None:
            return None
        return self.limit - self.value

    @property
    def overshoot_pct(self) -> Optional[float]:
        if self.value is None or not self.limit:
            return None
        return max(0.0, (self.value - self.limit) / self.limit * 100.0)


def _estimated(state: DesignState, ppa_key: str) -> Optional[float]:
    """The design's estimated value for a metric — from ppa_metrics (populated by
    evaluate_node/ppa_assessor) or, failing that, the knee design point."""
    ppa = state.get("ppa_metrics", {})
    if ppa.get(ppa_key) is not None:
        return float(ppa[ppa_key])
    knee = state.get("knee_point") or {}
    obj = knee.get("objectives", {})
    return float(obj[ppa_key]) if obj.get(ppa_key) is not None else None


def power_tool(state: DesignState) -> EstimatorResult:
    c = get_constraints(state)
    return EstimatorResult(
        metric=MetricAxis.POWER, value=_estimated(state, "power_watts"), limit=c.max_power_watts
    )


def latency_tool(state: DesignState) -> EstimatorResult:
    c = get_constraints(state)
    return EstimatorResult(
        metric=MetricAxis.LATENCY, value=_estimated(state, "latency_ms"), limit=c.max_latency_ms
    )


def area_tool(state: DesignState) -> EstimatorResult:
    c = get_constraints(state)
    return EstimatorResult(
        metric=MetricAxis.AREA, value=_estimated(state, "area_mm2"), limit=c.max_area_mm2
    )


def cost_tool(state: DesignState) -> EstimatorResult:
    c = get_constraints(state)
    return EstimatorResult(
        metric=MetricAxis.COST, value=_estimated(state, "cost_usd"), limit=c.max_cost_usd
    )


def thermal_tool(
    state: DesignState,
    *,
    theta_c_per_w: float = 6.0,
    ambient_temp_c: float = 40.0,
    max_junction_temp_c: float = 125.0,
) -> EstimatorResult:
    """Junction temperature from the design's power, via
    `physical_estimators.estimate_junction_temperature`."""
    power = _estimated(state, "power_watts")
    if power is None:
        return EstimatorResult(
            metric=MetricAxis.THERMAL,
            value=None,
            limit=max_junction_temp_c,
            detail="no power estimate",
        )
    tj = estimate_junction_temperature(power, theta_c_per_w, ambient_temp_c)
    return EstimatorResult(
        metric=MetricAxis.THERMAL,
        value=round(tj, 1),
        limit=max_junction_temp_c,
        detail=f"T_j={tj:.0f}°C at {power:.1f}W (θ={theta_c_per_w} °C/W, ambient {ambient_temp_c}°C)",
    )


def _severity_from(result: EstimatorResult) -> Severity:
    """Judgment: how bad is this violation?"""
    pct = result.overshoot_pct or 0.0
    if pct >= 50:
        return Severity.CRITICAL
    if pct >= 15:
        return Severity.HIGH
    return Severity.MEDIUM


# ---------------------------------------------------------------------------
# Specialist agents
# ---------------------------------------------------------------------------


class SpecialistAgent(ABC):
    """A specialist that reasons over the shared state and files DesignIssues."""

    name: str = "specialist"

    @abstractmethod
    def assess(self, state: DesignState) -> list[DesignIssue]:
        """Return DesignIssues for the bottlenecks this specialist owns."""


class PPASpecialist(SpecialistAgent):
    """Files PPA DesignIssues from the estimator tools, dominant bottleneck first."""

    name = "ppa_specialist"

    def assess(self, state: DesignState) -> list[DesignIssue]:
        iteration = int(state.get("iteration", 0))
        results = [t(state) for t in (power_tool, latency_tool, area_tool, cost_tool)]
        failing = [r for r in results if r.verdict == "FAIL"]
        # Judgment: order by overshoot so the worst bottleneck is filed first.
        failing.sort(key=lambda r: r.overshoot_pct or 0.0, reverse=True)
        return [
            DesignIssue(
                metric=r.metric,
                level=AbstractionLevel.SYSTEM,
                severity=_severity_from(r),
                summary=f"{r.metric.value} {r.value} exceeds budget {r.limit} "
                f"(+{(r.overshoot_pct or 0.0):.0f}%)",
                observed_value=r.value,
                target_value=r.limit,
                contribution_pct=r.overshoot_pct,
                raised_by=self.name,
                iteration_raised=iteration,
            )
            for r in failing
        ]


class ThermalSpecialist(SpecialistAgent):
    """Files a THERMAL DesignIssue when the design would exceed the junction limit."""

    name = "thermal_specialist"

    def __init__(self, *, theta_c_per_w: float = 6.0, max_junction_temp_c: float = 125.0):
        self.theta = theta_c_per_w
        self.max_junction = max_junction_temp_c

    def assess(self, state: DesignState) -> list[DesignIssue]:
        r = thermal_tool(state, theta_c_per_w=self.theta, max_junction_temp_c=self.max_junction)
        if r.verdict != "FAIL":
            return []
        return [
            DesignIssue(
                metric=MetricAxis.THERMAL,
                level=AbstractionLevel.PHYSICAL,
                severity=_severity_from(r),
                summary=f"junction temperature {r.value}°C exceeds {r.limit}°C limit",
                observed_value=r.value,
                target_value=r.limit,
                raised_by=self.name,
                iteration_raised=int(state.get("iteration", 0)),
                metadata={"detail": r.detail},
            )
        ]


DEFAULT_SPECIALISTS: list[SpecialistAgent] = [PPASpecialist(), ThermalSpecialist()]


def run_specialists(
    state: DesignState, agents: Optional[list[SpecialistAgent]] = None
) -> list[DesignIssue]:
    """Run each specialist and collect the DesignIssues they file."""
    issues: list[DesignIssue] = []
    for agent in agents or DEFAULT_SPECIALISTS:
        issues.extend(agent.assess(state))
    return issues


def file_specialist_issues(
    state: DesignState, agents: Optional[list[SpecialistAgent]] = None
) -> DesignState:
    """Run the specialists and fold their issues into the state's open_issues."""
    for issue in run_specialists(state, agents):
        add_issue(state, issue)
    return state


def specialist_registry() -> dict[str, SpecialistAgent]:
    """Name → agent, for the dispatcher's SPECIALIST_RETASK handling (S8)."""
    return {a.name: a for a in DEFAULT_SPECIALISTS}
