"""Evaluation data models for the agentic SoC design system.

Defines the core types used by the 9-dimension evaluation framework:
- RunTrace: captures a complete demo execution for scoring
- GoldStandard: expected behavior per demo for comparison
- DimensionScore: result for a single evaluation dimension
- Scorecard: composite result across all dimensions

Usage:
    from embodied_ai_architect.graphs.evaluation import (
        RunTrace, GoldStandard, DimensionScore, Scorecard,
    )
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RunTrace(BaseModel):
    """Captures a complete demo execution for evaluation scoring.

    Collected during or after a design session, this trace contains
    everything needed to score the agentic system's performance across
    all 9 evaluation dimensions.
    """

    demo_name: str = ""
    task_graph: dict[str, Any] = Field(
        default_factory=dict,
        description="Serialized TaskGraph from the run",
    )
    ppa_metrics: dict[str, Any] = Field(
        default_factory=dict,
        description="Final PPAMetrics from the run",
    )
    iteration_history: list[dict[str, Any]] = Field(
        default_factory=list,
        description="PPA snapshots per optimization iteration",
    )
    tool_calls: list[str] = Field(
        default_factory=list,
        description="Ordered list of agent names dispatched during the run",
    )
    audit_log: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Governance audit entries",
    )
    failures: int = Field(
        default=0,
        description="Number of task failures during the run",
    )
    recoveries: int = Field(
        default=0,
        description="Number of successful recoveries from failures",
    )
    duration_seconds: float = Field(
        default=0.0,
        description="Wall-clock duration of the run in seconds",
    )
    cost_tokens: int = Field(
        default=0,
        description="Total LLM tokens consumed",
    )
    human_interventions: int = Field(
        default=0,
        description="Number of times a human had to intervene",
    )
    design_rationale: list[str] = Field(
        default_factory=list,
        description="Rationale chain from the design session",
    )
    pareto_points: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Explored Pareto design points",
    )
    history: list[dict[str, Any]] = Field(
        default_factory=list,
        description="All DesignDecision entries",
    )


class GoldStandard(BaseModel):
    """Expected behavior for a demo, used as the scoring reference.

    Hand-crafted per demo to define what a "good" run looks like.
    """

    demo_name: str = ""
    expected_task_graph: dict[str, Any] = Field(
        default_factory=dict,
        description="Expected TaskGraph structure (node names and edges)",
    )
    expected_ppa: dict[str, Any] = Field(
        default_factory=dict,
        description="Expected PPA metric ranges (power_watts, latency_ms, etc.)",
    )
    governance_triggers: list[str] = Field(
        default_factory=list,
        description="Expected governance actions that should appear in the audit log",
    )
    expected_tool_calls: list[str] = Field(
        default_factory=list,
        description="Expected agent names that should be called during the run",
    )
    max_iterations: int = Field(
        default=10,
        description="Maximum acceptable optimization iterations",
    )
    max_duration_seconds: float = Field(
        default=60.0,
        description="Maximum acceptable wall-clock duration",
    )
    max_cost_tokens: int = Field(
        default=100000,
        description="Maximum acceptable token cost",
    )
    max_human_interventions: int = Field(
        default=3,
        description="Maximum acceptable human interventions",
    )
    expected_pareto_points: int = Field(
        default=0,
        description="Minimum expected Pareto front points (0 = no Pareto expected)",
    )
    rationale_keywords: list[str] = Field(
        default_factory=list,
        description="Keywords expected in design rationale for reasoning scoring",
    )


class DimensionScore(BaseModel):
    """Score for a single evaluation dimension."""

    dimension: str = Field(..., description="Name of the scored dimension")
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Score from 0.0 (worst) to 1.0 (best)",
    )
    weight: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Weight for composite scoring",
    )
    details: str = Field(
        default="",
        description="Human-readable explanation of the score",
    )


class Scorecard(BaseModel):
    """Composite evaluation result across all dimensions."""

    demo_name: str = ""
    dimensions: list[DimensionScore] = Field(default_factory=list)
    composite_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Weighted average of all dimension scores",
    )
    passed: bool = Field(
        default=False,
        description="True if composite_score >= passing threshold",
    )

    def summary(self) -> str:
        """Format a human-readable summary of the scorecard."""
        lines = [
            f"Scorecard: {self.demo_name} — {'PASS' if self.passed else 'FAIL'} ({self.composite_score:.2f})"
        ]
        for d in self.dimensions:
            bar = "#" * int(d.score * 10) + "." * (10 - int(d.score * 10))
            lines.append(f"  {d.dimension:25s} [{bar}] {d.score:.2f} (w={d.weight:.2f})")
        return "\n".join(lines)
