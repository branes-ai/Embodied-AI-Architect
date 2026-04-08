"""Pydantic response models for the REST API.

Typed schemas for every derived endpoint. These serve double duty:
1. FastAPI uses them for response validation and OpenAPI doc generation
2. The branes-frontend generates TypeScript types from the OpenAPI spec
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str = "ok"
    session_dir: str
    session_count: int


# ---------------------------------------------------------------------------
# Session list
# ---------------------------------------------------------------------------


class SessionSummary(BaseModel):
    session_id: str
    goal: str = ""
    status: str = "unknown"
    iteration: int = 0
    max_iterations: int = 20
    platform: str = ""
    use_case: str = ""
    created_at: str = ""
    saved_at: str = ""


# ---------------------------------------------------------------------------
# Pareto
# ---------------------------------------------------------------------------


class ParetoPoint(BaseModel):
    power: Optional[float] = None
    latency: Optional[float] = None
    cost: Optional[float] = None
    area: Optional[float] = None
    hardware: str = ""
    dominated: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class ParetoHistoryEntry(BaseModel):
    """One snapshot of the accumulated Pareto frontier after a MOO run."""

    iteration: int
    num_points: int
    hypervolume: float
    new_points_added: int = 0
    dominated_removed: int = 0


class ParetoResponse(BaseModel):
    points: list[ParetoPoint] = Field(default_factory=list)
    front: list[int] = Field(
        default_factory=list,
        description="Indices of non-dominated points",
    )
    knee_point_index: Optional[int] = None
    objectives: list[str] = Field(default_factory=lambda: ["power_watts", "latency_ms", "cost_usd"])
    pareto_front_size: int = 0
    hypervolume: Optional[float] = None
    frontier_history: list[ParetoHistoryEntry] = Field(
        default_factory=list,
        description="Per-iteration frontier evolution snapshots (issue #23)",
    )


# ---------------------------------------------------------------------------
# Sensitivity (issue #24)
# ---------------------------------------------------------------------------


class SensitivityEntry(BaseModel):
    """Per-variable impact across all objectives."""

    variable: str
    impacts: dict[str, float] = Field(
        default_factory=dict,
        description="objective_name -> impact_score (0-1)",
    )
    max_impact: float = Field(default=0.0, description="Max impact across all objectives")


class SensitivityResponse(BaseModel):
    """Response for the /sessions/{id}/sensitivity endpoint.

    Variables are sorted by max_impact descending so the architect sees the
    most influential design knobs first.
    """

    entries: list[SensitivityEntry] = Field(default_factory=list)
    objectives: list[str] = Field(
        default_factory=list,
        description="Objective names (column order for the impact table)",
    )


# ---------------------------------------------------------------------------
# Constraint slackness
# ---------------------------------------------------------------------------


class ConstraintSlackness(BaseModel):
    name: str
    target: Optional[float] = None
    actual: Optional[float] = None
    unit: str = ""
    verdict: str = "UNKNOWN"
    margin_pct: Optional[float] = Field(
        default=None,
        description="Positive = slack (within budget), negative = violated",
    )
    binding: bool = Field(
        default=False,
        description="True if margin < 5%",
    )
    trend: str = Field(
        default="stable",
        description="improving / worsening / stable",
    )


# ---------------------------------------------------------------------------
# KPU inner-loop slackness (issue #30)
# ---------------------------------------------------------------------------


class KPUFloorplanSlackness(BaseModel):
    """Floorplan slackness for the KPU inner loop."""

    compute_tile_width_mm: float = 0.0
    compute_tile_height_mm: float = 0.0
    memory_tile_width_mm: float = 0.0
    memory_tile_height_mm: float = 0.0
    pitch_ratio_width: float = 1.0
    pitch_ratio_height: float = 1.0
    pitch_tolerance: float = 0.15
    pitch_matched: bool = True
    total_area_mm2: float = 0.0
    max_die_area_mm2: float = 0.0
    area_utilization_pct: float = 0.0
    feasible: bool = True
    issues: list[str] = Field(default_factory=list)


class KPUBandwidthLink(BaseModel):
    """One link in the KPU bandwidth waterfall."""

    name: str
    source: str = ""
    sink: str = ""
    required_gbps: float = 0.0
    available_gbps: float = 0.0
    utilization_pct: float = 0.0
    bottleneck: bool = False
    status: str = "OK"


class KPUBandwidthSlackness(BaseModel):
    """Bandwidth slackness over the KPU memory hierarchy."""

    links: list[KPUBandwidthLink] = Field(default_factory=list)
    balanced: bool = True
    bottleneck_link: Optional[str] = None
    peak_utilization_pct: float = 0.0
    compute_demand_gbps: float = 0.0
    issues: list[str] = Field(default_factory=list)


class KPUSlacknessResponse(BaseModel):
    """Response for /api/sessions/{id}/kpu-slackness — both floorplan and bandwidth."""

    floorplan: Optional[KPUFloorplanSlackness] = None
    bandwidth: Optional[KPUBandwidthSlackness] = None


class KPUResponse(BaseModel):
    """Response for /api/sessions/{id}/kpu — full KPU snapshot for /architect-drill (issue #33).

    Combines the raw `kpu_config` (the architect's design knobs) with the
    derived floorplan and bandwidth slackness from issue #30. Each field is
    Optional so non-RTL sessions return a valid response with nulls instead
    of a 404.
    """

    config: Optional[dict] = Field(
        default=None,
        description="Full KPUMicroArchConfig dump (compute_tile, memory_tile, noc, dram, ...)",
    )
    floorplan: Optional[KPUFloorplanSlackness] = None
    bandwidth: Optional[KPUBandwidthSlackness] = None


# ---------------------------------------------------------------------------
# Optimization trajectory
# ---------------------------------------------------------------------------


class PPASnapshot(BaseModel):
    power_watts: Optional[float] = None
    latency_ms: Optional[float] = None
    area_mm2: Optional[float] = None
    cost_usd: Optional[float] = None


class TrajectoryEntry(BaseModel):
    iteration: int
    ppa_snapshot: PPASnapshot = Field(default_factory=PPASnapshot)
    verdicts: dict[str, str] = Field(default_factory=dict)
    strategy_applied: Optional[str] = None
    timestamp: Optional[str] = None


# ---------------------------------------------------------------------------
# Task graph
# ---------------------------------------------------------------------------


class TaskGraphNode(BaseModel):
    id: str
    name: str = ""
    agent: str = ""
    status: str = "pending"
    dependencies: list[str] = Field(default_factory=list)


class TaskGraphResponse(BaseModel):
    nodes: list[TaskGraphNode] = Field(default_factory=list)
    execution_order: list[str] = Field(default_factory=list)
    parallel_groups: list[list[str]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Workload
# ---------------------------------------------------------------------------


class OperatorBreakdown(BaseModel):
    name: str = "unknown"
    model_class: str = ""
    gflops: Optional[float] = None
    memory_mb: Optional[float] = None
    latency_ms: Optional[float] = None
    mapped_to: Optional[str] = Field(
        default=None,
        description="Hardware block this operator is mapped to (kpu, gpu, cpu)",
    )
    bound: Optional[str] = Field(
        default=None,
        description="What limits this operator: compute, memory, io",
    )
    scheduling: str = ""


class WorkloadResponse(BaseModel):
    operators: list[OperatorBreakdown] = Field(default_factory=list)
    total_gflops: Optional[float] = None
    total_memory_mb: Optional[float] = None
    dominant_op: str = ""
    source: str = ""
