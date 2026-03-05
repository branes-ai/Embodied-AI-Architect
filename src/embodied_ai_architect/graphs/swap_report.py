"""SWaP-C assessment report generation.

Produces scorecards and formatted reports for system-level SWaP-C analysis,
covering weight, volume, power, cost, and thermal feasibility.

Usage:
    from embodied_ai_architect.graphs.swap_report import (
        assess_design_point, generate_swap_report, format_swap_report_text,
    )

    scorecard = assess_design_point(design, bom, constraints)
    report = generate_swap_report(opt_result, constraints)
    print(format_swap_report_text(report))
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Report models
# ---------------------------------------------------------------------------


class SWaPCMetric(BaseModel):
    """Single SWaP-C metric with budget tracking."""

    name: str = Field(..., description="Metric name (e.g. weight_grams)")
    value: float = Field(..., description="Measured value")
    unit: str = Field(..., description="Unit of measurement")
    budget: Optional[float] = Field(default=None, description="Budget from constraints")
    utilization_pct: Optional[float] = Field(default=None, description="value / budget * 100")
    margin: Optional[float] = Field(default=None, description="budget - value")
    verdict: str = Field(default="PASS", description="PASS / FAIL / WARNING")


class ThermalAssessment(BaseModel):
    """Thermal feasibility assessment."""

    junction_temp_c: float = Field(default=0.0, description="Estimated junction temperature")
    max_junction_temp_c: float = Field(default=125.0, description="Maximum allowed junction temp")
    ambient_temp_c: float = Field(default=40.0, description="Ambient temperature")
    thermal_resistance_total: float = Field(
        default=0.0, description="Total thermal resistance (C/W)"
    )
    tdp_watts: float = Field(default=0.0, description="Thermal design power")
    cooling_type: str = Field(default="passive", description="Cooling solution type")
    feasible: bool = Field(default=True, description="Whether thermal constraints are met")
    derating_factor: float = Field(
        default=1.0, description="Power derating factor for sustained operation"
    )


class SWaPCScorecard(BaseModel):
    """Complete SWaP-C assessment for a single design point."""

    design_name: str = Field(default="", description="Design identifier")
    metrics: list[SWaPCMetric] = Field(default_factory=list, description="All SWaP-C metrics")
    thermal: ThermalAssessment = Field(
        default_factory=ThermalAssessment, description="Thermal assessment"
    )
    bom_summary: list[dict[str, Any]] = Field(
        default_factory=list, description="Flattened BOM table"
    )
    pareto_position: str = Field(default="unknown", description="non-dominated / dominated / knee")
    overall_verdict: str = Field(default="PASS", description="PASS / FAIL / MARGINAL")


# ---------------------------------------------------------------------------
# Assessment functions
# ---------------------------------------------------------------------------


def _make_metric(
    name: str,
    value: float,
    unit: str,
    budget: float | None = None,
) -> SWaPCMetric:
    """Build a SWaPCMetric with derived utilization, margin, and verdict."""
    utilization = None
    margin = None
    verdict = "PASS"

    if budget is not None and budget > 0:
        utilization = round((value / budget) * 100, 1)
        margin = round(budget - value, 2)
        if value > budget:
            verdict = "FAIL"
        elif utilization > 80:
            verdict = "WARNING"

    return SWaPCMetric(
        name=name,
        value=round(value, 3),
        unit=unit,
        budget=budget,
        utilization_pct=utilization,
        margin=margin,
        verdict=verdict,
    )


def assess_design_point(
    design: dict[str, Any],
    bom_summary: list[dict[str, Any]] | None = None,
    constraints: dict[str, Any] | None = None,
    thermal_data: dict[str, Any] | None = None,
    pareto_position: str = "unknown",
) -> SWaPCScorecard:
    """Build a SWaP-C scorecard for one design point.

    Args:
        design: Dict with 'objectives' and 'design_params' keys.
        bom_summary: Flattened BOM table from SystemBOM.summary_table().
        constraints: Constraint bounds (max_power_watts, max_weight_grams, etc.).
        thermal_data: Thermal feasibility dict from compute_thermal_feasibility().
        pareto_position: Position on Pareto front (non-dominated/dominated/knee).

    Returns:
        SWaPCScorecard with all metrics and verdicts.
    """
    constraints = constraints or {}
    objectives = design.get("objectives", design)
    params = design.get("design_params", design)

    metrics = [
        _make_metric(
            "power_watts",
            objectives.get("power_watts", 0.0),
            "W",
            constraints.get("max_power_watts"),
        ),
        _make_metric(
            "latency_ms",
            objectives.get("latency_ms", 0.0),
            "ms",
            constraints.get("max_latency_ms"),
        ),
        _make_metric(
            "area_mm2",
            objectives.get("area_mm2", 0.0),
            "mm²",
            constraints.get("max_area_mm2"),
        ),
        _make_metric(
            "cost_usd",
            objectives.get("cost_usd", 0.0),
            "USD",
            constraints.get("max_cost_usd"),
        ),
        _make_metric(
            "weight_grams",
            objectives.get("weight_grams", 0.0),
            "g",
            constraints.get("max_weight_grams"),
        ),
        _make_metric(
            "volume_cm3",
            objectives.get("volume_cm3", 0.0),
            "cm³",
            constraints.get("max_volume_cm3"),
        ),
    ]

    # Thermal assessment
    thermal = ThermalAssessment(
        junction_temp_c=0.0,
        max_junction_temp_c=125.0,
        ambient_temp_c=40.0,
        feasible=True,
    )
    if thermal_data:
        thermal = ThermalAssessment(
            junction_temp_c=thermal_data.get("junction_temp_c", 0.0),
            max_junction_temp_c=thermal_data.get("max_junction_temp_c", 125.0),
            ambient_temp_c=thermal_data.get("ambient_temp_c", 40.0),
            thermal_resistance_total=thermal_data.get("total_theta_c_per_w", 0.0),
            tdp_watts=thermal_data.get("tdp_watts", 0.0),
            cooling_type=params.get("cooling_type", "passive"),
            feasible=thermal_data.get("feasible", True),
        )

    # Overall verdict
    any_fail = any(m.verdict == "FAIL" for m in metrics)
    any_warning = any(m.verdict == "WARNING" for m in metrics)
    thermal_fail = not thermal.feasible

    if any_fail or thermal_fail:
        overall = "FAIL"
    elif any_warning:
        overall = "MARGINAL"
    else:
        overall = "PASS"

    return SWaPCScorecard(
        design_name=design.get("name", params.get("name", "")),
        metrics=metrics,
        thermal=thermal,
        bom_summary=bom_summary or [],
        pareto_position=pareto_position,
        overall_verdict=overall,
    )


def generate_swap_report(
    opt_result: dict[str, Any],
    constraints: dict[str, Any] | None = None,
) -> list[SWaPCScorecard]:
    """Generate scorecards for all designs on the Pareto front.

    Args:
        opt_result: OptimizationResult dict with pareto_front and knee_point.
        constraints: Constraint bounds dict.

    Returns:
        List of SWaPCScorecard, one per Pareto-front design.
    """
    constraints = constraints or {}
    front = opt_result.get("pareto_front", [])
    knee = opt_result.get("knee_point")
    knee_params = knee.get("design_params", {}) if knee else {}

    scorecards: list[SWaPCScorecard] = []
    for design in front:
        params = design.get("design_params", {})
        meta = design.get("metadata", {})

        # Determine Pareto position
        position = "non-dominated"
        if params == knee_params and knee_params:
            position = "knee"

        scorecard = assess_design_point(
            design=design,
            bom_summary=meta.get("bom_summary", []),
            constraints=constraints,
            thermal_data=meta.get("thermal"),
            pareto_position=position,
        )
        scorecards.append(scorecard)

    return scorecards


def format_swap_report_text(scorecards: list[SWaPCScorecard]) -> str:
    """Format scorecards as a CLI-friendly text table.

    Args:
        scorecards: List of SWaPCScorecard from generate_swap_report().

    Returns:
        Multi-line string with formatted table.
    """
    if not scorecards:
        return "No designs to report."

    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("SWaP-C Assessment Report")
    lines.append("=" * 78)

    for i, sc in enumerate(scorecards):
        lines.append("")
        label = sc.design_name or f"Design #{i + 1}"
        lines.append(f"--- {label} [{sc.pareto_position}] ---")
        lines.append(f"Overall: {sc.overall_verdict}")
        lines.append("")

        # Metrics table
        header = (
            f"{'Metric':<16} {'Value':>10} {'Unit':<6} {'Budget':>10} {'Util%':>8} {'Verdict':<8}"
        )
        lines.append(header)
        lines.append("-" * len(header))
        for m in sc.metrics:
            budget_str = f"{m.budget:.1f}" if m.budget is not None else "-"
            util_str = f"{m.utilization_pct:.0f}%" if m.utilization_pct is not None else "-"
            lines.append(
                f"{m.name:<16} {m.value:>10.2f} {m.unit:<6} "
                f"{budget_str:>10} {util_str:>8} {m.verdict:<8}"
            )

        # Thermal
        if sc.thermal.tdp_watts > 0:
            lines.append("")
            lines.append(
                f"Thermal: Tj={sc.thermal.junction_temp_c:.0f}C "
                f"(max {sc.thermal.max_junction_temp_c:.0f}C), "
                f"cooling={sc.thermal.cooling_type}, "
                f"{'PASS' if sc.thermal.feasible else 'FAIL'}"
            )

    lines.append("")
    lines.append("=" * 78)
    return "\n".join(lines)
