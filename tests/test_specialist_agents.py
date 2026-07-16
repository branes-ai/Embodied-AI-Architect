"""Seam S9 (issue #213): specialist agents reason over the shared DesignState via
verdict-first estimator tools and file DesignIssues."""

from embodied_ai_architect.graphs.design_state import (
    DesignConstraints,
    DesignState,
    MetricAxis,
    Severity,
    open_issues,
)
from embodied_ai_architect.graphs.physical_estimators import estimate_junction_temperature
from embodied_ai_architect.graphs.specialist_agents import (
    PPASpecialist,
    ThermalSpecialist,
    file_specialist_issues,
    power_tool,
    run_specialists,
    specialist_registry,
    thermal_tool,
)


def _state(**ppa: float) -> DesignState:
    return {
        "constraints": DesignConstraints(
            max_power_watts=5.0, max_latency_ms=33.0, max_area_mm2=50.0, max_cost_usd=30.0
        ).model_dump(),
        "ppa_metrics": {k: v for k, v in ppa.items()},
        "iteration": 1,
    }


# ---------------------------------------------------------------------------
# Verdict-first tools
# ---------------------------------------------------------------------------


def test_power_tool_is_verdict_first() -> None:
    over = power_tool(_state(power_watts=6.0))
    assert over.verdict == "FAIL"
    assert over.margin == -1.0
    assert round(over.overshoot_pct) == 20
    assert power_tool(_state(power_watts=4.0)).verdict == "PASS"
    assert power_tool(_state()).verdict == "UNKNOWN"  # no estimate


def test_tools_fall_back_to_knee_point() -> None:
    state = _state()
    state["knee_point"] = {"objectives": {"power_watts": 7.0}}
    assert power_tool(state).verdict == "FAIL"


# ---------------------------------------------------------------------------
# PPA specialist
# ---------------------------------------------------------------------------


def test_ppa_specialist_files_dominant_bottleneck_first() -> None:
    # power 60% over budget, latency 21% over -> both filed, power first, severity scaled.
    issues = PPASpecialist().assess(_state(power_watts=8.0, latency_ms=40.0))
    assert [i.metric for i in issues] == [MetricAxis.POWER, MetricAxis.LATENCY]
    assert issues[0].severity == Severity.CRITICAL  # 60% over
    assert issues[1].severity == Severity.HIGH  # 21% over
    assert issues[0].raised_by == "ppa_specialist"
    assert issues[0].observed_value == 8.0 and issues[0].target_value == 5.0


def test_ppa_specialist_silent_when_within_budget() -> None:
    assert PPASpecialist().assess(_state(power_watts=4.0, latency_ms=20.0)) == []


# ---------------------------------------------------------------------------
# Thermal specialist (uses the physical_estimators junction-temp tool)
# ---------------------------------------------------------------------------


def test_thermal_specialist_flags_overheating_design() -> None:
    issues = ThermalSpecialist().assess(_state(power_watts=20.0))  # 40 + 20*6 = 160C
    assert len(issues) == 1
    assert issues[0].metric == MetricAxis.THERMAL
    assert issues[0].observed_value == estimate_junction_temperature(20.0, 6.0, 40.0)
    assert issues[0].raised_by == "thermal_specialist"


def test_thermal_specialist_silent_when_cool() -> None:
    assert ThermalSpecialist().assess(_state(power_watts=5.0)) == []  # 70C < 125C


def test_thermal_tool_uses_estimator() -> None:
    r = thermal_tool(_state(power_watts=10.0), theta_c_per_w=6.0, ambient_temp_c=40.0)
    assert r.value == estimate_junction_temperature(10.0, 6.0, 40.0)


# ---------------------------------------------------------------------------
# Acceptance: >= 2 specialists reason via tools and file issues into the backlog
# ---------------------------------------------------------------------------


def test_two_specialists_file_issues_into_open_issues() -> None:
    state = _state(power_watts=20.0)  # over power budget AND thermally infeasible
    file_specialist_issues(state)
    filed = open_issues(state)
    raisers = {i.raised_by for i in filed}
    assert "ppa_specialist" in raisers
    assert "thermal_specialist" in raisers
    assert {MetricAxis.POWER, MetricAxis.THERMAL}.issubset({i.metric for i in filed})


def test_run_specialists_returns_issues_without_mutating() -> None:
    state = _state(power_watts=20.0)
    issues = run_specialists(state)
    assert len(issues) >= 2
    assert "open_issues" not in state or not state["open_issues"]  # run_ doesn't file


def test_registry_exposes_named_specialists() -> None:
    reg = specialist_registry()
    assert set(reg) == {"ppa_specialist", "thermal_specialist"}
