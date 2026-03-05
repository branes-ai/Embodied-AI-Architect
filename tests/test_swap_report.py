"""Tests for SWaP-C report generation (Phase 5)."""

from embodied_ai_architect.graphs.swap_report import (
    SWaPCMetric,
    SWaPCScorecard,
    assess_design_point,
    format_swap_report_text,
    generate_swap_report,
)

# ---------------------------------------------------------------------------
# SWaPCMetric
# ---------------------------------------------------------------------------


def test_metric_pass():
    m = SWaPCMetric(name="weight", value=100.0, unit="g", budget=200.0)
    assert m.name == "weight"


def test_metric_fields():
    m = SWaPCMetric(name="power", value=5.0, unit="W")
    assert m.verdict == "PASS"


# ---------------------------------------------------------------------------
# assess_design_point
# ---------------------------------------------------------------------------


def test_scorecard_all_pass():
    design = {
        "objectives": {
            "power_watts": 3.0,
            "latency_ms": 10.0,
            "area_mm2": 25.0,
            "cost_usd": 15.0,
            "weight_grams": 100.0,
            "volume_cm3": 50.0,
        },
        "design_params": {"package_type": "BGA", "cooling_type": "passive"},
    }
    constraints = {
        "max_power_watts": 10.0,
        "max_weight_grams": 500.0,
        "max_volume_cm3": 200.0,
    }
    sc = assess_design_point(design, constraints=constraints)
    assert sc.overall_verdict == "PASS"
    assert all(m.verdict == "PASS" for m in sc.metrics)


def test_scorecard_fail_weight():
    design = {
        "objectives": {
            "power_watts": 3.0,
            "latency_ms": 10.0,
            "area_mm2": 25.0,
            "cost_usd": 15.0,
            "weight_grams": 600.0,
            "volume_cm3": 50.0,
        },
    }
    constraints = {"max_weight_grams": 500.0}
    sc = assess_design_point(design, constraints=constraints)
    weight_metric = next(m for m in sc.metrics if m.name == "weight_grams")
    assert weight_metric.verdict == "FAIL"
    assert weight_metric.utilization_pct == 120.0
    assert weight_metric.margin == -100.0
    assert sc.overall_verdict == "FAIL"


def test_scorecard_warning():
    design = {
        "objectives": {
            "power_watts": 8.5,
            "latency_ms": 10.0,
            "area_mm2": 25.0,
            "cost_usd": 15.0,
            "weight_grams": 100.0,
            "volume_cm3": 50.0,
        },
    }
    constraints = {"max_power_watts": 10.0}  # 85% utilization -> WARNING
    sc = assess_design_point(design, constraints=constraints)
    power_metric = next(m for m in sc.metrics if m.name == "power_watts")
    assert power_metric.verdict == "WARNING"
    assert sc.overall_verdict == "MARGINAL"


def test_scorecard_thermal_fail():
    design = {
        "objectives": {
            "power_watts": 3.0,
            "latency_ms": 10.0,
            "area_mm2": 25.0,
            "cost_usd": 15.0,
            "weight_grams": 100.0,
            "volume_cm3": 50.0,
        },
        "design_params": {"cooling_type": "passive"},
    }
    thermal = {
        "feasible": False,
        "junction_temp_c": 150.0,
        "max_junction_temp_c": 125.0,
        "ambient_temp_c": 60.0,
        "total_theta_c_per_w": 15.0,
        "tdp_watts": 10.0,
    }
    sc = assess_design_point(design, thermal_data=thermal)
    assert sc.thermal.feasible is False
    assert sc.overall_verdict == "FAIL"


def test_scorecard_bom_summary():
    bom = [{"name": "die", "level": "die", "weight_grams": 0.1}]
    design = {"objectives": {"power_watts": 3.0}}
    sc = assess_design_point(design, bom_summary=bom)
    assert sc.bom_summary == bom


def test_scorecard_pareto_position():
    design = {"objectives": {"power_watts": 3.0}}
    sc = assess_design_point(design, pareto_position="knee")
    assert sc.pareto_position == "knee"


# ---------------------------------------------------------------------------
# Budget utilization
# ---------------------------------------------------------------------------


def test_utilization_correct():
    design = {
        "objectives": {
            "power_watts": 5.0,
            "latency_ms": 10.0,
            "area_mm2": 25.0,
            "cost_usd": 15.0,
            "weight_grams": 250.0,
            "volume_cm3": 100.0,
        },
    }
    constraints = {"max_weight_grams": 500.0}
    sc = assess_design_point(design, constraints=constraints)
    weight_metric = next(m for m in sc.metrics if m.name == "weight_grams")
    assert weight_metric.utilization_pct == 50.0
    assert weight_metric.margin == 250.0


# ---------------------------------------------------------------------------
# generate_swap_report
# ---------------------------------------------------------------------------


def test_generate_report_empty():
    result = generate_swap_report({"pareto_front": []})
    assert result == []


def test_generate_report_basic():
    opt_result = {
        "pareto_front": [
            {
                "objectives": {
                    "power_watts": 3.0,
                    "weight_grams": 100.0,
                },
                "design_params": {"package_type": "BGA"},
                "metadata": {},
            },
        ],
        "knee_point": None,
    }
    report = generate_swap_report(opt_result)
    assert len(report) == 1
    assert isinstance(report[0], SWaPCScorecard)


def test_generate_report_marks_knee():
    params = {"package_type": "BGA"}
    opt_result = {
        "pareto_front": [
            {
                "objectives": {"power_watts": 3.0},
                "design_params": params,
                "metadata": {},
            },
        ],
        "knee_point": {
            "objectives": {"power_watts": 3.0},
            "design_params": params,
        },
    }
    report = generate_swap_report(opt_result)
    assert report[0].pareto_position == "knee"


# ---------------------------------------------------------------------------
# format_swap_report_text
# ---------------------------------------------------------------------------


def test_format_empty():
    text = format_swap_report_text([])
    assert "No designs" in text


def test_format_basic():
    sc = SWaPCScorecard(
        design_name="test",
        metrics=[
            SWaPCMetric(name="weight_grams", value=100.0, unit="g", budget=500.0),
        ],
        overall_verdict="PASS",
        pareto_position="non-dominated",
    )
    text = format_swap_report_text([sc])
    assert "SWaP-C Assessment Report" in text
    assert "test" in text
    assert "PASS" in text
    assert "weight_grams" in text
