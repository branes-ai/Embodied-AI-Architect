"""Tests for SWaPCEvaluator and SWaP-C design space (Phase 3)."""

from embodied_ai_architect.graphs.moo.design_space import (
    create_soc_design_space,
    create_swap_design_space,
)
from embodied_ai_architect.graphs.moo.evaluator import (
    DesignEvaluator,
    EvaluationResult,
    SWaPCEvaluator,
)

# ---------------------------------------------------------------------------
# Design space factory
# ---------------------------------------------------------------------------


def test_swap_design_space_variables():
    ds = create_swap_design_space()
    names = [v.name for v in ds.variables]
    assert "package_type" in names
    assert "cooling_type" in names
    assert len(ds.variables) == 9  # 7 SoC + 2 SWaP-C


def test_swap_design_space_objectives():
    ds = create_swap_design_space()
    assert ds.n_obj == 6
    assert "weight_grams" in ds.objectives
    assert "volume_cm3" in ds.objectives


def test_swap_design_space_constraints():
    ds = create_swap_design_space({"max_weight_grams": 500, "max_volume_cm3": 200})
    assert "weight_grams" in ds.constraint_bounds
    assert "volume_cm3" in ds.constraint_bounds
    assert ds.constraint_bounds["weight_grams"] == (0.0, 500)
    assert ds.constraint_bounds["volume_cm3"] == (0.0, 200)


def test_swap_design_space_sampling():
    ds = create_swap_design_space()
    samples = ds.sample_random(5, seed=42)
    assert len(samples) == 5
    for s in samples:
        assert "package_type" in s
        assert "cooling_type" in s
        assert s["package_type"] in ["QFN", "BGA", "FCBGA", "WLCSP"]
        assert s["cooling_type"] in ["passive", "active_fan", "liquid"]


def test_swap_design_space_encode_decode():
    ds = create_swap_design_space()
    params = {
        "process_nm": 28,
        "clock_mhz": 1000.0,
        "array_rows": 8,
        "array_cols": 8,
        "sram_kb": 256,
        "num_compute_tiles": 4,
        "noc_link_width_bits": 256,
        "package_type": "BGA",
        "cooling_type": "passive",
    }
    encoded = ds.encode(params)
    decoded = ds.decode(encoded)
    assert decoded["package_type"] == "BGA"
    assert decoded["cooling_type"] == "passive"
    assert decoded["process_nm"] == 28


def test_soc_design_space_unchanged():
    """Existing 4-objective design space should not be affected."""
    ds = create_soc_design_space()
    assert ds.n_obj == 4
    assert "weight_grams" not in ds.objectives
    names = [v.name for v in ds.variables]
    assert "package_type" not in names


# ---------------------------------------------------------------------------
# SWaPCEvaluator
# ---------------------------------------------------------------------------


BASE_PARAMS = {
    "process_nm": 28,
    "clock_mhz": 1000.0,
    "array_rows": 8,
    "array_cols": 8,
    "sram_kb": 256,
    "num_compute_tiles": 4,
    "noc_link_width_bits": 256,
    "package_type": "BGA",
    "cooling_type": "passive",
}


def _make_evaluator(**kwargs):
    ds = create_swap_design_space(kwargs.get("constraints"))
    base_state = kwargs.get("base_state", {"workload_profile": {"total_estimated_gflops": 5.0}})
    thermal_config = kwargs.get("thermal_config")
    return SWaPCEvaluator(
        design_space=ds,
        base_state=base_state,
        thermal_config=thermal_config,
    )


def test_evaluator_returns_6_objectives():
    ev = _make_evaluator()
    result = ev.evaluate(BASE_PARAMS)
    assert isinstance(result, EvaluationResult)
    assert "weight_grams" in result.objectives
    assert "volume_cm3" in result.objectives
    assert "power_watts" in result.objectives
    assert "latency_ms" in result.objectives
    assert "area_mm2" in result.objectives
    assert "cost_usd" in result.objectives
    assert len(result.objectives) == 6


def test_first_4_objectives_match_base():
    """First 4 objectives should match what DesignEvaluator produces."""
    ds = create_swap_design_space()
    base_state = {"workload_profile": {"total_estimated_gflops": 5.0}}

    ppa_ev = DesignEvaluator(ds, base_state)
    swap_ev = SWaPCEvaluator(ds, base_state)

    ppa_result = ppa_ev.evaluate(BASE_PARAMS)
    swap_result = swap_ev.evaluate(BASE_PARAMS)

    for key in ["power_watts", "latency_ms", "area_mm2", "cost_usd"]:
        assert (
            swap_result.objectives[key] == ppa_result.objectives[key]
        ), f"Mismatch on {key}: {swap_result.objectives[key]} vs {ppa_result.objectives[key]}"


def test_package_type_affects_weight():
    ev = _make_evaluator()
    qfn_result = ev.evaluate({**BASE_PARAMS, "package_type": "QFN"})
    fcbga_result = ev.evaluate({**BASE_PARAMS, "package_type": "FCBGA"})
    assert fcbga_result.objectives["weight_grams"] != qfn_result.objectives["weight_grams"]


def test_cooling_type_affects_weight():
    ev = _make_evaluator()
    passive_result = ev.evaluate({**BASE_PARAMS, "cooling_type": "passive"})
    liquid_result = ev.evaluate({**BASE_PARAMS, "cooling_type": "liquid"})
    assert passive_result.objectives["weight_grams"] != liquid_result.objectives["weight_grams"]


def test_thermal_metadata_present():
    ev = _make_evaluator()
    result = ev.evaluate(BASE_PARAMS)
    assert "thermal" in result.metadata
    thermal = result.metadata["thermal"]
    assert "feasible" in thermal
    assert "junction_temp_c" in thermal


def test_constraint_enforcement_weight():
    ev = _make_evaluator(constraints={"max_weight_grams": 1.0})
    result = ev.evaluate(BASE_PARAMS)
    # System weight will exceed 1g, so should be infeasible
    assert result.objectives["weight_grams"] > 1.0
    assert result.constraints_satisfied.get("weight_grams") is False
    assert result.feasible is False


def test_constraint_enforcement_volume():
    ev = _make_evaluator(constraints={"max_volume_cm3": 0.001})
    result = ev.evaluate(BASE_PARAMS)
    assert result.objectives["volume_cm3"] > 0.001
    assert result.constraints_satisfied.get("volume_cm3") is False
    assert result.feasible is False


def test_bom_summary_in_metadata():
    ev = _make_evaluator()
    result = ev.evaluate(BASE_PARAMS)
    assert "bom_summary" in result.metadata
    assert isinstance(result.metadata["bom_summary"], list)
    assert len(result.metadata["bom_summary"]) > 0


def test_evaluate_batch():
    ev = _make_evaluator()
    params_list = [
        {**BASE_PARAMS, "package_type": "QFN"},
        {**BASE_PARAMS, "package_type": "BGA"},
    ]
    results = ev.evaluate_batch(params_list)
    assert len(results) == 2
    for r in results:
        assert len(r.objectives) == 6
