"""Tests for RTL specialist agents (rtl_generator, rtl_ppa_assessor)."""

from __future__ import annotations


from embodied_ai_architect.graphs.rtl_specialists import rtl_generator, rtl_ppa_assessor
from embodied_ai_architect.graphs.soc_state import create_initial_soc_state, DesignConstraints
from embodied_ai_architect.graphs.task_graph import TaskNode
from embodied_ai_architect.graphs.kpu_config import create_kpu_config


def _make_task(agent: str = "rtl_generator") -> TaskNode:
    """Create a minimal TaskNode for testing."""
    return TaskNode(id="t1", name="Test", agent=agent)


# ---------------------------------------------------------------------------
# rtl_generator tests
# ---------------------------------------------------------------------------


def test_rtl_generator_skips_when_disabled():
    """rtl_generator returns SKIP verdict when rtl_enabled is False."""
    state = create_initial_soc_state(
        goal="Test",
        constraints=DesignConstraints(max_area_mm2=100.0),
        use_case="delivery_drone",
        rtl_enabled=False,
    )
    task = _make_task("rtl_generator")
    result = rtl_generator(task, state)

    assert result["verdict"] == "SKIP"
    assert "skipped" in result["summary"].lower() or "rtl_enabled" in result["summary"]


def test_rtl_generator_skips_no_kpu_config():
    """rtl_generator returns SKIP when rtl_enabled=True but kpu_config is empty."""
    state = create_initial_soc_state(
        goal="Test",
        constraints=DesignConstraints(max_area_mm2=100.0),
        use_case="delivery_drone",
        rtl_enabled=True,
    )
    # kpu_config defaults to {} in create_initial_soc_state
    assert state["kpu_config"] == {}

    task = _make_task("rtl_generator")
    result = rtl_generator(task, state)

    assert result["verdict"] == "SKIP"
    assert "kpu" in result["summary"].lower() or "No KPU" in result["summary"]


def test_rtl_generator_produces_modules():
    """rtl_generator with a valid kpu_config produces RTL modules and synthesis results."""
    state = create_initial_soc_state(
        goal="Test",
        constraints=DesignConstraints(max_area_mm2=100.0),
        use_case="delivery_drone",
        rtl_enabled=True,
    )
    config = create_kpu_config("delivery_drone", {"max_area_mm2": 100.0}, {"gflops": 4.0})
    state["kpu_config"] = config.model_dump()

    task = _make_task("rtl_generator")
    result = rtl_generator(task, state)

    # Should have produced multiple RTL modules
    assert "rtl_modules" in result
    assert isinstance(result["rtl_modules"], list)
    assert len(result["rtl_modules"]) > 0

    # State updates should include rtl_modules and rtl_synthesis_results
    updates = result["_state_updates"]
    assert "rtl_modules" in updates
    assert "rtl_synthesis_results" in updates
    assert len(updates["rtl_modules"]) > 0


def test_rtl_generator_summary():
    """rtl_generator summary string contains the component count."""
    state = create_initial_soc_state(
        goal="Test",
        constraints=DesignConstraints(max_area_mm2=100.0),
        use_case="delivery_drone",
        rtl_enabled=True,
    )
    config = create_kpu_config("delivery_drone", {"max_area_mm2": 100.0}, {"gflops": 4.0})
    state["kpu_config"] = config.model_dump()

    task = _make_task("rtl_generator")
    result = rtl_generator(task, state)

    # Summary should mention how many components were generated
    num_modules = len(result["rtl_modules"])
    assert str(num_modules) in result["summary"]
    assert "components" in result["summary"].lower()


# ---------------------------------------------------------------------------
# rtl_ppa_assessor tests
# ---------------------------------------------------------------------------


def test_rtl_ppa_assessor_skips_when_disabled():
    """rtl_ppa_assessor returns SKIP verdict when rtl_enabled is False."""
    state = create_initial_soc_state(
        goal="Test",
        constraints=DesignConstraints(max_area_mm2=100.0),
        use_case="delivery_drone",
        rtl_enabled=False,
    )
    task = _make_task("rtl_ppa_assessor")
    result = rtl_ppa_assessor(task, state)

    assert result["verdict"] == "SKIP"
    assert "skipped" in result["summary"].lower() or "rtl_enabled" in result["summary"]


def test_rtl_ppa_assessor_aggregates():
    """rtl_ppa_assessor aggregates synthesis results into total_cells and total_area_mm2."""
    state = create_initial_soc_state(
        goal="Test",
        constraints=DesignConstraints(max_area_mm2=100.0),
        use_case="delivery_drone",
        rtl_enabled=True,
    )
    state["rtl_synthesis_results"] = {
        "mac_unit": {"success": True, "area_cells": 50000, "area_um2": 10000000.0},
        "l1_skew_buffer": {"success": True, "area_cells": 30000, "area_um2": 6000000.0},
    }
    state["kpu_config"] = {"process_nm": 28}

    task = _make_task("rtl_ppa_assessor")
    result = rtl_ppa_assessor(task, state)

    # Total cells should be 50000 + 30000 = 80000
    assert result["total_cells"] == 80000

    assert "total_area_mm2" in result
    assert result["total_area_mm2"] > 0

    # Module-level breakdown should be present
    assert "module_areas" in result
    assert "mac_unit" in result["module_areas"]
    assert "l1_skew_buffer" in result["module_areas"]

    # PPA metrics should be updated in state_updates
    ppa = result["_state_updates"]["ppa_metrics"]
    assert "area_mm2" in ppa
    assert ppa["area_mm2"] > 0
