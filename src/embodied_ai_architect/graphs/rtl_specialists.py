"""RTL specialist agents for template-based RTL generation and PPA assessment.

- rtl_generator: Generate RTL from KPU config via templates + EDA toolchain
- rtl_ppa_assessor: Aggregate synthesis metrics and refine area estimates

Usage:
    from embodied_ai_architect.graphs.rtl_specialists import rtl_generator
    result = rtl_generator(task, state)
"""

from __future__ import annotations

import logging
from typing import Any

from embodied_ai_architect.graphs.soc_state import (
    SoCDesignState,
    get_constraints,
)
from embodied_ai_architect.graphs.task_graph import TaskNode

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RTL Generator
# ---------------------------------------------------------------------------


def rtl_generator(task: TaskNode, state: SoCDesignState) -> dict[str, Any]:
    """Generate RTL for KPU sub-components from templates.

    Skips if rtl_enabled is False.
    Reads kpu_config from state, renders templates, runs lint + synthesis.

    Writes to state: rtl_modules, rtl_testbenches, rtl_synthesis_results,
                     rtl_lint_results, rtl_validation_results
    """
    from embodied_ai_architect.graphs.kpu_config import KPUMicroArchConfig
    from embodied_ai_architect.graphs.rtl_loop import RTLLoopConfig, run_rtl_loop
    from embodied_ai_architect.graphs.rtl_templates import RTLTemplateEngine

    if not state.get("rtl_enabled", False):
        return {
            "summary": "RTL generation skipped (rtl_enabled=False)",
            "verdict": "SKIP",
        }

    kpu_dict = state.get("kpu_config", {})
    if not kpu_dict:
        return {
            "summary": "No KPU config — skipping RTL generation",
            "verdict": "SKIP",
        }

    config = KPUMicroArchConfig(**kpu_dict)
    process_nm = config.process_nm

    engine = RTLTemplateEngine()
    components = engine.get_kpu_components(config)

    rtl_modules: dict[str, str] = {}
    rtl_testbenches: dict[str, str] = {}
    rtl_synthesis_results: dict[str, dict] = {}
    rtl_lint_results: dict[str, dict] = {}
    rtl_validation_results: dict[str, dict] = {}

    loop_config = RTLLoopConfig(
        process_nm=process_nm,
        max_iterations=1,  # Phase 3: single iteration
        skip_validation=True,  # Skip simulation for speed
    )

    for comp in components:
        comp_type = comp["component_type"]
        module_name = comp["module_name"]
        params = comp["params"]

        try:
            # Render RTL
            rtl_source = engine.render(comp_type, params)
            rtl_modules[module_name] = rtl_source

            # Render testbench
            try:
                tb_source = engine.render_testbench(comp_type, module_name, params)
                rtl_testbenches[module_name] = tb_source
            except Exception:
                tb_source = None

            # Run through EDA loop
            result = run_rtl_loop(module_name, rtl_source, loop_config, testbench_source=tb_source)

            rtl_lint_results[module_name] = result.lint_result
            rtl_synthesis_results[module_name] = result.synthesis_result
            if result.validation_result:
                rtl_validation_results[module_name] = result.validation_result

            logger.info(
                "RTL %s: %s (%d cells)",
                module_name,
                "PASS" if result.success else "FAIL",
                result.metrics.get("area_cells", 0),
            )

        except Exception as e:
            logger.warning("Failed to generate RTL for %s: %s", comp_type, e)
            rtl_lint_results[module_name] = {"success": False, "errors": [str(e)]}

    # Summary
    total_cells = sum(
        r.get("area_cells", 0) for r in rtl_synthesis_results.values() if r.get("success")
    )
    pass_count = sum(1 for r in rtl_synthesis_results.values() if r.get("success"))

    state_updates = {
        "rtl_modules": rtl_modules,
        "rtl_testbenches": rtl_testbenches,
        "rtl_synthesis_results": rtl_synthesis_results,
        "rtl_lint_results": rtl_lint_results,
        "rtl_validation_results": rtl_validation_results,
    }

    return {
        "summary": (
            f"Generated RTL for {len(components)} components: "
            f"{pass_count} passed, {len(components) - pass_count} failed, "
            f"{total_cells} total cells"
        ),
        "rtl_modules": list(rtl_modules.keys()),
        "total_cells": total_cells,
        "_state_updates": state_updates,
    }


# ---------------------------------------------------------------------------
# RTL PPA Assessor
# ---------------------------------------------------------------------------


def rtl_ppa_assessor(task: TaskNode, state: SoCDesignState) -> dict[str, Any]:
    """Aggregate synthesis metrics and refine PPA area estimate.

    Reads rtl_synthesis_results from state, sums cell counts,
    converts to area using technology.py, cross-checks floorplan.

    Writes to state: updated ppa_metrics.area_mm2
    """
    from embodied_ai_architect.graphs.technology import estimate_area_um2

    if not state.get("rtl_enabled", False):
        return {
            "summary": "RTL PPA assessment skipped (rtl_enabled=False)",
            "verdict": "SKIP",
        }

    synth_results = state.get("rtl_synthesis_results", {})
    if not synth_results:
        return {
            "summary": "No synthesis results — skipping RTL PPA assessment",
            "verdict": "SKIP",
        }

    kpu_dict = state.get("kpu_config", {})
    process_nm = kpu_dict.get("process_nm", 28)

    # Aggregate cell counts
    total_cells = 0
    module_areas: dict[str, dict] = {}
    for module_name, result in synth_results.items():
        if result.get("success"):
            cells = result.get("area_cells", 0)
            total_cells += cells
            area_um2 = result.get("area_um2", 0.0)
            module_areas[module_name] = {
                "cells": cells,
                "area_um2": area_um2,
                "area_mm2": area_um2 / 1e6,
            }

    # Convert total to mm²
    total_area_um2 = estimate_area_um2(total_cells, process_nm)
    total_area_mm2 = total_area_um2 / 1e6

    # Use floorplan area as primary estimate (includes SRAM macros, routing
    # overhead, and periphery which dominate die area).  Synthesis cell counts
    # cover only combinational/sequential logic — a small fraction of total.
    fp = state.get("floorplan_estimate", {})
    floorplan_area = fp.get("total_area_mm2", 0)

    if floorplan_area > 0:
        effective_area = floorplan_area
        cross_check = (
            f" (floorplan: {floorplan_area:.1f}mm², " f"synthesis logic: {total_area_mm2:.3f}mm²)"
        )
    else:
        effective_area = total_area_mm2
        cross_check = ""

    # Update PPA metrics
    ppa = dict(state.get("ppa_metrics", {}))
    ppa["area_mm2"] = round(effective_area, 2)

    constraints = get_constraints(state)
    if constraints.max_area_mm2:
        if effective_area <= constraints.max_area_mm2:
            ppa.setdefault("verdicts", {})["area"] = "PASS"
        else:
            ppa.setdefault("verdicts", {})["area"] = "FAIL"

    return {
        "summary": (
            f"RTL PPA area: {effective_area:.1f}mm² "
            f"({total_cells} cells at {process_nm}nm){cross_check}"
        ),
        "total_cells": total_cells,
        "total_area_mm2": effective_area,
        "module_areas": module_areas,
        "_state_updates": {"ppa_metrics": ppa},
    }
