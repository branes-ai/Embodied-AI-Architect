"""RTL inner loop: lint → synthesize → validate per RTL module.

Plain Python loop that processes RTL modules through the EDA toolchain.
Phase 3 runs one iteration per module (template-based, no LLM).

Usage:
    from embodied_ai_architect.graphs.rtl_loop import run_rtl_loop, RTLLoopConfig

    result = run_rtl_loop("mac_unit", rtl_source, RTLLoopConfig(process_nm=28))
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class RTLLoopConfig:
    """Configuration for the RTL inner loop.

    Args:
        max_iterations: Maximum lint→synth→validate cycles.
        process_nm: Technology node for area scaling.
        work_dir: Working directory for intermediate files.
        skip_validation: Skip simulation step.
        synth_timeout: Yosys timeout in seconds.
            -1 (default) estimates from RTL complexity.
    """

    max_iterations: int = 5
    process_nm: int = 28
    work_dir: Optional[Path] = None
    skip_validation: bool = False
    synth_timeout: int = -1


@dataclass
class RTLLoopResult:
    """Result of processing one RTL module."""

    module_name: str
    success: bool
    rtl_source: str = ""
    testbench_source: Optional[str] = None
    lint_result: dict = field(default_factory=dict)
    synthesis_result: dict = field(default_factory=dict)
    validation_result: Optional[dict] = None
    metrics: dict = field(default_factory=dict)
    iterations_used: int = 1
    history: list[dict] = field(default_factory=list)


def run_rtl_loop(
    module_name: str,
    rtl_source: str,
    config: Optional[RTLLoopConfig] = None,
    testbench_source: Optional[str] = None,
) -> RTLLoopResult:
    """Process an RTL module: lint → synthesize → [validate].

    Phase 3 runs a single iteration (no LLM-based optimization).
    The iteration hook structure is for Phase 4 LLM-based RTL optimization.

    Args:
        module_name: Name of the RTL module.
        rtl_source: SystemVerilog source code.
        config: Loop configuration.
        testbench_source: Optional testbench source for validation.

    Returns:
        RTLLoopResult with lint, synthesis, and validation results.
    """
    from embodied_ai_architect.graphs.eda_tools import EDAToolchain

    if config is None:
        config = RTLLoopConfig()

    work_dir = config.work_dir or Path(tempfile.mkdtemp(prefix=f"rtl_{module_name}_"))
    toolchain = EDAToolchain(
        work_dir=work_dir,
        process_nm=config.process_nm,
        synth_timeout=config.synth_timeout,
    )

    history: list[dict] = []

    for iteration in range(config.max_iterations):
        logger.info("RTL loop iteration %d for %s", iteration, module_name)

        # Step 1: Lint
        lint_result = toolchain.lint(rtl_source)
        lint_ok = lint_result.get("success", False)

        if not lint_ok:
            history.append(
                {
                    "iteration": iteration,
                    "stage": "lint",
                    "success": False,
                    "errors": lint_result.get("errors", []),
                }
            )
            # In Phase 3, no auto-fix — just report failure
            return RTLLoopResult(
                module_name=module_name,
                success=False,
                rtl_source=rtl_source,
                testbench_source=testbench_source,
                lint_result=lint_result,
                iterations_used=iteration + 1,
                history=history,
            )

        # Step 2: Synthesize
        synth_result = toolchain.synthesize(rtl_source, module_name)
        synth_ok = synth_result.get("success", False)

        if not synth_ok:
            history.append(
                {
                    "iteration": iteration,
                    "stage": "synthesis",
                    "success": False,
                    "errors": synth_result.get("errors", []),
                }
            )
            return RTLLoopResult(
                module_name=module_name,
                success=False,
                rtl_source=rtl_source,
                testbench_source=testbench_source,
                lint_result=lint_result,
                synthesis_result=synth_result,
                iterations_used=iteration + 1,
                history=history,
            )

        # Step 3: Validate (simulation) if testbench provided
        validation_result = None
        if testbench_source and not config.skip_validation:
            validation_result = toolchain.simulate(rtl_source, testbench_source, module_name)

        # Collect metrics
        metrics = {
            "area_cells": synth_result.get("area_cells", 0),
            "area_um2": synth_result.get("area_um2", 0.0),
            "num_wires": synth_result.get("num_wires", 0),
            "num_cells": synth_result.get("num_cells", 0),
        }

        history.append(
            {
                "iteration": iteration,
                "stage": "complete",
                "success": True,
                "metrics": metrics,
            }
        )

        # Phase 3: single iteration, no LLM-based optimization
        return RTLLoopResult(
            module_name=module_name,
            success=True,
            rtl_source=rtl_source,
            testbench_source=testbench_source,
            lint_result=lint_result,
            synthesis_result=synth_result,
            validation_result=validation_result,
            metrics=metrics,
            iterations_used=iteration + 1,
            history=history,
        )

    # Should not reach here in Phase 3 (single iteration)
    return RTLLoopResult(
        module_name=module_name,
        success=False,
        rtl_source=rtl_source,
        iterations_used=config.max_iterations,
        history=history,
    )
