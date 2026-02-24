"""Thread-safe design evaluator wrapping existing PPA specialists.

Converts design parameters into SoC compositions and evaluates power,
latency, area, and cost using the physics-based models from ip_blocks.py,
manufacturing.py, and technology.py.

Usage:
    from embodied_ai_architect.graphs.moo.evaluator import DesignEvaluator

    evaluator = DesignEvaluator(design_space, base_state)
    result = evaluator.evaluate({"process_nm": 28, "clock_mhz": 1000, ...})
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from embodied_ai_architect.graphs.moo.design_space import DesignSpace

logger = logging.getLogger(__name__)


class EvaluationResult(BaseModel):
    """Result of evaluating a single design point."""

    design_params: dict[str, Any] = Field(default_factory=dict)
    objectives: dict[str, float] = Field(default_factory=dict)
    constraints_satisfied: dict[str, bool] = Field(default_factory=dict)
    feasible: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class DesignEvaluator:
    """Wraps existing SoC PPA models into a callable objective function.

    Thread-safe: evaluate() is a pure function with no shared mutable state.
    Reuses existing SoCComposition, estimate_manufacturing_cost, and
    technology.py for all PPA calculations.
    """

    def __init__(
        self,
        design_space: DesignSpace,
        base_state: dict[str, Any] | None = None,
        constraint_bounds: dict[str, tuple[float, float]] | None = None,
    ):
        self.design_space = design_space
        self.base_state = base_state or {}
        self.constraint_bounds = constraint_bounds or design_space.constraint_bounds

        # Cache workload info from base_state
        workload = self.base_state.get("workload_profile", {})
        self._total_gflops = workload.get(
            "total_estimated_gflops", workload.get("estimated_gflops", 5.0)
        )
        self._target_volume = self.base_state.get("constraints", {}).get("target_volume", 10_000)

    def evaluate(self, params: dict[str, Any]) -> EvaluationResult:
        """Evaluate a single design point. Thread-safe, pure function.

        Args:
            params: Design parameter dict (e.g. from DesignSpace.decode()).

        Returns:
            EvaluationResult with objective values and feasibility.
        """
        from embodied_ai_architect.graphs.ip_blocks import (
            CPUCoreConfig,
            KPUBlockConfig,
            MemoryControllerConfig,
            NoCIPConfig,
            SoCComposition,
        )
        from embodied_ai_architect.graphs.manufacturing import estimate_manufacturing_cost
        from embodied_ai_architect.graphs.technology import get_technology

        try:
            process_nm = int(params.get("process_nm", 28))
            clock_mhz = float(params.get("clock_mhz", 500.0))
            array_rows = int(params.get("array_rows", 8))
            array_cols = int(params.get("array_cols", 8))
            sram_kb = int(params.get("sram_kb", 256))
            num_compute_tiles = int(params.get("num_compute_tiles", 4))
            noc_link_width = int(params.get("noc_link_width_bits", 256))

            # Build SoC composition from parameters
            kpu = KPUBlockConfig(
                name="KPU-opt",
                array_rows=array_rows,
                array_cols=array_cols,
                compute_frequency_mhz=clock_mhz,
                systolic_rows=16,
                systolic_cols=16,
                count=num_compute_tiles,
            )

            cpu = CPUCoreConfig(
                name="CPU-ctrl",
                isa="ARM_A55",
                frequency_mhz=min(clock_mhz, 1400.0),
                pipeline_stages=8,
                issue_width=2,
                l1i_bytes=32 * 1024,
                l1d_bytes=32 * 1024,
                l2_bytes=min(sram_kb, 256) * 1024,
                has_fpu=True,
                has_simd=True,
            )

            mem_ctrl = MemoryControllerConfig(
                name="LPDDR4X",
                channels=2,
                bandwidth_gbps_per_channel=6.4,
                capacity_gb=4.0,
            )

            num_routers = max(4, (num_compute_tiles + 3) * 2)
            noc = NoCIPConfig(
                name="NoC",
                topology="mesh_2d",
                link_width_bits=noc_link_width,
                frequency_mhz=clock_mhz,
                num_routers=num_routers,
            )

            soc = SoCComposition(
                name="opt-soc",
                process_nm=process_nm,
                blocks=[kpu, cpu, mem_ctrl, noc],
            )

            # Compute objectives
            power_w = soc.total_power_watts(utilization=0.8)
            area_mm2 = soc.total_area_mm2()
            peak_gops = soc.total_peak_gops()

            # Latency from workload GFLOPS / peak throughput
            utilization = 0.3
            if peak_gops > 0:
                tech = get_technology(process_nm)
                speed_scale = 25.0 / tech.gate_delay_ps
                latency_s = self._total_gflops / (peak_gops * utilization)
                latency_ms = (latency_s * 1000.0) / speed_scale + 2.0
            else:
                latency_ms = 1000.0

            # Manufacturing cost
            volume = self._target_volume or 10_000
            mfg = estimate_manufacturing_cost(area_mm2, process_nm, volume)
            cost_usd = mfg.total_unit_cost_usd

            objectives = {
                "power_watts": round(power_w, 3),
                "latency_ms": round(latency_ms, 3),
                "area_mm2": round(area_mm2, 3),
                "cost_usd": round(cost_usd, 2),
            }

            # Check feasibility
            constraints_satisfied = {}
            feasible = True
            for name, (lo, hi) in self.constraint_bounds.items():
                val = objectives.get(name, 0.0)
                sat = lo <= val <= hi
                constraints_satisfied[name] = sat
                if not sat:
                    feasible = False

            return EvaluationResult(
                design_params=params,
                objectives=objectives,
                constraints_satisfied=constraints_satisfied,
                feasible=feasible,
                metadata={
                    "peak_gops": round(peak_gops, 2),
                    "yield_percent": mfg.yield_percent,
                    "dies_per_wafer": mfg.dies_per_wafer,
                },
            )

        except Exception as e:
            logger.warning("Evaluation failed for params %s: %s", params, e)
            # Return infeasible result with penalty values
            return EvaluationResult(
                design_params=params,
                objectives={
                    "power_watts": 1e6,
                    "latency_ms": 1e6,
                    "area_mm2": 1e6,
                    "cost_usd": 1e6,
                },
                constraints_satisfied={},
                feasible=False,
                metadata={"error": str(e)},
            )

    def evaluate_batch(self, param_list: list[dict[str, Any]]) -> list[EvaluationResult]:
        """Evaluate a batch of design points sequentially.

        For parallel evaluation, use executor.LocalThreadExecutor.
        """
        return [self.evaluate(p) for p in param_list]
