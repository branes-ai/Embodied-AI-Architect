"""Joint design space: pipeline x model x compiler x hardware.

Extends the SoC-only design space with pipeline structure, NAS, and compiler
configuration variables. Enables simultaneous optimization across all four
dimensions of an embodied AI system.

Phase 3 of the agentic optimization loop.

Usage:
    from embodied_ai_architect.graphs.moo.joint_design_space import (
        create_joint_design_space,
        JointEvaluator,
    )

    ds = create_joint_design_space(constraints={"max_power_watts": 10})
    evaluator = JointEvaluator(ds)
    result = evaluator.evaluate(ds.sample_random(1)[0])
"""

from __future__ import annotations

import logging
from typing import Any

from embodied_ai_architect.graphs.moo.design_space import (
    DesignSpace,
    DesignVariable,
    ObjectiveDirection,
    VarType,
)
from embodied_ai_architect.graphs.moo.evaluator import DesignEvaluator, EvaluationResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline operator metadata — used to estimate GFLOPS and latency per stage
# ---------------------------------------------------------------------------

# Tracker overhead in GFLOPS (lightweight, mostly association logic)
_TRACKER_GFLOPS: dict[str, float] = {
    "bytetrack": 0.01,
    "sort": 0.005,
    "deepsort": 0.5,
    "botsort": 0.02,
    "ocsort": 0.01,
    "none": 0.0,
}

# Tracker latency overhead in ms at reference clock
_TRACKER_LATENCY_MS: dict[str, float] = {
    "bytetrack": 1.0,
    "sort": 0.5,
    "deepsort": 3.0,
    "botsort": 1.5,
    "ocsort": 1.0,
    "none": 0.0,
}

# State estimator compute in GFLOPS
_STATE_ESTIMATOR_GFLOPS: dict[str, float] = {
    "ekf": 0.001,
    "ukf": 0.01,
    "particle_filter": 0.1,
    "none": 0.0,
}

# Compiler/runtime utilization multipliers
# How much each runtime improves HW utilization beyond the base 30%
_RUNTIME_UTILIZATION: dict[str, float] = {
    "pytorch": 0.30,
    "tensorrt": 0.65,
    "openvino": 0.50,
    "tvm": 0.45,
    "onnxruntime": 0.40,
}

# Graph fusion speedup factor (reduction in overhead)
_FUSION_SPEEDUP: float = 1.15


def create_joint_design_space(
    constraints: dict[str, Any] | None = None,
    include_pipeline: bool = True,
    include_nas: bool = True,
    include_compiler: bool = True,
) -> DesignSpace:
    """Factory: create a joint design space across 4 dimensions.

    Dimensions:
    1. Hardware (7 vars): process, clock, array, SRAM, tiles, NoC
    2. Pipeline (3 vars): detector family, tracker type, state estimator
    3. NAS (3 vars): model variant scale, quantization, pruning ratio
    4. Compiler (4 vars): runtime, precision mode, tile size, graph fusion

    Args:
        constraints: Dict with constraint bounds and config options.
            Supports all keys from create_soc_design_space plus:
            - detector_families: list of detector families to search over
            - workload_type: detection, classification, segmentation
            - include_accuracy: bool (default True for joint space)
        include_pipeline: Include pipeline structure variables.
        include_nas: Include NAS variables.
        include_compiler: Include compiler config variables.

    Returns:
        DesignSpace with up to 17+ variables across 4 dimensions.
    """
    from embodied_ai_architect.graphs.technology import TECHNOLOGY_NODES

    constraints = constraints or {}
    process_nodes = sorted([int(k.replace("nm", "")) for k in TECHNOLOGY_NODES.keys()])

    # -----------------------------------------------------------------------
    # Dimension 1: Hardware (same as create_soc_design_space)
    # -----------------------------------------------------------------------
    variables = [
        DesignVariable(
            name="process_nm",
            var_type=VarType.CATEGORICAL,
            choices=process_nodes,
        ),
        DesignVariable(
            name="clock_mhz",
            var_type=VarType.CONTINUOUS,
            lower=100.0,
            upper=3000.0,
        ),
        DesignVariable(
            name="array_rows",
            var_type=VarType.INTEGER,
            lower=2,
            upper=64,
        ),
        DesignVariable(
            name="array_cols",
            var_type=VarType.INTEGER,
            lower=2,
            upper=64,
        ),
        DesignVariable(
            name="sram_kb",
            var_type=VarType.INTEGER,
            lower=32,
            upper=2048,
        ),
        DesignVariable(
            name="num_compute_tiles",
            var_type=VarType.INTEGER,
            lower=1,
            upper=16,
        ),
        DesignVariable(
            name="noc_link_width_bits",
            var_type=VarType.CATEGORICAL,
            choices=[64, 128, 256, 512],
        ),
    ]

    # -----------------------------------------------------------------------
    # Dimension 2: Pipeline structure
    # -----------------------------------------------------------------------
    if include_pipeline:
        detector_families = constraints.get(
            "detector_families", ["yolov8", "yolov5", "efficientdet", "ssd_mobilenet"]
        )
        variables.extend(
            [
                DesignVariable(
                    name="detector_family",
                    var_type=VarType.CATEGORICAL,
                    choices=detector_families,
                ),
                DesignVariable(
                    name="tracker_type",
                    var_type=VarType.CATEGORICAL,
                    choices=["bytetrack", "sort", "deepsort", "botsort", "ocsort", "none"],
                ),
                DesignVariable(
                    name="state_estimator",
                    var_type=VarType.CATEGORICAL,
                    choices=["ekf", "ukf", "particle_filter", "none"],
                ),
            ]
        )

    # -----------------------------------------------------------------------
    # Dimension 3: NAS (model scaling + compression)
    # -----------------------------------------------------------------------
    if include_nas:
        variables.extend(
            [
                DesignVariable(
                    name="model_variant_idx",
                    var_type=VarType.INTEGER,
                    lower=0,
                    upper=4,  # maps to n/s/m/l/x or equivalent
                ),
                DesignVariable(
                    name="quantization_dtype",
                    var_type=VarType.CATEGORICAL,
                    choices=["fp32", "fp16", "int8", "int4", "mixed_int8_fp16"],
                ),
                DesignVariable(
                    name="pruning_ratio",
                    var_type=VarType.CONTINUOUS,
                    lower=0.0,
                    upper=0.7,
                ),
            ]
        )

    # -----------------------------------------------------------------------
    # Dimension 4: Compiler / runtime configuration
    # -----------------------------------------------------------------------
    if include_compiler:
        variables.extend(
            [
                DesignVariable(
                    name="runtime",
                    var_type=VarType.CATEGORICAL,
                    choices=["pytorch", "tensorrt", "openvino", "tvm", "onnxruntime"],
                ),
                DesignVariable(
                    name="tile_size",
                    var_type=VarType.CATEGORICAL,
                    choices=[8, 16, 32, 64],
                ),
                DesignVariable(
                    name="batch_size",
                    var_type=VarType.INTEGER,
                    lower=1,
                    upper=8,
                ),
                DesignVariable(
                    name="graph_fusion",
                    var_type=VarType.CATEGORICAL,
                    choices=[True, False],
                ),
            ]
        )

    # -----------------------------------------------------------------------
    # Objectives — always include accuracy for joint space
    # -----------------------------------------------------------------------
    include_accuracy = constraints.get("include_accuracy", True)

    objectives = ["power_watts", "latency_ms", "area_mm2", "cost_usd"]
    directions = [ObjectiveDirection.MINIMIZE] * 4

    if include_accuracy:
        objectives.append("accuracy_percent")
        directions.append(ObjectiveDirection.MAXIMIZE)
        objectives.append("capability_per_watt")
        directions.append(ObjectiveDirection.MAXIMIZE)

    # -----------------------------------------------------------------------
    # Constraint bounds
    # -----------------------------------------------------------------------
    constraint_bounds: dict[str, tuple[float, float]] = {}
    if constraints.get("max_power_watts"):
        constraint_bounds["power_watts"] = (0.0, constraints["max_power_watts"])
    if constraints.get("max_latency_ms"):
        constraint_bounds["latency_ms"] = (0.0, constraints["max_latency_ms"])
    if constraints.get("max_area_mm2"):
        constraint_bounds["area_mm2"] = (0.0, constraints["max_area_mm2"])
    if constraints.get("max_cost_usd"):
        constraint_bounds["cost_usd"] = (0.0, constraints["max_cost_usd"])
    if constraints.get("min_accuracy_percent"):
        constraint_bounds["accuracy_percent"] = (
            constraints["min_accuracy_percent"],
            100.0,
        )

    return DesignSpace(
        variables=variables,
        objectives=objectives,
        directions=directions,
        constraint_bounds=constraint_bounds,
    )


# ---------------------------------------------------------------------------
# Model variant index → concrete family/variant mapping
# ---------------------------------------------------------------------------

# Ordered by increasing size within each family
_VARIANT_MAP: dict[str, list[str]] = {
    "yolov8": ["n", "s", "m", "l", "x"],
    "yolov5": ["n", "s", "m", "l"],
    "efficientdet": ["d0", "d1"],
    "ssd_mobilenet": ["v2"],
    "resnet": ["18", "34", "50", "101"],
    "mobilenet": ["v3_small", "v2", "v3_large"],
    "efficientnet": ["b0", "b1", "b2", "b3"],
    "deeplabv3": ["mobilenet_v3", "resnet50"],
    "fcn": ["resnet50"],
}


def _resolve_model(detector_family: str, variant_idx: int) -> tuple[str, str]:
    """Map detector_family + variant_idx → (family, variant)."""
    variants = _VARIANT_MAP.get(detector_family, ["s"])
    idx = max(0, min(variant_idx, len(variants) - 1))
    return detector_family, variants[idx]


class JointEvaluator:
    """Evaluator for the joint pipeline x model x compiler x hardware space.

    Composes a complete system from a design point:
    1. HW: delegates to DesignEvaluator for SoC PPA
    2. Pipeline: adds tracker + state estimator overhead
    3. NAS: selects model variant, applies quantization/pruning to accuracy
    4. Compiler: adjusts utilization and latency based on runtime config

    Thread-safe: evaluate() is a pure function.
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

        # Workload config
        constraints = self.base_state.get("constraints", {})
        self._workload_type = constraints.get("workload_type", "detection")

    def evaluate(self, params: dict[str, Any]) -> EvaluationResult:
        """Evaluate a joint design point across all 4 dimensions.

        Args:
            params: Design parameter dict from DesignSpace.decode() or sample_random().

        Returns:
            EvaluationResult with PPA + accuracy + capability/watt objectives.
        """
        try:
            return self._evaluate_inner(params)
        except Exception as e:
            logger.warning("Joint evaluation failed for params %s: %s", params, e)
            penalty = {
                "power_watts": 1e6,
                "latency_ms": 1e6,
                "area_mm2": 1e6,
                "cost_usd": 1e6,
            }
            if "accuracy_percent" in self.design_space.objectives:
                penalty["accuracy_percent"] = 0.0
            if "capability_per_watt" in self.design_space.objectives:
                penalty["capability_per_watt"] = 0.0
            return EvaluationResult(
                design_params=params,
                objectives=penalty,
                constraints_satisfied={},
                feasible=False,
                metadata={"error": str(e)},
            )

    def _evaluate_inner(self, params: dict[str, Any]) -> EvaluationResult:
        """Core evaluation logic."""
        from embodied_ai_architect.graphs.moo.accuracy_tables import (
            estimate_effective_accuracy,
            estimate_effective_gflops,
            get_model_spec,
            lookup_accuracy_for_workload,
        )

        # -------------------------------------------------------------------
        # 1. Resolve model from pipeline + NAS variables
        # -------------------------------------------------------------------
        detector_family = str(params.get("detector_family", "yolov8"))
        variant_idx = int(params.get("model_variant_idx", 1))
        family, variant = _resolve_model(detector_family, variant_idx)

        quant_dtype = str(params.get("quantization_dtype", "fp16"))
        pruning_ratio = float(params.get("pruning_ratio", 0.0))

        # Get model spec for compute requirements
        model_spec = get_model_spec(family, variant)
        if model_spec:
            base_gflops = model_spec.gflops
            model_gflops = estimate_effective_gflops(family, variant, pruning_ratio)
            if model_gflops is None:
                model_gflops = base_gflops
        else:
            # Fallback: use lookup by workload type
            lookup = lookup_accuracy_for_workload(self._workload_type, 100.0, quant_dtype)
            base_gflops = lookup.get("gflops", 28.6)
            model_gflops = base_gflops
            family = lookup.get("model_family", family)
            variant = lookup.get("model_variant", variant)

        # -------------------------------------------------------------------
        # 2. Pipeline overhead: tracker + state estimator
        # -------------------------------------------------------------------
        tracker_type = str(params.get("tracker_type", "bytetrack"))
        state_est = str(params.get("state_estimator", "ekf"))

        pipeline_gflops = (
            model_gflops
            + _TRACKER_GFLOPS.get(tracker_type, 0.01)
            + _STATE_ESTIMATOR_GFLOPS.get(state_est, 0.001)
        )

        tracker_latency = _TRACKER_LATENCY_MS.get(tracker_type, 1.0)

        # -------------------------------------------------------------------
        # 3. Compiler / runtime config
        # -------------------------------------------------------------------
        runtime = str(params.get("runtime", "tensorrt"))
        tile_size = int(params.get("tile_size", 32))
        batch_size = int(params.get("batch_size", 1))
        graph_fusion = params.get("graph_fusion", True)

        # Runtime determines HW utilization
        utilization = _RUNTIME_UTILIZATION.get(runtime, 0.30)

        # Tile size affects utilization: larger tiles = better utilization
        # but only up to the array dimension
        array_size = int(params.get("array_rows", 8)) * int(params.get("array_cols", 8))
        if tile_size <= 8:
            tile_util_factor = 0.85
        elif tile_size >= array_size:
            tile_util_factor = 1.0
        else:
            tile_util_factor = 0.90 + 0.10 * (tile_size / max(array_size, 1))
        tile_util_factor = min(tile_util_factor, 1.0)

        effective_utilization = utilization * tile_util_factor

        # Graph fusion reduces overhead
        fusion_factor = _FUSION_SPEEDUP if graph_fusion else 1.0

        # -------------------------------------------------------------------
        # 4. HW evaluation via base DesignEvaluator (PPA)
        # -------------------------------------------------------------------
        # Build base_state with the model's GFLOPS as the workload
        hw_base_state = dict(self.base_state)
        hw_base_state.setdefault("workload_profile", {})
        hw_base_state["workload_profile"]["total_estimated_gflops"] = pipeline_gflops
        hw_base_state["workload_profile"]["estimated_gflops"] = pipeline_gflops
        # Pass model info for accuracy computation
        hw_base_state["workload_profile"]["model_family"] = family
        hw_base_state["workload_profile"]["model_variant"] = variant
        hw_base_state.setdefault("constraints", {})
        hw_base_state["constraints"]["workload_type"] = self._workload_type
        hw_base_state["constraints"]["quantization_dtype"] = quant_dtype

        # Create a temporary HW-only design space (4 objectives, no accuracy)
        # We compute accuracy ourselves from the NAS variables
        from embodied_ai_architect.graphs.moo.design_space import create_soc_design_space

        hw_ds = create_soc_design_space()  # PPA-only, no accuracy objectives
        hw_evaluator = DesignEvaluator(
            design_space=hw_ds,
            base_state=hw_base_state,
            constraint_bounds={},
        )

        hw_result = hw_evaluator.evaluate(params)

        # -------------------------------------------------------------------
        # 5. Adjust latency based on compiler config
        # -------------------------------------------------------------------
        raw_power = hw_result.objectives.get("power_watts", 1e6)
        raw_area = hw_result.objectives.get("area_mm2", 1e6)
        raw_cost = hw_result.objectives.get("cost_usd", 1e6)

        # Re-derive latency with correct utilization
        peak_gops = hw_result.metadata.get("peak_gops", 0.0)
        if peak_gops > 0 and effective_utilization > 0:
            from embodied_ai_architect.graphs.technology import get_technology

            process_nm = int(params.get("process_nm", 28))
            tech = get_technology(process_nm)
            speed_scale = 25.0 / tech.gate_delay_ps
            latency_s = pipeline_gflops / (peak_gops * effective_utilization)
            hw_latency_ms = (latency_s * 1000.0) / speed_scale + 2.0
        else:
            hw_latency_ms = hw_result.objectives.get("latency_ms", 1000.0)

        # Add tracker + state estimator latency, apply fusion speedup
        total_latency_ms = (hw_latency_ms + tracker_latency) / fusion_factor

        # Batch size: amortizes per-inference overhead but increases latency
        # For real-time systems, batch_size > 1 trades latency for throughput
        if batch_size > 1:
            # Latency increases sub-linearly with batch (memory bound)
            total_latency_ms *= 1.0 + 0.3 * (batch_size - 1)

        # -------------------------------------------------------------------
        # 6. Compute accuracy from NAS variables
        # -------------------------------------------------------------------
        accuracy_pct = estimate_effective_accuracy(family, variant, quant_dtype, pruning_ratio)
        if accuracy_pct is None:
            lookup = lookup_accuracy_for_workload(self._workload_type, model_gflops, quant_dtype)
            accuracy_pct = lookup.get("accuracy", 0.0)

        capability_score = accuracy_pct / 100.0
        capability_per_watt = capability_score / raw_power if raw_power > 0 else 0.0
        gops_per_watt = peak_gops / raw_power if raw_power > 0 else 0.0

        # -------------------------------------------------------------------
        # 7. Assemble objectives
        # -------------------------------------------------------------------
        objectives: dict[str, float] = {
            "power_watts": round(raw_power, 3),
            "latency_ms": round(total_latency_ms, 3),
            "area_mm2": round(raw_area, 3),
            "cost_usd": round(raw_cost, 2),
        }

        if "accuracy_percent" in self.design_space.objectives:
            objectives["accuracy_percent"] = round(accuracy_pct, 2)
        if "capability_per_watt" in self.design_space.objectives:
            objectives["capability_per_watt"] = round(capability_per_watt, 4)

        # -------------------------------------------------------------------
        # 8. Check feasibility
        # -------------------------------------------------------------------
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
                "model_family": family,
                "model_variant": variant,
                "model_gflops": round(model_gflops, 2),
                "pipeline_gflops": round(pipeline_gflops, 2),
                "tracker_type": tracker_type,
                "state_estimator": state_est,
                "runtime": runtime,
                "effective_utilization": round(effective_utilization, 3),
                "graph_fusion": graph_fusion,
                "quantization_dtype": quant_dtype,
                "pruning_ratio": round(pruning_ratio, 3),
                "accuracy_percent": round(accuracy_pct, 2),
                "capability_score": round(capability_score, 4),
                "capability_per_watt": round(capability_per_watt, 4),
                "gops_per_watt": round(gops_per_watt, 2),
            },
        )

    def evaluate_batch(self, param_list: list[dict[str, Any]]) -> list[EvaluationResult]:
        """Evaluate a batch of design points sequentially."""
        return [self.evaluate(p) for p in param_list]
