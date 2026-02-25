"""Specialist agent executors for the agentic SoC designer.

Each specialist conforms to the AgentExecutor protocol:
    (task: TaskNode, state: SoCDesignState) -> dict[str, Any]

Specialists read from state (constraints, previous task results) and return
a result dict. If the result contains a '_state_updates' key, the Dispatcher
merges those into the top-level SoCDesignState.

Usage:
    from embodied_ai_architect.graphs.specialists import create_default_dispatcher

    dispatcher = create_default_dispatcher()
    state = dispatcher.run(state)
"""

from __future__ import annotations

import logging
from typing import Any

from embodied_ai_architect.graphs.dispatcher import Dispatcher
from embodied_ai_architect.graphs.soc_state import (
    DesignConstraints,
    PPAMetrics,
    SoCDesignState,
    get_constraints,
    get_dependency_results,
)
from embodied_ai_architect.graphs.task_graph import TaskNode
from embodied_ai_architect.graphs.manufacturing import estimate_manufacturing_cost
from embodied_ai_architect.graphs.technology import get_technology
from embodied_ai_architect.graphs.ip_blocks import (
    ARM_A55_PRESET,
    ARM_A78_PRESET,
    DEFAULT_IO_PRESET,
    DEFAULT_ISP_PRESET,
    IPBlockConfig,
    MemoryControllerConfig,
    NoCIPConfig,
    SMALL_GPU_PRESET,
    SoCComposition,
    kpu_preset_block,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Workload Analyzer
# ---------------------------------------------------------------------------


def workload_analyzer(task: TaskNode, state: SoCDesignState) -> dict[str, Any]:
    """Analyze AI workload: operator graph, compute/memory requirements.

    Wraps the existing ModelAnalyzerAgent when a model path is available.
    Falls back to constraint-based estimation from the goal description.

    Writes to state: workload_profile
    """
    goal = state.get("goal", "")
    use_case = state.get("use_case", "")
    constraints = get_constraints(state)

    # Try to use existing ModelAnalyzerAgent if available
    model_analysis = _try_model_analysis(state)

    if model_analysis:
        profile = _build_profile_from_model(model_analysis, use_case)
    else:
        profile = _estimate_workload_from_goal(goal, use_case, constraints)

    return {
        "summary": f"Workload analyzed for {use_case or 'target application'}",
        "workload_profile": profile,
        "_state_updates": {"workload_profile": profile},
    }


def _try_model_analysis(state: SoCDesignState) -> dict[str, Any] | None:
    """Try to run ModelAnalyzerAgent if a model path is in state."""
    try:
        from embodied_ai_architect.agents.model_analyzer import ModelAnalyzerAgent

        model_path = state.get("workload_profile", {}).get("model_path")
        if not model_path:
            return None

        agent = ModelAnalyzerAgent()
        result = agent.execute({"model": model_path})
        if result.success:
            return result.data
    except (ImportError, Exception) as e:
        logger.debug("ModelAnalyzerAgent not available: %s", e)
    return None


def _build_profile_from_model(model_data: dict[str, Any], use_case: str) -> dict[str, Any]:
    """Build workload profile from ModelAnalyzerAgent output."""
    layer_counts = model_data.get("layer_type_counts", {})
    total_params = model_data.get("total_parameters", 0)

    # Classify operator types
    operators = []
    if layer_counts.get("Conv2d", 0) > 0:
        operators.append({"type": "convolution", "count": layer_counts["Conv2d"]})
    if layer_counts.get("Linear", 0) > 0:
        operators.append({"type": "matrix_multiply", "count": layer_counts["Linear"]})
    if layer_counts.get("MultiheadAttention", 0) > 0:
        operators.append({"type": "attention", "count": layer_counts["MultiheadAttention"]})
    for ltype, count in layer_counts.items():
        if ltype not in ("Conv2d", "Linear", "MultiheadAttention"):
            operators.append({"type": ltype.lower(), "count": count})

    # Estimate compute requirements
    params_m = total_params / 1e6
    estimated_gflops = params_m * 2.0  # rough: 2 FLOPs per parameter per inference
    estimated_memory_mb = (total_params * 4) / (1024 * 1024)  # FP32

    return {
        "model_type": model_data.get("model_type", "unknown"),
        "total_parameters": total_params,
        "operators": operators,
        "estimated_gflops": round(estimated_gflops, 2),
        "estimated_memory_mb": round(estimated_memory_mb, 2),
        "dominant_op": operators[0]["type"] if operators else "unknown",
        "use_case": use_case,
        "source": "model_analysis",
    }


def _estimate_workload_from_goal(
    goal: str, use_case: str, constraints: DesignConstraints
) -> dict[str, Any]:
    """Estimate workload profile from goal description when no model is available.

    Uses heuristics based on common embodied AI workloads.
    """
    goal_lower = goal.lower()

    # Detect workload types from keywords
    workloads = []
    if any(kw in goal_lower for kw in ("detection", "yolo", "object")):
        workloads.append(
            {
                "name": "object_detection",
                "model_class": "YOLOv8",
                "operators": [
                    {"type": "convolution", "count": 75},
                    {"type": "batch_norm", "count": 70},
                    {"type": "activation", "count": 70},
                ],
                "estimated_gflops": 8.7,
                "estimated_memory_mb": 12.0,
                "estimated_params_m": 3.2,
            }
        )
    if any(kw in goal_lower for kw in ("tracking", "track", "bytetrack")):
        workloads.append(
            {
                "name": "object_tracking",
                "model_class": "ByteTrack",
                "operators": [
                    {"type": "matrix_multiply", "count": 10},
                    {"type": "sort", "count": 1},
                ],
                "estimated_gflops": 0.5,
                "estimated_memory_mb": 2.0,
                "estimated_params_m": 0.0,
            }
        )
    if any(kw in goal_lower for kw in ("slam", "localization", "mapping")):
        workloads.append(
            {
                "name": "visual_slam",
                "model_class": "ORB-SLAM3",
                "operators": [
                    {"type": "feature_extraction", "count": 1},
                    {"type": "matrix_multiply", "count": 20},
                    {"type": "sparse_solve", "count": 1},
                ],
                "estimated_gflops": 3.0,
                "estimated_memory_mb": 50.0,
                "estimated_params_m": 0.0,
            }
        )
    if any(kw in goal_lower for kw in ("perception", "vision", "camera")):
        workloads.append(
            {
                "name": "visual_perception",
                "model_class": "CNN-based",
                "operators": [
                    {"type": "convolution", "count": 50},
                    {"type": "pooling", "count": 10},
                ],
                "estimated_gflops": 6.0,
                "estimated_memory_mb": 10.0,
                "estimated_params_m": 2.5,
            }
        )
    if any(kw in goal_lower for kw in ("voice", "speech", "audio")):
        workloads.append(
            {
                "name": "voice_recognition",
                "model_class": "Whisper-tiny",
                "operators": [
                    {"type": "attention", "count": 4},
                    {"type": "convolution", "count": 2},
                    {"type": "matrix_multiply", "count": 20},
                ],
                "estimated_gflops": 1.5,
                "estimated_memory_mb": 150.0,
                "estimated_params_m": 39.0,
            }
        )
    if any(kw in goal_lower for kw in ("lidar", "point cloud")):
        workloads.append(
            {
                "name": "lidar_processing",
                "model_class": "PointPillars",
                "operators": [
                    {"type": "convolution", "count": 30},
                    {"type": "scatter", "count": 1},
                ],
                "estimated_gflops": 4.0,
                "estimated_memory_mb": 20.0,
                "estimated_params_m": 1.8,
            }
        )

    # Default if nothing matched
    if not workloads:
        workloads.append(
            {
                "name": "general_inference",
                "model_class": "Unknown",
                "operators": [{"type": "general_purpose", "count": 1}],
                "estimated_gflops": 5.0,
                "estimated_memory_mb": 20.0,
                "estimated_params_m": 2.0,
            }
        )

    # Add scheduling field per workload (concurrent/sequential/time_shared)
    for w in workloads:
        if "scheduling" not in w:
            w["scheduling"] = _infer_scheduling(w.get("name", ""))

    total_gflops = aggregate_workload_requirements(workloads)
    total_memory_mb = sum(w["estimated_memory_mb"] for w in workloads)

    # Determine dominant operation type
    all_ops: dict[str, int] = {}
    for w in workloads:
        for op in w["operators"]:
            all_ops[op["type"]] = all_ops.get(op["type"], 0) + op["count"]
    dominant_op = max(all_ops, key=all_ops.get) if all_ops else "unknown"  # type: ignore[arg-type]

    return {
        "workloads": workloads,
        "total_estimated_gflops": round(total_gflops, 2),
        "total_estimated_memory_mb": round(total_memory_mb, 2),
        "dominant_op": dominant_op,
        "workload_count": len(workloads),
        "use_case": use_case,
        "source": "goal_estimation",
    }


def _infer_scheduling(workload_name: str) -> str:
    """Infer scheduling mode from workload name."""
    concurrent = {"object_detection", "visual_perception", "visual_slam"}
    sequential = {"voice_recognition"}
    if workload_name in concurrent:
        return "concurrent"
    if workload_name in sequential:
        return "sequential"
    return "time_shared"


def aggregate_workload_requirements(workloads: list[dict]) -> float:
    """Aggregate GFLOPS across workloads considering scheduling mode.

    Concurrent workloads sum their GFLOPS (peak demand).
    Sequential workloads take the max (only one active at a time).
    Time-shared workloads contribute 70% of their GFLOPS (multiplexed).

    Args:
        workloads: List of workload dicts with estimated_gflops and scheduling.

    Returns:
        Peak aggregate GFLOPS requirement.
    """
    concurrent_gflops = 0.0
    sequential_gflops: list[float] = []
    time_shared_gflops = 0.0

    for w in workloads:
        gflops = w.get("estimated_gflops", 0.0)
        scheduling = w.get("scheduling", "concurrent")
        if scheduling == "concurrent":
            concurrent_gflops += gflops
        elif scheduling == "sequential":
            sequential_gflops.append(gflops)
        else:  # time_shared
            time_shared_gflops += gflops * 0.7

    return (
        concurrent_gflops
        + (max(sequential_gflops) if sequential_gflops else 0.0)
        + time_shared_gflops
    )


def map_workloads_to_accelerators(
    workloads: list[dict], hardware_candidates: list[dict]
) -> list[dict]:
    """Map workloads to available accelerators for heterogeneous execution.

    Each workload is assigned to the best-matching accelerator based on
    operator type compatibility and available compute.

    Args:
        workloads: List of workload dicts.
        hardware_candidates: Available hardware platforms.

    Returns:
        List of mapping dicts with workload, accelerator, and precision.
    """
    if not hardware_candidates:
        return []

    mappings = []
    for w in workloads:
        best_hw = hardware_candidates[0]  # default to top-ranked
        dominant_op = ""
        if w.get("operators"):
            dominant_op = max(w["operators"], key=lambda o: o.get("count", 0)).get("type", "")

        # Find best match for this workload's dominant operator
        for hw in hardware_candidates:
            strengths = hw.get("strengths", [])
            if dominant_op == "convolution" and any(
                s in strengths for s in ("dataflow", "int8", "gpu_acceleration")
            ):
                best_hw = hw
                break
            if dominant_op == "attention" and any(
                s in strengths for s in ("gpu_acceleration", "flexible")
            ):
                best_hw = hw
                break

        precision = "int8" if best_hw.get("peak_tops_int8", 0) > 0 else "fp16"
        mappings.append(
            {
                "workload": w.get("name", "unknown"),
                "model_class": w.get("model_class", "unknown"),
                "accelerator": best_hw.get("name", "unknown"),
                "precision": precision,
                "scheduling": w.get("scheduling", "concurrent"),
            }
        )

    return mappings


# ---------------------------------------------------------------------------
# Hardware Explorer
# ---------------------------------------------------------------------------


def hw_explorer(task: TaskNode, state: SoCDesignState) -> dict[str, Any]:
    """Enumerate and score hardware candidates against constraints.

    Wraps the existing HardwareProfileAgent when available, and adds
    constraint-based filtering and scoring.

    Writes to state: hardware_candidates
    """
    constraints = get_constraints(state)
    dep_results = get_dependency_results(state, task)

    # Collect workload profile from dependencies
    workload = None
    for dep_id, result in dep_results.items():
        if "workload_profile" in result:
            workload = result["workload_profile"]
            break

    # Also check top-level state
    if workload is None:
        workload = state.get("workload_profile", {})

    # Use static catalog for consistent field naming and KPU inclusion.
    # The existing HardwareProfileAgent uses a different schema — it will be
    # integrated once field names are normalized across embodied-schemas.
    candidates = _estimate_hardware_candidates(workload, constraints)

    # Filter by constraints
    filtered = _filter_by_constraints(candidates, constraints)

    return {
        "summary": f"Explored {len(filtered)} hardware candidates from {len(candidates)} evaluated",
        "hardware_candidates": filtered,
        "total_evaluated": len(candidates),
        "_state_updates": {"hardware_candidates": filtered},
    }


def _try_hardware_agent(
    workload: dict[str, Any], constraints: DesignConstraints
) -> list[dict[str, Any]]:
    """Try to use existing HardwareProfileAgent."""
    try:
        from embodied_ai_architect.agents.hardware_profile.agent import HardwareProfileAgent

        agent = HardwareProfileAgent()
        input_data: dict[str, Any] = {
            "model_analysis": {
                "total_parameters": workload.get("total_parameters", 0),
                "total_layers": 0,
                "layer_type_counts": {},
            },
            "constraints": {},
            "top_n": 10,
        }
        if constraints.max_latency_ms:
            input_data["constraints"]["max_latency_ms"] = constraints.max_latency_ms
        if constraints.max_power_watts:
            input_data["constraints"]["max_power_watts"] = constraints.max_power_watts
        if constraints.max_cost_usd:
            input_data["constraints"]["max_cost_usd"] = constraints.max_cost_usd

        result = agent.execute(input_data)
        if result.success:
            return result.data.get("recommendations", [])
    except (ImportError, Exception) as e:
        logger.debug("HardwareProfileAgent not available: %s", e)
    return []


# -- Stillwater KPU: not yet in embodied-schemas, always included manually --
_STILLWATER_KPU: dict[str, Any] = {
    "name": "Stillwater KPU",
    "vendor": "Stillwater Supercomputing",
    "type": "kpu",
    "compute_paradigm": "dataflow",
    "peak_tops_int8": 20.0,
    "peak_tflops_fp16": 5.0,
    "memory_gb": 2.0,
    "tdp_watts": 5.0,
    "cost_usd": 25.0,
    "strengths": ["low_power", "dataflow", "int8", "custom_operators"],
    "suitable_for": ["edge", "drone", "amr"],
}

# Fallback catalog used when embodied-schemas is not installed
_FALLBACK_CATALOG: list[dict[str, Any]] = [
    {
        "name": "NVIDIA Jetson Orin Nano",
        "vendor": "NVIDIA",
        "type": "gpu",
        "compute_paradigm": "simd",
        "peak_tops_int8": 40.0,
        "peak_tflops_fp16": 20.0,
        "memory_gb": 8.0,
        "tdp_watts": 15.0,
        "cost_usd": 199.0,
        "strengths": ["gpu_acceleration", "cuda", "tensorrt", "wide_framework_support"],
        "suitable_for": ["edge", "drone", "amr", "quadruped"],
    },
    {
        "name": "Google Coral Edge TPU",
        "vendor": "Google",
        "type": "tpu",
        "compute_paradigm": "systolic_array",
        "peak_tops_int8": 4.0,
        "peak_tflops_fp16": 0.0,
        "memory_gb": 0.008,
        "tdp_watts": 2.0,
        "cost_usd": 35.0,
        "strengths": ["ultra_low_power", "int8_only", "fast_inference"],
        "suitable_for": ["edge", "iot", "camera"],
    },
    {
        "name": "AMD Ryzen AI (Xilinx NPU)",
        "vendor": "AMD",
        "type": "npu",
        "compute_paradigm": "reconfigurable",
        "peak_tops_int8": 16.0,
        "peak_tflops_fp16": 8.0,
        "memory_gb": 16.0,
        "tdp_watts": 25.0,
        "cost_usd": 150.0,
        "strengths": ["heterogeneous", "cpu_plus_npu", "flexible"],
        "suitable_for": ["edge", "amr", "quadruped"],
    },
    {
        "name": "Raspberry Pi 5",
        "vendor": "Raspberry Pi Foundation",
        "type": "cpu",
        "compute_paradigm": "von_neumann",
        "peak_tops_int8": 0.5,
        "peak_tflops_fp16": 0.1,
        "memory_gb": 8.0,
        "tdp_watts": 8.0,
        "cost_usd": 80.0,
        "strengths": ["general_purpose", "low_cost", "ecosystem"],
        "suitable_for": ["edge", "education", "prototype"],
    },
    {
        "name": "Hailo-8",
        "vendor": "Hailo",
        "type": "npu",
        "compute_paradigm": "dataflow",
        "peak_tops_int8": 26.0,
        "peak_tflops_fp16": 0.0,
        "memory_gb": 0.0,  # No on-chip memory, uses host
        "tdp_watts": 2.5,
        "cost_usd": 70.0,
        "strengths": ["high_efficiency", "low_power", "dataflow"],
        "suitable_for": ["edge", "camera", "drone"],
    },
]


def _derive_strengths(entry) -> list[str]:
    """Derive strength tags from a HardwareEntry for scoring compatibility."""
    strengths: list[str] = []

    # Compute paradigm mapping
    paradigm = entry.compute_paradigm.value if entry.compute_paradigm else ""
    if paradigm == "dataflow":
        strengths.append("dataflow")
    elif paradigm == "simd":
        strengths.append("gpu_acceleration")
    elif paradigm == "reconfigurable":
        strengths.extend(["flexible", "heterogeneous"])
    elif paradigm == "von_neumann":
        strengths.append("general_purpose")
    elif paradigm == "systolic_array":
        strengths.append("int8")

    # Quantization support
    quant = entry.capabilities.quantization_support if entry.capabilities else []
    if "int8" in quant and "int8" not in strengths:
        strengths.append("int8")

    # Low power
    if entry.power and entry.power.tdp_watts is not None and entry.power.tdp_watts <= 5:
        strengths.append("low_power")

    # Framework / runtime tags from capabilities
    frameworks = [f.lower() for f in (entry.capabilities.frameworks if entry.capabilities else [])]
    runtimes = [
        r.lower() for r in (entry.capabilities.inference_runtimes if entry.capabilities else [])
    ]
    all_sw = frameworks + runtimes
    if any("cuda" in s for s in all_sw):
        strengths.append("cuda")
    if any("tensorrt" in s for s in all_sw):
        strengths.append("tensorrt")

    # GPU type bonus
    hw_type = entry.hardware_type.value if entry.hardware_type else ""
    if hw_type == "gpu" and "gpu_acceleration" not in strengths:
        strengths.append("gpu_acceleration")

    return strengths


def _load_registry_candidates() -> list[dict[str, Any]]:
    """Load hardware candidates from embodied-schemas Registry.

    Returns an empty list if embodied-schemas is not installed.
    """
    try:
        from embodied_schemas import Registry
    except ImportError:
        return []

    try:
        registry = Registry.load()
    except Exception as e:
        logger.debug("Failed to load embodied-schemas Registry: %s", e)
        return []

    candidates: list[dict[str, Any]] = []
    for entry in registry.hardware:
        candidates.append(
            {
                "name": entry.name,
                "vendor": entry.vendor,
                "type": entry.hardware_type.value if entry.hardware_type else "unknown",
                "compute_paradigm": (
                    entry.compute_paradigm.value if entry.compute_paradigm else "unknown"
                ),
                "peak_tops_int8": (
                    entry.capabilities.peak_tops_int8
                    if entry.capabilities and entry.capabilities.peak_tops_int8
                    else 0.0
                ),
                "peak_tflops_fp16": (
                    entry.capabilities.peak_tflops_fp16
                    if entry.capabilities and entry.capabilities.peak_tflops_fp16
                    else 0.0
                ),
                "memory_gb": (entry.capabilities.memory_gb if entry.capabilities else 0.0),
                "tdp_watts": entry.power.tdp_watts if entry.power else None,
                "cost_usd": entry.cost_usd,
                "strengths": _derive_strengths(entry),
                "suitable_for": entry.suitable_for or [],
            }
        )
    return candidates


def _estimate_hardware_candidates(
    workload: dict[str, Any], constraints: DesignConstraints
) -> list[dict[str, Any]]:
    """Generate hardware candidates from heuristics when agent not available."""
    total_gflops = workload.get("total_estimated_gflops", workload.get("estimated_gflops", 5.0))
    dominant_op = workload.get("dominant_op", "general_purpose")

    # Try registry first, fall back to static catalog
    registry_candidates = _load_registry_candidates()
    if registry_candidates:
        catalog = registry_candidates
    else:
        catalog = list(_FALLBACK_CATALOG)  # copy to avoid mutating the constant

    # Always include Stillwater KPU (not yet in schemas)
    if not any(c["name"] == _STILLWATER_KPU["name"] for c in catalog):
        catalog.append(dict(_STILLWATER_KPU))

    # Score each candidate
    scored = []
    for hw in catalog:
        score = _score_hardware(hw, total_gflops, dominant_op, constraints)
        scored.append({**hw, "score": score, "rank": 0})

    # Sort by score descending and assign ranks
    scored.sort(key=lambda x: x["score"], reverse=True)
    for i, hw in enumerate(scored):
        hw["rank"] = i + 1

    return scored


def _score_hardware(
    hw: dict[str, Any],
    required_gflops: float,
    dominant_op: str,
    constraints: DesignConstraints,
) -> float:
    """Score a hardware candidate (0-100) based on workload fit and constraints."""
    score = 50.0  # base score

    # Performance adequacy (can it handle the workload?)
    peak_int8 = hw.get("peak_tops_int8", 0)
    if peak_int8 > 0:
        perf_ratio = peak_int8 / max(required_gflops / 1000, 0.001)
        score += min(perf_ratio * 10, 20)  # up to +20 for headroom

    # Power fit
    if constraints.max_power_watts and hw.get("tdp_watts"):
        if hw["tdp_watts"] <= constraints.max_power_watts:
            power_headroom = 1 - (hw["tdp_watts"] / constraints.max_power_watts)
            score += power_headroom * 15  # up to +15 for power headroom
        else:
            score -= 30  # heavy penalty for exceeding power budget

    # Cost fit
    if constraints.max_cost_usd and hw.get("cost_usd"):
        if hw["cost_usd"] <= constraints.max_cost_usd:
            cost_headroom = 1 - (hw["cost_usd"] / constraints.max_cost_usd)
            score += cost_headroom * 10  # up to +10 for cost headroom
        else:
            score -= 20  # penalty for exceeding cost budget

    # Operation type match
    strengths = hw.get("strengths", [])
    if dominant_op == "convolution" and any(
        s in strengths for s in ("gpu_acceleration", "dataflow", "int8")
    ):
        score += 10
    if dominant_op == "attention" and any(s in strengths for s in ("gpu_acceleration", "flexible")):
        score += 10
    if dominant_op == "general_purpose" and "general_purpose" in strengths:
        score += 5

    # Multi-workload bonus: heterogeneous platforms handle diverse workloads better
    if "heterogeneous" in strengths or "flexible" in strengths:
        score += 5

    return round(max(0, min(100, score)), 1)


def _filter_by_constraints(
    candidates: list[dict[str, Any]], constraints: DesignConstraints
) -> list[dict[str, Any]]:
    """Add constraint verdict annotations to candidates."""
    for c in candidates:
        verdicts = {}
        if constraints.max_power_watts and c.get("tdp_watts"):
            verdicts["power"] = "PASS" if c["tdp_watts"] <= constraints.max_power_watts else "FAIL"
        if constraints.max_cost_usd and c.get("cost_usd"):
            verdicts["cost"] = "PASS" if c["cost_usd"] <= constraints.max_cost_usd else "FAIL"
        c["constraint_verdicts"] = verdicts
    return candidates


# ---------------------------------------------------------------------------
# Architecture Composer
# ---------------------------------------------------------------------------


def architecture_composer(task: TaskNode, state: SoCDesignState) -> dict[str, Any]:
    """Compose SoC architecture from workload analysis and hardware candidates.

    Maps workload operators to hardware accelerators, designs memory hierarchy,
    and defines interconnect topology.

    Writes to state: selected_architecture, ip_blocks, memory_map, interconnect
    """
    constraints = get_constraints(state)
    dep_results = get_dependency_results(state, task)

    # Gather inputs from dependencies and state
    workload = state.get("workload_profile", {})
    candidates = state.get("hardware_candidates", [])
    for result in dep_results.values():
        if "workload_profile" in result:
            workload = result["workload_profile"]
        if "hardware_candidates" in result:
            candidates = result["hardware_candidates"]

    # Select top hardware candidate that passes constraints
    selected_hw = None
    for hw in candidates:
        verdicts = hw.get("constraint_verdicts", {})
        if all(v == "PASS" for v in verdicts.values()):
            selected_hw = hw
            break
    if selected_hw is None and candidates:
        selected_hw = candidates[0]  # fallback to best-scored

    if selected_hw is None:
        raise ValueError("No hardware candidates available for architecture composition")

    # Compose heterogeneous SoC from IP blocks
    soc = _compose_soc(workload, selected_hw, constraints)
    architecture = _compose_architecture(workload, selected_hw, constraints, soc)
    ip_blocks = soc.to_ip_block_dicts()
    memory_map = _define_memory_map(ip_blocks, selected_hw)
    interconnect = _define_interconnect(ip_blocks)

    return {
        "summary": f"Composed SoC architecture around {selected_hw['name']}",
        "selected_architecture": architecture,
        "ip_blocks": ip_blocks,
        "memory_map": memory_map,
        "interconnect": interconnect,
        "soc_composition": soc.block_summary(),
        "_state_updates": {
            "selected_architecture": architecture,
            "ip_blocks": ip_blocks,
            "memory_map": memory_map,
            "interconnect": interconnect,
        },
    }


def _compose_soc(
    workload: dict[str, Any],
    hw: dict[str, Any],
    constraints: DesignConstraints,
) -> SoCComposition:
    """Build a heterogeneous SoCComposition from workload analysis and selected hardware.

    Always includes: CPU cores, memory controller, NoC, I/O subsystem.
    Conditionally includes: GPU clusters, KPU tiles, ISP.
    """
    process_nm = constraints.target_process_nm or 28
    max_power = constraints.max_power_watts or 15.0
    hw_type = hw.get("type", "unknown")
    dominant_op = workload.get("dominant_op", "general_purpose")
    workloads = workload.get("workloads", [])
    workload_names = {w.get("name", "") for w in workloads}

    blocks: list[IPBlockConfig] = []

    # --- CPU cores: always present ---
    if max_power < 5.0:
        # Edge: big.LITTLE with 1 big + 1 little
        blocks.append(ARM_A78_PRESET.model_copy(update={"count": 1}))
        blocks.append(ARM_A55_PRESET.model_copy(update={"count": 1}))
    elif max_power < 15.0:
        # Mid-range: 2 big + 2 little
        blocks.append(ARM_A78_PRESET.model_copy(update={"count": 2}))
        blocks.append(ARM_A55_PRESET.model_copy(update={"count": 2}))
    else:
        # High-power: 4 big cores
        blocks.append(ARM_A78_PRESET.model_copy(update={"count": 4}))

    # --- GPU clusters: when workload has attention/general ops AND power allows ---
    has_attention = any(
        op.get("type") == "attention" for w in workloads for op in w.get("operators", [])
    )
    has_general = dominant_op in ("general_purpose", "attention")
    if (has_attention or has_general or hw_type == "gpu") and max_power >= 5.0:
        gpu = SMALL_GPU_PRESET.model_copy(update={"count": 1})
        blocks.append(gpu)

    # --- KPU tiles: when workload has convolution/matrix_multiply AND fits budget ---
    has_conv = any(
        op.get("type") in ("convolution", "matrix_multiply")
        for w in workloads
        for op in w.get("operators", [])
    )
    if has_conv or hw_type == "kpu" or dominant_op in ("convolution", "matrix_multiply"):
        if max_power < 5.0:
            kpu = kpu_preset_block("drone_minimal")
        elif max_power < 15.0:
            kpu = kpu_preset_block("edge_balanced")
        else:
            kpu = kpu_preset_block("server_max")
        blocks.append(kpu)

    # --- Memory controller: always ---
    mem_gb = min(hw.get("memory_gb", 4.0), 8.0)
    channels = 2 if max_power < 10.0 else 4
    blocks.append(
        MemoryControllerConfig(
            name="LPDDR4X Controller",
            channels=channels,
            bandwidth_gbps_per_channel=6.4,
            capacity_gb=mem_gb,
        )
    )

    # --- NoC: always ---
    num_ip = len(blocks) + 2  # +2 for NoC and IO we're about to add
    num_routers = max(4, num_ip * 2)
    blocks.append(
        NoCIPConfig(
            name="Mesh NoC",
            topology="mesh_2d",
            link_width_bits=256 if num_routers <= 16 else 512,
            frequency_mhz=1000.0,
            num_routers=num_routers,
        )
    )

    # --- I/O subsystem: always ---
    blocks.append(DEFAULT_IO_PRESET.model_copy())

    # --- ISP: when vision workloads detected ---
    vision_workloads = {
        "object_detection",
        "visual_perception",
        "visual_slam",
    }
    if workload_names & vision_workloads or dominant_op in ("convolution", "feature_extraction"):
        blocks.append(DEFAULT_ISP_PRESET.model_copy())

    return SoCComposition(
        name=f"SoC-{hw['name'].replace(' ', '-')}",
        process_nm=process_nm,
        blocks=blocks,
    )


def _compose_architecture(
    workload: dict[str, Any],
    hw: dict[str, Any],
    constraints: DesignConstraints,
    soc: SoCComposition | None = None,
) -> dict[str, Any]:
    """Build top-level architecture description."""
    arch = {
        "name": f"SoC-{hw['name'].replace(' ', '-')}",
        "primary_compute": hw["name"],
        "compute_type": hw.get("type", "unknown"),
        "compute_paradigm": hw.get("compute_paradigm", "unknown"),
        "target_process_nm": constraints.target_process_nm or 28,
        "workload_mapping": _map_workloads_to_hw(workload, hw),
        "design_rationale": (
            f"Selected {hw['name']} as primary compute for "
            f"{workload.get('dominant_op', 'general')} workload "
            f"with {hw.get('tdp_watts', '?')}W TDP"
        ),
    }
    if soc is not None:
        # Enrich with per-block type summary
        block_types: dict[str, int] = {}
        for b in soc.blocks:
            block_types[b.block_type] = block_types.get(b.block_type, 0) + b.count
        arch["ip_block_types"] = block_types
    return arch


def _map_workloads_to_hw(workload: dict[str, Any], hw: dict[str, Any]) -> list[dict[str, str]]:
    """Map workload operators/subworkloads to hardware execution targets."""
    mappings = []
    workloads = workload.get("workloads", [])
    if not workloads:
        # Single-model workload
        mappings.append(
            {
                "workload": workload.get("use_case", "main"),
                "target": hw["name"],
                "precision": "int8" if hw.get("peak_tops_int8", 0) > 0 else "fp16",
            }
        )
    else:
        for w in workloads:
            precision = "int8" if hw.get("peak_tops_int8", 0) > 0 else "fp16"
            mappings.append(
                {
                    "workload": w["name"],
                    "model_class": w.get("model_class", "unknown"),
                    "target": hw["name"],
                    "precision": precision,
                }
            )
    return mappings


def _define_ip_blocks(workload: dict[str, Any], hw: dict[str, Any]) -> list[dict[str, Any]]:
    """Define IP blocks for the SoC."""
    blocks = [
        {
            "name": "cpu_subsystem",
            "type": "cpu",
            "description": "Application processor for control and orchestration",
            "config": {"cores": 2, "isa": "RISC-V", "frequency_mhz": 1000},
        },
        {
            "name": "compute_engine",
            "type": hw.get("type", "accelerator"),
            "description": f"{hw['name']} compute engine",
            "config": {
                "peak_tops_int8": hw.get("peak_tops_int8", 0),
                "peak_tflops_fp16": hw.get("peak_tflops_fp16", 0),
                "paradigm": hw.get("compute_paradigm", "unknown"),
            },
        },
        {
            "name": "memory_controller",
            "type": "memory",
            "description": "DDR/LPDDR memory controller",
            "config": {
                "type": "LPDDR4X",
                "capacity_gb": min(hw.get("memory_gb", 2), 4),
                "bandwidth_gbps": 25.6,
            },
        },
        {
            "name": "io_subsystem",
            "type": "io",
            "description": "Camera, sensor, and communication interfaces",
            "config": {"mipi_csi": 2, "i2c": 4, "spi": 2, "uart": 2, "gpio": 32},
        },
    ]

    # Add ISP if vision workload
    dominant = workload.get("dominant_op", "")
    if dominant in ("convolution", "feature_extraction") or any(
        w.get("name", "") in ("object_detection", "visual_perception", "visual_slam")
        for w in workload.get("workloads", [])
    ):
        blocks.append(
            {
                "name": "isp",
                "type": "isp",
                "description": "Image Signal Processor for camera input",
                "config": {"max_resolution": "4K", "pipeline_stages": 8},
            }
        )

    return blocks


def _define_memory_map(ip_blocks: list[dict[str, Any]], hw: dict[str, Any]) -> dict[str, Any]:
    """Define address space layout."""
    base = 0x0000_0000
    regions = []
    for block in ip_blocks:
        size = 0x0100_0000  # 16MB per block default
        regions.append(
            {
                "name": block["name"],
                "base_address": f"0x{base:08X}",
                "size_bytes": size,
            }
        )
        base += size

    # DRAM region
    dram_gb = min(hw.get("memory_gb", 2), 4)
    regions.append(
        {
            "name": "dram",
            "base_address": f"0x{0x8000_0000:08X}",
            "size_bytes": int(dram_gb * 1024 * 1024 * 1024),
        }
    )

    return {"regions": regions, "address_width_bits": 32}


def _define_interconnect(ip_blocks: list[dict[str, Any]]) -> dict[str, Any]:
    """Define bus/NoC topology."""
    return {
        "type": "AXI4" if len(ip_blocks) <= 6 else "NoC",
        "data_width_bits": 128,
        "frequency_mhz": 500,
        "topology": "star" if len(ip_blocks) <= 4 else "ring",
        "connected_blocks": [b["name"] for b in ip_blocks],
    }


# ---------------------------------------------------------------------------
# PPA Assessor
# ---------------------------------------------------------------------------


def ppa_assessor(task: TaskNode, state: SoCDesignState) -> dict[str, Any]:
    """Evaluate a proposed architecture against PPA constraints.

    Estimates power, performance (latency), and area from the architecture
    and hardware specs. Produces verdicts per constraint.

    Writes to state: ppa_metrics
    """
    constraints = get_constraints(state)
    process_nm = constraints.target_process_nm or 28
    architecture = state.get("selected_architecture", {})
    ip_blocks = state.get("ip_blocks", [])
    workload = state.get("workload_profile", {})
    candidates = state.get("hardware_candidates", [])

    # Also check dependency results
    dep_results = get_dependency_results(state, task)
    for result in dep_results.values():
        if "selected_architecture" in result:
            architecture = result["selected_architecture"]
        if "ip_blocks" in result:
            ip_blocks = result["ip_blocks"]

    # Find the selected hardware
    selected_hw = None
    for hw in candidates:
        if hw.get("name") == architecture.get("primary_compute"):
            selected_hw = hw
            break
    if selected_hw is None and candidates:
        selected_hw = candidates[0]

    # Estimate PPA
    power_w = _estimate_power(ip_blocks, selected_hw, workload, process_nm)
    latency_ms = _estimate_latency(workload, selected_hw, process_nm, ip_blocks)
    area_mm2 = _estimate_area(ip_blocks, constraints)

    # Manufacturing cost model (replaces static BOM lookup)
    volume = constraints.target_volume or 10_000
    mfg = estimate_manufacturing_cost(area_mm2, process_nm, volume)
    cost_usd = mfg.total_unit_cost_usd
    cost_breakdown = mfg.model_dump()

    memory_mb = sum(
        b.get("config", {}).get("capacity_gb", 0) * 1024
        for b in ip_blocks
        if b.get("type") == "memory"
    )

    # Generate verdicts
    verdicts = {}
    bottlenecks = []
    suggestions = []

    if constraints.max_power_watts and power_w is not None:
        if power_w <= constraints.max_power_watts:
            verdicts["power"] = "PASS"
        else:
            verdicts["power"] = "FAIL"
            bottlenecks.append(
                f"Power {power_w:.1f}W exceeds {constraints.max_power_watts}W budget"
            )
            suggestions.append("Consider lower-power accelerator or reducing clock frequency")

    if constraints.max_latency_ms and latency_ms is not None:
        if latency_ms <= constraints.max_latency_ms:
            verdicts["latency"] = "PASS"
        else:
            verdicts["latency"] = "FAIL"
            bottlenecks.append(
                f"Latency {latency_ms:.1f}ms exceeds {constraints.max_latency_ms}ms target"
            )
            suggestions.append("Consider INT8 quantization or model pruning")

    if constraints.max_cost_usd and cost_usd is not None:
        if cost_usd <= constraints.max_cost_usd:
            verdicts["cost"] = "PASS"
        else:
            verdicts["cost"] = "FAIL"
            bottlenecks.append(
                f"Cost ${cost_usd:.0f} exceeds ${constraints.max_cost_usd:.0f} budget"
            )

    if constraints.max_area_mm2 and area_mm2 is not None:
        if area_mm2 <= constraints.max_area_mm2:
            verdicts["area"] = "PASS"
        else:
            verdicts["area"] = "FAIL"
            bottlenecks.append(
                f"Area {area_mm2:.1f}mm² exceeds {constraints.max_area_mm2}mm² target"
            )

    all_pass = all(v == "PASS" for v in verdicts.values()) if verdicts else False
    overall = "PASS" if all_pass else "FAIL"

    ppa = PPAMetrics(
        power_watts=power_w,
        latency_ms=latency_ms,
        area_mm2=area_mm2,
        process_nm=process_nm,
        cost_usd=cost_usd,
        cost_breakdown=cost_breakdown,
        memory_mb=memory_mb,
        verdicts=verdicts,
        bottlenecks=bottlenecks,
        suggestions=suggestions,
    )

    return {
        "summary": f"PPA assessment: {overall} ({len(verdicts)} constraints checked)",
        "overall_verdict": overall,
        "ppa_metrics": ppa.model_dump(),
        "_state_updates": {"ppa_metrics": ppa.model_dump()},
    }


def _estimate_power(
    ip_blocks: list[dict[str, Any]],
    hw: dict[str, Any] | None,
    workload: dict[str, Any] | None = None,
    process_nm: int = 28,
) -> float:
    """Estimate total SoC power consumption.

    When ip_blocks contain physics-based block data (block_type + _power_watts),
    use per-block power sums. Otherwise falls back to TDP-based estimation.
    """
    # --- New path: physics-based IP block power ---
    if ip_blocks and _has_physics_blocks(ip_blocks):
        total = sum(b.get("_power_watts", 0.0) for b in ip_blocks)
        # Apply optimization scaling if workload was optimized
        if workload and workload.get("optimizations_applied"):
            opt_scale = _optimization_scale(workload)
            # Only scale compute blocks, not infra
            compute_types = {"cpu_core", "gpu_cluster", "kpu_tile"}
            compute_power = sum(
                b.get("_power_watts", 0.0)
                for b in ip_blocks
                if b.get("block_type") in compute_types
            )
            infra_power = total - compute_power
            total = compute_power * opt_scale + infra_power
        return round(total, 1)

    # --- Legacy path: TDP-based estimation ---
    if hw is None:
        return 10.0

    tdp = hw.get("tdp_watts", 10.0)
    base_utilization = 0.8

    optimization_scale = 1.0
    if workload and workload.get("optimizations_applied"):
        optimization_scale = _optimization_scale(workload)

    clock_scale = 1.0
    for block in ip_blocks:
        if block.get("type") in ("kpu", "gpu", "npu", "tpu", "accelerator"):
            config = block.get("config", {})
            freq = config.get("frequency_mhz", 1000)
            if freq < 1000:
                clock_scale = freq / 1000.0

    compute_w = tdp * base_utilization * optimization_scale * clock_scale
    cpu_w = 1.0
    io_w = 0.5
    memory_w = 0.8

    tech = get_technology(process_nm)
    voltage_scale = (tech.vdd_v / 0.90) ** 2
    compute_w *= voltage_scale
    cpu_w *= voltage_scale
    io_w *= voltage_scale
    memory_w *= voltage_scale

    return round(compute_w + cpu_w + io_w + memory_w, 1)


def _has_physics_blocks(ip_blocks: list[dict[str, Any]]) -> bool:
    """Check if ip_blocks contain physics-based block data (new format)."""
    return any("block_type" in b and "_area_mm2" in b for b in ip_blocks)


def _optimization_scale(workload: dict[str, Any]) -> float:
    """Compute power scaling factor from applied optimizations."""
    current_gflops = workload.get("total_estimated_gflops", workload.get("estimated_gflops", 0))
    reference_gflops = _reference_gflops(workload)
    if reference_gflops > 0 and current_gflops > 0:
        scale = min(current_gflops / reference_gflops, 1.0)
        return max(scale, 0.20)
    return 1.0


def _reference_gflops(workload: dict[str, Any]) -> float:
    """Get reference GFLOPS for a workload type (used for power scaling)."""
    # Sum up the workloads' GFLOPS from the model-class heuristics
    workloads = workload.get("workloads", [])
    if workloads:
        # Reference = sum of standard model GFLOPS for each workload type
        ref_map = {
            "object_detection": 8.7,
            "object_tracking": 0.5,
            "visual_perception": 6.0,
            "visual_slam": 3.0,
            "voice_recognition": 1.5,
            "lidar_processing": 4.0,
            "general_inference": 5.0,
        }
        total = sum(ref_map.get(w.get("name", ""), 5.0) for w in workloads)
        return total
    # Single-workload fallback
    return workload.get("total_estimated_gflops", workload.get("estimated_gflops", 5.0))


def _estimate_latency(
    workload: dict[str, Any],
    hw: dict[str, Any] | None,
    process_nm: int = 28,
    ip_blocks: list[dict[str, Any]] | None = None,
) -> float:
    """Estimate end-to-end inference latency.

    For heterogeneous SoCs with physics-based blocks, latency is the max
    across concurrent accelerators (they run in parallel).
    """
    if hw is None:
        return 50.0

    total_gflops = workload.get("total_estimated_gflops", workload.get("estimated_gflops", 5.0))
    tech = get_technology(process_nm)
    speed_scale = 25.0 / tech.gate_delay_ps
    overhead_ms = 2.0

    # --- New path: per-accelerator latency for heterogeneous blocks ---
    if ip_blocks and _has_physics_blocks(ip_blocks):
        compute_blocks = [
            b
            for b in ip_blocks
            if b.get("block_type") in ("cpu_core", "gpu_cluster", "kpu_tile")
            and b.get("_peak_gops", 0) > 0
        ]
        if compute_blocks:
            total_peak_gops = sum(b.get("_peak_gops", 0) for b in compute_blocks)
            utilization = 0.3
            if total_peak_gops > 0:
                latency_s = total_gflops / (total_peak_gops * utilization)
                latency_ms = latency_s * 1000.0
            else:
                latency_ms = 100.0
            latency_ms /= speed_scale
            return round(latency_ms + overhead_ms, 1)

    # --- Legacy path ---
    peak_tops = hw.get("peak_tops_int8", 1.0)
    if peak_tops <= 0:
        peak_tops = hw.get("peak_tflops_fp16", 1.0)

    utilization = 0.3
    if peak_tops > 0:
        latency_s = (total_gflops / 1000) / (peak_tops * utilization)
        latency_ms = latency_s * 1000
    else:
        latency_ms = 100.0

    latency_ms /= speed_scale
    return round(latency_ms + overhead_ms, 1)


def _estimate_area(ip_blocks: list[dict[str, Any]], constraints: DesignConstraints) -> float:
    """Estimate die area in mm².

    Uses physics-based per-block area when available (new format),
    falls back to type-based lookup table for legacy blocks.
    """
    process_nm = constraints.target_process_nm or 28

    # --- New path: physics-based IP block area ---
    if ip_blocks and _has_physics_blocks(ip_blocks):
        core_area = sum(b.get("_area_mm2", 0.0) for b in ip_blocks)
        return round(core_area * 1.2, 1)  # 20% interconnect overhead

    # --- Legacy path: type-based lookup ---
    area_per_type = {
        "cpu": 4.0,
        "kpu": 8.0,
        "gpu": 15.0,
        "npu": 10.0,
        "tpu": 12.0,
        "memory": 3.0,
        "io": 2.0,
        "isp": 3.0,
        "accelerator": 10.0,
    }

    scale = (process_nm / 28) ** 2
    total = sum(area_per_type.get(b.get("type", ""), 5.0) for b in ip_blocks)
    total *= scale
    total *= 1.2

    return round(total, 1)


# ---------------------------------------------------------------------------
# Critic
# ---------------------------------------------------------------------------


def critic(task: TaskNode, state: SoCDesignState) -> dict[str, Any]:
    """Review overall design quality and identify weaknesses.

    Acts as the "senior engineer" who challenges assumptions and
    identifies potential issues.

    Does not write to state — returns critique for planner/human review.
    """
    state.get("selected_architecture", {})
    ppa = state.get("ppa_metrics", {})
    constraints = get_constraints(state)
    workload = state.get("workload_profile", {})
    ip_blocks = state.get("ip_blocks", [])

    issues = []
    recommendations = []
    strengths = []

    # Check PPA verdicts
    verdicts = ppa.get("verdicts", {})
    failing = [k for k, v in verdicts.items() if v == "FAIL"]
    passing = [k for k, v in verdicts.items() if v == "PASS"]

    if failing:
        issues.append(f"Failing constraints: {', '.join(failing)}")
    if passing:
        strengths.append(f"Meeting constraints: {', '.join(passing)}")

    # Check for single point of failure
    compute_blocks = [b for b in ip_blocks if b.get("type") in ("kpu", "gpu", "npu", "tpu")]
    if len(compute_blocks) == 1:
        issues.append("Single compute engine — no fallback if accelerator fails or is overloaded")
        recommendations.append("Consider CPU fallback path for critical inference workloads")

    # Check power margin
    power_w = ppa.get("power_watts")
    max_power = constraints.max_power_watts
    if power_w and max_power:
        margin_pct = (1 - power_w / max_power) * 100
        if margin_pct < 10:
            issues.append(f"Tight power margin: only {margin_pct:.0f}% headroom")
            recommendations.append("Add power management modes (sleep/throttle)")
        elif margin_pct > 50:
            strengths.append(f"Generous power margin: {margin_pct:.0f}% headroom")

    # Check memory adequacy
    workload_mem = workload.get("total_estimated_memory_mb", workload.get("estimated_memory_mb", 0))
    available_mem = sum(
        b.get("config", {}).get("capacity_gb", 0) * 1024
        for b in ip_blocks
        if b.get("type") == "memory"
    )
    if workload_mem > 0 and available_mem > 0:
        if workload_mem > available_mem * 0.8:
            issues.append(
                f"Memory pressure: workload needs ~{workload_mem:.0f}MB, "
                f"available {available_mem:.0f}MB"
            )
            recommendations.append("Consider model compression or streaming execution")

    # Safety check
    if constraints.safety_critical and "redundancy" not in str(ip_blocks).lower():
        issues.append("Safety-critical application but no redundancy in architecture")
        recommendations.append("Add dual-lockstep CPU and ECC memory for safety compliance")

    # Overall assessment
    if not failing and not issues:
        assessment = "STRONG"
    elif not failing:
        assessment = "ADEQUATE"
    elif len(failing) <= 1:
        assessment = "NEEDS_WORK"
    else:
        assessment = "SIGNIFICANT_ISSUES"

    return {
        "summary": f"Design review: {assessment} ({len(issues)} issues, {len(strengths)} strengths)",
        "assessment": assessment,
        "issues": issues,
        "recommendations": recommendations,
        "strengths": strengths,
    }


# ---------------------------------------------------------------------------
# Report Generator
# ---------------------------------------------------------------------------


def report_generator(task: TaskNode, state: SoCDesignState) -> dict[str, Any]:
    """Generate a design report summarizing all artifacts and decisions.

    Produces a structured report dict (no file I/O for now).
    """
    architecture = state.get("selected_architecture", {})
    ppa = state.get("ppa_metrics", {})
    workload = state.get("workload_profile", {})
    candidates = state.get("hardware_candidates", [])
    ip_blocks = state.get("ip_blocks", [])
    history = state.get("history", [])
    rationale = state.get("design_rationale", [])

    # Find critic results in dependencies
    dep_results = get_dependency_results(state, task)
    critique = {}
    for result in dep_results.values():
        if "assessment" in result:
            critique = result

    report = {
        "title": f"SoC Design Report: {state.get('goal', 'Unknown')}",
        "session_id": state.get("session_id", "unknown"),
        "use_case": state.get("use_case", ""),
        "platform": state.get("platform", ""),
        "sections": {
            "executive_summary": {
                "goal": state.get("goal", ""),
                "architecture": architecture.get("name", "N/A"),
                "primary_compute": architecture.get("primary_compute", "N/A"),
                "overall_verdict": ppa.get("verdicts", {}),
                "assessment": critique.get("assessment", "N/A"),
            },
            "workload_analysis": workload,
            "hardware_exploration": {
                "candidates_evaluated": len(candidates),
                "top_candidates": candidates[:3] if candidates else [],
            },
            "architecture": {
                "description": architecture,
                "ip_blocks": ip_blocks,
                "memory_map": state.get("memory_map", {}),
                "interconnect": state.get("interconnect", {}),
            },
            "ppa_assessment": ppa,
            "design_review": critique,
            "decision_history": history,
            "design_rationale": rationale,
        },
    }

    return {
        "summary": "Design report generated",
        "report": report,
    }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_default_dispatcher() -> Dispatcher:
    """Create a Dispatcher pre-loaded with all specialist agents.

    Returns a Dispatcher with these agents registered:
    - workload_analyzer
    - hw_explorer
    - architecture_composer
    - ppa_assessor
    - critic
    - report_generator
    - design_optimizer
    - design_explorer (Pareto front)
    - safety_detector
    - experience_retriever
    """
    from embodied_ai_architect.graphs.optimizer import design_optimizer
    from embodied_ai_architect.graphs.kpu_specialists import (
        bandwidth_validator,
        floorplan_validator,
        kpu_configurator,
        kpu_optimizer,
    )
    from embodied_ai_architect.graphs.rtl_specialists import (
        rtl_generator,
        rtl_ppa_assessor,
    )
    from embodied_ai_architect.graphs.pareto import design_explorer
    from embodied_ai_architect.graphs.safety import safety_detector
    from embodied_ai_architect.graphs.experience_specialist import experience_retriever

    dispatcher = Dispatcher()

    specialists: dict = {
        "workload_analyzer": workload_analyzer,
        "hw_explorer": hw_explorer,
        "architecture_composer": architecture_composer,
        "ppa_assessor": ppa_assessor,
        "critic": critic,
        "report_generator": report_generator,
        "design_optimizer": design_optimizer,
        # KPU micro-architecture specialists
        "kpu_configurator": kpu_configurator,
        "floorplan_validator": floorplan_validator,
        "bandwidth_validator": bandwidth_validator,
        "kpu_optimizer": kpu_optimizer,
        # RTL specialists
        "rtl_generator": rtl_generator,
        "rtl_ppa_assessor": rtl_ppa_assessor,
        # Phase 4 specialists
        "design_explorer": design_explorer,
        "safety_detector": safety_detector,
        "experience_retriever": experience_retriever,
    }

    # MOO specialist (optional, requires numpy)
    try:
        from embodied_ai_architect.graphs.moo.specialist import moo_explorer

        specialists["moo_explorer"] = moo_explorer
    except ImportError:
        pass

    dispatcher.register_many(specialists)
    return dispatcher
