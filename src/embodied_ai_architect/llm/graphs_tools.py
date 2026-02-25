"""Tools integrating branes-ai/graphs for detailed model characterization.

Provides LLM-callable tools for:
- Detailed model analysis with roofline modeling
- Hardware comparison across 30+ targets
- Bottleneck identification
- Energy/power estimation

Verdict-first tools (NEW):
- analyze_latency: Check if model meets latency target
- analyze_power: Check if model meets power budget
- analyze_memory: Check if model fits in memory
- check_constraint: Generic constraint checking
"""

import json
import traceback
from typing import Any

# Optional import - graphs may not be installed
try:
    from graphs.analysis.unified_analyzer import UnifiedAnalyzer
    from graphs.hardware.resource_model import Precision
    from graphs.ir.structures import BottleneckType

    HAS_GRAPHS = True
except ImportError:
    HAS_GRAPHS = False
    UnifiedAnalyzer = None
    Precision = None
    BottleneckType = None

# Optional import - Pydantic adapters and schemas
try:
    from graphs.adapters import convert_to_pydantic
    from embodied_schemas import (
        Verdict,
        Confidence,
        GraphAnalysisResult,
    )

    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False
    convert_to_pydantic = None
    Verdict = None
    Confidence = None
    GraphAnalysisResult = None


# Hardware categories for easier discovery
HARDWARE_CATALOG = {
    "datacenter_gpu": [
        "H100-SXM5-80GB",
        "A100-SXM4-80GB",
        "A100-SXM4-40GB",
        "V100-SXM2-32GB",
        "L4",
        "T4",
    ],
    "edge_gpu": [
        "Jetson-Orin-AGX",
        "Jetson-Orin-Nano",
    ],
    "datacenter_cpu": [
        "Intel-Xeon-8490H",
        "AMD-EPYC-9654",
    ],
    "edge_cpu": [
        "Intel-i7-12700K",
    ],
    "tpu": [
        "TPU-v4",
        "Coral-Edge-TPU",
    ],
    "accelerators": [
        "KPU-T256",
        "KPU-T64",
        "Hailo-8",
    ],
    "automotive": [
        "TDA4VM",
        "TDA4VL",
    ],
}

# Flatten for quick lookup
ALL_HARDWARE = []
for category, hw_list in HARDWARE_CATALOG.items():
    ALL_HARDWARE.extend(hw_list)


def get_graphs_tool_definitions() -> list[dict[str, Any]]:
    """Get tool definitions for graphs-based analysis.

    Returns:
        List of tool definitions in Anthropic format
    """
    return [
        {
            "name": "analyze_model_detailed",
            "description": (
                "Perform detailed analysis of a neural network model on specific hardware "
                "using roofline modeling. Returns latency, energy, memory usage, hardware "
                "utilization, and bottleneck classification. More accurate than basic analysis. "
                "Supports 140+ models and 30+ hardware targets."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "model_name": {
                        "type": "string",
                        "description": (
                            "Model name (e.g., 'resnet18', 'resnet50', 'mobilenet_v2', "
                            "'efficientnet_b0', 'vit_b_16', 'yolov8n')"
                        ),
                    },
                    "hardware_name": {
                        "type": "string",
                        "description": (
                            "Hardware target (e.g., 'H100-SXM5-80GB', 'Jetson-Orin-AGX', "
                            "'TPU-v4', 'Coral-Edge-TPU', 'KPU-T256')"
                        ),
                    },
                    "batch_size": {
                        "type": "integer",
                        "description": "Batch size for inference (default: 1)",
                    },
                    "precision": {
                        "type": "string",
                        "enum": ["FP32", "FP16", "BF16", "INT8", "INT4", "TF32"],
                        "description": "Numerical precision (default: FP32)",
                    },
                },
                "required": ["model_name", "hardware_name"],
            },
        },
        {
            "name": "compare_hardware_targets",
            "description": (
                "Compare a model's performance across multiple hardware targets. "
                "Returns a ranked comparison by latency, energy, and efficiency. "
                "Useful for finding the best hardware for a specific model."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "model_name": {
                        "type": "string",
                        "description": "Model name to analyze",
                    },
                    "hardware_targets": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "List of hardware targets to compare. If not specified, "
                            "uses a default set of representative hardware."
                        ),
                    },
                    "batch_size": {
                        "type": "integer",
                        "description": "Batch size for inference (default: 1)",
                    },
                },
                "required": ["model_name"],
            },
        },
        {
            "name": "identify_bottleneck",
            "description": (
                "Identify whether a model is compute-bound, memory-bound, or balanced "
                "on specific hardware using roofline analysis. Provides optimization "
                "recommendations based on the bottleneck type."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "model_name": {
                        "type": "string",
                        "description": "Model name to analyze",
                    },
                    "hardware_name": {
                        "type": "string",
                        "description": "Hardware target",
                    },
                    "batch_size": {
                        "type": "integer",
                        "description": "Batch size (default: 1)",
                    },
                },
                "required": ["model_name", "hardware_name"],
            },
        },
        {
            "name": "list_available_hardware",
            "description": (
                "List all available hardware targets for analysis, organized by category "
                "(datacenter GPU, edge GPU, CPU, TPU, accelerators, automotive)."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": [
                            "all",
                            "datacenter_gpu",
                            "edge_gpu",
                            "datacenter_cpu",
                            "edge_cpu",
                            "tpu",
                            "accelerators",
                            "automotive",
                        ],
                        "description": "Filter by category (default: all)",
                    },
                },
                "required": [],
            },
        },
        {
            "name": "estimate_power_consumption",
            "description": (
                "Estimate power and energy consumption for running a model on hardware. "
                "Breaks down into compute energy, memory energy, and static/leakage power."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "model_name": {
                        "type": "string",
                        "description": "Model name",
                    },
                    "hardware_name": {
                        "type": "string",
                        "description": "Hardware target",
                    },
                    "batch_size": {
                        "type": "integer",
                        "description": "Batch size (default: 1)",
                    },
                    "inferences_per_second": {
                        "type": "number",
                        "description": "Target inference rate for power calculation",
                    },
                },
                "required": ["model_name", "hardware_name"],
            },
        },
        # === Verdict-first tools (NEW) ===
        {
            "name": "check_latency",
            "description": (
                "Check if a model meets a latency target on specific hardware. "
                "Returns verdict (PASS/FAIL), confidence level, and actionable suggestions. "
                "Use this when you need a clear yes/no answer about latency requirements."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "model_name": {
                        "type": "string",
                        "description": "Model name (e.g., 'resnet18', 'mobilenet_v2')",
                    },
                    "hardware_name": {
                        "type": "string",
                        "description": "Hardware target (e.g., 'H100', 'Jetson-Orin-AGX')",
                    },
                    "latency_target_ms": {
                        "type": "number",
                        "description": "Required latency in milliseconds",
                    },
                    "batch_size": {
                        "type": "integer",
                        "description": "Batch size (default: 1)",
                    },
                    "precision": {
                        "type": "string",
                        "enum": ["FP32", "FP16", "INT8"],
                        "description": "Numerical precision (default: FP32)",
                    },
                },
                "required": ["model_name", "hardware_name", "latency_target_ms"],
            },
        },
        {
            "name": "check_power",
            "description": (
                "Check if a model meets a power budget on specific hardware. "
                "Returns verdict (PASS/FAIL), confidence level, and actionable suggestions. "
                "Use this for power-constrained deployments."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "model_name": {
                        "type": "string",
                        "description": "Model name",
                    },
                    "hardware_name": {
                        "type": "string",
                        "description": "Hardware target",
                    },
                    "power_budget_w": {
                        "type": "number",
                        "description": "Maximum power budget in watts",
                    },
                    "batch_size": {
                        "type": "integer",
                        "description": "Batch size (default: 1)",
                    },
                },
                "required": ["model_name", "hardware_name", "power_budget_w"],
            },
        },
        {
            "name": "check_memory",
            "description": (
                "Check if a model fits within a memory budget on specific hardware. "
                "Returns verdict (PASS/FAIL), confidence level, and memory breakdown. "
                "Use this for memory-constrained edge deployments."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "model_name": {
                        "type": "string",
                        "description": "Model name",
                    },
                    "hardware_name": {
                        "type": "string",
                        "description": "Hardware target",
                    },
                    "memory_budget_mb": {
                        "type": "number",
                        "description": "Maximum memory budget in MB",
                    },
                    "batch_size": {
                        "type": "integer",
                        "description": "Batch size (default: 1)",
                    },
                },
                "required": ["model_name", "hardware_name", "memory_budget_mb"],
            },
        },
        {
            "name": "full_analysis",
            "description": (
                "Perform comprehensive analysis with optional constraint checking. "
                "Returns complete breakdown (roofline, energy, memory) with verdict. "
                "Use this when you need detailed metrics alongside pass/fail status."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "model_name": {
                        "type": "string",
                        "description": "Model name",
                    },
                    "hardware_name": {
                        "type": "string",
                        "description": "Hardware target",
                    },
                    "constraint_metric": {
                        "type": "string",
                        "enum": ["latency", "power", "memory", "energy"],
                        "description": "Metric to check against threshold",
                    },
                    "constraint_threshold": {
                        "type": "number",
                        "description": "Threshold value (ms for latency, W for power, MB for memory, mJ for energy)",
                    },
                    "batch_size": {
                        "type": "integer",
                        "description": "Batch size (default: 1)",
                    },
                    "precision": {
                        "type": "string",
                        "enum": ["FP32", "FP16", "INT8"],
                        "description": "Numerical precision (default: FP32)",
                    },
                },
                "required": ["model_name", "hardware_name"],
            },
        },
    ]


def _check_graphs_available() -> str | None:
    """Check if graphs package is available."""
    if not HAS_GRAPHS:
        return (
            "Error: graphs package not installed. "
            "Install from: /home/stillwater/dev/branes/clones/graphs"
        )
    return None


def _get_precision(precision_str: str | None) -> "Precision":
    """Convert precision string to Precision enum."""
    if precision_str is None:
        return Precision.FP32

    precision_map = {
        "FP32": Precision.FP32,
        "FP16": Precision.FP16,
        "BF16": Precision.BF16,
        "INT8": Precision.INT8,
        "INT4": Precision.INT4,
        "TF32": Precision.TF32,
    }
    return precision_map.get(precision_str.upper(), Precision.FP32)


def analyze_model_detailed(
    model_name: str,
    hardware_name: str,
    batch_size: int = 1,
    precision: str | None = None,
) -> str:
    """Perform detailed model analysis using graphs UnifiedAnalyzer."""
    error = _check_graphs_available()
    if error:
        return error

    try:
        analyzer = UnifiedAnalyzer(verbose=False)
        prec = _get_precision(precision)

        result = analyzer.analyze_model(
            model_name=model_name,
            hardware_name=hardware_name,
            batch_size=batch_size,
            precision=prec,
        )

        # Format result
        roofline = result.roofline_report

        # Determine dominant bottleneck from subgraph counts
        if roofline:
            if roofline.num_compute_bound > roofline.num_memory_bound:
                dominant = "compute_bound"
            elif roofline.num_memory_bound > roofline.num_compute_bound:
                dominant = "memory_bound"
            else:
                dominant = "balanced"
        else:
            dominant = "unknown"

        output = {
            "model": result.model_name,
            "hardware": result.hardware_name,
            "batch_size": batch_size,
            "precision": precision or "FP32",
            "metrics": {
                "latency_ms": round(result.total_latency_ms, 3),
                "throughput_fps": round(result.throughput_fps, 1),
                "energy_mj": round(result.total_energy_mj, 3),
                "peak_memory_mb": round(result.peak_memory_mb, 1),
                "average_utilization_pct": round(result.average_utilization_pct, 1),
            },
            "bottleneck": {
                "type": dominant,
                "compute_bound_ops": roofline.num_compute_bound if roofline else 0,
                "memory_bound_ops": roofline.num_memory_bound if roofline else 0,
                "balanced_ops": roofline.num_balanced if roofline else 0,
                "avg_compute_utilization": (
                    round(roofline.average_flops_utilization * 100, 1) if roofline else 0
                ),
                "avg_memory_utilization": (
                    round(roofline.average_bandwidth_utilization * 100, 1) if roofline else 0
                ),
            },
        }

        return json.dumps(output, indent=2)

    except Exception as e:
        return (
            f"Error analyzing {model_name} on {hardware_name}: {str(e)}\n{traceback.format_exc()}"
        )


def compare_hardware_targets(
    model_name: str,
    hardware_targets: list[str] | None = None,
    batch_size: int = 1,
) -> str:
    """Compare model performance across multiple hardware targets."""
    error = _check_graphs_available()
    if error:
        return error

    # Default hardware set for comparison
    if hardware_targets is None:
        hardware_targets = [
            "H100-SXM5-80GB",
            "A100-SXM4-80GB",
            "Jetson-Orin-AGX",
            "Jetson-Orin-Nano",
            "TPU-v4",
            "Coral-Edge-TPU",
        ]

    try:
        analyzer = UnifiedAnalyzer(verbose=False)
        results = []

        for hw in hardware_targets:
            try:
                result = analyzer.analyze_model(
                    model_name=model_name,
                    hardware_name=hw,
                    batch_size=batch_size,
                )
                roofline = result.roofline_report
                # Calculate efficiency: FPS per Watt
                # Power (W) = Energy (mJ) / Time (ms) = Energy (J) / Time (s)
                power_w = result.total_energy_mj / result.total_latency_ms  # mJ/ms = W
                efficiency = result.throughput_fps / power_w if power_w > 0 else 0

                # Determine dominant bottleneck
                if roofline:
                    if roofline.num_compute_bound > roofline.num_memory_bound:
                        dominant = "compute_bound"
                    elif roofline.num_memory_bound > roofline.num_compute_bound:
                        dominant = "memory_bound"
                    else:
                        dominant = "balanced"
                else:
                    dominant = "unknown"

                results.append(
                    {
                        "hardware": result.hardware_name,
                        "latency_ms": round(result.total_latency_ms, 3),
                        "throughput_fps": round(result.throughput_fps, 1),
                        "energy_mj": round(result.total_energy_mj, 3),
                        "efficiency_fps_per_watt": round(efficiency, 2),
                        "bottleneck": dominant,
                    }
                )
            except Exception as e:
                results.append(
                    {
                        "hardware": hw,
                        "error": str(e),
                    }
                )

        # Sort by latency (fastest first)
        valid_results = [r for r in results if "error" not in r]
        valid_results.sort(key=lambda x: x["latency_ms"])

        # Add ranking
        for i, r in enumerate(valid_results):
            r["rank"] = i + 1

        output = {
            "model": model_name,
            "batch_size": batch_size,
            "comparison": valid_results,
            "errors": [r for r in results if "error" in r],
            "fastest": valid_results[0]["hardware"] if valid_results else None,
            "most_efficient": (
                max(valid_results, key=lambda x: x["efficiency_fps_per_watt"])["hardware"]
                if valid_results
                else None
            ),
        }

        return json.dumps(output, indent=2)

    except Exception as e:
        return f"Error comparing hardware: {str(e)}\n{traceback.format_exc()}"


def identify_bottleneck(
    model_name: str,
    hardware_name: str,
    batch_size: int = 1,
) -> str:
    """Identify performance bottleneck using roofline analysis."""
    error = _check_graphs_available()
    if error:
        return error

    try:
        analyzer = UnifiedAnalyzer(verbose=False)
        result = analyzer.analyze_model(
            model_name=model_name,
            hardware_name=hardware_name,
            batch_size=batch_size,
        )

        roofline = result.roofline_report
        if not roofline:
            return json.dumps(
                {"error": "Roofline analysis not available for this model/hardware combination"}
            )

        # Determine dominant bottleneck from counts
        if roofline.num_compute_bound > roofline.num_memory_bound:
            bottleneck_type = BottleneckType.COMPUTE_BOUND
        elif roofline.num_memory_bound > roofline.num_compute_bound:
            bottleneck_type = BottleneckType.MEMORY_BOUND
        else:
            bottleneck_type = BottleneckType.BALANCED

        # Generate recommendations based on bottleneck
        recommendations = []
        if bottleneck_type == BottleneckType.MEMORY_BOUND:
            recommendations = [
                "Consider using lower precision (FP16, INT8) to reduce memory bandwidth",
                "Look for operator fusion opportunities to reduce data movement",
                "Consider model pruning to reduce memory traffic",
                "Try batching to improve compute-to-memory ratio",
            ]
        elif bottleneck_type == BottleneckType.COMPUTE_BOUND:
            recommendations = [
                "Model is efficiently using compute resources",
                "Consider quantization (INT8, INT4) for higher throughput",
                "Look for more powerful hardware if lower latency needed",
                "Consider model distillation for a smaller, faster model",
            ]
        elif bottleneck_type == BottleneckType.BANDWIDTH_BOUND:
            recommendations = [
                "Memory bandwidth is the limiting factor",
                "Consider hardware with higher memory bandwidth",
                "Optimize data layout for better cache utilization",
                "Consider model compression techniques",
            ]
        else:  # BALANCED
            recommendations = [
                "Model is well-balanced between compute and memory",
                "Current hardware is a good fit for this workload",
                "Scaling batch size may improve throughput",
            ]

        output = {
            "model": result.model_name,
            "hardware": result.hardware_name,
            "batch_size": batch_size,
            "bottleneck_analysis": {
                "type": bottleneck_type.value,
                "compute_bound_ops": roofline.num_compute_bound,
                "memory_bound_ops": roofline.num_memory_bound,
                "balanced_ops": roofline.num_balanced,
                "total_compute_time_ms": round(roofline.total_compute_time * 1000, 4),
                "total_memory_time_ms": round(roofline.total_memory_time * 1000, 4),
                "avg_compute_utilization_pct": round(roofline.average_flops_utilization * 100, 1),
                "avg_memory_utilization_pct": round(
                    roofline.average_bandwidth_utilization * 100, 1
                ),
            },
            "interpretation": _interpret_bottleneck(bottleneck_type, roofline),
            "recommendations": recommendations,
        }

        return json.dumps(output, indent=2)

    except Exception as e:
        return f"Error analyzing bottleneck: {str(e)}\n{traceback.format_exc()}"


def _interpret_bottleneck(bottleneck_type: "BottleneckType", roofline) -> str:
    """Generate human-readable interpretation of bottleneck."""
    compute_ms = roofline.total_compute_time * 1000
    memory_ms = roofline.total_memory_time * 1000

    if bottleneck_type == BottleneckType.MEMORY_BOUND:
        return (
            f"This model is MEMORY-BOUND on this hardware. "
            f"{roofline.num_memory_bound} of {roofline.num_memory_bound + roofline.num_compute_bound + roofline.num_balanced} "
            f"operators are memory-limited. Total memory time is {memory_ms:.3f}ms vs "
            f"compute time of {compute_ms:.3f}ms. The GPU spends most time waiting for data."
        )
    elif bottleneck_type == BottleneckType.COMPUTE_BOUND:
        return (
            f"This model is COMPUTE-BOUND. "
            f"{roofline.num_compute_bound} of {roofline.num_memory_bound + roofline.num_compute_bound + roofline.num_balanced} "
            f"operators are compute-limited. Total compute time is {compute_ms:.3f}ms vs "
            f"memory time of {memory_ms:.3f}ms. Hardware is well-utilized at "
            f"{roofline.average_flops_utilization*100:.1f}% of peak compute."
        )
    elif bottleneck_type == BottleneckType.BANDWIDTH_BOUND:
        return (
            f"This model is BANDWIDTH-BOUND. Memory bandwidth is saturated at "
            f"{roofline.average_bandwidth_utilization*100:.1f}% utilization. Consider hardware "
            f"with higher memory bandwidth (HBM, wider bus)."
        )
    else:
        return (
            f"This model is BALANCED between compute and memory on this hardware. "
            f"Both resources are being utilized effectively with compute at "
            f"{roofline.average_flops_utilization*100:.1f}% and memory at "
            f"{roofline.average_bandwidth_utilization*100:.1f}% utilization."
        )


def list_available_hardware(category: str = "all") -> str:
    """List available hardware targets by category."""
    if category == "all":
        output = {
            "total_hardware_targets": len(ALL_HARDWARE),
            "categories": HARDWARE_CATALOG,
        }
    elif category in HARDWARE_CATALOG:
        output = {
            "category": category,
            "hardware": HARDWARE_CATALOG[category],
        }
    else:
        output = {
            "error": f"Unknown category: {category}",
            "available_categories": list(HARDWARE_CATALOG.keys()),
        }

    return json.dumps(output, indent=2)


def estimate_power_consumption(
    model_name: str,
    hardware_name: str,
    batch_size: int = 1,
    inferences_per_second: float | None = None,
) -> str:
    """Estimate power consumption for model inference."""
    error = _check_graphs_available()
    if error:
        return error

    try:
        analyzer = UnifiedAnalyzer(verbose=False)
        result = analyzer.analyze_model(
            model_name=model_name,
            hardware_name=hardware_name,
            batch_size=batch_size,
        )

        # Calculate power from energy and latency
        latency_s = result.total_latency_ms / 1000.0
        energy_j = result.total_energy_mj / 1000.0

        # Dynamic power during inference
        dynamic_power_w = energy_j / latency_s

        # If target inference rate specified, calculate sustained power
        if inferences_per_second:
            inference_time_fraction = (result.total_latency_ms / 1000.0) * inferences_per_second
            if inference_time_fraction > 1.0:
                sustained_power_w = dynamic_power_w  # Can't keep up
                can_sustain = False
            else:
                # Assume idle power is 10% of dynamic
                idle_power_w = dynamic_power_w * 0.1
                sustained_power_w = (
                    inference_time_fraction * dynamic_power_w
                    + (1 - inference_time_fraction) * idle_power_w
                )
                can_sustain = True
        else:
            inferences_per_second = 1000.0 / result.total_latency_ms
            sustained_power_w = dynamic_power_w
            can_sustain = True

        output = {
            "model": model_name,
            "hardware": hardware_name,
            "batch_size": batch_size,
            "power_analysis": {
                "energy_per_inference_mj": round(result.total_energy_mj, 3),
                "dynamic_power_w": round(dynamic_power_w, 2),
                "max_throughput_fps": round(1000.0 / result.total_latency_ms, 1),
            },
        }

        if inferences_per_second:
            output["target_rate_analysis"] = {
                "target_fps": inferences_per_second,
                "can_sustain": can_sustain,
                "sustained_power_w": round(sustained_power_w, 2),
                "energy_per_second_j": round(
                    result.total_energy_mj / 1000.0 * inferences_per_second, 3
                ),
            }

        # Energy breakdown if available
        if hasattr(result, "energy_report") and result.energy_report:
            er = result.energy_report
            output["energy_breakdown"] = {
                "compute_energy_mj": round(er.compute_energy_mj, 3),
                "memory_energy_mj": round(er.memory_energy_mj, 3),
                "static_energy_mj": round(er.static_energy_mj, 3),
            }

        return json.dumps(output, indent=2)

    except Exception as e:
        return f"Error estimating power: {str(e)}\n{traceback.format_exc()}"


# =============================================================================
# Verdict-First Tool Executors (NEW)
# =============================================================================


def _check_pydantic_available() -> str | None:
    """Check if Pydantic adapters are available."""
    if not HAS_PYDANTIC:
        return (
            "Error: embodied-schemas or graphs[schemas] not installed. "
            "Install with: pip install embodied-schemas"
        )
    return None


def _format_verdict_result(result: "GraphAnalysisResult") -> str:
    """Format GraphAnalysisResult as JSON for LLM consumption."""
    # Convert to dict, handling datetime
    output = {
        "verdict": result.verdict.value,
        "confidence": result.confidence.value,
        "summary": result.summary,
        "model_id": result.model_id,
        "hardware_id": result.hardware_id,
        "batch_size": result.batch_size,
        "precision": result.precision,
        "metrics": {
            "latency_ms": round(result.latency_ms, 3),
            "throughput_fps": round(result.throughput_fps, 1),
            "energy_per_inference_mj": round(result.energy_per_inference_mj, 3),
            "peak_memory_mb": round(result.peak_memory_mb, 1),
        },
        "roofline": {
            "bottleneck": result.roofline.bottleneck.value,
            "utilization_pct": round(result.roofline.utilization_pct, 1),
            "arithmetic_intensity": round(result.roofline.arithmetic_intensity, 1),
        },
        "energy": {
            "compute_mj": round(result.energy.compute_energy_mj, 3),
            "memory_mj": round(result.energy.memory_energy_mj, 3),
            "static_mj": round(result.energy.static_energy_mj, 3),
            "average_power_w": round(result.energy.average_power_w, 1),
        },
        "memory": {
            "weights_mb": round(result.memory.weights_mb, 1),
            "activations_mb": round(result.memory.activations_mb, 1),
            "fits_in_l2": result.memory.fits_in_l2,
            "fits_in_device_memory": result.memory.fits_in_device_memory,
        },
    }

    # Add constraint info if present
    if result.constraint_metric:
        output["constraint"] = {
            "metric": result.constraint_metric,
            "threshold": result.constraint_threshold,
            "actual": result.constraint_actual,
            "margin_pct": (
                round(result.constraint_margin_pct, 1) if result.constraint_margin_pct else None
            ),
        }

    # Add suggestions if present
    if result.suggestions:
        output["suggestions"] = result.suggestions

    # Add warnings if present
    if result.warnings:
        output["warnings"] = result.warnings

    return json.dumps(output, indent=2)


def check_latency(
    model_name: str,
    hardware_name: str,
    latency_target_ms: float,
    batch_size: int = 1,
    precision: str | None = None,
) -> str:
    """Check if a model meets a latency target (verdict-first)."""
    error = _check_graphs_available()
    if error:
        return error
    error = _check_pydantic_available()
    if error:
        return error

    try:
        analyzer = UnifiedAnalyzer(verbose=False)
        prec = _get_precision(precision)

        result = analyzer.analyze_model(
            model_name=model_name,
            hardware_name=hardware_name,
            batch_size=batch_size,
            precision=prec,
        )

        # Convert to Pydantic with latency constraint
        pydantic_result = convert_to_pydantic(
            result,
            constraint_metric="latency",
            constraint_threshold=latency_target_ms,
        )

        return _format_verdict_result(pydantic_result)

    except Exception as e:
        return json.dumps(
            {
                "verdict": "UNKNOWN",
                "error": str(e),
                "traceback": traceback.format_exc(),
            },
            indent=2,
        )


def check_power(
    model_name: str,
    hardware_name: str,
    power_budget_w: float,
    batch_size: int = 1,
) -> str:
    """Check if a model meets a power budget (verdict-first)."""
    error = _check_graphs_available()
    if error:
        return error
    error = _check_pydantic_available()
    if error:
        return error

    try:
        analyzer = UnifiedAnalyzer(verbose=False)

        result = analyzer.analyze_model(
            model_name=model_name,
            hardware_name=hardware_name,
            batch_size=batch_size,
        )

        # Convert to Pydantic with power constraint
        pydantic_result = convert_to_pydantic(
            result,
            constraint_metric="power",
            constraint_threshold=power_budget_w,
        )

        return _format_verdict_result(pydantic_result)

    except Exception as e:
        return json.dumps(
            {
                "verdict": "UNKNOWN",
                "error": str(e),
                "traceback": traceback.format_exc(),
            },
            indent=2,
        )


def check_memory(
    model_name: str,
    hardware_name: str,
    memory_budget_mb: float,
    batch_size: int = 1,
) -> str:
    """Check if a model fits within a memory budget (verdict-first)."""
    error = _check_graphs_available()
    if error:
        return error
    error = _check_pydantic_available()
    if error:
        return error

    try:
        analyzer = UnifiedAnalyzer(verbose=False)

        result = analyzer.analyze_model(
            model_name=model_name,
            hardware_name=hardware_name,
            batch_size=batch_size,
        )

        # Convert to Pydantic with memory constraint
        pydantic_result = convert_to_pydantic(
            result,
            constraint_metric="memory",
            constraint_threshold=memory_budget_mb,
        )

        return _format_verdict_result(pydantic_result)

    except Exception as e:
        return json.dumps(
            {
                "verdict": "UNKNOWN",
                "error": str(e),
                "traceback": traceback.format_exc(),
            },
            indent=2,
        )


def full_analysis(
    model_name: str,
    hardware_name: str,
    constraint_metric: str | None = None,
    constraint_threshold: float | None = None,
    batch_size: int = 1,
    precision: str | None = None,
) -> str:
    """Perform comprehensive analysis with optional constraint checking."""
    error = _check_graphs_available()
    if error:
        return error
    error = _check_pydantic_available()
    if error:
        return error

    try:
        analyzer = UnifiedAnalyzer(verbose=False)
        prec = _get_precision(precision)

        result = analyzer.analyze_model(
            model_name=model_name,
            hardware_name=hardware_name,
            batch_size=batch_size,
            precision=prec,
        )

        # Convert to Pydantic with optional constraint
        pydantic_result = convert_to_pydantic(
            result,
            constraint_metric=constraint_metric,
            constraint_threshold=constraint_threshold,
        )

        return _format_verdict_result(pydantic_result)

    except Exception as e:
        return json.dumps(
            {
                "verdict": "UNKNOWN",
                "error": str(e),
                "traceback": traceback.format_exc(),
            },
            indent=2,
        )


# =============================================================================
# Tool Executor Registration
# =============================================================================


def create_graphs_tool_executors() -> dict[str, callable]:
    """Create executor functions for graphs tools.

    Returns:
        Dictionary mapping tool names to executor functions
    """
    executors = {
        # Original tools
        "analyze_model_detailed": analyze_model_detailed,
        "compare_hardware_targets": compare_hardware_targets,
        "identify_bottleneck": identify_bottleneck,
        "list_available_hardware": list_available_hardware,
        "estimate_power_consumption": estimate_power_consumption,
        # Verdict-first tools (NEW)
        "check_latency": check_latency,
        "check_power": check_power,
        "check_memory": check_memory,
        "full_analysis": full_analysis,
    }
    return executors
