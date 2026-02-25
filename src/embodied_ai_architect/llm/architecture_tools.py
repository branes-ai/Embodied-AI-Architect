"""Tools for architecture-level analysis of embodied AI systems.

Provides LLM-callable tools for:
- End-to-end architecture analysis on hardware targets
- Multi-rate scheduling feasibility checking
- Bottleneck identification at the architecture level
- Architecture listing and visualization
"""

import json
import traceback
from datetime import datetime
from typing import Any

# Import from embodied-schemas
try:
    from embodied_schemas import (
        Registry,
        SoftwareArchitecture,
        OperatorEntry,
        Verdict,
        Confidence,
        ExecutionTarget,
        EXECUTION_TARGET_INFO,
        architecture_to_mermaid,
    )
    from embodied_schemas.scheduling import (
        ArchitectureAnalysisResult,
        SchedulingAnalysisResult,
        OperatorTiming,
        RateFeasibility,
        ExecutionTargetUtilization,
        DataMovementAnalysis,
        DataTransfer,
        OperatorBenchmarkResult,
        ArchitectureBenchmarkResult,
    )

    HAS_SCHEMAS = True
except ImportError:
    HAS_SCHEMAS = False
    Registry = None
    SoftwareArchitecture = None
    ExecutionTarget = None
    EXECUTION_TARGET_INFO = {}


def _check_schemas_available() -> str | None:
    """Check if embodied-schemas is available."""
    if not HAS_SCHEMAS:
        return (
            "Error: embodied-schemas not installed or outdated. "
            "Install with: pip install -e ../embodied-schemas"
        )
    return None


def _normalize_execution_target(target: str | None) -> str:
    """Normalize execution target string to standard form.

    Args:
        target: Execution target string (e.g., 'gpu', 'GPU', 'kpu', None)

    Returns:
        Normalized lowercase target string, defaults to 'cpu' if None
    """
    if target is None:
        return "cpu"

    # Normalize to lowercase
    normalized = target.lower().strip()

    # Map common aliases
    aliases = {
        # GPU aliases
        "cuda": "gpu",
        "rocm": "gpu",
        "hip": "gpu",
        "radeon": "gpu",
        # NPU aliases
        "neural": "npu",
        "xdna": "npu",
        "ryzen_ai": "npu",
        "ryzenai": "npu",
        "hexagon_npu": "npu",
        "neural_engine": "npu",
        # TPU aliases
        "tensor": "tpu",
        "edge_tpu": "tpu",
        # Other accelerators
        "knowledge": "kpu",
        "vision": "vpu",
        "hailo": "cvu",
        "mobileye": "cvu",
        "eyeq": "cvu",
        "hexagon": "dsp",
        "hexagon_dsp": "dsp",
    }

    return aliases.get(normalized, normalized)


def _get_target_display_name(target: str) -> str:
    """Get display name for an execution target.

    Args:
        target: Normalized execution target string

    Returns:
        Human-readable display name
    """
    if ExecutionTarget is None:
        return target.upper()

    try:
        exec_target = ExecutionTarget(target)
        info = EXECUTION_TARGET_INFO.get(exec_target, {})
        return info.get("name", target.upper())
    except ValueError:
        return target.upper()


def _get_registry() -> "Registry":
    """Get or create the global registry instance."""
    return Registry.load()


def get_architecture_tool_definitions() -> list[dict[str, Any]]:
    """Get tool definitions for architecture analysis.

    Returns:
        List of tool definitions in Anthropic format
    """
    return [
        {
            "name": "list_architectures",
            "description": (
                "List all available software architectures in the catalog. "
                "Returns architecture IDs, names, platform types, and operator counts."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "platform_type": {
                        "type": "string",
                        "description": (
                            "Filter by platform type: drone, vehicle, manipulator, etc. "
                            "If not specified, returns all architectures."
                        ),
                    },
                },
                "required": [],
            },
        },
        {
            "name": "show_architecture",
            "description": (
                "Show detailed information about a specific architecture, including "
                "operators, dataflow edges, variants, and a Mermaid diagram."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "architecture_id": {
                        "type": "string",
                        "description": "Architecture ID (e.g., 'drone_perception_v1')",
                    },
                },
                "required": ["architecture_id"],
            },
        },
        {
            "name": "analyze_architecture",
            "description": (
                "Analyze a software architecture on a specific hardware target. "
                "Returns end-to-end latency, power, memory, critical path, and bottleneck. "
                "Use this to determine if an architecture meets its timing requirements."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "architecture_id": {
                        "type": "string",
                        "description": "Architecture ID (e.g., 'drone_perception_v1')",
                    },
                    "hardware_id": {
                        "type": "string",
                        "description": "Hardware ID (e.g., 'Jetson-Orin-Nano')",
                    },
                    "variant_id": {
                        "type": "string",
                        "description": "Optional variant ID to use instead of base architecture",
                    },
                },
                "required": ["architecture_id", "hardware_id"],
            },
        },
        {
            "name": "check_scheduling",
            "description": (
                "Check if all operators in an architecture can meet their target rates "
                "on a specific hardware platform. Returns PASS/FAIL verdict with "
                "per-operator feasibility analysis."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "architecture_id": {
                        "type": "string",
                        "description": "Architecture ID",
                    },
                    "hardware_id": {
                        "type": "string",
                        "description": "Hardware ID",
                    },
                    "variant_id": {
                        "type": "string",
                        "description": "Optional variant ID",
                    },
                },
                "required": ["architecture_id", "hardware_id"],
            },
        },
        {
            "name": "identify_architecture_bottleneck",
            "description": (
                "Identify the bottleneck operator in an architecture that limits overall "
                "throughput. Returns the bottleneck operator, its latency contribution, "
                "and classification (compute-bound, memory-bound, or I/O-bound)."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "architecture_id": {
                        "type": "string",
                        "description": "Architecture ID",
                    },
                    "hardware_id": {
                        "type": "string",
                        "description": "Hardware ID",
                    },
                    "variant_id": {
                        "type": "string",
                        "description": "Optional variant ID",
                    },
                },
                "required": ["architecture_id", "hardware_id"],
            },
        },
        {
            "name": "suggest_optimizations",
            "description": (
                "Suggest optimizations for an architecture on a hardware target. "
                "Returns actionable recommendations: operator swaps (e.g., YOLOv8s→YOLOv8n), "
                "quantization options, hardware upgrades, and parallelization opportunities."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "architecture_id": {
                        "type": "string",
                        "description": "Architecture ID",
                    },
                    "hardware_id": {
                        "type": "string",
                        "description": "Hardware ID",
                    },
                    "variant_id": {
                        "type": "string",
                        "description": "Optional variant ID",
                    },
                    "target_latency_ms": {
                        "type": "number",
                        "description": "Target latency to achieve (optional, uses architecture default)",
                    },
                },
                "required": ["architecture_id", "hardware_id"],
            },
        },
        {
            "name": "compare_variants",
            "description": (
                "Compare architecture variants on a hardware target. Returns a table "
                "showing latency, power, memory, accuracy trade-offs between variants."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "architecture_id": {
                        "type": "string",
                        "description": "Architecture ID",
                    },
                    "hardware_id": {
                        "type": "string",
                        "description": "Hardware ID",
                    },
                    "variant_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of variant IDs to compare (optional, compares all if not specified)",
                    },
                },
                "required": ["architecture_id", "hardware_id"],
            },
        },
        {
            "name": "benchmark_architecture",
            "description": (
                "Run a benchmark of an architecture pipeline on a hardware target. "
                "Measures per-operator latency and end-to-end throughput. Returns "
                "measured results with comparison to estimates. Uses simulation when "
                "real operators are not available."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "architecture_id": {
                        "type": "string",
                        "description": "Architecture ID",
                    },
                    "hardware_id": {
                        "type": "string",
                        "description": "Hardware ID",
                    },
                    "variant_id": {
                        "type": "string",
                        "description": "Optional variant ID",
                    },
                    "iterations": {
                        "type": "integer",
                        "description": "Number of benchmark iterations (default: 100)",
                    },
                    "warmup_iterations": {
                        "type": "integer",
                        "description": "Warmup iterations (default: 10)",
                    },
                    "input_source": {
                        "type": "string",
                        "enum": ["synthetic", "file"],
                        "description": "Input data source (default: synthetic)",
                    },
                },
                "required": ["architecture_id", "hardware_id"],
            },
        },
    ]


def list_architectures(platform_type: str | None = None) -> str:
    """List available architectures in the catalog."""
    error = _check_schemas_available()
    if error:
        return error

    try:
        registry = _get_registry()

        if platform_type:
            archs = registry.get_architectures_by_platform(platform_type)
        else:
            archs = list(registry.architectures)

        output = {
            "total": len(archs),
            "architectures": [
                {
                    "id": arch.id,
                    "name": arch.name,
                    "platform_type": arch.platform_type,
                    "num_operators": len(arch.operators),
                    "num_variants": len(arch.variants),
                    "end_to_end_latency_ms": arch.end_to_end_latency_ms,
                    "power_budget_w": arch.power_budget_w,
                }
                for arch in archs
            ],
        }

        return json.dumps(output, indent=2)

    except Exception as e:
        return json.dumps(
            {
                "error": str(e),
                "traceback": traceback.format_exc(),
            },
            indent=2,
        )


def show_architecture(architecture_id: str) -> str:
    """Show detailed information about an architecture."""
    error = _check_schemas_available()
    if error:
        return error

    try:
        registry = _get_registry()
        arch = registry.architectures.get(architecture_id)

        if not arch:
            return json.dumps(
                {
                    "error": f"Architecture '{architecture_id}' not found",
                    "available": list(registry.architectures.keys()),
                },
                indent=2,
            )

        # Get operator details
        operator_details = []
        for op_inst in arch.operators:
            op_entry = registry.operators.get(op_inst.operator_id)
            operator_details.append(
                {
                    "instance_id": op_inst.id,
                    "operator_id": op_inst.operator_id,
                    "operator_name": op_entry.name if op_entry else "Unknown",
                    "category": op_entry.category.value if op_entry else "unknown",
                    "rate_hz": op_inst.rate_hz,
                    "execution_target": op_inst.execution_target,
                    "config": op_inst.config if op_inst.config else None,
                }
            )

        # Generate Mermaid diagram
        mermaid = architecture_to_mermaid(arch)

        output = {
            "id": arch.id,
            "name": arch.name,
            "description": arch.description,
            "platform_type": arch.platform_type,
            "sensors": arch.sensors,
            "actuators": arch.actuators,
            "requirements": {
                "end_to_end_latency_ms": arch.end_to_end_latency_ms,
                "min_throughput_fps": arch.min_throughput_fps,
                "power_budget_w": arch.power_budget_w,
                "memory_budget_mb": arch.memory_budget_mb,
            },
            "operators": operator_details,
            "dataflow": [
                {
                    "from": f"{edge.source_op}.{edge.source_port}",
                    "to": f"{edge.target_op}.{edge.target_port}",
                }
                for edge in arch.dataflow
            ],
            "variants": [
                {
                    "id": var.id,
                    "name": var.name,
                    "description": var.description,
                    "operator_overrides": var.operator_overrides,
                    "target_hardware": var.target_hardware,
                    "expected_latency_ms": var.expected_latency_ms,
                }
                for var in arch.variants
            ],
            "mermaid_diagram": mermaid,
            "reference_impl": arch.reference_impl,
            "tags": arch.tags,
        }

        return json.dumps(output, indent=2)

    except Exception as e:
        return json.dumps(
            {
                "error": str(e),
                "traceback": traceback.format_exc(),
            },
            indent=2,
        )


def _get_operator_perf(
    op_entry: "OperatorEntry",
    hardware_id: str,
) -> tuple[float, float, float | None]:
    """Get performance metrics for an operator on specific hardware.

    Returns:
        Tuple of (latency_ms, memory_mb, power_w)
    """
    # Try to find a matching perf profile
    for profile in op_entry.perf_profiles:
        if profile.hardware_id == hardware_id:
            return (
                profile.latency_ms or op_entry.typical_latency_ms or 1.0,
                profile.memory_mb or op_entry.typical_memory_mb or 10.0,
                profile.power_w,
            )

    # Fall back to reference/typical values
    return (
        op_entry.typical_latency_ms or 1.0,
        op_entry.typical_memory_mb or 10.0,
        None,
    )


def _apply_variant(
    arch: "SoftwareArchitecture",
    variant_id: str,
    registry: "Registry",
) -> tuple[dict[str, str], dict[str, dict]]:
    """Apply variant overrides to get effective operator mapping.

    Returns:
        Tuple of (operator_id_map, config_map) where:
        - operator_id_map: instance_id -> effective operator_id
        - config_map: instance_id -> effective config
    """
    # Start with base architecture
    operator_id_map = {op.id: op.operator_id for op in arch.operators}
    config_map = {op.id: op.config for op in arch.operators}

    # Find and apply variant if specified
    if variant_id:
        for var in arch.variants:
            if var.id == variant_id:
                # Apply operator overrides
                for inst_id, new_op_id in var.operator_overrides.items():
                    if inst_id in operator_id_map:
                        operator_id_map[inst_id] = new_op_id

                # Apply config overrides
                for inst_id, config_override in var.config_overrides.items():
                    if inst_id in config_map:
                        merged = dict(config_map[inst_id])
                        merged.update(config_override)
                        config_map[inst_id] = merged
                break

    return operator_id_map, config_map


def analyze_architecture(
    architecture_id: str,
    hardware_id: str,
    variant_id: str | None = None,
) -> str:
    """Analyze an architecture on specific hardware."""
    error = _check_schemas_available()
    if error:
        return error

    try:
        registry = _get_registry()
        arch = registry.architectures.get(architecture_id)

        if not arch:
            return json.dumps(
                {
                    "error": f"Architecture '{architecture_id}' not found",
                },
                indent=2,
            )

        # Check if hardware exists
        hardware = registry.hardware.get(hardware_id)
        hardware_name = hardware.name if hardware else hardware_id

        # Apply variant if specified
        operator_id_map, config_map = _apply_variant(arch, variant_id, registry)

        # Analyze each operator
        operator_timings = []
        total_latency_ms = 0.0
        total_memory_mb = 0.0
        total_power_w = 0.0
        bottleneck_op = None
        bottleneck_latency = 0.0
        critical_path = []

        # Track operators and utilization by execution target (generalized)
        # Keys: normalized target names (cpu, gpu, kpu, tpu, cvu, etc.)
        target_operators: dict[str, list[str]] = {}
        target_latencies: dict[str, float] = {}

        for op_inst in arch.operators:
            effective_op_id = operator_id_map[op_inst.id]
            op_entry = registry.operators.get(effective_op_id)

            if not op_entry:
                # Unknown operator - use estimates
                latency_ms = 1.0
                memory_mb = 10.0
                power_w = None
            else:
                latency_ms, memory_mb, power_w = _get_operator_perf(op_entry, hardware_id)

            # Calculate rate feasibility
            target_rate = op_inst.rate_hz
            if target_rate:
                max_rate = 1000.0 / latency_ms if latency_ms > 0 else float("inf")
                rate_feasible = max_rate >= target_rate
            else:
                max_rate = None
                rate_feasible = True

            # Track by execution target (generalized for all accelerator types)
            exec_target = _normalize_execution_target(op_inst.execution_target)

            # Add to target tracking
            if exec_target not in target_operators:
                target_operators[exec_target] = []
                target_latencies[exec_target] = 0.0
            target_operators[exec_target].append(op_inst.id)
            target_latencies[exec_target] += latency_ms

            timing = OperatorTiming(
                operator_instance_id=op_inst.id,
                operator_id=effective_op_id,
                execution_target=exec_target,
                latency_ms=latency_ms,
                memory_mb=memory_mb,
                power_w=power_w,
                target_rate_hz=target_rate,
                achievable_rate_hz=max_rate,
                rate_feasible=rate_feasible,
                is_critical_path=True,  # Simplified: assume all on critical path
                limiting_factor=None,
            )
            operator_timings.append(timing)

            # Accumulate totals (simplified: sequential execution)
            total_latency_ms += latency_ms
            total_memory_mb = max(total_memory_mb, memory_mb)  # Peak concurrent
            if power_w:
                total_power_w += power_w

            # Track bottleneck (longest operator)
            if latency_ms > bottleneck_latency:
                bottleneck_latency = latency_ms
                bottleneck_op = op_inst.id

            critical_path.append(op_inst.id)

        # Calculate throughput
        throughput_fps = 1000.0 / total_latency_ms if total_latency_ms > 0 else 0.0

        # Determine verdict based on constraints
        latency_target = arch.end_to_end_latency_ms
        power_budget = arch.power_budget_w

        if latency_target and total_latency_ms > latency_target:
            verdict = Verdict.FAIL
            latency_margin = ((latency_target - total_latency_ms) / latency_target) * 100
        elif latency_target:
            verdict = Verdict.PASS
            latency_margin = ((latency_target - total_latency_ms) / latency_target) * 100
        else:
            verdict = Verdict.PASS
            latency_margin = None

        if power_budget:
            power_margin = ((power_budget - total_power_w) / power_budget) * 100
            if total_power_w > power_budget:
                verdict = Verdict.FAIL
        else:
            power_margin = None

        # Generate summary
        if verdict == Verdict.PASS:
            summary = (
                f"Architecture meets requirements on {hardware_id}: "
                f"{total_latency_ms:.1f}ms latency, {throughput_fps:.1f} FPS"
            )
        else:
            summary = (
                f"Architecture FAILS on {hardware_id}: "
                f"{total_latency_ms:.1f}ms latency exceeds {latency_target}ms target. "
                f"Bottleneck: {bottleneck_op} ({bottleneck_latency:.1f}ms)"
            )

        # Build suggestions
        suggestions = []
        if verdict == Verdict.FAIL:
            if bottleneck_op:
                suggestions.append(
                    f"Optimize or replace '{bottleneck_op}' operator ({bottleneck_latency:.1f}ms)"
                )
            if latency_margin and latency_margin < -20:
                suggestions.append("Consider using a faster variant or more powerful hardware")
            suggestions.append("Check if operators can run in parallel to reduce critical path")

        # Create result
        result = ArchitectureAnalysisResult(
            verdict=verdict,
            confidence=Confidence.MEDIUM,  # Estimates, not measured
            summary=summary,
            architecture_id=architecture_id,
            architecture_name=arch.name,
            hardware_id=hardware_id,
            hardware_name=hardware_name,
            variant_id=variant_id,
            timestamp=datetime.now(),
            end_to_end_latency_ms=total_latency_ms,
            throughput_fps=throughput_fps,
            total_power_w=total_power_w,
            total_memory_mb=total_memory_mb,
            latency_target_ms=latency_target,
            latency_margin_pct=latency_margin,
            power_budget_w=power_budget,
            power_margin_pct=power_margin,
            operator_timings=operator_timings,
            critical_path=critical_path,
            critical_path_latency_ms=total_latency_ms,
            bottleneck_operator=bottleneck_op,
            bottleneck_latency_ms=bottleneck_latency,
            bottleneck_type="latency",
            suggestions=suggestions,
            warnings=["Analysis uses estimated performance data, not measurements"],
        )

        # Convert to JSON
        output = {
            "verdict": result.verdict.value,
            "confidence": result.confidence.value,
            "summary": result.summary,
            "architecture_id": result.architecture_id,
            "hardware_id": result.hardware_id,
            "variant_id": result.variant_id,
            "metrics": {
                "end_to_end_latency_ms": round(result.end_to_end_latency_ms, 2),
                "throughput_fps": round(result.throughput_fps, 1),
                "total_power_w": round(result.total_power_w, 1),
                "total_memory_mb": round(result.total_memory_mb, 1),
            },
            "constraints": {
                "latency_target_ms": result.latency_target_ms,
                "latency_margin_pct": (
                    round(result.latency_margin_pct, 1) if result.latency_margin_pct else None
                ),
                "power_budget_w": result.power_budget_w,
                "power_margin_pct": (
                    round(result.power_margin_pct, 1) if result.power_margin_pct else None
                ),
            },
            "bottleneck": {
                "operator": result.bottleneck_operator,
                "latency_ms": (
                    round(result.bottleneck_latency_ms, 2) if result.bottleneck_latency_ms else None
                ),
                "type": result.bottleneck_type,
            },
            "critical_path": result.critical_path,
            "operator_timings": [
                {
                    "instance_id": t.operator_instance_id,
                    "operator_id": t.operator_id,
                    "target": t.execution_target,
                    "latency_ms": round(t.latency_ms, 2),
                    "memory_mb": round(t.memory_mb, 1),
                    "rate_feasible": t.rate_feasible,
                }
                for t in result.operator_timings
            ],
            # Per-execution-target breakdown (generalized for all accelerator types)
            "execution_targets": {
                target: {
                    "name": _get_target_display_name(target),
                    "operators": ops,
                    "total_latency_ms": round(target_latencies.get(target, 0.0), 2),
                    "operator_count": len(ops),
                }
                for target, ops in target_operators.items()
            },
            "suggestions": result.suggestions,
            "warnings": result.warnings,
        }

        return json.dumps(output, indent=2)

    except Exception as e:
        return json.dumps(
            {
                "verdict": "UNKNOWN",
                "error": str(e),
                "traceback": traceback.format_exc(),
            },
            indent=2,
        )


def check_scheduling(
    architecture_id: str,
    hardware_id: str,
    variant_id: str | None = None,
) -> str:
    """Check scheduling feasibility for an architecture on hardware."""
    error = _check_schemas_available()
    if error:
        return error

    try:
        registry = _get_registry()
        arch = registry.architectures.get(architecture_id)

        if not arch:
            return json.dumps(
                {
                    "error": f"Architecture '{architecture_id}' not found",
                },
                indent=2,
            )

        # Apply variant if specified
        operator_id_map, config_map = _apply_variant(arch, variant_id, registry)

        # Analyze rate feasibility for each operator
        rate_analysis = []
        all_feasible = True
        worst_latency = 0.0

        # Track utilization by execution target (generalized for all accelerator types)
        # Keys: normalized target names (cpu, gpu, kpu, tpu, cvu, etc.)
        target_util_lists: dict[str, list[float]] = {}
        target_operators: dict[str, list[str]] = {}

        for op_inst in arch.operators:
            effective_op_id = operator_id_map[op_inst.id]
            op_entry = registry.operators.get(effective_op_id)

            if op_entry:
                latency_ms, memory_mb, power_w = _get_operator_perf(op_entry, hardware_id)
            else:
                latency_ms = 1.0
                memory_mb = 10.0

            target_rate = op_inst.rate_hz
            exec_target = _normalize_execution_target(op_inst.execution_target)

            # Initialize tracking for this target if needed
            if exec_target not in target_util_lists:
                target_util_lists[exec_target] = []
                target_operators[exec_target] = []

            target_operators[exec_target].append(op_inst.id)

            if target_rate:
                # Maximum rate = 1 / latency
                max_rate = 1000.0 / latency_ms if latency_ms > 0 else float("inf")
                achievable = max_rate >= target_rate
                margin = ((max_rate - target_rate) / target_rate) * 100 if target_rate > 0 else 100

                # Calculate utilization at target rate
                # Utilization = (time per invocation) * (invocations per second)
                utilization = (latency_ms / 1000.0) * target_rate  # fraction of time busy
                target_util_lists[exec_target].append(utilization)
            else:
                max_rate = None
                achievable = True
                margin = None
                utilization = 0.0

            if not achievable:
                all_feasible = False
                limiting = "compute" if latency_ms > (1000.0 / target_rate) else "dependency"
            else:
                limiting = None

            rate_analysis.append(
                RateFeasibility(
                    operator_instance_id=op_inst.id,
                    target_rate_hz=target_rate or 0.0,
                    achievable=achievable,
                    actual_rate_hz=max_rate,
                    margin_pct=margin,
                    limiting_factor=limiting,
                )
            )

            if latency_ms > worst_latency:
                worst_latency = latency_ms

        # Aggregate utilization per target (as percentage)
        target_utilization_pct: dict[str, float] = {}
        for target, util_list in target_util_lists.items():
            target_utilization_pct[target] = sum(util_list) * 100 if util_list else 0.0

        # Check if any target is oversubscribed
        oversubscribed_targets = [t for t, u in target_utilization_pct.items() if u > 100]

        # Determine verdict
        if all_feasible and not oversubscribed_targets:
            verdict = Verdict.PASS
            # Build utilization summary for all targets
            util_parts = [
                f"{_get_target_display_name(t)}: {u:.1f}%"
                for t, u in sorted(target_utilization_pct.items())
            ]
            summary = (
                f"All operator rates achievable on {hardware_id}. "
                f"Utilization: {', '.join(util_parts)}."
            )
        else:
            verdict = Verdict.FAIL
            infeasible = [r for r in rate_analysis if not r.achievable]
            if infeasible:
                summary = (
                    f"Scheduling FAILS: {len(infeasible)} operators cannot meet target rate. "
                    f"Worst case: {infeasible[0].operator_instance_id}"
                )
            else:
                oversubscribed_names = [
                    f"{_get_target_display_name(t)}: {target_utilization_pct[t]:.1f}%"
                    for t in oversubscribed_targets
                ]
                summary = (
                    f"Scheduling FAILS: Utilization exceeds 100%. "
                    f"Oversubscribed: {', '.join(oversubscribed_names)}"
                )

        # Build suggestions
        suggestions = []
        if not all_feasible:
            for r in rate_analysis:
                if not r.achievable:
                    suggestions.append(
                        f"Reduce target rate for '{r.operator_instance_id}' "
                        f"(max: {r.actual_rate_hz:.1f}Hz, target: {r.target_rate_hz:.1f}Hz)"
                    )
        for target in oversubscribed_targets:
            target_name = _get_target_display_name(target)
            suggestions.append(
                f"{target_name} is oversubscribed ({target_utilization_pct[target]:.1f}%) - "
                f"reduce rates or use more powerful hardware"
            )

        # Create target utilization entries (generalized for all accelerator types)
        target_utilization = [
            ExecutionTargetUtilization(
                target_id=target,
                hardware_name=_get_target_display_name(target),
                utilization_pct=target_utilization_pct[target],
                assigned_operators=target_operators[target],
            )
            for target in sorted(target_utilization_pct.keys())
        ]

        # For backward compatibility, extract CPU utilization
        cpu_util = target_utilization_pct.get("cpu", 0.0)
        # Sum all non-CPU accelerator utilization
        accel_util = sum(u for t, u in target_utilization_pct.items() if t != "cpu")

        result = SchedulingAnalysisResult(
            verdict=verdict,
            confidence=Confidence.MEDIUM,
            summary=summary,
            architecture_id=architecture_id,
            hardware_id=hardware_id,
            variant_id=variant_id,
            timestamp=datetime.now(),
            target_utilization=target_utilization,
            rate_analysis=rate_analysis,
            all_rates_feasible=all_feasible,
            total_cpu_utilization_pct=cpu_util,
            total_accelerator_utilization_pct=accel_util if accel_util > 0 else None,
            worst_case_latency_ms=worst_latency,
            suggestions=suggestions,
        )

        # Convert to JSON with per-target utilization breakdown
        output = {
            "verdict": result.verdict.value,
            "confidence": result.confidence.value,
            "summary": result.summary,
            "architecture_id": result.architecture_id,
            "hardware_id": result.hardware_id,
            "all_rates_feasible": result.all_rates_feasible,
            # Per-target utilization (generalized)
            "utilization": {
                target: {
                    "name": _get_target_display_name(target),
                    "percent": round(util, 1),
                    "operators": target_operators[target],
                    "oversubscribed": util > 100,
                }
                for target, util in target_utilization_pct.items()
            },
            "worst_case_latency_ms": round(result.worst_case_latency_ms, 2),
            "rate_analysis": [
                {
                    "operator": r.operator_instance_id,
                    "target_hz": r.target_rate_hz,
                    "achievable": r.achievable,
                    "max_hz": round(r.actual_rate_hz, 1) if r.actual_rate_hz else None,
                    "margin_pct": round(r.margin_pct, 1) if r.margin_pct else None,
                    "limiting_factor": r.limiting_factor,
                }
                for r in result.rate_analysis
            ],
            "suggestions": result.suggestions,
        }

        return json.dumps(output, indent=2)

    except Exception as e:
        return json.dumps(
            {
                "verdict": "UNKNOWN",
                "error": str(e),
                "traceback": traceback.format_exc(),
            },
            indent=2,
        )


def _find_alternative_operators(
    registry: "Registry",
    op_entry: "OperatorEntry",
    hardware_id: str,
    current_latency: float,
) -> list[dict]:
    """Find alternative operators that could replace the given operator.

    Looks for operators in the same category/subcategory that are faster.

    Returns:
        List of alternatives with estimated latency improvement
    """
    alternatives = []

    # Get all operators in the same category
    for op in registry.operators:
        if op.id == op_entry.id:
            continue

        # Must be same category
        if op.category != op_entry.category:
            continue

        # Check if it's a lighter variant (same subcategory or compatible)
        subcategory_match = (
            hasattr(op_entry, "subcategory")
            and hasattr(op, "subcategory")
            and op_entry.subcategory == op.subcategory
        )

        # Or check for same base name (e.g., yolo_detector_n vs yolo_detector_s)
        base_name_match = op_entry.id.rsplit("_", 1)[0] == op.id.rsplit("_", 1)[0]

        if not (subcategory_match or base_name_match):
            continue

        # Get latency for the alternative
        alt_latency, alt_memory, _ = _get_operator_perf(op, hardware_id)

        if alt_latency < current_latency:
            reduction_pct = ((current_latency - alt_latency) / current_latency) * 100
            alternatives.append(
                {
                    "operator_id": op.id,
                    "operator_name": op.name,
                    "latency_ms": alt_latency,
                    "memory_mb": alt_memory,
                    "latency_reduction_pct": round(reduction_pct, 1),
                    "latency_improvement_ms": round(current_latency - alt_latency, 2),
                    "tags": op.tags[:3] if op.tags else [],
                }
            )

    # Sort by latency (fastest first)
    alternatives.sort(key=lambda x: x["latency_ms"])
    return alternatives[:5]  # Top 5 alternatives


def identify_bottleneck(
    architecture_id: str,
    hardware_id: str,
    variant_id: str | None = None,
) -> str:
    """Identify the bottleneck operator in an architecture."""
    error = _check_schemas_available()
    if error:
        return error

    try:
        registry = _get_registry()
        arch = registry.architectures.get(architecture_id)

        if not arch:
            return json.dumps(
                {
                    "error": f"Architecture '{architecture_id}' not found",
                },
                indent=2,
            )

        # Apply variant if specified
        operator_id_map, config_map = _apply_variant(arch, variant_id, registry)

        # Analyze each operator
        operator_analysis = []
        total_latency = 0.0

        for op_inst in arch.operators:
            effective_op_id = operator_id_map[op_inst.id]
            op_entry = registry.operators.get(effective_op_id)

            if op_entry:
                latency_ms, memory_mb, power_w = _get_operator_perf(op_entry, hardware_id)
                compute_flops = op_entry.compute_flops
            else:
                latency_ms = 1.0
                memory_mb = 10.0
                power_w = None
                compute_flops = None

            exec_target = _normalize_execution_target(op_inst.execution_target)

            operator_analysis.append(
                {
                    "instance_id": op_inst.id,
                    "operator_id": effective_op_id,
                    "latency_ms": latency_ms,
                    "memory_mb": memory_mb,
                    "power_w": power_w,
                    "compute_flops": compute_flops,
                    "execution_target": exec_target,
                    "op_entry": op_entry,  # Keep for alternative lookup
                }
            )
            total_latency += latency_ms

        # Find the bottleneck (operator with highest latency)
        bottleneck = max(operator_analysis, key=lambda x: x["latency_ms"])
        bottleneck_pct = (bottleneck["latency_ms"] / total_latency) * 100

        # Classify bottleneck type based on characteristics
        # - Compute-bound: High FLOPS, GPU/NPU target
        # - Memory-bound: Large memory footprint relative to compute
        # - I/O-bound: CPU target with low compute but data dependencies
        if bottleneck["execution_target"] in ["gpu", "npu", "tpu", "kpu", "cvu"]:
            if bottleneck["compute_flops"] and bottleneck["compute_flops"] > 1e9:
                bottleneck_type = "compute-bound"
                bottleneck_reason = (
                    f"High compute workload ({bottleneck['compute_flops']:.1e} FLOPS) "
                    f"on {_get_target_display_name(bottleneck['execution_target'])}"
                )
            else:
                bottleneck_type = "memory-bound"
                bottleneck_reason = (
                    f"Memory access patterns limiting throughput "
                    f"({bottleneck['memory_mb']:.0f} MB footprint)"
                )
        elif bottleneck["execution_target"] == "cpu":
            if bottleneck["latency_ms"] > 10:
                bottleneck_type = "compute-bound"
                bottleneck_reason = "Complex CPU computation limiting throughput"
            else:
                bottleneck_type = "io-bound"
                bottleneck_reason = "Data dependencies or I/O overhead"
        else:
            bottleneck_type = "compute-bound"
            bottleneck_reason = "Accelerator compute limiting throughput"

        # Find alternatives for the bottleneck
        alternatives = []
        if bottleneck["op_entry"]:
            alternatives = _find_alternative_operators(
                registry,
                bottleneck["op_entry"],
                hardware_id,
                bottleneck["latency_ms"],
            )

        # Rank all operators by latency contribution
        ranked_operators = sorted(
            operator_analysis,
            key=lambda x: x["latency_ms"],
            reverse=True,
        )

        output = {
            "architecture_id": architecture_id,
            "hardware_id": hardware_id,
            "variant_id": variant_id,
            "total_latency_ms": round(total_latency, 2),
            "bottleneck": {
                "operator_instance": bottleneck["instance_id"],
                "operator_id": bottleneck["operator_id"],
                "latency_ms": round(bottleneck["latency_ms"], 2),
                "latency_contribution_pct": round(bottleneck_pct, 1),
                "execution_target": bottleneck["execution_target"],
                "type": bottleneck_type,
                "reason": bottleneck_reason,
            },
            "alternatives": alternatives,
            "operator_ranking": [
                {
                    "rank": i + 1,
                    "instance_id": op["instance_id"],
                    "operator_id": op["operator_id"],
                    "latency_ms": round(op["latency_ms"], 2),
                    "contribution_pct": round((op["latency_ms"] / total_latency) * 100, 1),
                    "target": op["execution_target"],
                }
                for i, op in enumerate(ranked_operators)
            ],
            "summary": (
                f"Bottleneck: {bottleneck['instance_id']} ({bottleneck['operator_id']}) "
                f"takes {bottleneck['latency_ms']:.1f}ms ({bottleneck_pct:.0f}% of total). "
                f"Type: {bottleneck_type}. "
                + (
                    f"{len(alternatives)} faster alternatives available."
                    if alternatives
                    else "No direct alternatives in catalog."
                )
            ),
        }

        return json.dumps(output, indent=2, default=str)

    except Exception as e:
        return json.dumps(
            {
                "error": str(e),
                "traceback": traceback.format_exc(),
            },
            indent=2,
        )


def suggest_optimizations(
    architecture_id: str,
    hardware_id: str,
    variant_id: str | None = None,
    target_latency_ms: float | None = None,
) -> str:
    """Suggest optimizations for an architecture on hardware."""
    error = _check_schemas_available()
    if error:
        return error

    try:
        registry = _get_registry()
        arch = registry.architectures.get(architecture_id)

        if not arch:
            return json.dumps(
                {
                    "error": f"Architecture '{architecture_id}' not found",
                },
                indent=2,
            )

        # Use architecture's target if not specified
        if target_latency_ms is None:
            target_latency_ms = arch.end_to_end_latency_ms

        # Apply variant if specified
        operator_id_map, config_map = _apply_variant(arch, variant_id, registry)

        # Analyze current state
        operator_analysis = []
        total_latency = 0.0
        total_memory = 0.0
        gpu_ops = []
        cpu_ops = []

        for op_inst in arch.operators:
            effective_op_id = operator_id_map[op_inst.id]
            op_entry = registry.operators.get(effective_op_id)

            if op_entry:
                latency_ms, memory_mb, power_w = _get_operator_perf(op_entry, hardware_id)
            else:
                latency_ms = 1.0
                memory_mb = 10.0
                power_w = None
                op_entry = None

            exec_target = _normalize_execution_target(op_inst.execution_target)

            operator_analysis.append(
                {
                    "instance_id": op_inst.id,
                    "operator_id": effective_op_id,
                    "latency_ms": latency_ms,
                    "memory_mb": memory_mb,
                    "execution_target": exec_target,
                    "op_entry": op_entry,
                }
            )

            total_latency += latency_ms
            total_memory = max(total_memory, memory_mb)

            if exec_target in ["gpu", "npu", "tpu", "kpu", "cvu"]:
                gpu_ops.append(op_inst.id)
            else:
                cpu_ops.append(op_inst.id)

        # Generate suggestions
        suggestions = []

        # 1. Operator swap suggestions (for high-latency operators)
        for op in sorted(operator_analysis, key=lambda x: x["latency_ms"], reverse=True)[:3]:
            if op["op_entry"]:
                alternatives = _find_alternative_operators(
                    registry, op["op_entry"], hardware_id, op["latency_ms"]
                )
                if alternatives:
                    best_alt = alternatives[0]
                    new_total = total_latency - op["latency_ms"] + best_alt["latency_ms"]
                    suggestions.append(
                        {
                            "type": "operator_swap",
                            "priority": (
                                "high" if op["latency_ms"] > total_latency * 0.3 else "medium"
                            ),
                            "operator_instance": op["instance_id"],
                            "current_operator": op["operator_id"],
                            "suggested_operator": best_alt["operator_id"],
                            "current_latency_ms": round(op["latency_ms"], 2),
                            "new_latency_ms": round(best_alt["latency_ms"], 2),
                            "improvement_ms": round(op["latency_ms"] - best_alt["latency_ms"], 2),
                            "new_total_latency_ms": round(new_total, 2),
                            "description": (
                                f"Replace {op['operator_id']} with {best_alt['operator_id']} "
                                f"to save {op['latency_ms'] - best_alt['latency_ms']:.1f}ms "
                                f"({best_alt['latency_reduction_pct']:.0f}% reduction)"
                            ),
                        }
                    )

        # 2. Quantization suggestions (for GPU operators with high memory)
        for op in operator_analysis:
            if op["execution_target"] in ["gpu", "npu"] and op["memory_mb"] > 200:
                suggestions.append(
                    {
                        "type": "quantization",
                        "priority": "medium",
                        "operator_instance": op["instance_id"],
                        "current_memory_mb": round(op["memory_mb"], 1),
                        "estimated_memory_mb": round(op["memory_mb"] * 0.5, 1),
                        "estimated_speedup": "1.5-2x",
                        "description": (
                            f"Apply INT8 quantization to {op['operator_id']} "
                            f"to reduce memory from {op['memory_mb']:.0f}MB to ~{op['memory_mb']*0.5:.0f}MB "
                            f"and potentially improve throughput 1.5-2x"
                        ),
                    }
                )

        # 3. Hardware upgrade suggestions
        if target_latency_ms and total_latency > target_latency_ms:
            gap = total_latency - target_latency_ms
            gap_pct = (gap / target_latency_ms) * 100

            # Suggest more powerful hardware
            hardware_upgrades = []
            if "Nano" in hardware_id:
                hardware_upgrades.append(
                    {
                        "hardware_id": "Jetson-Orin-AGX",
                        "expected_speedup": "2-3x",
                        "reason": "More CUDA cores and memory bandwidth",
                    }
                )
            elif "Orin" in hardware_id and "AGX" not in hardware_id:
                hardware_upgrades.append(
                    {
                        "hardware_id": "Jetson-Orin-AGX",
                        "expected_speedup": "1.5-2x",
                        "reason": "Higher GPU frequency and more cores",
                    }
                )

            if hardware_upgrades:
                for hw in hardware_upgrades:
                    suggestions.append(
                        {
                            "type": "hardware_upgrade",
                            "priority": "high" if gap_pct > 50 else "medium",
                            "current_hardware": hardware_id,
                            "suggested_hardware": hw["hardware_id"],
                            "expected_speedup": hw["expected_speedup"],
                            "reason": hw["reason"],
                            "description": (
                                f"Upgrade from {hardware_id} to {hw['hardware_id']} "
                                f"for {hw['expected_speedup']} speedup"
                            ),
                        }
                    )

        # 4. Parallelization suggestions
        if len(gpu_ops) > 1:
            suggestions.append(
                {
                    "type": "parallelization",
                    "priority": "low",
                    "operators": gpu_ops,
                    "description": (
                        f"Consider running {len(gpu_ops)} GPU operators in parallel "
                        f"using CUDA streams if data dependencies allow"
                    ),
                }
            )

        # 5. Check existing variants
        if arch.variants and not variant_id:
            for var in arch.variants:
                if var.target_hardware and hardware_id in var.target_hardware:
                    suggestions.append(
                        {
                            "type": "use_variant",
                            "priority": "high",
                            "variant_id": var.id,
                            "variant_name": var.name,
                            "expected_latency_ms": var.expected_latency_ms,
                            "description": (
                                f"Use pre-defined variant '{var.name}' "
                                f"optimized for {hardware_id}"
                            ),
                        }
                    )

        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        suggestions.sort(key=lambda x: priority_order.get(x.get("priority", "low"), 2))

        # Calculate potential improvement
        potential_savings = sum(
            s.get("improvement_ms", 0) for s in suggestions if s["type"] == "operator_swap"
        )
        potential_latency = total_latency - potential_savings

        output = {
            "architecture_id": architecture_id,
            "hardware_id": hardware_id,
            "variant_id": variant_id,
            "current_state": {
                "total_latency_ms": round(total_latency, 2),
                "target_latency_ms": target_latency_ms,
                "meets_target": total_latency <= target_latency_ms if target_latency_ms else None,
                "gap_ms": (
                    round(total_latency - target_latency_ms, 2) if target_latency_ms else None
                ),
            },
            "potential_improvement": {
                "latency_savings_ms": round(potential_savings, 2),
                "new_latency_ms": round(potential_latency, 2),
                "would_meet_target": (
                    potential_latency <= target_latency_ms if target_latency_ms else None
                ),
            },
            "suggestions": suggestions,
            "summary": (
                f"Found {len(suggestions)} optimization opportunities. "
                + (
                    f"Potential latency reduction: {potential_savings:.1f}ms "
                    f"({potential_savings/total_latency*100:.0f}%)."
                    if potential_savings > 0
                    else ""
                )
            ),
        }

        return json.dumps(output, indent=2, default=str)

    except Exception as e:
        return json.dumps(
            {
                "error": str(e),
                "traceback": traceback.format_exc(),
            },
            indent=2,
        )


def compare_variants(
    architecture_id: str,
    hardware_id: str,
    variant_ids: list[str] | None = None,
) -> str:
    """Compare architecture variants on a hardware target."""
    error = _check_schemas_available()
    if error:
        return error

    try:
        registry = _get_registry()
        arch = registry.architectures.get(architecture_id)

        if not arch:
            return json.dumps(
                {
                    "error": f"Architecture '{architecture_id}' not found",
                },
                indent=2,
            )

        # Determine which variants to compare
        if variant_ids:
            variants_to_compare = [v for v in arch.variants if v.id in variant_ids]
        else:
            variants_to_compare = arch.variants

        # Always include base architecture
        results = []

        # Analyze base architecture
        base_analysis = json.loads(analyze_architecture(architecture_id, hardware_id))
        results.append(
            {
                "id": "base",
                "name": arch.name + " (base)",
                "latency_ms": base_analysis.get("metrics", {}).get("end_to_end_latency_ms"),
                "throughput_fps": base_analysis.get("metrics", {}).get("throughput_fps"),
                "power_w": base_analysis.get("metrics", {}).get("total_power_w"),
                "memory_mb": base_analysis.get("metrics", {}).get("total_memory_mb"),
                "verdict": base_analysis.get("verdict"),
                "meets_target": base_analysis.get("verdict") == "PASS",
            }
        )

        # Analyze each variant
        for var in variants_to_compare:
            var_analysis = json.loads(analyze_architecture(architecture_id, hardware_id, var.id))
            results.append(
                {
                    "id": var.id,
                    "name": var.name,
                    "description": var.description,
                    "target_hardware": var.target_hardware,
                    "latency_ms": var_analysis.get("metrics", {}).get("end_to_end_latency_ms"),
                    "throughput_fps": var_analysis.get("metrics", {}).get("throughput_fps"),
                    "power_w": var_analysis.get("metrics", {}).get("total_power_w"),
                    "memory_mb": var_analysis.get("metrics", {}).get("total_memory_mb"),
                    "verdict": var_analysis.get("verdict"),
                    "meets_target": var_analysis.get("verdict") == "PASS",
                    "expected_latency_ms": var.expected_latency_ms,
                    "operator_overrides": var.operator_overrides,
                }
            )

        # Sort by latency (fastest first)
        results.sort(key=lambda x: x.get("latency_ms") or float("inf"))

        # Find best option
        passing = [r for r in results if r.get("meets_target")]
        if passing:
            best = min(passing, key=lambda x: x.get("latency_ms") or float("inf"))
            recommendation = (
                f"Recommended: '{best['name']}' - fastest option that meets requirements"
            )
        else:
            best = results[0] if results else None
            recommendation = (
                f"No variant meets target. '{best['name']}' is fastest but still exceeds target."
                if best
                else "No variants available"
            )

        output = {
            "architecture_id": architecture_id,
            "hardware_id": hardware_id,
            "target_latency_ms": arch.end_to_end_latency_ms,
            "variants_compared": len(results),
            "comparison": results,
            "recommendation": recommendation,
            "summary": (
                f"Compared {len(results)} variants on {hardware_id}. "
                f"{len(passing)}/{len(results)} meet the latency target."
            ),
        }

        return json.dumps(output, indent=2, default=str)

    except Exception as e:
        return json.dumps(
            {
                "error": str(e),
                "traceback": traceback.format_exc(),
            },
            indent=2,
        )


def _simulate_operator_benchmark(
    estimated_latency_ms: float,
    iterations: int,
    noise_factor: float = 0.15,
) -> dict:
    """Simulate benchmark measurements for an operator.

    Generates realistic timing measurements based on the estimated latency,
    with appropriate variance to simulate real-world conditions.

    Args:
        estimated_latency_ms: Base latency from operator profile
        iterations: Number of simulated iterations
        noise_factor: Relative noise (0.15 = 15% variance)

    Returns:
        Dictionary with simulated timing statistics
    """
    import random
    import statistics

    # Generate samples with log-normal distribution (realistic for latency)
    # Mean is slightly higher than estimate (real often slower)
    mean_actual = estimated_latency_ms * random.uniform(0.95, 1.15)
    std = mean_actual * noise_factor

    samples = [max(0.1, random.gauss(mean_actual, std)) for _ in range(iterations)]
    samples.sort()

    return {
        "mean": statistics.mean(samples),
        "std": statistics.stdev(samples) if len(samples) > 1 else 0,
        "min": min(samples),
        "max": max(samples),
        "p50": samples[int(len(samples) * 0.5)],
        "p95": samples[int(len(samples) * 0.95)],
        "p99": samples[int(len(samples) * 0.99)] if len(samples) >= 100 else samples[-1],
    }


def benchmark_architecture(
    architecture_id: str,
    hardware_id: str,
    variant_id: str | None = None,
    iterations: int = 100,
    warmup_iterations: int = 10,
    input_source: str = "synthetic",
) -> str:
    """Benchmark an architecture pipeline on hardware.

    Runs (or simulates) the architecture pipeline and collects
    per-operator timing measurements with statistical analysis.
    """
    error = _check_schemas_available()
    if error:
        return error

    try:
        registry = _get_registry()
        arch = registry.architectures.get(architecture_id)

        if not arch:
            return json.dumps(
                {
                    "error": f"Architecture '{architecture_id}' not found",
                },
                indent=2,
            )

        # Check if hardware exists
        hardware = registry.hardware.get(hardware_id)
        hardware_name = hardware.name if hardware else hardware_id

        # Apply variant if specified
        operator_id_map, config_map = _apply_variant(arch, variant_id, registry)

        # Benchmark each operator
        operator_results = []
        total_measured_latency = 0.0
        total_estimated_latency = 0.0
        bottleneck_op = None
        bottleneck_latency = 0.0

        for op_inst in arch.operators:
            effective_op_id = operator_id_map[op_inst.id]
            op_entry = registry.operators.get(effective_op_id)

            if op_entry:
                estimated_latency, memory_mb, power_w = _get_operator_perf(op_entry, hardware_id)
            else:
                estimated_latency = 1.0
                memory_mb = 10.0

            exec_target = _normalize_execution_target(op_inst.execution_target)

            # Simulate benchmark
            timing = _simulate_operator_benchmark(estimated_latency, iterations)

            # Calculate estimate error
            estimate_error = ((timing["mean"] - estimated_latency) / estimated_latency) * 100

            op_result = OperatorBenchmarkResult(
                operator_instance_id=op_inst.id,
                operator_id=effective_op_id,
                execution_target=exec_target,
                iterations=iterations,
                mean_latency_ms=timing["mean"],
                std_latency_ms=timing["std"],
                min_latency_ms=timing["min"],
                max_latency_ms=timing["max"],
                p50_latency_ms=timing["p50"],
                p95_latency_ms=timing["p95"],
                p99_latency_ms=timing["p99"],
                peak_memory_mb=memory_mb,
                estimated_latency_ms=estimated_latency,
                estimate_error_pct=estimate_error,
                success=True,
            )
            operator_results.append(op_result)

            total_measured_latency += timing["mean"]
            total_estimated_latency += estimated_latency

            # Track bottleneck
            if timing["mean"] > bottleneck_latency:
                bottleneck_latency = timing["mean"]
                bottleneck_op = op_inst.id

        # Calculate end-to-end metrics
        measured_throughput = 1000.0 / total_measured_latency if total_measured_latency > 0 else 0
        latency_target = arch.end_to_end_latency_ms

        # Calculate overall estimate error
        overall_estimate_error = (
            (total_measured_latency - total_estimated_latency) / total_estimated_latency * 100
            if total_estimated_latency > 0
            else 0
        )

        # Determine verdict
        if latency_target:
            meets_target = total_measured_latency <= latency_target
            latency_margin = ((latency_target - total_measured_latency) / latency_target) * 100
            if meets_target:
                verdict = Verdict.PASS
                summary = (
                    f"Benchmark PASSED: Measured {total_measured_latency:.1f}ms "
                    f"(target: {latency_target}ms, margin: {latency_margin:.1f}%)"
                )
            else:
                verdict = Verdict.FAIL
                summary = (
                    f"Benchmark FAILED: Measured {total_measured_latency:.1f}ms "
                    f"exceeds {latency_target}ms target by {-latency_margin:.1f}%"
                )
        else:
            verdict = Verdict.PASS
            meets_target = None
            latency_margin = None
            summary = (
                f"Benchmark complete: {total_measured_latency:.1f}ms latency, "
                f"{measured_throughput:.1f} FPS"
            )

        # Calculate bottleneck contribution
        bottleneck_contribution = (
            (bottleneck_latency / total_measured_latency) * 100 if total_measured_latency > 0 else 0
        )

        # Create result
        result = ArchitectureBenchmarkResult(
            verdict=verdict,
            confidence=Confidence.MEDIUM,  # Simulated, not real measurements
            summary=summary,
            architecture_id=architecture_id,
            architecture_name=arch.name,
            hardware_id=hardware_id,
            hardware_name=hardware_name,
            variant_id=variant_id,
            timestamp=datetime.now(),
            iterations=iterations,
            warmup_iterations=warmup_iterations,
            input_source=input_source,
            measured_latency_ms=total_measured_latency,
            measured_latency_std_ms=sum(r.std_latency_ms for r in operator_results),
            measured_throughput_fps=measured_throughput,
            measured_memory_mb=max(r.peak_memory_mb or 0 for r in operator_results),
            operator_results=operator_results,
            estimated_latency_ms=total_estimated_latency,
            latency_estimate_error_pct=overall_estimate_error,
            latency_target_ms=latency_target,
            meets_latency_target=meets_target,
            latency_margin_pct=latency_margin,
            measured_bottleneck_operator=bottleneck_op,
            measured_bottleneck_latency_ms=bottleneck_latency,
            measured_bottleneck_contribution_pct=bottleneck_contribution,
            warnings=[
                "Results are simulated based on operator performance profiles",
                "Real benchmarks require operator implementations",
            ],
        )

        # Convert to JSON
        output = {
            "verdict": result.verdict.value,
            "confidence": result.confidence.value,
            "summary": result.summary,
            "architecture_id": result.architecture_id,
            "hardware_id": result.hardware_id,
            "variant_id": result.variant_id,
            "benchmark_config": {
                "iterations": result.iterations,
                "warmup_iterations": result.warmup_iterations,
                "input_source": result.input_source,
            },
            "measured_metrics": {
                "end_to_end_latency_ms": round(result.measured_latency_ms, 2),
                "latency_std_ms": round(result.measured_latency_std_ms, 2),
                "throughput_fps": round(result.measured_throughput_fps, 1),
                "peak_memory_mb": (
                    round(result.measured_memory_mb, 1) if result.measured_memory_mb else None
                ),
            },
            "comparison_to_estimate": {
                "estimated_latency_ms": round(result.estimated_latency_ms, 2),
                "estimate_error_pct": round(result.latency_estimate_error_pct, 1),
            },
            "constraint_check": {
                "target_latency_ms": result.latency_target_ms,
                "meets_target": result.meets_latency_target,
                "margin_pct": (
                    round(result.latency_margin_pct, 1) if result.latency_margin_pct else None
                ),
            },
            "bottleneck": {
                "operator": result.measured_bottleneck_operator,
                "latency_ms": round(result.measured_bottleneck_latency_ms, 2),
                "contribution_pct": round(result.measured_bottleneck_contribution_pct, 1),
            },
            "per_operator_results": [
                {
                    "instance_id": r.operator_instance_id,
                    "operator_id": r.operator_id,
                    "target": r.execution_target,
                    "measured_ms": round(r.mean_latency_ms, 2),
                    "std_ms": round(r.std_latency_ms, 2),
                    "p95_ms": round(r.p95_latency_ms, 2) if r.p95_latency_ms else None,
                    "estimated_ms": (
                        round(r.estimated_latency_ms, 2) if r.estimated_latency_ms else None
                    ),
                    "estimate_error_pct": (
                        round(r.estimate_error_pct, 1) if r.estimate_error_pct else None
                    ),
                }
                for r in result.operator_results
            ],
            "warnings": result.warnings,
        }

        return json.dumps(output, indent=2, default=str)

    except Exception as e:
        return json.dumps(
            {
                "verdict": "UNKNOWN",
                "error": str(e),
                "traceback": traceback.format_exc(),
            },
            indent=2,
        )


def create_architecture_tool_executors() -> dict[str, callable]:
    """Create executor functions for architecture tools.

    Returns:
        Dictionary mapping tool names to executor functions
    """
    return {
        "list_architectures": list_architectures,
        "show_architecture": show_architecture,
        "analyze_architecture": analyze_architecture,
        "check_scheduling": check_scheduling,
        "identify_architecture_bottleneck": identify_bottleneck,
        "suggest_optimizations": suggest_optimizations,
        "compare_variants": compare_variants,
        "benchmark_architecture": benchmark_architecture,
    }
