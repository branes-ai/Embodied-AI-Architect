"""Converter from CodebaseAnalysisResult to workload_profile format.

Maps compute kernels to operator types understood by the existing
workload_analyzer, hw_explorer, and ppa_assessor pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from embodied_ai_architect.codebase.models import (
    CodebaseAnalysisResult,
    ComputeKernel,
    SuggestedConstraint,
    SuggestedConstraints,
)

if TYPE_CHECKING:
    from embodied_ai_architect.graphs.soc_state import DesignConstraints, SoCDesignState

# Kernel type → operator mapping
KERNEL_OPERATOR_MAP: dict[str, list[dict]] = {
    "ml_inference": [
        {"type": "convolution", "count": 50},
        {"type": "matrix_multiply", "count": 20},
        {"type": "activation", "count": 40},
        {"type": "batch_norm", "count": 30},
    ],
    "signal_processing": [
        {"type": "fft", "count": 5},
        {"type": "filtering", "count": 10},
        {"type": "accumulate", "count": 5},
    ],
    "image_processing": [
        {"type": "convolution", "count": 10},
        {"type": "resize", "count": 2},
        {"type": "color_convert", "count": 2},
    ],
    "control_loop": [
        {"type": "matrix_multiply", "count": 5},
        {"type": "accumulate", "count": 10},
    ],
    "sensor_fusion": [
        {"type": "matrix_multiply", "count": 15},
        {"type": "accumulate", "count": 10},
    ],
    "io_bound": [
        {"type": "memory_copy", "count": 5},
    ],
    "general_compute": [
        {"type": "general_purpose", "count": 1},
    ],
}

# GFLOPS estimates per kernel type when ops_per_invocation is not set
DEFAULT_GFLOPS: dict[str, float] = {
    "ml_inference": 8.0,
    "signal_processing": 1.0,
    "image_processing": 2.0,
    "control_loop": 0.1,
    "sensor_fusion": 0.5,
    "io_bound": 0.01,
    "general_compute": 0.5,
}

# Memory estimates per kernel type (MB)
DEFAULT_MEMORY_MB: dict[str, float] = {
    "ml_inference": 50.0,
    "signal_processing": 5.0,
    "image_processing": 10.0,
    "control_loop": 1.0,
    "sensor_fusion": 5.0,
    "io_bound": 2.0,
    "general_compute": 5.0,
}

# Scheduling inference
SCHEDULING_MAP: dict[str, str] = {
    "ml_inference": "concurrent",
    "signal_processing": "concurrent",
    "image_processing": "concurrent",
    "control_loop": "concurrent",
    "sensor_fusion": "concurrent",
    "io_bound": "sequential",
    "general_compute": "time_shared",
}


class CodebaseConverter:
    """Converts CodebaseAnalysisResult into workload_profile dicts.

    The output format matches what specialists.py:workload_analyzer() and
    hw_explorer() consume.
    """

    def to_workload_profile(self, analysis: CodebaseAnalysisResult) -> dict:
        """Convert analysis results to a workload_profile dict.

        Args:
            analysis: The full codebase analysis result.

        Returns:
            Dict compatible with the existing PPA pipeline.
        """
        workloads = []
        all_operators: dict[str, int] = {}

        for kernel in analysis.kernels:
            workload = self._kernel_to_workload(kernel)
            workloads.append(workload)

            # Accumulate operator counts
            for op in workload["operators"]:
                all_operators[op["type"]] = all_operators.get(op["type"], 0) + op["count"]

        # If no kernels were found, create a default workload
        if not workloads:
            workloads.append(
                {
                    "name": "application",
                    "model_class": "Unknown",
                    "operators": [{"type": "general_purpose", "count": 1}],
                    "estimated_gflops": 1.0,
                    "estimated_memory_mb": 10.0,
                    "estimated_params_m": 0.0,
                    "scheduling": "sequential",
                }
            )
            all_operators["general_purpose"] = 1

        total_gflops = sum(w["estimated_gflops"] for w in workloads)
        total_memory_mb = sum(w["estimated_memory_mb"] for w in workloads)
        dominant_op = max(all_operators, key=all_operators.get) if all_operators else "unknown"

        # Issue #40: build the operator dataflow graph from the LLM's
        # DataflowLink edges. Falls back to a sequential chain when the
        # LLM didn't produce any dataflow links.
        operator_graph = self._build_operator_graph(analysis, workloads)

        return {
            "workloads": workloads,
            "total_estimated_gflops": round(total_gflops, 2),
            "total_estimated_memory_mb": round(total_memory_mb, 2),
            "dominant_op": dominant_op,
            "workload_count": len(workloads),
            "use_case": "application_analysis",
            "source": "codebase_analysis",
            "project_name": analysis.project_name,
            "languages": analysis.languages,
            "operator_graph": operator_graph,
        }

    @staticmethod
    def _build_operator_graph(
        analysis: CodebaseAnalysisResult,
        workloads: list[dict],
    ) -> dict:
        """Build an operator DAG from DataflowLink edges (issue #40).

        Nodes correspond 1:1 to the workloads list (same order / same names).
        Edges come from analysis.dataflow. When no dataflow links exist, the
        method falls back to a sequential chain: kernel_0 → kernel_1 → ... so
        downstream consumers always have a connected graph to schedule against.

        Returns a dict with `nodes` (list of node dicts) and `edges` (list of
        edge dicts), ready to be serialized directly onto the workload_profile.
        """
        if not workloads:
            return {"nodes": [], "edges": []}

        # Build node dicts — keyed by workload name so dataflow edges can
        # reference them. Disambiguate duplicate names with a numeric suffix
        # so the graph always has unique node IDs (CodeRabbit PR #91).
        nodes: list[dict] = []
        id_counts: dict[str, int] = {}
        for i, w in enumerate(workloads):
            base_id = w.get("name") or f"op_{i}"
            seen = id_counts.get(base_id, 0)
            node_id = base_id if seen == 0 else f"{base_id}_{seen}"
            id_counts[base_id] = seen + 1

            dominant_op_type = "general_purpose"
            ops = w.get("operators", [])
            if ops:
                dominant_op_type = max(ops, key=lambda o: o.get("count", 0))["type"]
            nodes.append(
                {
                    "id": node_id,
                    "kernel": w.get("name", ""),
                    "gflops": w.get("estimated_gflops"),
                    "memory_mb": w.get("estimated_memory_mb"),
                    "type": dominant_op_type,
                    "scheduling": w.get("scheduling", ""),
                }
            )

        node_ids = {n["id"] for n in nodes}

        # Build edges from analysis.dataflow (when available)
        edges: list[dict] = []
        if analysis.dataflow:
            for link in analysis.dataflow:
                # Only include edges where both endpoints exist in the graph
                if link.source_kernel in node_ids and link.sink_kernel in node_ids:
                    edges.append(
                        {
                            "source": link.source_kernel,
                            "sink": link.sink_kernel,
                            "data_bytes": link.data_size_bytes,
                            "transfer_type": link.transfer_type,
                        }
                    )

        # Fallback: when no dataflow links were found (or the LLM didn't
        # produce them), create a sequential chain so the graph is connected.
        if not edges and len(nodes) > 1:
            for i in range(len(nodes) - 1):
                edges.append(
                    {
                        "source": nodes[i]["id"],
                        "sink": nodes[i + 1]["id"],
                        "data_bytes": 0,
                        "transfer_type": "memory",
                    }
                )

        return {"nodes": nodes, "edges": edges}

    def _kernel_to_workload(self, kernel: ComputeKernel) -> dict:
        """Map a single ComputeKernel to a workload dict."""
        operators = KERNEL_OPERATOR_MAP.get(
            kernel.kernel_type,
            KERNEL_OPERATOR_MAP["general_compute"],
        )
        # Copy operators so we don't mutate the template
        operators = [dict(op) for op in operators]

        # Estimate GFLOPS from ops_per_invocation if available
        if kernel.estimated_ops_per_invocation > 0:
            gflops = kernel.estimated_ops_per_invocation / 1e9
        else:
            gflops = DEFAULT_GFLOPS.get(kernel.kernel_type, 0.5)

        memory_mb = DEFAULT_MEMORY_MB.get(kernel.kernel_type, 5.0)
        scheduling = SCHEDULING_MAP.get(kernel.kernel_type, "time_shared")

        # Determine model class from frameworks
        model_class = "Custom"
        if kernel.frameworks:
            fw_lower = [f.lower() for f in kernel.frameworks]
            if "pytorch" in fw_lower or "torch" in fw_lower:
                model_class = "PyTorch"
            elif "tensorflow" in fw_lower or "tf" in fw_lower:
                model_class = "TensorFlow"
            elif "opencv" in fw_lower:
                model_class = "OpenCV"
            elif "eigen" in fw_lower:
                model_class = "Eigen"

        return {
            "name": kernel.name,
            "model_class": model_class,
            "operators": operators,
            "estimated_gflops": round(gflops, 2),
            "estimated_memory_mb": round(memory_mb, 2),
            "estimated_params_m": 0.0,
            "scheduling": scheduling,
            "kernel_type": kernel.kernel_type,
            "data_types": kernel.data_types,
            "parallelism": kernel.parallelism,
            "invocation_frequency_hz": kernel.invocation_frequency_hz,
            # Source traceability — preserved from ComputeKernel for the
            # /architect-assess and /architect-drill skills (issue #42)
            "source_file": kernel.source_file,
            "line_range": list(kernel.line_range),
            "frameworks": list(kernel.frameworks),
        }


# ---------------------------------------------------------------------------
# Issue #37: Codebase → SoCDesignState bridge
# ---------------------------------------------------------------------------

# Canonical use_case for any session built from a codebase scan. The
# planner / qualifier / registry recognize this string; downstream
# consumers branch on it. The dominant-kernel-type signal is preserved
# separately on `codebase_metadata.dominant_kernel_type` so a future
# consumer can pick it up without us having to mint use_case labels that
# aren't wired through (CodeRabbit PR #88).
_CODEBASE_USE_CASE = "codebase_analysis"


def _dominant_kernel_type(analysis: CodebaseAnalysisResult) -> str:
    """Pick the dominant kernel type from the analysis (or 'general_compute')."""
    if not analysis.kernels:
        return "general_compute"
    counts: dict[str, int] = {}
    for kernel in analysis.kernels:
        counts[kernel.kernel_type] = counts.get(kernel.kernel_type, 0) + 1
    return max(counts, key=counts.get)


def _build_goal(analysis: CodebaseAnalysisResult) -> str:
    """Synthesize a one-line design goal from the codebase analysis."""
    name = analysis.project_name or "application"
    parts = [f"Design SoC for {name}"]
    if analysis.kernels:
        types = sorted({k.kernel_type for k in analysis.kernels})
        parts.append(f"({', '.join(types)})")
    if analysis.summary:
        # Trim summary to first sentence to keep the goal short
        first_sentence = analysis.summary.split(".")[0].strip()
        if first_sentence:
            parts.append(f"— {first_sentence}")
    return " ".join(parts)


def codebase_to_soc_state(
    analysis: CodebaseAnalysisResult,
    constraints: "Optional[DesignConstraints]" = None,
    project_path: Optional[str] = None,
    session_id: Optional[str] = None,
) -> "SoCDesignState":
    """Build a populated SoCDesignState from a CodebaseAnalysisResult.

    The bridge from "I have an application" to "design hardware for it"
    (issue #37). Auto-fills:

    - `goal` from project name + dominant kernel types + analysis summary
    - `workload_profile` from `CodebaseConverter.to_workload_profile`
    - `use_case` inferred from the dominant kernel type
    - `codebase_metadata` with the project path, languages, build system,
      and a brief scan summary so the architect skills (`/architect-assess`,
      `/architect-drill source:`) can resolve source files later

    Args:
        analysis: Full codebase analysis result.
        constraints: Optional design constraints (power, latency, area, cost).
        project_path: Path used to resolve source files (kept absolute on
            state so `/architect-drill source:` works regardless of cwd).
        session_id: Optional session identifier — auto-generated if None.

    Returns:
        SoCDesignState ready to feed into PlannerNode / SoCDesignRunner.
    """
    # Local imports to avoid pulling soc_state into module-load time and
    # creating a circular dep with the graphs subsystem.
    from embodied_ai_architect.graphs.soc_state import create_initial_soc_state

    converter = CodebaseConverter()
    workload_profile = converter.to_workload_profile(analysis)

    dominant_kernel_type = _dominant_kernel_type(analysis)
    goal = _build_goal(analysis)

    state = create_initial_soc_state(
        goal=goal,
        constraints=constraints,
        use_case=_CODEBASE_USE_CASE,  # canonical label recognized by planner / qualifier
        platform="custom",
        session_id=session_id,
    )

    state["workload_profile"] = workload_profile

    resolved_project_path = str(Path(project_path).resolve()) if project_path else ""
    state["codebase_metadata"] = {
        "project_path": resolved_project_path,
        "project_name": analysis.project_name,
        "languages": list(analysis.languages),
        "build_system": analysis.build_system,
        "kernel_count": len(analysis.kernels),
        "ml_models": list(analysis.ml_models),
        "dominant_kernel_type": dominant_kernel_type,
        "scan_summary": {
            "source_file_count": len(analysis.source_files),
            "dependencies": list(analysis.dependencies)[:50],
        },
    }

    return state


def codebase_data_to_soc_state(
    analysis_data: dict[str, Any],
    scan_data: Optional[dict[str, Any]] = None,
    project_path: Optional[str] = None,
    constraints: "Optional[DesignConstraints]" = None,
    session_id: Optional[str] = None,
) -> "SoCDesignState":
    """Same as `codebase_to_soc_state` but accepts dict payloads.

    Useful for callers that already have the analysis as a model_dump (e.g.
    the agent layer that returns `result.data`). Reconstructs the
    `CodebaseAnalysisResult` from the dict, falls back to scan_data fields
    when the analysis dict is missing them.
    """
    merged = dict(analysis_data) if analysis_data else {}
    if scan_data:
        # CodeRabbit PR #88: model_dump() of an empty CodebaseAnalysisResult
        # produces `[]`, `"unknown"`, etc. as defaults — `setdefault` would
        # then NOT backfill from scan_data. Only fall back to scan_data when
        # the analysis value is empty/missing/unknown.
        def _empty_or_default(value: Any, default_marker: Any = None) -> bool:
            if value is None:
                return True
            if isinstance(value, (list, tuple, dict, str)) and not value:
                return True
            if value == default_marker:
                return True
            return False

        if _empty_or_default(merged.get("project_name")):
            merged["project_name"] = scan_data.get("project_name", "application")
        if _empty_or_default(merged.get("languages")):
            merged["languages"] = list(scan_data.get("languages", []))
        if _empty_or_default(merged.get("build_system"), default_marker="unknown"):
            merged["build_system"] = scan_data.get("build_system", "unknown")
        if _empty_or_default(merged.get("source_files")):
            merged["source_files"] = list(scan_data.get("source_files", []))
        if _empty_or_default(merged.get("ml_models")):
            merged["ml_models"] = list(scan_data.get("ml_models", []))
        if _empty_or_default(merged.get("dependencies")):
            merged["dependencies"] = list(scan_data.get("dependencies", []))
    merged.setdefault("project_name", "application")
    merged.setdefault("kernels", [])
    merged.setdefault("source_files", [])

    analysis = CodebaseAnalysisResult(**merged)
    return codebase_to_soc_state(
        analysis,
        constraints=constraints,
        project_path=project_path,
        session_id=session_id,
    )


# ---------------------------------------------------------------------------
# Issue #38: Infer DesignConstraints from codebase characteristics
# ---------------------------------------------------------------------------

# Heuristic constants. Easy to tune as the inference layer matures.
# All values are crude order-of-magnitude estimates — they're meant to be
# starting points the architect can refine, not authoritative numbers.

# Power envelope: roughly 1W per 2 TOPS at 28nm. The catalog calls this
# "edge accelerator" territory; servers and HBM-class designs will be
# orders of magnitude higher.
_GFLOPS_PER_WATT_28NM = 2000.0  # 2 TOPS/W → 2000 GFLOPS/W

# Frequency thresholds for the high-frequency signal-processing heuristic
_SIGNAL_PROC_HIGH_FREQ_HZ = 1000.0  # 1 kHz+ → DSP territory

# Compute thresholds for the ML hardware-class hint
_ML_NPU_GFLOPS_THRESHOLD = 5.0  # 5 GFLOPS+ → NPU/GPU class
_ML_GPU_GFLOPS_THRESHOLD = 50.0  # 50 GFLOPS+ → GPU class


def _kernel_share_by_type(analysis: CodebaseAnalysisResult) -> dict[str, float]:
    """Compute the fraction of kernels of each type (sums to 1.0)."""
    if not analysis.kernels:
        return {}
    counts: dict[str, int] = {}
    for k in analysis.kernels:
        counts[k.kernel_type] = counts.get(k.kernel_type, 0) + 1
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()}


def _max_invocation_frequency_by_type(analysis: CodebaseAnalysisResult, kernel_type: str) -> float:
    """Find the maximum invocation_frequency_hz across kernels of a type."""
    freqs = [
        k.invocation_frequency_hz
        for k in analysis.kernels
        if k.kernel_type == kernel_type and k.invocation_frequency_hz > 0
    ]
    return max(freqs, default=0.0)


def _total_gflops(analysis: CodebaseAnalysisResult) -> float:
    """Aggregate compute throughput in GFLOPS (ops/sec ÷ 1e9).

    `estimated_ops_per_invocation` is per-call, not per-second — multiplying
    by `invocation_frequency_hz` gives the actual throughput. When the
    frequency is unknown, fall back to 1 Hz (one invocation per second) as
    a pessimistic floor so the kernel still contributes to the total
    (CodeRabbit PR #89).
    """
    total_ops_per_sec = 0.0
    for k in analysis.kernels:
        rate = k.invocation_frequency_hz if k.invocation_frequency_hz > 0 else 1.0
        total_ops_per_sec += k.estimated_ops_per_invocation * rate
    return total_ops_per_sec / 1e9


def infer_constraints(analysis: CodebaseAnalysisResult) -> SuggestedConstraints:
    """Infer suggested DesignConstraints from codebase characteristics (issue #38).

    Heuristic rules:
      - control_loop kernel with invocation_frequency_hz → max_latency_ms
        derived from the period (high confidence when frequency is set)
      - ML inference dominant + high GFLOPS → hardware_class hint
        (NPU at 5+ GFLOPS, GPU at 50+ GFLOPS)
      - Signal processing with high invocation frequency → DSP hint
      - Total GFLOPS → max_power_watts envelope at ~2 TOPS/W (28nm)
      - I/O bound dominant → memory_bw_critical=True

    Returns a `SuggestedConstraints` collection. The architect can pass the
    high-confidence numeric subset directly into a `DesignConstraints`
    constructor via `to_design_constraints_kwargs()`.
    """
    suggestions: list[SuggestedConstraint] = []
    shares = _kernel_share_by_type(analysis)

    # --- 1. Latency from control loop frequency ---------------------------
    cl_freq = _max_invocation_frequency_by_type(analysis, "control_loop")
    if cl_freq > 0:
        # period_ms = 1000 / freq_hz; deadline must beat the period
        period_ms = 1000.0 / cl_freq
        # Conservative: latency budget is the period itself (the next sample
        # is due then). High confidence when frequency is explicitly set.
        suggestions.append(
            SuggestedConstraint(
                name="max_latency_ms",
                value=round(period_ms, 2),
                confidence="high",
                rationale=(
                    f"control_loop kernel at {cl_freq:.0f}Hz → " f"{period_ms:.1f}ms per cycle"
                ),
            )
        )

    # --- 2. Total GFLOPS → power envelope ---------------------------------
    total_gflops = _total_gflops(analysis)
    if total_gflops > 0:
        # 2 TOPS/W at 28nm → power = gflops / 2000
        # Round up to a sensible engineering value (1W minimum)
        power_w = max(1.0, round(total_gflops / _GFLOPS_PER_WATT_28NM, 1))
        confidence = "medium" if total_gflops < 100 else "low"
        suggestions.append(
            SuggestedConstraint(
                name="max_power_watts",
                value=power_w,
                confidence=confidence,
                rationale=(f"{total_gflops:.1f} GFLOPS estimated, " f"~2 TOPS/W envelope at 28nm"),
            )
        )

    # --- 3. Hardware class hint (single winner) --------------------------
    # Score every candidate hardware class and emit ONE entry — issuing
    # multiple `hardware_class` suggestions would silently collide in
    # `to_dict()` (CodeRabbit PR #89). Score = (confidence_rank, share)
    # so high-confidence rules beat medium ones, and ties break by
    # kernel-type dominance.
    ml_share = shares.get("ml_inference", 0.0)
    sp_freq = _max_invocation_frequency_by_type(analysis, "signal_processing")
    sp_share = shares.get("signal_processing", 0.0)

    _CONF_RANK = {"high": 3, "medium": 2, "low": 1}
    hw_candidates: list[SuggestedConstraint] = []

    if ml_share > 0.5 and total_gflops >= _ML_GPU_GFLOPS_THRESHOLD:
        hw_candidates.append(
            SuggestedConstraint(
                name="hardware_class",
                value="gpu",
                confidence="high",
                rationale=(
                    f"{ml_share:.0%} of kernels are ML inference, "
                    f"{total_gflops:.1f} GFLOPS exceeds GPU threshold"
                ),
            )
        )
    elif ml_share > 0.5 and total_gflops >= _ML_NPU_GFLOPS_THRESHOLD:
        hw_candidates.append(
            SuggestedConstraint(
                name="hardware_class",
                value="npu",
                confidence="high",
                rationale=(
                    f"{ml_share:.0%} of kernels are ML inference, "
                    f"{total_gflops:.1f} GFLOPS suits an NPU"
                ),
            )
        )

    if sp_share > 0.3 and sp_freq >= _SIGNAL_PROC_HIGH_FREQ_HZ:
        hw_candidates.append(
            SuggestedConstraint(
                name="hardware_class",
                value="dsp",
                confidence="medium",
                rationale=(
                    f"{sp_share:.0%} signal processing kernels at " f"{sp_freq:.0f}Hz suggests DSP"
                ),
            )
        )

    if hw_candidates:
        winner = max(
            hw_candidates,
            key=lambda c: (
                _CONF_RANK.get(c.confidence, 0),
                ml_share if c.value in ("gpu", "npu") else sp_share,
            ),
        )
        suggestions.append(winner)

    # --- 5. Memory bandwidth flag from I/O bound dominance ---------------
    io_share = shares.get("io_bound", 0.0)
    if io_share >= 0.5:
        suggestions.append(
            SuggestedConstraint(
                name="memory_bw_critical",
                value=True,
                confidence="medium",
                rationale=(
                    f"{io_share:.0%} of kernels are I/O bound — memory "
                    "bandwidth likely the dominant constraint"
                ),
            )
        )

    summary = (
        f"Inferred {len(suggestions)} constraint(s) from "
        f"{len(analysis.kernels)} kernels across {len(shares)} type(s)"
    )
    return SuggestedConstraints(constraints=suggestions, summary=summary)
