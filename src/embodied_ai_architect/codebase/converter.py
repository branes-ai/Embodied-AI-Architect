"""Converter from CodebaseAnalysisResult to workload_profile format.

Maps compute kernels to operator types understood by the existing
workload_analyzer, hw_explorer, and ppa_assessor pipeline.
"""

from __future__ import annotations

from embodied_ai_architect.codebase.models import CodebaseAnalysisResult, ComputeKernel

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
        }

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
