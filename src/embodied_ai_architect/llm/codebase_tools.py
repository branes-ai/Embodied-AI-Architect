"""LLM tools for codebase analysis.

Provides three tools for the interactive chat agent:
1. scan_project — quick file scan
2. analyze_codebase — full LLM multi-pass analysis
3. assess_codebase_on_hardware — end-to-end scan → analyze → convert → assess
"""

import json
import traceback
from pathlib import Path
from typing import Any, Callable

from embodied_ai_architect.codebase.converter import CodebaseConverter
from embodied_ai_architect.codebase.scanner import CodebaseScanner


def get_codebase_tool_definitions() -> list[dict[str, Any]]:
    """Get tool definitions for codebase analysis in Anthropic format.

    Returns:
        List of tool definitions with name, description, and input_schema
    """
    return [
        {
            "name": "scan_project",
            "description": (
                "Quick scan of a project directory. Returns file inventory, detected "
                "languages, build system (cmake/cargo/pip), ML model files (.onnx, .pt, "
                ".tflite), and dependencies. No LLM needed — this is a fast static scan."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "project_path": {
                        "type": "string",
                        "description": "Path to the project directory to scan",
                    },
                },
                "required": ["project_path"],
            },
        },
        {
            "name": "analyze_codebase",
            "description": (
                "Full multi-pass LLM analysis of a project codebase. Reads source files "
                "to identify compute kernels (ML inference, signal processing, control "
                "loops, sensor fusion, etc.), estimates computational requirements, and "
                "maps dataflow between kernels. Use scan_project first for a quick look, "
                "then this for deep analysis."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "project_path": {
                        "type": "string",
                        "description": "Path to the project directory to analyze",
                    },
                },
                "required": ["project_path"],
            },
        },
        {
            "name": "design_from_codebase",
            "description": (
                "End-to-end bridge from a project codebase to a saved SoC design "
                "session (issue #37). Scans the project, runs LLM analysis, "
                "auto-generates a goal + use_case + workload_profile, and persists "
                "the result as a SoCDesignState in the session store. The architect "
                "can then inspect it with `branes session show` or drill into "
                "specific kernels with /architect-drill. Use this when the user "
                "says 'I have a project, design hardware for it'."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "project_path": {
                        "type": "string",
                        "description": "Path to the project directory",
                    },
                    "power_budget_watts": {
                        "type": "number",
                        "description": "Optional max power in watts",
                    },
                    "latency_target_ms": {
                        "type": "number",
                        "description": "Optional max latency in milliseconds",
                    },
                    "max_area_mm2": {
                        "type": "number",
                        "description": "Optional max die area in mm²",
                    },
                    "max_cost_usd": {
                        "type": "number",
                        "description": "Optional max unit cost in USD",
                    },
                },
                "required": ["project_path"],
            },
        },
        {
            "name": "assess_codebase_on_hardware",
            "description": (
                "End-to-end codebase hardware assessment. Scans the project, runs LLM "
                "analysis to extract compute kernels, converts to workload profile, and "
                "runs it through the hardware explorer and PPA pipeline. Returns hardware "
                "recommendations with scores."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "project_path": {
                        "type": "string",
                        "description": "Path to the project directory",
                    },
                    "target_hardware": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Hardware targets to assess (e.g., ['jetson_orin', 'custom_kpu']). "
                            "If not specified, explores all available hardware."
                        ),
                    },
                    "power_budget_watts": {
                        "type": "number",
                        "description": "Maximum power consumption in watts (optional)",
                    },
                    "latency_target_ms": {
                        "type": "number",
                        "description": "Target end-to-end latency in milliseconds (optional)",
                    },
                },
                "required": ["project_path"],
            },
        },
    ]


def create_codebase_tool_executors() -> dict[str, Callable]:
    """Create executor functions for codebase tools.

    Returns:
        Dictionary mapping tool names to executor functions
    """

    def scan_project(project_path: str) -> str:
        """Execute project scan."""
        try:
            path = Path(project_path).expanduser().resolve()
            if not path.is_dir():
                return json.dumps({"error": f"Not a directory: {project_path}"})

            scanner = CodebaseScanner()
            result = scanner.scan(path)
            return json.dumps(result.model_dump(), indent=2, default=str)
        except Exception as e:
            return f"Error scanning project: {str(e)}\n{traceback.format_exc()}"

    def analyze_codebase(project_path: str) -> str:
        """Execute full LLM codebase analysis."""
        try:
            from embodied_ai_architect.codebase.analyzer import CodeAnalyzer
            from embodied_ai_architect.codebase.converter import infer_constraints
            from embodied_ai_architect.llm.client import LLMClient

            path = Path(project_path).expanduser().resolve()
            if not path.is_dir():
                return json.dumps({"error": f"Not a directory: {project_path}"})

            # Scan first
            scanner = CodebaseScanner()
            scan_result = scanner.scan(path)

            # Analyze
            llm = LLMClient()
            analyzer = CodeAnalyzer(llm)
            analysis = analyzer.analyze(scan_result, path)

            # Issue #38: include suggested constraints in the response
            payload = analysis.model_dump()
            payload["suggested_constraints"] = infer_constraints(analysis).model_dump()
            return json.dumps(payload, indent=2, default=str)
        except ImportError:
            return json.dumps(
                {
                    "error": (
                        "LLM client not available. "
                        "Install with: pip install embodied-ai-architect[chat]"
                    )
                }
            )
        except Exception as e:
            return f"Error analyzing codebase: {str(e)}\n{traceback.format_exc()}"

    def assess_codebase_on_hardware(
        project_path: str,
        target_hardware: list[str] | None = None,
        power_budget_watts: float | None = None,
        latency_target_ms: float | None = None,
    ) -> str:
        """Execute end-to-end codebase hardware assessment."""
        try:
            path = Path(project_path).expanduser().resolve()
            if not path.is_dir():
                return json.dumps({"error": f"Not a directory: {project_path}"})

            # Step 1: Scan
            scanner = CodebaseScanner()
            scan_result = scanner.scan(path)

            # Step 2: LLM analysis (best-effort)
            analysis = None
            try:
                from embodied_ai_architect.llm.client import LLMClient
                from embodied_ai_architect.codebase.analyzer import CodeAnalyzer

                llm = LLMClient()
                analyzer = CodeAnalyzer(llm)
                analysis = analyzer.analyze(scan_result, path)
            except (ImportError, Exception):
                # Fall back to scan-only analysis
                from embodied_ai_architect.codebase.models import CodebaseAnalysisResult

                analysis = CodebaseAnalysisResult(
                    project_name=scan_result.project_name,
                    languages=scan_result.languages,
                    source_files=scan_result.source_files,
                    ml_models=scan_result.ml_models,
                    build_system=scan_result.build_system,
                    dependencies=scan_result.dependencies,
                )

            # Step 3: Convert to workload profile
            converter = CodebaseConverter()
            workload_profile = converter.to_workload_profile(analysis)

            # Step 4: Run through hardware explorer if available
            hw_results = _run_hw_assessment(
                workload_profile, target_hardware, power_budget_watts, latency_target_ms
            )

            # Step 5: When no specific targets were requested, also call the
            # workload-based recommender (issue #39) so the chat sees a
            # ranked list of candidates.
            recommendations: list[dict[str, Any]] = []
            if not target_hardware:
                from embodied_ai_architect.codebase.recommender import recommend_hardware

                recs = recommend_hardware(
                    workload_profile,
                    top_k=5,
                    power_envelope_watts=power_budget_watts or 0.0,
                )
                recommendations = [r.to_dict() for r in recs]

            output = {
                "project": scan_result.project_name,
                "languages": scan_result.languages,
                "build_system": scan_result.build_system,
                "kernel_count": len(analysis.kernels),
                "ml_models_found": len(analysis.ml_models),
                "workload_profile": workload_profile,
                "hardware_assessment": hw_results,
                "recommendations": recommendations,
                "summary": analysis.summary
                or (
                    f"Analyzed {scan_result.project_name}: "
                    f"{len(scan_result.source_files)} files, "
                    f"{len(analysis.kernels)} compute kernels"
                ),
            }

            return json.dumps(output, indent=2, default=str)
        except Exception as e:
            return f"Error: {str(e)}\n{traceback.format_exc()}"

    def design_from_codebase(
        project_path: str,
        power_budget_watts: float | None = None,
        latency_target_ms: float | None = None,
        max_area_mm2: float | None = None,
        max_cost_usd: float | None = None,
    ) -> str:
        """Bridge codebase analysis → saved SoC design session (issue #37)."""
        try:
            from embodied_ai_architect.codebase.analyzer import CodeAnalyzer
            from embodied_ai_architect.codebase.converter import codebase_to_soc_state
            from embodied_ai_architect.graphs.session_store import SessionStore
            from embodied_ai_architect.graphs.soc_state import DesignConstraints
            from embodied_ai_architect.llm.client import LLMClient

            path = Path(project_path).expanduser().resolve()
            if not path.is_dir():
                return json.dumps({"error": f"Not a directory: {project_path}"})

            # Scan + analyze
            scanner = CodebaseScanner()
            scan_result = scanner.scan(path)
            llm = LLMClient()
            analyzer = CodeAnalyzer(llm)
            analysis = analyzer.analyze(scan_result, path)

            # Build constraints from kwargs
            constraint_kwargs: dict[str, Any] = {}
            if power_budget_watts is not None:
                constraint_kwargs["max_power_watts"] = power_budget_watts
            if latency_target_ms is not None:
                constraint_kwargs["max_latency_ms"] = latency_target_ms
            if max_area_mm2 is not None:
                constraint_kwargs["max_area_mm2"] = max_area_mm2
            if max_cost_usd is not None:
                constraint_kwargs["max_cost_usd"] = max_cost_usd
            constraints = DesignConstraints(**constraint_kwargs) if constraint_kwargs else None

            state = codebase_to_soc_state(
                analysis,
                constraints=constraints,
                project_path=str(path),
            )

            store = SessionStore()
            session_id = store.save(state)

            return json.dumps(
                {
                    "session_id": session_id,
                    "goal": state.get("goal", ""),
                    "use_case": state.get("use_case", ""),
                    "kernel_count": len(analysis.kernels),
                    "workload_summary": {
                        "total_gflops": state.get("workload_profile", {}).get(
                            "total_estimated_gflops", 0
                        ),
                        "total_memory_mb": state.get("workload_profile", {}).get(
                            "total_estimated_memory_mb", 0
                        ),
                        "dominant_op": state.get("workload_profile", {}).get(
                            "dominant_op", "unknown"
                        ),
                    },
                    "next_steps": [
                        f"branes session show {session_id}",
                        "/architect-assess in chat for the metrics dashboard",
                        "/architect-drill <kernel> for source-level deep dive",
                    ],
                },
                indent=2,
                default=str,
            )
        except Exception as e:
            return f"Error: {str(e)}\n{traceback.format_exc()}"

    return {
        "scan_project": scan_project,
        "analyze_codebase": analyze_codebase,
        "assess_codebase_on_hardware": assess_codebase_on_hardware,
        "design_from_codebase": design_from_codebase,
    }


def _run_hw_assessment(
    workload_profile: dict,
    target_hardware: list[str] | None,
    power_budget_watts: float | None,
    latency_target_ms: float | None,
) -> dict:
    """Run hardware assessment using the existing pipeline.

    Attempts to use hw_explorer from the graphs module. Falls back to
    basic estimation if the graphs module is not available.
    """
    try:
        from embodied_ai_architect.graphs.specialists import hw_explorer
        from embodied_ai_architect.graphs.soc_state import SoCDesignState
        from embodied_ai_architect.graphs.task_graph import TaskNode

        # Build state with workload profile
        state = SoCDesignState()
        state["workload_profile"] = workload_profile
        state["use_case"] = "application_analysis"

        # Set constraints
        constraints: dict[str, Any] = {}
        if power_budget_watts is not None:
            constraints["power_w"] = power_budget_watts
        if latency_target_ms is not None:
            constraints["latency_ms"] = latency_target_ms
        if target_hardware:
            constraints["target_hardware"] = target_hardware
        state["constraints"] = constraints

        # Create a task node for hw_explorer
        task = TaskNode(
            id="hw_explore",
            name="Hardware Explorer",
            agent="hw_explorer",
            dependencies=[],
        )

        result = hw_explorer(task, state)
        return {
            "source": "hw_explorer",
            "candidates": result.get("hardware_candidates", []),
            "summary": result.get("summary", ""),
        }
    except (ImportError, Exception) as e:
        # Fall back to basic estimation
        total_gflops = workload_profile.get("total_estimated_gflops", 0)
        total_memory = workload_profile.get("total_estimated_memory_mb", 0)

        return {
            "source": "estimation",
            "note": f"Hardware explorer not available ({e}). Basic estimation only.",
            "requirements": {
                "compute_gflops": total_gflops,
                "memory_mb": total_memory,
                "dominant_op": workload_profile.get("dominant_op", "unknown"),
            },
            "suggestions": _basic_hw_suggestions(total_gflops, power_budget_watts),
        }


def _basic_hw_suggestions(total_gflops: float, power_budget: float | None) -> list[str]:
    """Generate basic hardware suggestions without the full pipeline."""
    suggestions = []

    if total_gflops < 1:
        suggestions.append("Microcontroller or low-power CPU sufficient")
    elif total_gflops < 10:
        suggestions.append("Edge CPU (ARM Cortex-A78) or lightweight NPU")
        if power_budget and power_budget <= 5:
            suggestions.append("Consider Coral Edge TPU or Hailo-8")
    elif total_gflops < 50:
        suggestions.append("Edge GPU (Jetson Orin Nano) recommended")
        suggestions.append("Consider dedicated accelerator (NPU/KPU) for sustained throughput")
    else:
        suggestions.append("High-performance edge GPU (Jetson Orin AGX) or datacenter GPU")

    return suggestions
