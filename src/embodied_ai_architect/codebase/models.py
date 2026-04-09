"""Data models for codebase analysis results.

Provides structured representations for source files, compute kernels,
dataflow links, and full codebase analysis results. Used by the scanner,
analyzer, and converter pipeline.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class SourceFile(BaseModel):
    """A source file discovered during project scanning."""

    path: str
    language: str  # "cpp", "rust", "python", "header", "config", "build", "unknown"
    lines: int = 0
    role: str = "library"  # "entry_point", "library", "config", "build", "test"
    entry_type: str | None = None  # "application", "example", "test", or None


class ComputeKernel(BaseModel):
    """A computational kernel identified in the codebase.

    Represents a unit of computation that can be mapped to hardware operators
    for PPA assessment.
    """

    name: str
    source_file: str
    line_range: tuple[int, int] = (0, 0)
    kernel_type: str = "general_compute"
    # Options: "ml_inference", "signal_processing", "control_loop",
    #          "image_processing", "sensor_fusion", "io_bound", "general_compute"
    estimated_ops_per_invocation: float = 0.0
    data_types: list[str] = Field(default_factory=lambda: ["float32"])
    parallelism: str = "sequential"  # "data_parallel", "pipeline", "sequential"
    memory_access_pattern: str = "streaming"  # "streaming", "random", "reuse_heavy"
    invocation_frequency_hz: float = 0.0
    frameworks: list[str] = Field(default_factory=list)


class DataflowLink(BaseModel):
    """A data dependency between two compute kernels."""

    source_kernel: str
    sink_kernel: str
    data_size_bytes: int = 0
    transfer_type: str = "memory"  # "memory", "dma", "network"


class ScanResult(BaseModel):
    """Result of the static file scanner pass.

    Contains the file inventory, detected build system, ML model files,
    and dependencies — everything needed to prioritize LLM analysis passes.
    """

    project_name: str
    project_path: str
    languages: list[str] = Field(default_factory=list)
    source_files: list[SourceFile] = Field(default_factory=list)
    ml_models: list[dict] = Field(default_factory=list)
    build_system: str = "unknown"  # "cmake", "cargo", "pip/poetry", "make", "unknown"
    dependencies: list[str] = Field(default_factory=list)
    total_lines: int = 0
    recommended_entry_point: dict | None = None
    # {"path": "...", "confidence": "high"|"medium"|"low", "alternatives": [...]}


class CodebaseAnalysisResult(BaseModel):
    """Full analysis result combining scanner + LLM analysis.

    This is the rich representation of the application that the converter
    maps into the workload_profile format consumed by the existing PPA pipeline.
    """

    project_name: str
    languages: list[str] = Field(default_factory=list)
    source_files: list[SourceFile] = Field(default_factory=list)
    kernels: list[ComputeKernel] = Field(default_factory=list)
    dataflow: list[DataflowLink] = Field(default_factory=list)
    ml_models: list[dict] = Field(default_factory=list)
    build_system: str = "unknown"
    dependencies: list[str] = Field(default_factory=list)
    summary: str = ""


# ---------------------------------------------------------------------------
# Issue #38: Constraint inference from codebase characteristics
# ---------------------------------------------------------------------------


class SuggestedConstraint(BaseModel):
    """A single inferred design constraint with provenance.

    Each suggestion carries the constraint name (one of the DesignConstraints
    fields, plus a few advisory keys), its inferred value, a confidence
    bucket, and a one-line rationale that names the heuristic that produced
    it. The architect uses these to populate `DesignConstraints` (or to
    override the heuristic when it's wrong).
    """

    name: str
    value: float | str | bool
    confidence: str = "medium"  # "high" | "medium" | "low"
    rationale: str = ""


class SuggestedConstraints(BaseModel):
    """Collection of inferred constraints from codebase characteristics."""

    constraints: list[SuggestedConstraint] = Field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict[str, dict]:
        """Render as a {name: {value, confidence, rationale}} dict for display."""
        return {
            c.name: {
                "value": c.value,
                "confidence": c.confidence,
                "rationale": c.rationale,
            }
            for c in self.constraints
        }

    def to_design_constraints_kwargs(
        self,
        min_confidence: str = "high",
    ) -> dict[str, float]:
        """Return numeric suggestions as DesignConstraints kwargs.

        Only the canonical numeric fields (max_power_watts, max_latency_ms,
        max_area_mm2, max_cost_usd) are included so the output can be
        spread directly into a `DesignConstraints` constructor. Advisory
        keys like `hardware_class` and `memory_bw_critical` are excluded.

        Filters by `min_confidence` (default "high") so callers don't
        silently propagate gut-feel guesses into binding design constraints
        (CodeRabbit PR #89). Pass `min_confidence="low"` to opt into all
        suggestions.
        """
        canonical = {
            "max_power_watts",
            "max_latency_ms",
            "max_area_mm2",
            "max_cost_usd",
        }
        rank = {"high": 3, "medium": 2, "low": 1}
        threshold = rank.get(min_confidence, 3)
        return {
            c.name: float(c.value)
            for c in self.constraints
            if c.name in canonical
            and isinstance(c.value, (int, float))
            and rank.get(c.confidence, 0) >= threshold
        }
