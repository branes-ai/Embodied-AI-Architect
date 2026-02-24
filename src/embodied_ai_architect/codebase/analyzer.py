"""LLM-powered multi-pass code analyzer.

Reads source files through focused LLM passes to extract compute kernels,
dataflow structure, and pipeline architecture from arbitrary codebases.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from embodied_ai_architect.codebase.models import (
    CodebaseAnalysisResult,
    ComputeKernel,
    DataflowLink,
    ScanResult,
)

logger = logging.getLogger(__name__)

# Maximum characters to include per LLM pass to stay within context limits
MAX_CONTENT_PER_PASS = 48_000


class CodeAnalyzer:
    """Multi-pass LLM code analyzer.

    Strategy (4 passes, each focused and within context limits):
      1. Build & Config — read build files and dependency manifests
      2. Entry Points — read main files, identify pipeline structure
      3. Compute Kernels — read implementation files, extract ComputeKernel instances
      4. Synthesis — combine passes into CodebaseAnalysisResult
    """

    def __init__(self, llm_client: Any):
        """Initialize the analyzer.

        Args:
            llm_client: An LLMClient instance from embodied_ai_architect.llm.client
        """
        self.llm = llm_client

    def analyze(
        self,
        scan_result: ScanResult,
        project_path: Path,
    ) -> CodebaseAnalysisResult:
        """Run multi-pass LLM analysis on a scanned project.

        Args:
            scan_result: Output from CodebaseScanner.scan()
            project_path: Root directory of the project

        Returns:
            CodebaseAnalysisResult with kernels, dataflow, and summary
        """
        project_path = Path(project_path).resolve()

        # Pass 1: Build & Config context
        build_context = self._pass_build_config(scan_result, project_path)

        # Pass 2: Entry points & pipeline structure
        pipeline_context = self._pass_entry_points(scan_result, project_path, build_context)

        # Pass 3: Compute kernels extraction
        kernels_raw = self._pass_compute_kernels(scan_result, project_path, pipeline_context)

        # Pass 4: Synthesis — combine into final result
        result = self._pass_synthesis(scan_result, build_context, pipeline_context, kernels_raw)

        return result

    # ------------------------------------------------------------------
    # Pass 1: Build & Config
    # ------------------------------------------------------------------

    def _pass_build_config(self, scan: ScanResult, project_path: Path) -> dict:
        """Read build files and dependency manifests."""
        file_contents = self._read_files_by_role(
            scan, project_path, roles=["config", "build"], max_chars=MAX_CONTENT_PER_PASS
        )

        # Also read the build system file directly
        build_file_names = {
            "cmake": "CMakeLists.txt",
            "cargo": "Cargo.toml",
            "pip/poetry": "pyproject.toml",
            "make": "Makefile",
        }
        build_name = build_file_names.get(scan.build_system)
        if build_name:
            bf = project_path / build_name
            if bf.exists():
                try:
                    content = bf.read_text(encoding="utf-8", errors="replace")[:8000]
                    file_contents[build_name] = content
                except OSError:
                    pass

        if not file_contents:
            return {"build_system": scan.build_system, "dependencies": scan.dependencies}

        prompt = self._build_prompt(
            "You are analyzing a software project's build configuration.",
            (
                "Analyze these build/config files and return a JSON object with:\n"
                '- "build_system": string (cmake, cargo, pip/poetry, make)\n'
                '- "dependencies": list of dependency names\n'
                '- "project_type": string (ml_app, embedded_app, hybrid, library)\n'
                '- "key_frameworks": list of frameworks used (pytorch, tensorflow, opencv, eigen, etc.)\n'
                "Return ONLY valid JSON, no markdown fences."
            ),
            file_contents,
        )

        raw = self._call_llm(prompt)
        return self._parse_json(
            raw,
            default={
                "build_system": scan.build_system,
                "dependencies": scan.dependencies,
                "project_type": "unknown",
                "key_frameworks": [],
            },
        )

    # ------------------------------------------------------------------
    # Pass 2: Entry Points
    # ------------------------------------------------------------------

    def _pass_entry_points(self, scan: ScanResult, project_path: Path, build_ctx: dict) -> dict:
        """Read entry-point files and identify pipeline structure."""
        file_contents = self._read_files_by_role(
            scan, project_path, roles=["entry_point"], max_chars=MAX_CONTENT_PER_PASS
        )

        if not file_contents:
            return {"pipeline_stages": [], "execution_model": "unknown"}

        frameworks = build_ctx.get("key_frameworks", [])

        prompt = self._build_prompt(
            "You are analyzing entry-point files of a software project.",
            (
                f"Known frameworks: {frameworks}\n\n"
                "Analyze these entry-point files and return a JSON object with:\n"
                '- "pipeline_stages": list of objects with "name", "description", '
                '"source_file" fields\n'
                '- "execution_model": one of "sequential", "multi_rate", "event_driven"\n'
                '- "main_loop_frequency_hz": estimated main loop rate (0 if unknown)\n'
                "Return ONLY valid JSON, no markdown fences."
            ),
            file_contents,
        )

        raw = self._call_llm(prompt)
        return self._parse_json(
            raw,
            default={
                "pipeline_stages": [],
                "execution_model": "unknown",
                "main_loop_frequency_hz": 0,
            },
        )

    # ------------------------------------------------------------------
    # Pass 3: Compute Kernels
    # ------------------------------------------------------------------

    def _pass_compute_kernels(
        self, scan: ScanResult, project_path: Path, pipeline_ctx: dict
    ) -> list[dict]:
        """Read implementation files and extract compute kernels."""
        # Prioritize library files (implementation), sorted by line count (largest first)
        file_contents = self._read_files_by_role(
            scan, project_path, roles=["library", "entry_point"], max_chars=MAX_CONTENT_PER_PASS
        )

        if not file_contents:
            return []

        stages = pipeline_ctx.get("pipeline_stages", [])
        stages_desc = json.dumps(stages, indent=2) if stages else "unknown"

        prompt = self._build_prompt(
            "You are extracting computational kernels from source code.",
            (
                f"Known pipeline stages: {stages_desc}\n\n"
                "For each computationally significant function/region, return a JSON list "
                "of kernel objects with these fields:\n"
                '- "name": string (function or kernel name)\n'
                '- "source_file": string (relative file path)\n'
                '- "line_range": [start, end]\n'
                '- "kernel_type": one of "ml_inference", "signal_processing", '
                '"control_loop", "image_processing", "sensor_fusion", "io_bound", '
                '"general_compute"\n'
                '- "estimated_ops_per_invocation": float (estimate FLOPS/call, '
                "use 1e6 for light, 1e9 for heavy, 1e12 for very heavy)\n"
                '- "data_types": list of strings (e.g., ["float32", "int8"])\n'
                '- "parallelism": one of "data_parallel", "pipeline", "sequential"\n'
                '- "memory_access_pattern": one of "streaming", "random", "reuse_heavy"\n'
                '- "invocation_frequency_hz": float (0 if unknown)\n'
                '- "frameworks": list of frameworks used (e.g., ["pytorch", "opencv"])\n'
                "\nReturn ONLY a valid JSON list, no markdown fences."
            ),
            file_contents,
        )

        raw = self._call_llm(prompt)
        result = self._parse_json(raw, default=[])
        if isinstance(result, dict):
            result = result.get("kernels", [])
        return result if isinstance(result, list) else []

    # ------------------------------------------------------------------
    # Pass 4: Synthesis
    # ------------------------------------------------------------------

    def _pass_synthesis(
        self,
        scan: ScanResult,
        build_ctx: dict,
        pipeline_ctx: dict,
        kernels_raw: list[dict],
    ) -> CodebaseAnalysisResult:
        """Combine all passes into a CodebaseAnalysisResult."""
        # Build kernel models
        kernels: list[ComputeKernel] = []
        for k in kernels_raw:
            line_range = k.get("line_range", [0, 0])
            if isinstance(line_range, list) and len(line_range) == 2:
                line_range = (int(line_range[0]), int(line_range[1]))
            else:
                line_range = (0, 0)

            kernels.append(
                ComputeKernel(
                    name=k.get("name", "unknown"),
                    source_file=k.get("source_file", ""),
                    line_range=line_range,
                    kernel_type=k.get("kernel_type", "general_compute"),
                    estimated_ops_per_invocation=float(k.get("estimated_ops_per_invocation", 0)),
                    data_types=k.get("data_types", ["float32"]),
                    parallelism=k.get("parallelism", "sequential"),
                    memory_access_pattern=k.get("memory_access_pattern", "streaming"),
                    invocation_frequency_hz=float(k.get("invocation_frequency_hz", 0)),
                    frameworks=k.get("frameworks", []),
                )
            )

        # Infer dataflow from pipeline stages
        dataflow: list[DataflowLink] = []
        if len(kernels) > 1:
            for i in range(len(kernels) - 1):
                dataflow.append(
                    DataflowLink(
                        source_kernel=kernels[i].name,
                        sink_kernel=kernels[i + 1].name,
                        data_size_bytes=0,
                        transfer_type="memory",
                    )
                )

        # Generate summary
        summary = self._generate_summary(scan, build_ctx, pipeline_ctx, kernels)

        return CodebaseAnalysisResult(
            project_name=scan.project_name,
            languages=scan.languages,
            source_files=scan.source_files,
            kernels=kernels,
            dataflow=dataflow,
            ml_models=scan.ml_models,
            build_system=build_ctx.get("build_system", scan.build_system),
            dependencies=build_ctx.get("dependencies", scan.dependencies),
            summary=summary,
        )

    def _generate_summary(
        self,
        scan: ScanResult,
        build_ctx: dict,
        pipeline_ctx: dict,
        kernels: list[ComputeKernel],
    ) -> str:
        """Generate a human-readable summary using LLM."""
        kernel_types = {}
        for k in kernels:
            kernel_types[k.kernel_type] = kernel_types.get(k.kernel_type, 0) + 1

        context = (
            f"Project: {scan.project_name}\n"
            f"Languages: {', '.join(scan.languages)}\n"
            f"Build system: {scan.build_system}\n"
            f"Source files: {len(scan.source_files)}, Total lines: {scan.total_lines}\n"
            f"ML models found: {len(scan.ml_models)}\n"
            f"Compute kernels: {len(kernels)}\n"
            f"Kernel types: {kernel_types}\n"
            f"Project type: {build_ctx.get('project_type', 'unknown')}\n"
            f"Execution model: {pipeline_ctx.get('execution_model', 'unknown')}\n"
        )

        prompt = (
            f"Summarize this project in 2-3 sentences for a hardware architect:\n\n"
            f"{context}\n\n"
            f"Focus on: what the application does, its computational characteristics, "
            f"and what kind of hardware it would benefit from."
        )

        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.llm.chat(messages)
            return response.text.strip()
        except Exception as e:
            logger.warning("Summary generation failed: %s", e)
            return (
                f"{scan.project_name}: {', '.join(scan.languages)} project with "
                f"{len(kernels)} compute kernels ({', '.join(kernel_types.keys())})"
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _read_files_by_role(
        self,
        scan: ScanResult,
        project_path: Path,
        roles: list[str],
        max_chars: int,
    ) -> dict[str, str]:
        """Read source files matching given roles, up to max_chars total."""
        contents: dict[str, str] = {}
        total_chars = 0

        for sf in scan.source_files:
            if sf.role not in roles:
                continue
            fpath = project_path / sf.path
            if not fpath.exists():
                continue
            try:
                text = fpath.read_text(encoding="utf-8", errors="replace")
                if total_chars + len(text) > max_chars:
                    remaining = max_chars - total_chars
                    if remaining > 500:  # Only include if we can fit meaningful content
                        contents[sf.path] = text[:remaining] + "\n... (truncated)"
                    break
                contents[sf.path] = text
                total_chars += len(text)
            except OSError:
                continue

        return contents

    def _build_prompt(
        self, system_ctx: str, instruction: str, file_contents: dict[str, str]
    ) -> str:
        """Build a prompt with file contents for the LLM."""
        parts = [system_ctx, "", instruction, ""]
        for path, content in file_contents.items():
            parts.append(f"--- {path} ---")
            parts.append(content)
            parts.append("")
        return "\n".join(parts)

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM and return the text response."""
        messages = [{"role": "user", "content": prompt}]
        try:
            response = self.llm.chat(messages)
            return response.text
        except Exception as e:
            logger.error("LLM call failed: %s", e)
            return "{}"

    def _parse_json(self, text: str, default: Any) -> Any:
        """Parse JSON from LLM response, stripping markdown fences if present."""
        text = text.strip()

        # Strip markdown code fences
        if text.startswith("```"):
            lines = text.splitlines()
            # Remove first and last fence lines
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON within the text
            for start_char, end_char in [("{", "}"), ("[", "]")]:
                start = text.find(start_char)
                end = text.rfind(end_char)
                if start >= 0 and end > start:
                    try:
                        return json.loads(text[start : end + 1])
                    except json.JSONDecodeError:
                        continue
            logger.warning("Failed to parse LLM JSON response: %s...", text[:200])
            return default
