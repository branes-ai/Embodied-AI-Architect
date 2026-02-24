"""CodebaseAnalyzer agent — wraps scanner + analyzer + converter pipeline.

Follows the BaseAgent pattern: accepts input_data dict, returns AgentResult.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

from embodied_ai_architect.agents.base import AgentResult, BaseAgent
from embodied_ai_architect.codebase.converter import CodebaseConverter
from embodied_ai_architect.codebase.scanner import CodebaseScanner

logger = logging.getLogger(__name__)


class CodebaseAnalyzerAgent(BaseAgent):
    """Agent that runs the full codebase analysis pipeline.

    Input:
        {
            "project_path": "/path/to/app",
            "target_hardware": ["jetson_orin", "custom_soc"],  # optional
            "skip_llm": false,  # optional, skip LLM analysis
        }

    Output AgentResult.data:
        {
            "scan_result": {...},
            "analysis": {...},          # if LLM analysis ran
            "workload_profile": {...},  # compatible with specialists pipeline
        }
    """

    def __init__(self):
        super().__init__(name="CodebaseAnalyzer")

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Execute the codebase analysis pipeline.

        Args:
            input_data: Dict with "project_path" and optional "target_hardware", "skip_llm".

        Returns:
            AgentResult with scan results, analysis, and workload profile.
        """
        project_path_str = input_data.get("project_path")
        if not project_path_str:
            return AgentResult(
                success=False,
                data={},
                error="Missing required 'project_path' in input_data",
            )

        project_path = Path(project_path_str).expanduser().resolve()
        if not project_path.is_dir():
            return AgentResult(
                success=False,
                data={},
                error=f"Not a directory: {project_path}",
            )

        skip_llm = input_data.get("skip_llm", False)

        try:
            # Step 1: Static scan
            scanner = CodebaseScanner()
            scan_result = scanner.scan(project_path)

            result_data: dict[str, Any] = {
                "scan_result": scan_result.model_dump(),
            }

            # Step 2: LLM analysis (optional)
            analysis = None
            if not skip_llm:
                try:
                    from embodied_ai_architect.llm.client import LLMClient
                    from embodied_ai_architect.codebase.analyzer import CodeAnalyzer

                    llm = LLMClient()
                    analyzer = CodeAnalyzer(llm)
                    analysis = analyzer.analyze(scan_result, project_path)
                    result_data["analysis"] = analysis.model_dump()
                except ImportError:
                    logger.info(
                        "LLM client not available, skipping code analysis. "
                        "Install with: pip install embodied-ai-architect[chat]"
                    )
                except Exception as e:
                    logger.warning("LLM analysis failed: %s", e)

            # Step 3: Convert to workload profile
            converter = CodebaseConverter()
            if analysis:
                workload_profile = converter.to_workload_profile(analysis)
            else:
                # Build a minimal analysis from scan results only
                from embodied_ai_architect.codebase.models import CodebaseAnalysisResult

                minimal = CodebaseAnalysisResult(
                    project_name=scan_result.project_name,
                    languages=scan_result.languages,
                    source_files=scan_result.source_files,
                    ml_models=scan_result.ml_models,
                    build_system=scan_result.build_system,
                    dependencies=scan_result.dependencies,
                )
                workload_profile = converter.to_workload_profile(minimal)

            result_data["workload_profile"] = workload_profile

            return AgentResult(
                success=True,
                data=result_data,
                metadata={
                    "project_path": str(project_path),
                    "languages": scan_result.languages,
                    "file_count": len(scan_result.source_files),
                    "llm_analysis": analysis is not None,
                },
            )

        except Exception as e:
            return AgentResult(
                success=False,
                data={},
                error=str(e),
            )
