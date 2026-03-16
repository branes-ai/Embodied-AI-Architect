"""LLM tool definitions for mission decomposition and research retrieval.

Follows the get_*_tool_definitions() + create_*_tool_executors() pattern.
Provides 3 tools: decompose_mission, retrieve_research, formulate_design_space.

Usage:
    from embodied_ai_architect.llm.decomposition_tools import (
        get_decomposition_tool_definitions,
        create_decomposition_tool_executors,
    )
"""

from __future__ import annotations

import json
import traceback
from typing import Any, Callable


def get_decomposition_tool_definitions() -> list[dict[str, Any]]:
    """Get tool definitions for mission decomposition in Anthropic format."""
    return [
        {
            "name": "decompose_mission",
            "description": (
                "Decompose a natural-language mission description into a structured "
                "MissionPlan with sub-capabilities, an operator graph, and design "
                "constraints. Uses research library context for realistic budgets. "
                "Works without an API key via rule-based heuristics."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "mission_description": {
                        "type": "string",
                        "description": (
                            "Natural-language mission description "
                            "(e.g., 'Drone perception for GPS-denied navigation at 5 m/s')"
                        ),
                    },
                    "max_context_tokens": {
                        "type": "integer",
                        "description": (
                            "Token budget for research context injection (default: 8000)"
                        ),
                    },
                },
                "required": ["mission_description"],
            },
        },
        {
            "name": "retrieve_research",
            "description": (
                "Retrieve relevant hardware accelerator research documents from the "
                "curated library. Returns formatted context blocks with specs, "
                "architecture patterns, efficiency data, and mission profiles. "
                "Tag-based retrieval — deterministic and auditable."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": ("Tags to match (e.g., ['drone', 'edge', 'detection'])"),
                    },
                    "domain": {
                        "type": "string",
                        "description": (
                            "Domain filter (e.g., 'accelerator', 'architecture', "
                            "'efficiency', 'workload', 'mission')"
                        ),
                    },
                    "relevance": {
                        "type": "string",
                        "description": (
                            "Relevance context (e.g., 'mission_decomposition', "
                            "'hardware_selection')"
                        ),
                    },
                    "mission_type": {
                        "type": "string",
                        "description": "Mission type to match against tags (e.g., 'drone')",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum documents to return (default: 5)",
                    },
                    "max_tokens": {
                        "type": "integer",
                        "description": "Token budget for the context block (default: 8000)",
                    },
                    "include_frontmatter": {
                        "type": "boolean",
                        "description": (
                            "Include quantitative frontmatter (TOPS, TOPS/W, etc.) "
                            "in output (default: true)"
                        ),
                    },
                },
                "required": [],
            },
        },
        {
            "name": "formulate_design_space",
            "description": (
                "Convert a MissionPlan's design space definition into optimization "
                "parameters for the MOO engine. Bridges mission decomposition to "
                "design space exploration by producing the constraints dict expected "
                "by create_soc_design_space()."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "mission_description": {
                        "type": "string",
                        "description": (
                            "Mission description to decompose and formulate. "
                            "If provided, runs decompose_mission first."
                        ),
                    },
                    "platform": {
                        "type": "string",
                        "description": (
                            "Override platform type (drone, ugv, vehicle, wearable, edge)"
                        ),
                    },
                    "power_budget_watts": {
                        "type": "number",
                        "description": "Override power budget in watts",
                    },
                    "latency_target_ms": {
                        "type": "number",
                        "description": "Override latency target in ms",
                    },
                },
                "required": ["mission_description"],
            },
        },
    ]


def create_decomposition_tool_executors(
    llm_client: Any | None = None,
) -> dict[str, Callable]:
    """Create executor functions for decomposition tools.

    Args:
        llm_client: Optional LLM client for LLM-powered decomposition.
            If None, decomposer falls back to rule-based heuristics.
    """
    from embodied_ai_architect.research.decomposer import MissionDecomposer
    from embodied_ai_architect.research.library import ResearchLibrary

    library = ResearchLibrary()
    decomposer = MissionDecomposer(llm_client=llm_client, research_library=library)

    # Cache last plan for formulate_design_space
    _state: dict[str, Any] = {"last_plan": None}

    def decompose_mission(
        mission_description: str,
        max_context_tokens: int = 8000,
    ) -> str:
        """Decompose a mission description into a structured plan."""
        try:
            plan = decomposer.decompose(
                mission_description,
                max_context_tokens=max_context_tokens,
            )
            _state["last_plan"] = plan

            result = {
                "mission": plan.mission,
                "platform": plan.platform,
                "mission_type": plan.mission_type,
                "environment": plan.environment,
                "sub_capabilities": [
                    {
                        "name": cap.name,
                        "description": cap.description,
                        "required": cap.required,
                        "latency_budget_ms": cap.latency_budget_ms,
                        "accuracy_requirement": cap.accuracy_requirement,
                        "rate_hz": cap.rate_hz,
                        "depends_on": cap.depends_on,
                    }
                    for cap in plan.sub_capabilities
                ],
                "operators": [
                    {
                        "name": op.name,
                        "type": op.operator_type.value,
                        "model": (
                            f"{op.model_family}/{op.model_variant}" if op.model_family else None
                        ),
                        "gflops": op.estimated_gflops,
                        "latency_ms": op.latency_budget_ms,
                        "rate_hz": op.rate_hz,
                    }
                    for op in plan.operator_graph.operators
                ],
                "total_gflops": plan.operator_graph.total_estimated_gflops,
                "constraints": [
                    {
                        "name": c.name,
                        "bound": f"{'<' if c.constraint_type == 'upper' else '>'} {c.value}",
                        "unit": c.unit,
                    }
                    for c in plan.design_space.constraints
                ],
                "design_space": {
                    "workload_type": plan.design_space.workload_type,
                    "quantization_dtype": plan.design_space.quantization_dtype,
                    "include_accuracy": plan.design_space.include_accuracy,
                    "platform": plan.design_space.platform,
                },
                "research_docs_used": plan.research_context_used,
                "rationale": plan.rationale,
            }
            return json.dumps(result, indent=2)
        except Exception as e:
            return f"Error decomposing mission: {e}\n{traceback.format_exc()}"

    def retrieve_research(
        tags: list[str] | None = None,
        domain: str | None = None,
        relevance: str | None = None,
        mission_type: str | None = None,
        max_results: int = 5,
        max_tokens: int = 8000,
        include_frontmatter: bool = True,
    ) -> str:
        """Retrieve relevant research documents."""
        try:
            docs = library.retrieve(
                tags=tags,
                domain=domain,
                relevance=relevance,
                mission_type=mission_type,
                max_results=max_results,
            )

            if not docs:
                available_tags = library.list_tags()
                available_domains = library.list_domains()
                return json.dumps(
                    {
                        "documents": [],
                        "message": "No matching documents found.",
                        "available_tags": available_tags[:20],
                        "available_domains": available_domains,
                    },
                    indent=2,
                )

            context_block = library.build_context_block(
                docs,
                max_tokens=max_tokens,
                include_frontmatter=include_frontmatter,
            )

            result = {
                "documents_found": len(docs),
                "documents": [
                    {
                        "title": d.title,
                        "path": d.path,
                        "domain": d.domain,
                        "tags": d.tags,
                        "estimated_tokens": d.estimated_tokens,
                    }
                    for d in docs
                ],
                "context_block": context_block,
            }
            return json.dumps(result, indent=2)
        except Exception as e:
            return f"Error retrieving research: {e}\n{traceback.format_exc()}"

    def formulate_design_space(
        mission_description: str,
        platform: str | None = None,
        power_budget_watts: float | None = None,
        latency_target_ms: float | None = None,
    ) -> str:
        """Convert a mission plan into MOO engine parameters."""
        try:
            # Always decompose to get a fresh plan
            plan = decomposer.decompose(mission_description)
            _state["last_plan"] = plan

            # Apply overrides
            if platform:
                plan.design_space.platform = platform
            constraints_dict = plan.design_space.to_constraints_dict()
            if power_budget_watts is not None:
                constraints_dict["max_power_watts"] = power_budget_watts
            if latency_target_ms is not None:
                constraints_dict["max_latency_ms"] = latency_target_ms

            result = {
                "mission": plan.mission,
                "platform": plan.design_space.platform,
                "constraints_for_create_soc_design_space": constraints_dict,
                "operator_graph_summary": {
                    "num_operators": len(plan.operator_graph.operators),
                    "total_gflops": plan.operator_graph.total_estimated_gflops,
                    "e2e_latency_ms": plan.operator_graph.end_to_end_latency_budget_ms,
                    "operators": [
                        f"{op.name} ({op.operator_type.value})"
                        for op in plan.operator_graph.operators
                    ],
                },
                "usage": (
                    "Pass constraints_for_create_soc_design_space to "
                    "explore_design_space tool or directly to "
                    "create_soc_design_space(**constraints)"
                ),
            }
            return json.dumps(result, indent=2)
        except Exception as e:
            return f"Error formulating design space: {e}\n{traceback.format_exc()}"

    return {
        "decompose_mission": decompose_mission,
        "retrieve_research": retrieve_research,
        "formulate_design_space": formulate_design_space,
    }
