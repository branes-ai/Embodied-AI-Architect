"""Research library for HW accelerator context injection.

Provides curated document retrieval for enriching LLM prompts with
domain knowledge about hardware accelerators, architecture patterns,
efficiency data, workload characteristics, and mission profiles.
"""

from .library import ResearchLibrary, Document
from .models import (
    MissionPlan,
    SubCapability,
    OperatorSpec,
    OperatorGraph,
    DesignSpaceDefinition,
)

__all__ = [
    "ResearchLibrary",
    "Document",
    "MissionPlan",
    "SubCapability",
    "OperatorSpec",
    "OperatorGraph",
    "DesignSpaceDefinition",
]
