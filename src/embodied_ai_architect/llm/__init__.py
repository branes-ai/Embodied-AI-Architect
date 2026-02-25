"""LLM integration for interactive Embodied AI Architect agent."""

from .client import LLMClient
from .agent import ArchitectAgent
from .tools import get_tool_definitions, create_tool_executors

# Optional graphs integration
try:
    from .graphs_tools import (
        get_graphs_tool_definitions,
        create_graphs_tool_executors,
        HAS_GRAPHS,
    )
except ImportError:
    HAS_GRAPHS = False
    get_graphs_tool_definitions = None
    create_graphs_tool_executors = None

# Architecture analysis tools
try:
    from .architecture_tools import (
        get_architecture_tool_definitions,
        create_architecture_tool_executors,
    )

    HAS_ARCHITECTURE_TOOLS = True
except ImportError:
    HAS_ARCHITECTURE_TOOLS = False
    get_architecture_tool_definitions = None
    create_architecture_tool_executors = None

__all__ = [
    "LLMClient",
    "ArchitectAgent",
    "get_tool_definitions",
    "create_tool_executors",
    "HAS_GRAPHS",
    "get_graphs_tool_definitions",
    "create_graphs_tool_executors",
    "HAS_ARCHITECTURE_TOOLS",
    "get_architecture_tool_definitions",
    "create_architecture_tool_executors",
]
