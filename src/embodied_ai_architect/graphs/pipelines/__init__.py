"""Pre-built LangGraph pipelines for common embodied AI workflows.

Requires the 'langgraph' optional dependency:

    pip install -e ".[langgraph]"
"""

__all__ = [
    "build_perception_graph",
    "build_autonomy_graph",
]


def __getattr__(name):
    """Lazy import for langgraph-dependent modules."""
    if name == "build_perception_graph":
        from embodied_ai_architect.graphs.pipelines.perception import build_perception_graph

        return build_perception_graph
    elif name == "build_autonomy_graph":
        from embodied_ai_architect.graphs.pipelines.autonomy import build_autonomy_graph

        return build_autonomy_graph
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
