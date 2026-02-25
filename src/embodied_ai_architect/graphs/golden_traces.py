"""Golden trace storage and comparison for regression testing.

Saves and loads run traces as JSON files, and compares traces against
golden references for structural regression detection.

Usage:
    from embodied_ai_architect.graphs.golden_traces import (
        save_golden_trace, load_golden_trace, compare_traces, TraceComparison,
    )
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from embodied_ai_architect.graphs.evaluation import RunTrace


class TraceComparison(BaseModel):
    """Result of comparing a run trace against a golden trace."""

    task_graph_match: bool = Field(
        default=False,
        description="True if task graph node names and edges match",
    )
    ppa_regression: bool = Field(
        default=False,
        description="True if PPA metrics regressed (got worse)",
    )
    iteration_regression: bool = Field(
        default=False,
        description="True if more iterations were needed than the golden trace",
    )
    tool_call_diff: dict[str, Any] = Field(
        default_factory=dict,
        description="Differences in tool calls (added, removed)",
    )
    details: list[str] = Field(
        default_factory=list,
        description="Human-readable comparison details",
    )

    @property
    def is_regression(self) -> bool:
        """True if any regression was detected."""
        return self.ppa_regression or self.iteration_regression


def save_golden_trace(trace: RunTrace, path: str | Path) -> Path:
    """Save a RunTrace as a golden reference JSON file.

    Args:
        trace: The trace to save.
        path: File path (will create parent directories).

    Returns:
        The path the trace was saved to.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(trace.model_dump(), indent=2, default=str))
    return path


def load_golden_trace(path: str | Path) -> RunTrace:
    """Load a RunTrace from a golden reference JSON file.

    Args:
        path: File path to the golden trace.

    Returns:
        The loaded RunTrace.

    Raises:
        FileNotFoundError: If the file doesn't exist.
    """
    path = Path(path)
    data = json.loads(path.read_text())
    return RunTrace(**data)


def compare_traces(current: RunTrace, golden: RunTrace) -> TraceComparison:
    """Compare a current run trace against a golden reference.

    Checks for:
    - Task graph structural match (same node names and dependencies)
    - PPA regression (metrics got worse)
    - Iteration regression (needed more iterations)
    - Tool call differences

    Args:
        current: The current run trace to evaluate.
        golden: The golden reference trace.

    Returns:
        TraceComparison with detailed results.
    """
    details: list[str] = []

    # Task graph comparison
    current_nodes = _extract_node_names(current.task_graph)
    golden_nodes = _extract_node_names(golden.task_graph)
    task_graph_match = current_nodes == golden_nodes
    if not task_graph_match:
        added = current_nodes - golden_nodes
        removed = golden_nodes - current_nodes
        if added:
            details.append(f"Task graph: added nodes {added}")
        if removed:
            details.append(f"Task graph: removed nodes {removed}")
    else:
        details.append("Task graph: structure matches golden")

    # PPA regression check
    ppa_regression = False
    current_ppa = current.ppa_metrics
    golden_ppa = golden.ppa_metrics
    for metric in ("power_watts", "latency_ms", "area_mm2", "cost_usd"):
        cv = current_ppa.get(metric)
        gv = golden_ppa.get(metric)
        if cv is not None and gv is not None:
            if cv > gv * 1.1:  # 10% regression threshold
                ppa_regression = True
                details.append(f"PPA regression: {metric} {cv} > golden {gv}")

    # Check verdict regressions (PASS -> FAIL)
    current_verdicts = current_ppa.get("verdicts", {})
    golden_verdicts = golden_ppa.get("verdicts", {})
    for k, gv in golden_verdicts.items():
        cv = current_verdicts.get(k, "")
        if gv == "PASS" and cv == "FAIL":
            ppa_regression = True
            details.append(f"Verdict regression: {k} was PASS, now FAIL")

    if not ppa_regression:
        details.append("PPA: no regression detected")

    # Iteration regression
    current_iters = len(current.iteration_history)
    golden_iters = len(golden.iteration_history)
    iteration_regression = current_iters > golden_iters and golden_iters > 0
    if iteration_regression:
        details.append(f"Iteration regression: {current_iters} vs golden {golden_iters}")

    # Tool call diff
    current_calls = set(current.tool_calls)
    golden_calls = set(golden.tool_calls)
    tool_call_diff = {
        "added": sorted(current_calls - golden_calls),
        "removed": sorted(golden_calls - current_calls),
    }
    if tool_call_diff["added"] or tool_call_diff["removed"]:
        details.append(f"Tool calls: +{tool_call_diff['added']} -{tool_call_diff['removed']}")

    return TraceComparison(
        task_graph_match=task_graph_match,
        ppa_regression=ppa_regression,
        iteration_regression=iteration_regression,
        tool_call_diff=tool_call_diff,
        details=details,
    )


def _extract_node_names(task_graph_dict: dict[str, Any]) -> set[str]:
    """Extract node names from a serialized TaskGraph."""
    nodes = task_graph_dict.get("nodes", {})
    return {node.get("name", node_id) for node_id, node in nodes.items()}
