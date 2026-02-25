"""RunTrace capture middleware for the dispatcher.

Wraps the Dispatcher to record timing, tool calls, and state diffs
for post-run evaluation.

Usage:
    from embodied_ai_architect.graphs.trace import TracingDispatcher, extract_trace_from_state

    dispatcher = TracingDispatcher(create_default_dispatcher())
    state = dispatcher.run(state)
    trace = extract_trace_from_state(initial_state, state, "demo_1")
"""

from __future__ import annotations

import logging
import time
from typing import Any

from embodied_ai_architect.graphs.dispatcher import Dispatcher
from embodied_ai_architect.graphs.evaluation import RunTrace
from embodied_ai_architect.graphs.soc_state import SoCDesignState, get_task_graph
from embodied_ai_architect.graphs.task_graph import TaskStatus

logger = logging.getLogger(__name__)


class TracingDispatcher:
    """Wraps a Dispatcher to capture execution traces.

    Records wall-clock timing, tool call order, and task outcomes
    for building RunTrace objects after a run.
    """

    def __init__(self, inner: Dispatcher) -> None:
        self._inner = inner
        self._tool_calls: list[str] = []
        self._start_time: float = 0.0
        self._end_time: float = 0.0

    @property
    def registered_agents(self) -> list[str]:
        return self._inner.registered_agents

    def register(self, agent_name: str, executor: Any) -> None:
        self._inner.register(agent_name, executor)

    def register_many(self, agents: dict[str, Any]) -> None:
        self._inner.register_many(agents)

    def step(self, state: SoCDesignState) -> SoCDesignState:
        """Execute one batch of ready tasks with tracing."""
        graph = get_task_graph(state)
        ready = graph.ready_tasks()
        for task in ready:
            self._tool_calls.append(task.agent)

        return self._inner.step(state)

    def run(self, state: SoCDesignState, max_steps: int = 50) -> SoCDesignState:
        """Execute all tasks with timing capture."""
        self._tool_calls = []
        self._start_time = time.monotonic()
        result = self._inner.run(state, max_steps=max_steps)
        self._end_time = time.monotonic()
        return result

    @property
    def duration_seconds(self) -> float:
        if self._end_time > self._start_time:
            return self._end_time - self._start_time
        return 0.0

    @property
    def tool_calls(self) -> list[str]:
        return list(self._tool_calls)

    def get_dispatch_summary(self, state: SoCDesignState) -> str:
        return self._inner.get_dispatch_summary(state)


def extract_trace_from_state(
    initial_state: SoCDesignState,
    final_state: SoCDesignState,
    demo_name: str,
    duration_seconds: float = 0.0,
    tool_calls: list[str] | None = None,
) -> RunTrace:
    """Build a RunTrace from before/after state diff.

    Args:
        initial_state: State before the run.
        final_state: State after the run.
        demo_name: Name of the demo.
        duration_seconds: Wall-clock duration.
        tool_calls: Ordered agent names from tracing dispatcher.

    Returns:
        Populated RunTrace.
    """
    # Extract tool calls from history if not provided
    if tool_calls is None:
        history = final_state.get("history", [])
        tool_calls = []
        for entry in history:
            agent = entry.get("agent", "")
            if agent and "Completed task" in entry.get("action", ""):
                tool_calls.append(agent)

    # Count failures and recoveries from task graph
    task_graph = final_state.get("task_graph", {})
    nodes = task_graph.get("nodes", {})
    failures = sum(1 for n in nodes.values() if n.get("status") == "failed")
    failed_agents = {n.get("agent") for n in nodes.values() if n.get("status") == "failed"}
    completed_agents = {n.get("agent") for n in nodes.values() if n.get("status") == "completed"}
    recoveries = len(failed_agents & completed_agents)

    # Audit log and cost
    audit_log = final_state.get("audit_log", [])
    cost_tokens = sum(entry.get("cost_tokens", 0) for entry in audit_log)
    human_interventions = sum(1 for entry in audit_log if entry.get("human_approved", False))

    return RunTrace(
        demo_name=demo_name,
        task_graph=task_graph,
        ppa_metrics=final_state.get("ppa_metrics", {}),
        iteration_history=final_state.get("optimization_history", []),
        tool_calls=tool_calls,
        audit_log=audit_log,
        failures=failures,
        recoveries=recoveries,
        duration_seconds=duration_seconds,
        cost_tokens=cost_tokens,
        human_interventions=human_interventions,
        design_rationale=final_state.get("design_rationale", []),
        pareto_points=final_state.get("pareto_results", {}).get("front", []),
        history=final_state.get("history", []),
    )
