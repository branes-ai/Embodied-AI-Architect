"""Dispatcher for agentic SoC design task execution.

The Dispatcher walks the TaskGraph, finds ready tasks, dispatches them to
registered specialist agents, and records results back into the SoCDesignState.

It replaces the hardcoded sequential Orchestrator with a DAG-aware scheduler
that supports parallel-ready tasks, failure handling, and re-planning.

Usage:
    from embodied_ai_architect.graphs.dispatcher import Dispatcher

    dispatcher = Dispatcher()
    dispatcher.register("workload_analyzer", my_analyzer_fn)
    dispatcher.register("hw_explorer", my_explorer_fn)

    # Single step (dispatch one batch of ready tasks)
    state = dispatcher.step(state)

    # Full run (loop until complete or stuck)
    state = dispatcher.run(state)
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from embodied_ai_architect.graphs.memory import WorkingMemoryStore
from embodied_ai_architect.graphs.soc_state import (
    DesignStatus,
    SoCDesignState,
    get_task_graph,
    record_decision,
    set_task_graph,
)
from embodied_ai_architect.graphs.task_graph import TaskGraph, TaskNode, TaskStatus

logger = logging.getLogger(__name__)


# Type for agent executor functions.
# Signature: (task: TaskNode, state: SoCDesignState) -> dict[str, Any]
# Returns a result dict that gets stored as task.result.
AgentExecutor = Callable[[TaskNode, SoCDesignState], dict[str, Any]]


class DispatchError(Exception):
    """Raised when the dispatcher encounters an unrecoverable problem."""


class Dispatcher:
    """DAG-aware task dispatcher for SoC design.

    Maintains a registry of agent executors and dispatches tasks from the
    TaskGraph in dependency order. Each step dispatches all currently-ready
    tasks (potential parallelism), records results, and updates the graph.
    """

    def __init__(self) -> None:
        self._agents: dict[str, AgentExecutor] = {}

    def register(self, agent_name: str, executor: AgentExecutor) -> None:
        """Register an agent executor.

        Args:
            agent_name: Name matching TaskNode.agent values.
            executor: Callable(task, state) -> result_dict.
        """
        self._agents[agent_name] = executor

    def register_many(self, agents: dict[str, AgentExecutor]) -> None:
        """Register multiple agent executors at once."""
        self._agents.update(agents)

    @property
    def registered_agents(self) -> list[str]:
        """List of registered agent names."""
        return list(self._agents.keys())

    def step(self, state: SoCDesignState) -> SoCDesignState:
        """Execute one batch of ready tasks.

        Finds all tasks that are ready (dependencies satisfied), dispatches
        each to its agent, and records the results. Returns updated state.

        If no tasks are ready and the graph is not complete, the design is
        stuck and needs re-planning.

        Args:
            state: Current SoC design state.

        Returns:
            Updated state with task results and graph changes.
        """
        graph = get_task_graph(state)
        ready = graph.ready_tasks()

        if not ready:
            if graph.is_complete:
                state = {**state, "status": DesignStatus.COMPLETE.value}
                return state
            elif graph.has_failures:
                state = {**state, "status": DesignStatus.FAILED.value}
                logger.warning("Dispatch stuck: tasks have failures and no tasks are ready")
                return state
            else:
                # Deadlock — shouldn't happen with valid DAG
                logger.error("Dispatch deadlock: no ready tasks but graph is not complete")
                state = {**state, "status": DesignStatus.FAILED.value}
                return state

        for task in ready:
            state = self._dispatch_task(task, graph, state)

        state = set_task_graph(state, graph)
        return state

    def run(self, state: SoCDesignState, max_steps: int = 50) -> SoCDesignState:
        """Execute tasks until the graph is complete or dispatch is stuck.

        Loops calling step() until:
        - All tasks are terminal (complete/failed/skipped)
        - No tasks are ready and graph is not complete (stuck)
        - max_steps is exceeded (safety bound)

        Args:
            state: Initial SoC design state with populated task_graph.
            max_steps: Maximum dispatch steps before forced stop.

        Returns:
            Final state after all tasks are dispatched.
        """
        for step_num in range(max_steps):
            graph = get_task_graph(state)

            if graph.is_complete:
                state = {**state, "status": DesignStatus.COMPLETE.value}  # type: ignore[assignment]
                logger.info("Dispatch complete: all tasks finished in %d steps", step_num)
                return state

            ready = graph.ready_tasks()
            if not ready:
                if graph.has_failures:
                    logger.warning(
                        "Dispatch stopped: failed tasks blocking progress at step %d",
                        step_num,
                    )
                    state = {**state, "status": DesignStatus.FAILED.value}  # type: ignore[assignment]
                else:
                    logger.error("Dispatch deadlock at step %d", step_num)
                    state = {**state, "status": DesignStatus.FAILED.value}  # type: ignore[assignment]
                return state

            state = self.step(state)

        logger.warning("Dispatch hit max_steps limit (%d)", max_steps)
        return state

    def _dispatch_task(
        self, task: TaskNode, graph: TaskGraph, state: SoCDesignState
    ) -> SoCDesignState:
        """Dispatch a single task to its agent and record the outcome."""
        executor = self._agents.get(task.agent)
        if executor is None:
            error_msg = (
                f"No executor registered for agent '{task.agent}'. "
                f"Registered: {self.registered_agents}"
            )
            logger.error(error_msg)
            graph.mark_running(task.id)
            graph.mark_failed(task.id, error=error_msg)
            return state

        graph.mark_running(task.id)
        state = {**state, "current_task_id": task.id}  # type: ignore[assignment]

        logger.info("Dispatching task '%s' (%s) to agent '%s'", task.id, task.name, task.agent)

        try:
            result = executor(task, state)
            graph.mark_completed(task.id, result=result)
            logger.info("Task '%s' completed successfully", task.id)

            # Merge state updates from task result if present
            state_updates = result.pop("_state_updates", None)
            if state_updates and isinstance(state_updates, dict):
                state = {**state, **state_updates}  # type: ignore[assignment]
                logger.debug(
                    "Merged state updates from task '%s': %s",
                    task.id,
                    list(state_updates.keys()),
                )

            # Record attempt in working memory (backward compatible: only if
            # working_memory key exists in state)
            if "working_memory" in state:
                wm_data = state.get("working_memory", {})
                store = WorkingMemoryStore(**wm_data) if wm_data else WorkingMemoryStore()
                store.record_attempt(
                    agent_name=task.agent,
                    description=f"task:{task.id}:{task.name}",
                    outcome=result.get("summary", "completed"),
                    iteration=state.get("iteration", 0),
                )
                state = {**state, "working_memory": store.model_dump()}  # type: ignore[assignment]

            state = record_decision(
                state,
                agent=task.agent,
                action=f"Completed task '{task.name}'",
                rationale=result.get("summary", "Task completed"),
                data={"task_id": task.id},
            )

        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            graph.mark_failed(task.id, error=error_msg)
            logger.error("Task '%s' failed: %s", task.id, error_msg)

            # Record failure in working memory
            if "working_memory" in state:
                wm_data = state.get("working_memory", {})
                store = WorkingMemoryStore(**wm_data) if wm_data else WorkingMemoryStore()
                store.record_attempt(
                    agent_name=task.agent,
                    description=f"task:{task.id}:{task.name}",
                    outcome=f"FAILED: {error_msg}",
                    iteration=state.get("iteration", 0),
                )
                state = {**state, "working_memory": store.model_dump()}  # type: ignore[assignment]

            state = record_decision(
                state,
                agent=task.agent,
                action=f"Failed task '{task.name}'",
                rationale=error_msg,
                data={"task_id": task.id, "error": error_msg},
            )

        return state

    def get_dispatch_summary(self, state: SoCDesignState) -> str:
        """Format a summary of the current dispatch state."""
        graph = get_task_graph(state)
        lines = [f"Status: {state.get('status', 'unknown')}"]

        summary = graph.summary()
        if summary:
            lines.append(f"Tasks: {summary}")

        ready = graph.ready_tasks()
        if ready:
            lines.append(f"Ready: {[t.id + ':' + t.agent for t in ready]}")

        blocked = graph.blocked_tasks
        if blocked:
            lines.append(f"Blocked: {[t.id for t in blocked]}")

        failed = [t for t in graph.nodes.values() if t.status == TaskStatus.FAILED]
        if failed:
            lines.append(f"Failed: {[t.id + ':' + (t.error or '?') for t in failed]}")

        return "\n".join(lines)
