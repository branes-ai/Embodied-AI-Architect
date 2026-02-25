"""Task graph engine for agentic SoC design.

Provides a DAG-based task graph with dependency tracking, status management,
and dynamic modification. The TaskGraph is the execution backbone of the
agentic system — the Planner produces it, and the Dispatcher walks it.

Usage:
    graph = TaskGraph()
    graph.add_task(TaskNode(id="t1", name="Analyze workload", agent="workload_analyzer"))
    graph.add_task(TaskNode(
        id="t2", name="Explore hardware", agent="hw_explorer",
        dependencies=["t1"],
    ))

    ready = graph.ready_tasks()  # ["t1"] — t2 is blocked
    graph.mark_running("t1")
    graph.mark_completed("t1", result={"profile": {...}})
    ready = graph.ready_tasks()  # ["t2"] — t1 is done, t2 unblocked
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator


class TaskStatus(str, Enum):
    """Lifecycle status of a task node."""

    PENDING = "pending"  # Not yet ready (dependencies incomplete)
    READY = "ready"  # All dependencies met, waiting for dispatch
    RUNNING = "running"  # Currently being executed by an agent
    COMPLETED = "completed"  # Finished successfully
    FAILED = "failed"  # Finished with error
    SKIPPED = "skipped"  # Bypassed (e.g., planner decided not needed)


class TaskNode(BaseModel):
    """A single task in the design DAG.

    Each task is owned by a specialist agent and produces a structured result.
    Pre/postconditions are natural-language descriptions used by the Planner
    and Critic to reason about task readiness and completion quality.
    """

    id: str = Field(..., description="Unique task identifier (e.g. 't1', 'analyze_workload')")
    name: str = Field(..., description="Human-readable task description")
    agent: str = Field(
        ...,
        description="Specialist agent responsible (e.g. 'workload_analyzer', 'hw_explorer')",
    )
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="Current lifecycle status")
    dependencies: list[str] = Field(
        default_factory=list,
        description="Task IDs that must complete before this task can run",
    )
    result: Optional[dict[str, Any]] = Field(
        default=None, description="Structured output when completed"
    )
    error: Optional[str] = Field(default=None, description="Error message if failed")
    preconditions: list[str] = Field(
        default_factory=list,
        description="What must be true before execution (natural language)",
    )
    postconditions: list[str] = Field(
        default_factory=list,
        description="What must be true after execution (natural language)",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary metadata (priority, estimated cost, etc.)",
    )

    @property
    def is_terminal(self) -> bool:
        """True if the task has reached a final state."""
        return self.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.SKIPPED)


class CycleError(Exception):
    """Raised when adding a task would create a cycle in the DAG."""


class TaskGraph(BaseModel):
    """DAG of tasks with dependency tracking and status management.

    The graph enforces:
    - No cycles (validated on add_task)
    - No dangling dependencies (all referenced task IDs must exist)
    - Correct status transitions (ready_tasks only returns truly unblocked tasks)
    """

    nodes: dict[str, TaskNode] = Field(default_factory=dict, description="Task ID -> TaskNode")

    def add_task(self, task: TaskNode) -> None:
        """Add a task to the graph.

        Validates that:
        - Task ID is unique
        - All dependencies reference existing tasks
        - Adding this task does not create a cycle

        Raises:
            ValueError: If task ID already exists or dependencies are invalid.
            CycleError: If adding this task would create a cycle.
        """
        if task.id in self.nodes:
            raise ValueError(f"Task '{task.id}' already exists in the graph")

        for dep_id in task.dependencies:
            if dep_id not in self.nodes:
                raise ValueError(
                    f"Task '{task.id}' depends on '{dep_id}' which does not exist. "
                    f"Add '{dep_id}' first, or add both in dependency order."
                )

        # Temporarily add to check for cycles
        self.nodes[task.id] = task
        if self._has_cycle():
            del self.nodes[task.id]
            raise CycleError(
                f"Adding task '{task.id}' would create a cycle in the dependency graph"
            )

        # Update status based on current dependency state
        self._update_task_readiness(task.id)

    def add_tasks(self, tasks: list[TaskNode]) -> None:
        """Add multiple tasks in dependency order.

        Tasks are sorted topologically before insertion so callers
        don't need to worry about ordering.

        Raises:
            ValueError: If any task has an invalid dependency.
            CycleError: If the tasks would create a cycle.
        """
        # Build a local lookup for the batch
        batch = {t.id: t for t in tasks}

        # Check for duplicate IDs within the batch
        if len(batch) != len(tasks):
            ids = [t.id for t in tasks]
            dupes = [x for x in ids if ids.count(x) > 1]
            raise ValueError(f"Duplicate task IDs in batch: {set(dupes)}")

        # Topological sort within the batch
        sorted_tasks = self._topo_sort_batch(tasks, batch)

        for task in sorted_tasks:
            self.add_task(task)

    def remove_task(self, task_id: str) -> TaskNode:
        """Remove a task from the graph.

        Only pending/ready tasks can be removed. Running/completed tasks
        are part of the execution history and should not be removed.

        Raises:
            KeyError: If task does not exist.
            ValueError: If task is running or completed, or if other tasks depend on it.
        """
        task = self._get_task(task_id)

        if task.status in (TaskStatus.RUNNING, TaskStatus.COMPLETED):
            raise ValueError(
                f"Cannot remove task '{task_id}' with status '{task.status.value}'. "
                f"Only pending/ready/failed/skipped tasks can be removed."
            )

        # Check no other task depends on this one
        dependents = [
            t.id for t in self.nodes.values() if task_id in t.dependencies and t.id != task_id
        ]
        if dependents:
            raise ValueError(f"Cannot remove task '{task_id}': tasks {dependents} depend on it")

        return self.nodes.pop(task_id)

    def ready_tasks(self) -> list[TaskNode]:
        """Return tasks whose dependencies are all satisfied and that are ready to run.

        A task is ready when:
        - Its status is PENDING or READY
        - All of its dependencies are COMPLETED or SKIPPED
        """
        ready = []
        for task in self.nodes.values():
            if task.status not in (TaskStatus.PENDING, TaskStatus.READY):
                continue
            if self._dependencies_satisfied(task.id):
                task.status = TaskStatus.READY
                ready.append(task)
        return ready

    def mark_running(self, task_id: str) -> None:
        """Mark a task as currently executing.

        Raises:
            KeyError: If task does not exist.
            ValueError: If task is not in READY state.
        """
        task = self._get_task(task_id)
        if task.status != TaskStatus.READY:
            raise ValueError(
                f"Cannot start task '{task_id}': status is '{task.status.value}', "
                f"expected 'ready'"
            )
        task.status = TaskStatus.RUNNING

    def mark_completed(self, task_id: str, result: dict[str, Any]) -> None:
        """Mark a task as successfully completed with its result.

        After completion, downstream tasks may become ready.

        Raises:
            KeyError: If task does not exist.
            ValueError: If task is not in RUNNING state.
        """
        task = self._get_task(task_id)
        if task.status != TaskStatus.RUNNING:
            raise ValueError(
                f"Cannot complete task '{task_id}': status is '{task.status.value}', "
                f"expected 'running'"
            )
        task.status = TaskStatus.COMPLETED
        task.result = result
        self._propagate_readiness(task_id)

    def mark_failed(self, task_id: str, error: str) -> None:
        """Mark a task as failed with an error message.

        Raises:
            KeyError: If task does not exist.
            ValueError: If task is not in RUNNING state.
        """
        task = self._get_task(task_id)
        if task.status != TaskStatus.RUNNING:
            raise ValueError(
                f"Cannot fail task '{task_id}': status is '{task.status.value}', "
                f"expected 'running'"
            )
        task.status = TaskStatus.FAILED
        task.error = error

    def mark_skipped(self, task_id: str, reason: str = "") -> None:
        """Mark a task as skipped (planner decided it's not needed).

        Skipped tasks count as satisfied for dependency resolution,
        so downstream tasks can still proceed.

        Raises:
            KeyError: If task does not exist.
            ValueError: If task is already completed or running.
        """
        task = self._get_task(task_id)
        if task.status in (TaskStatus.RUNNING, TaskStatus.COMPLETED):
            raise ValueError(f"Cannot skip task '{task_id}': status is '{task.status.value}'")
        task.status = TaskStatus.SKIPPED
        task.metadata["skip_reason"] = reason
        self._propagate_readiness(task_id)

    def reset_failed(self, task_id: str) -> None:
        """Reset a failed task back to pending for retry.

        Raises:
            KeyError: If task does not exist.
            ValueError: If task is not in FAILED state.
        """
        task = self._get_task(task_id)
        if task.status != TaskStatus.FAILED:
            raise ValueError(
                f"Cannot reset task '{task_id}': status is '{task.status.value}', "
                f"expected 'failed'"
            )
        task.status = TaskStatus.PENDING
        task.error = None
        task.result = None
        self._update_task_readiness(task_id)

    def get_task(self, task_id: str) -> TaskNode:
        """Get a task by ID.

        Raises:
            KeyError: If task does not exist.
        """
        return self._get_task(task_id)

    def get_result(self, task_id: str) -> dict[str, Any]:
        """Get the result of a completed task.

        Raises:
            KeyError: If task does not exist.
            ValueError: If task is not completed or has no result.
        """
        task = self._get_task(task_id)
        if task.status != TaskStatus.COMPLETED or task.result is None:
            raise ValueError(f"Task '{task_id}' has no result (status: {task.status.value})")
        return task.result

    @property
    def is_complete(self) -> bool:
        """True if all tasks are in a terminal state (completed, failed, or skipped)."""
        if not self.nodes:
            return True
        return all(task.is_terminal for task in self.nodes.values())

    @property
    def has_failures(self) -> bool:
        """True if any task has failed."""
        return any(task.status == TaskStatus.FAILED for task in self.nodes.values())

    @property
    def blocked_tasks(self) -> list[TaskNode]:
        """Tasks that cannot run because a dependency has failed."""
        blocked = []
        for task in self.nodes.values():
            if task.is_terminal:
                continue
            for dep_id in task.dependencies:
                dep = self.nodes.get(dep_id)
                if dep and dep.status == TaskStatus.FAILED:
                    blocked.append(task)
                    break
        return blocked

    def summary(self) -> dict[str, int]:
        """Return a count of tasks by status."""
        counts: dict[str, int] = {}
        for task in self.nodes.values():
            counts[task.status.value] = counts.get(task.status.value, 0) + 1
        return counts

    def execution_order(self) -> list[str]:
        """Return a valid topological execution order of all task IDs.

        Raises:
            CycleError: If the graph contains a cycle (should not happen
                        if tasks were added via add_task).
        """
        visited: set[str] = set()
        order: list[str] = []
        visiting: set[str] = set()

        def dfs(node_id: str) -> None:
            if node_id in visited:
                return
            if node_id in visiting:
                raise CycleError(f"Cycle detected involving task '{node_id}'")
            visiting.add(node_id)
            for dep_id in self.nodes[node_id].dependencies:
                if dep_id in self.nodes:
                    dfs(dep_id)
            visiting.remove(node_id)
            visited.add(node_id)
            order.append(node_id)

        for task_id in self.nodes:
            dfs(task_id)

        return order

    def downstream(self, task_id: str) -> list[str]:
        """Return all task IDs that transitively depend on the given task."""
        self._get_task(task_id)  # validate exists
        result: list[str] = []
        visited: set[str] = set()

        def collect(tid: str) -> None:
            for node in self.nodes.values():
                if tid in node.dependencies and node.id not in visited:
                    visited.add(node.id)
                    result.append(node.id)
                    collect(node.id)

        collect(task_id)
        return result

    def to_dict(self) -> dict[str, Any]:
        """Serialize the graph for checkpointing / state persistence."""
        return {
            "nodes": {tid: task.model_dump() for tid, task in self.nodes.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskGraph:
        """Restore a graph from serialized form."""
        graph = cls()
        graph.nodes = {tid: TaskNode(**node_data) for tid, node_data in data["nodes"].items()}
        return graph

    # --- Private helpers ---

    def _get_task(self, task_id: str) -> TaskNode:
        """Get a task or raise KeyError."""
        if task_id not in self.nodes:
            raise KeyError(f"Task '{task_id}' not found in graph")
        return self.nodes[task_id]

    def _dependencies_satisfied(self, task_id: str) -> bool:
        """Check if all dependencies of a task are in a terminal-success state."""
        task = self.nodes[task_id]
        for dep_id in task.dependencies:
            dep = self.nodes.get(dep_id)
            if dep is None:
                return False
            if dep.status not in (TaskStatus.COMPLETED, TaskStatus.SKIPPED):
                return False
        return True

    def _update_task_readiness(self, task_id: str) -> None:
        """Update a single task's status to READY if its dependencies are satisfied."""
        task = self.nodes[task_id]
        if task.status == TaskStatus.PENDING and self._dependencies_satisfied(task_id):
            task.status = TaskStatus.READY

    def _propagate_readiness(self, completed_task_id: str) -> None:
        """After a task completes/skips, check if downstream tasks become ready."""
        for task in self.nodes.values():
            if completed_task_id in task.dependencies:
                self._update_task_readiness(task.id)

    def _has_cycle(self) -> bool:
        """Detect cycles using DFS."""
        visited: set[str] = set()
        visiting: set[str] = set()

        def dfs(node_id: str) -> bool:
            if node_id in visiting:
                return True
            if node_id in visited:
                return False
            visiting.add(node_id)
            for dep_id in self.nodes[node_id].dependencies:
                if dep_id in self.nodes and dfs(dep_id):
                    return True
            visiting.remove(node_id)
            visited.add(node_id)
            return False

        for task_id in self.nodes:
            if dfs(task_id):
                return True
        return False

    def _topo_sort_batch(self, tasks: list[TaskNode], batch: dict[str, TaskNode]) -> list[TaskNode]:
        """Topological sort of a batch of tasks (for add_tasks).

        Tasks that depend on already-existing graph nodes sort first.
        """
        visited: set[str] = set()
        order: list[str] = []
        visiting: set[str] = set()

        def dfs(task_id: str) -> None:
            if task_id not in batch or task_id in visited:
                return
            if task_id in visiting:
                raise CycleError(f"Cycle detected in batch involving task '{task_id}'")
            visiting.add(task_id)
            for dep_id in batch[task_id].dependencies:
                if dep_id in batch:
                    dfs(dep_id)
            visiting.remove(task_id)
            visited.add(task_id)
            order.append(task_id)

        for t in tasks:
            dfs(t.id)

        return [batch[tid] for tid in order]
