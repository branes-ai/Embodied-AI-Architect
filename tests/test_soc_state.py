"""Tests for the SoC design state schema and task graph engine.

Tests cover:
- TaskGraph: DAG operations, dependency resolution, cycle detection, serialization
- SoCDesignState: creation, helper functions, constraint handling
"""

import pytest

from embodied_ai_architect.graphs.task_graph import (
    CycleError,
    TaskGraph,
    TaskNode,
    TaskStatus,
)
from embodied_ai_architect.graphs.soc_state import (
    DesignConstraints,
    DesignStatus,
    PPAMetrics,
    create_initial_soc_state,
    get_constraints,
    get_iteration_summary,
    get_optimization_history,
    get_ppa_metrics,
    get_task_graph,
    get_working_memory,
    is_design_complete,
    is_over_iteration_limit,
    record_audit,
    record_decision,
    set_task_graph,
    update_working_memory,
)

# ============================================================================
# TaskNode tests
# ============================================================================


class TestTaskNode:
    def test_defaults(self):
        node = TaskNode(id="t1", name="Test", agent="tester")
        assert node.status == TaskStatus.PENDING
        assert node.dependencies == []
        assert node.result is None
        assert node.error is None
        assert node.is_terminal is False

    def test_terminal_states(self):
        node = TaskNode(id="t1", name="Test", agent="tester")
        assert node.is_terminal is False

        node.status = TaskStatus.COMPLETED
        assert node.is_terminal is True

        node.status = TaskStatus.FAILED
        assert node.is_terminal is True

        node.status = TaskStatus.SKIPPED
        assert node.is_terminal is True

        node.status = TaskStatus.RUNNING
        assert node.is_terminal is False


# ============================================================================
# TaskGraph — basic operations
# ============================================================================


class TestTaskGraphBasic:
    def test_empty_graph(self):
        graph = TaskGraph()
        assert graph.is_complete is True
        assert graph.ready_tasks() == []
        assert graph.summary() == {}

    def test_add_single_task(self):
        graph = TaskGraph()
        task = TaskNode(id="t1", name="Analyze workload", agent="workload_analyzer")
        graph.add_task(task)

        assert "t1" in graph.nodes
        assert graph.nodes["t1"].status == TaskStatus.READY  # no deps -> ready

    def test_add_duplicate_raises(self):
        graph = TaskGraph()
        graph.add_task(TaskNode(id="t1", name="A", agent="a"))
        with pytest.raises(ValueError, match="already exists"):
            graph.add_task(TaskNode(id="t1", name="B", agent="b"))

    def test_add_with_missing_dependency_raises(self):
        graph = TaskGraph()
        with pytest.raises(ValueError, match="does not exist"):
            graph.add_task(TaskNode(id="t2", name="B", agent="b", dependencies=["t1"]))

    def test_ready_tasks_no_dependencies(self):
        graph = TaskGraph()
        graph.add_task(TaskNode(id="t1", name="A", agent="a"))
        graph.add_task(TaskNode(id="t2", name="B", agent="b"))

        ready = graph.ready_tasks()
        assert len(ready) == 2
        ids = {t.id for t in ready}
        assert ids == {"t1", "t2"}

    def test_ready_tasks_with_dependencies(self):
        graph = TaskGraph()
        graph.add_task(TaskNode(id="t1", name="A", agent="a"))
        graph.add_task(TaskNode(id="t2", name="B", agent="b", dependencies=["t1"]))

        ready = graph.ready_tasks()
        assert len(ready) == 1
        assert ready[0].id == "t1"

    def test_get_task(self):
        graph = TaskGraph()
        graph.add_task(TaskNode(id="t1", name="A", agent="a"))
        task = graph.get_task("t1")
        assert task.name == "A"

    def test_get_task_missing_raises(self):
        graph = TaskGraph()
        with pytest.raises(KeyError, match="not found"):
            graph.get_task("nonexistent")


# ============================================================================
# TaskGraph — lifecycle transitions
# ============================================================================


class TestTaskGraphLifecycle:
    @pytest.fixture()
    def linear_graph(self):
        """t1 -> t2 -> t3"""
        graph = TaskGraph()
        graph.add_task(TaskNode(id="t1", name="A", agent="a"))
        graph.add_task(TaskNode(id="t2", name="B", agent="b", dependencies=["t1"]))
        graph.add_task(TaskNode(id="t3", name="C", agent="c", dependencies=["t2"]))
        return graph

    def test_full_execution(self, linear_graph):
        g = linear_graph

        # Only t1 is ready
        ready = g.ready_tasks()
        assert [t.id for t in ready] == ["t1"]

        # Run t1
        g.mark_running("t1")
        assert g.nodes["t1"].status == TaskStatus.RUNNING
        assert g.ready_tasks() == []  # t2 still blocked

        # Complete t1 -> t2 becomes ready
        g.mark_completed("t1", result={"data": "workload_profile"})
        assert g.nodes["t1"].status == TaskStatus.COMPLETED
        ready = g.ready_tasks()
        assert [t.id for t in ready] == ["t2"]

        # Run and complete t2 -> t3 becomes ready
        g.mark_running("t2")
        g.mark_completed("t2", result={"data": "hw_candidates"})
        ready = g.ready_tasks()
        assert [t.id for t in ready] == ["t3"]

        # Run and complete t3
        g.mark_running("t3")
        g.mark_completed("t3", result={"data": "architecture"})
        assert g.is_complete is True

    def test_mark_running_not_ready_raises(self, linear_graph):
        with pytest.raises(ValueError, match="expected 'ready'"):
            linear_graph.mark_running("t2")  # t2 is pending, not ready

    def test_mark_completed_not_running_raises(self, linear_graph):
        with pytest.raises(ValueError, match="expected 'running'"):
            linear_graph.mark_completed("t1", result={})  # t1 is ready, not running

    def test_mark_failed(self, linear_graph):
        g = linear_graph
        g.mark_running("t1")
        g.mark_failed("t1", error="Model file not found")

        assert g.nodes["t1"].status == TaskStatus.FAILED
        assert g.nodes["t1"].error == "Model file not found"
        assert g.has_failures is True

    def test_blocked_tasks_after_failure(self, linear_graph):
        g = linear_graph
        g.mark_running("t1")
        g.mark_failed("t1", error="Oops")

        blocked = g.blocked_tasks
        # t2 is directly blocked (depends on failed t1)
        # t3 depends on t2 which is pending, not failed — so t3 isn't "blocked" per se,
        # just not ready. blocked_tasks only reports tasks with a failed dependency.
        assert len(blocked) == 1
        assert blocked[0].id == "t2"

    def test_reset_failed(self, linear_graph):
        g = linear_graph
        g.mark_running("t1")
        g.mark_failed("t1", error="Oops")

        g.reset_failed("t1")
        assert g.nodes["t1"].status in (TaskStatus.PENDING, TaskStatus.READY)
        assert g.nodes["t1"].error is None

    def test_mark_skipped_unblocks_downstream(self, linear_graph):
        g = linear_graph
        g.mark_skipped("t1", reason="Not needed for this use case")

        assert g.nodes["t1"].status == TaskStatus.SKIPPED
        # t2 should now be ready since t1 is skipped (counts as satisfied)
        ready = g.ready_tasks()
        assert [t.id for t in ready] == ["t2"]

    def test_get_result(self, linear_graph):
        g = linear_graph
        g.mark_running("t1")
        g.mark_completed("t1", result={"profile": "data"})
        assert g.get_result("t1") == {"profile": "data"}

    def test_get_result_not_completed_raises(self, linear_graph):
        with pytest.raises(ValueError, match="has no result"):
            linear_graph.get_result("t1")


# ============================================================================
# TaskGraph — DAG structure
# ============================================================================


class TestTaskGraphDAG:
    def test_diamond_dependency(self):
        """
        t1 -> t2 -+
                   |-> t4
        t1 -> t3 -+
        """
        graph = TaskGraph()
        graph.add_task(TaskNode(id="t1", name="A", agent="a"))
        graph.add_task(TaskNode(id="t2", name="B", agent="b", dependencies=["t1"]))
        graph.add_task(TaskNode(id="t3", name="C", agent="c", dependencies=["t1"]))
        graph.add_task(TaskNode(id="t4", name="D", agent="d", dependencies=["t2", "t3"]))

        # Only t1 is ready
        ready = graph.ready_tasks()
        assert [t.id for t in ready] == ["t1"]

        # Complete t1 -> t2 and t3 both ready
        graph.mark_running("t1")
        graph.mark_completed("t1", result={})
        ready = graph.ready_tasks()
        assert {t.id for t in ready} == {"t2", "t3"}

        # Complete t2 only -> t4 still blocked (needs t3)
        graph.mark_running("t2")
        graph.mark_completed("t2", result={})
        ready = graph.ready_tasks()
        assert [t.id for t in ready] == ["t3"]

        # Complete t3 -> t4 now ready
        graph.mark_running("t3")
        graph.mark_completed("t3", result={})
        ready = graph.ready_tasks()
        assert [t.id for t in ready] == ["t4"]

    def test_cycle_detection_self_loop(self):
        """Self-dependency should be rejected (dependency not found)."""
        graph = TaskGraph()
        # t1 depends on itself — but it doesn't exist yet, so ValueError
        with pytest.raises(ValueError, match="does not exist"):
            graph.add_task(
                TaskNode(id="t1", name="A", agent="a", dependencies=["t1"]),
            )

    def test_cycle_detection_in_batch(self):
        """Mutual dependencies in a batch should raise CycleError."""
        graph = TaskGraph()
        with pytest.raises(CycleError):
            graph.add_tasks(
                [
                    TaskNode(id="t1", name="A", agent="a", dependencies=["t2"]),
                    TaskNode(id="t2", name="B", agent="b", dependencies=["t1"]),
                ]
            )

    def test_execution_order(self):
        graph = TaskGraph()
        graph.add_task(TaskNode(id="t1", name="A", agent="a"))
        graph.add_task(TaskNode(id="t2", name="B", agent="b", dependencies=["t1"]))
        graph.add_task(TaskNode(id="t3", name="C", agent="c", dependencies=["t1"]))
        graph.add_task(TaskNode(id="t4", name="D", agent="d", dependencies=["t2", "t3"]))

        order = graph.execution_order()
        # t1 must come before t2, t3; t2 and t3 must come before t4
        assert order.index("t1") < order.index("t2")
        assert order.index("t1") < order.index("t3")
        assert order.index("t2") < order.index("t4")
        assert order.index("t3") < order.index("t4")

    def test_downstream(self):
        graph = TaskGraph()
        graph.add_task(TaskNode(id="t1", name="A", agent="a"))
        graph.add_task(TaskNode(id="t2", name="B", agent="b", dependencies=["t1"]))
        graph.add_task(TaskNode(id="t3", name="C", agent="c", dependencies=["t2"]))

        downstream = graph.downstream("t1")
        assert set(downstream) == {"t2", "t3"}

        downstream = graph.downstream("t2")
        assert set(downstream) == {"t3"}

        downstream = graph.downstream("t3")
        assert downstream == []

    def test_summary(self):
        graph = TaskGraph()
        graph.add_task(TaskNode(id="t1", name="A", agent="a"))
        graph.add_task(TaskNode(id="t2", name="B", agent="b", dependencies=["t1"]))

        summary = graph.summary()
        assert summary["ready"] == 1  # t1
        assert summary["pending"] == 1  # t2

    def test_remove_task(self):
        graph = TaskGraph()
        graph.add_task(TaskNode(id="t1", name="A", agent="a"))
        graph.add_task(TaskNode(id="t2", name="B", agent="b"))

        removed = graph.remove_task("t2")
        assert removed.id == "t2"
        assert "t2" not in graph.nodes

    def test_remove_task_with_dependents_raises(self):
        graph = TaskGraph()
        graph.add_task(TaskNode(id="t1", name="A", agent="a"))
        graph.add_task(TaskNode(id="t2", name="B", agent="b", dependencies=["t1"]))

        with pytest.raises(ValueError, match="depend on it"):
            graph.remove_task("t1")

    def test_remove_running_task_raises(self):
        graph = TaskGraph()
        graph.add_task(TaskNode(id="t1", name="A", agent="a"))
        graph.mark_running("t1")

        with pytest.raises(ValueError, match="Cannot remove"):
            graph.remove_task("t1")


# ============================================================================
# TaskGraph — batch operations
# ============================================================================


class TestTaskGraphBatch:
    def test_add_tasks_in_order(self):
        graph = TaskGraph()
        tasks = [
            TaskNode(id="t1", name="A", agent="a"),
            TaskNode(id="t2", name="B", agent="b", dependencies=["t1"]),
            TaskNode(id="t3", name="C", agent="c", dependencies=["t2"]),
        ]
        graph.add_tasks(tasks)
        assert set(graph.nodes.keys()) == {"t1", "t2", "t3"}

    def test_add_tasks_out_of_order(self):
        """add_tasks should sort topologically, so order doesn't matter."""
        graph = TaskGraph()
        tasks = [
            TaskNode(id="t3", name="C", agent="c", dependencies=["t2"]),
            TaskNode(id="t1", name="A", agent="a"),
            TaskNode(id="t2", name="B", agent="b", dependencies=["t1"]),
        ]
        graph.add_tasks(tasks)
        assert set(graph.nodes.keys()) == {"t1", "t2", "t3"}

    def test_add_tasks_duplicate_ids_raises(self):
        graph = TaskGraph()
        tasks = [
            TaskNode(id="t1", name="A", agent="a"),
            TaskNode(id="t1", name="B", agent="b"),
        ]
        with pytest.raises(ValueError, match="Duplicate"):
            graph.add_tasks(tasks)

    def test_add_tasks_cycle_raises(self):
        graph = TaskGraph()
        tasks = [
            TaskNode(id="t1", name="A", agent="a", dependencies=["t2"]),
            TaskNode(id="t2", name="B", agent="b", dependencies=["t1"]),
        ]
        with pytest.raises(CycleError):
            graph.add_tasks(tasks)


# ============================================================================
# TaskGraph — serialization
# ============================================================================


class TestTaskGraphSerialization:
    def test_round_trip(self):
        graph = TaskGraph()
        graph.add_task(TaskNode(id="t1", name="A", agent="a"))
        graph.add_task(TaskNode(id="t2", name="B", agent="b", dependencies=["t1"]))

        data = graph.to_dict()
        restored = TaskGraph.from_dict(data)

        assert set(restored.nodes.keys()) == {"t1", "t2"}
        assert restored.nodes["t2"].dependencies == ["t1"]
        assert restored.nodes["t1"].status == TaskStatus.READY

    def test_round_trip_with_results(self):
        graph = TaskGraph()
        graph.add_task(TaskNode(id="t1", name="A", agent="a"))
        graph.mark_running("t1")
        graph.mark_completed("t1", result={"key": "value"})

        data = graph.to_dict()
        restored = TaskGraph.from_dict(data)

        assert restored.nodes["t1"].status == TaskStatus.COMPLETED
        assert restored.nodes["t1"].result == {"key": "value"}


# ============================================================================
# DesignConstraints
# ============================================================================


class TestDesignConstraints:
    def test_empty_constraints(self):
        c = DesignConstraints()
        assert c.max_power_watts is None
        assert c.safety_critical is False

    def test_drone_constraints(self):
        c = DesignConstraints(
            max_power_watts=5.0,
            max_latency_ms=33.3,
            max_cost_usd=30.0,
            target_volume=100_000,
        )
        assert c.max_power_watts == 5.0
        assert c.max_latency_ms == 33.3
        assert c.max_cost_usd == 30.0
        assert c.target_volume == 100_000

    def test_safety_critical(self):
        c = DesignConstraints(
            safety_critical=True,
            safety_standard="IEC 62304 Class C",
        )
        assert c.safety_critical is True
        assert c.safety_standard == "IEC 62304 Class C"

    def test_serialization(self):
        c = DesignConstraints(max_power_watts=5.0, max_latency_ms=33.3)
        data = c.model_dump()
        restored = DesignConstraints(**data)
        assert restored.max_power_watts == 5.0
        assert restored.max_latency_ms == 33.3


# ============================================================================
# PPAMetrics
# ============================================================================


class TestPPAMetrics:
    def test_empty(self):
        m = PPAMetrics()
        assert m.power_watts is None
        assert m.verdicts == {}
        assert m.bottlenecks == []

    def test_with_verdicts(self):
        m = PPAMetrics(
            power_watts=4.9,
            latency_ms=32.0,
            verdicts={"power": "PASS", "latency": "PASS"},
            bottlenecks=[],
        )
        assert m.verdicts["power"] == "PASS"


# ============================================================================
# SoCDesignState creation and helpers
# ============================================================================


class TestSoCDesignState:
    def test_create_minimal(self):
        state = create_initial_soc_state(goal="Design an SoC for a drone")
        assert state["goal"] == "Design an SoC for a drone"
        assert state["status"] == "planning"
        assert state["iteration"] == 0
        assert state["session_id"].startswith("soc_")
        assert state["next_action"] == "planner"

    def test_create_with_constraints(self):
        constraints = DesignConstraints(
            max_power_watts=5.0,
            max_latency_ms=33.3,
            max_cost_usd=30.0,
        )
        state = create_initial_soc_state(
            goal="Delivery drone SoC",
            constraints=constraints,
            use_case="delivery_drone",
            platform="drone",
        )
        assert state["use_case"] == "delivery_drone"
        assert state["platform"] == "drone"
        assert state["constraints"]["max_power_watts"] == 5.0

    def test_get_task_graph(self):
        state = create_initial_soc_state(goal="Test")
        graph = get_task_graph(state)
        assert isinstance(graph, TaskGraph)
        assert len(graph.nodes) == 0

    def test_set_task_graph(self):
        state = create_initial_soc_state(goal="Test")
        graph = get_task_graph(state)
        graph.add_task(TaskNode(id="t1", name="A", agent="a"))

        updated = set_task_graph(state, graph)
        restored = get_task_graph(updated)
        assert "t1" in restored.nodes

    def test_get_constraints(self):
        constraints = DesignConstraints(max_power_watts=5.0)
        state = create_initial_soc_state(goal="Test", constraints=constraints)
        restored = get_constraints(state)
        assert restored.max_power_watts == 5.0

    def test_get_ppa_metrics(self):
        state = create_initial_soc_state(goal="Test")
        ppa = get_ppa_metrics(state)
        assert ppa.power_watts is None  # empty initially

    def test_record_decision(self):
        state = create_initial_soc_state(goal="Test")
        updated = record_decision(
            state,
            agent="workload_analyzer",
            action="Selected YOLOv8n as detection model",
            rationale="Best power/accuracy trade-off for <5W budget",
            alternatives=["YOLOv8s", "MobileNetV2-SSD"],
        )
        assert len(updated["history"]) == 1
        assert updated["history"][0]["agent"] == "workload_analyzer"
        assert len(updated["design_rationale"]) == 1
        assert "YOLOv8n" in updated["design_rationale"][0]

    def test_is_design_complete(self):
        state = create_initial_soc_state(goal="Test")
        assert is_design_complete(state) is False

        state["status"] = DesignStatus.COMPLETE.value
        assert is_design_complete(state) is True

        state["status"] = DesignStatus.FAILED.value
        assert is_design_complete(state) is True

    def test_is_over_iteration_limit(self):
        state = create_initial_soc_state(goal="Test", max_iterations=5)
        assert is_over_iteration_limit(state) is False

        state["iteration"] = 5
        assert is_over_iteration_limit(state) is True

    def test_get_iteration_summary(self):
        state = create_initial_soc_state(
            goal="Design an SoC for a drone",
            use_case="delivery_drone",
        )
        summary = get_iteration_summary(state)
        assert "Design an SoC for a drone" in summary
        assert "planning" in summary
        assert "0/20" in summary

    def test_custom_session_id(self):
        state = create_initial_soc_state(goal="Test", session_id="my_session_42")
        assert state["session_id"] == "my_session_42"


# ============================================================================
# Integration: TaskGraph + SoCDesignState together
# ============================================================================


class TestIntegration:
    def test_planner_workflow(self):
        """Simulate the planner producing a task graph and storing it in state."""
        state = create_initial_soc_state(
            goal="Design an SoC for a delivery drone: <5W, <50ms, <$30",
            constraints=DesignConstraints(
                max_power_watts=5.0,
                max_latency_ms=50.0,
                max_cost_usd=30.0,
            ),
            use_case="delivery_drone",
            platform="drone",
        )

        # Planner creates task graph
        graph = get_task_graph(state)
        graph.add_tasks(
            [
                TaskNode(
                    id="t1",
                    name="Analyze perception workload",
                    agent="workload_analyzer",
                    postconditions=["workload_profile populated"],
                ),
                TaskNode(
                    id="t2",
                    name="Analyze environmental constraints",
                    agent="workload_analyzer",
                    postconditions=["thermal and IP constraints identified"],
                ),
                TaskNode(
                    id="t3",
                    name="Enumerate feasible hardware",
                    agent="hw_explorer",
                    dependencies=["t1", "t2"],
                    preconditions=["workload_profile available", "env constraints available"],
                ),
                TaskNode(
                    id="t4",
                    name="PPA assessment",
                    agent="ppa_assessor",
                    dependencies=["t3"],
                ),
                TaskNode(
                    id="t5",
                    name="Generate design report",
                    agent="report_generator",
                    dependencies=["t4"],
                ),
            ]
        )

        state = set_task_graph(state, graph)
        state = record_decision(
            state,
            agent="planner",
            action="Created initial task graph with 5 tasks",
            rationale="Standard SoC design decomposition for drone use case",
        )

        # Verify state
        restored_graph = get_task_graph(state)
        assert len(restored_graph.nodes) == 5

        ready = restored_graph.ready_tasks()
        assert {t.id for t in ready} == {"t1", "t2"}  # parallel start

        order = restored_graph.execution_order()
        assert order.index("t1") < order.index("t3")
        assert order.index("t2") < order.index("t3")
        assert order.index("t3") < order.index("t4")
        assert order.index("t4") < order.index("t5")

        # Verify decision was recorded
        assert len(state["history"]) == 1
        assert len(state["design_rationale"]) == 1


# ============================================================================
# Phase 2 fields: working_memory, optimization_history, governance, audit_log
# ============================================================================


class TestPhase2StateFields:
    def test_new_fields_initialized_empty(self):
        state = create_initial_soc_state(goal="Test")
        assert state["working_memory"] == {}
        assert state["optimization_history"] == []
        assert state["governance"] == {}
        assert state["audit_log"] == []

    def test_governance_param(self):
        gov = {"iteration_limit": 5, "cost_budget_tokens": 10000}
        state = create_initial_soc_state(goal="Test", governance=gov)
        assert state["governance"]["iteration_limit"] == 5

    def test_get_working_memory(self):
        state = create_initial_soc_state(goal="Test")
        mem = get_working_memory(state)
        assert mem == {}

    def test_update_working_memory(self):
        state = create_initial_soc_state(goal="Test")
        mem = {"agents": {"optimizer": {"decisions_made": ["chose INT8"]}}}
        updated = update_working_memory(state, mem)
        assert updated["working_memory"]["agents"]["optimizer"]["decisions_made"] == ["chose INT8"]

    def test_record_audit(self):
        state = create_initial_soc_state(goal="Test")
        updated = record_audit(
            state,
            agent="ppa_assessor",
            action="Assessed PPA metrics",
            input_summary="architecture with KPU",
            output_summary="power=6.3W FAIL",
        )
        assert len(updated["audit_log"]) == 1
        entry = updated["audit_log"][0]
        assert entry["agent"] == "ppa_assessor"
        assert entry["action"] == "Assessed PPA metrics"
        assert entry["iteration"] == 0
        assert entry["cost_tokens"] == 0
        assert entry["human_approved"] is False

    def test_record_audit_multiple(self):
        state = create_initial_soc_state(goal="Test")
        state = record_audit(state, agent="a", action="first")
        state = record_audit(state, agent="b", action="second")
        assert len(state["audit_log"]) == 2

    def test_get_optimization_history(self):
        state = create_initial_soc_state(goal="Test")
        assert get_optimization_history(state) == []

    def test_optimization_history_append(self):
        state = create_initial_soc_state(goal="Test")
        snap = {"iteration": 0, "power_watts": 6.3, "verdict": "FAIL"}
        state["optimization_history"] = [snap]
        history = get_optimization_history(state)
        assert len(history) == 1
        assert history[0]["power_watts"] == 6.3
