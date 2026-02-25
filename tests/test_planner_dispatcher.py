"""Tests for the Planner agent and Dispatcher.

Tests cover:
- Planner: plan parsing, task graph construction, LLM integration (mocked)
- Dispatcher: dispatch loop, failure handling, completion detection
- Integration: planner -> dispatcher end-to-end with mock agents
"""

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from embodied_ai_architect.graphs.dispatcher import Dispatcher
from embodied_ai_architect.graphs.planner import (
    PlannerNode,
    parse_plan_json,
    tasks_to_graph,
)
from embodied_ai_architect.graphs.memory import WorkingMemoryStore
from embodied_ai_architect.graphs.soc_state import (
    DesignConstraints,
    DesignStatus,
    SoCDesignState,
    create_initial_soc_state,
    get_task_graph,
)
from embodied_ai_architect.graphs.task_graph import TaskNode, TaskStatus

# ============================================================================
# Mock LLM
# ============================================================================


@dataclass
class MockLLMResponse:
    text: str = ""
    tool_calls: list = field(default_factory=list)
    stop_reason: str = "end_turn"

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


class MockLLMClient:
    """Mock LLM that returns a canned plan response."""

    def __init__(self, response_text: str):
        self.response_text = response_text
        self.calls: list[dict] = []

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
    ) -> MockLLMResponse:
        self.calls.append({"messages": messages, "tools": tools, "system": system})
        return MockLLMResponse(text=self.response_text)


# ============================================================================
# Fixtures
# ============================================================================

SIMPLE_PLAN_JSON = json.dumps(
    {
        "tasks": [
            {
                "id": "t1",
                "name": "Analyze workload",
                "agent": "workload_analyzer",
                "dependencies": [],
            },
            {
                "id": "t2",
                "name": "Explore hardware",
                "agent": "hw_explorer",
                "dependencies": ["t1"],
            },
            {
                "id": "t3",
                "name": "Assess PPA",
                "agent": "ppa_assessor",
                "dependencies": ["t2"],
            },
            {
                "id": "t4",
                "name": "Generate report",
                "agent": "report_generator",
                "dependencies": ["t3"],
            },
        ]
    }
)

DIAMOND_PLAN_JSON = json.dumps(
    {
        "tasks": [
            {
                "id": "t1",
                "name": "Analyze perception workload",
                "agent": "workload_analyzer",
                "dependencies": [],
            },
            {
                "id": "t2",
                "name": "Analyze environmental constraints",
                "agent": "workload_analyzer",
                "dependencies": [],
            },
            {
                "id": "t3",
                "name": "Enumerate hardware",
                "agent": "hw_explorer",
                "dependencies": ["t1", "t2"],
            },
            {
                "id": "t4",
                "name": "Compose architecture",
                "agent": "architecture_composer",
                "dependencies": ["t3"],
            },
            {
                "id": "t5",
                "name": "PPA assessment",
                "agent": "ppa_assessor",
                "dependencies": ["t4"],
            },
            {
                "id": "t6",
                "name": "Generate report",
                "agent": "report_generator",
                "dependencies": ["t5"],
            },
        ]
    }
)


@pytest.fixture()
def drone_state():
    return create_initial_soc_state(
        goal="Design an SoC for a delivery drone: <5W, <50ms, <$30",
        constraints=DesignConstraints(
            max_power_watts=5.0,
            max_latency_ms=50.0,
            max_cost_usd=30.0,
        ),
        use_case="delivery_drone",
        platform="drone",
    )


def make_mock_executor(result: dict[str, Any] | None = None, fail: bool = False):
    """Create a mock agent executor that returns a canned result or raises."""

    def executor(task: TaskNode, state: SoCDesignState) -> dict[str, Any]:
        if fail:
            raise RuntimeError(f"Simulated failure in {task.agent}")
        return result or {"summary": f"Completed {task.name}", "data": {}}

    return executor


# ============================================================================
# parse_plan_json tests
# ============================================================================


class TestParsePlanJson:
    def test_parse_simple(self):
        tasks = parse_plan_json(SIMPLE_PLAN_JSON)
        assert len(tasks) == 4
        assert tasks[0]["id"] == "t1"
        assert tasks[1]["dependencies"] == ["t1"]

    def test_parse_with_markdown_fences(self):
        wrapped = f"```json\n{SIMPLE_PLAN_JSON}\n```"
        tasks = parse_plan_json(wrapped)
        assert len(tasks) == 4

    def test_parse_bare_array(self):
        raw = json.dumps(
            [
                {"id": "t1", "name": "A", "agent": "a", "dependencies": []},
            ]
        )
        tasks = parse_plan_json(raw)
        assert len(tasks) == 1

    def test_parse_missing_field_raises(self):
        raw = json.dumps({"tasks": [{"id": "t1", "name": "A"}]})
        with pytest.raises(ValueError, match="missing required field 'agent'"):
            parse_plan_json(raw)

    def test_parse_invalid_json_raises(self):
        with pytest.raises(ValueError, match="Failed to parse"):
            parse_plan_json("not json at all")

    def test_parse_defaults_dependencies(self):
        raw = json.dumps({"tasks": [{"id": "t1", "name": "A", "agent": "a"}]})
        tasks = parse_plan_json(raw)
        assert tasks[0]["dependencies"] == []


# ============================================================================
# tasks_to_graph tests
# ============================================================================


class TestTasksToGraph:
    def test_simple_chain(self):
        tasks = parse_plan_json(SIMPLE_PLAN_JSON)
        graph = tasks_to_graph(tasks)
        assert len(graph.nodes) == 4
        assert graph.nodes["t1"].status == TaskStatus.READY
        assert graph.nodes["t2"].status == TaskStatus.PENDING

    def test_diamond(self):
        tasks = parse_plan_json(DIAMOND_PLAN_JSON)
        graph = tasks_to_graph(tasks)
        assert len(graph.nodes) == 6
        ready = graph.ready_tasks()
        assert {t.id for t in ready} == {"t1", "t2"}

    def test_preserves_pre_postconditions(self):
        raw = json.dumps(
            {
                "tasks": [
                    {
                        "id": "t1",
                        "name": "A",
                        "agent": "a",
                        "dependencies": [],
                        "preconditions": ["model file exists"],
                        "postconditions": ["workload profile populated"],
                    }
                ]
            }
        )
        tasks = parse_plan_json(raw)
        graph = tasks_to_graph(tasks)
        assert graph.nodes["t1"].preconditions == ["model file exists"]
        assert graph.nodes["t1"].postconditions == ["workload profile populated"]


# ============================================================================
# PlannerNode tests
# ============================================================================


class TestPlannerNode:
    def test_static_plan(self, drone_state):
        static_plan = [
            {
                "id": "t1",
                "name": "Analyze workload",
                "agent": "workload_analyzer",
                "dependencies": [],
            },
            {
                "id": "t2",
                "name": "Explore hardware",
                "agent": "hw_explorer",
                "dependencies": ["t1"],
            },
        ]
        planner = PlannerNode(static_plan=static_plan)
        updates = planner(drone_state)

        assert updates["status"] == DesignStatus.EXPLORING.value
        graph = get_task_graph({**drone_state, **updates})
        assert len(graph.nodes) == 2
        assert "t1" in graph.nodes

    def test_llm_plan(self, drone_state):
        llm = MockLLMClient(response_text=SIMPLE_PLAN_JSON)
        planner = PlannerNode(llm=llm)
        updates = planner(drone_state)

        graph = get_task_graph({**drone_state, **updates})
        assert len(graph.nodes) == 4

        # Verify LLM was called with proper context
        assert len(llm.calls) == 1
        msg = llm.calls[0]["messages"][0]["content"]
        assert "delivery drone" in msg.lower() or "5W" in msg or "5.0" in msg

    def test_records_decision(self, drone_state):
        static_plan = [
            {"id": "t1", "name": "Analyze", "agent": "workload_analyzer", "dependencies": []},
        ]
        planner = PlannerNode(static_plan=static_plan)
        updates = planner(drone_state)

        assert len(updates["history"]) == 1
        assert updates["history"][0]["agent"] == "planner"
        assert len(updates["design_rationale"]) == 1

    def test_no_llm_or_plan_raises(self):
        with pytest.raises(ValueError, match="Either llm or static_plan"):
            PlannerNode()

    def test_empty_llm_response_raises(self, drone_state):
        llm = MockLLMClient(response_text="")
        planner = PlannerNode(llm=llm)
        with pytest.raises(ValueError, match="empty response"):
            planner(drone_state)


# ============================================================================
# Dispatcher tests
# ============================================================================


class TestDispatcherBasic:
    def test_empty_graph_completes(self):
        state = create_initial_soc_state(goal="Test")
        dispatcher = Dispatcher()
        result = dispatcher.step(state)
        assert result["status"] == DesignStatus.COMPLETE.value

    def test_register_agents(self):
        dispatcher = Dispatcher()
        dispatcher.register("a", make_mock_executor())
        assert "a" in dispatcher.registered_agents

    def test_register_many(self):
        dispatcher = Dispatcher()
        dispatcher.register_many(
            {
                "a": make_mock_executor(),
                "b": make_mock_executor(),
            }
        )
        assert set(dispatcher.registered_agents) == {"a", "b"}


class TestDispatcherExecution:
    @pytest.fixture()
    def planned_state(self, drone_state):
        """State with a task graph already populated."""
        planner = PlannerNode(static_plan=parse_plan_json(SIMPLE_PLAN_JSON))
        updates = planner(drone_state)
        return {**drone_state, **updates}

    @pytest.fixture()
    def full_dispatcher(self):
        """Dispatcher with all agents registered as mocks."""
        dispatcher = Dispatcher()
        dispatcher.register_many(
            {
                "workload_analyzer": make_mock_executor(
                    {"summary": "Workload analyzed", "operators": ["conv2d", "matmul"]}
                ),
                "hw_explorer": make_mock_executor(
                    {"summary": "Hardware explored", "candidates": ["jetson", "kpu"]}
                ),
                "ppa_assessor": make_mock_executor(
                    {"summary": "PPA assessed", "power_watts": 4.8, "latency_ms": 32.0}
                ),
                "report_generator": make_mock_executor(
                    {"summary": "Report generated", "format": "html"}
                ),
            }
        )
        return dispatcher

    def test_single_step(self, planned_state, full_dispatcher):
        """One step should dispatch t1 (the only ready task)."""
        result = full_dispatcher.step(planned_state)
        graph = get_task_graph(result)

        assert graph.nodes["t1"].status == TaskStatus.COMPLETED
        assert graph.nodes["t2"].status == TaskStatus.READY  # unblocked

    def test_full_run(self, planned_state, full_dispatcher):
        """Full run should complete all 4 tasks."""
        result = full_dispatcher.run(planned_state)

        assert result["status"] == DesignStatus.COMPLETE.value
        graph = get_task_graph(result)
        assert graph.is_complete
        assert all(t.status == TaskStatus.COMPLETED for t in graph.nodes.values())

    def test_decisions_recorded(self, planned_state, full_dispatcher):
        """Each completed task should produce a design decision."""
        result = full_dispatcher.run(planned_state)

        # Planner recorded 1 decision, each of 4 tasks adds 1 more = 5 total
        history = result.get("history", [])
        assert len(history) >= 4  # at least one per dispatched task

    def test_missing_agent_fails_task(self, planned_state):
        """Tasks for unregistered agents should fail."""
        dispatcher = Dispatcher()
        # Only register workload_analyzer, not hw_explorer
        dispatcher.register("workload_analyzer", make_mock_executor())

        result = dispatcher.run(planned_state)
        graph = get_task_graph(result)

        # t1 completed, t2 failed (no hw_explorer), rest stuck
        assert graph.nodes["t1"].status == TaskStatus.COMPLETED
        assert graph.nodes["t2"].status == TaskStatus.FAILED
        assert "No executor registered" in (graph.nodes["t2"].error or "")

    def test_agent_exception_fails_task(self, planned_state):
        """Agent exceptions should be caught and recorded as task failures."""
        dispatcher = Dispatcher()
        dispatcher.register_many(
            {
                "workload_analyzer": make_mock_executor(fail=True),
                "hw_explorer": make_mock_executor(),
                "ppa_assessor": make_mock_executor(),
                "report_generator": make_mock_executor(),
            }
        )

        result = dispatcher.run(planned_state)
        graph = get_task_graph(result)

        assert graph.nodes["t1"].status == TaskStatus.FAILED
        assert "Simulated failure" in (graph.nodes["t1"].error or "")

    def test_max_steps_safety(self, planned_state, full_dispatcher):
        """Run should stop at max_steps even if not complete."""
        # Only allow 1 step — should dispatch t1 but not complete all
        result = full_dispatcher.run(planned_state, max_steps=1)
        graph = get_task_graph(result)

        completed = [t for t in graph.nodes.values() if t.status == TaskStatus.COMPLETED]
        assert len(completed) == 1  # only t1

    def test_dispatch_summary(self, planned_state, full_dispatcher):
        summary = full_dispatcher.get_dispatch_summary(planned_state)
        assert "Ready:" in summary


class TestDispatcherDiamond:
    def test_parallel_start(self, drone_state):
        """Diamond graph should start t1 and t2 in parallel."""
        planner = PlannerNode(static_plan=parse_plan_json(DIAMOND_PLAN_JSON))
        updates = planner(drone_state)
        state = {**drone_state, **updates}

        dispatcher = Dispatcher()
        dispatcher.register_many(
            {
                "workload_analyzer": make_mock_executor({"summary": "Analyzed"}),
                "hw_explorer": make_mock_executor({"summary": "Explored"}),
                "architecture_composer": make_mock_executor({"summary": "Composed"}),
                "ppa_assessor": make_mock_executor({"summary": "Assessed"}),
                "report_generator": make_mock_executor({"summary": "Reported"}),
            }
        )

        # First step: t1 and t2 dispatched in parallel
        result = dispatcher.step(state)
        graph = get_task_graph(result)
        assert graph.nodes["t1"].status == TaskStatus.COMPLETED
        assert graph.nodes["t2"].status == TaskStatus.COMPLETED
        assert graph.nodes["t3"].status == TaskStatus.READY

    def test_full_diamond_run(self, drone_state):
        """Full diamond plan should complete in 5 steps."""
        planner = PlannerNode(static_plan=parse_plan_json(DIAMOND_PLAN_JSON))
        updates = planner(drone_state)
        state = {**drone_state, **updates}

        dispatcher = Dispatcher()
        dispatcher.register_many(
            {
                "workload_analyzer": make_mock_executor(),
                "hw_explorer": make_mock_executor(),
                "architecture_composer": make_mock_executor(),
                "ppa_assessor": make_mock_executor(),
                "report_generator": make_mock_executor(),
            }
        )

        result = dispatcher.run(state)
        assert result["status"] == DesignStatus.COMPLETE.value
        graph = get_task_graph(result)
        assert graph.is_complete


# ============================================================================
# Integration: Planner -> Dispatcher end-to-end
# ============================================================================


class TestIntegration:
    def test_full_pipeline(self):
        """End-to-end: create state -> plan -> dispatch -> complete."""
        # 1. Create initial state
        state = create_initial_soc_state(
            goal="Design an SoC for a warehouse AMR: <15W, <100ms, <$50",
            constraints=DesignConstraints(
                max_power_watts=15.0,
                max_latency_ms=100.0,
                max_cost_usd=50.0,
            ),
            use_case="warehouse_amr",
            platform="amr",
        )
        assert state["status"] == DesignStatus.PLANNING.value

        # 2. Plan
        llm = MockLLMClient(response_text=DIAMOND_PLAN_JSON)
        planner = PlannerNode(llm=llm)
        updates = planner(state)
        state = {**state, **updates}

        assert state["status"] == DesignStatus.EXPLORING.value
        graph = get_task_graph(state)
        assert len(graph.nodes) == 6

        # 3. Dispatch
        dispatcher = Dispatcher()

        results_log = []

        def tracking_executor(suffix: str):
            def executor(task: TaskNode, state: SoCDesignState) -> dict[str, Any]:
                result = {"summary": f"Completed {task.name}", "agent": task.agent}
                results_log.append(result)
                return result

            return executor

        dispatcher.register_many(
            {
                "workload_analyzer": tracking_executor("workload"),
                "hw_explorer": tracking_executor("hw"),
                "architecture_composer": tracking_executor("arch"),
                "ppa_assessor": tracking_executor("ppa"),
                "report_generator": tracking_executor("report"),
            }
        )

        state = dispatcher.run(state)

        # 4. Verify
        assert state["status"] == DesignStatus.COMPLETE.value
        assert len(results_log) == 6  # all 6 tasks executed
        graph = get_task_graph(state)
        assert graph.is_complete
        assert not graph.has_failures

        # Check all tasks have results
        for task in graph.nodes.values():
            assert task.status == TaskStatus.COMPLETED
            assert task.result is not None

        # Verify decision history
        history = state.get("history", [])
        # 1 planner decision + 6 task completions = 7
        assert len(history) == 7

    def test_partial_failure_recovery_info(self):
        """When a task fails, the dispatcher should record useful info."""
        state = create_initial_soc_state(goal="Test failure handling")
        plan = [
            {"id": "t1", "name": "Analyze", "agent": "analyzer", "dependencies": []},
            {"id": "t2", "name": "Explore", "agent": "explorer", "dependencies": ["t1"]},
        ]
        planner = PlannerNode(static_plan=plan)
        updates = planner(state)
        state = {**state, **updates}

        dispatcher = Dispatcher()
        dispatcher.register("analyzer", make_mock_executor(fail=True))
        dispatcher.register("explorer", make_mock_executor())

        result = dispatcher.run(state)
        graph = get_task_graph(result)

        # t1 failed, t2 never ran (blocked)
        assert graph.nodes["t1"].status == TaskStatus.FAILED
        assert graph.nodes["t2"].status == TaskStatus.PENDING
        assert result["status"] == DesignStatus.FAILED.value

        # Failure recorded in history
        history = result.get("history", [])
        fail_entries = [h for h in history if "Failed" in h.get("action", "")]
        assert len(fail_entries) == 1


# ============================================================================
# Dispatcher working memory integration
# ============================================================================


class TestDispatcherWorkingMemory:
    def test_records_in_working_memory(self):
        """Dispatcher should record each task execution in working memory."""
        state = create_initial_soc_state(goal="Test working memory")
        plan = [
            {"id": "t1", "name": "Analyze", "agent": "analyzer", "dependencies": []},
        ]
        planner = PlannerNode(static_plan=plan)
        updates = planner(state)
        state = {**state, **updates}

        dispatcher = Dispatcher()
        dispatcher.register("analyzer", make_mock_executor({"summary": "Analyzed OK"}))

        result = dispatcher.run(state)

        # Working memory should have a record for the analyzer agent
        wm = WorkingMemoryStore(**result.get("working_memory", {}))
        tried = wm.get_tried_descriptions("analyzer")
        assert len(tried) == 1
        assert "t1" in tried[0]

    def test_records_failure_in_working_memory(self):
        """Failed tasks should also be recorded in working memory."""
        state = create_initial_soc_state(goal="Test failure memory")
        plan = [
            {"id": "t1", "name": "Fail task", "agent": "failer", "dependencies": []},
        ]
        planner = PlannerNode(static_plan=plan)
        updates = planner(state)
        state = {**state, **updates}

        dispatcher = Dispatcher()
        dispatcher.register("failer", make_mock_executor(fail=True))

        result = dispatcher.run(state)

        wm = WorkingMemoryStore(**result.get("working_memory", {}))
        tried = wm.get_tried_descriptions("failer")
        assert len(tried) == 1
        assert "FAILED" in wm.get_agent_memory("failer").things_tried[0]["outcome"]

    def test_backward_compatible_no_working_memory(self):
        """States without working_memory key should still work."""
        state = create_initial_soc_state(goal="Test")
        # Remove working_memory key to simulate old state
        del state["working_memory"]

        plan = [
            {"id": "t1", "name": "Analyze", "agent": "analyzer", "dependencies": []},
        ]
        planner = PlannerNode(static_plan=plan)
        updates = planner(state)
        state = {**state, **updates}

        dispatcher = Dispatcher()
        dispatcher.register("analyzer", make_mock_executor())

        result = dispatcher.run(state)
        assert result["status"] == DesignStatus.COMPLETE.value
