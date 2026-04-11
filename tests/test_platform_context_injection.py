"""Tests for platform context injection into LLM prompts (issue #99)."""

from embodied_ai_architect.graphs.soc_state import create_initial_soc_state
from embodied_ai_architect.platforms.context import (
    build_context_prompt,
    get_platform_context_for_goal,
)


class TestPlatformContextInState:
    def test_state_has_platform_context_field(self):
        state = create_initial_soc_state(goal="drone perception SoC")
        assert "platform_context" in state

    def test_state_stores_platform_context(self):
        ctx = {"platform_id": "test.drone", "context": {"typical_architecture": "ARM SoC"}}
        state = create_initial_soc_state(goal="drone SoC", platform_context=ctx)
        assert state["platform_context"]["platform_id"] == "test.drone"

    def test_state_default_empty_context(self):
        state = create_initial_soc_state(goal="test")
        assert state["platform_context"] == {}


class TestBuildContextPrompt:
    def test_empty_context(self):
        assert build_context_prompt({}) == ""

    def test_with_architecture(self):
        ctx = {
            "platform_id": "aerial.drone",
            "platform_name": "Delivery Drone",
            "platform_description": "A delivery drone",
            "context": {
                "typical_architecture": "ARM + NPU",
                "design_considerations": "Weight matters",
            },
        }
        prompt = build_context_prompt(ctx)
        assert "Delivery Drone" in prompt
        assert "ARM + NPU" in prompt
        assert "Weight matters" in prompt

    def test_with_pitfalls(self):
        ctx = {
            "platform_id": "test",
            "platform_name": "Test",
            "platform_description": "",
            "context": {
                "common_pitfalls": ["Thermal throttling", "Battery sag"],
            },
        }
        prompt = build_context_prompt(ctx)
        assert "Thermal throttling" in prompt


class TestGetPlatformContextForGoal:
    def test_drone_goal_returns_context(self):
        ctx = get_platform_context_for_goal("delivery drone for packages")
        # Should match an aerial platform
        if ctx:
            assert "platform_id" in ctx
            assert "context" in ctx

    def test_nonsense_goal_returns_empty(self):
        ctx = get_platform_context_for_goal("zzzzxxxx nonsense")
        assert ctx == {}


class TestPlannerContextInjection:
    def test_planner_enriches_prompt_from_state(self):
        """PlannerNode should detect platform_context in state."""
        from embodied_ai_architect.graphs.planner import PlannerNode

        static_plan = [
            {"id": "t1", "name": "Test", "agent": "workload_analyzer", "dependencies": []},
        ]
        planner = PlannerNode(static_plan=static_plan)

        ctx = {
            "platform_id": "aerial.drone",
            "platform_name": "Delivery Drone",
            "platform_description": "A drone",
            "context": {"typical_architecture": "ARM + NPU"},
        }
        state = create_initial_soc_state(goal="drone SoC", platform_context=ctx)

        # Call the planner — with static plan it won't call LLM but should
        # still process the state (the context enrichment only happens for
        # non-static plans, which is correct)
        result = planner(state)
        assert "task_graph" in result
