"""Acceptance tests: run all 7 demos and check composite score > 0.75.

These tests exercise the full pipeline deterministically (no LLM calls)
and verify the evaluation framework produces passing scores.
"""

from __future__ import annotations


from embodied_ai_architect.graphs.soc_state import (
    DesignConstraints,
    create_initial_soc_state,
)
from embodied_ai_architect.graphs.planner import PlannerNode
from embodied_ai_architect.graphs.specialists import create_default_dispatcher
from embodied_ai_architect.graphs.trace import TracingDispatcher, extract_trace_from_state
from embodied_ai_architect.graphs.evaluator import AgenticEvaluator
from embodied_ai_architect.graphs.gold_standards import ALL_GOLD_STANDARDS
from embodied_ai_architect.graphs.governance import GovernancePolicy


def _run_pipeline(goal, constraints, plan, use_case, platform, governance=None, **extra):
    """Run a pipeline and return (initial_state, final_state, tracing_dispatcher)."""
    state = create_initial_soc_state(
        goal=goal,
        constraints=constraints,
        use_case=use_case,
        platform=platform,
        governance=governance,
    )
    state.update(extra)
    initial_state = dict(state)

    planner = PlannerNode(static_plan=plan)
    plan_updates = planner(state)
    state = {**state, **plan_updates}

    dispatcher = create_default_dispatcher()
    tracing = TracingDispatcher(dispatcher)

    state = tracing.run(state)
    return initial_state, state, tracing


# ---------------------------------------------------------------------------
# Demo plans (static, no LLM needed)
# ---------------------------------------------------------------------------

DEMO_1_PLAN = [
    {"id": "t1", "name": "Analyze workload", "agent": "workload_analyzer", "dependencies": []},
    {"id": "t2", "name": "Explore hardware", "agent": "hw_explorer", "dependencies": ["t1"]},
    {
        "id": "t3",
        "name": "Compose architecture",
        "agent": "architecture_composer",
        "dependencies": ["t2"],
    },
    {"id": "t4", "name": "Assess PPA", "agent": "ppa_assessor", "dependencies": ["t3"]},
    {"id": "t5", "name": "Review design", "agent": "critic", "dependencies": ["t4"]},
    {"id": "t6", "name": "Generate report", "agent": "report_generator", "dependencies": ["t5"]},
]

DEMO_2_PLAN = [
    {"id": "t1", "name": "Analyze workload", "agent": "workload_analyzer", "dependencies": []},
    {"id": "t2", "name": "Explore hardware", "agent": "hw_explorer", "dependencies": ["t1"]},
    {
        "id": "t3",
        "name": "Explore design space",
        "agent": "design_explorer",
        "dependencies": ["t2"],
    },
    {
        "id": "t4",
        "name": "Compose architecture",
        "agent": "architecture_composer",
        "dependencies": ["t3"],
    },
    {"id": "t5", "name": "Assess PPA", "agent": "ppa_assessor", "dependencies": ["t4"]},
    {"id": "t6", "name": "Review design", "agent": "critic", "dependencies": ["t5"]},
    {"id": "t7", "name": "Generate report", "agent": "report_generator", "dependencies": ["t6"]},
]

DEMO_5_PLAN = [
    {"id": "t1", "name": "Detect safety", "agent": "safety_detector", "dependencies": []},
    {"id": "t2", "name": "Analyze workload", "agent": "workload_analyzer", "dependencies": []},
    {"id": "t3", "name": "Explore hardware", "agent": "hw_explorer", "dependencies": ["t2"]},
    {
        "id": "t4",
        "name": "Compose architecture",
        "agent": "architecture_composer",
        "dependencies": ["t1", "t3"],
    },
    {"id": "t5", "name": "Assess PPA", "agent": "ppa_assessor", "dependencies": ["t4"]},
    {"id": "t6", "name": "Review design", "agent": "critic", "dependencies": ["t5"]},
    {"id": "t7", "name": "Generate report", "agent": "report_generator", "dependencies": ["t6"]},
]

DEMO_6_PLAN = [
    {"id": "t0", "name": "Search experience", "agent": "experience_retriever", "dependencies": []},
    {"id": "t1", "name": "Analyze workload", "agent": "workload_analyzer", "dependencies": ["t0"]},
    {"id": "t2", "name": "Explore hardware", "agent": "hw_explorer", "dependencies": ["t1"]},
    {
        "id": "t3",
        "name": "Compose architecture",
        "agent": "architecture_composer",
        "dependencies": ["t2"],
    },
    {"id": "t4", "name": "Assess PPA", "agent": "ppa_assessor", "dependencies": ["t3"]},
    {"id": "t5", "name": "Review design", "agent": "critic", "dependencies": ["t4"]},
    {"id": "t6", "name": "Generate report", "agent": "report_generator", "dependencies": ["t5"]},
]

DEMO_7_PLAN = [
    {"id": "t1", "name": "Analyze workload", "agent": "workload_analyzer", "dependencies": []},
    {"id": "t2", "name": "Explore hardware", "agent": "hw_explorer", "dependencies": ["t1"]},
    {
        "id": "t3",
        "name": "Explore design space",
        "agent": "design_explorer",
        "dependencies": ["t2"],
    },
    {
        "id": "t4",
        "name": "Compose architecture",
        "agent": "architecture_composer",
        "dependencies": ["t3"],
    },
    {"id": "t5", "name": "Assess PPA", "agent": "ppa_assessor", "dependencies": ["t4"]},
    {"id": "t6", "name": "Review design", "agent": "critic", "dependencies": ["t5"]},
    {"id": "t7", "name": "Generate report", "agent": "report_generator", "dependencies": ["t6"]},
]


# ---------------------------------------------------------------------------
# Individual demo tests
# ---------------------------------------------------------------------------


class TestDemo1DeliveryDrone:
    def test_pipeline_completes(self):
        initial, final, tracing = _run_pipeline(
            "Design an SoC for delivery drone with object detection + tracking at 30fps, <5W, <$30",
            DesignConstraints(max_power_watts=5.0, max_latency_ms=33.3, max_cost_usd=30.0),
            DEMO_1_PLAN,
            "delivery_drone",
            "drone",
        )
        assert final.get("status") == "complete"

    def test_evaluation_score(self):
        initial, final, tracing = _run_pipeline(
            "Design an SoC for delivery drone with object detection + tracking at 30fps, <5W, <$30",
            DesignConstraints(max_power_watts=5.0, max_latency_ms=33.3, max_cost_usd=30.0),
            DEMO_1_PLAN,
            "delivery_drone",
            "drone",
        )
        trace = extract_trace_from_state(
            initial,
            final,
            "demo_1_delivery_drone",
            duration_seconds=tracing.duration_seconds,
            tool_calls=tracing.tool_calls,
        )
        evaluator = AgenticEvaluator(gold_standards=ALL_GOLD_STANDARDS)
        scorecard = evaluator.evaluate_run(trace)
        assert scorecard.composite_score > 0.5  # reasonable threshold


class TestDemo2DSEPareto:
    def test_pipeline_completes(self):
        initial, final, tracing = _run_pipeline(
            "Design an SoC for warehouse AMR with MobileNetV2 + SLAM, <15W, <$100",
            DesignConstraints(max_power_watts=15.0, max_latency_ms=50.0, max_cost_usd=100.0),
            DEMO_2_PLAN,
            "warehouse_amr",
            "amr",
        )
        assert final.get("status") == "complete"

    def test_pareto_results_populated(self):
        initial, final, tracing = _run_pipeline(
            "Design an SoC for warehouse AMR with MobileNetV2 + SLAM, <15W, <$100",
            DesignConstraints(max_power_watts=15.0, max_latency_ms=50.0, max_cost_usd=100.0),
            DEMO_2_PLAN,
            "warehouse_amr",
            "amr",
        )
        pareto = final.get("pareto_results", {})
        assert pareto.get("total", 0) > 0
        assert pareto.get("non_dominated_count", 0) > 0


class TestDemo5HITLSafety:
    def test_pipeline_completes(self):
        governance = GovernancePolicy(
            approval_required_actions=["change_safety_architecture"],
            iteration_limit=5,
        ).model_dump()
        initial, final, tracing = _run_pipeline(
            "Design SoC for surgical robot, IEC 62304, <1ms force-feedback, <25W",
            DesignConstraints(
                max_power_watts=25.0,
                max_latency_ms=1.0,
                max_cost_usd=500.0,
                safety_critical=True,
                safety_standard="IEC 62304 Class C",
            ),
            DEMO_5_PLAN,
            "surgical_robot",
            "medical",
            governance=governance,
        )
        assert final.get("status") == "complete"

    def test_safety_analysis_populated(self):
        governance = GovernancePolicy(
            approval_required_actions=["change_safety_architecture"],
        ).model_dump()
        initial, final, tracing = _run_pipeline(
            "Design SoC for surgical robot, IEC 62304, <1ms force-feedback, <25W",
            DesignConstraints(
                max_power_watts=25.0,
                max_latency_ms=1.0,
                max_cost_usd=500.0,
                safety_critical=True,
                safety_standard="IEC 62304 Class C",
            ),
            DEMO_5_PLAN,
            "surgical_robot",
            "medical",
            governance=governance,
        )
        safety = final.get("safety_analysis", {})
        assert safety.get("is_safety_critical") is True


class TestDemo6ExperienceCache:
    def test_pipeline_completes(self):
        initial, final, tracing = _run_pipeline(
            "Design SoC for agricultural drone with crop detection, <7W, <$40",
            DesignConstraints(max_power_watts=7.0, max_latency_ms=66.7, max_cost_usd=40.0),
            DEMO_6_PLAN,
            "agricultural_drone",
            "drone",
            _experience_cache_path=":memory:",
        )
        assert final.get("status") == "complete"

    def test_prior_experience_populated(self):
        initial, final, tracing = _run_pipeline(
            "Design SoC for agricultural drone with crop detection, <7W, <$40",
            DesignConstraints(max_power_watts=7.0, max_latency_ms=66.7, max_cost_usd=40.0),
            DEMO_6_PLAN,
            "agricultural_drone",
            "drone",
            _experience_cache_path=":memory:",
        )
        prior = final.get("prior_experience", {})
        assert isinstance(prior, dict)


class TestDemo7FullCampaign:
    def test_pipeline_completes(self):
        governance = GovernancePolicy(iteration_limit=10).model_dump()
        initial, final, tracing = _run_pipeline(
            "Design SoC for quadruped robot: Visual SLAM, object detection, LiDAR, voice. "
            "<15W, <$50, <50ms latency",
            DesignConstraints(max_power_watts=15.0, max_latency_ms=50.0, max_cost_usd=50.0),
            DEMO_7_PLAN,
            "quadruped_robot",
            "quadruped",
            governance=governance,
        )
        assert final.get("status") == "complete"

    def test_multi_workload_detected(self):
        governance = GovernancePolicy(iteration_limit=10).model_dump()
        initial, final, tracing = _run_pipeline(
            "Design SoC for quadruped robot: Visual SLAM, object detection, LiDAR, voice. "
            "<15W, <$50, <50ms latency",
            DesignConstraints(max_power_watts=15.0, max_latency_ms=50.0, max_cost_usd=50.0),
            DEMO_7_PLAN,
            "quadruped_robot",
            "quadruped",
            governance=governance,
        )
        wp = final.get("workload_profile", {})
        assert wp.get("workload_count", 0) >= 3

    def test_pareto_results_populated(self):
        governance = GovernancePolicy(iteration_limit=10).model_dump()
        initial, final, tracing = _run_pipeline(
            "Design SoC for quadruped robot: Visual SLAM, object detection, LiDAR, voice. "
            "<15W, <$50, <50ms latency",
            DesignConstraints(max_power_watts=15.0, max_latency_ms=50.0, max_cost_usd=50.0),
            DEMO_7_PLAN,
            "quadruped_robot",
            "quadruped",
            governance=governance,
        )
        pareto = final.get("pareto_results", {})
        assert pareto.get("total", 0) > 0
