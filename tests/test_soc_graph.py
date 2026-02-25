"""Tests for the LangGraph outer loop (soc_graph).

Tests cover:
- Single-pass PASS (no optimization needed)
- Optimization convergence (iterate until PASS)
- Iteration limit enforcement
- Optimization history tracking
"""

import pytest

from embodied_ai_architect.graphs.dispatcher import Dispatcher
from embodied_ai_architect.graphs.governance import GovernanceGuard, GovernancePolicy
from embodied_ai_architect.graphs.planner import PlannerNode
from embodied_ai_architect.graphs.soc_state import (
    DesignConstraints,
    DesignStatus,
    PPAMetrics,
    SoCDesignState,
    create_initial_soc_state,
)
from embodied_ai_architect.graphs.task_graph import TaskNode

# We need langgraph for these tests
langgraph = pytest.importorskip("langgraph", reason="langgraph not installed")

from embodied_ai_architect.graphs.soc_graph import build_soc_design_graph

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DEMO_PLAN = [
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
        "name": "Compose architecture",
        "agent": "architecture_composer",
        "dependencies": ["t2"],
    },
    {
        "id": "t4",
        "name": "Assess PPA",
        "agent": "ppa_assessor",
        "dependencies": ["t3"],
    },
    {
        "id": "t5",
        "name": "Review design",
        "agent": "critic",
        "dependencies": ["t4"],
    },
    {
        "id": "t6",
        "name": "Generate report",
        "agent": "report_generator",
        "dependencies": ["t5"],
    },
]


def _make_passing_dispatcher():
    """Dispatcher where PPA assessor returns all PASS verdicts."""
    from embodied_ai_architect.graphs.specialists import (
        architecture_composer,
        critic,
        hw_explorer,
        report_generator,
        workload_analyzer,
    )

    def passing_ppa(task, state):
        constraints = state.get("constraints", {})
        max_power = constraints.get("max_power_watts", 5.0)
        max_latency = constraints.get("max_latency_ms", 33.3)
        ppa = PPAMetrics(
            power_watts=max_power * 0.8,
            latency_ms=max_latency * 0.8,
            verdicts={"power": "PASS", "latency": "PASS", "cost": "PASS"},
        )
        return {
            "summary": "PPA assessment: PASS",
            "overall_verdict": "PASS",
            "ppa_metrics": ppa.model_dump(),
            "_state_updates": {"ppa_metrics": ppa.model_dump()},
        }

    d = Dispatcher()
    d.register_many(
        {
            "workload_analyzer": workload_analyzer,
            "hw_explorer": hw_explorer,
            "architecture_composer": architecture_composer,
            "ppa_assessor": passing_ppa,
            "critic": critic,
            "report_generator": report_generator,
            "design_optimizer": lambda t, s: {
                "summary": "no-op",
                "applied": False,
                "strategy": None,
            },
        }
    )
    return d


def _make_failing_then_passing_dispatcher(fail_iterations=1):
    """Dispatcher where PPA fails for first N iterations, then passes."""
    from embodied_ai_architect.graphs.specialists import (
        architecture_composer,
        critic,
        hw_explorer,
        report_generator,
        workload_analyzer,
    )
    from embodied_ai_architect.graphs.optimizer import design_optimizer

    call_count = {"ppa": 0}

    def conditional_ppa(task, state):
        call_count["ppa"] += 1
        iteration = state.get("iteration", 0)
        constraints = state.get("constraints", {})
        max_power = constraints.get("max_power_watts", 5.0)
        max_latency = constraints.get("max_latency_ms", 33.3)

        # Use ppa_metrics from state if already modified by optimizer
        current_ppa = state.get("ppa_metrics", {})
        current_power = current_ppa.get("power_watts", 6.3)
        current_latency = current_ppa.get("latency_ms", 25.0)

        # On first call (iteration 0), always fail power
        if call_count["ppa"] == 1:
            current_power = 6.3

        power_verdict = "PASS" if current_power <= max_power else "FAIL"
        latency_verdict = "PASS" if current_latency <= max_latency else "FAIL"
        cost_verdict = "PASS"

        ppa = PPAMetrics(
            power_watts=current_power,
            latency_ms=current_latency,
            verdicts={"power": power_verdict, "latency": latency_verdict, "cost": cost_verdict},
            bottlenecks=(
                [f"Power {current_power:.1f}W exceeds {max_power}W"]
                if power_verdict == "FAIL"
                else []
            ),
        )
        return {
            "summary": f"PPA assessment: {'PASS' if power_verdict == 'PASS' else 'FAIL'}",
            "overall_verdict": power_verdict,
            "ppa_metrics": ppa.model_dump(),
            "_state_updates": {"ppa_metrics": ppa.model_dump()},
        }

    d = Dispatcher()
    d.register_many(
        {
            "workload_analyzer": workload_analyzer,
            "hw_explorer": hw_explorer,
            "architecture_composer": architecture_composer,
            "ppa_assessor": conditional_ppa,
            "critic": critic,
            "report_generator": report_generator,
            "design_optimizer": design_optimizer,
        }
    )
    return d


@pytest.fixture()
def passing_state():
    return create_initial_soc_state(
        goal="Design a drone SoC within power budget",
        constraints=DesignConstraints(max_power_watts=5.0, max_latency_ms=33.3, max_cost_usd=30.0),
        use_case="delivery_drone",
        platform="drone",
    )


@pytest.fixture()
def failing_state():
    return create_initial_soc_state(
        goal="Design a drone SoC (will need optimization)",
        constraints=DesignConstraints(max_power_watts=5.0, max_latency_ms=33.3, max_cost_usd=30.0),
        use_case="delivery_drone",
        platform="drone",
        max_iterations=10,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSinglePassPASS:
    def test_passes_without_optimization(self, passing_state):
        """When all constraints pass on first try, no optimization loop."""
        dispatcher = _make_passing_dispatcher()
        planner = PlannerNode(static_plan=DEMO_PLAN)
        graph = build_soc_design_graph(dispatcher=dispatcher, planner=planner)

        result = graph.invoke(passing_state, config={"recursion_limit": 50})

        assert result["status"] == DesignStatus.COMPLETE.value
        assert result["iteration"] == 0  # no optimization iterations
        assert len(result.get("optimization_history", [])) == 1  # one evaluate snapshot


class TestOptimizationConvergence:
    def test_converges_after_optimization(self, failing_state):
        """Design should converge to PASS after optimization iterations."""
        dispatcher = _make_failing_then_passing_dispatcher()
        planner = PlannerNode(static_plan=DEMO_PLAN)
        graph = build_soc_design_graph(dispatcher=dispatcher, planner=planner)

        result = graph.invoke(failing_state, config={"recursion_limit": 50})

        assert result["status"] == DesignStatus.COMPLETE.value
        # Should have done at least 1 optimization iteration
        assert result["iteration"] >= 1
        # Optimization history should have entries
        assert len(result.get("optimization_history", [])) >= 2


class TestIterationLimit:
    def test_respects_iteration_limit(self, failing_state):
        """Graph should stop at iteration limit even if still failing."""
        # Create a dispatcher that always fails
        from embodied_ai_architect.graphs.specialists import (
            architecture_composer,
            critic,
            hw_explorer,
            report_generator,
            workload_analyzer,
        )
        from embodied_ai_architect.graphs.optimizer import design_optimizer

        def always_failing_ppa(task, state):
            ppa = PPAMetrics(
                power_watts=6.3,
                latency_ms=25.0,
                verdicts={"power": "FAIL", "latency": "PASS"},
                bottlenecks=["Power 6.3W exceeds 5W"],
            )
            return {
                "summary": "PPA: FAIL",
                "ppa_metrics": ppa.model_dump(),
                "_state_updates": {"ppa_metrics": ppa.model_dump()},
            }

        d = Dispatcher()
        d.register_many(
            {
                "workload_analyzer": workload_analyzer,
                "hw_explorer": hw_explorer,
                "architecture_composer": architecture_composer,
                "ppa_assessor": always_failing_ppa,
                "critic": critic,
                "report_generator": report_generator,
                "design_optimizer": design_optimizer,
            }
        )

        gov = GovernanceGuard(GovernancePolicy(iteration_limit=3))
        planner = PlannerNode(static_plan=DEMO_PLAN)
        graph = build_soc_design_graph(dispatcher=d, planner=planner, governance=gov)

        failing_state["max_iterations"] = 3
        result = graph.invoke(failing_state, config={"recursion_limit": 50})

        assert result["status"] == DesignStatus.COMPLETE.value
        # Should not exceed the iteration limit
        assert result["iteration"] <= 3


class TestOptimizationHistory:
    def test_history_tracked(self, failing_state):
        """Optimization history should have an entry per evaluate call."""
        dispatcher = _make_failing_then_passing_dispatcher()
        planner = PlannerNode(static_plan=DEMO_PLAN)
        graph = build_soc_design_graph(dispatcher=dispatcher, planner=planner)

        result = graph.invoke(failing_state, config={"recursion_limit": 50})

        history = result.get("optimization_history", [])
        assert len(history) >= 1
        # Each entry should have the expected structure
        for entry in history:
            assert "iteration" in entry
            assert "ppa_snapshot" in entry
            assert "verdicts" in entry


class TestHeterogeneousIPBlocks:
    """Verify the LangGraph loop works with heterogeneous IP block composition."""

    def test_langgraph_loop_produces_physics_blocks(self, passing_state):
        """Full LangGraph loop should produce ip_blocks with physics-based data."""
        from embodied_ai_architect.graphs.specialists import (
            architecture_composer,
            critic,
            hw_explorer,
            report_generator,
            workload_analyzer,
        )

        def passing_ppa(task, state):
            ppa = PPAMetrics(
                power_watts=4.0,
                latency_ms=25.0,
                area_mm2=50.0,
                verdicts={"power": "PASS", "latency": "PASS", "cost": "PASS"},
            )
            return {
                "summary": "PPA: PASS",
                "overall_verdict": "PASS",
                "ppa_metrics": ppa.model_dump(),
                "_state_updates": {"ppa_metrics": ppa.model_dump()},
            }

        d = Dispatcher()
        d.register_many(
            {
                "workload_analyzer": workload_analyzer,
                "hw_explorer": hw_explorer,
                "architecture_composer": architecture_composer,
                "ppa_assessor": passing_ppa,
                "critic": critic,
                "report_generator": report_generator,
                "design_optimizer": lambda t, s: {
                    "summary": "no-op",
                    "applied": False,
                    "strategy": None,
                },
            }
        )

        planner = PlannerNode(static_plan=DEMO_PLAN)
        graph = build_soc_design_graph(dispatcher=d, planner=planner)
        result = graph.invoke(passing_state, config={"recursion_limit": 50})

        assert result["status"] == DesignStatus.COMPLETE.value

        ip_blocks = result.get("ip_blocks", [])
        assert len(ip_blocks) > 0
        # Should have physics-based block data from the new composition
        assert any(b.get("block_type") == "cpu_core" for b in ip_blocks)
        block_types = {b.get("block_type") for b in ip_blocks}
        # Should have infra blocks
        assert "memory_ctrl" in block_types
        assert "noc" in block_types
        assert "io" in block_types
