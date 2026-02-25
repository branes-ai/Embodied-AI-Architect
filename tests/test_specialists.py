"""Tests for specialist agent executors.

Tests each specialist individually and then the full end-to-end pipeline:
Planner -> Dispatcher(specialists) -> complete design.
"""

import pytest

from embodied_ai_architect.graphs.dispatcher import Dispatcher
from embodied_ai_architect.graphs.planner import PlannerNode
from embodied_ai_architect.graphs.soc_state import (
    DesignConstraints,
    DesignStatus,
    SoCDesignState,
    create_initial_soc_state,
    get_constraints,
    get_task_graph,
)
from embodied_ai_architect.graphs.specialists import (
    architecture_composer,
    create_default_dispatcher,
    critic,
    hw_explorer,
    ppa_assessor,
    report_generator,
    workload_analyzer,
)
from embodied_ai_architect.graphs.task_graph import TaskNode, TaskStatus

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture()
def drone_state():
    return create_initial_soc_state(
        goal="Design an SoC for a delivery drone with object detection and tracking at 30fps, <5W, <$30",
        constraints=DesignConstraints(
            max_power_watts=5.0,
            max_latency_ms=33.3,
            max_cost_usd=30.0,
        ),
        use_case="delivery_drone",
        platform="drone",
    )


@pytest.fixture()
def amr_state():
    return create_initial_soc_state(
        goal="Design an SoC for warehouse AMR with visual SLAM, object detection, and voice recognition, <15W, <$50",
        constraints=DesignConstraints(
            max_power_watts=15.0,
            max_latency_ms=100.0,
            max_cost_usd=50.0,
        ),
        use_case="warehouse_amr",
        platform="amr",
    )


def make_task(task_id: str, agent: str, deps: list[str] | None = None) -> TaskNode:
    return TaskNode(id=task_id, name=f"Test {agent}", agent=agent, dependencies=deps or [])


# ============================================================================
# Workload Analyzer
# ============================================================================


class TestWorkloadAnalyzer:
    def test_detection_keywords(self, drone_state):
        task = make_task("t1", "workload_analyzer")
        result = workload_analyzer(task, drone_state)

        assert "workload_profile" in result
        profile = result["workload_profile"]
        assert profile["source"] == "goal_estimation"
        assert profile["workload_count"] >= 1

        # Should detect "detection" and "tracking" from goal
        workload_names = [w["name"] for w in profile.get("workloads", [])]
        assert "object_detection" in workload_names
        assert "object_tracking" in workload_names

    def test_multi_workload_detection(self, amr_state):
        task = make_task("t1", "workload_analyzer")
        result = workload_analyzer(task, amr_state)

        profile = result["workload_profile"]
        workload_names = [w["name"] for w in profile.get("workloads", [])]
        assert "visual_slam" in workload_names
        assert "voice_recognition" in workload_names

    def test_state_updates(self, drone_state):
        task = make_task("t1", "workload_analyzer")
        result = workload_analyzer(task, drone_state)

        assert "_state_updates" in result
        assert "workload_profile" in result["_state_updates"]

    def test_has_summary(self, drone_state):
        task = make_task("t1", "workload_analyzer")
        result = workload_analyzer(task, drone_state)
        assert "summary" in result


# ============================================================================
# Hardware Explorer
# ============================================================================


class TestHWExplorer:
    def test_produces_candidates(self, drone_state):
        # Populate workload_profile in state
        drone_state["workload_profile"] = {
            "total_estimated_gflops": 9.2,
            "dominant_op": "convolution",
            "source": "goal_estimation",
        }

        task = make_task("t1", "hw_explorer")
        result = hw_explorer(task, drone_state)

        assert "hardware_candidates" in result
        candidates = result["hardware_candidates"]
        assert len(candidates) >= 3  # should have multiple options
        assert all("score" in c for c in candidates)
        assert all("rank" in c for c in candidates)

    def test_constraint_verdicts(self, drone_state):
        drone_state["workload_profile"] = {
            "total_estimated_gflops": 9.2,
            "dominant_op": "convolution",
        }
        task = make_task("t1", "hw_explorer")
        result = hw_explorer(task, drone_state)

        for c in result["hardware_candidates"]:
            assert "constraint_verdicts" in c
            # Should have power and cost verdicts given our constraints
            verdicts = c["constraint_verdicts"]
            assert "power" in verdicts
            assert "cost" in verdicts

    def test_kpu_scores_well_for_low_power(self, drone_state):
        """KPU should score well for <5W drone use case."""
        drone_state["workload_profile"] = {
            "total_estimated_gflops": 9.2,
            "dominant_op": "convolution",
        }
        task = make_task("t1", "hw_explorer")
        result = hw_explorer(task, drone_state)

        candidates = result["hardware_candidates"]
        kpu = next((c for c in candidates if "KPU" in c["name"]), None)
        assert kpu is not None
        assert kpu["constraint_verdicts"]["power"] == "PASS"
        assert kpu["constraint_verdicts"]["cost"] == "PASS"

    def test_state_updates(self, drone_state):
        drone_state["workload_profile"] = {
            "total_estimated_gflops": 5.0,
            "dominant_op": "convolution",
        }
        task = make_task("t1", "hw_explorer")
        result = hw_explorer(task, drone_state)
        assert "_state_updates" in result
        assert "hardware_candidates" in result["_state_updates"]


# ============================================================================
# Architecture Composer
# ============================================================================


class TestArchitectureComposer:
    @pytest.fixture()
    def state_with_hw(self, drone_state):
        drone_state["workload_profile"] = {
            "workloads": [
                {"name": "object_detection", "model_class": "YOLOv8n"},
            ],
            "dominant_op": "convolution",
            "use_case": "delivery_drone",
        }
        drone_state["hardware_candidates"] = [
            {
                "name": "Stillwater KPU",
                "type": "kpu",
                "compute_paradigm": "dataflow",
                "peak_tops_int8": 20.0,
                "peak_tflops_fp16": 5.0,
                "memory_gb": 2.0,
                "tdp_watts": 5.0,
                "cost_usd": 25.0,
                "score": 85.0,
                "rank": 1,
                "constraint_verdicts": {"power": "PASS", "cost": "PASS"},
            },
        ]
        return drone_state

    def test_produces_architecture(self, state_with_hw):
        task = make_task("t1", "architecture_composer")
        result = architecture_composer(task, state_with_hw)

        assert "selected_architecture" in result
        arch = result["selected_architecture"]
        assert "Stillwater KPU" in arch["primary_compute"]
        assert arch["compute_type"] == "kpu"

    def test_produces_ip_blocks(self, state_with_hw):
        task = make_task("t1", "architecture_composer")
        result = architecture_composer(task, state_with_hw)

        ip_blocks = result["ip_blocks"]
        block_types = [b.get("block_type", b.get("type")) for b in ip_blocks]
        assert "cpu_core" in block_types
        assert "kpu_tile" in block_types
        assert "memory_ctrl" in block_types
        assert "io" in block_types

    def test_vision_workload_adds_isp(self, state_with_hw):
        task = make_task("t1", "architecture_composer")
        result = architecture_composer(task, state_with_hw)

        ip_blocks = result["ip_blocks"]
        block_types = [b.get("block_type", b.get("type")) for b in ip_blocks]
        assert "isp" in block_types

    def test_produces_memory_map(self, state_with_hw):
        task = make_task("t1", "architecture_composer")
        result = architecture_composer(task, state_with_hw)

        assert "memory_map" in result
        assert "regions" in result["memory_map"]
        assert len(result["memory_map"]["regions"]) > 0

    def test_produces_interconnect(self, state_with_hw):
        task = make_task("t1", "architecture_composer")
        result = architecture_composer(task, state_with_hw)

        assert "interconnect" in result
        assert result["interconnect"]["type"] in ("AXI4", "NoC")

    def test_state_updates(self, state_with_hw):
        task = make_task("t1", "architecture_composer")
        result = architecture_composer(task, state_with_hw)

        updates = result["_state_updates"]
        assert "selected_architecture" in updates
        assert "ip_blocks" in updates
        assert "memory_map" in updates
        assert "interconnect" in updates

    def test_no_candidates_raises(self, drone_state):
        task = make_task("t1", "architecture_composer")
        with pytest.raises(ValueError, match="No hardware candidates"):
            architecture_composer(task, drone_state)


# ============================================================================
# PPA Assessor
# ============================================================================


class TestPPAAssessor:
    @pytest.fixture()
    def state_with_arch(self, drone_state):
        drone_state["workload_profile"] = {
            "total_estimated_gflops": 9.2,
            "dominant_op": "convolution",
        }
        drone_state["hardware_candidates"] = [
            {
                "name": "Stillwater KPU",
                "type": "kpu",
                "peak_tops_int8": 20.0,
                "tdp_watts": 5.0,
                "cost_usd": 25.0,
                "score": 85.0,
                "rank": 1,
            },
        ]
        drone_state["selected_architecture"] = {
            "name": "SoC-KPU",
            "primary_compute": "Stillwater KPU",
        }
        drone_state["ip_blocks"] = [
            {"name": "cpu", "type": "cpu", "config": {}},
            {"name": "kpu", "type": "kpu", "config": {}},
            {"name": "mem", "type": "memory", "config": {"capacity_gb": 2}},
            {"name": "io", "type": "io", "config": {}},
        ]
        return drone_state

    def test_produces_ppa_metrics(self, state_with_arch):
        task = make_task("t1", "ppa_assessor")
        result = ppa_assessor(task, state_with_arch)

        ppa = result["ppa_metrics"]
        assert "power_watts" in ppa
        assert "latency_ms" in ppa
        assert ppa["power_watts"] is not None
        assert ppa["latency_ms"] is not None

    def test_verdicts_present(self, state_with_arch):
        task = make_task("t1", "ppa_assessor")
        result = ppa_assessor(task, state_with_arch)

        ppa = result["ppa_metrics"]
        assert "power" in ppa["verdicts"]
        assert "cost" in ppa["verdicts"]
        assert "latency" in ppa["verdicts"]

    def test_overall_verdict(self, state_with_arch):
        task = make_task("t1", "ppa_assessor")
        result = ppa_assessor(task, state_with_arch)

        assert result["overall_verdict"] in ("PASS", "FAIL")

    def test_state_updates(self, state_with_arch):
        task = make_task("t1", "ppa_assessor")
        result = ppa_assessor(task, state_with_arch)

        assert "_state_updates" in result
        assert "ppa_metrics" in result["_state_updates"]

    def test_process_nm_stored(self, state_with_arch):
        """Default process_nm=28 is stored in PPA metrics."""
        task = make_task("t1", "ppa_assessor")
        result = ppa_assessor(task, state_with_arch)

        ppa = result["ppa_metrics"]
        assert ppa["process_nm"] == 28

    def test_custom_process_nm(self, state_with_arch):
        """Setting target_process_nm=7 stores process_nm=7."""
        state_with_arch["constraints"] = DesignConstraints(
            max_power_watts=5.0,
            max_latency_ms=33.3,
            max_cost_usd=30.0,
            target_process_nm=7,
        ).model_dump()
        task = make_task("t1", "ppa_assessor")
        result = ppa_assessor(task, state_with_arch)

        ppa = result["ppa_metrics"]
        assert ppa["process_nm"] == 7

    def test_7nm_reduces_power_and_latency(self, state_with_arch):
        """7nm PPA values should be lower than 28nm PPA values."""
        task = make_task("t1", "ppa_assessor")

        # 28nm baseline
        result_28 = ppa_assessor(task, state_with_arch)
        ppa_28 = result_28["ppa_metrics"]

        # 7nm
        state_7nm = dict(state_with_arch)
        state_7nm["constraints"] = DesignConstraints(
            max_power_watts=5.0,
            max_latency_ms=33.3,
            max_cost_usd=30.0,
            target_process_nm=7,
        ).model_dump()
        result_7 = ppa_assessor(task, state_7nm)
        ppa_7 = result_7["ppa_metrics"]

        assert ppa_7["power_watts"] < ppa_28["power_watts"]
        assert ppa_7["latency_ms"] < ppa_28["latency_ms"]

    def test_large_node_increases_power(self, state_with_arch):
        """65nm power should exceed default 28nm power."""
        task = make_task("t1", "ppa_assessor")

        # 28nm baseline
        result_28 = ppa_assessor(task, state_with_arch)
        ppa_28 = result_28["ppa_metrics"]

        # 65nm
        state_65nm = dict(state_with_arch)
        state_65nm["constraints"] = DesignConstraints(
            max_power_watts=5.0,
            max_latency_ms=33.3,
            max_cost_usd=30.0,
            target_process_nm=65,
        ).model_dump()
        result_65 = ppa_assessor(task, state_65nm)
        ppa_65 = result_65["ppa_metrics"]

        assert ppa_65["power_watts"] > ppa_28["power_watts"]

    def test_cost_breakdown_present(self, state_with_arch):
        """ppa_metrics contains cost_breakdown dict from manufacturing model."""
        task = make_task("t1", "ppa_assessor")
        result = ppa_assessor(task, state_with_arch)

        ppa = result["ppa_metrics"]
        assert "cost_breakdown" in ppa
        bd = ppa["cost_breakdown"]
        assert bd is not None
        assert "die_cost_usd" in bd
        assert "package_cost_usd" in bd
        assert "nre_per_unit_usd" in bd
        assert "yield_percent" in bd
        assert bd["total_unit_cost_usd"] > 0

    def test_cost_uses_manufacturing_model(self, state_with_arch):
        """cost_usd should come from manufacturing model, not static hw catalog price."""
        task = make_task("t1", "ppa_assessor")
        result = ppa_assessor(task, state_with_arch)

        ppa = result["ppa_metrics"]
        # The old static cost was hw["cost_usd"] = 25.0
        # Manufacturing model cost is derived from die area/process/volume
        assert ppa["cost_usd"] > 0
        assert ppa["cost_usd"] != 25.0  # should differ from catalog BOM price

    def test_volume_affects_cost(self, state_with_arch):
        """Higher volume = lower per-unit cost due to NRE amortization."""
        task = make_task("t1", "ppa_assessor")

        # Low volume (default: 10K since drone_state has no target_volume)
        result_low = ppa_assessor(task, state_with_arch)
        cost_low = result_low["ppa_metrics"]["cost_usd"]

        # High volume
        state_high = dict(state_with_arch)
        state_high["constraints"] = DesignConstraints(
            max_power_watts=5.0,
            max_latency_ms=33.3,
            max_cost_usd=30.0,
            target_volume=1_000_000,
        ).model_dump()
        result_high = ppa_assessor(task, state_high)
        cost_high = result_high["ppa_metrics"]["cost_usd"]

        assert cost_high < cost_low


# ============================================================================
# Critic
# ============================================================================


class TestCritic:
    @pytest.fixture()
    def state_with_ppa(self, drone_state):
        drone_state["selected_architecture"] = {
            "name": "SoC-KPU",
            "primary_compute": "Stillwater KPU",
        }
        drone_state["ip_blocks"] = [
            {"name": "cpu", "type": "cpu", "config": {}},
            {"name": "kpu", "type": "kpu", "config": {}},
            {"name": "mem", "type": "memory", "config": {"capacity_gb": 2}},
        ]
        drone_state["ppa_metrics"] = {
            "power_watts": 4.5,
            "latency_ms": 30.0,
            "verdicts": {"power": "PASS", "latency": "PASS"},
        }
        return drone_state

    def test_produces_assessment(self, state_with_ppa):
        task = make_task("t1", "critic")
        result = critic(task, state_with_ppa)

        assert "assessment" in result
        assert result["assessment"] in ("STRONG", "ADEQUATE", "NEEDS_WORK", "SIGNIFICANT_ISSUES")

    def test_identifies_single_compute_risk(self, state_with_ppa):
        task = make_task("t1", "critic")
        result = critic(task, state_with_ppa)

        # Should flag single compute engine
        assert any("Single compute" in issue for issue in result["issues"])

    def test_safety_critical_check(self, drone_state):
        drone_state["constraints"] = DesignConstraints(
            safety_critical=True,
            safety_standard="IEC 62304 Class C",
        ).model_dump()
        drone_state["ip_blocks"] = [
            {"name": "cpu", "type": "cpu", "config": {}},
        ]
        drone_state["ppa_metrics"] = {"verdicts": {}}

        task = make_task("t1", "critic")
        result = critic(task, drone_state)

        assert any("Safety-critical" in i for i in result["issues"])

    def test_has_strengths_for_good_design(self, state_with_ppa):
        task = make_task("t1", "critic")
        result = critic(task, state_with_ppa)

        assert "strengths" in result


# ============================================================================
# Report Generator
# ============================================================================


class TestReportGenerator:
    def test_produces_report(self, drone_state):
        drone_state["selected_architecture"] = {"name": "SoC-KPU"}
        drone_state["ppa_metrics"] = {"power_watts": 4.5}
        drone_state["workload_profile"] = {"dominant_op": "convolution"}
        drone_state["hardware_candidates"] = [{"name": "KPU", "score": 85}]

        task = make_task("t1", "report_generator")
        result = report_generator(task, drone_state)

        assert "report" in result
        report = result["report"]
        assert "title" in report
        assert "sections" in report
        assert "executive_summary" in report["sections"]
        assert "workload_analysis" in report["sections"]
        assert "ppa_assessment" in report["sections"]


# ============================================================================
# Factory
# ============================================================================


class TestFactory:
    def test_create_default_dispatcher(self):
        dispatcher = create_default_dispatcher()
        assert "workload_analyzer" in dispatcher.registered_agents
        assert "hw_explorer" in dispatcher.registered_agents
        assert "architecture_composer" in dispatcher.registered_agents
        assert "ppa_assessor" in dispatcher.registered_agents
        assert "critic" in dispatcher.registered_agents
        assert "report_generator" in dispatcher.registered_agents


# ============================================================================
# Integration: Full pipeline
# ============================================================================


class TestFullPipeline:
    """End-to-end: Planner creates plan -> Dispatcher runs all specialists."""

    DRONE_PLAN = [
        {
            "id": "t1",
            "name": "Analyze perception workload",
            "agent": "workload_analyzer",
            "dependencies": [],
        },
        {
            "id": "t2",
            "name": "Explore hardware candidates",
            "agent": "hw_explorer",
            "dependencies": ["t1"],
        },
        {
            "id": "t3",
            "name": "Compose SoC architecture",
            "agent": "architecture_composer",
            "dependencies": ["t2"],
        },
        {"id": "t4", "name": "Assess PPA metrics", "agent": "ppa_assessor", "dependencies": ["t3"]},
        {"id": "t5", "name": "Review design quality", "agent": "critic", "dependencies": ["t4"]},
        {
            "id": "t6",
            "name": "Generate design report",
            "agent": "report_generator",
            "dependencies": ["t5"],
        },
    ]

    def test_drone_soc_design(self):
        state = create_initial_soc_state(
            goal="Design an SoC for a delivery drone with object detection and tracking at 30fps, <5W, <$30",
            constraints=DesignConstraints(
                max_power_watts=5.0,
                max_latency_ms=33.3,
                max_cost_usd=30.0,
            ),
            use_case="delivery_drone",
            platform="drone",
        )

        # Plan
        planner = PlannerNode(static_plan=self.DRONE_PLAN)
        updates = planner(state)
        state = {**state, **updates}

        # Dispatch
        dispatcher = create_default_dispatcher()
        state = dispatcher.run(state)

        # Verify completion
        assert state["status"] == DesignStatus.COMPLETE.value
        graph = get_task_graph(state)
        assert graph.is_complete
        assert not graph.has_failures

        # Verify state was populated by specialists
        assert state.get("workload_profile", {}).get("source") == "goal_estimation"
        assert len(state.get("hardware_candidates", [])) > 0
        assert state.get("selected_architecture", {}).get("primary_compute") is not None
        assert state.get("ppa_metrics", {}).get("verdicts") is not None
        assert len(state.get("ip_blocks", [])) > 0

        # Verify decision trail
        assert len(state.get("history", [])) >= 7  # planner + 6 tasks
        assert len(state.get("design_rationale", [])) >= 7

    def test_amr_soc_design(self):
        state = create_initial_soc_state(
            goal="Design an SoC for warehouse AMR with visual SLAM, detection, voice, and LiDAR, <15W, <$50",
            constraints=DesignConstraints(
                max_power_watts=15.0,
                max_latency_ms=100.0,
                max_cost_usd=50.0,
            ),
            use_case="warehouse_amr",
            platform="amr",
        )

        planner = PlannerNode(static_plan=self.DRONE_PLAN)
        updates = planner(state)
        state = {**state, **updates}

        dispatcher = create_default_dispatcher()
        state = dispatcher.run(state)

        assert state["status"] == DesignStatus.COMPLETE.value

        # AMR should detect more workloads
        workloads = state.get("workload_profile", {}).get("workloads", [])
        workload_names = [w["name"] for w in workloads]
        assert "visual_slam" in workload_names
        assert "voice_recognition" in workload_names

    def test_kpu_selected_for_drone(self):
        """Verify the KPU is the top candidate for a <5W drone."""
        state = create_initial_soc_state(
            goal="Design an SoC for a delivery drone with object detection, <5W, <$30",
            constraints=DesignConstraints(
                max_power_watts=5.0,
                max_cost_usd=30.0,
            ),
            use_case="delivery_drone",
            platform="drone",
        )

        planner = PlannerNode(static_plan=self.DRONE_PLAN)
        updates = planner(state)
        state = {**state, **updates}

        dispatcher = create_default_dispatcher()
        state = dispatcher.run(state)

        # KPU should be selected (passes both power and cost constraints)
        arch = state.get("selected_architecture", {})
        assert "KPU" in arch.get("primary_compute", "")
