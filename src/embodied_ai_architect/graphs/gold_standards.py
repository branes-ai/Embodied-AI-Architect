"""Gold standards for all 7 demos in the evaluation framework.

Hand-crafted GoldStandard objects per demo that define expected task graphs,
PPA targets, governance triggers, and tool call expectations.

Usage:
    from embodied_ai_architect.graphs.gold_standards import ALL_GOLD_STANDARDS
"""

from __future__ import annotations

from embodied_ai_architect.graphs.evaluation import GoldStandard

# ---------------------------------------------------------------------------
# Demo 1: Delivery drone — single-pass design
# ---------------------------------------------------------------------------

DEMO_1_GOLD = GoldStandard(
    demo_name="demo_1_delivery_drone",
    expected_task_graph={
        "nodes": {
            "t1": {"agent": "workload_analyzer", "dependencies": []},
            "t2": {"agent": "hw_explorer", "dependencies": ["t1"]},
            "t3": {"agent": "architecture_composer", "dependencies": ["t2"]},
            "t4": {"agent": "ppa_assessor", "dependencies": ["t3"]},
            "t5": {"agent": "critic", "dependencies": ["t4"]},
            "t6": {"agent": "report_generator", "dependencies": ["t5"]},
        },
    },
    expected_ppa={
        "power_watts": 5.0,
        "latency_ms": 33.3,
        "cost_usd": 30.0,
    },
    governance_triggers=[],
    expected_tool_calls=[
        "workload_analyzer",
        "hw_explorer",
        "architecture_composer",
        "ppa_assessor",
        "critic",
        "report_generator",
    ],
    max_iterations=1,
    max_duration_seconds=30.0,
    max_cost_tokens=0,
    max_human_interventions=0,
    rationale_keywords=["drone", "perception", "detection", "power", "workload"],
)


# ---------------------------------------------------------------------------
# Demo 2: DSE Pareto — 3-way hardware comparison (warehouse AMR)
# ---------------------------------------------------------------------------

DEMO_2_GOLD = GoldStandard(
    demo_name="demo_2_dse_pareto",
    expected_task_graph={
        "nodes": {
            "t1": {"agent": "workload_analyzer", "dependencies": []},
            "t2": {"agent": "hw_explorer", "dependencies": ["t1"]},
            "t3": {"agent": "design_explorer", "dependencies": ["t2"]},
            "t4": {"agent": "architecture_composer", "dependencies": ["t3"]},
            "t5": {"agent": "ppa_assessor", "dependencies": ["t4"]},
            "t6": {"agent": "critic", "dependencies": ["t5"]},
            "t7": {"agent": "report_generator", "dependencies": ["t6"]},
        },
    },
    expected_ppa={
        "power_watts": 15.0,
        "latency_ms": 50.0,
        "cost_usd": 100.0,
    },
    governance_triggers=[],
    expected_tool_calls=[
        "workload_analyzer",
        "hw_explorer",
        "design_explorer",
        "architecture_composer",
        "ppa_assessor",
        "critic",
        "report_generator",
    ],
    max_iterations=1,
    max_duration_seconds=30.0,
    max_cost_tokens=0,
    max_human_interventions=0,
    expected_pareto_points=3,
    rationale_keywords=["amr", "warehouse", "slam", "pareto", "exploration"],
)


# ---------------------------------------------------------------------------
# Demo 3: Optimization loop convergence
# ---------------------------------------------------------------------------

DEMO_3_GOLD = GoldStandard(
    demo_name="demo_3_optimization",
    expected_task_graph={
        "nodes": {
            "t1": {"agent": "workload_analyzer", "dependencies": []},
            "t2": {"agent": "hw_explorer", "dependencies": ["t1"]},
            "t3": {"agent": "architecture_composer", "dependencies": ["t2"]},
            "t4": {"agent": "ppa_assessor", "dependencies": ["t3"]},
            "t5": {"agent": "critic", "dependencies": ["t4"]},
            "t6": {"agent": "report_generator", "dependencies": ["t5"]},
        },
    },
    expected_ppa={
        "power_watts": 5.0,
        "latency_ms": 33.3,
    },
    governance_triggers=["iteration_limit"],
    expected_tool_calls=[
        "workload_analyzer",
        "hw_explorer",
        "architecture_composer",
        "ppa_assessor",
        "critic",
        "report_generator",
        "design_optimizer",
    ],
    max_iterations=10,
    max_duration_seconds=60.0,
    max_cost_tokens=0,
    max_human_interventions=0,
    rationale_keywords=["optimization", "convergence", "quantize", "power", "strategy"],
)


# ---------------------------------------------------------------------------
# Demo 4: KPU + RTL pipeline
# ---------------------------------------------------------------------------

DEMO_4_GOLD = GoldStandard(
    demo_name="demo_4_kpu_rtl",
    expected_task_graph={
        "nodes": {
            "t1": {"agent": "workload_analyzer", "dependencies": []},
            "t2": {"agent": "hw_explorer", "dependencies": ["t1"]},
            "t3": {"agent": "architecture_composer", "dependencies": ["t2"]},
            "t4": {"agent": "ppa_assessor", "dependencies": ["t3"]},
            "t5": {"agent": "critic", "dependencies": ["t4"]},
            "t6": {"agent": "kpu_configurator", "dependencies": ["t5"]},
            "t7": {"agent": "floorplan_validator", "dependencies": ["t6"]},
            "t8": {"agent": "bandwidth_validator", "dependencies": ["t7"]},
            "t9": {"agent": "rtl_generator", "dependencies": ["t8"]},
            "t10": {"agent": "report_generator", "dependencies": ["t9"]},
        },
    },
    expected_ppa={
        "power_watts": 5.0,
        "latency_ms": 33.3,
    },
    governance_triggers=[],
    expected_tool_calls=[
        "workload_analyzer",
        "hw_explorer",
        "architecture_composer",
        "ppa_assessor",
        "critic",
        "kpu_configurator",
        "floorplan_validator",
        "bandwidth_validator",
        "rtl_generator",
        "report_generator",
    ],
    max_iterations=1,
    max_duration_seconds=60.0,
    max_cost_tokens=0,
    max_human_interventions=0,
    rationale_keywords=["kpu", "floorplan", "bandwidth", "rtl", "verilog"],
)


# ---------------------------------------------------------------------------
# Demo 5: HITL Safety — surgical robot
# ---------------------------------------------------------------------------

DEMO_5_GOLD = GoldStandard(
    demo_name="demo_5_hitl_safety",
    expected_task_graph={
        "nodes": {
            "t1": {"agent": "safety_detector", "dependencies": []},
            "t2": {"agent": "workload_analyzer", "dependencies": []},
            "t3": {"agent": "hw_explorer", "dependencies": ["t2"]},
            "t4": {"agent": "architecture_composer", "dependencies": ["t1", "t3"]},
            "t5": {"agent": "ppa_assessor", "dependencies": ["t4"]},
            "t6": {"agent": "critic", "dependencies": ["t5"]},
            "t7": {"agent": "report_generator", "dependencies": ["t6"]},
        },
    },
    expected_ppa={
        "power_watts": 25.0,
        "latency_ms": 1.0,
        "cost_usd": 500.0,
    },
    governance_triggers=[
        "SAFETY",
        "approval",
    ],
    expected_tool_calls=[
        "safety_detector",
        "workload_analyzer",
        "hw_explorer",
        "architecture_composer",
        "ppa_assessor",
        "critic",
        "report_generator",
    ],
    max_iterations=1,
    max_duration_seconds=30.0,
    max_cost_tokens=0,
    max_human_interventions=3,
    rationale_keywords=["safety", "surgical", "redundancy", "IEC", "lockstep"],
)


# ---------------------------------------------------------------------------
# Demo 6: Experience cache reuse
# ---------------------------------------------------------------------------

DEMO_6_GOLD = GoldStandard(
    demo_name="demo_6_experience_cache",
    expected_task_graph={
        "nodes": {
            "t0": {"agent": "experience_retriever", "dependencies": []},
            "t1": {"agent": "workload_analyzer", "dependencies": ["t0"]},
            "t2": {"agent": "hw_explorer", "dependencies": ["t1"]},
            "t3": {"agent": "architecture_composer", "dependencies": ["t2"]},
            "t4": {"agent": "ppa_assessor", "dependencies": ["t3"]},
            "t5": {"agent": "critic", "dependencies": ["t4"]},
            "t6": {"agent": "report_generator", "dependencies": ["t5"]},
        },
    },
    expected_ppa={
        "power_watts": 5.0,
        "latency_ms": 33.3,
        "cost_usd": 30.0,
    },
    governance_triggers=[],
    expected_tool_calls=[
        "experience_retriever",
        "workload_analyzer",
        "hw_explorer",
        "architecture_composer",
        "ppa_assessor",
        "critic",
        "report_generator",
    ],
    max_iterations=1,
    max_duration_seconds=30.0,
    max_cost_tokens=0,
    max_human_interventions=0,
    rationale_keywords=["experience", "prior", "drone", "warm-start"],
)


# ---------------------------------------------------------------------------
# Demo 7: Full campaign — quadruped robot
# ---------------------------------------------------------------------------

DEMO_7_GOLD = GoldStandard(
    demo_name="demo_7_full_campaign",
    expected_task_graph={
        "nodes": {
            "t1": {"agent": "workload_analyzer", "dependencies": []},
            "t2": {"agent": "hw_explorer", "dependencies": ["t1"]},
            "t3": {"agent": "design_explorer", "dependencies": ["t2"]},
            "t4": {"agent": "architecture_composer", "dependencies": ["t3"]},
            "t5": {"agent": "ppa_assessor", "dependencies": ["t4"]},
            "t6": {"agent": "critic", "dependencies": ["t5"]},
            "t7": {"agent": "report_generator", "dependencies": ["t6"]},
        },
    },
    expected_ppa={
        "power_watts": 15.0,
        "latency_ms": 50.0,
        "cost_usd": 50.0,
    },
    governance_triggers=["iteration_limit"],
    expected_tool_calls=[
        "workload_analyzer",
        "hw_explorer",
        "design_explorer",
        "architecture_composer",
        "ppa_assessor",
        "critic",
        "report_generator",
    ],
    max_iterations=10,
    max_duration_seconds=120.0,
    max_cost_tokens=0,
    max_human_interventions=3,
    expected_pareto_points=3,
    rationale_keywords=[
        "quadruped",
        "multi-workload",
        "slam",
        "detection",
        "lidar",
        "voice",
        "pareto",
        "optimization",
    ],
)


# ---------------------------------------------------------------------------
# All gold standards
# ---------------------------------------------------------------------------

ALL_GOLD_STANDARDS: dict[str, GoldStandard] = {
    "demo_1_delivery_drone": DEMO_1_GOLD,
    "demo_2_dse_pareto": DEMO_2_GOLD,
    "demo_3_optimization": DEMO_3_GOLD,
    "demo_4_kpu_rtl": DEMO_4_GOLD,
    "demo_5_hitl_safety": DEMO_5_GOLD,
    "demo_6_experience_cache": DEMO_6_GOLD,
    "demo_7_full_campaign": DEMO_7_GOLD,
}
