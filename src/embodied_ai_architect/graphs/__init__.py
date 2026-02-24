"""LangGraph-based operator orchestration for embodied AI pipelines.

This module provides LangGraph-based orchestration for embodied AI operators.
Requires the 'langgraph' optional dependency:

    pip install -e ".[langgraph]"

Usage:
    from embodied_ai_architect.graphs import (
        build_perception_graph,
        build_autonomy_graph,
        PipelineRunner,
    )

    graph = build_perception_graph()
    runner = PipelineRunner(graph)
    result = runner.run_batch(frame=image)
"""

from embodied_ai_architect.graphs.state import (
    EmbodiedPipelineState,
    PipelineStage,
    create_initial_state,
    format_timing_summary,
    get_total_latency_ms,
    is_over_budget,
)
from embodied_ai_architect.graphs.task_graph import (
    CycleError,
    TaskGraph,
    TaskNode,
    TaskStatus,
)
from embodied_ai_architect.graphs.soc_state import (
    DesignConstraints,
    DesignDecision,
    DesignStatus,
    PPAMetrics,
    SoCDesignState,
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
from embodied_ai_architect.graphs.planner import (
    PlannerNode,
    create_planner_node,
    parse_plan_json,
    tasks_to_graph,
)
from embodied_ai_architect.graphs.dispatcher import (
    Dispatcher,
    DispatchError,
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
from embodied_ai_architect.graphs.soc_state import (
    get_dependency_results,
    get_task_result,
)
from embodied_ai_architect.graphs.memory import (
    AgentWorkingMemory,
    WorkingMemoryStore,
)
from embodied_ai_architect.graphs.optimizer import (
    OPTIMIZATION_STRATEGIES,
    design_optimizer,
)
from embodied_ai_architect.graphs.governance import (
    AuditEntry,
    GovernanceGuard,
    GovernancePolicy,
)
from embodied_ai_architect.graphs.experience import (
    DesignEpisode,
    ExperienceCache,
)
from embodied_ai_architect.graphs.technology import (
    SRAM_DENSITY,
    TECHNOLOGY_NODES,
    TechnologyNode,
    estimate_area_um2,
    estimate_sram_area_mm2,
    estimate_timing_ps,
    get_technology,
)
from embodied_ai_architect.graphs.kpu_config import (
    ComputeTileConfig,
    DRAMConfig,
    KPUMicroArchConfig,
    KPU_PRESETS,
    MemoryTileConfig,
    NoCConfig,
    create_kpu_config,
)
from embodied_ai_architect.graphs.floorplan import (
    FloorplanEstimate,
    TileDimensions,
    estimate_floorplan,
)
from embodied_ai_architect.graphs.ip_blocks import (
    ARM_A55_PRESET,
    ARM_A78_PRESET,
    CPUCoreConfig,
    DEFAULT_IO_PRESET,
    DEFAULT_ISP_PRESET,
    GPUClusterConfig,
    IPBlockConfig,
    ISPConfig,
    KPUBlockConfig,
    LPDDR4X_CONTROLLER_PRESET,
    MEDIUM_GPU_PRESET,
    MESH_NOC_PRESET,
    MemoryControllerConfig,
    NoCIPConfig,
    RISCV_SMALL_PRESET,
    SMALL_GPU_PRESET,
    SoCComposition,
    kpu_preset_block,
)
from embodied_ai_architect.graphs.bandwidth import (
    BandwidthLink,
    BandwidthMatchResult,
    check_bandwidth_match,
)
from embodied_ai_architect.graphs.kpu_specialists import (
    bandwidth_validator,
    floorplan_validator,
    kpu_configurator,
    kpu_optimizer,
)
from embodied_ai_architect.graphs.kpu_loop import (
    KPULoopConfig,
    KPULoopResult,
    run_kpu_loop,
)
from embodied_ai_architect.graphs.rtl_loop import (
    RTLLoopConfig,
    RTLLoopResult,
    run_rtl_loop,
)
from embodied_ai_architect.graphs.rtl_specialists import (
    rtl_generator,
    rtl_ppa_assessor,
)
from embodied_ai_architect.graphs.soc_state import (
    get_bandwidth,
    get_floorplan,
    get_kpu_config,
    get_rtl_summary,
)
from embodied_ai_architect.graphs.evaluation import (
    DimensionScore,
    GoldStandard,
    RunTrace,
    Scorecard,
)
from embodied_ai_architect.graphs.scoring import (
    score_adaptability,
    score_convergence,
    score_decomposition,
    score_efficiency,
    score_exploration_efficiency,
    score_governance,
    score_ppa_accuracy,
    score_reasoning,
    score_tool_use,
)
from embodied_ai_architect.graphs.evaluator import (
    AgenticEvaluator,
    DEFAULT_WEIGHTS,
)
from embodied_ai_architect.graphs.gold_standards import (
    ALL_GOLD_STANDARDS,
)
from embodied_ai_architect.graphs.pareto import (
    ParetoPoint,
    compute_pareto_front,
    design_explorer,
    identify_knee_point,
)
from embodied_ai_architect.graphs.safety import (
    safety_detector,
    detect_safety_critical,
)
from embodied_ai_architect.graphs.experience_specialist import (
    compute_similarity,
    experience_retriever,
)
from embodied_ai_architect.graphs.trace import (
    TracingDispatcher,
    extract_trace_from_state,
)
from embodied_ai_architect.graphs.golden_traces import (
    TraceComparison,
    compare_traces,
    load_golden_trace,
    save_golden_trace,
)
from embodied_ai_architect.graphs.governance import (
    CostTracker,
)
from embodied_ai_architect.graphs.specialists import (
    aggregate_workload_requirements,
    map_workloads_to_accelerators,
)

__all__ = [
    # Perception pipeline state
    "EmbodiedPipelineState",
    "PipelineStage",
    "create_initial_state",
    "format_timing_summary",
    "get_total_latency_ms",
    "is_over_budget",
    # Task graph engine
    "CycleError",
    "TaskGraph",
    "TaskNode",
    "TaskStatus",
    # SoC design state
    "DesignConstraints",
    "DesignDecision",
    "DesignStatus",
    "PPAMetrics",
    "SoCDesignState",
    "create_initial_soc_state",
    "get_constraints",
    "get_iteration_summary",
    "get_optimization_history",
    "get_ppa_metrics",
    "get_task_graph",
    "get_working_memory",
    "is_design_complete",
    "is_over_iteration_limit",
    "record_audit",
    "record_decision",
    "set_task_graph",
    "update_working_memory",
    # Planner
    "PlannerNode",
    "create_planner_node",
    "parse_plan_json",
    "tasks_to_graph",
    # Dispatcher
    "Dispatcher",
    "DispatchError",
    # Specialists
    "architecture_composer",
    "create_default_dispatcher",
    "critic",
    "hw_explorer",
    "ppa_assessor",
    "report_generator",
    "workload_analyzer",
    # State helpers
    "get_dependency_results",
    "get_task_result",
    # Working memory
    "AgentWorkingMemory",
    "WorkingMemoryStore",
    # Optimizer
    "OPTIMIZATION_STRATEGIES",
    "design_optimizer",
    # Governance
    "AuditEntry",
    "GovernanceGuard",
    "GovernancePolicy",
    # Experience
    "DesignEpisode",
    "ExperienceCache",
    # Technology
    "SRAM_DENSITY",
    "TECHNOLOGY_NODES",
    "TechnologyNode",
    "estimate_area_um2",
    "estimate_sram_area_mm2",
    "estimate_timing_ps",
    "get_technology",
    # KPU config
    "ComputeTileConfig",
    "DRAMConfig",
    "KPUMicroArchConfig",
    "KPU_PRESETS",
    "MemoryTileConfig",
    "NoCConfig",
    "create_kpu_config",
    # Floorplan
    "FloorplanEstimate",
    "TileDimensions",
    "estimate_floorplan",
    # IP Blocks
    "ARM_A55_PRESET",
    "ARM_A78_PRESET",
    "CPUCoreConfig",
    "DEFAULT_IO_PRESET",
    "DEFAULT_ISP_PRESET",
    "GPUClusterConfig",
    "IPBlockConfig",
    "ISPConfig",
    "KPUBlockConfig",
    "LPDDR4X_CONTROLLER_PRESET",
    "MEDIUM_GPU_PRESET",
    "MESH_NOC_PRESET",
    "MemoryControllerConfig",
    "NoCIPConfig",
    "RISCV_SMALL_PRESET",
    "SMALL_GPU_PRESET",
    "SoCComposition",
    "kpu_preset_block",
    # Bandwidth
    "BandwidthLink",
    "BandwidthMatchResult",
    "check_bandwidth_match",
    # KPU specialists
    "bandwidth_validator",
    "floorplan_validator",
    "kpu_configurator",
    "kpu_optimizer",
    # KPU loop
    "KPULoopConfig",
    "KPULoopResult",
    "run_kpu_loop",
    # RTL loop
    "RTLLoopConfig",
    "RTLLoopResult",
    "run_rtl_loop",
    # RTL specialists
    "rtl_generator",
    "rtl_ppa_assessor",
    # State helpers (Phase 3)
    "get_bandwidth",
    "get_floorplan",
    "get_kpu_config",
    "get_rtl_summary",
    # Evaluation framework (Phase 4)
    "DimensionScore",
    "GoldStandard",
    "RunTrace",
    "Scorecard",
    "score_adaptability",
    "score_convergence",
    "score_decomposition",
    "score_efficiency",
    "score_exploration_efficiency",
    "score_governance",
    "score_ppa_accuracy",
    "score_reasoning",
    "score_tool_use",
    "AgenticEvaluator",
    "DEFAULT_WEIGHTS",
    "ALL_GOLD_STANDARDS",
    # Pareto
    "ParetoPoint",
    "compute_pareto_front",
    "design_explorer",
    "identify_knee_point",
    # Safety
    "safety_detector",
    "detect_safety_critical",
    # Experience specialist
    "compute_similarity",
    "experience_retriever",
    # Trace capture
    "TracingDispatcher",
    "extract_trace_from_state",
    # Golden traces
    "TraceComparison",
    "compare_traces",
    "load_golden_trace",
    "save_golden_trace",
    # Cost tracking
    "CostTracker",
    # Multi-workload
    "aggregate_workload_requirements",
    "map_workloads_to_accelerators",
]

# Lazy imports for langgraph-dependent modules
def __getattr__(name):
    """Lazy import for langgraph-dependent modules."""
    if name == "PipelineRunner":
        from embodied_ai_architect.graphs.runner import PipelineRunner
        return PipelineRunner
    elif name == "build_perception_graph":
        from embodied_ai_architect.graphs.pipelines.perception import build_perception_graph
        return build_perception_graph
    elif name == "build_autonomy_graph":
        from embodied_ai_architect.graphs.pipelines.autonomy import build_autonomy_graph
        return build_autonomy_graph
    elif name == "build_soc_design_graph":
        from embodied_ai_architect.graphs.soc_graph import build_soc_design_graph
        return build_soc_design_graph
    elif name == "SoCDesignRunner":
        from embodied_ai_architect.graphs.soc_runner import SoCDesignRunner
        return SoCDesignRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
