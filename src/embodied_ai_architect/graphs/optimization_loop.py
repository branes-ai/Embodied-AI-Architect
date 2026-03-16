"""LangGraph-orchestrated agentic optimization loop.

Closed-loop pipeline: NL mission → decompose → formulate → optimize →
evaluate → reason → (iterate | recommend). Claude reasons over Pareto
fronts with research library context and steers the search.

Phase 4 of the agentic optimization loop.

Usage:
    from embodied_ai_architect.graphs.optimization_loop import (
        build_optimization_loop,
        run_optimization_loop,
    )

    # Full end-to-end
    result = run_optimization_loop("Drone perception at 5 m/s within 10W")

    # Or build the graph for custom execution
    graph = build_optimization_loop()
    final_state = graph.invoke({"mission_description": "..."})
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional

from typing_extensions import TypedDict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State model
# ---------------------------------------------------------------------------


class OptimizationLoopState(TypedDict, total=False):
    """State flowing through the agentic optimization loop."""

    # --- Input ---
    mission_description: str

    # --- Decomposition output ---
    mission_plan: dict[str, Any]
    platform: str
    mission_type: str
    sub_capabilities: list[dict[str, Any]]
    research_docs_used: list[str]

    # --- Formulation output ---
    constraints: dict[str, Any]
    design_space_config: dict[str, Any]
    num_variables: int

    # --- Optimization output ---
    pareto_front: list[dict[str, Any]]
    hypervolume: float
    knee_point: Optional[dict[str, Any]]
    total_evaluations: int
    convergence_history: list[dict[str, Any]]

    # --- Reasoning output ---
    analysis: str
    recommendation: dict[str, Any]
    research_citations: list[str]
    should_iterate: bool
    refinements: dict[str, Any]

    # --- Loop control ---
    iteration: int
    max_iterations: int
    hypervolume_history: list[float]
    converged: bool

    # --- Final output ---
    final_report: str

    # --- Error tracking ---
    errors: list[str]

    # --- LLM config (optional, for reasoning node) ---
    llm_available: bool


# ---------------------------------------------------------------------------
# System prompt for the reasoning node
# ---------------------------------------------------------------------------

REASONING_SYSTEM_PROMPT = """\
You are an embodied AI systems architect analyzing multi-objective optimization \
results. You have deep knowledge of hardware accelerators, neural network \
architectures, and edge deployment.

Given a Pareto front from a joint design space exploration (pipeline × model × \
compiler × hardware), analyze the results and decide whether to:

1. **RECOMMEND**: The Pareto front is good enough — select the best design and \
explain why, citing research context.
2. **ITERATE**: The search can be improved — suggest specific refinements to the \
design space bounds, constraints, or variable selection.

Your analysis should cover:
- Key tradeoffs visible in the Pareto front
- Whether the accuracy/capability-per-watt is competitive with published results
- Which design variables have the most impact
- Whether constraints are too tight or too loose

Respond with JSON:
{
    "decision": "recommend" or "iterate",
    "analysis": "Your analysis text",
    "selected_design": {design point dict} or null,
    "refinements": {
        "tighten_constraints": {"name": new_value, ...},
        "loosen_constraints": {"name": new_value, ...},
        "narrow_variables": {"var_name": {"lower": x, "upper": y}, ...},
        "reason": "Why these refinements will help"
    } or null,
    "research_citations": ["doc_path", ...]
}
"""


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------


def decompose_node(state: OptimizationLoopState) -> dict:
    """Decompose mission description into structured plan."""
    from embodied_ai_architect.research.decomposer import MissionDecomposer

    mission = state.get("mission_description", "")
    if not mission:
        return {"errors": state.get("errors", []) + ["No mission description provided"]}

    try:
        decomposer = MissionDecomposer()
        plan = decomposer.decompose(mission)

        return {
            "mission_plan": plan.model_dump(),
            "platform": plan.platform,
            "mission_type": plan.mission_type,
            "sub_capabilities": [cap.model_dump() for cap in plan.sub_capabilities],
            "research_docs_used": plan.research_context_used,
        }
    except Exception as e:
        logger.error("Decomposition failed: %s", e)
        return {"errors": state.get("errors", []) + [f"Decomposition failed: {e}"]}


def formulate_node(state: OptimizationLoopState) -> dict:
    """Formulate joint design space from mission plan."""
    plan_data = state.get("mission_plan", {})
    if not plan_data:
        return {"errors": state.get("errors", []) + ["No mission plan available"]}

    try:
        # Extract design space config from the plan
        ds_def = plan_data.get("design_space", {})
        constraints = ds_def.get("constraints", [])

        # Build constraints dict for create_joint_design_space.
        # Mission constraints are passed as soft guidance (workload_type, etc.)
        # but NOT as hard feasibility bounds — the decomposer's budgets are
        # aspirational targets, not gates.  Hard bounds kill the search when
        # the SoC evaluator can't meet them exactly.
        constraints_dict: dict[str, Any] = {
            "include_accuracy": True,
            "workload_type": ds_def.get("workload_type", "detection"),
        }

        # Only propagate power budget as a hard constraint (most commonly
        # a real physical limit) — latency and cost stay soft.
        for c in constraints:
            name = c.get("name", "")
            value = c.get("value", 0)
            ctype = c.get("constraint_type", "upper")
            if name == "power_watts" and ctype == "upper":
                constraints_dict["max_power_watts"] = value

        # Apply any refinements from previous iteration
        refinements = state.get("refinements", {})
        if refinements:
            for name, val in refinements.get("tighten_constraints", {}).items():
                constraints_dict[name] = val
            for name, val in refinements.get("loosen_constraints", {}).items():
                constraints_dict[name] = val

        from embodied_ai_architect.graphs.moo.joint_design_space import (
            create_joint_design_space,
        )

        ds = create_joint_design_space(constraints=constraints_dict)

        return {
            "constraints": constraints_dict,
            "design_space_config": {
                "num_variables": ds.n_var,
                "num_objectives": ds.n_obj,
                "objectives": ds.objectives,
                "constraint_bounds": {k: list(v) for k, v in ds.constraint_bounds.items()},
            },
            "num_variables": ds.n_var,
        }
    except Exception as e:
        logger.error("Formulation failed: %s", e)
        return {"errors": state.get("errors", []) + [f"Formulation failed: {e}"]}


def optimize_node(state: OptimizationLoopState) -> dict:
    """Run MOO engine on the joint design space."""
    constraints_dict = state.get("constraints", {})
    if not constraints_dict:
        return {"errors": state.get("errors", []) + ["No constraints formulated"]}

    try:
        from embodied_ai_architect.graphs.moo.joint_design_space import (
            JointEvaluator,
            create_joint_design_space,
        )
        from embodied_ai_architect.graphs.moo.engine import (
            OptimizationConfig,
            OptimizationEngine,
        )
        from embodied_ai_architect.graphs.moo.map_elites import MAPElitesConfig

        ds = create_joint_design_space(constraints=constraints_dict)
        evaluator = JointEvaluator(
            design_space=ds,
            base_state={"constraints": constraints_dict},
            constraint_bounds=ds.constraint_bounds,
        )

        # Scale effort by iteration — first pass is fast, subsequent passes
        # get more budget as the search is more focused
        iteration = state.get("iteration", 0)
        n_iters = 15 + iteration * 10
        batch_size = 32 + iteration * 16
        initial_pop = 128 + iteration * 64

        me_config = MAPElitesConfig(
            n_iterations=n_iters,
            batch_size=batch_size,
            initial_population=initial_pop,
        )
        config = OptimizationConfig(
            layers="map_elites",
            map_elites=me_config,
            max_workers=4,
        )

        engine = OptimizationEngine(ds, evaluator, config)
        try:
            result = engine.run()
        finally:
            engine.shutdown()

        # Track hypervolume history for convergence
        hv_history = list(state.get("hypervolume_history", []))
        hv_history.append(result.hypervolume)

        # Build convergence entry
        conv_history = list(state.get("convergence_history", []))
        conv_history.append(
            {
                "iteration": state.get("iteration", 0),
                "hypervolume": result.hypervolume,
                "pareto_size": len(result.pareto_front),
                "total_evals": result.total_evaluations,
            }
        )

        return {
            "pareto_front": result.pareto_front[:20],
            "hypervolume": result.hypervolume,
            "knee_point": result.knee_point,
            "total_evaluations": (state.get("total_evaluations", 0) + result.total_evaluations),
            "hypervolume_history": hv_history,
            "convergence_history": conv_history,
        }
    except Exception as e:
        logger.error("Optimization failed: %s", e)
        return {"errors": state.get("errors", []) + [f"Optimization failed: {e}"]}


def evaluate_node(state: OptimizationLoopState) -> dict:
    """Evaluate Pareto front quality and check convergence."""
    pareto = state.get("pareto_front", [])
    if not pareto:
        return {
            "converged": True,
            "analysis": "No Pareto front produced — cannot optimize further.",
        }

    hv_history = state.get("hypervolume_history", [])
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 3)

    # Convergence checks
    converged = False

    # 1. Max iterations reached
    if iteration >= max_iterations - 1:
        converged = True
        logger.info("Converged: max iterations reached (%d)", max_iterations)

    # 2. Hypervolume improvement below threshold
    if len(hv_history) >= 2:
        prev_hv = hv_history[-2]
        curr_hv = hv_history[-1]
        if prev_hv > 0:
            improvement = abs(curr_hv - prev_hv) / prev_hv
            if improvement < 0.02:
                converged = True
                logger.info(
                    "Converged: hypervolume improvement %.1f%% < 2%%",
                    improvement * 100,
                )

    # Summarize Pareto front quality
    cap_per_watt_values = [
        p.get("objectives", {}).get(
            "capability_per_watt",
            p.get("metadata", {}).get("capability_per_watt", 0),
        )
        for p in pareto
    ]
    best_cpw = max(cap_per_watt_values) if cap_per_watt_values else 0

    accuracy_values = [
        p.get("objectives", {}).get(
            "accuracy_percent",
            p.get("metadata", {}).get("accuracy_percent", 0),
        )
        for p in pareto
    ]
    best_acc = max(accuracy_values) if accuracy_values else 0

    power_values = [p.get("objectives", {}).get("power_watts", 1e6) for p in pareto]
    min_power = min(power_values) if power_values else 1e6

    return {
        "converged": converged,
        "analysis": (
            f"Iteration {iteration}: {len(pareto)} Pareto designs, "
            f"best capability/watt={best_cpw:.4f}, "
            f"best accuracy={best_acc:.1f}%, "
            f"min power={min_power:.2f}W, "
            f"hypervolume={state.get('hypervolume', 0):.4f}"
        ),
    }


def reason_node(state: OptimizationLoopState) -> dict:
    """Claude reasons over results with research context.

    If LLM is not available, uses a rule-based fallback.
    """
    pareto = state.get("pareto_front", [])
    converged = state.get("converged", False)

    if converged or not pareto:
        # Select best design and generate recommendation
        return _generate_recommendation(state)

    # Try LLM reasoning if available
    if state.get("llm_available", False):
        try:
            return _reason_with_llm(state)
        except Exception as e:
            logger.warning("LLM reasoning failed, using heuristic: %s", e)

    # Heuristic reasoning fallback
    return _reason_heuristic(state)


def _reason_with_llm(state: OptimizationLoopState) -> dict:
    """Use Claude to reason over the Pareto front."""
    from embodied_ai_architect.llm.client import LLMClient
    from embodied_ai_architect.research.library import ResearchLibrary
    import re

    client = LLMClient()
    library = ResearchLibrary()

    # Retrieve relevant research for the analysis
    docs = library.retrieve(
        tags=[state.get("platform", "edge"), "efficiency"],
        relevance="design_tradeoffs",
        max_results=3,
    )
    research_context = library.build_context_block(docs, max_tokens=4000)

    # Build the analysis prompt
    pareto_summary = json.dumps(state.get("pareto_front", [])[:5], indent=2, default=str)
    prompt = f"""Analyze these optimization results for the mission:
"{state.get('mission_description', '')}"

Platform: {state.get('platform', 'unknown')}
Iteration: {state.get('iteration', 0)}
Hypervolume history: {state.get('hypervolume_history', [])}

Top 5 Pareto-optimal designs:
{pareto_summary}

{research_context}

Decide: RECOMMEND the best design or ITERATE with refinements.
Respond with JSON only."""

    response = client.chat(
        messages=[{"role": "user", "content": prompt}],
        system=REASONING_SYSTEM_PROMPT,
    )

    text = response.text
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*$", "", text)
    data = json.loads(text)

    if data.get("decision") == "iterate":
        return {
            "should_iterate": True,
            "refinements": data.get("refinements", {}),
            "analysis": data.get("analysis", ""),
            "research_citations": data.get("research_citations", []),
        }
    else:
        selected = (
            data.get("selected_design") or state.get("knee_point") or state["pareto_front"][0]
        )
        return _build_recommendation(
            state, selected, data.get("analysis", ""), data.get("research_citations", [])
        )


def _reason_heuristic(state: OptimizationLoopState) -> dict:
    """Rule-based reasoning when LLM is not available."""
    pareto = state.get("pareto_front", [])
    iteration = state.get("iteration", 0)

    # Check if Pareto front quality is acceptable
    best_cpw = max(
        (
            p.get("objectives", {}).get(
                "capability_per_watt",
                p.get("metadata", {}).get("capability_per_watt", 0),
            )
            for p in pareto
        ),
        default=0,
    )

    # Heuristic: if capability/watt > 0.1, we're in a reasonable range
    if best_cpw > 0.1 or iteration >= 1:
        return _generate_recommendation(state)

    # Suggest tightening constraints for focused search
    refinements = {
        "reason": (
            f"Capability/watt={best_cpw:.4f} is low. "
            "Narrowing search to efficient configurations."
        ),
    }

    return {
        "should_iterate": True,
        "refinements": refinements,
        "analysis": (
            f"Iteration {iteration}: capability/watt={best_cpw:.4f}. "
            "Heuristic suggests narrowing the search space."
        ),
        "research_citations": state.get("research_docs_used", []),
    }


def _generate_recommendation(state: OptimizationLoopState) -> dict:
    """Generate final recommendation from the best design."""
    pareto = state.get("pareto_front", [])
    if not pareto:
        return {
            "should_iterate": False,
            "recommendation": {},
            "final_report": "No feasible designs found.",
        }

    # Select best design: prefer knee point, then max capability/watt
    selected = state.get("knee_point")
    if not selected:
        selected = max(
            pareto,
            key=lambda p: p.get("objectives", {}).get(
                "capability_per_watt",
                p.get("metadata", {}).get("capability_per_watt", 0),
            ),
        )

    return _build_recommendation(
        state, selected, state.get("analysis", ""), state.get("research_docs_used", [])
    )


def _build_recommendation(
    state: OptimizationLoopState,
    selected: dict,
    analysis: str,
    citations: list[str],
) -> dict:
    """Build the final recommendation output."""
    objs = selected.get("objectives", {})
    meta = selected.get("metadata", {})
    params = selected.get("design_params", {})

    report_lines = [
        f"# Optimization Report: {state.get('mission_description', '')}",
        "",
        f"## Mission: {state.get('mission_type', 'perception')} on {state.get('platform', 'edge')}",
        f"- Sub-capabilities: {len(state.get('sub_capabilities', []))}",
        f"- Total evaluations: {state.get('total_evaluations', 0)}",
        f"- Iterations: {state.get('iteration', 0) + 1}",
        f"- Pareto front size: {len(state.get('pareto_front', []))}",
        "",
        "## Recommended Design",
        "",
        "### Hardware",
        f"- Process: {params.get('process_nm', '?')}nm",
        f"- Clock: {params.get('clock_mhz', 0)} MHz",
        f"- Array: {params.get('array_rows', '?')}x{params.get('array_cols', '?')}",
        f"- SRAM: {params.get('sram_kb', '?')} KB",
        f"- Tiles: {params.get('num_compute_tiles', '?')}",
        "",
        "### Pipeline",
        f"- Detector: {meta.get('model_family', '?')}/{meta.get('model_variant', '?')}",
        f"- Tracker: {meta.get('tracker_type', '?')}",
        f"- State estimator: {meta.get('state_estimator', '?')}",
        "",
        "### Compiler",
        f"- Runtime: {meta.get('runtime', '?')}",
        f"- Quantization: {meta.get('quantization_dtype', '?')}",
        f"- Pruning: {meta.get('pruning_ratio', 0):.0%}",
        f"- Graph fusion: {meta.get('graph_fusion', '?')}",
        f"- Utilization: {meta.get('effective_utilization', 0):.0%}",
        "",
        "### Objectives",
        f"- Power: {objs.get('power_watts', 0):.2f} W",
        f"- Latency: {objs.get('latency_ms', 0):.2f} ms",
        f"- Area: {objs.get('area_mm2', 0):.2f} mm2",
        f"- Cost: ${objs.get('cost_usd', 0):.2f}",
        f"- Accuracy: {objs.get('accuracy_percent', meta.get('accuracy_percent', 0)):.1f}%",
        f"- Capability/watt: {objs.get('capability_per_watt', meta.get('capability_per_watt', 0)):.4f}",
        "",
        "## Analysis",
        analysis or state.get("analysis", ""),
        "",
    ]

    if citations:
        report_lines.append("## Research Citations")
        for c in citations:
            report_lines.append(f"- {c}")
        report_lines.append("")

    if state.get("convergence_history"):
        report_lines.append("## Convergence History")
        for entry in state["convergence_history"]:
            report_lines.append(
                f"- Iteration {entry['iteration']}: "
                f"HV={entry['hypervolume']:.4f}, "
                f"Pareto={entry['pareto_size']}, "
                f"evals={entry['total_evals']}"
            )

    return {
        "should_iterate": False,
        "recommendation": selected,
        "research_citations": citations,
        "final_report": "\n".join(report_lines),
    }


def iterate_node(state: OptimizationLoopState) -> dict:
    """Prepare for next iteration by incrementing counter."""
    return {
        "iteration": state.get("iteration", 0) + 1,
    }


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def route_after_reason(state: OptimizationLoopState) -> str:
    """Route to iterate or recommend based on reasoning output."""
    if state.get("should_iterate", False) and not state.get("converged", False):
        return "iterate"
    return "recommend"


def route_after_iterate(state: OptimizationLoopState) -> str:
    """After iteration, go back to formulate."""
    return "formulate"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_optimization_loop():
    """Build the LangGraph optimization loop.

    Graph topology:
        decompose → formulate → optimize → evaluate → reason
                                  ↑                      ↓
                                  └── iterate ◄── (if should_iterate)
                                                         ↓
                                                    recommend → END

    Returns:
        Compiled LangGraph StateGraph.
    """
    from langgraph.graph import END, StateGraph

    workflow = StateGraph(OptimizationLoopState)

    # Add nodes
    workflow.add_node("decompose", decompose_node)
    workflow.add_node("formulate", formulate_node)
    workflow.add_node("optimize", optimize_node)
    workflow.add_node("evaluate", evaluate_node)
    workflow.add_node("reason", reason_node)
    workflow.add_node("iterate", iterate_node)
    # recommend is handled within reason_node — the final_report is in state

    # Entry point
    workflow.set_entry_point("decompose")

    # Linear edges
    workflow.add_edge("decompose", "formulate")
    workflow.add_edge("formulate", "optimize")
    workflow.add_edge("optimize", "evaluate")
    workflow.add_edge("evaluate", "reason")

    # Conditional: reason → iterate or END
    workflow.add_conditional_edges(
        "reason",
        route_after_reason,
        {
            "iterate": "iterate",
            "recommend": END,
        },
    )

    # iterate → formulate (loop back)
    workflow.add_edge("iterate", "formulate")

    return workflow.compile()


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------


def run_optimization_loop(
    mission_description: str,
    max_iterations: int = 3,
    llm_available: bool = False,
) -> dict[str, Any]:
    """Run the full optimization loop end-to-end.

    Args:
        mission_description: Natural-language mission description.
        max_iterations: Maximum reasoning/optimization iterations.
        llm_available: Whether to use Claude for reasoning.

    Returns:
        Final state dict with recommendation and report.
    """
    graph = build_optimization_loop()

    initial_state: OptimizationLoopState = {
        "mission_description": mission_description,
        "iteration": 0,
        "max_iterations": max_iterations,
        "hypervolume_history": [],
        "convergence_history": [],
        "errors": [],
        "converged": False,
        "should_iterate": False,
        "llm_available": llm_available,
    }

    start = time.time()
    final_state = graph.invoke(initial_state)
    elapsed = time.time() - start

    logger.info(
        "Optimization loop completed in %.1fs, %d evaluations, %d iterations",
        elapsed,
        final_state.get("total_evaluations", 0),
        final_state.get("iteration", 0) + 1,
    )

    return dict(final_state)
