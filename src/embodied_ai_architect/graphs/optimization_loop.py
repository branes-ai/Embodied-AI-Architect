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
    """Build a narrative optimization report with tradeoff analysis."""
    pareto = state.get("pareto_front", [])
    mission = state.get("mission_description", "")
    platform = state.get("platform", "edge")

    # --- Classify Pareto designs into archetypes ---
    archetypes = _classify_archetypes(pareto, selected)

    lines: list[str] = []

    # === 1. Executive Summary ===
    lines.extend(_build_executive_summary(state, archetypes, selected))

    # === 2. Key Tradeoffs ===
    lines.extend(_build_tradeoff_analysis(archetypes, mission, platform))

    # === 3. Design Alternatives (comparison table with archetype labels) ===
    lines.append("## Design Alternatives")
    lines.append("")
    lines.extend(_format_comparison_table(archetypes))
    lines.append("")

    # === 4. Per-design commentary ===
    lines.extend(_build_design_commentary(archetypes, mission))

    # === 5. Recommendation ===
    lines.extend(_build_recommendation_detail(selected, state))

    # === 6. Methodology ===
    lines.extend(
        [
            "",
            "## Methodology",
            f"- Explored {state.get('total_evaluations', 0)} design configurations "
            f"across {state.get('iteration', 0) + 1} iteration(s)",
            f"- Pareto front: {len(pareto)} non-dominated designs "
            "(no design is strictly better in all objectives)",
            f"- Optimization used MAP-Elites quality-diversity search "
            f"over a {state.get('num_variables', 17)}-variable joint design space",
            "- All designs satisfy the reticle limit " "(max die edge 26mm, max area 676 mm²)",
            "",
        ]
    )

    # === 7. Citations ===
    if citations:
        lines.append("## Research References")
        for c in citations:
            lines.append(f"- {c}")
        lines.append("")

    # === 8. Convergence ===
    if state.get("convergence_history"):
        lines.append("## Convergence History")
        for entry in state["convergence_history"]:
            lines.append(
                f"- Iteration {entry['iteration']}: "
                f"hypervolume={entry['hypervolume']:.1f}, "
                f"Pareto size={entry['pareto_size']}, "
                f"evaluations={entry['total_evals']}"
            )
        lines.append("")

    # === 9. Glossary ===
    lines.extend(_build_glossary())

    return {
        "should_iterate": False,
        "recommendation": selected,
        "research_citations": citations,
        "final_report": "\n".join(lines),
    }


# ---------------------------------------------------------------------------
# Archetype classification
# ---------------------------------------------------------------------------

# Archetype definitions: (key, label, description, sort_key, reverse)
_ARCHETYPE_DEFS = [
    ("best_efficiency", "Best Efficiency", "Highest intelligence per watt"),
    ("most_accurate", "Most Accurate", "Highest detection accuracy"),
    ("lowest_power", "Lowest Power", "Minimum power consumption"),
    ("lowest_cost", "Lowest Cost", "Cheapest to manufacture"),
    ("balanced", "Balanced (★)", "Best tradeoff across all objectives"),
]


def _classify_archetypes(pareto: list[dict], selected: dict) -> list[tuple[str, str, str, dict]]:
    """Classify Pareto front designs into named archetypes.

    Returns list of (key, label, description, design) tuples.
    """
    if not pareto:
        return [("balanced", "Balanced (★)", "Best tradeoff", selected)]

    def _obj(p: dict, name: str, default: float = 0.0) -> float:
        return p.get("objectives", {}).get(name, p.get("metadata", {}).get(name, default))

    best_eff = max(pareto, key=lambda p: _obj(p, "capability_per_watt"))
    most_acc = max(pareto, key=lambda p: _obj(p, "accuracy_percent"))
    low_power = min(pareto, key=lambda p: _obj(p, "power_watts", 1e6))
    low_cost = min(pareto, key=lambda p: _obj(p, "cost_usd", 1e6))

    # Deduplicate: if two archetypes pick the same design, keep the first
    seen_params: set[str] = set()
    result: list[tuple[str, str, str, dict]] = []

    candidates = [
        ("best_efficiency", "Best Efficiency", "Highest intelligence per watt", best_eff),
        ("most_accurate", "Most Accurate", "Highest detection accuracy", most_acc),
        ("lowest_power", "Lowest Power", "Minimum power consumption", low_power),
        ("lowest_cost", "Lowest Cost", "Cheapest to manufacture", low_cost),
    ]

    for key, label, desc, design in candidates:
        sig = str(sorted(design.get("design_params", {}).items()))
        if sig not in seen_params:
            seen_params.add(sig)
            result.append((key, label, desc, design))

    # Always include the selected/balanced design
    sel_sig = str(sorted(selected.get("design_params", {}).items()))
    if sel_sig not in seen_params:
        result.append(("balanced", "Balanced (★)", "Best tradeoff across all objectives", selected))
    else:
        # Mark the existing entry as the selected one
        for i, (key, label, desc, d) in enumerate(result):
            dsig = str(sorted(d.get("design_params", {}).items()))
            if dsig == sel_sig:
                result[i] = (key, label + " (★)", desc, d)
                break

    return result


# ---------------------------------------------------------------------------
# Executive summary
# ---------------------------------------------------------------------------


def _build_executive_summary(
    state: dict, archetypes: list[tuple[str, str, str, dict]], selected: dict
) -> list[str]:
    """One-paragraph summary of results."""
    mission = state.get("mission_description", "")
    platform = state.get("platform", "edge")
    n_evals = state.get("total_evaluations", 0)
    n_pareto = len(state.get("pareto_front", []))

    sel_objs = selected.get("objectives", {})
    sel_meta = selected.get("metadata", {})
    sel_params = selected.get("design_params", {})

    power = sel_objs.get("power_watts", 0)
    latency = sel_objs.get("latency_ms", 0)
    acc = sel_objs.get("accuracy_percent", sel_meta.get("accuracy_percent", 0))
    capw = sel_objs.get("capability_per_watt", sel_meta.get("capability_per_watt", 0))
    process = sel_params.get("process_nm", "?")

    # Find the range across archetypes
    powers = [d.get("objectives", {}).get("power_watts", 0) for _, _, _, d in archetypes]
    accs = [
        d.get("objectives", {}).get(
            "accuracy_percent", d.get("metadata", {}).get("accuracy_percent", 0)
        )
        for _, _, _, d in archetypes
    ]

    lines = [
        f"# Design Exploration: {mission}",
        "",
        "## Executive Summary",
        "",
        f"We explored {n_evals} hardware/software configurations for "
        f"**{mission}** on a {platform} platform, producing {n_pareto} "
        f"Pareto-optimal designs where no single design dominates all others.",
        "",
        f"The design space reveals a fundamental tension: across the Pareto "
        f"front, power ranges from {min(powers):.1f}W to {max(powers):.1f}W "
        f"while accuracy ranges from {min(accs):.0f}% to {max(accs):.0f}%. "
        f"Higher accuracy demands larger models and more compute, which "
        f"directly increases power and cost.",
        "",
        f"The recommended balanced design uses a **{process}nm** process at "
        f"**{power:.1f}W**, achieving **{acc:.0f}% accuracy** with a "
        f"capability-per-watt of **{capw:.4f}** and **{latency:.1f}ms** "
        f"inference latency.",
        "",
    ]
    return lines


# ---------------------------------------------------------------------------
# Tradeoff analysis
# ---------------------------------------------------------------------------


def _build_tradeoff_analysis(
    archetypes: list[tuple[str, str, str, dict]], mission: str, platform: str
) -> list[str]:
    """Explain the key tradeoffs that shape the Pareto front."""
    lines = [
        "## Key Tradeoffs",
        "",
    ]

    # --- Process node vs cost ---
    processes = set()
    for _, _, _, d in archetypes:
        processes.add(int(d.get("design_params", {}).get("process_nm", 28)))
    if len(processes) > 1:
        sorted_p = sorted(processes)
        lines.extend(
            [
                f"**Process node ({sorted_p[0]}nm–{sorted_p[-1]}nm):** "
                f"Advanced nodes (smaller nm) pack more transistors per mm², enabling "
                f"higher compute density and lower power per operation. However, "
                f"manufacturing cost scales super-linearly — a 2nm wafer costs ~$22K "
                f"vs ~$4K for 28nm. For cost-sensitive {platform} applications, "
                f"mature nodes (16–28nm) often deliver better value.",
                "",
            ]
        )

    # --- Model size vs accuracy vs latency ---
    lines.extend(
        [
            "**Model size vs accuracy:** Larger detector models (e.g., YOLOv8-l/x) "
            "achieve higher mAP but require proportionally more compute (GFLOPS). "
            "On power-constrained hardware, this means either higher power draw "
            "or longer inference latency. Smaller models (YOLOv8-n/s, MobileNet) "
            "trade accuracy for real-time capability.",
            "",
        ]
    )

    # --- Quantization tradeoff ---
    dtypes = set()
    for _, _, _, d in archetypes:
        dtypes.add(d.get("metadata", {}).get("quantization_dtype", "fp16"))
    if len(dtypes) > 1:
        lines.extend(
            [
                f"**Quantization ({', '.join(sorted(dtypes))}):** "
                "Reducing numerical precision (FP32 → FP16 → INT8 → INT4) "
                "cuts memory bandwidth and compute requirements roughly in "
                "proportion. INT8 typically loses 1–3% mAP vs FP32; INT4 can "
                "lose 5–15%. The right choice depends on whether the mission "
                "tolerates that accuracy reduction for the power savings.",
                "",
            ]
        )

    # --- Runtime/compiler ---
    runtimes = set()
    for _, _, _, d in archetypes:
        runtimes.add(d.get("metadata", {}).get("runtime", "pytorch"))
    if len(runtimes) > 1:
        lines.extend(
            [
                f"**Compiler runtime ({', '.join(sorted(runtimes))}):** "
                "The runtime determines how efficiently the model maps to hardware. "
                "TensorRT and TVM perform graph-level optimizations (operator fusion, "
                "memory planning) that can improve hardware utilization by 30–60% "
                "vs unoptimized PyTorch. OpenVINO targets Intel hardware specifically. "
                "Higher utilization means the same hardware delivers more useful "
                "compute per watt.",
                "",
            ]
        )

    # --- Pruning ---
    pruning_ratios = [d.get("metadata", {}).get("pruning_ratio", 0) for _, _, _, d in archetypes]
    if max(pruning_ratios) > 0.2:
        lines.extend(
            [
                f"**Pruning ({min(pruning_ratios):.0%}–{max(pruning_ratios):.0%}):** "
                "Structured pruning removes redundant weights, reducing effective "
                "GFLOPS and memory footprint. Moderate pruning (10–30%) typically "
                "preserves accuracy well. Aggressive pruning (>50%) can significantly "
                "degrade accuracy — the optimizer explores this boundary to find "
                "where the accuracy loss becomes unacceptable for the mission.",
                "",
            ]
        )

    return lines


# ---------------------------------------------------------------------------
# Comparison table with archetype labels
# ---------------------------------------------------------------------------


def _format_comparison_table(archetypes: list[tuple[str, str, str, dict]]) -> list[str]:
    """Format archetype designs as a comparison table."""
    lines: list[str] = []

    headers = [""] + [label for _, label, _, _ in archetypes]
    sep = ["-" * max(len(h), 3) for h in headers]

    def _row(name: str, values: list[str]) -> str:
        return "| " + " | ".join([name] + values) + " |"

    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(sep) + " |")

    # Hardware
    lines.append(
        _row(
            "**Hardware**",
            [""] * len(archetypes),
        )
    )
    lines.append(
        _row(
            "Process",
            [f"{d.get('design_params', {}).get('process_nm', '?')}nm" for _, _, _, d in archetypes],
        )
    )
    lines.append(
        _row(
            "Clock",
            [
                f"{d.get('design_params', {}).get('clock_mhz', 0):.0f} MHz"
                for _, _, _, d in archetypes
            ],
        )
    )
    lines.append(
        _row(
            "Systolic array",
            [
                f"{d.get('design_params', {}).get('array_rows', '?')}"
                f"×{d.get('design_params', {}).get('array_cols', '?')}"
                for _, _, _, d in archetypes
            ],
        )
    )
    lines.append(
        _row(
            "On-chip SRAM",
            [f"{d.get('design_params', {}).get('sram_kb', '?')} KB" for _, _, _, d in archetypes],
        )
    )
    lines.append(
        _row(
            "Compute tiles",
            [
                f"{d.get('design_params', {}).get('num_compute_tiles', '?')}"
                for _, _, _, d in archetypes
            ],
        )
    )

    # Pipeline
    lines.append(_row("**Pipeline**", [""] * len(archetypes)))
    lines.append(
        _row(
            "Detector",
            [
                f"{d.get('metadata', {}).get('model_family', '?')}"
                f"/{d.get('metadata', {}).get('model_variant', '?')}"
                for _, _, _, d in archetypes
            ],
        )
    )
    lines.append(
        _row(
            "Tracker",
            [f"{d.get('metadata', {}).get('tracker_type', 'none')}" for _, _, _, d in archetypes],
        )
    )
    lines.append(
        _row(
            "State estimator",
            [
                f"{d.get('metadata', {}).get('state_estimator', 'none')}"
                for _, _, _, d in archetypes
            ],
        )
    )

    # Compiler
    lines.append(_row("**Compiler**", [""] * len(archetypes)))
    lines.append(
        _row(
            "Runtime", [f"{d.get('metadata', {}).get('runtime', '?')}" for _, _, _, d in archetypes]
        )
    )
    lines.append(
        _row(
            "Quantization",
            [
                f"{d.get('metadata', {}).get('quantization_dtype', '?')}"
                for _, _, _, d in archetypes
            ],
        )
    )
    lines.append(
        _row(
            "Pruning",
            [f"{d.get('metadata', {}).get('pruning_ratio', 0):.0%}" for _, _, _, d in archetypes],
        )
    )
    lines.append(
        _row(
            "Graph fusion",
            [
                "yes" if d.get("metadata", {}).get("graph_fusion") else "no"
                for _, _, _, d in archetypes
            ],
        )
    )
    lines.append(
        _row(
            "HW utilization",
            [
                f"{d.get('metadata', {}).get('effective_utilization', 0):.0%}"
                for _, _, _, d in archetypes
            ],
        )
    )

    # Objectives
    lines.append(_row("**Results**", [""] * len(archetypes)))
    lines.append(
        _row(
            "Power",
            [f"{d.get('objectives', {}).get('power_watts', 0):.2f} W" for _, _, _, d in archetypes],
        )
    )
    lines.append(
        _row(
            "Latency",
            [f"{d.get('objectives', {}).get('latency_ms', 0):.1f} ms" for _, _, _, d in archetypes],
        )
    )
    lines.append(
        _row(
            "Die area",
            [f"{d.get('objectives', {}).get('area_mm2', 0):.0f} mm²" for _, _, _, d in archetypes],
        )
    )
    lines.append(
        _row(
            "Unit cost",
            [f"${d.get('objectives', {}).get('cost_usd', 0):,.0f}" for _, _, _, d in archetypes],
        )
    )
    lines.append(
        _row(
            "Accuracy (mAP)",
            [
                f"{d.get('objectives', {}).get('accuracy_percent', d.get('metadata', {}).get('accuracy_percent', 0)):.1f}%"
                for _, _, _, d in archetypes
            ],
        )
    )
    lines.append(
        _row(
            "**Cap/watt**",
            [
                f"**{d.get('objectives', {}).get('capability_per_watt', d.get('metadata', {}).get('capability_per_watt', 0)):.4f}**"
                for _, _, _, d in archetypes
            ],
        )
    )

    return lines


# ---------------------------------------------------------------------------
# Per-design commentary
# ---------------------------------------------------------------------------


def _build_design_commentary(
    archetypes: list[tuple[str, str, str, dict]], mission: str
) -> list[str]:
    """Generate per-design commentary with physical sanity observations."""
    lines = ["## Design Analysis", ""]

    for key, label, desc, design in archetypes:
        params = design.get("design_params", {})
        meta = design.get("metadata", {})
        objs = design.get("objectives", {})

        process = int(params.get("process_nm", 28))
        clock = float(params.get("clock_mhz", 0))
        rows = int(params.get("array_rows", 8))
        cols = int(params.get("array_cols", 8))
        sram = int(params.get("sram_kb", 0))
        tiles = int(params.get("num_compute_tiles", 1))
        power = objs.get("power_watts", 0)
        cost = objs.get("cost_usd", 0)
        acc = objs.get("accuracy_percent", meta.get("accuracy_percent", 0))
        capw = objs.get("capability_per_watt", meta.get("capability_per_watt", 0))
        runtime = meta.get("runtime", "?")
        quant = meta.get("quantization_dtype", "fp16")
        pruning = meta.get("pruning_ratio", 0)
        detector = f"{meta.get('model_family', '?')}/{meta.get('model_variant', '?')}"
        tracker = meta.get("tracker_type", "none")
        utilization = meta.get("effective_utilization", 0)

        lines.append(f"### {label}")
        lines.append(f"*{desc}*")
        lines.append("")

        # Core description
        array_macs = rows * cols * tiles
        lines.append(
            f"A {process}nm design with {rows}×{cols} systolic array "
            f"({array_macs} MACs across {tiles} tile{'s' if tiles > 1 else ''}), "
            f"clocked at {clock:.0f} MHz with {sram} KB on-chip SRAM. "
            f"Runs {detector} via {runtime} "
            f"({'with' if meta.get('graph_fusion') else 'without'} graph fusion) "
            f"at {quant} precision."
        )
        lines.append("")

        # Why this design is on the Pareto front
        lines.append(
            "**Why it's Pareto-optimal:** ",
        )
        if key == "best_efficiency":
            lines[-1] += (
                f"Achieves the highest capability-per-watt ({capw:.4f}) by "
                f"balancing accuracy ({acc:.0f}%) against power ({power:.1f}W). "
            )
        elif key == "most_accurate":
            lines[-1] += (
                f"Reaches the highest accuracy ({acc:.0f}%) at the cost of "
                f"higher power ({power:.1f}W) and cost (${cost:,.0f}). "
            )
        elif key == "lowest_power":
            lines[-1] += (
                f"Minimizes power to {power:.1f}W, but at reduced accuracy "
                f"({acc:.0f}%). Good for battery-powered {mission.split()[0]}s "
                f"where runtime matters more than peak performance. "
            )
        elif key == "lowest_cost":
            lines[-1] += (
                f"Cheapest to manufacture (${cost:,.0f}/unit) using a mature "
                f"process node. Suitable for high-volume production. "
            )
        else:
            lines[-1] += (
                f"Balances power ({power:.1f}W), accuracy ({acc:.0f}%), "
                f"and cost (${cost:,.0f}). "
            )

        # Physical observations / warnings
        observations: list[str] = []

        # Clock vs process sanity
        # Typical max clocks: 2nm~3GHz, 7nm~2GHz, 16nm~1.5GHz, 28nm~1GHz
        _typical_max = {
            2: 3000,
            3: 2800,
            5: 2500,
            7: 2000,
            10: 1800,
            12: 1600,
            14: 1500,
            16: 1500,
            22: 1200,
            28: 1000,
        }
        typical_max = _typical_max.get(process, 800)
        if clock < typical_max * 0.2:
            observations.append(
                f"Clock ({clock:.0f} MHz) is very low for {process}nm "
                f"(typical range: {typical_max * 0.3:.0f}–{typical_max:.0f} MHz). "
                f"This wastes the speed advantage of the process node — "
                f"a mature node at higher clock may be more cost-effective."
            )
        elif clock > typical_max * 0.95:
            observations.append(
                f"Clock ({clock:.0f} MHz) is near the limit for {process}nm. "
                f"Timing closure may be challenging; consider a faster process "
                f"node or lower clock target."
            )

        # Array size vs SRAM bandwidth
        total_macs = rows * cols * tiles
        if total_macs > 256 and utilization < 0.4:
            observations.append(
                f"Large array ({rows}×{cols}×{tiles} = {total_macs} MACs) "
                f"but only {utilization:.0%} utilization suggests the array "
                f"is starved for data. Consider more SRAM, wider NoC, or "
                f"smaller array."
            )

        # Aggressive quantization/pruning
        if quant == "int4" and acc < 30:
            observations.append(
                f"INT4 quantization with {acc:.0f}% accuracy — this aggressive "
                f"quantization likely degrades accuracy beyond acceptable limits "
                f"for most perception tasks. Consider INT8 for better "
                f"accuracy/efficiency tradeoff."
            )
        if pruning > 0.5 and acc < 30:
            observations.append(
                f"Pruning at {pruning:.0%} with only {acc:.0f}% accuracy — "
                f"the model has been pruned beyond its effective capacity. "
                f"Reduce pruning to <30% for meaningful accuracy."
            )

        # No tracker for drone perception
        if tracker == "none" and "drone" in mission.lower():
            observations.append(
                "No object tracker selected. Drone perception typically "
                "requires multi-frame tracking (ByteTrack, SORT, DeepSORT) "
                "to maintain object identity across frames and handle "
                "ego-motion. This design would need to add tracking "
                "in post-processing."
            )

        # Cost vs process
        if process <= 5 and cost > 10000:
            observations.append(
                f"At {process}nm, NRE costs dominate — mask sets alone cost "
                f"$25M+ at this node. Unit cost of ${cost:,.0f} assumes "
                f"moderate volume; this only makes sense for high-volume "
                f"production or applications where performance justifies "
                f"the premium."
            )

        if observations:
            lines.append("")
            lines.append("**Observations:**")
            for obs in observations:
                lines.append(f"- {obs}")

        lines.append("")

    return lines


# ---------------------------------------------------------------------------
# Recommendation detail
# ---------------------------------------------------------------------------


def _build_recommendation_detail(selected: dict, state: dict) -> list[str]:
    """Build the recommended design section."""
    objs = selected.get("objectives", {})
    meta = selected.get("metadata", {})
    mission = state.get("mission_description", "")

    power = objs.get("power_watts", 0)
    latency = objs.get("latency_ms", 0)
    acc = objs.get("accuracy_percent", meta.get("accuracy_percent", 0))

    lines = [
        "## Recommendation",
        "",
        "Based on the exploration, the **balanced design** (★) is recommended "
        "as the starting point. ",
    ]

    # Mission-specific guidance
    if "10w" in mission.lower() or "10 w" in mission.lower():
        if power <= 10:
            lines[-1] += f"It meets the 10W power budget at {power:.1f}W."
        else:
            lines[-1] += (
                f"**Warning:** at {power:.1f}W it exceeds the 10W budget. "
                f"Consider the Lowest Power alternative."
            )

    lines.extend(
        [
            "",
            "**Next steps:**",
            f"- If accuracy ({acc:.0f}%) is insufficient, move toward the "
            f"Most Accurate design (larger model, less pruning, higher precision)",
            f"- If power ({power:.1f}W) must be reduced further, move toward the "
            f"Lowest Power design (smaller model, more aggressive quantization)",
            f"- If latency ({latency:.1f}ms) is critical, enable graph fusion "
            f"and use TensorRT/TVM for higher utilization",
            "- Run with `--max-iterations 5` for deeper exploration "
            "(more evaluations, tighter convergence)",
        ]
    )

    return lines


# ---------------------------------------------------------------------------
# Glossary
# ---------------------------------------------------------------------------


def _build_glossary() -> list[str]:
    """Appendix: brief definitions of key terms."""
    return [
        "",
        "## Glossary",
        "",
        "- **Capability/watt (cap/watt):** Intelligence per watt — "
        "(accuracy / 100) / power. The primary optimization target: "
        "how much useful perception you get per watt of power.",
        "- **Pareto front:** The set of designs where no design is "
        "strictly better than another in all objectives simultaneously. "
        "Moving along the front always involves giving up something "
        "to gain something else.",
        "- **Systolic array:** A grid of multiply-accumulate (MAC) units "
        "that data flows through in a pipeline. Larger arrays = more "
        "parallel compute, but need proportionally more memory bandwidth "
        "to stay fed.",
        "- **Quantization:** Reducing the numerical precision of model weights "
        "and activations (FP32 → FP16 → INT8 → INT4). Each step roughly "
        "halves memory and bandwidth requirements but may reduce accuracy.",
        "- **Graph fusion:** Compiler optimization that merges consecutive "
        "operations (e.g., Conv + BatchNorm + ReLU) into a single kernel, "
        "eliminating intermediate memory reads/writes.",
        "- **mAP (mean Average Precision):** Standard accuracy metric for "
        "object detection, measuring how well the model identifies and "
        "localizes objects across confidence thresholds.",
        "- **Pruning:** Removing redundant weights from a neural network. "
        "Reduces compute and memory requirements at the cost of accuracy.",
        "- **Reticle limit:** Maximum die size imposed by the lithographic "
        "exposure field (~26×33mm). Dies larger than this cannot be "
        "manufactured with standard single-exposure lithography.",
        "",
    ]


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
