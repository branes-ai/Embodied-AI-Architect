"""Unified loop assembly — DRAFT sketch for the Loop Convergence milestone.

See `docs/plans/roadmap-loop-convergence.md` (Phase 4), `graphs/design_state.py`,
and `graphs/loop_agents.py`.

This module is a **proposal**, not yet wired into the CLI. It assembles the single
LangGraph loop over the unified `DesignState`, closing the critic ↔ optimizer cycle
that today is split across two disjoint graphs:

    seed → critic → (route) ──recommend──► recommend → END
                       │
                    optimize → evaluate ──► critic   (loop back)

Compared to `optimization_loop.py:build_optimization_loop`, the differences that
matter are:

  * one state schema (`DesignState`) instead of `OptimizationLoopState` +
    `SoCDesignState`;
  * the MOO engine is a **tool** (`MooTool`) the optimizer calls, injected at build
    time — the graph does not import the engine;
  * the loop stop condition is single (`route_after_critic` over the `open_issues`
    backlog + hypervolume), not two separate ones.

The `make_moo_engine_tool` adapter below wraps the real `OptimizationEngine` exactly
as `optimize_node` does, but maps its output onto `DesignState` field names.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from embodied_ai_architect.graphs.design_state import DesignState
from embodied_ai_architect.graphs.loop_agents import (
    MooTool,
    critic_node,
    optimizer_node,
    route_after_critic,
)


# ---------------------------------------------------------------------------
# MOO engine, as a tool (the boundary this milestone introduces)
# ---------------------------------------------------------------------------


def make_moo_engine_tool(*, max_workers: int = 4) -> MooTool:
    """Build a MooTool that runs the real OptimizationEngine over the joint space.

    Mirrors `optimization_loop.optimize_node`'s engine invocation, but returns the
    subset of results mapped onto unified `DesignState` field names. The optimizer
    agent calls this; the graph itself never sees the engine.
    """

    def _run(state: DesignState) -> dict:
        from embodied_ai_architect.graphs.moo.joint_design_space import (
            JointEvaluator,
            create_joint_design_space,
        )
        from embodied_ai_architect.graphs.moo.engine import (
            OptimizationConfig,
            OptimizationEngine,
        )
        from embodied_ai_architect.graphs.moo.map_elites import MAPElitesConfig

        constraints = state.get("constraints", {})
        iteration = int(state.get("iteration", 0))

        ds = create_joint_design_space(constraints=constraints)
        evaluator = JointEvaluator(
            design_space=ds,
            base_state={"constraints": constraints},
            constraint_bounds=ds.constraint_bounds,
        )
        # Effort scales with iteration, as in optimize_node.
        me_config = MAPElitesConfig(
            n_iterations=15 + iteration * 10,
            batch_size=32 + iteration * 16,
            initial_population=128 + iteration * 64,
        )
        engine = OptimizationEngine(
            ds,
            evaluator,
            OptimizationConfig(layers="map_elites", map_elites=me_config, max_workers=max_workers),
        )
        try:
            result = engine.run()
        finally:
            engine.shutdown()

        hv_history = list(state.get("hypervolume_history", [])) + [result.hypervolume]
        pareto_points = _merge_pareto(state.get("pareto_points", []), result.pareto_front)
        frontier_history = list(state.get("pareto_frontier_history", [])) + [result.pareto_front]

        # Map engine output → unified DesignState field names.
        return {
            "pareto_points": pareto_points,
            "pareto_frontier_history": frontier_history,
            "hypervolume_history": hv_history,
            "knee_point": result.knee_point,
            "sensitivity": result.sensitivity,
            "atlas": result.atlas or {},
            "moo_results": result.model_dump() if hasattr(result, "model_dump") else {},
        }

    return _run


def _merge_pareto(existing: list[dict], new: list[dict]) -> list[dict]:
    """Monotonically merge new Pareto points into the accumulated frontier (S6).

    Reuses `moo.specialist._merge_pareto_frontiers`, so dominated points are dropped
    and no non-dominated point is lost across iterations. Points flow through the
    ParetoPoint format (top-level power/latency/cost/area) for dominance computation
    and are recovered from `metadata` afterward, so the loop keeps its engine-native
    point shape (nested `objectives`).
    """
    from embodied_ai_architect.graphs.moo.specialist import (
        _merge_pareto_frontiers,
        _pareto_front_to_points,
    )

    existing_pp = _pareto_front_to_points(existing or [])
    new_pp = _pareto_front_to_points(new or [])
    merged_pp, _added, _dominated = _merge_pareto_frontiers(existing_pp, new_pp)
    # `metadata` holds the original engine-format point (_pareto_front_to_points).
    return [pp.get("metadata", pp) for pp in merged_pp]


# ---------------------------------------------------------------------------
# Evaluate node — turn the current best design into PPA verdicts the critic reads
# ---------------------------------------------------------------------------


def evaluate_node(state: DesignState) -> dict:
    """Score the current design against constraints via the real `ppa_assessor` (S7).

    Delegates to `graphs.specialists.ppa_assessor` — the same assessor the dispatcher
    pipeline uses — so the loop's `ppa_metrics.verdicts` are identical to the rest of
    the pipeline for the same state, rather than a separate divergent constraint check.
    """
    from embodied_ai_architect.graphs.specialists import ppa_assessor
    from embodied_ai_architect.graphs.task_graph import TaskNode

    task = TaskNode(id="loop_evaluate", name="Evaluate PPA", agent="ppa_assessor")
    result = ppa_assessor(task, state)
    return {"ppa_metrics": result.get("ppa_metrics", state.get("ppa_metrics", {}))}


# ---------------------------------------------------------------------------
# Seed + recommend endpoints
# ---------------------------------------------------------------------------


def seed_node(state: DesignState) -> dict:
    """Front door (S10): NL mission → constraints → joint design space.

    Given a `mission_description` (or `goal`), runs the `MissionDecomposer` and
    `create_joint_design_space` so a single mission string seeds a valid DesignState
    with no manual setup. If the state is already seeded, or no mission / decompose
    fails, falls back to a minimal design-space stub so the loop still runs.
    """
    updates: dict = {"status": "exploring"}
    if state.get("design_space_config"):
        return updates  # already seeded

    try:
        from embodied_ai_architect.graphs.moo.joint_design_space import create_joint_design_space
        from embodied_ai_architect.graphs.soc_state import DesignConstraints

        if state.get("constraints"):
            # Constraints already set — just materialize a joint design space.
            constraints = dict(state["constraints"])
        else:
            # No constraints: decompose the NL mission into constraints (S10).
            mission = state.get("mission_description") or state.get("goal")
            if not mission:
                updates["design_space_config"] = {"source": "default_joint_space"}
                return updates
            from embodied_ai_architect.research.decomposer import (
                MissionDecomposer,
                plan_to_constraints,
            )

            plan = MissionDecomposer(llm_client=None).decompose(mission)
            constraints = DesignConstraints(**plan_to_constraints(plan)).model_dump()
            updates["constraints"] = constraints
            updates["platform"] = plan.platform
            updates["mission_type"] = plan.mission_type
            updates["mission_plan"] = plan.model_dump()

        ds = create_joint_design_space(constraints={**constraints, "include_accuracy": True})
        updates["design_space_config"] = {
            "num_variables": ds.n_var,
            "num_objectives": ds.n_obj,
            "objectives": ds.objectives,
            "constraint_bounds": {k: list(v) for k, v in ds.constraint_bounds.items()},
            "source": "mission_decomposer",
        }
    except Exception:  # pragma: no cover - decompose/joint-space best effort
        updates.setdefault("design_space_config", {"source": "default_joint_space"})

    return updates


def recommend_node(state: DesignState) -> dict:
    """Terminal: select the final design and emit a short report."""
    point = state.get("knee_point") or _first(state.get("pareto_points", []))
    resolved = sum(1 for i in state.get("open_issues", []) if i.get("status") == "resolved")
    report = (
        f"Converged after {state.get('iteration', 0)} iteration(s). "
        f"Resolved {resolved} issue(s). Selected design: {point}."
    )
    return {"status": "complete", "recommendation": point or {}, "final_report": report}


# ---------------------------------------------------------------------------
# Router with an iteration cap
# ---------------------------------------------------------------------------


def make_router(max_iterations: int = 6) -> Callable[[DesignState], str]:
    """Route critic → optimize|recommend, with a hard iteration cap as a backstop.

    Delegates the convergence decision to the single canonical `route_after_critic`
    (in `loop_agents`) and only adds the iteration-cap backstop on top, so there is
    one stop-condition implementation, not two.
    """

    def _route(state: DesignState) -> str:
        if int(state.get("iteration", 0)) >= max_iterations:
            return "recommend"
        return route_after_critic(state)

    return _route


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------


def build_loop_convergence_graph(
    *,
    moo_tool: Optional[MooTool] = None,
    max_iterations: int = 6,
) -> Any:  # langgraph CompiledStateGraph (untyped to avoid a hard import at module load)
    """Assemble and compile the unified loop.

    Args:
        moo_tool: the MOO-engine boundary the optimizer calls. Defaults to the real
            engine adapter; inject a fake in tests.
        max_iterations: backstop so the loop always terminates.

    Returns:
        A compiled LangGraph app over `DesignState`.
    """
    from functools import partial

    from langgraph.graph import END, StateGraph

    tool = moo_tool if moo_tool is not None else make_moo_engine_tool()

    workflow = StateGraph(DesignState)
    workflow.add_node("seed", seed_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("optimize", partial(optimizer_node, moo_tool=tool))
    workflow.add_node("evaluate", evaluate_node)
    workflow.add_node("recommend", recommend_node)

    workflow.set_entry_point("seed")
    workflow.add_edge("seed", "critic")
    workflow.add_conditional_edges(
        "critic",
        make_router(max_iterations),
        {"optimize": "optimize", "recommend": "recommend"},
    )
    workflow.add_edge("optimize", "evaluate")
    workflow.add_edge("evaluate", "critic")  # loop back
    workflow.add_edge("recommend", END)

    return workflow.compile()


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------


def run_loop_convergence(
    state: DesignState,
    *,
    moo_tool: Optional[MooTool] = None,
    max_iterations: int = 6,
) -> DesignState:
    """Build the graph and invoke it once over `state`."""
    app = build_loop_convergence_graph(moo_tool=moo_tool, max_iterations=max_iterations)
    return app.invoke(state)


def _first(seq: list[Any]) -> Optional[Any]:
    return seq[0] if seq else None
