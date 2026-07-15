"""Observable, deterministic driver + tracer for the unified Loop Convergence loop.

The compiled `loop_convergence_graph` runs its nodes inside LangGraph, which hides
the intermediate steps. This module drives the same loop
(seed → (critic → route → optimize → evaluate)* → recommend) *manually* using the
real node functions, and records a `LoopTrace` — one `LoopStep` per node with the
DECISION it made and the REASON why. It also logs each step to the
`embodied_ai_architect.loop` logger.

Use it to review loop behaviour ("why did it iterate / converge / apply that
edit?") and as the engine behind the deterministic acceptance harness in
`tests/test_loop_convergence_acceptance.py`.

    trace = run_loop_traced(state, moo_tool=fake_tool)
    print(trace.render())
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

from embodied_ai_architect.graphs.design_state import DesignState, open_issues
from embodied_ai_architect.graphs.loop_agents import MooTool, critic_node, optimizer_node
from embodied_ai_architect.graphs.loop_convergence_graph import (
    evaluate_node,
    make_router,
    recommend_node,
    seed_node,
)

logger = logging.getLogger("embodied_ai_architect.loop")

# An evaluate step: state -> {"ppa_metrics": ...}. Defaults to the real evaluate_node
# (which delegates to ppa_assessor). Scenarios can inject a scripted one to drive a
# known verdict trajectory.
EvaluateFn = Callable[[DesignState], dict]


@dataclass
class LoopStep:
    """One node invocation: what it decided and why."""

    iteration: int
    node: str  # seed | critic | route | optimize | evaluate | recommend
    decision: str  # short outcome
    reason: str  # WHY the decision was made
    detail: dict = field(default_factory=dict)


@dataclass
class LoopTrace:
    steps: list[LoopStep]
    final_state: DesignState

    @property
    def iterations(self) -> int:
        return int(self.final_state.get("iteration", 0))

    @property
    def converged(self) -> bool:
        return bool(self.final_state.get("converged", False))

    @property
    def final_verdicts(self) -> dict:
        return dict(self.final_state.get("ppa_metrics", {}).get("verdicts", {}))

    def nodes(self) -> list[str]:
        return [s.node for s in self.steps]

    def render(self) -> str:
        """Human-readable trace: each step with its decision and reason."""
        lines = ["Loop Convergence trace", "=" * 60]
        for s in self.steps:
            lines.append(f"[iter {s.iteration}] {s.node:<9} → {s.decision}")
            if s.reason:
                lines.append(f"            reason: {s.reason}")
            for key in ("issues", "deltas"):
                if s.detail.get(key):
                    for item in s.detail[key]:
                        lines.append(f"              - {item}")
        lines.append("=" * 60)
        lines.append(
            f"converged={self.converged}  iterations={self.iterations}  "
            f"verdicts={self.final_verdicts}"
        )
        return "\n".join(lines)


# A steering hook is called after the critic files its issues/deltas, before the
# router runs. The operator receives the live state and may mutate it in place —
# relax/tighten a constraint, drop or add an issue, or edit `pending_deltas` before
# the optimizer applies them — to intercept and steer the loop. Return value ignored.
SteerFn = Callable[[DesignState], None]


def run_loop_traced(
    state: DesignState,
    *,
    moo_tool: MooTool,
    evaluate_fn: Optional[EvaluateFn] = None,
    steer: Optional[SteerFn] = None,
    max_iterations: int = 6,
) -> LoopTrace:
    """Drive the unified loop over the real node functions, recording each step.

    Args:
        state: initial DesignState (constraints, and any seeded ppa_metrics).
        moo_tool: the MOO-engine boundary the optimizer calls (a fake in tests).
        evaluate_fn: the evaluate step; defaults to the real `evaluate_node`
            (ppa_assessor). Inject a scripted one to drive a known verdict path.
        steer: optional human-in-the-loop hook, called after the critic and before
            routing with the live state, so an operator can intercept/steer the loop
            (edit constraints, issues, or `pending_deltas`).
        max_iterations: backstop; also the router's convergence cap.
    """
    evaluate = evaluate_fn or evaluate_node
    router = make_router(max_iterations)
    steps: list[LoopStep] = []
    state = dict(state)  # local copy; we mutate a working state

    def record(node: str, decision: str, reason: str, **detail: object) -> None:
        step = LoopStep(
            iteration=int(state.get("iteration", 0)),
            node=node,
            decision=decision,
            reason=reason,
            detail=dict(detail),
        )
        steps.append(step)
        logger.info("[iter %d] %-9s → %s :: %s", step.iteration, node, decision, reason)

    state.update(seed_node(state))
    record("seed", "initialized", "ensured design space + status before first review")

    while True:
        upd = critic_node(state)
        state.update(upd)
        issues = open_issues(state)
        issue_lines = [f"{i.metric.value}: {i.summary} [{i.severity.value}]" for i in issues]
        record(
            "critic",
            f"{len(issues)} open issue(s), converged={upd.get('converged')}",
            upd.get("analysis", "") or "(no analysis)",
            issues=issue_lines,
        )

        if steer is not None:
            before = len(state.get("pending_deltas", []))
            steer(state)
            record(
                "steer",
                "operator intervened",
                f"human-in-the-loop hook ran (pending_deltas {before} → "
                f"{len(state.get('pending_deltas', []))})",
            )

        route = router(state)
        record(
            "route",
            route,
            (
                "converged / backlog empty / iteration cap → recommend"
                if route == "recommend"
                else "open issues remain → optimize"
            ),
        )
        if route == "recommend":
            upd = recommend_node(state)
            state.update(upd)
            record("recommend", "final design selected", upd.get("final_report", ""))
            break

        upd = optimizer_node(state, moo_tool=moo_tool)
        state.update(upd)
        applied = state.get("applied_deltas", [])
        recent = applied[-3:]
        delta_lines = [
            f"{d.get('kind')} {d.get('target')} :: {d.get('rationale', '')}"
            + (
                f"  (research: {', '.join(d.get('research_refs', []))})"
                if d.get("research_refs")
                else ""
            )
            for d in recent
        ]
        record(
            "optimize",
            f"applied {len(applied)} delta(s) cumulative",
            "applied the critic's edits and re-ran the MOO tool",
            deltas=delta_lines,
        )

        upd = evaluate(state)
        state.update(upd)
        record(
            "evaluate",
            f"verdicts={state.get('ppa_metrics', {}).get('verdicts', {})}",
            "re-scored the design against constraints",
        )

    return LoopTrace(steps=steps, final_state=state)
