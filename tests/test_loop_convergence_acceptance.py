"""Deterministic acceptance harness for the unified Loop Convergence loop (Phase 2).

Analogous to `gold_standards.py` for the dispatcher pipeline: each LoopScenario
runs the real critic → optimize → evaluate → recommend loop over a fake MOO tool
and a scripted verdict trajectory, and is checked against KNOWN targets
(iterations, convergence, final verdicts, applied-delta count). The `LoopTrace`
render is asserted to capture the reasoning ("why") behind each decision.
"""

from dataclasses import dataclass
from typing import Callable, Optional

from embodied_ai_architect.graphs.design_state import (
    DesignState,
    MetricAxis,
    create_initial_design_state,
)
from embodied_ai_architect.graphs.loop_trace import EvaluateFn, LoopTrace, run_loop_traced
from embodied_ai_architect.graphs.soc_state import DesignConstraints


# ---------------------------------------------------------------------------
# Harness building blocks
# ---------------------------------------------------------------------------


def _initial_state() -> DesignState:
    state = create_initial_design_state(
        "Drone perception at 5 m/s within 5W",
        constraints=DesignConstraints(max_power_watts=5.0, max_latency_ms=33.0),
    )
    state["llm_available"] = False  # deterministic heuristic critic
    state["ppa_metrics"] = {"verdicts": {"power": "FAIL"}}  # seed the first review
    return state


def _fake_moo() -> Callable[[DesignState], dict]:
    """MOO tool whose knee design improves power each call, with a strictly
    improving hypervolume (so the loop converges on the backlog/verdicts, not on a
    hypervolume plateau — that plateau path is exercised separately)."""
    calls = {"n": 0}

    def tool(state: DesignState) -> dict:
        calls["n"] += 1
        power = max(8.0 - calls["n"] * 2.0, 3.0)
        knee = {"objectives": {"power_watts": power, "latency_ms": 20.0}}
        return {
            "knee_point": knee,
            "pareto_points": [knee],
            # strictly increasing -> no plateau-triggered convergence
            "hypervolume_history": state.get("hypervolume_history", []) + [float(calls["n"])],
        }

    return tool


def _scripted_evaluate(pass_at_iter: int) -> EvaluateFn:
    """Verdicts follow a known trajectory: power FAIL until `pass_at_iter`, then PASS."""

    def ev(state: DesignState) -> dict:
        it = int(state.get("iteration", 0))
        verdict = "PASS" if it >= pass_at_iter else "FAIL"
        ppa = dict(state.get("ppa_metrics", {}))
        ppa["verdicts"] = {"power": verdict}
        return {"ppa_metrics": ppa}

    return ev


@dataclass
class LoopScenario:
    name: str
    description: str
    evaluate_fn: Optional[EvaluateFn]
    max_iterations: int
    # Known targets:
    expected_iterations: int
    expected_converged: bool
    expected_verdicts: dict
    expected_applied_deltas: int


def _run(scenario: LoopScenario) -> LoopTrace:
    return run_loop_traced(
        _initial_state(),
        moo_tool=_fake_moo(),
        evaluate_fn=scenario.evaluate_fn,
        max_iterations=scenario.max_iterations,
    )


# ---------------------------------------------------------------------------
# Scenarios with known output targets
# ---------------------------------------------------------------------------

CONVERGES = LoopScenario(
    name="converges_after_fixing_power",
    description="power fails for 2 iterations, then the fix lands and the loop converges",
    evaluate_fn=_scripted_evaluate(pass_at_iter=2),
    max_iterations=6,
    expected_iterations=2,
    expected_converged=True,
    expected_verdicts={"power": "PASS"},
    expected_applied_deltas=2,
)

HITS_CAP = LoopScenario(
    name="hits_iteration_cap",
    description="power never passes; the loop stops at the iteration cap, not converged",
    evaluate_fn=_scripted_evaluate(pass_at_iter=999),
    max_iterations=3,
    expected_iterations=3,
    expected_converged=False,
    expected_verdicts={"power": "FAIL"},
    expected_applied_deltas=3,
)


def test_scenario_converges_after_fix() -> None:
    t = _run(CONVERGES)
    assert t.iterations == CONVERGES.expected_iterations
    assert t.converged is CONVERGES.expected_converged
    assert t.final_verdicts == CONVERGES.expected_verdicts
    assert len(t.final_state.get("applied_deltas", [])) == CONVERGES.expected_applied_deltas
    # the critic flagged power as the bottleneck it was optimizing
    critic_steps = [s for s in t.steps if s.node == "critic"]
    assert any(
        MetricAxis.POWER.value in i for s in critic_steps for i in s.detail.get("issues", [])
    )
    # the loop ended by recommending, not by hitting the cap
    assert t.steps[-1].node == "recommend"


def test_scenario_converges_on_hypervolume_plateau() -> None:
    """When the MOO frontier stops improving (constant hypervolume) the loop
    converges even though the verdict still fails — the plateau is a valid stop
    signal (`has_converged`)."""

    def flat_moo(state: DesignState) -> dict:
        knee = {"objectives": {"power_watts": 6.0, "latency_ms": 20.0}}
        return {
            "knee_point": knee,
            "pareto_points": [knee],
            "hypervolume_history": state.get("hypervolume_history", []) + [1.0],  # flat
        }

    t = run_loop_traced(
        _initial_state(),
        moo_tool=flat_moo,
        evaluate_fn=_scripted_evaluate(pass_at_iter=999),  # power never passes
        max_iterations=6,
    )
    # Stopped well before the cap, via the hypervolume-plateau signal.
    assert t.iterations < 6
    assert t.steps[-1].node == "recommend"


def test_scenario_hits_iteration_cap() -> None:
    t = _run(HITS_CAP)
    assert t.iterations == HITS_CAP.expected_iterations
    assert t.converged is HITS_CAP.expected_converged
    assert t.final_verdicts == HITS_CAP.expected_verdicts
    assert len(t.final_state.get("applied_deltas", [])) == HITS_CAP.expected_applied_deltas


def test_real_evaluate_integration_terminates() -> None:
    """With the real evaluate_node (ppa_assessor), the loop still terminates and
    produces verdicts — proving the S7 assessor integrates into the traced loop."""
    t = run_loop_traced(_initial_state(), moo_tool=_fake_moo(), evaluate_fn=None, max_iterations=4)
    assert t.steps[-1].node == "recommend"
    assert t.iterations <= 4
    assert isinstance(t.final_verdicts, dict)


def test_operator_can_steer_the_loop() -> None:
    """The steer hook lets an operator intercept the loop: here it force-converges
    by clearing the backlog, so the loop recommends immediately at iteration 0."""

    def force_converge(state: DesignState) -> None:
        for issue in state.get("open_issues", []):
            issue["status"] = "wontfix"  # operator accepts the trade-off
        state["pending_deltas"] = []

    t = run_loop_traced(
        _initial_state(),
        moo_tool=_fake_moo(),
        evaluate_fn=_scripted_evaluate(pass_at_iter=999),  # would never converge on its own
        steer=force_converge,
        max_iterations=6,
    )
    assert t.iterations == 0  # operator short-circuited it
    assert any(s.node == "steer" for s in t.steps)
    assert t.steps[-1].node == "recommend"


def test_trace_captures_the_reasoning() -> None:
    """The trace render must surface WHY the loop made its decisions."""
    rendered = _run(CONVERGES).render()
    assert "power" in rendered  # the bottleneck the critic reasoned about
    assert "reason:" in rendered  # every decision carries a reason
    assert "→ recommend" in rendered or "recommend" in rendered
    assert "converged=True" in rendered
    # the optimizer's edits and their rationale appear
    assert "applied" in rendered
