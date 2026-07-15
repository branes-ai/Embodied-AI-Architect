#!/usr/bin/env python
"""Loop Convergence demo — an agentic loop solving a real SoC design problem.

Problem
-------
Design the compute SoC for a delivery-drone perception pipeline that must fit:

    power   <= 5.0 W
    latency <= 33.0 ms
    cost    <= 30.0 $

We start from a naive design that busts the power and latency budgets, and let the
loop converge to a feasible design. At each iteration:

    critic    finds the bottlenecks (as typed DesignIssues) and proposes edits
              (typed DesignDeltas)
    optimizer applies the edits to the design space and re-runs the MOO tool
    evaluate  re-scores the resulting design against the constraints
    router    decides whether to iterate again or recommend

Everything the agents do is streamed to the `embodied_ai_architect.loop` logger,
so you can watch the reasoning live (see `--log`), and an operator can intercept
and steer the loop via a hook (see `steered_run`).

Run
---
    python docs/demos/loop_convergence_demo.py          # autonomous solve
    python docs/demos/loop_convergence_demo.py --log    # + live agent log
    python docs/demos/loop_convergence_demo.py --steer  # operator intervention

This demo uses a deterministic *surrogate* MOO tool (no botorch/pymoo/API key
needed) that models how the SoC design space responds to the loop's edits, so the
solve is reproducible. Swap `surrogate_moo` for
`graphs.loop_convergence_graph.make_moo_engine_tool()` to drive the real
MAP-Elites/Bayesian optimizer instead.
"""

from __future__ import annotations

import argparse
import logging
import sys

from embodied_ai_architect.graphs.design_state import (
    DeltaKind,
    DesignDelta,
    DesignState,
    create_initial_design_state,
)
from embodied_ai_architect.graphs.loop_trace import run_loop_traced
from embodied_ai_architect.graphs.soc_state import DesignConstraints

# --- The problem ---------------------------------------------------------------

CONSTRAINTS = DesignConstraints(max_power_watts=5.0, max_latency_ms=33.0, max_cost_usd=30.0)


def initial_state() -> DesignState:
    """A naive drone-perception SoC that busts power and latency."""
    state = create_initial_design_state(
        "Delivery-drone perception SoC: detection + tracking + VIO at 5 m/s",
        constraints=CONSTRAINTS,
    )
    state["platform"] = "drone"
    state["llm_available"] = False  # deterministic heuristic critic (no API key)
    # Naive starting point: FP32 on a big generic accelerator -> over budget.
    state["ppa_metrics"] = {"verdicts": {"power": "FAIL", "latency": "FAIL"}}
    return state


# --- Surrogate MOO tool: models how the design space responds to the loop's edits


def surrogate_moo(state: DesignState) -> dict:
    """Return a knee design whose PPA reflects the edits the loop has applied.

    Baseline FP32 / small systolic array: 8.0 W, 45 ms, $26. The loop's edits move
    it: INT8 quantization cuts power and latency; a larger MAC array cuts latency.
    """
    space = state.get("design_space_config", {})
    hw = space.get("hardware", {})

    power, latency, cost = 8.0, 45.0, 26.0
    if space.get("quantization_dtype") == "int8":
        power *= 0.55  # INT8 MACs are far cheaper
        latency *= 0.7
    if hw.get("array_rows") == 32:
        latency *= 0.55  # 4x MACs -> lower latency
        power *= 1.05  # slightly more power for the bigger array
    if space.get("sparsity") == "structured":
        power *= 0.83  # structured sparsity skips zero MACs (operator's lever)

    knee = {
        "objectives": {
            "power_watts": round(power, 1),
            "latency_ms": round(latency, 1),
            "cost_usd": round(cost, 1),
        },
        "design_params": {
            "quantization_dtype": space.get("quantization_dtype", "fp32"),
            "array_rows": hw.get("array_rows", 8),
        },
    }
    return {
        "knee_point": knee,
        "pareto_points": [knee],
        # improving hypervolume so convergence is driven by the verdicts, not a plateau
        "hypervolume_history": state.get("hypervolume_history", [])
        + [float(len(state.get("hypervolume_history", [])) + 1)],
    }


def score_knee(state: DesignState) -> dict:
    """Evaluate step: score the current knee design against the constraints.

    (The shipped `evaluate_node` delegates to the architecture-level `ppa_assessor`;
    here we score the MOO knee point directly so the demo narrative tracks the
    design the loop is actually optimizing.)
    """
    point = state.get("knee_point", {})
    obj = point.get("objectives", {})
    c = state.get("constraints", {})
    verdicts = {}
    for name, key, limit in (
        ("power", "power_watts", c.get("max_power_watts")),
        ("latency", "latency_ms", c.get("max_latency_ms")),
        ("cost", "cost_usd", c.get("max_cost_usd")),
    ):
        val = obj.get(key)
        if val is not None and limit is not None:
            verdicts[name] = "PASS" if float(val) <= float(limit) else "FAIL"
    ppa = dict(state.get("ppa_metrics", {}))
    ppa.update({k: obj.get(k) for k in ("power_watts", "latency_ms", "cost_usd")})
    ppa["verdicts"] = verdicts
    return {"ppa_metrics": ppa}


# --- Runs ----------------------------------------------------------------------


def autonomous_run() -> None:
    print("\n=== Autonomous solve: drone SoC, <=5W / <=33ms / <=$30 ===\n")
    trace = run_loop_traced(initial_state(), moo_tool=surrogate_moo, evaluate_fn=score_knee)
    print(trace.render())
    final = trace.final_state.get("knee_point", {}).get("objectives", {})
    print(f"\nFinal design: {final}")
    print(f"Converged: {trace.converged} in {trace.iterations} iteration(s)\n")


def steered_run() -> None:
    """Two operator interventions show steering is *productive*:

    1. A late requirement change tightens the power budget 5.0 -> 4.0 W.
    2. INT8 alone only reaches 4.6 W, so the critic's power lever is exhausted and
       the loop would stall. The operator, watching the log, injects a design move
       the heuristic critic didn't know about — structured sparsity — which the
       optimizer applies and the loop then converges under 4.0 W.
    """
    print("\n=== Steered solve: operator tightens budget, then unblocks the loop ===\n")

    tightened = {"done": False}

    def steer(state: DesignState) -> None:
        if not tightened["done"]:
            tightened["done"] = True
            state.setdefault("constraints", {})["max_power_watts"] = 4.0
            print("   >>> operator: late requirement — tighten max_power_watts 5.0 -> 4.0 W\n")
            return
        # Loop stuck on power with INT8 already applied? Inject a fresh lever.
        verdicts = state.get("ppa_metrics", {}).get("verdicts", {})
        space = state.get("design_space_config", {})
        if (
            verdicts.get("power") == "FAIL"
            and space.get("quantization_dtype") == "int8"
            and space.get("sparsity") != "structured"
        ):
            print("   >>> operator: INT8 exhausted at 4.6W — injecting structured sparsity\n")
            state.setdefault("pending_deltas", []).append(
                DesignDelta(
                    kind=DeltaKind.DESIGN_SPACE_EDIT,
                    target="sparsity",
                    change={"value": "structured"},
                    rationale="operator: structured sparsity to close the last 0.6W",
                    proposed_by="operator",
                ).model_dump(mode="json")
            )

    trace = run_loop_traced(
        initial_state(), moo_tool=surrogate_moo, evaluate_fn=score_knee, steer=steer
    )
    print(trace.render())
    final = trace.final_state.get("knee_point", {}).get("objectives", {})
    print(f"\nFinal design: {final}  (met the tightened 4.0W budget via operator's sparsity move)")
    print(f"Converged: {trace.converged} in {trace.iterations} iteration(s)\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", action="store_true", help="stream the agent log live")
    ap.add_argument("--steer", action="store_true", help="run the operator-steered variant")
    args = ap.parse_args()

    if args.log:
        # Concurrent tracking: everything the agents do streams to this logger.
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("  LOG | %(message)s"))
        loop_logger = logging.getLogger("embodied_ai_architect.loop")
        loop_logger.setLevel(logging.INFO)
        loop_logger.addHandler(h)

    if args.steer:
        steered_run()
    else:
        autonomous_run()


if __name__ == "__main__":
    main()
