# Loop Convergence — a demonstration

This walks through the **Loop Convergence** agentic loop solving a real SoC design
problem end-to-end: you can watch each agent reason, track the log live, and
intercept the loop to steer it. It exists to build confidence that the loop is
*productive* — that it actually converges on a feasible design and tells you *why*.

> Run it yourself:
> ```bash
> python docs/demos/loop_convergence_demo.py            # autonomous solve
> python docs/demos/loop_convergence_demo.py --log      # + live agent log
> python docs/demos/loop_convergence_demo.py --steer    # operator intervention
> ```
> The demo uses a deterministic *surrogate* MOO tool (no botorch/pymoo/API key
> needed) that models how the SoC design space responds to the loop's edits, so
> the solve is reproducible. Swap it for `make_moo_engine_tool()` to drive the
> real MAP-Elites/Bayesian optimizer.

---

## The problem

Design the compute SoC for a delivery-drone perception pipeline (detection +
tracking + VIO at 5 m/s) that must fit a hard budget:

| Constraint | Budget |
|---|---|
| Power | ≤ 5.0 W |
| Latency | ≤ 33.0 ms |
| Cost | ≤ 30.0 $ |

We start from a **naive design that busts power and latency** (FP32 on a small
generic accelerator: 8.0 W / 45 ms) and let the loop find a feasible design.

## The loop

Every iteration runs four agents over one shared `DesignState`:

```
        ┌─────────── critic ──────────┐
        │ finds bottlenecks as typed  │
        │ DesignIssues, proposes typed│
        │ DesignDeltas (the edits)    │
        └──────────────┬──────────────┘
                       │  (optional operator steer hook)
                       ▼
        ┌──────────── router ─────────┐   recommend
        │ converged / backlog empty / ├──────────────▶ done
        │ iteration cap → recommend;  │
        │ else → optimize             │
        └──────────────┬──────────────┘
                       ▼ optimize
        ┌─────────── optimizer ───────┐
        │ applies the deltas to the   │
        │ design space, re-runs the   │
        │ MOO tool                    │
        └──────────────┬──────────────┘
                       ▼
        ┌─────────── evaluate ────────┐
        │ re-scores the design vs the │
        │ constraints → verdicts      │
        └──────────────┬──────────────┘
                       └──────────▶ back to critic
```

- **critic** — reviews `ppa_metrics.verdicts` + the Pareto front + the open-issue
  backlog (and, with an API key, retrieved research) and emits a `CriticVerdict`
  of typed `DesignIssue`s and `DesignDelta`s.
- **optimizer** — applies the deltas as concrete design-space edits and calls the
  MOO tool (a boundary, not the loop body).
- **evaluate** — re-scores the resulting design (the shipped node delegates to the
  same `ppa_assessor` the dispatcher pipeline uses).
- **router** — one stop condition: converged, empty backlog, or the iteration cap.

## 1. Autonomous solve

```
[iter 0] critic    → 2 open issue(s), converged=False
            reason: 2 failing constraint(s); proposed 2 edit(s).
              - power: power constraint failing [critical]
              - latency: latency constraint failing [critical]
[iter 0] route     → optimize
[iter 1] optimize  → applied 2 delta(s) cumulative
              - design_space_edit quantization_dtype  :: relieve power bottleneck
              - design_space_edit hardware.array_rows :: relieve latency bottleneck
[iter 1] evaluate  → verdicts={'power': 'PASS', 'latency': 'PASS', 'cost': 'PASS'}
[iter 1] critic    → 0 open issue(s), converged=True
            reason: No failing constraints; frontier stable — recommend.
[iter 1] route     → recommend
============================================================
converged=True  iterations=1  verdicts={'power': 'PASS', 'latency': 'PASS', 'cost': 'PASS'}

Final design: {'power_watts': 4.6, 'latency_ms': 17.3, 'cost_usd': 26.0}
```

**8.0 W / 45 ms (both failing) → 4.6 W / 17.3 ms, all constraints met, in one
iteration.** The critic identified *both* bottlenecks, chose the right levers
(INT8 quantization for power, a 4× larger MAC array for latency), and the loop
confirmed feasibility and stopped. Every decision carries a reason.

## 2. Tracking the log concurrently

Everything the agents do streams to the `embodied_ai_architect.loop` logger. Attach
a handler (the demo's `--log` flag does this) and you get a live feed you can tail,
pipe to a file, or forward to a dashboard while the loop runs:

```python
import logging, sys
h = logging.StreamHandler(sys.stdout)
h.setFormatter(logging.Formatter("  LOG | %(message)s"))
lg = logging.getLogger("embodied_ai_architect.loop")
lg.setLevel(logging.INFO)
lg.addHandler(h)
```

```
  LOG | [iter 0] critic    → 2 open issue(s), converged=False :: 2 failing constraint(s); proposed 2 edit(s).
  LOG | [iter 0] route     → optimize :: open issues remain → optimize
  LOG | [iter 1] optimize  → applied 2 delta(s) cumulative :: applied the critic's edits and re-ran the MOO tool
  LOG | [iter 1] evaluate  → verdicts={'power': 'PASS', 'latency': 'PASS', 'cost': 'PASS'} :: re-scored...
  LOG | [iter 1] critic    → 0 open issue(s), converged=True :: No failing constraints; frontier stable — recommend.
```

The same data is also captured structurally in the `LoopTrace` (`trace.steps`,
`trace.render()`) for post-hoc review.

## 3. Intercepting and steering the loop

`run_loop_traced(..., steer=hook)` invokes an operator hook **after the critic and
before the optimizer**, with the live `DesignState`. The operator can relax/tighten
a constraint, drop or add an issue, or **edit `pending_deltas`** — injecting a design
move the critic didn't propose. This is how a human keeps the loop on-mission.

In this run the operator makes two interventions:

1. A **late requirement change** tightens the power budget 5.0 → 4.0 W.
2. INT8 alone only reaches 4.6 W, so the critic's power lever is *exhausted* and the
   loop would stall. Watching the log, the operator **injects structured sparsity**
   — a lever the heuristic critic didn't know about — and the loop converges.

```
[iter 0] steer     → operator intervened  (late requirement: tighten 5.0 → 4.0 W)
[iter 1] optimize  → applied 2 delta(s)   (INT8 + bigger MAC array)
[iter 1] evaluate  → verdicts={'power': 'FAIL', 'latency': 'PASS', 'cost': 'PASS'}   # 4.6 W > 4.0 W
[iter 1] critic    → 1 open issue(s)       (power still failing; only lever is INT8, already applied)
[iter 1] steer     → operator intervened  (inject: sparsity=structured)
[iter 2] optimize  → applied 4 delta(s)
              - design_space_edit sparsity :: operator: structured sparsity to close the last 0.6W
[iter 2] evaluate  → verdicts={'power': 'PASS', 'latency': 'PASS', 'cost': 'PASS'}   # 3.8 W < 4.0 W
[iter 2] critic    → 0 open issue(s), converged=True
============================================================
converged=True  iterations=2  verdicts={'power': 'PASS', 'latency': 'PASS', 'cost': 'PASS'}

Final design: {'power_watts': 3.8, 'latency_ms': 17.3, 'cost_usd': 26.0}
```

This is the point of the demo: **the loop and the engineer collaborate.** The loop
does the mechanical search and reports honestly when a lever is exhausted; the human
supplies domain knowledge (a new design move, a requirement change) and the loop
incorporates it and reconverges — under a *harder* 4.0 W budget.

## Why this builds confidence

- **It converges on a feasible design**, not just "runs" — 8 W → 3.8 W under a
  tightened budget, with all constraints met.
- **Every step is explained** — which bottleneck, which edit, why it iterated or
  stopped — so the output is auditable, not a black box.
- **It fails honestly** — when a lever is exhausted it says so instead of pretending;
  that's what makes the steer hook necessary and useful.
- **A human can steer it** at any iteration, so it's a co-pilot, not an oracle.

## What's real vs. illustrative here

- **Real:** the `critic` → `optimizer` → `evaluate` → `router` loop, the typed
  `DesignIssue`/`DesignDelta` flow, the convergence logic, the trace/logger, and the
  steer hook are the shipped Phase-2 code (`graphs/loop_agents.py`,
  `graphs/loop_convergence_graph.py`, `graphs/loop_trace.py`).
- **Illustrative:** the `surrogate_moo` design-space response and `score_knee`
  evaluate are demo stand-ins so the solve is deterministic and dependency-free.
  In production, `make_moo_engine_tool()` drives the real MAP-Elites/Bayesian
  optimizer over the 17-variable joint design space, and `evaluate_node` delegates
  to the real `ppa_assessor`. Wiring these into a CLI command with an LLM critic is
  the remaining Phase 4 work (S10–S11).
