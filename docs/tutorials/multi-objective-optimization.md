# Tutorial: Multi-Objective Optimization in the SoC Design Pipeline

This tutorial walks through using the integrated MOO pipeline end-to-end:
running a design session that produces a Pareto frontier, inspecting the
results, and using the sensitivity data to guide bottleneck resolution.

By the end you will know:

- How `moo_explorer` fits into the LangGraph design pipeline
- How to read the Pareto frontier, sensitivity, and atlas coverage
- How to use MOO data to make a steering decision
- How to configure MOO budget vs. wall time for your environment

## Prerequisites

```bash
.venv/bin/pip install -e ".[dev,optimization,api]"
```

The `[optimization]` extra installs `botorch`, `gpytorch`, and `pymoo` —
required for the Bayesian and NSGA-III layers. Without it the pipeline
falls back to MAP-Elites only.

## Part 1: A Minimal End-to-End Run

The fastest path to a populated frontier is `SoCDesignRunner.run()` with a
static plan. The default planner already includes `moo_explorer` after
`hw_explorer`, so any session you start through `branes design` will exercise
the MOO pipeline.

```python
from embodied_ai_architect.graphs.soc_runner import SoCDesignRunner
from embodied_ai_architect.graphs.soc_state import DesignConstraints

PLAN_WITH_MOO = [
    {"id": "t1", "name": "Analyze workload", "agent": "workload_analyzer", "dependencies": []},
    {"id": "t2", "name": "Explore hardware", "agent": "hw_explorer", "dependencies": ["t1"]},
    {
        "id": "t3",
        "name": "Compose architecture",
        "agent": "architecture_composer",
        "dependencies": ["t2"],
    },
    {
        "id": "t4",
        "name": "Explore Pareto frontier",
        "agent": "moo_explorer",
        "dependencies": ["t2"],
        # fast_mode → MAP-Elites only with a moderate budget; production
        # default (no metadata) auto-selects layers (MAP-Elites + BO/NSGA-III).
        "metadata": {"fast_mode": True},
    },
    {"id": "t5", "name": "Assess PPA", "agent": "ppa_assessor", "dependencies": ["t3", "t4"]},
    {"id": "t6", "name": "Review design", "agent": "critic", "dependencies": ["t5"]},
    {"id": "t7", "name": "Generate report", "agent": "report_generator", "dependencies": ["t6"]},
]

runner = SoCDesignRunner(static_plan=PLAN_WITH_MOO)
state = runner.run(
    goal="Design an SoC for warehouse AMR with detection + tracking",
    constraints=DesignConstraints(
        max_power_watts=15.0,
        max_latency_ms=50.0,
        max_cost_usd=100.0,
    ),
    use_case="warehouse_amr",
    platform="amr",
)

print(f"Pareto points: {len(state['pareto_points'])}")
print(f"Hypervolume:   {state['moo_results']['hypervolume']:.2f}")
print(f"Layers used:   {state['moo_results']['layers_used']}")
```

That's it — `moo_explorer` runs as a normal task in the dispatch DAG and
its results land in the state alongside `ppa_metrics` and `selected_architecture`.

## Part 2: Inspecting the Frontier

After a run, you have several entry points:

### CLI: `branes session show`

```bash
.venv/bin/branes session show --latest
```

Shows the MOO summary block:

```
MOO Summary
  Pareto front:    12 non-dominated designs
  Accumulated:     27 points (across iterations)
  Total evals:     6,656
  Hypervolume:     613.40
  Layers used:     map_elites, bayesian
```

### REST API

Start the API server and hit the Pareto endpoint:

```bash
.venv/bin/uvicorn embodied_ai_architect.api.server:app --reload
```

```bash
curl http://localhost:8000/api/sessions/<id>/pareto | jq
```

Returns the full point list, the indices of non-dominated points, the
knee-point index, and the per-iteration `frontier_history`.

The sensitivity endpoint (issue #24) gives you the ranked-by-impact view:

```bash
curl http://localhost:8000/api/sessions/<id>/sensitivity | jq
```

```json
{
  "entries": [
    {"variable": "clock_mhz", "max_impact": 0.90,
     "impacts": {"power_watts": 0.90, "latency_ms": 0.80}},
    {"variable": "process_nm", "max_impact": 0.40,
     "impacts": {"power_watts": 0.40, "latency_ms": 0.30}}
  ],
  "objectives": ["power_watts", "latency_ms"]
}
```

### Programmatic snapshot

```python
from embodied_ai_architect.graphs.optimization_review import (
    build_optimization_review_snapshot,
)

snap = build_optimization_review_snapshot(state)
print(snap.pareto_front_size)            # 12
print(snap.hypervolume)                  # 613.4
print(snap.moo_summary["layers_used"])   # ['map_elites', 'bayesian']
print(snap.moo_summary["total_evaluations"])
```

## Part 3: Using MOO Data to Guide Optimization

The integrated `design_optimizer` (issue #25) reads the BO sensitivity to
pick which knob to turn when a constraint fails. Here's the contract:

```python
# In src/embodied_ai_architect/graphs/optimizer.py

# When a strategy targets a MOO design space variable, its score is boosted by
#   1.0 + sensitivity[variable][failing_objective]
#
# So if the failing objective is power_watts and clock_mhz has impact 0.90,
# then clock_scaling (which targets clock_mhz) gets a 1.9x boost over
# alternatives that don't target a high-impact knob.
```

The architect skills (`/architect-loop`, `/architect-assess`) consume the same
sensitivity data. Without MOO, the optimizer falls back to a static reduction-
factor heuristic — which works, but is blind to *which* knobs actually matter
in this design space.

## Part 4: Configuration — Budget vs. Wall Time

The MOO budget directly controls wall time. Three configuration points matter:

### 1. `enable_moo` on design state — the kill switch

```python
state = create_initial_soc_state(..., enable_moo=False)
```

Setting `enable_moo=False` causes the planner to **strip `moo_explorer` tasks**
from the plan even if the static plan includes it. Use this for fast iterations
or when MOO is genuinely overkill (e.g., feasibility checks).

### 2. `metadata.fast_mode` on the moo_explorer task

```python
{"agent": "moo_explorer", "metadata": {"fast_mode": True}, ...}
```

`fast_mode` skips the Bayesian layer entirely (`layers="map_elites"`) and uses
a moderate MAP-Elites budget (~2,500 evals). This is the right setting for
**tests and CI** — typically 10-30 seconds vs minutes for the full pipeline.

Production runs (no `fast_mode`) use `layers="auto"`, which selects MAP-Elites
+ BO for ≤4 objectives or MAP-Elites + NSGA-III for >4. Budget: ~6,500 evals
plus the BO loop.

### 3. `OptimizationConfig` directly (advanced)

If you call the engine yourself (e.g., for the joint design space loop in
`graphs/optimization_loop.py`), you can dial individual knobs:

```python
from embodied_ai_architect.graphs.moo.engine import OptimizationConfig
from embodied_ai_architect.graphs.moo.map_elites import MAPElitesConfig
from embodied_ai_architect.graphs.moo.bayesian_opt import BayesianOptConfig

config = OptimizationConfig(
    layers="auto",
    map_elites=MAPElitesConfig(
        n_iterations=100,
        batch_size=64,
        initial_population=256,
    ),
    bayesian=BayesianOptConfig(
        n_initial=20,
        n_iterations=50,
        batch_size=4,
        # Acquisition optimizer knobs (lower = faster, less optimal)
        acq_num_restarts=5,
        acq_raw_samples=256,
    ),
    max_workers=8,
)
```

### Picking sensible defaults

| Goal | `fast_mode` | Layers | Wall time (CI runner) | When |
|---|---|---|---|---|
| Test / CI smoke | True | map_elites | ~30s | Verify data flows |
| Quick "what does the frontier look like?" | True | map_elites | ~30s | Architect exploration |
| Production design session | False | auto | ~5–15 min | Real architectural decisions |
| Many objectives (>4) | False | auto (→ NSGA-III) | ~10–30 min | SWaP-C with ≥6 objectives |

## Part 5: Common Pitfalls

**0 feasible designs in the frontier.** The constraints are too tight or the
budget is too small. Loosen one constraint at a time and re-run; check
`moo_results.atlas.coverage` — if it's <30% the search hasn't explored enough.

**`pareto_points` is empty after a session.** Either `moo_explorer` was stripped
(`enable_moo=False`) or it wasn't scheduled. Look at
`state["task_graph"]["nodes"]` for an `agent: moo_explorer` entry and check its
status — `completed` means it ran but found nothing; missing means it wasn't
scheduled.

**Sensitivity is empty.** Only the BO layer produces sensitivity. If you ran
with `fast_mode=True` (MAP-Elites only) or with >4 objectives (auto-selects
NSGA-III), there's no sensitivity data — the optimizer falls back to greedy
reduction-factor selection. Run without `fast_mode` to enable BO.

**Frontier shrinks across iterations.** This is allowed in principle (new
points can dominate many old ones). The accumulated count is monotonic in
*coverage* but not in *cardinality*. If you need to verify monotonic
improvement, compare hypervolume across `pareto_frontier_history` entries
rather than counting points.

## Further Reading

- `docs/interactive-design-review.md` — human steering at plan/optimization stages
- `docs/multi-objective-optimization-research.md` — design rationale for the 3-layer pipeline
- `.claude/commands/architect-loop.md` — bottleneck-hunting workflow that consumes sensitivity
- `tests/test_moo_integration.py` — end-to-end contract tests for the data flow
- Issues #21–#27 — the integration arc (wire output → schedule → accumulate → sensitivity → optimizer-aware → integration test)
