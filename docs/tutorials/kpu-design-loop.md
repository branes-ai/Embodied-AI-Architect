# Tutorial: The KPU Design Loop

This tutorial walks through designing a custom KPU SoC end-to-end using
the dual-loop architecture: the outer SoC optimization loop and the inner
KPU micro-architecture convergence loop. By the end you will know:

- How to enable the KPU pipeline on a `SoCDesignRunner` session
- How the architect inspects and steers KPU sizing at every stage
- How to override KPU parameters at plan review time
- How the catalog strategies and the RTL feedback loop interact
- How to drill into specific KPU subsystems when something goes wrong

The full design arc was implemented across issues #29–#35; this tutorial
ties them together.

## Prerequisites

```bash
.venv/bin/pip install -e ".[dev,api]"
```

For RTL synthesis you'll also want Yosys installed (the EDA toolchain
falls back to mock synthesis if Yosys is missing, so you don't strictly
need it for a smoke run).

## Part 1: A Minimal KPU Run

The fastest path to a populated KPU pipeline is `SoCDesignRunner.run()`
with `rtl_enabled=True` and a static plan that schedules the KPU
specialists in dependency order.

```python
from embodied_ai_architect.graphs.soc_runner import SoCDesignRunner
from embodied_ai_architect.graphs.soc_state import DesignConstraints

KPU_PLAN = [
    {"id": "t1", "name": "Analyze workload", "agent": "workload_analyzer", "dependencies": []},
    {"id": "t2", "name": "Explore hardware", "agent": "hw_explorer", "dependencies": ["t1"]},
    {"id": "t3", "name": "Compose architecture", "agent": "architecture_composer", "dependencies": ["t2"]},
    {"id": "t4", "name": "Configure KPU", "agent": "kpu_configurator", "dependencies": ["t3"]},
    {"id": "t5", "name": "Validate floorplan", "agent": "floorplan_validator", "dependencies": ["t4"]},
    {"id": "t6", "name": "Validate bandwidth", "agent": "bandwidth_validator", "dependencies": ["t4"]},
    {"id": "t7", "name": "Generate RTL", "agent": "rtl_generator", "dependencies": ["t5", "t6"]},
    {"id": "t8", "name": "Assess RTL PPA", "agent": "rtl_ppa_assessor", "dependencies": ["t7"]},
    {"id": "t9", "name": "Assess PPA", "agent": "ppa_assessor", "dependencies": ["t8"]},
    {"id": "t10", "name": "Review design", "agent": "critic", "dependencies": ["t9"]},
    {"id": "t11", "name": "Generate report", "agent": "report_generator", "dependencies": ["t10"]},
]

runner = SoCDesignRunner(static_plan=KPU_PLAN)
state = runner.run(
    goal="Design a KPU SoC for delivery drone perception",
    constraints=DesignConstraints(
        max_power_watts=5.0,
        max_latency_ms=33.3,
        max_cost_usd=30.0,
        max_area_mm2=100.0,
    ),
    use_case="delivery_drone",
    platform="drone",
    rtl_enabled=True,            # turn on KPU + floorplan + bandwidth + RTL
    rtl_area_feedback=False,     # we'll enable this in Part 4
)

print(f"KPU: {state['kpu_config']['name']}")
print(f"Floorplan feasible: {state['floorplan_estimate']['feasible']}")
print(f"Bandwidth balanced: {state['bandwidth_match']['balanced']}")
print(f"History entries:    {len(state['kpu_optimization_history'])}")
```

The pipeline runs the KPU specialists in dependency order, the design
optimizer iterates if any constraint fails, and the final state has
everything the architect needs.

## Part 2: Inspecting the KPU at Three Levels

You have three views into the KPU config — pick the right one for the
task at hand.

### CLI: `branes session show`

```bash
.venv/bin/branes session show --latest
```

Shows the architect-facing summary:

```
KPU Configuration — swkpu-delivery-drone at 28nm
  Grid:        3 x 3 checkerboard
  Systolic:    16 x 16 @ 500.0MHz
  SRAM:        L1=32KB | L2=256KB | L3=512KB
  NoC:         mesh_2d, 256-bit, 1000.0MHz
  DRAM:        LPDDR4X, 2 controllers, 4.0GB

KPU Convergence History (4 entries)
  iter 0 [kpu_configurator] — swkpu-delivery-drone
    systolic=16x16  |  L2=256KB  |  NoC=256b
  iter 0 [floorplan_validator]
    pitch=PASS/area=48.2mm²
  iter 0 [bandwidth_validator]
    BW=PASS
  ...
```

### REST API

The `/kpu` endpoint serves the full snapshot to programmatic consumers:

```bash
curl http://localhost:8000/api/sessions/<id>/kpu | jq
```

Returns `config + floorplan + bandwidth` in one response. For just the
slackness view (no full config), use `/kpu-slackness`.

### Programmatic snapshot

```python
from embodied_ai_architect.graphs.optimization_review import (
    build_optimization_review_snapshot,
)

snap = build_optimization_review_snapshot(state)
print(snap.kpu_floorplan.pitch_matched)
print(snap.kpu_floorplan.area_utilization_pct)
print(snap.kpu_bandwidth.bottleneck_link)
for entry in snap.kpu_history:
    print(entry["source"], entry.get("compute_array"))
```

## Part 3: Steering at Plan Review (Architect Overrides)

The architect can override any KPU parameter before dispatch starts using
`PlanReviewInput.kpu_overrides` with dotted-path keys. This is the
intervention point — the heuristic configurator gives you a starting
point, but the architect knows the workload's quirks.

```python
from embodied_ai_architect.graphs.review import PlanReviewInput, ReviewDecision

# Run interactive mode
runner = SoCDesignRunner(
    static_plan=KPU_PLAN,
    human_review=True,
    optimization_review=True,
)
status, state = runner.start(
    goal=...,
    constraints=...,
    rtl_enabled=True,
)
# status == "review_plan"
# state["review_snapshot"]["kpu_preview"] has the heuristic preview

# Override the systolic array dimensions and the NoC link width
review = PlanReviewInput(
    decision=ReviewDecision.MODIFY,
    kpu_overrides={
        "compute_tile.array_rows": 8,      # smaller MAC grid (less area)
        "compute_tile.array_cols": 8,
        "noc.link_width_bits": 512,        # wider NoC (more BW, more area)
        "dram.num_controllers": 4,         # more DRAM channels
    },
)
status, state = runner.step(review_input=review.model_dump())
```

The overrides survive across review passes (additive merge) and are
applied on top of the heuristic when `kpu_configurator` runs during
dispatch. Architect's choices win over the auto-sizer.

## Part 4: The Inner Loop — How It Converges

### The four KPU specialists

| Specialist | Reads | Writes |
|---|---|---|
| `kpu_configurator` | `workload_profile`, `constraints`, `kpu_config_overrides` | `kpu_config` |
| `floorplan_validator` | `kpu_config`, `constraints` | `floorplan_estimate` |
| `bandwidth_validator` | `kpu_config`, `workload_profile` | `bandwidth_match` |
| `kpu_optimizer` | `floorplan_estimate`, `bandwidth_match` | adjusted `kpu_config` |

Every one of them appends a snapshot to `kpu_optimization_history` so
the architect can replay the full trail (issue #34).

### When the outer optimizer touches `kpu_config`

When `design_optimizer` picks a KPU strategy from #32 (e.g.
`reduce_systolic_array`), it mutates `kpu_config` and clears
`floorplan_estimate / bandwidth_match`. On the next dispatch iteration,
`dispatch_node` notices that `kpu_config` exists and re-runs
`floorplan_validator` + `bandwidth_validator` (issue #35) so the
slackness views stay honest. This is the dual-loop coupling — the
outer optimizer's catalog strategies trigger inner-loop re-validation
automatically.

### RTL → KPU area feedback (issue #31)

After RTL synthesis, the `rtl_area_feedback` specialist compares
synthesis area against the floorplan estimate. If synthesis exceeds
floorplan × tolerance (default 1.1), it re-runs the KPU sizing loop
with the synthesis area as a tightened budget — bounded at 3 iterations.
Enable it on session creation:

```python
runner.run(..., rtl_enabled=True, rtl_area_feedback=True)
```

Or schedule a separate `rtl_area_feedback` task in the plan after
`rtl_ppa_assessor`. The default plan does NOT schedule it — opt-in.

## Part 5: Drilling Into Bottlenecks

When something fails, use `/architect-drill` with a KPU target:

```text
/architect-drill bandwidth_chain
```

Returns the per-link waterfall, identifies the bottleneck link, and
suggests the right catalog strategy:

```text
BANDWIDTH CHAIN — KPU Memory Hierarchy
Total: 12.8 GB/s demand | 25.6 GB/s DRAM supply

Link              Demand    Supply    Util    Bound?  Fix
DRAM → L3         12.8 GB/s  25.6 GB/s  50%     No    —
L3 → L2            9.0 GB/s  16.4 GB/s  55%     No    —
L2 → L1            4.5 GB/s   4.8 GB/s  94%     YES   add_sram_banks (L2)
L1 → compute       1.4 GB/s   3.2 GB/s  43%     No    —

Root cause: L2→L1 bandwidth limited by 2 streamers × 2.4 GB/s each
Options:
  A. add_sram_banks — bump l2_num_banks +2 (~0.3 mm² area, removes bottleneck)
  B. add streamers (no catalog strategy yet — flag as gap)
  C. reduce compute throughput (shrink array, matches current BW)
```

Other targets: `kpu`, `systolic_array`, `sram_hierarchy`, `noc`.

## Part 6: Common Pitfalls

**`kpu_config` is empty after the run.** The KPU specialists were never
scheduled. Make sure your static plan includes `kpu_configurator` and
that `rtl_enabled=True` was passed to `runner.run()`. Non-RTL pipelines
intentionally skip the KPU specialists.

**`floorplan_estimate` empty even though `kpu_optimization_history` has
entries.** This was the issue #35 bug — pre-fix, the LangGraph
`dispatch_node` and `optimize_node` were dropping these fields across
state-merge boundaries. If you see this on current main, it's a
regression — file an issue.

**Architect overrides not applied.** Check that
`PlanReviewInput.kpu_overrides` uses **dotted-path keys**, not nested
dicts. `{"compute_tile.array_rows": 8}` works;
`{"compute_tile": {"array_rows": 8}}` does not. The helper
`apply_kpu_overrides` walks dotted paths and silently ignores unknown
keys, so a typo just becomes a no-op.

**Bandwidth always saturated.** The default workload profile assumes
moderate arithmetic intensity (~10 FLOPs/byte). For memory-bound
workloads (transformers, attention) you may need to widen the NoC and
add SRAM banks aggressively, OR drop to `add_sram_banks` directly via
the catalog. See `docs/designs/kpu-optimization-knobs.md` for the
limitations of the current strategy set.

**RTL synthesis hangs.** Yosys can be slow on the L3 tile. The toolchain
has a 30s timeout per module and falls back to mock synthesis on timeout
— check the logs for `Yosys timed out after 30s`. Mock synthesis is fine
for development; for real area numbers install Yosys 0.40+ and let it
run longer.

## Further Reading

- `docs/interactive-design-review.md` — the full plan-review and
  optimization-review surface area, with the new KPU Review section
- `docs/designs/kpu-optimization-knobs.md` — design notes on the KPU
  catalog and the planned redesign in epic #83
- `docs/plans/architect-workflows.md` — Workflow 6 (KPU SoC) walks
  through the architect's session-by-session pattern
- `tests/test_kpu_integration.py::TestKPUFullPipeline` — the contract
  tests for the full pipeline; the simplest reference for what state
  fields should be populated after a run
- `examples/demo_kpu_rtl.py` — runnable demo using `SoCDesignRunner`
  that writes a saved session inspectable via `branes session show`
- Issues #29–#35 — the integration arc (overrides → slackness → RTL
  feedback → catalog strategies → drill → convergence history →
  integration test)
