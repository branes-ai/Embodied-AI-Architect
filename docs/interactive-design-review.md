# Interactive SoC Design with Human-in-the-Loop Review

The SoC design pipeline supports two human review stages that make the
design process transparent and steerable. This is the "shift left"
approach: getting human insight into the design early — at plan time and
during optimization — rather than just validating at the end.

## Architecture

```
User Prompt
    ↓
┌─────────┐     ┌──────────────┐     ┌──────────┐     ┌──────────┐
│ Planner │ ──> │ PLAN REVIEW  │ ──> │ Dispatch │ ──> │ Evaluate │
└─────────┘     │  (human)     │     └──────────┘     └────┬─────┘
                └──────────────┘                           │
                                                    ┌──────┴──────┐
                                            ┌───────┤ OPT REVIEW  │
                                            │       │  (human)    │
                                            ▼       └─────────────┘
                                       ┌──────────┐        │
                                       │ Optimize │ <──────┘
                                       └──────────┘
                                            │
                                            ▼
                                       ┌──────────┐
                                       │ Report   │ > END
                                       └──────────┘
```

## Plan Review

After the planner decomposes a goal into a task graph (DAG), the human
architect can inspect and modify the plan before execution begins.

### What you see

- **Goal interpretation** — how the planner understood your objective
- **Constraint summary** — Power: 5.0W | Latency: 33.3ms | Cost: $30
- **Task graph** — ASCII tree showing tasks, agents, and dependencies
- **Execution schedule** — parallel groups (which tasks run concurrently)
- **Available agents** — all specialist agents you can assign to tasks

### What you can do

| Action | Example |
|--------|---------|
| **Approve** | Accept the plan as-is |
| **Add tasks** | Insert a safety check before the report |
| **Remove tasks** | Remove the critic if time is tight |
| **Reassign agents** | Swap hw_explorer for a different specialist |
| **Reorder** | Change dependencies to parallelize more |
| **Override constraints** | Relax power to 8W to explore the space |
| **Add notes** | Record reasoning for the design history |

### Validation

After edits, the system validates:
- No cycles in the dependency graph
- All agents are registered
- No dangling dependencies
- At least one task exists

## Optimization Review

At each optimization iteration, you see the full state of the design —
not just pass/fail, but *how close* each constraint is, *how metrics
evolved*, and *what strategies are available*.

### What you see

**Constraint slackness** — the key table:

```
Constraint       Target     Actual   Margin  Verdict      Trend
──────────── ────────── ────────── ──────── ──────── ──────────
power              5.0W       6.2W   -24.0%     FAIL improving ^
latency          33.3ms     28.5ms   +14.4%     PASS improving ^
cost            30.0USD    35.0USD   -16.7%     FAIL improving ^
```

- **Margin** — positive means slack (within budget), negative means violated
- **Binding** — marked with `!` when margin < 5% (constraint almost active)
- **Trend** — improving/worsening/stable over recent iterations

**Optimization trajectory** — how each metric evolved:

```
iter 0: P=8.0W, L=35.0ms, C=$40  [0/3 PASS]
iter 1: P=7.0W, L=30.0ms, C=$37  [1/3 PASS]
iter 2: P=6.2W, L=28.5ms, C=$35  [1/3 PASS]
```

**Strategy analysis** — what's available, what's been tried:

```
  [avail] quantize_int8             P-20.0% L-15.0%      acc: minor
  [tried] reduce_resolution         P-25.0% L-30.0%      acc: moderate
  [n/a]   clock_scaling             P-15.0% L--10.0%     acc: none
```

**Pareto frontier** — when MOO is active: point count, hypervolume,
convergence metrics.

**Design rationale** — recent decisions with agent attribution.

### Steering options

| Decision | Effect |
|----------|--------|
| `continue` | Let the optimizer choose the next strategy |
| `accept` | Accept current design, generate report |
| `redirect` | Focus on a specific objective or force a strategy |
| `explore_more` | Request broader exploration |
| `stop` | Stop and report the current best |

When redirecting, you can:
- **Focus objective**: `focus_objective="power"` — bias strategy selection
- **Force strategy**: `force_strategy="quantize_int8"` — apply specific strategy
- **Relax constraints**: `constraint_relaxation={"max_power_watts": 8.0}`
- **Tighten constraints**: `constraint_tightening={"max_latency_ms": 20.0}`

## Programmatic Usage

### Batch mode (no review)

```python
from embodied_ai_architect.graphs.soc_runner import SoCDesignRunner
from embodied_ai_architect.graphs.soc_state import DesignConstraints

runner = SoCDesignRunner(static_plan=MY_PLAN)
result = runner.run(
    goal="Drone perception SoC: <5W, <33ms, <$30",
    constraints=DesignConstraints(
        max_power_watts=5.0,
        max_latency_ms=33.3,
        max_cost_usd=30.0,
    ),
    use_case="delivery_drone",
    platform="drone",
)
```

### Interactive mode (with review)

```python
from embodied_ai_architect.graphs.soc_runner import SoCDesignRunner
from embodied_ai_architect.graphs.soc_state import DesignConstraints

runner = SoCDesignRunner(
    static_plan=MY_PLAN,
    human_review=True,
    optimization_review=True,
)

# Start — pauses at plan review
status, state = runner.start(
    goal="Drone perception SoC: <5W, <33ms, <$30",
    constraints=DesignConstraints(
        max_power_watts=5.0,
        max_latency_ms=33.3,
        max_cost_usd=30.0,
    ),
    use_case="delivery_drone",
    platform="drone",
)

# status == "review_plan"
# Display state["review_snapshot"] to the human

# Approve the plan
status, state = runner.step(review_input={"decision": "approve"})

# status == "review_optimization" (if constraints fail)
# Display state["optimization_review_snapshot"] to the human

# Steer the optimization
status, state = runner.step(steering_input={
    "decision": "redirect",
    "focus_objective": "power",
    "notes": "Power is the tightest constraint for airborne operation",
})

# Keep stepping until complete
while status != "complete":
    status, state = runner.step(steering_input={"decision": "continue"})
```

### Using the building blocks directly

For maximum control, use the review modules without the runner:

```python
from embodied_ai_architect.graphs.review import (
    PlanReviewInput,
    ReviewDecision,
    apply_review_edits,
    build_review_snapshot,
    render_plan_review_rich,
)
from embodied_ai_architect.graphs.optimization_review import (
    OptimizationSteeringInput,
    SteeringDecision,
    apply_steering_input,
    build_optimization_review_snapshot,
    render_optimization_review,
)

# After planner runs, build and display the snapshot
snapshot = build_review_snapshot(state, available_agents)
print(render_plan_review_rich(snapshot))

# Modify the plan
review = PlanReviewInput(
    decision=ReviewDecision.MODIFY,
    tasks_to_add=[{
        "id": "t_safety",
        "name": "Validate flight-critical safety",
        "agent": "safety_detector",
        "dependencies": ["t4"],
    }],
    tasks_to_remove=["t6"],
    notes="Safety validation required for flight-critical system",
)
updates = apply_review_edits(state, review, available_agents)
state = {**state, **updates}

# During optimization, build and display the optimization snapshot
opt_snapshot = build_optimization_review_snapshot(state)
print(render_optimization_review(opt_snapshot))

# Steer
steering = OptimizationSteeringInput(
    decision=SteeringDecision.REDIRECT,
    focus_objective="power",
    constraint_relaxation={"max_cost_usd": 35.0},
    notes="Relaxing cost by $5 to explore lower-power designs",
)
updates = apply_steering_input(state, steering)
state = {**state, **updates}
```

## Chat Interface (branes chat)

The interactive design flow is available through `branes chat` via five
tools. Here's a typical session:

### Starting a design

```
You: Design a drone perception SoC with YOLO detection at 30fps,
     under 5W power and $30 BOM cost.

Architect: I'll start an interactive design session.
           [calls design_soc tool]

           ========================================================
             PLAN REVIEW — SoC Design Task Graph
           ========================================================

             Goal:        Design a drone perception SoC...
             Constraints: Power: 5.0 W | Latency: 33.3 ms | Cost: 30.0 USD

           ────────────────────────────────────────────────────────
             TASK GRAPH (dependency tree)
           ────────────────────────────────────────────────────────

             [t1] Analyze perception workload (workload_analyzer) [READY]
             ├── [t2] Enumerate feasible hardware (hw_explorer)
             │   └── [t3] Compose SoC architecture (architecture_composer)
             │       └── [t4] Assess PPA (ppa_assessor)
             └── [t5] Design review (critic)
                   └── [t6] Generate report (report_generator)

             EXECUTION SCHEDULE: Step 1: t1 | Step 2: t2,t5 | ...

           Use review_plan tool to approve, modify, or reject.
```

### Reviewing and modifying the plan

```
You: The plan looks good but I want a safety check for the flight
     system. Add one after PPA assessment.

Architect: I'll modify the plan to add safety validation.
           [calls review_plan with tasks_to_add]

           Plan updated: added safety_detector task after t4.
           Execution now has 7 tasks across 6 parallel groups.

You: Approve it.

Architect: [calls review_plan(decision="approve")]
           Plan approved. Starting execution...
```

### Steering the optimization

```
Architect: Pipeline complete. Power is 6.2W (FAIL), latency 28.5ms (PASS).
           Entering optimization loop.

           [calls show_optimization_status]

           ========================================================
             OPTIMIZATION REVIEW — Iteration 2/20
           ========================================================

             Constraint       Target     Actual   Margin  Verdict
             power              5.0W       6.2W   -24.0%     FAIL
             latency          33.3ms     28.5ms   +14.4%     PASS
             cost            30.0USD    35.0USD   -16.7%     FAIL

             Strategy: Recommends 'smaller_model' targeting power.

           What would you like to do?

You: Power is the priority — we're airborne. But I don't want to
     sacrifice accuracy. Try quantization first.

Architect: [calls steer_optimization(
               decision="redirect",
               focus_objective="power",
               force_strategy="quantize_int8",
               notes="Power critical for airborne; preserve accuracy"
           )]

           Applied quantize_int8. Power: 6.2W > 5.0W.
           Verdict: power PASS, latency PASS, cost still FAIL.

You: Cost is $35 vs target $30. Can we relax to $35?

Architect: [calls steer_optimization(
               decision="redirect",
               constraint_relaxation={"max_cost_usd": 35.0}
           )]

           Constraints updated. All verdicts now PASS.

You: Accept this design.

Architect: [calls steer_optimization(decision="accept")]

           Design accepted. Generating final report...
```

### Available chat tools

| Tool | When to use |
|------|-------------|
| `design_soc` | Start a new design session |
| `review_plan` | At plan review: approve/modify/reject |
| `steer_optimization` | At optimization review: direct the optimizer |
| `show_optimization_status` | View current optimization state (read-only) |
| `show_plan` | View current task graph (read-only) |

## Running the Example

```bash
# Auto-approve demo (no interaction needed)
.venv/bin/python examples/demo_interactive_review.py

# With plan modification
.venv/bin/python examples/demo_interactive_review.py --modify

# Interactive prompts at each stage
.venv/bin/python examples/demo_interactive_review.py --interactive

# Limit optimization iterations
.venv/bin/python examples/demo_interactive_review.py --max-iterations 3
```

## Design Philosophy

The worst possible output is just the final answer. A human architect
cannot evaluate whether a result is good, bad, or whether more
exploration is needed if they only see the end state.

This review system is built around three principles:

1. **Show the journey, not just the destination** — constraint slackness,
   optimization trajectory, strategies tried and available, Pareto
   frontier data. Every iteration tells a story.

2. **Enable steering, not just approval** — the human can redirect focus,
   relax constraints, force strategies, or accept early. The optimizer
   is a tool, not an oracle.

3. **Record everything** — every human decision, every architect note,
   every strategy applied is recorded in the design history. The
   rationale chain is the institutional memory of the design.
