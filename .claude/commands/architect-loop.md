Run one iteration of the architect's bottleneck-hunting loop on the current design.

This is the core expert skill: assess → rank bottlenecks → drill down → propose options.

## How to get the state

Load the most recent design session:

```bash
.venv/bin/branes session show --latest --json
```

This returns the full `SoCDesignState` JSON. The key fields you need:
- `ppa_metrics` — current power, latency, area, cost with verdicts
- `constraints` — target budgets
- `optimization_review_snapshot` — constraint slackness, trajectory, strategy analysis
- `optimization_history` — PPA snapshots across iterations
- `workload_profile` — per-operator compute requirements
- `selected_architecture` — hardware mapping
- `design_rationale` — decision trail
- `iteration` — which iteration we're on

If no session exists, tell the user:
```
No active design session. Start one with:
  branes design qualify "your design goal"
  branes design plan "your qualified goal" --power X --latency Y
```

## Steps

1. **Assess current state**: From the session JSON, extract:
   - All PPA metrics vs constraints (PASS/FAIL for each)
   - Constraint slackness (margin %, trend direction)
   - Current optimization iteration number

2. **Rank the top 3 bottlenecks across ALL metrics**: Look at:
   - Which constraints are FAIL? By how much?
   - Which passing constraints have <10% headroom? (about to fail)
   - Which operator/subsystem consumes the most of each resource?
   - Is memory bandwidth saturated? (`workload_profile` has BW data)
   - Is utilization unbalanced? (one IP block at 90%, another at 20%)

   For each of the top 3:
   - Name it and locate it (system/subsystem/operator/kernel level)
   - Quantify the gap (actual vs target, % over budget)
   - Show trend from `optimization_history` (improving, worsening, stable)
   - Classify: compute-bound, memory-bound, thermally-limited, cost-dominated

3. **Drill down on bottleneck #1**: Run additional analysis:
   - For compute: `.venv/bin/branes mcp analyze --model <model>` for kernel breakdown
   - For cost: `.venv/bin/branes swap estimate --area X --power Y --process Z` for cost decomposition
   - For power: check `ip_blocks` config for clock/voltage settings
   - For latency: check per-operator latency from `workload_profile`
   - **Use sensitivity to find the right knob**: Read
     `optimization_review_snapshot.sensitivity` (issue #24). It maps each
     design variable to its impact on each objective (0–1 score from the BO
     layer). For the failing metric, sort variables by their impact on that
     specific objective and pick the top 1-2 — those are the highest-leverage
     knobs to turn. Example: if power is failing and `quantization_dtype` has
     impact 0.82 on power but `noc_link_width_bits` has impact 0.04, propose
     changing the dtype not the NoC width.

4. **Propose 3-5 concrete options** from the strategy catalog:
   Read `optimization_review_snapshot.strategies` for what's available vs tried.
   For each option: estimated impact on bottleneck AND side effects on other metrics.
   Flag any option that would flip a passing constraint to FAIL.

5. **Summarize as situation report**:
   ```
   ARCHITECT LOOP — Iteration N/M

   TOP 3 BOTTLENECKS:
     1. [name] at [level]: [metric] = [value] vs [target] ([margin]% headroom, [trend])
     2. ...
     3. ...

   DRILL-DOWN on #1:
     [detailed analysis from step 3]

   OPTIONS:
     A. [action] — est. [impact] on [metric], side effect: [effect]
     B. ...

   RECOMMENDATION: [which option and reasoning]
   ```

Do NOT choose or execute — present the analysis and let the architect decide.

## How to apply the architect's decision

When the architect picks an option, the next step depends on whether we have an active interactive session:

- If running interactively: use `steer_optimization` tool in `branes chat`
- If using the pipeline: modify the state and re-run the dispatch step
- For quick what-if: `.venv/bin/branes swap score --area X --power Y --process Z --profile drone`
