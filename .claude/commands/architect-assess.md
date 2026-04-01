Generate the multi-level metrics dashboard for the current design state.

This is the "where am I" command — shows metrics at every level of abstraction so the architect can quickly identify what needs attention.

## How to get the state

Load the most recent design session from disk:

```bash
.venv/bin/branes session show --latest --json
```

This returns the full `SoCDesignState` JSON with PPA metrics, optimization history, constraint slackness, task graph, and design rationale. If no sessions exist, tell the user to start one with `branes design qualify` or `branes design plan`.

You can also list all sessions:
```bash
.venv/bin/branes session list
```

Or inspect a specific one:
```bash
.venv/bin/branes session show <session_id>
```

## What to present

1. **System-level overview**: Show all SWaP-C metrics with budget utilization:
   ```
   Power:   [value]W / [budget]W  [████░░░░░░] [%]  [PASS/FAIL]
   Latency: [value]ms / [budget]ms ...
   Area:    [value]mm² / [budget]mm² ...
   Cost:    $[value] / $[budget] ...
   ```
   Read `ppa_metrics` and `constraints` from the session state.

2. **Constraint slackness**: Read `optimization_review_snapshot.constraint_slackness` for per-constraint margin analysis with trends.

3. **Operator breakdown**: Read `workload_profile` for per-operator GFLOPS, memory, latency, and hardware mapping.

4. **Efficiency metrics**: Compute from PPA data:
   - Capability/Watt, GOPS/Watt
   - KPU/GPU/CPU utilization
   - Memory BW utilization
   - Power/latency/thermal headroom

5. **Highlight top 3 concerns**: The three metrics closest to their limits.
   - GREEN: >20% headroom
   - YELLOW: 5-20% headroom
   - RED: <5% headroom or exceeded

6. **Design journey**: Read `design_rationale` for the trail of decisions that got us here.

Present as a single cohesive dashboard. The architect should scan it in 30 seconds.
