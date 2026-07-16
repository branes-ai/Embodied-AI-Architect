Generate the multi-level metrics dashboard for the current design state.

This is the "where am I" command — shows metrics at every level of abstraction so the architect can quickly identify what needs attention.

> **Bottlenecks are now typed `DesignIssue`s.** When the state has `open_issues`
> (filed by the loop's critic + specialist agents — see `/architect-loop`), treat
> that as the authoritative "what needs attention" list: each issue carries
> `metric`, `level`, `severity`, `observed_value` vs `target_value`, and
> `contribution_pct`. Surface the open issues alongside the raw metrics.

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

   **MOO frontier (if present)**: When `moo_explorer` ran (the default), read `pareto_points`, `moo_results`, and `optimization_review_snapshot.pareto_front_size` to surface a one-line summary right under the slackness table:

   ```text
   Pareto frontier:  <pareto_front_size> non-dominated designs   HV=<moo_results.hypervolume>
   Search effort:    <moo_results.total_evaluations> evals       layers=<moo_results.layers_used>
   ```

   If `pareto_front_size == 0` and `moo_results.total_evaluations > 0`, that's a signal — the search ran but found no feasible designs. Tell the architect the constraints may be too tight, and suggest loosening one to widen the feasible region. Also offer `/architect-drill pareto:<id>` to inspect a specific design point on the frontier.

3. **Operator breakdown**: Read `workload_profile` for per-operator GFLOPS, memory, latency, and hardware mapping.

   **If `workload_profile.source == "codebase_analysis"`** (the session was created from a codebase scan), show a richer **source-mapped operator breakdown** that ties each workload back to the actual source code. Read `codebase_metadata` for the project context and use the per-workload `source_file`, `line_range`, `kernel_type`, `estimated_gflops`, `estimated_memory_mb`, and `frameworks` fields:

   ```text
   OPERATOR BREAKDOWN (from codebase: <codebase_metadata.project_path>)
     Operator             Source                       Lines     Type              Est. Ops  Memory  Hardware
     ───────────────────  ───────────────────────────  ───────   ───────────────   ────────  ──────  ─────────
     <workload.name>      <workload.source_file>       <a-b>     <kernel_type>     <gflops>  <mb>    <ip block>
     ...
   ```

   The hardware mapping comes from looking up which `ip_block` the workload was assigned to (check `selected_architecture`). If no explicit mapping exists, fall back to inferring from `kernel_type` using this complete table:

   | kernel_type | Default hardware |
   |---|---|
   | `ml_inference` | KPU / GPU |
   | `signal_processing` | DSP / CPU |
   | `image_processing` | GPU / DSP |
   | `control_loop` | MCU / CPU |
   | `sensor_fusion` | CPU / GPU |
   | `io_bound` | MCU / CPU |
   | `general_compute` | CPU |

   Below the table, surface key metadata from `codebase_metadata`:
   - **Project**: name, languages, build system
   - **Code**: total lines of code, kernel count
   - **ML frameworks**: aggregate `workload.frameworks` across all workloads (e.g., pytorch, ultralytics, opencv)

   Suggest: *"To deep-dive a specific kernel, run /architect-drill source:<kernel_name>"*

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
