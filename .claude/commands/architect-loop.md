Run one iteration of the architect's bottleneck-hunting loop on the current design.

This is the core expert skill: assess → rank bottlenecks → drill down → propose options.

## Steps

1. **Assess current state**: Find the most recent design session state. Check:
   - `examples/demo_interactive_review.py` output or any recent `branes design plan` results
   - Look for PPA metrics, optimization history, and design rationale in the session
   - If no session exists, tell the user to start one with `branes design qualify` or `branes design plan`

2. **Rank bottlenecks across ALL metrics**: Identify the top 3 issues. For each:
   - Name the bottleneck and locate it (system/subsystem/operator/kernel level)
   - Quantify: how much does it contribute to the constraint violation?
   - Show headroom: how close are we to the limit? (percentage margin)
   - Show trend: across optimization iterations, is this getting better or worse?
   - Classify the bound: compute-bound, memory-bound, thermally-limited, cost-dominated, bandwidth-limited

   Metrics to check: power (W), latency (ms), area (mm²), cost ($), weight (g), volume (cm³), thermal margin (°C), memory bandwidth utilization (%), compute utilization (%), efficiency (GOPS/W, capability/W)

3. **Drill down on bottleneck #1**: Run the appropriate detailed analysis:
   - If compute-bound: `.venv/bin/branes mcp analyze <model>` for kernel breakdown
   - If memory-bound: check bandwidth_validator results for BW oversubscription
   - If cost-dominated: `.venv/bin/branes swap estimate` for cost breakdown by component
   - If thermally-limited: check thermal margin and power density
   - If latency-tight: show per-operator latency waterfall

4. **Propose 3-5 concrete options** for the #1 bottleneck:
   - For each option: estimated impact on the bottleneck metric AND side effects on other metrics
   - Flag any option that would make another currently-passing constraint fail
   - Rank options by impact-to-risk ratio

5. **Summarize as a situation report**:
   ```
   ARCHITECT LOOP — Iteration N

   TOP 3 BOTTLENECKS:
     1. [name] at [level]: [metric] = [value] vs [target] ([margin]% headroom)
     2. ...
     3. ...

   DRILL-DOWN on #1:
     [detailed analysis]

   OPTIONS:
     A. [action] — estimated [impact] on [metric], side effect: [effect]
     B. ...

   RECOMMENDATION: [which option and why]
   NEXT ACTION: [what the architect should do]
   ```

Do NOT choose an option or execute changes — present the analysis and let the architect decide.
