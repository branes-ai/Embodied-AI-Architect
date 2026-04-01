Deep-dive analysis of a specific bottleneck target: $ARGUMENTS

The target can be a subsystem (perception, control), an operator (yolo_detector, tracker, vio), a kernel (conv2d, fft, matmul), or a physical component (kpu, sram, interconnect).

## How to get the state

Load the design session:

```bash
.venv/bin/branes session show --latest --json
```

The JSON contains `workload_profile`, `selected_architecture`, `ip_blocks`, `ppa_metrics`, `optimization_review_snapshot` — all the data needed for drill-down.

For additional hardware analysis, use the branes MCP/CLI:
```bash
.venv/bin/branes mcp analyze --model <model_name> --hardware <hw_name>
.venv/bin/branes swap estimate --area <mm2> --power <watts> --process <nm>
.venv/bin/branes swap sensitivity --area <mm2> --power <watts> --process <nm> --mode tornado
```

## Steps

1. **Identify the target**: Parse $ARGUMENTS. Match against:
   - `workload_profile.workloads[].name` for operators
   - `ip_blocks[].name` for hardware blocks
   - `ppa_metrics` fields for constraints (power, latency, cost)
   - If ambiguous, list available targets from the session and ask.

2. **Gather metrics for the target** from the session JSON:

   For an **operator**: Extract from `workload_profile`:
   - GFLOPS, memory footprint, data type, input/output shapes
   - Latency contribution and percentage of total pipeline
   - Which hardware block it's mapped to
   - Whether it's compute-bound or memory-bound

   For a **hardware block**: Extract from `ip_blocks` and `ppa_metrics`:
   - Configuration (frequency, SRAM size, data width)
   - Power contribution and percentage of total
   - Utilization (compute % and memory BW %)

   For a **constraint** (power, cost, latency): Extract from `ppa_metrics` and `constraints`:
   - Per-component breakdown (which operators/blocks contribute how much)
   - What's reducible vs fixed overhead
   - Cost breakdown from `ppa_metrics.cost_breakdown` if available

3. **Present detailed breakdown** in tabular format showing the composition of this target across its sub-components.

4. **Identify root cause**: Why is this a bottleneck?
   - Inherent to workload? (fundamental compute requirement)
   - Mapping issue? (wrong operator → hardware assignment)
   - Configuration issue? (clock speed, SRAM sizing, data width)
   - Physical limitation? (process node, die area budget)

5. **Propose 3-5 targeted actions** with estimated impact and side effects.

## Running what-if analysis

For quick exploration of how changes would affect metrics:
```bash
# Cost sensitivity to area and process node
.venv/bin/branes swap sensitivity --area <mm2> --power <watts> --process <nm> --mode tornado

# Score a design point against a mission profile
.venv/bin/branes swap score --area <mm2> --power <watts> --process <nm> --profile drone

# Compare two hardware targets
.venv/bin/branes mcp compare --hardware <hw1> --hardware <hw2> --model <model>
```
