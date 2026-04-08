Deep-dive analysis of a specific bottleneck target: $ARGUMENTS

The target can be:
- A subsystem (perception, control)
- An operator (yolo_detector, tracker, vio)
- A kernel (conv2d, fft, matmul)
- A physical component (kpu, sram, interconnect)
- **A KPU micro-architecture target** — `kpu`, `systolic_array`, `sram_hierarchy`, `noc`, or `bandwidth_chain` (issue #33). See the "KPU drill targets" section below.
- A source-level kernel via `source:<kernel_name>` (only when the session was created from a codebase scan — see "Source drill" below).

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
   - **Special prefix `source:<kernel_name>`** — drill into the actual source
     code for a kernel (only valid when `workload_profile.source ==
     "codebase_analysis"`). See "Source drill" below.
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

   **Use sensitivity to find which design variables matter most** (issue #24):
   Read `optimization_review_snapshot.sensitivity` — a `{variable_name:
   {objective_name: impact_score}}` map from the BO layer. Sort by impact on
   the failing objective; the top variables tell you which design knobs to
   prioritize. If the target is a constraint (power/latency/cost), the
   highest-impact variables for that objective ARE the answer.

5. **Propose 3-5 targeted actions** with estimated impact and side effects.
   When sensitivity data is available, prefer actions that turn high-impact
   variables. Cite the impact score in each option (e.g., "Lower
   `quantization_dtype` to int8 — sensitivity to power = 0.82, est. 30%
   power reduction").

## Source drill (`source:<kernel_name>`)

When the target starts with `source:`, the user wants to see the actual code
behind a workload. Only works for sessions with `workload_profile.source ==
"codebase_analysis"`.

Steps:
1. Strip the `source:` prefix to get the kernel name.
2. Find the matching workload in `workload_profile.workloads[]` by name.
3. Read its `source_file`, `line_range`, `kernel_type`, `frameworks`,
   `estimated_gflops`, `estimated_memory_mb`, and `invocation_frequency_hz`.
4. Resolve the source file path **safely** to prevent path traversal:
   - Compute `project_root = Path(codebase_metadata.project_path).resolve()`
   - Compute `resolved = (project_root / source_file).resolve()`
   - Verify `resolved` is still under `project_root` (use
     `resolved.is_relative_to(project_root)`). If not, **reject snippet
     loading** and report "invalid source path: outside project root".
5. Read the snippet using the line_range (e.g., `Read tool with offset/limit`).
   If the file is missing or unreadable, report the path and skip the snippet.
6. Present:

   ```text
   KERNEL: <name>
   ─────────────────────────────────────────────────────
   Source:    <project_path>/<source_file>
   Lines:     <a-b>
   Type:      <kernel_type>
   Frameworks: <list>
   Estimated: <gflops> GFLOPS, <memory_mb> MB, <freq_hz> Hz
   ```

   Then the actual code:

   ```python
   <snippet from source file>
   ```

   Then analysis:

   ```text
   Performance hotspots:
   - <root cause analysis: why this kernel is interesting>

   Optimization options:
   - <3-5 targeted actions, e.g., quantize to INT8, fuse with adjacent
     kernel, move to KPU, reduce input resolution>
   ```

## KPU drill targets (issue #33)

When the target is one of the KPU micro-architecture keywords below, the
skill switches modes and reads from `state["kpu_config"]`,
`state["floorplan_estimate"]`, and `state["bandwidth_match"]` (populated
when `rtl_enabled=True` — issues #29 / #30 / #31). All five targets are
read-only inspections; no state mutation.

### Recognized KPU targets

| Target | What it shows |
|---|---|
| `kpu` | Full config summary + floorplan + bandwidth status (the "where am I" view) |
| `systolic_array` | Per-tile array dims, peak TOPS, vector lanes, frequency, utilization vs workload demand |
| `sram_hierarchy` | L1 / L2 / L3 sizes, banks, total capacity, area contribution |
| `noc` | Topology, link width, frequency, router count, link bandwidth |
| `bandwidth_chain` | DRAM → L3 → L2 → L1 → compute waterfall, with per-link demand/supply/utilization and the bottleneck link |

### How each target reads from state

For all KPU targets, first verify `state.get("kpu_config")` is non-empty.
If the session was run without `rtl_enabled=True`, tell the architect:

> No KPU micro-architecture data on this session — re-run with
> `rtl_enabled=True` to populate `kpu_config`, `floorplan_estimate`, and
> `bandwidth_match`.

Then dispatch by target name:

#### `kpu`

The "show me everything" view. Render three subsections:

```text
KPU CONFIGURATION — <kpu_config.name> at <kpu_config.process_nm>nm
─────────────────────────────────────────────────────────────────
  Grid:        <array_rows> x <array_cols> checkerboard
               (<num_compute_tiles> compute + <num_memory_tiles> memory tiles)
  Peak TOPS:   <peak_tops_int8> TOPS INT8
  SRAM total:  L1=<total_l1>KB | L2=<total_l2>KB | L3=<total_l3>KB
  NoC:         <noc.topology>, <noc.link_width_bits>-bit, <noc.frequency_mhz>MHz
  DRAM:        <dram.technology>, <dram.num_controllers>x<dram.channels_per_controller>ch,
               <total_dram_bandwidth_gbps> GB/s
```

Then the floorplan and bandwidth one-liners (full breakdowns are available
via the dedicated targets `systolic_array`, `bandwidth_chain`, etc.).

#### `systolic_array`

Read `kpu_config.compute_tile`. Compute utilization from `workload_profile`:

```text
SYSTOLIC ARRAY (per compute tile)
─────────────────────────────────
  Dimensions:   <array_rows> x <array_cols> INT8 MACs
  Vector lanes: <vector_lanes>
  Frequency:    <frequency_mhz> MHz
  Per-tile TOPS: <ct.peak_tops_int8>
  Tiles in grid: <num_compute_tiles>
  Total peak:   <kpu_config.peak_tops_int8> TOPS

  Workload demand: <workload_profile.total_estimated_gflops> GFLOPS
  Utilization:    <demand / (peak * 1000) * 100>%
```

If utilization < 30%, suggest `reduce_systolic_array` or
`reduce_compute_tiles` (the strategies from issue #32). If > 90%, suggest
growing or splitting workload across tiles.

#### `sram_hierarchy`

Read all three SRAM levels. Use `kpu_config.total_l1_bytes`,
`total_l2_bytes`, `total_l3_bytes`, plus per-tile fields:

```text
SRAM HIERARCHY
──────────────
  Level   Per-tile  Tiles    Total      Banks    Notes
  ──────  ────────  ───────  ─────────  ───────  ─────────────────
  L1      <l1>KB    <ct>     <l1*ct>KB  <l1bk>   Skew buffer (compute-local)
  L2      <l2>KB    <ct>     <l2*ct>KB  <l2bk>   Inside compute tile
  L3      <l3>KB    <mt>     <l3*mt>KB  <l3bk>   Memory tiles (checkerboard)

  Total SRAM: <total_sram_bytes / 1024> KB
```

Pull area contribution from `floorplan_estimate.compute_tile.sub_blocks`
and `floorplan_estimate.memory_tile.sub_blocks` if present (the
sub-blocks include SRAM macros). Hit rates aren't currently modeled —
note that and suggest `add_sram_banks` if `bandwidth_match` shows L2 or
L3 bottlenecks.

#### `noc`

Read `kpu_config.noc`. Show topology, link width, frequency, derived
link bandwidth, and router count:

```text
NoC
───
  Topology:     <topology>
  Link width:   <link_width_bits> bits
  Frequency:    <frequency_mhz> MHz
  Link BW:      <link_bandwidth_gbps> GB/s   (link_width_bits * frequency_mhz / 8 / 1000)
  Routers:      <num_routers>
```

Cross-reference `bandwidth_match.links` for any link whose name contains
"noc" or "l3" — if it's the bottleneck, suggest `widen_noc` (the strategy
from issue #32) and quantify the expected impact.

#### `bandwidth_chain`

The headline view of issue #30's KPU bandwidth waterfall. Read
`bandwidth_match` (or `optimization_review_snapshot.kpu_bandwidth` if
present — the snapshot version has a status classification already
applied):

```text
BANDWIDTH CHAIN — KPU Memory Hierarchy
Total: <compute_demand_gbps> GB/s demand | <dram.total_bandwidth_gbps> GB/s DRAM supply

Link              Demand    Supply    Util    Bound?  Fix
────────────────  ────────  ────────  ──────  ──────  ──────────────────
DRAM → L3         <d> GB/s  <s> GB/s  <u>%    <yn>    <hint>
L3 → L2           <d> GB/s  <s> GB/s  <u>%    <yn>    <hint>
L2 → L1           <d> GB/s  <s> GB/s  <u>%    <yn>    <hint>
L1 → compute      <d> GB/s  <s> GB/s  <u>%    <yn>    <hint>
```

For the `Fix` column, map each saturated link to the right strategy:
- DRAM bottleneck → `upgrade_dram_technology` or add controllers
- L3 / NoC bottleneck → `widen_noc`
- L2 bottleneck → `add_sram_banks` (L2 banks)
- L1 bottleneck → add streamers (no catalog strategy yet — flag as a gap)

Then the root-cause analysis the issue example shows: identify the
bottleneck link from `bandwidth_match.bottleneck_link`, name the
constraint, and propose 3 options.

### Examples

```text
/architect-drill kpu
/architect-drill systolic_array
/architect-drill bandwidth_chain
```

The bandwidth_chain example matches the issue #33 spec — see the issue
body for the expected output shape.

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
