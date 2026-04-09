# Architect Workflows — Five Design Scenarios

## The Core Loop

Every embodied AI system design follows the same iterative loop, regardless
of the specific platform or application:

```
Characterize → Identify Bottlenecks → Analyze Bottlenecks → Explore Solutions → Re-evaluate
     ↑                                                                              │
     └──────────────────────────────────────────────────────────────────────────────┘
```

This is the fundamental skill of a systems architect: at every pass through
the loop, the architect identifies the **top 3 issues** across all metrics
(energy, performance, latency, cost, power, complexity, size, weight, heat,
variability, headroom, efficiency, utilization), drills down to understand
why each is a bottleneck, explores solution options at the appropriate level
of abstraction, applies the best option, and re-evaluates the whole system
to find the new set of bottlenecks.

The platform generates metrics at **multiple levels of abstraction and
composition**:

```
System Level     ─────  Total power: 4.8W | Total latency: 28ms | BOM: $42
  │
  ├─ Subsystem   ─────  Perception: 3.2W/22ms | Control: 0.8W/4ms | Comms: 0.8W/2ms
  │    │
  │    ├─ Operator ────  YOLO detector: 2.1W/15ms | Tracker: 0.6W/4ms | VIO: 0.5W/3ms
  │    │    │
  │    │    └─ Kernel ─  Conv2D: 1.2W/8ms | FC: 0.4W/3ms | NMS: 0.5W/4ms
  │    │
  │    └─ Hardware ────  KPU: 2.5W/18ms (72% util) | CPU: 0.5W/6ms (35% util)
  │
  └─ Physical    ─────  Die: 42mm² | Package: 8x8mm | Thermal: 38°C (margin: 12°C)
```

At each level, the architect can see:
- **Absolute metrics**: power, latency, area, cost
- **Relative metrics**: utilization, efficiency, headroom
- **Derived metrics**: capability/watt, GOPS/watt, cost/performance
- **Variability**: p50/p95/p99 latency, thermal cycling, voltage droop

---

## Workflow 1: Reconnaissance Drone — Multi-Session Design Campaign

### The Problem

Design a long-range, long-duration reconnaissance drone that:
- Gathers multi-spectral surveillance data (visible + thermal + SAR)
- Monitors three points of interest with adaptive mission planning
- Adapts mission when observing behavioral or asset changes
- Operates for 8+ hours at 100km+ range
- Processes data onboard for real-time alerting, stores raw for post-mission

### Why This Is Multiple Sessions

This design cannot be done in one sitting. The architect works across
multiple sessions, each with a specific focus:

**Session 1: Application Characterization** (2-3 hours)
```
Goal: Understand the computational workload
Tasks:
  - Decompose the perception pipeline: visible detection + thermal detection
    + SAR processing + multi-spectral fusion + change detection + alert logic
  - For each operator: identify model family, estimate GFLOPS, memory footprint,
    latency budget, data rate
  - Build the operator graph: what runs in parallel, what is sequential,
    what is triggered conditionally (change detection → alert only on change)
  - Characterize the adaptive mission planner: how often does it re-plan?
    What inputs does it need? What latency can it tolerate?

Output: Operator graph with per-operator metrics, total system compute
        profile, data flow diagram with bandwidth requirements

Key Metrics Generated:
  System level:  Total GFLOPS, total memory BW, total sensor data rate
  Per-operator:  GFLOPS, memory, latency budget, duty cycle
  Data flow:     GB/s between operators, storage write rate
```

**Session 2: Bottleneck Identification** (1-2 hours)
```
Goal: Find the top 3 bottlenecks in energy, performance, and cost
Tasks:
  - Map operators to candidate hardware (KPU, GPU, CPU, DSP)
  - Run PPA assessment against drone constraints (15W compute, 8hr endurance)
  - Identify: which operator consumes the most power? Which has the least
    latency headroom? Which drives the die area?

Output: Ranked bottleneck list with per-operator contribution analysis

Expected Discovery:
  1. SAR processing dominates power (40% of compute budget)
  2. Multi-spectral fusion has tightest latency margin (2ms headroom)
  3. On-chip SRAM for change detection buffer drives die area

Drill-Down:
  SAR processing → which kernels? → FFT is 60% of SAR → FFT is memory-bound
  on current config → bandwidth_validator shows 2x oversubscription
```

**Session 3: Design Exploration for SAR Bottleneck** (2-3 hours)
```
Goal: Resolve the SAR power bottleneck
Tasks:
  - Explore: dedicated SAR accelerator vs. time-shared with detection
  - Explore: reduce SAR resolution (trade accuracy for power)
  - Explore: process SAR in bursts (duty-cycle to reduce average power)
  - Explore: offload SAR to ground station (trade comms BW for compute)
  - Run MOO across these options: objectives = power, latency, accuracy, cost

Output: Pareto frontier of SAR design options with trade-off analysis
        Recommended option with rationale

How the architect actually consumes the MOO output:

  After moo_explorer runs (it's in the default plan, no extra setup), the
  session state has:

    state["pareto_points"]              # accumulated non-dominated designs
    state["moo_results"]["sensitivity"] # which knobs drive which objective
    state["moo_results"]["layers_used"] # which layers ran (map_elites, bayesian)
    state["optimization_review_snapshot"]["pareto_front_size"]

  The architect skills wire these into action:

    /architect-assess  → shows MOO summary block + sensitivity-ranked knobs
    /architect-loop    → uses sensitivity to pick the highest-leverage knob
                         for the failing objective (issue #24, #25)
    /architect-drill   → can deep-dive a specific design point from the front

  Or programmatically via the API:

    GET /api/sessions/{id}/pareto       # frontier points + history
    GET /api/sessions/{id}/sensitivity  # ranked variables × objectives

Key Decision: Burst-mode SAR processing at 25% duty cycle
  - Reduces SAR average power from 6W to 1.5W
  - Increases SAR latency from 50ms to 200ms (acceptable for recon)
  - No accuracy loss
  - Saves $8 in die area cost
```

**Session 4: Re-evaluation and Next Bottleneck** (1-2 hours)
```
Goal: After SAR fix, what are the new top 3?
Tasks:
  - Re-run full system PPA with burst-mode SAR
  - New bottleneck ranking
  - Drill into the new #1

Expected Discovery:
  1. Thermal imaging now dominates (was #4, moved to #1 after SAR fix)
  2. Change detection memory buffer still large
  3. Cost target still missed due to multi-spectral sensor BOM

This is the loop: fix one bottleneck, the ranking shifts, new issues emerge.
```

**Session 5: Final Integration and Report** (1-2 hours)
```
Goal: Converge on final design, generate report
Tasks:
  - Apply thermal imaging optimization (INT8 quantization, sufficient for recon)
  - Apply memory optimization (streaming change detection, no full-frame buffer)
  - Final PPA assessment with all optimizations
  - Generate design report with full decision trail
  - Generate BOM estimate at 1K/10K/100K volumes
```

### Metrics at Each Level

At every session, the architect sees a dashboard like:

```text
┌─────────────────────────────────────────────────────────┐
│ SYSTEM OVERVIEW                                         │
│                                                         │
│ Power:   12.3W / 15.0W budget  [====████    ] 82%       │
│ Latency: 45ms / 100ms budget   [===█        ] 45%       │
│ Area:    78mm² / 100mm²        [======██    ] 78%       │
│ Cost:    $38 / $50 target      [=====██     ] 76%       │
│ Weight:  180g / 250g           [====█       ] 72%       │
│ Thermal: 62°C / 85°C max       [=====██     ] 73%       │
│                                                         │
│ TOP 3 BOTTLENECKS:                                      │
│   1. SAR processing: 6.0W (49% of compute power)        │
│   2. Fusion latency: 48ms (2ms headroom)                │
│   3. Change detection SRAM: 32mm² (41% of die area)     │
│                                                         │
│ EFFICIENCY:                                             │
│   Capability/Watt: 0.42                                 │
│   KPU utilization: 78%                                  │
│   Memory BW utilization: 91% ← near saturation          │
│   Power headroom: 2.7W (18%)                            │
└─────────────────────────────────────────────────────────┘
```

Drilling into bottleneck #1:

```text
┌─────────────────────────────────────────────────────────┐
│ SAR PROCESSING — Detailed Breakdown                     │
│                                                         │
│ Total: 6.0W | 50ms | compute-bound                      │
│                                                         │
│ Kernel          Power   Latency   Bound     Utilization │
│ ──────────────  ──────  ────────  ────────  ─────────── │
│ range_FFT       2.4W     18ms     memory    91% BW      │
│ azimuth_FFT     1.8W     14ms     memory    87% BW      │
│ matched_filter  1.2W     12ms     compute   72% ALU     │
│ CFAR_detect     0.6W      6ms     compute   45% ALU     │
│                                                         │
│ FFT kernels are memory-bandwidth-bound:                 │
│   Required: 25.6 GB/s  |  Available: 19.2 GB/s          │
│   → 1.33x oversubscribed                                │
│                                                         │
│ OPTIONS:                                                │
│   A. Add SRAM bank (+4mm², -30% BW pressure)            │
│   B. Burst-mode duty cycling (25% → avg 1.5W)           │
│   C. Reduce range resolution (2x → -50% BW)             │
│   D. Dedicated SAR accelerator (+15mm², -60% power)     │
└─────────────────────────────────────────────────────────┘
```

---

## Workflow 2: Warehouse AMR Fleet — Cost-Optimized at Scale

### The Problem
Design the perception compute for a warehouse AMR fleet (10K units).
At volume, every dollar of BOM cost matters. The architect's primary
metric is cost/unit, with performance as a constraint rather than an
objective.

### Session Flow
1. **Characterize**: 2D LiDAR SLAM + obstacle avoidance + barcode reading
2. **Bottleneck**: NRE amortization dominates at 10K volume — custom SoC
   too expensive, need COTS solution
3. **Explore**: Jetson Orin Nano vs. RK3588 vs. Hailo-8 + host CPU
4. **Re-evaluate**: Hailo-8 wins on cost but fails latency for barcode
   reading — need hybrid approach
5. **Converge**: Hailo-8 for SLAM inference + ARM CPU for barcode processing

### Key Metrics Focus
- Cost breakdown: die, package, test, NRE/unit at 10K, 100K volumes
- Power at idle vs. active (fleet power = mostly idle)
- Mean time between failures (fleet reliability)

---

## Workflow 3: Surgical Cobot — Safety-Critical Design

### The Problem
Design the perception + control compute for a surgical collaborative
robot arm with IEC 62304 Class C requirements.

### Session Flow
1. **Characterize**: Dual nervous system — brain (vision + planning) +
   peripheral (joint safety controllers)
2. **Bottleneck**: Brain-to-peripheral communication latency (EtherCAT
   cycle time) limits force-feedback bandwidth
3. **Explore**: Faster EtherCAT vs. dedicated safety bus vs. local
   processing at joints
4. **Re-evaluate**: Safety certification requires formal verification
   of joint controllers — drives selection toward certified MCUs
5. **Converge**: Certified ARM Cortex-R per joint + central Cortex-A
   for perception, with safety-rated EtherCAT

### Key Metrics Focus
- Safety integrity (SIL-4 for joint control, SIL-2 for perception)
- Worst-case execution time (WCET) for all safety-critical paths
- Diagnostic coverage percentage
- Force-feedback loop latency (must be <1ms)

---

## Workflow 4: Agricultural Drone Fleet — Multi-Spectral Survey

### The Problem
Design a fleet of agricultural survey drones that capture NDVI, thermal,
and RGB imagery for precision farming. Edge processing for real-time
anomaly detection, cloud upload for full analysis.

### Session Flow
1. **Characterize**: 3 camera streams + GPS/IMU + edge inference for
   anomaly flagging + raw storage for post-processing
2. **Bottleneck**: Storage write bandwidth — 3 raw streams at 4K exceed
   SD card write speed
3. **Explore**: Compressed storage vs. selective recording vs. faster
   storage medium vs. reduced resolution for non-anomaly frames
4. **Re-evaluate**: After storage fix, thermal camera cost dominates BOM
5. **Converge**: Shared optics with filter wheel (trade temporal for cost)

### Key Metrics Focus
- Data rate: sensor ingestion vs. storage write vs. comms uplink
- Coverage rate: acres/hour at required resolution
- Edge inference accuracy vs. cloud accuracy (acceptable gap)

---

## Workflow 5: Racing FPV Drone — Latency-Dominated Design

### The Problem
Design perception compute for an autonomous racing drone. Total
perception-to-actuation latency must be <15ms at 120fps. Every
millisecond matters.

### Session Flow
1. **Characterize**: Monocular gate detection + IMU fusion + trajectory
   prediction — the pipeline is short but every stage has microsecond budgets
2. **Bottleneck**: Gate detection model inference is 8ms — leaves only
   7ms for everything else
3. **Explore**: Smaller model (4ms but lower accuracy) vs. hardware
   acceleration (3ms but higher power) vs. predictive pre-computation
4. **Re-evaluate**: With faster inference, IMU-to-perception
   synchronization becomes the bottleneck (jitter in frame timestamps)
5. **Converge**: Hardware-timestamped capture + pipelined inference

### Key Metrics Focus
- p99 latency (not average — worst case matters at 40m/s)
- Jitter: frame-to-frame latency variance
- Pipeline depth: how many frames in flight simultaneously
- Thermal throttling probability during 5-minute race

---

## Workflow 6: KPU SoC — Inner Loop Convergence

### The Problem

When the design includes a custom KPU (Knowledge Processing Unit) and
RTL generation, the architect drives **two nested optimization loops**:

- **Outer loop**: SoC-level constraints (power, latency, area, cost) —
  same as Workflows 1–5
- **Inner loop**: KPU micro-architecture convergence — systolic array
  sizing, SRAM hierarchy, NoC topology, DRAM technology, floorplan
  pitch matching, bandwidth waterfall — issues #29–#35

These loops interact: the outer loop's `design_optimizer` can pick
KPU-targeted strategies (#32) that mutate `kpu_config`, which in turn
forces the inner-loop validators to re-run on the next dispatch
iteration.

### Setting Up the Session

Enable the inner loop on session creation (or via `SoCDesignRunner.run()`):

```python
runner = SoCDesignRunner(static_plan=KPU_PLAN)
state = runner.run(
    goal="Design a KPU-based SoC for delivery drone perception",
    constraints=DesignConstraints(max_power_watts=5.0, max_area_mm2=100.0),
    use_case="delivery_drone",
    platform="drone",
    rtl_enabled=True,            # turn on KPU + floorplan + bandwidth + RTL
    rtl_area_feedback=True,      # optional: re-size on synthesis area > floorplan
)
```

### The KPU Plan

Schedule KPU specialists in dependency order:

```
workload_analyzer
   → hw_explorer
       → architecture_composer
           → kpu_configurator       (sizes the micro-architecture)
               → floorplan_validator (parallel)
               → bandwidth_validator (parallel)
                   → rtl_generator
                       → rtl_ppa_assessor
                           → ppa_assessor → critic → report_generator
```

### What the Architect Sees

**At plan review time** (#29):
The plan review snapshot includes a **KPU configuration preview** — the
architect can override any field (compute_tile.array_rows,
noc.link_width_bits, dram.num_controllers, etc.) using dotted-path keys
in `PlanReviewInput.kpu_overrides`. Overrides survive across review
passes.

**During optimization review** (#30):
Two new sections in `OptimizationReviewSnapshot`:

```text
KPU FLOORPLAN
  Compute tile: 2.10 x 2.30 mm  |  Memory tile: 2.00 x 2.40 mm
  Pitch ratio: 1.05 (within 15% tolerance)  PASS
  Total die area: 48.2 mm² / 100 mm² budget  48%  PASS

KPU BANDWIDTH CHAIN
  Link              Demand    Supply    Util    Status
  DRAM → L3         12.8 GB/s  25.6 GB/s  50%   OK
  L3 → L2            9.0 GB/s  16.4 GB/s  55%   OK
  L2 → L1            4.5 GB/s   4.8 GB/s  94%   TIGHT!
  L1 → compute       1.4 GB/s   3.2 GB/s  43%   OK
```

**Across the session** (#34):
`branes session show --latest` renders a **KPU Convergence History**
block showing each iteration's specialist invocation, before/after
metrics, and the changes the optimizer applied. The history is monotonic
— a 10-iteration session shows the full trail.

### Drilling Into KPU Internals

`/architect-drill` understands KPU-specific targets (#33):

```text
/architect-drill kpu              # full config + floorplan + bandwidth
/architect-drill systolic_array   # array dims, peak TOPS, utilization
/architect-drill sram_hierarchy   # L1/L2/L3 sizes, banks, area
/architect-drill noc              # topology, link width, frequency
/architect-drill bandwidth_chain  # DRAM → L3 → L2 → L1 → compute waterfall
```

The skill cross-references the bandwidth bottleneck against the catalog
strategies from #32 (`upgrade_dram_technology`, `widen_noc`,
`add_sram_banks`) and proposes the right knob to turn.

### The RTL Feedback Loop (#31)

After RTL synthesis runs, the `rtl_area_feedback` specialist compares
synthesis area against the floorplan estimate. If synthesis exceeds
floorplan × 1.1, it triggers up to 3 KPU re-sizing iterations using the
existing optimizer adjustments and then re-validates floorplan +
bandwidth. Bounded to prevent infinite loops; every iteration is
recorded in `kpu_optimization_history`.

### Verification

The end-to-end integration test (`tests/test_kpu_integration.py::TestKPUFullPipeline`)
asserts that a real `SoCDesignRunner` run with `rtl_enabled=True`
produces:

- non-empty `kpu_config`, `floorplan_estimate`, `bandwidth_match`
- non-empty `rtl_synthesis_results`
- `kpu_optimization_history` with at least one entry per specialist
- session round-trips through `SessionStore`
- snapshot exposes `kpu_floorplan` + `kpu_bandwidth` + `kpu_history`
- `branes session show` renders the KPU Configuration block

That test was the forcing function for the dual-loop wiring fixes in
issue #35.

---

## Workflow 7: Customer Application Assessment (issues #37–#43)

### The Problem

A customer hands the architect a project directory (C++/Python/Rust) and
says "what hardware do I need to run this?" The architect needs to go from
source code to a hardware recommendation without manually characterizing
the workload.

### The One-Command Path

```bash
branes codebase design /path/to/customer_app --power 5 --latency 33
```

This runs the full chain:

1. **Scanner** (no LLM) → file inventory, build system, dependencies, ML models
2. **Analyzer** (LLM) → compute kernels, dataflow links, data types, frequencies
3. **Converter** → workload_profile + operator_graph (#40) for pipeline visualization
4. **Constraint inference** (#38) → suggested DesignConstraints from kernel
   characteristics (control_loop at 100Hz → 10ms latency, total GFLOPS → power
   envelope, ML dominance → NPU/GPU hint)
5. **Hardware recommender** (#39) → ranked list of hardware targets from the
   embodied-schemas registry, scored by compute / memory / power / cost fit
6. **Session creation** (#37) → `SoCDesignState` persisted to SessionStore
7. **Plan review** → the planner's task graph rendered for architect approval

### What the Architect Sees

```
Session saved: soc_abc123
  Goal:        Design SoC for customer_app (ml_inference, control_loop)
  Use case:    codebase_analysis
  Workload:    3 kernels, 10.4 GFLOPS, 70.0 MB

SUGGESTED CONSTRAINTS (inferred from codebase analysis)
  max_latency_ms   10.0     high    control_loop at 100Hz → 10ms per cycle
  max_power_watts  1.0      medium  10.4 GFLOPS estimated, ~2 TOPS/W at 28nm
  hardware_class   npu      high    67% ML inference, 10.4 GFLOPS suits an NPU

RECOMMENDED HARDWARE (for ml_heavy workload, 10.4 GFLOPS)
  Rank  Hardware             Fit    Strengths             Weaknesses
  1     Hailo-8              0.85   26.0 TOPS, $200       2GB memory (tight)
  2     Jetson Orin NX       0.82   100.0 TOPS, INT8      $400 cost
  ...

PLAN REVIEW — SoC Design Task Graph
  [t1] Analyze workload (workload_analyzer)
    └── [t2] Explore hardware (hw_explorer)
          ├── [t3] Compose architecture (architecture_composer)
          └── [t4] Explore Pareto frontier (moo_explorer)
                └── [t5] Assess PPA (ppa_assessor) ...
```

### After the Session Is Created

The architect can:

- `branes session show --latest` → PPA metrics, constraint slackness, MOO summary
- `/architect-assess` in chat → source-mapped operator breakdown (ties each
  workload back to the actual source file and line range)
- `/architect-drill source:yolo_detection` → see the actual code behind a kernel
- `/architect-drill bandwidth_chain` → DRAM→L3→L2→L1 waterfall (if KPU enabled)
- `branes design plan` → run the optimizer to find the Pareto frontier

### The Operator Graph for Frontend Visualization

The workload profile carries an `operator_graph` (#40) with nodes (one per
kernel) and edges (from the LLM's DataflowLink analysis). The API serves
it at `/api/sessions/{id}/workload` for rendering as a pipeline DAG:

```json
{
  "operator_graph": {
    "nodes": [
      {"id": "yolo_detection", "gflops": 8.4, "type": "convolution"},
      {"id": "tracker", "gflops": 2.0, "type": "matmul"},
      {"id": "pid_control", "gflops": 0.001, "type": "accumulate"}
    ],
    "edges": [
      {"source": "yolo_detection", "sink": "tracker", "data_bytes": 262144},
      {"source": "tracker", "sink": "pid_control", "data_bytes": 4096}
    ]
  }
}
```

### Verification

The end-to-end integration test (`tests/test_codebase_integration.py`)
asserts the full chain: scanner → analysis → workload with operator graph →
constraint inference (latency from control loop, power from GFLOPS,
hardware_class for ML-dominant) → SoC state with codebase metadata → session
round-trip → API workload endpoint with operators and graph.

---

## The Architect Skill — Codifying the Expert Loop

The workflows above all follow the same cognitive pattern. This
pattern should be codified as a Claude Code skill that amplifies the
architect's ability to systematically hunt bottlenecks and explore
solutions.

### The Skill: `/architect-loop`

```
Invoke: /architect-loop
Purpose: Run one iteration of the architect's bottleneck-hunting loop
```

**What the skill does:**

1. **Assess current state**: Read the latest PPA metrics, identify which
   constraints are passing/failing, compute utilization and headroom at
   every level of the hierarchy

2. **Rank bottlenecks**: Identify the top 3 issues across ALL metrics
   (power, latency, area, cost, thermal, bandwidth, utilization). For
   each bottleneck, show:
   - What it is and where (operator, kernel, IP block, physical)
   - How much it contributes to the constraint violation
   - What the headroom/margin is
   - Trend: is it getting better or worse across iterations?

3. **Drill down**: For the #1 bottleneck, automatically run the
   appropriate detailed analysis:
   - If compute-bound: show kernel breakdown, ALU utilization per IP block
   - If memory-bound: show bandwidth analysis, SRAM vs DRAM traffic
   - If thermally-limited: show power density map, hotspot analysis
   - If cost-dominated: show BOM breakdown, volume sensitivity

4. **Propose options**: For the #1 bottleneck, enumerate 3-5 concrete
   actions the architect can take, with estimated impact on the
   bottleneck AND side effects on other metrics

5. **Summarize**: Present a concise "situation report" that the architect
   can use to decide what to do next

**What the skill does NOT do:**
- It does not choose an option — the architect decides
- It does not execute changes — the architect directs
- It does not skip steps — every iteration produces the full analysis

### The Skill: `/architect-assess`

```
Invoke: /architect-assess
Purpose: Generate the multi-level metrics dashboard for current design state
```

Produces the system overview with metrics at every level of abstraction:
system → subsystem → operator → kernel, with utilization, headroom, and
efficiency metrics. This is the "where am I" command.

### The Skill: `/architect-drill <target>`

```
Invoke: /architect-drill SAR_processing
Purpose: Deep-dive analysis of a specific subsystem, operator, or kernel
```

Runs the detailed analysis for a named target: kernel breakdown, bandwidth
analysis, power density, or cost breakdown depending on what the target is
and what bound it's hitting.

### Implementation Approach

These skills are Claude Code commands (`.claude/commands/*.md`) that
instruct Claude to:

1. Read the current design state from the most recent run/session
2. Run the appropriate `branes` CLI commands to gather metrics
3. Synthesize the results into the structured dashboard format
4. Present bottleneck rankings with drill-down analysis
5. Propose concrete next actions

The skills compose: `/architect-assess` gives the overview,
`/architect-loop` runs one full iteration, `/architect-drill` goes deep
on a specific target. Together they implement the expert's cognitive
loop at the speed of an AI assistant.

### Why This Matters

Without this skill, the architect must:
- Remember which metrics to check after each change
- Manually identify which bottleneck shifted
- Know which analysis tool to run for each type of bottleneck
- Track the decision history across sessions

With this skill, the architect's cognitive load is reduced to the
essential creative work: deciding which trade-off to make, which
constraint to relax, and when the design is good enough.

The skill doesn't replace the architect — it amplifies them by handling
the systematic analysis and bookkeeping, freeing them to focus on
engineering judgment.
