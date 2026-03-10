# Workflow: Drone Perception Pipeline Bottleneck Analysis

End-to-end workflow demonstrating how to take an embodied AI application,
characterize pipeline delays and energies, isolate the bottleneck stage,
evaluate its computational graph, and produce a ranked hardware comparison
spanning both custom SoC designs and COTS hardware.

---

## Prerequisites

```bash
# embodied-ai-architect venv with graphs on PYTHONPATH
export PYTHONPATH=/path/to/graphs/src
# or for the standard clone layout:
export PYTHONPATH=../graphs/src
```

All commands below assume you are in the `embodied-ai-architect` repo root
with the `.venv` activated.

---

## Step 1 — Scan the Application

Static file enumeration — identifies source files, languages, build system,
ML model files, dependencies, and classifies entry points:

```bash
.venv/bin/branes codebase scan prototypes/drone_perception
```

**What scan does:**
- Walks the directory tree and lists every source file with language, line count,
  and role (`entry_point`, `library`, `test`, `config`).
- Sub-classifies entry points as `application`, `example`, or `test` using
  path and filename heuristics.
- Detects the build system, ML model files (`.onnx`, `.pt`, etc.), and
  extracts dependency names.
- Recommends the most likely application entry point and suggests the next command.

**What scan does NOT do:**
- It does not parse code semantics, identify pipeline stages, extract compute
  kernels, or run any LLM analysis.  That happens in Step 2.

**Output:** A file table with five columns (File, Language, Lines, Role,
Entry Type) plus a recommendation of which entry point is the main application.

The scan output tells you *what files exist* and *which one to analyze next*.
Pipeline stages, compute costs, and dataflow are extracted by `branes codebase
analyze` in Step 2.

---

## Step 2 — Characterize Each DNN Stage on a Baseline Target

Pick a representative edge target as baseline.  `jetson_orin_nano` is a
common drone-class device:

```bash
# Detection stage — the heavy one
.venv/bin/branes --quiet mcp analyze yolov8n jetson_orin_nano

# Lighter detection variant for comparison
.venv/bin/branes --quiet mcp analyze yolov8s jetson_orin_nano

# Tracking backbone (if DNN-based)
.venv/bin/branes --quiet mcp analyze mobilenet_v2 jetson_orin_nano
```

Each `analyze` call returns:

- **Latency** (ms) — roofline-predicted inference time
- **Energy** (mJ) — compute + memory + static breakdown
- **Peak memory** (MB) — activations + weights + workspace
- **Utilization** (%) — how much of the hardware is actually used
- **Bottleneck** — compute-bound vs memory-bound

---

## Step 3 — Isolate the Bottleneck Stage

From Step 2, the detection stage (YOLOv8) dominates latency and energy.
Get the detailed breakdown to understand *why*:

### Roofline analysis — where is the time going?

```bash
.venv/bin/branes --quiet mcp latency yolov8n jetson_orin_nano
```

Returns the top-10 subgraphs by time, each tagged as compute-bound or
memory-bound.  Typical finding: backbone convolutions are compute-bound,
detection head is memory-bound.

### Energy breakdown — compute vs memory vs static

```bash
.venv/bin/branes --quiet mcp energy yolov8n jetson_orin_nano
```

Shows how Joules are split across compute operations, data movement, and
leakage.  High static energy relative to compute energy indicates the
hardware is oversized for this workload.

### Memory analysis — does it fit?

```bash
.venv/bin/branes --quiet mcp memory yolov8n jetson_orin_nano
```

Reports peak memory, activation timeline, weight footprint, and whether
the model fits in on-device SRAM / L2 cache.  Critical for determining
whether a smaller accelerator is viable.

---

## Step 4 — Evaluate Hardware Candidates

The key question: *what hardware should run the bottleneck stage?*

Compare across COTS edge devices and custom Stillwater KPU designs in a
single command:

```bash
.venv/bin/branes --quiet mcp compare yolov8n \
    jetson_orin_nano \
    jetson_orin_nx_16gb_gpu \
    jetson_orin_agx_gpu \
    coral_edge_tpu \
    hailo_8 \
    stillwater_kpu_t64 \
    stillwater_kpu_t256 \
    stillwater_kpu_t768 \
    qrb5165_dsp \
    --sort latency
```

**Example output** (illustrative — actual values from calibrated models):

```
           Comparison: yolov8n (fp16), sorted by latency
┏━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━┓
┃ # ┃ Hardware              ┃ Latency (ms) ┃  Throughput ┃ Energy (mJ) ┃ Memory(MB) ┃ Util(%) ┃
┡━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━┩
│ 1 │ stillwater_kpu_t768   │        0.812 │  1231.5 FPS │       0.081 │      12.50 │    32.1 │
│ 2 │ stillwater_kpu_t256   │        1.234 │   810.4 FPS │       0.062 │      12.50 │    78.3 │
│ 3 │ jetson_orin_agx_gpu   │        2.891 │   345.9 FPS │       0.173 │      14.20 │    45.2 │
│ 4 │ hailo_8               │        3.456 │   289.3 FPS │       0.052 │      11.80 │    71.5 │
│ 5 │ jetson_orin_nx_16gb   │        4.102 │   243.8 FPS │       0.205 │      14.20 │    38.7 │
│ 6 │ stillwater_kpu_t64    │        6.789 │   147.3 FPS │       0.034 │       8.00 │    85.2 │
│ 7 │ jetson_orin_nano      │        8.234 │   121.5 FPS │       0.123 │      14.20 │    52.1 │
│ 8 │ coral_edge_tpu        │       12.567 │    79.6 FPS │       0.025 │       8.00 │    68.9 │
│ 9 │ qrb5165_dsp           │       15.234 │    65.7 FPS │       0.046 │      10.50 │    42.8 │
└───┴───────────────────────┴──────────────┴─────────────┴─────────────┴────────────┴─────────┘
```

### Reading the table

- **KPU-T256** is the sweet spot: 810 FPS at 0.062 mJ with 78% utilization.
  High utilization means the accelerator is well-matched to the workload.
- **KPU-T768** is faster (1231 FPS) but only 32% utilized — oversized for
  YOLOv8n.  Better suited for YOLOv8x or multi-model pipelines.
- **Hailo-8** is competitive on energy (0.052 mJ) with good utilization.
- **Jetson Orin AGX** is fast but power-hungry (0.173 mJ) and under-utilized.
- **Coral Edge TPU** and **QRB5165 DSP** are too slow for real-time 30 FPS
  drone perception with this model.

---

## Step 5 — Deep-Dive the Top Candidates

### Hardware specs

```bash
.venv/bin/branes --quiet mcp specs stillwater_kpu_t256
.venv/bin/branes --quiet mcp specs hailo_8
.venv/bin/branes --quiet mcp specs jetson_orin_agx_gpu
```

### Detailed energy breakdown for the leaders

```bash
# Where is the KPU spending Joules?
.venv/bin/branes --quiet mcp energy yolov8n stillwater_kpu_t256

# Compare: where is the Jetson spending Joules?
.venv/bin/branes --quiet mcp energy yolov8n jetson_orin_agx_gpu
```

Typical finding: KPU has lower static energy (smaller die, power-gated
unused tiles) while Jetson has higher compute throughput but wastes energy
on memory traffic.

### Re-rank with INT8 quantization

```bash
.venv/bin/branes --quiet mcp compare yolov8n \
    stillwater_kpu_t256 \
    jetson_orin_agx_gpu \
    hailo_8 \
    --precision int8 --sort energy
```

INT8 typically improves all candidates but changes the relative ranking.
Accelerators with native INT8 datapaths (Hailo, KPU) benefit more than
GPU architectures.

---

## Step 6 — System-Level SWaP-C Assessment

The graph-level analysis tells you which hardware is best for the
*compute workload*.  The SWaP-C analysis tells you whether that hardware
fits in the *physical system*.

### Estimate system physical properties

```bash
# KPU-T256 at 7nm process: weight, volume, cost
.venv/bin/branes swap estimate --area 25 --power 5 --process 7

# Score against the drone mission profile
.venv/bin/branes swap score --area 25 --power 5 --process 7 --profile drone
```

### Compare packaging options

```bash
# Lightweight drone package vs heavier ruggedized package
.venv/bin/branes swap compare --area 25 --power 5 --process 7 \
    --left "QFN,passive,abs_plastic" --right "FCBGA,active_fan,aluminum"
```

### Sensitivity analysis

```bash
# Which parameter most affects the SWaP-C score?
.venv/bin/branes swap sensitivity --area 25 --power 5 --process 7 --mode tornado
```

---

## Step 7 — Machine-Readable Output for Reports

Every command supports `--json` for scripting and report generation:

```bash
# Hardware comparison as JSON
.venv/bin/branes --quiet --json mcp compare yolov8n \
    stillwater_kpu_t256 \
    jetson_orin_agx_gpu \
    hailo_8 \
    coral_edge_tpu \
    > /tmp/hw_comparison.json

# SWaP-C score as JSON
.venv/bin/branes --quiet --json swap score \
    --area 25 --power 5 --process 7 --profile drone \
    > /tmp/swap_score.json

# Combine in a downstream tool or notebook
```

---

## Architecture: What Happens Under the Hood

```
                 embodied-ai-architect                           graphs
                 ─────────────────────                           ──────

Step 1   branes codebase scan ──→ scanner.py
           file inventory, entry point classification,
           languages, build system, dependencies

Step 2   branes mcp analyze ────→ graphs.mcp.server
                                    ├── UnifiedAnalyzer.analyze_model()
                                    │     ├── torch.fx graph extraction
                                    │     ├── FusionPartitioner → subgraphs
                                    │     ├── HardwareMapper → resource model
                                    │     ├── RooflineAnalyzer → latency
                                    │     ├── EnergyAnalyzer → energy
                                    │     └── MemoryEstimator → peak memory
                                    └── returns executive summary JSON

Step 4   branes mcp compare ────→ runs analyze_model() × N targets
                                    ├── HardwareRegistry (51 targets)
                                    │     ├── COTS: Jetson, Coral, Hailo, QRB
                                    │     └── Custom: KPU-T64, T256, T768
                                    └── returns ranked comparison table

Step 6   branes swap score ──→ physical_estimators.py
                                    ├── die cost (Murphy yield model)
                                    ├── package weight / volume
                                    ├── thermal feasibility check
                                    └── mission profile scoring (AHP)
```

### Confidence levels

Each graphs estimate carries a confidence tag:

| Level | Meaning | Source |
|-------|---------|--------|
| **CALIBRATED** | Measured on real hardware | `calibration/` profiles |
| **INTERPOLATED** | Interpolated from nearby measurements | Calibration + curve fit |
| **THEORETICAL** | Derived from vendor specs | Datasheet peaks × efficiency |
| **UNKNOWN** | No data available | Default fallback |

Calibrated results (e.g., Jetson Orin family) are accurate to ~10%.
Theoretical results (e.g., custom KPU) are order-of-magnitude guides
useful for architecture exploration, not deployment sign-off.

---

## Decision Matrix Template

After running Steps 1–6, fill in the decision matrix for the team review:

| Criterion | Weight | KPU-T256 | Jetson AGX | Hailo-8 | Coral TPU |
|-----------|--------|----------|------------|---------|-----------|
| Latency (ms) | 25% | 1.23 | 2.89 | 3.46 | 12.57 |
| Energy (mJ/inf) | 25% | 0.062 | 0.173 | 0.052 | 0.025 |
| Peak memory (MB) | 10% | 12.5 | 14.2 | 11.8 | 8.0 |
| Utilization (%) | 10% | 78.3 | 45.2 | 71.5 | 68.9 |
| System weight (g) | 10% | TBD | 60 | 15 | 5 |
| Unit cost ($) | 10% | TBD | $399 | $99 | $25 |
| SW ecosystem | 10% | Custom SDK | CUDA/TRT | Dataflow compiler | TFLite |
| **Confidence** | — | THEORETICAL | CALIBRATED | THEORETICAL | THEORETICAL |

The `branes mcp compare --json` output provides the first four rows.
The `branes swap estimate` output provides weight and cost.
Ecosystem maturity and confidence level are qualitative inputs for the team.
