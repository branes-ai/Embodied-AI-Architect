---
title: SWaP-C Analysis Walkthrough
description: Hands-on tutorial estimating system weight, volume, and thermal feasibility for NUC and Jetson form factors.
---

This tutorial walks through two real-world SWaP-C scenarios: an AMD Ryzen AI NUC for desktop edge inference, and an NVIDIA Jetson Orin Nano for drone perception. You'll learn to estimate, compare, check, and optimize physical system designs.

## Overview

You'll learn how to:
- Estimate system weight, volume, and cost from SoC parameters
- Inspect the BOM to understand where weight comes from
- Check designs against form factor budgets
- Compare packaging and cooling alternatives
- Run 6-objective optimization with SWaP-C

## Prerequisites

- **Branes** installed (`pip install -e ".[dev]"`)
- No API keys or optional dependencies required for SWaP-C commands

## Example 1: AMD Ryzen 7 AI in NUC Form Factor

The AMD Ryzen 7 8845HS is a modern heterogeneous SoC integrating CPU (Zen 4), GPU (RDNA 3), and NPU (XDNA) on TSMC 4nm. In a NUC chassis, it ships as a compact desktop for edge AI workloads — always-on video analytics, local LLM inference, industrial vision.

**Target specs:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Die area | 220 mm² | Zen 4 + RDNA 3 + XDNA monolithic die |
| Process | 4 nm | TSMC N4 |
| TDP | 45 W | Performance mode (configurable 15-45W) |
| Package | FCBGA | Flip-chip BGA for high TDP thermal path |
| Cooling | Active fan | Required for sustained 45W operation |
| Enclosure | Aluminum | NUC standard — thermal mass + EMI shielding |

**NUC form factor budgets:** ~800 g weight, ~700 cm³ volume (117×112×54 mm).

### Step 1: Quick Estimate

```bash
branes swap estimate \
    --area 220 --power 45 --process 4 \
    --package FCBGA --cooling active_fan --enclosure aluminum
```

**Expected Output:**

```
╭── SWaP-C Estimate: 220mm² / 4nm / FCBGA / active_fan ──╮
│  Weight:    261.3 g                                      │
│  Volume:    148.6 cm³                                    │
│  Cost:      $4168.52                                     │
│  Dims:      43×43×70 mm                                  │
│  Thermal:   Tj=57°C (margin: 68°C) ✓                    │
╰─────────────────────────────────────────────────────────╯
```

Key takeaways:
- **261 g** is well under the NUC's ~800 g budget — plenty of room for memory, storage, and I/O
- **Thermal margin of 68°C** — the active fan handles 45W easily with FCBGA's low θ_jc
- **Cost is dominated by silicon** — the 4nm die with NRE amortization drives the BOM

### Step 2: BOM Breakdown

Where does the weight come from?

```bash
branes swap bom \
    --area 220 --power 45 --process 4 \
    --package FCBGA --cooling active_fan --enclosure aluminum
```

**Expected Output:**

```
                  System BOM Breakdown
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━┓
┃ Component           ┃ Level   ┃ Wt (g)   ┃ Vol(cm³) ┃ Cost($) ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━┩
│ die                 │ die     │     0.40 │     0.16 │ 4158.92 │
│ FCBGA-256           │ package │     1.93 │     1.24 │    3.00 │
│ pcb                 │ pcb     │     3.04 │     1.53 │    5.72 │
│ heatsink-active_fan │ system  │   135.00 │    90.00 │    0.00 │
│ enclosure-aluminum  │ system  │   120.93 │    55.67 │    0.88 │
├─────────────────────┼─────────┼──────────┼──────────┼─────────┤
│ TOTAL               │         │   261.30 │   148.60 │ 4168.52 │
└─────────────────────┴─────────┴──────────┴──────────┴─────────┘
```

**Insight:** The heatsink (135 g) and enclosure (121 g) dominate weight — the silicon, package, and PCB together are under 6 g. For weight reduction, material choice and cooling design matter far more than die size.

### Step 3: Check Against NUC Budgets

```bash
branes swap check \
    --area 220 --power 45 --process 4 \
    --package FCBGA --cooling active_fan --enclosure aluminum \
    --max-weight 800 --max-volume 700 --max-power 65 --max-cost 5000
```

**Expected Output:**

```
          SWaP-C Scorecard
┏━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━┳━━━━━━┳━━━━━━━━━┓
┃ Metric        ┃ Value    ┃ Budget  ┃ Util ┃ Verdict ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━╇━━━━━━╇━━━━━━━━━┩
│ power_watts   │    45.00 │   65.00 │  69% │ PASS    │
│ cost_usd      │  4168.52 │ 5000.00 │  83% │ WARNING │
│ weight_grams  │   261.30 │  800.00 │  33% │ PASS    │
│ volume_cm3    │   148.60 │  700.00 │  21% │ PASS    │
└───────────────┴──────────┴─────────┴──────┴─────────┘
 Thermal: Tj=57°C (max 125°C), active_fan cooling — PASS
 Overall: MARGINAL
```

The design passes all hard constraints but gets a **WARNING** on cost (83% utilization). This is expected for a 4nm process — NRE amortization at production volumes drives the die cost. At higher volumes (100K+), cost drops significantly.

:::tip
Use `--json-output` for CI pipelines. The command returns exit code 1 on FAIL, making it easy to gate deployments on SWaP-C compliance.
:::

### Step 4: What If We Used Magnesium?

NUC designs sometimes use magnesium alloy for lighter chassis. Let's compare:

```bash
branes swap compare \
    --area 220 --power 45 --process 4 \
    --left "FCBGA,active_fan,aluminum" \
    --right "FCBGA,active_fan,magnesium"
```

**Expected Output:**

```
       Configuration Comparison (220mm² / 4nm / 45.0W)
┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Metric       ┃ FCBGA/active_fan/aluminum ┃ FCBGA/active_fan/magnesium ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Weight (g)   │                    261.30 │                     183.35 │
│ Volume (cm³) │                    148.60 │                     148.60 │
│ Cost ($)     │                   4168.52 │                   4168.84 │
│ Tj (°C)      │                57 (68°C)  │                 57 (68°C)  │
│ Thermal      │                      PASS │                       PASS │
│ Dims (mm)    │                  43×43×70 │                   43×43×70 │
└──────────────┴───────────────────────────┴────────────────────────────┘
 Delta: right is -30% weight, +0% volume, +0% cost, +0°C Tj
```

Switching to magnesium saves 30% on weight (78 g lighter) with negligible cost difference — identical thermal performance since the cooling path goes through the heatsink, not the enclosure.

---

## Example 2: NVIDIA Jetson Orin Nano for Drone Perception

The Jetson Orin Nano is a 15W SoM for autonomous machines — drones, AMRs, inspection robots. It packs 40 TOPS of INT8 inference in an 88 g module. For our drone, we need active cooling to sustain 15W in a compact airframe.

**Target specs:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Die area | 160 mm² | Orin Nano SoC (6× A78AE + 1024 CUDA cores) |
| Process | 8 nm | Samsung 8LPP |
| TDP | 15 W | Maximum power mode |
| Package | BGA | Standard for SoM mounting |
| Cooling | Active fan | Sustained 15W in enclosed drone airframe |
| Enclosure | ABS plastic | Lightweight for drone payload |

**Drone payload budgets:** 200 g for compute module + cooling, 150 cm³ volume.

### Step 1: Quick Estimate

```bash
branes swap estimate \
    --area 160 --power 15 --process 8 \
    --package BGA --cooling active_fan --enclosure abs_plastic \
    --ambient-temp 45
```

We set ambient to 45°C — realistic for an enclosed drone airframe in summer conditions.

**Expected Output:**

```
╭── SWaP-C Estimate: 160mm² / 8nm / BGA / active_fan ─╮
│  Weight:    76.8 g                                    │
│  Volume:    44.3 cm³                                  │
│  Cost:      $1098.42                                  │
│  Dims:      31×31×41 mm                               │
│  Thermal:   Tj=63°C (margin: 62°C) ✓                 │
╰──────────────────────────────────────────────────────╯
```

At 77 g and 44 cm³, this fits comfortably in the drone's payload budget. The active fan keeps Tj well below limits even at 45°C ambient.

### Step 2: Can We Use Passive Cooling Instead?

Drones benefit from passive cooling: no fan noise, no vibration, no moving parts to fail. But can 15W be passively cooled?

```bash
branes swap compare \
    --area 160 --power 15 --process 8 \
    --left "BGA,active_fan,abs_plastic" \
    --right "BGA,passive,abs_plastic"
```

**Expected Output:**

```
      Configuration Comparison (160mm² / 8nm / 15.0W)
┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Metric       ┃ BGA/active_fan/plastic   ┃ BGA/passive/plastic   ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━┩
│ Weight (g)   │                    76.80 │                148.50 │
│ Volume (cm³) │                    44.30 │                 80.20 │
│ Cost ($)     │                  1098.42 │               1098.42 │
│ Tj (°C)      │                63 (62°C) │             220 (-95°C)│
│ Thermal      │                     PASS │                  FAIL │
│ Dims (mm)    │                31×31×41  │               38×38×53 │
└──────────────┴──────────────────────────┴───────────────────────┘
 Delta: right is +93% weight, +81% volume, +0% cost, +157°C Tj
```

**Passive cooling fails** at 15W — junction temperature would reach 220°C, far beyond the 125°C limit. The passive heatsink also nearly doubles the weight. For this drone, active fan cooling is the right choice.

:::note
At a lower power mode (7W), passive cooling becomes feasible. You can test this by changing `--power 7` in the comparison.
:::

### Step 3: Check Against Drone Budgets

```bash
branes swap check \
    --area 160 --power 15 --process 8 \
    --package BGA --cooling active_fan --enclosure abs_plastic \
    --max-weight 200 --max-volume 150 --max-power 20 --max-cost 1500 \
    --ambient-temp 45
```

**Expected Output:**

```
          SWaP-C Scorecard
┏━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━┳━━━━━━┳━━━━━━━━━┓
┃ Metric        ┃ Value    ┃ Budget  ┃ Util ┃ Verdict ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━╇━━━━━━╇━━━━━━━━━┩
│ power_watts   │    15.00 │   20.00 │  75% │ PASS    │
│ cost_usd      │  1098.42 │ 1500.00 │  73% │ PASS    │
│ weight_grams  │    76.80 │  200.00 │  38% │ PASS    │
│ volume_cm3    │    44.30 │  150.00 │  30% │ PASS    │
└───────────────┴──────────┴─────────┴──────┴─────────┘
 Thermal: Tj=63°C (max 125°C), active_fan cooling — PASS
 Overall: PASS
```

All green. The design uses 38% of the weight budget and 30% of the volume budget, leaving room for cameras, batteries, and structural frame.

### Step 4: Design Space Exploration

Now let's find the full Pareto front — what are the best possible designs across all 6 objectives?

```bash
branes swap explore \
    --goal "drone perception SoC" \
    --power 20 --weight 200 --volume 150 \
    --fast --workers 4
```

**Expected Output:**

```
SWaP-C Design Space Exploration
Goal: drone perception SoC
Constraints: {'max_power_watts': 20.0, 'max_weight_grams': 200.0, 'max_volume_cm3': 150.0}
Objectives: 6 (power, latency, area, cost, weight, volume)

Optimization Complete
Total evaluations: 1344
Layers used: map_elites
Pareto front size: 8
Hypervolume: 2.8341
```

### Step 5: Inspect the Pareto Front

```bash
branes swap show-front --top 5
```

The output shows designs spanning different process nodes, packages, and cooling strategies — each representing a different tradeoff between power, latency, size, weight, volume, and cost.

### Step 6: Explain a Tradeoff

Compare the knee point (best balance) against the lightest design:

```bash
branes swap explain --points 0,2
```

**Expected Output:**

```
Tradeoff: Design #0 vs Design #2

Objective Changes
 Objective      Design #0  Design #2  Delta    Change %
 power_watts       2.83       0.65    -2.18     -77.0%
 latency_ms       15.37      36.57   +21.20    +137.9%
 area_mm2         65.75       7.35   -58.40     -88.8%
 cost_usd        435.12    2082.50  +1647.38   +378.6%
 weight_grams     18.10       9.10     -9.00    -49.7%
 volume_cm3       10.30       4.40     -5.90    -57.3%

Parameter Changes
 Parameter       Design #0  Design #2
 cooling_type    active_fan active_fan
 package_type    WLCSP      FCBGA
 process_nm      40         16
```

Design #2 is 50% lighter and 77% lower power — but costs 4× more (16nm process) and is 138% slower. The optimizer found this tradeoff automatically; whether it's worthwhile depends on your priorities.

---

## Typical Workflow

```bash
# 1. Quick estimate to validate feasibility
branes swap estimate --area 160 --power 15 --process 8

# 2. Compare packaging options
branes swap compare --area 160 --power 15 --process 8 \
    --left "BGA,active_fan,abs_plastic" \
    --right "QFN,passive,abs_plastic"

# 3. Check against budgets
branes swap check --area 160 --power 15 --process 8 \
    --package BGA --cooling active_fan --enclosure abs_plastic \
    --max-weight 200 --max-volume 150

# 4. Full optimization
branes swap explore -g "drone SoC" --power 20 --weight 200 --volume 150 --fast

# 5. Inspect and compare results
branes swap show-front --top 5
branes swap explain --points 0,2
```

## JSON Output for Automation

All commands support `--json-output` for pipeline integration:

```bash
# Get machine-readable estimate
branes swap estimate --area 160 --power 15 --process 8 --json-output | jq .weight_grams

# CI gate: exits 1 if over budget
branes swap check --area 160 --power 15 --process 8 \
    --max-weight 200 --json-output || echo "SWaP-C budget exceeded!"
```

## Next Steps

- [SWaP-C Feature Overview](/features/swap-analysis/) for methodology details
- [Design Optimization](/features/design-optimization/) for the underlying MOO engine
- [Constraint Checking](/features/constraint-checking/) for model-level pass/fail checks
- [CLI Reference](/reference/cli/) for all `swap` command options
