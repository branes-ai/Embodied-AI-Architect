---
title: Mission-Driven Workflow Tutorial
description: End-to-end walkthrough of designing a vineyard sprayer drone from mission creation through validation.
---

This tutorial walks through the complete mission-driven design lifecycle using a
**vineyard sprayer drone** as the example. By the end you'll have a fully specified
system with sensor/actuator selections, a design plan, and validation results.

## Prerequisites

```bash
pip install -e ".[dev]"
branes --help
```

No API key needed — we use `--auto` and `--static` flags throughout.

---

## Step 1: Create a Mission

> See also: [Mission Management](/features/mission-management/)

Every design starts with a named mission that persists across sessions.

```bash
branes mission new vineyard-sprayer \
  --goal "Autonomous vineyard sprayer with weed detection, <15W compute, <100ms latency"
```

```text
Created mission: vineyard-sprayer
  Goal: Autonomous vineyard sprayer with weed detection, <15W compute, <100ms latency
  Status: draft
```

**Mission state after this step:**
- Status: `draft`
- Goal: set
- Everything else: empty

---

## Step 2: Qualify the Goal

> See also: [CLI: design qualify](/reference/cli/#design-qualify)

The qualifier walks through structured questions to derive specific constraints.
Use `--auto` to accept defaults:

```bash
branes design qualify --mission vineyard-sprayer --auto
```

```text
Design Qualification — Autonomous vineyard sprayer

  Tangibility:  ████████░░  80%
  Dimensions:   perception ✓  power ✓  platform ✓  environment ✓

Goal qualified. Design inputs:
  Goal:     Autonomous vineyard sprayer with weed detection
  Platform: aerial.agricultural_sprayer
  Use case: precision_agriculture
  max_power_watts: 15.0
  max_latency_ms: 100.0

  Mission 'vineyard-sprayer' updated → qualified
```

**Mission state after this step:**
- Status: `qualified`
- `platform_id`: `aerial.agricultural_sprayer`
- `constraints`: `{max_power_watts: 15.0, max_latency_ms: 100.0}`
- `use_case`: `precision_agriculture`

---

## Step 3: Search and Select Sensors

> See also: [Sensor & Actuator Selection](/features/sensor-actuator-selection/)

Search the registry (80 sensors, TF-IDF ranked) for what your sprayer needs:

```bash
branes sensor search "multispectral camera for weed detection"
```

```text
                    Search: multispectral camera for weed detection
┌────────────────────────────────────┬──────────────────────────┬──────────┬───────┐
│ ID                                 │ Name                     │ Category │ Score │
├────────────────────────────────────┼──────────────────────────┼──────────┼───────┤
│ visual.multispectral_camera        │ Multispectral Camera     │ visual   │ 1.000 │
│ visual.hyperspectral_camera        │ Hyperspectral Camera     │ visual   │ 0.542 │
│ visual.rgb_camera                  │ RGB Camera               │ visual   │ 0.321 │
└────────────────────────────────────┴──────────────────────────┴──────────┴───────┘
```

Select sensors for the mission:

```bash
branes sensor select vineyard-sprayer \
  visual.multispectral_camera \
  inertial.imu_6dof \
  position.gps_rtk
```

```text
Added 3 sensor(s) to mission 'vineyard-sprayer':
  + visual.multispectral_camera (Multispectral Camera)
  + inertial.imu_6dof (6-DOF IMU)
  + position.gps_rtk (RTK GPS)
```

---

## Step 4: Search and Select Actuators

```bash
branes actuator search "sprayer pump for agriculture"
```

```text
                    Search: sprayer pump for agriculture
┌────────────────────────┬──────────────────────┬──────────┬───────┐
│ ID                     │ Name                 │ Category │ Score │
├────────────────────────┼──────────────────────┼──────────┼───────┤
│ fluid.sprayer          │ Agricultural Sprayer │ fluid    │ 1.000 │
│ fluid.pump             │ Centrifugal Pump     │ fluid    │ 0.456 │
└────────────────────────┴──────────────────────┴──────────┴───────┘
```

```bash
branes actuator select vineyard-sprayer fluid.sprayer
```

```text
Added 1 actuator(s) to mission 'vineyard-sprayer':
  + fluid.sprayer (Agricultural Sprayer)
```

**Mission state after steps 3-4:**
- `selected_sensors`: `[visual.multispectral_camera, inertial.imu_6dof, position.gps_rtk]`
- `selected_actuators`: `[fluid.sprayer]`

---

## Step 5: Check Budgets

Aggregate power, weight, and cost across all selected sensors:

```bash
branes sensor budget vineyard-sprayer
```

```text
                      Sensor Budget — vineyard-sprayer
┌──────────────────────────────┬────────────────────────┬───────┬────────┬───────┐
│ ID                           │ Name                   │ Power │ Weight │ Cost  │
├──────────────────────────────┼────────────────────────┼───────┼────────┼───────┤
│ visual.multispectral_camera  │ Multispectral Camera   │ 5.0W  │ 200g   │ $3000 │
│ inertial.imu_6dof            │ 6-DOF IMU              │ 0.1W  │ 5g     │ $25   │
│ position.gps_rtk             │ RTK GPS                │ 1.0W  │ 30g    │ $200  │
├──────────────────────────────┼────────────────────────┼───────┼────────┼───────┤
│ TOTAL                        │                        │ 6.1W  │ 235g   │ $3225 │
└──────────────────────────────┴────────────────────────┴───────┴────────┴───────┘
```

6.1W sensor subsystem leaves 8.9W headroom from the 15W compute budget.

---

## Step 6: Fusion Analysis

Get recommendations for how to combine your selected sensor modalities:

```bash
branes sensor fusion vineyard-sprayer
```

```text
Sensor Fusion — vineyard-sprayer

  Selected categories: inertial, position, visual

  Recommendations:
    • Visual-Inertial Odometry (VIO) — fuse camera + IMU for ego-motion
    • INS/GNSS fusion — fuse IMU + GPS for robust localization
    • Full SLAM stack — VIO + GPS for global + local mapping
```

The system recommends VIO + INS/GNSS fusion — a solid navigation stack
for outdoor agricultural operations.

---

## Step 7: Generate a Design Plan

> See also: [Design Optimization](/features/design-optimization/)

The planner decomposes the goal into a task graph of specialist agents.
Use `--static` for a demo plan without an API key:

```bash
branes design plan --mission vineyard-sprayer --static
```

```text
Platform context loaded: aerial.agricultural_sprayer

Task Graph (7 tasks, 4 stages)
  t1: Analyze workload          → workload_analyzer
  t2: Enumerate hardware         → hw_explorer           [after t1]
  t3: Compose architecture       → architecture_composer [after t2]
  t4: Explore Pareto frontier    → moo_explorer          [after t2]
  t5: Assess PPA metrics         → ppa_assessor          [after t3, t4]
  t6: Review design              → critic                [after t5]
  t7: Generate report            → report_generator      [after t6]

Mission 'vineyard-sprayer' updated → designed
```

**Mission state after this step:**
- Status: `designed`
- `design_state`: full task graph + platform context

---

## Step 8: Synthesize the System

View a summary of all selected components and constraints:

```bash
branes synthesize system vineyard-sprayer
```

```text
System Synthesis — vineyard-sprayer

  Sensors:    visual.multispectral_camera, inertial.imu_6dof, position.gps_rtk
  Actuators:  fluid.sprayer
  Compute:    none
  Models:     none
  Constraints: {'max_power_watts': 15.0, 'max_latency_ms': 100.0}

  Missing selections: compute
  Use 'branes select compute vineyard-sprayer' to add components.
```

Generate a Mermaid architecture diagram:

```bash
branes synthesize architecture vineyard-sprayer
```

---

## Step 9: Validate

Run all validation checks on the mission:

```bash
branes validate mission vineyard-sprayer
```

This checks constraints, completeness, safety, and scheduling feasibility.

---

## Step 10: Explore Variants

Fork the mission to try a lower-power alternative:

```bash
branes mission fork vineyard-sprayer vineyard-low-power
branes mission edit vineyard-low-power --goal "Same sprayer but under 5W compute budget"
```

Now you can run the same workflow on the fork without affecting the original.

---

## Summary

The lifecycle follows this flow:

```text
mission new        Define the system goal
    │
design qualify     Derive constraints from the goal
    │
sensor/actuator    Search, select, compare components
select
    │
sensor budget      Validate power/bandwidth/weight budgets
sensor fusion      Analyze sensor fusion opportunities
    │
design plan        Generate the compute pipeline
    │
synthesize system  Produce architecture, BOM, interconnects
    │
validate mission   Verify constraints, completeness, safety
    │
mission fork       Explore design variants
```

Each step enriches the mission state. You can revisit any step, change selections,
and re-run downstream stages to explore the design space iteratively.

## Next Steps

- [Mission Management](/features/mission-management/) — lifecycle, refinement, forking
- [Sensor & Actuator Selection](/features/sensor-actuator-selection/) — deep dive into registries
- [SWaP-C Analysis](/features/swap-analysis/) — system-level analysis
- [Design Optimization](/features/design-optimization/) — multi-objective Pareto exploration
- [CLI Reference](/reference/cli/) — complete command reference
