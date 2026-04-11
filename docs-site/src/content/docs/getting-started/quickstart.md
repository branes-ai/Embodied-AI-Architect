---
title: Quickstart
description: Get up and running with the Branes Embodied AI Platform in 5 minutes.
---

This guide walks you through the mission-driven design lifecycle — from defining
what you want to build, through component selection, to system synthesis.

## Mission-Driven Workflow

### 1. Create a Mission

> See also: [Mission Management](/features/mission-management/) | [CLI: mission](/reference/cli/#mission)

Every design starts with a mission — a persistent entity that captures your goal,
constraints, selected components, and design state across sessions.

```bash
branes mission new my-drone --goal "Drone perception SoC for YOLO at 30fps under 5W"
```

```text
Created mission: my-drone
  Goal: Drone perception SoC for YOLO at 30fps under 5W
  Status: draft
```

### 2. Qualify the Goal

> See also: [CLI: design qualify](/reference/cli/#design-qualify)

The qualifier walks through structured questions to derive specific constraints
from your goal. Use `--auto` for default answers:

```bash
branes design qualify --mission my-drone --auto
```

```text
Design Qualification — Drone perception SoC

  Tangibility:  ████████░░  80%
  Dimensions:   perception ✓  power ✓  platform ✓  environment ✓

  Mission 'my-drone' updated → qualified
  Next: branes design plan --mission my-drone
```

### 3. Search and Select Sensors

> See also: [Sensor & Actuator Selection](/features/sensor-actuator-selection/) | [CLI: sensor](/reference/cli/#sensor)

Browse the sensor registry (80+ sensors with TF-IDF search) and select
components for your mission:

```bash
# Search for sensors
branes sensor search "stereo camera for VIO"

# Select sensors for the mission
branes sensor select my-drone visual.stereo_camera inertial.imu_6dof position.gps_l1
```

```text
Added 3 sensor(s) to mission 'my-drone':
  + visual.stereo_camera (Stereo Camera)
  + inertial.imu_6dof (6-DOF IMU)
  + position.gps_l1 (GPS L1)
```

### 4. Check Budgets and Fusion

> See also: [CLI: sensor budget](/reference/cli/#sensor-budget) | [CLI: sensor fusion](/reference/cli/#sensor-fusion)

Verify that your selected sensors fit within power/weight/cost constraints
and get fusion strategy recommendations:

```bash
# Power, weight, cost budget
branes sensor budget my-drone

# Sensor fusion recommendations
branes sensor fusion my-drone
```

```text
Sensor Fusion — my-drone

  Selected categories: inertial, position, visual

  Recommendations:
    • Visual-Inertial Odometry (VIO) — fuse camera + IMU for ego-motion
    • INS/GNSS fusion — fuse IMU + GPS for robust localization
    • Full SLAM stack — VIO + GPS for global + local mapping
```

### 5. Generate a Design Plan

> See also: [CLI: design plan](/reference/cli/#design-plan) | [Design Optimization](/features/design-optimization/)

The planner decomposes your goal into a task graph (DAG of specialist agents).
Use `--static` for a demo plan without an API key:

```bash
branes design plan --mission my-drone --static
```

```text
Task Graph (7 tasks, 4 stages)
  t1: Analyze workload          → workload_analyzer
  t2: Enumerate hardware         → hw_explorer        [after t1]
  t3: Compose architecture       → architecture_composer [after t2]
  t4: Explore Pareto frontier    → moo_explorer        [after t2]
  t5: Assess PPA metrics         → ppa_assessor        [after t3, t4]
  t6: Review design              → critic              [after t5]
  t7: Generate report            → report_generator    [after t6]

  Mission 'my-drone' updated → designed
```

### 6. Synthesize the System

> See also: [CLI: synthesize](/reference/cli/#synthesize)

View a summary of the composed system from all selected components:

```bash
branes synthesize system my-drone
```

```text
System Synthesis — my-drone

  Sensors:    visual.stereo_camera, inertial.imu_6dof, position.gps_l1
  Actuators:  none
  Compute:    none
  Models:     none
  Constraints: {'max_power_watts': 5.0, 'max_latency_ms': 33.0}
```

### 7. Validate

> See also: [CLI: validate](/reference/cli/#validate)

Run all validation checks on the mission:

```bash
branes validate mission my-drone
```

---

## Model-First Workflow

If you already have a trained model and want to evaluate it against hardware,
you can skip the mission workflow and analyze directly:

### Analyze a Model

```bash
branes analyze yolov8n.pt
```

This outputs model architecture summary, parameter count, memory footprint,
and computational requirements (FLOPs).

### Check Hardware Fit

```bash
branes mcp analyze yolov8n jetson_orin_nano
```

You'll get predicted latency, memory utilization, bottleneck classification,
and hardware utilization percentage.

### Compare Hardware Options

```bash
branes mcp compare yolov8n jetson_orin_nano,jetson_orin_agx,coral_edge_tpu
```

### Interactive Chat

For exploratory analysis, use the interactive chat:

```bash
export ANTHROPIC_API_KEY=your-key-here
branes chat
```

Ask questions in natural language:

```text
You: Can I run YOLOv8s at 30fps on a Jetson Orin Nano under 5W?
```

---

## Next Steps

- [Sensor & Actuator Selection](/features/sensor-actuator-selection/) — deep dive into component search and comparison
- [Mission Management](/features/mission-management/) — mission lifecycle, refinement, and forking
- [SWaP-C Analysis](/features/swap-analysis/) — system-level size, weight, power, cost analysis
- [Design Optimization](/features/design-optimization/) — multi-objective Pareto exploration
- [CLI Reference](/reference/cli/) — complete command reference for all 30 command groups
- [Hardware Catalog](/catalog/hardware/) — browse 266 platforms and 62 product configs
