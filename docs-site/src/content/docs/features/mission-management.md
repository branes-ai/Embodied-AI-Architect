---
title: Mission Management
description: Create, manage, and evolve persistent design missions that tie together the entire design lifecycle.
---

The **Mission** is the central entity in the Branes design workflow. It captures
your goal, constraints, selected components, and design state — persisting across
sessions so you can resume, compare, and audit the full lifecycle.

## Why Missions?

Before missions, every command was stateless. Running `design qualify` produced
a dict that you had to manually copy into `design plan`. Sensor selections lived
in your head. Optimization results vanished when the session ended.

With missions:
- **Persistent state** — your goal, constraints, and selections survive across sessions
- **Connected commands** — `--mission` flag loads context into qualify, plan, swap, optimize
- **Auditable** — every stage records what was decided and why
- **Explorable** — fork a mission to try a design variant without losing the original

## Lifecycle

A mission progresses through 5 stages:

```text
DRAFT → QUALIFIED → DESIGNED → OPTIMIZED → VALIDATED
  │         │           │           │           │
  │    design qualify   │    optimize explore   │
  │                design plan            validate mission
  mission new                        
```

| Status | What happened | Next step |
|--------|---------------|-----------|
| **draft** | Mission created with a goal | `design qualify --mission` |
| **qualified** | Goal refined, constraints derived, platform matched | `design plan --mission` |
| **designed** | Task graph generated, architecture composed | `optimize explore --mission` |
| **optimized** | Pareto frontier explored, best design selected | `validate mission` |
| **validated** | All checks pass | Deploy or report |

## Mission Fields

| Field | Set by | Description |
|-------|--------|-------------|
| `id` | `mission new` | Unique identifier |
| `name` | `mission new` | Human-readable name |
| `goal` | `mission new` / `design qualify` | Natural language design objective |
| `status` | Automatic | Current lifecycle stage |
| `constraints` | `design qualify` | Power, latency, cost, area limits |
| `platform_id` | `design qualify` | Matched platform from registry |
| `use_case` | `design qualify` | Application type |
| `spec` | `spec set` / `design qualify` | Full SystemSpec with subsystem details |
| `selected_sensors` | `sensor select` | Sensor IDs chosen for the mission |
| `selected_actuators` | `actuator select` | Actuator IDs chosen for the mission |
| `selected_compute` | Manual | Compute platform ID |
| `selected_models` | Manual | ML model IDs |
| `design_state` | `design plan` | Full SoC design state (task graph, PPA) |
| `optimization_history` | `optimize` / `swap` | Trail of optimization runs |

## Commands

### Create a Mission

```bash
branes mission new vineyard-sprayer --goal "Autonomous vineyard sprayer with weed detection"
```

The `--goal` flag sets the initial design objective. You can also set it later
with `mission edit`.

### List Missions

```bash
branes mission list
```

```text
Missions
┌──────────────────┬───────────┬─────────────────────────────────────────┐
│ ID               │ Status    │ Goal                                    │
├──────────────────┼───────────┼─────────────────────────────────────────┤
│ vineyard-sprayer │ qualified │ Autonomous vineyard sprayer with weed…  │
│ my-drone         │ designed  │ Drone perception SoC for YOLO at 30fps… │
└──────────────────┴───────────┴─────────────────────────────────────────┘
```

### Show Mission Details

```bash
branes mission show vineyard-sprayer
```

Displays all fields: goal, status, constraints, selected components,
design state summary, and timestamps.

### Edit a Mission

```bash
branes mission edit vineyard-sprayer --goal "Updated goal text"
branes mission edit vineyard-sprayer --status qualified
```

### Delete a Mission

```bash
branes mission delete vineyard-sprayer
```

### Refine with LLM

Use the AI to suggest constraint improvements based on the current mission state:

```bash
branes mission refine vineyard-sprayer
```

This analyzes the mission's goal, selected components, and constraints,
then suggests refinements to improve feasibility or performance.

### Fork a Mission

Create a variant to explore an alternative design without modifying the original:

```bash
branes mission fork vineyard-sprayer vineyard-sprayer-v2
```

The fork copies all fields (goal, constraints, selections, spec) into a new
mission. You can then modify the fork independently.

## Connecting Missions to Commands

The `--mission` flag loads constraints and context from a mission into any
command that needs them:

```bash
# Qualification reads/writes the mission
branes design qualify --mission vineyard-sprayer --auto

# Planning loads goal + constraints from mission, saves task graph back
branes design plan --mission vineyard-sprayer --static

# SWaP-C analysis uses mission constraints
branes swap check --mission vineyard-sprayer --area 50 --power 5

# Optimization loads constraints from mission
branes optimize explore --mission vineyard-sprayer --fast

# MCP analysis loads selected compute/models from mission
branes mcp analyze --mission vineyard-sprayer
```

Explicit CLI flags always override mission values. For example,
`--power 10 --mission my-drone` uses 10W even if the mission says 5W.

## Example: Vineyard Sprayer

```bash
# 1. Create the mission
branes mission new vineyard-sprayer \
  --goal "Autonomous vineyard sprayer with weed detection, <15W compute, <100ms latency"

# 2. Qualify — derives constraints from the goal
branes design qualify --mission vineyard-sprayer --auto

# 3. Select sensors
branes sensor search "multispectral camera for weed detection"
branes sensor select vineyard-sprayer visual.multispectral_camera inertial.imu_6dof position.gps_rtk

# 4. Select actuators
branes actuator search "sprayer pump"
branes actuator select vineyard-sprayer fluid.sprayer

# 5. Check budgets
branes sensor budget vineyard-sprayer
branes sensor fusion vineyard-sprayer

# 6. Generate design plan
branes design plan --mission vineyard-sprayer --static

# 7. Synthesize
branes synthesize system vineyard-sprayer

# 8. Validate
branes validate mission vineyard-sprayer

# 9. Fork to try a lower-power variant
branes mission fork vineyard-sprayer vineyard-low-power
branes mission edit vineyard-low-power --goal "Same but under 5W compute"
```

## Storage

Missions are stored as JSON files in `.branes/missions/<id>/manifest.json`.
Writes are atomic (temp file + rename) to prevent corrupt reads. The store
auto-generates UUIDs for unnamed missions.

## See Also

- [Quickstart](/getting-started/quickstart/) — mission workflow in 7 steps
- [CLI Reference: mission](/reference/cli/#mission) — complete command reference
- [Sensor & Actuator Selection](/features/sensor-actuator-selection/) — component selection workflow
