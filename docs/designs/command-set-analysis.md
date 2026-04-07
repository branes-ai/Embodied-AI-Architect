# Command Set Analysis: Embodied AI Design Workflow

**Date**: 2026-04-07
**Status**: Design analysis — gap assessment and recommendations

## Purpose

The branes platform must support the full design workflow for the **computational
subsystem of an embodied AI system**. An embodied AI system has four orthogonal
component classes that all need design and design-time optimization:

1. **Application** — the code that delivers mission objectives (planning, scheduling, control logic)
2. **Sensors** — perception inputs supporting mission requirements (cameras, LiDAR, radar, IMU, etc.)
3. **Compute** — model-based or AI-based processing (perception models, decision models, control models)
4. **Actuators** — physical outputs supporting mission requirements (motors, grippers, valves, displays)

These four classes interact through tight constraints: sensor data rates set the
compute load; compute latency sets the actuator control rate; actuator power dominates
the system power budget; mission requirements constrain everything.

A complete design workflow needs commands that answer questions at every level
of abstraction, from high-level mission definition down to detailed component
selection, system synthesis, optimization, and validation.

This document hypothesizes the **ideal command set** for that workflow,
articulates how each command would be used in concrete design scenarios, then
compares to the **existing command set** and identifies gaps.

---

## 1. Hypothesized Ideal Command Set

Organized by **design lifecycle phase**, with each command's purpose and usage.

### Phase 1: Mission Definition

What the system must accomplish.

```
branes mission new [name]               # Start a new mission spec interactively
branes mission show [name]              # Display mission details
branes mission edit [name]              # Edit mission parameters
branes mission validate [name]          # Check mission completeness
branes mission requirements [name]      # Extract quantitative requirements
branes mission decompose [name]         # Break mission into subsystem requirements
```

**Workflow example — designing a vineyard sprayer:**
```bash
branes mission new vineyard-sprayer
# Interactive: platform=ground_wheeled, environment=outdoor_farmyard,
#              tasks=[weed_detection, precision_spraying, row_following],
#              endurance=8h, payload=200L, terrain=sloped_vineyard

branes mission requirements vineyard-sprayer
# Extracts: max_speed=12km/h, weed_detection_latency<25ms,
#           gps_accuracy<2cm, spray_drift<1m, IP65, -10°C to 50°C

branes mission decompose vineyard-sprayer
# Output: perception_subsystem={tasks, FPS, accuracy targets}
#         compute_subsystem={TOPS, memory, power budget}
#         actuator_subsystem={DOF, control_rate, payload, speed}
#         sensor_subsystem={modalities, resolution, FOV}
```

### Phase 2: Subsystem Design

Each subsystem gets dedicated commands.

```
# Sensor subsystem
branes sensor select <mission> [--task]    # Recommend sensors for a perception task
branes sensor compare <s1> <s2> ...         # Side-by-side sensor comparison
branes sensor budget <mission>              # Sensor power, weight, data rate budget
branes sensor list [--modality]             # Browse sensor catalog
branes sensor show <sensor_id>              # Detailed sensor specs
branes sensor fusion <mission>              # Recommend sensor fusion strategy

# Actuator subsystem
branes actuator select <mission>            # Recommend actuators for mission
branes actuator compare <a1> <a2> ...       # Compare actuator options
branes actuator budget <mission>            # Power, weight, control bandwidth budget
branes actuator list [--type]               # Browse actuator catalog
branes actuator control-rate <actuator>     # Required control loop rate

# Compute subsystem
branes compute select <mission>             # Recommend compute hardware
branes compute compare <hw1> <hw2> ...      # Side-by-side hardware comparison
branes compute budget <mission>             # Power/thermal/area budget
branes compute place <model> --hw <target>  # Map model layers to accelerators

# Application/Model subsystem
branes app analyze <mission>                # Application complexity analysis
branes model select <task>                  # Recommend ML models for a task
branes model compare <m1> <m2> ...          # Compare model accuracy/cost
branes model fit <model> --hw <target>      # Check if model fits on hardware
branes model latency <model> --hw <target>  # Predict latency on hardware
```

**Workflow example — vineyard sprayer continued:**
```bash
# Pick sensors for weed detection
branes sensor select vineyard-sprayer --task weed_detection
# Recommends: RGB camera 1920x1080 @ 30fps, multispectral 5-band @ 10fps
# Considers: lighting variability, plant occlusion, ground sampling distance

# Pick compute hardware
branes compute select vineyard-sprayer
# Recommends: Jetson Orin NX (15W, 100 TOPS), justified by:
#   - 30fps × 12 cameras × 1080p inference budget
#   - GPS+IMU sensor fusion at 100Hz
#   - Spray nozzle solenoid control at 50Hz
#   - <25ms perception-to-actuator latency

# Check if a specific model fits
branes model fit yolov8m --hw jetson_orin_nx
# PASS: 18ms latency (target <25ms), 6.2GB memory (target 8GB)

# Compare alternatives
branes compute compare jetson_orin_nx jetson_orin_nano coral_dev_board
# Table: power, latency, throughput, cost, thermal, mature SDK
```

### Phase 3: System Synthesis

Compose the full system from selected components.

```
branes synthesize system <mission>          # Compose full system from components
branes synthesize architecture <mission>    # Generate architecture diagram
branes synthesize bom <mission>             # Generate hierarchical bill of materials
branes synthesize cabling <mission>         # Inter-component data/power connections
branes synthesize thermal <mission>         # Thermal map and cooling requirements
branes synthesize software <mission>        # Software stack composition
```

**Workflow example:**
```bash
branes synthesize system vineyard-sprayer
# Composes: platform + sensors + compute + actuators + power + comms
# Generates: SystemSpec with all subsystem fields populated

branes synthesize bom vineyard-sprayer --volume 100
# Hierarchical BOM: dies → packages → boards → enclosure → vehicle
# Cost: $12,400/unit at 100 volume, $4,200 at 10K volume
# Weight: 580kg total, 23kg compute+sensors

branes synthesize architecture vineyard-sprayer
# Outputs: Mermaid diagram, JSON architecture spec
```

### Phase 4: Analysis

Quantitative system-level analysis.

```
branes analyze power <mission>              # System power breakdown
branes analyze latency <mission>            # End-to-end latency analysis
branes analyze thermal <mission>            # Thermal feasibility
branes analyze swap <mission>               # SWaP-C analysis
branes analyze safety <mission>             # Safety/redundancy analysis
branes analyze cost <mission> [--volume]    # BOM cost at volume
branes analyze bandwidth <mission>          # Data bandwidth (sensor → compute → actuator)
branes analyze scheduling <mission>         # Multi-rate scheduling feasibility
branes analyze failure <mission>            # Failure mode and effect analysis
```

**Workflow example:**
```bash
branes analyze latency vineyard-sprayer
# End-to-end: camera → ISP → CNN → fusion → planner → spray valve
# Total: 22ms (target <25ms)
# Bottleneck: CNN inference (14ms = 64% of total)

branes analyze power vineyard-sprayer
# Compute: 15W (Jetson Orin NX)
# Sensors: 8W (12 cameras × 0.5W + multispectral + IMU + GPS)
# Actuators: 480W (motors + spray pump)
# Comms: 5W (LTE + WiFi)
# Total: 508W → battery sized for 8h endurance

branes analyze scheduling vineyard-sprayer
# Verifies: perception 30Hz feasible, GPS 10Hz feasible,
#           spray control 50Hz feasible, all within compute budget
```

### Phase 5: Optimization

Multi-objective design space exploration.

```
branes optimize design <mission>            # Multi-objective MOO across all subsystems
branes optimize sensor <mission>            # Sensor configuration optimization
branes optimize compute <mission>           # Compute placement optimization
branes optimize swap <mission>              # SWaP-C Pareto exploration
branes optimize sensitivity <mission>       # Parameter sensitivity analysis
branes optimize tradeoff <mission>          # Explain trade-offs between Pareto points
```

**Workflow example:**
```bash
branes optimize design vineyard-sprayer --layers map_elites+bayesian
# Explores 5000 design points across:
#   variables: [sensor_count, model_size, hardware_choice, battery_size, ...]
#   objectives: [power, latency, cost, weight, accuracy]
# Returns: Pareto frontier with knee-point recommendation

branes optimize sensitivity vineyard-sprayer
# Tornado plot: model_size has 65% impact on latency,
#               battery_capacity has 80% impact on weight,
#               sensor_count has 40% impact on cost
```

### Phase 6: Validation

Verify the design meets all requirements.

```
branes validate mission <mission>           # Validate against mission requirements
branes validate constraints <mission>       # Check all constraints satisfied
branes validate scheduling <mission>        # Multi-rate scheduling proof
branes validate safety <mission>            # Safety integrity verification
branes validate thermal <mission>           # Thermal budget verification
branes validate failure <mission>           # FMEA completeness check
branes validate compliance <mission>        # Regulatory compliance check
```

### Phase 7: Reporting & Export

Generate artifacts for downstream consumers.

```
branes report design <mission>              # Comprehensive design report
branes report bom <mission>                 # Bill of materials
branes report compliance <mission>          # Regulatory compliance report
branes export rtl <mission>                 # RTL for custom SoC
branes export pipeline <mission>            # Deployment pipeline
branes export bom <mission>                 # BOM in standard format
branes export specs <mission>               # System specs for procurement
```

### Phase 8: Iteration & Refinement

Re-enter the loop with new information.

```
branes mission refine <mission>             # Update mission requirements
branes mission compare <m1> <m2>            # Compare two mission designs
branes mission fork <mission> [name]        # Fork a mission for what-if analysis
```

---

## 2. Mapping to Existing Commands

Here's how the existing 22 command groups map to the ideal command set:

### Direct matches (work as-is)

| Ideal Command | Existing Command | Status |
|---------------|------------------|--------|
| `optimize design` | `optimize explore` | ✅ Works |
| `optimize sensitivity` | `optimize sensitivity` | ✅ Works |
| `optimize tradeoff` | `optimize explain` | ✅ Works |
| `analyze swap` | `swap explore` / `swap check` | ✅ Works |
| `analyze sensitivity` | `swap sensitivity` | ✅ Works |
| `synthesize bom` | `swap bom` | ✅ Works |
| `compute compare` | `mcp compare` | ✅ Works |
| `compute select` | `mcp hardware` + `mcp analyze` | ⚠️ Partial — no direct "select" |
| `model latency` | `mcp latency` | ✅ Works |
| `model fit` | `mcp memory` + `mcp latency` | ⚠️ Partial — must compose |
| `analyze power` | `mcp energy` | ✅ Works (per-model only) |

### Partial matches (need extension)

| Ideal Command | Closest Existing | Gap |
|---------------|------------------|-----|
| `mission new` | `design qualify` | qualify is goal-text-driven, no persistent mission concept |
| `mission decompose` | `design plan` | plan generates a task DAG, not a subsystem requirement spec |
| `mission requirements` | `design new` (requirements wizard) | exists but disconnected from qualify/plan flow |
| `synthesize system` | `design synthesize` | synthesizes a *pipeline* from a YAML, not a full system from a mission |
| `synthesize architecture` | (none) | architecture only exists inside SoCDesignState |
| `validate constraints` | `swap check` | only validates SWaP-C, not full design |
| `report design` | `report view` | views existing reports, no comprehensive design report generator |

### Missing entirely

| Ideal Command | Status |
|---------------|--------|
| `mission *` (all) | ❌ No mission concept as first-class entity |
| `sensor *` (all) | ❌ Sensors are buried in spec subsystem, no dedicated commands |
| `actuator *` (all) | ❌ Actuators are buried in spec subsystem, no dedicated commands |
| `compute place` | ❌ Layer-to-accelerator mapping not exposed |
| `synthesize cabling` | ❌ No inter-component connection analysis |
| `synthesize thermal` | ❌ No system-level thermal map |
| `analyze bandwidth` | ❌ No data bandwidth flow analysis |
| `analyze scheduling` | ❌ Multi-rate scheduling feasibility (scheduling check is hidden in mcp/architecture tools) |
| `analyze failure` | ❌ No FMEA tooling |
| `validate scheduling` | ❌ No scheduling proof |
| `validate safety` | ❌ No safety integrity verification |
| `validate compliance` | ❌ No regulatory compliance check |
| `mission compare/fork` | ❌ No iteration support |
| `app analyze` | ❌ Application logic complexity not modeled |

---

## 3. Workflow Comparison

### Ideal Workflow: Vineyard Sprayer (8 commands)

```bash
branes mission new vineyard-sprayer
branes mission decompose vineyard-sprayer
branes sensor select vineyard-sprayer --task weed_detection
branes compute select vineyard-sprayer
branes synthesize system vineyard-sprayer
branes analyze latency vineyard-sprayer
branes optimize design vineyard-sprayer
branes report design vineyard-sprayer
```

**Each command operates on a named mission entity that persists across calls.**
The user thinks in terms of *the design*, not in terms of *files and sessions*.

### Existing Workflow: Same scenario (8+ commands, fragmented)

```bash
branes design qualify "vineyard sprayer for autonomous weeding"
# Q&A flow, output is a goal string + design inputs

branes design plan "<refined goal>" --power 500 --latency 25
# Generates a task graph, but no persistent mission

branes spec new vineyard-sprayer --template amr-warehouse
# Spec is separate from design qualification!

branes spec set vineyard-sprayer perception.detection_classes weed
branes spec set vineyard-sprayer power.compute_power_watts 15
# Manual field-by-field editing

branes mcp hardware
branes mcp analyze yolov8m --hw jetson_orin_nx
# Per-model analysis, no link to spec

branes swap explore --power 500 --latency 25 --weight 600 --cost 15000
# SWaP-C MOO, but disconnected from mission/spec

branes swap check ...
branes report view --latest
```

**Three problems with the existing flow:**

1. **Fragmentation**: `design`, `spec`, `mcp`, `swap`, `optimize` are silos.
   The user has to manually pass goal/constraints/parameters between them.

2. **No persistent mission entity**: `design qualify` produces a one-shot
   design input dict, `spec new` creates an unrelated YAML, `swap explore`
   takes raw flags. Nothing ties them together.

3. **Subsystem invisibility**: Sensors and actuators are fields inside
   `spec`, not first-class concepts. There's no `branes sensor select` or
   `branes actuator compare`.

---

## 4. Gap Assessment

### What is Missing (4 major gaps)

**Gap 1: Mission as a first-class entity**
- The current `design qualify` flow produces a one-shot result with no persistent
  identity. Specs exist but are disconnected from qualification.
- **Impact**: User cannot say "let me re-run the latency analysis on my vineyard
  sprayer" — they must restart from the goal text every time.

**Gap 2: Sensor and actuator commands**
- All other components have dedicated commands (`model`, `platform`, `swap`),
  but the two physical I/O subsystems do not.
- **Impact**: Users must hand-edit YAML to specify sensors and actuators.
  No way to get recommendations, comparisons, or budgets.

**Gap 3: System-level synthesis**
- `design synthesize` exists but takes a requirements YAML and produces a
  *deployment pipeline* — not a complete embodied AI system specification.
- **Impact**: There is no command that takes "mission X" and outputs a full
  system architecture + BOM + power tree + thermal map + scheduling proof.

**Gap 4: Validation as a top-level concept**
- `testbench` validates models against datasets. `swap check` validates
  SWaP-C budgets. There is no `branes validate` that runs a comprehensive
  design verification across all dimensions.
- **Impact**: No single command answers "is this design valid?"

### What Needs to Be Added (new commands)

| Priority | Command | Justification |
|----------|---------|---------------|
| **HIGH** | `mission new/show/edit/refine` | First-class mission entity |
| **HIGH** | `sensor select/compare/budget/list` | Sensor subsystem CLI |
| **HIGH** | `actuator select/compare/budget/list` | Actuator subsystem CLI |
| **HIGH** | `validate <subcommand>` | Top-level validation command |
| **MED** | `synthesize architecture/cabling/thermal` | System-level synthesis beyond BOM |
| **MED** | `analyze scheduling/bandwidth/failure` | Missing analysis dimensions |
| **MED** | `compute place` | Model-to-hardware layer mapping |
| **LOW** | `mission compare/fork` | What-if analysis |
| **LOW** | `app analyze` | Application complexity modeling |
| **LOW** | `export rtl/specs` | Procurement and tape-out artifacts |

### What Needs to Be Improved (modify existing)

| Command | Current Behavior | Recommended Change |
|---------|------------------|-------------------|
| `design qualify` | Produces one-shot dict | Should write to a named mission entity, persistent across sessions |
| `design plan` | Takes goal string | Should accept `--mission <name>` to load from mission store |
| `spec new` | Creates standalone YAML | Should be tied to a mission, updated automatically as design progresses |
| `swap *` commands | Take raw `--power --latency --cost` flags | Should accept `--mission <name>` to inherit constraints |
| `optimize explore` | Takes goal string + flags | Should accept `--mission <name>` |
| `mcp analyze` | Per-model analysis | Should accept `--mission <name>` to analyze all models in the mission |
| `report view` | Views existing reports | Should auto-generate a report when `validate mission` passes |

### What Needs to Be Modified (architectural)

**1. Unified state model**
- Today: `SoCDesignState` (LangGraph internal), `SystemSpec` (specs CLI),
  design inputs dict (from qualify), goal string (planner input).
- Recommended: A single **`Mission`** entity that owns everything: requirements,
  spec, design state, optimization history, reports. All commands operate on
  named missions.

**2. Subsystem registries**
- Today: `PlatformRegistry` (266 platforms), no sensor or actuator registries.
- Recommended: `SensorRegistry` and `ActuatorRegistry` following the same
  pattern — YAML files under `data/sensors/` and `data/actuators/` with
  attributes, keywords, and search.

**3. Command grouping by lifecycle phase**
- Today: 22 command groups by *function* (analyze, benchmark, optimize, etc.)
- Recommended: Reorganize so the workflow phases are obvious:
  ```
  branes mission ...        # Phase 1: definition
  branes select ...         # Phase 2: components (sensor/actuator/compute/model)
  branes synthesize ...     # Phase 3: system composition
  branes analyze ...        # Phase 4: quantitative analysis
  branes optimize ...       # Phase 5: design space exploration
  branes validate ...       # Phase 6: verification
  branes report ...         # Phase 7: artifacts
  ```
  Existing commands stay as aliases for backward compatibility.

**4. Command output should feed forward**
- Today: Commands print tables. The user copies values manually.
- Recommended: Every command writes its result to the mission state.
  `branes sensor select` should populate `mission.sensors`, which
  `branes synthesize system` then reads.

---

## 5. Recommendations

### Immediate (small effort, high value)

1. **Add `mission` command group** — wrap existing `qualify` + `spec` + `session`
   into a unified mission entity. Persistent across sessions.
2. **Add `sensor list/show/search`** — even without full registry data, browsing
   the embodied-schemas sensor catalog would be valuable.
3. **Add `actuator list/show/search`** — same as sensor.
4. **Add `validate mission`** — a top-level command that runs all existing
   checks (SWaP-C, scheduling, constraints) and reports PASS/FAIL.

### Medium-term (moderate effort, structural improvement)

5. **Create `SensorRegistry` and `ActuatorRegistry`** — YAML files with rich
   keywords, attribute ranges, and domain context (mirror `PlatformRegistry`).
6. **Add `synthesize architecture`** — generate a system architecture diagram
   (Mermaid) from mission state.
7. **Add `analyze bandwidth/scheduling`** — these analyses exist inside
   `architecture_tools` but aren't exposed as CLI verbs.
8. **Refactor `design qualify` to produce a mission** — instead of a dict,
   create a persistent named mission entity.

### Long-term (significant effort, completes the vision)

9. **Reorganize commands by lifecycle phase** — group by `mission/select/synthesize/analyze/optimize/validate/report`. Maintain old verbs as aliases.
10. **Unified `Mission` state model** — replace `SoCDesignState`, `SystemSpec`, and
    design inputs dict with a single mission entity. All commands read/write it.
11. **Mission-aware optimization** — `optimize explore --mission X` runs MOO
    against the mission's constraints, components, and design space, then
    writes the result back into the mission.
12. **Add `app analyze`** — model the application code complexity (control loops,
    state machines, decision trees) as part of compute load estimation.

---

## 6. Visualization: Current vs Ideal

### Current: 22 fragmented command groups

```
analyze | api | backends | benchmark | chat | codebase | config | demo
deploy  | design | mcp | model | optimize | pipeline | platform | report
secrets | session | spec | swap | testbench | workflow | zoo
```

Cognitive load: the user has to remember which subsystem owns which command.

### Ideal: 7 lifecycle phases + 5 utility groups

```
LIFECYCLE:
  mission     # Phase 1: define what to build
  select      # Phase 2: pick components (sensor/actuator/compute/model)
  synthesize  # Phase 3: compose the system
  analyze     # Phase 4: quantitative analysis
  optimize    # Phase 5: design space exploration
  validate    # Phase 6: verification
  report      # Phase 7: artifacts

UTILITY:
  platform    # Browse platform registry
  sensor      # Browse sensor catalog
  actuator    # Browse actuator catalog
  model       # Browse model zoo
  compute     # Browse compute hardware

INFRASTRUCTURE:
  api | session | spec | config | secrets | backends | chat
```

Cognitive load: the user follows the phases left-to-right.

---

## 7. Conclusion

The current command set has the **right building blocks** but lacks the
**unifying narrative** of an embodied AI design lifecycle. The biggest gaps are:

1. **No mission entity** to tie everything together
2. **No sensor/actuator commands** (the two missing subsystem CLIs)
3. **No top-level validation** command
4. **Fragmented state** across `design`, `spec`, `session`, `swap`, `optimize`

The recommended path forward is **incremental**:
- **Phase 1**: Add `mission`, `sensor`, `actuator`, `validate` as new top-level
  commands without breaking existing ones.
- **Phase 2**: Build `SensorRegistry` and `ActuatorRegistry` following the
  proven `PlatformRegistry` pattern.
- **Phase 3**: Refactor existing commands to accept `--mission <name>` and
  read/write the unified mission state.
- **Phase 4**: Reorganize the CLI help by lifecycle phase, with old command
  groups remaining as aliases for backward compatibility.

Each phase delivers user value independently, while building toward the
complete vision of a **mission-driven embodied AI design environment**.
