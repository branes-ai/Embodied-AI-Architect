# Mission-Driven Embodied AI Design Lifecycle — Implementation Plan

**Date**: 2026-04-07
**Status**: Plan ready for execution
**Related**: `command-set-analysis.md`, Epic issue (TBD)

## Goal

Transform the branes CLI from 22 fragmented command groups into a coherent
**mission-driven design lifecycle**. Every command should operate on a named
**Mission** entity that persists across sessions and ties together qualification,
component selection, synthesis, analysis, optimization, validation, and reporting.

## Approach: 4 Incremental Phases

Each phase delivers user-facing value and is independently mergeable. Backward
compatibility is preserved throughout — existing commands continue to work.

```
Phase 1: Add new top-level commands (no breakage)
   ↓
Phase 2: Build sensor and actuator registries (mirror PlatformRegistry)
   ↓
Phase 3: Mission-aware existing commands (--mission flag)
   ↓
Phase 4: Lifecycle reorganization (group commands by phase)
```

---

## Phase 1: Foundation — Mission Entity + New Commands

**Goal**: Add the missing pieces without touching existing commands.

### Step 1.1: Mission Entity + MissionStore
**Effort**: Small (~300 LOC + tests)

Create a first-class Mission entity that owns the full design state.

**Files to create:**
- `src/embodied_ai_architect/mission/__init__.py`
- `src/embodied_ai_architect/mission/models.py` — `Mission` Pydantic model
- `src/embodied_ai_architect/mission/store.py` — `MissionStore` (mirrors `SpecStore`)
- `tests/test_mission_store.py`

**Mission model fields:**
```python
class Mission(BaseModel):
    id: str                          # slug-style identifier
    name: str                        # human-readable name
    description: str
    created_at: datetime
    updated_at: datetime
    status: MissionStatus            # draft | qualified | designed | optimized | validated

    # Inputs (from qualification)
    goal: str                        # natural-language goal
    platform_id: str | None          # from PlatformRegistry
    use_case: str | None
    constraints: DesignConstraints

    # Components (from selection)
    selected_sensors: list[str]
    selected_actuators: list[str]
    selected_compute: str | None
    selected_models: list[str]

    # State (from design pipeline)
    spec: dict                       # SystemSpec
    design_state: dict | None        # SoCDesignState if running
    optimization_history: list

    # Artifacts
    bom: dict | None
    architecture: dict | None
    reports: list[str]
```

**MissionStore** persists to `.branes/missions/<mission_id>/manifest.json` (mirrors `.branes/specs/`).

### Step 1.2: `branes mission` Command Group
**Effort**: Small (~200 LOC)

```bash
branes mission new <name> [--goal "..."] [--platform <id>]
branes mission show <name>
branes mission list
branes mission edit <name>
branes mission delete <name>
branes mission refine <name>     # re-enter qualification with current state
branes mission fork <src> <dst>  # what-if analysis
```

**Files to create:**
- `src/embodied_ai_architect/cli/commands/mission.py`
- Register in `cli/__init__.py`

### Step 1.3: `branes sensor` Command Group (stub)
**Effort**: Small (~150 LOC)

Initially backed by an empty registry. Phase 2 populates it.

```bash
branes sensor list [--modality]
branes sensor show <sensor_id>
branes sensor search "<query>"
branes sensor categories
```

### Step 1.4: `branes actuator` Command Group (stub)
**Effort**: Small (~150 LOC)

Same shape as sensor.

```bash
branes actuator list [--type]
branes actuator show <actuator_id>
branes actuator search "<query>"
branes actuator categories
```

### Step 1.5: `branes validate` Command Group
**Effort**: Medium (~400 LOC)

A top-level validation command that runs all existing checks against a mission.

```bash
branes validate mission <name>           # run all checks
branes validate constraints <name>       # SWaP-C budgets
branes validate scheduling <name>        # multi-rate feasibility
branes validate safety <name>            # safety integrity
branes validate completeness <name>      # all subsystems specified
```

**Files to create:**
- `src/embodied_ai_architect/cli/commands/validate.py`
- `src/embodied_ai_architect/validation/__init__.py`
- `src/embodied_ai_architect/validation/runner.py` — orchestrates all checks

**Phase 1 deliverables**:
- 5 new commands working end-to-end (no breakage to existing)
- Mission entity persistent across sessions
- Foundation for Phase 2-4

---

## Phase 2: Sensor and Actuator Registries

**Goal**: Build the missing two registries following the proven `PlatformRegistry` pattern.

### Step 2.1: SensorRegistry Schema and Taxonomy
**Effort**: Small (design only)

**Files to create:**
- `data/sensors/schema.yaml`
- `data/sensors/taxonomy.yaml`

**Sensor categories:**
```
visual: rgb_camera, monochrome_camera, stereo_camera, depth_camera,
        thermal_camera, multispectral, hyperspectral, event_camera,
        ir_camera, fisheye_camera, omnidirectional_camera

ranging: lidar_2d, lidar_3d_spinning, lidar_3d_solid_state, radar_2d,
         radar_4d, ultrasonic, tof_camera, sonar

inertial: imu_6dof, imu_9dof, gyro_only, accelerometer_only

position: gps_l1, gps_rtk, gps_ppk, magnetometer, wheel_encoder, vio

environmental: barometer, thermometer, humidity, gas_sensor,
               particulate_pm25, light_sensor, uv_sensor

force: 6dof_force_torque, strain_gauge, pressure_sensor, tactile_array

audio: microphone, microphone_array, hydrophone

biological: ecg, eeg, emg, ppg, glucose, spo2, blood_pressure
```

### Step 2.2: Populate Sensor Registry (~80 sensors)
**Effort**: Medium (data authoring)

**Files to create:**
- `data/sensors/visual/*.yaml` (~20 files)
- `data/sensors/ranging/*.yaml` (~12 files)
- `data/sensors/inertial/*.yaml` (~8 files)
- `data/sensors/position/*.yaml` (~8 files)
- `data/sensors/environmental/*.yaml` (~10 files)
- `data/sensors/force/*.yaml` (~8 files)
- `data/sensors/audio/*.yaml` (~5 files)
- `data/sensors/biological/*.yaml` (~10 files)

Each sensor YAML follows a schema with:
- Identity, keywords, classification
- Attributes: resolution, frame_rate, FOV, range, accuracy, power, weight, cost
- Interface: protocol, data_rate_mbps
- Environmental: IP rating, operating_temp, vibration
- Reference designs (real products)

### Step 2.3: SensorRegistry Class
**Effort**: Small (~200 LOC, mirrors PlatformRegistry)

**Files to create:**
- `src/embodied_ai_architect/sensors/__init__.py`
- `src/embodied_ai_architect/sensors/registry.py`
- `tests/test_sensor_registry.py`

### Step 2.4: ActuatorRegistry Schema and Taxonomy
**Effort**: Small

**Actuator categories:**
```
motor: brushless_dc, brushed_dc, stepper, servo, linear_actuator, voice_coil
hydraulic: hydraulic_cylinder, hydraulic_motor, hydraulic_valve
pneumatic: pneumatic_cylinder, pneumatic_valve, vacuum_generator
gripper: parallel_gripper, suction_cup, magnetic_gripper, soft_gripper,
         multi_finger_hand, vacuum_gripper
locomotion: wheel_motor, propeller, jet_thruster, leg_actuator,
            track_drive, omnidirectional_wheel
fluid: pump, peristaltic_pump, syringe_pump, dispensing_nozzle, sprayer
display: led_array, oled_display, projector, speaker, haptic_actuator
specialty: laser_cutter, plasma_torch, welding_torch, paint_sprayer
```

### Step 2.5: Populate Actuator Registry (~80 actuators)
**Effort**: Medium

Each actuator YAML follows a schema with:
- Attributes: torque, force, speed, payload, control_rate, position_accuracy
- Interface: protocol (CAN, EtherCAT, PWM, USB)
- Power: peak_watts, continuous_watts, voltage
- Physical: weight, dimensions, mounting

### Step 2.6: ActuatorRegistry Class
**Effort**: Small (~200 LOC)

### Step 2.7: Wire Registries into CLI Commands
**Effort**: Medium (~300 LOC)

Replace the stub implementations from Phase 1 with real registry-backed commands:

```bash
branes sensor select <mission>           # recommend sensors for mission tasks
branes sensor compare <s1> <s2> ...      # side-by-side comparison
branes sensor budget <mission>           # power, weight, data rate budget
branes sensor fusion <mission>           # recommend fusion strategy

branes actuator select <mission>         # recommend actuators for mission
branes actuator compare <a1> <a2> ...
branes actuator budget <mission>
branes actuator control-rate <actuator>
```

**Phase 2 deliverables**:
- 80+ sensor definitions, 80+ actuator definitions
- Two new registries with TF-IDF matching
- Sensor and actuator selection/comparison commands

---

## Phase 3: Mission-Aware Existing Commands

**Goal**: Make existing commands operate on the Mission entity instead of taking raw flags.

### Step 3.1: Refactor `design qualify` and `design plan`
**Effort**: Medium (~400 LOC modified)

**Changes:**
- `branes design qualify "goal"` should auto-create a draft Mission
- `branes design qualify --mission <name>` should resume an existing Mission
- `branes design plan --mission <name>` should load constraints from the Mission
- After qualification, the Mission's `selected_sensors`, `selected_actuators`,
  `platform_id`, and `constraints` should be populated

**Files to modify:**
- `src/embodied_ai_architect/cli/commands/design.py`
- `src/embodied_ai_architect/qualification/qualifier.py` — write to mission
- `src/embodied_ai_architect/graphs/planner.py` — read from mission

### Step 3.2: Add `--mission` to `swap`, `optimize`, `mcp`
**Effort**: Medium (~500 LOC modified)

Every command that takes `--power --latency --cost` flags should also accept
`--mission <name>` and load those values from the Mission's constraints.

**Commands to update:**
- `branes swap explore/check/score/rank/budget/sensitivity` — all 11 swap subcommands
- `branes optimize explore/sensitivity/explain/show-front` — all 5 optimize subcommands
- `branes mcp analyze/latency/energy/memory/compare` — all 5 mcp subcommands
- `branes report view` — auto-generate from mission

After running, results should be **written back** into the mission state.

### Step 3.3: Migrate `spec` to be Backed by Mission
**Effort**: Medium (~300 LOC modified)

The `spec` command currently has its own SpecStore. Refactor so:
- A Mission's `spec` field IS the SystemSpec
- `branes spec show <mission>` works directly on missions
- Old `branes spec new <name>` creates a Mission with a draft spec
- SpecStore migrates to MissionStore (with backward-compat shim)

**Phase 3 deliverables**:
- Every command works with `--mission <name>`
- Results flow back into mission state
- User can run a full design loop without tracking files

---

## Phase 4: Lifecycle Reorganization

**Goal**: Reorganize the CLI help so the workflow phases are obvious.

### Step 4.1: Create Lifecycle Command Groups
**Effort**: Medium (~400 LOC)

Create top-level groups that mirror the lifecycle phases:

```
branes select   sensor|actuator|compute|model  # Phase 2: components
branes synthesize  system|architecture|bom|cabling|thermal  # Phase 3
branes analyze     power|latency|thermal|swap|safety|cost|bandwidth|scheduling  # Phase 4
```

These groups don't introduce new logic — they delegate to existing commands.

**Files to create:**
- `src/embodied_ai_architect/cli/commands/select.py`
- `src/embodied_ai_architect/cli/commands/synthesize.py`
- `src/embodied_ai_architect/cli/commands/analyze_group.py` (rename existing `analyze.py`)

### Step 4.2: Reorganize CLI Help
**Effort**: Small (~100 LOC)

Group commands in `branes --help` output by lifecycle phase:

```
LIFECYCLE COMMANDS:
  mission       Define what to build
  select        Pick components (sensor/actuator/compute/model)
  synthesize    Compose the system
  analyze       Quantitative analysis
  optimize      Design space exploration
  validate      Verification
  report        Artifacts

CATALOG COMMANDS:
  platform      Browse 266 embodied AI platforms
  sensor        Browse sensor catalog
  actuator      Browse actuator catalog
  model         Browse model zoo
  compute       Browse compute hardware

INFRASTRUCTURE:
  api | session | spec | config | secrets | backends | chat

LEGACY (aliases for backward compat):
  design | swap | mcp | benchmark | testbench | workflow
```

### Step 4.3: Update README and Documentation
**Effort**: Small (~200 LOC docs)

- Update `README.md` with the new lifecycle workflow
- Update `CLAUDE.md` with new command structure
- Update `docs/designs/system-architecture.md` to show mission-driven flow
- Add `docs/quickstart-mission.md` walking through a complete example

**Phase 4 deliverables**:
- Coherent CLI help organized by lifecycle phase
- Documentation reflects new mental model
- Backward compatibility preserved via aliases

---

## Issue Breakdown

The 4 phases break down into **18 sub-issues** under one parent epic:

| Phase | Step | Title | Effort |
|-------|------|-------|--------|
| 1 | 1.1 | Mission entity + MissionStore | Small |
| 1 | 1.2 | `branes mission` command group | Small |
| 1 | 1.3 | `branes sensor` stub commands | Small |
| 1 | 1.4 | `branes actuator` stub commands | Small |
| 1 | 1.5 | `branes validate` command group | Medium |
| 2 | 2.1 | SensorRegistry schema + taxonomy | Small |
| 2 | 2.2 | Populate sensor registry (~80 sensors) | Medium |
| 2 | 2.3 | SensorRegistry class | Small |
| 2 | 2.4 | ActuatorRegistry schema + taxonomy | Small |
| 2 | 2.5 | Populate actuator registry (~80 actuators) | Medium |
| 2 | 2.6 | ActuatorRegistry class | Small |
| 2 | 2.7 | Wire registries into select/compare commands | Medium |
| 3 | 3.1 | Refactor `design` to use mission | Medium |
| 3 | 3.2 | Add `--mission` flag to swap/optimize/mcp | Medium |
| 3 | 3.3 | Migrate `spec` to mission backing | Medium |
| 4 | 4.1 | Lifecycle command groups (select/synthesize/analyze) | Medium |
| 4 | 4.2 | Reorganize CLI help by phase | Small |
| 4 | 4.3 | Update README and docs | Small |

**Total estimated effort**: ~5000 LOC across 18 PRs.

Each PR is independently mergeable and adds user value. Phases are sequential
(Phase 2 depends on Phase 1, etc.) but steps within a phase can be parallelized.

## Success Criteria

When complete, this scenario should work end-to-end:

```bash
# Define the mission
branes mission new vineyard-sprayer
  --goal "autonomous vineyard sprayer with weed detection"

# Pick components
branes sensor select vineyard-sprayer --task weed_detection
branes compute select vineyard-sprayer
branes actuator select vineyard-sprayer

# Synthesize
branes synthesize system vineyard-sprayer
branes synthesize bom vineyard-sprayer --volume 100

# Analyze
branes analyze latency vineyard-sprayer
branes analyze power vineyard-sprayer
branes analyze swap vineyard-sprayer

# Optimize
branes optimize design vineyard-sprayer

# Validate
branes validate mission vineyard-sprayer

# Report
branes report design vineyard-sprayer
```

Every command operates on the same Mission entity. State flows forward.
The user never has to copy values between commands.
