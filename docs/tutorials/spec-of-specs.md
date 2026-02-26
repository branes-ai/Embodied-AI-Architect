# Spec-of-Specs: Defining Embodied AI System Requirements

## Why

Embodied AI systems are complex. A single drone has requirements spanning perception
(cameras, detection models, latency), compute (SoC, memory, quantization), power
(battery, thermal, mission duration), sensors (IMU, GPS, LiDAR), actuators, comms,
autonomy, and safety. These requirements are deeply interdependent — changing a
perception FPS target ripples into compute TDP, power budget, and battery life.

Traditional approaches store requirements in flat documents or spreadsheets. This
creates three problems:

1. **No structure.** A flat list of 200 fields doesn't show that `max_latency_ms`
   and `min_fps` are both perception constraints, or that `compute_power_watts` and
   `power_budget_watts` need to be consistent.

2. **No provenance.** When a field value changes from 30 to 60, nobody knows who
   changed it, when, or why. Was it the user? An optimization agent? A template
   default? This makes requirements reviews impossible.

3. **No versioning.** You can't diff "what we had last week" against "what we have
   now", tag a baseline, or roll back a bad change.

The spec-of-specs system solves all three with a hierarchical model, an append-only
event log, and content-addressed snapshots.

## What

### Hierarchical Model

A `SystemSpec` organizes requirements into 8 subsystems:

```
SystemSpec (root)
├── name, description, platform_type, tags
├── perception    — cameras, detection, tracking, accuracy, latency, FPS
├── compute       — SoC, CPU, GPU, accelerators, memory, quantization, TDP
├── power         — battery, power budget, thermal, cooling, mission duration
├── sensors       — modalities, rates, data rate, environmental rating
├── actuators     — DOF, control rate, speed, payload
├── comms         — protocols, bandwidth, latency, range
├── autonomy      — autonomy level, planning, navigation, decision rate
├── safety        — safety level, redundancy, failsafe, certifications
├── constraints   — cross-cutting success criteria
└── custom        — user-defined extensibility
```

Every subsystem is **optional**. You start with an empty spec and build it up
incrementally — or start from a template with sensible defaults.

### Event-Sourced Provenance

Every mutation is recorded as an event:

```
seq=0  CREATE   author=template  reason="Created from template 'drone-perception'"
seq=1  SET      /perception/min_fps = 60   author=user   reason="need 60fps for tracking"
seq=2  SET      /compute/soc = "Orin NX"   author=agent  reason="fits 15W TDP budget"
seq=3  SNAPSHOT author=user  reason="baseline requirements"
```

You can ask "why is `/perception/min_fps` set to 60?" and get the full history —
who set it, when, and their stated reason.

### Content-Addressed Versioning

When you commit a snapshot, the spec state is hashed (SHA-256) and stored as a blob.
You can tag versions (`v1.0`, `baseline`, `review-ready`), diff any two versions,
and retrieve historical states by hash or tag.

### Cross-Subsystem Validation

The validator checks for inconsistencies across subsystems:

- Compute TDP exceeding power budget
- Perception latency exceeding frame time at target FPS
- SLAM navigation without depth sensing
- Mission duration exceeding battery capacity
- Safety level without matching redundancy
- Sensor data rate exceeding comms bandwidth

### Templates

Six predefined archetypes provide starting points:

| Template | Platform | Description |
|----------|----------|-------------|
| `drone-perception` | Drone | Real-time perception for obstacle avoidance |
| `quadruped-nav` | Quadruped | Autonomous navigation with terrain adaptation |
| `industrial-inspection` | Industrial Arm | Anomaly detection on production lines |
| `amr-warehouse` | AMR | Warehouse pick-and-place with multi-agent |
| `edge-camera` | Fixed Camera | Monitoring and video analytics |
| `biped-humanoid` | Biped | Human-environment interaction |

## How

All commands use `branes spec <subcommand>`. Add `--json` before `spec` for
machine-readable output.

### Creating and Viewing

```bash
# Create an empty spec
branes spec new my-system

# Create from a template
branes spec new my-drone --template drone-perception

# List available templates
branes spec new _ --template list

# List all specs
branes spec list

# View a spec as a tree
branes spec show my-drone

# View a specific version
branes spec show my-drone --version v1.0
```

### Setting and Deleting Fields

Fields are addressed with JSON pointer paths (`/subsystem/field`):

```bash
# Set a field (auto-creates the subsystem if needed)
branes spec set my-drone /perception/min_fps 60 -m "need 60fps for tracking"

# Set a string value
branes spec set my-drone /compute/soc "Jetson Orin NX" -m "fits power budget"

# Set a boolean
branes spec set my-drone /safety/geofencing true -m "regulatory requirement"

# Set a list (JSON syntax)
branes spec set my-drone /perception/detection_classes '["person","vehicle","bird"]'

# Delete a field
branes spec delete my-drone /perception/model_family -m "no longer constraining model choice"
```

### Versioning

```bash
# Commit a snapshot
branes spec commit my-drone -m "baseline requirements after stakeholder review"

# Tag a version
branes spec tag my-drone v1.0 -m "approved baseline"

# View version history
branes spec history my-drone

# Diff two versions (by tag or hash prefix)
branes spec diff my-drone v1.0 v2.0
```

### Provenance and Validation

```bash
# Why is a field set to its current value?
branes spec why my-drone /perception/min_fps

# Run cross-subsystem consistency checks
branes spec validate my-drone
```

### Export and Import

```bash
# Export to YAML
branes spec export my-drone --format yaml > my-drone-spec.yaml

# Export to JSON
branes spec export my-drone --format json > my-drone-spec.json

# Import from file (creates new spec or overwrites)
branes spec import my-drone-v2 my-drone-spec.yaml
```

### Agent Consumption

```bash
# Flatten to dot-notation for passing to agents
branes spec resolve my-drone
```

This produces output like:
```json
{
  "name": "my-drone",
  "platform_type": "drone",
  "perception.cameras": 2,
  "perception.min_fps": 60.0,
  "compute.soc": "Jetson Orin NX",
  "power.power_budget_watts": 25.0
}
```

## Walkthrough: Designing a Warehouse Inspection Drone

This walkthrough takes you through a realistic scenario: your team needs to design
a drone that inspects warehouse shelving for inventory management. You'll create the
spec, refine it through several iterations, validate it, and produce a versioned
baseline.

### Step 1: Start from a Template

The `drone-perception` template is the closest starting point. It gives you a
fully-populated drone spec with sensible defaults.

```bash
branes spec new warehouse-inspector --template drone-perception \
  -d "Drone for warehouse shelf inventory inspection"
```

View what the template gave you:

```bash
branes spec show warehouse-inspector
```

You'll see 8 fully-populated subsystems: perception with stereo cameras at 30fps,
Jetson Orin NX compute, 99Wh battery with 25-minute mission, and more.

### Step 2: Adapt Perception for Inventory Scanning

The default template targets outdoor obstacle avoidance. For warehouse inventory, you
need different detection classes, higher accuracy (reading labels), and a better
resolution camera.

```bash
# Change detection targets to inventory items
branes spec set warehouse-inspector /perception/detection_classes \
  '["barcode","label","box","shelf","gap"]' \
  -m "inventory scanning requires reading barcodes and detecting shelf gaps"

# Higher accuracy for barcode/label reading
branes spec set warehouse-inspector /perception/min_accuracy 0.9 \
  -m "need 90% accuracy for reliable inventory counts"

# Higher resolution for reading labels
branes spec set warehouse-inspector /perception/resolution "1920x1080" \
  -m "1080p needed to read barcodes at 2m distance"

# Lower FPS is fine for slow-moving inspection
branes spec set warehouse-inspector /perception/min_fps 15 \
  -m "drone moves slowly during inspection, 15fps sufficient"

# Adjust latency budget accordingly
branes spec set warehouse-inspector /perception/max_latency_ms 66 \
  -m "relaxed latency to match 15fps target"
```

### Step 3: Adjust for Indoor Operation

Warehouses are indoor — different sensors, comms, and environmental rating.

```bash
# Indoor environmental rating
branes spec set warehouse-inspector /sensors/environmental_rating indoor \
  -m "warehouse is a controlled indoor environment"

# Add ultrasonic for close-range shelf proximity
branes spec set warehouse-inspector /sensors/modalities \
  '["imu","barometer","ultrasonic"]' \
  -m "ultrasonic for shelf proximity, no GPS indoors"

# Swap navigation from VIO to SLAM (better for structured indoor environments)
branes spec set warehouse-inspector /autonomy/navigation slam \
  -m "SLAM better suited to structured warehouse environment"

# Need WiFi for fleet coordination, not mavlink
branes spec set warehouse-inspector /comms/protocols '["wifi","mqtt"]' \
  -m "MQTT for fleet management, WiFi for connectivity"
```

### Step 4: Validate

Check for cross-subsystem inconsistencies:

```bash
branes spec validate warehouse-inspector
```

You might see:

```
  WARN  SLAM navigation requires depth sensing (stereo/depth camera or LiDAR)
        Path: /autonomy/navigation
        Fix:  Add a depth camera to /perception/camera_types or 'lidar' to /sensors/modalities
```

Good catch — we switched to SLAM but the template already has stereo+depth cameras,
so this warning shouldn't fire. But if you had removed the depth camera, the
validator would flag it. This is the value of cross-subsystem checks.

### Step 5: Commit the Baseline

Everything looks good. Commit and tag this as the v1 baseline:

```bash
branes spec commit warehouse-inspector -m "initial warehouse inspection requirements"
branes spec tag warehouse-inspector v1.0 -m "stakeholder-approved baseline"
```

### Step 6: Iterate After Testing

After a prototype test flight, the team discovers the drone needs more battery life
and a wider power budget for the higher-resolution camera processing:

```bash
# Longer missions needed
branes spec set warehouse-inspector /power/mission_duration_min 45 \
  -m "warehouse zones take 40min to scan, need margin"

# Bigger battery
branes spec set warehouse-inspector /power/battery_wh 150 \
  -m "upgraded to 150Wh pack to support 45min missions"

# Slightly higher compute power for 1080p processing
branes spec set warehouse-inspector /compute/max_tdp_watts 20 \
  -m "1080p inference requires more compute headroom"
branes spec set warehouse-inspector /power/compute_power_watts 20 \
  -m "match compute TDP allocation"
```

Validate again to make sure nothing broke:

```bash
branes spec validate warehouse-inspector
```

Commit the iteration:

```bash
branes spec commit warehouse-inspector -m "post-flight-test adjustments: battery, compute, mission duration"
branes spec tag warehouse-inspector v2.0 -m "post-prototype iteration"
```

### Step 7: Review Changes

See what changed between v1 and v2:

```bash
branes spec diff warehouse-inspector v1.0 v2.0
```

Output:
```
~ /compute/max_tdp_watts: 15.0 → 20.0
~ /power/battery_wh: 99.0 → 150.0
~ /power/compute_power_watts: 15.0 → 20.0
~ /power/mission_duration_min: 25.0 → 45.0
```

### Step 8: Check Provenance

Months later, someone asks "why is the battery 150Wh and not the standard 99Wh?"

```bash
branes spec why warehouse-inspector /power/battery_wh
```

Output:
```
  Seq  Op   Value  Author  Reason                                    Timestamp
    8  set  150    user    upgraded to 150Wh pack to support 45min    2026-02-26...
```

The reason is right there — traceable back to the post-flight-test decision.

### Step 9: Export for Sharing

Share the spec with another team or tool:

```bash
# YAML for human reading
branes spec export warehouse-inspector --format yaml > warehouse-inspector.yaml

# JSON for tool consumption
branes spec export warehouse-inspector --format json > warehouse-inspector.json

# Flat dict for agent pipelines
branes spec resolve warehouse-inspector > warehouse-inspector-flat.json
```

## Chat Agent Integration

The spec system is also available through the interactive chat (`branes chat`).
The AI architect agent can create, read, modify, and validate specs directly:

```
You: Create a spec for an outdoor quadruped robot that needs to navigate rough terrain

Agent: I'll create a spec starting from the quadruped-nav template and customize it
       for rough terrain operation.
       [calls create_spec("terrain-quad", template="quadruped-nav")]
       [calls modify_spec("terrain-quad", "/sensors/environmental_rating", "ip67",
                          reason="outdoor rough terrain requires weatherproofing")]
       [calls validate_spec("terrain-quad")]
```

Agent modifications are tracked with `author=agent` in the event log, so you can
always distinguish human decisions from AI suggestions in the provenance history.

## Summary

| Task | Command |
|------|---------|
| Create from template | `branes spec new <name> --template <template>` |
| View current state | `branes spec show <name>` |
| Set a field | `branes spec set <name> <path> <value> -m "<reason>"` |
| Delete a field | `branes spec delete <name> <path> -m "<reason>"` |
| Validate consistency | `branes spec validate <name>` |
| Commit snapshot | `branes spec commit <name> -m "<message>"` |
| Tag a version | `branes spec tag <name> <tag> -m "<message>"` |
| View history | `branes spec history <name>` |
| Diff versions | `branes spec diff <name> <v1> <v2>` |
| Trace a field | `branes spec why <name> <path>` |
| Export | `branes spec export <name> --format yaml` |
| Import | `branes spec import <name> <file>` |
| Flatten for agents | `branes spec resolve <name>` |
