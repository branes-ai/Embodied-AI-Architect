# Mission-Driven Workflow Quickstart

This guide walks through a complete mission lifecycle using a vineyard sprayer drone
as the example. By the end, you will have a fully specified system with sensor/actuator
selections, a synthesized architecture, and system-level feasibility analysis.

## Prerequisites

```bash
# Install in development mode
pip install -e ".[dev]"

# Verify the CLI is available
branes --help
```

## Step 1: Create a Mission

A mission captures the high-level goal and constraints for your embodied AI system.

```bash
branes mission new vineyard-sprayer \
  --goal "Autonomous vineyard spraying drone with row-following navigation, \
          weed detection, and precision spot-spraying"
```

Expected output:
```
Mission 'vineyard-sprayer' created.
Goal: Autonomous vineyard spraying drone with row-following navigation,
      weed detection, and precision spot-spraying
Status: draft
```

You can view and manage missions at any time:

```bash
branes mission list              # list all missions
branes mission show vineyard-sprayer  # show mission details
branes mission edit vineyard-sprayer  # edit mission interactively
branes mission refine vineyard-sprayer  # refine goals with LLM assistance
branes mission fork vineyard-sprayer vineyard-sprayer-v2  # fork a variant
```

## Step 2: Qualify Design Goals

The qualifier analyzes the mission goal and derives concrete design constraints
(latency targets, power budget, safety requirements, etc.).

```bash
branes design qualify --mission vineyard-sprayer --auto
```

Expected output:
```
Qualifying mission: vineyard-sprayer
Derived constraints:
  Latency class:   real-time (< 33 ms)
  Power budget:    15 W (airborne platform)
  Safety level:    agricultural-outdoor
  Compute profile: edge-gpu
  Operating temp:  -10 C to +50 C
Qualification: PASS
```

## Step 3: Search and Select Sensors

Search the sensor catalog, then bind selections to the mission.

### Search for sensors

```bash
branes sensor search "stereo camera for VIO"
```

Expected output:
```
Found 4 matching sensors:
  1. intel_realsense_d435i   Stereo + IMU, 1280x720@30fps, USB-C, 2.5W
  2. oak_d_pro              Stereo + IMU, 1280x800@60fps, USB-C, 3.5W
  3. zed_mini               Stereo + IMU, 2208x1242@15fps, USB-C, 5.0W
  4. arducam_stereo_hat     Stereo, 1600x1200@30fps, CSI, 1.2W
```

### Select sensors for the mission

```bash
branes sensor select vineyard-sprayer visual.stereo_camera
```

You can also search for and select additional sensors:

```bash
branes sensor search "GNSS RTK module"
branes sensor select vineyard-sprayer navigation.gnss

branes sensor search "downward optical flow"
branes sensor select vineyard-sprayer navigation.optical_flow
```

### Compare selected sensors

```bash
branes sensor compare vineyard-sprayer
```

Expected output:
```
Sensor comparison for mission 'vineyard-sprayer':
  Slot                        Sensor              Power   Bandwidth   Weight
  visual.stereo_camera        intel_realsense_d435i  2.5W   1.2 Gbps    72g
  navigation.gnss             ublox_f9p              0.5W   0.1 Mbps    15g
  navigation.optical_flow     pmw3901                0.1W   0.5 Mbps     3g
  -----------------------------------------------------------------------
  Total                                              3.1W   1.2 Gbps    90g
```

## Step 4: Search and Select Actuators

```bash
branes actuator search "pump for spraying"
```

Expected output:
```
Found 3 matching actuators:
  1. peristaltic_12v_small   Peristaltic pump, 100 mL/min, 12V, 3W
  2. diaphragm_micro         Micro diaphragm pump, 500 mL/min, 12V, 8W
  3. solenoid_valve_3way     3-way solenoid valve, 12V, 2W
```

```bash
branes actuator select vineyard-sprayer fluid.sprayer
```

Select additional actuators as needed:

```bash
branes actuator search "servo for nozzle aiming"
branes actuator select vineyard-sprayer nozzle.pan_servo
branes actuator select vineyard-sprayer nozzle.tilt_servo
```

### Review actuator budget

```bash
branes actuator budget vineyard-sprayer
```

Expected output:
```
Actuator budget for mission 'vineyard-sprayer':
  Slot                  Actuator                Power   Weight   Control Rate
  fluid.sprayer         peristaltic_12v_small     3.0W    85g     10 Hz
  nozzle.pan_servo      sg90_micro_servo          0.5W    10g     50 Hz
  nozzle.tilt_servo     sg90_micro_servo          0.5W    10g     50 Hz
  -------------------------------------------------------------------
  Total                                           4.0W   105g
```

## Step 5: Sensor Budget and Fusion Analysis

### Sensor power and bandwidth budget

```bash
branes sensor budget vineyard-sprayer
```

Expected output:
```
Sensor budget for mission 'vineyard-sprayer':
  Total sensor power:     3.1 W  (of 15.0 W system budget -> 20.7%)
  Total sensor bandwidth: 1.2 Gbps
  Total sensor weight:    90 g
  Verdict: PASS - within platform constraints
```

### Sensor fusion analysis

Analyze how selected sensors can be fused for state estimation and perception.

```bash
branes sensor fusion vineyard-sprayer
```

Expected output:
```
Sensor fusion analysis for mission 'vineyard-sprayer':

  Fusion group: visual_inertial_odometry
    Sensors: stereo_camera + gnss + optical_flow
    Method:  EKF / MSCKF
    Output:  6-DOF pose @ 200 Hz
    Latency: < 5 ms

  Fusion group: weed_detection
    Sensors: stereo_camera (RGB + depth)
    Method:  DNN inference (YOLOv8-seg)
    Output:  segmentation mask @ 15 Hz
    Latency: < 33 ms

  Coverage: position, orientation, velocity, weed-map
  Gaps:     wind estimation (consider adding anemometer)
```

## Step 6: Generate a Design Plan

Create a static design plan that lays out the compute pipeline.

```bash
branes design plan --mission vineyard-sprayer --static
```

Expected output:
```
Design plan for mission 'vineyard-sprayer':

  Pipeline stages:
    1. sensor_ingest     -> stereo frames @ 30 Hz, GNSS @ 10 Hz, flow @ 100 Hz
    2. vio_frontend      -> feature extraction + tracking (GPU, < 10 ms)
    3. state_estimator   -> EKF fusion (CPU, < 2 ms)
    4. weed_detector     -> YOLOv8-seg inference (GPU, < 25 ms)
    5. path_planner      -> row-following + obstacle avoidance (CPU, < 10 ms)
    6. spray_controller  -> nozzle aim + pump trigger (CPU, < 1 ms)

  Compute estimate:
    GPU: 4.2 TOPS required
    CPU: 1.8 GFLOPS required
    Memory: 1.2 GB peak

  Status: plan generated
```

## Step 7: Synthesize the System

Generate the complete system architecture, including compute allocation and interconnects.

```bash
branes synthesize system vineyard-sprayer
```

Expected output:
```
System synthesis for mission 'vineyard-sprayer':

  Compute target: NVIDIA Jetson Orin Nano (8 GB)
    GPU: 1024-core Ampere (40 TOPS)
    CPU: 6-core Arm Cortex-A78AE
    Power: 7-15 W configurable

  Interconnects:
    stereo_camera -> Jetson (USB 3.2, 5 Gbps)
    gnss          -> Jetson (UART, 115200 baud)
    optical_flow  -> Jetson (SPI, 10 MHz)
    servos        -> Jetson (PWM via GPIO)
    pump          -> Jetson (GPIO + relay)

  Software stack:
    OS:        JetPack 6.0 (Ubuntu 22.04)
    VIO:       ORB-SLAM3 / VINS-Fusion
    Detection: YOLOv8s-seg (TensorRT FP16)
    Planner:   custom ROS 2 node

  Estimated SWaP-C:
    Size:   85 x 55 x 25 mm (compute module)
    Weight: 260 g (compute + sensors + actuators)
    Power:  12.1 W (sensors 3.1W + compute 5.0W + actuators 4.0W)
    Cost:   $380 (BOM estimate)
```

You can also synthesize individual aspects:

```bash
branes synthesize architecture vineyard-sprayer  # architecture diagram
branes synthesize bom vineyard-sprayer           # bill of materials
```

## Step 8: Run System-Level Analysis

Verify the synthesized system meets all constraints.

> **Note:** The `analyze-system` subcommands are stubs in the current release.
> They show "Coming in a future release" and point to existing alternatives.
> The examples below show the planned output format.

### Power analysis (coming soon)

```bash
branes analyze-system power vineyard-sprayer
# Currently prints: Coming in a future release.
# Alternative: branes mcp energy <model> <hardware>
```

### Latency analysis (coming soon)

```bash
branes analyze-system latency vineyard-sprayer
# Currently prints: Coming in a future release.
# Alternative: branes mcp latency <model> <hardware>
```

### SWaP-C analysis (coming soon)

```bash
branes analyze-system swap vineyard-sprayer
# Currently prints: Coming in a future release.
# Alternative: branes swap check --mission vineyard-sprayer
```

### Safety analysis (coming soon)

```bash
branes analyze-system safety vineyard-sprayer
# Currently prints: Coming in a future release.
# Alternative: branes validate safety vineyard-sprayer
```

## Summary

The mission-driven workflow follows this lifecycle:

```text
mission new        Define the system goal
    |
design qualify     Derive constraints from the goal
    |
sensor/actuator    Search, select, compare components
select
    |
sensor budget      Validate power/bandwidth/weight budgets
sensor fusion      Analyze sensor fusion opportunities
    |
design plan        Generate the compute pipeline
    |
synthesize system  Produce architecture, BOM, interconnects
    |
analyze-system     Verify power, latency, thermal, SWaP-C, safety
```

Each step enriches the mission state. You can revisit any step, change selections,
and re-run downstream stages to explore the design space iteratively.

## Next Steps

- Use `branes mission refine` to iterate on the goal with LLM assistance
- Use `branes mission fork` to explore design variants
- Use `branes optimize` to run multi-objective optimization on the design space
- Use `branes chat` for interactive design sessions with the AI architect
