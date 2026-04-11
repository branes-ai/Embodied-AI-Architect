---
title: Sensor & Actuator Selection
description: Search, compare, and select sensors and actuators for your mission using TF-IDF keyword matching.
---

The Branes platform includes registries of **80+ sensors** across 8 categories and
**80+ actuators** across 8 categories. Each entry has rich keyword sets, typical
attribute ranges, and reference products — all searchable via TF-IDF matching.

## Registries at a Glance

### Sensor Categories

| Category | Types | Examples |
|----------|-------|---------|
| **visual** | RGB, stereo, depth, thermal, event, fisheye, multispectral | Sony IMX477, Intel RealSense D435, FLIR Lepton |
| **ranging** | LiDAR 2D/3D, radar, ultrasonic, ToF, sonar | Velodyne VLP-16, Ouster OS1, TI AWR1843 |
| **inertial** | IMU 6-DOF/9-DOF, gyro, accelerometer | Bosch BMI088, InvenSense ICM-42688, VectorNav VN-100 |
| **position** | GPS L1/RTK/PPK, magnetometer, wheel encoder, VIO | u-blox ZED-F9P, SwiftNav Piksi |
| **environmental** | Barometer, thermometer, humidity, gas, particulate, UV | Bosch BME680, Sensirion SCD40 |
| **force** | Force/torque 6-DOF, strain gauge, pressure, tactile array | ATI Mini45, OnRobot HEX-E |
| **audio** | Microphone, microphone array, hydrophone | MEMS arrays, Bruel & Kjaer |
| **biological** | ECG, EEG, EMG, PPG, glucose, SpO2 | Polar H10, Muse 2 |

### Actuator Categories

| Category | Types | Examples |
|----------|-------|---------|
| **motor** | Brushless DC, brushed DC, stepper, servo, linear, voice coil | Maxon EC-i, Dynamixel XM430, Nema 17 |
| **hydraulic** | Cylinder, motor, valve | Parker HMA, Bosch Rexroth A4VG |
| **pneumatic** | Cylinder, valve, vacuum generator | Festo DGEA, SMC SY series |
| **gripper** | Parallel, suction, magnetic, soft, multi-finger, vacuum | Robotiq 2F-85, OnRobot RG2-FT |
| **locomotion** | Wheel motor, propeller, jet thruster, leg actuator, track | T-Motor U15II, KDE 14215XF |
| **fluid** | Pump, peristaltic, syringe, dispensing nozzle, sprayer | KNF NF300, Watson-Marlow 530 |
| **display** | LED array, OLED, projector, speaker, haptic | Neopixel, Haptuator |
| **specialty** | Laser cutter, plasma torch, welding torch, paint sprayer | IPG fiber laser, Lincoln Electric |

## TF-IDF Keyword Search

Every sensor and actuator has 20-50 keywords across 6 groups: identity,
descriptions, application, industry, components, and related concepts. The
registry uses TF-IDF scoring with three phases:

1. **Phrase matching** — multi-word keywords get a bonus (e.g., "stereo camera" scores higher than separate "stereo" + "camera")
2. **Token matching** — individual words match against the inverted index
3. **Bigram matching** — adjacent word pairs provide intermediate specificity

```bash
# Search sensors by natural language
branes sensor search "stereo camera for VIO"
branes sensor search "lidar for autonomous driving"
branes sensor search "IMU for drone navigation"

# Search actuators
branes actuator search "gripper for fragile objects"
branes actuator search "brushless motor for drone propulsion"
branes actuator search "servo for robot joint"
```

## Selection Workflow

The typical workflow for component selection:

```text
search → compare → select → budget → fusion/control-rate
```

### 1. Search

Find sensors or actuators matching your requirements:

```bash
branes sensor search "depth camera for indoor mapping"
```

```text
                   Search: depth camera for indoor mapping
┌──────────────────────────┬──────────────────┬──────────┬───────┐
│ ID                       │ Name             │ Category │ Score │
├──────────────────────────┼──────────────────┼──────────┼───────┤
│ visual.depth_camera      │ Depth Camera     │ visual   │ 1.000 │
│ visual.stereo_camera     │ Stereo Camera    │ visual   │ 0.654 │
│ ranging.tof_camera       │ ToF Camera       │ ranging  │ 0.432 │
└──────────────────────────┴──────────────────┴──────────┴───────┘
```

### 2. Compare

View sensor specs side by side:

```bash
branes sensor compare visual.depth_camera visual.stereo_camera ranging.tof_camera
```

Shows a Rich table with all attributes (power, weight, cost, resolution, range, etc.)
from each sensor lined up for easy comparison.

### 3. Select

Add chosen sensors to your mission:

```bash
branes sensor select my-drone visual.stereo_camera inertial.imu_6dof position.gps_rtk
```

```text
Added 3 sensor(s) to mission 'my-drone':
  + visual.stereo_camera (Stereo Camera)
  + inertial.imu_6dof (6-DOF IMU)
  + position.gps_rtk (RTK GPS)
```

### 4. Budget

Aggregate power, weight, and cost across all selected sensors:

```bash
branes sensor budget my-drone
```

```text
                    Sensor Budget — my-drone
┌──────────────────────┬──────────────────┬───────┬────────┬───────┐
│ ID                   │ Name             │ Power │ Weight │ Cost  │
├──────────────────────┼──────────────────┼───────┼────────┼───────┤
│ visual.stereo_camera │ Stereo Camera    │ 3.0W  │ 150g   │ $300  │
│ inertial.imu_6dof    │ 6-DOF IMU       │ 0.1W  │ 5g     │ $25   │
│ position.gps_rtk     │ RTK GPS         │ 0.5W  │ 20g    │ $200  │
├──────────────────────┼──────────────────┼───────┼────────┼───────┤
│ TOTAL                │                  │ 3.6W  │ 175g   │ $525  │
└──────────────────────┴──────────────────┴───────┴────────┴───────┘
```

### 5. Fusion (Sensors) / Control Rate (Actuators)

**Sensor fusion** recommends cross-modal strategies based on selected categories:

```bash
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

**Actuator control rate** shows loop timing requirements:

```bash
branes actuator control-rate motor.servo
```

```text
Dynamixel Servo  (motor.servo)
  Control rate:   1000 Hz (1.0 ms loop period)
  Response time:  5.0 ms
```

## JSON Output

All commands support `--json` for machine-readable output:

```bash
branes sensor search "lidar" --json
branes sensor budget my-drone --json
branes actuator compare motor.brushless_dc motor.servo --json
```

## Integration with Missions

Selected sensors and actuators are stored on the mission entity:

- `mission.selected_sensors` — list of sensor IDs
- `mission.selected_actuators` — list of actuator IDs

These flow into downstream commands:
- `synthesize system` reads them to compose the full system
- `sensor budget` / `actuator budget` aggregate their attributes
- `sensor fusion` recommends strategies based on selected categories

## See Also

- [Mission Management](/features/mission-management/) — how missions tie components together
- [CLI Reference: sensor](/reference/cli/#sensor) — complete sensor command reference
- [CLI Reference: actuator](/reference/cli/#actuator) — complete actuator command reference
- [Sensor Catalog](/catalog/sensors/) — browse all 8 sensor categories
- [Hardware Catalog](/catalog/hardware/) — browse 266 platforms and 62 product configs
