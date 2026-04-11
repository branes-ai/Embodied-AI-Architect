---
title: Sensor Catalog
description: Browse 80+ sensors across 8 categories for embodied AI systems.
---

The sensor registry contains **80+ sensor definitions** across 8 modality categories,
each with rich keyword sets for TF-IDF search, typical attribute ranges, and
reference products.

## Browse with the CLI

```bash
# List all sensors
branes sensor list

# Filter by category
branes sensor list --category visual
branes sensor list --category ranging

# Search by natural language
branes sensor search "stereo camera for VIO"
branes sensor search "lidar for autonomous driving"

# Show details
branes sensor show visual.stereo_camera

# List categories
branes sensor categories
```

## Categories

### Visual (11 types)

Image-based sensors for 2D/3D visual information.

| Type | Typical Use | Power | Cost |
|------|-------------|-------|------|
| RGB Camera | Object detection, classification | 1-3W | $10-500 |
| Stereo Camera | Depth estimation, VIO, SLAM | 2-5W | $100-1000 |
| Depth Camera | Indoor mapping, obstacle avoidance | 2-5W | $100-500 |
| Thermal Camera | Night vision, industrial inspection | 1-3W | $200-5000 |
| Event Camera | High-speed motion, low latency | 1-2W | $500-3000 |
| Fisheye Camera | Wide-angle surround view | 1-3W | $50-300 |
| Multispectral Camera | Agriculture, environmental monitoring | 3-10W | $1000-10000 |
| Monochrome Camera | Machine vision, feature tracking | 1-2W | $50-500 |
| IR Camera | Night vision, thermal imaging | 1-3W | $100-2000 |
| Hyperspectral Camera | Material identification | 5-15W | $5000-50000 |
| Omnidirectional Camera | 360-degree coverage | 3-8W | $200-2000 |

### Ranging (8 types)

Distance and depth measurement sensors.

| Type | Range | Typical Use |
|------|-------|-------------|
| LiDAR 2D | 10-30m | Indoor mapping, safety zones |
| LiDAR 3D Spinning | 50-200m | Autonomous vehicles, outdoor mapping |
| LiDAR 3D Solid State | 30-150m | ADAS, compact mobile robots |
| Radar 2D | 50-200m | Automotive, maritime |
| Radar 4D | 50-300m | Autonomous driving, imaging |
| Ultrasonic | 0.1-5m | Proximity, parking assist |
| Time-of-Flight Camera | 0.1-10m | Gesture recognition, indoor |
| Sonar | 1-100m | Underwater, marine |

### Inertial (4 types)

Motion and orientation measurement.

| Type | Typical Rate | Use |
|------|-------------|-----|
| IMU 6-DOF | 200-2000 Hz | Navigation, VIO, stabilization |
| IMU 9-DOF | 200-1000 Hz | AHRS, compass heading |
| Gyro Only | 100-8000 Hz | Rate sensing, stabilization |
| Accelerometer Only | 100-4000 Hz | Vibration, impact detection |

### Position (6 types)

Absolute and relative position measurement.

| Type | Accuracy | Use |
|------|----------|-----|
| GPS L1 | 2-5m | Outdoor navigation |
| GPS RTK | 1-2cm | Precision agriculture, surveying |
| GPS PPK | 1-2cm | Mapping, post-processing |
| Magnetometer | 1-3 degrees | Heading reference |
| Wheel Encoder | 0.1-1mm | Odometry |
| VIO | 0.1-1% drift | Indoor navigation |

### Environmental (7 types)

Ambient condition measurement: barometer, thermometer, humidity, gas, particulate, light, UV.

### Force (4 types)

Force, torque, and pressure measurement: 6-DOF force/torque, strain gauge, pressure, tactile array.

### Audio (3 types)

Sound capture: microphone, microphone array, hydrophone.

### Biological (7 types)

Biological signal measurement: ECG, EEG, EMG, PPG, glucose, SpO2, blood pressure.

## Reference Products

Many sensor definitions include reference products with specific specs and pricing:

| Product | Category | Vendor | Approx. Price |
|---------|----------|--------|--------------|
| Sony IMX477 | Visual (RGB) | Sony | $25 |
| Intel RealSense D435 | Visual (Depth) | Intel | $179 |
| Stereolabs ZED 2i | Visual (Stereo) | Stereolabs | $449 |
| Velodyne VLP-16 | Ranging (LiDAR 3D) | Velodyne | $4000 |
| Bosch BMI088 | Inertial (IMU) | Bosch | $8 |
| u-blox ZED-F9P | Position (RTK GPS) | u-blox | $200 |
| FLIR Lepton 3.5 | Visual (Thermal) | FLIR | $250 |

## See Also

- [Sensor & Actuator Selection](/features/sensor-actuator-selection/) — selection workflow with search, compare, budget
- [CLI Reference: sensor](/reference/cli/#sensor) — complete sensor command reference
