---
title: "Autonomous Driving Compute Architecture"
domain: mission
tags: [automotive, adas, l2, l4, perception, planning, sensor-fusion]
relevance:
  - mission_decomposition
  - hardware_selection
last_updated: 2026-03-16
---

## Compute Requirements by Autonomy Level

| Level | Description | Compute (TOPS) | Power Budget | Latency |
|-------|-------------|---------------|-------------|---------|
| L2 (ADAS) | Lane keeping + ACC | 10-30 | 15-30W | <100ms |
| L2+ | Highway autopilot | 30-100 | 30-50W | <50ms |
| L3 | Conditional automation | 100-300 | 50-100W | <50ms |
| L4 (Robotaxi) | Full urban autonomy | 500-2000 | 100-500W | <30ms |

## Perception Pipeline (L2+ Example)

```
Cameras (8×) → Preprocess → Detect → Track → Fuse → Predict → Plan → Control
Radar (5×)  ↗                              ↗
LiDAR (1×) ↗                              ↗
                                                    Total: <50ms
```

### Per-Operator Breakdown

| Operator | Models | TOPS Required | Latency | Rate |
|----------|--------|---------------|---------|------|
| Camera preprocessing | ISP + distortion | 2 | <5ms | 30 Hz |
| Object detection | 8× camera inference | 20-50 | <15ms | 30 Hz |
| LiDAR processing | PointPillars/CenterPoint | 10-30 | <20ms | 10 Hz |
| Radar processing | CFAR + clustering | 1 | <5ms | 20 Hz |
| Sensor fusion | Late/mid fusion | 2-5 | <5ms | 30 Hz |
| Tracking + prediction | Multi-object tracking | 5-10 | <5ms | 30 Hz |
| Path planning | A*/lattice planner | 1-2 | <20ms | 10 Hz |
| Control | MPC | 0.5 | <2ms | 100 Hz |

## Safety & Redundancy

- **ASIL-D** requires redundant compute paths. Typically: primary SoC +
  safety MCU (lockstep Cortex-R).
- **Deterministic latency** is required — GPUs with variable scheduling are
  problematic. DLA/NPU with bounded execution is preferred for safety paths.
- **Graceful degradation**: If primary compute fails, reduced-capability mode
  must still maintain safe stop (ASIL-B minimum).

## Hardware Landscape

| Platform | TOPS | TDP | TOPS/W | Target Level |
|----------|------|-----|--------|-------------|
| Mobileye EyeQ6H | 34 | 12W | 2.8 | L2/L2+ |
| TI TDA4VM | 8 | 5-20W | 0.4-1.6 | L2 |
| NVIDIA Orin (drive) | 254 | 45-65W | 3.9-5.6 | L2+/L3 |
| NVIDIA Thor | 2000 | 100W | 20 | L4 |
| Qualcomm SA8650P | 200 | 45W | 4.4 | L2+/L3 |

## Lessons for Design Space Exploration

1. **Automotive is a different design regime** from drones: higher power budgets
   (15-500W), stricter latency requirements, safety certification overhead.
2. **Sensor fusion creates parallel compute demands**: 8 cameras + LiDAR + radar
   running simultaneously. Pipeline parallelism across tiles is essential.
3. **Safety certification** adds 20-40% cost overhead and constrains the design
   space (must use deterministic compute, lockstep cores, ECC memory).
4. **L2 ADAS** at 10-30 TOPS is achievable with KPU-T256 class designs. L4
   requires datacenter-class compute that's outside the current design space.
