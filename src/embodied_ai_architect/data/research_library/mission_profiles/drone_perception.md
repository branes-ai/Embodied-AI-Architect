---
title: "Drone Perception Pipeline Requirements"
domain: mission
tags: [drone, perception, detection, tracking, slam, real-time]
relevance:
  - mission_decomposition
  - design_space_definition
  - hardware_selection
last_updated: 2026-03-16
---

## Mission Requirements by Flight Speed

| Speed | Frame Rate | E2E Latency | Reaction Distance | Use Case |
|-------|-----------|-------------|-------------------|----------|
| Hover (0 m/s) | 10 FPS | <100ms | N/A | Inspection |
| Slow (2 m/s) | 15 FPS | <66ms | 0.13m | Indoor nav |
| Medium (5 m/s) | 30 FPS | <33ms | 0.17m | Outdoor nav |
| Fast (10 m/s) | 30 FPS | <33ms | 0.33m | Racing/pursuit |
| Very fast (20 m/s) | 60 FPS | <16ms | 0.33m | High-speed |

**Rule**: At speed V (m/s), the drone travels V × latency meters before
reacting. For obstacle avoidance, this must be less than the stopping distance.

## Standard Perception Pipeline

```
Camera → Preprocess → Detect → Track → Scene Graph → Reason → Control
  30Hz     0.5ms       8ms     2ms      1ms          2ms      0.5ms
                                                          Total: ~14ms
```

### Per-Operator Requirements

| Operator | Model | GFLOPS | Latency Budget | Rate |
|----------|-------|--------|----------------|------|
| Preprocess | Resize + normalize | 0.1 | <1ms | 30 Hz |
| Detection | YOLOv8s (640×640) | 28.6 | <10ms | 30 Hz |
| Tracking | ByteTrack | 0.01 | <3ms | 30 Hz |
| Scene Graph | Rule-based | 0.001 | <2ms | 30 Hz |
| Depth (optional) | MiDaS-small | 6.0 | <8ms | 15 Hz |
| SLAM (optional) | ORB-SLAM3 | 2.0 | <15ms | 15 Hz |
| State Est. | EKF (12-state) | 0.001 | <0.1ms | 100 Hz |
| Control | PID/MPC | 0.01 | <1ms | 100 Hz |

## Power & Weight Budget

| Drone Class | MTOW | Compute Budget | Battery | Flight Time |
|-------------|------|---------------|---------|-------------|
| Nano (<250g) | 250g | 2-5W | 1S 450mAh | 5-8 min |
| Mini (250g-2kg) | 1.5kg | 5-15W | 4S 2200mAh | 15-25 min |
| Medium (2-7kg) | 5kg | 15-50W | 6S 5000mAh | 20-35 min |
| Large (>7kg) | 15kg | 50-200W | 12S 16000mAh | 30-60 min |

**Compute impact on flight time**: Every additional watt of compute reduces
flight time by ~2-5% for mini drones. A 15W compute module on a mini drone
consumes ~15% of total power.

## Lessons for Design Space Exploration

1. **Latency budget decomposition**: For 5 m/s flight, the 33ms total budget
   must be split across operators. Detection gets the lion's share (~60%).
   The decomposer should output per-operator budgets.
2. **Multi-rate is mandatory**: Perception at 30 Hz, state estimation at
   100+ Hz, control at 100+ Hz. Pipeline design must support this.
3. **Power is the binding constraint** for nano/mini drones. A 2W KPU-T64
   is viable; a 15W Jetson Orin NX is not for a nano drone.
4. **Detection model choice** is the single highest-impact decision — it
   determines both accuracy (mission capability) and compute demand (power).
   YOLOv8n (8.7 GFLOPS) vs YOLOv8s (28.6 GFLOPS) is a 3× power difference
   for ~7% accuracy gap.
