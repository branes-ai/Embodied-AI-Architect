---
title: "Cross-Architecture TOPS/W Survey"
domain: efficiency
tags: [tops, efficiency, power, comparison, benchmark]
relevance:
  - mission_decomposition
  - hardware_selection
  - design_tradeoffs
last_updated: 2026-03-16
---

## Survey Data (INT8 Inference, 2024-2025)

| Accelerator | Peak TOPS | TDP (W) | TOPS/W | Process | Target |
|-------------|-----------|---------|--------|---------|--------|
| NVIDIA H100 | 3,958 | 700 | 5.7 | 4nm | Datacenter |
| NVIDIA A100 | 624 | 400 | 1.6 | 7nm | Datacenter |
| Google TPU v4 | 275 | 170 | 1.6 | 7nm | Datacenter |
| Apple M2 Ultra ANE | 31 | 15* | 2.1 | 5nm | Desktop |
| Jetson AGX Orin | 275 | 60 | 4.6 | 8nm | Edge |
| Jetson Orin NX | 100 | 25 | 4.0 | 8nm | Edge |
| Jetson Orin Nano | 40 | 15 | 2.7 | 8nm | Edge |
| Hailo-8 | 26 | 5 | 5.2 | 16nm | Edge |
| Hailo-8L | 13 | 2.5 | 5.2 | 16nm | Edge |
| Intel Movidius MA2085 | 4 | 1.5 | 2.7 | 16nm | Edge |
| Google Coral TPU | 4 | 2 | 2.0 | 28nm | Edge |
| Arm Ethos-U85 | 0.5 | 0.1 | 5.0 | 7nm | MCU |
| Arm Ethos-U55-256 | 0.256 | 0.05 | 5.0 | 7nm | MCU |
| Stillwater KPU-T256 | 16 | 2 | 8.0 | 16nm | Edge |
| Stillwater KPU-T64 | 4 | 0.5 | 8.0 | 16nm | Edge |

*ANE power is estimated; Apple does not publish TDP per block.

## Key Observations

1. **TOPS/W efficiency frontier**: 5-8 TOPS/W at the edge (Hailo, Ethos, KPU),
   1.5-6 TOPS/W at datacenter scale. Specialization beats scale for efficiency.

2. **Process node helps but isn't decisive**: Hailo at 16nm achieves better
   TOPS/W than Jetson Orin at 8nm — architecture matters more than process.

3. **Sweet spot for embodied AI**: 1-30 TOPS at 1-15W covers drone to UGV
   workloads. This maps to Hailo-8, KPU-T256, Orin NX/Nano.

4. **Diminishing returns above 30 TOPS** at the edge: most perception pipelines
   need 2-10 TOPS sustained. Higher TOPS is wasted unless running multiple
   concurrent models.

## Lessons for Design Space Exploration

1. **Set realistic TOPS/W targets**: 5-8 TOPS/W is SOTA at 16nm for custom
   designs. Exceeding 10 TOPS/W requires sub-10nm process or extreme workload
   specialization.
2. **Capability per watt is more meaningful than TOPS/W** — a 2 TOPS/W
   accelerator running a better model may outperform a 8 TOPS/W accelerator
   running a worse model. This is why Phase 1's accuracy loop matters.
3. **Power budgets by platform**: Drone (5-15W), UGV (15-50W), Vehicle (50-200W),
   Wearable (0.1-1W). Use these to scope the design space search.
