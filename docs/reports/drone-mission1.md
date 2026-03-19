```text
╭─────────────────────────────╮
│ Branes Embodied AI Platform │
│ Version 0.9.1               │
╰─────────────────────────────╯

Agentic Optimization Loop
Mission: drone perception at 5 m/s within 10W
Max iterations: 3
LLM reasoning: heuristic
```

# Design Exploration: drone perception at 5 m/s within 10W

## Executive Summary

We explored 608 hardware/software configurations for **drone perception at 5 m/s within 10W** on a drone platform, producing 11 Pareto-optimal designs where 
no single design dominates all others.

Throughout this report, the primary optimization metric is **capability per watt** (cap/watt): the ratio of detection accuracy to power consumption, computed
as (accuracy / 100) / power_watts. It answers the question *"how much useful perception do I get for each watt of power?"* Higher is better — a design with 
50% accuracy at 5W scores 0.10, while 50% accuracy at 2.5W scores 0.20 (twice as power-efficient for the same capability). Across the Pareto front explored 
here, cap/watt ranges from **0.0687** to **0.2056**.

The design space reveals a fundamental tension: power ranges from 2.3W to 6.9W while accuracy ranges from 16% to 57%. Higher accuracy demands larger models 
and more compute, which directly increases power and cost. The optimizer seeks designs that maximize the accuracy gained per watt spent.

The recommended balanced design uses a **7nm** process at **2.3W**, achieving **16% accuracy** (cap/watt = **0.0687**) with **5.8ms** inference latency.

## Key Tradeoffs

**Process node (7nm–28nm):** Advanced nodes (smaller nm) pack more transistors per mm², enabling higher compute density and lower power per operation. 
However, manufacturing cost scales super-linearly — a 2nm wafer costs ~$22K vs ~$4K for 28nm. For cost-sensitive drone applications, mature nodes (16–28nm) 
often deliver better value.

**Model size vs accuracy:** Larger detector models (e.g., YOLOv8-l/x) achieve higher mAP but require proportionally more compute (GFLOPS). On 
power-constrained hardware, this means either higher power draw or longer inference latency. Smaller models (YOLOv8-n/s, MobileNet) trade accuracy for 
real-time capability.

**Quantization (fp16, fp32, mixed_int8_fp16):** Reducing numerical precision (FP32 → FP16 → INT8 → INT4) cuts memory bandwidth and compute requirements 
roughly in proportion. INT8 typically loses 1–3% mAP vs FP32; INT4 can lose 5–15%. The right choice depends on whether the mission tolerates that accuracy 
reduction for the power savings.

**Compiler runtime (onnxruntime, openvino, pytorch, tvm):** The runtime determines how efficiently the model maps to hardware. TensorRT and TVM perform 
graph-level optimizations (operator fusion, memory planning) that can improve hardware utilization by 30–60% vs unoptimized PyTorch. OpenVINO targets Intel 
hardware specifically. Higher utilization means the same hardware delivers more useful compute per watt.

**Pruning (27%–55%):** Structured pruning removes redundant weights, reducing effective GFLOPS and memory footprint. Moderate pruning (10–30%) typically 
preserves accuracy well. Aggressive pruning (>50%) can significantly degrade accuracy — the optimizer explores this boundary to find where the accuracy loss 
becomes unacceptable for the mission.

## Design Alternatives

|  | Best Efficiency | Most Accurate | Lowest Power (★) | Lowest Cost |
| --- | --------------- | ------------- | ---------------- | ----------- |
| **Hardware** |  |  |  |  |
| Process | 28nm | 28nm | 7nm | 28nm |
| Clock | 100 MHz | 896 MHz | 191 MHz | 896 MHz |
| Systolic array | 3×28 | 2×28 | 11×4 | 2×20 |
| On-chip SRAM | 898 KB | 898 KB | 1409 KB | 1484 KB |
| Compute tiles | 1 | 1 | 6 | 1 |
| **Pipeline** |  |  |  |  |
| Detector | efficientdet/d1 | efficientdet/d1 | ssd_mobilenet/v2 | yolov8/l |
| Tracker | deepsort | deepsort | bytetrack | sort |
| State estimator | ekf | ekf | none | particle_filter |
| **Compiler** |  |  |  |  |
| Runtime | pytorch | tvm | openvino | onnxruntime |
| Quantization | mixed_int8_fp16 | fp16 | fp32 | fp16 |
| Pruning | 27% | 32% | 55% | 38% |
| Graph fusion | yes | yes | yes | yes |
| HW utilization | 26% | 45% | 50% | 34% |
| **Results** |  |  |  |  |
| Power | 2.76 W | 6.90 W | 2.34 W | 5.20 W |
| Latency | 31.5 ms | 11.2 ms | 5.8 ms | 53.6 ms |
| Die area | 429 mm² | 287 mm² | 86 mm² | 205 mm² |
| Unit cost | $796 | $777 | $6,667 | $768 |
| Accuracy (mAP) | 56.8% | 57.0% | 16.1% | 50.0% |
| **Cap/watt** | **0.2056** | **0.0826** | **0.0687** | **0.0961** |

## Design Analysis

### Best Efficiency
*Highest intelligence per watt*

A 28nm design with 3×28 systolic array (84 MACs across 1 tile), clocked at 100 MHz with 898 KB on-chip SRAM. Runs efficientdet/d1 via pytorch (with graph 
fusion) at mixed_int8_fp16 precision.

**Why it's Pareto-optimal:** Achieves the highest capability-per-watt (0.2056) by balancing accuracy (57%) against power (2.8W). 

**Observations:**
- Clock (100 MHz) is very low for 28nm (typical range: 300–1000 MHz). This wastes the speed advantage of the process node — a mature node at higher clock may
be more cost-effective.

### Most Accurate
*Highest detection accuracy*

A 28nm design with 2×28 systolic array (56 MACs across 1 tile), clocked at 896 MHz with 898 KB on-chip SRAM. Runs efficientdet/d1 via tvm (with graph fusion)
at fp16 precision.

**Why it's Pareto-optimal:** Reaches the highest accuracy (57%) at the cost of higher power (6.9W) and cost ($777). 

### Lowest Power (★)
*Minimum power consumption*

A 7nm design with 11×4 systolic array (264 MACs across 6 tiles), clocked at 191 MHz with 1409 KB on-chip SRAM. Runs ssd_mobilenet/v2 via openvino (with graph
fusion) at fp32 precision.

**Why it's Pareto-optimal:** Minimizes power to 2.3W, but at reduced accuracy (16%). Good for battery-powered drones where runtime matters more than peak 
performance. 

**Observations:**
- Clock (191 MHz) is very low for 7nm (typical range: 600–2000 MHz). This wastes the speed advantage of the process node — a mature node at higher clock may 
be more cost-effective.
- Pruning at 55% with only 16% accuracy — the model has been pruned beyond its effective capacity. Reduce pruning to <30% for meaningful accuracy.

### Lowest Cost
*Cheapest to manufacture*

A 28nm design with 2×20 systolic array (40 MACs across 1 tile), clocked at 896 MHz with 1484 KB on-chip SRAM. Runs yolov8/l via onnxruntime (with graph 
fusion) at fp16 precision.

**Why it's Pareto-optimal:** Cheapest to manufacture ($768/unit) using a mature process node. Suitable for high-volume production. 

## Recommendation

Based on the exploration, the **balanced design** (★) is recommended as the starting point. It meets the 10W power budget at 2.3W.

**Next steps:**
- If accuracy (16%) is insufficient, move toward the Most Accurate design (larger model, less pruning, higher precision)
- If power (2.3W) must be reduced further, move toward the Lowest Power design (smaller model, more aggressive quantization)
- If latency (5.8ms) is critical, enable graph fusion and use TensorRT/TVM for higher utilization
- Run with `--max-iterations 5` for deeper exploration (more evaluations, tighter convergence)

## Methodology
- Explored 608 design configurations across 1 iteration(s)
- Pareto front: 11 non-dominated designs (no design is strictly better in all objectives)
- Optimization used MAP-Elites quality-diversity search over a 17-variable joint design space
- All designs satisfy the reticle limit (max die edge 26mm, max area 676 mm²)

## Research References
- mission_profiles/drone_perception.md
- accelerators/nvidia_jetson_orin.md
- accelerators/stillwater_kpu.md
- mission_profiles/autonomous_driving.md
- workloads/transformer_attention.md
- efficiency_studies/tops_per_watt_survey.md
- accelerators/google_coral_edge_tpu.md
- accelerators/hailo_8.md

## Convergence History
- Iteration 0: hypervolume=43120394782.4, Pareto size=11, evaluations=608


## Glossary

- **Capability/watt (cap/watt):** Intelligence per watt — (accuracy / 100) / power. The primary optimization target: how much useful perception you get per 
watt of power.
- **Pareto front:** The set of designs where no design is strictly better than another in all objectives simultaneously. Moving along the front always 
involves giving up something to gain something else.
- **Systolic array:** A grid of multiply-accumulate (MAC) units that data flows through in a pipeline. Larger arrays = more parallel compute, but need 
proportionally more memory bandwidth to stay fed.
- **Quantization:** Reducing the numerical precision of model weights and activations (FP32 → FP16 → INT8 → INT4). Each step roughly halves memory and 
bandwidth requirements but may reduce accuracy.
- **Graph fusion:** Compiler optimization that merges consecutive operations (e.g., Conv + BatchNorm + ReLU) into a single kernel, eliminating intermediate 
memory reads/writes.
- **mAP (mean Average Precision):** Standard accuracy metric for object detection, measuring how well the model identifies and localizes objects across 
confidence thresholds.
- **Pruning:** Removing redundant weights from a neural network. Reduces compute and memory requirements at the cost of accuracy.
- **Reticle limit:** Maximum die size imposed by the lithographic exposure field (~26×33mm). Dies larger than this cannot be manufactured with standard 
single-exposure lithography.


                    Convergence History                     
┏━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃ Iteration ┃ Hypervolume      ┃ Pareto Size ┃ Evaluations ┃
┡━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
│ 0         │ 43120394782.4461 │ 11          │ 608         │
└───────────┴──────────────────┴─────────────┴─────────────┘
