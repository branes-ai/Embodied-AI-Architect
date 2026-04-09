---
title: Quickstart
description: Get up and running with Embodied AI Architect in 5 minutes.
---

This guide will walk you through analyzing a model and checking if it meets your deployment requirements.

## 1. Analyze a Model

Start by analyzing a PyTorch model:

```bash
branes analyze yolov8n.pt
```

This outputs:
- Model architecture summary
- Parameter count and memory footprint
- Computational requirements (FLOPs)
- Layer breakdown

## 2. Check Hardware Fit

See how the model performs on specific hardware:

```bash
branes analyze yolov8n.pt --hardware jetson-orin-nano
```

You'll get:
- Predicted latency
- Memory utilization
- Bottleneck classification (compute-bound vs memory-bound)
- Hardware utilization percentage

## 3. Verify Constraints

Check if your model meets deployment requirements:

```bash
# Check latency target
branes check-latency yolov8n --hardware jetson-orin-nano --target 33ms

# Check power budget
branes check-power yolov8n --hardware jetson-orin-nano --budget 15W

# Check memory limit
branes check-memory yolov8n --hardware jetson-orin-nano --limit 4096MB
```

Each command returns a clear verdict:

```
PASS - Latency: 28.5ms (target: 33ms, headroom: 13.6%)
```

or

```
FAIL - Latency: 45.2ms exceeds target 33ms by 37%
Suggestions:
  - Try YOLOv8n instead of YOLOv8s (2x faster)
  - Enable INT8 quantization (1.5-2x speedup)
  - Consider Jetson Orin NX for 2x more compute
```

## 4. Compare Hardware Options

Find the best hardware for your model:

```bash
branes compare yolov8n --hardware jetson-orin-nano,jetson-orin-agx,coral-edge-tpu
```

Output:

```
Hardware Comparison for YOLOv8n
┌─────────────────────┬───────────┬───────────┬─────────┐
│ Hardware            │ Latency   │ Power     │ $/Perf  │
├─────────────────────┼───────────┼───────────┼─────────┤
│ Jetson Orin AGX     │ 8.2ms     │ 45W       │ $$$     │
│ Jetson Orin Nano    │ 28.5ms    │ 15W       │ $$      │
│ Coral Edge TPU      │ 12.1ms    │ 2W        │ $       │
└─────────────────────┴───────────┴───────────┴─────────┘
```

## 5. Interactive Chat

For exploratory analysis, use the interactive chat:

```bash
export ANTHROPIC_API_KEY=your-key-here
branes chat
```

Ask questions in natural language:

```
> Can I run YOLOv8s at 30fps on a Jetson Orin Nano?

Based on my analysis, YOLOv8s achieves approximately 22 FPS on the
Jetson Orin Nano in FP16 mode, which falls short of your 30 FPS target.

**Recommendations:**
1. Use YOLOv8n instead - it achieves 35 FPS on the same hardware
2. Enable INT8 quantization for YOLOv8s - estimated 28-32 FPS
3. Consider Jetson Orin NX which can run YOLOv8s at 45 FPS

Would you like me to analyze any of these options in detail?
```

## Next Steps

- [Learn about roofline analysis](/features/roofline-analysis/)
- [Explore the hardware catalog](/catalog/hardware/)
- [Deploy your model](/features/deployment/)
