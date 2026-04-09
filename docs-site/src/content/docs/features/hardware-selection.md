---
title: Hardware Selection
description: Find the optimal hardware platform for your AI workload.
---

Choosing the right hardware is critical for edge AI deployments. Embodied AI Architect helps you compare 50+ hardware targets and find the best fit for your requirements.

## Hardware Categories

### Edge GPUs

| Platform | Compute | Power | Memory | Use Case |
|----------|---------|-------|--------|----------|
| Jetson Orin AGX | 275 TOPS | 60W | 64GB | High-end robotics |
| Jetson Orin NX | 100 TOPS | 25W | 16GB | Mid-range edge |
| Jetson Orin Nano | 40 TOPS | 15W | 8GB | Cost-sensitive edge |

### AI Accelerators

| Platform | Compute | Power | Memory | Use Case |
|----------|---------|-------|--------|----------|
| Coral Edge TPU | 4 TOPS | 2W | Shared | Ultra-low power |
| Hailo-8 | 26 TOPS | 2.5W | Shared | Automotive |
| Intel NCS2 | 1 TOPS | 1W | Shared | Prototyping |

### Cloud GPUs

| Platform | Compute | Power | Memory | Use Case |
|----------|---------|-------|--------|----------|
| H100 | 1,979 TFLOPS | 700W | 80GB | Training, serving |
| A100 | 312 TFLOPS | 400W | 80GB | Training, inference |
| T4 | 65 TFLOPS | 70W | 16GB | Inference |

## Comparing Hardware

### CLI

```bash
branes compare yolov8n \
  --hardware jetson-orin-nano,jetson-orin-agx,coral-edge-tpu \
  --metric latency,power,cost
```

### Interactive Chat

```
> What's the best hardware for running YOLOv8n at 30fps with under 10W power?

Based on your requirements (30 FPS, <10W), here are the best options:

1. **Coral Edge TPU** - Best fit
   - Latency: 12.1ms (82 FPS)
   - Power: 2W
   - Cost: $60

2. **Hailo-8** - Good alternative
   - Latency: 8.5ms (117 FPS)
   - Power: 2.5W
   - Cost: $100

3. **Jetson Orin Nano** - Overkill but flexible
   - Latency: 28.5ms (35 FPS)
   - Power: 7-15W configurable
   - Cost: $200

Recommendation: Coral Edge TPU for your constraints. It meets your
requirements with significant margin and lowest power/cost.
```

## Decision Factors

### Performance vs Power

For battery-powered systems, optimize for performance per watt:

```bash
branes compare yolov8n --sort efficiency
```

### Cost vs Performance

For production deployments, consider $/performance:

```bash
branes compare yolov8n --sort cost-performance
```

### Flexibility vs Optimization

- **Coral Edge TPU**: Best for INT8-only, fixed models
- **Jetson**: Best for flexibility, multiple models, FP16/INT8
- **Hailo**: Best for automotive-grade reliability

## API Usage

```python
from embodied_ai_architect.agents import HardwareProfileAgent

profiler = HardwareProfileAgent()
result = profiler.execute({
    "model_analysis": model_analysis,
    "constraints": {
        "power_watts": 10,
        "latency_ms": 33,
    },
    "top_n": 5,
})

for rec in result.data["recommendations"]:
    print(f"{rec['hardware']}: {rec['score']:.2f}")
```

## Next Steps

- [Explore the hardware catalog](/catalog/hardware/)
- [Deploy to your chosen hardware](/features/deployment/)
