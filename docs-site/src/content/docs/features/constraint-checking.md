---
title: Constraint Checking
description: Verify that models meet deployment requirements with clear PASS/FAIL verdicts.
---

Constraint checking gives you definitive answers: Can your model meet the requirements or not?

## Available Constraints

### Latency

Check if a model meets a latency target:

```bash
branes check-latency resnet18 --hardware h100 --target 10ms
```

Output:
```
PASS - Latency: 0.8ms (target: 10ms)
       Headroom: 92%
       Confidence: HIGH
```

### Power

Check if a model fits within a power budget:

```bash
branes check-power yolov8s --hardware jetson-orin-nano --budget 15W
```

Output:
```
PASS - Power: 12.4W (budget: 15W)
       Headroom: 17%
       Confidence: MEDIUM
```

### Memory

Check if a model fits in available memory:

```bash
branes check-memory resnet152 --hardware jetson-orin-nano --limit 8192MB
```

Output:
```
FAIL - Memory: 240MB required (limit: 8192MB)
       Weights: 232MB
       Activations: 8MB @ batch=1

       Note: Model fits in device memory.
       Verdict considers working memory only.
```

## Verdict Levels

| Verdict | Meaning |
|---------|---------|
| **PASS** | Constraint is met with headroom |
| **MARGINAL** | Constraint is met but with <10% headroom |
| **FAIL** | Constraint cannot be met |
| **UNKNOWN** | Insufficient data to determine |

## Confidence Levels

| Confidence | Meaning |
|------------|---------|
| **HIGH** | Based on calibrated measurements |
| **MEDIUM** | Based on roofline modeling |
| **LOW** | Based on estimates or extrapolation |

## Using in Chat

```
> Can ResNet-50 achieve 5ms latency on a Jetson Orin Nano?

**FAIL** - ResNet-50 cannot meet 5ms latency on Jetson Orin Nano.

Predicted latency: 18.2ms (target: 5ms)
Gap: 264% over budget

**Suggestions:**
1. Use MobileNetV3 instead - estimated 4.8ms
2. Use ResNet-18 - estimated 6.2ms
3. Upgrade to Jetson Orin AGX - ResNet-50 achieves 4.1ms
4. Apply INT8 quantization - estimated 9.5ms (still over)
```

## API Usage

```python
from embodied_ai_architect.llm.graphs_tools import check_latency

result = check_latency(
    model_name="resnet50",
    hardware_name="Jetson-Orin-Nano",
    latency_target_ms=5.0,
)

import json
data = json.loads(result)
print(f"Verdict: {data['verdict']}")
print(f"Confidence: {data['confidence']}")
print(f"Actual: {data['metrics']['latency_ms']}ms")
```

## Batch Constraint Checking

Check multiple constraints at once:

```bash
branes check yolov8n --hardware jetson-orin-nano \
  --latency 33ms \
  --power 15W \
  --memory 4096MB
```

## Next Steps

- [Deploy your model](/features/deployment/)
- [Compare hardware options](/features/hardware-selection/)
