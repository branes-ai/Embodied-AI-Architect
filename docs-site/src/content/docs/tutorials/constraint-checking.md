---
title: Checking Deployment Constraints
description: Step-by-step guide to verifying models meet latency, power, and memory requirements before deployment.
---

This tutorial walks you through checking whether your models can meet real-world deployment constraints — latency targets, power budgets, and memory limits — and what to do when they fail.

## Overview

You'll learn how to:
- Check latency, power, and memory constraints individually
- Run batch checks with multiple constraints at once
- Interpret verdicts, confidence levels, and headroom
- Act on failures with concrete next steps

## Prerequisites

- **Embodied AI Architect** installed (`pip install -e ".[dev]"`)
- Know your target hardware and deployment requirements

## 1. Check Latency

The most common constraint: can your model hit the frame rate target?

For 30 FPS real-time processing, you need 33ms per frame:

```bash
branes check-latency yolov8n \
  --hardware jetson-orin-nano \
  --target 33ms
```

**Expected Output:**

```
PASS - Latency: 28.5ms (target: 33ms)
       Headroom: 13.6%
       Confidence: HIGH
```

**What the output means:**
- **PASS** — the model meets the target
- **Headroom: 13.6%** — you have 4.5ms of margin before hitting the target
- **Confidence: HIGH** — this estimate is based on calibrated measurements

Now try a model that's too large:

```bash
branes check-latency yolov8m \
  --hardware jetson-orin-nano \
  --target 33ms
```

```
FAIL - Latency: 92ms (target: 33ms)
       Gap: 179% over budget
       Confidence: HIGH
```

The gap percentage tells you how far off you are — 179% means you'd need roughly a 3x speedup.

## 2. Check Power

Edge devices have strict power budgets. A drone running on battery can't afford a 30W inference pipeline:

```bash
branes check-power yolov8s \
  --hardware jetson-orin-nano \
  --budget 15W
```

**Expected Output:**

```
PASS - Power: 12.4W (budget: 15W)
       Headroom: 17%
       Confidence: MEDIUM
```

**Confidence: MEDIUM** means this is based on roofline modeling rather than measured power draw. Real power may vary by 10-20%.

Try a tighter budget:

```bash
branes check-power yolov8s \
  --hardware jetson-orin-nano \
  --budget 10W
```

```
MARGINAL - Power: 12.4W (budget: 10W)
           Gap: 24% over budget
           Confidence: MEDIUM
```

**MARGINAL** means the constraint is not met but the gap is close enough that optimization might close it (e.g., INT8 quantization, clock gating).

## 3. Check Memory

Ensure your model fits in available device memory:

```bash
branes check-memory resnet152 \
  --hardware jetson-orin-nano \
  --limit 4096MB
```

**Expected Output:**

```
PASS - Memory: 240MB required (limit: 4096MB)
       Weights: 232MB
       Activations: 8MB @ batch=1
       Headroom: 94.1%
```

Memory checks include both weight storage and activation memory at the specified batch size.

## 4. Batch Check All Constraints

In practice, you need to check everything at once. Run all constraints in a single command:

```bash
branes check yolov8n --hardware jetson-orin-nano \
  --latency 33ms \
  --power 15W \
  --memory 4096MB
```

**Expected Output:**

```
Constraint Check: yolov8n on Jetson Orin Nano
─────────────────────────────────────────────
 Constraint  | Actual   | Target  | Verdict  | Headroom
 Latency     | 28.5ms   | 33ms    | PASS     | 13.6%
 Power       | 12.1W    | 15W     | PASS     | 19.3%
 Memory      | 25.4MB   | 4096MB  | PASS     | 99.4%

Overall: PASS ✓
```

If any constraint fails, the overall verdict is FAIL.

## 5. Compare Models Under Constraints

Check several models against the same constraints to find the best fit:

```bash
# Check YOLOv8 variants
branes check-latency yolov8n --hardware jetson-orin-nano --target 33ms
branes check-latency yolov8s --hardware jetson-orin-nano --target 33ms
branes check-latency yolov8m --hardware jetson-orin-nano --target 33ms
```

**Summary:**

| Model | Latency | Verdict | Headroom |
|-------|---------|---------|----------|
| YOLOv8n | 28ms | PASS | 15.2% |
| YOLOv8s | 45ms | FAIL | -36.4% |
| YOLOv8m | 92ms | FAIL | -178.8% |

YOLOv8n is the only variant that fits. If you need better accuracy than YOLOv8n provides, consider upgrading hardware instead of model size.

## 6. What to Do When Constraints Fail

### Latency FAIL

**Try these in order:**

1. **Reduce precision**: FP32 → FP16 → INT8 (often 2-4x speedup)
   ```bash
   branes check-latency yolov8s --hardware jetson-orin-nano \
     --target 33ms --precision int8
   ```

2. **Use a smaller model**: YOLOv8m → YOLOv8s → YOLOv8n

3. **Upgrade hardware**: Orin Nano → Orin NX → Orin AGX
   ```bash
   branes check-latency yolov8s --hardware jetson-orin-agx --target 33ms
   ```

### Power FAIL

1. **Lower clock speed** (trades latency for power)
2. **Reduce precision** (less compute = less power)
3. **Use a smaller model**
4. **Choose more efficient hardware** (TPU vs GPU for inference-heavy workloads)

### Memory FAIL

1. **Reduce batch size** (activation memory scales linearly)
2. **Quantize weights** (FP32 → FP16 halves weight memory, INT8 quarters it)
3. **Use model pruning** to remove redundant parameters
4. **Choose hardware with more memory**

## 7. Interactive Chat

Constraint checking is natural in conversation:

```
You: Can YOLOv8s run at 30fps on a Jetson Orin Nano?

Architect: FAIL — YOLOv8s latency is 45ms on Jetson Orin Nano, above
  the 33ms target for 30fps.

  Suggestions:
  1. Use YOLOv8n instead — 28ms latency, PASS with 15% headroom
  2. Upgrade to Jetson Orin NX — YOLOv8s achieves 24ms
  3. Apply INT8 quantization — estimated 22ms (PASS)

You: What about power? Budget is 10W.

Architect: MARGINAL — YOLOv8n draws 12.1W on Orin Nano at max clocks.
  At 10W power mode (nvpmodel -m 1), latency increases to 35ms — FAIL
  on latency.

  Recommendation: Budget 15W or use Coral Edge TPU for inference (2W).
```

## 8. Python API

```python
from embodied_ai_architect.llm.graphs_tools import (
    check_latency,
    check_power,
    check_memory,
)
import json

# Check latency
result = json.loads(check_latency(
    model_name="yolov8n",
    hardware_name="Jetson-Orin-Nano",
    latency_target_ms=33.0,
))
print(f"Verdict: {result['verdict']}")
print(f"Latency: {result['metrics']['latency_ms']}ms")
print(f"Margin: {result['constraint']['margin_pct']}%")

# Check power
result = json.loads(check_power(
    model_name="yolov8n",
    hardware_name="Jetson-Orin-Nano",
    power_budget_w=15.0,
))
print(f"Power verdict: {result['verdict']}")
```

## Verdict and Confidence Reference

**Verdicts:**

| Verdict | Meaning | Action |
|---------|---------|--------|
| **PASS** | Meets constraint with headroom | Safe to proceed |
| **MARGINAL** | Meets constraint but <10% headroom | Proceed with caution, test on real hardware |
| **FAIL** | Does not meet constraint | Change model, hardware, or precision |
| **UNKNOWN** | Insufficient data | Try a different hardware or model that has calibration data |

**Confidence:**

| Level | Basis | Expected Accuracy |
|-------|-------|-------------------|
| **HIGH** | Calibrated measurements | Within 5% |
| **MEDIUM** | Roofline modeling | Within 15-20% |
| **LOW** | Extrapolation or estimates | Order of magnitude |

## Tips

- **Always check all three constraints** — a model that passes latency may fail on power
- **MARGINAL means test on real hardware** — roofline models are approximate, real silicon may surprise you
- **Headroom matters** — 5% headroom sounds like PASS but leaves no room for OS overhead, thermal throttling, or framework overhead
- **Check at your actual batch size** — memory and power scale with batch size
- **INT8 is often free accuracy** — modern quantization loses <0.5 mAP on detection models

## Next Steps

- [Deploy your model](/features/deployment/) once constraints pass
- [Run roofline analysis](/tutorials/roofline-analysis/) to understand why a constraint fails
- [Analyze your full codebase](/tutorials/codebase-analysis/) for multi-kernel workloads
- See the [CLI reference](/reference/cli/) for all check command options
