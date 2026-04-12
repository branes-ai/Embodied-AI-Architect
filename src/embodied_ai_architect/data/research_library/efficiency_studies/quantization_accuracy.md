---
title: "Quantization Accuracy Impact Across Model Families"
domain: efficiency
tags: [quantization, accuracy, int8, int4, fp16, mixed-precision]
relevance:
  - design_tradeoffs
  - design_space_definition
last_updated: 2026-03-16
---

## Post-Training Quantization (PTQ) Impact

| Model | Task | FP32 | FP16 | INT8 PTQ | INT4 PTQ |
|-------|------|------|------|----------|----------|
| YOLOv8s | Det (mAP50) | 44.9 | 44.8 | 43.5 | 39.1 |
| YOLOv8m | Det (mAP50) | 50.2 | 50.1 | 49.0 | 44.8 |
| ResNet-50 | Cls (top1) | 76.1 | 76.0 | 75.2 | 71.5 |
| MobileNetV2 | Cls (top1) | 71.9 | 71.8 | 70.5 | 65.2 |
| EfficientNet-B0 | Cls (top1) | 77.1 | 77.0 | 76.1 | 72.0 |
| DeepLabV3-R50 | Seg (mIoU) | 73.5 | 73.4 | 72.0 | 67.1 |

## Quantization-Aware Training (QAT) Recovery

QAT typically recovers 50-80% of PTQ accuracy loss:

| Model | FP32 | INT8 PTQ | INT8 QAT | Recovery |
|-------|------|----------|----------|----------|
| YOLOv8s | 44.9 | 43.5 | 44.4 | 64% |
| ResNet-50 | 76.1 | 75.2 | 75.8 | 67% |
| MobileNetV2 | 71.9 | 70.5 | 71.4 | 64% |

## Key Findings

1. **FP16 is essentially free**: <0.1% accuracy loss on all tested models.
   Should be the default for GPU targets.

2. **INT8 PTQ is practical**: 1-2% accuracy loss. Acceptable for most
   applications. 3-4× speedup and memory reduction.

3. **INT4 is aggressive**: 4-7% accuracy loss with PTQ. Only viable with
   QAT and for non-safety-critical applications.

4. **Smaller models are more sensitive**: MobileNetV2 loses 1.4% at INT8 vs
   0.9% for ResNet-50. Compact architectures have less redundancy to absorb
   quantization error.

5. **Mixed precision** (INT8 compute + FP16 for sensitive layers like softmax,
   batch norm) achieves 80% of INT8 speedup with only 0.3-0.5% accuracy loss.

## Lessons for Design Space Exploration

1. **quantization_dtype is a high-impact design variable**: INT8 gives 3-4×
   perf/W improvement at 1-2% accuracy cost. This is often the best trade.
2. **Model-specific tables** (like accuracy_tables.py) are essential — generic
   "INT8 drops 1.5%" is too coarse for design space exploration.
3. **QAT should be modeled as a recovery factor**: If the pipeline includes a
   QAT step, reduce the accuracy drop by 50-80%.
