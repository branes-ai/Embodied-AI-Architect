---
title: Deployment
description: Deploy optimized models to edge hardware.
---

Once you've selected your hardware and verified constraints, deploy your model with optimized configurations.

## Deployment Targets

| Target | Framework | Platforms |
|--------|-----------|-----------|
| `jetson` | TensorRT | NVIDIA Jetson devices |
| `openvino` | OpenVINO | Intel CPUs, AMD x86 |
| `coral` | Edge TPU | Google Coral devices |

## Basic Deployment

### To Jetson (TensorRT)

```bash
embodied-ai deploy model.pt \
  --target jetson \
  --precision fp16 \
  --input-shape 1,3,640,640 \
  --output-dir ./deployment
```

### To Coral (Edge TPU)

```bash
embodied-ai deploy model.pt \
  --target coral \
  --precision int8 \
  --calibration-data ./calibration_images/ \
  --input-shape 1,3,320,320
```

### To OpenVINO

```bash
embodied-ai deploy model.pt \
  --target openvino \
  --precision fp16 \
  --input-shape 1,3,640,640
```

## Quantization

### INT8 Quantization

For best edge performance, use INT8 with calibration:

```bash
embodied-ai deploy model.pt \
  --target jetson \
  --precision int8 \
  --calibration-data ./calibration_images/ \
  --calibration-samples 100
```

Calibration data should be representative of your inference inputs (100-500 images).

### Precision Comparison

| Precision | Speed | Accuracy | Memory |
|-----------|-------|----------|--------|
| FP32 | 1x | Baseline | 1x |
| FP16 | 1.5-2x | ~Same | 0.5x |
| INT8 | 2-4x | -0.5-2% | 0.25x |

## Deployment Output

```
Deployment Complete
==================

Target: Jetson (TensorRT)
Precision: INT8

Artifacts:
  ./deployment/
  ├── model.engine        # TensorRT engine
  ├── model.onnx          # Intermediate ONNX
  ├── calibration.cache   # INT8 calibration
  └── config.json         # Deployment config

Validation:
  - Input shape: [1, 3, 640, 640]
  - Output shape: [1, 84, 8400]
  - Latency: 8.2ms (122 FPS)
  - Memory: 45MB

Next steps:
  1. Copy ./deployment/ to your Jetson device
  2. Use TensorRT runtime to load model.engine
  3. See deployment guide: ./deployment/README.md
```

## API Usage

```python
from embodied_ai_architect.agents import DeploymentAgent

agent = DeploymentAgent()
result = agent.execute({
    "model": "yolov8n.pt",
    "target": "jetson",
    "precision": "int8",
    "input_shape": (1, 3, 640, 640),
    "calibration_data": "./calibration/",
    "output_dir": "./deployment",
})

print(f"Engine: {result.data['engine_path']}")
print(f"Latency: {result.data['validation']['latency_ms']}ms")
```

## Deployment Guides

For detailed platform-specific guides:

- [YOLOv8 on Jetson Orin](/tutorials/yolo-on-jetson/)
- [Coral Edge TPU Deployment](/tutorials/coral-deployment/)
- [OpenVINO Deployment](/tutorials/openvino-deployment/)
- [AMD Ryzen AI NUC Deployment](/tutorials/ryzen-ai-deployment/)

## Next Steps

- [Follow a deployment tutorial](/tutorials/)
- [Explore the hardware catalog](/catalog/hardware/)
