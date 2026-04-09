---
title: Common Issues
description: Solutions to frequently encountered problems.
---

## Installation Issues

### "ModuleNotFoundError: No module named 'embodied_ai_architect'"

**Solution**: Install the package in development mode:

```bash
pip install -e ".[dev]"
```

### Missing optional dependencies

**Error**: `ImportError: Kubernetes backend requires kubernetes package`

**Solution**: Install with optional dependencies:

```bash
pip install -e ".[dev,remote,kubernetes]"
```

## Analysis Issues

### "Model not found in catalog"

**Error**: `Unknown model: custom_model`

**Solution**: For custom models, provide the model file directly:

```bash
branes analyze ./my_model.pt --input-shape 1,3,640,640
```

### "Hardware not found"

**Error**: `Unknown hardware: nvidia-rtx-5090`

**Solution**: List available hardware:

```bash
branes chat
> What hardware targets are available?
```

## Chat Issues

### "ANTHROPIC_API_KEY not set"

**Solution**: Set your API key:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

Or add to your shell profile:

```bash
echo 'export ANTHROPIC_API_KEY=sk-ant-...' >> ~/.bashrc
source ~/.bashrc
```

### Rate limiting errors

**Error**: `RateLimitError: Too many requests`

**Solution**: Wait and retry. For batch operations, add delays between requests.

## Deployment Issues

### TensorRT conversion fails

**Error**: `Unsupported ONNX operator`

**Solutions**:
1. Simplify the ONNX model:
   ```bash
   python -m onnxsim model.onnx model_simplified.onnx
   ```
2. Try a different opset:
   ```python
   torch.onnx.export(model, dummy_input, "model.onnx", opset_version=11)
   ```

### INT8 calibration fails

**Error**: `Calibration failed: insufficient data`

**Solutions**:
1. Provide more calibration images (100-500)
2. Ensure images are representative of inference data
3. Check image preprocessing matches training

## Performance Issues

### Lower than expected FPS

**Causes and solutions**:

1. **Wrong power mode** (Jetson):
   ```bash
   sudo nvpmodel -m 0  # Maximum performance
   sudo jetson_clocks
   ```

2. **Thermal throttling**:
   - Check temperature: `tegrastats`
   - Improve cooling

3. **Memory swapping**:
   - Monitor memory: `free -h`
   - Reduce batch size or model size

### High latency variance

**Causes**:
- Background processes
- Thermal throttling
- Memory pressure

**Solutions**:
1. Run warmup iterations before benchmarking
2. Use process isolation
3. Monitor system resources during inference

## Getting Help

If you can't find a solution:

1. Check [GitHub Issues](https://github.com/branes-ai/embodied-ai-architect/issues)
2. Search existing discussions
3. Open a new issue with:
   - Full error message
   - Steps to reproduce
   - System information (`branes --version`, Python version, OS)
