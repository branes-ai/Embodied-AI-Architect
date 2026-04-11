---
title: Hardware Catalog
description: Browse 328 platforms and 62 product configurations for embodied AI systems.
---

The platform registry contains **328 platform definitions** across 36 categories,
**62 real-product configurations** (DJI, Boston Dynamics, Universal Robots, etc.),
and **36 category index files** with shared defaults.

```bash
# Browse platforms
branes platform list
branes platform list --category aerial
branes platform search "delivery drone"
branes platform show aerial.dji_matrice_350_rtk
```

## Edge GPUs

### NVIDIA Jetson

| Platform | Compute | Memory | Power | Status |
|----------|---------|--------|-------|--------|
| [Jetson Orin AGX](/catalog/hardware/jetson-orin-agx) | 275 TOPS | 64GB | 15-60W | Full support |
| [Jetson Orin NX](/catalog/hardware/jetson-orin-nx) | 100 TOPS | 16GB | 10-25W | Full support |
| [Jetson Orin Nano](/catalog/hardware/jetson-orin-nano) | 40 TOPS | 8GB | 7-15W | Full support |
| [Jetson Xavier NX](/catalog/hardware/jetson-xavier-nx) | 21 TOPS | 8GB | 10-20W | Full support |

## AI Accelerators

### Google Coral

| Platform | Compute | Power | Status |
|----------|---------|-------|--------|
| [Coral Edge TPU](/catalog/hardware/coral-edge-tpu) | 4 TOPS | 2W | Full support |
| [Coral Dev Board](/catalog/hardware/coral-dev-board) | 4 TOPS | 2W | Full support |

### Hailo

| Platform | Compute | Power | Status |
|----------|---------|-------|--------|
| [Hailo-8](/catalog/hardware/hailo-8) | 26 TOPS | 2.5W | Partial support |
| [Hailo-8L](/catalog/hardware/hailo-8l) | 13 TOPS | 1.5W | Partial support |

### Custom Accelerators

| Platform | Compute | Power | Status |
|----------|---------|-------|--------|
| [KPU-T256](/catalog/hardware/kpu-t256) | 256 TOPS | 25W | Simulation only |
| [KPU-T64](/catalog/hardware/kpu-t64) | 64 TOPS | 8W | Simulation only |

## Cloud GPUs

### NVIDIA Datacenter

| Platform | Compute | Memory | Status |
|----------|---------|--------|--------|
| [H100 SXM5](/catalog/hardware/h100) | 1,979 TFLOPS | 80GB | Full support |
| [A100 SXM4](/catalog/hardware/a100) | 312 TFLOPS | 80GB | Full support |
| [V100 SXM2](/catalog/hardware/v100) | 125 TFLOPS | 32GB | Full support |
| [T4](/catalog/hardware/t4) | 65 TFLOPS | 16GB | Full support |
| [L4](/catalog/hardware/l4) | 121 TFLOPS | 24GB | Full support |

## CPUs

### Datacenter

| Platform | Cores | Status |
|----------|-------|--------|
| [Intel Xeon 8490H](/catalog/hardware/xeon-8490h) | 60 | Partial support |
| [AMD EPYC 9654](/catalog/hardware/epyc-9654) | 96 | Partial support |

### Edge

| Platform | Cores | Status |
|----------|-------|--------|
| [Intel i7-12700K](/catalog/hardware/i7-12700k) | 12 | Partial support |

## TPUs

| Platform | Compute | Status |
|----------|---------|--------|
| [TPU v4](/catalog/hardware/tpu-v4) | 275 TFLOPS | Partial support |
| [Coral Edge TPU](/catalog/hardware/coral-edge-tpu) | 4 TOPS | Full support |

## Automotive

| Platform | Compute | Power | Status |
|----------|---------|-------|--------|
| [TDA4VM](/catalog/hardware/tda4vm) | 8 TOPS | 20W | Partial support |
| [TDA4VL](/catalog/hardware/tda4vl) | 4 TOPS | 10W | Partial support |

---

## Adding New Hardware

To add a new hardware platform to the catalog:

1. Create a YAML file in `embodied-schemas/data/hardware/`
2. Include all required fields (see schema)
3. Run validation tests
4. Submit a PR with datasheet links

See [Contributing Guide](/reference/contributing) for details.
