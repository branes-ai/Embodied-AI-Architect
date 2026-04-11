---
title: Introduction
description: Meet the Embodied AI Architect—your AI partner for designing differentiated embodied AI solutions.
---

The **Embodied AI Architect** is an agentic AI that partners with product architects and engineers to design embodied AI solutions that deliver capabilities commodity hardware cannot match.

## The Problem We Solve

Building breakthrough embodied AI products requires answering hard questions:

- **Build vs. Buy**: Should you use commercial hardware or invest in custom silicon?
- **Competitive Position**: How will your solution compare to competitors on performance, cost, and power?
- **Design Tradeoffs**: Which model architecture, quantization, and hardware combination meets your constraints?
- **Pre-Silicon Validation**: Can you validate custom hardware designs before committing to tape-out?

These questions require expertise across ML, hardware architecture, and systems engineering—plus access to data about hardware you may not have.

## What Makes Us Different

### 1. Agentic Design Partner

The Architect isn't just a tool—it's an AI that reasons about your requirements, explores the design space, and recommends solutions. Ask questions in natural language:

```
> I need to run a perception pipeline at 60Hz under 10W for a surgical robot.
> What are my options, and when does custom silicon make sense?
```

### 2. COTS + Custom Hardware

We analyze solutions across **commercial off-the-shelf** platforms (NVIDIA Jetson, Google Coral, Hailo, Intel) **and** custom AI accelerators:

- **328 platform definitions** across 36 categories with 62 real-product configs
- **80 sensors** and **80 actuators** searchable via TF-IDF keyword matching
- **Pre-silicon modeling** for custom accelerators before tape-out
- **Comparative analysis** across your options and competitor systems

### 3. Competitive Intelligence

Our characterization methodology lets you predict performance, cost, and energy for systems you don't have access to:

- Estimate competitor hardware capabilities
- Generate quantitative competitive analysis
- Validate your differentiation before you build

### 4. Mission-Driven Design Lifecycle

The platform organizes all work around **missions** — persistent design entities
that flow through 7 lifecycle phases:

| Phase | Command | What happens |
|-------|---------|-------------|
| **Define** | `branes mission new` | Create a named mission with a goal |
| **Qualify** | `branes design qualify --mission` | Derive constraints, match platforms |
| **Select** | `branes sensor/actuator select` | Choose components from registries |
| **Synthesize** | `branes synthesize system` | Compose architecture from selections |
| **Analyze** | `branes analyze-system` | Power, latency, thermal, SWaP-C checks |
| **Optimize** | `branes optimize explore --mission` | Multi-objective Pareto exploration |
| **Validate** | `branes validate mission` | Verify all constraints pass |

Every command reads from and writes to the same mission. State persists across
sessions — you can close the terminal, come back tomorrow, and pick up where
you left off.

## Why Custom Matters

Tesla's FSD computer demonstrates the power of purpose-built solutions. Instead of using commodity hardware, Tesla designed a custom accelerator optimized for their neural networks—achieving capabilities no off-the-shelf hardware could deliver at their cost and power targets.

The Embodied AI Architect helps you determine when custom hardware delivers competitive advantage:

| Factor | COTS | Custom |
|--------|------|--------|
| Time to market | Fast | 18-24 months |
| NRE cost | Low | High |
| Unit cost at volume | Higher | Lower |
| Performance/watt | Good | Optimized |
| Differentiation | Limited | Unique |

## The Branes Platform

The Embodied AI Architect is the design interface to the **Branes Embodied AI Platform**:

```
┌─────────────────────────────────────────────────────────┐
│              Embodied AI Architect                       │
│         (Agentic Design Interface)                       │
├─────────────────────────────────────────────────────────┤
│  Hardware      │  Analysis      │  Deployment           │
│  Catalog       │  Engine        │  Automation           │
│  ─────────     │  ─────────     │  ─────────            │
│  COTS specs    │  Roofline      │  Quantization         │
│  Custom models │  Constraints   │  Runtime config       │
│  Calibrations  │  Predictions   │  Validation           │
└─────────────────────────────────────────────────────────┘
```

## Who Should Use This

- **Product Architects** exploring hardware options for new embodied AI products
- **Systems Engineers** optimizing perception pipelines for edge deployment
- **Hardware Teams** validating custom accelerator designs against application requirements
- **Technical Leadership** making build-vs-buy decisions with quantitative analysis

## Next Steps

- [Install the platform](/getting-started/installation/)
- [Run your first mission](/getting-started/quickstart/)
- [Follow the mission workflow tutorial](/tutorials/mission-workflow/)
- [Explore the hardware catalog](/catalog/hardware/)
- [Browse the CLI reference](/reference/cli/)
