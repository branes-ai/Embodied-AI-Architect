---
title: Design Space Optimization
description: Explore SoC design tradeoffs with multi-objective optimization.
---

When designing a system-on-chip, no single "best" design exists. Lower power means higher latency; smaller area means less on-chip memory; faster clocks cost more. Greedy optimizers find one point and call it done — but real design decisions require understanding the full tradeoff surface.

Branes' multi-objective optimization (MOO) engine maps the feasibility region, identifies Pareto-optimal designs, and explains parameter sensitivity so stakeholders can make informed tradeoffs between power, latency, area, and cost.

## The 3-Layer Pipeline

The engine uses a layered approach where each layer builds on the previous one:

```
┌─────────────────────┐
│  Layer 1: MAP-Elites │  Fast atlas: 5K-10K evals, seconds
│  Quality-diversity   │  Fills a grid of design niches
└──────────┬──────────┘
           │ warm-start best designs
           ▼
┌─────────────────────┐
│  Layer 2: Bayesian   │  Refined front: 100-200 evals, minutes
│  BO (qNEHVI)         │  GP surrogate + sensitivity analysis
└──────────┬──────────┘
           │
           ▼
       Pareto front + sensitivity + knee point
```

**Layer 1 — MAP-Elites** generates a quality-diversity atlas using Latin Hypercube Sampling and mutation. It rapidly explores the design space (thousands of evaluations in seconds) and fills a grid where each cell represents a unique design niche. This gives broad coverage of what's feasible.

**Layer 2 — Bayesian Optimization** takes the best designs from MAP-Elites as seed points and refines the Pareto front using Gaussian Process surrogate models with the qNEHVI (q-Noisy Expected Hypervolume Improvement) acquisition function. The GP lengthscales provide parameter sensitivity for free.

**Layer 3 — NSGA-III** is used as a fallback for many-objective problems (>4 objectives) or when BoTorch is not installed. It uses pymoo's evolutionary algorithm with reference-direction-based selection.

**Auto layer selection:** The engine automatically picks layers based on objective count:
- 4 or fewer objectives: MAP-Elites + Bayesian BO
- More than 4 objectives: MAP-Elites + NSGA-III

## Quick Start

### CLI

```bash
# Fast exploration (MAP-Elites only, seconds)
branes optimize explore --goal "drone SoC" --power 5 --latency 33 --fast

# Full pipeline (MAP-Elites + Bayesian BO)
branes optimize explore --goal "drone SoC" --power 5 --latency 33

# View the Pareto front
branes optimize show-front --top 10

# Parameter sensitivity analysis
branes optimize sensitivity

# Compare two designs from the front
branes optimize explain --points 0,3
```

### Interactive Chat

```
You: Explore designs for a drone perception SoC under 5W and 33ms latency

Architect: Starting design space exploration...
  MAP-Elites: 5120 evaluations, 78.3% coverage
  Bayesian BO: 200 evaluations, hypervolume 12.45

  Pareto Front (Top 5):
  # | Process | Clock    | Power | Latency | Area    | Cost
  0 | 7nm     | 1200MHz  | 3.2W  | 28.1ms  | 42mm²   | $18.50
  1*| 7nm     | 800MHz   | 2.1W  | 31.2ms  | 38mm²   | $16.20
  2 | 5nm     | 600MHz   | 1.8W  | 30.5ms  | 31mm²   | $24.10
  ...
  * = knee point (best balance)

You: Compare designs 0 and 2

Architect: Design #0 vs #2:
  Power:   3.2W → 1.8W  (-43.8%)  ✓ improves
  Latency: 28.1 → 30.5ms (+8.5%)  ✗ worsens
  Area:    42 → 31mm²    (-26.2%)  ✓ improves
  Cost:    $18.50 → $24.10 (+30.3%) ✗ worsens
  Key change: 5nm process shrinks area/power but increases cost.
```

### Python API

```python
from embodied_ai_architect.graphs.moo.design_space import create_soc_design_space
from embodied_ai_architect.graphs.moo.evaluator import DesignEvaluator
from embodied_ai_architect.graphs.moo.engine import OptimizationEngine, OptimizationConfig

# Define design space with constraints
ds = create_soc_design_space({"max_power_watts": 5, "max_latency_ms": 33})

# Create evaluator
evaluator = DesignEvaluator(
    design_space=ds,
    base_state={"constraints": {"max_power_watts": 5}},
    constraint_bounds=ds.constraint_bounds,
)

# Run optimization
config = OptimizationConfig(layers="auto", max_workers=8)
engine = OptimizationEngine(ds, evaluator, config)
result = engine.run()
engine.shutdown()

# Inspect results
print(f"Pareto front: {len(result.pareto_front)} designs")
print(f"Hypervolume: {result.hypervolume:.4f}")
print(f"Knee point: {result.knee_point}")
print(f"Sensitivity: {result.sensitivity}")
```

## Understanding Results

### Pareto Front

Each row in the Pareto front is a non-dominated design — no other design is better in every objective simultaneously. The columns show:

| Column | Meaning |
|--------|---------|
| Process | Technology node (e.g., 7nm, 5nm) |
| Clock | Operating frequency |
| Power | Total power dissipation |
| Latency | End-to-end inference latency |
| Area | Die area |
| Cost | Estimated manufacturing cost |

### Knee Point

The knee point is the design closest to the utopia point (the hypothetical design that is best in every objective). It represents the best balanced tradeoff — moving away from it means sacrificing a lot in one objective for only a small gain in another.

### Sensitivity Analysis

Sensitivity comes from the Bayesian BO layer's GP lengthscales. A short lengthscale means the objective changes rapidly with that parameter (high sensitivity). A long lengthscale means the parameter has little effect.

```
Sensitivity: power_watts
  Parameter         Lengthscale  Importance
  clock_mhz         0.1234       0.8912    ← most influential
  process_nm        0.2341       0.7234
  num_compute_tiles 0.5612       0.3456
  sram_kb           0.8901       0.1234    ← least influential
```

### Tradeoff Explanation

The `explain` command shows what changes between two designs and quantifies each objective delta as a percentage, making it easy to judge whether a tradeoff is worthwhile.

## Design Space Variables

The SoC design space includes 7 parameters spanning process technology, microarchitecture, and memory:

| Variable | Type | Range | Description |
|----------|------|-------|-------------|
| `process_nm` | Categorical | Technology nodes | Fabrication process node |
| `clock_mhz` | Continuous | 100–3000 | Operating frequency |
| `array_rows` | Integer | 2–64 | Systolic array row count |
| `array_cols` | Integer | 2–64 | Systolic array column count |
| `sram_kb` | Integer | 32–2048 | On-chip SRAM capacity |
| `num_compute_tiles` | Integer | 1–16 | Number of compute tiles |
| `noc_link_width_bits` | Categorical | 64, 128, 256, 512 | NoC interconnect width |

## When to Use What

| Mode | Command | Time | Use Case |
|------|---------|------|----------|
| Fast | `--fast` | Seconds | Quick design survey, feasibility check |
| Full pipeline | Default (`auto`) | Minutes | Production design exploration |
| MAP-Elites only | `--layers map_elites` | Seconds | Coverage atlas without refinement |
| Bayesian only | `--layers bayesian` | Minutes | Small design space, need sensitivity |
| NSGA-III | `--layers nsga3` | Minutes | >4 objectives, requires `pymoo` |

## Optional Dependencies

The full pipeline requires optional packages:

```bash
# For Bayesian BO (Layer 2, recommended)
pip install botorch

# For NSGA-III (Layer 3, many-objective fallback)
pip install pymoo
```

Without these, the engine falls back to MAP-Elites only, which still provides a useful exploration atlas.

## Next Steps

- [CLI reference](/reference/cli/) for all `optimize` command options
- [MCP tools reference](/reference/mcp-tools/) for programmatic access via Claude
- [Hardware catalog](/catalog/hardware/) for available hardware targets
- [Constraint checking](/features/constraint-checking/) for pass/fail verification
