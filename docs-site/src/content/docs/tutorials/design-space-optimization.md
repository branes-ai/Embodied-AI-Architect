---
title: Drone SoC Design Optimization
description: Step-by-step guide to exploring SoC design tradeoffs for a drone perception workload.
---

This tutorial walks you through using the multi-objective optimization engine to find Pareto-optimal SoC designs for a drone perception pipeline under power and latency constraints.

## Overview

You'll learn how to:
- Run a fast design space exploration with MAP-Elites
- Refine the Pareto front with Bayesian optimization
- Interpret sensitivity analysis to understand which parameters matter
- Compare designs and explain tradeoffs

## Prerequisites

- **Embodied AI Architect** installed (`pip install -e ".[dev]"`)
- **Optional**: `botorch` for Bayesian optimization (`pip install botorch`)
- **Optional**: `pymoo` for NSGA-III fallback (`pip install pymoo`)

Without the optional packages, the engine uses MAP-Elites only — still useful for a quick survey.

## 1. Fast Exploration

Start with a quick MAP-Elites run to survey the design space. This evaluates thousands of designs in seconds:

```bash
branes optimize explore \
  --goal "drone perception SoC" \
  --power 5 \
  --latency 33 \
  --fast
```

**Expected Output:**

```
Design Space Exploration
Goal: drone perception SoC
Constraints: {'max_power_watts': 5.0, 'max_latency_ms': 33.0}
Layers: map_elites

Optimization Complete
Total evaluations: 1344
Layers used: map_elites
Pareto front size: 8
Hypervolume: 6.2341
Atlas coverage: 127/256 (49.6%)

Pareto Front (Top 10)
 # | Process | Clock    | Power (W) | Latency (ms) | Area (mm²) | Cost ($)
 0 | 7nm     | 1247MHz  | 3.18      | 28.14        | 41.92      | 18.47
 1*| 7nm     | 843MHz   | 2.14      | 31.25        | 37.88      | 16.23
 2 | 5nm     | 612MHz   | 1.82      | 30.53        | 31.14      | 24.08
 3 | 7nm     | 1891MHz  | 4.52      | 22.67        | 48.31      | 21.15
 4 | 12nm    | 1634MHz  | 4.87      | 31.89        | 62.45      | 12.30
 ...
* = knee point (best balance)
```

The atlas coverage tells you how much of the design space is feasible — 49.6% means about half the parameter combinations meet your constraints.

## 2. Full Pipeline Exploration

Now run the full pipeline with Bayesian refinement for a tighter Pareto front and sensitivity analysis:

```bash
branes optimize explore \
  --goal "drone perception SoC" \
  --power 5 \
  --latency 33
```

This runs MAP-Elites first, then warm-starts Bayesian BO with the best designs. Takes a few minutes but produces better results:

```
Layers used: map_elites, bayesian
Pareto front size: 14
Hypervolume: 8.9127
```

The hypervolume increased from 6.23 to 8.91, meaning the optimizer found a larger region of feasible tradeoffs.

## 3. Inspect the Pareto Front

View the top designs from your exploration:

```bash
branes optimize show-front --top 10
```

Each row is a Pareto-optimal design — no other design is strictly better in every objective. The knee point (marked with `*`) is the best balanced design: moving away from it means giving up a lot in one objective for only a small gain in another.

## 4. Analyze Parameter Sensitivity

See which design parameters have the most impact on each objective:

```bash
branes optimize sensitivity
```

**Expected Output:**

```
Sensitivity: power_watts
 Parameter           Lengthscale  Importance
 clock_mhz           0.1234       0.8912
 process_nm           0.2341       0.7234
 num_compute_tiles    0.4123       0.4567
 array_rows           0.5612       0.3456
 sram_kb              0.8901       0.1234

Sensitivity: latency_ms
 Parameter           Lengthscale  Importance
 array_rows           0.1567       0.8456
 array_cols           0.1892       0.8012
 clock_mhz           0.2456       0.6789
 sram_kb              0.3421       0.5432
 num_compute_tiles    0.4567       0.4123
```

**How to read this:** A short lengthscale means the objective changes rapidly with that parameter — it's highly sensitive. Here we can see:
- **Power** is dominated by clock frequency and process node
- **Latency** is dominated by systolic array dimensions (rows/cols)
- **SRAM** has relatively low impact on both — you may be over-provisioning

This guides where to focus your design effort.

## 5. Compare Two Designs

Pick two interesting designs from the front and compare them:

```bash
branes optimize explain --points 0,2
```

**Expected Output:**

```
Tradeoff: Design #0 vs Design #2

Objective Changes
 Objective     Design #0  Design #2  Delta   Change %
 power_watts   3.18       1.82       -1.36   -42.8%    ← improves
 latency_ms    28.14      30.53      +2.39   +8.5%     ← worsens
 area_mm2      41.92      31.14      -10.78  -25.7%    ← improves
 cost_usd      18.47      24.08      +5.61   +30.4%    ← worsens

Parameter Changes
 Parameter      Design #0  Design #2
 process_nm     7          5
 clock_mhz      1247       612
 array_rows      16         12
```

**Interpretation:** Moving from Design #0 to #2 saves 43% power and 26% area by switching to 5nm and halving the clock — but costs 30% more (advanced process) and adds 8% latency. Whether that's worth it depends on your priorities.

## 6. Python API

For programmatic use or integration into scripts:

```python
from embodied_ai_architect.graphs.moo.design_space import create_soc_design_space
from embodied_ai_architect.graphs.moo.evaluator import DesignEvaluator
from embodied_ai_architect.graphs.moo.engine import OptimizationEngine, OptimizationConfig

# Define constraints
constraints = {"max_power_watts": 5, "max_latency_ms": 33}
ds = create_soc_design_space(constraints)

# Set up evaluator and engine
evaluator = DesignEvaluator(
    design_space=ds,
    base_state={"constraints": constraints},
    constraint_bounds=ds.constraint_bounds,
)
engine = OptimizationEngine(ds, evaluator, OptimizationConfig(max_workers=8))

# Run and inspect
result = engine.run()
engine.shutdown()

print(f"Pareto front: {len(result.pareto_front)} designs")
print(f"Knee point power: {result.knee_point['objectives']['power_watts']:.2f}W")

# Compare two designs
tradeoff = engine.explain_tradeoff(result.pareto_front[0], result.pareto_front[2])
print(tradeoff["summary"])
```

## 7. Interactive Chat

You can also explore designs conversationally:

```
You: Find SoC designs for drone perception under 5W and 33ms

Architect: I'll run a design space exploration with those constraints.
  [starts optimization, shows progress]
  Found 14 Pareto-optimal designs. The knee point uses 7nm at 843MHz
  with 2.14W power and 31.25ms latency.

You: What drives the power consumption?

Architect: From the sensitivity analysis, clock frequency has the highest
  impact (importance 0.89) followed by process node (0.72). SRAM size
  has minimal effect on power (0.12).

You: Compare the cheapest and lowest-power designs

Architect: [shows tradeoff table]
  The cheapest design (#4, 12nm, $12.30) uses 4.87W — near your budget.
  The lowest-power design (#2, 5nm, 1.82W) costs $24.08. You save 63%
  power but pay 96% more.
```

## Design Space Variables

The SoC design space spans 7 parameters:

| Variable | Type | Range | What it Controls |
|----------|------|-------|------------------|
| `process_nm` | Categorical | Technology nodes | Die area, power, cost |
| `clock_mhz` | Continuous | 100–3000 | Performance vs power |
| `array_rows` | Integer | 2–64 | Compute throughput |
| `array_cols` | Integer | 2–64 | Compute throughput |
| `sram_kb` | Integer | 32–2048 | On-chip buffer capacity |
| `num_compute_tiles` | Integer | 1–16 | Parallelism |
| `noc_link_width_bits` | Categorical | 64–512 | Data movement bandwidth |

## Tips

- **Start with `--fast`** to check feasibility before a full run
- **Watch atlas coverage** — low coverage (<30%) suggests constraints are too tight
- **Focus on high-importance parameters** from sensitivity analysis and fix the rest
- **Use JSON output** (`--json-output`) for scripting and downstream tools
- **Hypervolume** is the single best metric for comparing runs — higher is better

## Next Steps

- Read the [Design Optimization feature page](/features/design-optimization/) for methodology details
- Check [hardware targets](/catalog/hardware/) to see available technology nodes
- Use [constraint checking](/features/constraint-checking/) to verify individual designs
- See the [CLI reference](/reference/cli/) for all `optimize` command options
