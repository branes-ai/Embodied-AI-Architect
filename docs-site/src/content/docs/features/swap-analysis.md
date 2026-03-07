---
title: SWaP-C Analysis
description: Estimate and optimize Size, Weight, Power, and Cost for complete embedded systems.
---

A silicon design isn't done when the die tape-out finishes. The die goes into a package, the package onto a PCB, the PCB into an enclosure with a cooling solution — and each layer adds weight, volume, and cost. A 50 mm² die that looks great on paper can become a 200 g system that breaks your drone's payload budget.

Branes' SWaP-C analysis closes the gap between chip-level PPA (Power, Performance, Area) and system-level reality. It answers the questions hardware teams ask every day: *What will this weigh? Will it fit? Can we cool it? What does packaging cost?*

## Why SWaP-C Matters

Traditional design tools stop at the die boundary. But for embodied AI — drones, robots, wearables, edge appliances — the physical system is what ships. A chip designer optimizing for area alone may pick a package that's thermally infeasible, or an enclosure material that doubles the weight.

SWaP-C brings four system-level objectives into the design loop alongside power and latency:

| Objective | Why It Matters |
|-----------|---------------|
| **Weight** (grams) | Payload budgets for drones, wearables, handheld devices |
| **Volume** (cm³) | Enclosure size, rack density, form factor compliance |
| **Power** (watts) | Battery life, thermal envelope, cooling requirements |
| **Cost** (USD) | Bill of materials, NRE amortization, volume pricing |

Plus **thermal feasibility** — because a design that can't dissipate its own heat is a design that doesn't work.

## What Gets Analyzed

The SWaP-C pipeline builds a hierarchical Bill of Materials (BOM) from SoC parameters:

```
System
├── PCB (FR4, copper layers, connectors)
│   └── Package (QFN / BGA / FCBGA / WLCSP)
│       └── Die (silicon, process-dependent)
├── Heatsink / Cooling (passive / active fan / liquid)
└── Enclosure (aluminum / ABS plastic / magnesium)
```

Each component contributes weight, volume, cost, and thermal resistance. The pipeline walks this tree to produce system-level totals and checks thermal feasibility against junction temperature limits.

### Physical Estimators

| Component | Key Parameters | What's Estimated |
|-----------|---------------|-----------------|
| **Die** | Area (mm²), process (nm) | Weight, dimensions, manufacturing cost |
| **Package** | Type, pin count | Substrate weight, thermal resistance (θ_jc) |
| **PCB** | Layer count, connectors | Board area, weight, routing cost |
| **Heatsink** | TDP, cooling type | Weight per watt, volume per watt, θ_sa |
| **Enclosure** | Material, wall thickness | Shell weight from surface area × density |

### Thermal Model

Junction temperature is computed as:

```
T_junction = T_ambient + TDP × (θ_jc + θ_sa)
```

Where θ_jc comes from the package type and θ_sa from the cooling solution. If T_junction exceeds the maximum (typically 125°C), the design is thermally infeasible.

## Five Optimization Methodologies

Raw SWaP-C numbers tell you *what*. The five optimization methodologies tell you *what to do about it*. Each methodology builds on the physical estimators to answer a specific design question:

| Step | Methodology | Question | Command |
|------|-------------|----------|---------|
| **Score** | Weighted FoM | How good is this design for my mission? | `branes swap score` |
| **Understand** | Sensitivity | Which parameters matter most? | `branes swap sensitivity` |
| **Explore** | Pareto + TOPSIS | What are my best options? | `branes swap rank` |
| **Compare** | Delta & Sweep | What happens if I change one thing? | `branes swap sweep` |
| **Commit** | Monte Carlo | Will this design meet my budgets? | `branes swap budget` |

### Mission Profiles

Mission profiles define what "good" means for a deployment context. Each profile assigns weights to the six SWaP-C objectives:

| Profile | Top Priorities | Use Case |
|---------|---------------|----------|
| `drone` | Weight 0.35, Power 0.25 | Payload + battery life dominate |
| `rack` | Cost 0.30, Power 0.30 | TCO + cooling capacity |
| `wearable` | Weight 0.25, Volume 0.25, Power 0.25 | Every physical dimension tight |
| `vehicle` | Cost 0.35, Latency 0.30 | Unit economics + real-time response |

Custom profiles can be derived from pairwise comparison matrices using the built-in AHP (Analytic Hierarchy Process) helper.

### Scoring (M1)

Score a design against a mission profile to get a composite Figure of Merit (0–100):

```bash
branes swap score --area 50 --power 5 --process 28 --profile drone
```

Each objective is normalized against ideal (best possible) and anti-ideal (worst possible) bounds, then weighted by the mission profile. A score of 78 means the design captures 78% of the theoretically achievable performance for that mission.

### Sensitivity Analysis (M2)

Understand which design parameters have the biggest impact:

```bash
# Tornado diagram — one-at-a-time perturbation
branes swap sensitivity --area 50 --power 5 --process 28 --mode tornado

# Taguchi L18 screening — orthogonal array, 18 evaluations
branes swap sensitivity --area 50 --power 5 --process 28 --mode taguchi
```

**Tornado** shows the swing (high − low) for each variable on each objective, sorted by impact. **Taguchi** computes main effects and signal-to-noise ratios to rank the top factors driving design quality.

### Pareto Ranking (M3)

After an `explore` run, rank the Pareto-front designs by mission profile using TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution):

```bash
branes swap rank --profile drone --method topsis --top 5
```

The enhanced `show-front` command adds clustering and scoring columns:

```bash
branes swap show-front --cluster --profile drone --top 10
```

Clustering groups similar designs into families (e.g., "low-power passively cooled"), while the profile score shows each design's TOPSIS closeness to the mission ideal.

### Parametric Sweep (M4)

Sweep one variable while holding others fixed to see how objectives respond:

```bash
branes swap sweep --area 50 --power 5 --process 28 \
    --param process_nm --from 28 --to 5 --steps 5
```

The output table shows all six objectives at each step, making it easy to spot diminishing returns or constraint violations.

### Monte Carlo Budget Feasibility (M5)

Before committing to a design, run Monte Carlo simulation to assess the probability of meeting budgets under manufacturing uncertainty:

```bash
branes swap budget --area 50 --power 5 --process 28 \
    --max-weight 200 --max-cost 1000 --samples 1000
```

Each sample perturbs continuous BOM inputs (area ±5%, power ±15%, thermal ±10%) and checks all budgets. The output shows P10/P50/P90 percentiles and a traffic-light indicator:

- **Green** (≥90% feasible): Commit with confidence
- **Yellow** (50–90%): Design margins are tight — consider de-risking
- **Red** (<50%): High probability of budget violation — redesign needed

## The `branes swap` Commands

Twelve commands covering the full workflow:

```
Questions:     branes swap estimate      →  What does this design weigh?
               branes swap bom          →  Where does the weight come from?

Assertions:    branes swap check        →  Does it pass my budgets?

Optimization:  branes swap explore      →  Find the 6-objective Pareto front
               branes swap show-front   →  View results (with clustering/scoring)

Methodologies: branes swap score        →  Weighted FoM for a mission profile
               branes swap rank         →  TOPSIS/FoM ranking of Pareto designs
               branes swap sensitivity  →  Tornado + Taguchi screening
               branes swap sweep        →  Single-variable parametric sweep
               branes swap budget       →  Monte Carlo budget feasibility

Comparisons:   branes swap compare      →  QFN vs FCBGA side-by-side
               branes swap explain      →  Why is design #0 better than #3?
```

### Quick Estimate

Get system-level numbers for a single design point in one command:

```bash
branes swap estimate --area 50 --power 5 --process 28
```

```
╭─── SWaP-C Estimate: 50mm² / 28nm / BGA / passive ───╮
│  Weight:    51.9 g                                    │
│  Volume:    26.2 cm³                                  │
│  Cost:      $758.18                                   │
│  Dims:      27×27×44 mm                               │
│  Thermal:   Tj=100°C (margin: 25°C) ✓                │
╰──────────────────────────────────────────────────────╯
```

### BOM Breakdown

See where weight and cost come from:

```bash
branes swap bom --area 50 --power 5 --process 28 --package FCBGA
```

```
          System BOM Breakdown
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┓
┃ Component       ┃ Level   ┃ Wt (g)   ┃ Vol(cm³)┃ Cost($)┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━┩
│ die             │ die     │     0.09 │    0.04 │ 753.39 │
│ FCBGA-256       │ package │     0.44 │    0.27 │   3.00 │
│ pcb             │ pcb     │     1.01 │    0.39 │   3.18 │
│ heatsink        │ system  │    40.00 │   19.90 │   0.00 │
│ enclosure-alum  │ system  │    11.28 │    6.49 │   0.56 │
├─────────────────┼─────────┼──────────┼─────────┼────────┤
│ TOTAL           │         │    52.82 │   27.09 │ 760.13 │
└─────────────────┴─────────┴──────────┴─────────┴────────┘
```

### Budget Checking

Assert SWaP-C budgets and get PASS/FAIL verdicts — with exit code 1 on failure for CI integration:

```bash
branes swap check --area 50 --power 5 --process 28 \
    --max-weight 500 --max-volume 200 --max-cost 1000
```

You can also read budgets from the [spec store](/features/constraint-checking/):

```bash
branes swap check --area 50 --power 5 --process 28 --from-spec my-drone
```

### 6-Objective Optimization

Extend the [design optimization](/features/design-optimization/) engine with weight and volume objectives:

```bash
branes swap explore --goal "drone SoC" \
    --power 10 --weight 500 --volume 200 --fast
```

This runs the same 3-layer optimization pipeline (MAP-Elites → Bayesian/NSGA-III) but with 9 design variables (adding package type and cooling type) and 6 objectives (adding weight and volume).

### Configuration Comparison

Compare two packaging approaches on the same SoC:

```bash
branes swap compare --area 50 --power 5 --process 28 \
    --left "QFN,passive,abs_plastic" \
    --right "FCBGA,active_fan,aluminum"
```

## Design Variables

The SWaP-C design space extends the SoC variables with packaging choices:

| Variable | Type | Options | What it Controls |
|----------|------|---------|-----------------|
| `package_type` | Categorical | QFN, BGA, FCBGA, WLCSP | Weight, thermal resistance, cost |
| `cooling_type` | Categorical | passive, active_fan, liquid | Weight per watt, max TDP, thermal path |
| `enclosure_material` | Choice | aluminum, abs_plastic, magnesium | Shell weight, EMI shielding, cost |

Plus the 7 SoC architecture variables (process, clock, array size, SRAM, tiles, NoC width).

## Packaging Options

| Package | θ_jc (°C/W) | Body Height | Best For |
|---------|-------------|-------------|----------|
| **QFN** | 3.0 | 0.85 mm | Low-power edge (<5W), cost-sensitive |
| **BGA** | 2.0 | 1.70 mm | General purpose (5–15W) |
| **FCBGA** | 1.0 | 2.50 mm | High performance (>15W), best thermal |
| **WLCSP** | 5.0 | 0.50 mm | Ultra-compact, lowest weight |

## Cooling Options

| Cooling | Weight/Watt | Max TDP | Parasitic Power | Best For |
|---------|-------------|---------|----------------|----------|
| **Passive** | 8 g/W | 15W | 0W | Battery-powered, silent operation |
| **Active fan** | 3 g/W | 100W | 1.5W | Desktop/NUC, performance workloads |
| **Liquid** | 2 g/W | 500W | 5W | Data center, extreme thermal loads |

## Python API

```python
from embodied_ai_architect.graphs.physical_estimators import (
    estimate_system_bom, compute_thermal_feasibility,
)
from embodied_ai_architect.graphs.swap_report import assess_design_point

# Build system BOM
bom = estimate_system_bom(
    soc_area_mm2=220, soc_power_watts=45, process_nm=4,
    package_type="FCBGA", cooling_type="active_fan",
    enclosure_material="aluminum",
)

# Check thermal feasibility
thermal = compute_thermal_feasibility(bom, tdp_watts=45, ambient_temp_c=35)
print(f"Tj = {thermal['junction_temp_c']}°C, feasible = {thermal['feasible']}")

# Assess against budgets
design = {
    "objectives": {
        "power_watts": 45, "weight_grams": bom.total_weight_grams(),
        "volume_cm3": bom.total_volume_cm3(), "cost_usd": bom.total_cost_usd(),
    },
    "design_params": {"package_type": "FCBGA", "cooling_type": "active_fan"},
}
scorecard = assess_design_point(
    design, constraints={"max_weight_grams": 800, "max_volume_cm3": 700},
    thermal_data=thermal,
)
print(f"Overall: {scorecard.overall_verdict}")
```

## Next Steps

- [SWaP-C Tutorial](/tutorials/swap-analysis/) — Hands-on walkthrough with NUC and Jetson examples
- [Optimization Workflow Tutorial](/tutorials/swap-optimization-workflow/) — End-to-end Score → Commit workflow
- [Design Optimization](/features/design-optimization/) — The underlying MOO engine
- [CLI Reference](/reference/cli/) — All `swap` command options
- [Hardware Catalog](/catalog/hardware/) — Available hardware targets
