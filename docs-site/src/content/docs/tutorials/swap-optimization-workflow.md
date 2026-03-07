---
title: SWaP-C Optimization Workflow
description: End-to-end guide to scoring, analyzing, exploring, and de-risking system-level SWaP-C designs using mission profiles and the five optimization methodologies.
---

You have a design. It weighs something, costs something, draws some power. The [SWaP-C Analysis tutorial](/tutorials/swap-analysis/) showed you how to estimate, compare, and check those numbers. But how do you go from a single design point to a confident, optimized design *decision*?

This tutorial walks through the five optimization methodologies that turn raw SWaP-C data into actionable engineering decisions. You'll follow a single design scenario — a drone perception SoC — from initial scoring through sensitivity analysis, design exploration, and probabilistic budget sign-off.

## The Five Methodologies

Each methodology answers a different question in the design process:

| Step | Methodology | Question It Answers | Command |
|------|------------|---------------------|---------|
| 1. **Score** | Weighted Figure of Merit | How good is this design *for my application*? | `branes swap score` |
| 2. **Understand** | Sensitivity Analysis | Which design knobs matter most? | `branes swap sensitivity` |
| 3. **Explore** | Pareto + TOPSIS Ranking | What are the best designs, and which wins for my mission? | `branes swap explore` + `rank` |
| 4. **Compare** | Parametric Sweep | How does one parameter affect all six objectives? | `branes swap sweep` |
| 5. **Commit** | Monte Carlo Feasibility | Will this design meet its budgets in production? | `branes swap budget` |

They build on each other. Scoring defines what "good" means. Sensitivity tells you where to focus. Exploration finds the frontier. Ranking picks winners. Monte Carlo gives you the confidence to commit.

## Prerequisites

- **Branes** installed (`pip install -e ".[dev]"`)
- No API keys or optional dependencies required
- Complete the [SWaP-C Analysis tutorial](/tutorials/swap-analysis/) first — we'll build on those concepts

## The Design Scenario

You're designing the compute module for a delivery drone. The SoC runs a real-time perception stack: YOLO object detection, stereo depth, obstacle avoidance.

**Baseline design:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Die area | 100 mm² | Mid-range perception SoC |
| Process | 14 nm | Mature node, good cost/performance |
| TDP | 10 W | Battery-constrained, 45 min flight time |
| Package | BGA | Standard for SoM mounting |
| Cooling | Active fan | Needed for sustained 10W in sealed airframe |
| Enclosure | ABS plastic | Lightweight for drone payload |

**Drone payload budgets:** 150 g compute weight, 100 cm³ volume, $800 cost.

---

## Step 1: Score — How Good Is This Design?

A single number doesn't capture a 6-objective design. But a *weighted* single number — tailored to your deployment — tells you how well this design serves *your* mission.

### Mission Profiles

Branes ships four preset profiles. Each assigns weights to the six SWaP-C objectives based on what matters most for that deployment:

| Profile | Top Priorities | Rationale |
|---------|---------------|-----------|
| `drone` | Weight (0.35), Volume (0.25), Power (0.25) | Payload and battery life dominate |
| `rack` | Cost (0.30), Power (0.30) | TCO and cooling capacity |
| `wearable` | Weight (0.25), Volume (0.25), Power (0.25) | Every physical dimension is tight |
| `vehicle` | Cost (0.35), Latency (0.30) | Unit economics and real-time response |

### Score the Baseline

```bash
branes swap score \
    --area 100 --power 10 --process 14 \
    --package BGA --cooling active_fan --enclosure abs_plastic \
    --profile drone
```

The output is a panel showing a composite score (0-100) with per-objective breakdowns:

```
╭──── SWaP-C Score: 72/100 (drone profile) ────────────────────────╮
│  power_watts        10.0W    ████████████████░░░░  75/100         │
│  latency_ms          0.0ms   ████████████████████ 100/100         │
│  area_mm2          100.0mm²  ██████████░░░░░░░░░░  50/100         │
│  cost_usd          $952      ██████████████░░░░░░  52/100         │
│  weight_grams       53.2g    ████████████████████░  92/100         │
│  volume_cm3         30.1cm³  ████████████████████░  85/100         │
╰──────────────────────────────────────────────────────────────────╯
```

**Interpretation:** 72/100 for a drone mission. Weight and volume score well (the physical package is small and light), but cost is mediocre and area is middling. The composite score is dominated by weight (0.35 weight) and power (0.25), where this design performs reasonably.

### Compare Across Profiles

The same design scores differently for different missions:

```bash
# How does it look for a rack deployment?
branes swap score \
    --area 100 --power 10 --process 14 \
    --package BGA --cooling active_fan --enclosure abs_plastic \
    --profile rack
```

For a rack, the score drops because rack deployments weight cost and power more heavily — and this 14nm design's cost is its weakness.

:::tip
Use `--json-output` to capture scores programmatically. Pipe into `jq .composite_score` for CI dashboards.
:::

---

## Step 2: Understand — What Knobs Matter Most?

Before exploring the design space, you need to know which parameters have the biggest impact. Twiddling a knob that moves objectives by 0.1% is a waste of engineering time.

### Tornado Analysis

Tornado analysis varies each parameter one-at-a-time across its full range while holding others fixed, then ranks variables by the size of the resulting swing:

```bash
branes swap sensitivity \
    --area 100 --power 10 --process 14 \
    --package BGA --cooling active_fan --enclosure abs_plastic \
    --mode tornado
```

```
               Tornado Sensitivity Analysis
┏━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┓
┃ Variable    ┃ Objective      ┃    Low ┃   Base ┃   High ┃  Swing ┃
┡━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━┩
│ power       │ weight_grams   │  21.30 │  53.20 │ 173.40 │ 152.10 │
│ area        │ cost_usd       │ 345.00 │ 952.00 │3720.00 │3375.00 │
│ power       │ volume_cm3     │   9.50 │  30.10 │  97.80 │  88.30 │
│ cooling     │ weight_grams   │  32.10 │  53.20 │ 103.40 │  71.30 │
│ area        │ weight_grams   │  41.20 │  53.20 │  85.70 │  44.50 │
│ ...         │                │        │        │        │        │
└─────────────┴────────────────┴────────┴────────┴────────┴────────┘
```

**Insight:** Power is the dominant driver of weight (152g swing across the power range). Die area dominates cost (a 20× swing from small to large die). Cooling type matters significantly for weight. Process node barely registers — for *physical* SWaP-C, the package and cooling choices dwarf the semiconductor process.

### Focus on One Objective

If you only care about weight:

```bash
branes swap sensitivity \
    --area 100 --power 10 --process 14 \
    --package BGA --cooling active_fan --enclosure abs_plastic \
    --mode tornado --objective weight_grams
```

### Taguchi L18 Screening

For a more rigorous multi-factor screening that accounts for interactions between variables, use Taguchi's L18 orthogonal array. It evaluates all 7 design variables simultaneously in exactly 18 runs:

```bash
branes swap sensitivity \
    --area 100 --power 10 --process 14 \
    --package BGA --cooling active_fan --enclosure abs_plastic \
    --mode taguchi
```

```
Top factors: power, area, cooling, enclosure, layer_count, package, process

               Taguchi L18 Main Effects
┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃ Variable     ┃ weight_grams┃   cost_usd  ┃  volume_cm3 ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
│ power        │     132.4200│     204.8100│      72.3800│
│ area         │      38.1200│    2847.6300│      12.4500│
│ cooling      │      64.7300│       0.0000│      38.2100│
│ ...          │             │             │             │
└──────────────┴─────────────┴─────────────┴─────────────┘
```

**Insight:** Taguchi confirms the tornado findings — power and area are the biggest levers — but also reveals that cooling has a strong interaction effect on volume that the one-at-a-time tornado missed. For the drone, TDP and cooling type are the two decisions that matter most.

---

## Step 3: Explore — Find the Pareto Front and Rank by Mission

Now that you know which knobs matter, explore the full design space to find all non-dominated designs, then rank them by your mission profile.

### Run the Exploration

```bash
branes swap explore \
    --goal "drone perception SoC" \
    --power 15 --weight 150 --volume 100 --cost 800 \
    --fast --workers 4
```

This runs the 6-objective optimizer (MAP-Elites in fast mode) across 9 design variables — the 7 SoC architecture variables plus package type and cooling type.

### View the Pareto Front with Clustering

A raw Pareto front can have dozens of non-dominated designs. Clustering groups them into families so you can see the major design archetypes:

```bash
branes swap show-front --top 10 --cluster --profile drone
```

The `--cluster` flag adds a "Family" column labeling each design by its dominant characteristic (e.g., "low-power", "low-weight", "low-cost"). The `--profile drone` flag adds a TOPSIS score column so you can immediately see which designs best match the drone mission.

### Rank by Mission Profile

For a definitive ranking, use TOPSIS — a multi-criteria decision method that accounts for distance to both the ideal and anti-ideal points, weighted by your mission profile:

```bash
branes swap rank --profile drone --method topsis --top 5
```

```
         TOPSIS Ranking (drone profile, top 5)
┏━━━━━━┳━━━━┳━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┓
┃ Rank ┃  # ┃  Score ┃ Power (W) ┃ Cost ($) ┃ Weight(g) ┃ Vol (cm³) ┃
┡━━━━━━╇━━━━╇━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━┩
│    1 │  3 │ 0.8214 │      4.20 │   385.00 │     28.40 │     14.80 │
│    2 │  7 │ 0.7856 │      6.10 │   290.00 │     35.60 │     19.20 │
│    3 │  1 │ 0.7102 │      3.80 │   720.00 │     22.10 │     12.50 │
│    4 │  0 │ 0.6543 │      8.50 │   180.00 │     48.20 │     27.30 │
│    5 │  5 │ 0.5891 │     12.00 │   150.00 │     62.40 │     35.10 │
└──────┴────┴────────┴───────────┴──────────┴───────────┴───────────┘
```

**Interpretation:** Design #3 wins for the drone mission — it's light (28g), low-power (4.2W), and compact (14.8 cm³). Design #0 is the cheapest but heavier. The TOPSIS score quantifies how close each design is to the weighted ideal.

### Switch Profiles to See Different Winners

The same Pareto front, ranked for a vehicle mission:

```bash
branes swap rank --profile vehicle --method topsis --top 5
```

Different weights produce a different ranking — the cheapest, lowest-latency design rises to the top because vehicle missions prioritize cost and real-time response over physical size.

:::note
You can also rank by simple weighted FoM instead of TOPSIS: `--method fom`. TOPSIS is preferred because it considers distance to both the ideal and anti-ideal points, which handles outliers better.
:::

---

## Step 4: Compare — Sweep a Critical Parameter

You've identified that power is the biggest lever (from Step 2) and Design #3 looks promising (from Step 3). Before committing, you want to understand exactly how TDP affects all six objectives.

### Parametric Sweep

Sweep TDP from 3W to 15W while holding all other parameters at the baseline:

```bash
branes swap sweep \
    --area 100 --power 10 --process 14 \
    --package BGA --cooling active_fan --enclosure abs_plastic \
    --param power --from 3 --to 15 --steps 5
```

```
                Parametric Sweep: power
┏━━━━━━━━┳━━━━━━━┳━━━━━━━━━┳━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┓
┃  power ┃ power ┃ latency ┃  area ┃   cost ┃ weight ┃ volume ┃
┡━━━━━━━━╇━━━━━━━╇━━━━━━━━━╇━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━┩
│    3.0 │  3.00 │    0.00 │100.00 │ 952.18 │  32.20 │  16.50 │
│    6.0 │  6.00 │    0.00 │100.00 │ 952.18 │  41.20 │  22.30 │
│    9.0 │  9.00 │    0.00 │100.00 │ 952.18 │  50.20 │  28.10 │
│   12.0 │ 12.00 │    0.00 │100.00 │ 952.18 │  59.20 │  33.90 │
│   15.0 │ 15.00 │    0.00 │100.00 │ 952.18 │  68.20 │  39.70 │
└────────┴───────┴─────────┴───────┴────────┴────────┴────────┘
```

**Insight:** Weight scales linearly with power at ~3g/W (from the active fan heatsink at 3 g/W). Volume follows the same pattern. Cost and area don't change because they're driven by die size, not TDP. Going from 10W to 5W saves 15g — meaningful for a drone with a 150g budget.

### Sweep a Categorical Variable

For categorical variables like cooling type, the sweep automatically cycles through all options:

```bash
branes swap sweep \
    --area 100 --power 10 --process 14 \
    --package BGA --enclosure abs_plastic \
    --param cooling
```

This shows how switching from passive to active fan to liquid changes all six objectives in one table. No `--from` or `--to` needed.

---

## Step 5: Commit — Probabilistic Budget Feasibility

You've picked a design. The point estimate looks good. But physical manufacturing has tolerances — die area varies wafer to wafer, TDP varies with binning, thermal resistance depends on assembly quality. Will this design *reliably* meet its budgets in production?

### Monte Carlo Simulation

Monte Carlo perturbs the continuous design parameters (area ±5%, power ±15%, ambient temperature ±5%) across 1000 samples to build a probability distribution for each objective:

```bash
branes swap budget \
    --area 100 --power 10 --process 14 \
    --package BGA --cooling active_fan --enclosure abs_plastic \
    --max-weight 150 --max-volume 100 --max-cost 800 \
    --samples 1000
```

```
          Monte Carlo Budget Feasibility (1000 samples)
┏━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┓
┃ Metric        ┃ Budget ┃    P10 ┃    P50 ┃    P90 ┃  P(OK) ┃ Status ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━┩
│ weight_grams  │  150.0 │   44.8 │   53.1 │   63.2 │  99.2% │ GREEN  │
│ volume_cm3    │  100.0 │   24.7 │   30.0 │   36.8 │  99.8% │ GREEN  │
│ cost_usd      │  800.0 │  768.2 │  952.1 │1152.3  │  18.4% │ RED    │
└───────────────┴────────┴────────┴────────┴────────┴────────┴────────┘
 Overall feasibility: 18.2%
```

### Reading the Traffic Lights

| Color | Meaning | P(OK) | Action |
|-------|---------|-------|--------|
| **GREEN** | Budget met with high confidence | >= 90% | Proceed |
| **YELLOW** | Marginal — may fail in some production lots | 50-90% | Add margin or re-examine |
| **RED** | Budget will likely be violated | < 50% | Redesign or relax constraint |

**Interpretation:** Weight and volume are green — even with manufacturing variation, this design comfortably fits the drone. But cost is red at 18.4% — the 14nm die's manufacturing cost exceeds the $800 budget in most scenarios. The P50 (median) is $952, well above budget.

### What to Do About the Red Light

Options:
1. **Relax the cost budget** — if $1000 is acceptable, re-run with `--max-cost 1000`
2. **Reduce die area** — use `branes swap sweep --param area --from 50 --to 100` to find the area where cost drops below $800
3. **Change process node** — a mature 28nm process has much lower NRE; sweep it with `branes swap sweep --param process`
4. **Increase production volume** — use `--volume 100000` (amortizes NRE across more units)

### Increase Samples for Confidence

For final design reviews, use more samples:

```bash
branes swap budget \
    --area 100 --power 10 --process 14 \
    --package BGA --cooling active_fan --enclosure abs_plastic \
    --max-weight 150 --max-volume 100 --max-cost 1200 \
    --samples 5000
```

More samples narrow the confidence intervals. For production sign-off, 5000-10000 samples is typical.

---

## Putting It All Together

Here's the complete workflow as a shell script. A hardware team can run this in 30 seconds to go from baseline design to budget sign-off:

```bash
#!/bin/bash
# SWaP-C Optimization Workflow: Drone Perception SoC

COMMON="--area 100 --power 10 --process 14 \
    --package BGA --cooling active_fan --enclosure abs_plastic"

# 1. Score: How good is this design for a drone?
echo "=== Step 1: Score ==="
branes swap score $COMMON --profile drone

# 2. Understand: Which parameters matter most?
echo "=== Step 2: Sensitivity ==="
branes swap sensitivity $COMMON --mode tornado --objective weight_grams

# 3. Explore: Find the Pareto front
echo "=== Step 3: Explore ==="
branes swap explore --goal "drone perception SoC" \
    --power 15 --weight 150 --volume 100 --fast

# 4. Rank: Which Pareto-front design wins for a drone?
echo "=== Step 4: Rank ==="
branes swap rank --profile drone --method topsis --top 5

# 5. Sweep: How does TDP affect the selected design?
echo "=== Step 5: Sweep ==="
branes swap sweep $COMMON --param power --from 3 --to 15 --steps 5

# 6. Commit: Will it meet budgets in production?
echo "=== Step 6: Budget ==="
branes swap budget $COMMON \
    --max-weight 150 --max-volume 100 --max-cost 1200 \
    --samples 1000
```

### The Workflow Loop

In practice, the five steps form a loop, not a line:

```
                ┌─────────────────────────────────────────┐
                │                                         │
  Score ──→ Understand ──→ Explore ──→ Compare ──→ Commit │
    ▲                                               │     │
    │         adjust design, relax budget,           │     │
    └──────── or change mission profile ◄────────────┘     │
                                                           │
                                       GREEN on all ──→ Ship
```

A red traffic light sends you back to an earlier step — adjust the design (explore again with different constraints), relax the budget (if the system can tolerate it), or change the mission profile (if priorities have shifted).

---

## JSON Output for Automation

Every command supports `--json-output` for pipeline integration:

```bash
# Score as JSON
branes swap score --area 100 --power 10 --process 14 \
    --profile drone --json-output | jq .composite_score

# Sensitivity as JSON (for custom plotting)
branes swap sensitivity --area 100 --power 10 --process 14 \
    --mode tornado --json-output | jq '.bars[:5]'

# Sweep as JSON (for charting in Python/JS)
branes swap sweep --area 100 --power 10 --process 14 \
    --param power --from 3 --to 15 --steps 10 \
    --json-output > sweep_results.json

# Budget as JSON (for design review dashboards)
branes swap budget --area 100 --power 10 --process 14 \
    --max-weight 150 --max-cost 1200 --samples 1000 \
    --json-output | jq '{overall: .overall_feasibility, lights: .traffic_light}'
```

## Python API

All five methodologies are available as pure functions for scripting:

```python
from embodied_ai_architect.graphs.swap_profiles import get_profile, list_profiles
from embodied_ai_architect.graphs.swap_analysis import (
    compute_fom_score,
    tornado_analysis,
    topsis_rank,
    parametric_sweep,
    monte_carlo_feasibility,
)

# Load a mission profile
profile = get_profile("drone")

# Score a design
objectives = {
    "power_watts": 10.0, "latency_ms": 5.0, "area_mm2": 100.0,
    "cost_usd": 952.0, "weight_grams": 53.2, "volume_cm3": 30.1,
}
fom = compute_fom_score(objectives, profile.weights, profile_name="drone")
print(f"Score: {fom.composite_score}/100")

# Rank a set of designs
designs = [objectives, {**objectives, "power_watts": 5, "weight_grams": 30}]
ranked = topsis_rank(designs, profile.weights)
for r in ranked.rankings:
    print(f"  #{r['_original_index']}: TOPSIS={r['topsis_score']:.4f}")
```

## Command Reference

| Command | Purpose | Key Options |
|---------|---------|-------------|
| `branes swap score` | Weighted FoM (0-100) | `--profile`, common SoC options |
| `branes swap sensitivity` | Tornado or Taguchi | `--mode tornado\|taguchi`, `--objective` |
| `branes swap sweep` | Single-parameter sweep | `--param`, `--from`, `--to`, `--steps` |
| `branes swap rank` | TOPSIS/FoM ranking of Pareto front | `--profile`, `--method fom\|topsis`, `--top` |
| `branes swap budget` | Monte Carlo feasibility | `--max-weight`, `--max-cost`, `--samples` |
| `branes swap show-front` | View Pareto front | `--cluster`, `--profile`, `--top` |

See the [CLI Reference](/reference/cli/) for all options.

## Next Steps

- [SWaP-C Analysis Tutorial](/tutorials/swap-analysis/) — Estimate, compare, and check with NUC and Jetson examples
- [SWaP-C Feature Overview](/features/swap-analysis/) — How the BOM estimators and thermal model work
- [Design Optimization](/features/design-optimization/) — The underlying 6-objective MOO engine
- [CLI Reference](/reference/cli/) — All `branes swap` command options
