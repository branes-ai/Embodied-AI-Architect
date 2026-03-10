# Plan: Five SWaP-C Optimization Methodology Workflow

## Context

The `branes swap` CLI (7 commands) and its supporting infrastructure (physical estimators,
BOM model, scorecard, SWaPCEvaluator, MOO engine) are complete and shipping. The research
document `docs/research/swap-optimization-methodologies.md` identifies five interconnected
optimization methodologies — Score, Understand, Explore, Compare, Commit — that transform
the raw SWaP-C data into actionable design decisions.

This plan adds the five methodology implementations as a library + CLI surface, making
SWaP-C optimization practical and productive for hardware design teams.

---

## Methodology Dependency Graph

```
M1 (Mission Profiles + FoM Scoring) ─── foundation for M3 (TOPSIS weights)
                                    └── foundation for M5 (budget definitions)
M2 (Sensitivity / Tornado / Taguchi) ── independent, uses evaluator directly
M3 (Pareto + TOPSIS + MRS + Cluster) ── depends on M1 for weights
M4 (Delta Attribution + Sweep) ──────── independent, extends compare/explain
M5 (Monte Carlo Budget Feasibility) ─── depends on M1 for budget profiles
```

---

## New CLI Commands

| Command | Methodology | Purpose |
|---------|-------------|---------|
| `branes swap score` | M1 | Weighted FoM score (0–100) for a single design point |
| `branes swap rank` | M1+M3 | TOPSIS/FoM ranking of Pareto-front designs by mission profile |
| `branes swap sensitivity` | M2 | Tornado diagrams, Taguchi L18 screening |
| `branes swap sweep` | M4 | Parametric sweep of one variable across all 6 objectives |
| `branes swap budget` | M5 | Monte Carlo probabilistic budget feasibility |

Existing `show-front` enhanced with `--cluster` and `--profile` flags (M3).

---

## File Plan

### New Files

| File | Lines (est.) | Purpose |
|------|-------------|---------|
| `src/embodied_ai_architect/graphs/swap_profiles.py` | ~120 | Mission profile model, 4 presets, AHP helper |
| `src/embodied_ai_architect/graphs/swap_analysis.py` | ~600 | Pure-function library for all 5 methodologies |
| `tests/test_swap_profiles.py` | ~80 | Profile model tests |
| `tests/test_swap_analysis.py` | ~350 | Analysis library tests (~30 functions) |

### Modified Files

| File | Change |
|------|--------|
| `src/embodied_ai_architect/cli/commands/swap.py` | Add 5 new subcommands, enhance `show-front` |
| `src/embodied_ai_architect/graphs/swap_report.py` | Add `fom_score` field to SWaPCScorecard |
| `tests/test_swap_cli.py` | Add ~15 CLI tests for new commands |

---

## Phase 1: Mission Profiles (`swap_profiles.py`)

### `MissionProfile` model (Pydantic)

```python
class MissionProfile(BaseModel):
    name: str
    description: str = ""
    weights: dict[str, float]      # 6 objectives → weights summing to 1.0
    budgets: dict[str, float] = {}  # optional max_weight_grams, etc.
```

### 4 Preset Profiles

| Profile | Top weights | Rationale |
|---------|-------------|-----------|
| `drone` | weight 0.35, power 0.25 | Payload + battery life dominate |
| `rack` | cost 0.30, power 0.30 | TCO + cooling capacity |
| `wearable` | weight 0.25, volume 0.25, power 0.25 | Every physical dimension tight |
| `vehicle` | cost 0.35, latency 0.30 | Unit economics + real-time response |

### Functions

- `get_profile(name) → MissionProfile` — look up preset or raise KeyError
- `list_profiles() → list[str]` — list available preset names
- `ahp_weights(pairwise_matrix, objectives) → dict[str, float]` — derive weights from
  AHP pairwise comparison matrix (geometric mean method, consistency check)

---

## Phase 2: Analysis Library (`swap_analysis.py`)

Pure functions, no CLI/Rich dependencies. All accept `evaluator_fn: Callable[[dict], dict]`
so they work with mock evaluators in tests and with the real BOM estimator in production.

### M1: Weighted FoM Scoring

```python
class FoMResult(BaseModel):
    composite_score: float        # 0–100
    per_objective_scores: dict[str, float]
    profile_name: str = ""

def compute_fom_score(objectives, weights, ideal=None, anti_ideal=None) → FoMResult
def rank_designs_by_fom(designs, weights, ...) → list[tuple[int, FoMResult]]
```

Normalization: `score_i = 100 × (anti_ideal_i − value_i) / (anti_ideal_i − ideal_i)`,
clamped to [0, 100]. Composite = Σ(weight_i × score_i).

If ideal/anti_ideal not provided, derive from the design set (min/max per objective).

### M2: Sensitivity Analysis

```python
class TornadoBar(BaseModel):
    variable_name: str; objective_name: str
    base_value: float; low_value: float; high_value: float; swing: float

class TornadoResult(BaseModel):
    bars: list[TornadoBar]; base_design: dict

class TaguchiResult(BaseModel):
    main_effects: dict[str, dict[str, float]]  # var → obj → effect
    sn_ratios: dict[str, dict[str, float]]     # var → obj → S/N
    top_factors: list[str]

def tornado_analysis(base_params, evaluator_fn, design_space_variables, ...) → TornadoResult
def taguchi_l18_screening(evaluator_fn, design_space_variables, ...) → TaguchiResult
```

**Tornado:** For each of 9 variables, evaluate at ±1σ (continuous/integer) or all levels
(categorical). ~18–24 evaluations total. Bars sorted by swing magnitude.

**Taguchi L18:** Constant L18(2¹×3⁷) orthogonal array stored as a list. Each variable
discretized to 3 levels via `_discretize_variable(var) → [low, mid, high]`. Main effects
computed as average objective at each level minus grand mean. S/N ratio: −10·log₁₀(mean(y²))
("smaller is better" for all SWaP-C objectives). 18 evaluations exactly.

### M3: Pareto-Guided Exploration

```python
class TOPSISResult(BaseModel):
    rankings: list[dict]       # designs with topsis_score added
    ideal_point: dict[str, float]
    anti_ideal_point: dict[str, float]

class MRSResult(BaseModel):
    knee_index: int
    exchange_rates: dict[str, float]  # "weight_grams/power_watts" → rate

class ClusterResult(BaseModel):
    n_clusters: int
    clusters: list[dict]       # {id, centroid, n_members, label}
    labels: list[int]

def topsis_rank(designs, weights, objectives=None) → TOPSISResult
def marginal_rate_of_substitution(pareto_front, knee_index, ...) → MRSResult
def cluster_pareto_front(pareto_front, n_clusters=4, ...) → ClusterResult
```

**TOPSIS:** Vector normalization → weighted matrix → ideal/anti-ideal → Euclidean distances
→ closeness coefficient C_i = d⁻/(d⁺+d⁻). Pure numpy.

**MRS:** At knee point, compute finite-difference exchange rates to nearest neighbors on
the front for each objective pair. Output: "1W saved costs 12g of weight."

**Clustering:** Simple k-means on normalized objective space (~30 lines numpy). Label
clusters by dominant characteristic (e.g., "low-power passively cooled").

### M4: Delta Attribution & Parametric Sweep

```python
class DeltaAttribution(BaseModel):
    component_deltas: list[dict]  # {name, weight_delta, volume_delta, cost_delta}
    total_deltas: dict[str, float]

class SweepResult(BaseModel):
    variable_name: str
    sweep_values: list[Any]
    objective_traces: dict[str, list[float]]

def delta_attribution(left_bom_summary, right_bom_summary, ...) → DeltaAttribution
def parametric_sweep(base_params, variable_name, sweep_values, evaluator_fn) → SweepResult
```

**Delta attribution:** Match BOM components by name between two summaries. Compute
per-component weight/volume/cost delta. Identify largest contributor.

**Parametric sweep:** Vary one parameter, hold others fixed, evaluate at each step.
Return objective traces for table/chart display.

### M5: Monte Carlo Budget Feasibility

```python
class MonteCarloResult(BaseModel):
    n_samples: int
    p10: dict[str, float]
    p50: dict[str, float]
    p90: dict[str, float]
    mean: dict[str, float]
    std: dict[str, float]
    feasibility_prob: dict[str, float]  # metric → P(within budget)
    overall_feasibility: float          # P(all budgets met)
    traffic_light: dict[str, str]       # metric → green/yellow/red

def monte_carlo_feasibility(base_params, budgets, evaluator_fn,
                            n_samples=1000, seed=None) → MonteCarloResult
```

**Uncertainty model:** Perturb continuous BOM inputs (area ±5%, power ±15%, material
densities ±5%, thermal resistance ±10%) using normal distributions. Categorical variables
(package, cooling) are not perturbed — they're design choices, not uncertain quantities.

**Traffic light:** green ≥ 90%, yellow 50–90%, red < 50%.

---

## Phase 3: CLI Commands

### Common pattern

All new commands follow the existing pattern:
- `_common_soc_options` decorator for shared args
- Rich console output (Table, Panel)
- `--json-output` flag for machine-readable output
- `_make_evaluator_fn()` helper to create the `evaluator_fn` callable from CLI args

### `branes swap score`

```
branes swap score --area 50 --power 5 --process 28 --profile drone
```

Options: all `_common_soc_options` + `--profile` (default: drone) + `--json-output`.

Output: Rich Panel with composite score and per-objective breakdown:
```
╭─── SWaP-C Score: 78/100 (drone profile) ────────────╮
│  power_watts     5.0W    ██████████████████░░  85/100 │
│  weight_grams   52g      ████████████████████░  92/100│
│  volume_cm3     26cm³    ████████████████░░░░  78/100 │
│  cost_usd       $758     ██████░░░░░░░░░░░░░░  32/100 │
│  ...                                                   │
╰────────────────────────────────────────────────────────╯
```

### `branes swap rank`

```
branes swap rank --profile drone --method topsis --top 5
```

Options: `--profile`, `--method` (fom|topsis, default topsis), `--top`, `--json-output`.

Loads saved explore result via `_load_swap_result()`. Applies TOPSIS or FoM ranking.

### `branes swap sensitivity`

```
branes swap sensitivity --area 50 --power 5 --process 28 --mode tornado
branes swap sensitivity --area 50 --power 5 --process 28 --mode taguchi
```

Options: all `_common_soc_options` + `--mode` (tornado|taguchi, default tornado) +
`--objective` (focus on one, default all) + `--json-output`.

Tornado output: Rich Table with horizontal bars showing swing per variable.
Taguchi output: Rich Table with main effects and top factors.

### `branes swap sweep`

```
branes swap sweep --area 50 --power 5 --process 28 \
    --param process_nm --from 28 --to 5 --steps 5
```

Options: all `_common_soc_options` + `--param` (required) + `--from` + `--to` +
`--steps` (default 10) + `--json-output`.

For categorical params (package_type, cooling_type), `--from` and `--to` are ignored;
sweeps all choices.

Output: Rich Table with one row per step, columns for all 6 objectives.

### `branes swap budget`

```
branes swap budget --area 50 --power 5 --process 28 \
    --max-weight 200 --max-volume 150 --max-cost 1000 \
    --samples 1000 --confidence 0.90
```

Options: all `_common_soc_options` + `--max-weight` + `--max-volume` + `--max-power` +
`--max-cost` + `--confidence` (default 0.90) + `--samples` (default 1000) +
`--from-spec` + `--profile` + `--json-output`.

Output: Rich Table with P10/P50/P90 columns + P(OK) + traffic light status.

### Enhance `show-front`

Add `--cluster` flag and `--profile` flag to existing `show-front`:
- `--cluster`: adds a "Family" column showing cluster labels
- `--profile`: adds a "Score" column showing TOPSIS closeness coefficient

---

## Phase 4: Extend `swap_report.py`

Add optional `fom_score` field to `SWaPCScorecard` (backward-compatible):

```python
class SWaPCScorecard(BaseModel):
    # ... existing fields unchanged ...
    fom_score: float | None = Field(default=None)
    fom_per_objective: dict[str, float] = Field(default_factory=dict)
```

---

## Phase 5: Tests

### `tests/test_swap_profiles.py` (~8 tests)

- `test_preset_weights_sum_to_one` — all 4 presets
- `test_get_profile_known` — returns MissionProfile
- `test_get_profile_unknown` — raises KeyError
- `test_ahp_identity_matrix_equal_weights` — 6×6 identity → 1/6 each
- `test_ahp_known_ranking` — verify weight ordering matches dominance
- `test_ahp_consistency_check` — inconsistent matrix raises or warns
- `test_list_profiles` — returns 4 names
- `test_profile_all_objectives_present` — weights have all 6 keys

### `tests/test_swap_analysis.py` (~30 tests)

**M1 FoM:** 4 tests (perfect score, worst score, weighted ranking, design list ranking)
**M2 Sensitivity:** 5 tests (tornado bars per var, sorted by swing, taguchi 18 evals,
taguchi top factors, response surface)
**M3 Pareto:** 6 tests (topsis 2D, topsis with weights, ideal/anti-ideal correct,
MRS at knee, clustering k=4, cluster labels)
**M4 Delta:** 4 tests (same BOM zero delta, different enclosure, sweep steps, sweep monotonic)
**M5 MC:** 5 tests (zero uncertainty matches point, P90 > P50, feasibility range,
traffic light green, traffic light red, overall < individual)

Most tests use a mock evaluator (simple lambda returning deterministic results) or
call the real `estimate_system_bom` + `compute_thermal_feasibility` for integration tests.

### `tests/test_swap_cli.py` additions (~15 tests)

One `test_<command>_basic` and `test_<command>_json` per new command, plus edge cases.

---

## Phase 6: Documentation

Update existing `docs-site/src/content/docs/features/swap-analysis.md` to add a section
on the five methodologies, or create a dedicated tutorial page.

Update `docs-site/src/content/docs/reference/cli.md` with the 5 new commands.

---

## Implementation Order

```
1. swap_profiles.py + test_swap_profiles.py        (foundation, no deps)
2. swap_analysis.py M1 (FoM) + M4 (delta/sweep)   (independent of each other)
3. swap_analysis.py M2 (sensitivity)               (independent)
4. swap_analysis.py M3 (TOPSIS/MRS/cluster)        (needs M1 for weights)
5. swap_analysis.py M5 (Monte Carlo)               (needs evaluator pattern)
6. test_swap_analysis.py                           (all analysis tests)
7. CLI commands in swap.py                         (all 5 + show-front enhancement)
8. test_swap_cli.py additions                      (CLI tests)
9. swap_report.py fom_score extension              (backward-compatible)
10. docs + CLI reference updates                   (last)
```

---

## Key Design Decisions

1. **Single analysis module** (`swap_analysis.py`) — all 5 methodology algorithms in one
   file. Functions share data types (objectives dicts, BOM summaries). ~600 lines.

2. **Evaluator function pattern** — all analysis functions accept
   `evaluator_fn: Callable[[dict], dict]` not the concrete evaluator class. Enables
   mock testing and decouples from the MOO infrastructure.

3. **k-means without sklearn** — simple ~30-line numpy implementation. Pareto fronts have
   10–100 points; full sklearn is overkill.

4. **Taguchi L18 as a constant** — well-known orthogonal array stored as a nested list.
   No computation needed.

5. **No new dependencies** — everything uses numpy (core dep), pydantic, rich, click.

6. **MC perturbs continuous params only** — categorical variables (package, cooling) are
   design choices, not uncertain quantities.

---

## Verification

```bash
# Lint + format
.venv/bin/black --check src/ tests/ --line-length 100
.venv/bin/ruff check src/ tests/

# Unit tests
.venv/bin/pytest tests/test_swap_profiles.py tests/test_swap_analysis.py -v
.venv/bin/pytest tests/test_swap_cli.py -v

# Full test suite (must remain green)
.venv/bin/pytest tests/ -q

# Smoke tests
.venv/bin/branes swap score --area 50 --power 5 --process 28 --profile drone
.venv/bin/branes swap sensitivity --area 50 --power 5 --process 28 --mode tornado
.venv/bin/branes swap sensitivity --area 50 --power 5 --process 28 --mode taguchi
.venv/bin/branes swap sweep --area 50 --power 5 --process 28 --param process_nm --from 28 --to 5
.venv/bin/branes swap budget --area 50 --power 5 --process 28 --max-weight 200 --samples 500
.venv/bin/branes swap explore -g "test" --power 10 --weight 500 --fast
.venv/bin/branes swap rank --profile drone --top 5
.venv/bin/branes swap show-front --cluster --profile drone
```
