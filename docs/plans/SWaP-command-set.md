# Plan: `branes swap` CLI Subcommand Set

## Context

The SWaP-C data models, physical estimators, evaluator, and report generation are now
implemented (Phases 1-5 of the prior plan). However, there is no CLI surface for users to
interact with SWaP-C analysis. Users need to:
- Quickly estimate system weight/volume/thermal for a design point
- Set SWaP-C budgets and check designs against them
- Run 6-objective optimization with weight/volume
- Compare packaging and cooling alternatives side-by-side
- View BOM breakdowns and thermal assessments
- Bridge from existing specs to SWaP-C analysis

This plan adds `branes swap` — a Click command group following the same patterns as
`branes spec` and `branes optimize`.

---

## Command Design

### Group: `branes swap`

```
branes swap estimate   — Quick single-point SWaP-C estimation (Questions)
branes swap bom        — Detailed BOM breakdown (Questions)
branes swap check      — Scorecard against budgets (Assertions + Tests)
branes swap explore    — 6-objective MOO optimization (Tests)
branes swap show-front — Display Pareto front from last explore (Tests)
branes swap compare    — Side-by-side comparison of two configurations (Comparisons)
branes swap explain    — Tradeoff analysis between two Pareto-front designs (Comparisons)
```

Seven commands covering the four workflow stages: Questions → Assertions → Tests → Comparisons.

---

### 1. `branes swap estimate` — Quick single-point estimation

**Purpose:** Answer "what does this design weigh?" without running optimization.

```
branes swap estimate --area 50 --power 5 --process 28 \
    --package BGA --cooling passive --enclosure aluminum
```

**Options:**
| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--area` / `-a` | float | required | Die area (mm²) |
| `--power` / `-p` | float | required | SoC TDP (watts) |
| `--process` | int | 28 | Process node (nm) |
| `--package` | choice | BGA | Package type (QFN/BGA/FCBGA/WLCSP) |
| `--cooling` | choice | passive | Cooling type (passive/active_fan/liquid) |
| `--enclosure` | choice | aluminum | Enclosure material (aluminum/abs_plastic/magnesium) |
| `--volume` | int | 10000 | Production volume |
| `--layers` | int | 4 | PCB layer count |
| `--connectors` | int | 0 | Board connectors |
| `--ambient-temp` | float | 40.0 | Ambient temperature (°C) |
| `--json-output` | flag | false | JSON output |

**Calls:**
1. `estimate_system_bom(area, power, process, package, cooling, enclosure, volume, layers, connectors)`
2. `compute_thermal_feasibility(bom, power, ambient_temp)`

**Output (human):** Rich Panel with summary metrics:
```
╭─ SWaP-C Estimate: 50mm² / 28nm / BGA / passive ─╮
│  Weight:    142.3 g                                │
│  Volume:     68.5 cm³                              │
│  Cost:      $23.40                                 │
│  Dims:      54×54×32 mm                            │
│  Thermal:   Tj=105°C (margin: 20°C) ✓             │
╰────────────────────────────────────────────────────╯
```

---

### 2. `branes swap bom` — Detailed BOM breakdown

**Purpose:** Show the full hierarchical BOM tree with per-component weight/volume/cost.

```
branes swap bom --area 50 --power 5 --process 28 --package BGA --cooling passive
```

**Options:** Same as `estimate` (shared option decorator).

**Calls:**
1. `estimate_system_bom(...)` → SystemBOM
2. `bom.summary_table()` → flat rows

**Output (human):** Rich Table with per-level breakdown:
```
                    System BOM Breakdown
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━┓
┃ Component          ┃ Level   ┃ Weight(g)┃ Vol(cm³) ┃ Cost($) ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━┩
│ system             │ system  │          │          │         │
│ ├─ pcb             │ pcb     │    12.40 │     4.20 │    3.80 │
│ │  └─ BGA-256      │ package │     0.25 │     0.17 │    1.50 │
│ │     └─ die       │ die     │     0.18 │     0.04 │   12.30 │
│ ├─ heatsink-passive│ system  │    40.00 │    20.00 │    0.00 │
│ └─ enclosure-alum  │ system  │    89.50 │    44.10 │    4.48 │
├────────────────────┼─────────┼──────────┼──────────┼─────────┤
│ TOTAL              │         │   142.33 │    68.51 │   22.08 │
└────────────────────┴─────────┴──────────┴──────────┴─────────┘
```

---

### 3. `branes swap check` — Scorecard against budgets

**Purpose:** Assert SWaP-C budgets and get PASS/FAIL/WARNING verdicts. Bridges to the
spec system: can read budgets from an existing spec.

```
# Explicit budgets
branes swap check --area 50 --power 5 --process 28 --package BGA --cooling passive \
    --max-weight 500 --max-volume 200 --max-power 10 --max-cost 50

# From a spec (reads constraints from spec store)
branes swap check --area 50 --power 5 --process 28 --from-spec my-drone
```

**Options:** All `estimate` options plus:
| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--max-weight` | float | None | Weight budget (grams) |
| `--max-volume` | float | None | Volume budget (cm³) |
| `--max-power` | float | None | Power budget (watts) |
| `--max-cost` | float | None | Cost budget (USD) |
| `--max-latency` | float | None | Latency budget (ms) |
| `--from-spec` | str | None | Read budgets from a named spec |

**Calls:**
1. If `--from-spec`: `SpecStore.get(name)` → extract budgets from `spec.power`, `spec.compute`, `spec.constraints`
2. `estimate_system_bom(...)`
3. `compute_thermal_feasibility(...)`
4. `assess_design_point(design, bom_summary, constraints, thermal_data)`

**Output (human):** Rich Table scorecard:
```
              SWaP-C Scorecard
┏━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━┳━━━━━━━━━┓
┃ Metric         ┃ Value    ┃ Budget   ┃ Util ┃ Verdict ┃
┡━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━╇━━━━━━━━━┩
│ weight_grams   │   142.33 │   500.00 │  28% │ PASS    │
│ volume_cm3     │    68.51 │   200.00 │  34% │ PASS    │
│ power_watts    │     5.00 │    10.00 │  50% │ PASS    │
│ cost_usd       │    22.08 │    50.00 │  44% │ PASS    │
└────────────────┴──────────┴──────────┴──────┴─────────┘
 Thermal: Tj=105°C (max 125°C), passive cooling — PASS
 Overall: PASS
```

**Exit code:** 0 if PASS, 1 if FAIL (enables CI/scripting).

---

### 4. `branes swap explore` — 6-objective MOO optimization

**Purpose:** Run the full 6-objective design space exploration.

```
branes swap explore --goal "drone perception SoC" \
    --power 10 --latency 33 --cost 50 --weight 500 --volume 200 \
    --fast
```

**Options:**
| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--goal` / `-g` | str | required | Design goal |
| `--power` / `-p` | float | None | Max power (W) |
| `--latency` / `-l` | float | None | Max latency (ms) |
| `--cost` / `-c` | float | None | Max cost (USD) |
| `--area` / `-a` | float | None | Max area (mm²) |
| `--weight` / `-w` | float | None | Max weight (g) |
| `--volume` | float | None | Max volume (cm³) |
| `--fast` | flag | false | Reduced evaluations |
| `--layers` | str | auto | Optimizer layer selection |
| `--workers` | int | 8 | Thread pool size |
| `--from-spec` | str | None | Read constraints from spec |
| `--json-output` | flag | false | JSON output |

**Calls:**
1. `create_swap_design_space(constraints)`
2. `SWaPCEvaluator(design_space, base_state, constraint_bounds, thermal_config)`
3. `OptimizationEngine(ds, evaluator, config).run()`

**Output:** Pareto front table with 6 objective columns + design params. Saves result
via `_save_swap_result()` for `show-front`, `compare`, `explain`.

---

### 5. `branes swap show-front` — Display Pareto front

**Purpose:** Show the Pareto front from the last `explore` run.

```
branes swap show-front [--top 10]
```

**Options:**
| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--top` | int | 10 | Number of designs to show |

**Calls:** `_load_swap_result()` → display table.

**Output:** Rich Table with process, package, cooling, power, latency, area, cost,
weight, volume columns. Knee point marked with `*`.

---

### 6. `branes swap compare` — Side-by-side configuration comparison

**Purpose:** Compare two packaging/cooling configurations on the same SoC.

```
branes swap compare --area 50 --power 5 --process 28 \
    --left "QFN,passive,abs_plastic" \
    --right "FCBGA,active_fan,aluminum"
```

**Options:** Core SoC params (`--area`, `--power`, `--process`) plus:
| Flag | Type | Description |
|------|------|-------------|
| `--left` | str | Comma-separated: package,cooling,enclosure |
| `--right` | str | Comma-separated: package,cooling,enclosure |

**Calls:**
1. `estimate_system_bom(...)` for left config
2. `estimate_system_bom(...)` for right config
3. `compute_thermal_feasibility(...)` for each

**Output:** Side-by-side Rich Table:
```
         Configuration Comparison (50mm² / 28nm / 5W)
┏━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Metric         ┃ QFN/passive/plastic  ┃ FCBGA/active_fan/alum  ┃
┡━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Weight (g)     │               62.10  │                148.30  │
│ Volume (cm³)   │               38.20  │                 72.50  │
│ Cost ($)       │               14.80  │                 28.60  │
│ Tj (°C)        │            118 (7°C) │             58 (67°C)  │
│ Thermal        │              WARNING │                  PASS  │
│ Dims (mm)      │          42×42×24    │            62×62×36    │
└────────────────┴──────────────────────┴────────────────────────┘
 Delta: right is +138% weight, +90% volume, +93% cost, but 60°C cooler
```

---

### 7. `branes swap explain` — Tradeoff between Pareto-front designs

**Purpose:** Explain the tradeoff between two designs from the last `explore` run.
Same pattern as `branes optimize explain`.

```
branes swap explain --points 0,3
```

**Options:**
| Flag | Type | Description |
|------|------|-------------|
| `--points` / `-p` | str | Two comma-separated point indices |

**Calls:** `_load_swap_result()` → compare two points across all 6 objectives.

**Output:** Objective deltas table (with all 6 objectives including weight/volume)
+ parameter change table (including package_type, cooling_type).

---

## Workflow Composition

```
Questions:     branes swap estimate ...   →  Quick numbers
               branes swap bom ...        →  Where does weight come from?

Assertions:    branes swap check ...      →  Does it pass budgets?
               branes swap check --from-spec my-drone ...  →  Against spec budgets

Tests:         branes swap explore ...    →  Find the Pareto front
               branes swap show-front     →  View results
               branes swap check ...      →  Validate specific point

Comparisons:   branes swap compare ...    →  QFN vs FCBGA side-by-side
               branes swap explain ...    →  Why is design #0 better than #3?
```

Typical user session:
```bash
# 1. Quick estimate
branes swap estimate --area 50 --power 5 --process 28

# 2. Try different packages
branes swap compare --area 50 --power 5 --process 28 \
    --left "QFN,passive,abs_plastic" --right "BGA,passive,aluminum"

# 3. Check against drone spec budgets
branes swap check --area 50 --power 5 --process 28 --package BGA \
    --from-spec my-drone

# 4. Full optimization
branes swap explore -g "drone SoC" --power 10 --weight 500 --volume 200 --fast

# 5. Inspect results
branes swap show-front --top 5
branes swap explain --points 0,2
```

---

## Implementation

### New file: `src/embodied_ai_architect/cli/commands/swap.py`

Single file containing the `swap` Click group and all 7 subcommands.

**Internal helpers:**
- `_common_soc_options(f)` — Shared Click options decorator for `--area`, `--power`,
  `--process`, `--package`, `--cooling`, `--enclosure`, `--volume`, `--layers`, `--connectors`,
  `--ambient-temp`
- `_save_swap_result(result)` / `_load_swap_result()` — Save/load to
  `$TMPDIR/branes_swap_result.json` (same pattern as optimize.py)
- `_parse_config(config_str)` — Parse "package,cooling,enclosure" for `compare`
- `_read_spec_budgets(spec_name)` — Extract SWaP-C budgets from a spec via SpecStore

**Functions called from graphs layer:**
- `estimate_system_bom()` — estimate, bom, check, compare
- `compute_thermal_feasibility()` — estimate, check, compare
- `assess_design_point()` — check
- `format_swap_report_text()` — check (with --json-output off)
- `create_swap_design_space()` — explore
- `SWaPCEvaluator` — explore
- `OptimizationEngine` — explore

### Modified file: `src/embodied_ai_architect/cli/__init__.py`

Add `from .commands.swap import swap; cli.add_command(swap)` in `main()`.

### New file: `tests/test_swap_cli.py`

Test each subcommand via Click's `CliRunner`:
- `test_estimate_basic` — returns 0, output contains "Weight"
- `test_estimate_json` — valid JSON with weight_grams key
- `test_bom_table` — output contains "die", "package", "pcb"
- `test_check_pass` — exit code 0 when within budgets
- `test_check_fail` — exit code 1 when over budget
- `test_check_from_spec` — reads budgets from spec store (mock)
- `test_compare_two_configs` — output contains both config names
- `test_explore_fast` — runs without error (fast mode)
- `test_show_front_no_result` — error message when no prior explore
- `test_explain_basic` — shows delta table

---

## File Summary

| Action | File |
|--------|------|
| **Create** | `src/embodied_ai_architect/cli/commands/swap.py` |
| **Modify** | `src/embodied_ai_architect/cli/__init__.py` — register `swap` group |
| **Create** | `tests/test_swap_cli.py` |

## Verification

```bash
# Lint + format
.venv/bin/black --check src/ tests/ --line-length 100
.venv/bin/ruff check src/ tests/

# Tests
.venv/bin/pytest tests/test_swap_cli.py -v

# Smoke test
.venv/bin/branes swap estimate --area 50 --power 5 --process 28
.venv/bin/branes swap bom --area 50 --power 5 --process 28 --package FCBGA
.venv/bin/branes swap check --area 50 --power 5 --process 28 --max-weight 500
.venv/bin/branes swap compare --area 50 --power 5 --process 28 \
    --left "QFN,passive,abs_plastic" --right "FCBGA,active_fan,aluminum"
.venv/bin/branes swap explore -g "drone" --power 10 --weight 500 --fast
.venv/bin/branes swap show-front
.venv/bin/branes swap explain --points 0,1
```
