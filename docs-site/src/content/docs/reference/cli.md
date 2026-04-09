---
title: CLI Reference
description: Complete reference for the branes command-line interface.
---

## Global Options

```bash
branes [OPTIONS] COMMAND [ARGS]...
```

| Option | Description |
|--------|-------------|
| `--version` | Show version and exit |
| `--help` | Show help and exit |

## Commands

### analyze

Analyze a model's structure and characteristics.

```bash
branes analyze MODEL [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--hardware TEXT` | Target hardware for analysis |
| `--input-shape TEXT` | Input tensor shape (e.g., 1,3,640,640) |
| `--batch-size INT` | Batch size for analysis |
| `--precision TEXT` | Precision (fp32, fp16, int8) |

### benchmark

Benchmark model performance on target backends.

```bash
branes benchmark MODEL [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--backend TEXT` | Backend (local, ssh, kubernetes) |
| `--iterations INT` | Number of iterations |
| `--warmup INT` | Warmup iterations |

### chat

Start interactive AI architect session.

```bash
branes chat [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--model TEXT` | Claude model to use |
| `--verbose` | Show tool calls |

Requires `ANTHROPIC_API_KEY` environment variable.

### workflow

Run the full analysis workflow.

```bash
branes workflow run MODEL [OPTIONS]
```

### deploy

Deploy a model to target hardware.

```bash
branes deploy MODEL [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--target TEXT` | Target (jetson, openvino, coral) |
| `--precision TEXT` | Precision (fp32, fp16, int8) |
| `--input-shape TEXT` | Input shape |
| `--calibration-data PATH` | Calibration images for INT8 |
| `--output-dir PATH` | Output directory |

### codebase

Analyze full application codebases for hardware assessment.

```bash
branes codebase [scan|analyze|assess] PROJECT_PATH [OPTIONS]
```

#### codebase scan

Quick static scan — no LLM or API key needed.

```bash
branes codebase scan /path/to/project
```

#### codebase analyze

Full LLM-powered multi-pass analysis. Requires `ANTHROPIC_API_KEY`.

```bash
branes codebase analyze /path/to/project
```

#### codebase assess

End-to-end hardware assessment: scan, analyze, convert, and assess.

```bash
branes codebase assess /path/to/project [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--hardware TEXT` | Comma-separated hardware targets (e.g., jetson_orin,custom_kpu) |
| `--power-budget FLOAT` | Maximum power budget in watts |
| `--latency-target FLOAT` | Target end-to-end latency in milliseconds |

### optimize

Multi-objective design space optimization.

```bash
branes optimize [explore|show-front|sensitivity|explain] [OPTIONS]
```

#### optimize explore

Explore the design space with multi-objective optimization.

```bash
branes optimize explore --goal "drone SoC" --power 5 --latency 33
```

| Option | Description |
|--------|-------------|
| `--goal, -g TEXT` | Design goal description (required) |
| `--power, -p FLOAT` | Power budget in watts |
| `--latency, -l FLOAT` | Latency target in ms |
| `--cost, -c FLOAT` | Cost budget in USD |
| `--area, -a FLOAT` | Area budget in mm² |
| `--fast` | Fast mode (reduced evaluations, MAP-Elites only) |
| `--layers TEXT` | Layer selection: auto, map_elites, bayesian, nsga3 (default: auto) |
| `--workers INT` | Thread pool size (default: 8) |
| `--json-output` | Output raw JSON |

#### optimize show-front

Show the Pareto front from the last exploration.

```bash
branes optimize show-front --top 10
```

| Option | Description |
|--------|-------------|
| `--top INT` | Number of designs to show (default: 10) |

#### optimize sensitivity

Show parameter sensitivity from the last exploration. Requires the Bayesian optimization layer to have run.

```bash
branes optimize sensitivity
```

#### optimize explain

Explain the tradeoff between two Pareto-front designs.

```bash
branes optimize explain --points 0,3
```

| Option | Description |
|--------|-------------|
| `--points, -p TEXT` | Two point indices, comma-separated (required) |

### swap

System-level SWaP-C (Size, Weight, Power, Cost) analysis.

```bash
branes swap [estimate|bom|check|explore|show-front|score|rank|sensitivity|sweep|budget|compare|explain] [OPTIONS]
```

#### swap estimate

Quick single-point SWaP-C estimation.

```bash
branes swap estimate --area 50 --power 5 --process 28
```

| Option | Description |
|--------|-------------|
| `--area, -a FLOAT` | Die area in mm² (required) |
| `--power, -p FLOAT` | SoC TDP in watts (required) |
| `--process INT` | Process node in nm (default: 28) |
| `--package TEXT` | Package type: QFN, BGA, FCBGA, WLCSP (default: BGA) |
| `--cooling TEXT` | Cooling type: passive, active_fan, liquid (default: passive) |
| `--enclosure TEXT` | Enclosure material: aluminum, abs_plastic, magnesium (default: aluminum) |
| `--volume INT` | Production volume (default: 10000) |
| `--layers INT` | PCB layer count (default: 4) |
| `--connectors INT` | Board connectors (default: 0) |
| `--ambient-temp FLOAT` | Ambient temperature in °C (default: 40.0) |
| `--json-output` | Output JSON |

#### swap bom

Detailed BOM breakdown with per-component weight, volume, and cost.

```bash
branes swap bom --area 50 --power 5 --process 28 --package FCBGA
```

Options: same as `swap estimate`.

#### swap check

Scorecard against SWaP-C budgets. Returns exit code 1 on FAIL.

```bash
branes swap check --area 50 --power 5 --process 28 \
    --max-weight 500 --max-volume 200 --max-cost 1000
```

Options: all `swap estimate` options plus:

| Option | Description |
|--------|-------------|
| `--max-weight FLOAT` | Weight budget in grams |
| `--max-volume FLOAT` | Volume budget in cm³ |
| `--max-power FLOAT` | Power budget in watts |
| `--max-cost FLOAT` | Cost budget in USD |
| `--max-latency FLOAT` | Latency budget in ms |
| `--from-spec TEXT` | Read budgets from a named spec |
| `--json-output` | Output JSON |

#### swap explore

6-objective design space exploration with SWaP-C (power, latency, area, cost, weight, volume).

```bash
branes swap explore --goal "drone SoC" --power 10 --weight 500 --fast
```

| Option | Description |
|--------|-------------|
| `--goal, -g TEXT` | Design goal description (required) |
| `--power, -p FLOAT` | Max power in watts |
| `--latency, -l FLOAT` | Max latency in ms |
| `--cost, -c FLOAT` | Max cost in USD |
| `--area, -a FLOAT` | Max area in mm² |
| `--weight, -w FLOAT` | Max weight in grams |
| `--volume FLOAT` | Max volume in cm³ |
| `--fast` | Reduced evaluations (MAP-Elites only) |
| `--layers TEXT` | Optimizer layer: auto, map_elites, bayesian, nsga3 |
| `--workers INT` | Thread pool size (default: 8) |
| `--from-spec TEXT` | Read constraints from a spec |
| `--json-output` | Output raw JSON |

#### swap show-front

Display Pareto front from the last `swap explore` run.

```bash
branes swap show-front --top 5
branes swap show-front --cluster --profile drone --top 10
```

| Option | Description |
|--------|-------------|
| `--top INT` | Number of designs to show (default: 10) |
| `--cluster` | Add a "Family" column with k-means cluster labels |
| `--profile TEXT` | Add a "Score" column with TOPSIS closeness for a mission profile |

#### swap score

Compute a weighted Figure of Merit (0–100) for a single design point.

```bash
branes swap score --area 50 --power 5 --process 28 --profile drone
```

Options: all `swap estimate` options plus:

| Option | Description |
|--------|-------------|
| `--profile TEXT` | Mission profile: drone, rack, wearable, vehicle (default: drone) |
| `--json-output` | Output JSON |

#### swap rank

Rank Pareto-front designs from the last `swap explore` by mission profile.

```bash
branes swap rank --profile drone --method topsis --top 5
```

| Option | Description |
|--------|-------------|
| `--profile TEXT` | Mission profile (default: drone) |
| `--method TEXT` | Ranking method: topsis, fom (default: topsis) |
| `--top INT` | Number of designs to show (default: 10) |
| `--json-output` | Output JSON |

#### swap sensitivity

Tornado diagrams or Taguchi L18 screening for parameter sensitivity.

```bash
branes swap sensitivity --area 50 --power 5 --process 28 --mode tornado
branes swap sensitivity --area 50 --power 5 --process 28 --mode taguchi
```

Options: all `swap estimate` options plus:

| Option | Description |
|--------|-------------|
| `--mode TEXT` | Analysis mode: tornado, taguchi (default: tornado) |
| `--objective TEXT` | Focus on a single objective (default: all) |
| `--json-output` | Output JSON |

#### swap sweep

Parametric sweep of one design variable across all 6 objectives.

```bash
branes swap sweep --area 50 --power 5 --process 28 \
    --param process_nm --from 28 --to 5 --steps 5
```

Options: all `swap estimate` options plus:

| Option | Description |
|--------|-------------|
| `--param TEXT` | Variable to sweep (required) |
| `--from FLOAT` | Start value |
| `--to FLOAT` | End value |
| `--steps INT` | Number of steps (default: 10) |
| `--json-output` | Output JSON |

For categorical parameters (`package_type`, `cooling_type`), `--from` and `--to` are ignored; all choices are swept automatically.

#### swap budget

Monte Carlo probabilistic budget feasibility analysis.

```bash
branes swap budget --area 50 --power 5 --process 28 \
    --max-weight 200 --max-cost 1000 --samples 1000
```

Options: all `swap estimate` options plus:

| Option | Description |
|--------|-------------|
| `--max-weight FLOAT` | Weight budget in grams |
| `--max-volume FLOAT` | Volume budget in cm³ |
| `--max-power FLOAT` | Power budget in watts |
| `--max-cost FLOAT` | Cost budget in USD |
| `--samples INT` | Number of Monte Carlo samples (default: 1000) |
| `--from-spec TEXT` | Read budgets from a named spec |
| `--profile TEXT` | Load budgets from a mission profile |
| `--json-output` | Output JSON |

Traffic-light output: green (≥90% feasible), yellow (50–90%), red (<50%).

#### swap compare

Side-by-side comparison of two packaging/cooling configurations.

```bash
branes swap compare --area 50 --power 5 --process 28 \
    --left "QFN,passive,abs_plastic" --right "FCBGA,active_fan,aluminum"
```

| Option | Description |
|--------|-------------|
| `--area, -a FLOAT` | Die area in mm² (required) |
| `--power, -p FLOAT` | SoC TDP in watts (required) |
| `--process INT` | Process node in nm (default: 28) |
| `--left TEXT` | Left config: package,cooling,enclosure (required) |
| `--right TEXT` | Right config: package,cooling,enclosure (required) |
| `--json-output` | Output JSON |

#### swap explain

Explain the tradeoff between two designs from the last `swap explore` run.

```bash
branes swap explain --points 0,3
```

| Option | Description |
|--------|-------------|
| `--points, -p TEXT` | Two point indices, comma-separated (required) |

### report

View and manage analysis reports.

```bash
branes report [list|view|export] [OPTIONS]
```

### backends

Manage benchmark backends.

```bash
branes backends [list|add|remove] [OPTIONS]
```

### secrets

Manage credentials for remote backends.

```bash
branes secrets [set|get|delete] [OPTIONS]
```
