---
title: CLI Reference
description: Complete reference for the branes command-line interface.
---

The `branes` CLI has 30 command groups organized into five sections:
**Lifecycle**, **Catalog**, **Analysis & Benchmarking**, **Infrastructure**, and **Deployment**.

All commands support the `--json` global flag for machine-readable output.

## Global Options

```bash
branes [OPTIONS] COMMAND [ARGS]...
```

| Option | Description |
|--------|-------------|
| `--version` | Show version and exit |
| `--json` | Enable JSON output globally for all subcommands |
| `--quiet` | Suppress non-essential output |
| `--verbose` | Increase verbosity |
| `--help` | Show help and exit |

---

## Lifecycle Commands

Commands that follow the mission-driven design flow: define a mission, qualify the goal,
plan a task graph, select components, synthesize a system, analyze subsystems, optimize,
validate, and generate reports.

### mission

Manage design missions -- the persistent backbone that owns the full design state.

```bash
branes mission [new|list|show|edit|delete|refine|fork] [OPTIONS]
```

#### mission new

Create a new design mission.

```bash
branes mission new "Drone Perception SoC" --goal "30fps detection at <5W"
branes mission new "Warehouse AMR" --platform ground_wheeled.warehouse_amr
```

| Option | Description |
|--------|-------------|
| `--goal TEXT` | Design objective |
| `--platform TEXT` | Platform registry ID |
| `--use-case TEXT` | Use case label |

#### mission list

List all design missions.

```bash
branes mission list
```

#### mission show

Inspect a mission in detail.

```bash
branes mission show <mission_id>
branes mission show <mission_id> --json
```

| Option | Description |
|--------|-------------|
| `--json` | Output as JSON |

#### mission edit

Edit mission fields.

```bash
branes mission edit <mission_id> --name "Renamed" --status qualified
branes mission edit <mission_id> --goal "Updated goal"
```

| Option | Description |
|--------|-------------|
| `--name TEXT` | New name |
| `--goal TEXT` | New goal |
| `--status TEXT` | New status (draft/qualified/designed/optimized/validated) |
| `--platform TEXT` | Platform ID |
| `--use-case TEXT` | Use case label |

#### mission delete

Delete a mission.

```bash
branes mission delete <mission_id>
branes mission delete <mission_id> --yes
```

| Option | Description |
|--------|-------------|
| `--yes, -y` | Skip confirmation |

#### mission refine

Re-enter qualification with the mission's current state.

```bash
branes mission refine <mission_id>
```

#### mission fork

Fork a mission for what-if analysis. Creates a copy with a new ID and name.

```bash
branes mission fork <source_id> "What-If: 7nm Process"
```

### design

Design perception pipelines from requirements, qualify goals, and plan task graphs.

```bash
branes design [new|from-usecase|show|synthesize|qualify|plan] [OPTIONS]
```

#### design qualify

Qualify a design goal through structured Q&A. Checks whether a goal is specific
enough to produce meaningful design results and walks through domain-specific
questions to refine it.

```bash
branes design qualify "drone perception SoC"
branes design qualify --mission vineyard-sprayer
branes design qualify "cobot for assembly" --domain robot_arm
branes design qualify "warehouse AMR" --auto
```

| Option | Description |
|--------|-------------|
| `--mission TEXT` | Read/write a mission entity |
| `--domain, -d TEXT` | Force a domain (drone, ugv, robot_arm) |
| `--auto` | Auto-answer with defaults (non-interactive) |

#### design plan

Show the task plan the LLM generates for a design goal. Decomposes a natural-language
goal into a task graph (DAG of specialist agents), then displays for review.

```bash
branes design plan "Drone perception SoC: YOLO at 30fps, <5W, <$30"
branes design plan --mission vineyard-sprayer
branes design plan "Drone SoC" --static   # no API key needed
```

| Option | Description |
|--------|-------------|
| `--mission TEXT` | Load goal/constraints from a mission |
| `--power FLOAT` | Max power budget in watts |
| `--latency FLOAT` | Max latency in ms |
| `--cost FLOAT` | Max BOM cost in USD |
| `--area FLOAT` | Max die area in mm2 |
| `--process INT` | Target process node in nm (e.g., 28, 16, 7) |
| `--use-case TEXT` | Application type (e.g., delivery_drone) |
| `--platform TEXT` | Platform type (e.g., drone, amr, quadruped) |
| `--static` | Use static demo plan instead of LLM |

#### design new

Create new pipeline requirements interactively via wizard.

```bash
branes design new
branes design new -o my-pipeline.yaml
branes design new --no-interactive
```

| Option | Description |
|--------|-------------|
| `-o, --output PATH` | Output YAML file path |
| `--no-interactive` | Skip interactive wizard, use defaults |

#### design from-usecase

Create requirements from an embodied-schemas use case.

```bash
branes design from-usecase drone_obstacle_avoidance
branes design from-usecase industrial_inspection -o reqs.yaml
```

| Option | Description |
|--------|-------------|
| `-o, --output PATH` | Output YAML file path |

#### design show

Display requirements from a YAML file.

```bash
branes design show requirements.yaml
```

#### design synthesize

Synthesize a pipeline from requirements: select models, download them, and
generate a pipeline configuration.

```bash
branes design synthesize requirements.yaml
branes design synthesize requirements.yaml -o my-pipeline.yaml
branes design synthesize requirements.yaml --dry-run
```

| Option | Description |
|--------|-------------|
| `-o, --output PATH` | Output pipeline YAML file (default: pipeline.yaml) |
| `--download/--no-download` | Download required models (default: yes) |
| `--dry-run` | Show what would be done without executing |
| `--validate/--no-validate` | Run inference benchmark on downloaded models |

### select

Select components for a mission. Delegates to the sensor/actuator registries.

```bash
branes select [sensor|actuator|compute|model] MISSION_ID [ARGS]
```

#### select sensor

```bash
branes select sensor <mission_id> <sensor_id> [<sensor_id> ...]
```

#### select actuator

```bash
branes select actuator <mission_id> <actuator_id> [<actuator_id> ...]
```

#### select compute

Select compute hardware for a mission (coming soon).

```bash
branes select compute <mission_id>
```

#### select model

Select ML models for a mission (coming soon).

```bash
branes select model <mission_id>
```

### synthesize

Synthesize a system design from mission components.

```bash
branes synthesize [system|architecture|bom|cabling|thermal] MISSION_ID
```

#### synthesize system

Compose a full system from the mission's selected components (sensors, actuators,
compute, models).

```bash
branes synthesize system <mission_id>
```

#### synthesize architecture

Generate a Mermaid-compatible architecture diagram for a mission's system.

```bash
branes synthesize architecture <mission_id>
```

#### synthesize bom

Generate bill of materials for a mission (coming soon).

```bash
branes synthesize bom <mission_id>
```

#### synthesize cabling

Generate cabling and interconnect plan (coming soon).

```bash
branes synthesize cabling <mission_id>
```

#### synthesize thermal

Generate thermal management plan (coming soon).

```bash
branes synthesize thermal <mission_id>
```

### analyze-system

Analyze mission subsystems. Distinct from `branes analyze` (model analysis).

```bash
branes analyze-system [power|latency|thermal|swap|safety|cost|bandwidth|scheduling] MISSION_ID
```

#### analyze-system power

Analyze power budget for a mission.

```bash
branes analyze-system power <mission_id>
```

#### analyze-system latency

Analyze end-to-end latency for a mission.

```bash
branes analyze-system latency <mission_id>
```

#### analyze-system thermal

Analyze thermal feasibility for a mission.

```bash
branes analyze-system thermal <mission_id>
```

#### analyze-system swap

Run SWaP-C analysis for a mission.

```bash
branes analyze-system swap <mission_id>
```

#### analyze-system safety

Analyze safety requirements for a mission.

```bash
branes analyze-system safety <mission_id>
```

#### analyze-system cost

Analyze cost breakdown for a mission.

```bash
branes analyze-system cost <mission_id>
```

#### analyze-system bandwidth

Analyze data bandwidth requirements for a mission.

```bash
branes analyze-system bandwidth <mission_id>
```

#### analyze-system scheduling

Analyze task scheduling feasibility for a mission.

```bash
branes analyze-system scheduling <mission_id>
```

### optimize

Multi-objective design space optimization.

```bash
branes optimize [explore|show-front|sensitivity|explain] [OPTIONS]
```

#### optimize explore

Explore the design space with multi-objective optimization.

```bash
branes optimize explore --goal "drone SoC" --power 5 --latency 33
branes optimize explore --mission vineyard-sprayer
```

| Option | Description |
|--------|-------------|
| `--goal, -g TEXT` | Design goal description (required unless --mission) |
| `--mission TEXT` | Load constraints from a mission |
| `--power, -p FLOAT` | Power budget in watts |
| `--latency, -l FLOAT` | Latency target in ms |
| `--cost, -c FLOAT` | Cost budget in USD |
| `--area, -a FLOAT` | Area budget in mm2 |
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

### validate

Run design validation checks against a mission.

```bash
branes validate [mission|constraints|completeness|safety|scheduling] MISSION_ID
```

#### validate mission

Run ALL validation checks on a mission. Returns a verdict-first pass/fail report.

```bash
branes validate mission <mission_id>
```

#### validate constraints

Check SWaP-C constraint budgets.

```bash
branes validate constraints <mission_id>
```

#### validate completeness

Check that all required subsystems are specified.

```bash
branes validate completeness <mission_id>
```

#### validate safety

Check safety integrity requirements.

```bash
branes validate safety <mission_id>
```

#### validate scheduling

Check multi-rate scheduling feasibility.

```bash
branes validate scheduling <mission_id>
```

### report

View and manage analysis reports.

```bash
branes report [list|view|export] [OPTIONS]
```

---

## Catalog Commands

Commands for browsing and searching the platform, sensor, actuator, model,
and model zoo registries.

### platform

Browse the platform registry -- predefined platform definitions with classification,
attributes, and domain context.

```bash
branes platform [list|show|search|categories] [OPTIONS]
```

#### platform list

List all platforms in the registry.

```bash
branes platform list
branes platform list --category aerial
branes platform list --json
```

| Option | Description |
|--------|-------------|
| `--category, -c TEXT` | Filter by category |
| `--json` | Output as JSON |

#### platform show

Show full details of a platform definition, including classification, attributes,
domain context, and keywords.

```bash
branes platform show aerial.delivery_drone
branes platform show ground_wheeled
```

| Option | Description |
|--------|-------------|
| `--json` | Output as JSON |

#### platform search

Search platforms by free-text query with relevance scoring.

```bash
branes platform search "outdoor drone delivery"
branes platform search "warehouse robot" --top 5
```

| Option | Description |
|--------|-------------|
| `--top, -n INT` | Number of results (default: 10) |
| `--json` | Output as JSON |

#### platform categories

List available platform categories with counts.

```bash
branes platform categories
```

### sensor

Browse the sensor registry, select sensors for missions, compare specs, and
compute budgets.

```bash
branes sensor [list|show|search|categories|select|compare|budget|fusion] [OPTIONS]
```

#### sensor list

List all sensors in the registry.

```bash
branes sensor list
branes sensor list --category visual
```

| Option | Description |
|--------|-------------|
| `--category TEXT` | Filter by category (visual, ranging, inertial, ...) |

#### sensor show

Show details for a specific sensor.

```bash
branes sensor show <sensor_id>
```

#### sensor search

Search sensors by keyword with relevance scoring.

```bash
branes sensor search "stereo camera 30fps"
```

#### sensor categories

List sensor categories.

```bash
branes sensor categories
```

#### sensor select

Add sensors to a mission's selected sensors list.

```bash
branes sensor select <mission_id> <sensor_id> [<sensor_id> ...]
```

#### sensor compare

Compare sensors side by side in a table.

```bash
branes sensor compare <sensor_id_1> <sensor_id_2> [<sensor_id_3> ...]
```

#### sensor budget

Show power, weight, and cost budget for mission sensors.

```bash
branes sensor budget <mission_id>
```

#### sensor fusion

Recommend sensor fusion strategy based on selected sensor modalities.

```bash
branes sensor fusion <mission_id>
```

### actuator

Browse the actuator registry, select actuators for missions, compare specs,
and compute budgets.

```bash
branes actuator [list|show|search|categories|select|compare|budget|control-rate] [OPTIONS]
```

#### actuator list

List all actuators in the registry.

```bash
branes actuator list
branes actuator list --category motor
```

| Option | Description |
|--------|-------------|
| `--category TEXT` | Filter by category (motor, gripper, locomotion, ...) |

#### actuator show

Show details for a specific actuator.

```bash
branes actuator show <actuator_id>
```

#### actuator search

Search actuators by keyword with relevance scoring.

```bash
branes actuator search "brushless motor 100W"
```

#### actuator categories

List actuator categories.

```bash
branes actuator categories
```

#### actuator select

Add actuators to a mission's selected actuators list.

```bash
branes actuator select <mission_id> <actuator_id> [<actuator_id> ...]
```

#### actuator compare

Compare actuators side by side in a table.

```bash
branes actuator compare <actuator_id_1> <actuator_id_2>
```

#### actuator budget

Show power, weight, and cost budget for mission actuators.

```bash
branes actuator budget <mission_id>
```

#### actuator control-rate

Show control rate requirements for an actuator (loop period and response time).

```bash
branes actuator control-rate <actuator_id>
```

### model

Manage the local model registry. Models are analyzed once and cached for
querying and reasoning.

```bash
branes model [register|list|show|analyze|update|remove] [OPTIONS]
```

#### model register

Register a model in the registry. Analyzes the model and stores metadata.

```bash
branes model register yolov8s.pt
branes model register model.pt --name "My Model" --tags perception detection
branes model register model.pt --input-shape 1,3,640,640
```

| Option | Description |
|--------|-------------|
| `--name, -n TEXT` | Model name (default: filename) |
| `--input-shape, -i TEXT` | Input shape for analysis (e.g., 1,3,640,640) |
| `--tags, -t TEXT` | Tags for filtering (repeatable) |
| `--description, -d TEXT` | Model description |
| `--overwrite` | Overwrite if model ID exists |

#### model list

List registered models with optional filters.

```bash
branes model list
branes model list --architecture cnn
branes model list --max-params 15000000
branes model list --tags perception --tags detection
```

| Option | Description |
|--------|-------------|
| `--architecture, -a TEXT` | Filter by architecture type (cnn, transformer, mlp) |
| `--family, -f TEXT` | Filter by architecture family (yolo, resnet, vit) |
| `--min-params INT` | Minimum parameters |
| `--max-params INT` | Maximum parameters |
| `--tags, -t TEXT` | Filter by tags (models must have ALL tags) |

#### model show

Show full details of a registered model.

```bash
branes model show yolov8-small
```

#### model analyze

Analyze a model without registering it.

```bash
branes model analyze custom_model.pt
branes model analyze model.onnx --input-shape 1,3,224,224
```

| Option | Description |
|--------|-------------|
| `--input-shape, -i TEXT` | Input shape for analysis |

#### model update

Update model metadata (name, description, tags).

```bash
branes model update yolov8-small --name "YOLOv8 Small"
branes model update mymodel --add-tag perception --add-tag detection
```

| Option | Description |
|--------|-------------|
| `--name, -n TEXT` | Update model name |
| `--description, -d TEXT` | Update description |
| `--add-tag, -t TEXT` | Add tag(s) (repeatable) |
| `--remove-tag, -r TEXT` | Remove tag(s) (repeatable) |

#### model remove

Remove a model from the registry (does not delete the model file).

```bash
branes model remove yolov8-small
branes model remove old-model --force
```

| Option | Description |
|--------|-------------|
| `--force, -f` | Skip confirmation |

### zoo

Manage models from the unified Model Zoo. Search, download, and cache
models from multiple providers (Ultralytics, TorchVision, HuggingFace).

```bash
branes zoo [search|download|list|info|clear] [OPTIONS]
```

#### zoo search

Search for models in the zoo.

```bash
branes zoo search --task detection
branes zoo search --task detection --max-params 5000000
branes zoo search -q yolo --benchmarked
```

| Option | Description |
|--------|-------------|
| `--task TEXT` | Filter by task (detection, classification, segmentation, pose) |
| `--max-params INT` | Maximum parameters |
| `--min-accuracy FLOAT` | Minimum accuracy (0.0-1.0) |
| `--provider TEXT` | Filter by provider (ultralytics, torchvision, huggingface) |
| `--benchmarked` | Only show models with benchmark data |
| `--query, -q TEXT` | Search query |

#### zoo download

Download a model from the zoo.

```bash
branes zoo download yolov8n
branes zoo download yolov8s --format onnx
branes zoo download yolov8n --force
```

| Option | Description |
|--------|-------------|
| `--format, -f TEXT` | Model format: onnx, pytorch, torchscript, tensorrt, openvino, coreml (default: onnx) |
| `--provider TEXT` | Provider name (auto-detected if not specified) |
| `--force` | Force re-download even if cached |

#### zoo list

List available or cached models.

```bash
branes zoo list
branes zoo list --cached
branes zoo list --provider ultralytics
branes zoo list -t detection -n yolo --max-params 5000000
```

| Option | Description |
|--------|-------------|
| `--cached` | List downloaded models only |
| `--provider TEXT` | Filter by provider |
| `--format, -f TEXT` | Filter by format |
| `--task, -t TEXT` | Filter by task |
| `--name, -n TEXT` | Filter by name (substring match) |
| `--min-params INT` | Minimum parameters |
| `--max-params INT` | Maximum parameters |

#### zoo info

Show detailed information about a model.

```bash
branes zoo info yolov8n
branes zoo info yolov8s-seg
```

#### zoo clear

Clear the model cache.

```bash
branes zoo clear
branes zoo clear --provider ultralytics
branes zoo clear -y
```

| Option | Description |
|--------|-------------|
| `--provider TEXT` | Clear only specific provider cache |
| `--yes, -y` | Skip confirmation |

---

## Analysis & Benchmarking Commands

Commands for analyzing models, estimating hardware performance, running
benchmarks, and performing SWaP-C analysis.

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
| `--area, -a FLOAT` | Die area in mm2 (required) |
| `--power, -p FLOAT` | SoC TDP in watts (required) |
| `--process INT` | Process node in nm (default: 28) |
| `--package TEXT` | Package type: QFN, BGA, FCBGA, WLCSP (default: BGA) |
| `--cooling TEXT` | Cooling type: passive, active_fan, liquid (default: passive) |
| `--enclosure TEXT` | Enclosure material: aluminum, abs_plastic, magnesium (default: aluminum) |
| `--volume INT` | Production volume (default: 10000) |
| `--layers INT` | PCB layer count (default: 4) |
| `--connectors INT` | Board connectors (default: 0) |
| `--ambient-temp FLOAT` | Ambient temperature in C (default: 40.0) |
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
branes swap check --mission vineyard-sprayer
```

Options: all `swap estimate` options plus:

| Option | Description |
|--------|-------------|
| `--mission TEXT` | Load constraints from a mission |
| `--max-weight FLOAT` | Weight budget in grams |
| `--max-volume FLOAT` | Volume budget in cm3 |
| `--max-power FLOAT` | Power budget in watts |
| `--max-cost FLOAT` | Cost budget in USD |
| `--max-latency FLOAT` | Latency budget in ms |
| `--from-spec TEXT` | Read budgets from a named spec |
| `--json-output` | Output JSON |

#### swap explore

6-objective design space exploration with SWaP-C (power, latency, area, cost, weight, volume).

```bash
branes swap explore --goal "drone SoC" --power 10 --weight 500 --fast
branes swap explore --mission vineyard-sprayer
```

| Option | Description |
|--------|-------------|
| `--goal, -g TEXT` | Design goal description (required unless --mission) |
| `--mission TEXT` | Load constraints from a mission |
| `--power, -p FLOAT` | Max power in watts |
| `--latency, -l FLOAT` | Max latency in ms |
| `--cost, -c FLOAT` | Max cost in USD |
| `--area, -a FLOAT` | Max area in mm2 |
| `--weight, -w FLOAT` | Max weight in grams |
| `--volume FLOAT` | Max volume in cm3 |
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

Compute a weighted Figure of Merit (0-100) for a single design point.

```bash
branes swap score --area 50 --power 5 --process 28 --profile drone
branes swap score --mission vineyard-sprayer --profile drone
```

Options: all `swap estimate` options plus:

| Option | Description |
|--------|-------------|
| `--mission TEXT` | Load constraints from a mission |
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
branes swap sensitivity --mission vineyard-sprayer --mode taguchi
```

Options: all `swap estimate` options plus:

| Option | Description |
|--------|-------------|
| `--mission TEXT` | Load constraints from a mission |
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
branes swap budget --mission vineyard-sprayer --samples 1000
```

Options: all `swap estimate` options plus:

| Option | Description |
|--------|-------------|
| `--mission TEXT` | Load constraints from a mission |
| `--max-weight FLOAT` | Weight budget in grams |
| `--max-volume FLOAT` | Volume budget in cm3 |
| `--max-power FLOAT` | Power budget in watts |
| `--max-cost FLOAT` | Cost budget in USD |
| `--samples INT` | Number of Monte Carlo samples (default: 1000) |
| `--from-spec TEXT` | Read budgets from a named spec |
| `--profile TEXT` | Load budgets from a mission profile |
| `--json-output` | Output JSON |

Traffic-light output: green (>=90% feasible), yellow (50-90%), red (<50%).

#### swap compare

Side-by-side comparison of two packaging/cooling configurations.

```bash
branes swap compare --area 50 --power 5 --process 28 \
    --left "QFN,passive,abs_plastic" --right "FCBGA,active_fan,aluminum"
```

| Option | Description |
|--------|-------------|
| `--area, -a FLOAT` | Die area in mm2 (required) |
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

### mcp

Client for the graphs MCP server -- roofline, energy, and memory estimators.

```bash
branes mcp [tools|hardware|specs|analyze|latency|energy|memory|compare|server] [OPTIONS]
```

#### mcp tools

List available MCP tools and their descriptions.

```bash
branes mcp tools
```

#### mcp hardware

List available hardware targets with optional filtering.

```bash
branes mcp hardware
branes mcp hardware --type gpu
branes mcp hardware --query orin
```

| Option | Description |
|--------|-------------|
| `--type TEXT` | Filter: cpu, gpu, dsp, tpu, kpu, accelerator |
| `--query, -q TEXT` | Fuzzy search (e.g., "orin", "jetson") |

#### mcp specs

Show detailed specifications for a hardware target.

```bash
branes mcp specs jetson_orin_nano
```

#### mcp analyze

Run full roofline + energy + memory analysis. Supports `--mission` to load
model and hardware from a mission entity.

```bash
branes mcp analyze resnet18 jetson_orin_nano
branes mcp analyze yolov8n h100_sxm5 --precision int8
branes mcp analyze --mission vineyard-sprayer
```

| Option | Description |
|--------|-------------|
| `--mission TEXT` | Load model/hardware from a mission |
| `--batch INT` | Batch size (default: 1) |
| `--precision, -p TEXT` | Precision: fp32, fp16, bf16, int8, int4 (default: fp16) |
| `--thermal TEXT` | Thermal profile (e.g., "15W") |

#### mcp latency

Predict inference latency using roofline model.

```bash
branes mcp latency resnet50 jetson_orin_nano --thermal 15W
branes mcp latency --mission vineyard-sprayer
```

| Option | Description |
|--------|-------------|
| `--mission TEXT` | Load model/hardware from a mission |
| `--batch INT` | Batch size (default: 1) |
| `--precision, -p TEXT` | Precision: fp32, fp16, bf16, int8, int4 (default: fp16) |
| `--thermal TEXT` | Thermal profile (e.g., "15W") |

#### mcp energy

Estimate energy consumption with component breakdown.

```bash
branes mcp energy resnet18 h100_sxm5 --power-gating
```

| Option | Description |
|--------|-------------|
| `--batch INT` | Batch size (default: 1) |
| `--precision, -p TEXT` | Precision: fp32, fp16, bf16, int8, int4 (default: fp16) |
| `--power-gating` | Enable power gating |
| `--thermal TEXT` | Thermal profile |

#### mcp memory

Analyse peak memory usage and activation timeline.

```bash
branes mcp memory yolov8n jetson_orin_nano
```

| Option | Description |
|--------|-------------|
| `--batch INT` | Batch size (default: 1) |
| `--precision, -p TEXT` | Precision: fp32, fp16, bf16, int8, int4 (default: fp16) |

#### mcp compare

Compare model performance across multiple hardware targets.

```bash
branes mcp compare resnet18 jetson_orin_nano h100_sxm5 coral_edge_tpu
```

| Option | Description |
|--------|-------------|
| `--batch INT` | Batch size (default: 1) |
| `--precision, -p TEXT` | Precision (default: fp16) |
| `--sort TEXT` | Sort by: latency, energy, memory (default: latency) |

#### mcp server

Start the graphs MCP server.

```bash
branes mcp server                          # stdio transport
branes mcp server --sse --port 8100        # HTTP/SSE transport
```

| Option | Description |
|--------|-------------|
| `--sse` | Use SSE/HTTP transport |
| `--host TEXT` | Bind address (default: 127.0.0.1) |
| `--port INT` | Port (default: 8100) |

### codebase

Analyze full application codebases for hardware assessment.

```bash
branes codebase [scan|analyze|assess] PROJECT_PATH [OPTIONS]
```

#### codebase scan

Quick static scan -- no LLM or API key needed.

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

### testbench

Model validation and drift monitoring.

```bash
branes testbench [validate|benchmark|drift|history|list] [OPTIONS]
```

#### testbench validate

Validate model accuracy against a dataset.

```bash
branes testbench validate yolov8n.onnx --dataset coco_val.json
branes testbench validate model.onnx --task classification --threshold 0.8
```

| Option | Description |
|--------|-------------|
| `--dataset PATH` | Path to validation dataset (JSON or directory) |
| `--task TEXT` | Model task type: detection, classification, segmentation (default: detection) |
| `--threshold FLOAT` | Accuracy threshold for pass/fail |
| `--record/--no-record` | Record result for drift monitoring (default: yes) |

#### testbench benchmark

Benchmark model inference latency.

```bash
branes testbench benchmark model.onnx
branes testbench benchmark model.onnx --iterations 1000
```

| Option | Description |
|--------|-------------|
| `--iterations INT` | Number of inference iterations (default: 100) |
| `--warmup INT` | Warmup iterations (default: 10) |
| `--input-shape TEXT` | Input tensor shape (default: 1,3,640,640) |

#### testbench drift

Check for performance drift against the baseline.

```bash
branes testbench drift yolov8n
branes testbench drift resnet50 --metric top1
```

| Option | Description |
|--------|-------------|
| `--metric TEXT` | Metric to check: mAP@50, top1, mIoU (default: mAP@50) |

#### testbench history

Show validation history for a model.

```bash
branes testbench history yolov8n
branes testbench history yolov8n --limit 20
branes testbench history yolov8n --clear
```

| Option | Description |
|--------|-------------|
| `--limit INT` | Number of entries to show (default: 10) |
| `--clear` | Clear history for this model |

#### testbench list

List models with validation history.

```bash
branes testbench list
```

---

## Infrastructure Commands

Commands for managing the API server, design sessions, specifications,
configuration, secrets, backends, and interactive chat.

### api

Manage the REST API server.

```bash
branes api [serve] [OPTIONS]
```

#### api serve

Start the REST API server. Serves design session data for the branes-frontend dashboard.

```bash
branes api serve
branes api serve --port 9000 --cors-origin http://localhost:3000
branes api serve --host 0.0.0.0 --reload
```

| Option | Description |
|--------|-------------|
| `--host TEXT` | Host to bind to (default: 127.0.0.1) |
| `--port INT` | Port to listen on (default: 8000) |
| `--cors-origin TEXT` | Allowed CORS origins (repeatable) |
| `--reload` | Enable auto-reload for development |

Requires `fastapi` and `uvicorn`. Docs at `http://host:port/docs`.

### session

Manage saved design sessions. Sessions are auto-saved by the SoC design pipeline
after every step.

```bash
branes session [list|show|delete] [OPTIONS]
```

#### session list

List all saved design sessions.

```bash
branes session list
```

#### session show

Inspect a saved session. Displays PPA metrics, constraint slackness, KPU
configuration, MOO summary, and design journey.

```bash
branes session show soc_abc123
branes session show --latest
branes session show --latest --json
```

| Option | Description |
|--------|-------------|
| `--latest` | Show the most recent session |
| `--json` | Output as JSON |

#### session delete

Delete a saved session.

```bash
branes session delete soc_abc123
branes session delete soc_abc123 --yes
```

| Option | Description |
|--------|-------------|
| `--yes, -y` | Skip confirmation |

### spec

Manage system specifications with versioning and provenance tracking.

```bash
branes spec [new|list|show|set|delete|commit|history|diff|tag|export|import|why|validate|resolve] [OPTIONS]
```

#### spec new

Create a new spec, optionally from a template.

```bash
branes spec new my-drone --template drone-perception
branes spec new my-robot -d "Warehouse picking robot"
branes spec new my-drone --template list   # list available templates
```

| Option | Description |
|--------|-------------|
| `--template, -t TEXT` | Template archetype (drone-perception, quadruped-nav, etc.) |
| `--description, -d TEXT` | Spec description |

#### spec list

List all specs.

```bash
branes spec list
```

#### spec show

Display a spec as a tree.

```bash
branes spec show my-drone
branes spec show my-drone --version v1.0
```

| Option | Description |
|--------|-------------|
| `--version, -v TEXT` | Version hash or tag |

#### spec set

Set a field on a spec.

```bash
branes spec set my-drone /perception/min_fps 60 -m "need 60fps"
branes spec set my-drone /compute/soc "Jetson Orin NX"
branes spec set my-drone /tags '["drone","outdoor"]'
```

| Option | Description |
|--------|-------------|
| `--message, -m TEXT` | Reason for the change |
| `--author TEXT` | Author of the change (default: user) |

#### spec delete

Remove a field from a spec.

```bash
branes spec delete my-drone /perception/model_family -m "no longer needed"
```

| Option | Description |
|--------|-------------|
| `--message, -m TEXT` | Reason for deletion |
| `--author TEXT` | Author (default: user) |

#### spec commit

Snapshot the current spec state.

```bash
branes spec commit my-drone -m "initial requirements"
```

| Option | Description |
|--------|-------------|
| `--message, -m TEXT` | Commit message (required) |
| `--author TEXT` | Author (default: user) |

#### spec history

Show version history for a spec.

```bash
branes spec history my-drone
```

#### spec diff

Show differences between two versions.

```bash
branes spec diff my-drone v1.0 v2.0
branes spec diff my-drone abc123 def456
```

#### spec tag

Tag a version with a human-readable name.

```bash
branes spec tag my-drone v1.0 -m "first baseline"
branes spec tag my-drone review-ready --version abc123
```

| Option | Description |
|--------|-------------|
| `--message, -m TEXT` | Tag message |
| `--version TEXT` | Version hash to tag (default: latest) |
| `--author TEXT` | Author (default: user) |

#### spec export

Export a spec to JSON or YAML.

```bash
branes spec export my-drone
branes spec export my-drone --format yaml > spec.yaml
```

| Option | Description |
|--------|-------------|
| `--format, -f TEXT` | Output format: json, yaml (default: json) |

#### spec import

Import a spec from a JSON or YAML file.

```bash
branes spec import my-drone-copy spec.json
branes spec import my-drone-copy spec.yaml
```

| Option | Description |
|--------|-------------|
| `--author TEXT` | Author (default: user) |

#### spec why

Show provenance of a field -- who changed it, when, and why.

```bash
branes spec why my-drone /perception/min_fps
```

#### spec validate

Run cross-subsystem consistency checks.

```bash
branes spec validate my-drone
```

#### spec resolve

Flatten a spec to dot-notation for agent consumption.

```bash
branes spec resolve my-drone
```

### config

Manage configuration settings.

```bash
branes config [OPTIONS]
```

### secrets

Manage credentials for remote backends.

```bash
branes secrets [set|get|delete] [OPTIONS]
```

### backends

Manage benchmark backends.

```bash
branes backends [list|add|remove] [OPTIONS]
```

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

---

## Deployment Commands

Commands for deploying models, running pipelines, executing workflows,
and running demos.

### deploy

Deploy models to edge/embedded targets.

```bash
branes deploy [run|list|info] [OPTIONS]
```

Supported targets: jetson, openvino, coral, swkpu (Stillwater KPU), nvdla.

#### deploy run

Deploy a model to target hardware.

```bash
branes deploy run model.pt --target jetson --precision int8 \
    --calibration-data ./calib_images --input-shape 1,3,224,224

branes deploy run model.onnx --target openvino --precision fp16 \
    --input-shape 1,3,640,640

branes deploy run model.onnx --target swkpu --precision int8 \
    --calibration-data ./calib --input-shape 1,3,224,224 \
    --power-budget 5.0 --measure-power
```

| Option | Description |
|--------|-------------|
| `--target, -t TEXT` | Deployment target: jetson, openvino, coral, swkpu, nvdla (default: jetson) |
| `--precision, -p TEXT` | Target precision: fp32, fp16, int8 (default: int8) |
| `--input-shape TEXT` | Model input shape (required, e.g., 1,3,224,224) |
| `--calibration-data, -c PATH` | Path to calibration dataset (required for INT8) |
| `--calibration-samples INT` | Number of calibration samples (default: 100) |
| `--calibration-preprocessing TEXT` | Preprocessing: imagenet, yolo, coco, none (default: imagenet) |
| `--test-data PATH` | Path to test dataset for validation |
| `--validate/--no-validate` | Run validation after deployment (default: yes) |
| `--output-dir, -o PATH` | Output directory for artifacts (default: ./deployments) |
| `--tolerance FLOAT` | Max accuracy drop tolerance in percent (default: 1.0) |
| `--power-budget FLOAT` | Power budget in watts for validation |
| `--measure-power/--no-measure-power` | Measure actual power during inference |

#### deploy list

List available deployment targets with availability and requirements.

```bash
branes deploy list
```

#### deploy info

Show detailed information about a deployment target.

```bash
branes deploy info jetson
```

### pipeline

Run operator pipelines with LangGraph orchestration.

```bash
branes pipeline [run|benchmark|list] [OPTIONS]
```

#### pipeline run

Run a pipeline on input data.

```bash
branes pipeline run perception --input image.jpg
branes pipeline run autonomy --input video.mp4 --stream
branes pipeline run autonomy-ekf --input video.mp4 --stream --max-frames 100
```

| Option | Description |
|--------|-------------|
| `--input, -i PATH` | Input image or video file (required) |
| `--execution-target, -t TEXT` | Hardware target: cpu, gpu, npu (default: cpu) |
| `--yolo-variant, -y TEXT` | YOLO variant: n, s, m, l, x (default: s) |
| `--stream` | Enable streaming mode for video input |
| `--max-frames INT` | Maximum frames to process in streaming mode |
| `--latency-budget FLOAT` | Target latency in milliseconds (default: 100.0) |
| `--checkpoint` | Enable LangGraph checkpointing for persistence |

#### pipeline benchmark

Benchmark pipeline performance over multiple iterations.

```bash
branes pipeline benchmark perception --input image.jpg --iterations 100
```

| Option | Description |
|--------|-------------|
| `--input, -i PATH` | Input image for benchmarking (required) |
| `--iterations, -n INT` | Number of timed iterations (default: 100) |
| `--warmup, -w INT` | Warmup iterations (default: 10) |
| `--execution-target, -t TEXT` | Hardware target: cpu, gpu, npu (default: cpu) |
| `--yolo-variant, -y TEXT` | YOLO variant: n, s, m, l, x (default: s) |

#### pipeline list

List available pipelines.

```bash
branes pipeline list
```

### workflow

Run the full analysis workflow (model analysis, hardware profiling, benchmarking,
report generation).

```bash
branes workflow [run|list] [OPTIONS]
```

#### workflow run

Run complete workflow on a model.

```bash
branes workflow run my_model.pt
branes workflow run my_model.pt --backend kubernetes
branes workflow run my_model.pt --max-latency 50 --max-power 100 --max-cost 3000
```

| Option | Description |
|--------|-------------|
| `--backend TEXT` | Benchmark backend: local_cpu, remote_ssh, kubernetes (default: local_cpu) |
| `--input-shape TEXT` | Model input shape (e.g., 1,3,224,224) |
| `--iterations INT` | Number of benchmark iterations (default: 100) |
| `--warmup INT` | Warmup iterations (default: 10) |
| `--max-latency FLOAT` | Maximum latency constraint in ms |
| `--max-power FLOAT` | Maximum power constraint in watts |
| `--max-cost FLOAT` | Maximum cost constraint in USD |

#### workflow list

List past workflow executions.

```bash
branes workflow list
branes workflow list --limit 20
```

| Option | Description |
|--------|-------------|
| `--limit INT` | Maximum number of workflows to list (default: 10) |

### demo

Discover and run platform demos.

```bash
branes demo [list|info|run] [OPTIONS]
```

#### demo list

List all available demos.

```bash
branes demo list
```

#### demo info

Show detailed info about a specific demo.

```bash
branes demo info kpu-rtl
```

#### demo run

Run a demo by name, or "all" to run every demo sequentially.

```bash
branes demo run dse-pareto
branes demo run soc-optimizer --power 4.0
branes demo run soc-designer --goal "Warehouse AMR" --llm
branes demo run all
```

| Option | Description |
|--------|-------------|
| `--goal TEXT` | Custom design goal (soc-designer demo) |
| `--power FLOAT` | Max power budget in watts |
| `--latency FLOAT` | Max latency in ms |
| `--cost FLOAT` | Max BOM cost in USD |
| `--max-iterations INT` | Max optimization iterations |
| `--llm` | Use LLM planner (soc-designer demo) |

Available demos: soc-designer, dse-pareto, soc-optimizer, kpu-rtl, hitl-safety, experience-cache, full-campaign.
