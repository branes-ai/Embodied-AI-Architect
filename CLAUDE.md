# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Embodied AI Architect is a design environment for creating, evaluating, optimizing and deploying embodied AI hw/sw
to imbue a device or system with intelligence and/or autonomy. 

It provides:
- Application architecture analysis and characterization
- Transformational analysis, sensor data acquisition, signal conditioning, signal processing, DNN inference, state estimation, actuation
- DNN and Linear Algebra Computational Graph extraction, analysis, characterization and estimation
- Model analysis and benchmarking across different hardware targets
- Hardware profiling with recommendations for edge/cloud deployment
- Pre-silicon Knowledge Processing Unit SoC hardware targets to leapfrog COTS designs and enable new use-cases
- Multi-objective optimization (MAP-Elites, Bayesian, NSGA-III)
- SWaP-C (Size, Weight, Power, Cost) analysis and optimization
- SoC design pipeline with RTL generation and EDA tool integration
- Multi-hardware benchmark execution (local CPU, remote SSH, Kubernetes)
- Report generation for model-to-hardware fit analysis

## Build & Development Commands

```bash
# Install in development mode
.venv/bin/pip install -e ".[dev]"

# Install with optional dependencies
.venv/bin/pip install -e ".[dev,remote,kubernetes]"

# Install with interactive chat support
.venv/bin/pip install -e ".[chat]"

# Run tests
.venv/bin/pytest tests/

# Run single test
.venv/bin/pytest tests/test_file.py::test_function -v

# Linting and formatting
.venv/bin/black src/ tests/ --line-length 100
.venv/bin/ruff check src/ tests/

# Pre-push quality gate (mandatory)
.venv/bin/black --check src/ tests/ --line-length 100
.venv/bin/ruff check src/ tests/
.venv/bin/pytest tests/ -q

# Run CLI
.venv/bin/branes --help
.venv/bin/branes workflow run model.pt
.venv/bin/branes analyze model.pt
.venv/bin/branes benchmark model.pt --backend local

# Codebase analysis (scan requires no API key)
.venv/bin/branes codebase scan /path/to/project
.venv/bin/branes codebase analyze /path/to/project
.venv/bin/branes codebase assess /path/to/project --hardware jetson_orin --power-budget 15

# SWaP-C analysis
.venv/bin/branes swap estimate --area 50 --power 5 --process 28
.venv/bin/branes swap score --area 50 --power 5 --process 28 --profile drone
.venv/bin/branes swap sensitivity --area 50 --power 5 --process 28 --mode tornado

# Start interactive chat session (requires ANTHROPIC_API_KEY)
export ANTHROPIC_API_KEY=your-key-here
.venv/bin/branes chat
```

## CLI Commands

The `branes` CLI has 20 command groups, organized by function:

**Analysis & Benchmarking:**
- `analyze` - Analyze model architecture and complexity
- `benchmark` - Run performance benchmarks
- `codebase` - Analyze application codebases for hardware assessment
- `testbench` - Model validation and drift monitoring

**Optimization & Design:**
- `optimize` - Multi-objective design space optimization
- `design` - Design perception pipelines from requirements
- `swap` - System-level SWaP-C analysis (score, rank, sensitivity, sweep, budget)
- `spec` - Manage system specifications with versioning and provenance

**Deployment:**
- `deploy` - Deploy models to edge/embedded targets
- `pipeline` - Run operator pipelines with LangGraph orchestration
- `workflow` - Run complete workflows for model evaluation

**Model Management:**
- `model` - Manage model registry
- `zoo` - Manage models from the Model Zoo

**Environment & Config:**
- `chat` - Interactive AI architect session (Claude Code-style)
- `config` - Manage configuration settings
- `report` - View and manage reports
- `backends` - Manage benchmark backends
- `secrets` - Manage secrets and credentials
- `demo` - Discover and run platform demos
- `mcp` - Graphs estimator CLI client (hardware, analyze, latency, energy, memory, compare, specs)

## Architecture

### Core System (`src/embodied_ai_architect/`)

**Orchestrator Pattern**: The `Orchestrator` class coordinates agent execution in a pipeline:
1. ModelAnalyzer → analyzes PyTorch model structure
2. HardwareProfile → recommends hardware based on model characteristics
3. Benchmark → measures actual performance on target backends
4. ReportSynthesis → generates HTML/JSON reports

**Agent System** (`agents/`):
- All agents extend `BaseAgent` and implement `execute(input_data) -> AgentResult`
- Agents are registered with the Orchestrator and executed in sequence
- Each agent produces an `AgentResult` with success status, data, and optional error

**Benchmark Backends** (`agents/benchmark/backends/`):
- `LocalCPUBackend`: Local CPU inference
- `RemoteSSHBackend`: Execute on remote machines via SSH (requires `paramiko`)
- `KubernetesBackend`: Distributed benchmarking on K8s (requires `kubernetes`)

**Deployment Targets** (`agents/deployment/targets/`):
- `JetsonTarget` - NVIDIA Jetson (Orin AGX/NX/Nano) with TensorRT
- `CoralTarget` - Google Coral Edge TPU (USB Accelerator, Dev Board)
- `OpenVINOTarget` - Intel OpenVINO with NNCF quantization
- `StillwaterKPUTarget` - Stillwater KPU custom accelerator
- `NVDLATarget` - NVIDIA Deep Learning Accelerator (licensable IP)
- Power monitoring (`power/monitor.py`) and prediction (`power/predictor.py`)

**Codebase Analysis** (`codebase/`): Full application analysis pipeline:
- `scanner.py` - Static file scanner (languages, build system, ML models, deps)
- `analyzer.py` - LLM multi-pass code analyzer (4 passes: build→entry→kernels→synthesis)
- `converter.py` - Maps `CodebaseAnalysisResult` → `workload_profile` for PPA pipeline
- `models.py` - Pydantic models (`ComputeKernel`, `ScanResult`, `CodebaseAnalysisResult`)

**CLI** (`cli/`): Click-based CLI with 19 command groups (see CLI Commands above)

**LLM Integration** (`llm/`): Interactive agent system:
- `LLMClient` - Claude API wrapper with tool use support
- `ArchitectAgent` - Agentic loop that reasons and calls tools
- `tools.py` - Tool definitions wrapping existing agents
- `codebase_tools.py` - Tools for codebase scan/analyze/assess in chat

**Specs System** (`specs/`): Hierarchical system specifications with versioning:
- `branes spec create` / `show` / `list` / `diff` / `validate`
- YAML-based specs with provenance tracking

### Graphs Subsystem (`src/embodied_ai_architect/graphs/`)

The largest subsystem (60+ files) — the computational core for SoC design,
optimization, and physical analysis.

**SoC Design Pipeline:**
- `soc_state.py`, `soc_runner.py`, `soc_graph.py` - SoC topology and state machine
- `specialists.py` - Architecture, power, thermal, timing specialists
- `ip_blocks.py` - IP block specifications and integration
- `planner.py`, `dispatcher.py`, `runner.py` - Design orchestration

**Multi-Objective Optimization** (`moo/`): 3-layer pipeline integrated into the
LangGraph SoC design pipeline as the `moo_explorer` specialist (issues #21–#27):

- Layer 1: `map_elites.py` - MAP-Elites quality-diversity search (5K-10K evals)
- Layer 2: `bayesian_opt.py` - Bayesian BO with qLogNEHVI (sample-efficient, ≤4 objectives)
- Layer 3: `nsga3.py` - NSGA-III many-objective optimization (>4 objectives)
- `engine.py` - Orchestrates the 3-layer pipeline; emits `OptimizationResult`
- `design_space.py` - Design space definition and sampling (17-var SoC space)
- `evaluator.py`, `k8s_evaluator.py` - Local and distributed evaluation
- `executor.py` - Execution orchestration
- `specialist.py` - `moo_explorer` (4-obj) and `swap_explorer` (6-obj) specialists
  that bridge `OptimizationEngine` to the LangGraph state and merge per-iteration
  Pareto frontiers (`_merge_pareto_frontiers` for monotonic accumulation)

**MOO data flow through the LangGraph pipeline:**

```
planner → dispatch ──┬──> hw_explorer ────┐
                     │                    ├──> moo_explorer ──> ppa_assessor → ...
                     │                    │   (writes:
                     │                    │     pareto_points
                     │                    │     pareto_frontier_history
                     │                    │     moo_results)
                     └──> architecture_composer ──┘
```

The `moo_explorer` task is **always scheduled in the default plan** (after
`hw_explorer`, in parallel with `architecture_composer`), so any `SoCDesignRunner`
session automatically produces a populated Pareto frontier. The escape hatch is
`enable_moo=False` on the design state (planner strips MOO tasks from the plan).

**Key state fields produced by MOO** (consumed by API, snapshot, and architect skills):

| Field | Producer | Consumer |
|---|---|---|
| `pareto_points` | `moo_explorer` (merged across iterations) | `/api/sessions/{id}/pareto`, snapshot |
| `pareto_frontier_history` | `moo_explorer` per iteration | trajectory views, frontier monotonicity check |
| `moo_results` | `OptimizationResult.model_dump()` | snapshot, sensitivity API, optimizer #25 |
| `moo_results.sensitivity` | BO layer hyperparameters | `/api/sessions/{id}/sensitivity`, `/architect-loop`, `design_optimizer` MOO-aware ranking |
| `moo_results.layers_used` | engine | snapshot, `branes session show` MOO summary |
| `moo_results.atlas` | MAP-Elites | snapshot (coverage %) |
| `optimization_review_snapshot.pareto_front_size` | snapshot builder | `branes session show`, API, architect skills |
| `last_strategy_rationale` | `design_optimizer` (MOO-aware selector) | snapshot, architect skills |

**SWaP-C Analysis:**
- `physical_estimators.py` - Physical dimension, thermal, cost estimators
- `bom.py` - Bill of Materials estimation
- `swap_report.py` - SWaP-C report generation
- `swap_profiles.py` - Profile definitions and templates (drone, UGV, etc.)
- `swap_analysis.py` - SWaP-C analysis engine

**KPU Design:**
- `kpu_loop.py` - KPU design iteration loop
- `kpu_config.py` - Stillwater KPU configuration
- `kpu_specialists.py` - KPU architecture specialists

**RTL Generation:**
- `rtl_loop.py` - RTL design iteration loop
- `rtl_specialists.py` - RTL design specialists (routing, timing)
- `rtl_templates/` - RTL template library

**EDA Tools** (`eda_tools/`):
- `synthesis.py` - Logic synthesis
- `simulation.py` - RTL simulation
- `lint.py` - Design linting
- `toolchain.py` - EDA toolchain integration

**Supporting Modules:**
- `memory.py` - Memory subsystem modeling
- `bandwidth.py` - Bandwidth analysis
- `technology.py` - Process technology models
- `manufacturing.py` - Manufacturing process/yield modeling
- `floorplan.py` - Physical floorplanning
- `pareto.py` - Pareto frontier analysis
- `optimizer.py` - Optimization orchestration
- `governance.py` - Design review workflows
- `gold_standards.py` - Golden reference designs
- `safety.py` - Safety constraint validation
- `scoring.py` - Design scoring metrics

### Prototypes (`prototypes/`)

**drone_perception/**: Real-time perception pipeline for drones
- `app/` - Deployable perception application (`main.py`)
- `lib/` - Pipeline library: sensors, detection, tracking, scene_graph, reasoning, visualization
- `examples/` - Basic usage (simple_detection, full_pipeline)
- `demos/` - Advanced sensor demos (stereo, wide-angle, LiDAR, reasoning)
- `scripts/` - Dev utilities (calibration, depth maps, comparison)
- `tests/` - Test runners and validation

```bash
# Run drone perception pipeline
cd prototypes/drone_perception
.venv/bin/pip install -r requirements.txt
python app/main.py --sensor mono --video 0             # deployable app
python examples/full_pipeline.py --video 0             # example
python demos/reasoning_pipeline.py --camera 0 --model s  # demo
```

**multi_rate_framework/**: Multi-rate control system using Zenoh pub/sub
- Components run at different frequencies (1Hz, 10Hz, 100Hz)
- `@control_loop` decorator for rate-specified execution

```bash
cd prototypes/multi_rate_framework
.venv/bin/pip install eclipse-zenoh
python example_multirate.py
```

## Key Design Patterns

- **Pydantic models** for data validation (`AgentResult`, `WorkflowResult`)
- **Optional dependencies** with try/except imports (see `backends/__init__.py`)
- **Rich console** output for CLI
- **Jinja2 templates** for HTML report generation

## Related Repositories

This project is part of a multi-repo architecture:

```
embodied-schemas (shared dependency)
       ↑              ↑
       │              │
   graphs      embodied-ai-architect (this repo)
```

### embodied-schemas (`../embodied-schemas`)
Shared Pydantic schemas and factual data catalog. This repo imports:
- `HardwareEntry`, `ModelEntry`, `SensorEntry`, `UseCaseEntry` - Data models
- `BenchmarkResult` - Verdict-first output schema for tools
- `Registry` - Unified data access API
- Constraint tier definitions (latency, power classes)

**Usage:**
```python
from embodied_schemas import HardwareEntry, Registry, BenchmarkResult
from embodied_schemas.constraints import LatencyTier, get_latency_tier
```

### graphs (`../graphs`)
Sibling analysis library — roofline models, hardware simulation, calibration.
- `estimation/` - Roofline, energy, memory analysis (`unified_analyzer.py`)
- `hardware/` - Hardware models for datacenter, edge, automotive, mobile, accelerators
- `hardware/mappers/` - Architecture-specific execution models (CPU, GPU, DSP, KPU, TPU)
- `calibration/` - Measured performance data (e.g., Jetson Orin profiles by power mode)
- `benchmarks/` - Micro-benchmarks (GEMM, Conv2D, attention, TensorRT)
- `research/` - Dataflow analysis, systolic array design, tiling optimization
- `subgraphs/` - Kernel patterns (attention, conv2d_stack, MLP, ResNet block)

### Data Split
- **Datasheet specs** (vendor-published facts) → `embodied-schemas`
- **Analysis-specific data** (roofline params, calibration) → `graphs`

## Commit Convention

Conventional commits required: `type(scope): description`

Types: `feat`, `fix`, `docs`, `chore`, `refactor`, `test`, `ci`, `perf`, `style`, `build`

Examples:
```
feat(swap): add thermal feasibility scoring
fix(moo): correct NSGA-III crowding distance
docs: update CLI command reference
chore(release): 0.8.0 [skip ci]
```

Enforced by PR template and semantic-release.

## Code Style

- Line length: 100 characters
- Python target: 3.11+
- Use type hints
- Format with Black, lint with Ruff
- Always use `.venv/bin/` prefix for tool commands
