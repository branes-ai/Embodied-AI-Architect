# Plan: Multi-Objective Optimization Engine

## Context

The current optimizer (`optimizer.py`) is a greedy constraint-fixer that tries one strategy per iteration. The Pareto code (`pareto.py`) does brute-force non-dominated sorting on a fixed hardware catalog with 3 hardcoded objectives. Design exploration — understanding the feasible region, tradeoff surfaces, and parameter sensitivity — is almost more valuable than any single optimal point. We need a proper MOO engine with concurrent evaluation, optional K8s offloading, and MCP server integration for the LLM agent.

**Research basis:** `docs/multi-objective-optimization-research.md` (34 references, DAC/NeurIPS/ICML papers on MOO for chip design).

## Architecture: 3-Layer Pipeline

```
User requirements + constraints
        |
        v
  +---------------------------+
  |  Layer 1: MAP-Elites      |  Fast atlas on analytical models
  |  (5K-10K evals, ~seconds) |  "Here's what the design space looks like"
  +-------------+-------------+
                |  seed points + feasibility map
                v
  +---------------------------+
  |  Layer 2: Bayesian BO     |  Refined Pareto front + sensitivity
  |  qNEHVI + exploration     |  "Here's the tradeoff surface"
  |  (100-200 evals)          |  "Here's where we're uncertain"
  +-------------+-------------+
                |  Pareto front + GP posteriors
                v
  +---------------------------+
  |  Layer 3: LLM Interpret   |  ArchitectAgent explains tradeoffs,
  |  (existing agent)         |  suggests design point based on priorities
  +---------------------------+

For >4 objectives: Layer 2 swaps from BO to NSGA-III/MOEA/D automatically.
```

## Deliverables

### 1. NEW: `src/embodied_ai_architect/graphs/moo/__init__.py`

Package init exporting public API: `DesignSpace`, `DesignEvaluator`, `OptimizationEngine`, `OptimizationResult`.

### 2. NEW: `src/embodied_ai_architect/graphs/moo/design_space.py`

Formal design space definition with mixed discrete/continuous variables.

- `DesignVariable(BaseModel)` — name, var_type (integer/continuous/categorical), bounds, choices
- `DesignSpace(BaseModel)` — variables, objectives, directions, constraint bounds
  - `sample_random(n)` → LHS initial sample
  - `encode(params_dict)` → numpy array for optimizer
  - `decode(x_array)` → params dict for evaluator
  - `to_pymoo_problem_kwargs()` → pymoo problem definition
  - `to_botorch_bounds()` → torch tensor bounds
- `create_soc_design_space(constraints)` — factory with default SoC variables:
  - process_nm: categorical [2..180] (from `technology.py` TECHNOLOGY_NODES)
  - clock_mhz: continuous [100, 3000]
  - array_rows/cols: integer [2, 64]
  - sram_kb: integer [32, 2048]
  - num_compute_tiles: integer [1, 16]
  - noc_link_width_bits: categorical [64, 128, 256, 512]

### 3. NEW: `src/embodied_ai_architect/graphs/moo/evaluator.py`

Wraps existing specialists into a callable objective function. **Thread-safe, pure function.**

- `EvaluationResult(BaseModel)` — design_params, objectives dict, constraints_satisfied, feasible
- `DesignEvaluator` class:
  - `__init__(design_space, base_state, constraint_bounds)` — captures the workload context
  - `evaluate(params_dict) -> EvaluationResult` — called 100-10K times
  - `evaluate_batch(param_list) -> list[EvaluationResult]`

**Reuses** (not duplicates):
- `SoCComposition` + `IPBlockConfig` from `graphs/ip_blocks.py` — builds design from params
- `estimate_manufacturing_cost()` from `graphs/manufacturing.py` — cost objective
- `get_technology()` from `graphs/technology.py` — process-aware PPA
- PPA estimation logic from `specialists.py` `_estimate_power`, `_estimate_latency`, `_estimate_area`

### 4. NEW: `src/embodied_ai_architect/graphs/moo/executor.py`

Concurrent evaluation backends with a shared `EvalBackend` protocol.

- `EvalBackend(Protocol)` — `submit_batch(params) -> list[EvaluationResult]`, `shutdown()`
- `LocalThreadExecutor` — `ThreadPoolExecutor(max_workers=8)` for analytical models
- `KubernetesEvalExecutor` — extends existing K8s pattern for batch evaluation jobs

### 5. NEW: `src/embodied_ai_architect/graphs/moo/map_elites.py`

Layer 1: Fast design space illumination.

- `MAPElitesConfig(BaseModel)` — feature_dimensions, resolution, n_iterations, batch_size
- `MAPElitesResult(BaseModel)` — grid (cell → best design), coverage, best_per_objective
- `MAPElites` class:
  - `run(callback)` → fills the grid atlas
  - `_mutate()`, `_crossover()`, `_feature_to_cell()`
  - Uses `executor.submit_batch()` for parallel evaluation (batch_size=64 per iteration)

Custom implementation (~200 lines). pymoo lacks MAP-Elites; the algorithm is simple enough.

### 6. NEW: `src/embodied_ai_architect/graphs/moo/bayesian_opt.py`

Layer 2: Sample-efficient Pareto refinement. **Requires BoTorch (optional dep).**

- `BayesianOptConfig(BaseModel)` — n_initial, n_iterations, batch_size, exploration_schedule
- `SensitivityResult(BaseModel)` — parameter_name, sobol_total, sobol_first_order, lengthscale
- `BayesianOptResult(BaseModel)` — pareto_front, hypervolume, sensitivity, convergence_history
- `BayesianMOO` class:
  - `run(callback)` → qNEHVI loop with exploration-exploitation schedule
  - `get_sensitivity()` → extracts from GP lengthscales (free after fitting)
  - `predict(params)` → mean + std per objective
  - `explain_tradeoff(point_a, point_b)` → parameter deltas + GP gradient analysis
  - Warm-starts from MAP-Elites results (Layer 1 → Layer 2 handoff)

Exploration-exploitation schedule:
```python
alpha = max(0.0, 1.0 - 2.0 * (iteration / budget))
acquisition = alpha * GP_variance + (1 - alpha) * EHVI
```

### 7. NEW: `src/embodied_ai_architect/graphs/moo/nsga3.py`

Layer 3: Many-objective fallback (>4 objectives). **Requires pymoo.**

- `NSGA3Config(BaseModel)` — pop_size, n_generations, reference_directions
- `NSGA3Result(BaseModel)` — pareto_front, hypervolume, generation_history
- `NSGA3MOO` class wrapping pymoo's `NSGA3` with our `DesignSpace` and evaluator

### 8. NEW: `src/embodied_ai_architect/graphs/moo/engine.py`

Orchestrates the 3-layer pipeline.

- `OptimizationConfig(BaseModel)` — layers (auto/map_elites/bayesian/nsga3), per-layer configs, backend, max_workers
- `OptimizationResult(BaseModel)` — pareto_front, hypervolume, knee_point, sensitivity, atlas, convergence_history
- `OptimizationEngine` class:
  - Auto-selects layers: <=4 objectives → MAP-Elites + BO; >4 → MAP-Elites + NSGA-III
  - `run(callback)` → executes pipeline, Layer 1 seeds Layer 2
  - `get_pareto_front()`, `get_sensitivity()`, `explain_tradeoff()` — queryable during/after run

### 9. NEW: `src/embodied_ai_architect/graphs/moo/specialist.py`

Bridge to existing dispatcher pattern: `moo_explorer(task, state) -> dict`.

- Same signature as all specialists: `(TaskNode, SoCDesignState) -> dict`
- Reads: constraints, workload_profile
- Writes: `pareto_results` (backward compatible), `moo_results` (new field)
- Supports `task.metadata["fast_mode"]` for reduced evaluation budgets

### 10. NEW: `src/embodied_ai_architect/graphs/moo/k8s_evaluator.py`

K8s evaluation backend for expensive evaluations (synthesis, simulation).

- Extends existing K8s Job pattern from `agents/benchmark/backends/kubernetes.py`
- Concurrent job submission (not sequential like existing `execute_parallel`)
- Semaphore-based concurrency control (`max_parallel_jobs`)
- EDA license management via K8s secrets (`eda_license_secret` mount)
- Result collection as jobs complete (not in submission order)

### 11. NEW: `src/embodied_ai_architect/mcp/__init__.py` + `server.py` + `session.py`

MCP server exposing optimization as tools for the LLM agent. **NOT in the hot evaluation loop** — MCP is the interface layer, not the compute layer.

**Tools (Dual-Response pattern):**
- `start_exploration(goal, constraints, config)` → session_id (async, optimization runs in background thread)
- `get_pareto_front(session_id, top_n=5)` → preview (top-5 for LLM) + ResourceLink (full front)
- `get_sensitivity(session_id)` → Sobol indices + GP lengthscales
- `explain_tradeoff(session_id, point_a, point_b)` → parameter/objective deltas
- `get_exploration_status(session_id)` → layer, iteration, progress

**Resources:**
- `hardware_catalog` → Dual-Response: 5-entry preview + paginated ResourceLink
- `process_nodes` → 17-node technology database

**Session management** (`session.py`):
- `OptimizationSession` — runs engine in background thread, queryable status
- `SessionManager` — tracks active sessions, cleanup expired

### 12. NEW: `src/embodied_ai_architect/llm/optimization_tools.py`

LLM tool definitions following the `get_*_tool_definitions()` + `create_*_tool_executors()` pattern from `llm/codebase_tools.py`.

5 tools: `explore_design_space`, `get_pareto_front`, `get_design_sensitivity`, `explain_design_tradeoff`, `suggest_optimal_design`.

When MCP server available: tools proxy to MCP (async, stateful sessions).
When MCP not available: tools run optimization inline (blocking).

### 13. MODIFY: `src/embodied_ai_architect/llm/tools.py`

Register optimization tools with try/except guard (`HAS_MOO` flag).

### 14. NEW: `src/embodied_ai_architect/cli/commands/optimize.py`

Click command group:
```
branes optimize explore --goal "..." --power 5 --latency 33 [--fast] [--backend kubernetes]
branes optimize show-front [session-id] [--top 10]
branes optimize sensitivity [session-id]
branes optimize explain [session-id] --points 0,3
```

### 15. MODIFY: `src/embodied_ai_architect/cli/__init__.py`

Register `optimize` command group.

### 16. MODIFY: `src/embodied_ai_architect/graphs/soc_state.py`

Add `moo_results: dict` field to SoCDesignState and `create_initial_soc_state()`.

### 17. MODIFY: `src/embodied_ai_architect/graphs/specialists.py`

Register `moo_explorer` in `create_default_dispatcher()` with try/except guard.

### 18. MODIFY: `pyproject.toml`

New optional dependency groups:
```toml
optimization = ["pymoo>=0.6.0", "botorch>=0.11.0", "gpytorch>=1.12", "scipy>=1.10.0"]
mcp = ["mcp>=1.0.0", "uvicorn>=0.25.0", "httpx>=0.25.0"]
```

### 19. Tests

| File | Tests | Coverage |
|------|-------|---------|
| `tests/test_design_space.py` | ~10 | Variable types, encode/decode roundtrip, factory, pymoo/botorch export |
| `tests/test_moo_evaluator.py` | ~8 | Single eval, batch, feasibility, thread safety (100 concurrent), reuse of PPA logic |
| `tests/test_map_elites.py` | ~6 | Init, small run, mutation bounds, feature→cell, result structure, coverage |
| `tests/test_bayesian_moo.py` | ~8 | Init, warm-start, small run, sensitivity, predict, explain (skip if no botorch) |
| `tests/test_moo_engine.py` | ~8 | Auto layer selection, full pipeline, result compatibility, specialist integration |
| `tests/test_mcp_server.py` | ~6 | Tool calls, dual-response, session lifecycle (skip if no mcp) |

## File Summary

| File | Action | Depends On |
|------|--------|-----------|
| `graphs/moo/__init__.py` | CREATE | — |
| `graphs/moo/design_space.py` | CREATE | technology.py, soc_state.py |
| `graphs/moo/evaluator.py` | CREATE | ip_blocks.py, manufacturing.py, specialists.py |
| `graphs/moo/executor.py` | CREATE | evaluator.py |
| `graphs/moo/map_elites.py` | CREATE | design_space.py, executor.py (numpy only) |
| `graphs/moo/bayesian_opt.py` | CREATE | design_space.py, executor.py (needs botorch) |
| `graphs/moo/nsga3.py` | CREATE | design_space.py, executor.py (needs pymoo) |
| `graphs/moo/engine.py` | CREATE | map_elites, bayesian_opt, nsga3 |
| `graphs/moo/specialist.py` | CREATE | engine.py |
| `graphs/moo/k8s_evaluator.py` | CREATE | executor.py, kubernetes backend pattern |
| `mcp/__init__.py` | CREATE | — |
| `mcp/server.py` | CREATE | engine.py, session.py |
| `mcp/session.py` | CREATE | engine.py |
| `llm/optimization_tools.py` | CREATE | engine.py |
| `cli/commands/optimize.py` | CREATE | engine.py |
| `llm/tools.py` | MODIFY | optimization_tools.py |
| `cli/__init__.py` | MODIFY | optimize.py |
| `graphs/soc_state.py` | MODIFY | — |
| `graphs/specialists.py` | MODIFY | moo/specialist.py |
| `pyproject.toml` | MODIFY | — |
| 6 test files | CREATE | — |

## Key Design Decisions

1. **Evaluation is a pure function** — `DesignEvaluator.evaluate()` has no side effects, safe for `ThreadPoolExecutor`. Reuses existing `SoCComposition` + `estimate_manufacturing_cost`, no duplication.

2. **MCP is the interface layer, NOT the compute layer** — 5K-10K evaluations/sec need microsecond overhead. MCP handles LLM↔engine interaction (start exploration, query front, explain tradeoff). The hot loop is `ThreadPoolExecutor` or K8s Jobs.

3. **Layer handoff** — MAP-Elites results seed BO initial observations. Avoids redundant random sampling. BO warm-starts from the atlas's best points.

4. **K8s for expensive evaluations** — Analytical model evals stay local (ThreadPool). Synthesis/simulation jobs go to K8s with concurrency control, license management via secrets.

5. **Auto layer selection** — <=4 objectives: MAP-Elites → Bayesian BO. >4 objectives: MAP-Elites → NSGA-III. User can override.

6. **Backward compatible** — Results flow into existing `pareto_results` format. New `moo_results` field for rich data. Existing `design_explorer` consumers see no change.

## Implementation Sequence

Build in this order (each step testable independently):

1. `design_space.py` + tests — Pure data model, no external deps
2. `evaluator.py` + tests — Wraps existing specialists, validate thread safety
3. `executor.py` — LocalThreadExecutor first
4. `map_elites.py` + tests — Layer 1, needs only numpy
5. `engine.py` + tests — Orchestration with MAP-Elites only initially
6. `specialist.py` — Register with dispatcher, integration test
7. `cli/commands/optimize.py` — CLI commands
8. `bayesian_opt.py` + tests — Layer 2, requires botorch
9. `nsga3.py` — Layer 3, requires pymoo
10. `mcp/` + tests — MCP server integration
11. `llm/optimization_tools.py` — LLM tool wrappers
12. `k8s_evaluator.py` — K8s evaluation backend
13. `pyproject.toml` — Optional dependency groups

## Verification

```bash
# Install with optimization support
pip install -e ".[optimization]"

# Run core tests (no optional deps needed for MAP-Elites)
pytest tests/test_design_space.py tests/test_moo_evaluator.py tests/test_map_elites.py tests/test_moo_engine.py -v

# Run BO tests (requires botorch)
pytest tests/test_bayesian_moo.py -v

# Run MCP tests (requires mcp)
pip install -e ".[mcp]"
pytest tests/test_mcp_server.py -v

# CLI test
branes optimize explore --goal "drone perception SoC" --power 5 --latency 33 --fast

# Chat test (requires ANTHROPIC_API_KEY)
branes chat
> explore the design space for a drone SoC with 5W power budget and 33ms latency

# Existing tests still pass
pytest tests/ -v

# Lint
ruff check src/embodied_ai_architect/graphs/moo/ src/embodied_ai_architect/mcp/
black src/embodied_ai_architect/graphs/moo/ src/embodied_ai_architect/mcp/ --check --line-length 100
```
