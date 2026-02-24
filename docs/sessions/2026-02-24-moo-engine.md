# Session: Multi-Objective Optimization Engine

**Date:** 2026-02-24

## Summary

Implemented a comprehensive 3-layer multi-objective optimization engine for SoC design space exploration, based on the plan in `docs/plans/multi_objective_optimization_engine.md`. The engine provides MAP-Elites for fast design space illumination, Bayesian optimization with qNEHVI for sample-efficient Pareto refinement, and NSGA-III for many-objective problems. Integrated with the existing specialist dispatcher, LLM agent tools, MCP server, and CLI.

## Architecture

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
  |  (100-200 evals)          |
  +-------------+-------------+
                |  Pareto front + GP posteriors
                v
  +---------------------------+
  |  Layer 3: LLM Interpret   |  ArchitectAgent explains tradeoffs,
  |  (existing agent)         |  suggests design point based on priorities
  +---------------------------+

For >4 objectives: Layer 2 swaps from BO to NSGA-III/MOEA/D automatically.
```

## Files Created

| File | Description |
|------|-------------|
| `src/embodied_ai_architect/graphs/moo/__init__.py` | Package init with public API exports |
| `src/embodied_ai_architect/graphs/moo/design_space.py` | DesignVariable, DesignSpace, LHS sampling, encode/decode, factory |
| `src/embodied_ai_architect/graphs/moo/evaluator.py` | Thread-safe DesignEvaluator wrapping existing PPA models |
| `src/embodied_ai_architect/graphs/moo/executor.py` | EvalBackend protocol, LocalThreadExecutor |
| `src/embodied_ai_architect/graphs/moo/map_elites.py` | Layer 1: quality-diversity algorithm (~250 lines) |
| `src/embodied_ai_architect/graphs/moo/bayesian_opt.py` | Layer 2: BoTorch qNEHVI with GP surrogates (requires botorch) |
| `src/embodied_ai_architect/graphs/moo/nsga3.py` | Layer 3: pymoo NSGA-III wrapper (requires pymoo) |
| `src/embodied_ai_architect/graphs/moo/engine.py` | Pipeline orchestrator with auto layer selection |
| `src/embodied_ai_architect/graphs/moo/specialist.py` | Bridge to dispatcher: moo_explorer(task, state) |
| `src/embodied_ai_architect/graphs/moo/k8s_evaluator.py` | Kubernetes evaluation backend |
| `src/embodied_ai_architect/mcp/__init__.py` | MCP package init |
| `src/embodied_ai_architect/mcp/session.py` | OptimizationSession, SessionManager |
| `src/embodied_ai_architect/mcp/server.py` | 5 MCP tools with dual-response pattern |
| `src/embodied_ai_architect/llm/optimization_tools.py` | 5 LLM tools following existing pattern |
| `src/embodied_ai_architect/cli/commands/optimize.py` | Click CLI commands: explore, show-front, sensitivity, explain |
| `tests/test_design_space.py` | 15 tests |
| `tests/test_moo_evaluator.py` | 9 tests |
| `tests/test_map_elites.py` | 8 tests |
| `tests/test_bayesian_moo.py` | 8 tests (skip if no botorch) |
| `tests/test_moo_engine.py` | 11 tests |
| `tests/test_mcp_server.py` | 9 tests |

## Files Modified

| File | Change |
|------|--------|
| `src/embodied_ai_architect/graphs/soc_state.py` | Added `moo_results` field to SoCDesignState |
| `src/embodied_ai_architect/graphs/specialists.py` | Registered moo_explorer with try/except guard |
| `src/embodied_ai_architect/llm/tools.py` | Added HAS_MOO flag and optimization tools |
| `src/embodied_ai_architect/cli/__init__.py` | Registered optimize command group |
| `pyproject.toml` | Added optimization and mcp optional dependency groups |

## Key Design Decisions

1. **Evaluation is a pure function** — `DesignEvaluator.evaluate()` has no side effects, safe for `ThreadPoolExecutor`. Reuses existing `SoCComposition` + `estimate_manufacturing_cost`, no duplication.

2. **MCP is the interface layer, NOT the compute layer** — 5K-10K evaluations/sec need microsecond overhead. MCP handles LLM-engine interaction. The hot loop is `ThreadPoolExecutor` or K8s Jobs.

3. **Layer handoff** — MAP-Elites results seed BO initial observations. Avoids redundant random sampling. BO warm-starts from the atlas's best points.

4. **Auto layer selection** — <=4 objectives: MAP-Elites + Bayesian BO. >4 objectives: MAP-Elites + NSGA-III. User can override.

5. **Backward compatible** — Results flow into existing `pareto_results` format via `to_pareto_results()`. New `moo_results` field for rich data. Existing consumers see no change.

## Test Results

- 52 new MOO tests: all pass
- 8 Bayesian tests: skipped (botorch not installed)
- Full test suite: 642 passed, 48 skipped, 39 errors (all pre-existing)
- Ruff: clean (0 issues)
- Black: formatted (line-length 100)

## Usage

```bash
# Install with optimization support
pip install -e ".[optimization]"

# Fast exploration (MAP-Elites only)
branes optimize explore --goal "drone perception SoC" --power 5 --latency 33 --fast

# Full pipeline
branes optimize explore --goal "drone SoC" --power 5 --latency 33

# View results
branes optimize show-front --top 10
branes optimize sensitivity
branes optimize explain --points 0,3

# From chat
embodied-ai chat
> explore the design space for a drone SoC with 5W power budget and 33ms latency
```
