# Session 2026-07-16 -- Loop Convergence: Phases 3 & 4 (epic complete)

**Repo:** `branes-ai/embodied-ai-architect`
**Epic:** `#203` (Loop Convergence) -- **CLOSED**, all 12 seams (S1-S12) merged
**Landed on `main`:** PRs #230-#235

## Context

Phases 1-2 unified the two design loops onto one `DesignState` and made the
critic/optimizer real. This session added review tooling + a demonstration, then
completed Phases 3 (specialist agents) and 4 (wire the loop to a CLI + tune
convergence), closing the epic.

## Review tooling + demonstration (PRs #230, #231)

- **`graphs/loop_trace.py`** -- there was no step logger for the unified loop (only
  the dispatcher had `TracingDispatcher`). `run_loop_traced` drives the real
  seed -> (critic -> route -> optimize -> evaluate)* -> recommend loop over the
  actual node functions and records a `LoopTrace`: one `LoopStep` per node with the
  decision and the *reason why* (critic's issues + analysis, the deltas applied and
  their rationale/research, verdicts, routing, convergence). It logs to the
  `embodied_ai_architect.loop` logger and has a human-in-the-loop `steer` hook.
- **`tests/test_loop_convergence_acceptance.py`** -- a gold-standard-style harness:
  `LoopScenario`s with known targets (iterations, convergence, final verdicts,
  applied-delta count) over a fake MOO tool + scripted verdicts. It caught that a
  constant-hypervolume tool triggers the plateau convergence path -- a real
  `has_converged` signal, then covered as an explicit scenario.
- **`docs/loop-convergence-demo.md` + `docs/demos/loop_convergence_demo.py`** -- a
  user-facing demo that solves a drone-perception SoC (<=5W/<=33ms/<=$30): 8W/45ms
  -> 4.6W/17.3ms autonomously in one iteration; a steered run tightens power to 4W,
  the INT8 lever stalls at 4.6W, the operator injects structured sparsity, and it
  reconverges at 3.8W. Honest about real (the loop) vs illustrative (surrogate MOO
  tool + knee-scoring evaluate).

## Phase 3 -- specialist agents (S9 #213, S8 #214, PR #232)

- **`graphs/specialist_agents.py`** -- verdict-first estimator tools
  (power/latency/area/cost/thermal, exposing verdict/margin/overshoot) and two
  reasoning agents that file typed `DesignIssue`s: `PPASpecialist` (dominant
  bottleneck first, severity by overshoot) and `ThermalSpecialist` (junction
  temperature via a new `physical_estimators.estimate_junction_temperature`). The
  numeric estimation stays deterministic in the tools; the judgment lives in the
  agents. `specialist_registry()` names them.
- **S8** -- `loop_agents._run_pending_retasks` consumes `SPECIALIST_RETASK`'s
  `pending_specialist_tasks`, re-runs the named specialist, files fresh issues, and
  drains the queue; wired into `optimizer_node`.
- CodeRabbit caught a real `TypeError` on a 0.0 budget and pushed `EstimatorResult`
  to a Pydantic model; both fixed.

## Phase 4 -- wire + tune (PRs #233, #234, #235)

- **S10 front door (#215)** -- `seed_node` turns an NL mission into a valid
  `DesignState` (constraints + 17-var joint design space) via `MissionDecomposer` +
  `create_joint_design_space`; `research.decomposer.plan_to_constraints` maps a
  `MissionPlan` onto `DesignConstraints` fields. Verified: a mission-only state
  seeds constraints + a 17-var space and the loop runs to completion.
- **S11 CLI + skills (#216)** -- `branes session iterate [SESSION|--latest] [-n N]`
  loads the persisted `DesignState`, runs N real loop iterations (critic +
  specialists -> optimizer -> evaluate over the real MOO engine, safe-wrapped),
  saves the mutated state, and prints the trace. Verified end-to-end with the real
  MAP-Elites engine (returns a real joint-space design point). `architect-loop.md`
  rewritten to invoke it; `architect-drill`/`architect-assess` read `open_issues`.
- **S12 convergence tuning (#217)** -- single `has_converged()`: empty backlog OR
  critic diminishing-returns OR a *relative, windowed* hypervolume plateau (a lone
  flat step no longer prematurely stops a still-improving search). The critic emits
  a `diminishing_returns` judgment when the levers are exhausted.

## Outcome

Epic `#203` closed -- all 12 seams merged. The system is now one multi-agent loop
over a unified `DesignState`: reasoning critic + specialist agents exchanging typed
issues/deltas, a real MOO-engine tool with monotonic Pareto accumulation,
research-grounded S3-validated edits, pipeline-consistent evaluation, an
observable/steerable trace, and a real `branes session iterate` entry point.

## Follow-ups (non-blocking)

- **Two pre-existing flaky tests** surfaced repeatedly and deserve their own fix:
  `tests/agents/test_deployment.py::test_preprocessing_imagenet` (random-data
  assertion) and `tests/test_session_store.py::test_load_latest` (st_mtime
  tiebreak).
- **The demo's `surrogate_moo`/`score_knee` are illustrative**; the real
  `make_moo_engine_tool()` + `ppa_assessor` are wired (S11 runs them). A
  productionized `branes` demo command with an LLM critic is the natural next
  polish.
- **The `optimization_loop` `refinements` Pydantic-validation** hardening is still
  tracked on #209 (a different loop/file).
