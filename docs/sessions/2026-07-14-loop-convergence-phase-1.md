# Session 2026-07-14 -- Loop Convergence epic: Phase 1 (state unification)

**Repo:** `branes-ai/embodied-ai-architect`
**Epic:** `#203` (Loop Convergence) -- Phase 1 complete
**Landed on `main`:** PRs #218, #219, #220, #221, #222, #223 (v1.2.0 -> v1.6.0)

## Context

The project had a real closed-loop optimization pipeline, but it was **two
disjoint loops with two state schemas** that never shared design state:

- a population/Pareto **MOO loop** (`graphs/optimization_loop.py`, keyed on
  `OptimizationLoopState`), and
- a single-design **dispatcher loop** (`graphs/soc_graph.py` / `dispatcher.py`,
  keyed on `SoCDesignState`).

The dispatcher's "specialists" were deterministic estimator functions, and the
critic's feedback was coarse ("narrow the search"), never concrete design-space
edits. This session created the **Loop Convergence** milestone to fix that, and
executed all of **Phase 1: unify onto one `DesignState`**.

## What landed

### Planning + skeleton (PR #218)

- `docs/plans/roadmap-loop-convergence.md`: 4 phases, seam checklist S1-S12,
  critical path, risks; wired into `roadmap-v2.md` (risk row, dependency lines,
  graph node, critical path, staffing).
- DRAFT import-only skeleton:
  - `graphs/design_state.py` -- unified `DesignState` + `DesignIssue` /
    `DesignDelta` (the structured currency between a reasoning Critic and
    Optimizer) + lifecycle helpers.
  - `graphs/loop_agents.py` -- `Critic` / `Optimizer` agents (LLM-plus-heuristic
    fallback), node wrappers, single router. MOO engine injected as a **tool**.
  - `graphs/loop_convergence_graph.py` -- `StateGraph` assembly + `MooTool`
    adapter over the real `OptimizationEngine`.

### S1 -- channel audit (#204, PR #219)

A LangGraph `TypedDict` schema is a **channel allowlist**: any key a node
returns that isn't a declared channel is silently dropped at runtime (no error).
Added `declared_channels()` / `undeclared_keys()` / `assert_declared_channels()`
and `tests/test_design_state_channels.py` which drives each loop node and asserts
it writes only declared channels. (A CodeRabbit finding here made the
duplicate-channel test parse the class AST instead of `__annotations__`, which is
already de-duplicated.)

### S2a -- migrate the MOO loop (#205, PR #220)

`optimization_loop.py` now flows through `DesignState`; `OptimizationLoopState`
deleted. Computed the exact gap first -- 24 of 31 fields were already channels,
only 7 MOO-search fields needed adding. `TestChannelCompliance` (with a
stubbed + `caplog`-guarded LLM-branch case) proves no node drops a write.

### S2b -- migrate the dispatcher loop (#206, PR #221)

`soc_graph.py` / `dispatcher.py` / `specialists.py` bind
`StateGraph(DesignState)`; declared the 18 remaining `SoCDesignState`-only
channels so `DesignState` is a **superset** -- the invariant that makes the
rebinding safe (guarded by `test_designstate_is_superset_of_socdesignstate`).
Fixed two **pre-existing** `dispatch_node` forwarding bugs the migration
surfaced: a phantom `swap_results` (vs the declared `swap_assessment`) and four
dropped specialist channels.

### S2c -- delete the legacy schema (#207, PR #222)

`SoCDesignState` is referenced across ~30 files. A scripted mass-rename went
badly (mangled signatures), so reverted and chose the **compat-alias** route the
issue title calls for: delete the `SoCDesignState` TypedDict class, keep the name
resolvable via a lazy module `__getattr__` returning `DesignState`. Lazy import
avoids the runtime cycle a direct alias would create (`design_state` imports the
base models defined in `soc_state`). One file changed, -71 lines.

### S3 -- typed DesignDelta payloads (#208, PR #223)

Replaced the free-form `DesignDelta.change: dict` with a payload model per
`DeltaKind`, validated at construction via a `model_validator`; malformed edits
now raise immediately instead of failing later in `_apply_delta`, which now
consumes typed fields. (CodeRabbit caught that a payload extra named
`specialist` could override `delta.target` -- fixed by spreading the payload
first.)

## Outcome

- **One state schema:** `graphs/design_state.py:DesignState` (~92 channels).
- Both loops and all specialists flow through it; the legacy `SoCDesignState` /
  `OptimizationLoopState` classes are gone.
- Full test suite green in CI (the loop skeleton remains DRAFT / unwired).

## Notes / follow-ups

- **Phase 2-4 (S4-S12) remain** -- the actual agent implementation:
  `Critic._review_with_llm` (S4 #209), research-grounded deltas, real
  `_merge_pareto`, real `evaluate_node` via `ppa_assessor`, specialist agents,
  the decompose front door, wiring the architect skills, convergence tuning.
- **Two pre-existing flaky tests** surfaced (unrelated to this work):
  `test_session_store::test_load_latest` (st_mtime tiebreak) and
  `test_verdict_tools_cli::TestVerdictToolsLive` (live Anthropic calls to the
  retired model id `claude-sonnet-4-20250514`; skip in CI, run locally with a
  key set).
- **Process:** CI lint pinned to `black==25.12.0` / `ruff==0.14.10` to stop
  version drift; every PR went draft -> CodeRabbit review -> resolve ->
  squash-merge.
