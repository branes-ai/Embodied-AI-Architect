# Session 2026-07-15 -- Loop Convergence epic: Phase 2 (agent critic + optimizer)

**Repo:** `branes-ai/embodied-ai-architect`
**Epic:** `#203` (Loop Convergence) -- Phase 2 complete
**Landed on `main`:** PRs #225, #226, #227, #228 (v1.7.0 -> v1.10.0)

## Context

Phase 1 (2026-07-14) unified the two disjoint design loops onto one
`DesignState` and shipped a DRAFT critic/optimizer skeleton whose reasoning was
stubbed. Phase 2 makes the critic real and hardens the loop's tool boundaries.

## What landed

### S4 -- LLM critic path (#209, PR #225)

`Critic._review_with_llm` was a `NotImplementedError`. It now assembles a prompt
from PPA verdicts + top Pareto designs + the `open_issues` backlog + retrieved
research (`CRITIC_SYSTEM_PROMPT` pins a strict JSON schema and the per-`DeltaKind`
payloads), and parses the response into a `CriticVerdict` of typed `DesignIssue`s
and `DesignDelta`s linked by `addresses_issue` index. It is robust to model slop:
a delta whose `change` fails per-kind S3 validation (or an unknown `DeltaKind`) is
skipped, unknown metric/level/severity strings fall back, and any client/parse
error drops to the deterministic heuristic. Two CodeRabbit-caught bugs were fixed
in review: delta<->issue links were resolved through the *original* LLM array
index (a skipped issue had been compacting the list and misaligning links), and
`converged` is coerced from JSON (the string `"false"` is falsey) and refused
while any verdict is `FAIL`.

### S5 -- research-grounded deltas (#210, PR #226)

`DesignDelta.research_refs` lets deltas cite research. `_research_tags_for_state`
derives retrieval tags from the failing metrics / open issues via a per-`MetricAxis`
tag map (made exhaustive after review), so a power-bound drone and a latency-bound
AMR retrieve different research -- verified against the real library (distinct 8.5k
vs 7.9k-char context blocks). Per-delta `research_refs` are parsed from the verdict.

### S6 -- monotonic Pareto merge (#211, PR #227)

The loop MOO tool's `_merge_pareto` was a concatenation stub; it now reuses
`moo.specialist._merge_pareto_frontiers`. Dominated points are dropped and no
non-dominated point is lost across iterations. Points flow through the ParetoPoint
format for dominance and are recovered from `metadata`, so the loop keeps its
engine-native (nested `objectives`) shape.

### S7 -- pipeline-consistent evaluation (#212, PR #228)

The loop's `evaluate_node` had its own constraint check (keyed `power_watts`/...).
It now delegates to the real `ppa_assessor` -- the same assessor the dispatcher
pipeline uses -- so its `ppa_metrics.verdicts` (keyed `power`/`latency`/...) equal
the standalone assessor's for the same state, by construction.

## Test & review

Ran the existing **known-target test loop** to confirm Phase-1 state unification
did not regress the shipped pipeline:

- `tests/test_demo_acceptance.py` runs the SoC dispatcher graph for **7 gold
  standards** (`graphs/gold_standards.py`) -- each with an expected task graph,
  expected PPA (power/latency/cost), expected tool calls, iteration/duration
  bounds, and rationale keywords -- scored by `AgenticEvaluator`
  (`composite_score` threshold).
- `tests/test_golden_traces.py` covers `RunTrace` save/load + `compare_traces`
  regression detection (task-graph match, PPA regression, iteration regression).
- **Result: 27 passed.** The dispatcher loop -- which S2b/S2c migrated to
  `DesignState` -- matches its known targets.

**Gap:** the new unified `loop_convergence_graph` (the Phase-2 critic/optimizer
loop) has unit + smoke coverage but **no gold-standard end-to-end with known
outputs yet** -- it is still DRAFT/unwired and LLM-driven (non-deterministic
without a stubbed client). A deterministic acceptance loop for it (stubbed LLM +
fake MOO tool + known verdict targets) is the natural companion to S11 (wiring the
architect skills).

## Notes / follow-ups

- **Phase 3-4 remain** (the bigger, design-heavier stretch): S8 #214 (retask
  execution in the dispatcher), S9 #213 (promote specialists to reasoning agents
  with estimator tools), S10 #215 (decompose/formulate front door), S11 #216
  (wire architect skills to the code loop), S12 #217 (convergence tuning).
- **Loop-topology note (pre-existing):** without a seeded `ppa_metrics`, the
  unified loop converges at iteration 0 -- the graph runs `critic` before the
  first `evaluate` (seed -> critic -> optimize -> evaluate -> critic). Belongs
  with the front-door wiring (S10), not S7.
- The `optimization_loop` `refinements` Pydantic-validation hardening remains
  tracked on #209 (different file/loop; the loop_agents critic emits typed,
  S3-validated deltas already).
