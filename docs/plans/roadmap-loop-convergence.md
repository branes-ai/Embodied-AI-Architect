# Roadmap — Loop Convergence

**Created:** 2026-07-09
**Status:** Draft
**Owner:** Architect team
**Depends on:** `roadmap-v2.md` R0.8 (Optimization Loop), `agentic-optimization-loop.md` (Phases 1–4, all landed)
**Slots between:** roadmap-v2 R0.8 → R0.9

**Goal:** Turn the two existing, disjoint closed loops into a single unified multi-agent
design loop where reasoning specialists, a critic, and an optimizer iterate over **one
evolving design state** — the difference between "LLM-supervised search plus a
human-driven CLI skill loop" (today) and "a proper loop-engineering multi-agent design
automation system" (target).

---

## Why This Release Exists

Phases 1–3 of `agentic-optimization-loop.md` are implemented and Phase 4's loop runs.
But the system is **not yet one loop**. A code audit (2026-07-09) found three structural
gaps that no current roadmap milestone addresses:

1. **Two disjoint loops with two state schemas.**
   - Population/Pareto MOO loop: `graphs/optimization_loop.py` over `OptimizationLoopState`
     (decompose → formulate → optimize → evaluate → reason → {iterate | recommend}).
   - Single-design dispatcher loop: `graphs/soc_graph.py` / `graphs/dispatcher.py` over
     `SoCDesignState` (planner → dispatch → evaluate → {optimize → dispatch | report}).
   - Neither shares design state with the other; the architect skills only target the latter.

2. **The "specialists" are not agents.** In the dispatcher loop they are deterministic
   estimator functions (`graphs/specialists.py`, `graphs/physical_estimators.py`), not
   reasoning agents negotiating over shared state. Only the Phase-4 `reason_node` uses an LLM.

3. **Feedback is shallow.** When the critic (`reason_node`) says "iterate," it emits coarse
   moves (narrow the search, tighten a budget) and `optimize_node` just re-runs MAP-Elites
   with iteration-scaled effort (`optimization_loop.py:352`). It does not translate insight
   into concrete design-space edits or re-task specific specialists.

4. **The best "loop" UX is human-in-the-CLI.** `architect-loop`, `architect-drill`, and
   `architect-assess` read a real `SoCDesignState` via `branes session show --json`, but they
   are markdown prompt-skills that shell out to the CLI — not code-invoked graph nodes.

**Business value:** This is the credibility milestone. It is what lets us say the agent
*designs*, not just *searches*. It is a prerequisite for the co-simulation trust story
(roadmap-v2 R0.9) because a unified, inspectable design state is what co-sim validates against.

---

## Target Architecture

```
                        ┌────────────────────────────────┐
                        │        Unified DesignState      │  ◄── one schema, one source of truth
                        │  (constraints, artifacts, PPA,  │
                        │   pareto, history, open_issues) │
                        └───────────────┬────────────────┘
                                        │  read / mutate
        ┌───────────────┬───────────────┼───────────────┬───────────────┐
        ▼               ▼               ▼               ▼               ▼
  ┌──────────┐   ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐
  │ Planner  │   │ Specialists│  │  Optimizer │  │   Critic   │  │  Architect │
  │ (agent)  │   │ (agents +  │  │ (MOO engine│  │  (agent)   │  │   skills   │
  │          │   │  estimator │  │  as a tool)│  │ emits      │  │ (CLI ->    │
  │          │   │  tools)    │  │            │  │ design     │  │  graph     │
  │          │   │            │  │            │  │ deltas)    │  │  iteration)│
  └────┬─────┘   └─────┬──────┘  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘
       └───────────────┴───────────────┴───────────────┴───────────────┘
                                        │
                        ┌───────────────▼────────────────┐
                        │      Convergence controller      │  ◄── one loop condition,
                        │  (hypervolume Δ, issue backlog,  │      not two
                        │   critic "diminishing returns")  │
                        └──────────────────────────────────┘
```

The MOO engine stops being *the* loop and becomes a **tool the optimizer agent calls**.
The physical estimators stop being *the* specialists and become **tools the specialist
agents call**. The two state schemas collapse into one `DesignState`.

---

## Phase 1 — Unify the State

**Goal:** One `DesignState` that both loops read and mutate; delete the schema seam.

| # | Task | Priority | Effort |
|---|------|----------|--------|
| 1 | Define `DesignState` superset that subsumes `SoCDesignState` + `OptimizationLoopState` (design point, joint design space, pareto_points, pareto_frontier_history, moo_results, ppa_metrics, kpu_config, `open_issues`, history) | P0 | L |
| 2 | Add `open_issues: list[DesignIssue]` — structured, typed bottleneck records (metric, level, contribution, severity, proposed action) — the shared currency between critic and optimizer | P0 | M |
| 3 | Migrate `optimization_loop.py` nodes to read/write `DesignState` (keep the graph shape) | P0 | M |
| 4 | Migrate `soc_graph.py` / `dispatcher.py` to the same `DesignState` | P0 | L |
| 5 | Adapter/compat shims so existing snapshot builders, `/api/sessions/*`, and `branes session show` keep working | P1 | M |

**Exit criteria:** A single `branes` session produces one `DesignState` that carries both the
Pareto frontier and the single-design dispatcher artifacts; both graphs run against it; all
existing MOO state-field consumers (per CLAUDE.md table) still resolve.

---

## Phase 2 — Agent-ify the Critic and the Optimizer

**Goal:** Replace coarse "iterate" with structured, actionable feedback.

| # | Task | Priority | Effort |
|---|------|----------|--------|
| 1 | Promote `reason_node` to a **Critic agent** that emits `DesignDelta` objects (design-space variable edits, constraint relaxations, specialist re-tasking) instead of a free-text "iterate" verdict | P0 | L |
| 2 | Promote `design_optimizer` to an **Optimizer agent** that consumes `DesignDelta`s, edits the joint design space concretely, and calls the MOO engine **as a tool** (not as the loop body) | P0 | L |
| 3 | Wire critic → `open_issues` → optimizer so each iteration closes specific issues rather than globally re-running MAP-Elites | P0 | M |
| 4 | Keep a deterministic fallback path (no LLM key) for both agents — mirror the existing `_reason_with_llm` / heuristic-fallback pattern (`optimization_loop.py:507,576`) | P1 | M |
| 5 | Cost + audit trail per agent turn (reuse existing token cost tracking + governance) | P1 | S |

**Exit criteria:** Given a design that fails a constraint, the Critic emits ≥1 concrete
`DesignDelta`, the Optimizer applies it as a specific design-space edit, and the next
iteration's Pareto front reflects that edit — verifiable in `open_issues` closing out.

---

## Phase 3 — Specialists as Agents with Estimator Tools

**Goal:** Turn 2–3 deterministic specialists into reasoning agents; keep the estimators as tools.

| # | Task | Priority | Effort |
|---|------|----------|--------|
| 1 | Wrap `physical_estimators.py` / `specialists.py` functions as **callable tools** (verdict-first schema) | P0 | M |
| 2 | Promote the **PPA/critic-adjacent specialist** and the **architecture composer** to agents that call those tools and reason over `open_issues` | P0 | L |
| 3 | Leave pure-numeric specialists (thermal, cost, floorplan, bandwidth) as deterministic tools — do **not** agent-ify what has no judgment content | P1 | S |
| 4 | KPU inner-loop specialists (`kpu_specialists.py`) stay deterministic but publish issues into the shared `open_issues` backlog | P2 | M |

**Exit criteria:** At least two specialists reason over shared state and call estimator tools;
their outputs land as `DesignIssue`s the critic and optimizer act on — no separate specialist
state schema remains.

---

## Phase 4 — Wire Architect Skills to the Code Loop

**Goal:** `architect-loop` triggers a real graph iteration instead of orchestrating CLI calls by hand.

| # | Task | Priority | Effort |
|---|------|----------|--------|
| 1 | Add `branes session iterate` (or equivalent) that runs one code-level iteration of the unified loop and mutates the persisted `DesignState` | P0 | M |
| 2 | Rewrite `architect-loop.md` to invoke that command and read back the new `open_issues` / Pareto delta — same bottleneck-hunting UX, real code path underneath | P0 | S |
| 3 | Point `architect-drill` at `DesignIssue` records; `architect-assess` at the unified multi-level metrics (`architect-workflows.md`) | P1 | M |
| 4 | Convergence controller: single loop condition (hypervolume Δ < ε for N iters **OR** empty `open_issues` **OR** critic "diminishing returns") replacing the two separate stop conditions | P1 | M |

**Exit criteria:** A user runs `architect-loop` and each pass is a genuine code iteration of the
unified loop — the critic finds bottlenecks, the optimizer/specialists resolve them against one
`DesignState`, and the loop halts on the shared convergence condition. The human is *steering*
the loop, not *being* it.

---

## Critical Path

```
Phase 1 (unify state) ──→ Phase 2 (agent critic+optimizer) ──→ Phase 4 (wire skills + convergence)
        └──────────────────→ Phase 3 (specialist agents) ──────┘
```

Phase 1 gates everything. Phases 2 and 3 can proceed in parallel once the state is unified.
Phase 4 needs both.

---

## Implementation Seams (from the executable skeleton)

A DRAFT skeleton exists and passes the Black+Ruff gate; the compiled graph drives a full
failing→converged cycle with a fake MOO tool. It is wired to nothing (import-only). The
remaining work is filling these named seams:

- `src/embodied_ai_architect/graphs/design_state.py` — unified `DesignState` + `DesignIssue` / `DesignDelta` + lifecycle helpers
- `src/embodied_ai_architect/graphs/loop_agents.py` — `Critic` / `Optimizer` agents + node wrappers + router
- `src/embodied_ai_architect/graphs/loop_convergence_graph.py` — graph assembly + `MooTool` engine adapter + endpoints

| # | Seam | Location | Phase | Status |
|---|------|----------|-------|--------|
| S1 | **Schema audit** — every field any node writes must be declared in `DesignState`, or LangGraph silently drops it at runtime (found `final_report` / `research_citations` / `pending_specialist_tasks` this way). Exhaustively reconcile all node writes against the schema before deleting the old schemas. | `design_state.py` | 1 | ☐ |
| S2 | **State migration + compat shims** — subsume `SoCDesignState` + `OptimizationLoopState`; keep existing `moo_results` / snapshot / API consumers (CLAUDE.md field table) resolving; run the full test suite as the gate. | `design_state.py`, `soc_state.py`, `optimization_loop.py` | 1 | ☐ |
| S3 | **`DesignDelta.change` discriminated union** — replace the free-form `dict` payload with a per-`DeltaKind` typed model so edits validate at the boundary. | `design_state.py` | 1 | ☐ |
| S4 | **LLM critic path** — `Critic._review_with_llm` is a documented `NotImplementedError`. Assemble prompt from PPA + Pareto + `open_issues` + retrieved research (reuse `optimization_loop._reason_with_llm` shape), require JSON, parse into `CriticVerdict`. | `loop_agents.py` | 2 | ☐ |
| S5 | **Research-grounded deltas** — `_default_delta_for` presets are crude per-metric guesses (power→int8, latency→more MAC rows). The LLM path should propose targeted deltas grounded in the research library. | `loop_agents.py` | 2 | ☐ |
| S6 | **Real `_merge_pareto`** — the accumulation stub in the MOO tool must reuse `moo.specialist._merge_pareto_frontiers` (monotonic; regression test for frontier monotonicity). | `loop_convergence_graph.py` | 2 | ☐ |
| S7 | **Real `evaluate_node`** — replace the deterministic constraint check with the actual `ppa_assessor` so verdicts match the rest of the pipeline. | `loop_convergence_graph.py` | 2 | ☐ |
| S8 | **Specialist retask execution** — `SPECIALIST_RETASK` deltas enqueue `pending_specialist_tasks`; the dispatcher must consume them and re-run the named specialist (issue #35-style re-validation after a config change). | `loop_agents.py`, `dispatcher.py` | 3 | ☐ |
| S9 | **Specialist agents + estimator tools** — wrap `physical_estimators.py` / `specialists.py` as verdict-first tools; promote the 2–3 judgment-bearing specialists to agents. | `specialists.py`, `physical_estimators.py` | 3 | ☐ |
| S10 | **Decompose/formulate front door** — `seed_node` stands in for the real mission → constraints → joint design space entry (reuse `MissionDecomposer` + `create_joint_design_space`). | `loop_convergence_graph.py`, `research/decomposer.py` | 4 | ☐ |
| S11 | **Wire architect skills to the loop** — add `branes session iterate` running one code-level loop iteration; rewrite `architect-loop.md` to invoke it instead of orchestrating CLI calls. | CLI, `.claude/commands/architect-loop.md` | 4 | ☐ |
| S12 | **Convergence tuning** — `has_converged` uses a fixed hypervolume epsilon; calibrate against real runs and add the critic's "diminishing returns" judgment as a third signal. | `design_state.py`, `loop_agents.py` | 4 | ☐ |

---

## Success Criteria (release-level)

| Metric | Target |
|--------|--------|
| One `DesignState` schema; the second schema deleted | Phase 1 |
| Critic emits structured `DesignDelta` / `DesignIssue`, not free text | Phase 2 |
| Optimizer applies critic deltas as concrete design-space edits | Phase 2 |
| ≥2 specialists reason over shared state via estimator tools | Phase 3 |
| `architect-loop` runs a real code iteration, not manual CLI orchestration | Phase 4 |
| Single convergence condition governs the unified loop | Phase 4 |
| End-to-end: NL mission → unified loop → design meeting all constraints, with an auditable trail of which issues each iteration closed | Release |

---

## Non-Goals

- New objectives or design-space dimensions (Phases 1–3 of `agentic-optimization-loop.md`
  already delivered these — this release restructures the loop, it does not widen it).
- Co-simulation validation (that is roadmap-v2 R0.9, and it consumes this release's output).
- Agent-ifying purely numeric specialists with no judgment content.
- Replacing the MOO engine — it becomes a tool, unchanged internally.

---

## Risks

| Risk | Mitigation |
|------|-----------|
| State unification breaks the many existing `moo_results` / snapshot / API consumers (CLAUDE.md field table) | Phase 1 compat shims + run the existing test suite as the gate before deleting the old schema |
| LLM-in-the-loop cost/latency per iteration | Deterministic fallback paths (Phase 2/4); cap agent turns per iteration; reuse token cost tracking |
| Over-agent-ifying deterministic estimators adds nondeterminism without value | Phase 3 explicitly keeps numeric specialists as tools; only judgment-bearing nodes become agents |
| Two-loop merge changes convergence behavior / regresses Pareto monotonicity | Keep `_merge_pareto_frontiers` monotonic-accumulation invariant; add a frontier-monotonicity regression test |
