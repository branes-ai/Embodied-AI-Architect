Run one iteration of the architect's bottleneck-hunting loop on the current design.

As of the Loop Convergence work (epic #203), this loop is a **real code path** — the
`critic`, the specialist agents (PPA, thermal), the `optimizer`, and `evaluate` — not
manual CLI orchestration. Your job is to invoke it and interpret the result for the
user, and to steer it when domain knowledge is needed.

## 1. Run one iteration

```bash
.venv/bin/branes session iterate --latest          # or: iterate <session_id> -n <N>
```

This runs the unified loop over the persisted `DesignState`:

- the **critic** + **specialist agents** file typed `DesignIssue`s for the bottlenecks
  (with research-grounded rationale when an LLM key is set),
- the **optimizer** applies concrete `DesignDelta` edits and re-runs the MOO engine
  over the 17-variable joint design space,
- **evaluate** re-scores the design via the same `ppa_assessor` the pipeline uses,
- the mutated state is **saved**, and the command prints the full reasoning trace —
  what each agent decided and *why*, step by step.

If there is no session, tell the user:
```
No active design session. Start one with:
  branes design qualify "your design goal"
  branes design plan "your qualified goal" --power X --latency Y
```

## 2. Read back what the loop did

```bash
.venv/bin/branes session show --latest --json
```

The bottleneck analysis is now structured data (you no longer hand-derive it):

- `open_issues` — the typed bottlenecks the critic + specialists filed: `metric`,
  `level` (system/subsystem/operator/kernel/hardware/physical), `severity`,
  `summary`, `observed_value` vs `target_value`, `contribution_pct`.
- `applied_deltas` — the concrete edits the optimizer applied: `kind`, `target`,
  `rationale`, and `research_refs` grounding them.
- `ppa_metrics.verdicts` — PASS/FAIL per constraint; `converged`; `iteration`.
- `optimization_review_snapshot.sensitivity` — MOO per-variable leverage (use it
  only to add color on which knob actually moves a metric).

## 3. Present a situation report

1. **Top bottlenecks** — from `open_issues`, ranked by `severity` then
   `contribution_pct`. For each: the metric, its level, the gap (observed vs target,
   % over budget), and why it's the dominant one.
2. **What the loop did** — from `applied_deltas` and the printed trace: which edits it
   applied, their rationale, any research it cited.
3. **Where it stands** — the verdicts, whether it `converged`, and iteration `N`. If it
   didn't converge, name what still fails and the loop's next lever.
4. **Recommendation / next step** — run another iteration, or steer.

```
ARCHITECT LOOP — iteration N

TOP BOTTLENECKS (from open_issues):
  1. [metric] at [level]: [observed] vs [target] (+[contribution_pct]%, [severity])
  2. ...
WHAT THE LOOP DID (from applied_deltas):
  - [kind] [target] :: [rationale]  (research: [research_refs])
STATUS: verdicts=[...], converged=[bool]
NEXT: [iterate again | steer]
```

## 4. Steer when the loop needs domain knowledge

The loop reports honestly when a lever is exhausted (e.g. INT8 alone can't hit a
tightened power budget). When that happens — or when the user wants a different
trade-off — steer it:

- **Relax/tighten a constraint** and re-run `session iterate`.
- **Inject a design move** the critic didn't propose (e.g. structured sparsity, a
  smaller process node) via the loop's steer hook — see
  `docs/loop-convergence-demo.md` for a worked example.

The old manual bottleneck-ranking is now the critic + specialists' job. You are the
interpreter and the human-in-the-loop steersman.
