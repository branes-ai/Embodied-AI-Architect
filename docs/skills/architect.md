# Architect Skills

All three skills are now live in Claude Code — you can see them in the skill list above:

- /architect-loop — Run one iteration of bottleneck-hunting
- /architect-assess — Generate multi-level metrics dashboard
- /architect-drill <target> — Deep-dive on a specific bottleneck

## What was built

Five Workflow Scenarios (docs/plans/architect-workflows.md)

All five follow the same expert loop but highlight different dominant constraints:


|      Workflow      |         Platform         |     Dominant Constraint      |                         Key Insight                         |
|--------------------|--------------------------|------------------------------|-------------------------------------------------------------|
| Recon drone        | Fixed-wing long-duration | Power (SAR processing)       | Multi-session campaign, bottleneck shifts after each fix    |
| Warehouse AMR      | Indoor fleet (10K units) | Cost/unit (NRE amortization) | Volume economics drive COTS vs custom                       |
| Surgical cobot     | 7-DOF arm                | Safety (SIL-4, WCET)         | Dual nervous system, certification constrains MCU selection |
| Agricultural drone | Multi-spectral survey    | Storage bandwidth            | Data rate vs. write speed, edge vs. cloud trade-off         |
| Racing FPV         | Autonomous racer         | p99 latency (<15ms total)    | Jitter matters more than average, pipeline depth trade-offs |


The recon drone workflow is detailed across 5 sessions showing how the architect works over time: 

```text
characterize → find SAR bottleneck → explore burst-mode → re-evaluate → new bottleneck (thermal) → iterate.
```

## Three Composable Claude Code Skills

The skills implement the expert cognitive loop at three granularities:

```text
  **/architect-assess** — "Where am I?" Shows metrics at every level with utilization bars and headroom indicators (green/yellow/red).

  **/architect-loop** — "What should I do next?" One full iteration: assess → rank top 3 bottlenecks → drill into #1 → propose 3-5 options with side effects →
  situation report.

  **/architect-drill <target>** — "Why is this a bottleneck?" Deep-dive with kernel breakdown, bandwidth analysis, or cost decomposition depending on what's
  bound.
```

The skills compose naturally: /architect-assess to orient, /architect-loop to iterate, /architect-drill to go deep when needed.

## Claude Code with Skills vs Embodied AI Architect with Orchestrator Loop

The skills makes Claude Code the embodied ai architect. How does this now relate to the Agentic loop we have been building with LangGraph?

That's the key architectural question. Right now there are two separate systems that both claim to be "the architect":

1. Claude Code + skills (/architect-loop, /architect-assess, /architect-drill) — Claude reads code, runs branes CLI commands, synthesizes results, and
presents analysis. The human directs every step. Claude is the analyst, the human is the architect.
2. LangGraph pipeline (planner → review → dispatch → optimize → report) — An autonomous agent loop with specialist nodes that executes a task graph, runs
PPA assessment, applies optimization strategies, and generates reports. The human reviews at checkpoints.

These are not connected. The skills don't read LangGraph state. The LangGraph pipeline doesn't know about the skills. They operate on the same codebase but
share no runtime state.

Here's what the relationship should be:

```text
┌─────────────────────────────────────────────────────────────┐
│  Claude Code + Skills (the human-facing interface)          │
│                                                             │
│  /architect-assess  →  reads LangGraph state, renders it    │
│  /architect-loop    →  runs one LangGraph iteration,        │
│                        analyzes results, proposes options   │
│  /architect-drill   →  calls branes CLI for deep analysis   │
│                        of a specific bottleneck             │
│                                                             │
│  The skills are the STEERING LAYER over the agentic loop    │
└──────────────────────────┬──────────────────────────────────┘
                           │ reads/writes
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  LangGraph Pipeline (the execution engine)                  │
│                                                             │
│  qualify → plan → [review] → dispatch → evaluate →          │
│            [optimize with steering] → report                │
│                                                             │
│  The pipeline is the EXECUTION LAYER — it runs specialists, │
│  tracks state, manages the task graph, applies strategies   │
└──────────────────────────┬──────────────────────────────────┘
                           │ calls
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Specialist Agents + branes CLI (the computation layer)     │
│                                                             │
│  workload_analyzer, hw_explorer, ppa_assessor,              │
│  design_optimizer, kpu_configurator, etc.                   │
│                                                             │
│  These do the actual work — PPA estimation, HW scoring,     │
│  bandwidth validation, cost modeling                        │
└─────────────────────────────────────────────────────────────┘
```

The skills should be the human-readable steering interface over the LangGraph execution engine. /architect-loop doesn't do its own ad-hoc analysis — it
runs one step of the LangGraph pipeline (via SoCDesignRunner.step()), reads the optimization_review_snapshot that the pipeline produces, synthesizes it
into the bottleneck ranking format, and presents options that map to OptimizationSteeringInput decisions.

The connection point already exists — it's the review.py and optimization_review.py modules we built. They produce exactly the snapshots that the skills
need to render. The missing piece is that the skills don't yet invoke SoCDesignRunner — they assume the human will separately run branes commands.

This is what needs to happen to unify them:

1. /architect-assess reads SoCDesignState (from a running or saved session) and renders optimization_review_snapshot in the dashboard format
2. /architect-loop calls runner.step() to advance the pipeline, reads the new snapshot, ranks bottlenecks from constraint_slackness, and maps proposed
options to steer_optimization inputs
3. /architect-drill calls the appropriate branes mcp or specialist agent for deep analysis of a named target from the current state

The skills become the human-friendly wrapper around the same LangGraph state machine. Not a parallel system — the steering layer over the execution engine.

The main work is making the skills read/write SoCDesignState from a persistent session file rather than doing ad-hoc CLI analysis.

## Workflow Architecture

Here's the architecture that's now wired:

  ┌──────────────────────────────────────────────────────────┐
  │  Claude Code + Skills                                    │
  │                                                          │
  │  /architect-assess ─┐                                    │
  │  /architect-loop  ──┤── branes session show --latest ──┐ │
  │  /architect-drill ──┘                    --json        │ │
  │                                                        │ │
  └────────────────────────────────────────────────────────┘ │
                                                             │
                      reads JSON from ~/.embodied-ai/sessions/
                                                             │
  ┌────────────────────────────────────────────────────────┘ │
  │   Session Store (auto-save after every step)             │
  │   ~/.embodied-ai/sessions/soc_<id>.json                  │
  └──────────────────────────────────────────────────────────┘
           ▲ writes                    ▲ writes
           │                           │
  ┌────────┴───────────┐    ┌──────────┴──────────┐
  │  SoCDesignRunner   │    │  SoCDesignRunner    │
  │  .run() (batch)    │    │  .start()/.step()   │
  │                    │    │  (interactive)      │
  └────────────────────┘    └─────────────────────┘
           │                           │
           ▼                           ▼
  ┌─────────────────────────────────────────────────────────┐
  │  LangGraph Pipeline                                     │
  │  qualify → plan → review → dispatch →                   │
  │              → evaluate → optimize → report             │
  └─────────────────────────────────────────────────────────┘

The skills now read the same state that the LangGraph pipeline writes. Run a design pipeline → session auto-saves → /architect-assess reads it → presents
the dashboard → /architect-loop identifies bottlenecks → architect decides → pipeline resumes.

