# Claude Code Developer Guide for Embodied AI Architect

This document describes the Claude Code (CC) configuration for the
`embodied-ai-architect` repository: what it provides, why each piece exists,
how the parts fit together, and practical examples of using them to move faster.

---

## What's Configured

The CC environment for this repo consists of five layers:

| Layer | File(s) | Purpose |
|-------|---------|---------|
| **Project guide** | `CLAUDE.md` | Teaches Claude the codebase architecture, CLI commands, code style, and commit conventions so it starts every session with accurate context |
| **Hooks** | `.claude/settings.local.json` (`hooks`) | Automated checks that run *around* tool calls — before pushes, after file edits, and before commits |
| **Skills** | `.claude/commands/*.md` | One-shot slash commands that bundle multi-step workflows into a single `/command` |
| **MCP server** | `.claude/settings.local.json` (`mcpServers`) | Connects the branes analysis engine as native Claude tools — SWaP-C estimates, BOM, thermal checks, TOPSIS ranking — without shelling out |
| **Subagents** | Built-in (Explore, Plan, general-purpose) | Parallel agents Claude can spawn for deep codebase research, architecture planning, or independent tasks |

---

## Why This Configuration Exists

### The problems it solves

1. **Stale context at session start.** Without `CLAUDE.md`, Claude doesn't know the
   CLI is called `branes`, that all tools live under `.venv/bin/`, that the graphs
   subsystem is 60+ files, or that there are 19 CLI command groups. It guesses wrong,
   writes commands that fail, and wastes turns recovering.

2. **Lint failures discovered too late.** Without hooks, a developer (or Claude)
   writes code, pushes, waits for CI, discovers a ruff violation, fixes it, pushes
   again. The pre-push hook and post-edit auto-lint hook collapse that loop to zero
   round-trips.

3. **Repetitive multi-step workflows.** Running the quality gate means remembering
   three commands in the right order. Running SWaP-C smoke tests means four commands
   with specific flags. Skills turn these into one-word invocations.

4. **Context-expensive analysis.** Shelling out to `branes swap estimate ...` and
   parsing CLI output burns context tokens and requires Claude to format arguments
   as strings. The MCP server exposes the same functions as structured tool calls
   with typed inputs and outputs.

5. **Large codebase navigation.** With 190+ source files across three repos, a naive
   grep sweep can flood context. Subagents (Explore, Plan) run in isolated context
   windows and return only the relevant findings.

---

## Layer-by-Layer: How It Works

### 1. CLAUDE.md — The Project Guide

**What:** A markdown file at the repo root that Claude Code reads at the start of
every session. It contains:

- Correct CLI name (`branes`), venv paths, Python version (3.11+)
- All 19 CLI command groups, organized by category
- Full architecture map: orchestrator, agents, graphs subsystem (MOO, SWaP-C, KPU, RTL, EDA), deployment targets, specs system
- Commit convention (conventional commits)
- Related repository structure (embodied-schemas, graphs)

**Why:** Claude's training data doesn't include this repo. Without `CLAUDE.md`,
it hallucinates file paths, uses the wrong CLI name, and misses entire subsystems
when planning changes. The guide acts as a compressed, authoritative map.

**How it helps:** When you say "add a new SWaP-C scoring method," Claude already
knows that scoring lives in `graphs/scoring.py`, profiles are in
`graphs/swap_profiles.py`, the CLI entry point is `cli/commands/swap.py`, and tests
follow `tests/test_swap_*.py`. It doesn't need to spend 5 turns searching.

### 2. Hooks — Automated Guardrails

Three hooks are configured in `.claude/settings.local.json`:

#### a) Pre-push quality gate (`PreToolUse` on `git push`)

```
Trigger: Any Bash command containing "git push"
Action:  Run black --check + ruff check before allowing the push
Effect:  If lint fails, the push is blocked and Claude sees the errors
```

This is the single highest-value hook. It prevents the push-wait-fail-fix-push
cycle that wastes 5-10 minutes per iteration.

#### b) Conventional commit enforcement (`PreToolUse` on `git commit`)

```
Trigger: Any Bash command containing "git commit"
Action:  Extract the -m message and validate it matches type(scope): description
Effect:  Non-conforming commits are blocked with an explanatory error
```

Ensures every commit Claude creates follows the project's conventional commit
format, which drives semantic-release versioning.

#### c) Auto-lint on file write (`PostToolUse` on `Edit|Write`)

```
Trigger: After any Edit or Write tool call
Action:  If the file is *.py, run ruff check on it
Effect:  Lint errors appear immediately, not at push time
```

This gives Claude instant feedback on every Python file it touches. Lint errors
are fixed in the same turn they're introduced, not discovered 20 turns later.

### 3. Skills — Slash Commands

Skills are markdown files in `.claude/commands/` that define multi-step workflows
invoked with `/command-name`.

| Skill | File | What it does |
|-------|------|-------------|
| `/quality-gate` | `quality-gate.md` | Runs black + ruff + pytest — the full pre-push checklist |
| `/smoke-swap` | `smoke-swap.md` | Runs SWaP-C unit tests + CLI smoke commands (estimate, score, sensitivity) |
| `/cross-repo-check` | `cross-repo-check.md` | Validates that embodied-schemas and graphs are importable, then runs the full test suite |

Skills are different from hooks: hooks run automatically and silently, while skills
run on demand and produce visible output. Use a skill when you want Claude to
actively report results and fix problems.

### 4. MCP Server — Branes as Native Tools

The branes MCP server (`embodied_ai_architect.mcp.server`) is registered in
`settings.local.json`:

```json
"mcpServers": {
  "branes": {
    "command": ".venv/bin/python",
    "args": ["-m", "embodied_ai_architect.mcp.server"]
  }
}
```

This exposes the project's analysis engine as structured tool calls that Claude
can invoke directly, without constructing shell commands and parsing text output.
Available tools include SWaP-C estimation, BOM calculation, thermal feasibility
checks, and TOPSIS multi-criteria ranking.

**Why MCP instead of CLI?** MCP tools have typed inputs and structured JSON
outputs. Claude doesn't need to remember CLI flag syntax or parse Rich-formatted
console tables. The feedback loop is tighter and less error-prone.

### 5. Subagents — Parallel Research and Planning

Claude Code can spawn specialized subagents that run in isolated context windows:

| Agent | Use case |
|-------|----------|
| **Explore** | Fast codebase search — find files, grep for patterns, answer "where does X live?" |
| **Plan** | Architecture planning — design an implementation strategy before writing code |
| **general-purpose** | Complex multi-step research — web searches, deep code analysis, multi-file investigation |

Subagents are valuable because they don't pollute the main conversation's context
window. A search that returns 200 lines of grep output stays contained in the
subagent; only the summarized answer comes back.

---

## Practical Examples

### Example 1: Adding a New SWaP-C Methodology

**Scenario:** You want to add a `branes swap budget` command that allocates a
total power budget across subsystems.

**Workflow using the full CC stack:**

1. **Claude reads `CLAUDE.md`** at session start and already knows:
   - SWaP-C code lives in `graphs/swap_analysis.py`, `graphs/swap_profiles.py`
   - CLI entry point is `cli/commands/swap.py`
   - Tests follow `tests/test_swap_*.py`

2. **You describe the feature.** Claude enters Plan mode (subagent) to design the
   implementation across the relevant files.

3. **Claude writes the code.** After each file edit, the **auto-lint hook** runs
   `ruff check` on the modified `.py` file. If there's an unused import or a
   line-length violation, Claude sees it immediately and fixes it in the same turn.

4. **You type `/smoke-swap`.** Claude runs all SWaP-C tests plus the CLI smoke
   commands to verify nothing broke.

5. **Claude commits.** The **conventional commit hook** validates the message
   format: `feat(swap): add power budget allocation command`.

6. **Claude pushes.** The **pre-push quality gate hook** runs black + ruff across
   the entire `src/` and `tests/` tree. If anything slipped through, the push is
   blocked and Claude fixes it before retrying.

**Time saved:** The hooks eliminate the push-fail-fix cycle (5-10 min per round).
The skill eliminates manually typing four test commands. CLAUDE.md eliminates
3-5 turns of "where does SWaP-C code live?" exploration.

---

### Example 2: Investigating a Cross-Repo Import Failure

**Scenario:** After updating `embodied-schemas`, tests in this repo start failing
with `ImportError: cannot import name 'BenchmarkResult'`.

**Workflow:**

1. **You type `/cross-repo-check`.** Claude runs the three-step validation:
   - Checks if `embodied-schemas` is importable (finds the error)
   - Checks if `graphs` is importable
   - Runs the full test suite

2. **Claude sees the import error** and spawns an **Explore subagent** to search
   both repos in parallel:
   - One search in `../embodied-schemas` for the current `BenchmarkResult` definition
   - One search in this repo for all `from embodied_schemas import BenchmarkResult` usages

3. The subagent reports back: `BenchmarkResult` was renamed to `BenchmarkVerdict`
   in the schemas repo. Claude knows from `CLAUDE.md` that "verdict-first output
   schema" is the design pattern, so it updates all imports.

4. **Auto-lint hook** catches any remaining import issues as files are edited.

5. **You type `/quality-gate`** to confirm everything passes before pushing.

**Time saved:** The `/cross-repo-check` skill immediately pinpoints which repo
boundary broke. The Explore subagent searches two repos in parallel without
flooding the main context. Without these, you'd be manually running imports and
grepping across three repos.

---

### Example 3: Designing a New Deployment Target

**Scenario:** You want to add an FPGA deployment target for Xilinx Zynq.

**Workflow:**

1. **Claude reads `CLAUDE.md`** and knows the deployment target interface:
   - All targets extend the base class in `agents/deployment/targets/base.py`
   - Existing targets: Jetson, Coral, OpenVINO, KPU, NVDLA
   - Each has a `deploy()` method and power monitoring

2. **Claude spawns a Plan subagent** that:
   - Reads `base.py` to understand the `DeploymentTarget` ABC
   - Reads an existing target (e.g., `jetson.py`) as a template
   - Reads `graphs/technology.py` for Xilinx process node parameters
   - Produces a step-by-step implementation plan

3. **You approve the plan.** Claude writes `agents/deployment/targets/fpga.py`,
   registers it in `__init__.py`, and adds tests.

4. **The auto-lint hook** validates each file as it's written — no ruff errors
   accumulate.

5. **Claude uses the MCP server** to run a quick SWaP-C estimate for the Zynq
   target, calling `estimate_system_bom` with structured parameters instead of
   constructing a CLI command string. The result comes back as typed JSON that
   Claude can reason about directly.

6. **You type `/quality-gate`.** Full suite passes. Claude commits with
   `feat(deploy): add Xilinx Zynq FPGA deployment target` — validated by the
   conventional commit hook — and pushes, gated by the pre-push lint check.

**Time saved:** The Plan subagent produces a coherent multi-file design in one
pass instead of Claude discovering the interface incrementally. The MCP server
gives structured analysis results without CLI parsing. The hooks prevent any
quality regressions from reaching CI.

---

### Example 4: Refactoring the MOO Engine

**Scenario:** You want to extract the MAP-Elites grid storage into a separate
module for reuse.

**Workflow:**

1. **Claude reads `CLAUDE.md`** and knows the MOO pipeline structure:
   - 3-layer architecture in `graphs/moo/`
   - `map_elites.py` is Layer 1, `engine.py` orchestrates

2. **Claude spawns an Explore subagent** with the prompt "find all references to
   MAP-Elites grid storage across the codebase." The subagent searches `graphs/moo/`,
   tests, and any CLI commands that reference the grid, returning a concise list
   of touchpoints.

3. **Claude writes the refactored code.** Every `.py` file triggers the
   **auto-lint hook** — immediate feedback on imports, unused variables, line length.

4. **You type `/quality-gate`.** Tests pass, lint is clean.

5. **Claude commits:** `refactor(moo): extract MAP-Elites grid into reusable module`.
   The **commit hook** confirms the format. The **push hook** runs the full lint
   suite one final time.

**Time saved:** The Explore subagent found all grid references without Claude
reading every file in `moo/` into the main context. The auto-lint hook caught
broken imports from the refactor immediately, not in a batch at the end.

---

### Example 5: Using MCP for Interactive Design Exploration

**Scenario:** You're iterating on a drone SoC design and want to quickly compare
thermal feasibility across process nodes.

**Workflow:**

1. **You ask Claude** to compare 28nm vs 16nm vs 7nm for a 50mm^2, 5W design.

2. **Claude calls the MCP server tools directly** — three parallel calls to
   `estimate_system_bom` with different process parameters. No shell commands,
   no CLI output parsing. Each call returns structured JSON with area, power,
   thermal, and cost breakdowns.

3. **Claude calls `topsis_rank`** via MCP to rank the three options against your
   weighted criteria (power 40%, cost 30%, area 20%, thermal 10%).

4. **Claude presents a comparison table** synthesized from the structured tool
   outputs, with a recommendation and trade-off analysis.

**Time saved:** Three CLI invocations would require constructing flag strings,
capturing stdout, parsing Rich tables, and reassembling the data. MCP gives
typed inputs and JSON outputs — Claude reasons about the data directly. The
entire exploration takes one conversational turn instead of six.

---

## Quick Reference

| I want to... | Do this |
|--------------|---------|
| Run the full lint + test suite | Type `/quality-gate` |
| Smoke-test SWaP-C after changes | Type `/smoke-swap` |
| Check cross-repo health | Type `/cross-repo-check` |
| Push code | Just push — the hook runs lint automatically |
| Commit code | Just commit — the hook validates the message format |
| Run a SWaP-C estimate interactively | Ask Claude (MCP tools are available as native calls) |
| Find where something lives in the codebase | Ask Claude (it spawns an Explore subagent) |
| Plan a multi-file change | Ask Claude to plan (it spawns a Plan subagent) |

---

## File Inventory

```
CLAUDE.md                              # Project guide (read at session start)
.claude/settings.local.json            # Hooks, permissions, MCP server config
.claude/commands/quality-gate.md       # /quality-gate skill
.claude/commands/smoke-swap.md         # /smoke-swap skill
.claude/commands/cross-repo-check.md   # /cross-repo-check skill
```
