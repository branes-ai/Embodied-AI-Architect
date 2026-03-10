# Plan: Dev Environment Improvements

## Context

The CLAUDE.md file has accumulated inaccuracies (wrong CLI name, missing `.venv` prefix,
wrong Python version, only 8 of 19 CLI commands documented) and is missing architecture
docs for the largest subsystems (graphs/, MOO, SWaP-C, deploy). No hooks or custom skills
exist. The branes MCP server could be connected for interactive analysis. These gaps
cause Claude Code to start every session with stale context, miss lint errors until
the end, and lack quick access to quality gates.

Implements the 9 items from `docs/dev-environment-improvements.md`.

---

## Items to Implement

### Item 1: Fix CLAUDE.md Inaccuracies (Priority 1)

**File:** `CLAUDE.md`

Changes:
- Replace all `embodied-ai` with `branes` in CLI examples (19 occurrences)
- Prefix all bare `pip`, `pytest`, `black`, `ruff` commands with `.venv/bin/`
- Change Python target from `3.9+` to `3.11+`
- Update CLI command list from 8 to 19 groups

### Item 2: Add graphs/ Architecture Section to CLAUDE.md (Priority 2)

**File:** `CLAUDE.md`

Add a new section after the existing "Core System" section covering:

- **`graphs/` computational core** — the largest subsystem (61+ files), organized as:
  - SoC design pipeline: `soc_state.py`, `soc_runner.py`, `soc_graph.py`, `specialists.py`
  - Multi-objective optimization (`moo/`): `engine.py` (3-layer pipeline), `map_elites.py`,
    `bayesian_opt.py`, `nsga3.py`, `design_space.py`, `evaluator.py`
  - SWaP-C analysis: `physical_estimators.py`, `bom.py`, `swap_report.py`,
    `swap_profiles.py`, `swap_analysis.py`
  - KPU design: `kpu_loop.py`, `kpu_config.py`, `kpu_specialists.py`
  - RTL: `rtl_loop.py`, `rtl_specialists.py`, `rtl_templates/`
  - EDA tools: `eda_tools/` (lint, simulation, synthesis)
  - Supporting: `ip_blocks.py`, `memory.py`, `bandwidth.py`, `technology.py`,
    `manufacturing.py`, `floorplan.py`, `pareto.py`

- **Optimization subsystem** — 3-layer MOO pipeline:
  - Layer 1: MAP-Elites (quality-diversity, 5K-10K evals)
  - Layer 2: Bayesian BO with qNEHVI (sample-efficient, ≤4 objectives)
  - Layer 3: NSGA-III (many-objective, >4 objectives)

- **Deploy targets** — Jetson, OpenVINO, Coral, KPU, NVDLA
  in `agents/deployment/targets/`

- **Specs system** — hierarchical system specs with versioning (`branes spec`)

### Item 3: Add Missing CLI Commands to CLAUDE.md (Priority 7)

**File:** `CLAUDE.md`

Replace the existing 8-command CLI list with the full 19-command set:
- `chat`, `workflow`, `analyze`, `benchmark`, `codebase`, `report`, `backends`,
  `secrets`, `config`, `pipeline`, `model`, `zoo`, `design`, `testbench`,
  `deploy`, `demo`, `optimize`, `spec`, `swap`

Group them by category (analysis, optimization, deployment, management).

### Item 4: Add Conventional Commit Convention to CLAUDE.md

**File:** `CLAUDE.md`

Add a "Commit Convention" section noting:
- Conventional commits required: `type(scope): description`
- Types: feat, fix, docs, chore, refactor, test, ci, perf, style, build
- Enforced by PR template and semantic-release

### Item 5: Pre-push Quality Gate Hook (Priority 3)

**File:** `.claude/settings.local.json`

Add a `hooks.PreToolUse` entry that intercepts `git push` commands and runs
black + ruff before allowing the push. Read the existing file first to merge
with existing permissions.

### Item 6: Auto-lint on Edit/Write Hook (Priority 5)

**File:** `.claude/settings.local.json`

Add a `hooks.PostToolUse` entry that runs `ruff check` on Python files after
Edit or Write tool use.

### Item 7: Conventional Commit Hook (Priority 9)

**File:** `.claude/settings.local.json`

Add a `hooks.PreToolUse` entry that validates commit messages match conventional
commit format.

### Item 8: `/quality-gate` Skill (Priority 4)

**File:** `.claude/commands/quality-gate.md` (new, create directory first)

Slash command running black + ruff + pytest.

### Item 9: `/smoke-swap` Skill (Priority 8)

**File:** `.claude/commands/smoke-swap.md` (new)

Slash command running SWaP-C tests and smoke commands.

### Item 10: `/cross-repo-check` Skill (Priority — not in original 9, but in the doc)

**File:** `.claude/commands/cross-repo-check.md` (new)

Slash command validating cross-repo imports.

### Item 11: Connect Branes MCP Server (Priority 6)

**File:** `.claude/settings.local.json`

Add `mcpServers.branes` entry pointing to the MCP server module. Need to verify
the server can be started with `python -m embodied_ai_architect.mcp.server`.

---

## Implementation Order

1. Read `.claude/settings.local.json` to understand existing structure
2. Rewrite `CLAUDE.md` with all fixes (items 1-4)
3. Create `.claude/commands/` directory
4. Write skill files (items 8-10)
5. Update `.claude/settings.local.json` with hooks and MCP server (items 5-7, 11)
6. Verify: read back all files to confirm correctness

---

## Verification

```bash
# Confirm CLAUDE.md has no stale references
grep -c 'embodied-ai' CLAUDE.md          # should be 0
grep -c '\.venv/bin/' CLAUDE.md           # should be > 0
grep -c '3\.9' CLAUDE.md                  # should be 0

# Confirm skill files exist
ls .claude/commands/

# Confirm settings.local.json is valid JSON
.venv/bin/python -c "import json; json.load(open('.claude/settings.local.json'))"

# Confirm MCP server module is importable
.venv/bin/python -c "import embodied_ai_architect.mcp.server; print('MCP OK')"
```
