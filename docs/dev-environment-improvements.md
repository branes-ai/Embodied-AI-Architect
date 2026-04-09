# Development Environment Improvements

Recommendations for CLAUDE.md updates, hooks, skills, and MCP servers to make
Embodied AI Architect development faster and higher quality with Claude Code.

---

## 1. CLAUDE.md — Stale & Missing Content

The file has several inaccuracies and gaps that actively mislead Claude Code.

### Bugs to Fix

| Issue | Current | Correct |
|-------|---------|---------|
| CLI entrypoint | `branes` (19 occurrences) | `branes` |
| Tool commands | bare `pip`, `pytest`, `black`, `ruff` | `.venv/bin/` prefix |
| Python target | 3.9+ | >=3.11 (per pyproject.toml) |
| CLI command groups | 8 listed | 19+ exist |

### Missing Architecture Sections

- **`graphs/` in-repo subsystem** (50+ files) — MOO engine, physical estimators,
  SWaP-C analysis, KPU config, RTL loop, SoC state machine, specialists. This is
  the computational core and CLAUDE.md barely mentions it.
- **SWaP-C optimization methodologies** — profiles, analysis library, 5 new commands
  (score, rank, sensitivity, sweep, budget).
- **Optimization subsystem** — MAP-Elites, NSGA-III, Bayesian, design space.
- **Deploy targets** — Jetson, OpenVINO, Coral, KPU, NVDLA.
- **Specs system** — `branes spec` commands.
- **Conventional commit format** — enforced by PR template but not documented for Claude.

### Missing Cross-Repo Guidance

- How to test changes that span embodied-schemas + graphs + this repo.
- Which repo owns which Pydantic models (avoids duplicate definitions).

---

## 2. Hooks — Catch Errors Earlier

### a) Pre-push quality gate (highest value)

Prevents the #1 time sink — pushing code, waiting for CI, finding lint failures.

```jsonc
// .claude/settings.json (or settings.local.json)
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "if echo \"$TOOL_INPUT\" | grep -qE 'git push'; then .venv/bin/black --check src/ tests/ --line-length 100 && .venv/bin/ruff check src/ tests/ && echo 'Quality gate passed'; fi"
          }
        ]
      }
    ]
  }
}
```

### b) Auto-lint on file write (medium value)

Run ruff on the specific file after every Edit/Write so lint errors surface immediately
instead of batching up at the end.

```jsonc
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [
          {
            "type": "command",
            "command": "FILE=$(echo \"$TOOL_INPUT\" | jq -r '.file_path // empty'); if [ -n \"$FILE\" ] && echo \"$FILE\" | grep -qE '\\.py$'; then .venv/bin/ruff check \"$FILE\" --no-fix 2>&1 || true; fi"
          }
        ]
      }
    ]
  }
}
```

### c) Conventional commit enforcement (low effort, high consistency)

Reject commit messages that don't match `type(scope): description`.

```jsonc
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "if echo \"$TOOL_INPUT\" | grep -qE 'git commit'; then MSG=$(echo \"$TOOL_INPUT\" | grep -oP '(?<=-m \").*?(?=\")' || true); if [ -n \"$MSG\" ] && ! echo \"$MSG\" | head -1 | grep -qE '^(feat|fix|docs|chore|refactor|test|ci|perf|style|build)(\\(.*\\))?: '; then echo 'BLOCKED: Commit message must follow conventional commits: type(scope): description' >&2; exit 1; fi; fi"
          }
        ]
      }
    ]
  }
}
```

---

## 3. Custom Skills (slash commands)

### a) `/quality-gate` — The mandatory pre-push sequence

```markdown
<!-- .claude/commands/quality-gate.md -->
Run the mandatory pre-push quality gate:
1. `.venv/bin/black --check src/ tests/ --line-length 100`
2. `.venv/bin/ruff check src/ tests/`
3. `.venv/bin/pytest tests/ -q`
Report results concisely. If anything fails, fix it.
```

### b) `/smoke-swap` — Quick SWaP-C subsystem validation

```markdown
<!-- .claude/commands/smoke-swap.md -->
Run SWaP-C smoke tests:
1. `.venv/bin/pytest tests/test_swap_profiles.py tests/test_swap_analysis.py tests/test_swap_cli.py -v`
2. `.venv/bin/branes swap estimate --area 50 --power 5 --process 28`
3. `.venv/bin/branes swap score --area 50 --power 5 --process 28 --profile drone`
4. `.venv/bin/branes swap sensitivity --area 50 --power 5 --process 28 --mode tornado`
Report results concisely.
```

### c) `/cross-repo-check` — Validate the three-repo boundary

```markdown
<!-- .claude/commands/cross-repo-check.md -->
Check cross-repo consistency:
1. Verify embodied-schemas is importable: `.venv/bin/python -c "from embodied_schemas import Registry; print('schemas OK')"`
2. Verify graphs is importable: `.venv/bin/python -c "from graphs.analysis.unified_analyzer import UnifiedAnalyzer; print('graphs OK')"`
3. Run this repo's full test suite: `.venv/bin/pytest tests/ -q`
4. Check for import errors in any test file.
Report which repos are healthy and flag any cross-boundary breakage.
```

---

## 4. MCP Servers

### Connect the branes MCP server to Claude Code

The project's own MCP server (`src/embodied_ai_architect/mcp/server.py`) already
exposes the MOO engine as MCP tools. Connecting it to Claude Code means Claude can
run SWaP-C estimates, explore design spaces, and rank designs interactively during
development — instead of shelling out to CLI commands.

```jsonc
// .claude/settings.json
{
  "mcpServers": {
    "branes": {
      "command": ".venv/bin/python",
      "args": ["-m", "embodied_ai_architect.mcp.server"],
      "env": {}
    }
  }
}
```

This would let Claude call `estimate_system_bom`, `compute_thermal_feasibility`,
`topsis_rank`, etc. as native tools — faster feedback loops when developing new
analysis features.

### Memory/context MCP server (lower priority)

For a project this size (190+ source files, 50+ graph modules, 3 repos), a semantic
search MCP server over the codebase would help with "find the module that handles X"
queries without burning context on broad grep sweeps. Nice-to-have — the Explore
subagent already handles this reasonably well.

---

## 5. Priority Order

| Priority | Item | Impact | Effort |
|----------|------|--------|--------|
| 1 | Fix CLAUDE.md inaccuracies (CLI name, .venv, Python version) | High — every session starts wrong | 15 min |
| 2 | Add `graphs/` architecture section to CLAUDE.md | High — it's 60% of the code | 30 min |
| 3 | Pre-push quality gate hook | High — prevents CI round-trips | 5 min |
| 4 | `/quality-gate` skill | Medium — quick access to the gate | 2 min |
| 5 | Auto-lint on Edit/Write hook | Medium — catches errors inline | 5 min |
| 6 | Connect branes MCP server | Medium — interactive analysis | 15 min |
| 7 | Add missing CLI commands to CLAUDE.md | Medium — completeness | 20 min |
| 8 | `/smoke-swap` skill | Low — useful for SWaP-C work | 2 min |
| 9 | Conventional commit hook | Low — nice guardrail | 5 min |
