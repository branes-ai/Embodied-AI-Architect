# Session 2026-05-09 -- Graphs Phase 5: dynamic catalog + profile-aliased SKUs

**Repo:** `branes-ai/embodied-ai-architect`
**Landed on `main`:** PR #201 (commit `889aee5`)
**Cross-repo origin:** `branes-ai/graphs#136` (Phase 5)

## Context

This session's change in this repo is the consumer-side payoff for a
multi-PR arc in `branes-ai/graphs` that introduced **profile-as-SKU
addressing** -- making `Jetson-Orin-Nano-8GB@7W` a first-class
addressable deployment target rather than just "Jetson Orin Nano" with
some power mode hidden in a kwarg. Until this session, the orchestrator
(this repo) carried a hardcoded `HARDWARE_CATALOG` dict that listed
hardware names by hand. That created two persistent failure modes:

1. **Drift.** Every time a new mapper landed in `graphs`, the
   orchestrator's catalog lagged. Worse, when the canonical naming for
   existing chips changed (e.g., `Jetson-Orin-AGX` → `Jetson-Orin-AGX-64GB`
   to disambiguate from the 32GB variant), the orchestrator listed names
   the analysis tools would reject.
2. **No deployment-target granularity.** The orchestrator could ask
   "analyze on Jetson-Orin-Nano-8GB" but couldn't say "analyze on
   Jetson-Orin-Nano-8GB at the 7W power profile." That distinction
   matters: the same silicon at 7W vs 25W vs MAXN gives very different
   throughput / latency answers.

## What landed

### Live-sourced catalog

`llm/graphs_tools.py` now has `_get_hardware_catalog(include_profile_aliases
=False)` that walks `graphs.hardware.mappers.list_all_mappers()` /
`list_all_skus()`, buckets entries by category from `mapper.category`, and
returns a sorted dict. The `HARDWARE_CATALOG` and `ALL_HARDWARE`
module-level names are now thin wrappers that call this function. The
orchestrator's hardware list is whatever the registry says it is.

### Profile-alias expansion

When called with `include_profile_aliases=True`, the function expands every
multi-profile chip into one row per (silicon × profile) using the alias
form `<silicon>@<profile>`. Today that's:

| Silicon | Profiles | Aliases produced |
|---------|----------|------------------|
| `Jetson-Orin-Nano-8GB` | 7W / 15W / 25W / MAXN | 4 |
| `Jetson-Orin-NX-16GB` | 10W / 15W / 20W / 25W / MAXN | 5 |
| `Jetson-Orin-AGX-64GB` | 15W / 30W / 50W / MAXN | 4 |
| `Jetson-Thor-128GB` | 40W / 60W / 75W / 100W / MAXN | 5 |

The total count goes from 46 silicon-bin entries to 122 addressable SKUs
when aliases are expanded. The expansion is opt-in to keep the default
prompt small (the LLM doesn't need to see all 122 unless the user is
asking specifically about deployment-target tradeoffs).

### Tool-surface change

`list_available_hardware(include_profile_aliases: bool = False)` is the
LLM-callable wrapper. Five hardware-consuming tools (`analyze_workload`,
`check_latency`, `check_memory`, `check_energy`, `compare_hardware_targets`,
`full_analysis`) had their `hardware_name` parameter docstrings updated to
say:

> Hardware name (e.g., 'H100', 'Jetson-Orin-Nano-8GB' for default profile,
> or 'Jetson-Orin-Nano-8GB@7W' for a specific power profile).

This is what tells the LLM that the alias form is legal -- without that
docstring update the model would only ever generate silicon-bin names.

### Fallback catalog (CodeRabbit Major fix)

When `graphs` isn't installed -- it's an optional dep -- the catalog
falls back to a static `_FALLBACK_HARDWARE_CATALOG` constant. CodeRabbit
caught that the fallback's bucket keys still used the legacy names from
the pre-Phase-5 catalog: `datacenter_gpu`, `edge_gpu`, `accelerators`,
`automotive`. The dynamic path used the registry-canonical names: `gpu`,
`tpu`, `kpu`, `accelerator`, `dsp`. So in fallback mode,
`list_available_hardware(category="gpu")` would return "Unknown category"
because there was no `gpu` bucket -- only `datacenter_gpu` and
`edge_gpu`.

Fixed by renaming the fallback buckets to match the registry. Two
regression guards added to `tests/test_hardware_catalog.py`:

- `test_fallback_catalog_keys_match_registry_categories`: asserts the
  legacy names are absent and `gpu` / `tpu` are present.
- `test_list_available_hardware_in_fallback_mode`: whole-tool round-trip
  in fallback mode -- `list_available_hardware(category="gpu")` must
  return a populated bucket, not the "Unknown category" error payload.

### Tests

`tests/test_hardware_catalog.py` -- 15 tests across three classes:

- `TestDynamicHardwareCatalog` (live-source path, gated on `HAS_GRAPHS`):
  default vs expanded shape, real Jetson names present, alias expansion
  produces strictly more entries, aliases inherit silicon category,
  sorted-output determinism for prompt-cache stability.
- `TestListAvailableHardwareTool` (LLM-tool wrapper): default and
  expanded calls, category filter, alias-aware filter, unknown-category
  error path.
- `TestFallbackCatalog` (runs unconditionally via `monkeypatch`):
  fallback identity, registry-canonical bucket keys, whole-tool
  round-trip in fallback mode.

## Wrinkles

### Stacked-PR auto-close

The Phase 5 work was originally opened as PR #200 stacked on top of
another in-flight PR. When the parent PR merged with `--delete-branch`,
GitHub auto-closed #200. Recovery: rebased onto current main, force-
pushed to a fresh branch, opened as PR #201. Lesson re-learned (this
isn't the first time): for cross-repo arcs where the parent PR will
merge first, branch off `main` directly and don't stack.

### Black formatting

`black --line-length 100` wanted reformatting on the Phase 5 files. Easy
fix -- ran black, committed as `bf8b136` -- but the lesson is to wire
black into a pre-commit hook here so it doesn't keep showing up as a CI
failure on otherwise-clean PRs.

### CodeRabbit on stale tool description

My initial pass used `sed` to update the `hardware_name` docstrings
across all the hardware tools. The sed missed `check_latency`'s
parameter description. CodeRabbit caught it in review; explicit `Edit`
call fixed it. When the change-set is "update the same field across
several files," prefer Edit per file over a global sed -- it's slightly
more verbose but it's harder to skip a hit by accident.

## What this enables

The orchestrator can now say things like:

> "I'd recommend deploying YOLOv8n on `Jetson-Orin-Nano-8GB@7W` if you're
> battery-bound, but `@25W` if you can afford the headroom -- here's the
> latency and energy breakdown for both."

Before Phase 5 it had no syntax to express that distinction. The
dynamic catalog also means that as new mappers land in `graphs` (the 41
non-PhysicalSpec-populated ones being the obvious candidates), they
become immediately addressable from the orchestrator without a code
change here.

## Cross-repo deployment

| Repo | Role this session |
|------|-------------------|
| `branes-ai/graphs` | Phases 1-4 + 3.5 of #136 -- `PhysicalSpec`, profile aliases, YAML loader, memory clocks. 9 PRs merged. |
| `branes-ai/embodied-schemas` | RFC 0001 (ComputeProduct unification) merged as PR #7. Issue #8 filed for two YAML bugs. |
| `branes-ai/embodied-ai-architect` (this repo) | Phase 5 -- consume the new graphs registry contract. PR #201. |

## Files of note

| Path | Purpose |
|------|---------|
| `src/embodied_ai_architect/llm/graphs_tools.py` | Dynamic catalog + alias-aware tool surface |
| `tests/test_hardware_catalog.py` | 15 tests; CodeRabbit regression guards |
