# KPU Micro-Architecture Optimization Knobs — Design Notes

**Status**: design discussion / gap analysis (created alongside issue #32 PR)
**Audience**: anyone touching `design_optimizer` or the KPU pipeline

## Why this document exists

Issue #32 added a first cut of KPU-targeting strategies to
`design_optimizer.OPTIMIZATION_STRATEGIES`. The point of that PR was to make
the optimizer *aware* of the KPU as something it could mutate — before #32
the catalog only knew how to change models, frequencies on generic IP
blocks, and the process node. The optimizer was forced to suggest model-level
changes even when KPU sizing was the actual bottleneck.

The strategies that landed in #32 are these six:

```
reduce_systolic_array     (area, power)
upgrade_dram_technology   (latency)
add_sram_banks            (latency)
widen_noc                 (latency)
reduce_compute_tiles      (area, power)
clock_scale_kpu           (power)
```

**These are not the right set.** They are *plausible* knobs in the sense that
each one corresponds to a real KPU parameter and the wiring through
`apply_kpu_overrides` works. But:

1. The catalog was assembled to demonstrate the *plumbing* — it was not
   derived from a principled analysis of the KPU design space.
2. Several legitimate knobs are missing (see below).
3. Several knobs are too coarse — they bundle decisions that should be
   independent.
4. Some knobs are mis-classified by `applicable_when` and would be selected
   for the wrong constraint.
5. The reduction factors are static guesses, not derived from the actual
   physical estimators.

This document captures what #32 actually implemented, why it isn't enough,
and what the proper knob set should look like. The follow-on epic (issue
created alongside this doc) tracks the work to fix it.

---

## What landed in #32

### Catalog entries

| Strategy | `applies_to` | `applicable_when` | What it mutates |
|---|---|---|---|
| `reduce_systolic_array` | `kpu_config` | area, power | `compute_tile.array_rows -= 4`, `compute_tile.array_cols -= 4` (floor 4) |
| `upgrade_dram_technology` | `kpu_config` | latency | walks `dram.technology` through `LPDDR4X → LPDDR5 → HBM2E` and bumps `dram.bandwidth_per_channel_gbps` accordingly |
| `add_sram_banks` | `kpu_config` | latency | `compute_tile.l2_num_banks += 2`, `memory_tile.l3_num_banks += 1` |
| `widen_noc` | `kpu_config` | latency | `noc.link_width_bits *= 2` (cap 1024) |
| `reduce_compute_tiles` | `kpu_config` | area, power | drop `array_rows -= 1`, fall back to `array_cols -= 1` |
| `clock_scale_kpu` | `kpu_config` | power | `compute_tile.frequency_mhz *= 0.8` (floor 100 MHz) |

### Plumbing

- `design_optimizer` filters out `kpu_config`-strategies when no `kpu_config`
  is on state, so non-RTL pipelines never see them as candidates.
- `_apply_kpu_strategy` builds a dotted-path override dict per strategy and
  applies it via `kpu_config.apply_kpu_overrides` (the helper introduced in
  issue #29 for plan-review overrides).
- After mutation, `floorplan_estimate` and `bandwidth_match` are cleared on
  state so the next dispatch iteration re-runs the validators against the
  new config.
- `STRATEGY_VARIABLE_MAP` ties each KPU strategy to a placeholder MOO design
  space variable name (`systolic_array_size`, `dram_bandwidth_gbps`,
  `sram_size_kb`, `noc_link_width_bits`, `num_compute_tiles`, `clock_mhz`)
  so the MOO-aware selector from issue #25 can boost the strategy via BO
  sensitivity. **These variable names are not currently in the design space
  emitted by `moo/design_space.py`** — the boost falls through to the
  neutral 1.0× factor, so the coupling is mostly aspirational.
- Reduction factors (`power_reduction_factor`, `latency_reduction_factor`)
  are static guesses (0.18 to 0.25) and not derived from the
  `physical_estimators` / `bandwidth` / `floorplan` modules that compute
  the actual physics.

### What that buys us

- The optimizer can now write to `kpu_config` at all, where before it
  couldn't.
- An area-failing run with `rtl_enabled=True` will get a KPU-targeted
  suggestion instead of `smaller_model` or `quantize_int8`.
- The next dispatch iteration sees a fresh KPU config and re-validates.
- The structural pattern (filter → select → apply → clear validators) is
  in place so adding more strategies later is mechanical.

---

## What's wrong with the current set

### 1. The strategies bundle decisions that should be independent

`reduce_systolic_array` shrinks both `array_rows` and `array_cols` by 4.
That's two knobs masquerading as one, and there's no reason a square shrink
is always the right move — many AI workloads have asymmetric data shapes
where a tall-narrow array dominates.

Similarly, `add_sram_banks` always bumps both L2 and L3. The L2 bottleneck
case and the L3 bottleneck case are different bandwidth bottlenecks; they
should be addressed independently and the bandwidth waterfall (see issue
\#30 / `KPUBandwidthSlackness`) tells us *which* link is saturated.

### 2. Missing knobs that the architect actually has

Looking at `KPUMicroArchConfig`, here are real, independently-tunable
parameters that the catalog does not address:

**Memory hierarchy**:
- `compute_tile.l1_size_bytes` and `l1_num_banks` — L1 skew buffer sizing
- `compute_tile.l2_size_bytes` (capacity, not just banks) — the optimizer
  only knows how to add banks, not grow capacity
- `memory_tile.l3_tile_size_bytes` — same problem at L3
- `compute_tile.num_streamers`, `streamer_prefetch_depth`, `streamer_buffer_bytes`
- `memory_tile.num_block_movers`, `block_mover_bw_gbps`
- `memory_tile.num_dma_engines`, `dma_max_transfer_bytes`, `dma_queue_depth`

**Compute**:
- `compute_tile.vector_lanes` — the vector unit is independent of the
  systolic array but they're often coupled in shape
- `compute_tile.supported_precisions` — adding/removing INT4 / FP8 / BF16
  changes both area and power meaningfully
- `compute_tile.frequency_mhz` floor (today the strategy hardcodes 100 MHz)

**NoC**:
- `noc.topology` — `mesh_2d` is the only option in presets, but a torus or
  fat-tree shifts the bandwidth/area tradeoff. Even within mesh, the
  routing strategy matters.
- `noc.frequency_mhz` independently of `noc.link_width_bits`
- `noc.num_routers` — currently derived from `array_rows × array_cols` but
  could be independent in some topologies

**DRAM** (more granular than the chain):
- `dram.num_controllers` and `dram.channels_per_controller` independently
  of technology — sometimes you just want more channels of LPDDR4X without
  paying for HBM
- `dram.capacity_gb` — capacity vs bandwidth are decoupled in real DRAM
  selection

**Process / packaging**:
- `process_nm` is touched by `shrink_process_node` / `grow_process_node`
  but those work on `constraints.target_process_nm`, not on `kpu_config`,
  and the two can drift out of sync.
- 2.5D / chiplet partitioning, advanced packaging — not represented at all

### 3. Mis-classified strategies

- `clock_scale_kpu` is marked `applicable_when=["power"]` but reducing the
  clock also helps `latency` slack indirectly via slower-but-more-efficient
  designs that may then permit *not* using a strategy that increases latency.
  More importantly, it should *also* be applicable to `area` failures
  (since lower clocks let you remove pipeline stages), but the current
  catalog ignores that.
- `widen_noc`, `add_sram_banks`, `upgrade_dram_technology` are all marked
  `latency`-only, but they all have side effects on power and area that
  might trigger the *next* iteration's bottleneck. The catalog is monotonic
  per-strategy but the system is not.
- `upgrade_dram_technology` has a static `power_reduction_factor=-0.05`
  meaning "5% power *increase*" but the actual delta depends heavily on
  which step of the chain we're climbing (LPDDR5 vs HBM2E differ by an
  order of magnitude in IO power).

### 4. Reduction factors are not derived from physics

Every entry in the catalog has hand-typed `power_reduction_factor` and
`latency_reduction_factor` numbers like `0.25`, `0.18`. These were chosen
by gut. The codebase already has the physics:

- `graphs/physical_estimators.py` knows how to estimate power, area, cost
  from a `KPUMicroArchConfig`.
- `graphs/floorplan.py` knows how to estimate die area from the same.
- `graphs/bandwidth.py` knows how to estimate the per-link bandwidth chain.
- `moo/evaluator.py` already glues these together for the MOO loop.

So the right thing to do is: for each strategy, **call the physical
estimators on the before-and-after config and report the actual delta**.
That gives the optimizer real numbers to score on, and it removes the
maintenance burden of keeping hand-tuned factors aligned with reality.

### 5. The MOO sensitivity coupling is aspirational

`STRATEGY_VARIABLE_MAP` ties each KPU strategy to a variable name like
`systolic_array_size` or `sram_size_kb`. These names don't match anything
in the design space that `moo/design_space.create_soc_design_space` actually
emits. The MOO-aware selector falls through to a 1.0× neutral boost, so
the integration with issue #25's sensitivity-driven selection is decorative.

To fix this we either need:

(a) Add the corresponding variables to `create_soc_design_space` so MOO
    actually optimizes them and produces sensitivity for them, or

(b) Map the strategy to existing MOO variables that *do* drive its target
    objective (e.g. `clock_scale_kpu` should map to whatever `clock_mhz`
    variable already exists in the design space, not to a placeholder).

Option (a) is the right answer but requires expanding the MOO design space
intentionally, which is a significant scope.

---

## Proposed knob set (sketch)

The follow-on epic should produce a real catalog. Here's a starting sketch
to argue against. Each entry should ultimately compute its reduction
factors from the physical estimators rather than carrying static numbers.

### Compute knobs

- `shrink_systolic_rows` — shrink only `array_rows` by N
- `shrink_systolic_cols` — shrink only `array_cols` by N
- `shrink_systolic_uniform` — square shrink (the current bundled behavior)
- `grow_systolic_rows` / `grow_systolic_cols` — for performance pressure
- `add_vector_lanes` / `remove_vector_lanes`
- `enable_int4` / `disable_int4` (similarly for fp8, bf16)

### Memory knobs (per level)

- `grow_l1_capacity` / `shrink_l1_capacity`
- `add_l1_banks` / `remove_l1_banks`
- `grow_l2_capacity` / `shrink_l2_capacity`
- `add_l2_banks` / `remove_l2_banks`
- `grow_l3_capacity` / `shrink_l3_capacity`
- `add_l3_banks` / `remove_l3_banks`
- `add_streamers` / `remove_streamers`
- `grow_streamer_buffer`
- `add_block_movers` / `remove_block_movers`
- `add_dma_engines`

### NoC knobs

- `widen_noc_links` / `narrow_noc_links` (current `widen_noc` is half this)
- `clock_up_noc` / `clock_down_noc`
- `change_noc_topology` (parameterized over `mesh_2d`, `torus`, `fat_tree`)

### DRAM knobs

- `add_dram_channels` / `remove_dram_channels`
- `add_dram_controllers` / `remove_dram_controllers`
- `upgrade_dram_tech` / `downgrade_dram_tech` (the chain walk, kept)
- `grow_dram_capacity` / `shrink_dram_capacity`

### Grid / packaging knobs

- `add_compute_tile_row` / `remove_compute_tile_row`
- `add_compute_tile_col` / `remove_compute_tile_col`
- (deferred) chiplet partitioning, 2.5D integration

### Cross-cutting

- Each strategy must be tagged with **all** the constraints it actually
  affects (power, latency, area, cost), not just its primary intent.
- Each strategy should declare its **direction** explicitly so the optimizer
  can pick the inverse when the bottleneck flips.
- Each strategy should expose its **floor / ceiling** so the optimizer
  doesn't suggest something that can't actually be applied.
- Static reduction factors should be **replaced** by `compute_delta(config,
  workload)` callbacks that hit the physical estimators and return real
  numbers. The catalog becomes a registry of (name, applicability, delta_fn)
  tuples instead of a static dict.

---

## Open design questions

1. **Granularity vs optimizer churn.** Splitting `reduce_systolic_array`
   into rows/cols/uniform variants triples the number of candidate
   strategies the optimizer has to evaluate per iteration. Is that worth
   it, or do we need a meta-strategy that picks the right granularity?

2. **Coupling to MOO design space.** Should we add every knob above as a
   MOO variable? That would explode the search space. Or do we keep MOO
   focused on a tractable subset and let the optimizer handle the long
   tail of detailed knobs?

3. **Floor enforcement.** Each strategy needs a floor (e.g. systolic array
   can't go below 4×4). Should that be a hardcoded constant per strategy,
   or a constraint pulled from `kpu_config` constructor validation, or
   from a separate `KPUDesignLimits` model?

4. **Direction inference.** When the bottleneck flips (was area, now
   latency), the optimizer should be able to *unwind* a previous decision.
   Today the catalog has separate strategies for shrink and grow — that
   doubles the entry count. An alternative is a single strategy with a
   direction parameter, but that complicates `applicable_when`.

5. **Architect veto.** Issue #29 already gives the architect plan-review
   overrides for the KPU. Should the optimizer respect those overrides as
   floor/ceiling pins (i.e. never suggest a strategy that would un-do an
   architect override), or treat them as starting points only?

---

## Pointers

- `src/embodied_ai_architect/graphs/optimizer.py` — current catalog and
  selector
- `src/embodied_ai_architect/graphs/kpu_config.py` — `KPUMicroArchConfig`
  and `apply_kpu_overrides`
- `src/embodied_ai_architect/graphs/physical_estimators.py` — the physics
  the proposed delta callbacks would call into
- `src/embodied_ai_architect/graphs/moo/design_space.py` — current MOO
  variables (the `STRATEGY_VARIABLE_MAP` should converge here)
- Issues `#21–#27`, `#29–#32` — the integration arc that this doc closes
