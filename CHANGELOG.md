# Changelog

All notable changes to the Embodied AI Architect project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Loop Convergence epic — COMPLETE: Phases 3 & 4 + tooling** (2026-07-16,
  epic `#203` closed, PRs #230–#235):
  - **Observable loop tracer + steerable acceptance harness** (PR #230) —
    `graphs/loop_trace.py`: `run_loop_traced` drives the real
    seed→critic→optimize→evaluate→recommend loop over the actual node functions,
    recording a `LoopTrace` (one step per node with the decision and the *reason
    why*) to the `embodied_ai_architect.loop` logger, with a human-in-the-loop
    `steer` hook. Plus a gold-standard-style deterministic acceptance harness.
  - **User-facing demonstration** (PR #231) — `docs/loop-convergence-demo.md` +
    `docs/demos/loop_convergence_demo.py`: solves a drone-perception SoC
    (≤5W/≤33ms/≤$30) end-to-end (8W/45ms → 4.6W/17.3ms autonomously; 3.8W under a
    tightened 4W budget after an operator injects structured sparsity), with live
    agent logging and steering.
  - **Phase 3 — specialist agents (S9 #213, S8 #214, PR #232)** —
    `graphs/specialist_agents.py`: verdict-first estimator tools + `PPASpecialist`
    and `ThermalSpecialist` reasoning agents that file typed `DesignIssue`s
    (thermal via a new `physical_estimators.estimate_junction_temperature`).
    `SPECIALIST_RETASK` deltas now re-run the named specialist and drain
    `pending_specialist_tasks`.
  - **Phase 4 — wire + tune:**
    - **S10 front door (#215, PR #233)** — `seed_node` turns an NL mission into a
      valid `DesignState` (constraints + 17-var joint design space) via
      `MissionDecomposer` + `create_joint_design_space` (`plan_to_constraints`).
    - **S11 CLI + skills (#216, PR #234)** — `branes session iterate` runs real
      loop iterations over a persisted `DesignState` (verified with the real
      MAP-Elites engine); `architect-loop.md` rewritten to invoke it;
      `architect-drill`/`architect-assess` read `open_issues`.
    - **S12 convergence tuning (#217, PR #235)** — single `has_converged()`:
      empty backlog OR critic diminishing-returns OR a *relative, windowed*
      hypervolume plateau; the critic emits a `diminishing_returns` judgment when
      levers are exhausted.
- **Loop Convergence epic — Phase 2: agent critic + optimizer** (2026-07-15,
  epic `#203`, PRs #225–#228, v1.7.0–v1.10.0):
  - **LLM critic path (S4, #209)** — `Critic._review_with_llm` assembles a
    prompt from PPA verdicts + Pareto front + `open_issues` + research context
    (`CRITIC_SYSTEM_PROMPT` pins a strict JSON schema and the per-`DeltaKind`
    payloads) and parses it into a `CriticVerdict` of typed `DesignIssue`s +
    `DesignDelta`s, linked by index. Robust to model slop (skips deltas failing
    S3 validation or with unknown kinds; unknown enum strings fall back);
    `converged` is coerced from JSON (not Python truthiness) and refused while
    any verdict is `FAIL`; heuristic fallback preserved for the no-key path.
  - **Research-grounded deltas (S5, #210)** — `DesignDelta.research_refs`;
    retrieval tags derived from the failing metrics / open issues
    (`_research_tags_for_state`, exhaustive over `MetricAxis`), so a power-bound
    drone and a latency-bound AMR retrieve different research; per-delta
    citations parsed from the verdict.
  - **Monotonic Pareto merge (S6, #211)** — the loop MOO tool's `_merge_pareto`
    reuses `moo.specialist._merge_pareto_frontiers`; dominated points dropped,
    no non-dominated point lost across iterations (points keep their
    engine-native shape via `metadata`).
  - **Pipeline-consistent evaluation (S7, #212)** — the loop's `evaluate_node`
    delegates to the real `ppa_assessor`, so its `ppa_metrics.verdicts` equal
    the dispatcher pipeline's for the same state (no divergent second check).
- **Loop Convergence epic — Phase 1: state unification** (2026-07-14, epic
  `#203`, PRs #218–#223, v1.2.0–v1.6.0):
  - New milestone doc `docs/plans/roadmap-loop-convergence.md` — unify the two
    disjoint design loops (population/Pareto MOO + single-design dispatcher)
    into one multi-agent loop over a shared `DesignState`; 4 phases, seam
    checklist S1–S12, wired into `roadmap-v2.md` (PR #218).
  - **Unified `DesignState`** (`graphs/design_state.py`) — a single TypedDict
    (~92 channels) that subsumes the former `SoCDesignState` and
    `OptimizationLoopState`, plus `DesignIssue` / `DesignDelta` — the structured
    currency between a reasoning Critic and Optimizer.
  - **DRAFT loop skeleton** (import-only, not yet wired): `loop_agents.py`
    (`Critic`, `Optimizer`, node wrappers, single router) and
    `loop_convergence_graph.py` (StateGraph assembly + MOO-engine-as-a-tool
    adapter). `Critic._review_with_llm` is a documented `NotImplementedError`
    pending S4.
  - **Channel audit (S1, #204/#219)** — `declared_channels()`,
    `undeclared_keys()`, `assert_declared_channels()` guard the LangGraph
    invariant that any node write not declared as a channel is silently dropped;
    `tests/test_design_state_channels.py` enforces it per node.
  - **Typed `DesignDelta` payloads (S3, #208/#223)** — one payload model per
    `DeltaKind`, validated at construction (malformed edits raise
    `ValidationError`); `DesignDelta.typed_change()` returns the parsed model.
- **Dynamic hardware catalog from `graphs` registry** (2026-05-09, PR #201,
  `graphs#136` Phase 5):
  - `llm/graphs_tools.py`: replaced the hand-maintained `HARDWARE_CATALOG`
    constant with `_get_hardware_catalog(include_profile_aliases=False)`.
    The orchestrator's hardware list is now live-sourced from
    `graphs.hardware.mappers.list_all_mappers()`, which keeps it in sync
    with what the analysis tools can actually accept (no more drift between
    "what the agent thinks exists" and "what the analyzers will run").
  - **Profile-aliased SKUs** opt-in via `include_profile_aliases=True`:
    surfaces `Jetson-Orin-Nano-8GB@7W`, `@15W`, `@25W`, `@MAXN` as
    addressable deployment targets alongside the silicon-bin name. The
    `list_available_hardware` tool exposes the same flag.
  - 5 hardware-tool docstrings updated to document the profile-alias form
    (`<silicon>` for default profile, `<silicon>@<profile>` for a specific
    power point); `compare_hardware_targets` defaults updated to current
    Jetson naming.
  - **Static fallback catalog** for environments without `graphs` installed:
    `_FALLBACK_HARDWARE_CATALOG` uses the same registry-canonical bucket
    keys (`gpu` / `tpu` / `kpu` / `accelerator` / `dsp`), so
    `list_available_hardware(category="gpu")` returns a populated bucket
    in fallback mode instead of an "Unknown category" error.
  - `tests/test_hardware_catalog.py`: 15 tests covering live-source path,
    profile-alias expansion, alias category inheritance, sorted-output
    determinism, fallback-mode round-trip, and a regression guard for the
    fallback bucket-key contract that CodeRabbit caught in review.

### Changed
- **Both design loops now flow through the unified `DesignState`** (Loop
  Convergence Phase 1, 2026-07-14):
  - S2a (#205/#220): `graphs/optimization_loop.py` — the MOO loop nodes
    (decompose/formulate/optimize/evaluate/reason) read/write `DesignState`.
  - S2b (#206/#221): `graphs/soc_graph.py`, `dispatcher.py`, `specialists.py` —
    the dispatcher loop binds `StateGraph(DesignState)`; `DesignState` is a
    superset of the former `SoCDesignState`, so no specialist write is dropped.
  - CI linters pinned in `ci.yml` (`black==25.12.0`, `ruff==0.14.10`) to match
    the project `.venv` and stop version drift from failing the lint gate.

### Removed
- **Legacy per-loop state TypedDicts** (Loop Convergence Phase 1):
  - `OptimizationLoopState` deleted (S2a) — merged into `DesignState`.
  - `SoCDesignState` class deleted (S2c #207/#222); the name is kept as a lazy
    module `__getattr__` compat alias to `DesignState`, so existing
    `from graphs.soc_state import SoCDesignState` imports keep working without a
    codebase-wide rename and without a runtime import cycle.

### Fixed
- **`dispatch_node` state-forwarding bugs** surfaced by the S2b migration
  (2026-07-14, PR #221): the outer LangGraph merge dropped SWaP-C results
  (forwarded a phantom `swap_results` key instead of the declared
  `swap_assessment`) and four specialist channels (`rtl_testbenches`,
  `safety_analysis`, `prior_experience`, `last_strategy_rationale`). All now
  forwarded in both return paths.
- **Module docstring tool-name drift** in `llm/graphs_tools.py`: the file's
  module-level documentation referred to deprecated tool names
  (`analyze_*`, `check_constraint`) that had been renamed to `check_*`
  and `full_analysis` in earlier work. Refreshed alongside the Phase 5
  changes (PR #201).

- Multi-Objective Optimization Engine (2026-02-24):
  - New `graphs/moo/` package: 3-layer pipeline (MAP-Elites → Bayesian BO → NSGA-III) for SoC design space exploration
  - `design_space.py`: Mixed discrete/continuous/categorical variable definitions, LHS sampling, encode/decode for pymoo and BoTorch
  - `evaluator.py`: Thread-safe design evaluator wrapping existing PPA specialists (ip_blocks, manufacturing, technology) — pure function, no duplication
  - `executor.py`: `EvalBackend` protocol with `LocalThreadExecutor` (ThreadPoolExecutor) for parallel analytical model evaluation
  - `map_elites.py`: Layer 1 — quality-diversity algorithm filling a behavioral feature grid (5K-10K evals in seconds), custom implementation (~250 lines)
  - `bayesian_opt.py`: Layer 2 — sample-efficient Pareto refinement using BoTorch qNEHVI with GP surrogates, sensitivity from lengthscales, warm-start from MAP-Elites (requires botorch)
  - `nsga3.py`: Layer 3 — many-objective fallback (>4 objectives) wrapping pymoo NSGA-III with warm-start support (requires pymoo)
  - `engine.py`: Pipeline orchestrator with auto layer selection (<=4 obj → ME+BO, >4 → ME+NSGA3), hypervolume computation, knee point identification, tradeoff explanation
  - `specialist.py`: Bridge to existing dispatcher pattern — `moo_explorer(task, state)` writes both `pareto_results` (backward compat) and `moo_results`
  - `k8s_evaluator.py`: Kubernetes evaluation backend with semaphore-based concurrency control, EDA license management via secrets
  - New `mcp/` package: MCP server exposing 5 optimization tools (start_exploration, get_pareto_front, get_sensitivity, explain_tradeoff, get_exploration_status) with dual-response pattern and session management
  - `llm/optimization_tools.py`: 5 LLM tools (explore_design_space, get_pareto_front, get_design_sensitivity, explain_design_tradeoff, suggest_optimal_design) following existing tool pattern
  - `cli/commands/optimize.py`: CLI commands — `branes optimize explore|show-front|sensitivity|explain` with Rich table output
  - Optional dependency groups: `optimization` (pymoo, botorch, gpytorch, scipy), `mcp` (mcp, uvicorn, httpx)
  - 52 new tests across 6 test files (design_space, evaluator, map_elites, engine, mcp_server, bayesian_moo)
  - Documentation: `docs/sessions/2026-02-24-moo-engine.md`

- Full application codebase analysis pipeline (2026-02-24):
  - New `codebase/` package: static scanner, LLM-powered 4-pass code analyzer, workload profile converter
  - `CodebaseScanner`: detects languages (15 extensions), build systems (CMake/Cargo/pip/Make), ML model files, entry points, dependencies — no API key needed
  - `CodeAnalyzer`: 4-pass LLM analysis (build → entry points → compute kernels → synthesis) with 48K char per-pass limit
  - `CodebaseConverter`: maps 7 kernel types (ml_inference, signal_processing, control_loop, image_processing, sensor_fusion, io_bound, general_compute) to hardware operators and outputs workload_profile compatible with existing PPA pipeline
  - `CodebaseAnalyzerAgent(BaseAgent)`: wraps full pipeline, gracefully falls back to scan-only when LLM unavailable
  - 3 interactive chat tools: `scan_project`, `analyze_codebase`, `assess_codebase_on_hardware`
  - CLI: `embodied-ai codebase scan|analyze|assess` subcommands with `--hardware`, `--power-budget`, `--latency-target` options
  - 34 new tests with 3 synthetic test fixture projects (C++ drone, Python ML, Rust embedded)
  - Documentation: methodology guide (`docs/codebase-analysis-guide.md`), Starlight feature page, CLI reference updates
  - Documentation: `docs/sessions/2026-02-24-codebase-analysis-pipeline.md`

- Manufacturing cost model with process node optimizer (2026-02-18):
  - `manufacturing.py`: Physics-based cost model — dies per wafer (300mm circular packing), Murphy's yield model, manufacturing cost breakdown (die + package + test + NRE)
  - `PROCESS_ECONOMICS` database: 16 nodes (2nm–180nm) with wafer cost, defect density, mask set, design NRE, and test NRE
  - `ManufacturingCostBreakdown` Pydantic model with itemized variable costs (die, package, test) and amortized fixed costs (NRE/volume)
  - `get_adjacent_nodes()` in `technology.py` for process node traversal
  - `shrink_process_node` / `grow_process_node` optimizer strategies with dynamic naming for multi-step node changes
  - `cost_breakdown` field on `PPAMetrics` for full cost visibility
  - Cost breakdown display (Die Cost, Package, Test, NRE/unit, Yield, Dies/wafer) in 6 demo scripts
  - Convergence failure analysis in `demo_soc_optimizer.py`: per-constraint diagnostics, break-even volume calculation, actionable next steps
  - 21 new tests across `test_manufacturing.py` (14), `test_specialists.py` (3), `test_optimizer.py` (4)
  - Documentation: `docs/sessions/2026-02-18-manufacturing-cost-model.md`

### Changed
- Fix truncated goal display and enrich demo prompts (2026-02-18):
  - Replace `goal[:80-90]...` truncation with `textwrap.fill()` in 6 demo scripts so full goal text is always visible
  - Enrich all 7 demo goal strings with specific workload keywords (YOLOv8, ByteTrack, SLAM/localization/mapping, camera/vision, voice/speech, LiDAR) to trigger distinct `_estimate_workload_from_goal()` paths
  - Demo 5 (hitl-safety) now triggers voice + detection (unique combo); Demo 2 (dse-pareto) triggers slam + detection (no tracking); Demo 7 (full-campaign) triggers all 5 workload types
  - Demo 3 (soc-optimizer) dynamic goal reused for both display and `create_initial_soc_state()`
  - Demo 6 (experience-cache) delivery vs agricultural drone goals made much more distinct
  - Documentation: `docs/sessions/2026-02-18-fix-truncated-goals-enrich-prompts.md`

- Complexity-based synthesis timeout with configurable override (2026-02-17):
  - `RTLSynthesisTool.estimate_timeout()`: 3-tier heuristic (30s/120s/300s) from line count, always blocks, and memory declarations
  - `RTLSynthesisTool(timeout=)`, `EDAToolchain(synth_timeout=)`, `RTLLoopConfig(synth_timeout=)`: configurable override at every layer, defaults to `AUTO_TIMEOUT` (-1) for auto-detection
  - `SynthesisResult.source` field: `"yosys"` vs `"mock"` lets callers detect heuristic estimates
  - Mock fallback warnings now include reason (`yosys_not_found` or `timeout_Ns`)
  - Documentation: `docs/sessions/2026-02-17-synthesis-timeout-design.md`

### Added
- Phase 4: Evaluation Harness, Missing Demos & Regression Suite (2026-02-13):
  - **9-Dimension Evaluation Framework** (`src/embodied_ai_architect/graphs/`):
    - `evaluation.py`: Data models — `RunTrace`, `GoldStandard`, `DimensionScore`, `Scorecard`
    - `scoring.py`: 9 scoring functions — decomposition, PPA accuracy, exploration efficiency, reasoning quality, convergence, governance compliance, tool use, adaptability, efficiency
    - `evaluator.py`: `AgenticEvaluator` with weighted composite scoring (DCS=15%, PPA=20%, exploration=10%, reasoning=15%, convergence=10%, governance=10%, tool_use=10%, adaptability=5%, efficiency=5%)
    - `gold_standards.py`: Hand-crafted `GoldStandard` objects for all 7 demos in `ALL_GOLD_STANDARDS`
  - **3 New Specialist Agents**:
    - `pareto.py`: `design_explorer` — Pareto front non-dominated sorting, knee point identification (min normalized Euclidean distance), `ParetoPoint` model
    - `safety.py`: `safety_detector` — IEC 62304/ISO 26262/DO-178C/IEC 61508 detection, redundancy injection (dual-lockstep CPU, ECC memory, watchdog), governance gate integration
    - `experience_specialist.py`: `experience_retriever` — similarity-based episode matching (use_case 40%, platform 20%, constraints 40%), warm-start hardware candidates from prior episodes (+15 score boost)
  - **Multi-Workload Support** (`specialists.py`):
    - `_infer_scheduling()`: Concurrent/sequential/time-shared scheduling per workload
    - `aggregate_workload_requirements()`: Peak concurrent GFLOPS computation
    - `map_workloads_to_accelerators()`: Heterogeneous workload-to-hardware mapping
    - Multi-workload bonus in `_score_hardware()` for hardware supporting multiple accelerators
  - **Regression Infrastructure**:
    - `trace.py`: `TracingDispatcher` wrapping agent calls with timing/call logging, `extract_trace_from_state()` for building RunTrace from state diffs
    - `golden_traces.py`: `save_golden_trace()`, `load_golden_trace()` (JSON), `compare_traces()` with 10% PPA regression threshold, `TraceComparison` model
    - `CostTracker` in `governance.py`: Per-agent token accumulation, `estimated_cost_usd()`, `format_cost_report()`
  - **Governance Extensions** (`governance.py`):
    - `GovernancePolicy.safety_critical_actions` field
    - `GovernanceGuard.auto_detect_safety_critical()` and `flag_safety_decision()` methods
  - **State Schema Extensions** (`soc_state.py`):
    - New fields: `pareto_results`, `safety_analysis`, `prior_experience`, `cost_tracking`, `evaluation_scorecard`
  - **4 New Demo Scripts** (`examples/`):
    - `demo_dse_pareto.py` (Demo 2): Warehouse AMR with MobileNetV2 + SLAM, Pareto front across 6 hardware candidates
    - `demo_hitl_safety.py` (Demo 5): Surgical robot with IEC 62304, safety detection + governance gates
    - `demo_experience_cache.py` (Demo 6): Two-run demo — delivery drone then agricultural drone with experience retrieval
    - `demo_full_campaign.py` (Demo 7): Quadruped robot with 4 concurrent workloads (SLAM, detection, LiDAR, voice)
  - **144 Phase 4 Tests** across 9 test files:
    - `test_evaluation.py` (11): Data model validation, RunTrace/GoldStandard/Scorecard round-trips
    - `test_scoring.py` (28): All 9 scoring functions with known inputs and edge cases
    - `test_evaluator.py` (20): AgenticEvaluator with synthetic traces, batch evaluation
    - `test_pareto.py` (15): Pareto front computation, knee point, design_explorer specialist
    - `test_safety.py` (16): Safety detector, redundancy requirements, governance gates
    - `test_experience_specialist.py` (10): Experience retriever, similarity scoring, warm-start
    - `test_golden_traces.py` (13): Trace save/load/compare, regression detection
    - `test_cost_tracking.py` (14): CostTracker accumulation, cost reports
    - `test_demo_acceptance.py` (17): All 7 demos run end-to-end, composite score > 0.5
  - **564 Total Tests** all passing (144 Phase 4 + 420 existing)
  - **Demo Guide** (`docs/demo-guide.md`): Updated with Demos 2, 5, 6, 7 and architecture progression
  - Documentation: `docs/sessions/2026-02-13-phase4-evaluation-demos.md`, `docs/plans/phase4-integration-and-evaluation.md`

- Phase 3: KPU Micro-Architecture, Floorplan Validation & RTL Generation (2026-02-12):
  - **KPU Micro-Architecture Configuration** (`src/embodied_ai_architect/graphs/`):
    - `kpu_config.py`: `KPUMicroArchConfig` Pydantic model with compute tile, memory tile, DRAM, NoC sub-models
    - 3 presets (`drone_minimal`, `edge_balanced`, `server_class`) with `create_kpu_config()` factory
    - `technology.py`: Process node scaling (7nm–65nm), cell area estimation, power density models
    - `floorplan.py`: Checkerboard tile layout with pitch matching validation (±15% tolerance)
    - `bandwidth.py`: 4-link bandwidth matching through DRAM→L3→L2→L1→compute hierarchy
  - **KPU Optimization Loop** (`kpu_loop.py`):
    - `run_kpu_loop()`: Iterative config→floorplan→bandwidth convergence
    - Auto-adjusts tile counts and DRAM controllers when constraints are violated
    - `KPULoopResult` with config, floorplan, bandwidth, iteration history
  - **KPU Specialist Agents** (`kpu_specialists.py`):
    - `kpu_configurator`: Sizes KPU micro-architecture from use-case and workload
    - `floorplan_validator`: Validates checkerboard pitch matching
    - `bandwidth_validator`: Validates memory hierarchy bandwidth balance
    - `kpu_optimizer`: Adjusts config when floorplan or bandwidth fails
  - **EDA Toolchain** (`eda_tools/`):
    - `lint.py`: Verilator lint with mock fallback
    - `synthesis.py`: Yosys synthesis with cell count extraction and mock fallback
    - `simulation.py`: Icarus Verilog simulation with mock fallback
    - `toolchain.py`: Unified `run_eda_pipeline()` orchestrating lint→synthesis→simulation
  - **RTL Template Engine** (`rtl_templates/`):
    - Jinja2 templates for 12 KPU sub-components (MAC unit, systolic array, compute tile, memory tile, L1/L2/L3 SRAM, skew buffer, block mover, DMA, router, top-level)
    - `RTLTemplateEngine` with `get_kpu_components()` deriving template params from KPU config
  - **RTL Inner Loop** (`rtl_loop.py`):
    - `run_rtl_loop()`: lint→synthesize→validate per module
    - `RTLLoopResult` with success, metrics (area_cells, area_um2), iteration history
  - **RTL Specialist Agents** (`rtl_specialists.py`):
    - `rtl_generator`: Renders templates for all KPU sub-components, runs EDA pipeline
    - `rtl_ppa_assessor`: Aggregates synthesis cell counts→area via technology scaling
  - **State Schema Extensions** (`soc_state.py`):
    - New fields: `rtl_enabled`, `kpu_config`, `floorplan_estimate`, `bandwidth_match`, `kpu_optimization_history`, `rtl_modules`, `rtl_testbenches`, `rtl_synthesis_results`
    - Backward compatible: all new fields default to empty/False
  - **Experience Episode Extensions** (`experience.py`):
    - New fields: `kpu_config_name`, `kpu_process_nm`, `floorplan_area_mm2`, `bandwidth_balanced`, `rtl_modules_generated`, `rtl_total_cells`
  - **Demo 4** (`examples/demo_kpu_rtl.py`):
    - Full 4-level flow: Outer Loop→KPU Config→Floorplan+Bandwidth→RTL Generation
    - 10-task static plan with parallel floorplan/bandwidth validation
    - Prints KPU config, floorplan pitch matching, bandwidth links, RTL synthesis cell counts, PPA summary
  - **Demo Guide** (`docs/demo-guide.md`):
    - Comprehensive guide covering all 3 demos and 3 utility examples
    - Usage, CLI options, output sections, prerequisites, architecture progression
  - **88 Phase 3 Tests** across 10 test files:
    - `test_technology.py` (6): Process node scaling, area estimation
    - `test_kpu_config.py` (7): Presets, factory, model validation
    - `test_floorplan.py` (7): Tile sizing, pitch matching, area budgets
    - `test_bandwidth.py` (6): Link bandwidth, bottleneck detection
    - `test_eda_tools.py` (7): Lint, synthesis, simulation with mocks
    - `test_rtl_templates.py` (5): Template rendering, component coverage
    - `test_kpu_specialists.py` (8): All 4 KPU specialists
    - `test_kpu_loop.py` (5): Loop convergence, tight constraints
    - `test_rtl_loop.py` (5): RTL pipeline, metrics extraction
    - `test_rtl_specialists.py` (6): RTL generator, PPA assessor
    - `test_kpu_integration.py` (6): End-to-end KPU loop convergence
    - `test_rtl_integration.py` (5): Template→RTL→synthesis pipeline
  - **207 Total Tests** all passing (88 Phase 3 + 119 existing)
  - Documentation: `docs/sessions/2026-02-12-phase3-kpu-rtl.md`, `docs/plans/phase3-kpu-floorplan-rtl-generation.md`

- Agentic SoC Designer Foundation — Phase 0 + Phase 1 Implementation (2026-02-11):
  - **Phase 0.1 — State Schema & Task Graph Engine** (`src/embodied_ai_architect/graphs/`):
    - `task_graph.py`: DAG engine with `TaskNode`, `TaskGraph`, `TaskStatus`, `CycleError`
    - Dependency tracking, lifecycle management (PENDING→READY→RUNNING→COMPLETED/FAILED)
    - Cycle detection, batch add with topological sort, serialization round-trip
    - `soc_state.py`: `SoCDesignState` TypedDict — the single state object flowing through the design pipeline
    - `DesignConstraints` model (power, latency, area, cost, process node, safety)
    - `PPAMetrics` model with per-constraint verdicts (PASS/FAIL)
    - `DesignDecision` model for audit trail
    - Helper functions: `get_task_graph()`, `set_task_graph()`, `record_decision()`, `get_iteration_summary()`
  - **Phase 0.2 — Planner & Dispatcher** (`src/embodied_ai_architect/graphs/`):
    - `planner.py`: `PlannerNode` — LLM-powered goal decomposition into TaskGraph DAGs
    - System prompt encoding SoC design decomposition expertise with 7 specialist agents
    - Structured JSON output parsing with markdown fence stripping
    - Supports static plans for deterministic testing
    - `dispatcher.py`: `Dispatcher` — DAG-walking task scheduler replacing hardcoded Orchestrator
    - Finds ready tasks, dispatches to registered agent executors, records decisions
    - Failure handling, blocked-task detection, max-steps safety bound
    - `_state_updates` merging: task results can write to top-level state fields
  - **Phase 1 — Specialist Agents** (`src/embodied_ai_architect/graphs/specialists.py`):
    - `workload_analyzer`: Keyword-based workload estimation from goal description
      (detection, tracking, SLAM, voice, LiDAR → operator profiles with GFLOPS/memory estimates)
    - `hw_explorer`: Static hardware catalog (KPU, Jetson Orin Nano, Coral Edge TPU, AMD Ryzen AI,
      Raspberry Pi 5, Hailo-8) with constraint-aware scoring and verdict annotations
    - `architecture_composer`: Operator-to-hardware mapping, IP block generation (CPU, accelerator,
      memory controller, I/O, ISP), memory map layout, interconnect topology (AXI4/NoC)
    - `ppa_assessor`: Power/latency/area/cost estimation with per-constraint PASS/FAIL verdicts,
      bottleneck identification, and improvement suggestions
    - `critic`: Design review identifying single-compute risk, power margin analysis,
      memory pressure, safety-critical compliance checks
    - `report_generator`: Structured design report with executive summary, all artifacts, decision trail
    - `create_default_dispatcher()`: Factory wiring all 6 specialists into a ready-to-run Dispatcher
  - **End-to-End Pipeline Validated**:
    - Drone SoC design: goal → planner → workload → hw explore → compose → PPA → critic → report
    - KPU selected as top candidate for <5W/<$30 drone constraint profile
    - AMR SoC design: multi-workload detection (SLAM + detection + voice + LiDAR)
    - Full decision audit trail (7+ decisions per design session)
  - **108 Tests** across 3 test files:
    - `tests/test_soc_state.py` (52): TaskGraph DAG ops, lifecycle, serialization, state helpers
    - `tests/test_planner_dispatcher.py` (28): Plan parsing, LLM integration (mocked), dispatch loop
    - `tests/test_specialists.py` (28): Each specialist, factory, full pipeline integration
  - Documentation: `docs/sessions/2026-02-11-agentic-phase0-phase1.md`

- Agentic System Implementation Plan (2026-02-10):
  - **Current State Assessment**: Honest analysis of the two existing execution modes
    (hardcoded Orchestrator pipeline and LLM tool-use chat loop) and why neither
    qualifies as a true agentic system
  - **Gap Analysis**: 12-dimension gap matrix covering goal decomposition, task graph,
    orchestration, memory, design space exploration, optimization loops, governance,
    constraint propagation, domain knowledge, validation, feedback learning, and RTL generation
  - **CrewAI vs. LangGraph Analysis**: Evaluation of CrewAI's specialist agent model
    (RTL Designer, PPA Assessor, etc.) vs. LangGraph state graph approach. Recommendation:
    adopt the specialist role *concept*, implement on LangGraph with typed state and checkpointing
  - **4-Phase Implementation Plan** (~24 weeks):
    - Phase 0: Foundation (state schema, task graph engine, planner, dispatcher)
    - Phase 1: Specialist agents (workload analyzer, HW explorer, PPA assessor, critic, etc.)
    - Phase 2: Memory + governance (working memory, experience cache, audit trail, HITL)
    - Phase 3: EDA integration (containerized Yosys/Verilator/OpenROAD, RTL generation loop)
  - **7 Demo Prompts** for validation: delivery drone SoC, 3-way hardware Pareto comparison,
    iterative power optimization, RTL MAC unit generation, safety-critical surgical robot,
    experience cache reuse, full end-to-end quadruped robot design campaign
  - **9-Dimension Evaluation Framework**: task decomposition quality, PPA accuracy,
    exploration efficiency, reasoning quality, convergence behavior, governance compliance,
    tool use accuracy, adaptability, session efficiency -- with composite scorecard
  - Documentation: `docs/sessions/2026-02-10-agentic-system-plan.md`

- Documentation Site Implementation (2026-01-15):
  - **Astro Starlight Docs Site** (`docs-site/`):
    - Complete Phase 1 implementation of documentation roadmap
    - 20 pages: landing, getting started, features, catalog, reference, tutorials, troubleshooting
    - GitHub Pages deployment with automatic base path handling via rehype-rewrite plugin
    - `llms.txt` for AI agent discovery
    - Custom branding with logo SVGs and hero image
  - **GitHub Actions Workflow** (`.github/workflows/deploy-docs.yml`):
    - Automatic deployment to GitHub Pages on push to main
    - Build artifact upload for Vercel/Netlify alternatives
  - **Value Proposition Messaging**:
    - Refined landing page to convey agentic AI identity
    - Emphasis on custom + COTS hardware differentiation
    - Tesla FSD example for competitive advantage illustration
    - Competitive intelligence and build-vs-buy guidance positioning
  - Documentation: `docs/sessions/2026-01-15-docs-site-implementation.md`

### Fixed
- Duplicate Tool Name Bug (2026-01-15):
  - Renamed `identify_bottleneck` in `architecture_tools.py` to `identify_architecture_bottleneck`
  - Both `graphs_tools.py` and `architecture_tools.py` had tools with same name
  - Fixed 17 of 18 test failures in `test_verdict_tools_cli.py`

- Knowledge Base Architecture and Documentation Roadmap (2026-01-14):
  - **LLM-Native Knowledge Base Architecture** (`docs/knowledge-base-architecture.md`):
    - Text-first design: LLMs reason over markdown/YAML, not SQL queries
    - Three-tier content model: Structured Catalog, Analytical Data, Experiential Knowledge
    - Vector retrieval with semantic search for context injection
    - Tools only for computation (roofline analysis); everything else is text
    - Token budget management (~8K tokens default for retrieved context)
    - Content chunking strategy: one file = one chunk (~500-2000 tokens)
  - **Documentation Site Roadmap** (`docs/plans/roadmap-documentation.md`):
    - Astro Starlight for modern, LLM-friendly static documentation
    - 12-week implementation plan across 5 phases
    - Phase 1: Foundation (landing page, getting started)
    - Phase 2: Catalog (auto-generated from embodied-schemas)
    - Phase 3: Tutorials (7 deployment/optimization guides)
    - Phase 4: Reference (CLI, MCP tools, API docs)
    - Phase 5: Interactive (hardware finder, compatibility matrix)
    - `llms.txt` and `llms-full.txt` generation for AI agent discovery
    - Single source of truth: YAML → website + knowledge base + LLM metadata
  - **Phase 1 Roadmap** (`docs/plans/roadmap-phase-1.md`):
    - Complete user stories and acceptance criteria for Digital Twin Foundation
    - Milestone 1.1: Coupled Simulation Pipeline (AirSim ↔ CSim integration)
    - Milestone 1.2: Baseline Profiling (10,000+ flight minutes dataset)
    - Monthly breakdown and success criteria
  - **Roadmap Summary** (`docs/plans/roadmap-summary.md`):
    - Complete 12-month product roadmap overview
    - All 3 phases with milestones and acceptance criteria
    - Team structure (6 people, AI-leveraged)
    - Supporting infrastructure roadmaps
    - Risk matrix and end state definition
  - Updated main roadmap with Supporting Infrastructure Roadmaps section
  - Documentation: `docs/sessions/2026-01-14-knowledge-base-and-documentation-roadmap.md`
- Architecture Benchmark Harness and Runnable Operators (2026-01-03):
  - **Benchmark Harness** (`src/embodied_ai_architect/benchmark/`):
    - `ArchitectureRunner`: Executes complete architecture pipelines with per-operator timing
    - `PowerMonitor`: Power measurement via Intel RAPL, AMD SMU, or external meters
    - `ArchitectureBenchmarkResult`: Comprehensive results with timing, power, and requirement validation
    - Automatic requirement checking against architecture specs (latency, throughput, power)
  - **Runnable Operators** (`src/embodied_ai_architect/operators/`):
    - Base `Operator` ABC with `setup()`, `process()`, `teardown()`, `benchmark()` interface
    - Perception: `YOLOv8ONNX` (NPU support), `ImagePreprocessor`, `ByteTrack`, `SceneGraphManager`
    - State Estimation: `EKF6DOF`, `TrajectoryPredictor`, `CollisionDetector`
    - Control: `PIDController`, `PathFollower`, `PathPlannerAStar`
    - Operator registry with 14 operators mapped to embodied-schemas catalog IDs
  - **CLI Commands**:
    - `embodied-ai benchmark arch <id>`: Benchmark architecture pipeline
    - `embodied-ai benchmark arch-list`: List available architectures
    - JSON output support, result export to file
  - **Test Results**: All 3 reference architectures benchmark successfully
    - `pick_and_place_v1`: 24.4 fps (target: 10 fps)
    - `drone_perception_v1`: 15.6 fps (target: 30 fps)
    - `simple_adas_v1`: All operators functional
  - Documentation: `docs/benchmark-harness.md`, `docs/sessions/2026-01-03-benchmark-harness-operators.md`
- Operator Benchmarking Infrastructure with Graphs Auto-Detect Integration (2026-01-02):
  - Created `benchmarks/operators/` framework for systematic operator profiling
  - Base classes: `OperatorBenchmark`, `OperatorBenchmarkResult` with standardized timing methodology
  - Benchmark runner with warmup, iterations, percentile statistics (p50, p95, p99)
  - Perception benchmarks: YOLO detector (PyTorch/ONNX), ByteTrack, ImagePreprocessor
  - State estimation benchmarks: Kalman filter, Scene graph manager, EKF 6-DOF
  - Control benchmarks: PID controller, A* path planner, Trajectory follower
  - Integrated with `graphs.hardware.calibration.auto_detect` for SHA fingerprinting
  - Hardware fingerprint: SHA256 of hw identity (CPU stepping, microcode, GPU VBIOS)
  - Software fingerprint: SHA256 of sw stack (PyTorch, CUDA, drivers, etc.)
  - NPU detection: ONNX Runtime providers (RyzenAI, QNN, CoreML), AMD XDNA driver
  - Fallback to simplified detection when graphs not installed
  - Catalog updater for embodied-schemas operator YAML files
  - CLI script `run_operator_benchmarks.py` with `--detect` flag
  - Fingerprints included in YAML/JSON benchmark results for reproducibility
  - Documentation: docs/sessions/2026-01-02-operator-benchmarking-graphs-integration.md
- SoC Optimizer LangGraph Workflow Improvements (2025-12-23):
  - Fixed GraphRecursionError by adding dynamic recursion_limit to LangGraph config
  - Formula: `(max_iterations + 1) * 6 + 10` accounts for 5 nodes per iteration plus syntax retry loops
  - Verified full optimization loop works with Claude LLM integration
  - Identified need for functional correctness gate (test count validation)
  - Identified need for composite PPA scoring (not just area)
  - Documentation: docs/sessions/2025-12-23-soc-optimizer-recursion-fix.md
- Cross-Repository Claude Documentation (2025-12-21):
  - Created CLAUDE.md for embodied-schemas repository
  - Documents repository role as shared dependency for graphs and embodied-ai-architect
  - Documents data split (datasheet specs vs roofline/calibration)
  - Schema design patterns (verdict-first outputs, ID conventions)
  - Guidelines for adding new data and making schema changes
  - Updated embodied-ai-architect/CLAUDE.md with Related Repositories section
  - Updated graphs/CLAUDE.md with Related Repositories section and data split table
  - Each repo's CLAUDE.md now establishes cross-repo awareness
  - Documentation: docs/sessions/2025-12-21-cross-repo-claude-documentation.md
- Agentic AI Architecture Design (2025-12-20):
  - Tool registration pattern analysis and recommendations
  - Subtask decomposition design for embodied AI codesign
  - Dimension-specific tool granularity decision (analyze_latency, analyze_power, etc.)
  - Verdict-first tool output schema design (PASS/FAIL with confidence, evidence, suggestions)
  - Knowledge base architecture with hybrid storage (YAML + optional graph + vector)
  - Research on current agentic memory systems (A-MEM, Zep/Graphiti, Mem0)
  - Documentation: docs/plans/agentic-tool-architecture.md
- Embodied AI Codesign Subtask Enumeration (2025-12-20):
  - 70+ discrete subtasks organized into 7 categories
  - Knowledge Base subtasks (21): hardware, models, sensors, use cases, constraints
  - Analysis Tool subtasks (19): latency, power, memory, accuracy, cost, physical
  - Recommendation Tool subtasks (12): hardware, model, sensor, software stack
  - Synthesis Tool subtasks (11): trade-offs, configuration, reports
  - Decomposition & Planning subtasks (14): query understanding, orchestration
  - Validation subtasks (10): benchmark execution, accuracy verification
  - Phased implementation plan (Foundation → Recommendation → Validation → Advanced)
  - Documentation: docs/plans/embodied-ai-codesign-subtasks.md
- Knowledge Base Schema Design (2025-12-20):
  - Complete schema for hardware platforms (physical, environmental, power, interfaces)
  - Model catalog schema (architecture, variants, accuracy benchmarks)
  - Sensor catalog schema (cameras, depth, LiDAR)
  - Use case template schema (constraints, success criteria, recommendations)
  - Benchmark result schema (verdict-first output format)
  - Constraint ontology (latency tiers, power classes, implication rules)
  - Directory structure design for YAML data catalog
  - Query interface design with filtering and relationships
  - Documentation: docs/plans/knowledge-base-schema.md
- Shared Schema Repository Architecture (2025-12-20):
  - Created branes-ai/embodied-schemas as shared package
  - Dependency flow: embodied-schemas ← graphs, embodied-ai-architect
  - Data split decision between graphs/hardware_registry and embodied-schemas
  - Migration strategy for existing graphs hardware data
  - Integration patterns with code examples
  - Documentation: docs/plans/shared-schema-repo-architecture.md
- Created branes-ai/embodied-schemas repository:
  - Complete Pydantic schema models (hardware, models, sensors, usecases, benchmarks, constraints)
  - YAML data loaders with validation
  - Registry API for unified data access
  - Data directory structure for hardware, chips, models, sensors, usecases
  - 15 passing tests for schema validation
  - pyproject.toml, README.md, LICENSE, .gitignore
  - Session documentation: embodied-schemas/docs/sessions/2025-12-20-initial-setup.md
- Design Assistant CLI Verification (2025-12-18):
  - Verified all 10 LLM tools are functional and properly integrated
  - Confirmed CLI `chat` command is registered and accessible
  - Validated hardware catalog with 30+ targets across 7 categories
  - Tested tool executor argument handling and error reporting
  - Documentation: docs/sessions/2025-12-18-chat-cli-verification.md
- Interactive Chat Interface: Claude Code-style conversational agent for AI architecture design
  - LLM Integration Layer (src/embodied_ai_architect/llm/):
    - LLMClient: Claude API wrapper with tool use support
    - ArchitectAgent: Agentic loop that reasons and calls tools iteratively
    - Tool definitions wrapping existing agents (analyze_model, recommend_hardware, run_benchmark)
    - File exploration tools (list_files, read_file)
  - CLI Command: `embodied-ai chat` for interactive sessions
    - Rich terminal UI with spinners and formatted output
    - Tool execution visibility (shows tool calls and results)
    - Session commands: exit, reset, help
    - Verbose mode for detailed execution info
  - branes-ai/graphs Integration (src/embodied_ai_architect/llm/graphs_tools.py):
    - analyze_model_detailed: Roofline-based analysis with latency, energy, memory, utilization
    - compare_hardware_targets: Multi-hardware comparison (30+ targets) ranked by speed/efficiency
    - identify_bottleneck: Compute vs memory bound analysis with recommendations
    - list_available_hardware: Hardware catalog by category (datacenter GPU, edge GPU, CPU, TPU, accelerators, automotive)
    - estimate_power_consumption: Power and energy estimation for deployment planning
  - System Prompt: Domain-specific guidance for embodied AI architecture assistance
  - 10 total tools available for comprehensive model-to-hardware analysis
  - Documentation: docs/interactive-chat.md with architecture, examples, and usage guide
  - Optional dependency: `pip install -e ".[chat]"` for anthropic package
- CLAUDE.md: Claude Code guidance file for repository onboarding
  - Project overview and purpose
  - Build, test, and development commands
  - Architecture documentation (Orchestrator, agents, backends, CLI)
  - Prototype documentation (drone_perception, multi_rate_framework)
  - Key design patterns and code style guidelines
- Initial project structure with Python package layout
- Documentation directories: `docs/plans` for architecture plans, `docs/sessions` for session logs
- Architecture plan: Multi-agent system design for Embodied AI evaluation (docs/plans/agent-architecture.md)
- Agent packaging strategy document (docs/plans/agent-packaging-strategy.md)
- Core Orchestrator class for coordinating agent workflows
- BaseAgent abstract class and AgentResult data model
- ModelAnalyzerAgent: Extracts architecture information from PyTorch models
- BenchmarkAgent: Performance profiling with extensible backend architecture
  - Backend plugin system for dispatching benchmarks to different execution environments
  - LocalCPUBackend: Measures inference latency and throughput on CPU
  - Support for future backends (GPU, remote cluster, edge devices, robots)
- HardwareProfileAgent: Hardware recommendation system with comprehensive knowledge base
  - Knowledge base with 8 hardware profiles (CPUs, GPUs, TPUs, edge devices)
  - Profiles include: Intel Xeon, NVIDIA A100, NVIDIA Jetson AGX Orin, Google TPU v4, Google Coral Edge TPU, Intel NCS2, Raspberry Pi 4, AMD MI250X
  - Scoring algorithm for hardware-model fitness based on memory, power, operations, and compute
  - Constraint-aware recommendations (latency, power, cost)
  - Use-case filtering (edge, cloud, datacenter, embedded)
  - Detailed reasoning and warnings for each recommendation
- ReportSynthesisAgent: Comprehensive report generation with visualizations
  - Filesystem-based architecture (Producer pattern) - generates artifacts, doesn't serve HTTP
  - Multiple output formats: HTML (human-readable), JSON (machine-readable)
  - Matplotlib-based visualizations: hardware comparison bar chart, layer distribution pie chart
  - Executive summary with key metrics and constraint satisfaction
  - Automated insights generation from workflow data
  - Actionable recommendations based on results
  - Jinja2-templated HTML reports with embedded CSS
  - Reports saved to `./reports/{workflow_id}/` directory structure
- Optional Report Server: Simple HTTP server for viewing reports (separate from orchestrator)
  - Static file serving from reports directory
  - Auto-generated index page listing all reports
  - Run independently: `python -m embodied_ai_architect.report_server`
  - Clean separation of concerns: generation vs. serving
- Architecture documentation: Reporting architecture patterns and design (docs/plans/reporting-architecture.md)
- Security Architecture: Comprehensive secrets management system
  - SecretsManager: Multi-provider secrets management with audit logging
  - EnvironmentSecretsProvider: Load secrets from environment variables
  - FileSecretsProvider: Load secrets from files with permission validation
  - Automatic secret masking in logs and error messages
  - Configuration reference resolution (${secret:key}, ${env:VAR})
  - Audit trail for all secret accesses
  - Secure defaults: file permission checks, ownership validation
  - .env.example template for secure configuration
  - Updated .gitignore to prevent accidental secret commits
  - Security architecture documentation (docs/plans/security-architecture.md)
- RemoteSSHBackend: Execute benchmarks on remote machines via SSH
  - Secure SSH connection using SecretsManager
  - Model serialization and transfer via SFTP
  - Remote code execution with proper cleanup
  - Result retrieval and parsing
  - Optional dependency (install with [remote] extra)
  - Demonstrates proper secrets handling pattern
- Configuration management examples
  - examples/remote_benchmark_example.py: Secure remote benchmarking demo
  - Security features demonstration (masking, audit, multi-provider)
  - Graceful handling of missing credentials
- KubernetesBackend: Cloud-native benchmarking with horizontal scaling
  - Job-based execution (create, monitor, cleanup)
  - Parallel benchmark execution (horizontal scaling)
  - GPU allocation with node selectors
  - Resource management (CPU/memory requests and limits)
  - ConfigMap for model data storage
  - Automatic cleanup with TTL
  - Secure RBAC configuration
  - Optional dependency (install with [kubernetes] extra)
  - Kubernetes manifests (namespace, service account, RBAC)
  - Configuration examples and README (config/kubernetes/)
  - examples/kubernetes_scaling_example.py: Demonstrates parallel execution
- Architecture documentation: Kubernetes backend design (docs/plans/kubernetes-backend-architecture.md)
- Working example demonstrating full workflow with all four agents (examples/simple_workflow.py)
- Project dependencies via pyproject.toml and requirements.txt (added matplotlib, jinja2, paramiko, kubernetes as optional)
- This CHANGELOG.md file to track project changes
- Command-Line Interface (CLI): Human-friendly CLI inspired by Claude Code
  - Architecture documentation: CLI design patterns and commands (docs/plans/cli-architecture.md)
  - Core CLI framework using Click and Rich for beautiful terminal output
  - Entry point: `embodied-ai` command registered in pyproject.toml
  - Global options: --verbose/-v, --json, --quiet for different output modes
  - Rich output with panels, tables, progress indicators, and colored text
  - Workflow commands: Complete end-to-end model evaluation pipeline
    - `embodied-ai workflow run <model>`: Run full workflow (analyze, profile, benchmark, report)
    - `embodied-ai workflow list`: List past workflow executions
  - Model analysis commands:
    - `embodied-ai analyze <model>`: Analyze model architecture and complexity
    - Layer breakdown table with counts by type
    - Parameter and memory statistics
  - Benchmarking commands:
    - `embodied-ai benchmark run <model>`: Run performance benchmarks
    - `embodied-ai benchmark list`: List available backends
    - Progress indicators for long-running benchmarks
    - Support for --backend, --iterations, --warmup options
  - Report management commands:
    - `embodied-ai report view <workflow_id>`: Open report in browser
    - `embodied-ai report list`: List all available reports
    - `embodied-ai report compare <id1> <id2>`: Compare two reports
  - Configuration commands:
    - `embodied-ai config init`: Initialize configuration file
    - `embodied-ai config show`: Display current configuration with syntax highlighting
    - `embodied-ai config validate`: Validate configuration
  - Backend management commands:
    - `embodied-ai backends list`: List available backends with status
    - `embodied-ai backends test <backend>`: Test backend connection
    - Shows installation requirements for optional backends
  - Secrets management commands:
    - `embodied-ai secrets list`: List secrets (keys only, not values)
    - `embodied-ai secrets validate`: Validate secrets configuration
    - Security-focused: never displays secret values
  - Error handling with helpful messages and tips
  - JSON output mode for scripting and CI/CD integration
  - Dependencies: click>=8.1.0, rich>=13.0.0
- Embodied AI Application Framework: Architecture analysis and implementation planning
  - Architecture analysis document: Evaluated 3 architecture options for application support (docs/plans/embodied-ai-application-architecture.md)
  - Selected hybrid architecture (Option 3): Python Framework + Graph IR + DSL
  - Comprehensive implementation plan: 18-week timeline for core functionality (docs/plans/embodied-ai-application-implementation-plan.md)
  - Planned integration with existing infrastructure:
    - PyTorch FX for graph capture
    - MLIR/IREE for compilation
    - Existing simulators as benchmark backends
    - Existing compiler/runtime modeler for hardware mapping
  - Application framework will support heterogeneous operators:
    - DNNs (PyTorch models)
    - Classical algorithms (Kalman filters, PID controllers)
    - Path planners (RRT*, A*)
    - Custom operators
  - Complete control loop applications (sensors → processing → decision → actuation)
  - End-to-end benchmarking with real datasets (ROS bags, custom formats)
  - Graph IR for analysis and optimization
  - DSL serialization for storage and portability
  - Future: C++/Rust transpilation for production deployment
  - Planned reference applications: drone navigation, robot manipulation, autonomous vehicle, legged robot, humanoid balance
- Multi-Rate Control Framework: Deep research and architecture proposal
  - Comprehensive research document on multi-rate control systems (docs/research/multi-rate-control-architecture.md)
  - Analyzed requirements for humanoid/quadruped robots:
    - Joint control: 100-1000 Hz (tight sensor-control-actuator loops)
    - Perception: 30-60 Hz (vision, state estimation)
    - Planning: 1-10 Hz (path planning, high-level decisions)
  - Selected Zenoh as communication backbone:
    - Zero-copy, real-time pub/sub
    - Automotive-grade certified
    - Open source (Apache 2.0)
    - Multi-transport (shared memory, UDP, DDS bridge)
    - Superior to ROS2/DDS for real-time requirements
  - Evaluated 3 multi-rate scheduling patterns:
    - Time-triggered (deterministic, simple)
    - Data-triggered (event-driven, low latency)
    - Hybrid (selected): Time-triggered with async data caching
  - Designed component model with declarative I/O:
    - `@control_loop` decorator for rate specification
    - `Input`/`Output` descriptors for automatic Zenoh wiring
    - Explicit rate, priority, deadline, CPU affinity
  - Hybrid Python/Rust execution strategy:
    - Python for slow loops (≤ 100 Hz): perception, planning
    - Rust for fast loops (> 500 Hz): joint control, balance
    - All communication via Zenoh (language-agnostic)
  - Phase 1 implementation proposal (docs/plans/phase1-framework-architecture.md):
    - Core framework classes (Component, Application, Scheduler)
    - Multi-rate scheduler (one thread per component)
    - Automatic Zenoh pub/sub creation
    - Deadline monitoring and diagnostics
    - 8-week implementation timeline
  - Example quadruped robot architecture with 4 control rates
  - Real-time considerations (RT-PREEMPT, CPU isolation, zero-copy)
- Multi-Rate Framework Prototype: Working validation of architecture
  - Created minimal working prototype (prototypes/multi_rate_framework/)
  - Implemented core framework classes (~400 lines):
    - Component base class with lifecycle hooks
    - @control_loop decorator for rate specification
    - Input/Output descriptors for automatic Zenoh wiring
    - MultiRateScheduler with thread-per-component
    - Application container
  - Successfully integrated Zenoh (eclipse-zenoh 1.6.2)
  - Created two working examples:
    - Simple: 100 Hz producer, 1 Hz consumer
    - Multi-rate: 100 Hz sensor, 10 Hz control, 1 Hz planning
  - Validation tests passed (test_framework.py):
    - Decorators work correctly
    - Input/Output descriptors functional
    - Component lifecycle correct
    - Timing accuracy < 1% (100.1ms vs 100ms expected)
  - Architecture validated for full implementation
  - Demonstrates hybrid time/data triggered pattern
  - Multi-threading with independent rates working
  - Clean declarative API validated
- Drone Perception Pipeline: Complete end-to-end perception system (prototypes/drone_perception/)
  - **Object Detection**: YOLOv8 integration with multiple model sizes (nano to xlarge)
    - CPU and CUDA support
    - Class filtering and confidence thresholds
    - Batch processing capability
  - **Multi-Object Tracking**: ByteTrack implementation with re-identification
    - ID persistence across frames and occlusions
    - Two-stage association (high-conf + low-conf detections)
    - Kalman filter for 2D bbox tracking (8D state)
    - Lost track recovery and management
  - **3D Scene Graph**: World state management with position/velocity/acceleration
    - 9D Kalman filtering per object [x, y, z, vx, vy, vz, ax, ay, az]
    - Constant acceleration motion model
    - Depth estimation: heuristic (bbox height) with stereo/LiDAR ready
    - Object pruning with time-to-live
  - **Sensor Abstraction**: Progressive complexity support
    - BaseSensor interface for all camera types
    - MonocularCamera: Video file and webcam input with FPS control
    - Ready for stereo (RealSense, OAK-D) and LiDAR extensions
  - **Visualization**: Real-time 3D matplotlib viewer
    - Object positions, velocities, and trajectories
    - Interactive controls and screenshot saving
    - Dual view (2D video + 3D scene graph)
  - **Common Data Structures**: Frame, Detection, Track, TrackedObject
    - 2D→3D projection utilities
    - Bounding box operations (IOU, conversion)
    - Camera intrinsics handling
  - **Examples**: Full pipeline and simple detection demos
    - Live processing with dual visualization
    - Performance monitoring (FPS, detection/track counts)
    - Video saving capability
  - **Documentation**: Architecture, quick start, session summary
    - Design principles (sensor progression, visual channel approach)
    - Progressive implementation roadmap
    - Integration path to multi-rate framework
  - **Performance**: 20-30 FPS on CPU, 60+ FPS on GPU (YOLOv8n)
- Drone Perception Test Infrastructure: Comprehensive testing and validation system
  - **Test Data Structure**: Organized directory hierarchy
    - Categories: traffic, pedestrian, mixed, visdrone, synthetic
    - Separate folders for videos, annotations, results, recordings
    - Git-safe (videos excluded, metadata tracked)
  - **Video Catalog**: Curated test video collection (test_data/video_catalog.yaml)
    - 9 curated videos with detailed metadata
    - Sources: VisDrone (academic), Pixabay, Mixkit, Videezy (free stock)
    - Test suites: quick (~50MB), traffic_focus, pedestrian_focus, comprehensive (~200MB)
    - Metadata: duration, resolution, FPS, features, license, download method
  - **Download Script**: Automated test video acquisition (scripts/download_test_videos.py)
    - Download by suite, category, or individual video
    - Progress bars and error handling
    - YouTube support (via yt-dlp)
    - Direct HTTP downloads from free stock sites
  - **Test Runner**: Automated pipeline validation (scripts/run_test_suite.py)
    - Run on video suites or individual files
    - Metrics collection: FPS, detection counts, track counts, 3D objects
    - Per-frame statistics and timing
    - JSON output for regression testing
  - **Baseline Targets**: Expected performance metrics
    - Detection recall: cars 85%, person 80%, bicycle 75%
    - Tracking: <2 ID switches per 100 frames, 25+ frame persistence
    - Speed: 20-30 FPS CPU, 60-100 FPS GPU (YOLOv8n)
  - **Documentation**: Test data setup guide, scripts README
    - Quick start workflow
    - CI/CD integration examples
    - Troubleshooting guide
- Drone Perception Pipeline Phase 2: Stereo camera support and depth fusion
  - **Stereo Camera Integration** (sensors/stereo.py):
    - StereoCamera class with dual backend support (RealSense D435, OAK-D)
    - Real-time depth stream acquisition at 30 FPS
    - Automatic depth-to-color alignment
    - Depth scale calibration and camera intrinsics
    - Robust depth extraction with median filtering (center, median, bottom methods)
    - Valid depth range validation (0.1-20m)
    - StereoRecordedCamera for playback testing without hardware
  - **Depth Map Generation** (scripts/generate_depth_maps.py):
    - MiDaS monocular depth estimation integration
    - Three model options: dpt_large, dpt_hybrid, midas_small
    - GPU acceleration (CUDA, MPS for Apple Silicon)
    - Batch processing mode for test suites
    - Physical depth normalization (0-20m range)
    - YAML metadata generation (source, model, scale, resolution)
    - Optional live preview during generation
  - **Stereo Pipeline Example** (examples/stereo_pipeline.py):
    - Dedicated stereo demonstration with all three backends
    - Multiple depth extraction methods (configurable)
    - Live depth visualization toggle
    - Interactive controls (pause, depth view, screenshot)
    - Real-time metrics display
    - Integration with existing detection/tracking
  - **Full Pipeline Integration**:
    - Updated full_pipeline.py with --stereo and --stereo-backend flags
    - Automatic sensor mode detection
    - Depth mode switching (heuristic vs. stereo)
    - Compatible with existing monocular workflow
    - Shared visualization and tracking components
  - **Test Suite Infrastructure**:
    - run_stereo_test_suite.sh: Automated testing for all videos
    - Auto-generates depth maps if missing
    - Parallel test structure to monocular suite
    - Separate logs and outputs for comparison
  - **Comparison Tools** (scripts/compare_mono_vs_stereo.py):
    - Parse and compare monocular vs. stereo log files
    - Metrics: FPS, detection counts, tracking stability, 3D object counts
    - Statistical analysis (mean, std, min, max)
    - Batch mode for all videos
    - Quantified improvements display
  - **Validation Tools** (scripts/validate_stereo_accuracy.py):
    - Interactive measurement mode (click to measure depth)
    - Known distance testing with ground truth validation
    - Temporal consistency analysis (drift, stability)
    - Statistical depth measurement (20+ samples)
    - Real-time depth map visualization
  - **Documentation**:
    - STEREO_TESTING.md: Comprehensive testing guide
    - Updated README.md with stereo usage examples
    - Updated ARCHITECTURE.md marking Phase 2 complete
    - Updated test_data/README.md with stereo data format
    - Session log: docs/sessions/2025-11-20-drone-perception-phase2-stereo.md
  - **Dependencies**: Added pyrealsense2>=2.54.0, depthai>=2.23.0
  - **Performance**: ~20-25 FPS stereo (10-20% overhead), ~30% more stable tracking
  - **Testing**: Works with/without hardware using synthetic MiDaS depth maps
- Research Documentation: Drone architecture and design principles
  - Drone pipeline research (docs/research/drone-pipeline.md)
    - Hardware recommendations (Qualcomm RB5, Hailo-8, OAK-D)
    - Visual tracking vs digital twin comparison
    - Energy efficiency analysis (5-8W vs 30-60W SLAM)
  - SoC design assistant principles (docs/research/design-assistant.md)
    - AI assistant architecture for chip design
    - HW/SW co-design methodology
    - Design space exploration principles

### Changed

### Deprecated

### Removed

### Fixed

### Security

---

## Guidelines for Changelog Entries

### Categories
- **Added**: New features
- **Changed**: Changes to existing functionality
- **Deprecated**: Soon-to-be removed features
- **Removed**: Removed features
- **Fixed**: Bug fixes
- **Security**: Vulnerability fixes

### Format
Each release should include:
- Version number following semantic versioning (MAJOR.MINOR.PATCH)
- Release date in YYYY-MM-DD format
- Organized list of changes by category
