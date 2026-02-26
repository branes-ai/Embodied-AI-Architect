# Embodied AI Architect — Product Roadmap v2

> **Baseline:** v0.6.0 (Feb 2026) | **Target:** First paying customer by Release 1.0
> **Timeline:** 18 months (Mar 2026 – Aug 2027)
> **Feature Inventory:** [feature-inventory.md](feature-inventory.md)
> **Prior Roadmap:** [roadmap.md](roadmap.md) (original 12-month R&D plan)

---

## Roadmap Overview

```
Mar–Apr 2026     May–Jul 2026      Aug–Nov 2026       Dec 2026–Mar 2027
┌──────────┐    ┌──────────┐      ┌──────────┐       ┌──────────────┐
│ 0.7      │    │ 0.8      │      │ 0.9      │       │ 1.0          │
│ Prod CLI │───→│ Opt Loop │──┐──→│ Co-Sim   │──────→│ Customer     │
└──────────┘    └──────────┘  │   └──────────┘       │ Ready        │
                              │                       └──────────────┘
                              │                              │
                     (parallel if ≥2 eng)          Apr–Jun 2027   Jul–Aug 2027
                                                   ┌──────────┐  ┌──────────┐
                                                   │ 1.1      │  │ 1.2      │
                                                   │ Enterprise│→ │ Sim2Real │
                                                   └──────────┘  └──────────┘
```

**Critical path:** 0.7 → 0.8 → 1.0 (optimization loop is the core differentiator)
**Parallel path:** 0.9 (co-simulation) can proceed alongside 0.8 with a second engineer

---

## Release 0.7 — "Production CLI"

**Timeline:** Month 1–2 (Mar–Apr 2026)
**Business Value:** Tool is usable by early adopters without workarounds. Users can run benchmarks on remote hardware and manage configuration through the CLI instead of editing YAML files.

### Features

| # | Feature | Inventory Ref | Priority | Effort | Description |
|---|---------|--------------|----------|--------|-------------|
| 1 | Wire SSH/K8s backends into CLI | 6.5 | P0 | M | Replace "not yet implemented" fallbacks in `workflow.py:101-106` and `benchmark.py:88-92` with actual backend dispatch |
| 2 | Implement `branes config set` | 6.6, 6.7 | P0 | S | YAML-validated config editing from CLI (replace manual file editing) |
| 3 | Implement `branes backends add/test/configure` | 6.8, 6.9, 6.10 | P0 | M | Full backend lifecycle: add SSH host/K8s cluster, test connectivity, configure resources |
| 4 | Fix SSH model serialization | 4.5 | P0 | M | Replace `_save_model_architecture()` placeholder with ONNX-based serialization + transfer |
| 5 | Fix K8s model reconstruction | 4.10 | P0 | M | Replace simplified Sequential placeholder with ONNX model loading from ConfigMap |
| 6 | Add Local GPU benchmark backend | 4.3 | P1 | M | CUDA benchmarking with memory profiling and GPU utilization |
| 7 | Spec → pipeline bridge | 10.28, 10.30 | P1 | M | Auto-extract DesignConstraints from SystemSpec; spec drives workload analysis → hardware → PPA |
| 8 | Bayesian MOO `predict()` | 7.8 | P2 | S | Implement fitted model persistence and prediction for rapid design exploration |
| 9 | Complete spec tutorial | 10.27 | P2 | S | Finish the spec-of-specs tutorial walkthrough |

### Exit Criteria

- `branes benchmark run model.pt --backend remote_ssh` returns results from a remote machine
- `branes benchmark run model.pt --backend kubernetes` runs on a K8s cluster
- `branes config set benchmark.timeout 120` updates config with validation
- `branes backends add my-gpu-box --type ssh --host 192.168.1.100` works end-to-end
- `branes spec new my-drone --template drone-perception && branes design from-spec my-drone` runs full pipeline from spec
- All 640+ existing tests still pass

### Risks

| Risk | Probability | Impact | Mitigation |
|------|:-----------:|:------:|------------|
| ONNX serialization doesn't cover all model architectures | Medium | High | Support PyTorch scripting as fallback; document supported model types |
| K8s ConfigMap size limit (1MB) for large models | Medium | Medium | Use PersistentVolumeClaim for models >1MB |

---

## Release 0.8 — "Optimization Loop"

**Timeline:** Month 3–5 (May–Jul 2026)
**Business Value:** The agent autonomously improves designs — the core differentiator that separates this from static analysis tools. Customers see the AI architect iterate and converge on better designs.

### Features

| # | Feature | Inventory Ref | Priority | Effort | Description |
|---|---------|--------------|----------|--------|-------------|
| 1 | Iterative optimization loop | 1.30 | P0 | XL | Main loop: optimizer → ppa_assessor → critic → re-plan. Convergence detection, max iterations, improvement threshold |
| 2 | Experience cache (SQLite + similarity) | 1.31, 1.32 | P0 | L | Persistent storage of past designs with nearest-neighbor lookup for warm-starting new explorations |
| 3 | Human-in-the-loop approval blocking | 9.18 | P0 | M | Replace auto-approve placeholder with real blocking gate (CLI prompt + API endpoint for future web UI) |
| 4 | Dispatcher retry policies | 1.39 | P1 | M | Transient failure recovery with exponential backoff; configurable retry count per task type |
| 5 | Dispatcher streaming output | 1.40 | P1 | M | Stream task progress to CLI/API consumers; real-time optimization status |
| 6 | Expanded critic vocabulary | 1.21 | P1 | M | Add thermal analysis, reliability (MTTF/electromigration), supply chain risk critique |
| 7 | Surrogate model training | 7.16 | P1 | L | Train lightweight surrogate from experience cache for 100x faster design space exploration |
| 8 | DAG dataflow topology | 9.23 | P2 | M | Extend bandwidth analysis from linear chain to arbitrary DAG (parallel branches, fan-in/fan-out) |

### Exit Criteria

- Agent takes a use-case description, iterates through ≥3 design alternatives, converges on a design meeting all constraints
- ≥20% improvement over naive single-pass baseline on a standard benchmark goal
- Experience cache persists across sessions; warm-started exploration converges 2x faster
- Human approval gate blocks on safety-critical actions; CLI shows approval prompt

### Risks

| Risk | Probability | Impact | Mitigation |
|------|:-----------:|:------:|------------|
| Optimization loop diverges (oscillates between strategies) | Medium | High | Monotonicity check: reject iterations that regress on primary metric |
| Experience cache grows unbounded | Low | Medium | TTL + LRU eviction; archive old designs |
| Surrogate model inaccuracy misleads search | Medium | Medium | Validation against full PPA assessor every N iterations |

### Dependencies

- Requires Release 0.7 (CLI must work for testing the loop end-to-end)

---

## Release 0.9 — "Co-Simulation Foundation"

**Timeline:** Month 6–9 (Aug–Nov 2026)
**Business Value:** Designs are validated against physics simulation. Customers can trust that the predicted power/latency numbers match reality within ±15%.

### Features

| # | Feature | Inventory Ref | Priority | Effort | Description |
|---|---------|--------------|----------|--------|-------------|
| 1 | AirSim/Gazebo ↔ compute simulator bridge | 2.1, 2.2, 2.3 | P0 | XL | Bi-directional bridge: physics simulator sends sensor data, compute simulator returns actuator commands, power/latency instrumented |
| 2 | Per-component energy instrumentation | 2.4 | P0 | L | Attribute energy to perception, planning, control subsystems during simulation |
| 3 | End-to-end latency correlation | 2.5, 2.6 | P0 | L | Timestamp synchronization between physics and compute; sensor-to-actuator latency measurement |
| 4 | Hardware model calibration (Jetson Orin) | 2.7, 2.8 | P0 | L | Run physical benchmarks on Jetson Orin; calibrate PPA models to ±15% accuracy |
| 5 | Mission profile library | — | P1 | M | Standard scenarios: hover, cruise, obstacle avoidance, search pattern, emergency landing |
| 6 | Software configuration matrix (50+ configs) | 2.10 | P1 | L | Combinatorial sweep: model variants × quantization × scheduling × rate parameters |
| 7 | Parallel simulation infrastructure | 2.9 | P1 | L | Run 10+ simulations concurrently for dataset generation (10K flight minutes target) |
| 8 | External power monitoring (SCPI/USB) | 4.14 | P2 | M | Integration with Keithley/NI USB power meters for calibration measurements |

### Exit Criteria

- Run a simulated drone mission (5 minutes, obstacle avoidance scenario)
- Get per-component energy breakdown (perception X%, planning Y%, control Z%)
- Predicted vs measured energy on physical Jetson Orin is within ±15%
- Sensor-to-actuator latency measured and correlated with compute load

### Risks

| Risk | Probability | Impact | Mitigation |
|------|:-----------:|:------:|------------|
| AirSim deprecation/maintenance issues | High | High | Support Gazebo as primary; AirSim as secondary. Abstract behind simulator interface |
| Simulation accuracy < ±15% target | Medium | High | Iterative calibration with increasing fidelity; publish confidence intervals |
| Parallel sim requires expensive GPU fleet | Medium | Medium | Start with CPU-only physics; add GPU acceleration incrementally |

### Dependencies

- Requires Release 0.7 (remote backends for running sims on cluster)
- Can run in parallel with Release 0.8 (independent engineering track)

---

## Release 1.0 — "Customer-Ready Platform"

**Timeline:** Month 10–12 (Dec 2026–Mar 2027)
**Business Value:** First paying customer. Production-quality tool with professional output, documentation, and CI/CD gating. A customer can install, analyze their workload, get recommendations, benchmark, validate, and receive reports.

### Features

| # | Feature | Inventory Ref | Priority | Effort | Description |
|---|---------|--------------|----------|--------|-------------|
| 1 | PDF report export | 1.26 | P0 | M | WeasyPrint/ReportLab HTML→PDF with executive summary, charts, appendix |
| 2 | Comparison reports (design A vs B) | 1.27 | P0 | M | Side-by-side PPA comparison with delta annotations and recommendation |
| 3 | API reference auto-generation | 9.8 | P0 | M | Sphinx/MkDocs autodoc from docstrings → docs site integration |
| 4 | Integration tests in CI | 9.4 | P0 | L | Separate CI stage for `[remote]`, `[kubernetes]`, `[chat]` optional dep groups |
| 5 | Test coverage reporting (80%+ gate) | 9.10 | P1 | M | pytest-cov with coverage badge and CI gate |
| 6 | RISC-V deployment target | 5.6 | P1 | L | RISC-V cross-compilation + runtime integration |
| 7 | FPGA deployment target (Vivado/Quartus) | 5.7 | P1 | XL | Bitstream generation from RTL templates; placement + routing |
| 8 | Multi-accelerator heterogeneous mapping | 1.10 | P1 | L | Architecture composer splits workload across CPU + GPU + KPU simultaneously |
| 9 | Production multi-rate framework | 8.24–8.26 | P1 | L | Graduate prototype to core package; real-time scheduling with priority inheritance |
| 10 | LiDAR drivers (Velodyne, Ouster) | 8.13, 8.14 | P2 | M | Production-quality VLP-16/32 and Ouster OS1/OS2 integration |

### Exit Criteria

- Customer can `pip install branes[all]` and run complete workflow
- `branes codebase assess /path/to/project --hardware jetson_orin --power-budget 15` produces PDF report
- All public APIs have generated reference documentation
- CI runs integration tests for all optional dependency groups
- Test coverage ≥80%

### Risks

| Risk | Probability | Impact | Mitigation |
|------|:-----------:|:------:|------------|
| FPGA synthesis is unbounded complexity | High | Medium | Start with Lattice iCE40 (small, open-source toolchain); defer Xilinx/Intel to 1.1 |
| Coverage gate blocks releases | Low | Medium | Gradual ramp: 60% at 1.0-rc1 → 80% at 1.0 GA |

### Dependencies

- Requires Release 0.8 (optimization loop is the core product)
- Benefits from Release 0.9 (co-simulation results in reports)

---

## Release 1.1 — "Enterprise Features"

**Timeline:** Month 13–15 (Apr–Jun 2027)
**Business Value:** Enterprise readiness — team collaboration, regulated-industry compliance, commercial EDA integration. Unlocks defense/aerospace/medical device customers.

### Features

| # | Feature | Inventory Ref | Priority | Effort | Description |
|---|---------|--------------|----------|--------|-------------|
| 1 | Team collaboration | — | P0 | XL | Shared design specs, reports, and optimization results across team members (Git-backed or server-backed) |
| 2 | Commercial EDA wrappers | 3.30 | P0 | L | Synopsys Design Compiler and Cadence Genus integration for production ASIC flows |
| 3 | IP block licensing model + catalog | 3.34 | P1 | L | Standardized IP catalog with licensing terms, integration complexity, and qualification status |
| 4 | Safety certification compliance | — | P1 | XL | IEC 62304 (medical), ISO 26262 (automotive), DO-178C (aerospace) artifact generation |
| 5 | Stereo vision fusion (bilateral/guided filters) | 8.20 | P2 | M | Replace basic averaging with production-quality depth filtering |
| 6 | HDF5 recording/replay | 8.29 | P2 | M | Drone perception Phase 4: synchronized multi-sensor HDF5 recording with indexed replay |
| 7 | Volume discounts + packaging cost | 3.24, 3.25 | P2 | S | Volume pricing curves, chiplet/2.5D/3D packaging cost models |

### Exit Criteria

- Team of 3 can share design workspace with conflict resolution
- RTL synthesis runs through Synopsys DC with timing reports
- Generated documentation artifacts satisfy IEC 62304 Class B traceability requirements
- Enterprise customer can use the tool in a regulated environment with audit trails

### Dependencies

- Requires Release 1.0 (customer-ready baseline)

---

## Release 1.2 — "Sim-to-Real Validation"

**Timeline:** Month 16–18 (Jul–Aug 2027)
**Business Value:** Proof that the co-design thesis works. Measurable, published improvement on real hardware. This is the demo that closes enterprise deals.

### Features

| # | Feature | Inventory Ref | Priority | Effort | Description |
|---|---------|--------------|----------|--------|-------------|
| 1 | Sim-to-real accuracy (±10% predicted vs measured) | 2.12 | P0 | XL | Physical drone with instrumented power/latency; compare against simulation predictions |
| 2 | Physical drone deployment + flight test | — | P0 | XL | End-to-end: co-designed system running on physical drone in controlled environment |
| 3 | ≥50% flight time improvement proof point | — | P0 | L | A/B test: baseline drone vs co-designed drone, same mission profile |
| 4 | Blueprint package (binary + HW config + BoM + report) | — | P0 | M | Deliverable package that a customer can hand to a contract manufacturer |
| 5 | NoC topology exploration | 3.12 | P1 | L | Move beyond hardcoded 2D mesh; explore ring, tree, and custom topologies |
| 6 | RTL timing closure flow | 3.18 | P1 | XL | Static timing analysis integration for sign-off quality RTL |
| 7 | LiDAR drivers (Livox, ROS bridge, file formats) | 8.15, 8.16 | P2 | M | Complete the sensor driver matrix |
| 8 | Learned workload estimation | 1.2 | P2 | L | Replace keyword-based heuristics with ML model trained on workload corpus |

### Exit Criteria

- Demonstrate a drone that flies ≥50% longer on co-designed HW/SW vs baseline
- Predictions match reality within ±10% on energy and ±15% on latency
- Blueprint package accepted by contract manufacturer for quoting
- Published case study with quantified results

### Dependencies

- Requires Release 0.9 (co-simulation must be calibrated)
- Requires Release 1.0 (reporting and documentation for publication)

---

## Effort Estimation Key

| Size | Description | Approximate Duration (1 engineer) |
|------|-------------|-----------------------------------|
| **S** | Small — single file, well-understood change | 1–3 days |
| **M** | Medium — 2-5 files, some design decisions | 1–2 weeks |
| **L** | Large — cross-cutting, new subsystem integration | 2–4 weeks |
| **XL** | Extra large — new domain, research-grade, multiple subsystems | 4–8 weeks |

---

## Dependency Graph

```
                    ┌────────────────────────────────────────┐
                    │ embodied-schemas (shared catalog)       │
                    │ graphs (roofline, calibration)          │
                    └───────────┬────────────────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │ Release 0.7           │
                    │ Production CLI        │
                    │ • CLI backend wiring  │
                    │ • Config management   │
                    │ • Model serialization │
                    └───┬───────────┬───────┘
                        │           │
           ┌────────────▼──┐   ┌───▼──────────────┐
           │ Release 0.8   │   │ Release 0.9      │
           │ Opt Loop      │   │ Co-Simulation    │
           │ • Iterative   │   │ • AirSim/Gazebo  │
           │ • Experience  │   │ • Calibration    │
           │ • Governance  │   │ • Energy tracing │
           └────────┬──────┘   └───┬──────────────┘
                    │              │
                    └──────┬───────┘
                           │
                    ┌──────▼──────────────┐
                    │ Release 1.0         │
                    │ Customer-Ready      │
                    │ • PDF reports       │
                    │ • API docs          │
                    │ • Integration CI    │
                    │ • RISC-V / FPGA     │
                    └──────┬──────────────┘
                           │
                    ┌──────▼──────────────┐
                    │ Release 1.1         │
                    │ Enterprise          │
                    │ • Team collab       │
                    │ • Commercial EDA    │
                    │ • Safety certs      │
                    └──────┬──────────────┘
                           │
                    ┌──────▼──────────────┐
                    │ Release 1.2         │
                    │ Sim-to-Real         │
                    │ • Physical drone    │
                    │ • ±10% accuracy     │
                    │ • 50% flight time   │
                    └─────────────────────┘
```

---

## Critical Path

The longest dependency chain determines the minimum timeline:

1. **0.7 → 0.8 → 1.0** (10 months) — The optimization loop is the core differentiator. It must work before we can sell.
2. **0.9** runs in parallel with 0.8 if there's a second engineer on co-simulation.
3. **1.1 → 1.2** is sequential — enterprise features enable the customer relationships needed for sim-to-real hardware access.

### Staffing Implications

| Engineers | Timeline to 1.0 | Notes |
|-----------|-----------------|-------|
| 1 | 14 months | Sequential: 0.7 → 0.8 → 0.9 → 1.0 |
| 2 | 12 months | Parallel: 0.8 ‖ 0.9, then converge at 1.0 |
| 3 | 10 months | Also parallelize RISC-V/FPGA targets |

---

## Risk Matrix (Cross-Release)

| Risk | Releases Affected | Probability | Impact | Mitigation |
|------|:-----------------:|:-----------:|:------:|------------|
| Optimization loop divergence | 0.8 | Medium | High | Monotonicity check, max iterations, human escalation |
| AirSim project deprecation | 0.9, 1.2 | High | High | Gazebo as primary simulator; abstract behind interface |
| Sim-to-real gap exceeds ±15% | 0.9, 1.2 | Medium | High | Iterative calibration; domain randomization; publish confidence intervals |
| FPGA synthesis complexity explosion | 1.0 | High | Medium | Start with open-source iCE40; defer Xilinx/Intel |
| Commercial EDA licensing cost | 1.1 | Medium | Medium | University program licenses; mock backends for CI |
| Physical drone test failures | 1.2 | Medium | High | Incremental: bench test → indoor flight → outdoor flight |
| Single-engineer bottleneck | All | High | High | Prioritize 0.7 → 0.8 → 1.0 path; defer 0.9 if needed |

---

## Relationship to Prior Roadmap

This roadmap supersedes [roadmap.md](roadmap.md) (the original 12-month R&D plan). Key changes:

| Original Plan | This Roadmap | Rationale |
|--------------|-------------|-----------|
| Phase 1: Co-Simulation (Month 1-4) | Release 0.9 (Month 6-9) | Deferred — CLI and optimization loop have higher customer value |
| Phase 2: Software Architect (Month 5-8) | Release 0.8 (Month 3-5) | Pulled forward — this is the core differentiator |
| Phase 3: Hardware Co-Design (Month 9-12) | Release 1.2 (Month 16-18) | Pushed back — need co-simulation + customer base first |
| No CLI/DX milestone | Release 0.7 (Month 1-2) | Added — current CLI has stubs that block early adoption |
| No customer-ready milestone | Release 1.0 (Month 10-12) | Added — explicit first-customer gate |
| No enterprise milestone | Release 1.1 (Month 13-15) | Added — safety certification unlocks regulated industries |

### Release 0.6 (Current — In Progress)

Release 0.6 is the current release. Key features delivered:

- **Spec-of-Specs system** (Domain 10): Event-sourced requirements management with 8 subsystem models, 14 CLI commands, 6 LLM tools, 6 templates, 7 validation checks, content-addressed versioning, and 82 tests. See [spec-of-specs-design.md](spec-of-specs-design.md).
- **Multi-objective optimization engine**: MAP-Elites + Bayesian BO + NSGA-III with MCP server
- **Documentation site**: Astro Starlight with tutorials and reference docs

The spec system is standalone — it doesn't yet drive the design pipeline (Release 0.7 bridges this gap).

### What's Preserved

All original Phase 1–3 acceptance criteria are preserved in the feature inventory and mapped to releases:
- Phase 1 acceptance criteria → Release 0.9 exit criteria
- Phase 2 acceptance criteria → Release 0.8 exit criteria
- Phase 3 acceptance criteria → Release 1.2 exit criteria

---

## Quarterly Business Milestones

| Quarter | Release | Business Milestone |
|---------|---------|-------------------|
| Q2 2026 | 0.7 | Early adopter program: 3-5 users running benchmarks on remote hardware |
| Q3 2026 | 0.8 | Technical differentiator demo: AI architect iterates and improves designs autonomously |
| Q4 2026 | 0.9 | Validation credibility: show ±15% prediction accuracy against physical hardware |
| Q1 2027 | 1.0 | **First paying customer.** Professional reports, documentation, CI/CD quality. |
| Q2 2027 | 1.1 | Enterprise pipeline: regulated industry customers (defense, medical, automotive) |
| Q3 2027 | 1.2 | Published proof point: 50%+ flight time improvement, ±10% prediction accuracy |
