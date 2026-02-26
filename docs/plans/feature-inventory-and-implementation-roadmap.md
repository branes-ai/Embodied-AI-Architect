# Embodied AI Architect — Feature Inventory & Implementation Roadmap

## Context

The Embodied AI Architect is an agentic AI platform for autonomous HW/SW co-design of embedded AI systems. After 3 months of development (Dec 2025 – Feb 2026), significant infrastructure exists: 6 core specialists, KPU RTL generation, multi-objective optimization, codebase analysis, 7 demos, and 560+ tests. However, critical gaps remain — the co-simulation validation pipeline was skipped, CLI commands have stubs, distributed backends aren't wired, and the optimization loop (Phase 2) is framework-only.

**Goal:** Enumerate all features/subfeatures, organize into an 18-month roadmap targeting first paying customer, with both engineering milestones and business value framing.

**Approach:** We will generate this as a structured document (`docs/plans/roadmap-v2.md`) containing:
1. Complete feature inventory (what exists, what's missing, what's needed)
2. Prioritized milestone plan organized by releases
3. Business-value framing per milestone
4. Dependencies and critical path

---

## Plan: Generate the Roadmap Document

### Step 1: Create Feature Inventory (`docs/plans/feature-inventory.md`)

A comprehensive matrix of every feature and subfeature, classified by status. Organized into these **9 feature domains**:

#### Domain 1: Core Design Pipeline
| Feature | Subfeature | Status | Gap |
|---------|-----------|--------|-----|
| Workload Analysis | Keyword-based estimation | DONE | Need learned estimation |
| Hardware Explorer | Static catalog + scoring | DONE | Need dynamic catalog from embodied-schemas |
| Architecture Composer | Operator→hardware mapping | DONE | Need multi-accelerator heterogeneous mapping |
| PPA Assessor | Power/latency/area/cost verdicts | DONE | Need calibrated models (±15% accuracy) |
| Critic Agent | Single-compute risk, power margin | DONE | Need expanded critique vocabulary |
| Report Generator | Structured reports + audit trail | DONE | Need PDF export, comparison reports |
| Optimization Loop | Framework defined (Phase 2 plan) | PARTIAL | Need full implementation: working memory, experience cache, strategy catalog |
| Planner + Dispatcher | LLM goal decomposition, DAG scheduler | DONE | Need streaming, retry policies |

#### Domain 2: Co-Simulation & Validation (PHASE 1 — NOT STARTED)
| Feature | Subfeature | Status | Gap |
|---------|-----------|--------|-----|
| Simulator Integration | AirSim/Gazebo ↔ CSim bridge | NOT STARTED | Full implementation needed |
| Energy Traceability | Per-component power breakdown | NOT STARTED | Instrumentation layer needed |
| Latency Correlation | Sensor→actuator timing | NOT STARTED | Timestamp synchronization needed |
| Hardware Model Accuracy | ±15% vs physical hardware | NOT STARTED | Empirical calibration needed |
| Dataset Generation | 10K+ flight minutes, 50+ configs | NOT STARTED | Parallel simulation infra needed |
| Sim-to-Real Validation | Predicted vs measured comparison | NOT STARTED | Physical hardware test rig needed |

#### Domain 3: Hardware Design & Synthesis
| Feature | Subfeature | Status | Gap |
|---------|-----------|--------|-----|
| KPU Micro-Architecture Config | 3 presets, technology scaling | DONE | Need more presets, user-defined configs |
| Floorplan Validation | Checkerboard pitch matching | DONE | Need area-aware placement |
| Bandwidth Validation | 4-link memory hierarchy check | DONE | Need NoC topology exploration |
| RTL Template Engine | 12 Verilog modules via Jinja2 | DONE | Need FPGA synthesis flow, timing closure |
| Manufacturing Cost Model | 16 process nodes, yield model | DONE | Need packaging cost, volume discounts |
| EDA Tool Integration | Yosys/Verilator/Icarus pipeline | DONE | Need commercial EDA wrappers (Synopsys, Cadence) |
| IP Block Library | Basic MAC, SRAM, DMA, router | DONE | Need standardized IP catalog, licensing model |

#### Domain 4: Benchmarking & Profiling
| Feature | Subfeature | Status | Gap |
|---------|-----------|--------|-----|
| Local CPU Backend | PyTorch inference timing | DONE | — |
| Local GPU Backend | CUDA benchmarking | NOT STARTED | Need GPU backend implementation |
| Remote SSH Backend | Framework + SFTP transfer | PARTIAL | Model serialization is stub |
| Kubernetes Backend | Job-based parallel benchmarking | PARTIAL | Model loading is stub |
| Power Monitoring | RAPL + AMD SMU backends | DONE | External power meters (SCPI/USB) not implemented |
| Architecture Benchmarking | Operator pipeline timing | DONE | — |
| Roofline Analysis | Via graphs package | DONE | — |

#### Domain 5: Deployment Targets
| Feature | Subfeature | Status | Gap |
|---------|-----------|--------|-----|
| Jetson (TensorRT) | FP32/FP16/INT8 + validation | DONE | Need on-device profiling integration |
| Coral (EdgeTPU) | INT8/UINT8 + compiler | DONE | — |
| OpenVINO | FP32/FP16/INT8 + NNCF | DONE | — |
| Stillwater KPU | Full ISA + deployment | DONE | Need real silicon validation |
| NVDLA | Full spec + deployment | DONE | — |
| RISC-V targets | — | NOT STARTED | Need RISC-V toolchain integration |
| FPGA targets | — | NOT STARTED | Need Vivado/Quartus integration |

#### Domain 6: CLI & Developer Experience
| Feature | Subfeature | Status | Gap |
|---------|-----------|--------|-----|
| 19 CLI subcommands | Core commands registered | DONE | — |
| Backend CLI integration | SSH/K8s backend selection | STUB | CLI says "not yet implemented", falls back to local |
| Backend management | `branes backends add/test/configure` | STUB | Must edit YAML manually |
| Config management | `branes config set` | STUB | Must edit YAML manually |
| Design synthesis | `branes design synthesize` | STUB | Synthesis engine placeholder |
| Spec template operations | Template variant wiring | PARTIAL | Some templates not fully wired |
| Interactive chat | Claude-powered agentic loop | DONE | — |
| Model Zoo | Multi-provider download/cache | DONE | — |
| Demo system | 7 discoverable demos | DONE | — |

#### Domain 7: Multi-Objective Optimization
| Feature | Subfeature | Status | Gap |
|---------|-----------|--------|-----|
| MAP-Elites | Quality-diversity search | DONE | — |
| Bayesian Optimization | BoTorch qNEHVI | DONE | `predict()` not implemented |
| NSGA-III fallback | Many-objective (>4) | DONE | — |
| MCP Server | 5 optimization tools | DONE | — |
| CLI Integration | explore/show-front/sensitivity/explain | DONE | — |
| Surrogate Models | Fast approximation | NOT STARTED | Need trained surrogates for rapid iteration |

#### Domain 8: Perception & Prototypes
| Feature | Subfeature | Status | Gap |
|---------|-----------|--------|-----|
| Drone Perception Pipeline | Detection + tracking + reasoning | DONE (Phase 3) | — |
| LiDAR Drivers | Velodyne/Ouster/Livox/ROS | STUB | 5 drivers unimplemented |
| Stereo Fusion | Depth map integration | PARTIAL | Basic averaging only |
| Multi-Rate Framework | Zenoh pub/sub scheduler | DONE (prototype) | Need production hardening |
| Recording/Replay | HDF5 recording (Phase 4 planned) | NOT STARTED | — |

#### Domain 9: Infrastructure & Quality
| Feature | Subfeature | Status | Gap |
|---------|-----------|--------|-----|
| CI/CD | Lint + test + release + docs deploy | DONE | Need integration test stage for optional deps |
| Documentation Site | Astro Starlight, 28 pages | DONE | Need API reference auto-generation |
| Test Suite | 560+ tests, 47 test files | DONE | Need coverage reporting, integration tests in CI |
| Golden Traces | Regression detection | DONE | — |
| Cost Tracking | Token cost per agent | DONE | — |
| Safety Governance | Audit trail + flagging | DONE | Human approval is placeholder |
| MCP Protocol | Server implementation | DONE | — |
| Dataflow Inference | Kernel→kernel bandwidth | STUB | Hardcoded to 0 bytes, linear chains only |

---

### Step 2: Create Milestone Roadmap (`docs/plans/roadmap-v2.md`)

**18-month timeline, 6 releases, targeting first paying customer.**

#### Release 0.7 — "Production CLI" (Month 1-2)
**Business Value:** Tool is usable by early adopters without workarounds.

| # | Feature | Priority | Effort |
|---|---------|----------|--------|
| 1 | Wire SSH/K8s backends into CLI benchmark and workflow commands | P0 | M |
| 2 | Implement `branes config set` with YAML validation | P0 | S |
| 3 | Implement `branes backends add/test/configure` | P0 | M |
| 4 | Fix Remote SSH model serialization (ONNX-based transfer) | P0 | M |
| 5 | Fix Kubernetes model loading (ONNX from ConfigMap) | P0 | M |
| 6 | Implement `branes design synthesize` (perception pipeline generation) | P1 | L |
| 7 | Add Local GPU benchmark backend | P1 | M |
| 8 | Bayesian MOO `predict()` implementation | P2 | S |
| 9 | Wire all spec template operations | P2 | S |

**Exit Criteria:** User can `branes benchmark run model.pt --backend remote_ssh` and get results. Config/backends manageable via CLI.

---

#### Release 0.8 — "Optimization Loop" (Month 3-5)
**Business Value:** The agent autonomously improves designs — the core differentiator.

| # | Feature | Priority | Effort |
|---|---------|----------|--------|
| 1 | Implement Phase 2 optimization loop (working memory, strategy catalog) | P0 | XL |
| 2 | Experience cache with SQLite + similarity search | P0 | L |
| 3 | Governance: real human-in-the-loop approval blocking | P0 | M |
| 4 | Dispatcher retry policies and streaming support | P1 | M |
| 5 | Expanded critic vocabulary (thermal, reliability, supply chain) | P1 | M |
| 6 | Surrogate model training for fast design space exploration | P1 | L |
| 7 | Dataflow bandwidth inference (tensor shape → bytes) | P2 | M |
| 8 | Complex dataflow topology (DAG support, not just linear chains) | P2 | M |

**Exit Criteria:** Agent takes a use-case description, iterates through ≥3 design alternatives, and converges on a design that meets all constraints with ≥20% improvement over naive baseline.

---

#### Release 0.9 — "Co-Simulation Foundation" (Month 6-9)
**Business Value:** Designs are validated against physics — customers can trust results.

| # | Feature | Priority | Effort |
|---|---------|----------|--------|
| 1 | AirSim/Gazebo ↔ compute simulator bridge | P0 | XL |
| 2 | Per-component energy instrumentation layer | P0 | L |
| 3 | End-to-end latency correlation (sensor → actuator) | P0 | L |
| 4 | Hardware model calibration against physical boards (Jetson Orin) | P0 | L |
| 5 | Mission profile library (hover, cruise, obstacle avoidance, etc.) | P1 | M |
| 6 | Software configuration matrix (50+ configs) | P1 | L |
| 7 | Parallel simulation infrastructure (10K flight minutes) | P1 | L |
| 8 | External power monitoring (SCPI/USB meter integration) | P2 | M |

**Exit Criteria:** Run a simulated drone mission, get ±15% accurate energy/latency predictions vs physical Jetson Orin hardware.

---

#### Release 1.0 — "Customer-Ready Platform" (Month 10-12)
**Business Value:** First paying customer. Production-quality with documentation and support.

| # | Feature | Priority | Effort |
|---|---------|----------|--------|
| 1 | Report generation: PDF export, comparison reports | P0 | M |
| 2 | API reference auto-generation from docstrings | P0 | M |
| 3 | Integration tests in CI for all optional dependency groups | P0 | L |
| 4 | RISC-V deployment target | P1 | L |
| 5 | FPGA deployment target (Vivado/Quartus) | P1 | XL |
| 6 | Multi-accelerator heterogeneous mapping in architecture composer | P1 | L |
| 7 | Production multi-rate framework (from prototype to core) | P1 | L |
| 8 | Test coverage reporting and 80%+ gate | P2 | M |
| 9 | LiDAR driver implementations (Velodyne, Ouster) | P2 | M |

**Exit Criteria:** A customer can install `pip install branes[all]`, analyze their codebase, get hardware recommendations, run benchmarks on remote hardware, validate in simulation, and receive a professional PDF report.

---

#### Release 1.1 — "Enterprise Features" (Month 13-15)
**Business Value:** Enterprise readiness — team collaboration, compliance, scale.

| # | Feature | Priority | Effort |
|---|---------|----------|--------|
| 1 | Team collaboration (shared specs, designs, reports) | P0 | XL |
| 2 | Commercial EDA tool wrappers (Synopsys DC, Cadence Genus) | P0 | L |
| 3 | IP block licensing model and catalog | P1 | L |
| 4 | Safety certification compliance (IEC 62304, ISO 26262, DO-178C) | P1 | XL |
| 5 | Proper stereo vision fusion (bilateral/guided filters) | P2 | M |
| 6 | Recording/replay with HDF5 (drone perception Phase 4) | P2 | M |
| 7 | Volume discount and packaging cost models | P2 | S |

**Exit Criteria:** Enterprise customer can use the tool in a regulated environment with audit trails, team sharing, and commercial EDA integration.

---

#### Release 1.2 — "Sim-to-Real Validation" (Month 16-18)
**Business Value:** Proof that the co-design thesis works — measurable improvement on real hardware.

| # | Feature | Priority | Effort |
|---|---------|----------|--------|
| 1 | Sim-to-real accuracy validation (±10% predicted vs measured) | P0 | XL |
| 2 | Physical drone deployment and flight test | P0 | XL |
| 3 | ≥50% flight time improvement proof point | P0 | L |
| 4 | Blueprint package (binary + HW config + BoM + report) | P0 | M |
| 5 | NoC topology exploration in bandwidth validation | P1 | L |
| 6 | Timing closure flow for RTL | P1 | XL |
| 7 | LiDAR drivers: Livox, ROS bridge, file-based (.pcd/.ply) | P2 | M |
| 8 | Learned workload estimation (replace keyword-based) | P2 | L |

**Exit Criteria:** Demonstrate a drone that flies 50% longer on co-designed HW/SW vs baseline, with predictions matching reality within 10%.

---

### Step 3: Process for Generating These Documents

We will create **two documents**:

1. **`docs/plans/feature-inventory.md`** — The exhaustive feature matrix above, formatted as markdown tables with links to source files. This is the living inventory that gets updated as features are completed.

2. **`docs/plans/roadmap-v2.md`** — The milestone roadmap above, with:
   - Release timeline with dates
   - Per-release feature table with effort estimates (S/M/L/XL)
   - Business value statement per release
   - Exit criteria per release
   - Dependency graph showing critical path
   - Risk matrix per release

### Step 4: Critical Path Analysis

```
Release 0.7 (CLI polish) ──┐
                            ├──→ Release 0.8 (Optimization loop) ──┐
                            │                                       │
                            └──→ Release 0.9 (Co-simulation) ──────┤
                                                                    │
                                                    Release 1.0 (Customer-ready) ──→ 1.1 ──→ 1.2
```

**Critical path:** 0.7 → 0.8 → 1.0 (the optimization loop is the core differentiator and must work before customer release)

**Parallel path:** 0.9 (co-simulation) can proceed in parallel with 0.8 if there are multiple engineers.

### Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `docs/plans/feature-inventory.md` | CREATE | Exhaustive feature matrix with status and gaps |
| `docs/plans/roadmap-v2.md` | CREATE | 18-month milestone roadmap with business framing |

### Verification

- Review both documents for completeness against the codebase exploration findings
- Ensure every TODO/STUB/gap from the codebase audit appears in the feature inventory
- Verify critical path makes sense given dependencies
- Cross-reference with existing `docs/plans/roadmap.md` to ensure nothing from the original plan is lost
