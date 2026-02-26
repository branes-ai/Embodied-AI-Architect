# Embodied AI Architect — Feature Inventory

> **Baseline:** v0.6.0 (Feb 2026, commit 513fec6)
> **Test suite:** 640+ tests across 46 test files
> **Source:** `src/embodied_ai_architect/` (20,500+ LOC across agents + graphs modules)

This living document catalogs every feature and subfeature in the Embodied AI Architect platform, organized by domain. Each entry records implementation status, source location, and remaining gaps.

**Status legend:**
- **DONE** — Functional, tested, integrated
- **PARTIAL** — Framework/core exists, key pieces missing
- **STUB** — Interface defined, implementation placeholder
- **NOT STARTED** — No code exists

---

## Domain 1: Core Design Pipeline

The agentic pipeline that takes a use-case goal through workload analysis → hardware selection → architecture composition → PPA assessment → critique → reporting.

| # | Feature | Subfeature | Status | Source | Gap |
|---|---------|-----------|--------|--------|-----|
| 1.1 | Workload Analysis | PyTorch model introspection | DONE | `agents/model_analyzer.py` | — |
| 1.2 | Workload Analysis | Keyword-based workload estimation | DONE | `graphs/specialists.py:54-78` | Heuristic only; no learned estimation models |
| 1.3 | Workload Analysis | Operator classification (conv, matmul, attention) | DONE | `graphs/specialists.py:54-313` | Limited to predefined workload templates |
| 1.4 | Workload Analysis | Scheduling mode inference (concurrent/sequential/time-shared) | DONE | `graphs/specialists.py` | — |
| 1.5 | Hardware Explorer | Static catalog + embodied-schemas Registry | DONE | `agents/hardware_profile/agent.py`, `graphs/specialists.py:374-409` | Static integration; not truly dynamic catalog refresh |
| 1.6 | Hardware Explorer | Constraint-based filtering + scoring (0-100) | DONE | `agents/hardware_profile/agent.py:9-326` | No supply chain risk or availability modeling |
| 1.7 | Hardware Explorer | Always-include Stillwater KPU | DONE | `graphs/specialists.py:374-409` | — |
| 1.8 | Architecture Composer | Operator → hardware accelerator mapping | DONE | `graphs/specialists.py:720-869` | Single primary accelerator only |
| 1.9 | Architecture Composer | SoC composition (CPU + GPU + KPU + memory + NoC + ISP) | DONE | `graphs/specialists.py:720-869` | — |
| 1.10 | Architecture Composer | Power-budget-aware IP block selection | DONE | `graphs/specialists.py:720-869` | No multi-accelerator heterogeneous mapping |
| 1.11 | Architecture Composer | KPU variant selection (drone_minimal, edge_balanced, server_max) | DONE | `graphs/ip_blocks.py` | IP block placement not area-aware |
| 1.12 | PPA Assessor | Power estimation (block-level sum + optimization scaling) | DONE | `graphs/specialists.py:1031-1188` | No thermal analysis |
| 1.13 | PPA Assessor | Latency estimation (GFLOPS / peak TFLOPS × efficiency) | DONE | `graphs/specialists.py:1031-1188` | 50% efficiency ceiling assumption |
| 1.14 | PPA Assessor | Area estimation (IP block sum × process node scaling) | DONE | `graphs/specialists.py:1031-1188` | — |
| 1.15 | PPA Assessor | Cost estimation (manufacturing cost model) | DONE | `graphs/specialists.py:1031-1188` | No packaging/assembly complexity |
| 1.16 | PPA Assessor | Constraint verdicts (PASS/FAIL per metric) | DONE | `graphs/specialists.py:1031-1188` | ±15% accuracy not yet validated |
| 1.17 | PPA Assessor | Bottleneck identification + fix suggestions | DONE | `graphs/specialists.py:1031-1188` | — |
| 1.18 | Critic Agent | Constraint verdict checking | DONE | `graphs/governance.py` | — |
| 1.19 | Critic Agent | Improvement tracking vs baseline | DONE | `graphs/governance.py` | — |
| 1.20 | Critic Agent | Iteration loop control (max_iterations) | DONE | `graphs/governance.py` | — |
| 1.21 | Critic Agent | Critique vocabulary | PARTIAL | `graphs/governance.py` | Missing: thermal, reliability (MTTF), supply chain, PDN, EMI/EMC |
| 1.22 | Report Generator | HTML report with executive summary | DONE | `agents/report_synthesis/agent.py:1-794` | — |
| 1.23 | Report Generator | JSON metadata export | DONE | `agents/report_synthesis/agent.py` | — |
| 1.24 | Report Generator | Visualization charts (matplotlib) | DONE | `agents/report_synthesis/agent.py` | Static PNG only; no interactive charts |
| 1.25 | Report Generator | Hardware comparison + layer distribution charts | DONE | `agents/report_synthesis/agent.py` | — |
| 1.26 | Report Generator | PDF export | NOT STARTED | — | Full implementation needed |
| 1.27 | Report Generator | Comparison reports (design A vs B) | NOT STARTED | — | Full implementation needed |
| 1.28 | Optimization Loop | 8-strategy catalog (quantize, prune, clock scale, etc.) | DONE | `graphs/optimizer.py:102-193` | Deterministic greedy selection only |
| 1.29 | Optimization Loop | Working memory (attempt tracking) | PARTIAL | `graphs/optimizer.py`, `graphs/memory.py` | Skeleton hooks, no serialization across steps |
| 1.30 | Optimization Loop | Iterative convergence loop | NOT STARTED | — | Optimizer runs once then stops; no re-planning |
| 1.31 | Optimization Loop | Experience cache (SQLite + similarity search) | NOT STARTED | — | Full implementation needed |
| 1.32 | Optimization Loop | Strategy learning from past designs | NOT STARTED | — | Full implementation needed |
| 1.33 | Planner | LLM goal → TaskGraph (DAG) decomposition | DONE | `graphs/planner.py:200-344` | — |
| 1.34 | Planner | 13 specialist agent vocabulary | DONE | `graphs/planner.py:60-118` | — |
| 1.35 | Planner | Static plan mode for testing | DONE | `graphs/planner.py` | — |
| 1.36 | Dispatcher | DAG-aware task execution with dependency tracking | DONE | `graphs/dispatcher.py:51-263` | — |
| 1.37 | Dispatcher | Parallel-ready task batching | DONE | `graphs/dispatcher.py` | — |
| 1.38 | Dispatcher | State update merging from task results | DONE | `graphs/dispatcher.py` | — |
| 1.39 | Dispatcher | Retry policies | NOT STARTED | — | All tasks are synchronous, no transient failure recovery |
| 1.40 | Dispatcher | Streaming task output | NOT STARTED | — | Full implementation needed |

---

## Domain 2: Co-Simulation & Validation

The Phase 1 validation pipeline: physics simulation → compute simulation → energy traceability → calibration. **This entire domain was deferred.**

| # | Feature | Subfeature | Status | Source | Gap |
|---|---------|-----------|--------|--------|-----|
| 2.1 | Simulator Integration | AirSim ↔ compute simulator bridge | NOT STARTED | — | Full implementation needed |
| 2.2 | Simulator Integration | Gazebo ↔ compute simulator bridge | NOT STARTED | — | Full implementation needed |
| 2.3 | Simulator Integration | Co-simulation loop (physics + compute + power) | NOT STARTED | — | Full implementation needed |
| 2.4 | Energy Traceability | Per-component power breakdown (perception, planning, control) | NOT STARTED | — | Instrumentation layer needed |
| 2.5 | Latency Correlation | Sensor → actuator end-to-end timing | NOT STARTED | — | Timestamp synchronization needed |
| 2.6 | Latency Correlation | Latency vs power cross-metric analysis | NOT STARTED | — | No regression or correlation tooling |
| 2.7 | Hardware Model Accuracy | Calibration against physical Jetson Orin | NOT STARTED | — | Empirical calibration workflow needed |
| 2.8 | Hardware Model Accuracy | ±15% accuracy validation | NOT STARTED | — | Measurement test rig needed |
| 2.9 | Dataset Generation | 10K+ simulated flight minutes | NOT STARTED | — | Parallel simulation infrastructure needed |
| 2.10 | Dataset Generation | 50+ software configurations | NOT STARTED | — | Configuration matrix needed |
| 2.11 | Dataset Generation | Synthetic 3D scene generation | NOT STARTED | — | No Blender/Unity integration |
| 2.12 | Sim-to-Real Validation | Predicted vs measured comparison | NOT STARTED | — | Physical hardware test rig needed |
| 2.13 | Sim-to-Real Validation | Domain randomization (wind, noise, drift) | NOT STARTED | — | Requires simulator integration first |

---

## Domain 3: Hardware Design & Synthesis

KPU micro-architecture, floorplanning, RTL generation, manufacturing cost modeling, and EDA tool integration.

| # | Feature | Subfeature | Status | Source | Gap |
|---|---------|-----------|--------|--------|-----|
| 3.1 | KPU Micro-Architecture | 3 presets (drone_minimal, edge_balanced, server_max) | DONE | `graphs/kpu_config.py` | Need more presets, user-defined configs |
| 3.2 | KPU Micro-Architecture | Process node scaling (28nm to 7nm) | DONE | `graphs/kpu_config.py` | — |
| 3.3 | KPU Micro-Architecture | Heuristic sizing from workload GFLOPS + power | DONE | `graphs/kpu_config.py` | — |
| 3.4 | KPU Micro-Architecture | Checkerboard 2D layout (compute/memory tiles) | DONE | `graphs/kpu_config.py` | — |
| 3.5 | Floorplan Validation | Checkerboard pitch matching (15% tolerance) | DONE | `graphs/floorplan.py` | — |
| 3.6 | Floorplan Validation | Area estimation (compute + memory + periphery) | DONE | `graphs/floorplan.py` | — |
| 3.7 | Floorplan Validation | Die reticle limit check (~33mm edge) | DONE | `graphs/floorplan.py` | — |
| 3.8 | Floorplan Validation | Auto-adjustment loop for L3 pitch matching | DONE | `graphs/floorplan.py` | No area-aware placement |
| 3.9 | Bandwidth Validation | 4-link memory hierarchy (DRAM → L3 → L2 → L1 → Compute) | DONE | `graphs/bandwidth.py` | — |
| 3.10 | Bandwidth Validation | Hierarchical reuse factors (0.7, 0.5, 0.3) | DONE | `graphs/bandwidth.py` | — |
| 3.11 | Bandwidth Validation | Per-link utilization + bottleneck detection (85% threshold) | DONE | `graphs/bandwidth.py` | — |
| 3.12 | Bandwidth Validation | NoC topology exploration | NOT STARTED | — | Hardcoded to 2D mesh |
| 3.13 | RTL Template Engine | 11 component types (MAC, compute tile, L1/L2/L3, NoC, DMA, etc.) | DONE | `graphs/rtl_templates/__init__.py` | — |
| 3.14 | RTL Template Engine | Jinja2 rendering + fallback to built-in templates | DONE | `graphs/rtl_templates/__init__.py` | — |
| 3.15 | RTL Template Engine | Testbench generation per component | DONE | `graphs/rtl_templates/__init__.py` | — |
| 3.16 | RTL Template Engine | Parameter-driven RTL (data widths, array sizes, buffer depths) | DONE | `graphs/rtl_templates/__init__.py` | — |
| 3.17 | RTL Template Engine | FPGA synthesis flow | NOT STARTED | — | Need Vivado/Quartus integration |
| 3.18 | RTL Template Engine | Timing closure | NOT STARTED | — | Need STA integration |
| 3.19 | Manufacturing Cost Model | Process node economics (180nm to 2nm) | DONE | `graphs/manufacturing.py` | — |
| 3.20 | Manufacturing Cost Model | Murphy's yield model | DONE | `graphs/manufacturing.py` | — |
| 3.21 | Manufacturing Cost Model | Wafer die packing (300mm, 3mm exclusion) | DONE | `graphs/manufacturing.py` | — |
| 3.22 | Manufacturing Cost Model | Cost breakdown (NRE + die + package + test) | DONE | `graphs/manufacturing.py` | — |
| 3.23 | Manufacturing Cost Model | Package types (QFN, BGA, FCBGA, WLCSP) | DONE | `graphs/manufacturing.py` | — |
| 3.24 | Manufacturing Cost Model | Volume discounts | NOT STARTED | — | Need volume pricing curves |
| 3.25 | Manufacturing Cost Model | Advanced packaging cost (chiplet, 2.5D, 3D) | NOT STARTED | — | — |
| 3.26 | EDA Tool Integration | Yosys synthesis with auto-fallback to mock | DONE | `graphs/eda_tools/synthesis.py` | — |
| 3.27 | EDA Tool Integration | Complexity-based timeout estimation | DONE | `graphs/eda_tools/synthesis.py` | — |
| 3.28 | EDA Tool Integration | Process node area scaling from Yosys stats | DONE | `graphs/eda_tools/synthesis.py` | — |
| 3.29 | EDA Tool Integration | Icarus Verilog simulation + VCD generation | DONE | `graphs/eda_tools/simulation.py` | — |
| 3.30 | EDA Tool Integration | Commercial EDA wrappers (Synopsys DC, Cadence Genus) | NOT STARTED | — | Full implementation needed |
| 3.31 | IP Block Library | 8 block types (CPU, GPU, KPU, memory controller, NoC, I/O, ISP) | DONE | `graphs/ip_blocks.py` | — |
| 3.32 | IP Block Library | Area/power/GOPS estimation per block | DONE | `graphs/ip_blocks.py` | — |
| 3.33 | IP Block Library | SoCComposition aggregator | DONE | `graphs/ip_blocks.py` | — |
| 3.34 | IP Block Library | Standardized IP catalog + licensing model | NOT STARTED | — | Need IP marketplace/catalog |

---

## Domain 4: Benchmarking & Profiling

Performance measurement across local, remote, and cluster backends.

| # | Feature | Subfeature | Status | Source | Gap |
|---|---------|-----------|--------|--------|-----|
| 4.1 | Local CPU Backend | PyTorch inference timing (latency, throughput) | DONE | `agents/benchmark/backends/local_cpu.py` | — |
| 4.2 | Local CPU Backend | Warmup/iteration statistics | DONE | `agents/benchmark/backends/local_cpu.py` | — |
| 4.3 | Local GPU Backend | CUDA benchmarking | NOT STARTED | — | Full implementation needed |
| 4.4 | Remote SSH Backend | SSH connection via paramiko + secrets manager | DONE | `agents/benchmark/backends/remote_ssh.py` | — |
| 4.5 | Remote SSH Backend | Model serialization and transfer | PARTIAL | `agents/benchmark/backends/remote_ssh.py:282-286` | `_save_model_architecture()` is placeholder; generates minimal wrapper, not full architecture |
| 4.6 | Remote SSH Backend | Remote script execution + result parsing | DONE | `agents/benchmark/backends/remote_ssh.py` | — |
| 4.7 | Kubernetes Backend | Job creation + ConfigMap for models | DONE | `agents/benchmark/backends/kubernetes.py` | — |
| 4.8 | Kubernetes Backend | Horizontal scaling (`execute_parallel()`) | DONE | `agents/benchmark/backends/kubernetes.py` | — |
| 4.9 | Kubernetes Backend | GPU allocation + node selectors | DONE | `agents/benchmark/backends/kubernetes.py` | — |
| 4.10 | Kubernetes Backend | Model reconstruction in pod | PARTIAL | `agents/benchmark/backends/kubernetes.py:336-341` | Uses simplified Sequential model instead of real reconstruction |
| 4.11 | Power Monitoring | RAPL backend (Intel sysfs) | DONE | `benchmark/power.py` | — |
| 4.12 | Power Monitoring | AMD SMU backend | PARTIAL | `benchmark/power.py` | Stub; requires ryzen_monitor tool subprocess |
| 4.13 | Power Monitoring | Auto-detect backend + background sampling | DONE | `benchmark/power.py` | — |
| 4.14 | Power Monitoring | External power meters (SCPI/USB) | STUB | `benchmark/power.py` | Placeholder interface only |
| 4.15 | Architecture Benchmarking | Operator pipeline timing | DONE | `graphs/specialists.py` | — |
| 4.16 | Roofline Analysis | Via graphs package integration | DONE | — | — |

---

## Domain 5: Deployment Targets

Model compilation and optimization for edge/cloud hardware.

| # | Feature | Subfeature | Status | Source | Gap |
|---|---------|-----------|--------|--------|-----|
| 5.1 | NVIDIA Jetson | TensorRT FP32/FP16/INT8 + validation | DONE | `agents/deployment/targets/jetson.py` | Need on-device profiling integration |
| 5.2 | Google Coral | EdgeTPU INT8/UINT8 + compiler | DONE | `agents/deployment/targets/coral.py` | — |
| 5.3 | Intel OpenVINO | FP32/FP16/INT8 + NNCF | DONE | `agents/deployment/targets/openvino.py` | — |
| 5.4 | Stillwater KPU | Full ISA + deployment | DONE | Custom SoC target via graphs pipeline | Need real silicon validation |
| 5.5 | NVDLA | Full spec + deployment | DONE | Research deployment target | — |
| 5.6 | RISC-V Targets | RISC-V toolchain integration | NOT STARTED | — | Need cross-compilation + runtime |
| 5.7 | FPGA Targets | Vivado/Quartus integration | NOT STARTED | — | Need bitstream generation flow |
| 5.8 | AMD Ryzen AI | NPU deployment | PARTIAL | `docs/plans/ryzen-ai-nuc-demo.md` | Plan exists; implementation TBD |

---

## Domain 6: CLI & Developer Experience

Click-based CLI with 17+ subcommands, interactive chat, demos, and model zoo.

| # | Feature | Subfeature | Status | Source | Gap |
|---|---------|-----------|--------|--------|-----|
| 6.1 | CLI Framework | 17 registered subcommands | DONE | `cli/__init__.py`, `cli/commands/` | — |
| 6.2 | Workflow Command | Full orchestrator pipeline | DONE | `cli/commands/workflow.py` | — |
| 6.3 | Analyze Command | Model structure analysis | DONE | `cli/commands/` | — |
| 6.4 | Benchmark Command | Performance benchmarking | DONE | `cli/commands/benchmark.py` | — |
| 6.5 | Backend Selection (CLI) | `--backend remote_ssh\|kubernetes` | STUB | `cli/commands/workflow.py:101-106`, `benchmark.py:88-92` | Prints "not yet implemented"; falls back to local_cpu |
| 6.6 | Config Management | `branes config set` | STUB | `cli/commands/config.py:112-115` | Prints "not yet implemented"; must edit YAML manually |
| 6.7 | Config Management | YAML validation | NOT STARTED | `cli/commands/config.py:136` (TODO comment) | — |
| 6.8 | Backend Management | `branes backends add` | STUB | `cli/commands/backends.py:204-207` | Prints "not yet implemented" |
| 6.9 | Backend Management | `branes backends test` (SSH) | PARTIAL | `cli/commands/backends.py:128-146` | Only checks paramiko import; no connection test |
| 6.10 | Backend Management | `branes backends test` (K8s) | PARTIAL | `cli/commands/backends.py:177` | TODO: "Actually test K8s connection" |
| 6.11 | Design Commands | `branes design new/from-usecase/show` | DONE | `cli/commands/design.py` | — |
| 6.12 | Design Commands | `branes design synthesize` | DONE | `cli/commands/design.py` | Pipeline synthesis from requirements |
| 6.13 | Interactive Chat | Claude-powered agentic loop | DONE | `cli/commands/chat.py` | — |
| 6.14 | Model Zoo | Multi-provider search/download/cache/info | DONE | `cli/commands/zoo.py` | — |
| 6.15 | Model Registry | register/list/show/remove/update | DONE | `cli/commands/model.py` | — |
| 6.16 | Demo System | 7 discoverable demos | DONE | `cli/commands/demo.py` | — |
| 6.17 | Report Management | View/manage reports | DONE | `cli/commands/report.py` | — |
| 6.18 | Secrets Management | Credential storage | DONE | `cli/commands/secrets.py` | — |
| 6.19 | Codebase Analysis | scan/analyze/assess subcommands | DONE | `cli/commands/codebase.py` | — |
| 6.20 | Deploy Command | Target-specific deployment | DONE | `cli/commands/deploy.py` | — |
| 6.21 | Spec Commands | 14 subcommands (new/list/show/set/delete/commit/history/diff/tag/export/import/why/validate/resolve) | DONE | `cli/commands/spec.py:1-691` | — |
| 6.22 | Testbench Validation | `branes testbench validate` | DONE | `cli/commands/testbench.py` | — |
| 6.23 | Pipeline Command | Perception/autonomy pipeline execution | DONE | `cli/commands/pipeline.py` | Streaming + checkpoint support |
| 6.24 | Optimize Command | explore/show-front/sensitivity/explain | DONE | `cli/commands/optimize.py` | — |

---

## Domain 7: Multi-Objective Optimization

Three-layer MOO pipeline: MAP-Elites → Bayesian BO → NSGA-III, with MCP server and CLI integration.

| # | Feature | Subfeature | Status | Source | Gap |
|---|---------|-----------|--------|--------|-----|
| 7.1 | MAP-Elites | Quality-diversity search with grid cells | DONE | `graphs/moo/map_elites.py` | — |
| 7.2 | MAP-Elites | Mutation/crossover operators | DONE | `graphs/moo/map_elites.py` | — |
| 7.3 | MAP-Elites | Coverage metrics + best-per-objective | DONE | `graphs/moo/map_elites.py` | — |
| 7.4 | Bayesian Optimization | BoTorch qNEHVI acquisition function | DONE | `graphs/moo/bayesian_opt.py` | — |
| 7.5 | Bayesian Optimization | Independent GP models per objective | DONE | `graphs/moo/bayesian_opt.py` | — |
| 7.6 | Bayesian Optimization | Sensitivity analysis from GP lengthscales | DONE | `graphs/moo/bayesian_opt.py` | — |
| 7.7 | Bayesian Optimization | Hypervolume computation + Pareto front extraction | DONE | `graphs/moo/bayesian_opt.py` | — |
| 7.8 | Bayesian Optimization | `predict()` method | STUB | `graphs/moo/bayesian_opt.py` | Raises NotImplementedError; requires fitted model persistence |
| 7.9 | NSGA-III | Pymoo wrapper for many-objective (>4) | DONE | `graphs/moo/nsga3.py` | — |
| 7.10 | NSGA-III | Reference direction generation | DONE | `graphs/moo/nsga3.py` | — |
| 7.11 | NSGA-III | Hypervolume indicator | DONE | `graphs/moo/nsga3.py` | — |
| 7.12 | MCP Server | 5 optimization tools | DONE | `mcp/server.py` | — |
| 7.13 | MCP Server | Session manager for background optimization | DONE | `mcp/server.py` | — |
| 7.14 | MCP Server | Dual-response pattern (LLM preview + full data) | DONE | `mcp/server.py` | — |
| 7.15 | CLI Integration | explore/show-front/sensitivity/explain | DONE | `cli/commands/optimize.py` | — |
| 7.16 | Surrogate Models | Fast approximation for rapid iteration | NOT STARTED | — | Need trained surrogates |

---

## Domain 8: Perception & Prototypes

Drone perception pipeline, multi-rate control, and sensor integration.

| # | Feature | Subfeature | Status | Source | Gap |
|---|---------|-----------|--------|--------|-----|
| 8.1 | Detection | YOLOv8 wrapper (n/s/m/l/x variants) | DONE | `prototypes/drone_perception/detection/yolo.py` | — |
| 8.2 | Detection | Batch inference, class filtering, GPU/CPU | DONE | `prototypes/drone_perception/detection/yolo.py` | — |
| 8.3 | Tracking | ByteTrack with Kalman filter | DONE | `prototypes/drone_perception/tracking/bytetrack.py` | — |
| 8.4 | Tracking | IOU association + track state machine | DONE | `prototypes/drone_perception/tracking/bytetrack.py` | — |
| 8.5 | Reasoning | Trajectory prediction (constant vel/accel + physics) | DONE | `prototypes/drone_perception/reasoning/trajectory_predictor.py` | — |
| 8.6 | Reasoning | Collision detection + risk assessment | PARTIAL | `prototypes/drone_perception/reasoning/collision_detector.py` | Framework exists; core logic implemented |
| 8.7 | Reasoning | Behavior classification (9 types + threat assessment) | PARTIAL | `prototypes/drone_perception/reasoning/behavior_classifier.py` | Framework exists; heuristic-based |
| 8.8 | Reasoning | Spatial analysis | DONE | `prototypes/drone_perception/reasoning/` | — |
| 8.9 | Sensors | Monocular camera (OpenCV webcam/video) | DONE | `prototypes/drone_perception/sensors/monocular.py` | — |
| 8.10 | Sensors | Stereo camera (RealSense D435/D455 + OAK-D) | DONE | `prototypes/drone_perception/sensors/stereo.py` | — |
| 8.11 | Sensors | Recorded stereo playback | DONE | `prototypes/drone_perception/sensors/stereo.py` | — |
| 8.12 | Sensors | Wide-angle camera | DONE | `prototypes/drone_perception/sensors/wide_angle.py` | — |
| 8.13 | LiDAR | Velodyne driver (VLP-16/32, HDL-64E) | STUB | `prototypes/drone_perception/sensors/lidar.py` | Placeholder with TODO |
| 8.14 | LiDAR | Ouster driver | STUB | `prototypes/drone_perception/sensors/lidar.py` | Placeholder; needs ouster-sdk |
| 8.15 | LiDAR | Livox driver | STUB | `prototypes/drone_perception/sensors/lidar.py` | Placeholder; needs livox-sdk |
| 8.16 | LiDAR | ROS subscriber bridge | STUB | `prototypes/drone_perception/sensors/lidar.py` | Placeholder |
| 8.17 | LiDAR | File-based (.pcd, .ply, .bin, .npy) | DONE | `prototypes/drone_perception/sensors/lidar.py` | — |
| 8.18 | LiDAR-Camera Fusion | Point cloud → image plane projection | PARTIAL | `prototypes/drone_perception/sensors/lidar.py` | Framework good; depends on driver stubs |
| 8.19 | Stereo Fusion | Hardware depth extraction (RealSense/OAK-D) | DONE | `prototypes/drone_perception/sensors/stereo.py` | — |
| 8.20 | Stereo Fusion | Depth filtering and interpolation | PARTIAL | `prototypes/drone_perception/sensors/stereo.py` | Basic averaging; no bilateral/guided filters |
| 8.21 | Scene Graph | 3D state management with Kalman filter (9D) | DONE | `prototypes/drone_perception/scene_graph/manager.py` | — |
| 8.22 | Scene Graph | Object persistence + TTL pruning | DONE | `prototypes/drone_perception/scene_graph/manager.py` | — |
| 8.23 | Scene Graph | Monocular depth from bbox height heuristic | DONE | `prototypes/drone_perception/scene_graph/manager.py` | — |
| 8.24 | Multi-Rate Framework | `@control_loop(rate_hz=X)` decorator | DONE | `prototypes/multi_rate_framework/framework.py` | Prototype quality |
| 8.25 | Multi-Rate Framework | Zenoh pub/sub message bus | DONE | `prototypes/multi_rate_framework/framework.py` | — |
| 8.26 | Multi-Rate Framework | Per-component rate control + deadline monitoring | DONE | `prototypes/multi_rate_framework/framework.py` | No real-time guarantees (Python GIL) |
| 8.27 | Recording | Webcam capture utility | DONE | `prototypes/drone_perception/scripts/record_webcam.py` | — |
| 8.28 | Recording | MiDaS monocular depth synthesis | DONE | `prototypes/drone_perception/scripts/generate_depth_maps.py` | — |
| 8.29 | Recording/Replay | HDF5 recording (Phase 4) | NOT STARTED | — | Full implementation needed |

---

## Domain 9: Infrastructure & Quality

CI/CD, documentation, testing, governance, safety, and observability.

| # | Feature | Subfeature | Status | Source | Gap |
|---|---------|-----------|--------|--------|-----|
| 9.1 | CI/CD | Lint (black + ruff) + test (pytest) gate | DONE | `.github/workflows/ci.yml` | — |
| 9.2 | CI/CD | Release workflow (version bump + tag) | DONE | `.github/workflows/release.yml` | — |
| 9.3 | CI/CD | Documentation site deployment | DONE | `.github/workflows/deploy-docs.yml` | — |
| 9.4 | CI/CD | Integration tests for optional deps | NOT STARTED | — | Need separate CI stage |
| 9.5 | Documentation Site | Astro Starlight, 28+ pages | DONE | `docs-site/` | — |
| 9.6 | Documentation Site | Tutorials (Coral, OpenVINO, Ryzen AI, drone YOLO) | DONE | `docs-site/` | — |
| 9.7 | Documentation Site | MCP tools reference | DONE | `docs-site/` | — |
| 9.8 | Documentation Site | API reference auto-generation | NOT STARTED | — | Need docstring → API docs pipeline |
| 9.9 | Test Suite | 560+ tests, 44 test files | DONE | `tests/` | — |
| 9.10 | Test Suite | Coverage reporting | NOT STARTED | — | Need pytest-cov integration |
| 9.11 | Golden Traces | Regression detection via TracingDispatcher | DONE | `graphs/trace.py` | — |
| 9.12 | Golden Traces | RunTrace capture (tool calls, timing, PPA, audit) | DONE | `graphs/trace.py` | — |
| 9.13 | Cost Tracking | Token cost per agent + USD estimation | DONE | `graphs/governance.py` | — |
| 9.14 | Cost Tracking | Per-agent breakdown + reporting | DONE | `graphs/governance.py` | — |
| 9.15 | Safety Governance | Audit trail (append-only AuditEntry log) | DONE | `graphs/governance.py` | — |
| 9.16 | Safety Governance | Safety-critical detection (keywords + standards) | DONE | `graphs/safety.py` | — |
| 9.17 | Safety Governance | Redundancy injection (dual-lockstep, ECC, watchdog) | DONE | `graphs/safety.py` | — |
| 9.18 | Safety Governance | Human approval gate | STUB | `graphs/governance.py` | Auto-approves; "in production, this would block" comment |
| 9.19 | Safety Governance | Policy-based action gating | DONE | `graphs/governance.py` | Framework complete; no blocking endpoint |
| 9.20 | MCP Protocol | Server implementation (5 tools) | DONE | `mcp/server.py` | — |
| 9.21 | Dataflow Inference | 4-link bandwidth demand estimation | DONE | `graphs/bandwidth.py` | — |
| 9.22 | Dataflow Inference | Cache reuse factors (hierarchical) | DONE | `graphs/bandwidth.py` | — |
| 9.23 | Dataflow Inference | Complex DAG topology (not just linear chains) | NOT STARTED | — | Bandwidth analysis is linear chain only |
| 9.24 | Codebase Analysis | Static file scanner | DONE | `codebase/scanner.py` | — |
| 9.25 | Codebase Analysis | LLM multi-pass analyzer (4 passes) | DONE | `codebase/analyzer.py` | — |
| 9.26 | Codebase Analysis | Workload profile converter | DONE | `codebase/converter.py` | — |
| 9.27 | KPU Simulator | Compiler + runtime + cycle-accurate sim | NOT STARTED | `docs/kpu-simulator-requirements.md` (spec only) | Requirements documented; no code |
| 9.28 | Model Validation | Detection/classification/segmentation testbench | DONE | `testbench/validation.py`, `testbench/metrics.py` | — |

---

## Domain 10: Spec-of-Specs (Requirements Management)

Event-sourced, versioned, hierarchical requirements system. Released in v0.6.0.

| # | Feature | Subfeature | Status | Source | Gap |
|---|---------|-----------|--------|--------|-----|
| 10.1 | System Spec Model | 8 subsystems (Perception, Compute, Power, Sensor, Actuator, Comms, Autonomy, Safety) | DONE | `specs/models.py:104-368` | — |
| 10.2 | System Spec Model | Platform type enums (drone, quadruped, biped, AMR, etc.) | DONE | `specs/models.py:36-90` | — |
| 10.3 | System Spec Model | Path utilities (parse/get/set/delete at JSON pointer path) | DONE | `specs/models.py:477-632` | — |
| 10.4 | System Spec Model | Type coercion for numeric strings and booleans | DONE | `specs/models.py` | — |
| 10.5 | System Spec Model | `extra="forbid"` to catch typos | DONE | `specs/models.py` | — |
| 10.6 | Event Sourcing | Append-only JSONL event log | DONE | `specs/events.py:51-184` | — |
| 10.7 | Event Sourcing | Thread-safe file locking (fcntl) | DONE | `specs/events.py` | — |
| 10.8 | Event Sourcing | Event replay with snapshot optimization | DONE | `specs/events.py` | — |
| 10.9 | Event Sourcing | Auto-snapshot after 50 events | DONE | `specs/events.py:181-183` | — |
| 10.10 | Event Sourcing | Field-level provenance (who/when/why per change) | DONE | `specs/events.py` | — |
| 10.11 | Spec Store | CRUD (create, get, list, delete) | DONE | `specs/store.py:96-212` | — |
| 10.12 | Spec Store | Field mutations (set/delete at path) | DONE | `specs/store.py:213-295` | — |
| 10.13 | Spec Store | Content-addressed versioning (SHA256 blobs) | DONE | `specs/store.py:296-398` | — |
| 10.14 | Spec Store | Tag management (label versions) | DONE | `specs/store.py:296-398` | — |
| 10.15 | Spec Store | Version history with parent chain | DONE | `specs/store.py:296-398` | — |
| 10.16 | Spec Store | Structured diff between versions | DONE | `specs/diff.py:1-124` | — |
| 10.17 | Spec Store | Export (JSON/YAML) + Import | DONE | `specs/store.py:448-538` | — |
| 10.18 | Spec Store | Resolve (flatten to dot-notation for agents) | DONE | `specs/store.py:539-632` | — |
| 10.19 | Spec Store | Path traversal prevention | DONE | `specs/store.py` | — |
| 10.20 | Templates | 6 templates (drone-perception, quadruped-nav, industrial-inspection, amr-warehouse, edge-camera, biped-humanoid) | DONE | `specs/templates.py:1-389` | — |
| 10.21 | Validation | 7 cross-subsystem checks (power/compute, latency, platform, safety, mission duration, comms, autonomy sensors) | DONE | `specs/validation.py:1-296` | — |
| 10.22 | CLI Integration | 14 subcommands (new/list/show/set/delete/commit/history/diff/tag/export/import/why/validate/resolve) | DONE | `cli/commands/spec.py:1-691` | — |
| 10.23 | LLM Tools | 6 chat agent tools (list/create/read/modify/validate/field_history) | DONE | `llm/spec_tools.py:1-308` | — |
| 10.24 | Exceptions | 5 typed exceptions (SpecError, NotFound, AlreadyExists, InvalidPath, VersionNotFound) | DONE | `specs/exceptions.py:1-42` | — |
| 10.25 | Tests | 82 tests (44 model + 38 store) | DONE | `tests/test_spec_models.py`, `tests/test_spec_store.py` | — |
| 10.26 | Documentation | Design document | DONE | `docs/plans/spec-of-specs-design.md` | — |
| 10.27 | Documentation | Tutorial walkthrough | PARTIAL | `docs/tutorials/spec-of-specs.md` | Needs completion |
| 10.28 | Spec → Pipeline | Spec drives design pipeline (workload → hardware → PPA) | NOT STARTED | — | Spec is standalone; not yet wired as pipeline input |
| 10.29 | Spec Comparison | Compare specs across projects/versions | NOT STARTED | — | Diff exists per-spec but no cross-spec comparison |
| 10.30 | Spec Constraints | Auto-extract DesignConstraints from SystemSpec | NOT STARTED | — | Manual bridge needed between spec subsystems and pipeline constraints |

---

## Summary Statistics

| Status | Count | Percentage |
|--------|-------|------------|
| **DONE** | 140 | 76% |
| **PARTIAL** | 14 | 8% |
| **STUB** | 11 | 6% |
| **NOT STARTED** | 19 | 10% |
| **Total** | 184 | 100% |

### Critical Gaps by Priority

**Must-fix for production CLI (Release 0.7):**
- CLI backend selection falls back to local_cpu (6.5)
- `branes config set` is stub (6.6)
- `branes backends add/test` are stubs (6.8, 6.9, 6.10)
- SSH model serialization is placeholder (4.5)
- K8s model reconstruction is placeholder (4.10)

**Must-fix for core differentiator (Release 0.8):**
- No iterative optimization loop (1.30)
- No experience cache (1.31)
- Human approval is auto-approve placeholder (9.18)

**Must-fix for validation (Release 0.9):**
- Entire co-simulation domain is not started (Domain 2)

**Must-fix for customer release (Release 1.0):**
- No PDF report export (1.26)
- No API reference docs (9.8)
- No integration tests in CI (9.4)
