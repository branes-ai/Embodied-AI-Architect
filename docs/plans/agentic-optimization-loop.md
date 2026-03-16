# Agentic Optimization Loop — Design Space Exploration Plan

**Created:** 2026-03-16
**Status:** Draft
**Goal:** Close the loop from natural-language mission description → joint design space optimization → intelligence/capability per watt as the top-level objective.

---

## Current State

| Layer | Status | Key Files |
|-------|--------|-----------|
| **LLM Agent Loop** | Working | `src/embodied_ai_architect/llm/agent.py` — tool-use loop with Claude |
| **40+ Agent Tools** | Working | `llm/tools.py`, `llm/graphs_tools.py`, `llm/architecture_tools.py`, `llm/optimization_tools.py` |
| **3-Layer MOO** | Working | MAP-Elites → Bayesian BO → NSGA-III in `graphs/moo/` |
| **SoC Design Space** | Working | 7 HW variables (process, clock, array, SRAM, tiles, NoC) in `graphs/moo/design_space.py` |
| **SWaP-C Analysis** | Working | 5 methodologies (FoM, sensitivity, Pareto, delta, Monte Carlo) in `graphs/swap_analysis.py` |
| **LangGraph Pipelines** | Working | Perception + autonomy graphs in `graphs/pipelines/` |
| **Testbench Metrics** | Working | mAP, F1, IoU, accuracy in `testbench/metrics.py` |
| **9-Dim Evaluation** | Working | `graphs/scoring.py` scores the agentic system itself |

---

## Gap Analysis: 5 Missing Design Spaces

### 1. Mission-to-Plan Decomposition (The "Front Door")

The agent can call tools, but there is no structured **prompt → plan → task graph** pipeline. When Claude receives "build a drone perception system that can navigate at 5m/s in GPS-denied environments," there is no mechanism to:

- Decompose into **sub-capabilities** (obstacle detection, SLAM, path planning, motor control)
- Map sub-capabilities to **pipeline operators** with rate/latency requirements
- Generate a **LangGraph state machine** definition from the plan
- Create an **optimization problem formulation** (objectives, constraints, design variables)

### 2. Pipeline Structure as a Design Variable

Current design space is **hardware-only** (7 SoC parameters). The pipeline topology is fixed. Missing:

- **Operator selection** — which detector? tracker? state estimator?
- **Operator ordering** — parallel vs. sequential stages
- **Operator fusion** — can detection + tracking share features?
- **Multi-rate scheduling** — which operators run at which rate?
- **Data layout** — NCHW vs NHWC, tiling strategy

`architecture_tools.py` can *analyze* variants but cannot *search* over them. The `design.py` CLI builds pipelines from requirements but uses a fixed mapping, not optimization.

### 3. Neural Architecture Search (NAS) Integration

**Completely absent.** The system can analyze a given model on hardware, but cannot:

- Explore model families (width/depth scaling, attention vs conv)
- Search quantization schemes (FP16, INT8, INT4, mixed-precision)
- Trade accuracy vs. compute (accuracy-aware Pareto)
- Use proxy tasks or supernet evaluation

### 4. Compiler/Runtime Configuration Space

**Not represented.** The evaluator assumes fixed 30% utilization. Missing:

- Compiler optimization level (graph fusion, constant folding, dead code elimination)
- Precision mode (FP32/FP16/BF16/INT8/mixed)
- Tiling parameters (spatial tile size, batch tile)
- Memory allocation strategy (static vs dynamic, workspace size)
- Runtime batch size
- TensorRT/OpenVINO/TVM optimization profiles

### 5. Intelligence/Capability Per Watt (The "Top-Level Objective")

Current objectives are PPA-only: power, latency, area, cost. Missing:

| Metric | Definition | Why It Matters |
|--------|-----------|----------------|
| **Accuracy** | Task-specific (mAP, F1, classification accuracy) | Can't trade accuracy vs power without it |
| **Capability score** | Normalized mission success rate (0-1) | Integrates detection + tracking + planning quality |
| **Intelligence/watt** | `capability_score / power_watts` | The metric to maximize |
| **GOPS/watt** | `peak_gops / power_watts` | Hardware efficiency |
| **Ops/joule** | `gops / energy_per_inference` | Energy efficiency per inference |
| **Mission endurance** | `battery_wh / power_watts × mission_success_rate` | Effective operational time |

---

## Target Architecture

```
Mission Description (NL)
        │
        ▼
┌──────────────────────────┐
│  MissionDecomposer        │ ◄── Claude decomposes mission, informed by
│  + Research Library (RAG) │     HW accelerator research & domain knowledge
│  (Phase 2)                │
└────────┬─────────────────┘
         │  MissionRequirements + DesignSpace
         ▼
┌──────────────────────────┐
│  Joint Design Space       │ ◄── Pipeline × Model × Compiler × Hardware
│  (Phase 3)                │     Currently: Hardware only
└────────┬─────────────────┘
         │  Design point (pipeline + model + compiler + HW)
         ▼
┌──────────────────────────┐
│  Unified Evaluator        │ ◄── Evaluates BOTH capability AND cost
│  (Phase 1 + 3)            │     Currently: PPA only, no accuracy
└────────┬─────────────────┘
         │  (accuracy, latency, power, area, cost, weight)
         ▼
┌──────────────────────────┐
│  MOO Engine               │ ◄── Existing 3-layer pipeline
│  (EXISTS)                 │     Extended with more objectives
└────────┬─────────────────┘
         │  Pareto front in (capability/watt, latency, cost, ...)
         ▼
┌──────────────────────────┐
│  Claude Reasoning         │ ◄── Interpret Pareto, recommend, iterate
│  + Research Library (RAG) │     Informed by SOTA accelerator research
│  (Phase 4)                │
└──────────────────────────┘
```

---

## Phase 1 — Close the Accuracy Loop

**Goal:** Enable `capability_per_watt` as a first-class optimization objective.

### 1.1 Accuracy Lookup Tables

Create accuracy scaling data for common model families so the evaluator can estimate accuracy without running inference:

```python
# Model family accuracy curves (published/measured)
ACCURACY_TABLES = {
    "yolov8": {
        "n":  {"mAP50": 37.3, "params_M": 3.2,  "gflops": 8.7},
        "s":  {"mAP50": 44.9, "params_M": 11.2, "gflops": 28.6},
        "m":  {"mAP50": 50.2, "params_M": 25.9, "gflops": 78.9},
        "l":  {"mAP50": 52.9, "params_M": 43.7, "gflops": 165.2},
        "x":  {"mAP50": 53.9, "params_M": 68.2, "gflops": 257.8},
    },
    # + EfficientNet, ResNet, MobileNet, DETR, ...
}

# Quantization accuracy degradation curves
QUANTIZATION_IMPACT = {
    "FP16": {"accuracy_drop_pct": 0.1, "speedup": 1.8},
    "INT8":  {"accuracy_drop_pct": 1.5, "speedup": 3.2},
    "INT4":  {"accuracy_drop_pct": 5.0, "speedup": 5.5},
}
```

### 1.2 Extend PPAMetrics

Add accuracy and derived metrics to `PPAMetrics` in `soc_state.py`:

```python
# New fields
accuracy_percent: Optional[float]       # Task-specific accuracy (mAP, F1, top-1)
capability_score: Optional[float]       # Normalized mission success [0, 1]
capability_per_watt: Optional[float]    # capability_score / power_watts
gops_per_watt: Optional[float]          # peak_gops / power_watts
energy_per_inference_mj: Optional[float]  # Already exists, wire it up
```

### 1.3 Wire Testbench → Evaluator

Connect the existing testbench metrics (`testbench/metrics.py`) to the MOO evaluator so accuracy flows through the optimization loop.

### 1.4 Add Accuracy as MOO Objective

Extend the MOO engine to support `accuracy_percent` (MAXIMIZE) alongside the existing MINIMIZE objectives (power, latency, area, cost).

### Deliverables

- `graphs/accuracy_tables.py` — Model accuracy lookup data
- Extended `PPAMetrics` with capability fields
- Extended `DesignEvaluator` computing `capability_per_watt`
- MOO engine configured with accuracy as an objective
- CLI: `branes optimize` reports capability/watt on Pareto front

---

## Phase 2 — Mission Decomposition + Research Library

**Goal:** Claude can take a natural-language mission description and produce a structured optimization problem, informed by HW accelerator research literature.

### 2.1 HW Accelerator Research Library

Build a curated document library of hardware accelerator research that can be retrieved and mixed into the LLM context window during mission decomposition and design reasoning.

#### Library Structure

```
data/
  research_library/
    index.yaml                    # Master index with metadata per document
    accelerators/
      nvidia_dla.md               # NVDLA architecture deep-dive
      google_tpu_v4.md            # TPU v4 architecture & efficiency data
      apple_ane.md                # Apple Neural Engine design choices
      qualcomm_hexagon.md         # Hexagon DSP for always-on inference
      hailo_8.md                  # Hailo-8 dataflow architecture
      cerebras_wse.md             # Wafer-scale engine (cloud reference)
      ethos_u55_u85.md            # Arm Ethos-U for microcontrollers
      stillwater_kpu.md           # KPU architecture & design rationale
    architectures/
      systolic_arrays.md          # Systolic array design space & tradeoffs
      dataflow_architectures.md   # Spatial, temporal, row-stationary dataflow
      sparsity_acceleration.md    # Structured/unstructured sparsity HW support
      mixed_precision_hw.md       # HW support for INT4/INT8/FP16/BF16
      memory_hierarchies.md       # On-chip SRAM, HBM, LPDDR tradeoffs
      noc_topologies.md           # Mesh, ring, torus, crossbar interconnects
    efficiency_studies/
      tops_per_watt_survey.md     # Cross-architecture TOPS/W comparison
      roofline_case_studies.md    # Roofline analysis examples by workload
      quantization_accuracy.md    # Accuracy vs precision across model families
      nas_hw_codesign.md          # HW-aware NAS papers (EfficientNet, OFA, etc.)
      edge_ai_benchmarks.md       # MLPerf Edge, AI Benchmark results
    workloads/
      transformer_attention.md    # Attention compute patterns & HW implications
      convolution_variants.md     # Standard, depthwise, grouped conv HW mapping
      state_estimation.md         # EKF/UKF/PF compute characteristics
      sensor_fusion.md            # Multi-modal fusion architectures
      control_loops.md            # PID/MPC compute requirements
    mission_profiles/
      drone_perception.md         # Drone perception pipeline requirements & SOTA
      autonomous_driving.md       # L2-L5 compute budgets & architecture choices
      industrial_inspection.md    # Inspection pipeline design patterns
      humanoid_locomotion.md      # Real-time balance & gait control requirements
```

#### Document Format

Each document follows a structured format for consistent retrieval:

```markdown
---
title: "NVDLA Architecture Analysis"
domain: accelerator
tags: [nvidia, dla, inference, edge, open-source]
relevance:
  - mission_decomposition    # When to retrieve this document
  - hardware_selection
  - design_space_definition
compute_density_tops_w: 2.5  # Key quantitative facts in frontmatter
peak_tops: 5.0               # for quick filtering without full retrieval
process_nm: 16
target_workloads: [cnn, object_detection]
last_updated: 2026-03-16
---

## Architecture Overview
...

## Key Design Decisions & Tradeoffs
...

## Performance Characteristics
(TOPS, TOPS/W, utilization by workload type, memory bandwidth requirements)

## Lessons for Design Space Exploration
(What this architecture teaches about the design space — which knobs matter most)
```

#### Retrieval Strategy

The library is retrieved via **tag-based context injection**, not full vector RAG (keeping it simple and auditable):

```python
class ResearchLibrary:
    """Curated HW accelerator research for LLM context enrichment."""

    def __init__(self, library_path: Path):
        self.index = self._load_index(library_path / "index.yaml")
        self.documents = self._load_documents(library_path)

    def retrieve(
        self,
        mission_type: str | None = None,
        tags: list[str] | None = None,
        relevance: str | None = None,  # e.g. "mission_decomposition"
        max_tokens: int = 8000,
    ) -> list[Document]:
        """Retrieve documents matching criteria, fit within token budget."""
        ...

    def build_context_block(self, documents: list[Document]) -> str:
        """Format retrieved docs as a context block for the LLM prompt."""
        ...
```

**Integration points:**
- `MissionDecomposer` retrieves by `mission_type` + `relevance=mission_decomposition`
- `DesignSpaceFormulator` retrieves by `tags` matching workload types
- `ParetoReasoner` (Phase 4) retrieves by `relevance=design_tradeoffs`
- `suggest_optimal_design` tool retrieves architecture comparisons for rationale

#### Why Phase 2?

The research library is most impactful at the **mission decomposition** stage because:

1. **Informed decomposition** — knowing that drone perception at 5m/s requires <33ms end-to-end (from `drone_perception.md`) lets Claude set realistic latency budgets per operator
2. **Architecture priors** — knowing that depthwise convolutions are 10× more efficient on DSPs than GPUs (from `convolution_variants.md`) steers operator-to-hardware mapping
3. **Design space scoping** — knowing the TOPS/W frontier from `tops_per_watt_survey.md` prevents wasting optimization cycles on infeasible regions
4. **Quantitative grounding** — published efficiency numbers anchor Claude's estimates instead of hallucinating performance figures

The library also feeds Phase 4 (Claude reasoning over Pareto fronts), but decomposition is where bad assumptions compound most — a wrong latency budget at decomposition time wastes the entire optimization run.

### 2.2 MissionDecomposer

```python
class MissionDecomposer:
    """NL mission → structured requirements → operator graph → design space."""

    def decompose(self, mission_description: str) -> MissionPlan:
        # 1. Retrieve relevant research documents
        context = self.research_library.retrieve(
            mission_type=self._classify_mission(mission_description),
            relevance="mission_decomposition",
        )

        # 2. Claude decomposes with research context
        plan = self.llm_client.chat(
            messages=[{"role": "user", "content": mission_description}],
            system=DECOMPOSITION_PROMPT + context,
            tools=self.decomposition_tools,
        )

        # 3. Structured output
        return MissionPlan(
            sub_capabilities=[...],
            operator_graph=OperatorGraph(...),
            design_space=DesignSpaceDefinition(...),
            objectives=[...],
            constraints=[...],
        )
```

### 2.3 Plan → LangGraph Generation

Convert `MissionPlan.operator_graph` into a runnable LangGraph `StateGraph`:

- Map sub-capabilities to operator nodes
- Set conditional edges for branching logic (e.g., collision → evasive action)
- Assign rate requirements per node
- Generate the graph builder code or configuration

### Deliverables

- `data/research_library/` — Curated document collection (30-50 documents)
- `data/research_library/index.yaml` — Master index with tags and metadata
- `src/embodied_ai_architect/research/library.py` — Retrieval and context formatting
- `src/embodied_ai_architect/research/decomposer.py` — Mission decomposition agent
- `MissionPlan` Pydantic model in `embodied-schemas`
- LangGraph generation from operator graphs
- New tools: `decompose_mission`, `retrieve_research`, `formulate_design_space`

---

## Phase 3 — Joint Design Space

**Goal:** Optimization searches over pipeline structure + model + compiler + hardware simultaneously.

### 3.1 Extend DesignSpace with Pipeline Variables

```python
# Pipeline structure variables
DesignVariable("detector_variant", VariableType.CATEGORICAL,
               categories=["yolov8n", "yolov8s", "yolov8m", "yolov8l"]),
DesignVariable("tracker_variant", VariableType.CATEGORICAL,
               categories=["bytetrack", "sort", "deepsort", "botsort"]),
DesignVariable("state_estimator", VariableType.CATEGORICAL,
               categories=["ekf", "ukf", "particle_filter", "none"]),
DesignVariable("enable_fusion", VariableType.CATEGORICAL,
               categories=[True, False]),
DesignVariable("pipeline_parallelism", VariableType.CATEGORICAL,
               categories=["sequential", "pipelined", "parallel"]),
```

### 3.2 Add NAS Variables

```python
DesignVariable("model_family", VariableType.CATEGORICAL,
               categories=["yolov8", "efficientnet", "mobilenet", "detr"]),
DesignVariable("width_scale", VariableType.CONTINUOUS, bounds=(0.25, 2.0)),
DesignVariable("quantization", VariableType.CATEGORICAL,
               categories=["fp32", "fp16", "int8", "int4", "mixed"]),
DesignVariable("pruning_ratio", VariableType.CONTINUOUS, bounds=(0.0, 0.8)),
```

### 3.3 Add Compiler Config Variables

```python
DesignVariable("precision_mode", VariableType.CATEGORICAL,
               categories=["fp32", "fp16", "bf16", "int8", "mixed"]),
DesignVariable("tile_size", VariableType.INTEGER, bounds=(8, 64)),
DesignVariable("batch_size", VariableType.INTEGER, bounds=(1, 16)),
DesignVariable("graph_fusion", VariableType.CATEGORICAL,
               categories=[True, False]),
DesignVariable("runtime", VariableType.CATEGORICAL,
               categories=["pytorch", "tensorrt", "openvino", "tvm"]),
```

### 3.4 Unified Evaluator

Extend `DesignEvaluator` to compose a complete system from a joint design point:

1. Select pipeline operators from pipeline variables
2. Look up model accuracy from accuracy tables (Phase 1)
3. Apply quantization/pruning accuracy degradation
4. Estimate HW performance via roofline (parameterized by compiler config)
5. Compute all objectives including `capability_per_watt`

### Deliverables

- Extended `DesignSpace` with 15+ new variables across 4 dimensions
- Extended `DesignEvaluator` composing full system configurations
- Research library context used to scope design spaces per mission type
- Accuracy-aware Pareto analysis

---

## Phase 4 — Closed-Loop Agentic Optimization

**Goal:** LangGraph-orchestrated loop where Claude reasons over results, steers exploration, and converges on a recommended design.

### 4.1 Optimization LangGraph

```
decompose → formulate → optimize → evaluate → reason → (iterate | recommend)
```

Nodes:
- **decompose**: `MissionDecomposer` + research library context
- **formulate**: Define joint design space, objectives, constraints
- **optimize**: Run MOO engine (MAP-Elites → BO/NSGA-III)
- **evaluate**: Score Pareto front on capability/watt and mission metrics
- **reason**: Claude analyzes results with research library context, decides next action
- **iterate**: Refine design space bounds, add/remove variables, re-run
- **recommend**: Final design with full rationale and research citations

### 4.2 Convergence Detection

- Hypervolume improvement < threshold for N iterations
- Pareto front stability (no new non-dominated points)
- Claude judges diminishing returns from reasoning node

### 4.3 Research-Informed Reasoning

At the reasoning node, Claude receives:
- Current Pareto front visualization
- Retrieved research documents relevant to observed tradeoffs
- Historical optimization trajectory
- Comparison against known SOTA designs from the library

This enables Claude to make informed decisions like:
- "The Pareto front shows a knee at 2 TOPS/W — literature suggests dataflow architectures break through this at the cost of programmability. Should we add a dataflow variant to the design space?"
- "INT8 quantization drops mAP by 3% on this model family — the research shows structured pruning recovers 1.5% with minimal latency impact. Adding pruning_ratio to the search."

### Deliverables

- `graphs/optimization_loop.py` — LangGraph state machine for the outer loop
- Research library integration at reasoning node
- Convergence detection and early stopping
- Final recommendation report with research citations
- CLI: `branes optimize mission "description"` end-to-end command

---

## Research Library Curation Guidelines

### Document Selection Criteria

1. **Quantitative** — must include concrete performance numbers (TOPS, TOPS/W, accuracy, latency)
2. **Architectural** — must describe design decisions and tradeoffs, not just results
3. **Actionable** — must contain lessons that map to design space variables or constraint values
4. **Current** — prefer post-2022 publications; update when new benchmarks arrive

### Maintenance

- Review and update quarterly or when major new accelerators ship
- Tag documents with which phases/tools consume them
- Track which documents Claude actually retrieves (log retrieval patterns)
- Prune documents with zero retrievals over 3 months

### Bootstrapping

Start with 10-15 high-value documents covering:
- Top 5 edge accelerators (Jetson Orin, Hailo-8, Coral, Ethos-U, KPU)
- Key architecture patterns (systolic arrays, dataflow, sparsity)
- Cross-architecture efficiency survey (TOPS/W by workload class)
- 2-3 mission profiles matching current prototypes (drone, UGV)
- Quantization/NAS co-design papers

---

## Success Criteria

| Metric | Target |
|--------|--------|
| Can optimize `capability_per_watt` | Phase 1 |
| Claude decomposes a drone mission into ≥5 operators with latency budgets | Phase 2 |
| Research library provides relevant context in ≥80% of decompositions | Phase 2 |
| Joint design space covers ≥15 variables across 4 dimensions | Phase 3 |
| Pareto front includes accuracy as an axis | Phase 3 |
| End-to-end: NL mission → optimized design in single CLI command | Phase 4 |
| Claude cites research library in final recommendation | Phase 4 |
