# SWaP-C Optimization Methodologies: Unified Metrics, Industry Practice, and Design Space Exploration

> Research synthesis: Is there a unified SWaP-C figure of merit analogous to the Energy-Delay
> Product? What has industry developed over four decades for multi-objective SWaP-C optimization?
> Five optimization methodologies for practical, productive SWaP-C design space exploration.

---

## Table of Contents

1. [The Energy-Delay Product Paradigm](#1-the-energy-delay-product-paradigm)
2. [The Search for a Unified SWaP-C Metric](#2-the-search-for-a-unified-swap-c-metric)
3. [Industry Figures of Merit Bridging Chip to System](#3-industry-figures-of-merit-bridging-chip-to-system)
4. [Four Decades of SWaP-C Optimization Practice](#4-four-decades-of-swap-c-optimization-practice)
5. [Multi-Objective Optimization Algorithms for SWaP-C](#5-multi-objective-optimization-algorithms-for-swap-c)
6. [Decision-Making Frameworks and Trade Study Methods](#6-decision-making-frameworks-and-trade-study-methods)
7. [Sensitivity Analysis and What-If Methods](#7-sensitivity-analysis-and-what-if-methods)
8. [Constraint Satisfaction and Feasibility Approaches](#8-constraint-satisfaction-and-feasibility-approaches)
9. [AI/ML-Driven Design Space Exploration](#9-aiml-driven-design-space-exploration)
10. [Composite Figures of Merit Used in Practice](#10-composite-figures-of-merit-used-in-practice)
11. [Five Optimization Methodologies for Branes](#11-five-optimization-methodologies-for-branes)
12. [References](#12-references)

---

## 1. The Energy-Delay Product Paradigm

The **Energy-Delay Product (EDP)** is the canonical composite figure of merit in VLSI/SoC
design, popularized by Gonzalez and Horowitz at Stanford in the late 1990s [1]. Its formula:

```
EDP = E × D    (units: Joule-seconds)
```

Where E = total energy consumed per operation and D = delay (latency). Minimizing EDP yields
a design point where "1% of energy can be traded for 1% of delay" — an equal-weight balance.

The **generalized form** allows adjustable weighting via exponents:

| Product | Formula | Emphasis |
|---------|---------|----------|
| E¹D¹ | Energy × Delay | Equal weight (standard EDP) |
| E¹D² | Energy × Delay² | Prioritizes speed over energy |
| E²D¹ | Energy² × Delay | Prioritizes energy over speed |

For VLSI gates specifically:

```
EDP = 2K(V_DD - V_t)^1.5 × C² × V_DD³
```

The related **Power-Delay Product (PDP)** measures energy per switching event:

```
PDP = P_avg × t_pd    (units: Joules)
```

The elegance of EDP comes from the fact that energy and delay are **commensurable** — they
reduce to the same underlying physics (voltage, capacitance, switching). The E^i D^j
generalization lets designers dial the tradeoff weight via exponents without arbitrary
normalization. This is precisely the property that makes a SWaP-C analog so difficult.

### References for Section 1

- [1] R. Gonzalez and M. Horowitz, "Energy Dissipation in General Purpose Microprocessors,"
  IEEE JSSC, 1996. [Stanford EE371 Handout](https://web.stanford.edu/class/archive/ee/ee371/ee371.1066/handouts/gonzalez_97.pdf)
- [2] "Energy Delay Product," ScienceDirect Topics.
  [Link](https://www.sciencedirect.com/topics/computer-science/energy-delay-product)
- [3] "Power-Delay Product," Wikipedia.
  [Link](https://en.wikipedia.org/wiki/Power%E2%80%93delay_product)

---

## 2. The Search for a Unified SWaP-C Metric

### 2.1 The Fundamental Challenge: Incommensurability

**Key finding: There is no universally adopted "SWaP-C Product" or single composite metric
analogous to EDP.** Despite 40+ years of defense/aerospace SWaP optimization, the field has
not converged on a single formula like `S^a × W^b × P^c × C^d`.

The reasons are structural:

1. **Incommensurable dimensions.** Volume (cm³), mass (g), power (W), and cost ($) have no
   natural physical product — unlike EDP where both factors reduce to physics. Any product
   requires arbitrary normalization.

2. **Mission-dependent weighting.** A UAV cares most about weight; a submarine about volume;
   a ground vehicle about cost; a wearable about power. No single set of exponents serves
   all applications.

3. **Non-linear interactions.** Reducing power doesn't linearly reduce weight — it changes
   the cooling solution, which changes the enclosure, which changes the volume. The
   cascading dependencies defeat simple multiplicative products.

4. **Community preference for MCDA.** The defense/aerospace community has relied on
   multi-criteria decision analysis (weighted scoring, AHP, TOPSIS) rather than single-number
   compression, precisely because stakeholders need to see and debate the individual tradeoffs.

### 2.2 The Closest Analogs

**PPAC / PPACt (Power-Performance-Area-Cost[-Time])**

The semiconductor industry's nearest equivalent to a unified SWaP-C framework. Used by
TSMC, Intel, Samsung, and Imec to evaluate process node transitions:

- Evolved from PPA (Power-Performance-Area) when wafer costs became dominant post-28nm
- TSMC uses **PPACt** (adding time-to-market) as a fifth dimension for N2 and beyond
- Not a single product formula but a multi-dimensional evaluation framework
- Each process node is characterized by its PPAC improvement over the predecessor

Imec has demonstrated PPAC benefits of heterogeneous sequential 3D integration at advanced
nodes, showing that system-level integration strategies can improve all four dimensions
simultaneously [4][5][6].

**SWaP-C2 (Size, Weight, Power, Cost, Cooling)**

The DoD's expansion of SWaP-C recognizing that thermal management is a dominant constraint
as systems shrink. Formalized by JIFCO (Joint Improvised-Threat Defeat). DARPA envisions
large SWaP reductions if cooling can be integrated at the chip level rather than added as
external mass and volume [7][8].

### 2.3 The Opportunity: Mission-Tunable Composite Scores

The E^i D^j pattern **is** directly adaptable to SWaP-C — not as a physics-derived product,
but as a **normalized weighted score** with application-class presets:

```
FoM = w_S · S̃ + w_W · W̃ + w_P · P̃ + w_C · C̃
```

Where S̃, W̃, P̃, C̃ are normalized to [0,1] relative to budgets or min/max across
candidates. The weights can be:

- **Pre-tuned** for application classes (drone, NUC, rack server, wearable)
- **Derived** via AHP pairwise comparison from stakeholder input
- **Swept** parametrically to show how rankings change with priorities

This is less elegant than EDP but more honest — it makes the weighting explicit rather than
hiding it behind physics.

### References for Section 2

- [4] "TSMC's N2 and the Power of PPACt," TSPA Semiconductor.
  [Link](https://tspasemiconductor.substack.com/p/tsmcs-n2-and-the-power-of-ppact-driving)
- [5] "Imec Demonstrates PPAC Benefit of Heterogeneous Sequential 3D Integration," Imec.
  [Link](https://www.imec-int.com/en/articles/imec-demonstrates-power-performance-area-cost-benefit-of-heterogeneous-sequential-3d-integration-for-advanced-cmos-nodes)
- [6] "PPA (Power, Performance, and Area)," Semiconductor Engineering.
  [Link](https://semiengineering.com/knowledge_centers/eda-design/definitions/ppa/)
- [7] "SWaP-C2 Fact Sheet," JIFCO/DoD.
  [Link](https://jifco.defense.gov/Press-Room/Fact-Sheets/Article-View-Fact-sheets/Article/1488195/size-weight-power-cost-and-cooling-swapc2/)
- [8] "SWaP-C and SWaP-C2 Principles," Sealevel Systems.
  [Link](https://www.sealevel.com/swap-swapc2)

---

## 3. Industry Figures of Merit Bridging Chip to System

Understanding how chip-level efficiency metrics cascade to system-level SWaP-C is essential
for any optimization framework.

### 3.1 Performance-Per-Watt Metrics

| Metric | Formula | Domain | Limitation |
|--------|---------|--------|------------|
| **FLOPS/W** | Peak FLOPS / TDP | HPC, GPU | Ignores utilization, size, cost |
| **TOPS/W** | INT8 TOPS / TDP | Edge AI accelerators | Precision-dependent; 100 TOPS at INT8 ≈ 25 TOPS at FP16 |
| **Samples/Joule** | Throughput / Wall Power | MLPerf Power | Full-system, standardized |
| **IPW** | Mean Accuracy / Mean Power | LLM inference | Capability-aware, not just throughput |
| **Tokens/W** | Token throughput / Power | LLM serving | Workload-specific |
| **GFLOPS/W** | LINPACK GFLOPS / Power | Green500 ranking | Single benchmark |

**TOPS/W** is the dominant edge AI metric but has a critical caveat from Hailo: "Comparing
AI accelerators based on TOPS alone will not help uncover the performance KPIs that are
interesting to the edge AI product designer — namely, throughput or latency." Real-world
performance depends on memory bandwidth, data precision, and workload characteristics [9].

**Intelligence Per Watt (IPW)** is an emerging metric from Stanford's Hazy Research group
(November 2025). It measures `mean accuracy across tasks / mean power draw during inference`,
benchmarked across 20+ LLMs on 8 hardware platforms with 1M real-world queries. Key finding:
5.3× improvement in IPW from 2023–2025 (3.1× from models, 1.7× from hardware) [10][11].

**MLPerf Power** (MLCommons) standardizes efficiency measurement using SPEC-certified power
analyzers (Yokogawa WT310), measuring full-system wall power during the execution phase.
Organizations sacrificed up to 50% energy efficiency going from 99% to 99.9% accuracy in
early rounds; the gap has narrowed with quantization [12].

### 3.2 Area and Density Metrics

| Metric | Formula | Use |
|--------|---------|-----|
| **TOPS/mm²** | Compute throughput / Die area | Silicon efficiency |
| **GOPS/mm²** | Operations / Die area | FPGA vs DSP comparison |
| **TOPS/cm³** | Throughput / System volume | Volumetric compute density |
| **TOPS/kg** | Throughput / System mass | Gravimetric compute density |

Quadric's Chimera GPNPU achieves 2.7× higher TOPS/mm² than competitors, demonstrating
that silicon area efficiency varies dramatically across architectures [13].

### 3.3 The Chip-to-System Bridge

| Chip Metric | System SWaP Impact |
|-------------|-------------------|
| TOPS/W or FLOPS/W | Determines power budget → cooling → weight, volume |
| Die area (mm²) | Drives package size → board area → enclosure size |
| TDP (Watts) | Determines cooling solution → weight, volume, parasitic power |
| Process node cost ($/wafer) | Drives unit cost (dominates BOM at advanced nodes) |
| Memory bandwidth (GB/s) | Determines memory chip count → size, weight, power |
| EDP | Composite energy-speed metric feeding system power/thermal |

AMD Versal AI Edge explicitly markets "low-SWaP" as a key differentiator for multi-mission
UAVs, bridging TOPS/W at the chip level to system-level SWaP requirements [14].

### 3.4 The Roofline Model

Not a single metric but a visual analysis framework:

```
Attainable Performance = min(Peak_FLOPS, Peak_BW × Operational_Intensity)
```

Where Operational Intensity = FLOPs / Bytes transferred. The roofline reveals whether a
workload is **compute-bound** or **memory-bound**, directly informing SWaP tradeoffs:
memory-bound workloads need bandwidth (bigger, heavier, more power for memory);
compute-bound need ALUs [15][16].

### References for Section 3

- [9] "Why TOPS Are Not Enough," Hailo.
  [Link](https://hailo.ai/blog/evaluating-edge-ai-accelerator-performance-why-tops-are-not-enough/)
- [10] "Intelligence Per Watt," Stanford Hazy Research.
  [Link](https://hazyresearch.stanford.edu/blog/2025-11-11-ipw)
- [11] "Intelligence Per Watt," Stanford Scaling Intelligence Lab.
  [Link](https://scalingintelligence.stanford.edu/pubs/ipw/)
- [12] "MLPerf Power Measurement," arXiv:2410.12032.
  [Link](https://arxiv.org/html/2410.12032v1)
- [13] "Quadric Chimera GPNPU," SemiWiki Forum.
  [Link](https://semiwiki.com/forum/threads/quadric-chimera-gpnpu.20672/)
- [14] "Versal AI Edge Series," AMD.
  [Link](https://www.amd.com/en/products/adaptive-socs-and-fpgas/versal/ai-edge-series.html)
- [15] S. Williams, A. Waterman, D. Patterson, "Roofline: An Insightful Visual Performance
  Model," Communications of the ACM, 2009.
  [Link](https://people.eecs.berkeley.edu/~kubitron/cs252/handouts/papers/RooflineVyNoYellow.pdf)
- [16] "Roofline Performance Model," Wikipedia.
  [Link](https://en.wikipedia.org/wiki/Roofline_model)

---

## 4. Four Decades of SWaP-C Optimization Practice

### 4.1 1980s–1990s: Weight and Power Budgets (NASA/DoD)

The foundational methodology. NASA's Systems Engineering Handbook treats mass and power as
**budgeted resources** with margin allocations [17]:

- **Phase B (Formulation):** ~30% margin on mass, power, cost
- **Margins consumed** as design matures through lifecycle phases
- **Iterative allocation:** System engineer provides initial estimates from historical data;
  subsystem designers refine
- "The fundamental resource of a spacecraft is mass" — system design must stay within
  launch vehicle limits

Typical subsystem mass allocation from spacecraft heritage data:

| Subsystem | Mass % |
|-----------|--------|
| Payload | 31% |
| Structure/Mechanisms | 27% |
| Power (incl. harness) | 21% |
| Attitude Control | 6% |
| On-Board Processing | 5% |
| Thermal Control | 2% |
| Telemetry/Comms | 2% |
| Propulsion | 3% |
| Margin | 3% |

**Economic impact:** Reducing a DoD ISR UAV platform by one pound saves approximately
$30,000 in operational costs [18].

### 4.2 1990s–2000s: Trade Study Matrices and MCDA

Weighted decision matrices became the standard tool. Key frameworks:

- **MITRE Trade Study Process** — Structured methodology for comparing alternatives against
  weighted criteria [19]
- **Pugh Matrix** — Binary (+/-/S) comparison against a datum design for concept screening
- **AHP (Analytic Hierarchy Process)** — Mathematically rigorous pairwise comparison for
  deriving criterion weights (Thomas L. Saaty, 1970s) [20]
- **INCOSE Swing Weight Matrix** — Weights depend on both importance AND variation range;
  published by Parnell & Trainor (2009) [21]

These methods work well for down-selecting from a handful of candidates but don't scale to
continuous design spaces with thousands of possible configurations.

### 4.3 2000s–2010s: Evolutionary Multi-Objective Optimization

Deb's **NSGA-II** (2002) brought Pareto-based optimization to hardware design [22]:

- Fast non-dominated sorting: O(MN²) complexity
- Crowding distance for diversity preservation
- Elitism to preserve Pareto-optimal solutions
- Applied to defense weapon-target assignment, system-of-systems optimization, and
  electronic system design

**NSGA-III** (2014) extended this to many-objective problems (>3 objectives) using
reference-direction selection — directly relevant to SWaP-C's 4–6 objective regime [23].

### 4.4 2010s–2020s: Surrogate-Assisted and Bayesian Optimization

When evaluations are expensive (FPGA synthesis, RTL simulation, physical prototyping):

- **HyperMapper** (Stanford DAWN / Luigi Nardi et al., 2019): Multi-objective black-box
  optimizer using Bayesian Optimization for hardware DSE. Handles categorical/ordinal
  variables, unknown feasibility constraints, user prior knowledge. Achieved 8× improvement
  in sampling budget for FPGA accelerator tuning [24].

- **Bayesian Optimization for DNN Accelerator DSE** (Reagen et al., 2017, Harvard):
  Found optimal configurations in fewer than 50 samples, 2–5× more sample-efficient
  than genetic algorithms [25].

### 4.5 2020s: AI-Driven Design Space Exploration

- **Synopsys DSO.ai**: Industry's first autonomous AI for chip design. Uses RL to search
  design spaces for optimal PPA, operating tens-to-thousands of exploration vectors in
  real time [26].
- **Cadence Cerebrus**: ML-driven full-flow optimization where engineers specify goals and
  the system automatically meets PPA targets [27].
- **Google Chip Placement** (2020): Poses VLSI placement as RL, generating placements
  comparable to human experts in under 6 hours vs. weeks of manual effort [28].
- **Transfer learning across design families**: Train on one chip block, warm-start on
  another. Active-CEM achieves 1.58× performance improvement over AutoDSE and 2.7×
  runtime reduction when transitioning to new toolchains [29].

### 4.6 Standards, Programs, and Tools

**DoD/DARPA Programs:**

| Program | Year | Investment | Focus |
|---------|------|-----------|-------|
| **ERI** (Electronics Resurgence Initiative) | 2017 | $1.5B / 5yr | Beyond Moore's Law |
| **CHIPS** (Common Heterogeneous Integration) | 2017 | Part of ERI | Modular chiplets |
| **DAHI** (Diverse Accessible Heterogeneous Integration) | 2018 | Part of ERI | InP + GaN + Si CMOS |
| **NGMM** (Next-Gen Microelectronics Manufacturing) | 2024 | $840M+ | 3D military chiplets |
| **PIPES** (Photonics in the Package) | 2018 | Part of ERI | Optical interconnects |

**Standards:**

| Standard | Scope |
|----------|-------|
| **MOSA / SOSA / OpenVPX** (VITA 65) | Modular open systems, standardized form factors |
| **MIL-HDBK-338** | Electronic reliability design handbook |
| **MIL-HDBK-61B** | Configuration management (SWaP budget baselines) |
| **IRDS** (IEEE) | International Roadmap for Devices and Systems (successor to ITRS) |
| **DoD Source Selection (2022)** | VATEP methodology for scoring SWaP-C in procurement |

**Estimation Tools:**

- **Galorath SEER Suite**: Industry-standard parametric cost estimation for DoD. SEER-H
  (Hardware), SEER-SEM (Software), SEER-MFG (Manufacturing). Integrates with digital
  twins for "cost twin" modeling. Evaluates module configurations against SWaP-C [30][31].

### References for Section 4

- [17] NASA Systems Engineering Handbook, SP-2016-6105 Rev2.
  [Link](https://www.nasa.gov/wp-content/uploads/2018/09/nasa_systems_engineering_handbook_0.pdf)
- [18] "Optimizing SWaP-C in Defense & Aerospace 2025," Galorath.
  [Link](https://galorath.com/blog/optimizing-swap-c-defense-aerospace-2025/)
- [19] "The Trade Study Process," MITRE.
  [Link](https://www.mitre.org/sites/default/files/2021-11/prs-21-0522-the-trade-study-process.pdf)
- [20] "Analytic Hierarchy Process," Wikipedia.
  [Link](https://en.wikipedia.org/wiki/Analytic_hierarchy_process)
- [21] G. Parnell and T. Trainor, "Using the Swing Weight Matrix to Weight Multiple
  Objectives," INCOSE International Symposium, 2009.
  [Link](https://incose.onlinelibrary.wiley.com/doi/abs/10.1002/j.2334-5837.2009.tb00949.x)
- [22] K. Deb et al., "A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II,"
  IEEE TEC, 2002. [Link](https://www.cse.unr.edu/~sushil/class/gas/papers/nsga2.pdf)
- [23] "NSGA-III in pymoo." [Link](https://pymoo.org/algorithms/moo/nsga3.html)
- [24] L. Nardi et al., "HyperMapper: Practical Design Space Exploration," Stanford DAWN.
  [Link](https://dawn.cs.stanford.edu/publications/hypermapper/practical-design-space-exploration)
- [25] B. Reagen et al., "A Case for Efficient Accelerator Design Space Exploration via
  Bayesian Optimization," IEEE ISLPED, 2017.
  [Link](https://ieeexplore.ieee.org/document/8009208/)
- [26] "DSO.ai," Synopsys. [Link](https://www.synopsys.com/ai/ai-powered-eda/dso-ai.html)
- [27] "AI in Chip Design," Cadence.
  [Link](https://www.cadence.com/en_US/home/explore/ai-chip-design.html)
- [28] A. Mirhoseini et al., "Chip Placement with Deep Reinforcement Learning,"
  arXiv:2004.10746. [Link](https://arxiv.org/abs/2004.10746)
- [29] "Deep Learning for Chip Design," UCLA VAST Lab.
  [Link](https://vast.cs.ucla.edu/projects/deep-learning-chip-design)
- [30] "SEER Suite," Galorath. [Link](https://galorath.com/seer/)
- [31] "Galorath SWaP-C Optimization with SEER," ExecutiveBiz.
  [Link](https://executivebiz.com/2025/01/galorath-matt-mcdonald-swap-c-optimization-seer/)

---

## 5. Multi-Objective Optimization Algorithms for SWaP-C

### 5.1 NSGA-II (Non-dominated Sorting Genetic Algorithm II)

The most widely used evolutionary MOO algorithm, directly applicable to SWaP-C where size,
weight, power, and cost form 4 conflicting objectives.

**Algorithm:**

1. Initialize a random population of N solutions
2. **Non-dominated sorting:** Partition into fronts F₁, F₂, ... where F₁ = Pareto-optimal
3. **Crowding distance:** Sum of normalized distances to neighbors along each objective axis
4. **Binary tournament selection:** Compare by (a) rank, then (b) crowding distance
5. Crossover and mutation to produce offspring
6. **Elitism:** Merge parent + offspring, re-sort, fill next generation from best fronts

**Constraint handling** (Deb's feasibility rules):
- Both feasible → choose better objective value
- One feasible, one infeasible → choose feasible
- Both infeasible → choose smaller constraint violation

**Typical SWaP-C formulation:**

```
minimize  f₁(x) = Volume(x)     [cm³]
minimize  f₂(x) = Weight(x)     [grams]
minimize  f₃(x) = Power(x)      [watts]
minimize  f₄(x) = Cost(x)       [USD]
subject to:
  g₁(x): Performance(x) >= P_min     [TOPS or FPS threshold]
  g₂(x): T_junction(x) <= T_max      [thermal envelope]
```

**Tool:** pymoo — Python framework implementing NSGA-II, NSGA-III, R-NSGA-II, MOEA/D,
AGE-MOEA, SMS-EMOA. Handles mixed-variable types, constraint handling, and provides
visualization [32][33].

### 5.2 NSGA-III (Reference-Point Based)

Extends NSGA-II to **many-objective** problems (4+ objectives). NSGA-II's crowding distance
breaks down with >3 objectives because nearly all solutions become non-dominated. NSGA-III
replaces crowding distance with **reference-point association**: uniformly distributed
reference points on a normalized hyperplane guide selection, ensuring well-spread Pareto
fronts in high dimensions [23].

Directly relevant to SWaP-C with 6 objectives (power, latency, area, cost, weight, volume).

### 5.3 Bayesian Multi-Objective Optimization

When evaluations are expensive, Gaussian Process surrogates + acquisition functions minimize
the number of evaluations needed.

**qNEHVI (q-Noisy Expected Hypervolume Improvement):**
- Acquisition function that maximizes expected improvement in the Pareto front hypervolume
- Handles noisy observations (manufacturing variation, measurement error)
- GP lengthscales provide parameter sensitivity for free
- Typically converges in 100–200 evaluations

**Key frameworks:**
- **BoTorch** (Meta): GP-based Bayesian optimization with qNEHVI for multi-objective [34]
- **HyperMapper** (Stanford): Handles categorical variables, unknown feasibility constraints,
  user prior knowledge [24]

### 5.4 MAP-Elites and Quality-Diversity

**MAP-Elites** (Multi-dimensional Archive of Phenotypic Elites) takes a different approach:
instead of finding a single Pareto front, it **illuminates the entire design space** by
filling a grid where each cell represents a unique design niche.

For SWaP-C, the grid dimensions (descriptors) could be:
- Power envelope bracket (<5W, 5–15W, 15–30W, 30–75W, >75W)
- Form factor class (chip-scale, module, board, box)
- Deployment tier (edge, fog, cloud)

**MOME (Multi-Objective MAP-Elites)** extends this by storing a local Pareto front in each
cell rather than a single solution. The principal metric is the MOQD-score (sum of local
hypervolumes across all cells) [35][36].

### 5.5 Scalarization Methods

**Weighted Sum:**
```
minimize F(x) = w_S·Size(x) + w_W·Weight(x) + w_P·Power(x) + w_C·Cost(x)
```
Simple but cannot find solutions on non-convex Pareto fronts.

**Epsilon-Constraint (preferred for Pareto front generation):**
```
minimize Power(x)
subject to: Size(x) ≤ ε_S,  Weight(x) ≤ ε_W,  Cost(x) ≤ ε_C
```
Sweep ε values to trace the complete front. Handles non-convex fronts [37].

### References for Section 5

- [32] "NSGA-II," pymoo. [Link](https://pymoo.org/algorithms/moo/nsga2.html)
- [33] K. Deb et al., "A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II."
  [Link](https://www.cse.unr.edu/~sushil/class/gas/papers/nsga2.pdf)
- [34] "BoTorch: Bayesian Optimization in PyTorch." [Link](https://botorch.org/)
- [35] "Multi-Objective MAP-Elites (MOME)," arXiv:2202.03057.
  [Link](https://arxiv.org/abs/2202.03057)
- [36] "Quality-Diversity Optimization," QD Papers Collection.
  [Link](https://quality-diversity.github.io/papers.html)
- [37] "Multi-Objective Optimization," Wikipedia.
  [Link](https://en.wikipedia.org/wiki/Multi-objective_optimization)

---

## 6. Decision-Making Frameworks and Trade Study Methods

### 6.1 Pugh Matrix (Concept Selection)

Developed by Stuart Pugh for early-stage hardware downselection [38]:

1. Define evaluation criteria (size, weight, power, cost, reliability, thermal, availability)
2. Select a **reference/datum** design (current baseline)
3. Score each alternative: **+1** (better), **0** (same), **-1** (worse) per criterion
4. Sum scores; highest total wins initial screening
5. Optionally apply weights (1–5 scale) before summing

**Strengths for SWaP-C:** Quick to execute, forces structured comparison, surfaces relative
strengths/weaknesses. Good for narrowing from 10+ candidates to 3–4 finalists.

### 6.2 Analytic Hierarchy Process (AHP)

Developed by Thomas L. Saaty (1970s), AHP derives criteria weights from expert pairwise
comparisons [20]:

1. **Decompose** decision into hierarchy: Goal → Criteria (S, W, P, C) → Alternatives
2. **Pairwise comparison** using the Saaty 1–9 Scale:
   - 1 = Equal importance, 3 = Moderate, 5 = Strong, 7 = Very strong, 9 = Extreme
3. **Build comparison matrix A** where a_ij = preference of criterion i over j
4. **Priority vector** = principal eigenvector of A, normalized
5. **Consistency check:**
   - CI = (λ_max − n) / (n − 1)
   - CR = CI / RI (Random Index from Monte Carlo tables)
   - CR ≤ 0.10 is acceptable

**For SWaP-C:** Particularly valuable when stakeholders disagree on relative importance.
The pairwise comparison forces explicit articulation and mathematical verification.

### 6.3 TOPSIS (Technique for Order Preference by Similarity to Ideal Solution)

Developed by Hwang and Yoon (1981), ranks alternatives by geometric distance to ideal and
anti-ideal solutions [39]:

1. **Normalize:** r_ij = x_ij / √(Σ_k x_kj²)
2. **Weight:** v_ij = w_j × r_ij
3. **Ideal solutions:**
   - V⁺ = {max(v_ij) for benefit criteria, min(v_ij) for cost criteria}
   - V⁻ = {min(v_ij) for benefit criteria, max(v_ij) for cost criteria}
4. **Distances:**
   - d_i⁺ = √(Σ_j (v_ij − V_j⁺)²) (distance to ideal)
   - d_i⁻ = √(Σ_j (v_ij − V_j⁻)²) (distance to anti-ideal)
5. **Closeness coefficient:** C_i = d_i⁻ / (d_i⁺ + d_i⁻) ∈ [0,1]; higher = better

**For SWaP-C:** All four dimensions are "cost criteria" (lower is better), while performance
metrics are "benefit criteria." TOPSIS naturally handles this mixed landscape.

### 6.4 Quality Function Deployment (QFD) / House of Quality

Originated by Yoji Akao (1966, Japan). Maps customer requirements (WHATs) to engineering
characteristics (HOWs) through a matrix structure [40][41]:

- **WHATs:** "Fits in 2U payload bay," "Battery life > 4 hours," "Unit cost < $5000"
- **HOWs:** Processor TDP, heatsink volume, PCB layer count, DRAM capacity
- **Roof:** Correlation matrix capturing engineering parameter interactions
- **Relationship matrix:** Strong/Medium/Weak correlations

### 6.5 Value Engineering (VA/VE)

Pioneered by Lawrence Miles at GE during WWII. Core formula [42][43]:

```
Value = Function / Cost
```

Systematically evaluates every component: "Does this function justify its cost? Can a
cheaper alternative deliver the same function?"

Formalized in FAR Part 48 for defense acquisition. DoD guidebook SD-24 (February 2025
revision) provides the current framework. VE Change Proposals (VECPs) provide contractors
substantial financial incentives for cost reduction [44].

### References for Section 6

- [38] "Pugh Matrix," ASQ. [Link](https://asq.org/quality-resources/decision-matrix)
- [39] "TOPSIS," Wikipedia. [Link](https://en.wikipedia.org/wiki/TOPSIS)
- [40] "Quality Function Deployment," Wikipedia.
  [Link](https://en.wikipedia.org/wiki/Quality_function_deployment)
- [41] J. Hauser and D. Clausing, "The House of Quality," Harvard Business Review, 1988.
  [Link](https://hbr.org/1988/05/the-house-of-quality)
- [42] "VAVE Explained for OEMs," ESCATEC.
  [Link](https://www.escatec.com/blog/value-analysis-and-value-engineering-va/ve-explained-for-oems)
- [43] "VAVE Cost Reduction," Titoma.
  [Link](https://titoma.com/blog/vave-cost-reduction/)
- [44] "SD-24 Value Engineering Guidebook," DoD CTO, February 2025.
  [Link](https://www.cto.mil/wp-content/uploads/2025/02/SD-24-VE-Guidebook-25Feb2025-Cleared-1.pdf)

---

## 7. Sensitivity Analysis and What-If Methods

### 7.1 Tornado Diagrams

A horizontal bar chart showing how one-at-a-time variation of each input parameter affects
a key output, ordered from most to least influential [45]:

- Vary each design parameter ±30% while holding others at baseline
- Plot resulting range of each SWaP-C metric
- Longest bars = most sensitive parameters = where to focus engineering effort

**For SWaP-C:** Resolves disagreements about resource allocation. If processor TDP dominates
the tornado for weight, thermal design gets priority over connector miniaturization.

### 7.2 Monte Carlo Simulation

Propagates probability distributions through the SWaP model to produce output distributions
with confidence intervals [46]:

- **Inputs:** TDP = Normal(15W, 2W), Mass = Uniform(80g, 120g), Cost = Triangular($45, $60, $90)
- **Process:** 10,000+ samples from each distribution, propagated through BOM model
- **Output:** P10/P50/P90 estimates for system-level weight, volume, cost
- **Key insight:** Identifies probability of exceeding budget thresholds

For embedded systems, uncertainty-aware mapping has been formulated as MOO, generating
robust 3D Pareto frontiers across reliability, performance, and energy [47].

### 7.3 Taguchi Methods / Design of Experiments (DOE)

Developed by Dr. Genichi Taguchi, uses orthogonal arrays to systematically explore factor
effects with minimal experiments [48][49]:

**Signal-to-Noise (S/N) ratios:**
- Smaller is better: S/N = −10 log₁₀(mean(y²)) — for minimizing SWaP
- Larger is better: S/N = −10 log₁₀(mean(1/y²)) — for maximizing performance
- Nominal is best: S/N = 10 log₁₀(mean²/variance)

**Orthogonal arrays:** An L18 array explores 7 factors at 3 levels in only 18 experiments
instead of 3⁷ = 2,187 full factorial. Factors: processor model, memory type, power
regulator, heatsink material, PCB layers, clock speed, bus width.

**Robust design:** Find settings where performance is insensitive to noise factors
(temperature variation, manufacturing tolerance, supply voltage fluctuation).

### 7.4 Response Surface Methodology (RSM)

Fits polynomial surrogate models to experimental data for continuous optimization [50]:

```
y = β₀ + Σ βᵢxᵢ + Σ βᵢᵢxᵢ² + Σ βᵢⱼxᵢxⱼ + ε
```

After Taguchi screening identifies the top 3–4 factors, RSM builds a continuous surface.
Contour plots and 3D surfaces reveal the optimal operating region. Multi-response overlays
identify the feasible design window where all objectives are satisfactorily met.

### References for Section 7

- [45] "Tornado Diagram," Wikipedia.
  [Link](https://en.wikipedia.org/wiki/Tornado_diagram)
- [46] "Mastering Uncertainty Analysis in Systems Engineering," NumberAnalytics.
  [Link](https://www.numberanalytics.com/blog/mastering-uncertainty-analysis-systems-engineering)
- [47] "Uncertainty in NoC Mapping," ScienceDirect.
  [Link](https://www.sciencedirect.com/science/article/abs/pii/S0141933120306554)
- [48] "Taguchi Methods," Wikipedia.
  [Link](https://en.wikipedia.org/wiki/Taguchi_methods)
- [49] "Taguchi Approach to Design Optimization," NASA.
  [Link](https://ntrs.nasa.gov/api/citations/20040121019/downloads/20040121019.pdf)
- [50] "Response Surface Methodology," Wikipedia.
  [Link](https://en.wikipedia.org/wiki/Response_surface_methodology)

---

## 8. Constraint Satisfaction and Feasibility Approaches

### 8.1 Top-Down SWaP Budgeting

The standard systems engineering approach [17][51]:

1. Establish total system target (from platform or launch vehicle constraint)
2. Subtract fixed allocations (propellant, structure)
3. Reserve system-level margin (10–25% depending on design maturity)
4. Distribute remaining budget to subsystems using historical ratios
5. Iterate as subsystem designers refine estimates

**Margin management:** "If you hold too much margin, subsystem designs become overly
constrained and cost rises. If you hold too little, you end up with an assembled system
that doesn't work" [52].

### 8.2 Constraint Handling in Evolutionary Optimization

Three approaches for handling SWaP budgets as hard constraints [53]:

**Penalty function:**
```
F_penalized(x) = f(x) + R × Σ max(0, gᵢ(x))²
```
Problem: ideal penalty factor R is unknown a priori.

**Deb's feasibility rules** (used in NSGA-II):
- Feasible vs. feasible → better objective
- Feasible vs. infeasible → always feasible
- Infeasible vs. infeasible → smaller violation

**Adaptive/staged approaches:**
1. Infeasible phase: focus on reducing constraint violation
2. Semi-feasible phase: balance feasibility and objective improvement
3. Feasible phase: standard objective optimization

### 8.3 SWaP-Constrained Embedded Computing for AI

The Air Force Research Laboratory (2019) explicitly defined SWaP envelopes for edge AI,
framing SWaP constraints as hard feasibility boundaries within which AI systems must
operate [54][55].

### References for Section 8

- [51] "NanoStar Systems Engineering," GitLab.
  [Link](https://nanostar-project.gitlab.io/main/source/preliminary-design/systems.html)
- [52] "Margins," NASA Psyche Mission Blog.
  [Link](https://medium.com/the-nasa-psyche-mission-journey-to-a-metal-world/margins-e19f3dce28b6)
- [53] C. A. Coello, "Theoretical and Numerical Constraint-Handling Techniques."
  [Link](https://www2.cs.uh.edu/~ceick/6367/Coello_CHNOPT.pdf)
- [54] "SWaP-Constrained Embedded Computing for AI," Medium.
  [Link](https://alexmoltzau.medium.com/what-is-swap-constrained-embedded-computing-for-artificial-intelligence-f846cbccdf1e)
- [55] "SWaP Embedded Computing AI," Military Aerospace Electronics.
  [Link](https://www.militaryaerospace.com/computers/article/14174721/swap-embedded-computing-artificial-intelligence)

---

## 9. AI/ML-Driven Design Space Exploration

### 9.1 Hardware-Aware Neural Architecture Search (HW-NAS)

Explicitly incorporates hardware constraints into the architecture search [56][57]:

- **MARCO:** Multi-agent RL with conformal optimization for edge devices
- **ESC-NAS:** Hardware-aware NAS for audio classification
- **Early-exit NAS:** Integrates quantization and hardware allocation for edge accelerators

Typical formulation:

```
maximize  Accuracy(architecture)
subject to:
  Latency(architecture, target_hw) ≤ L_max
  Energy(architecture, target_hw) ≤ E_max
  ModelSize(architecture) ≤ Memory_max
```

### 9.2 Reinforcement Learning for Design Exploration

Commercial RL-based EDA tools:

| Tool | Vendor | Approach |
|------|--------|----------|
| **DSO.ai** | Synopsys | RL for PPA design space exploration |
| **Cerebrus** | Cadence | ML-driven full-flow optimization |

Research advances:
- **Multi-Agent RL (MARL)** for microprocessor DSE: each agent optimizes a subsystem.
  MARL consistently outperforms single-agent RL [58].
- **FastTuner:** GNN + Transformer for rapid DSE with attention-based parameter tuning.

### 9.3 Surrogate-Assisted Optimization

**Sherlock Framework** (ACM TODAES) adaptively selects between surrogates [59]:

| Surrogate | Best For |
|-----------|----------|
| Gaussian Process (GP/Kriging) | Smooth, low-dimensional spaces; provides uncertainty |
| Random Forest (RF) | Rough, high-dimensional spaces with categorical variables |
| Radial Basis Functions (RBF) | Good interpolation for continuous spaces |

Key finding: "The random forest better models some design spaces while the Gaussian process
is best for others." Adaptive selection is crucial.

### 9.4 Transfer Learning Across Design Families

- Google's chip placement RL agent improves at generating optimized placements for
  previously unseen chip blocks as it trains on more examples [28]
- **Active-CEM:** Achieves 1.58× performance improvement and 2.7× runtime reduction when
  transferring to new toolchains [29]
- **Chiplet reuse:** Designs composed like building blocks, with placement/routing templates
  transferable across device sizes

### References for Section 9

- [56] "Hardware-Aware NAS for Edge," arXiv:2512.04705.
  [Link](https://arxiv.org/abs/2512.04705)
- [57] "NAS for Resource-Constrained Devices," IET.
  [Link](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/cps2.12058)
- [58] "MARL for Microprocessor DSE," arXiv:2211.16385.
  [Link](https://arxiv.org/abs/2211.16385)
- [59] "Sherlock: Adaptive Surrogate Selection for FPGA HLS," ACM TODAES.
  [Link](https://dl.acm.org/doi/10.1145/3511472)

---

## 10. Composite Figures of Merit Used in Practice

### 10.1 Constructible Density Metrics

No single standardized "SWaP density" metric exists, but the following are constructible
from first principles and used in various contexts:

| Metric | Formula | Units | Use Case |
|--------|---------|-------|----------|
| Volumetric compute density | TOPS / Volume | TOPS/cm³ | Rack density, form factor |
| Gravimetric compute density | TOPS / Mass | TOPS/kg | Payload budgets |
| Cost-normalized performance | TOPS / (W × $) | TOPS/(W·$) | TCO optimization |
| Energy efficiency | TOPS / W | TOPS/W | Battery life, thermal |
| Area efficiency | TOPS / mm² | TOPS/mm² | Silicon utilization |

### 10.2 Application-Class Composite Scores

Mission-specific weighted scores with pre-tuned profiles:

```
FoM_drone     = 0.10·S̃ + 0.35·W̃ + 0.30·P̃ + 0.15·C̃ + 0.10·T̃
FoM_rack      = 0.30·S̃ + 0.05·W̃ + 0.25·P̃ + 0.25·C̃ + 0.15·T̃
FoM_wearable  = 0.25·S̃ + 0.25·W̃ + 0.35·P̃ + 0.10·C̃ + 0.05·T̃
FoM_vehicle   = 0.10·S̃ + 0.10·W̃ + 0.20·P̃ + 0.40·C̃ + 0.20·T̃
```

Where S̃, W̃, P̃, C̃, T̃ ∈ [0,1] are normalized relative to budgets (0 = at budget,
1 = zero consumption).

### 10.3 Multidisciplinary Design Optimization Frameworks

**OpenMDAO** (NASA Glenn): Open-source Python framework for multidisciplinary design
optimization [60]:
- Automatic analytic multidisciplinary derivatives
- Handles hundreds to thousands of design variables
- Epsilon-constraint and weighted-sum scalarization for multi-objective

**IRDS (IEEE):** International Roadmap for Devices and Systems tracks PPAC scaling for
logic and memory, predicting ground-rule scaling saturation ~2027. GAA transistor
transition on track. 3D integration accommodates area reduction limits [61].

### References for Section 10

- [60] "OpenMDAO," NASA Glenn. [Link](https://openmdao.org/)
- [61] "IRDS 2022 Executive Summary," IEEE.
  [Link](https://irds.ieee.org/images/files/pdf/2022/2022IRDS_ES.pdf)

---

## 11. Five Optimization Methodologies for Branes

Based on the research above, here are five methodologies ordered from simplest to most
sophisticated, each addressing a distinct user need.

### Methodology 1: SWaP-C Scorecard with Configurable Figures of Merit

**What it is:** A normalized scoring system where each SWaP-C metric is measured against a
budget, producing utilization percentages and composite scores. Users define **mission
profiles** that set the relative importance of each dimension.

**The key insight:** Instead of a universal SWaP-C product, define application-class FoMs
with pre-tuned weights:

```
FoM_drone     = 0.10·S̃ + 0.35·W̃ + 0.30·P̃ + 0.15·C̃ + 0.10·T̃
FoM_rack      = 0.30·S̃ + 0.05·W̃ + 0.25·P̃ + 0.25·C̃ + 0.15·T̃
FoM_wearable  = 0.25·S̃ + 0.25·W̃ + 0.35·P̃ + 0.10·C̃ + 0.05·T̃
```

Users can also derive weights via AHP pairwise comparison or define custom profiles.

**What the user gets:**
- A single SWaP-C score (0–100) for any design point
- Radar/spider charts showing budget envelope utilization
- Utilization heatmaps across the Pareto front colored by composite score
- The knee point as the design with the highest FoM on the Pareto front

**Why it works:** Answers the question every PM asks: "which design is better?" with a
single number, while keeping the weighting transparent and adjustable. The SWaP-C analog
of EDP — but with explicit, mission-tunable weights instead of physics-derived exponents.

**Implementation:** Extends `branes swap check` scorecard with composite scoring and named
mission profiles in the spec system.

---

### Methodology 2: Sensitivity-First What-If Explorer

**What it is:** Before optimizing, understand which parameters matter. Combines tornado
diagrams with Taguchi DOE for interaction effects, producing a ranked list of design levers
ordered by impact.

**How it works:**

1. **Tornado analysis:** Vary each of the 9+ design variables ±1σ while holding others at
   baseline. Plot the resulting swing in each objective.

2. **Taguchi L18 screening:** For 9 factors at mixed levels, 18 evaluations (vs. thousands
   for full factorial). Reveals interaction effects that one-at-a-time misses.

3. **Response surface:** For the top 3–4 factors, fit a 2nd-order polynomial. Contour plots
   show the feasible design window.

**What the user gets:**
- Tornado diagram: "Clock frequency dominates power; package type dominates weight"
- Factor ranking table with importance scores
- Contour maps showing the design sweet spot
- Actionable guidance: "Don't bother optimizing NoC width — it barely moves the needle"

**Why it works:** Engineers waste enormous effort optimizing parameters that barely matter.
This methodology answers "where should I focus?" before "what's the optimal point?"

**Implementation:** New `branes swap sensitivity` command running tornado + Taguchi pipeline.

---

### Methodology 3: Pareto-Guided Exploration with TOPSIS Ranking

**What it is:** The existing MAP-Elites → Bayesian BO / NSGA-III pipeline, enhanced with
three additions that make the Pareto front actionable:

**Enhancement A — Marginal Rate of Substitution (MRS):**
At the knee point, compute the slope of the Pareto front in each pairwise objective plane:
"Moving from the knee costs 12g of weight per watt saved" or "$3.20 per cm³ of volume
reduction." Makes tradeoffs concrete.

**Enhancement B — Pareto Front Clustering:**
For fronts with 50+ points, cluster into 3–5 design families using k-means on the
objective space. Each cluster = a qualitatively different design strategy (e.g., "low-power
passively cooled," "high-performance actively cooled," "cost-optimized commodity").

**Enhancement C — TOPSIS Ranking:**
Apply TOPSIS to the Pareto front using mission-profile weights from Methodology 1.
Produces a single ranked list — the best of the non-dominated designs for this specific
mission. Closeness coefficient C_i ∈ [0,1] gives a natural score.

**What the user gets:**
- Pareto front with clustered design families
- Marginal exchange rates at the knee
- TOPSIS-ranked shortlist with scores
- `branes swap rank --profile drone` to re-rank for different missions

**Why it works:** Raw 6-objective Pareto fronts with 50+ points are overwhelming.
Clustering + TOPSIS + MRS transforms "here are your options" into "here are your three
strategies, here's the best for your mission, and here's the marginal tradeoff."

---

### Methodology 4: Delta Attribution Comparator

**What it is:** A structured methodology for comparing two design points that attributes
every SWaP-C delta to a specific design change. The "diff" tool for hardware.

**How it works:**

1. **Decompose the delta:** Walk the BOM tree and attribute each metric change to the
   component(s) that changed. If A uses passive cooling and B uses active fan:
   - Weight: +X g from fan, −Y g from smaller heatsink
   - Volume: +Z cm³ from fan, −W cm³ from heatsink
   - Cost: +$F for fan, −$H for heatsink
   - Thermal: θ_sa drops from 5.0 to 1.5 °C/W

2. **Waterfall visualization:** Each bar segment = one component's contribution to the
   total delta.

3. **Dominance analysis:** For each objective, flag improvement (✓) or regression (✗).
   If B dominates A, state explicitly.

4. **Parametric sweep:** Sweep a single parameter and show how all 6 objectives evolve.

**What the user gets:**
- `branes swap diff --left "28nm,BGA,passive" --right "7nm,FCBGA,active_fan"`
- Waterfall charts showing which component drove each change
- Parametric sweep curves
- Clear dominance/tradeoff verdict

**Why it works:** Answers "why?" — not just "B is lighter" but "B is lighter because the
5nm die is 60% smaller → smaller package → smaller PCB → 30% less enclosure volume."

---

### Methodology 5: Budget-Constrained Feasibility with Monte Carlo Margins

**What it is:** Top-down systems engineering with uncertainty bands. Answers: "What's the
probability this design stays within my 200g payload budget given manufacturing tolerances?"

**How it works:**

1. **Budget allocation:** User specifies system-level budgets. Tool decomposes into
   subsystem allocations using BOM hierarchy, with configurable margin reserves (15%).

2. **Uncertainty propagation:** Each estimator returns a distribution:
   - Die cost: Triangular($12, $18, $45) from yield uncertainty
   - Heatsink weight: Normal(40g, 5g) from manufacturing tolerance
   - PCB area: Uniform(±10%) from routing complexity
   Monte Carlo (10K samples) propagates through BOM to produce P10/P50/P90 estimates.

3. **Margin tracking:** As design matures (concept → preliminary → detailed → prototype),
   margins are consumed. Traffic-light status: green > 15%, yellow 5–15%, red < 5%.

4. **Probabilistic feasibility gates:** Instead of binary PASS/FAIL: "92% probability of
   meeting weight budget, 67% probability of meeting cost budget."

**What the user gets:**
- `branes swap budget --weight 200 --volume 150 --power 15 --cost 500 --confidence 0.90`
- Probability distributions for each metric
- Margin waterfall tracking design maturity
- Risk-ranked metrics most likely to blow budget

**Why it works:** This is how spacecraft and military programs actually manage SWaP. Monte
Carlo transforms "it fits" into "it fits with 92% confidence" — what program managers
need for go/no-go decisions.

---

### Methodology Summary: The Five-Layer Workflow

```
                    ┌─────────────────────────────┐
                    │  1. Scorecard + FoM          │  "How good is this design?"
                    │     Single-number scoring     │  Quick assessment, comparison
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │  2. Sensitivity Explorer      │  "What parameters matter?"
                    │     Tornado + Taguchi + RSM   │  Focus engineering effort
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │  3. Pareto Explorer           │  "What are my options?"
                    │     MOO + Clustering + TOPSIS │  Map the tradeoff surface
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │  4. Configuration Comparator  │  "Why is B better than A?"
                    │     Delta attribution + sweep │  Understand specific tradeoffs
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │  5. Budget + Margins          │  "Will it fit? How sure am I?"
                    │     Monte Carlo + allocation  │  Risk management, go/no-go
                    └─────────────────────────────┘
```

Each methodology builds on the previous: **Score → Understand → Explore → Compare → Commit**.
A user might use just one, or flow through all five as a design matures from concept to
tape-out.

**Mapping to existing `branes swap` commands:**

| Methodology | Existing Command(s) | New Capability Needed |
|------------|--------------------|-----------------------|
| 1. Scorecard + FoM | `swap check` | Mission profiles, composite score, radar chart |
| 2. Sensitivity | (GP lengthscales in BO layer) | Tornado diagrams, Taguchi DOE, RSM |
| 3. Pareto + TOPSIS | `swap explore`, `swap show-front` | Clustering, TOPSIS ranking, MRS |
| 4. Delta Attribution | `swap compare`, `swap explain` | BOM waterfall, parametric sweep |
| 5. Budget + MC | `swap check` (deterministic) | Distribution inputs, Monte Carlo, margin tracking |

---

## 12. References

### Energy-Delay Product and Circuit Metrics

1. R. Gonzalez and M. Horowitz, "Energy Dissipation in General Purpose Microprocessors,"
   IEEE JSSC, 1996.
   [Stanford Handout](https://web.stanford.edu/class/archive/ee/ee371/ee371.1066/handouts/gonzalez_97.pdf)
2. "Energy Delay Product," ScienceDirect Topics.
   [Link](https://www.sciencedirect.com/topics/computer-science/energy-delay-product)
3. "Power-Delay Product," Wikipedia.
   [Link](https://en.wikipedia.org/wiki/Power%E2%80%93delay_product)

### PPAC and Process Node Evaluation

4. "TSMC's N2 and the Power of PPACt," TSPA Semiconductor.
   [Link](https://tspasemiconductor.substack.com/p/tsmcs-n2-and-the-power-of-ppact-driving)
5. "Imec Demonstrates PPAC Benefit of Heterogeneous Sequential 3D Integration," Imec.
   [Link](https://www.imec-int.com/en/articles/imec-demonstrates-power-performance-area-cost-benefit-of-heterogeneous-sequential-3d-integration-for-advanced-cmos-nodes)
6. "PPA (Power, Performance, and Area)," Semiconductor Engineering.
   [Link](https://semiengineering.com/knowledge_centers/eda-design/definitions/ppa/)

### SWaP-C / SWaP-C2 Standards and Definitions

7. "SWaP-C2 Fact Sheet," JIFCO/DoD.
   [Link](https://jifco.defense.gov/Press-Room/Fact-Sheets/Article-View-Fact-sheets/Article/1488195/size-weight-power-cost-and-cooling-swapc2/)
8. "SWaP-C and SWaP-C2 Principles," Sealevel Systems.
   [Link](https://www.sealevel.com/swap-swapc2)

### Performance Efficiency Metrics

9. "Why TOPS Are Not Enough," Hailo.
   [Link](https://hailo.ai/blog/evaluating-edge-ai-accelerator-performance-why-tops-are-not-enough/)
10. "Intelligence Per Watt," Stanford Hazy Research, November 2025.
    [Link](https://hazyresearch.stanford.edu/blog/2025-11-11-ipw)
11. "Intelligence Per Watt," Stanford Scaling Intelligence Lab.
    [Link](https://scalingintelligence.stanford.edu/pubs/ipw/)
12. "MLPerf Power Measurement," arXiv:2410.12032.
    [Link](https://arxiv.org/html/2410.12032v1)
13. "Quadric Chimera GPNPU TOPS/mm²," SemiWiki Forum.
    [Link](https://semiwiki.com/forum/threads/quadric-chimera-gpnpu.20672/)
14. "Versal AI Edge Series," AMD.
    [Link](https://www.amd.com/en/products/adaptive-socs-and-fpgas/versal/ai-edge-series.html)
15. S. Williams, A. Waterman, D. Patterson, "Roofline: An Insightful Visual Performance
    Model," Communications of the ACM, 2009.
    [Link](https://people.eecs.berkeley.edu/~kubitron/cs252/handouts/papers/RooflineVyNoYellow.pdf)
16. "Roofline Performance Model," Wikipedia.
    [Link](https://en.wikipedia.org/wiki/Roofline_model)

### Systems Engineering and Budget Management

17. NASA Systems Engineering Handbook, SP-2016-6105 Rev2.
    [Link](https://www.nasa.gov/wp-content/uploads/2018/09/nasa_systems_engineering_handbook_0.pdf)
18. "Optimizing SWaP-C in Defense & Aerospace 2025," Galorath.
    [Link](https://galorath.com/blog/optimizing-swap-c-defense-aerospace-2025/)

### Trade Study and MCDA Methods

19. "The Trade Study Process," MITRE.
    [Link](https://www.mitre.org/sites/default/files/2021-11/prs-21-0522-the-trade-study-process.pdf)
20. "Analytic Hierarchy Process," Wikipedia.
    [Link](https://en.wikipedia.org/wiki/Analytic_hierarchy_process)
21. G. Parnell and T. Trainor, "Using the Swing Weight Matrix to Weight Multiple
    Objectives," INCOSE International Symposium, 2009.
    [Link](https://incose.onlinelibrary.wiley.com/doi/abs/10.1002/j.2334-5837.2009.tb00949.x)

### Evolutionary Multi-Objective Optimization

22. K. Deb et al., "A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II,"
    IEEE Transactions on Evolutionary Computation, 2002.
    [Link](https://www.cse.unr.edu/~sushil/class/gas/papers/nsga2.pdf)
23. "NSGA-III in pymoo." [Link](https://pymoo.org/algorithms/moo/nsga3.html)

### Bayesian and Surrogate-Assisted Optimization

24. L. Nardi et al., "HyperMapper: Practical Design Space Exploration," Stanford DAWN.
    [Link](https://dawn.cs.stanford.edu/publications/hypermapper/practical-design-space-exploration)
25. B. Reagen et al., "A Case for Efficient Accelerator Design Space Exploration via
    Bayesian Optimization," IEEE ISLPED, 2017.
    [Link](https://ieeexplore.ieee.org/document/8009208/)

### AI-Driven EDA

26. "DSO.ai," Synopsys.
    [Link](https://www.synopsys.com/ai/ai-powered-eda/dso-ai.html)
27. "AI in Chip Design," Cadence.
    [Link](https://www.cadence.com/en_US/home/explore/ai-chip-design.html)
28. A. Mirhoseini et al., "Chip Placement with Deep Reinforcement Learning,"
    arXiv:2004.10746. [Link](https://arxiv.org/abs/2004.10746)
29. "Deep Learning for Chip Design," UCLA VAST Lab.
    [Link](https://vast.cs.ucla.edu/projects/deep-learning-chip-design)

### Estimation Tools and DoD Programs

30. "SEER Suite," Galorath. [Link](https://galorath.com/seer/)
31. "Galorath SWaP-C Optimization with SEER," ExecutiveBiz.
    [Link](https://executivebiz.com/2025/01/galorath-matt-mcdonald-swap-c-optimization-seer/)
32. "NSGA-II in pymoo." [Link](https://pymoo.org/algorithms/moo/nsga2.html)
33. K. Deb et al., "NSGA-II" (duplicate of [22]).

### Quality-Diversity and MAP-Elites

34. "BoTorch: Bayesian Optimization in PyTorch." [Link](https://botorch.org/)
35. "Multi-Objective MAP-Elites (MOME)," arXiv:2202.03057.
    [Link](https://arxiv.org/abs/2202.03057)
36. "Quality-Diversity Optimization Papers Collection."
    [Link](https://quality-diversity.github.io/papers.html)
37. "Multi-Objective Optimization," Wikipedia.
    [Link](https://en.wikipedia.org/wiki/Multi-objective_optimization)

### Decision-Making Frameworks

38. "Decision Matrix (Pugh Matrix)," ASQ.
    [Link](https://asq.org/quality-resources/decision-matrix)
39. "TOPSIS," Wikipedia. [Link](https://en.wikipedia.org/wiki/TOPSIS)
40. "Quality Function Deployment," Wikipedia.
    [Link](https://en.wikipedia.org/wiki/Quality_function_deployment)
41. J. Hauser and D. Clausing, "The House of Quality," Harvard Business Review, 1988.
    [Link](https://hbr.org/1988/05/the-house-of-quality)
42. "VAVE Explained for OEMs," ESCATEC.
    [Link](https://www.escatec.com/blog/value-analysis-and-value-engineering-va/ve-explained-for-oems)
43. "VAVE Cost Reduction," Titoma.
    [Link](https://titoma.com/blog/vave-cost-reduction/)
44. "SD-24 Value Engineering Guidebook," DoD CTO, February 2025.
    [Link](https://www.cto.mil/wp-content/uploads/2025/02/SD-24-VE-Guidebook-25Feb2025-Cleared-1.pdf)

### Sensitivity and What-If Analysis

45. "Tornado Diagram," Wikipedia.
    [Link](https://en.wikipedia.org/wiki/Tornado_diagram)
46. "Mastering Uncertainty Analysis in Systems Engineering," NumberAnalytics.
    [Link](https://www.numberanalytics.com/blog/mastering-uncertainty-analysis-systems-engineering)
47. "Uncertainty in NoC Mapping," ScienceDirect.
    [Link](https://www.sciencedirect.com/science/article/abs/pii/S0141933120306554)
48. "Taguchi Methods," Wikipedia.
    [Link](https://en.wikipedia.org/wiki/Taguchi_methods)
49. "Taguchi Approach to Design Optimization," NASA.
    [Link](https://ntrs.nasa.gov/api/citations/20040121019/downloads/20040121019.pdf)
50. "Response Surface Methodology," Wikipedia.
    [Link](https://en.wikipedia.org/wiki/Response_surface_methodology)

### Constraint Handling and Feasibility

51. "NanoStar Systems Engineering," GitLab.
    [Link](https://nanostar-project.gitlab.io/main/source/preliminary-design/systems.html)
52. "Margins," NASA Psyche Mission Blog.
    [Link](https://medium.com/the-nasa-psyche-mission-journey-to-a-metal-world/margins-e19f3dce28b6)
53. C. A. Coello, "Theoretical and Numerical Constraint-Handling Techniques."
    [Link](https://www2.cs.uh.edu/~ceick/6367/Coello_CHNOPT.pdf)
54. "SWaP-Constrained Embedded Computing for AI," Medium.
    [Link](https://alexmoltzau.medium.com/what-is-swap-constrained-embedded-computing-for-artificial-intelligence-f846cbccdf1e)
55. "SWaP Embedded Computing AI," Military Aerospace Electronics.
    [Link](https://www.militaryaerospace.com/computers/article/14174721/swap-embedded-computing-artificial-intelligence)

### AI/ML for Design Space Exploration

56. "Hardware-Aware NAS for Edge," arXiv:2512.04705.
    [Link](https://arxiv.org/abs/2512.04705)
57. "NAS for Resource-Constrained Devices," IET.
    [Link](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/cps2.12058)
58. "MARL for Microprocessor DSE," arXiv:2211.16385.
    [Link](https://arxiv.org/abs/2211.16385)
59. "Sherlock: Adaptive Surrogate Selection for FPGA HLS," ACM TODAES.
    [Link](https://dl.acm.org/doi/10.1145/3511472)

### Frameworks and Roadmaps

60. "OpenMDAO," NASA Glenn. [Link](https://openmdao.org/)
61. "IRDS 2022 Executive Summary," IEEE.
    [Link](https://irds.ieee.org/images/files/pdf/2022/2022IRDS_ES.pdf)

### Additional Sources

62. "SWaP-C Fundamentals," Amphenol Aerospace.
    [Link](https://www.amphenol-aerospace.com/blog/swap-c-fundamentals-are-the-key-to-future-defense-architectures)
63. "SWaP-C Technologies," Curtiss-Wright Defense Solutions.
    [Link](https://defense-solutions.curtisswright.com/capabilities/technologies/swap-c)
64. "Performance Per Watt," Wikipedia.
    [Link](https://en.wikipedia.org/wiki/Performance_per_watt)
65. "Performance Per Watt Is the New Moore's Law," Arm Newsroom.
    [Link](https://newsroom.arm.com/blog/performance-per-watt)
66. "Green500," Wikipedia. [Link](https://en.wikipedia.org/wiki/Green500)
67. "95.6 TOPS/W Deep Learning Inference Accelerator," NVIDIA Research.
    [Link](https://research.nvidia.com/publication/2023-01_956-topsw-deep-learning-inference-accelerator-vector-scaled-4-bit-quantization)
68. "TOPS in Computing," Lenovo.
    [Link](https://www.lenovo.com/us/en/glossary/tops-in-computing/)
69. "Why Tokens Per Watt Is Crucial," Schneider Electric.
    [Link](https://blog.se.com/datacenter/2025/12/04/why-tokens-per-watt-is-crucial-for-measuring-ai-efficiency/)
70. "Figure of Merit Scoring," Altium Resources.
    [Link](https://resources.altium.com/p/figure-merit-way-score-your-opinions)
71. "Figure of Merit: A Refresher Course," Electronic Design.
    [Link](https://www.electronicdesign.com/technologies/power/article/21749890/figure-of-merit-a-refresher-course)
72. "DARPA Electronics Resurgence Initiative," DARPA News, 2017.
    [Link](https://www.darpa.mil/news/2017/electronics-resurgence-initiative)
73. "DARPA CHIPS Program." [Link](https://www.darpa.mil/program/common-heterogeneous-integration-and-ip-reuse-strategies)
74. "DARPA DAHI Program." [Link](https://www.darpa.mil/research/programs/dahi-electronic-photonic-heterogenous-integration)
75. "DARPA Awards $840M for 3D Chiplet Research," The Register, 2024.
    [Link](https://www.theregister.com/2024/07/18/darpa_awards_840m_to_utaustin/)
76. "Pareto Optimal Benchmarking for ARM Cortex," arXiv:2602.17508.
    [Link](https://arxiv.org/html/2602.17508v1)
77. "AHP Consistency Ratio," SpiceLogic.
    [Link](https://spicelogic.com/docs/ahpsoftware/intro/ahp-consistency-ratio-transitivity-rule-388)
78. "TOPSIS Step-by-Step," GeeksforGeeks.
    [Link](https://www.geeksforgeeks.org/data-science/topsis-method-for-multiple-criteria-decision-making-mcdm/)
79. "Constraint Handling Review," Springer, 2022.
    [Link](https://link.springer.com/article/10.1007/s11831-022-09859-9)
80. "MIL-HDBK-338," GlobalSpec.
    [Link](https://standards.globalspec.com/std/1017119/MIL-HDBK-338)
81. "DoD Source Selection Procedures," 2022.
    [Link](https://www.acq.osd.mil/dpap/policy/policyvault/USA000740-22-DPC.pdf)
82. "Low-SWaP in Military Communications," REDCOM.
    [Link](https://www.redcom.com/what-is-low-swap-size-weight-and-power/)
83. "SWaP-C2 Applications," Sealevel Systems.
    [Link](https://www.sealevel.com/swap-c2-applications)
84. "MARCO: Multi-Agent RL for NAS," arXiv:2506.13755.
    [Link](https://arxiv.org/abs/2506.13755)
85. "Surrogate-Based Optimization for System Architectures," arXiv:2504.08721.
    [Link](https://arxiv.org/html/2504.08721v1)
86. "Enhanced Monte Carlo for Electronic Circuits," Springer, 2025.
    [Link](https://link.springer.com/article/10.1007/s10836-025-06202-5)
87. "HyperMapper," GitHub. [Link](https://github.com/luinardi/hypermapper)
88. "Comprehensive Survey on NSGA-II," Springer, 2023.
    [Link](https://link.springer.com/article/10.1007/s10462-023-10526-z)
89. "OpenMDAO Paper," Structural and Multidisciplinary Optimization, 2019.
    [Link](https://link.springer.com/article/10.1007/s00158-019-02211-z)
90. "SWaP-C Definition," NSTXL. [Link](https://nstxl.org/what-is-swap-c/)

---

*Document generated: March 2026*
*Complements: [swap-metrics.md](./swap-metrics.md) — Industry landscape and product design flows*
