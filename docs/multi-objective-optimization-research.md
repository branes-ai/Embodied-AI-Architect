# Multi-Objective Optimization for Design Space Exploration

**Date:** 2026-02-24
**Purpose:** Research survey and methodology assessment for strengthening the optimization pipeline in the Embodied AI Architect.

## Motivation

When a designer brings a system description and requirements to the Embodied AI Architect, the design *exploration* is almost more valuable than any single final design point. The exploration reveals:

- **What is feasible** — which combinations of objectives are achievable given the constraints
- **What trades off against what** — the shape of the Pareto surface and the marginal rates of substitution between objectives
- **Where the boundaries are** — which constraints are binding, which have slack, and how sensitive the solution is to requirement changes
- **Which design knobs matter** — sensitivity of objectives to each design parameter

This is inherently a multi-objective optimization (MOO) problem across performance (throughput, latency), power, cost, reliability, accuracy, and robustness. Different requirement profiles call for different exploration strategies — a cost-constrained drone needs different insight than a safety-critical surgical robot.

## Current State: What We Have

The existing optimizer (`optimizer.py`) uses a **greedy constraint-fixer**: it picks the strategy with the highest reduction factor for whichever constraint is failing, tries it, and marks it as used. The Pareto code (`pareto.py`) performs O(n^2) non-dominated sorting over a fixed hardware catalog with 3 hardcoded objectives (power, latency, cost) and identifies a knee point via Euclidean distance to the utopia point. Hardware scoring (`specialists.py`) uses an additive heuristic on a 0-100 scale.

**Gaps relative to proper MOO:**

| Aspect | Current | Gap |
|--------|---------|-----|
| Algorithm | Greedy heuristic | No evolutionary or Bayesian search |
| Objectives | 3 fixed (power, latency, cost) | Should support arbitrary objectives + weights |
| Design space | Discrete hardware catalog | Missing parametric/continuous variables |
| Pareto quality | Basic dominance sorting | No hypervolume, IGD, or coverage metrics |
| Sensitivity | None | No analysis of how objectives respond to design changes |
| Constraints | Binary PASS/FAIL | No soft constraints, margins, or feasibility gradients |
| Learning | None | No surrogates, no transfer from prior campaigns |

---

## Part 1: Classical Evolutionary Multi-Objective Optimization

### NSGA-II — Non-dominated Sorting Genetic Algorithm II

The de facto baseline for MOO. Maintains a population ranked by non-domination level, with crowding distance as a secondary diversity criterion.

**Strengths:** Well-understood, abundant implementations, handles mixed variables with appropriate operators, crowding distance explicitly rewards coverage of sparse front regions.

**Weaknesses:** Crowding distance breaks down for >3 objectives (nearly every solution becomes non-dominated). Requires 10,000-100,000+ function evaluations. No surrogate model.

**DSE assessment:** Excellent for 2-3 objective front discovery when evaluations are cheap. Not sample-efficient enough for expensive evaluations.

> **Reference:** K. Deb, A. Pratap, S. Agarwal, and T. Meyarivan, "A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II," *IEEE Transactions on Evolutionary Computation*, vol. 6, no. 2, pp. 182-197, Apr. 2002. DOI: [10.1109/4235.996017](https://doi.org/10.1109/4235.996017)

### NSGA-III — Reference-Point-Based Many-Objective NSGA

Replaces crowding distance with reference-point-based selection. Uniformly distributed reference points on the objective hyperplane guide diversity. Solutions are associated with the nearest reference point, and selection prefers under-represented directions.

**Strengths:** Designed for many-objective problems (4-15 objectives). Produces well-distributed fronts even in high-dimensional objective spaces. Reference point structure provides interpretability.

**Weaknesses:** Same sample efficiency problem as NSGA-II. For 6 objectives with resolution parameter 4, you need C(9,4) = 126 reference points and a population of at least 128.

**DSE assessment:** Best classical method for our 6-objective case. Use as ground truth when evaluations are cheap (analytical models).

> **Reference:** K. Deb and H. Jain, "An Evolutionary Many-Objective Optimization Algorithm Using Reference-Point-Based Nondominated Sorting Approach, Part I: Solving Problems With Box Constraints," *IEEE Transactions on Evolutionary Computation*, vol. 18, no. 4, pp. 577-601, Aug. 2014. DOI: [10.1109/TEVC.2013.2281535](https://doi.org/10.1109/TEVC.2013.2281535)

### MOEA/D — Multi-Objective Evolutionary Algorithm Based on Decomposition

Decomposes the multi-objective problem into N scalar subproblems using weight vectors (Tchebycheff, weighted sum, or PBI). Neighboring subproblems share solutions. Each subproblem is optimized simultaneously.

**Strengths:** Computationally efficient (scalar subproblems). Scales gracefully to many objectives. Weight vectors can be adapted online (MOEA/D-AWA) to concentrate search in regions of interest. Produces uniformly distributed fronts.

**Weaknesses:** Quality depends on decomposition scheme choice. Neighborhood size is sensitive.

**DSE assessment:** Strong alternative to NSGA-III for many-objective problems. Particularly useful when you want to weight certain objectives — bias the weight vector distribution toward the objectives that matter most.

> **Reference:** Q. Zhang and H. Li, "MOEA/D: A Multiobjective Evolutionary Algorithm Based on Decomposition," *IEEE Transactions on Evolutionary Computation*, vol. 11, no. 6, pp. 712-731, Dec. 2007. DOI: [10.1109/TEVC.2007.892759](https://doi.org/10.1109/TEVC.2007.892759)

### SPEA2 — Strength Pareto Evolutionary Algorithm 2

Assigns fitness based on the strengths of dominating individuals plus k-th nearest neighbor density. Maintains an external archive.

**Strengths:** Fine-grained fitness discrimination. Archive preserves non-dominated solutions.

**Weaknesses:** More expensive per generation than NSGA-II. Not designed for >3 objectives. Largely superseded by NSGA-III.

> **Reference:** E. Zitzler, M. Laumanns, and L. Thiele, "SPEA2: Improving the Strength Pareto Evolutionary Algorithm," *TIK-Report 103*, ETH Zurich, 2001. DOI: [10.3929/ethz-a-004284029](https://doi.org/10.3929/ethz-a-004284029)

### SMS-EMOA — S-Metric Selection Evolutionary Multi-Objective Algorithm

Uses hypervolume contribution as the selection criterion. The individual with the smallest hypervolume contribution in the worst front is removed each generation.

**Strengths:** Hypervolume is the only known unary Pareto-compliant quality indicator — maximizing it provably converges to the true front. Produces very high-quality fronts in 2-3 objectives.

**Weaknesses:** Hypervolume computation is O(n^(m/2)) for m objectives — exponential beyond 4-6 objectives. Very slow for large populations.

**DSE assessment:** The hypervolume concept is critically important (it appears in Bayesian methods below), but SMS-EMOA itself is too expensive for many-objective problems.

> **Reference:** N. Beume, B. Naujoks, and M. Emmerich, "SMS-EMOA: Multiobjective selection based on dominated hypervolume," *European Journal of Operational Research*, vol. 181, no. 3, pp. 1653-1669, 2007. DOI: [10.1016/j.ejor.2006.08.008](https://doi.org/10.1016/j.ejor.2006.08.008)

### Summary: Classical Methods

| Method | Best for # objectives | Evaluations needed | Mixed vars | Front quality |
|--------|----------------------|-------------------|------------|---------------|
| NSGA-II | 2-3 | 10K-100K | Yes | Good |
| NSGA-III | 4-15 | 10K-100K | Yes | Good |
| MOEA/D | 3-15 | 10K-100K | Yes | Very good |
| SPEA2 | 2-3 | 10K-100K | Yes | Good |
| SMS-EMOA | 2-4 | 10K-100K | Yes | Excellent |

---

## Part 2: Bayesian Multi-Objective Optimization

These methods build surrogate models (typically Gaussian processes) of each objective and use acquisition functions to select the most informative next evaluation point. Designed for expensive black-box optimization where each function evaluation is costly.

### qEHVI / qNEHVI — (Noisy) Expected Hypervolume Improvement

Fits independent GP models to each objective. The acquisition function computes the expected increase in hypervolume of the current Pareto front if a candidate point were evaluated. qEHVI handles parallel (q-batch) evaluation; qNEHVI handles noisy observations.

**Strengths:**
- State-of-the-art sample efficiency: convergence in 50-200 evaluations (100x fewer than evolutionary methods)
- Hypervolume improvement is Pareto-compliant — provably converges to the true front
- Handles noisy observations (critical for hardware simulation)
- Parallel batch proposals for concurrent evaluation
- Well-maintained implementation in BoTorch/Ax

**Weaknesses:**
- GP scales O(n^3) in observations (sparse approximations exist)
- Hypervolume computation exponential in #objectives — practical limit ~4-6 objectives
- Assumes continuous design space (mixed variables require special GP kernels)
- GP may struggle with highly non-smooth landscapes

**DSE assessment:** Primary recommendation when evaluations are expensive. 50-200 evaluations is compatible with our hardware assessment pipeline budget. The 4-6 objective limit is manageable by combining related objectives (e.g., reliability + robustness → "dependability").

> **Reference (qEHVI):** S. Daulton, M. Balandat, and E. Bakshy, "Differentiable Expected Hypervolume Improvement for Parallel Multi-Objective Bayesian Optimization," *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 33, pp. 9851-9864, 2020. arXiv: [2006.05078](https://arxiv.org/abs/2006.05078)
>
> **Reference (qNEHVI):** S. Daulton, M. Balandat, and E. Bakshy, "Parallel Bayesian Optimization of Multiple Noisy Objectives with Expected Hypervolume Improvement," *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 34, pp. 2187-2200, 2021. arXiv: [2105.08195](https://arxiv.org/abs/2105.08195)

### ParEGO — Efficient Global Optimization for Multi-Objective Problems

Scalarizes using random weight vectors (augmented Tchebycheff). Each iteration: sample a random weight, scalarize objectives, fit a GP to the scalar, maximize Expected Improvement.

**Strengths:** Simple to implement (single-objective BO with changing weights). Scales to many objectives (no hypervolume). Good sample efficiency (50-200 evaluations).

**Weaknesses:** Each iteration optimizes one direction — does not efficiently fill the entire front. Random weight sampling can be inefficient. Less uniform fronts than EHVI.

**DSE assessment:** Good fallback for >6 objectives where EHVI becomes intractable. Less suitable for systematic DSE because it doesn't explore the full front.

> **Reference:** J. Knowles, "ParEGO: A Hybrid Algorithm with On-Line Landscape Approximation for Expensive Multiobjective Optimization Problems," *IEEE Transactions on Evolutionary Computation*, vol. 10, no. 1, pp. 50-66, Feb. 2006. DOI: [10.1109/TEVC.2005.851274](https://doi.org/10.1109/TEVC.2005.851274)

### BoTorch — Bayesian Optimization in PyTorch

The reference implementation framework for modern Bayesian optimization, including all EHVI variants, multi-fidelity methods, and constrained optimization.

> **Reference:** M. Balandat, B. Karrer, D. R. Jiang, S. Daulton, B. Letham, A. G. Wilson, and E. Bakshy, "BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization," *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 33, pp. 21524-21538, 2020. arXiv: [1910.06403](https://arxiv.org/abs/1910.06403)

### Non-GP Surrogates for Mixed Design Spaces

Recent work has explored replacing GPs with:
- **Random forests** (SMAC3): Handle mixed discrete/continuous naturally. Lower sample efficiency than GPs but much more scalable.
- **Deep kernel learning**: Neural network feature extraction + GP layer. Better for high-dimensional inputs.

Random forest surrogates are particularly attractive for SoC design because the space is inherently mixed (process node is discrete, clock frequency is continuous, IP block count is integer).

> **Reference (SMAC3):** M. Lindauer, K. Eggensperger, M. Feurer, A. Biedenkapp, D. Deng, C. Benjamins, T. Ruhkopf, R. Sass, and F. Hutter, "SMAC3: A Versatile Bayesian Optimization Package for Hyperparameter Optimization," *Journal of Machine Learning Research*, vol. 23, no. 54, pp. 1-9, 2022.

### Summary: Bayesian Methods

| Method | Evaluations | # objectives | Mixed vars | Front quality |
|--------|------------|-------------|------------|---------------|
| qEHVI/qNEHVI | 50-200 | 2-6 | With extensions | Excellent |
| ParEGO | 50-200 | 2-15+ | With scalarization | Good |
| RF-based BO | 100-500 | 2-10 | Native | Good |

---

## Part 3: Information-Theoretic Approaches

These methods are most aligned with the goal of "generating the most information about the design space" rather than just finding optimal points.

### MESMO — Max-value Entropy Search for Multi-Objective Optimization

Measures the mutual information between a candidate evaluation and the optimal Pareto values. Maximizes information gain about where the Pareto front is in objective space.

**Strengths:** Directly optimizes information gain. Computationally cheaper than PESMO. Naturally encourages exploration — even points far from the current front are valuable if they reduce uncertainty about the front's location.

**Weaknesses:** Requires approximate sampling from the posterior Pareto front. Less mature implementations than EHVI.

**DSE assessment:** Highly aligned with DSE philosophy. The information-gain objective means it explores uncertain regions rather than greedily improving known good solutions — exactly what a designer wants in early-stage exploration.

> **Reference:** S. Belakaria, A. Deshwal, and J. R. Doppa, "Max-value Entropy Search for Multi-Objective Bayesian Optimization," *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 32, 2019. arXiv: [1905.08879](https://arxiv.org/abs/1905.08879)

### PFES — Pareto Front Entropy Search

Directly minimizes the entropy (uncertainty) of the Pareto front distribution. Uses Thompson sampling to generate front samples from the posterior, then selects the candidate that maximally reduces front entropy.

**Strengths:** The most principled information-theoretic approach — directly targets uncertainty reduction on the Pareto front. Handles input-space and output-space exploration simultaneously.

**Weaknesses:** Thompson sampling from the GP posterior requires careful implementation. Entropy estimation is noisy and expensive. Limited to ~4-6 objectives.

**DSE assessment:** Theoretical gold standard for design space exploration. Practical challenge is implementation quality.

> **Reference:** S. Suzuki, S. Takeno, T. Tamura, K. Shitara, and M. Karasuyama, "Multi-Objective Bayesian Optimization using Pareto-Frontier Entropy," *Proceedings of the 37th International Conference on Machine Learning (ICML)*, pp. 9279-9288, 2020. arXiv: [2004.01566](https://arxiv.org/abs/2004.01566) *(verify arXiv ID)*

### JES — Joint Entropy Search

Extends entropy search to jointly reduce uncertainty about both the Pareto set (which design parameters) and Pareto front (which objective values). Reveals the mapping from design parameters to objective tradeoffs.

**Strengths:** Answers "which design knobs matter and how do they trade off?" — the core DSE deliverable.

**Weaknesses:** Computationally expensive. Very new, limited empirical validation.

> **Reference:** B. Tu, A. Gandy, N. Kantas, and B. Sheridan, "Joint Entropy Search for Multi-Objective Bayesian Optimization," *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 35, 2022. arXiv: [2210.02905](https://arxiv.org/abs/2210.02905)

### PESMO — Predictive Entropy Search for Multi-Objective Optimization

Maximizes mutual information between the next observation and the Pareto front location using expectation propagation.

> **Reference:** D. Hernandez-Lobato, J. M. Hernandez-Lobato, A. Shah, and R. P. Adams, "Predictive Entropy Search for Multi-objective Bayesian Optimization," *Proceedings of the 33rd International Conference on Machine Learning (ICML)*, pp. 1492-1501, 2016. arXiv: [1511.05467](https://arxiv.org/abs/1511.05467)

### Practical Approximation: Exploration-Exploitation Schedule

A practical middle ground captures 80% of the information-theoretic benefit without implementation complexity:

```python
alpha = max(0.0, 1.0 - 2.0 * (iteration / budget))  # linear decay
acquisition(x) = alpha * GP_variance(x) + (1 - alpha) * EHVI(x)
```

Early iterations (alpha ≈ 1) maximize GP posterior variance (pure exploration of uncertain regions). Later iterations (alpha → 0) maximize EHVI (pure Pareto front refinement). Implementable in ~20 lines of BoTorch code.

---

## Part 4: Quality-Diversity Optimization

### MAP-Elites — Illuminating Search Spaces by Mapping Elites

Not traditional MOO — MAP-Elites *illuminates* the design space by filling a grid of objective-space cells with the best-performing solution found for each cell.

**How it works:** Discretize a "behavior space" (e.g., power bucket × latency bucket) into a grid. For each cell, maintain the best solution found. Explore via random perturbation + selection. The goal is to fill as many cells as possible with high-quality solutions.

**Why it matters for DSE:** MAP-Elites produces an "atlas" of the design space. While MOO finds the Pareto front (the boundary of what's achievable), MAP-Elites fills in the interior, answering:
- "Show me all designs under 5W" → filter by power column
- "What's the cost difference between 10ms and 20ms latency?" → compare rows
- "Which regions are empty/infeasible?" → empty cells

**Strengths:** Directly produces the design space understanding that a human designer needs. Fast when evaluations are cheap (5,000-10,000 evaluations on analytical models = seconds). Reveals feasibility boundaries and objective correlations.

**Weaknesses:** Not sample-efficient (needs many evaluations). Grid discretization can miss structure between cells. Doesn't compute a precise Pareto front.

> **Reference (original):** J.-B. Mouret and J. Clune, "Illuminating search spaces by mapping elites," arXiv preprint, 2015. arXiv: [1504.04909](https://arxiv.org/abs/1504.04909)
>
> **Reference (CMA-ME):** M. Fontaine, J. Togelius, S. Nikolaidis, and A. Hoover, "Covariance Matrix Adaptation for the Rapid Illumination of Behavior Space," *Proceedings of the Genetic and Evolutionary Computation Conference (GECCO)*, 2020. arXiv: [1912.02400](https://arxiv.org/abs/1912.02400)
>
> **Reference (Differentiable QD):** M. Fontaine and S. Nikolaidis, "Differentiable Quality Diversity," *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 36, 2023. arXiv: [2106.03894](https://arxiv.org/abs/2106.03894)

### MAP-Elites for HLS Design Space Exploration (DAC 2024)

This is a key reference that directly applies quality-diversity optimization to hardware design. Rather than finding only Pareto-optimal HLS configurations, it builds a complete map of the design space showing which pragma configurations are feasible and why.

**Key insight:** In HLS DSE, understanding *why* certain regions of the design space are infeasible is as valuable as finding the optimal configurations. MAP-Elites reveals the structure of feasibility boundaries.

> **Reference:** B. Liu and B. C. Schafer, "Efficient and Reliable High-Level Synthesis Design Space Exploration via Quality-Diversity Optimization," *Proceedings of the 61st ACM/IEEE Design Automation Conference (DAC)*, 2024.

---

## Part 5: Chip Design and Hardware DSE — What Has Worked

### Surrogate-Assisted Optimization (SAO) Pattern

The dominant pattern in electronic design automation:

1. Build a cheap surrogate of the expensive evaluation (synthesis, simulation, place-and-route)
2. Run MOO algorithm (NSGA-II, MOEA/D) on the surrogate to generate candidates
3. Evaluate best candidates on the real objective (EDA tool run)
4. Update surrogate with new data
5. Repeat

> **Reference (ReSPIR):** C. Palermo, V. Catania, and D. Patti, "ReSPIR: A Response Surface-based Pareto Iterative Refinement for application-specific design space exploration," *IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems (TCAD)*, vol. 28, no. 12, pp. 1816-1829, 2009. DOI: [10.1109/TCAD.2009.2028681](https://doi.org/10.1109/TCAD.2009.2028681)

### Bayesian Optimization for Analog Circuit Sizing

GP-based multi-objective BO for transistor sizing. Achieved comparable quality to commercial tools with 10-50x fewer simulations.

> **Reference:** W. Lyu, P. Xue, F. Yang, C. Yan, Z. Hong, X. Zeng, and D. Zhou, "An Efficient Bayesian Optimization Approach for Automated Optimization of Analog Circuits," *IEEE Transactions on Circuits and Systems I*, vol. 65, no. 6, pp. 1954-1967, 2018. DOI: [10.1109/TCSI.2017.2768826](https://doi.org/10.1109/TCSI.2017.2768826)

### COMBA — Multi-Objective BO for HLS

Demonstrated 5x fewer synthesis runs than random search to achieve the same Pareto front quality in HLS design space exploration.

> **Reference:** B. C. Schafer and Z. Wang, "High-Level Synthesis Design Space Exploration: Past, Present, and Future," *IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems (TCAD)*, vol. 39, no. 10, pp. 2628-2639, 2020. DOI: [10.1109/TCAD.2019.2943570](https://doi.org/10.1109/TCAD.2019.2943570)

### Chip Placement with Reinforcement Learning

Treated chip placement as an RL problem with multiple reward components (timing, area, congestion). Subsequent work explored multi-objective RL formulations.

> **Reference:** A. Mirhoseini, A. Goldie, M. Yazgan, et al., "A graph placement methodology for fast chip design," *Nature*, vol. 594, pp. 207-212, 2021. DOI: [10.1038/s41586-021-03544-w](https://doi.org/10.1038/s41586-021-03544-w)

### Industry Usage

| Company | Tool | Approach | Notes |
|---------|------|----------|-------|
| Synopsys | DSO.ai | RL-based search | Single-objective RL with constraint handling. Deployed at Samsung, Qualcomm. |
| Cadence | Cerebrus | ML-guided optimization | Primarily single-objective P&R optimization. |
| Arm | Internal | MOEA/D | Published work on configurable processor DSE (2019-2020). |
| Intel | Internal | GP-based BO | Microprocessor design parameters (cache, pipeline, clock). |

### Neural Architecture Search (NAS) for Edge Deployment

NAS for edge devices is a directly analogous problem: searching over architectures to optimize accuracy, latency, power, and size simultaneously.

> **Reference (MnasNet):** M. Tan, B. Chen, R. Pang, V. Vasudevan, M. Sandler, A. Howard, and Q. V. Le, "MnasNet: Platform-Aware Neural Architecture Search for Mobile," *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 2820-2828, 2019. DOI: [10.1109/CVPR.2019.00293](https://doi.org/10.1109/CVPR.2019.00293)
>
> **Reference (OFA):** H. Cai, C. Gan, T. Wang, Z. Zhang, and S. Han, "Once-for-All: Train One Network and Specialize it for Efficient Deployment," *International Conference on Learning Representations (ICLR)*, 2020. arXiv: [1908.09791](https://arxiv.org/abs/1908.09791)

---

## Part 6: Sensitivity Analysis and Tradeoff Characterization

### Marginal Rate of Substitution (MRS)

On a Pareto front, the MRS between objectives i and j:

```
MRS_{i,j}(x) = -df_i / df_j  along the Pareto front
```

This tells the designer: "how much power do I save per millisecond of latency I give up?" Directly interpretable and maps to the Lagrange multipliers (lambda values) in the existing `cost-functions-tradeoffs.md` thermodynamic-informational Lagrangian.

**Computation:** Numerically approximate from neighboring Pareto points, or analytically from GP posterior gradients in Bayesian methods.

### Knee Point Detection

A knee point is where a small sacrifice in one objective yields a large gain in another. The current `identify_knee_point` in `pareto.py` uses normalized Euclidean distance to the utopia point. More sophisticated methods:

- **Angle-based** (Branke et al., 2004): Knee is where the angle between adjacent front segments is maximized. More robust for non-convex fronts.
- **Marginal utility-based** (Das, 1999): Knee is where MRS changes most rapidly (inflection point).
- **Reflex angle** (Deb & Gupta, 2011): Works in many-objective spaces via deviation from hyperplane through neighbors.

> **Reference:** J. Branke, K. Deb, H. Dierolf, and M. Osswald, "Finding Knees in Multi-objective Optimization," *Parallel Problem Solving from Nature (PPSN VIII)*, pp. 722-731, 2004. DOI: [10.1007/978-3-540-30217-9_73](https://doi.org/10.1007/978-3-540-30217-9_73)

### Robustness and Sensitivity Analysis

For each Pareto point, assess objective sensitivity to design parameter perturbations:

- **Local sensitivity indices**: Partial derivatives df_i/dx_j — which knobs most affect which objectives
- **Sobol indices** (global): Decompose objective variance into per-parameter contributions. Can be estimated from the GP surrogate.
- **Robustness measure**: Worst-case objective degradation within an epsilon-ball in design space

**Practical approach:** Use GP surrogates to compute gradient-based sensitivity at each Pareto point — essentially free once the GP is fitted, producing a "sensitivity heatmap" across the front.

---

## Part 7: Modern and Hybrid Approaches (2023-2025)

### LLM-Guided Design Space Exploration

**LLM as initialization heuristic (LLMOPT):** Use an LLM to suggest initial design points based on natural language requirements. The LLM's world knowledge about "good" designs provides a much better starting point than random sampling. Directly applicable — our `ArchitectAgent` already reasons about design choices and can seed the initial sample.

**LLM as optimizer (OPRO):** Iteratively propose solutions based on the history of previous solutions and their scores. Surprisingly competitive with traditional optimizers.

**LLM for constraint elicitation:** Perhaps the most valuable role — helping the designer articulate objectives, suggest decompositions ("should NRE and unit cost be separate?"), and explain tradeoff results ("this saves 2W but costs $3 more per unit — acceptable for 10K volume?").

> **Reference (OPRO):** C. Yang, X. Wang, Y. Lu, H. Liu, Q. V. Le, D. Zhou, and X. Chen, "Large Language Models as Optimizers," *International Conference on Learning Representations (ICLR)*, 2024. arXiv: [2309.03409](https://arxiv.org/abs/2309.03409)
>
> **Reference (EvoPrompting):** A. Chen, D. M. Dohan, and D. R. So, "EvoPrompting: Language Models for Code-Level Neural Architecture Search," *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 36, 2023. arXiv: [2302.14838](https://arxiv.org/abs/2302.14838)

### Multi-Fidelity Multi-Objective Optimization

When multiple evaluation fidelity levels exist (analytical model → RTL simulation → synthesis → P&R), multi-fidelity methods choose both the candidate and the fidelity level:

- **MF-EHVI:** Extends EHVI to select fidelity. Cheap fidelities for exploration, expensive for refinement.
- **Knowledge gradient with multi-fidelity:** Maximally improves Pareto front value accounting for evaluation cost.

Our system already has a fidelity ladder: analytical models in `technology.py`/`manufacturing.py` (fast), RTL synthesis via `eda_tools/` (medium), place-and-route (high, if added). Multi-fidelity BO can automatically decide when to use which.

### Transfer Learning Across Design Campaigns

When exploring related problems (two drone designs with similar requirements):

- **Warm-starting BO:** Initialize GPs with data from prior campaigns
- **Multi-task GP:** Share prior across related problems, improving per-campaign sample efficiency

Directly relevant — the architect will run many campaigns for similar use cases. Transfer could reduce per-campaign budget from 200 to 50-100 evaluations.

---

## Part 8: Recommended Implementation Strategy

Given our system's characteristics — 4-6 objectives, mixed discrete/continuous design space, multi-fidelity evaluation, human-in-the-loop with LLM interface — we recommend three complementary approaches:

### Layer 1: MAP-Elites for Design Space Illumination

**Purpose:** Fast, coarse atlas of the design space.
**Budget:** 5,000-10,000 evaluations on analytical models (runs in seconds).
**Output:** Grid showing which objective combinations are feasible, where boundaries are, which regions are empty.
**Library:** pymoo (BSD, has QD algorithms and all classical MOO).

This runs first and gives the designer immediate orientation before any expensive optimization.

### Layer 2: Bayesian BO with qNEHVI + Exploration Bonus

**Purpose:** Sample-efficient Pareto front refinement with sensitivity analysis.
**Budget:** 100-200 evaluations (compatible with our hardware assessment pipeline).
**Output:** Refined Pareto front, GP uncertainty maps (= sensitivity analysis for free), marginal rates of substitution at each front point.
**Library:** BoTorch (MIT, PyTorch-native).

Uses an exploration-exploitation schedule: early iterations maximize GP variance (information gain), later iterations maximize EHVI (front refinement). MAP-Elites results seed the initialization.

### Layer 3: NSGA-III / MOEA/D as Many-Objective Fallback

**Purpose:** Robust fallback when objectives exceed 4-6 (where hypervolume-based BO becomes intractable).
**Budget:** 10,000+ evaluations on analytical models.
**Output:** Well-distributed Pareto front via reference-point-based selection.
**Library:** pymoo (has NSGA-II, NSGA-III, MOEA/D with reference directions).

### Integration Architecture

```
User requirements + constraints
        |
        v
  +---------------------------+
  |  Layer 1: MAP-Elites      |  Fast atlas on analytical models
  |  (5K-10K evals, ~seconds) |  "Here's what the design space looks like"
  +-------------+-------------+
                |  seed points + feasibility map
                v
  +---------------------------+
  |  Layer 2: Bayesian BO     |  Refined Pareto front + sensitivity
  |  qNEHVI + exploration     |  "Here's the tradeoff surface"
  |  (100-200 evals)          |  "Here's where we're uncertain"
  +-------------+-------------+
                |  Pareto front + GP posteriors
                v
  +---------------------------+
  |  Layer 3: LLM Interpret   |  ArchitectAgent explains tradeoffs,
  |  (existing agent)         |  suggests design point based on priorities
  +---------------------------+
```

For >4 objectives, Layer 2 swaps from BO to MOEA/D automatically.

### Key Library Dependencies

| Library | License | Purpose | PyPI |
|---------|---------|---------|------|
| **pymoo** | Apache 2.0 | NSGA-II/III, MOEA/D, MAP-Elites, quality indicators | `pymoo` |
| **BoTorch** | MIT | qNEHVI, GP models, multi-fidelity BO | `botorch` |
| **Ax** | MIT | Higher-level BO API on BoTorch | `ax-platform` |

> **Reference (pymoo):** J. Blank and K. Deb, "pymoo: Multi-Objective Optimization in Python," *IEEE Access*, vol. 8, pp. 89497-89509, 2020. DOI: [10.1109/ACCESS.2020.2990567](https://doi.org/10.1109/ACCESS.2020.2990567)

---

## References (Complete)

### Classical Multi-Objective Optimization

1. K. Deb, A. Pratap, S. Agarwal, and T. Meyarivan, "A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II," *IEEE Trans. Evol. Comput.*, vol. 6, no. 2, pp. 182-197, 2002. [DOI](https://doi.org/10.1109/4235.996017)

2. K. Deb and H. Jain, "An Evolutionary Many-Objective Optimization Algorithm Using Reference-Point-Based Nondominated Sorting Approach, Part I," *IEEE Trans. Evol. Comput.*, vol. 18, no. 4, pp. 577-601, 2014. [DOI](https://doi.org/10.1109/TEVC.2013.2281535)

3. Q. Zhang and H. Li, "MOEA/D: A Multiobjective Evolutionary Algorithm Based on Decomposition," *IEEE Trans. Evol. Comput.*, vol. 11, no. 6, pp. 712-731, 2007. [DOI](https://doi.org/10.1109/TEVC.2007.892759)

4. E. Zitzler, M. Laumanns, and L. Thiele, "SPEA2: Improving the Strength Pareto Evolutionary Algorithm," *TIK-Report 103*, ETH Zurich, 2001. [DOI](https://doi.org/10.3929/ethz-a-004284029)

5. N. Beume, B. Naujoks, and M. Emmerich, "SMS-EMOA: Multiobjective selection based on dominated hypervolume," *Eur. J. Oper. Res.*, vol. 181, no. 3, pp. 1653-1669, 2007. [DOI](https://doi.org/10.1016/j.ejor.2006.08.008)

### Bayesian Multi-Objective Optimization

6. S. Daulton, M. Balandat, and E. Bakshy, "Differentiable Expected Hypervolume Improvement for Parallel Multi-Objective Bayesian Optimization," *NeurIPS*, vol. 33, pp. 9851-9864, 2020. [arXiv](https://arxiv.org/abs/2006.05078)

7. S. Daulton, M. Balandat, and E. Bakshy, "Parallel Bayesian Optimization of Multiple Noisy Objectives with Expected Hypervolume Improvement," *NeurIPS*, vol. 34, pp. 2187-2200, 2021. [arXiv](https://arxiv.org/abs/2105.08195)

8. J. Knowles, "ParEGO: A Hybrid Algorithm with On-Line Landscape Approximation for Expensive Multiobjective Optimization Problems," *IEEE Trans. Evol. Comput.*, vol. 10, no. 1, pp. 50-66, 2006. [DOI](https://doi.org/10.1109/TEVC.2005.851274)

9. M. Balandat, B. Karrer, D. R. Jiang, et al., "BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization," *NeurIPS*, vol. 33, pp. 21524-21538, 2020. [arXiv](https://arxiv.org/abs/1910.06403)

10. M. Lindauer, K. Eggensperger, M. Feurer, et al., "SMAC3: A Versatile Bayesian Optimization Package for Hyperparameter Optimization," *JMLR*, vol. 23, no. 54, pp. 1-9, 2022.

### Information-Theoretic Approaches

11. S. Belakaria, A. Deshwal, and J. R. Doppa, "Max-value Entropy Search for Multi-Objective Bayesian Optimization," *NeurIPS*, vol. 32, 2019. [arXiv](https://arxiv.org/abs/1905.08879)

12. S. Suzuki, S. Takeno, T. Tamura, K. Shitara, and M. Karasuyama, "Multi-Objective Bayesian Optimization using Pareto-Frontier Entropy," *ICML*, pp. 9279-9288, 2020. [arXiv](https://arxiv.org/abs/2004.01566) *(verify arXiv ID)*

13. B. Tu, A. Gandy, N. Kantas, and B. Sheridan, "Joint Entropy Search for Multi-Objective Bayesian Optimization," *NeurIPS*, vol. 35, 2022. [arXiv](https://arxiv.org/abs/2210.02905)

14. D. Hernandez-Lobato, J. M. Hernandez-Lobato, A. Shah, and R. P. Adams, "Predictive Entropy Search for Multi-objective Bayesian Optimization," *ICML*, pp. 1492-1501, 2016. [arXiv](https://arxiv.org/abs/1511.05467)

### Quality-Diversity Optimization

15. J.-B. Mouret and J. Clune, "Illuminating search spaces by mapping elites," arXiv preprint, 2015. [arXiv](https://arxiv.org/abs/1504.04909)

16. M. Fontaine and S. Nikolaidis, "Differentiable Quality Diversity," *NeurIPS*, vol. 36, 2023. [arXiv](https://arxiv.org/abs/2106.03894)

17. B. Liu and B. C. Schafer, "Efficient and Reliable High-Level Synthesis Design Space Exploration via Quality-Diversity Optimization," *DAC*, 2024.

### Chip Design and Hardware DSE

18. W. Lyu, P. Xue, F. Yang, et al., "An Efficient Bayesian Optimization Approach for Automated Optimization of Analog Circuits," *IEEE Trans. Circuits Syst. I*, vol. 65, no. 6, pp. 1954-1967, 2018. [DOI](https://doi.org/10.1109/TCSI.2017.2768826)

19. B. C. Schafer and Z. Wang, "High-Level Synthesis Design Space Exploration: Past, Present, and Future," *IEEE TCAD*, vol. 39, no. 10, pp. 2628-2639, 2020. [DOI](https://doi.org/10.1109/TCAD.2019.2943570)

20. C. Palermo, V. Catania, and D. Patti, "ReSPIR: A Response Surface-based Pareto Iterative Refinement for application-specific design space exploration," *IEEE TCAD*, vol. 28, no. 12, pp. 1816-1829, 2009. [DOI](https://doi.org/10.1109/TCAD.2009.2028681)

21. A. Mirhoseini, A. Goldie, M. Yazgan, et al., "A graph placement methodology for fast chip design," *Nature*, vol. 594, pp. 207-212, 2021. [DOI](https://doi.org/10.1038/s41586-021-03544-w)

### Neural Architecture Search for Edge

22. M. Tan, B. Chen, R. Pang, et al., "MnasNet: Platform-Aware Neural Architecture Search for Mobile," *CVPR*, pp. 2820-2828, 2019. [DOI](https://doi.org/10.1109/CVPR.2019.00293)

23. H. Cai, C. Gan, T. Wang, Z. Zhang, and S. Han, "Once-for-All: Train One Network and Specialize it for Efficient Deployment," *ICLR*, 2020. [arXiv](https://arxiv.org/abs/1908.09791)

### LLM + Optimization

24. C. Yang, X. Wang, Y. Lu, et al., "Large Language Models as Optimizers," *ICLR*, 2024. [arXiv](https://arxiv.org/abs/2309.03409)

25. A. Chen, D. M. Dohan, and D. R. So, "EvoPrompting: Language Models for Code-Level Neural Architecture Search," *NeurIPS*, vol. 36, 2023. [arXiv](https://arxiv.org/abs/2302.14838)

### Sensitivity and Knee Points

26. J. Branke, K. Deb, H. Dierolf, and M. Osswald, "Finding Knees in Multi-objective Optimization," *PPSN VIII*, pp. 722-731, 2004. [DOI](https://doi.org/10.1007/978-3-540-30217-9_73)

### Frameworks

27. J. Blank and K. Deb, "pymoo: Multi-Objective Optimization in Python," *IEEE Access*, vol. 8, pp. 89497-89509, 2020. [DOI](https://doi.org/10.1109/ACCESS.2020.2990567)

### Quality-Diversity Variants

28. M. Fontaine, J. Togelius, S. Nikolaidis, and A. Hoover, "Covariance Matrix Adaptation for the Rapid Illumination of Behavior Space," *GECCO*, 2020. [arXiv](https://arxiv.org/abs/1912.02400)

### Multi-Fidelity Multi-Objective Optimization

29. S. Belakaria, A. Deshwal, and J. R. Doppa, "Multi-Fidelity Multi-Objective Bayesian Optimization: An Output Space Entropy Search Approach," *AAAI*, 2020. [arXiv](https://arxiv.org/abs/1911.01667)

30. K. Kandasamy, G. Dasarathy, J. Schneider, and B. Poczos, "Multi-Fidelity Bayesian Optimisation with Continuous Approximations," *ICML*, 2017.

### Additional Chip Design DSE

31. S. Bai et al., "BOOM-Explorer: RISC-V BOOM Microarchitecture Design Space Exploration Framework," *ICCAD*, 2021. *(Multi-objective BO for RISC-V core DSE)*

32. M. Konakovic Lukovic, Y. Tian, and W. Matusik, "Diversity-Guided Multi-Objective Bayesian Optimization With Batch Evaluations," *NeurIPS*, 2020. [arXiv](https://arxiv.org/abs/2006.13571)

### Hardware-Aware NAS (Additional)

33. B. Wu, X. Dai, P. Zhang, et al., "FBNet: Hardware-Aware Efficient ConvNet Design via Differentiable Neural Architecture Search," *CVPR*, 2019.

34. K. Wang, Z. Liu, Y. Lin, J. Lin, and S. Han, "HAQ: Hardware-Aware Automated Quantization with Mixed Precision," *CVPR*, 2019.

---

*Note: arXiv IDs and DOIs should be verified before citation in formal publications. References marked with "(verify)" have lower confidence on the exact identifier.*
