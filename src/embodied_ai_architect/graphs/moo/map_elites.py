"""MAP-Elites: quality-diversity algorithm for design space illumination.

Layer 1 of the MOO pipeline. Fills a grid of behavioral feature bins with
the best design found for each bin, providing a fast atlas of the design
space in seconds (5K-10K evaluations on analytical models).

Custom implementation (~200 lines) since pymoo lacks MAP-Elites.

Usage:
    from embodied_ai_architect.graphs.moo.map_elites import MAPElites, MAPElitesConfig

    config = MAPElitesConfig(n_iterations=100, batch_size=64)
    me = MAPElites(design_space, evaluator, executor, config)
    result = me.run()
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np
from pydantic import BaseModel, Field

from embodied_ai_architect.graphs.moo.design_space import DesignSpace, VarType
from embodied_ai_architect.graphs.moo.evaluator import DesignEvaluator, EvaluationResult
from embodied_ai_architect.graphs.moo.executor import EvalBackend

logger = logging.getLogger(__name__)


class MAPElitesConfig(BaseModel):
    """Configuration for MAP-Elites algorithm."""

    feature_dimensions: list[str] = Field(
        default_factory=lambda: ["power_watts", "latency_ms"],
        description="Objective names to use as feature dimensions for the grid",
    )
    resolution: int = Field(
        default=20,
        description="Grid resolution per feature dimension",
    )
    n_iterations: int = Field(default=100, description="Number of iterations")
    batch_size: int = Field(default=64, description="Evaluations per iteration")
    initial_population: int = Field(
        default=256,
        description="Random designs for initial population",
    )
    mutation_strength: float = Field(
        default=0.2,
        description="Mutation strength (fraction of variable range)",
    )
    crossover_rate: float = Field(
        default=0.3,
        description="Probability of crossover vs mutation",
    )


class MAPElitesCell(BaseModel):
    """A single cell in the MAP-Elites grid."""

    result: EvaluationResult
    fitness: float  # aggregate fitness (lower is better for minimization)


class MAPElitesResult(BaseModel):
    """Result of MAP-Elites run."""

    grid_cells: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Cell key -> best design in that cell",
    )
    coverage: float = Field(
        default=0.0,
        description="Fraction of grid cells filled (0-1)",
    )
    best_per_objective: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Best design found for each objective",
    )
    total_evaluations: int = 0
    total_cells: int = 0
    filled_cells: int = 0
    best_designs: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Top designs across all cells for seeding Layer 2",
    )

    model_config = {"arbitrary_types_allowed": True}


class MAPElites:
    """MAP-Elites quality-diversity algorithm.

    Maintains a grid indexed by behavioral features (objective values).
    Each cell stores the best design found for that region of objective space.
    """

    def __init__(
        self,
        design_space: DesignSpace,
        evaluator: DesignEvaluator,
        executor: EvalBackend,
        config: MAPElitesConfig | None = None,
    ):
        self.ds = design_space
        self.evaluator = evaluator
        self.executor = executor
        self.config = config or MAPElitesConfig()

        self._rng = np.random.default_rng(42)
        self._grid: dict[tuple[int, ...], MAPElitesCell] = {}
        self._feature_mins: dict[str, float] = {}
        self._feature_maxs: dict[str, float] = {}
        self._all_results: list[EvaluationResult] = []

    def run(
        self,
        callback: Callable[[int, int, float], None] | None = None,
    ) -> MAPElitesResult:
        """Run MAP-Elites algorithm.

        Args:
            callback: Optional (iteration, total_evals, coverage) callback.

        Returns:
            MAPElitesResult with the filled grid atlas.
        """
        cfg = self.config
        total_evals = 0

        # Phase 1: Initial random population
        initial_samples = self.ds.sample_random(cfg.initial_population, seed=42)
        initial_results = self.executor.submit_batch(initial_samples)
        self._all_results.extend(initial_results)
        total_evals += len(initial_results)

        # Initialize feature bounds from initial results
        self._init_feature_bounds(initial_results)

        # Place initial results into grid
        for result in initial_results:
            self._try_place(result)

        if callback:
            coverage = len(self._grid) / max(1, cfg.resolution ** len(cfg.feature_dimensions))
            callback(0, total_evals, coverage)

        # Phase 2: Iterative improvement
        for iteration in range(cfg.n_iterations):
            # Generate batch via mutation and crossover
            batch = self._generate_batch(cfg.batch_size)
            results = self.executor.submit_batch(batch)
            self._all_results.extend(results)
            total_evals += len(results)

            # Update feature bounds and place results
            self._update_feature_bounds(results)
            for result in results:
                self._try_place(result)

            if callback:
                total_cells = max(1, cfg.resolution ** len(cfg.feature_dimensions))
                coverage = len(self._grid) / total_cells
                callback(iteration + 1, total_evals, coverage)

        return self._build_result(total_evals)

    def _init_feature_bounds(self, results: list[EvaluationResult]) -> None:
        """Initialize feature dimension bounds from initial evaluations."""
        for feat in self.config.feature_dimensions:
            vals = [r.objectives.get(feat, 0.0) for r in results if r.feasible]
            if not vals:
                vals = [r.objectives.get(feat, 0.0) for r in results]
            if vals:
                self._feature_mins[feat] = min(vals)
                self._feature_maxs[feat] = max(vals)
                # Add margin to avoid edge effects
                rng = self._feature_maxs[feat] - self._feature_mins[feat]
                if rng < 1e-10:
                    rng = max(abs(self._feature_mins[feat]) * 0.1, 1.0)
                self._feature_mins[feat] -= rng * 0.05
                self._feature_maxs[feat] += rng * 0.05

    def _update_feature_bounds(self, results: list[EvaluationResult]) -> None:
        """Expand feature bounds if new results exceed current range."""
        for feat in self.config.feature_dimensions:
            for r in results:
                val = r.objectives.get(feat, 0.0)
                if feat in self._feature_mins:
                    self._feature_mins[feat] = min(self._feature_mins[feat], val)
                    self._feature_maxs[feat] = max(self._feature_maxs[feat], val)

    def _feature_to_cell(self, result: EvaluationResult) -> tuple[int, ...]:
        """Map evaluation result to grid cell indices."""
        res = self.config.resolution
        indices = []
        for feat in self.config.feature_dimensions:
            val = result.objectives.get(feat, 0.0)
            lo = self._feature_mins.get(feat, 0.0)
            hi = self._feature_maxs.get(feat, 1.0)
            rng = hi - lo
            if rng < 1e-10:
                idx = res // 2
            else:
                idx = int((val - lo) / rng * res)
                idx = max(0, min(res - 1, idx))
            indices.append(idx)
        return tuple(indices)

    def _fitness(self, result: EvaluationResult) -> float:
        """Compute aggregate fitness (lower is better).

        Uses sum of normalized objectives for minimization problems.
        """
        total = 0.0
        for obj_name in self.ds.objectives:
            val = result.objectives.get(obj_name, 1e6)
            # Normalize by observed range
            lo = self._feature_mins.get(obj_name, 0.0)
            hi = self._feature_maxs.get(obj_name, 1.0)
            rng = hi - lo
            if rng > 1e-10:
                total += (val - lo) / rng
            else:
                total += val
        return total

    def _try_place(self, result: EvaluationResult) -> bool:
        """Try to place a result in the grid. Returns True if placed."""
        cell_key = self._feature_to_cell(result)
        fitness = self._fitness(result)

        existing = self._grid.get(cell_key)
        if existing is None or fitness < existing.fitness:
            self._grid[cell_key] = MAPElitesCell(result=result, fitness=fitness)
            return True
        return False

    def _generate_batch(self, batch_size: int) -> list[dict[str, Any]]:
        """Generate a batch of new designs via mutation and crossover."""
        batch = []
        elites = list(self._grid.values())
        if not elites:
            return self.ds.sample_random(batch_size)

        for _ in range(batch_size):
            if self._rng.random() < self.config.crossover_rate and len(elites) >= 2:
                # Crossover two random elites
                idx1, idx2 = self._rng.choice(len(elites), size=2, replace=False)
                child = self._crossover(
                    elites[idx1].result.design_params,
                    elites[idx2].result.design_params,
                )
            else:
                # Mutate a random elite
                parent = elites[self._rng.integers(len(elites))]
                child = self._mutate(parent.result.design_params)
            batch.append(child)
        return batch

    def _mutate(self, params: dict[str, Any]) -> dict[str, Any]:
        """Mutate a design point."""
        child = dict(params)
        # Mutate 1-3 variables
        n_mutations = self._rng.integers(1, min(4, self.ds.n_var + 1))
        var_indices = self._rng.choice(self.ds.n_var, size=n_mutations, replace=False)

        for idx in var_indices:
            var = self.ds.variables[idx]
            if var.var_type == VarType.CATEGORICAL:
                child[var.name] = var.choices[self._rng.integers(len(var.choices))]
            elif var.var_type == VarType.INTEGER:
                current = int(child.get(var.name, var.lower))
                rng = var.upper - var.lower
                delta = int(self._rng.normal(0, self.config.mutation_strength * rng))
                new_val = max(int(var.lower), min(int(var.upper), current + delta))
                child[var.name] = new_val
            elif var.var_type == VarType.CONTINUOUS:
                current = float(child.get(var.name, var.lower))
                rng = var.upper - var.lower
                delta = self._rng.normal(0, self.config.mutation_strength * rng)
                new_val = max(var.lower, min(var.upper, current + delta))
                child[var.name] = new_val
        return child

    def _crossover(self, parent1: dict[str, Any], parent2: dict[str, Any]) -> dict[str, Any]:
        """Uniform crossover between two designs."""
        child = {}
        for var in self.ds.variables:
            if self._rng.random() < 0.5:
                child[var.name] = parent1.get(var.name)
            else:
                child[var.name] = parent2.get(var.name)
        return child

    def _build_result(self, total_evals: int) -> MAPElitesResult:
        """Build the final result from the grid."""
        total_cells = max(1, self.config.resolution ** len(self.config.feature_dimensions))
        filled = len(self._grid)

        # Extract grid cells as serializable dicts
        grid_cells = {}
        for key, cell in self._grid.items():
            cell_key = "_".join(str(k) for k in key)
            grid_cells[cell_key] = {
                "design_params": cell.result.design_params,
                "objectives": cell.result.objectives,
                "feasible": cell.result.feasible,
                "fitness": cell.fitness,
            }

        # Find best per objective
        best_per_obj: dict[str, dict[str, Any]] = {}
        for obj_name in self.ds.objectives:
            best = None
            best_val = float("inf")
            for cell in self._grid.values():
                val = cell.result.objectives.get(obj_name, float("inf"))
                if val < best_val:
                    best_val = val
                    best = cell
            if best is not None:
                best_per_obj[obj_name] = {
                    "design_params": best.result.design_params,
                    "objectives": best.result.objectives,
                    "value": best_val,
                }

        # Top designs for seeding Layer 2 (sorted by fitness)
        all_cells = sorted(self._grid.values(), key=lambda c: c.fitness)
        best_designs = [
            {
                "design_params": c.result.design_params,
                "objectives": c.result.objectives,
                "feasible": c.result.feasible,
            }
            for c in all_cells[:50]  # Top 50 for seeding BO
        ]

        return MAPElitesResult(
            grid_cells=grid_cells,
            coverage=filled / total_cells,
            best_per_objective=best_per_obj,
            total_evaluations=total_evals,
            total_cells=total_cells,
            filled_cells=filled,
            best_designs=best_designs,
        )
