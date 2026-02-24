"""Layer 3: NSGA-III for many-objective optimization (>4 objectives).

Wraps pymoo's NSGA-III with the DesignSpace and evaluator abstractions.
Falls back to this layer when BoTorch is not available or when the
problem has more than 4 objectives.

Requires pymoo (optional dependency).

Usage:
    from embodied_ai_architect.graphs.moo.nsga3 import NSGA3MOO, NSGA3Config

    nsga = NSGA3MOO(design_space, evaluator, config)
    result = nsga.run(seed_points=me_best)
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np
from pydantic import BaseModel, Field

from embodied_ai_architect.graphs.moo.design_space import DesignSpace
from embodied_ai_architect.graphs.moo.evaluator import DesignEvaluator

logger = logging.getLogger(__name__)


class NSGA3Config(BaseModel):
    """Configuration for NSGA-III."""

    pop_size: int = Field(default=100, description="Population size")
    n_generations: int = Field(default=100, description="Number of generations")
    reference_directions: str = Field(
        default="energy",
        description="Reference direction generation method: 'energy' or 'das_dennis'",
    )


class NSGA3Result(BaseModel):
    """Result of NSGA-III optimization."""

    pareto_front: list[dict[str, Any]] = Field(default_factory=list)
    hypervolume: float = 0.0
    generation_history: list[dict[str, float]] = Field(default_factory=list)
    total_evaluations: int = 0


class NSGA3MOO:
    """NSGA-III wrapper for many-objective optimization.

    Uses pymoo's implementation with our DesignSpace encoding and evaluator.
    """

    def __init__(
        self,
        design_space: DesignSpace,
        evaluator: DesignEvaluator,
        config: NSGA3Config | None = None,
    ):
        self.ds = design_space
        self.evaluator = evaluator
        self.config = config or NSGA3Config()

    def run(
        self,
        seed_points: list[dict[str, Any]] | None = None,
        callback: Callable[[int, int, float], None] | None = None,
    ) -> NSGA3Result:
        """Run NSGA-III optimization.

        Args:
            seed_points: Optional initial population from Layer 1.
            callback: Optional (generation, total_evals, hypervolume) callback.

        Returns:
            NSGA3Result with Pareto front.
        """
        from pymoo.algorithms.moo.nsga3 import NSGA3
        from pymoo.core.problem import ElementwiseProblem
        from pymoo.optimize import minimize
        from pymoo.util.ref_dirs import get_reference_directions
        from pymoo.indicators.hv import HV

        cfg = self.config

        # Reference directions
        ref_dirs = get_reference_directions(
            cfg.reference_directions,
            self.ds.n_obj,
            n_points=cfg.pop_size,
        )

        # Define pymoo problem
        problem_kwargs = self.ds.to_pymoo_problem_kwargs()

        evaluator = self.evaluator
        ds = self.ds

        class MOOProblem(ElementwiseProblem):
            def __init__(self):
                super().__init__(**problem_kwargs)

            def _evaluate(self, x, out, *args, **kwargs):
                params = ds.decode(x)
                result = evaluator.evaluate(params)
                # pymoo minimizes by default
                f = [result.objectives.get(name, 1e6) for name in ds.objectives]
                out["F"] = np.array(f)

        problem = MOOProblem()

        # Algorithm
        algorithm = NSGA3(
            pop_size=cfg.pop_size,
            ref_dirs=ref_dirs,
        )

        # Initial population from seed points
        if seed_points:
            from pymoo.core.population import Population

            X0 = np.array([ds.encode(p) for p in seed_points[: cfg.pop_size]])
            if len(X0) < cfg.pop_size:
                # Pad with random
                random_samples = ds.sample_random(cfg.pop_size - len(X0))
                X_random = np.array([ds.encode(s) for s in random_samples])
                X0 = np.vstack([X0, X_random])
            algorithm.initialization = Population.new("X", X0)

        # Run optimization
        generation_history = []
        total_evals = [0]

        class GenerationCallback:
            def __call__(self, algorithm_instance):
                gen = algorithm_instance.n_gen
                n_evals = algorithm_instance.evaluator.n_eval
                total_evals[0] = n_evals

                # Compute hypervolume if possible
                try:
                    F = algorithm_instance.pop.get("F")
                    ref_point = np.max(F, axis=0) * 1.1
                    hv_calc = HV(ref_point=ref_point)
                    hv = float(hv_calc(F))
                except Exception:
                    hv = 0.0

                generation_history.append(
                    {
                        "generation": gen,
                        "evaluations": n_evals,
                        "hypervolume": hv,
                    }
                )

                if callback:
                    callback(gen, n_evals, hv)

        try:
            res = minimize(
                problem,
                algorithm,
                ("n_gen", cfg.n_generations),
                verbose=False,
                callback=GenerationCallback(),
            )
        except Exception as e:
            logger.warning("NSGA-III failed with callback, retrying without: %s", e)
            # Retry without callback if it caused issues
            res = minimize(
                problem,
                algorithm,
                ("n_gen", cfg.n_generations),
                verbose=False,
            )
            total_evals[0] = cfg.pop_size * cfg.n_generations

        # Extract Pareto front
        pareto_front = []
        if res.F is not None:
            for i in range(len(res.X)):
                params = ds.decode(res.X[i])
                objs = {}
                for j, name in enumerate(ds.objectives):
                    objs[name] = round(float(res.F[i, j]), 4)
                pareto_front.append(
                    {
                        "design_params": params,
                        "objectives": objs,
                        "feasible": True,
                    }
                )

        # Hypervolume
        hv = 0.0
        if res.F is not None and len(res.F) > 0:
            try:
                ref_point = np.max(res.F, axis=0) * 1.1
                hv_calc = HV(ref_point=ref_point)
                hv = float(hv_calc(res.F))
            except Exception:
                pass

        return NSGA3Result(
            pareto_front=pareto_front,
            hypervolume=hv,
            generation_history=generation_history,
            total_evaluations=total_evals[0],
        )
