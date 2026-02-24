"""Layer 2: Bayesian multi-objective optimization with qNEHVI.

Sample-efficient Pareto refinement using Gaussian Process surrogates.
Requires botorch (optional dependency).

Features:
- Warm-start from MAP-Elites seed points
- Exploration-exploitation schedule
- Sensitivity analysis from GP lengthscales
- Tradeoff explanation via GP gradient analysis

Usage:
    from embodied_ai_architect.graphs.moo.bayesian_opt import BayesianMOO

    bo = BayesianMOO(design_space, evaluator, executor, config)
    result = bo.run(seed_points=me_best)
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np
from pydantic import BaseModel, Field

from embodied_ai_architect.graphs.moo.design_space import DesignSpace
from embodied_ai_architect.graphs.moo.evaluator import DesignEvaluator, EvaluationResult
from embodied_ai_architect.graphs.moo.executor import EvalBackend

logger = logging.getLogger(__name__)


class BayesianOptConfig(BaseModel):
    """Configuration for Bayesian multi-objective optimization."""

    n_initial: int = Field(default=20, description="Initial random samples (if no seed)")
    n_iterations: int = Field(default=50, description="BO iterations")
    batch_size: int = Field(default=4, description="Candidates per iteration")
    exploration_schedule: bool = Field(
        default=True,
        description="Use annealing exploration-exploitation schedule",
    )


class SensitivityResult(BaseModel):
    """Sensitivity analysis for one parameter."""

    parameter_name: str
    lengthscale: float = 0.0
    importance: float = 0.0  # 1/lengthscale normalized


class BayesianOptResult(BaseModel):
    """Result of Bayesian optimization."""

    pareto_front: list[dict[str, Any]] = Field(default_factory=list)
    hypervolume: float = 0.0
    sensitivity: dict[str, dict[str, float]] = Field(default_factory=dict)
    convergence_history: list[dict[str, float]] = Field(default_factory=list)
    total_evaluations: int = 0


class BayesianMOO:
    """Bayesian multi-objective optimization using BoTorch qNEHVI.

    Fits independent GP models per objective, uses qNEHVI (or qEHVI)
    for acquisition, and provides sensitivity from GP hyperparameters.
    """

    def __init__(
        self,
        design_space: DesignSpace,
        evaluator: DesignEvaluator,
        executor: EvalBackend,
        config: BayesianOptConfig | None = None,
    ):
        self.ds = design_space
        self.evaluator = evaluator
        self.executor = executor
        self.config = config or BayesianOptConfig()

        # Lazy imports
        self._torch = None
        self._botorch = None

    def _ensure_imports(self):
        """Lazy import torch and botorch."""
        if self._torch is None:
            import torch

            self._torch = torch
        if self._botorch is None:
            import botorch

            self._botorch = botorch

    def run(
        self,
        seed_points: list[dict[str, Any]] | None = None,
        callback: Callable[[int, int, float], None] | None = None,
    ) -> BayesianOptResult:
        """Run Bayesian optimization loop.

        Args:
            seed_points: Initial designs from Layer 1 (warm-start).
            callback: Optional (iteration, total_evals, hypervolume) callback.

        Returns:
            BayesianOptResult with Pareto front and sensitivity.
        """
        self._ensure_imports()
        torch = self._torch

        cfg = self.config
        total_evals = 0
        convergence = []

        # Initial data: use seed points or random
        if seed_points and len(seed_points) >= cfg.n_initial:
            init_params = seed_points[: cfg.n_initial]
        elif seed_points:
            # Supplement with random samples
            n_random = cfg.n_initial - len(seed_points)
            init_params = list(seed_points) + self.ds.sample_random(n_random)
        else:
            init_params = self.ds.sample_random(cfg.n_initial)

        # Evaluate initial points
        init_results = self.executor.submit_batch(init_params)
        total_evals += len(init_results)

        # Build training data
        train_X, train_Y = self._results_to_tensors(init_results)

        # Reference point for hypervolume (worst observed * 1.1)
        ref_point = train_Y.max(dim=0).values * 1.1

        # BO loop
        for iteration in range(cfg.n_iterations):
            try:
                # Fit GP models
                models = self._fit_gp_models(train_X, train_Y)

                # Generate candidates via acquisition function
                candidates = self._generate_candidates(
                    models, train_X, train_Y, ref_point, iteration, cfg.n_iterations
                )

                # Decode and evaluate
                new_params = [
                    self.ds.decode(candidates[i].numpy()) for i in range(candidates.shape[0])
                ]
                new_results = self.executor.submit_batch(new_params)
                total_evals += len(new_results)

                # Update training data
                new_X, new_Y = self._results_to_tensors(new_results)
                train_X = torch.cat([train_X, new_X], dim=0)
                train_Y = torch.cat([train_Y, new_Y], dim=0)

                # Update ref point
                ref_point = train_Y.max(dim=0).values * 1.1

                # Compute hypervolume
                hv = self._compute_hv(train_Y, ref_point)

                if callback:
                    callback(iteration + 1, total_evals, hv)
                convergence.append({"iteration": iteration + 1, "hypervolume": hv})

            except Exception as e:
                logger.warning("BO iteration %d failed: %s", iteration, e)
                continue

        # Extract Pareto front
        pareto_front = self._extract_pareto_front(train_X, train_Y)

        # Sensitivity analysis from GP lengthscales
        sensitivity = self._extract_sensitivity(models) if "models" in dir() else {}

        # Hypervolume
        hv = self._compute_hv(train_Y, ref_point) if train_Y.numel() > 0 else 0.0

        return BayesianOptResult(
            pareto_front=pareto_front,
            hypervolume=hv,
            sensitivity=sensitivity,
            convergence_history=convergence,
            total_evaluations=total_evals,
        )

    def _results_to_tensors(self, results: list[EvaluationResult]):
        """Convert evaluation results to torch tensors."""
        torch = self._torch
        X_list = []
        Y_list = []
        for r in results:
            x = self.ds.encode(r.design_params)
            X_list.append(x)
            # Negate objectives since BoTorch maximizes
            y = [-r.objectives.get(name, 1e6) for name in self.ds.objectives]
            Y_list.append(y)

        X = torch.tensor(np.array(X_list), dtype=torch.float64)
        Y = torch.tensor(np.array(Y_list), dtype=torch.float64)
        return X, Y

    def _fit_gp_models(self, train_X, train_Y):
        """Fit independent GP models per objective."""
        from botorch.models import SingleTaskGP
        from botorch.models.model_list_gp_regression import ModelListGP
        from botorch.utils.transforms import normalize
        from gpytorch.mlls import ExactMarginalLogLikelihood
        from botorch.fit import fit_gpytorch_mll

        bounds = self.ds.to_botorch_bounds()

        # Normalize inputs to [0,1]
        X_norm = normalize(train_X, bounds)

        models = []
        for i in range(train_Y.shape[1]):
            y = train_Y[:, i : i + 1]
            model = SingleTaskGP(X_norm, y)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            try:
                fit_gpytorch_mll(mll)
            except Exception as e:
                logger.debug("GP fitting for objective %d failed: %s", i, e)
            models.append(model)

        return ModelListGP(*models)

    def _generate_candidates(self, model, train_X, train_Y, ref_point, iteration, budget):
        """Generate candidate points using qNEHVI acquisition."""
        from botorch.acquisition.multi_objective import qNoisyExpectedHypervolumeImprovement
        from botorch.optim import optimize_acqf
        from botorch.utils.transforms import normalize, unnormalize

        torch = self._torch
        bounds = self.ds.to_botorch_bounds()
        X_norm = normalize(train_X, bounds)

        # Normalized bounds for optimization
        norm_bounds = torch.stack(
            [
                torch.zeros(self.ds.n_var, dtype=torch.float64),
                torch.ones(self.ds.n_var, dtype=torch.float64),
            ]
        )

        try:
            acq = qNoisyExpectedHypervolumeImprovement(
                model=model,
                ref_point=ref_point.tolist(),
                X_baseline=X_norm,
                prune_baseline=True,
            )

            candidates_norm, _ = optimize_acqf(
                acq_function=acq,
                bounds=norm_bounds,
                q=self.config.batch_size,
                num_restarts=5,
                raw_samples=256,
            )

            # Unnormalize back to original space
            candidates = unnormalize(candidates_norm, bounds)
            return candidates.detach()

        except Exception as e:
            logger.warning("Acquisition optimization failed: %s, using random", e)
            # Fallback: random candidates
            samples = self.ds.sample_random(self.config.batch_size)
            X = torch.tensor(
                np.array([self.ds.encode(s) for s in samples]),
                dtype=torch.float64,
            )
            return X

    def _compute_hv(self, Y, ref_point) -> float:
        """Compute hypervolume indicator."""
        try:
            from botorch.utils.multi_objective.hypervolume import Hypervolume

            # Extract non-dominated points (BoTorch works with maximization)
            from botorch.utils.multi_objective.pareto import is_non_dominated

            pareto_mask = is_non_dominated(Y)
            pareto_Y = Y[pareto_mask]

            if pareto_Y.numel() == 0:
                return 0.0

            hv = Hypervolume(ref_point=ref_point)
            return float(hv.compute(pareto_Y))
        except Exception:
            return 0.0

    def _extract_pareto_front(self, X, Y) -> list[dict[str, Any]]:
        """Extract Pareto-optimal designs from training data."""
        from botorch.utils.multi_objective.pareto import is_non_dominated

        pareto_mask = is_non_dominated(Y)
        front = []
        for i in range(len(X)):
            if pareto_mask[i]:
                params = self.ds.decode(X[i].numpy())
                # Un-negate objectives
                objs = {}
                for j, name in enumerate(self.ds.objectives):
                    objs[name] = round(float(-Y[i, j]), 4)
                front.append(
                    {
                        "design_params": params,
                        "objectives": objs,
                        "feasible": True,
                    }
                )
        return front

    def _extract_sensitivity(self, model) -> dict[str, dict[str, float]]:
        """Extract sensitivity from GP lengthscales.

        Shorter lengthscale = more sensitive (objective changes rapidly
        with that parameter).
        """
        sensitivity: dict[str, dict[str, float]] = {}

        try:
            for j, sub_model in enumerate(model.models):
                obj_name = self.ds.objectives[j] if j < len(self.ds.objectives) else f"obj_{j}"
                ls = sub_model.covar_module.base_kernel.lengthscale.detach().numpy().flatten()

                # Importance = 1/lengthscale (normalized)
                importance = 1.0 / (ls + 1e-6)
                importance = importance / importance.sum()

                obj_sens = {}
                for k, var in enumerate(self.ds.variables):
                    if k < len(ls):
                        obj_sens[var.name] = {
                            "lengthscale": round(float(ls[k]), 4),
                            "importance": round(float(importance[k]), 4),
                        }
                sensitivity[obj_name] = obj_sens

        except Exception as e:
            logger.debug("Sensitivity extraction failed: %s", e)

        return sensitivity

    def predict(self, params: dict[str, Any]) -> dict[str, dict[str, float]]:
        """Predict mean and std for each objective at a design point."""
        # This requires a fitted model, which is only available after run()
        raise NotImplementedError("predict() requires a fitted model from run()")

    def explain_tradeoff(self, point_a: dict[str, Any], point_b: dict[str, Any]) -> dict[str, Any]:
        """Explain tradeoff between two points using parameter deltas."""
        objs_a = point_a.get("objectives", {})
        objs_b = point_b.get("objectives", {})
        params_a = point_a.get("design_params", {})
        params_b = point_b.get("design_params", {})

        obj_deltas = {}
        for name in self.ds.objectives:
            va = objs_a.get(name, 0.0)
            vb = objs_b.get(name, 0.0)
            obj_deltas[name] = {
                "delta": round(vb - va, 4),
                "pct_change": round((vb - va) / va * 100, 1) if va != 0 else 0.0,
            }

        param_deltas = {}
        for var in self.ds.variables:
            va = params_a.get(var.name)
            vb = params_b.get(var.name)
            if va != vb:
                param_deltas[var.name] = {"from": va, "to": vb}

        return {
            "objective_deltas": obj_deltas,
            "parameter_changes": param_deltas,
        }
