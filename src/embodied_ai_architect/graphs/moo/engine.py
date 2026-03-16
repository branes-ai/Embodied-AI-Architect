"""Orchestrates the 3-layer MOO pipeline.

Auto-selects layers based on objective count:
    <=4 objectives: MAP-Elites -> Bayesian BO (qNEHVI)
    >4 objectives:  MAP-Elites -> NSGA-III

Usage:
    from embodied_ai_architect.graphs.moo.engine import OptimizationEngine

    engine = OptimizationEngine(design_space, evaluator, config)
    result = engine.run()
"""

from __future__ import annotations

import logging
import math
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field

from embodied_ai_architect.graphs.moo.design_space import DesignSpace
from embodied_ai_architect.graphs.moo.evaluator import DesignEvaluator
from embodied_ai_architect.graphs.moo.executor import LocalThreadExecutor
from embodied_ai_architect.graphs.moo.map_elites import (
    MAPElites,
    MAPElitesConfig,
    MAPElitesResult,
)

logger = logging.getLogger(__name__)


class OptimizationConfig(BaseModel):
    """Configuration for the optimization engine."""

    layers: str = Field(
        default="auto",
        description="Layer selection: 'auto', 'map_elites', 'bayesian', 'nsga3', 'map_elites+bayesian', 'map_elites+nsga3'",
    )
    map_elites: MAPElitesConfig = Field(default_factory=MAPElitesConfig)
    bayesian_n_initial: int = Field(default=20, description="BO initial samples")
    bayesian_n_iterations: int = Field(default=50, description="BO iterations")
    bayesian_batch_size: int = Field(default=4, description="BO batch size")
    nsga3_pop_size: int = Field(default=100, description="NSGA-III population size")
    nsga3_n_generations: int = Field(default=100, description="NSGA-III generations")
    max_workers: int = Field(default=8, description="Thread pool size")
    backend: str = Field(default="local", description="Evaluation backend: 'local' or 'kubernetes'")


class OptimizationResult(BaseModel):
    """Complete result from the optimization pipeline."""

    pareto_front: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Non-dominated design points",
    )
    hypervolume: float = Field(default=0.0, description="Hypervolume indicator")
    knee_point: Optional[dict[str, Any]] = Field(
        default=None,
        description="Knee point on the Pareto front",
    )
    sensitivity: dict[str, dict[str, float]] = Field(
        default_factory=dict,
        description="Parameter sensitivity per objective",
    )
    atlas: dict[str, Any] = Field(
        default_factory=dict,
        description="MAP-Elites atlas summary",
    )
    convergence_history: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Per-iteration convergence data",
    )
    total_evaluations: int = 0
    layers_used: list[str] = Field(default_factory=list)

    # Backward-compatible with existing pareto_results format
    def to_pareto_results(self) -> dict[str, Any]:
        """Convert to the format expected by existing pareto_results consumers."""
        front_dicts = []
        for p in self.pareto_front:
            front_dicts.append(
                {
                    "hardware_name": f"custom-{p.get('objectives', {}).get('process_nm', '?')}nm",
                    "power": p.get("objectives", {}).get("power_watts", 0),
                    "latency": p.get("objectives", {}).get("latency_ms", 0),
                    "cost": p.get("objectives", {}).get("cost_usd", 0),
                    "dominated": False,
                    "knee_point": p == self.knee_point if self.knee_point else False,
                    "metadata": p,
                }
            )
        return {
            "front": front_dicts,
            "knee_point": front_dicts[0] if front_dicts else None,
            "total": len(front_dicts),
            "non_dominated_count": len(front_dicts),
            "hypervolume": self.hypervolume,
        }


class OptimizationEngine:
    """Orchestrates the multi-layer optimization pipeline.

    Auto-selects layers based on objective count:
    - <=4 objectives: MAP-Elites -> Bayesian BO
    - >4 objectives:  MAP-Elites -> NSGA-III
    """

    def __init__(
        self,
        design_space: DesignSpace,
        evaluator: DesignEvaluator,
        config: OptimizationConfig | None = None,
    ):
        self.ds = design_space
        self.evaluator = evaluator
        self.config = config or OptimizationConfig()
        self._executor = LocalThreadExecutor(evaluator, max_workers=self.config.max_workers)
        self._result: OptimizationResult | None = None

    def run(
        self,
        callback: Callable[[str, int, int, float], None] | None = None,
    ) -> OptimizationResult:
        """Execute the optimization pipeline.

        Args:
            callback: Optional (layer_name, iteration, total_evals, metric) callback.

        Returns:
            OptimizationResult with Pareto front and analysis.
        """
        layers = self._select_layers()
        total_evals = 0
        convergence = []
        layers_used = []

        # Layer 1: MAP-Elites (always runs first)
        me_result: MAPElitesResult | None = None
        if "map_elites" in layers:
            layers_used.append("map_elites")
            logger.info("Starting Layer 1: MAP-Elites")

            def me_callback(iteration: int, evals: int, coverage: float) -> None:
                if callback:
                    callback("map_elites", iteration, evals, coverage)
                convergence.append(
                    {
                        "layer": "map_elites",
                        "iteration": iteration,
                        "evaluations": evals,
                        "coverage": coverage,
                    }
                )

            me = MAPElites(self.ds, self.evaluator, self._executor, self.config.map_elites)
            me_result = me.run(callback=me_callback)
            total_evals += me_result.total_evaluations
            logger.info(
                "MAP-Elites complete: %d/%d cells filled (%.1f%% coverage)",
                me_result.filled_cells,
                me_result.total_cells,
                me_result.coverage * 100,
            )

        # Layer 2: Bayesian Optimization (optional, requires botorch)
        bo_result = None
        if "bayesian" in layers:
            layers_used.append("bayesian")
            bo_result = self._run_bayesian(me_result, callback, convergence)
            if bo_result:
                total_evals += bo_result.get("total_evaluations", 0)

        # Layer 3: NSGA-III (optional, requires pymoo)
        nsga_result = None
        if "nsga3" in layers:
            layers_used.append("nsga3")
            nsga_result = self._run_nsga3(me_result, callback, convergence)
            if nsga_result:
                total_evals += nsga_result.get("total_evaluations", 0)

        # Build final result
        self._result = self._build_result(
            me_result, bo_result, nsga_result, total_evals, convergence, layers_used
        )
        return self._result

    def _select_layers(self) -> list[str]:
        """Select which layers to run based on config and objectives."""
        cfg = self.config.layers

        if cfg == "auto":
            layers = ["map_elites"]
            if self.ds.n_obj <= 4:
                # Try BO, fall back to NSGA-III
                try:
                    import botorch  # noqa: F401

                    layers.append("bayesian")
                except ImportError:
                    try:
                        import pymoo  # noqa: F401

                        layers.append("nsga3")
                    except ImportError:
                        pass  # MAP-Elites only
            else:
                try:
                    import pymoo  # noqa: F401

                    layers.append("nsga3")
                except ImportError:
                    pass
            return layers

        if cfg == "map_elites":
            return ["map_elites"]
        if cfg == "bayesian":
            return ["bayesian"]
        if cfg == "nsga3":
            return ["nsga3"]
        if cfg == "map_elites+bayesian":
            return ["map_elites", "bayesian"]
        if cfg == "map_elites+nsga3":
            return ["map_elites", "nsga3"]

        return ["map_elites"]

    def _run_bayesian(
        self,
        me_result: MAPElitesResult | None,
        callback: Callable | None,
        convergence: list,
    ) -> dict[str, Any] | None:
        """Run Layer 2: Bayesian Optimization with qNEHVI."""
        try:
            from embodied_ai_architect.graphs.moo.bayesian_opt import (
                BayesianMOO,
                BayesianOptConfig,
            )

            cfg = BayesianOptConfig(
                n_initial=self.config.bayesian_n_initial,
                n_iterations=self.config.bayesian_n_iterations,
                batch_size=self.config.bayesian_batch_size,
            )

            def bo_callback(iteration: int, evals: int, hv: float) -> None:
                if callback:
                    callback("bayesian", iteration, evals, hv)
                convergence.append(
                    {
                        "layer": "bayesian",
                        "iteration": iteration,
                        "evaluations": evals,
                        "hypervolume": hv,
                    }
                )

            bo = BayesianMOO(self.ds, self.evaluator, self._executor, cfg)

            # Warm-start from MAP-Elites
            seed_points = None
            if me_result and me_result.best_designs:
                seed_points = [d["design_params"] for d in me_result.best_designs[: cfg.n_initial]]

            result = bo.run(seed_points=seed_points, callback=bo_callback)
            return result.model_dump()

        except ImportError:
            logger.info("BoTorch not available, skipping Bayesian optimization")
            return None
        except Exception as e:
            logger.warning("Bayesian optimization failed: %s", e)
            return None

    def _run_nsga3(
        self,
        me_result: MAPElitesResult | None,
        callback: Callable | None,
        convergence: list,
    ) -> dict[str, Any] | None:
        """Run Layer 3: NSGA-III for many-objective optimization."""
        try:
            from embodied_ai_architect.graphs.moo.nsga3 import (
                NSGA3MOO,
                NSGA3Config,
            )

            cfg = NSGA3Config(
                pop_size=self.config.nsga3_pop_size,
                n_generations=self.config.nsga3_n_generations,
            )

            def nsga_callback(generation: int, evals: int, hv: float) -> None:
                if callback:
                    callback("nsga3", generation, evals, hv)
                convergence.append(
                    {
                        "layer": "nsga3",
                        "generation": generation,
                        "evaluations": evals,
                        "hypervolume": hv,
                    }
                )

            nsga = NSGA3MOO(self.ds, self.evaluator, cfg)

            # Warm-start from MAP-Elites
            seed_points = None
            if me_result and me_result.best_designs:
                seed_points = [d["design_params"] for d in me_result.best_designs]

            result = nsga.run(seed_points=seed_points, callback=nsga_callback)
            return result.model_dump()

        except ImportError:
            logger.info("pymoo not available, skipping NSGA-III")
            return None
        except Exception as e:
            logger.warning("NSGA-III optimization failed: %s", e)
            return None

    def _build_result(
        self,
        me_result: MAPElitesResult | None,
        bo_result: dict[str, Any] | None,
        nsga_result: dict[str, Any] | None,
        total_evals: int,
        convergence: list,
        layers_used: list[str],
    ) -> OptimizationResult:
        """Combine results from all layers into a unified result."""
        # Collect all Pareto-front candidates
        all_points: list[dict[str, Any]] = []

        if me_result:
            for design in me_result.best_designs:
                if design.get("feasible", True):
                    all_points.append(design)

        if bo_result:
            for point in bo_result.get("pareto_front", []):
                all_points.append(point)

        if nsga_result:
            for point in nsga_result.get("pareto_front", []):
                all_points.append(point)

        # Compute non-dominated front from all collected points
        pareto_front = self._compute_pareto_front(all_points)

        # Compute hypervolume
        hv = self._compute_hypervolume(pareto_front)

        # Find knee point
        knee = self._find_knee_point(pareto_front)

        # Build sensitivity from BO if available
        sensitivity: dict[str, dict[str, float]] = {}
        if bo_result and bo_result.get("sensitivity"):
            sensitivity = bo_result["sensitivity"]

        # Atlas from MAP-Elites
        atlas: dict[str, Any] = {}
        if me_result:
            atlas = {
                "coverage": me_result.coverage,
                "filled_cells": me_result.filled_cells,
                "total_cells": me_result.total_cells,
                "best_per_objective": me_result.best_per_objective,
            }

        return OptimizationResult(
            pareto_front=pareto_front,
            hypervolume=hv,
            knee_point=knee,
            sensitivity=sensitivity,
            atlas=atlas,
            convergence_history=convergence,
            total_evaluations=total_evals,
            layers_used=layers_used,
        )

    def _compute_pareto_front(self, points: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Compute non-dominated front from a list of design points.

        Handles mixed MINIMIZE/MAXIMIZE objectives correctly by normalizing
        all objectives to a "lower is better" convention before dominance check.
        """
        from embodied_ai_architect.graphs.moo.design_space import ObjectiveDirection

        if not points:
            return []

        obj_names = self.ds.objectives
        n = len(points)

        # Build sign vector: +1 for MINIMIZE, -1 for MAXIMIZE
        signs = []
        for i, name in enumerate(obj_names):
            direction = (
                self.ds.directions[i]
                if i < len(self.ds.directions)
                else ObjectiveDirection.MINIMIZE
            )
            signs.append(1.0 if direction == ObjectiveDirection.MINIMIZE else -1.0)

        # Normalize to "lower is better" convention
        vectors = []
        for p in points:
            objs = p.get("objectives", {})
            vec = [signs[k] * objs.get(name, 1e6) for k, name in enumerate(obj_names)]
            vectors.append(vec)

        # Non-dominated sorting (all objectives now in MINIMIZE convention)
        dominated = [False] * n
        for i in range(n):
            if dominated[i]:
                continue
            for j in range(n):
                if i == j or dominated[j]:
                    continue
                vi, vj = vectors[i], vectors[j]
                all_leq = all(vj[k] <= vi[k] for k in range(len(obj_names)))
                any_lt = any(vj[k] < vi[k] for k in range(len(obj_names)))
                if all_leq and any_lt:
                    dominated[i] = True
                    break

        return [p for p, d in zip(points, dominated) if not d]

    def _compute_hypervolume(self, front: list[dict[str, Any]]) -> float:
        """Compute hypervolume indicator for the Pareto front.

        Uses a simple reference-point based approximation.
        """
        if not front:
            return 0.0

        obj_names = self.ds.objectives
        n_obj = len(obj_names)

        # Reference point: worst value per objective * 1.1
        ref = []
        for name in obj_names:
            vals = [p.get("objectives", {}).get(name, 0.0) for p in front]
            ref.append(max(vals) * 1.1 if vals else 1.0)

        if n_obj == 2:
            # Exact 2D hypervolume
            points_2d = []
            for p in front:
                objs = p.get("objectives", {})
                points_2d.append((objs.get(obj_names[0], ref[0]), objs.get(obj_names[1], ref[1])))
            points_2d.sort(key=lambda x: x[0])

            hv = 0.0
            prev_y = ref[1]
            for x, y in points_2d:
                if y < prev_y:
                    hv += (ref[0] - x) * (prev_y - y)
                    prev_y = y
            return round(hv, 4)

        # For >2 objectives, use product of ranges as approximation
        hv = 1.0
        for i, name in enumerate(obj_names):
            vals = [p.get("objectives", {}).get(name, 0.0) for p in front]
            hv *= (ref[i] - min(vals)) if vals else 0.0
        return round(hv, 4)

    def _find_knee_point(self, front: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Find the knee point on the Pareto front."""
        if not front:
            return None

        obj_names = self.ds.objectives

        # Normalize objectives
        ranges = {}
        mins = {}
        for name in obj_names:
            vals = [p.get("objectives", {}).get(name, 0.0) for p in front]
            mn, mx = min(vals), max(vals)
            ranges[name] = mx - mn if mx > mn else 1.0
            mins[name] = mn

        # Find point closest to utopia
        best_dist = float("inf")
        best_point = None
        for p in front:
            objs = p.get("objectives", {})
            dist = sum(
                ((objs.get(name, 0.0) - mins[name]) / ranges[name]) ** 2 for name in obj_names
            )
            dist = math.sqrt(dist)
            if dist < best_dist:
                best_dist = dist
                best_point = p

        return best_point

    def get_pareto_front(self) -> list[dict[str, Any]]:
        """Get the current Pareto front (after run())."""
        if self._result:
            return self._result.pareto_front
        return []

    def get_sensitivity(self) -> dict[str, dict[str, float]]:
        """Get parameter sensitivity (after run(), requires BO layer)."""
        if self._result:
            return self._result.sensitivity
        return {}

    def explain_tradeoff(self, point_a: dict[str, Any], point_b: dict[str, Any]) -> dict[str, Any]:
        """Explain the tradeoff between two design points."""
        objs_a = point_a.get("objectives", {})
        objs_b = point_b.get("objectives", {})
        params_a = point_a.get("design_params", {})
        params_b = point_b.get("design_params", {})

        obj_deltas = {}
        for name in self.ds.objectives:
            va = objs_a.get(name, 0.0)
            vb = objs_b.get(name, 0.0)
            obj_deltas[name] = {
                "point_a": va,
                "point_b": vb,
                "delta": round(vb - va, 4),
                "pct_change": round((vb - va) / va * 100, 1) if va != 0 else 0.0,
            }

        param_deltas = {}
        for var in self.ds.variables:
            va = params_a.get(var.name)
            vb = params_b.get(var.name)
            param_deltas[var.name] = {
                "point_a": va,
                "point_b": vb,
                "changed": va != vb,
            }

        return {
            "objective_deltas": obj_deltas,
            "parameter_deltas": param_deltas,
            "summary": _summarize_tradeoff(obj_deltas),
        }

    def shutdown(self) -> None:
        """Release executor resources."""
        self._executor.shutdown()


def _summarize_tradeoff(obj_deltas: dict[str, dict]) -> str:
    """Generate human-readable tradeoff summary."""
    improved = []
    degraded = []
    for name, d in obj_deltas.items():
        pct = d["pct_change"]
        if pct < -1:
            improved.append(f"{name} improves {abs(pct):.0f}%")
        elif pct > 1:
            degraded.append(f"{name} worsens {pct:.0f}%")

    parts = []
    if improved:
        parts.append("Gains: " + ", ".join(improved))
    if degraded:
        parts.append("Costs: " + ", ".join(degraded))
    return "; ".join(parts) if parts else "Negligible difference"
