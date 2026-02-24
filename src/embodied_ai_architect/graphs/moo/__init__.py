"""Multi-Objective Optimization engine for SoC design space exploration.

Three-layer pipeline:
    Layer 1: MAP-Elites — fast design space illumination (~seconds)
    Layer 2: Bayesian BO — refined Pareto front with qNEHVI (requires botorch)
    Layer 3: NSGA-III — many-objective fallback for >4 objectives (requires pymoo)

Usage:
    from embodied_ai_architect.graphs.moo import (
        DesignSpace, DesignVariable, DesignEvaluator, OptimizationEngine,
        OptimizationResult, create_soc_design_space,
    )
"""

from embodied_ai_architect.graphs.moo.design_space import (
    DesignSpace,
    DesignVariable,
    create_soc_design_space,
)
from embodied_ai_architect.graphs.moo.evaluator import (
    DesignEvaluator,
    EvaluationResult,
)
from embodied_ai_architect.graphs.moo.engine import (
    OptimizationConfig,
    OptimizationEngine,
    OptimizationResult,
)

__all__ = [
    "DesignSpace",
    "DesignVariable",
    "DesignEvaluator",
    "EvaluationResult",
    "OptimizationConfig",
    "OptimizationEngine",
    "OptimizationResult",
    "create_soc_design_space",
]
