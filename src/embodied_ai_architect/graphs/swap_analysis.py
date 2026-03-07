"""SWaP-C optimization methodology library.

Pure-function implementations of five interconnected optimization methodologies:
  M1 - Weighted Figure of Merit (FoM) scoring
  M2 - Sensitivity analysis (Tornado, Taguchi L18)
  M3 - Pareto-guided exploration (TOPSIS, MRS, clustering)
  M4 - Delta attribution & parametric sweep
  M5 - Monte Carlo budget feasibility

All analysis functions accept an ``evaluator_fn: Callable[[dict], dict]`` so
they decouple from the concrete BOM estimator and work with mock evaluators in
tests.

Usage:
    from embodied_ai_architect.graphs.swap_analysis import (
        compute_fom_score, tornado_analysis, topsis_rank,
        delta_attribution, monte_carlo_feasibility,
    )
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from pydantic import BaseModel, Field

from embodied_ai_architect.graphs.swap_profiles import OBJECTIVES

# ---------------------------------------------------------------------------
# M1: Weighted FoM Scoring
# ---------------------------------------------------------------------------


class FoMResult(BaseModel):
    """Result of a Figure-of-Merit score computation."""

    composite_score: float = Field(..., description="Weighted composite score (0-100)")
    per_objective_scores: dict[str, float] = Field(
        default_factory=dict, description="Per-objective normalized scores"
    )
    profile_name: str = Field(default="", description="Profile used for weighting")


def compute_fom_score(
    objectives: dict[str, float],
    weights: dict[str, float],
    ideal: dict[str, float] | None = None,
    anti_ideal: dict[str, float] | None = None,
    profile_name: str = "",
) -> FoMResult:
    """Compute weighted Figure-of-Merit score for a single design point.

    Normalization: score_i = 100 * (anti_ideal_i - value_i) / (anti_ideal_i - ideal_i)
    All objectives are "smaller is better" (minimized).

    Args:
        objectives: Objective name -> measured value.
        weights: Objective name -> weight (should sum to 1.0).
        ideal: Best-case values per objective. If None, uses 0 for all.
        anti_ideal: Worst-case values per objective. If None, uses 2x the value.
        profile_name: Optional label for the profile used.

    Returns:
        FoMResult with composite and per-objective scores.
    """
    obj_names = [k for k in OBJECTIVES if k in objectives]

    per_scores: dict[str, float] = {}
    for name in obj_names:
        val = objectives[name]
        best = ideal[name] if ideal else 0.0
        worst = anti_ideal[name] if anti_ideal else max(val * 2.0, 1e-9)
        span = worst - best
        if span <= 0:
            per_scores[name] = 100.0 if val <= best else 0.0
        else:
            raw = 100.0 * (worst - val) / span
            per_scores[name] = max(0.0, min(100.0, raw))

    composite = sum(weights.get(k, 0.0) * per_scores.get(k, 0.0) for k in obj_names)
    return FoMResult(
        composite_score=round(composite, 1),
        per_objective_scores={k: round(v, 1) for k, v in per_scores.items()},
        profile_name=profile_name,
    )


def rank_designs_by_fom(
    designs: list[dict[str, float]],
    weights: dict[str, float],
    profile_name: str = "",
) -> list[tuple[int, FoMResult]]:
    """Rank a list of design objective dicts by weighted FoM.

    Ideal and anti-ideal are derived from the design set (min/max per objective).

    Args:
        designs: List of objective dicts.
        weights: Objective name -> weight.
        profile_name: Optional profile label.

    Returns:
        List of (original_index, FoMResult) sorted by score descending.
    """
    if not designs:
        return []

    obj_names = [k for k in OBJECTIVES if k in designs[0]]
    ideal = {k: min(d[k] for d in designs) for k in obj_names}
    anti_ideal = {k: max(d[k] for d in designs) for k in obj_names}

    results = []
    for i, d in enumerate(designs):
        fom = compute_fom_score(d, weights, ideal, anti_ideal, profile_name)
        results.append((i, fom))

    results.sort(key=lambda x: x[1].composite_score, reverse=True)
    return results


# ---------------------------------------------------------------------------
# M2: Sensitivity Analysis
# ---------------------------------------------------------------------------


class TornadoBar(BaseModel):
    """One bar in a tornado chart."""

    variable_name: str
    objective_name: str
    base_value: float
    low_value: float
    high_value: float
    swing: float = Field(..., description="abs(high - low)")


class TornadoResult(BaseModel):
    """Tornado sensitivity analysis result."""

    bars: list[TornadoBar] = Field(default_factory=list)
    base_design: dict[str, Any] = Field(default_factory=dict)


class TaguchiResult(BaseModel):
    """Taguchi L18 screening result."""

    main_effects: dict[str, dict[str, float]] = Field(
        default_factory=dict, description="variable -> objective -> effect"
    )
    sn_ratios: dict[str, dict[str, float]] = Field(
        default_factory=dict, description="variable -> objective -> S/N ratio"
    )
    top_factors: list[str] = Field(
        default_factory=list, description="Variables ranked by total effect magnitude"
    )


def _discretize_variable(var: dict[str, Any]) -> list[Any]:
    """Discretize a design variable to 3 levels for Taguchi screening."""
    vtype = var.get("var_type", var.get("type", "continuous"))
    if vtype == "categorical":
        choices = var["choices"]
        if len(choices) <= 3:
            return list(choices)
        # Pick first, middle, last
        mid = len(choices) // 2
        return [choices[0], choices[mid], choices[-1]]
    lower = var.get("lower", var.get("low", 0))
    upper = var.get("upper", var.get("high", 1))
    mid = (lower + upper) / 2.0
    if vtype == "integer":
        return [int(lower), int(round(mid)), int(upper)]
    return [float(lower), float(mid), float(upper)]


# Taguchi L18(2^1 x 3^7) orthogonal array — standard 18-run design
# Columns represent factor levels (0-indexed: 0, 1, 2)
# We use columns 1-7 (the 3-level factors) for up to 7 variables.
_L18_ARRAY = [
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1],
    [0, 0, 2, 2, 2, 2, 2],
    [0, 1, 0, 0, 1, 1, 2],
    [0, 1, 1, 1, 2, 2, 0],
    [0, 1, 2, 2, 0, 0, 1],
    [0, 2, 0, 1, 0, 2, 1],
    [0, 2, 1, 2, 1, 0, 2],
    [0, 2, 2, 0, 2, 1, 0],
    [1, 0, 0, 2, 1, 2, 0],
    [1, 0, 1, 0, 2, 0, 1],
    [1, 0, 2, 1, 0, 1, 2],
    [1, 1, 0, 1, 2, 0, 2],
    [1, 1, 1, 2, 0, 1, 0],
    [1, 1, 2, 0, 1, 2, 1],
    [1, 2, 0, 2, 2, 1, 1],
    [1, 2, 1, 0, 0, 2, 2],
    [1, 2, 2, 1, 1, 0, 0],
]


def tornado_analysis(
    base_params: dict[str, Any],
    evaluator_fn: Callable[[dict[str, Any]], dict[str, float]],
    design_space_variables: list[dict[str, Any]],
    objectives: list[str] | None = None,
) -> TornadoResult:
    """One-at-a-time tornado sensitivity analysis.

    For each variable, evaluate at low and high bounds while holding others
    at base values. Categorical variables cycle through all choices.

    Args:
        base_params: Baseline parameter dict.
        evaluator_fn: Callable that takes params dict and returns objectives dict.
        design_space_variables: List of variable specs with name, var_type, lower/upper/choices.
        objectives: Objective names to analyze. Defaults to all 6 SWaP-C objectives.

    Returns:
        TornadoResult with bars sorted by swing magnitude.
    """
    objectives = objectives or OBJECTIVES
    base_objs = evaluator_fn(base_params)

    bars: list[TornadoBar] = []
    for var in design_space_variables:
        vname = var["name"]
        vtype = var.get("var_type", var.get("type", "continuous"))

        if vtype == "categorical":
            choices = var["choices"]
            obj_at_choices: list[dict[str, float]] = []
            for c in choices:
                p = {**base_params, vname: c}
                obj_at_choices.append(evaluator_fn(p))
            for obj_name in objectives:
                if obj_name not in base_objs:
                    continue
                vals = [o.get(obj_name, 0.0) for o in obj_at_choices]
                lo, hi = min(vals), max(vals)
                bars.append(
                    TornadoBar(
                        variable_name=vname,
                        objective_name=obj_name,
                        base_value=base_objs.get(obj_name, 0.0),
                        low_value=lo,
                        high_value=hi,
                        swing=abs(hi - lo),
                    )
                )
        else:
            lower = var.get("lower", var.get("low", 0))
            upper = var.get("upper", var.get("high", 1))
            lo_params = {**base_params, vname: lower}
            hi_params = {**base_params, vname: upper}
            if vtype == "integer":
                lo_params[vname] = int(lower)
                hi_params[vname] = int(upper)

            lo_objs = evaluator_fn(lo_params)
            hi_objs = evaluator_fn(hi_params)

            for obj_name in objectives:
                if obj_name not in base_objs:
                    continue
                lo_val = lo_objs.get(obj_name, 0.0)
                hi_val = hi_objs.get(obj_name, 0.0)
                bars.append(
                    TornadoBar(
                        variable_name=vname,
                        objective_name=obj_name,
                        base_value=base_objs.get(obj_name, 0.0),
                        low_value=min(lo_val, hi_val),
                        high_value=max(lo_val, hi_val),
                        swing=abs(hi_val - lo_val),
                    )
                )

    bars.sort(key=lambda b: b.swing, reverse=True)
    return TornadoResult(bars=bars, base_design=base_params)


def taguchi_l18_screening(
    evaluator_fn: Callable[[dict[str, Any]], dict[str, float]],
    design_space_variables: list[dict[str, Any]],
    base_params: dict[str, Any] | None = None,
    objectives: list[str] | None = None,
) -> TaguchiResult:
    """Taguchi L18 orthogonal-array screening.

    Uses 18 runs to screen up to 7 variables at 3 levels each.
    Computes main effects and S/N ratios (smaller-is-better).

    Args:
        evaluator_fn: Callable that takes params dict and returns objectives dict.
        design_space_variables: Variable specs (at most 7 used).
        base_params: Base parameter dict for non-screened variables.
        objectives: Objective names. Defaults to all 6 SWaP-C objectives.

    Returns:
        TaguchiResult with main effects, S/N ratios, and ranked factors.
    """
    objectives = objectives or OBJECTIVES
    base_params = base_params or {}
    vars_to_screen = design_space_variables[:7]

    # Discretize each variable to 3 levels
    levels_map: dict[str, list[Any]] = {}
    for var in vars_to_screen:
        levels_map[var["name"]] = _discretize_variable(var)

    # Run the 18 experiments
    results: list[dict[str, float]] = []
    for row in _L18_ARRAY:
        params = dict(base_params)
        for j, var in enumerate(vars_to_screen):
            level_idx = row[j] % len(levels_map[var["name"]])
            params[var["name"]] = levels_map[var["name"]][level_idx]
        results.append(evaluator_fn(params))

    # Compute main effects and S/N ratios
    main_effects: dict[str, dict[str, float]] = {}
    sn_ratios: dict[str, dict[str, float]] = {}

    for j, var in enumerate(vars_to_screen):
        vname = var["name"]
        n_levels = len(levels_map[vname])
        main_effects[vname] = {}
        sn_ratios[vname] = {}

        for obj_name in objectives:
            # Group results by level
            level_values: dict[int, list[float]] = {k: [] for k in range(n_levels)}
            for i, row in enumerate(_L18_ARRAY):
                level_idx = row[j] % n_levels
                val = results[i].get(obj_name, 0.0)
                level_values[level_idx].append(val)

            # Main effect = max level mean - min level mean
            level_means = [np.mean(level_values[k]) for k in range(n_levels) if level_values[k]]
            effect = float(max(level_means) - min(level_means)) if level_means else 0.0
            main_effects[vname][obj_name] = round(effect, 4)

            # S/N ratio: smaller-is-better = -10 * log10(mean(y^2))
            all_vals = [v for vs in level_values.values() for v in vs]
            if all_vals:
                mean_sq = np.mean([v**2 for v in all_vals])
                sn = float(-10.0 * np.log10(max(mean_sq, 1e-30)))
            else:
                sn = 0.0
            sn_ratios[vname][obj_name] = round(sn, 2)

    # Rank factors by total effect magnitude across all objectives
    total_effects = {}
    for vname in main_effects:
        total_effects[vname] = sum(abs(v) for v in main_effects[vname].values())
    top_factors = sorted(total_effects, key=total_effects.get, reverse=True)

    return TaguchiResult(
        main_effects=main_effects,
        sn_ratios=sn_ratios,
        top_factors=top_factors,
    )


# ---------------------------------------------------------------------------
# M3: Pareto-Guided Exploration (TOPSIS, MRS, Clustering)
# ---------------------------------------------------------------------------


class TOPSISResult(BaseModel):
    """TOPSIS multi-criteria ranking result."""

    rankings: list[dict[str, Any]] = Field(
        default_factory=list, description="Designs with topsis_score added"
    )
    ideal_point: dict[str, float] = Field(default_factory=dict)
    anti_ideal_point: dict[str, float] = Field(default_factory=dict)


class MRSResult(BaseModel):
    """Marginal Rate of Substitution at a knee point."""

    knee_index: int
    exchange_rates: dict[str, float] = Field(
        default_factory=dict,
        description="Objective pair -> exchange rate (e.g. weight_grams/power_watts -> rate)",
    )


class ClusterResult(BaseModel):
    """K-means clustering of a Pareto front."""

    n_clusters: int
    clusters: list[dict[str, Any]] = Field(
        default_factory=list, description="Cluster info: id, centroid, n_members, label"
    )
    labels: list[int] = Field(default_factory=list, description="Cluster label per design")


def topsis_rank(
    designs: list[dict[str, float]],
    weights: dict[str, float],
    objectives: list[str] | None = None,
) -> TOPSISResult:
    """TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution).

    Steps: vector normalization -> weighted matrix -> ideal/anti-ideal ->
    Euclidean distances -> closeness coefficient C_i = d-/(d+ + d-).

    All objectives assumed to be minimized.

    Args:
        designs: List of objective dicts.
        weights: Objective name -> weight.
        objectives: Subset of objectives to use. Defaults to all 6.

    Returns:
        TOPSISResult with ranked designs and ideal/anti-ideal points.
    """
    if not designs:
        return TOPSISResult()

    objectives = objectives or [k for k in OBJECTIVES if k in designs[0]]
    n = len(designs)
    m = len(objectives)

    # Build decision matrix
    X = np.zeros((n, m))
    for i, d in enumerate(designs):
        for j, obj in enumerate(objectives):
            X[i, j] = d.get(obj, 0.0)

    # Vector normalization
    norms = np.sqrt(np.sum(X**2, axis=0))
    norms = np.where(norms == 0, 1.0, norms)
    R = X / norms

    # Weighted normalized matrix
    W = np.array([weights.get(obj, 1.0 / m) for obj in objectives])
    V = R * W

    # Ideal (min for all since all minimized) and anti-ideal (max)
    ideal = V.min(axis=0)
    anti_ideal = V.max(axis=0)

    # Distances
    d_plus = np.sqrt(np.sum((V - ideal) ** 2, axis=1))
    d_minus = np.sqrt(np.sum((V - anti_ideal) ** 2, axis=1))

    # Closeness coefficient
    denom = d_plus + d_minus
    denom = np.where(denom == 0, 1.0, denom)
    C = d_minus / denom

    # Build ranked results
    ranked_indices = np.argsort(-C)  # descending
    rankings = []
    for idx in ranked_indices:
        entry = dict(designs[int(idx)])
        entry["topsis_score"] = round(float(C[idx]), 4)
        entry["_original_index"] = int(idx)
        rankings.append(entry)

    ideal_point = {obj: float(X[:, j].min()) for j, obj in enumerate(objectives)}
    anti_ideal_point = {obj: float(X[:, j].max()) for j, obj in enumerate(objectives)}

    return TOPSISResult(
        rankings=rankings,
        ideal_point=ideal_point,
        anti_ideal_point=anti_ideal_point,
    )


def marginal_rate_of_substitution(
    pareto_front: list[dict[str, float]],
    knee_index: int,
    objectives: list[str] | None = None,
) -> MRSResult:
    """Compute exchange rates at a knee point on the Pareto front.

    For each pair of objectives, computes the finite-difference rate of
    exchange to the nearest neighbors on the front.

    Args:
        pareto_front: List of objective dicts on the Pareto front.
        knee_index: Index of the knee point.
        objectives: Objectives to consider.

    Returns:
        MRSResult with exchange rates like "1W saved costs Xg of weight".
    """
    objectives = objectives or [k for k in OBJECTIVES if k in pareto_front[0]]
    knee = pareto_front[knee_index]

    exchange_rates: dict[str, float] = {}
    for neighbor_idx in range(len(pareto_front)):
        if neighbor_idx == knee_index:
            continue
        neighbor = pareto_front[neighbor_idx]
        for i, obj_a in enumerate(objectives):
            for j, obj_b in enumerate(objectives):
                if i >= j:
                    continue
                da = neighbor.get(obj_a, 0.0) - knee.get(obj_a, 0.0)
                db = neighbor.get(obj_b, 0.0) - knee.get(obj_b, 0.0)
                if abs(da) > 1e-9:
                    key = f"{obj_b}/{obj_a}"
                    rate = db / da
                    # Keep the exchange rate with smallest absolute value
                    # (nearest neighbor gives most local gradient)
                    if key not in exchange_rates or abs(rate) < abs(exchange_rates[key]):
                        exchange_rates[key] = round(rate, 4)

    return MRSResult(knee_index=knee_index, exchange_rates=exchange_rates)


def cluster_pareto_front(
    pareto_front: list[dict[str, float]],
    n_clusters: int = 4,
    objectives: list[str] | None = None,
    max_iter: int = 50,
    seed: int | None = None,
) -> ClusterResult:
    """K-means clustering of Pareto-front designs in normalized objective space.

    Simple numpy implementation (~30 lines). Labels clusters by their dominant
    characteristic (lowest-value objective relative to centroid).

    Args:
        pareto_front: List of objective dicts.
        n_clusters: Number of clusters.
        objectives: Objectives to cluster on.
        max_iter: Maximum k-means iterations.
        seed: Random seed.

    Returns:
        ClusterResult with cluster assignments and labels.
    """
    if not pareto_front:
        return ClusterResult(n_clusters=0, clusters=[], labels=[])

    objectives = objectives or [k for k in OBJECTIVES if k in pareto_front[0]]
    n = len(pareto_front)
    m = len(objectives)
    n_clusters = min(n_clusters, n)

    # Build and normalize matrix
    X = np.zeros((n, m))
    for i, d in enumerate(pareto_front):
        for j, obj in enumerate(objectives):
            X[i, j] = d.get(obj, 0.0)

    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    spans = maxs - mins
    spans = np.where(spans == 0, 1.0, spans)
    Xn = (X - mins) / spans

    # K-means
    rng = np.random.default_rng(seed)
    indices = rng.choice(n, size=n_clusters, replace=False)
    centroids = Xn[indices].copy()

    labels = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        # Assign
        dists = np.zeros((n, n_clusters))
        for k in range(n_clusters):
            dists[:, k] = np.sum((Xn - centroids[k]) ** 2, axis=1)
        new_labels = np.argmin(dists, axis=1)

        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

        # Update centroids
        for k in range(n_clusters):
            members = Xn[labels == k]
            if len(members) > 0:
                centroids[k] = members.mean(axis=0)

    # Build cluster info with labels
    clusters = []
    for k in range(n_clusters):
        members = Xn[labels == k]
        if len(members) == 0:
            continue
        centroid = centroids[k]
        # Label by the objective where this cluster has the lowest normalized value
        min_obj_idx = int(np.argmin(centroid))
        label = f"low-{objectives[min_obj_idx].replace('_', '-')}"
        clusters.append(
            {
                "id": k,
                "centroid": {obj: round(float(centroid[j]), 4) for j, obj in enumerate(objectives)},
                "n_members": int(np.sum(labels == k)),
                "label": label,
            }
        )

    return ClusterResult(
        n_clusters=n_clusters,
        clusters=clusters,
        labels=[int(lbl) for lbl in labels],
    )


# ---------------------------------------------------------------------------
# M4: Delta Attribution & Parametric Sweep
# ---------------------------------------------------------------------------


class DeltaAttribution(BaseModel):
    """Component-level delta between two BOM summaries."""

    component_deltas: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Per-component deltas: name, weight_delta, volume_delta, cost_delta",
    )
    total_deltas: dict[str, float] = Field(
        default_factory=dict, description="Aggregate weight/volume/cost deltas"
    )


class SweepResult(BaseModel):
    """Result of a single-variable parametric sweep."""

    variable_name: str
    sweep_values: list[Any] = Field(default_factory=list)
    objective_traces: dict[str, list[float]] = Field(
        default_factory=dict, description="Objective name -> list of values at each sweep step"
    )


def delta_attribution(
    left_bom_summary: list[dict[str, Any]],
    right_bom_summary: list[dict[str, Any]],
) -> DeltaAttribution:
    """Attribute differences between two BOM summaries to individual components.

    Matches components by name. Computes per-component weight, volume, and cost
    deltas (right - left). Identifies the largest contributor.

    Args:
        left_bom_summary: BOM summary table (list of dicts with name, weight_grams, etc.)
        right_bom_summary: Second BOM summary table.

    Returns:
        DeltaAttribution with per-component and total deltas.
    """
    left_by_name = {c["name"]: c for c in left_bom_summary}
    right_by_name = {c["name"]: c for c in right_bom_summary}
    all_names = list(dict.fromkeys(list(left_by_name.keys()) + list(right_by_name.keys())))

    component_deltas = []
    total_w, total_v, total_c = 0.0, 0.0, 0.0

    for name in all_names:
        lc = left_by_name.get(name, {})
        rc = right_by_name.get(name, {})
        dw = rc.get("weight_grams", 0.0) - lc.get("weight_grams", 0.0)
        dv = rc.get("volume_cm3", 0.0) - lc.get("volume_cm3", 0.0)
        dc = rc.get("cost_usd", 0.0) - lc.get("cost_usd", 0.0)
        component_deltas.append(
            {
                "name": name,
                "weight_delta": round(dw, 4),
                "volume_delta": round(dv, 4),
                "cost_delta": round(dc, 4),
            }
        )
        total_w += dw
        total_v += dv
        total_c += dc

    return DeltaAttribution(
        component_deltas=component_deltas,
        total_deltas={
            "weight_grams": round(total_w, 4),
            "volume_cm3": round(total_v, 4),
            "cost_usd": round(total_c, 4),
        },
    )


def parametric_sweep(
    base_params: dict[str, Any],
    variable_name: str,
    sweep_values: list[Any],
    evaluator_fn: Callable[[dict[str, Any]], dict[str, float]],
    objectives: list[str] | None = None,
) -> SweepResult:
    """Sweep one parameter across a range, holding others fixed.

    Args:
        base_params: Baseline parameter dict.
        variable_name: Name of the variable to sweep.
        sweep_values: Values to evaluate at.
        evaluator_fn: Callable that takes params dict and returns objectives dict.
        objectives: Objectives to track. Defaults to all 6.

    Returns:
        SweepResult with objective traces.
    """
    objectives = objectives or OBJECTIVES

    traces: dict[str, list[float]] = {obj: [] for obj in objectives}
    for val in sweep_values:
        params = {**base_params, variable_name: val}
        objs = evaluator_fn(params)
        for obj in objectives:
            traces[obj].append(round(objs.get(obj, 0.0), 4))

    return SweepResult(
        variable_name=variable_name,
        sweep_values=list(sweep_values),
        objective_traces=traces,
    )


# ---------------------------------------------------------------------------
# M5: Monte Carlo Budget Feasibility
# ---------------------------------------------------------------------------


class MonteCarloResult(BaseModel):
    """Monte Carlo budget feasibility analysis result."""

    n_samples: int
    p10: dict[str, float] = Field(default_factory=dict)
    p50: dict[str, float] = Field(default_factory=dict)
    p90: dict[str, float] = Field(default_factory=dict)
    mean: dict[str, float] = Field(default_factory=dict)
    std: dict[str, float] = Field(default_factory=dict)
    feasibility_prob: dict[str, float] = Field(
        default_factory=dict, description="Metric -> P(within budget)"
    )
    overall_feasibility: float = Field(default=0.0, description="P(all budgets met simultaneously)")
    traffic_light: dict[str, str] = Field(
        default_factory=dict, description="Metric -> green/yellow/red"
    )


# Continuous parameters and their relative uncertainty (1-sigma as fraction)
_UNCERTAINTY_MODEL: dict[str, float] = {
    "area": 0.05,  # +-5% die area
    "power": 0.15,  # +-15% TDP
    "ambient_temp": 0.05,  # +-5% ambient
}


def monte_carlo_feasibility(
    base_params: dict[str, Any],
    budgets: dict[str, float],
    evaluator_fn: Callable[[dict[str, Any]], dict[str, float]],
    n_samples: int = 1000,
    seed: int | None = None,
    objectives: list[str] | None = None,
) -> MonteCarloResult:
    """Monte Carlo simulation for probabilistic budget feasibility.

    Perturbs continuous BOM inputs using normal distributions. Categorical
    variables (package, cooling) are not perturbed.

    Args:
        base_params: Baseline parameter dict.
        budgets: Metric name -> maximum allowed value (e.g. weight_grams -> 200).
        evaluator_fn: Callable that takes params dict and returns objectives dict.
        n_samples: Number of Monte Carlo samples.
        seed: Random seed for reproducibility.
        objectives: Objectives to track. Defaults to all keys in budgets + OBJECTIVES.

    Returns:
        MonteCarloResult with percentiles, feasibility, and traffic lights.
    """
    rng = np.random.default_rng(seed)
    obj_set = objectives or [k for k in OBJECTIVES]

    # Collect all samples
    all_results: list[dict[str, float]] = []
    for _ in range(n_samples):
        params = dict(base_params)
        for param_name, sigma_frac in _UNCERTAINTY_MODEL.items():
            if param_name in params:
                nominal = params[param_name]
                perturbed = rng.normal(nominal, abs(nominal) * sigma_frac)
                # Ensure positive for physical quantities
                params[param_name] = max(perturbed, nominal * 0.1)
        all_results.append(evaluator_fn(params))

    # Compute statistics
    p10, p50, p90 = {}, {}, {}
    mean_d, std_d = {}, {}
    feasibility_prob: dict[str, float] = {}
    all_feasible = np.ones(n_samples, dtype=bool)

    for obj in obj_set:
        vals = np.array([r.get(obj, 0.0) for r in all_results])
        p10[obj] = round(float(np.percentile(vals, 10)), 4)
        p50[obj] = round(float(np.percentile(vals, 50)), 4)
        p90[obj] = round(float(np.percentile(vals, 90)), 4)
        mean_d[obj] = round(float(np.mean(vals)), 4)
        std_d[obj] = round(float(np.std(vals)), 4)

        if obj in budgets:
            within = vals <= budgets[obj]
            prob = float(np.mean(within))
            feasibility_prob[obj] = round(prob, 4)
            all_feasible &= within

    overall = float(np.mean(all_feasible)) if budgets else 1.0

    # Traffic lights
    traffic_light: dict[str, str] = {}
    for obj, prob in feasibility_prob.items():
        if prob >= 0.90:
            traffic_light[obj] = "green"
        elif prob >= 0.50:
            traffic_light[obj] = "yellow"
        else:
            traffic_light[obj] = "red"

    return MonteCarloResult(
        n_samples=n_samples,
        p10=p10,
        p50=p50,
        p90=p90,
        mean=mean_d,
        std=std_d,
        feasibility_prob=feasibility_prob,
        overall_feasibility=round(overall, 4),
        traffic_light=traffic_light,
    )
