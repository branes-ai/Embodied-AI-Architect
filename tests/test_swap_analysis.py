"""Tests for SWaP-C optimization analysis library."""

from __future__ import annotations

import numpy as np

from embodied_ai_architect.graphs.swap_analysis import (
    ClusterResult,
    DeltaAttribution,
    MonteCarloResult,
    SweepResult,
    TaguchiResult,
    TOPSISResult,
    TornadoResult,
    cluster_pareto_front,
    compute_fom_score,
    delta_attribution,
    marginal_rate_of_substitution,
    monte_carlo_feasibility,
    parametric_sweep,
    rank_designs_by_fom,
    taguchi_l18_screening,
    topsis_rank,
    tornado_analysis,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

OBJECTIVES = ["power_watts", "latency_ms", "area_mm2", "cost_usd", "weight_grams", "volume_cm3"]


def _mock_evaluator(params: dict) -> dict[str, float]:
    """Simple deterministic mock: objectives are linear functions of area and power."""
    area = params.get("area", 50.0)
    power = params.get("power", 5.0)
    return {
        "power_watts": power,
        "latency_ms": 100.0 / max(power, 0.1),
        "area_mm2": area,
        "cost_usd": area * 0.5 + power * 10,
        "weight_grams": area * 0.3 + power * 5,
        "volume_cm3": area * 0.1 + power * 2,
    }


def _make_design(power, weight, cost, **kwargs):
    """Shorthand for building an objective dict."""
    d = {
        "power_watts": power,
        "latency_ms": 10.0,
        "area_mm2": 50.0,
        "cost_usd": cost,
        "weight_grams": weight,
        "volume_cm3": 30.0,
    }
    d.update(kwargs)
    return d


_DS_VARS = [
    {"name": "area", "var_type": "continuous", "lower": 10.0, "upper": 200.0},
    {"name": "power", "var_type": "continuous", "lower": 1.0, "upper": 50.0},
    {"name": "layer_count", "var_type": "integer", "lower": 2, "upper": 8},
]


# ---------------------------------------------------------------------------
# M1: FoM Scoring
# ---------------------------------------------------------------------------


class TestFoMScoring:
    def test_perfect_score(self):
        """Design at ideal gets 100."""
        ideal = {k: 0.0 for k in OBJECTIVES}
        anti_ideal = {k: 100.0 for k in OBJECTIVES}
        weights = {k: 1.0 / 6 for k in OBJECTIVES}
        result = compute_fom_score(ideal, weights, ideal, anti_ideal)
        assert result.composite_score == 100.0

    def test_worst_score(self):
        """Design at anti-ideal gets 0."""
        ideal = {k: 0.0 for k in OBJECTIVES}
        anti_ideal = {k: 100.0 for k in OBJECTIVES}
        weights = {k: 1.0 / 6 for k in OBJECTIVES}
        result = compute_fom_score(anti_ideal, weights, ideal, anti_ideal)
        assert result.composite_score == 0.0

    def test_weighted_ranking(self):
        """Higher weight on lower-value objectives increases score."""
        design = {k: 50.0 for k in OBJECTIVES}
        design["power_watts"] = 10.0  # much better on power
        ideal = {k: 0.0 for k in OBJECTIVES}
        anti_ideal = {k: 100.0 for k in OBJECTIVES}

        # Equal weights
        w_equal = {k: 1.0 / 6 for k in OBJECTIVES}
        s_equal = compute_fom_score(design, w_equal, ideal, anti_ideal)

        # Weight heavily on power
        w_power = {k: 0.02 for k in OBJECTIVES}
        w_power["power_watts"] = 0.90
        s_power = compute_fom_score(design, w_power, ideal, anti_ideal)

        assert s_power.composite_score > s_equal.composite_score

    def test_rank_designs(self):
        """Rank correctly orders designs by score."""
        designs = [
            _make_design(10, 100, 50),  # moderate
            _make_design(1, 10, 5),  # best
            _make_design(50, 500, 200),  # worst
        ]
        weights = {k: 1.0 / 6 for k in OBJECTIVES}
        ranked = rank_designs_by_fom(designs, weights)
        assert ranked[0][0] == 1  # best design
        assert ranked[-1][0] == 2  # worst design


# ---------------------------------------------------------------------------
# M2: Sensitivity Analysis
# ---------------------------------------------------------------------------


class TestSensitivity:
    def test_tornado_has_bars_per_variable(self):
        """Each variable produces bars for each tracked objective."""
        base = {"area": 50.0, "power": 5.0}
        result = tornado_analysis(base, _mock_evaluator, _DS_VARS)
        assert isinstance(result, TornadoResult)
        assert len(result.bars) > 0
        # Each of 3 variables x 6 objectives = 18 bars
        var_names = {b.variable_name for b in result.bars}
        assert len(var_names) == 3

    def test_tornado_sorted_by_swing(self):
        """Bars are sorted by swing magnitude descending."""
        base = {"area": 50.0, "power": 5.0}
        result = tornado_analysis(base, _mock_evaluator, _DS_VARS)
        swings = [b.swing for b in result.bars]
        assert swings == sorted(swings, reverse=True)

    def test_taguchi_18_evaluations(self):
        """Taguchi L18 calls the evaluator exactly 18 times."""
        call_count = [0]

        def counting_eval(params):
            call_count[0] += 1
            return _mock_evaluator(params)

        result = taguchi_l18_screening(counting_eval, _DS_VARS, {"area": 50, "power": 5})
        assert call_count[0] == 18
        assert isinstance(result, TaguchiResult)

    def test_taguchi_top_factors(self):
        """Top factors list is non-empty and contains variable names."""
        result = taguchi_l18_screening(_mock_evaluator, _DS_VARS, {"area": 50, "power": 5})
        assert len(result.top_factors) > 0
        for f in result.top_factors:
            assert f in [v["name"] for v in _DS_VARS]

    def test_taguchi_main_effects_structure(self):
        """Main effects have the right structure."""
        result = taguchi_l18_screening(_mock_evaluator, _DS_VARS, {"area": 50, "power": 5})
        for vname, effects in result.main_effects.items():
            assert isinstance(effects, dict)
            assert len(effects) > 0


# ---------------------------------------------------------------------------
# M3: Pareto-Guided Exploration
# ---------------------------------------------------------------------------


class TestParetoExploration:
    def test_topsis_2d(self):
        """TOPSIS correctly ranks 2 designs."""
        designs = [
            _make_design(5, 100, 50),  # balanced
            _make_design(1, 200, 100),  # low power but heavy/expensive
        ]
        weights = {k: 1.0 / 6 for k in OBJECTIVES}
        result = topsis_rank(designs, weights)
        assert isinstance(result, TOPSISResult)
        assert len(result.rankings) == 2
        assert "topsis_score" in result.rankings[0]

    def test_topsis_with_weights(self):
        """Weighting changes the winner."""
        designs = [
            _make_design(1, 500, 500),  # low power
            _make_design(50, 10, 10),  # low weight/cost
        ]
        # Weight power heavily -> design 0 wins
        w_power = {k: 0.02 for k in OBJECTIVES}
        w_power["power_watts"] = 0.90
        r_power = topsis_rank(designs, w_power)
        assert r_power.rankings[0]["_original_index"] == 0

        # Weight weight/cost heavily -> design 1 wins
        w_weight = {k: 0.02 for k in OBJECTIVES}
        w_weight["weight_grams"] = 0.45
        w_weight["cost_usd"] = 0.45
        r_weight = topsis_rank(designs, w_weight)
        assert r_weight.rankings[0]["_original_index"] == 1

    def test_topsis_ideal_anti_ideal(self):
        """Ideal and anti-ideal points are correct."""
        designs = [
            _make_design(1, 10, 5),
            _make_design(10, 100, 50),
        ]
        weights = {k: 1.0 / 6 for k in OBJECTIVES}
        result = topsis_rank(designs, weights)
        assert result.ideal_point["power_watts"] == 1.0
        assert result.anti_ideal_point["power_watts"] == 10.0

    def test_mrs_at_knee(self):
        """MRS computes exchange rates between neighbors."""
        front = [
            _make_design(5, 100, 50),
            _make_design(3, 150, 70),  # knee
            _make_design(1, 200, 100),
        ]
        result = marginal_rate_of_substitution(front, knee_index=1)
        assert result.knee_index == 1
        assert len(result.exchange_rates) > 0

    def test_clustering_k4(self):
        """Clustering produces 4 clusters with labels."""
        rng = np.random.default_rng(42)
        front = []
        for _ in range(20):
            front.append(
                _make_design(
                    power=rng.uniform(1, 50),
                    weight=rng.uniform(10, 500),
                    cost=rng.uniform(5, 200),
                )
            )
        result = cluster_pareto_front(front, n_clusters=4, seed=42)
        assert isinstance(result, ClusterResult)
        assert len(result.labels) == 20
        assert all(0 <= lbl < 4 for lbl in result.labels)

    def test_cluster_labels(self):
        """Cluster labels reference objective names."""
        front = [
            _make_design(1, 500, 500),
            _make_design(50, 10, 10),
            _make_design(25, 250, 250),
        ]
        result = cluster_pareto_front(front, n_clusters=2, seed=42)
        for c in result.clusters:
            assert "label" in c
            assert "low-" in c["label"]


# ---------------------------------------------------------------------------
# M4: Delta Attribution & Sweep
# ---------------------------------------------------------------------------


class TestDeltaAndSweep:
    def test_same_bom_zero_delta(self):
        """Identical BOMs produce zero deltas."""
        bom = [
            {"name": "die", "weight_grams": 1.0, "volume_cm3": 0.1, "cost_usd": 5.0},
            {"name": "pcb", "weight_grams": 10.0, "volume_cm3": 5.0, "cost_usd": 3.0},
        ]
        result = delta_attribution(bom, bom)
        assert isinstance(result, DeltaAttribution)
        assert result.total_deltas["weight_grams"] == 0.0
        assert result.total_deltas["cost_usd"] == 0.0

    def test_different_enclosure(self):
        """Different enclosure material shows up in deltas."""
        left = [
            {"name": "enclosure", "weight_grams": 100.0, "volume_cm3": 50.0, "cost_usd": 10.0},
        ]
        right = [
            {"name": "enclosure", "weight_grams": 60.0, "volume_cm3": 50.0, "cost_usd": 15.0},
        ]
        result = delta_attribution(left, right)
        enc_delta = [d for d in result.component_deltas if d["name"] == "enclosure"][0]
        assert enc_delta["weight_delta"] == -40.0
        assert enc_delta["cost_delta"] == 5.0

    def test_sweep_steps(self):
        """Sweep produces the requested number of steps."""
        base = {"area": 50.0, "power": 5.0}
        values = [10.0, 30.0, 50.0, 70.0, 100.0]
        result = parametric_sweep(base, "area", values, _mock_evaluator)
        assert isinstance(result, SweepResult)
        assert len(result.sweep_values) == 5
        assert len(result.objective_traces["weight_grams"]) == 5

    def test_sweep_monotonic(self):
        """Weight increases monotonically with area (for mock evaluator)."""
        base = {"area": 50.0, "power": 5.0}
        values = [10.0, 50.0, 100.0, 200.0]
        result = parametric_sweep(base, "area", values, _mock_evaluator)
        weights = result.objective_traces["weight_grams"]
        assert all(weights[i] <= weights[i + 1] for i in range(len(weights) - 1))


# ---------------------------------------------------------------------------
# M5: Monte Carlo Budget Feasibility
# ---------------------------------------------------------------------------


class TestMonteCarlo:
    def test_zero_uncertainty(self):
        """With a deterministic evaluator, all samples match the point estimate."""

        def deterministic(params):
            return {"weight_grams": 100.0, "cost_usd": 50.0}

        result = monte_carlo_feasibility(
            {"area": 50, "power": 5},
            {"weight_grams": 200.0},
            deterministic,
            n_samples=100,
            seed=42,
        )
        assert isinstance(result, MonteCarloResult)
        assert result.p50["weight_grams"] == 100.0
        assert result.feasibility_prob["weight_grams"] == 1.0

    def test_p90_gte_p50(self):
        """P90 >= P50 for all metrics."""
        result = monte_carlo_feasibility(
            {"area": 50, "power": 5},
            {"weight_grams": 200.0, "cost_usd": 100.0},
            _mock_evaluator,
            n_samples=200,
            seed=42,
        )
        for obj in ["weight_grams", "cost_usd"]:
            assert result.p90[obj] >= result.p50[obj]

    def test_feasibility_range(self):
        """Feasibility probabilities are between 0 and 1."""
        result = monte_carlo_feasibility(
            {"area": 50, "power": 5},
            {"weight_grams": 200.0},
            _mock_evaluator,
            n_samples=100,
            seed=42,
        )
        for prob in result.feasibility_prob.values():
            assert 0.0 <= prob <= 1.0

    def test_traffic_light_green(self):
        """Very generous budget -> green light."""
        result = monte_carlo_feasibility(
            {"area": 50, "power": 5},
            {"weight_grams": 10000.0},  # huge budget
            _mock_evaluator,
            n_samples=100,
            seed=42,
        )
        assert result.traffic_light["weight_grams"] == "green"
        assert result.feasibility_prob["weight_grams"] >= 0.90

    def test_traffic_light_red(self):
        """Impossibly tight budget -> red light."""
        result = monte_carlo_feasibility(
            {"area": 50, "power": 5},
            {"weight_grams": 0.001},  # impossibly tight
            _mock_evaluator,
            n_samples=100,
            seed=42,
        )
        assert result.traffic_light["weight_grams"] == "red"
        assert result.feasibility_prob["weight_grams"] < 0.50

    def test_overall_lte_individual(self):
        """Overall feasibility <= min individual feasibility."""
        result = monte_carlo_feasibility(
            {"area": 50, "power": 5},
            {"weight_grams": 200.0, "cost_usd": 100.0},
            _mock_evaluator,
            n_samples=200,
            seed=42,
        )
        min_individual = min(result.feasibility_prob.values())
        assert result.overall_feasibility <= min_individual + 1e-9
