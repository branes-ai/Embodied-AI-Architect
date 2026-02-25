"""Tests for the manufacturing cost model."""

from embodied_ai_architect.graphs.manufacturing import (
    ManufacturingCostBreakdown,
    dies_per_wafer,
    estimate_manufacturing_cost,
    murphy_yield,
)
from embodied_ai_architect.graphs.technology import _AVAILABLE_NODES


class TestDiesPerWafer:
    def test_small_die(self):
        """30mm² die should give ~2000+ dies on a 300mm wafer."""
        n = dies_per_wafer(30.0)
        assert n >= 2000

    def test_large_die(self):
        """200mm² die should give ~290+ dies on a 300mm wafer."""
        n = dies_per_wafer(200.0)
        assert n >= 280

    def test_zero_area(self):
        assert dies_per_wafer(0.0) == 0

    def test_negative_area(self):
        assert dies_per_wafer(-1.0) == 0

    def test_monotonic_decrease(self):
        """Larger dies → fewer dies per wafer."""
        n_small = dies_per_wafer(10.0)
        n_large = dies_per_wafer(100.0)
        assert n_small > n_large


class TestMurphyYield:
    def test_small_die_low_defect(self):
        """Small die + low defect density = high yield (>90%)."""
        y = murphy_yield(defect_density=0.02, die_area_cm2=0.30)
        assert y > 0.90

    def test_large_die_leading_edge(self):
        """Large die + high defect density = lower yield."""
        y = murphy_yield(defect_density=0.14, die_area_cm2=2.0)
        assert y < 0.80

    def test_zero_defects(self):
        """Zero defect density → 100% yield."""
        y = murphy_yield(defect_density=0.0, die_area_cm2=1.0)
        assert y == 1.0

    def test_yield_bounded(self):
        """Yield should always be between 0 and 1."""
        for d in [0.01, 0.1, 0.5, 1.0]:
            for a in [0.1, 1.0, 5.0]:
                y = murphy_yield(d, a)
                assert 0.0 < y <= 1.0


class TestEstimateManufacturingCost:
    def test_basic(self):
        """Returns ManufacturingCostBreakdown with positive values."""
        mfg = estimate_manufacturing_cost(30.0, 28, 100_000)
        assert isinstance(mfg, ManufacturingCostBreakdown)
        assert mfg.total_unit_cost_usd > 0
        assert mfg.die_cost_usd > 0
        assert mfg.package_cost_usd > 0
        assert mfg.yield_percent > 0
        assert mfg.dies_per_wafer > 0

    def test_mature_node_cheaper_than_leading_edge(self):
        """40nm cost < 7nm cost for the same logic (same effective die area at 28nm = 30mm²)."""
        # At 40nm, same logic has die area scaled by (40/28)² ≈ 2.04x → ~61mm²
        area_40nm = 30.0 * (40 / 28) ** 2
        mfg_40 = estimate_manufacturing_cost(area_40nm, 40, 100_000)

        # At 7nm, die shrinks to (7/28)² ≈ 0.0625x → ~1.9mm²
        area_7nm = 30.0 * (7 / 28) ** 2
        mfg_7 = estimate_manufacturing_cost(area_7nm, 7, 100_000)

        # 40nm should be cheaper due to lower wafer cost and much lower NRE
        assert mfg_40.total_unit_cost_usd < mfg_7.total_unit_cost_usd

    def test_volume_reduces_nre_per_unit(self):
        """1M volume NRE/unit < 10K volume NRE/unit."""
        mfg_10k = estimate_manufacturing_cost(30.0, 28, 10_000)
        mfg_1m = estimate_manufacturing_cost(30.0, 28, 1_000_000)

        assert mfg_1m.nre_per_unit_usd < mfg_10k.nre_per_unit_usd
        assert mfg_1m.total_unit_cost_usd < mfg_10k.total_unit_cost_usd

    def test_process_economics_coverage(self):
        """All nodes in _AVAILABLE_NODES should have economics data or nearest-match."""
        for node in _AVAILABLE_NODES:
            # Should not raise — nearest-match fallback
            mfg = estimate_manufacturing_cost(30.0, node, 10_000)
            assert mfg.total_unit_cost_usd > 0

    def test_package_types(self):
        """Different package types produce different costs."""
        mfg_qfn = estimate_manufacturing_cost(30.0, 28, 10_000, package_type="QFN")
        mfg_fcbga = estimate_manufacturing_cost(30.0, 28, 10_000, package_type="FCBGA")
        assert mfg_qfn.package_cost_usd < mfg_fcbga.package_cost_usd
