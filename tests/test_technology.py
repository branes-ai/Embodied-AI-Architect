"""Tests for the technology node database."""

import pytest

from embodied_ai_architect.graphs.technology import (
    SRAM_DENSITY,
    TECHNOLOGY_NODES,
    TechnologyNode,
    estimate_area_um2,
    estimate_sram_area_mm2,
    estimate_timing_ps,
    get_technology,
)


class TestTechnologyDatabase:
    def test_all_nodes_present(self):
        expected = {
            "2nm",
            "3nm",
            "4nm",
            "5nm",
            "7nm",
            "8nm",
            "10nm",
            "12nm",
            "14nm",
            "16nm",
            "22nm",
            "28nm",
            "40nm",
            "65nm",
            "90nm",
            "130nm",
            "180nm",
        }
        assert set(TECHNOLOGY_NODES.keys()) == expected

    def test_sram_density_has_entries(self):
        assert len(SRAM_DENSITY) >= 10
        assert SRAM_DENSITY[7] == 45.0
        assert SRAM_DENSITY[28] == 8.0


class TestGetTechnology:
    def test_exact_match(self):
        tech = get_technology(28)
        assert tech.process_nm == 28
        assert tech.cell_area_um2 == 0.200
        assert tech.gate_delay_ps == 25

    def test_nearest_match(self):
        tech = get_technology(6)  # no 6nm, should snap to 5nm or 7nm
        assert tech.process_nm in (5, 7)

    def test_returns_technology_node(self):
        tech = get_technology(7)
        assert isinstance(tech, TechnologyNode)
        assert tech.name == "7nm FinFET (N7)"
        assert tech.sram_density_mb_per_mm2 == 45.0


class TestEstimation:
    def test_estimate_area_um2(self):
        area = estimate_area_um2(1000, 28)
        assert area == pytest.approx(200.0)  # 1000 * 0.2

    def test_estimate_sram_area_mm2(self):
        # 1MB at 28nm with density 8 Mb/mm²
        # Function converts bytes -> MB then divides by density
        # 1048576 bytes = 1 MB, 1 MB / 8 Mb/mm² = 0.125 mm²
        area = estimate_sram_area_mm2(1024 * 1024, 28)
        expected = 1.0 / 8.0  # 1 MB / 8 Mb/mm²
        assert area == pytest.approx(expected)

    def test_estimate_timing_ps(self):
        timing = estimate_timing_ps(10, 28)
        assert timing == pytest.approx(250.0)  # 10 * 25ps

    def test_process_scaling(self):
        area_28 = estimate_area_um2(1000, 28)
        area_7 = estimate_area_um2(1000, 7)
        assert area_7 < area_28  # smaller process = less area
