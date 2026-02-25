"""Tests for floorplan estimator."""

from embodied_ai_architect.graphs.kpu_config import (
    ComputeTileConfig,
    KPUMicroArchConfig,
    KPU_PRESETS,
    MemoryTileConfig,
)
from embodied_ai_architect.graphs.floorplan import (
    FloorplanEstimate,
    estimate_compute_tile_dimensions,
    estimate_floorplan,
    estimate_memory_tile_dimensions,
)


class TestComputeTileDimensions:
    def test_has_sub_blocks(self):
        ct = ComputeTileConfig()
        dims = estimate_compute_tile_dimensions(ct, process_nm=28)
        names = [sb.name for sb in dims.sub_blocks]
        assert "systolic_array" in names
        assert "l2_sram" in names
        assert "l1_sram" in names
        assert dims.area_mm2 > 0
        assert dims.width_mm > 0
        assert dims.height_mm > 0

    def test_larger_array_more_area(self):
        small = ComputeTileConfig(array_rows=8, array_cols=8)
        large = ComputeTileConfig(array_rows=32, array_cols=32)
        small_dims = estimate_compute_tile_dimensions(small, 28)
        large_dims = estimate_compute_tile_dimensions(large, 28)
        assert large_dims.area_mm2 > small_dims.area_mm2


class TestMemoryTileDimensions:
    def test_has_sub_blocks(self):
        mt = MemoryTileConfig()
        dims = estimate_memory_tile_dimensions(mt, process_nm=28)
        names = [sb.name for sb in dims.sub_blocks]
        assert "l3_sram" in names
        assert "block_movers" in names
        assert dims.area_mm2 > 0

    def test_larger_l3_more_area(self):
        small = MemoryTileConfig(l3_tile_size_bytes=256 * 1024)
        large = MemoryTileConfig(l3_tile_size_bytes=2 * 1024 * 1024)
        small_dims = estimate_memory_tile_dimensions(small, 28)
        large_dims = estimate_memory_tile_dimensions(large, 28)
        assert large_dims.area_mm2 > small_dims.area_mm2


class TestFloorplanEstimate:
    def test_default_config_feasible(self):
        config = KPU_PRESETS["edge_balanced"]
        fp = estimate_floorplan(config, max_die_area_mm2=200.0)
        assert isinstance(fp, FloorplanEstimate)
        assert fp.total_area_mm2 > 0
        assert fp.die_edge_mm > 0

    def test_pitch_ratio_close_to_one(self):
        config = KPU_PRESETS["edge_balanced"]
        fp = estimate_floorplan(config)
        # Ratios should be somewhat close to 1.0
        assert 0.1 < fp.pitch_ratio_width < 10.0
        assert 0.1 < fp.pitch_ratio_height < 10.0

    def test_area_exceeds_budget_reported(self):
        config = KPU_PRESETS["server_max"]
        fp = estimate_floorplan(config, max_die_area_mm2=1.0)
        assert not fp.feasible
        assert any("exceeds" in issue.lower() for issue in fp.issues)

    def test_process_scaling(self):
        config_28 = KPUMicroArchConfig(process_nm=28)
        config_7 = KPUMicroArchConfig(process_nm=7)
        fp_28 = estimate_floorplan(config_28)
        fp_7 = estimate_floorplan(config_7)
        # 7nm should be smaller area than 28nm
        assert fp_7.total_area_mm2 < fp_28.total_area_mm2

    def test_drone_preset_reasonable(self):
        config = KPU_PRESETS["drone_minimal"]
        fp = estimate_floorplan(config, max_die_area_mm2=200.0)
        # Drone should be small
        assert fp.total_area_mm2 < 200.0

    def test_has_periphery(self):
        config = KPU_PRESETS["edge_balanced"]
        fp = estimate_floorplan(config)
        assert fp.periphery_area_mm2 > 0
        assert fp.total_area_mm2 > fp.core_area_mm2

    def test_pitch_tolerance_parameter(self):
        config = KPU_PRESETS["edge_balanced"]
        strict = estimate_floorplan(config, pitch_tolerance=0.01)
        loose = estimate_floorplan(config, pitch_tolerance=0.99)
        # Loose tolerance should be at least as feasible as strict
        if strict.feasible:
            assert loose.feasible

    def test_floorplan_returns_issues_list(self):
        config = KPU_PRESETS["edge_balanced"]
        fp = estimate_floorplan(config)
        assert isinstance(fp.issues, list)
