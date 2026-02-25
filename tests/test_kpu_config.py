"""Tests for KPU micro-architecture configuration."""

import pytest

from embodied_ai_architect.graphs.kpu_config import (
    ComputeTileConfig,
    DRAMConfig,
    KPUMicroArchConfig,
    KPU_PRESETS,
    MemoryTileConfig,
    NoCConfig,
    create_kpu_config,
)


class TestDRAMConfig:
    def test_total_bandwidth(self):
        dram = DRAMConfig(
            num_controllers=2, channels_per_controller=2, bandwidth_per_channel_gbps=6.4
        )
        assert dram.total_bandwidth_gbps == pytest.approx(25.6)


class TestComputeTileConfig:
    def test_peak_tops(self):
        ct = ComputeTileConfig(
            num_tiles=4,
            array_rows=16,
            array_cols=16,
            frequency_mhz=500.0,
        )
        # 16*16*2 = 512 ops/cycle * 4 tiles * 500e6 = 1.024 TOPS
        assert ct.peak_tops_int8 == pytest.approx(1.024)


class TestKPUMicroArchConfig:
    def test_default_config(self):
        config = KPUMicroArchConfig()
        assert config.process_nm == 28
        assert config.array_rows == 3
        assert config.array_cols == 3

    def test_checkerboard_tile_counts(self):
        config = KPUMicroArchConfig(array_rows=3, array_cols=3)
        # 3x3 = 9 tiles: 5 compute + 4 memory
        assert config.num_compute_tiles == 5
        assert config.num_memory_tiles == 4

    def test_even_grid_tile_counts(self):
        config = KPUMicroArchConfig(array_rows=2, array_cols=2)
        # 2x2 = 4 tiles: 2 compute + 2 memory
        assert config.num_compute_tiles == 2
        assert config.num_memory_tiles == 2

    def test_total_sram(self):
        config = KPUMicroArchConfig(
            array_rows=3,
            array_cols=3,
            compute_tile=ComputeTileConfig(l2_size_bytes=256 * 1024, l1_size_bytes=32 * 1024),
            memory_tile=MemoryTileConfig(l3_tile_size_bytes=512 * 1024),
        )
        # 5 compute tiles * (256KB L2 + 32KB L1) + 4 memory tiles * 512KB L3
        expected = 5 * (256 + 32) * 1024 + 4 * 512 * 1024
        assert config.total_sram_bytes == expected

    def test_peak_tops(self):
        config = KPUMicroArchConfig(
            array_rows=3,
            array_cols=3,
            compute_tile=ComputeTileConfig(
                array_rows=16,
                array_cols=16,
                frequency_mhz=500.0,
            ),
        )
        # 5 compute tiles * 16*16*2 * 500e6 / 1e12
        expected = 5 * 16 * 16 * 2 * 500e6 / 1e12
        assert config.peak_tops_int8 == pytest.approx(expected)


class TestPresets:
    def test_presets_exist(self):
        assert "drone_minimal" in KPU_PRESETS
        assert "edge_balanced" in KPU_PRESETS
        assert "server_max" in KPU_PRESETS

    def test_drone_minimal_is_small(self):
        config = KPU_PRESETS["drone_minimal"]
        assert config.array_rows <= 3
        assert config.array_cols <= 3
        assert config.dram.technology == "LPDDR4X"

    def test_server_max_is_large(self):
        config = KPU_PRESETS["server_max"]
        assert config.array_rows >= 4
        assert config.dram.technology == "HBM2E"


class TestCreateKPUConfig:
    def test_drone_use_case(self):
        config = create_kpu_config(
            "delivery_drone",
            {"max_power_watts": 5.0, "target_process_nm": 28},
            {"total_estimated_gflops": 8.7, "total_estimated_memory_mb": 12.0},
        )
        assert config.process_nm == 28
        assert config.dram.num_controllers <= 2

    def test_high_compute_workload(self):
        config = create_kpu_config(
            "server_inference",
            {"max_power_watts": 100.0, "target_process_nm": 7},
            {"total_estimated_gflops": 100.0, "total_estimated_memory_mb": 500.0},
        )
        assert config.dram.technology == "HBM2E"
        assert config.compute_tile.l2_size_bytes >= 256 * 1024
