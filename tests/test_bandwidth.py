"""Tests for bandwidth matching analysis."""

from embodied_ai_architect.graphs.bandwidth import (
    BandwidthMatchResult,
    check_bandwidth_match,
)
from embodied_ai_architect.graphs.kpu_config import (
    DRAMConfig,
    KPUMicroArchConfig,
    KPU_PRESETS,
)


class TestBandwidthMatch:
    def test_balanced_config(self):
        config = KPU_PRESETS["edge_balanced"]
        workload = {"total_estimated_gflops": 8.7}
        result = check_bandwidth_match(config, workload, arithmetic_intensity=50.0)
        assert isinstance(result, BandwidthMatchResult)
        assert len(result.links) == 4
        # Moderate workload on balanced config should be fine
        assert result.balanced

    def test_has_four_links(self):
        config = KPU_PRESETS["edge_balanced"]
        result = check_bandwidth_match(config, {"total_estimated_gflops": 5.0})
        link_names = {link.name for link in result.links}
        assert link_names == {"dram_to_l3", "l3_to_l2", "l2_to_l1", "l1_to_compute"}

    def test_dram_bottleneck_with_low_bw(self):
        # Tiny DRAM bandwidth + large workload
        config = KPUMicroArchConfig(
            dram=DRAMConfig(
                num_controllers=1,
                channels_per_controller=1,
                bandwidth_per_channel_gbps=1.0,
            ),
        )
        result = check_bandwidth_match(
            config,
            {"total_estimated_gflops": 100.0},
            arithmetic_intensity=1.0,  # Very memory-bound
        )
        # Should detect DRAM bottleneck
        dram_link = next(link for link in result.links if link.name == "dram_to_l3")
        assert dram_link.utilization > 0.85

    def test_high_arithmetic_intensity_reduces_demand(self):
        config = KPU_PRESETS["edge_balanced"]
        workload = {"total_estimated_gflops": 8.7}
        # High arithmetic intensity = compute-bound = low BW demand
        result = check_bandwidth_match(config, workload, arithmetic_intensity=100.0)
        assert result.compute_demand_gbps < 1.0

    def test_peak_utilization_positive(self):
        config = KPU_PRESETS["edge_balanced"]
        result = check_bandwidth_match(config, {"total_estimated_gflops": 5.0})
        assert result.peak_utilization >= 0.0

    def test_issues_list(self):
        config = KPU_PRESETS["edge_balanced"]
        result = check_bandwidth_match(config, {"total_estimated_gflops": 5.0})
        assert isinstance(result.issues, list)
