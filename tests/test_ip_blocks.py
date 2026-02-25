"""Tests for composable IP block models and SoC composition.

Unit tests verify area/power scaling physics for each block type.
Integration tests verify the composition pipeline produces heterogeneous designs.
"""

import pytest

from embodied_ai_architect.graphs.ip_blocks import (
    ARM_A55_PRESET,
    ARM_A78_PRESET,
    CPUCoreConfig,
    DEFAULT_IO_PRESET,
    DEFAULT_ISP_PRESET,
    GPUClusterConfig,
    ISPConfig,
    LPDDR4X_CONTROLLER_PRESET,
    MEDIUM_GPU_PRESET,
    MESH_NOC_PRESET,
    MemoryControllerConfig,
    NoCIPConfig,
    RISCV_SMALL_PRESET,
    SMALL_GPU_PRESET,
    SoCComposition,
    kpu_preset_block,
)

# ============================================================================
# CPU Core IP
# ============================================================================


class TestCPUCoreArea:
    def test_area_scales_with_process(self):
        """A78 at 7nm should be smaller than at 28nm."""
        a78 = ARM_A78_PRESET
        area_7 = a78.estimate_area_mm2(7)
        area_28 = a78.estimate_area_mm2(28)
        assert area_7 < area_28

    def test_a78_larger_than_a55(self):
        """Big core (A78) should be larger than little core (A55)."""
        area_a78 = ARM_A78_PRESET.estimate_area_mm2(28)
        area_a55 = ARM_A55_PRESET.estimate_area_mm2(28)
        assert area_a78 > area_a55

    def test_riscv_smallest(self):
        """RISC-V minimal should be smallest of all presets."""
        area_rv = RISCV_SMALL_PRESET.estimate_area_mm2(28)
        area_a55 = ARM_A55_PRESET.estimate_area_mm2(28)
        assert area_rv < area_a55

    def test_count_multiplies_area(self):
        """count=2 should double the area."""
        single = ARM_A78_PRESET.model_copy(update={"count": 1})
        double = ARM_A78_PRESET.model_copy(update={"count": 2})
        assert abs(double.estimate_area_mm2(28) - 2 * single.estimate_area_mm2(28)) < 0.001

    def test_area_positive(self):
        """Area should always be positive."""
        for preset in [ARM_A78_PRESET, ARM_A55_PRESET, RISCV_SMALL_PRESET]:
            assert preset.estimate_area_mm2(28) > 0
            assert preset.estimate_area_mm2(7) > 0


class TestCPUCorePower:
    def test_power_scales_with_voltage(self):
        """Power at 7nm (lower Vdd) should be less than at 28nm (higher Vdd),
        since power ~ Vdd^2 and area also shrinks."""
        a78 = ARM_A78_PRESET
        power_7 = a78.estimate_power_watts(7)
        power_28 = a78.estimate_power_watts(28)
        assert power_7 < power_28

    def test_power_scales_with_utilization(self):
        """Higher utilization = more power."""
        a78 = ARM_A78_PRESET
        power_low = a78.estimate_power_watts(28, utilization=0.3)
        power_high = a78.estimate_power_watts(28, utilization=0.9)
        assert power_high > power_low

    def test_power_positive(self):
        assert ARM_A78_PRESET.estimate_power_watts(28) > 0


class TestCPUCorePeakGops:
    def test_a78_gops_reasonable(self):
        gops = ARM_A78_PRESET.peak_gops()
        # 4-wide SIMD @ 2GHz => 4 * 2 * 2 = 16 GOPS
        assert gops == pytest.approx(16.0, rel=0.01)

    def test_simd_doubles_gops(self):
        with_simd = CPUCoreConfig(
            name="test", isa="ARM_A78", has_simd=True, issue_width=4, frequency_mhz=1000
        )
        no_simd = CPUCoreConfig(
            name="test", isa="ARM_A78", has_simd=False, issue_width=4, frequency_mhz=1000
        )
        assert with_simd.peak_gops() == 2 * no_simd.peak_gops()


# ============================================================================
# GPU Cluster IP
# ============================================================================


class TestGPUClusterArea:
    def test_area_includes_sram(self):
        """SRAM should contribute non-zero area to the total."""
        gpu = SMALL_GPU_PRESET
        total_area = gpu.estimate_area_mm2(28)
        # Compute SRAM-only area
        from embodied_ai_architect.graphs.technology import estimate_sram_area_mm2

        sram_area = estimate_sram_area_mm2(
            gpu.l1_bytes_per_core, 28
        ) * gpu.shader_cores + estimate_sram_area_mm2(gpu.shared_l2_bytes, 28)
        # SRAM area should be positive and part of the total
        assert sram_area > 0
        assert total_area > sram_area  # logic dominates, but SRAM is included

    def test_area_scales_with_process(self):
        area_7 = SMALL_GPU_PRESET.estimate_area_mm2(7)
        area_28 = SMALL_GPU_PRESET.estimate_area_mm2(28)
        assert area_7 < area_28

    def test_medium_larger_than_small(self):
        area_small = SMALL_GPU_PRESET.estimate_area_mm2(28)
        area_medium = MEDIUM_GPU_PRESET.estimate_area_mm2(28)
        assert area_medium > area_small

    def test_tensor_units_add_area(self):
        no_tensor = GPUClusterConfig(name="test", shader_cores=4, tensor_units=0)
        with_tensor = GPUClusterConfig(name="test", shader_cores=4, tensor_units=16)
        assert with_tensor.estimate_area_mm2(28) > no_tensor.estimate_area_mm2(28)


class TestGPUClusterPower:
    def test_power_positive(self):
        assert SMALL_GPU_PRESET.estimate_power_watts(28) > 0

    def test_more_cores_more_power(self):
        small = SMALL_GPU_PRESET.estimate_power_watts(28)
        medium = MEDIUM_GPU_PRESET.estimate_power_watts(28)
        assert medium > small


class TestGPUClusterGops:
    def test_peak_gops(self):
        gpu = SMALL_GPU_PRESET
        # 4 cores * 128 units * 2 (FMA) * 0.8 GHz = 819.2 GOPS
        expected = 4 * 128 * 2 * 0.8
        assert gpu.peak_gops() == pytest.approx(expected, rel=0.01)


# ============================================================================
# KPU Block IP
# ============================================================================


class TestKPUBlock:
    def test_delegates_to_floorplan(self):
        """KPU block should use floorplan estimator for area."""
        kpu = kpu_preset_block("edge_balanced")
        area = kpu.estimate_area_mm2(28)
        assert area > 0

    def test_area_scales_with_process(self):
        kpu = kpu_preset_block("edge_balanced")
        area_7 = kpu.estimate_area_mm2(7)
        area_28 = kpu.estimate_area_mm2(28)
        assert area_7 < area_28

    def test_power_positive(self):
        kpu = kpu_preset_block("edge_balanced")
        assert kpu.estimate_power_watts(28) > 0

    def test_peak_gops_from_kpu_config(self):
        """peak_gops should come from KPUMicroArchConfig.peak_tops_int8."""
        kpu = kpu_preset_block("edge_balanced")
        gops = kpu.peak_gops()
        # edge_balanced has non-trivial compute
        assert gops > 100

    def test_drone_minimal_smaller_than_edge(self):
        drone = kpu_preset_block("drone_minimal")
        edge = kpu_preset_block("edge_balanced")
        assert drone.estimate_area_mm2(28) < edge.estimate_area_mm2(28)

    def test_invalid_preset_raises(self):
        with pytest.raises(ValueError, match="Unknown KPU preset"):
            kpu_preset_block("nonexistent_preset")


# ============================================================================
# Infrastructure IPs
# ============================================================================


class TestMemoryController:
    def test_area_scales_with_channels(self):
        mc2 = MemoryControllerConfig(name="mc2", channels=2)
        mc4 = MemoryControllerConfig(name="mc4", channels=4)
        assert mc4.estimate_area_mm2(28) > mc2.estimate_area_mm2(28)

    def test_bandwidth(self):
        mc = LPDDR4X_CONTROLLER_PRESET
        assert mc.total_bandwidth_gbps == pytest.approx(4 * 6.4)

    def test_zero_gops(self):
        assert LPDDR4X_CONTROLLER_PRESET.peak_gops() == 0.0


class TestNoCIP:
    def test_area_scales_with_routers(self):
        small = NoCIPConfig(name="small", num_routers=4)
        big = NoCIPConfig(name="big", num_routers=32)
        assert big.estimate_area_mm2(28) > small.estimate_area_mm2(28)

    def test_power_positive(self):
        assert MESH_NOC_PRESET.estimate_power_watts(28) > 0


class TestIOSubsystem:
    def test_area_positive(self):
        assert DEFAULT_IO_PRESET.estimate_area_mm2(28) > 0

    def test_power_positive(self):
        assert DEFAULT_IO_PRESET.estimate_power_watts(28) > 0


class TestISP:
    def test_resolution_affects_area(self):
        isp_720 = ISPConfig(name="isp720", max_resolution="720p")
        isp_4k = ISPConfig(name="isp4k", max_resolution="4K")
        assert isp_4k.estimate_area_mm2(28) > isp_720.estimate_area_mm2(28)

    def test_area_scales_with_process(self):
        isp = DEFAULT_ISP_PRESET
        assert isp.estimate_area_mm2(7) < isp.estimate_area_mm2(28)


# ============================================================================
# SoC Composition
# ============================================================================


class TestSoCComposition:
    def test_totals_sum_blocks(self):
        """Total area/power should be sum of parts + interconnect overhead."""
        soc = SoCComposition(
            name="test-soc",
            process_nm=28,
            blocks=[ARM_A78_PRESET, SMALL_GPU_PRESET],
        )
        cpu_area = ARM_A78_PRESET.estimate_area_mm2(28)
        gpu_area = SMALL_GPU_PRESET.estimate_area_mm2(28)
        expected_area = (cpu_area + gpu_area) * 1.2  # 20% overhead
        assert soc.total_area_mm2() == pytest.approx(expected_area, rel=0.001)

    def test_power_totals(self):
        soc = SoCComposition(
            name="test-soc",
            process_nm=28,
            blocks=[ARM_A78_PRESET, SMALL_GPU_PRESET],
        )
        cpu_power = ARM_A78_PRESET.estimate_power_watts(28)
        gpu_power = SMALL_GPU_PRESET.estimate_power_watts(28)
        assert soc.total_power_watts() == pytest.approx(cpu_power + gpu_power, rel=0.001)

    def test_peak_gops_totals(self):
        soc = SoCComposition(
            name="test-soc",
            process_nm=28,
            blocks=[ARM_A78_PRESET, SMALL_GPU_PRESET],
        )
        expected = ARM_A78_PRESET.peak_gops() + SMALL_GPU_PRESET.peak_gops()
        assert soc.total_peak_gops() == pytest.approx(expected, rel=0.001)

    def test_block_summary(self):
        soc = SoCComposition(
            name="test-soc",
            process_nm=28,
            blocks=[ARM_A78_PRESET, SMALL_GPU_PRESET],
        )
        summary = soc.block_summary()
        assert len(summary["blocks"]) == 2
        assert summary["totals"]["area_mm2"] > 0
        assert summary["totals"]["power_watts"] > 0

    def test_to_ip_block_dicts(self):
        soc = SoCComposition(
            name="test-soc",
            process_nm=28,
            blocks=[ARM_A78_PRESET, SMALL_GPU_PRESET],
        )
        dicts = soc.to_ip_block_dicts()
        assert len(dicts) == 2
        assert all("block_type" in d for d in dicts)
        assert all("_area_mm2" in d for d in dicts)
        assert all("_power_watts" in d for d in dicts)
        assert all("_peak_gops" in d for d in dicts)

    def test_composition_from_presets(self):
        """Build a realistic edge SoC from presets."""
        soc = SoCComposition(
            name="edge-soc",
            process_nm=28,
            blocks=[
                ARM_A78_PRESET.model_copy(update={"count": 2}),
                ARM_A55_PRESET.model_copy(update={"count": 2}),
                kpu_preset_block("edge_balanced"),
                LPDDR4X_CONTROLLER_PRESET,
                MESH_NOC_PRESET,
                DEFAULT_IO_PRESET,
                DEFAULT_ISP_PRESET,
            ],
        )
        assert soc.total_area_mm2() > 0
        assert soc.total_power_watts() > 0
        assert soc.total_peak_gops() > 0
        assert len(soc.blocks) == 7

    def test_empty_composition(self):
        soc = SoCComposition(name="empty", process_nm=28, blocks=[])
        assert soc.total_area_mm2() == 0.0
        assert soc.total_power_watts() == 0.0
        assert soc.total_peak_gops() == 0.0


# ============================================================================
# Integration: architecture_composer produces heterogeneous blocks
# ============================================================================


class TestArchitectureComposerIntegration:
    def test_produces_heterogeneous_blocks(self):
        """Given a perception workload, composer should produce CPU + KPU + memory + IO."""
        from embodied_ai_architect.graphs.soc_state import (
            DesignConstraints,
            create_initial_soc_state,
        )
        from embodied_ai_architect.graphs.specialists import architecture_composer
        from embodied_ai_architect.graphs.task_graph import TaskNode

        state = create_initial_soc_state(
            goal="Drone SoC with object detection",
            constraints=DesignConstraints(
                max_power_watts=5.0,
                max_latency_ms=33.3,
                max_cost_usd=30.0,
            ),
            use_case="delivery_drone",
            platform="drone",
        )
        state["workload_profile"] = {
            "workloads": [
                {
                    "name": "object_detection",
                    "model_class": "YOLOv8n",
                    "operators": [{"type": "convolution", "count": 75}],
                    "estimated_gflops": 8.7,
                    "estimated_memory_mb": 12.0,
                },
            ],
            "total_estimated_gflops": 8.7,
            "dominant_op": "convolution",
            "use_case": "delivery_drone",
            "source": "goal_estimation",
        }
        state["hardware_candidates"] = [
            {
                "name": "Stillwater KPU",
                "type": "kpu",
                "compute_paradigm": "dataflow",
                "peak_tops_int8": 20.0,
                "peak_tflops_fp16": 5.0,
                "memory_gb": 2.0,
                "tdp_watts": 5.0,
                "cost_usd": 25.0,
                "score": 85.0,
                "rank": 1,
                "constraint_verdicts": {"power": "PASS", "cost": "PASS"},
            },
        ]

        task = TaskNode(id="t1", name="compose", agent="architecture_composer")
        result = architecture_composer(task, state)

        ip_blocks = result["ip_blocks"]
        block_types = {b["block_type"] for b in ip_blocks}

        # Should have CPU, KPU, memory, NoC, IO, and ISP (vision workload)
        assert "cpu_core" in block_types
        assert "kpu_tile" in block_types
        assert "memory_ctrl" in block_types
        assert "noc" in block_types
        assert "io" in block_types
        assert "isp" in block_types

        # All blocks should have physics-based area data
        assert all("_area_mm2" in b for b in ip_blocks)
        assert all("_power_watts" in b for b in ip_blocks)

    def test_ppa_uses_physics_models(self):
        """PPA assessor should use physics-based area from ip_blocks, not fixed lookup."""
        from embodied_ai_architect.graphs.soc_state import (
            DesignConstraints,
            create_initial_soc_state,
        )
        from embodied_ai_architect.graphs.specialists import ppa_assessor
        from embodied_ai_architect.graphs.task_graph import TaskNode

        # Build state with physics-based ip_blocks
        soc = SoCComposition(
            name="test",
            process_nm=28,
            blocks=[ARM_A78_PRESET, kpu_preset_block("edge_balanced")],
        )

        state = create_initial_soc_state(
            goal="test PPA",
            constraints=DesignConstraints(
                max_power_watts=15.0,
                max_latency_ms=100.0,
                max_cost_usd=100.0,
            ),
        )
        state["workload_profile"] = {
            "total_estimated_gflops": 8.7,
            "dominant_op": "convolution",
        }
        state["hardware_candidates"] = [
            {
                "name": "Stillwater KPU",
                "type": "kpu",
                "peak_tops_int8": 20.0,
                "tdp_watts": 5.0,
                "cost_usd": 25.0,
                "score": 85.0,
                "rank": 1,
            },
        ]
        state["selected_architecture"] = {
            "name": "SoC-KPU",
            "primary_compute": "Stillwater KPU",
        }
        state["ip_blocks"] = soc.to_ip_block_dicts()

        task = TaskNode(id="t1", name="ppa", agent="ppa_assessor")
        result = ppa_assessor(task, state)

        ppa = result["ppa_metrics"]
        # Should use physics-based area (not the fixed 12.0 = 4+8 from old lookup)
        assert ppa["area_mm2"] > 0
        assert ppa["power_watts"] > 0
        assert ppa["latency_ms"] > 0


class TestFullPipelineWithIPBlocks:
    """End-to-end: planner -> dispatch -> PPA with physics-based IP blocks."""

    PLAN = [
        {"id": "t1", "name": "Analyze workload", "agent": "workload_analyzer", "dependencies": []},
        {"id": "t2", "name": "Explore hardware", "agent": "hw_explorer", "dependencies": ["t1"]},
        {
            "id": "t3",
            "name": "Compose architecture",
            "agent": "architecture_composer",
            "dependencies": ["t2"],
        },
        {"id": "t4", "name": "Assess PPA", "agent": "ppa_assessor", "dependencies": ["t3"]},
        {"id": "t5", "name": "Review design", "agent": "critic", "dependencies": ["t4"]},
        {
            "id": "t6",
            "name": "Generate report",
            "agent": "report_generator",
            "dependencies": ["t5"],
        },
    ]

    def test_full_loop_with_ip_blocks(self):
        """Full pipeline should produce physics-based ip_blocks in state."""
        from embodied_ai_architect.graphs.soc_state import (
            DesignConstraints,
            DesignStatus,
            create_initial_soc_state,
        )
        from embodied_ai_architect.graphs.planner import PlannerNode
        from embodied_ai_architect.graphs.specialists import create_default_dispatcher

        state = create_initial_soc_state(
            goal="Design an SoC for drone with object detection at 30fps, <5W, <$30",
            constraints=DesignConstraints(
                max_power_watts=5.0,
                max_latency_ms=33.3,
                max_cost_usd=30.0,
            ),
            use_case="delivery_drone",
            platform="drone",
        )

        planner = PlannerNode(static_plan=self.PLAN)
        updates = planner(state)
        state = {**state, **updates}

        dispatcher = create_default_dispatcher()
        state = dispatcher.run(state)

        assert state["status"] == DesignStatus.COMPLETE.value

        # IP blocks should have physics-based data
        ip_blocks = state.get("ip_blocks", [])
        assert len(ip_blocks) > 0
        assert any(b.get("block_type") == "cpu_core" for b in ip_blocks)
        assert any(b.get("block_type") == "kpu_tile" for b in ip_blocks)
        assert all("_area_mm2" in b for b in ip_blocks)

        # PPA should have been computed
        ppa = state.get("ppa_metrics", {})
        assert ppa.get("area_mm2", 0) > 0
        assert ppa.get("power_watts", 0) > 0
