"""Integration tests for RTL generation pipeline.

Tests RTL template generation driven by KPU config and experience episode fields.
"""

from __future__ import annotations


from embodied_ai_architect.graphs.kpu_config import (
    KPU_PRESETS,
)
from embodied_ai_architect.graphs.rtl_templates import RTLTemplateEngine
from embodied_ai_architect.graphs.rtl_loop import RTLLoopConfig, run_rtl_loop
from embodied_ai_architect.graphs.technology import estimate_area_um2


class TestRTLFromKPUConfig:
    """RTL templates driven by KPU config."""

    def test_kpu_config_drives_template_rendering(self):
        config = KPU_PRESETS["drone_minimal"]
        engine = RTLTemplateEngine()
        components = engine.get_kpu_components(config)

        assert len(components) > 0
        for comp in components:
            rtl = engine.render(comp["component_type"], comp["params"])
            assert "module" in rtl
            assert len(rtl) > 50

    def test_most_components_pass_rtl_loop(self):
        config = KPU_PRESETS["drone_minimal"]
        engine = RTLTemplateEngine()
        components = engine.get_kpu_components(config)

        loop_config = RTLLoopConfig(
            process_nm=config.process_nm,
            max_iterations=1,
            skip_validation=True,
        )

        passed = 0
        for comp in components:
            rtl = engine.render(comp["component_type"], comp["params"])
            result = run_rtl_loop(
                comp["module_name"],
                rtl,
                loop_config,
            )
            if result.success:
                passed += 1
                assert result.metrics.get("area_cells", 0) > 0

        # Most components should pass; some SV features may fail with real Yosys
        assert (
            passed >= len(components) // 2
        ), f"Only {passed}/{len(components)} components passed RTL loop"

    def test_synthesis_area_uses_technology(self):
        # Verify technology.py area estimation works with synthesis cell counts
        cells = 500
        area_28nm = estimate_area_um2(cells, 28)
        area_7nm = estimate_area_um2(cells, 7)
        assert area_28nm > area_7nm  # smaller process = smaller area
        assert area_28nm > 0
        assert area_7nm > 0


class TestExperienceEpisodeIncludesKPU:
    """DesignEpisode has KPU/RTL fields."""

    def test_episode_has_kpu_fields(self):
        from embodied_ai_architect.graphs.experience import DesignEpisode

        episode = DesignEpisode(
            goal="Test",
            kpu_config_name="drone_minimal",
            kpu_process_nm=28,
            floorplan_area_mm2=45.0,
            bandwidth_balanced=True,
            rtl_modules_generated=8,
            rtl_total_cells=2500,
        )
        assert episode.kpu_config_name == "drone_minimal"
        assert episode.kpu_process_nm == 28
        assert episode.floorplan_area_mm2 == 45.0
        assert episode.bandwidth_balanced is True
        assert episode.rtl_modules_generated == 8
        assert episode.rtl_total_cells == 2500

    def test_episode_defaults_none(self):
        from embodied_ai_architect.graphs.experience import DesignEpisode

        episode = DesignEpisode(goal="Test")
        assert episode.kpu_config_name is None
        assert episode.rtl_modules_generated == 0
        assert episode.rtl_total_cells == 0
