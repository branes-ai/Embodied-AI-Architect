"""Tests for RTL template engine."""

import pytest

from embodied_ai_architect.graphs.rtl_templates import (
    RTLTemplateEngine,
)
from embodied_ai_architect.graphs.kpu_config import KPU_PRESETS


class TestRTLTemplateEngine:
    def test_available_templates(self):
        engine = RTLTemplateEngine()
        templates = engine.available_templates()
        assert "mac_unit" in templates
        assert "compute_tile" in templates
        assert len(templates) >= 10

    def test_can_generate(self):
        engine = RTLTemplateEngine()
        assert engine.can_generate("mac_unit")
        assert not engine.can_generate("nonexistent")

    def test_render_mac_unit(self):
        engine = RTLTemplateEngine()
        rtl = engine.render("mac_unit", {"data_width": 8, "accum_width": 32})
        assert "module mac_unit" in rtl
        assert "endmodule" in rtl
        assert "DATA_WIDTH" in rtl

    def test_render_all_components(self):
        engine = RTLTemplateEngine()
        for comp_type in engine.available_templates():
            rtl = engine.render(comp_type)
            assert "module" in rtl
            assert "endmodule" in rtl

    def test_render_testbench(self):
        engine = RTLTemplateEngine()
        tb = engine.render_testbench("mac_unit", "mac_unit_test", {"data_width": 8})
        assert "module" in tb
        assert "endmodule" in tb

    def test_get_kpu_components(self):
        engine = RTLTemplateEngine()
        config = KPU_PRESETS["edge_balanced"]
        components = engine.get_kpu_components(config)
        assert len(components) >= 8
        for comp in components:
            assert "component_type" in comp
            assert "module_name" in comp
            assert "params" in comp

    def test_unknown_component_raises(self):
        engine = RTLTemplateEngine()
        with pytest.raises(ValueError, match="Unknown"):
            engine.render("nonexistent_component")
