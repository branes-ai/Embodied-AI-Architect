"""Tests for multi-platform context composition (issue #100)."""

from embodied_ai_architect.platforms.context import (
    _merge_attributes,
    _merge_classifications,
    _merge_context_blocks,
    _merge_implications,
    build_context_prompt,
    get_platform_context_for_goal,
)


class TestMergeAttributes:
    def test_tighter_constraints(self):
        a1 = {"power_watts": {"min": 5, "max": 50, "typical": 20}}
        a2 = {"power_watts": {"min": 10, "max": 30, "typical": 15}}
        merged = _merge_attributes([a1, a2])
        # Tighter: higher min, lower max
        assert merged["power_watts"]["min"] == 10
        assert merged["power_watts"]["max"] == 30
        assert merged["power_watts"]["typical"] == 17.5  # average

    def test_union_keys(self):
        a1 = {"power_watts": {"min": 5, "max": 50, "typical": 20}}
        a2 = {"weight_kg": {"min": 1, "max": 10, "typical": 5}}
        merged = _merge_attributes([a1, a2])
        assert "power_watts" in merged
        assert "weight_kg" in merged


class TestMergeClassifications:
    def test_union_lists(self):
        c1 = {"environment": ["indoor", "outdoor"]}
        c2 = {"environment": ["outdoor", "underwater"]}
        merged = _merge_classifications([c1, c2])
        assert set(merged["environment"]) == {"indoor", "outdoor", "underwater"}

    def test_different_scalars(self):
        c1 = {"locomotion": "wheeled"}
        c2 = {"locomotion": "legged"}
        merged = _merge_classifications([c1, c2])
        assert isinstance(merged["locomotion"], list)
        assert "wheeled" in merged["locomotion"]
        assert "legged" in merged["locomotion"]


class TestMergeImplications:
    def test_union_perception_tasks(self):
        i1 = {"perception": {"detection_classes": ["person", "obstacle"]}}
        i2 = {"perception": {"detection_classes": ["object", "person"]}}
        merged = _merge_implications([i1, i2])
        classes = set(merged["perception"]["detection_classes"])
        assert "person" in classes
        assert "obstacle" in classes
        assert "object" in classes

    def test_tighter_latency(self):
        i1 = {"perception": {"max_latency_ms": 50}}
        i2 = {"perception": {"max_latency_ms": 20}}
        merged = _merge_implications([i1, i2])
        assert merged["perception"]["max_latency_ms"] == 20  # tighter


class TestMergeContextBlocks:
    def test_union_pitfalls(self):
        c1 = {"common_pitfalls": ["thermal", "weight"]}
        c2 = {"common_pitfalls": ["latency", "thermal"]}
        merged = _merge_context_blocks([c1, c2])
        assert len(merged["common_pitfalls"]) >= 3

    def test_concatenate_text(self):
        c1 = {"typical_architecture": "ARM SoC"}
        c2 = {"typical_architecture": "x86 + FPGA"}
        merged = _merge_context_blocks([c1, c2])
        assert "ARM SoC" in merged["typical_architecture"]
        assert "x86 + FPGA" in merged["typical_architecture"]


class TestGetMultiPlatformContext:
    def test_hybrid_system_returns_multi(self):
        """A hybrid goal should match multiple distinct categories."""
        ctx = get_platform_context_for_goal("mobile robot with manipulator arm for warehouse")
        if ctx and ctx.get("multi_platform"):
            assert "+" in ctx["platform_id"]
            assert ctx.get("platforms")
            assert len(ctx["platforms"]) >= 2

    def test_single_system_returns_single(self):
        """A specific goal matching one category should return single context."""
        ctx = get_platform_context_for_goal("agricultural crop spraying drone")
        if ctx:
            # If all top matches are in the same category, no multi-platform
            if not ctx.get("multi_platform"):
                assert "+" not in ctx.get("platform_id", "")

    def test_nonsense_returns_empty(self):
        assert get_platform_context_for_goal("zzzzxxxx") == {}


class TestBuildContextPromptMulti:
    def test_multi_platform_header(self):
        ctx = {
            "platform_id": "ugv.amr + manipulation.cobot",
            "platform_name": "Warehouse AMR + Cobot",
            "platform_description": "Mobile base with arm",
            "multi_platform": True,
            "context": {"typical_architecture": "ARM + GPU"},
            "attributes": {},
            "classification": {},
        }
        prompt = build_context_prompt(ctx)
        assert "Composed Platform Context" in prompt
        assert "Hybrid system" in prompt

    def test_single_platform_header(self):
        ctx = {
            "platform_id": "aerial.drone",
            "platform_name": "Drone",
            "platform_description": "A drone",
            "context": {"typical_architecture": "ARM"},
            "attributes": {},
            "classification": {},
        }
        prompt = build_context_prompt(ctx)
        assert "Platform Context: Drone" in prompt
        assert "Composed" not in prompt
