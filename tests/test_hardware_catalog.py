"""Tests for the dynamic hardware catalog (graphs#136 Phase 5).

Locks in the contract that ``list_available_hardware`` and the
``HARDWARE_CATALOG`` / ``ALL_HARDWARE`` module-level names live-source
from ``graphs.hardware.mappers`` so the orchestrator's hardware list
always matches what the analysis tools can actually accept.

Skipped when graphs isn't installed in the test environment -- the
fallback path is exercised separately.
"""

import json

import pytest

from embodied_ai_architect.llm.graphs_tools import (
    HAS_GRAPHS,
    _FALLBACK_HARDWARE_CATALOG,
    _get_hardware_catalog,
    list_available_hardware,
)


@pytest.mark.skipif(not HAS_GRAPHS, reason="graphs package not installed")
class TestDynamicHardwareCatalog:
    """Live-source path: graphs is available, catalog comes from the
    registry. These tests cover the happy path plus the profile-alias
    expansion that Phase 5 of branes-ai/graphs#136 added."""

    def test_default_catalog_is_silicon_bins_only(self):
        catalog = _get_hardware_catalog(include_profile_aliases=False)
        # No alias entries (those contain '@').
        for entries in catalog.values():
            for sku in entries:
                assert "@" not in sku

    def test_default_catalog_has_real_jetson_names(self):
        # Regression guard for the pre-Phase-5 catalog drift, where
        # 'Jetson-Orin-AGX' (without size) and 'V100-SXM2-32GB' were
        # listed -- neither matches the graphs registry.
        catalog = _get_hardware_catalog(include_profile_aliases=False)
        gpu = catalog.get("gpu", [])
        assert "Jetson-Orin-AGX-64GB" in gpu
        assert "Jetson-Orin-Nano-8GB" in gpu
        assert "Jetson-Orin-NX-16GB" in gpu

    def test_expanded_catalog_includes_profile_aliases(self):
        catalog = _get_hardware_catalog(include_profile_aliases=True)
        flat = [sku for entries in catalog.values() for sku in entries]
        # Orin Nano's 4 modes (7W / 15W / 25W / MAXN per graphs#136
        # Phase 1) all surface in the expanded view.
        assert "Jetson-Orin-Nano-8GB@7W" in flat
        assert "Jetson-Orin-Nano-8GB@15W" in flat
        assert "Jetson-Orin-Nano-8GB@25W" in flat
        assert "Jetson-Orin-Nano-8GB@MAXN" in flat

    def test_expanded_catalog_has_more_entries_than_default(self):
        default = _get_hardware_catalog(include_profile_aliases=False)
        expanded = _get_hardware_catalog(include_profile_aliases=True)
        d_count = sum(len(v) for v in default.values())
        e_count = sum(len(v) for v in expanded.values())
        # Every silicon-bin contributes >= 1 entry to expanded; chips
        # with multiple thermal_operating_points contribute more. So
        # expanded must be strictly larger.
        assert e_count > d_count

    def test_aliases_inherit_silicon_category(self):
        # An alias of a 'gpu' silicon (e.g., Jetson-Orin-Nano-8GB) must
        # also be classified under 'gpu' -- the alias mechanism doesn't
        # invent new categories. Regression guard against the bucket
        # logic in _get_hardware_catalog.
        catalog = _get_hardware_catalog(include_profile_aliases=True)
        gpu = catalog.get("gpu", [])
        assert "Jetson-Orin-Nano-8GB@7W" in gpu

    def test_categories_are_sorted_for_determinism(self):
        # LLM consumes the JSON output -- stable ordering across runs
        # avoids triggering spurious cache misses in prompt caches.
        catalog = _get_hardware_catalog(include_profile_aliases=False)
        for category, entries in catalog.items():
            assert entries == sorted(
                entries
            ), f"Category {category!r} entries not sorted: {entries}"


@pytest.mark.skipif(not HAS_GRAPHS, reason="graphs package not installed")
class TestListAvailableHardwareTool:
    """The LLM-callable wrapper around _get_hardware_catalog."""

    def test_default_call_returns_silicon_bins_only(self):
        result = json.loads(list_available_hardware())
        assert result["include_profile_aliases"] is False
        # Every entry across categories is a silicon-bin name (no '@').
        for entries in result["categories"].values():
            assert all("@" not in sku for sku in entries)

    def test_include_profile_aliases_flag_expands(self):
        bare = json.loads(list_available_hardware())
        expanded = json.loads(list_available_hardware(include_profile_aliases=True))
        assert expanded["include_profile_aliases"] is True
        assert expanded["total_hardware_targets"] > bare["total_hardware_targets"]

    def test_category_filter_for_gpu_returns_jetsons(self):
        result = json.loads(list_available_hardware(category="gpu"))
        assert result["category"] == "gpu"
        # At least one Jetson present.
        assert any("Jetson-Orin" in sku for sku in result["hardware"])

    def test_category_filter_with_aliases(self):
        result = json.loads(list_available_hardware(category="gpu", include_profile_aliases=True))
        # Profile aliases now visible in the gpu bucket.
        assert any("@" in sku for sku in result["hardware"])

    def test_unknown_category_returns_clear_error(self):
        result = json.loads(list_available_hardware(category="not-a-category"))
        assert "error" in result
        assert "available_categories" in result


class TestFallbackCatalog:
    """When ``graphs`` isn't available, ``_get_hardware_catalog`` returns
    the static fallback. Test that path independently of HAS_GRAPHS so
    it runs regardless of the test environment's optional-dep state.

    We do this by patching the HAS_GRAPHS flag in the module, since
    the function checks it lazily.
    """

    def test_fallback_catalog_used_when_graphs_unavailable(self, monkeypatch):
        from embodied_ai_architect.llm import graphs_tools as gt

        monkeypatch.setattr(gt, "HAS_GRAPHS", False)
        result = gt._get_hardware_catalog()
        # Returns the fallback verbatim (same dict identity).
        assert result is gt._FALLBACK_HARDWARE_CATALOG

    def test_fallback_catalog_has_real_jetson_names(self):
        # Even the fallback uses the registry-correct names so users
        # don't get a confusingly-different list when graphs is missing.
        flat = [sku for entries in _FALLBACK_HARDWARE_CATALOG.values() for sku in entries]
        assert "Jetson-Orin-Nano-8GB" in flat
        assert "Jetson-Orin-AGX-64GB" in flat
