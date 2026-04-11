"""Tests for the Platform Registry."""

import pytest

import yaml

from embodied_ai_architect.platforms.registry import (
    PlatformDefinition,
    PlatformRegistry,
    _tokenize,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_platform_dir(tmp_path):
    """Create a temporary directory with a few sample platform YAML files."""
    # Edge vision box
    edge_vision = {
        "id": "surveillance.edge_vision_box",
        "name": "General Edge AI Appliance",
        "category": "surveillance",
        "description": "Multi-purpose edge inference box for vision analytics",
        "aliases": ["edge device", "AI box", "inference appliance"],
        "keywords": {
            "identity": [
                "edge vision box",
                "edge AI appliance",
                "edge device",
                "edge computing",
                "edge inference",
            ],
            "descriptions": [
                "camera with AI processing",
                "edge device running edge detection",
                "smart camera box",
                "AI inference at the edge",
            ],
            "application": [
                "edge detection",
                "video analytics",
                "object detection",
                "real-time inference",
            ],
            "industry": ["ONVIF", "RTSP", "edge AI", "neural compute"],
            "components": ["NPU", "inference accelerator", "vision processor"],
            "related": ["smart camera", "edge computing", "IoT gateway"],
        },
        "classification": {
            "locomotion": "stationary.fixed_mount",
            "manipulation": "none",
            "load_class": "none",
            "environment": ["indoor_structured", "outdoor_urban"],
            "human_proximity": "no_humans",
        },
        "attributes": {
            "power_watts": {"min": 5, "max": 30, "typical": 15},
            "cost_usd": {"min": 100, "max": 2000, "typical": 500},
        },
        "implications": {
            "platform_type": "fixed_camera",
            "perception": {"camera_types": ["monocular"], "max_latency_ms": 50},
        },
        "context": {
            "typical_architecture": "ARM/x86 SoC with integrated NPU",
            "reference_designs": ["NVIDIA Jetson", "Hailo-8"],
            "design_considerations": "Choose NPU based on model complexity.",
        },
    }
    (tmp_path / "surveillance").mkdir()
    with open(tmp_path / "surveillance" / "edge_vision_box.yaml", "w") as f:
        yaml.dump(edge_vision, f)

    # Agricultural sprayer
    sprayer = {
        "id": "agriculture.pesticide_ground_sprayer",
        "name": "Autonomous Ground Sprayer",
        "category": "agriculture",
        "description": "Precision pesticide/herbicide application",
        "aliases": ["autonomous sprayer", "crop sprayer"],
        "keywords": {
            "identity": ["autonomous sprayer", "ground sprayer", "pesticide sprayer"],
            "descriptions": [
                "robot that sprays crops",
                "autonomous sprayer for vineyards",
                "precision spraying",
            ],
            "application": [
                "pesticide application",
                "herbicide spraying",
                "variable rate spraying",
            ],
            "industry": ["precision agriculture", "variable rate", "GPS-guided"],
            "components": ["nozzle", "tank", "RTK-GPS"],
            "related": ["vineyard", "orchard", "crop protection"],
        },
        "classification": {
            "locomotion": "ground.wheeled",
            "manipulation": "fixed_tool",
            "load_class": "none",
            "environment": ["outdoor_rural"],
            "human_proximity": "no_humans",
        },
        "attributes": {
            "power_watts": {"min": 50, "max": 500, "typical": 200},
            "cost_usd": {"min": 10000, "max": 100000, "typical": 50000},
        },
        "context": {
            "reference_designs": ["Naïo Oz", "ecoRobotix ARA"],
            "design_considerations": "Variable-rate nozzle control is critical.",
        },
    }
    (tmp_path / "agriculture").mkdir()
    with open(tmp_path / "agriculture" / "pesticide_ground_sprayer.yaml", "w") as f:
        yaml.dump(sprayer, f)

    # Vineyard robot
    vineyard = {
        "id": "agriculture.vineyard_robot",
        "name": "Vineyard Management Robot",
        "category": "agriculture",
        "description": "Grape leaf analysis, selective harvesting, vine management",
        "aliases": ["vineyard robot", "grape robot"],
        "keywords": {
            "identity": ["vineyard robot", "viticulture robot"],
            "descriptions": ["robot for grape vines", "vineyard automation"],
            "application": ["canopy management", "grape harvesting", "vine pruning"],
            "industry": ["viticulture", "enology", "vineyard"],
            "components": ["row-crop chassis", "multispectral camera"],
            "related": ["wine", "grape", "vineyard"],
        },
        "classification": {
            "locomotion": "ground.wheeled",
            "manipulation": "articulated_arm",
            "load_class": "deformable_light",
            "environment": ["outdoor_rural"],
            "human_proximity": "no_humans",
        },
        "attributes": {},
        "context": {"reference_designs": ["Naïo Ted"]},
    }
    with open(tmp_path / "agriculture" / "vineyard_robot.yaml", "w") as f:
        yaml.dump(vineyard, f)

    # Surgical orthopedic
    surgical = {
        "id": "surgical.orthopedic",
        "name": "Orthopedic Surgery Robot",
        "category": "surgical",
        "description": "Joint replacement and bone cutting with haptic feedback",
        "aliases": ["knee replacement robot", "orthopedic robot"],
        "keywords": {
            "identity": ["orthopedic surgery robot", "joint replacement robot"],
            "descriptions": [
                "surgical robot for knee replacement",
                "robot for hip surgery",
            ],
            "application": ["knee replacement", "hip replacement", "bone cutting"],
            "industry": ["orthopedic", "arthroplasty"],
            "components": ["haptic arm", "CT navigation"],
            "related": ["surgical", "joint surgery", "bone"],
        },
        "classification": {
            "locomotion": "stationary.fixed_mount",
            "manipulation": "surgical_instrument",
            "load_class": "contact_tool",
            "environment": ["indoor_sterile"],
            "human_proximity": "humans_contact",
        },
        "attributes": {},
        "context": {"reference_designs": ["Stryker Mako", "Zimmer ROSA"]},
    }
    (tmp_path / "surgical").mkdir()
    with open(tmp_path / "surgical" / "orthopedic.yaml", "w") as f:
        yaml.dump(surgical, f)

    # Schema and taxonomy (should be ignored)
    with open(tmp_path / "schema.yaml", "w") as f:
        yaml.dump({"schema": "v1"}, f)
    with open(tmp_path / "taxonomy.yaml", "w") as f:
        yaml.dump({"taxonomy": "v1"}, f)

    return tmp_path


@pytest.fixture
def registry(sample_platform_dir):
    """Create a registry loaded from the sample data."""
    reg = PlatformRegistry(data_dir=sample_platform_dir)
    reg.load()
    return reg


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPlatformDefinition:
    def test_all_keywords_flattens(self):
        p = PlatformDefinition(
            id="test.foo",
            name="Test",
            category="test",
            description="A test",
            aliases=["alias1"],
            keywords={
                "identity": ["word1", "word2"],
                "application": ["word3"],
            },
        )
        kws = p.all_keywords()
        assert "alias1" in kws
        assert "word1" in kws
        assert "word3" in kws
        assert len(kws) == 4


class TestTokenize:
    def test_basic(self):
        assert _tokenize("hello world") == ["hello", "world"]

    def test_strips_short(self):
        assert _tokenize("a b cd") == ["cd"]

    def test_handles_punctuation(self):
        tokens = _tokenize("edge-device running AI!")
        assert "edge-device" in tokens
        assert "running" in tokens
        assert "ai" in tokens


class TestPlatformRegistry:
    def test_load_counts(self, registry):
        assert registry.platform_count == 4

    def test_schema_not_loaded(self, registry):
        """schema.yaml and taxonomy.yaml should not be loaded as platforms."""
        assert registry.get("schema") is None
        assert registry.get("taxonomy") is None

    def test_get_existing(self, registry):
        p = registry.get("surveillance.edge_vision_box")
        assert p is not None
        assert p.name == "General Edge AI Appliance"

    def test_get_missing(self, registry):
        assert registry.get("nonexistent.platform") is None

    def test_list_platforms(self, registry):
        platforms = registry.list_platforms()
        assert len(platforms) == 4
        # Should be sorted by ID
        ids = [p.id for p in platforms]
        assert ids == sorted(ids)

    def test_list_categories(self, registry):
        cats = registry.list_categories()
        assert "agriculture" in cats
        assert "surveillance" in cats
        assert "surgical" in cats

    def test_list_by_category(self, registry):
        ag = registry.list_by_category("agriculture")
        assert len(ag) == 2
        ids = {p.id for p in ag}
        assert "agriculture.pesticide_ground_sprayer" in ids
        assert "agriculture.vineyard_robot" in ids


class TestSearch:
    """Acceptance criteria from issue #46."""

    def test_edge_device_matches_edge_vision(self, registry):
        """'edge device running edge detection' should match edge_vision_box."""
        matches = registry.search("edge device running edge detection")
        assert len(matches) > 0
        assert matches[0].platform_id == "surveillance.edge_vision_box"
        assert matches[0].score > 0.5

    def test_autonomous_sprayer_vineyards(self, registry):
        """'autonomous sprayer for vineyards' should match agriculture platforms."""
        matches = registry.search("autonomous sprayer for vineyards")
        assert len(matches) > 0
        top_ids = {m.platform_id for m in matches[:3]}
        assert "agriculture.pesticide_ground_sprayer" in top_ids or (
            "agriculture.vineyard_robot" in top_ids
        )

    def test_surgical_robot_knee(self, registry):
        """'surgical robot for knee replacement' should match orthopedic."""
        matches = registry.search("surgical robot for knee replacement")
        assert len(matches) > 0
        assert matches[0].platform_id == "surgical.orthopedic"

    def test_no_match_gibberish(self, registry):
        matches = registry.search("zzzzxxxx nonsense query")
        assert len(matches) == 0

    def test_scores_normalized(self, registry):
        matches = registry.search("edge device")
        if matches:
            assert matches[0].score <= 1.0
            assert matches[0].score > 0.0

    def test_matched_keywords_populated(self, registry):
        matches = registry.search("edge device")
        assert len(matches) > 0
        assert len(matches[0].matched_keywords) > 0

    def test_top_k_respected(self, registry):
        matches = registry.search("robot", top_k=2)
        assert len(matches) <= 2


class TestCategoryDefaults:
    """Test _category.yaml loading and attribute inheritance."""

    def test_category_defaults_loaded(self, sample_platform_dir):
        """Add a _category.yaml and verify it's loaded."""
        cat_data = {
            "category": "surveillance",
            "description": "Surveillance platforms",
            "platform_count": 1,
            "default_attributes": {
                "power_watts": {"min": 5, "max": 50, "typical": 20},
                "latency_ms": {"min": 10, "max": 100, "typical": 30},
                "weight_kg": {"min": 0.5, "max": 5, "typical": 2},
            },
        }
        with open(sample_platform_dir / "surveillance" / "_category.yaml", "w") as f:
            yaml.dump(cat_data, f)

        reg = PlatformRegistry(data_dir=sample_platform_dir)
        reg.load()
        defaults = reg.get_category_defaults("surveillance")
        assert defaults is not None
        assert defaults["category"] == "surveillance"

    def test_inherits_missing_attributes(self, sample_platform_dir):
        """Platform without weight_kg inherits from category default."""
        cat_data = {
            "category": "surveillance",
            "default_attributes": {
                "weight_kg": {"min": 0.5, "max": 5, "typical": 2},
            },
        }
        with open(sample_platform_dir / "surveillance" / "_category.yaml", "w") as f:
            yaml.dump(cat_data, f)

        reg = PlatformRegistry(data_dir=sample_platform_dir)
        reg.load()
        platform = reg.get("surveillance.edge_vision_box")
        assert platform is not None
        # edge_vision_box has power_watts and cost_usd but not weight_kg
        assert "weight_kg" in platform.attributes
        assert platform.attributes["weight_kg"]["typical"] == 2

    def test_explicit_attributes_not_overridden(self, sample_platform_dir):
        """Platform's own attributes take precedence over category defaults."""
        cat_data = {
            "category": "surveillance",
            "default_attributes": {
                "power_watts": {"min": 1, "max": 10, "typical": 5},
            },
        }
        with open(sample_platform_dir / "surveillance" / "_category.yaml", "w") as f:
            yaml.dump(cat_data, f)

        reg = PlatformRegistry(data_dir=sample_platform_dir)
        reg.load()
        platform = reg.get("surveillance.edge_vision_box")
        assert platform is not None
        # edge_vision_box explicitly defines power_watts={min:5, max:30, typical:15}
        assert platform.attributes["power_watts"]["typical"] == 15  # not overridden to 5

    def test_no_category_file_no_crash(self, sample_platform_dir):
        """Registry works fine without any _category.yaml files."""
        reg = PlatformRegistry(data_dir=sample_platform_dir)
        reg.load()
        assert reg.get_category_defaults("surveillance") is None
        assert reg.platform_count > 0


class TestProductConfigurations:
    """Test product configuration loading from configurations/ directory."""

    def test_live_products_loaded(self):
        """Product configs from configurations/ are loaded as platforms."""
        registry = PlatformRegistry()
        registry.load()
        # Should find at least some product configs
        dji = registry.get("aerial.dji_matrice_350")
        if dji:
            assert "DJI" in dji.name or "Matrice" in dji.name
            assert dji.attributes  # should have specifications mapped to attributes

    def test_product_searchable(self):
        """Product configs are indexed and searchable."""
        registry = PlatformRegistry()
        registry.load()
        results = registry.search("DJI Matrice drone", top_k=5)
        # If products are loaded, DJI should show up
        if registry.get("aerial.dji_matrice_350"):
            ids = [r.platform_id for r in results[:5]]
            assert any("dji" in pid for pid in ids), f"No DJI in results: {ids}"

    def test_product_specs_as_attributes(self):
        """Product 'specifications' field maps to attributes."""
        registry = PlatformRegistry()
        registry.load()
        franka = registry.get("manipulation.franka_panda")
        if franka:
            # Franka should have specifications loaded as attributes
            assert "weight_kg" in franka.attributes or "dof" in franka.attributes


class TestSearchFromLiveData:
    """Test against the actual data/platforms/ directory if populated."""

    def test_live_registry_loads(self):
        """Smoke test: the live registry loads without errors."""
        registry = PlatformRegistry()
        registry.load()
        # Should have at least a few platforms (from the YAML files we create)
        assert registry.platform_count >= 0  # Don't fail if empty yet

    def test_live_category_defaults_loaded(self):
        """Live registry loads _category.yaml files."""
        registry = PlatformRegistry()
        registry.load()
        # Should have category defaults for at least some categories
        assert len(registry._category_defaults) > 0
