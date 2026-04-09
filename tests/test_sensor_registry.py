"""Tests for the SensorRegistry with TF-IDF matching (issue #59)."""

from embodied_ai_architect.sensors import SensorDefinition, SensorMatchResult, SensorRegistry


class TestSensorRegistryLoad:
    """Loading 80 sensor YAML definitions."""

    def test_loads_all_sensors(self):
        registry = SensorRegistry()
        registry.load()
        assert registry.sensor_count >= 80

    def test_auto_loads_on_first_access(self):
        registry = SensorRegistry()
        # sensor_count triggers _ensure_loaded
        assert registry.sensor_count >= 80

    def test_all_categories_represented(self):
        registry = SensorRegistry()
        categories = {s.category for s in registry.list_sensors()}
        for cat in [
            "visual",
            "ranging",
            "inertial",
            "position",
            "environmental",
            "force",
            "audio",
            "biological",
        ]:
            assert cat in categories, f"Category '{cat}' not represented"

    def test_get_known_sensor(self):
        registry = SensorRegistry()
        sensor = registry.get("visual.rgb_camera")
        assert sensor is not None
        assert sensor.name == "RGB Camera"
        assert sensor.category == "visual"

    def test_get_nonexistent_returns_none(self):
        registry = SensorRegistry()
        assert registry.get("nonexistent.sensor") is None

    def test_list_by_category(self):
        registry = SensorRegistry()
        visual = registry.list_by_category("visual")
        assert len(visual) >= 10
        assert all(s.category == "visual" for s in visual)

    def test_list_sensors_with_modality_filter(self):
        registry = SensorRegistry()
        ranging = registry.list_sensors(modality="ranging")
        assert len(ranging) >= 5
        assert all(s.category == "ranging" for s in ranging)


class TestSensorRegistrySearch:
    """TF-IDF search returns ranked results."""

    def test_stereo_camera_for_vio(self):
        """The headline acceptance criterion from the issue."""
        registry = SensorRegistry()
        results = registry.search("stereo camera for VIO", top_k=5)
        assert len(results) >= 1
        # Stereo camera should be in top 3
        top_ids = [r.sensor_id for r in results[:3]]
        assert "visual.stereo_camera" in top_ids

    def test_lidar_for_autonomous_driving(self):
        registry = SensorRegistry()
        results = registry.search("lidar for autonomous driving")
        assert len(results) >= 1
        # A 3D LiDAR should rank high
        top_types = [r.sensor.sensor_type for r in results[:3]]
        assert any("lidar" in t for t in top_types)

    def test_imu_for_drone(self):
        registry = SensorRegistry()
        results = registry.search("IMU for drone navigation")
        assert len(results) >= 1
        top_ids = [r.sensor_id for r in results[:3]]
        assert any("imu" in sid for sid in top_ids)

    def test_empty_query_returns_empty(self):
        registry = SensorRegistry()
        results = registry.search("")
        assert results == []

    def test_top_k_limits_results(self):
        registry = SensorRegistry()
        results = registry.search("camera", top_k=3)
        assert len(results) <= 3

    def test_top_k_zero_returns_empty(self):
        registry = SensorRegistry()
        assert registry.search("camera", top_k=0) == []

    def test_results_are_sorted_by_score(self):
        registry = SensorRegistry()
        results = registry.search("thermal infrared camera")
        if len(results) >= 2:
            scores = [r.score for r in results]
            assert scores == sorted(scores, reverse=True)

    def test_match_result_has_matched_keywords(self):
        registry = SensorRegistry()
        results = registry.search("stereo camera depth")
        if results:
            assert len(results[0].matched_keywords) >= 1

    def test_search_returns_sensor_match_result(self):
        registry = SensorRegistry()
        results = registry.search("GPS RTK")
        if results:
            assert isinstance(results[0], SensorMatchResult)
            assert isinstance(results[0].sensor, SensorDefinition)
            assert results[0].score > 0


class TestSensorDefinition:
    """The sensor model loaded from YAML."""

    def test_all_keywords_flattens(self):
        sensor = SensorDefinition(
            id="test.sensor",
            name="Test",
            keywords={
                "identity": ["test sensor", "ts"],
                "application": ["testing", "validation"],
            },
        )
        kws = sensor.all_keywords()
        assert "test sensor" in kws
        assert "ts" in kws
        assert "testing" in kws
        assert len(kws) == 4

    def test_sensor_with_reference_products(self):
        registry = SensorRegistry()
        sensor = registry.get("visual.stereo_camera")
        if sensor:
            assert isinstance(sensor.reference_products, list)
            # The stereo_camera YAML has Intel RealSense and Stereolabs
            if sensor.reference_products:
                assert any(
                    "stereolabs" in str(p).lower() or "intel" in str(p).lower()
                    for p in sensor.reference_products
                )
