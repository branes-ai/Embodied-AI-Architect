"""Performance baseline tests (issue #151).

Establishes timing baselines for registry loading and search operations.
Assertions use generous 2-3x multipliers to avoid flaky CI failures
while still catching catastrophic regressions.
"""

import time

import pytest

pytestmark = [pytest.mark.slow, pytest.mark.cli]


class TestRegistryLoadTimes:
    """Registry loading should complete within reasonable bounds."""

    def test_platform_registry_load(self):
        """328 platforms + 62 products + 36 categories: <20s."""
        from embodied_ai_architect.platforms import PlatformRegistry

        t0 = time.perf_counter()
        registry = PlatformRegistry()
        registry.load()
        elapsed = time.perf_counter() - t0

        assert registry.platform_count >= 300
        assert elapsed < 20, f"Platform registry took {elapsed:.1f}s (limit: 20s)"

    def test_sensor_registry_load(self):
        """80 sensors: <3s."""
        from embodied_ai_architect.sensors import SensorRegistry

        t0 = time.perf_counter()
        registry = SensorRegistry()
        registry.load()
        elapsed = time.perf_counter() - t0

        assert registry.sensor_count >= 80
        assert elapsed < 3, f"Sensor registry took {elapsed:.1f}s (limit: 3s)"

    def test_actuator_registry_load(self):
        """80 actuators: <3s."""
        from embodied_ai_architect.actuators import ActuatorRegistry

        t0 = time.perf_counter()
        registry = ActuatorRegistry()
        registry.load()
        elapsed = time.perf_counter() - t0

        assert registry.actuator_count >= 80
        assert elapsed < 3, f"Actuator registry took {elapsed:.1f}s (limit: 3s)"


class TestSearchPerformance:
    """TF-IDF search should be fast after initial load."""

    def test_platform_search_latency(self):
        """Single platform search: <200ms."""
        from embodied_ai_architect.platforms import PlatformRegistry

        registry = PlatformRegistry()
        registry.load()

        t0 = time.perf_counter()
        results = registry.search("delivery drone for packages", top_k=5)
        elapsed = time.perf_counter() - t0

        assert len(results) >= 1
        assert elapsed < 0.2, f"Platform search took {elapsed*1000:.0f}ms (limit: 200ms)"

    def test_sensor_search_latency(self):
        """Single sensor search: <100ms."""
        from embodied_ai_architect.sensors import SensorRegistry

        registry = SensorRegistry()
        registry.load()

        t0 = time.perf_counter()
        results = registry.search("stereo camera for VIO", top_k=5)
        elapsed = time.perf_counter() - t0

        assert len(results) >= 1
        assert elapsed < 0.1, f"Sensor search took {elapsed*1000:.0f}ms (limit: 100ms)"

    def test_actuator_search_latency(self):
        """Single actuator search: <100ms."""
        from embodied_ai_architect.actuators import ActuatorRegistry

        registry = ActuatorRegistry()
        registry.load()

        t0 = time.perf_counter()
        results = registry.search("brushless motor for drone", top_k=5)
        elapsed = time.perf_counter() - t0

        assert len(results) >= 1
        assert elapsed < 0.1, f"Actuator search took {elapsed*1000:.0f}ms (limit: 100ms)"

    def test_batch_search_throughput(self):
        """15 searches across all registries: <1s (excluding load)."""
        from embodied_ai_architect.actuators import ActuatorRegistry
        from embodied_ai_architect.platforms import PlatformRegistry
        from embodied_ai_architect.sensors import SensorRegistry

        # Load registries first (not timed)
        pr = PlatformRegistry()
        pr.load()
        sr = SensorRegistry()
        sr.load()
        ar = ActuatorRegistry()
        ar.load()

        queries = [
            "delivery drone",
            "stereo camera for VIO",
            "brushless motor",
            "surgical robot",
            "warehouse AMR",
        ]

        # Time only the search operations
        t0 = time.perf_counter()
        for q in queries:
            pr.search(q, top_k=5)
            sr.search(q, top_k=5)
            ar.search(q, top_k=5)
        elapsed = time.perf_counter() - t0

        assert elapsed < 1.0, f"15 searches took {elapsed:.2f}s (limit: 1.0s)"


class TestMissionStoreThroughput:
    """Mission CRUD should be fast."""

    def test_create_and_list_missions(self, tmp_path, monkeypatch):
        """Create 10 missions, list them: <2s total."""
        import embodied_ai_architect.mission.store as store_mod
        from embodied_ai_architect.mission.models import Mission
        from embodied_ai_architect.mission.store import MissionStore

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(store_mod, "DEFAULT_MISSIONS_DIR", tmp_path / ".branes" / "missions")

        store = MissionStore()

        t0 = time.perf_counter()
        for i in range(10):
            m = Mission(name=f"perf-test-{i}", goal=f"Goal {i}")
            store.save(m)
        missions = store.list_missions()
        elapsed = time.perf_counter() - t0

        assert len(missions) == 10
        assert elapsed < 2.0, f"10 mission CRUD took {elapsed:.2f}s (limit: 2.0s)"
