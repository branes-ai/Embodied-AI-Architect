"""Tests for the ActuatorRegistry with TF-IDF matching (issue #62)."""

from embodied_ai_architect.actuators import (
    ActuatorDefinition,
    ActuatorMatchResult,
    ActuatorRegistry,
)


class TestActuatorRegistryLoad:
    """Loading 80 actuator YAML definitions."""

    def test_loads_all_actuators(self):
        registry = ActuatorRegistry()
        registry.load()
        assert registry.actuator_count >= 80

    def test_auto_loads_on_first_access(self):
        registry = ActuatorRegistry()
        assert registry.actuator_count >= 80

    def test_all_categories_represented(self):
        registry = ActuatorRegistry()
        categories = {a.category for a in registry.list_actuators()}
        for cat in [
            "motor",
            "hydraulic",
            "pneumatic",
            "gripper",
            "locomotion",
            "fluid",
            "display",
            "specialty",
        ]:
            assert cat in categories, f"Category '{cat}' not represented"

    def test_get_known_actuator(self):
        registry = ActuatorRegistry()
        actuator = registry.get("motor.brushless_dc")
        assert actuator is not None
        assert actuator.name == "Brushless DC Motor"
        assert actuator.category == "motor"

    def test_get_nonexistent_returns_none(self):
        registry = ActuatorRegistry()
        assert registry.get("nonexistent.actuator") is None

    def test_list_by_category(self):
        registry = ActuatorRegistry()
        motors = registry.list_by_category("motor")
        assert len(motors) >= 6
        assert all(a.category == "motor" for a in motors)

    def test_list_actuators_with_category_filter(self):
        registry = ActuatorRegistry()
        grippers = registry.list_actuators(category="gripper")
        assert len(grippers) >= 5
        assert all(a.category == "gripper" for a in grippers)


class TestActuatorRegistrySearch:
    """TF-IDF search returns ranked results."""

    def test_gripper_for_fragile_objects(self):
        """The headline acceptance criterion from the issue."""
        registry = ActuatorRegistry()
        results = registry.search("gripper for fragile objects", top_k=5)
        assert len(results) >= 1
        # Soft gripper should be in top 3
        top_ids = [r.actuator_id for r in results[:3]]
        assert any("soft" in aid for aid in top_ids), f"No soft gripper in top 3: {top_ids}"

    def test_brushless_motor_for_drone(self):
        registry = ActuatorRegistry()
        results = registry.search("brushless motor for drone propulsion")
        assert len(results) >= 1
        top_ids = [r.actuator_id for r in results[:3]]
        assert any("brushless" in aid for aid in top_ids)

    def test_servo_for_robot_joint(self):
        registry = ActuatorRegistry()
        results = registry.search("servo for robot joint")
        assert len(results) >= 1
        top_ids = [r.actuator_id for r in results[:3]]
        assert any("servo" in aid for aid in top_ids)

    def test_empty_query_returns_empty(self):
        registry = ActuatorRegistry()
        results = registry.search("")
        assert results == []

    def test_top_k_limits_results(self):
        registry = ActuatorRegistry()
        results = registry.search("motor", top_k=3)
        assert len(results) <= 3

    def test_top_k_zero_returns_empty(self):
        registry = ActuatorRegistry()
        assert registry.search("motor", top_k=0) == []

    def test_results_are_sorted_by_score(self):
        registry = ActuatorRegistry()
        results = registry.search("hydraulic cylinder high force")
        if len(results) >= 2:
            scores = [r.score for r in results]
            assert scores == sorted(scores, reverse=True)

    def test_match_result_has_matched_keywords(self):
        registry = ActuatorRegistry()
        results = registry.search("parallel gripper")
        if results:
            assert len(results[0].matched_keywords) >= 1

    def test_search_returns_actuator_match_result(self):
        registry = ActuatorRegistry()
        results = registry.search("stepper motor")
        if results:
            assert isinstance(results[0], ActuatorMatchResult)
            assert isinstance(results[0].actuator, ActuatorDefinition)
            assert results[0].score > 0


class TestActuatorDefinition:
    """The actuator model loaded from YAML."""

    def test_all_keywords_flattens(self):
        actuator = ActuatorDefinition(
            id="test.actuator",
            name="Test",
            keywords={
                "identity": ["test actuator", "ta"],
                "application": ["testing", "validation"],
            },
        )
        kws = actuator.all_keywords()
        assert "test actuator" in kws
        assert "ta" in kws
        assert "testing" in kws
        assert len(kws) == 4

    def test_actuator_with_reference_products(self):
        registry = ActuatorRegistry()
        actuator = registry.get("motor.servo")
        if actuator:
            assert isinstance(actuator.reference_products, list)
            if actuator.reference_products:
                assert any(
                    "dynamixel" in str(p).lower() or "maxon" in str(p).lower()
                    for p in actuator.reference_products
                )
