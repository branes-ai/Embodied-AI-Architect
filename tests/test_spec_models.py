"""Tests for spec models, events, and path utilities."""

import pytest

from embodied_ai_architect.specs.models import (
    ActuatorSpec,
    AutonomyLevel,
    AutonomySpec,
    CommsSpec,
    ComputeSpec,
    CoolingType,
    EnvironmentalRating,
    PerceptionSpec,
    PlatformType,
    PowerSpec,
    SafetyLevel,
    SafetySpec,
    SensorSpec,
    SystemSpec,
    delete_at_path,
    get_at_path,
    parse_path,
    set_at_path,
)
from embodied_ai_architect.specs.events import EventLog, EventOp, SpecEvent, make_event
from embodied_ai_architect.specs.exceptions import InvalidPathError
from embodied_ai_architect.specs.diff import ChangeType, diff_specs
from embodied_ai_architect.specs.validation import Severity, validate_spec
from embodied_ai_architect.specs.templates import get_template, get_templates

# ── Model Serialization ───────────────────────────────────────────────────


class TestSystemSpec:
    def test_empty_spec(self):
        spec = SystemSpec(name="test")
        assert spec.name == "test"
        assert spec.perception is None
        assert spec.compute is None
        assert spec.subsystem_names() == []

    def test_full_spec_round_trip(self):
        spec = SystemSpec(
            name="full-test",
            description="A test spec",
            platform_type=PlatformType.DRONE,
            perception=PerceptionSpec(
                cameras=2,
                min_fps=30.0,
                detection_classes=["person", "vehicle"],
            ),
            compute=ComputeSpec(
                soc="Jetson Orin NX",
                memory_gb=8.0,
                max_tdp_watts=15.0,
            ),
            power=PowerSpec(
                battery_wh=99.0,
                power_budget_watts=25.0,
            ),
            tags=["test", "drone"],
        )
        data = spec.model_dump(mode="json")
        restored = SystemSpec.model_validate(data)
        assert restored.name == "full-test"
        assert restored.platform_type == PlatformType.DRONE
        assert restored.perception.cameras == 2
        assert restored.perception.min_fps == 30.0
        assert restored.compute.soc == "Jetson Orin NX"
        assert restored.tags == ["test", "drone"]

    def test_subsystem_names(self):
        spec = SystemSpec(
            name="test",
            perception=PerceptionSpec(),
            compute=ComputeSpec(),
        )
        assert spec.subsystem_names() == ["perception", "compute"]

    def test_summary(self):
        spec = SystemSpec(
            name="drone-1",
            platform_type=PlatformType.DRONE,
            perception=PerceptionSpec(),
            tags=["outdoor"],
        )
        summary = spec.summary()
        assert "drone-1" in summary
        assert "drone" in summary
        assert "perception" in summary

    def test_exclude_none_serialization(self):
        spec = SystemSpec(name="minimal")
        data = spec.model_dump(exclude_none=True, mode="json")
        assert "perception" not in data
        assert "compute" not in data
        assert data["name"] == "minimal"

    def test_all_subsystems(self):
        spec = SystemSpec(
            name="all",
            perception=PerceptionSpec(),
            compute=ComputeSpec(),
            power=PowerSpec(),
            sensors=SensorSpec(),
            actuators=ActuatorSpec(),
            comms=CommsSpec(),
            autonomy=AutonomySpec(),
            safety=SafetySpec(),
        )
        assert len(spec.subsystem_names()) == 8


class TestEnums:
    def test_platform_type_values(self):
        assert PlatformType.DRONE.value == "drone"
        assert PlatformType.QUADRUPED.value == "quadruped"

    def test_safety_level_values(self):
        assert SafetyLevel.SIL_2.value == "sil_2"

    def test_autonomy_level_values(self):
        assert AutonomyLevel.FULL.value == "full"


# ── Path Utilities ─────────────────────────────────────────────────────────


class TestPathUtilities:
    def test_parse_path(self):
        assert parse_path("/perception/min_fps") == ["perception", "min_fps"]
        assert parse_path("/tags") == ["tags"]
        assert parse_path("/custom/my_field") == ["custom", "my_field"]

    def test_parse_path_invalid(self):
        with pytest.raises(InvalidPathError):
            parse_path("no-leading-slash")
        with pytest.raises(InvalidPathError):
            parse_path("/")

    def test_get_at_path(self):
        spec = SystemSpec(
            name="test",
            perception=PerceptionSpec(min_fps=30.0),
        )
        assert get_at_path(spec, "/perception/min_fps") == 30.0
        assert get_at_path(spec, "/name") == "test"

    def test_get_at_path_nested(self):
        spec = SystemSpec(
            name="test",
            perception=PerceptionSpec(detection_classes=["person", "car"]),
        )
        result = get_at_path(spec, "/perception/detection_classes")
        assert result == ["person", "car"]

    def test_get_at_path_invalid(self):
        spec = SystemSpec(name="test")
        with pytest.raises(InvalidPathError):
            get_at_path(spec, "/perception/min_fps")  # perception is None

    def test_set_at_path(self):
        spec = SystemSpec(name="test")
        new_spec = set_at_path(spec, "/perception/min_fps", 60)
        assert new_spec.perception is not None
        assert new_spec.perception.min_fps == 60

    def test_set_at_path_creates_subsystem(self):
        spec = SystemSpec(name="test")
        new_spec = set_at_path(spec, "/compute/soc", "Jetson Orin NX")
        assert new_spec.compute is not None
        assert new_spec.compute.soc == "Jetson Orin NX"

    def test_set_at_path_coercion(self):
        spec = SystemSpec(name="test")
        # String "true" should coerce to bool True
        new_spec = set_at_path(spec, "/perception/tracking", "true")
        assert new_spec.perception.tracking is True

        # Numeric strings should coerce
        new_spec = set_at_path(spec, "/perception/min_fps", "30")
        assert new_spec.perception.min_fps == 30.0

    def test_delete_at_path(self):
        spec = SystemSpec(
            name="test",
            perception=PerceptionSpec(min_fps=30.0, cameras=2),
        )
        new_spec = delete_at_path(spec, "/perception/min_fps")
        assert new_spec.perception.min_fps is None
        assert new_spec.perception.cameras == 2

    def test_set_at_path_top_level(self):
        spec = SystemSpec(name="test")
        new_spec = set_at_path(spec, "/description", "A test system")
        assert new_spec.description == "A test system"


# ── Events ─────────────────────────────────────────────────────────────────


class TestSpecEvent:
    def test_event_serialization(self):
        event = make_event(
            op=EventOp.SET,
            spec_name="test",
            sequence=1,
            author="user",
            path="/perception/min_fps",
            value=60,
            reason="need 60fps",
        )
        json_str = event.model_dump_json()
        restored = SpecEvent.model_validate_json(json_str)
        assert restored.op == EventOp.SET
        assert restored.path == "/perception/min_fps"
        assert restored.value == 60
        assert restored.author == "user"
        assert restored.reason == "need 60fps"
        assert restored.sequence == 1


class TestEventLog:
    def test_append_and_read(self, tmp_path):
        log = EventLog(tmp_path / "events.jsonl")
        event = make_event(
            op=EventOp.CREATE,
            spec_name="test",
            sequence=0,
            author="user",
            value=SystemSpec(name="test").model_dump(mode="json"),
        )
        log.append(event)
        events = log.read_all()
        assert len(events) == 1
        assert events[0].op == EventOp.CREATE

    def test_next_sequence(self, tmp_path):
        log = EventLog(tmp_path / "events.jsonl")
        assert log.next_sequence() == 0

        event = make_event(
            op=EventOp.CREATE,
            spec_name="test",
            sequence=0,
            author="user",
            value=SystemSpec(name="test").model_dump(mode="json"),
        )
        log.append(event)
        assert log.next_sequence() == 1

    def test_replay(self, tmp_path):
        log = EventLog(tmp_path / "events.jsonl")

        # Create
        spec = SystemSpec(name="test")
        log.append(
            make_event(
                op=EventOp.CREATE,
                spec_name="test",
                sequence=0,
                author="user",
                value=spec.model_dump(mode="json"),
            )
        )

        # Set
        log.append(
            make_event(
                op=EventOp.SET,
                spec_name="test",
                sequence=1,
                author="user",
                path="/perception/min_fps",
                value=30,
            )
        )

        log.append(
            make_event(
                op=EventOp.SET,
                spec_name="test",
                sequence=2,
                author="agent",
                path="/perception/min_fps",
                value=60,
            )
        )

        result = log.replay("test")
        assert result.perception.min_fps == 60

    def test_replay_with_snapshot(self, tmp_path):
        log = EventLog(tmp_path / "events.jsonl")

        # Create initial
        spec = SystemSpec(name="test")
        log.append(
            make_event(
                op=EventOp.CREATE,
                spec_name="test",
                sequence=0,
                author="user",
                value=spec.model_dump(mode="json"),
            )
        )

        # Set a field
        log.append(
            make_event(
                op=EventOp.SET,
                spec_name="test",
                sequence=1,
                author="user",
                path="/perception/min_fps",
                value=30,
            )
        )

        # Snapshot
        snap_spec = SystemSpec(name="test", perception=PerceptionSpec(min_fps=30.0))
        log.append(
            make_event(
                op=EventOp.SNAPSHOT,
                spec_name="test",
                sequence=2,
                author="user",
                value=snap_spec.model_dump(mode="json"),
            )
        )

        # More changes after snapshot
        log.append(
            make_event(
                op=EventOp.SET,
                spec_name="test",
                sequence=3,
                author="user",
                path="/compute/soc",
                value="Jetson Orin NX",
            )
        )

        result = log.replay("test")
        assert result.perception.min_fps == 30.0
        assert result.compute.soc == "Jetson Orin NX"

    def test_field_history(self, tmp_path):
        log = EventLog(tmp_path / "events.jsonl")

        log.append(
            make_event(
                op=EventOp.CREATE,
                spec_name="test",
                sequence=0,
                author="user",
                value=SystemSpec(name="test").model_dump(mode="json"),
            )
        )
        log.append(
            make_event(
                op=EventOp.SET,
                spec_name="test",
                sequence=1,
                author="user",
                path="/perception/min_fps",
                value=30,
            )
        )
        log.append(
            make_event(
                op=EventOp.SET,
                spec_name="test",
                sequence=2,
                author="agent",
                path="/perception/min_fps",
                value=60,
            )
        )
        log.append(
            make_event(
                op=EventOp.SET,
                spec_name="test",
                sequence=3,
                author="user",
                path="/compute/soc",
                value="Jetson Orin NX",
            )
        )

        history = log.field_history("/perception/min_fps")
        assert len(history) == 2
        assert history[0].value == 30
        assert history[1].value == 60


# ── Diff ───────────────────────────────────────────────────────────────────


class TestDiff:
    def test_no_changes(self):
        spec = SystemSpec(name="test", perception=PerceptionSpec(min_fps=30.0))
        changes = diff_specs(spec, spec)
        assert changes == []

    def test_added_field(self):
        old = SystemSpec(name="test")
        new = SystemSpec(name="test", description="hello")
        changes = diff_specs(old, new)
        added = [c for c in changes if c.change_type == ChangeType.ADDED]
        assert any(c.path == "/description" for c in added)

    def test_changed_field(self):
        old = SystemSpec(name="test", description="old")
        new = SystemSpec(name="test", description="new")
        changes = diff_specs(old, new)
        assert len(changes) == 1
        assert changes[0].change_type == ChangeType.CHANGED
        assert changes[0].old_value == "old"
        assert changes[0].new_value == "new"

    def test_subsystem_diff(self):
        old = SystemSpec(name="test", perception=PerceptionSpec(min_fps=30.0))
        new = SystemSpec(name="test", perception=PerceptionSpec(min_fps=60.0))
        changes = diff_specs(old, new)
        assert len(changes) == 1
        assert changes[0].path == "/perception/min_fps"


# ── Templates ──────────────────────────────────────────────────────────────


class TestTemplates:
    def test_get_templates(self):
        templates = get_templates()
        assert "drone-perception" in templates
        assert "quadruped-nav" in templates
        assert len(templates) == 6

    def test_drone_template(self):
        spec = get_template("drone-perception", name="my-drone")
        assert spec.name == "my-drone"
        assert spec.platform_type == PlatformType.DRONE
        assert spec.perception is not None
        assert spec.perception.min_fps == 30.0
        assert spec.compute is not None
        assert spec.power is not None

    def test_all_templates_valid(self):
        for template_name in get_templates():
            spec = get_template(template_name, name=f"test-{template_name}")
            assert spec.name == f"test-{template_name}"
            # Round-trip
            data = spec.model_dump(mode="json")
            restored = SystemSpec.model_validate(data)
            assert restored.name == spec.name

    def test_unknown_template(self):
        with pytest.raises(ValueError, match="Unknown template"):
            get_template("nonexistent")


# ── Validation ─────────────────────────────────────────────────────────────


class TestValidation:
    def test_power_vs_compute_error(self):
        spec = SystemSpec(
            name="test",
            compute=ComputeSpec(max_tdp_watts=30.0),
            power=PowerSpec(compute_power_watts=15.0),
        )
        issues = validate_spec(spec)
        errors = [i for i in issues if i.severity == Severity.ERROR]
        assert len(errors) >= 1
        assert "TDP" in errors[0].message

    def test_perception_latency_warning(self):
        spec = SystemSpec(
            name="test",
            perception=PerceptionSpec(max_latency_ms=50.0, min_fps=30.0),
        )
        issues = validate_spec(spec)
        # 50ms > 33.3ms frame time at 30fps
        warnings = [i for i in issues if i.severity == Severity.WARNING]
        assert len(warnings) >= 1

    def test_drone_geofencing_info(self):
        spec = SystemSpec(
            name="test",
            platform_type=PlatformType.DRONE,
            safety=SafetySpec(geofencing=False),
        )
        issues = validate_spec(spec)
        infos = [i for i in issues if i.severity == Severity.INFO]
        assert any("geofencing" in i.message.lower() for i in infos)

    def test_clean_spec_no_issues(self):
        # A well-configured spec should have no errors
        spec = SystemSpec(
            name="test",
            compute=ComputeSpec(max_tdp_watts=10.0),
            power=PowerSpec(
                compute_power_watts=15.0,
                power_budget_watts=25.0,
            ),
            perception=PerceptionSpec(max_latency_ms=20.0, min_fps=30.0),
        )
        issues = validate_spec(spec)
        errors = [i for i in issues if i.severity == Severity.ERROR]
        assert len(errors) == 0

    def test_mission_duration_error(self):
        spec = SystemSpec(
            name="test",
            power=PowerSpec(
                battery_wh=50.0,
                power_budget_watts=25.0,
                mission_duration_min=180.0,  # 3 hours on 2 hours of battery
            ),
        )
        issues = validate_spec(spec)
        errors = [i for i in issues if i.severity == Severity.ERROR]
        assert len(errors) >= 1
        assert "duration" in errors[0].message.lower()

    def test_slam_needs_depth(self):
        spec = SystemSpec(
            name="test",
            autonomy=AutonomySpec(navigation="slam"),
            perception=PerceptionSpec(camera_types=["monocular"]),
        )
        issues = validate_spec(spec)
        warnings = [i for i in issues if i.severity == Severity.WARNING]
        assert any("slam" in w.message.lower() for w in warnings)
