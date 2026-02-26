"""Integration tests for SpecStore — full CRUD, commit, tag, diff, history, why."""

import json

import pytest

from embodied_ai_architect.specs.store import SpecStore
from embodied_ai_architect.specs.models import PlatformType
from embodied_ai_architect.specs.exceptions import (
    SpecAlreadyExistsError,
    SpecNotFoundError,
    VersionNotFoundError,
)
from embodied_ai_architect.specs.diff import ChangeType


@pytest.fixture
def store(tmp_path):
    """Create a SpecStore with a temporary root."""
    return SpecStore(root=tmp_path / "specs")


class TestCreateAndGet:
    def test_create_empty(self, store):
        spec = store.create("test-spec", description="A test spec")
        assert spec.name == "test-spec"
        assert spec.description == "A test spec"

    def test_create_from_template(self, store):
        spec = store.create("drone-1", template="drone-perception")
        assert spec.name == "drone-1"
        assert spec.platform_type == PlatformType.DRONE
        assert spec.perception is not None
        assert spec.perception.min_fps == 30.0

    def test_create_duplicate_raises(self, store):
        store.create("test")
        with pytest.raises(SpecAlreadyExistsError):
            store.create("test")

    def test_get(self, store):
        store.create("test", description="hello")
        spec = store.get("test")
        assert spec.name == "test"
        assert spec.description == "hello"

    def test_get_nonexistent_raises(self, store):
        with pytest.raises(SpecNotFoundError):
            store.get("nonexistent")


class TestListAndDelete:
    def test_list_empty(self, store):
        specs = store.list_specs()
        assert specs == {}

    def test_list_after_create(self, store):
        store.create("a")
        store.create("b")
        specs = store.list_specs()
        assert "a" in specs
        assert "b" in specs
        assert len(specs) == 2

    def test_delete(self, store):
        store.create("to-delete")
        assert "to-delete" in store.list_specs()
        store.delete_spec("to-delete")
        assert "to-delete" not in store.list_specs()

    def test_delete_nonexistent_raises(self, store):
        with pytest.raises(SpecNotFoundError):
            store.delete_spec("nonexistent")


class TestMutations:
    def test_set_field(self, store):
        store.create("test")
        spec = store.set_field("test", "/perception/min_fps", 60, reason="need 60fps")
        assert spec.perception.min_fps == 60

    def test_set_field_creates_subsystem(self, store):
        store.create("test")
        spec = store.set_field("test", "/compute/soc", "Jetson Orin NX")
        assert spec.compute.soc == "Jetson Orin NX"

    def test_multiple_sets(self, store):
        store.create("test")
        store.set_field("test", "/perception/min_fps", 30)
        store.set_field("test", "/perception/cameras", 2)
        store.set_field("test", "/compute/soc", "Jetson Orin NX")
        spec = store.get("test")
        assert spec.perception.min_fps == 30
        assert spec.perception.cameras == 2
        assert spec.compute.soc == "Jetson Orin NX"

    def test_delete_field(self, store):
        store.create("test", template="drone-perception")
        spec = store.get("test")
        assert spec.perception.model_family is not None

        spec = store.delete_field("test", "/perception/model_family")
        assert spec.perception.model_family is None

    def test_set_field_nonexistent_spec_raises(self, store):
        with pytest.raises(SpecNotFoundError):
            store.set_field("nonexistent", "/perception/min_fps", 60)


class TestVersioning:
    def test_commit(self, store):
        store.create("test")
        store.set_field("test", "/perception/min_fps", 30)
        blob_hash = store.commit("test", message="initial", author="user")
        assert len(blob_hash) == 64  # SHA256 hex

    def test_history(self, store):
        store.create("test")
        store.set_field("test", "/perception/min_fps", 30)
        store.commit("test", message="v1")
        store.set_field("test", "/perception/min_fps", 60)
        store.commit("test", message="v2")

        versions = store.history("test")
        assert len(versions) == 2
        assert versions[0]["message"] == "v1"
        assert versions[1]["message"] == "v2"
        assert versions[1]["parent"] == versions[0]["hash"]

    def test_get_version(self, store):
        store.create("test")
        store.set_field("test", "/perception/min_fps", 30)
        h1 = store.commit("test", message="v1")
        store.set_field("test", "/perception/min_fps", 60)
        store.commit("test", message="v2")

        # Get v1 by full hash
        v1_spec = store.get_version("test", h1)
        assert v1_spec.perception.min_fps == 30

    def test_get_version_by_prefix(self, store):
        store.create("test")
        store.set_field("test", "/perception/min_fps", 30)
        h1 = store.commit("test", message="v1")

        # Get by prefix
        v1_spec = store.get_version("test", h1[:8])
        assert v1_spec.perception.min_fps == 30

    def test_get_version_nonexistent_raises(self, store):
        store.create("test")
        store.commit("test", message="v1")
        with pytest.raises(VersionNotFoundError):
            store.get_version("test", "nonexistent_hash")


class TestTagging:
    def test_tag_latest(self, store):
        store.create("test")
        store.set_field("test", "/perception/min_fps", 30)
        store.commit("test", message="v1")
        store.tag("test", "baseline")

        versions = store.history("test")
        assert "baseline" in versions[0]["tags"]

    def test_get_version_by_tag(self, store):
        store.create("test")
        store.set_field("test", "/perception/min_fps", 30)
        store.commit("test", message="v1")
        store.tag("test", "baseline")

        spec = store.get_version("test", "baseline")
        assert spec.perception.min_fps == 30

    def test_tag_without_commit_raises(self, store):
        store.create("test")
        with pytest.raises(ValueError, match="No committed versions"):
            store.tag("test", "v1")


class TestDiff:
    def test_diff_between_versions(self, store):
        store.create("test")
        store.set_field("test", "/perception/min_fps", 30)
        h1 = store.commit("test", message="v1")

        store.set_field("test", "/perception/min_fps", 60)
        store.set_field("test", "/compute/soc", "Jetson Orin NX")
        h2 = store.commit("test", message="v2")

        changes = store.diff("test", h1, h2)
        assert len(changes) > 0

        # Find the fps change
        fps_change = [c for c in changes if "min_fps" in c.path]
        assert len(fps_change) == 1
        assert fps_change[0].change_type == ChangeType.CHANGED
        assert fps_change[0].old_value == 30.0
        assert fps_change[0].new_value == 60.0

    def test_diff_by_tag(self, store):
        store.create("test")
        store.set_field("test", "/perception/min_fps", 30)
        store.commit("test", message="v1")
        store.tag("test", "v1")

        store.set_field("test", "/perception/min_fps", 60)
        store.commit("test", message="v2")
        store.tag("test", "v2")

        changes = store.diff("test", "v1", "v2")
        assert len(changes) > 0


class TestProvenance:
    def test_why(self, store):
        store.create("test")
        store.set_field("test", "/perception/min_fps", 30, author="user", reason="initial guess")
        store.set_field(
            "test", "/perception/min_fps", 60, author="agent", reason="30fps insufficient"
        )

        events = store.why("test", "/perception/min_fps")
        assert len(events) == 2
        assert events[0].author == "user"
        assert events[0].reason == "initial guess"
        assert events[1].author == "agent"
        assert events[1].reason == "30fps insufficient"

    def test_why_empty(self, store):
        store.create("test")
        events = store.why("test", "/perception/min_fps")
        assert events == []


class TestExportImport:
    def test_export_json(self, store):
        store.create("test", template="drone-perception")
        output = store.export_spec("test", fmt="json")
        data = json.loads(output)
        assert data["name"] == "test"
        assert "perception" in data

    def test_import_json(self, store, tmp_path):
        # Create and export
        store.create("original", template="drone-perception")
        output = store.export_spec("original", fmt="json")

        # Import as new spec
        data = json.loads(output)
        spec = store.import_spec("copy", data, reason="testing import")
        assert spec.name == "copy"
        assert spec.perception is not None

        # Verify it persists
        loaded = store.get("copy")
        assert loaded.name == "copy"

    def test_export_yaml(self, store):
        pytest.importorskip("yaml")
        store.create("test", template="edge-camera")
        output = store.export_spec("test", fmt="yaml")
        assert "name:" in output
        assert "perception:" in output


class TestValidation:
    def test_validate_drone(self, store):
        store.create("test", template="drone-perception")
        issues = store.validate("test")
        # Drone template should be well-formed, no errors
        errors = [i for i in issues if i.severity.value == "error"]
        assert len(errors) == 0

    def test_validate_bad_spec(self, store):
        store.create("test")
        store.set_field("test", "/compute/max_tdp_watts", 50)
        store.set_field("test", "/power/compute_power_watts", 10)
        issues = store.validate("test")
        errors = [i for i in issues if i.severity.value == "error"]
        assert len(errors) >= 1


class TestResolve:
    def test_resolve(self, store):
        store.create("test", template="drone-perception")
        flat = store.resolve("test")
        assert flat["name"] == "test"
        assert flat["platform_type"] == "drone"
        assert flat["perception.min_fps"] == 30.0
        assert flat["compute.soc"] == "Jetson Orin NX"
