"""Tests for Mission entity and MissionStore (issue #52)."""

from embodied_ai_architect.mission import Mission, MissionStatus, MissionStore


class TestMissionModel:
    """The Mission Pydantic model."""

    def test_default_fields(self):
        m = Mission()
        assert m.id.startswith("mission_")
        assert m.status == MissionStatus.DRAFT
        assert m.goal == ""
        assert m.constraints == {}
        assert m.selected_sensors == []
        assert m.design_state is None
        assert m.bom is None
        assert m.reports == []
        assert m.created_at
        assert m.updated_at

    def test_custom_fields(self):
        m = Mission(
            name="Drone Perception SoC",
            goal="30fps detection at <5W",
            platform_id="aerial.multirotor_delivery",
            use_case="delivery_drone",
            constraints={"max_power_watts": 5.0, "max_latency_ms": 33.3},
            status=MissionStatus.QUALIFIED,
        )
        assert m.name == "Drone Perception SoC"
        assert m.goal == "30fps detection at <5W"
        assert m.platform_id == "aerial.multirotor_delivery"
        assert m.status == MissionStatus.QUALIFIED
        assert m.constraints["max_power_watts"] == 5.0

    def test_touch_updates_timestamp(self):
        m = Mission()
        original = m.updated_at
        import time

        time.sleep(0.01)
        m.touch()
        assert m.updated_at >= original

    def test_model_dump_round_trips(self):
        m = Mission(
            name="Test",
            goal="Test goal",
            selected_sensors=["camera", "imu"],
            optimization_history=[{"iteration": 0, "power": 5.0}],
        )
        data = m.model_dump()
        m2 = Mission(**data)
        assert m2.name == m.name
        assert m2.selected_sensors == ["camera", "imu"]
        assert m2.optimization_history == [{"iteration": 0, "power": 5.0}]

    def test_status_enum_values(self):
        assert MissionStatus.DRAFT.value == "draft"
        assert MissionStatus.QUALIFIED.value == "qualified"
        assert MissionStatus.DESIGNED.value == "designed"
        assert MissionStatus.OPTIMIZED.value == "optimized"
        assert MissionStatus.VALIDATED.value == "validated"


class TestMissionStore:
    """JSON persistence for missions."""

    def test_save_and_load(self, tmp_path):
        store = MissionStore(root=tmp_path)
        m = Mission(name="Test Mission", goal="Test goal")
        mission_id = store.save(m)

        loaded = store.load(mission_id)
        assert loaded is not None
        assert loaded.id == m.id
        assert loaded.name == "Test Mission"
        assert loaded.goal == "Test goal"

    def test_save_creates_directory(self, tmp_path):
        store = MissionStore(root=tmp_path)
        m = Mission()
        store.save(m)
        assert (tmp_path / m.id / "manifest.json").exists()

    def test_save_is_atomic(self, tmp_path):
        """No .tmp file left after save (atomic rename completed)."""
        store = MissionStore(root=tmp_path)
        m = Mission()
        store.save(m)
        tmp_files = list((tmp_path / m.id).glob("*.tmp"))
        assert tmp_files == []

    def test_load_nonexistent_returns_none(self, tmp_path):
        store = MissionStore(root=tmp_path)
        assert store.load("nonexistent_id") is None

    def test_load_latest(self, tmp_path):
        store = MissionStore(root=tmp_path)

        m1 = Mission(name="First")
        store.save(m1)
        import time

        time.sleep(0.05)
        m2 = Mission(name="Second")
        store.save(m2)

        latest = store.load_latest()
        assert latest is not None
        assert latest.name == "Second"

    def test_load_latest_empty_store(self, tmp_path):
        store = MissionStore(root=tmp_path)
        assert store.load_latest() is None

    def test_list_missions(self, tmp_path):
        store = MissionStore(root=tmp_path)

        m1 = Mission(name="Alpha", goal="Goal A", status=MissionStatus.DRAFT)
        m2 = Mission(name="Beta", goal="Goal B", status=MissionStatus.QUALIFIED)
        store.save(m1)
        store.save(m2)

        missions = store.list_missions()
        assert len(missions) == 2
        names = {m["name"] for m in missions}
        assert names == {"Alpha", "Beta"}
        # Each summary has expected fields
        for s in missions:
            assert "id" in s
            assert "name" in s
            assert "status" in s
            assert "goal" in s
            assert "created_at" in s
            assert "updated_at" in s

    def test_list_empty_store(self, tmp_path):
        store = MissionStore(root=tmp_path)
        assert store.list_missions() == []

    def test_delete(self, tmp_path):
        store = MissionStore(root=tmp_path)
        m = Mission(name="To Delete")
        store.save(m)
        assert store.exists(m.id)

        result = store.delete(m.id)
        assert result is True
        assert not store.exists(m.id)
        assert store.load(m.id) is None

    def test_delete_nonexistent_returns_false(self, tmp_path):
        store = MissionStore(root=tmp_path)
        result = store.delete("nonexistent_id")
        assert result is False

    def test_exists(self, tmp_path):
        store = MissionStore(root=tmp_path)
        m = Mission()
        assert not store.exists(m.id)
        store.save(m)
        assert store.exists(m.id)

    def test_save_updates_timestamp(self, tmp_path):
        store = MissionStore(root=tmp_path)
        m = Mission(name="Timestamped")
        original = m.updated_at
        import time

        time.sleep(0.01)
        store.save(m)
        loaded = store.load(m.id)
        assert loaded.updated_at >= original

    def test_overwrite_preserves_id(self, tmp_path):
        """Saving the same mission twice overwrites without changing ID."""
        store = MissionStore(root=tmp_path)
        m = Mission(name="V1", goal="Original")
        store.save(m)

        m.name = "V2"
        m.goal = "Updated"
        m.status = MissionStatus.DESIGNED
        store.save(m)

        loaded = store.load(m.id)
        assert loaded.name == "V2"
        assert loaded.goal == "Updated"
        assert loaded.status == MissionStatus.DESIGNED
        # Only one mission in the store
        assert len(store.list_missions()) == 1

    def test_multiple_missions_isolated(self, tmp_path):
        """Each mission has its own directory."""
        store = MissionStore(root=tmp_path)
        m1 = Mission(name="Mission A")
        m2 = Mission(name="Mission B")
        store.save(m1)
        store.save(m2)

        assert store.load(m1.id).name == "Mission A"
        assert store.load(m2.id).name == "Mission B"
        assert len(store.list_missions()) == 2
