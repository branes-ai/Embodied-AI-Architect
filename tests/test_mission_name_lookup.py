"""Tests for mission name-based lookup (issue #140)."""

import pytest

from embodied_ai_architect.mission.models import Mission
from embodied_ai_architect.mission.store import MissionStore


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return MissionStore()


@pytest.fixture()
def saved_mission(store):
    m = Mission(name="vineyard-sprayer", goal="autonomous vineyard sprayer")
    store.save(m)
    return m


class TestLoadByName:
    def test_load_by_id(self, store, saved_mission):
        loaded = store.load(saved_mission.id)
        assert loaded is not None
        assert loaded.name == "vineyard-sprayer"

    def test_load_by_name(self, store, saved_mission):
        loaded = store.load("vineyard-sprayer")
        assert loaded is not None
        assert loaded.id == saved_mission.id
        assert loaded.goal == "autonomous vineyard sprayer"

    def test_load_nonexistent_returns_none(self, store):
        assert store.load("nonexistent") is None

    def test_id_takes_precedence_over_name(self, store):
        """If a mission's ID matches, return it even if name differs."""
        m = Mission(id="my-id", name="my-name", goal="test")
        store.save(m)
        loaded = store.load("my-id")
        assert loaded is not None
        assert loaded.name == "my-name"


class TestExistsByName:
    def test_exists_by_id(self, store, saved_mission):
        assert store.exists(saved_mission.id)

    def test_exists_by_name(self, store, saved_mission):
        assert store.exists("vineyard-sprayer")

    def test_not_exists(self, store):
        assert not store.exists("nonexistent")


class TestDeleteByName:
    def test_delete_by_name(self, store, saved_mission):
        result = store.delete("vineyard-sprayer")
        assert result
        assert store.load(saved_mission.id) is None

    def test_delete_by_id(self, store, saved_mission):
        result = store.delete(saved_mission.id)
        assert result
        assert store.load("vineyard-sprayer") is None
        deleted = store.delete("nonexistent")
        assert not deleted
    def test_delete_nonexistent(self, store):
        result = store.delete("nonexistent")
        assert not result
