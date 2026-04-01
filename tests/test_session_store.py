"""Tests for session persistence."""

import json

from embodied_ai_architect.graphs.session_store import SessionStore
from embodied_ai_architect.graphs.soc_state import (
    DesignConstraints,
    create_initial_soc_state,
)


class TestSessionStore:
    def _make_store(self, tmp_path):
        return SessionStore(session_dir=tmp_path)

    def _make_state(self, goal="Test drone SoC", session_id=None):
        state = create_initial_soc_state(
            goal=goal,
            constraints=DesignConstraints(max_power_watts=5.0, max_latency_ms=33.3),
            use_case="test_drone",
            platform="drone",
            session_id=session_id,
        )
        return state

    def test_save_and_load(self, tmp_path):
        store = self._make_store(tmp_path)
        state = self._make_state()
        session_id = store.save(state)

        loaded = store.load(session_id)
        assert loaded is not None
        assert loaded["goal"] == "Test drone SoC"
        assert loaded["platform"] == "drone"
        assert loaded["session_id"] == session_id

    def test_load_nonexistent(self, tmp_path):
        store = self._make_store(tmp_path)
        assert store.load("nonexistent") is None

    def test_load_latest(self, tmp_path):
        store = self._make_store(tmp_path)

        state1 = self._make_state(goal="First", session_id="soc_first")
        store.save(state1)

        state2 = self._make_state(goal="Second", session_id="soc_second")
        store.save(state2)

        latest = store.load_latest()
        assert latest is not None
        assert latest["goal"] == "Second"

    def test_list_sessions(self, tmp_path):
        store = self._make_store(tmp_path)

        for i in range(3):
            state = self._make_state(goal=f"Goal {i}", session_id=f"soc_test{i}")
            store.save(state)

        sessions = store.list_sessions()
        assert len(sessions) == 3
        assert all(s["platform"] == "drone" for s in sessions)
        ids = {s["session_id"] for s in sessions}
        assert ids == {"soc_test0", "soc_test1", "soc_test2"}

    def test_list_empty(self, tmp_path):
        store = self._make_store(tmp_path)
        assert store.list_sessions() == []

    def test_delete(self, tmp_path):
        store = self._make_store(tmp_path)
        state = self._make_state(session_id="soc_to_delete")
        store.save(state)

        assert store.exists("soc_to_delete")
        assert store.delete("soc_to_delete")
        assert not store.exists("soc_to_delete")
        assert store.load("soc_to_delete") is None

    def test_delete_nonexistent(self, tmp_path):
        store = self._make_store(tmp_path)
        assert not store.delete("nonexistent")

    def test_state_roundtrip_preserves_fields(self, tmp_path):
        store = self._make_store(tmp_path)
        state = self._make_state()

        # Add some PPA metrics
        state["ppa_metrics"] = {
            "power_watts": 4.5,
            "latency_ms": 28.0,
            "verdicts": {"power": "PASS", "latency": "PASS"},
        }
        state["iteration"] = 3
        state["design_rationale"] = ["[planner] Created plan", "[optimizer] Applied INT8"]

        session_id = store.save(state)
        loaded = store.load(session_id)

        assert loaded["ppa_metrics"]["power_watts"] == 4.5
        assert loaded["iteration"] == 3
        assert len(loaded["design_rationale"]) == 2

    def test_saved_at_metadata_stripped(self, tmp_path):
        store = self._make_store(tmp_path)
        state = self._make_state()
        session_id = store.save(state)

        # Raw file should have _saved_at
        path = tmp_path / f"{session_id}.json"
        with open(path) as f:
            raw = json.load(f)
        assert "_saved_at" in raw

        # Loaded state should NOT have _saved_at
        loaded = store.load(session_id)
        assert "_saved_at" not in loaded
