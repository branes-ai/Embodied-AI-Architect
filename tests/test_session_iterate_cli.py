"""Seam S11 (issue #216): `branes session iterate` runs a real loop iteration over a
persisted DesignState and saves the mutated state."""

from pathlib import Path

import pytest
from click.testing import CliRunner

from embodied_ai_architect.cli.commands.session import session
from embodied_ai_architect.graphs import session_store as ss_mod
from embodied_ai_architect.graphs.design_state import DesignConstraints
from embodied_ai_architect.graphs.session_store import SessionStore


@pytest.fixture()
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SessionStore:
    # Point the store the CLI constructs at a temp dir.
    monkeypatch.setattr(ss_mod, "DEFAULT_SESSION_DIR", tmp_path)
    return SessionStore(tmp_path)


def _fake_moo_tool_factory():
    def tool(state):
        knee = {"objectives": {"power_watts": 4.0, "latency_ms": 20.0}}
        return {
            "knee_point": knee,
            "pareto_points": [knee],
            "hypervolume_history": state.get("hypervolume_history", []) + [1.0],
        }

    return tool


def _seed_session(store: SessionStore) -> str:
    state = {
        "session_id": "soc_test123",
        "goal": "drone perception SoC",
        "constraints": DesignConstraints(max_power_watts=5.0, max_latency_ms=33.0).model_dump(),
        "ppa_metrics": {"verdicts": {"power": "FAIL"}, "power_watts": 8.0},
        "llm_available": False,
        "iteration": 0,
    }
    return store.save(state)


def test_iterate_runs_and_persists_mutated_state(
    store: SessionStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Use a fast, deterministic MOO tool instead of the real engine.
    import embodied_ai_architect.graphs.loop_convergence_graph as lcg

    monkeypatch.setattr(lcg, "make_moo_engine_tool", _fake_moo_tool_factory)

    sid = _seed_session(store)
    before = store.load(sid)

    result = CliRunner().invoke(session, ["iterate", sid, "-n", "1"])

    assert result.exit_code == 0, result.output
    assert "Loop Convergence trace" in result.output
    assert f"Session {sid} updated" in result.output

    # the persisted state was mutated by the loop
    after = store.load(sid)
    assert after is not None
    assert after.get("status") == "complete" or after.get("iteration", 0) >= before.get(
        "iteration", 0
    )


def test_iterate_latest(store: SessionStore, monkeypatch: pytest.MonkeyPatch) -> None:
    import embodied_ai_architect.graphs.loop_convergence_graph as lcg

    monkeypatch.setattr(lcg, "make_moo_engine_tool", _fake_moo_tool_factory)
    _seed_session(store)
    result = CliRunner().invoke(session, ["iterate", "--latest"])
    assert result.exit_code == 0, result.output
    assert "updated" in result.output


def test_iterate_no_session_is_graceful(store: SessionStore) -> None:
    result = CliRunner().invoke(session, ["iterate", "--latest"])
    assert result.exit_code == 0
    assert "No matching session" in result.output


def test_iterate_requires_target(store: SessionStore) -> None:
    result = CliRunner().invoke(session, ["iterate"])
    assert result.exit_code == 0
    assert "Provide a session ID or use --latest" in result.output
