"""End-to-end integration tests for the full MOO pipeline (issue #27).

Exercises the complete flow:

    qualify → plan → dispatch → moo_explorer → ppa_assessor → critic → report

through `SoCDesignRunner`, then verifies that Pareto data flows through to:

  - the runner's returned state (`pareto_points`, `moo_results`)
  - the on-disk session store (`SessionStore.load`)
  - the REST API (`/api/sessions/{id}/pareto`)
  - the `OptimizationReviewSnapshot` (`pareto_front_size > 0`)
  - frontier accumulation across iterations (#23 merge logic)
  - the `branes session show` rendering

These tests are slower than unit tests because they actually run MAP-Elites
(in fast mode), but they pin the contract that the whole pipeline produces
a populated frontier — which is the architect's primary deliverable.
"""

from __future__ import annotations

import pytest

try:
    from fastapi.testclient import TestClient

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

from embodied_ai_architect.graphs.optimization_review import (
    build_optimization_review_snapshot,
)
from embodied_ai_architect.graphs.session_store import SessionStore
from embodied_ai_architect.graphs.soc_runner import SoCDesignRunner
from embodied_ai_architect.graphs.soc_state import DesignConstraints

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

# A static plan that includes moo_explorer in the position the LLM planner is
# instructed to place it: after hw_explorer, in parallel with architecture_composer,
# joining at ppa_assessor. fast_mode keeps MAP-Elites cheap (~20 iterations) so
# the test runs in a few seconds rather than a minute.
PLAN_WITH_MOO = [
    {"id": "t1", "name": "Analyze workload", "agent": "workload_analyzer", "dependencies": []},
    {"id": "t2", "name": "Explore hardware", "agent": "hw_explorer", "dependencies": ["t1"]},
    {
        "id": "t3",
        "name": "Compose architecture",
        "agent": "architecture_composer",
        "dependencies": ["t2"],
    },
    {
        "id": "t4",
        "name": "Explore Pareto frontier",
        "agent": "moo_explorer",
        "dependencies": ["t2"],
        "metadata": {"fast_mode": True},
    },
    {"id": "t5", "name": "Assess PPA", "agent": "ppa_assessor", "dependencies": ["t3", "t4"]},
    {"id": "t6", "name": "Review design", "agent": "critic", "dependencies": ["t5"]},
    {"id": "t7", "name": "Generate report", "agent": "report_generator", "dependencies": ["t6"]},
]

# AMR-class constraints — looser than tight drone targets so MOO is virtually
# guaranteed to find feasible designs in the design space (avoids flakiness).
INTEGRATION_CONSTRAINTS = DesignConstraints(
    max_power_watts=15.0,
    max_latency_ms=50.0,
    max_cost_usd=100.0,
)
INTEGRATION_GOAL = "Design an SoC for warehouse AMR with detection + tracking"
INTEGRATION_USE_CASE = "warehouse_amr"
INTEGRATION_PLATFORM = "amr"


@pytest.fixture(scope="module")
def session_dir(tmp_path_factory):
    """Module-scoped isolated session directory.

    Module-scoped (not function-scoped) so the expensive `moo_run_state`
    fixture only runs the full pipeline once per module. tmp_path_factory
    is the module-safe equivalent of pytest's per-test tmp_path fixture.
    """
    return str(tmp_path_factory.mktemp("moo_integration_sessions"))


@pytest.fixture(scope="module")
def moo_run_state(session_dir):
    """Run the full MOO pipeline ONCE per module and share the result.

    The full pipeline (planner → dispatch → moo_explorer → ppa_assessor →
    critic → report) costs ~5-10s locally and significantly more on CI
    runners. Module-scoping cuts the integration suite from ~60s → ~10s
    by sharing one fully-executed final state across every test that just
    inspects the result.
    """
    runner = SoCDesignRunner(
        static_plan=PLAN_WITH_MOO,
        session_dir=session_dir,
    )
    state = runner.run(
        goal=INTEGRATION_GOAL,
        constraints=INTEGRATION_CONSTRAINTS,
        use_case=INTEGRATION_USE_CASE,
        platform=INTEGRATION_PLATFORM,
        session_id="soc_moo_integration",
    )
    return state


# ---------------------------------------------------------------------------
# Acceptance criterion 1: SoCDesignRunner.run() produces pareto_points
# ---------------------------------------------------------------------------


class TestRunnerProducesParetoData:
    def test_run_produces_pareto_points(self, moo_run_state):
        """The runner's returned state must contain non-empty pareto_points."""
        pareto_points = moo_run_state.get("pareto_points", [])
        assert len(pareto_points) > 0, (
            "MOO ran but produced no pareto_points — the moo_explorer task "
            "either was not scheduled or found no feasible designs"
        )

    def test_run_produces_moo_results_with_evaluations(self, moo_run_state):
        """moo_results must contain a populated OptimizationResult dump."""
        moo_results = moo_run_state.get("moo_results", {})
        assert moo_results, "moo_results was not written to state"
        assert moo_results.get("total_evaluations", 0) > 0
        # MAP-Elites is the always-on first layer
        assert "map_elites" in moo_results.get("layers_used", [])

    def test_run_produces_hypervolume(self, moo_run_state):
        """A populated frontier must yield a positive hypervolume."""
        moo_results = moo_run_state.get("moo_results", {})
        hv = moo_results.get("hypervolume", 0.0)
        assert hv > 0, f"hypervolume should be > 0 with feasible designs, got {hv}"


# ---------------------------------------------------------------------------
# Acceptance criterion 2: session saved to disk contains pareto_points
# ---------------------------------------------------------------------------


class TestSessionStorePersistsParetoData:
    def test_session_file_round_trips_pareto_points(self, moo_run_state, session_dir):
        """A SessionStore reload must preserve pareto_points and moo_results."""
        store = SessionStore(session_dir=session_dir)
        loaded = store.load("soc_moo_integration")
        assert loaded is not None, "session was not persisted to the override dir"
        assert len(loaded.get("pareto_points", [])) > 0
        assert loaded.get("moo_results", {}).get("total_evaluations", 0) > 0

    def test_load_latest_returns_the_moo_session(self, moo_run_state, session_dir):
        """`branes session show --latest` calls store.load_latest() — verify it
        returns the session we just ran (the only one in the isolated dir)."""
        store = SessionStore(session_dir=session_dir)
        latest = store.load_latest()
        assert latest is not None
        assert latest.get("session_id") == "soc_moo_integration"
        assert len(latest.get("pareto_points", [])) > 0


# ---------------------------------------------------------------------------
# Acceptance criterion 3: /api/sessions/{id}/pareto returns non-empty points
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_FASTAPI, reason="fastapi not installed")
class TestAPIServesParetoData:
    def test_pareto_endpoint_returns_non_empty_points(self, moo_run_state, session_dir):
        from embodied_ai_architect.api.server import create_app

        app = create_app(session_dir=session_dir)
        client = TestClient(app)

        response = client.get("/api/sessions/soc_moo_integration/pareto")
        assert response.status_code == 200
        body = response.json()
        assert len(body["points"]) > 0
        assert body["pareto_front_size"] > 0
        # hypervolume is wired from optimization_review_snapshot — may be None
        # if no snapshot was built; the points themselves are the contract here

    def test_pareto_endpoint_front_indices_are_valid(self, moo_run_state, session_dir):
        from embodied_ai_architect.api.server import create_app

        app = create_app(session_dir=session_dir)
        client = TestClient(app)

        body = client.get("/api/sessions/soc_moo_integration/pareto").json()
        # Every "front" index must point at a non-dominated point
        for idx in body["front"]:
            assert 0 <= idx < len(body["points"])
            assert body["points"][idx].get("dominated", False) is False


# ---------------------------------------------------------------------------
# Acceptance criterion 4: optimization_review_snapshot.pareto_front_size > 0
# ---------------------------------------------------------------------------


class TestOptimizationReviewSnapshot:
    def test_snapshot_reports_pareto_front_size(self, moo_run_state):
        """Building the snapshot from the final state must report pareto_front_size > 0."""
        snapshot = build_optimization_review_snapshot(moo_run_state)
        assert snapshot.pareto_front_size > 0, (
            f"snapshot.pareto_front_size should be > 0 after MOO ran, "
            f"got {snapshot.pareto_front_size}"
        )

    def test_snapshot_carries_moo_summary(self, moo_run_state):
        """The MOO summary should mirror moo_results — total evals + layers_used."""
        snapshot = build_optimization_review_snapshot(moo_run_state)
        assert snapshot.moo_summary, "moo_summary should be populated when MOO ran"
        assert snapshot.moo_summary.get("total_evaluations", 0) > 0
        assert "map_elites" in snapshot.moo_summary.get("layers_used", [])


# ---------------------------------------------------------------------------
# Acceptance criterion 5: Pareto frontier grows across 2+ iterations
# ---------------------------------------------------------------------------


class TestFrontierAccumulation:
    def test_frontier_grows_or_holds_across_two_runs(self, session_dir):
        """Calling moo_explorer twice on the same state must merge the frontiers
        (issue #23 — `_merge_pareto_frontiers`). The accumulated count must
        never shrink, and pareto_frontier_history must record both iterations."""
        from embodied_ai_architect.graphs.moo.specialist import moo_explorer
        from embodied_ai_architect.graphs.task_graph import TaskNode
        from embodied_ai_architect.graphs.soc_state import create_initial_soc_state

        state = create_initial_soc_state(
            goal=INTEGRATION_GOAL,
            constraints=INTEGRATION_CONSTRAINTS,
            use_case=INTEGRATION_USE_CASE,
            platform=INTEGRATION_PLATFORM,
            session_id="soc_frontier_growth",
        )

        task = TaskNode(
            id="t_moo",
            name="Explore Pareto frontier",
            agent="moo_explorer",
            metadata={"fast_mode": True},
        )

        # Iteration 0
        state["iteration"] = 0
        result0 = moo_explorer(task, state)
        for k, v in result0["_state_updates"].items():
            state[k] = v
        first_size = len(state["pareto_points"])
        assert first_size > 0

        # Iteration 1
        state["iteration"] = 1
        result1 = moo_explorer(task, state)
        for k, v in result1["_state_updates"].items():
            state[k] = v
        second_size = len(state["pareto_points"])

        # Accumulated frontier never shrinks — points only get added or
        # supersede dominated peers; the merged set is monotonic in coverage.
        assert (
            second_size >= first_size
        ), f"Pareto frontier shrank from {first_size} to {second_size} points"
        # pareto_frontier_history records both iterations
        history = state.get("pareto_frontier_history", [])
        assert len(history) == 2
        assert history[0]["iteration"] == 0
        assert history[1]["iteration"] == 1


# ---------------------------------------------------------------------------
# Acceptance criterion 6: `branes session show --latest` displays MOO summary
# ---------------------------------------------------------------------------


class TestSessionShowDisplaysMOOSummary:
    def test_session_show_renders_moo_block(self, moo_run_state, session_dir, monkeypatch):
        """The session show command (after issue #27) must render a `MOO Summary`
        block when the session has moo_results — so the architect doesn't have
        to drop into --json or the API to confirm MOO actually ran."""
        from click.testing import CliRunner
        from embodied_ai_architect.cli.commands.session import session as session_cmd

        # Force the CLI's SessionStore() to use the test dir
        from embodied_ai_architect.graphs import session_store as session_store_mod

        original_store = session_store_mod.SessionStore

        def _patched_store(*args, **kwargs):
            return original_store(session_dir=session_dir)

        monkeypatch.setattr(session_store_mod, "SessionStore", _patched_store)

        runner = CliRunner()
        result = runner.invoke(session_cmd, ["show", "--latest"])
        assert result.exit_code == 0, result.output
        assert "MOO Summary" in result.output
        assert "Pareto front" in result.output
        assert "Total evals" in result.output
