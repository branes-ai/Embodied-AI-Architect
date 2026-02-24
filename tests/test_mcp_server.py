"""Tests for the MCP server and session management.

Core session management tests run without MCP dependency.
MCP-specific protocol tests are skipped if mcp is not installed.
"""

import time

import pytest

from embodied_ai_architect.mcp.session import (
    OptimizationSession,
    SessionManager,
    SessionStatus,
)


class TestOptimizationSession:
    def test_session_creation(self):
        session = OptimizationSession(
            session_id="test_001",
            goal="test SoC",
            constraints={"max_power_watts": 5.0},
        )
        assert session.status == SessionStatus.PENDING
        assert session.session_id == "test_001"
        assert session.result is None

    def test_session_runs_to_completion(self):
        session = OptimizationSession(
            session_id="test_002",
            goal="test SoC",
            constraints={"max_power_watts": 10.0},
            config={
                "layers": "map_elites",
                "map_elites": {
                    "n_iterations": 3,
                    "batch_size": 4,
                    "initial_population": 8,
                    "resolution": 3,
                },
                "max_workers": 2,
            },
        )
        session.start()

        # Wait for completion (max 30s)
        timeout = 30
        while session.is_alive() and timeout > 0:
            time.sleep(0.5)
            timeout -= 0.5

        assert session.status == SessionStatus.COMPLETED
        assert session.result is not None
        assert "pareto_front" in session.result


class TestSessionManager:
    def test_create_session(self):
        manager = SessionManager(max_sessions=5)
        session_id = manager.create_session(
            goal="test",
            constraints={"max_power_watts": 10.0},
            config={
                "layers": "map_elites",
                "map_elites": {
                    "n_iterations": 2,
                    "batch_size": 4,
                    "initial_population": 8,
                    "resolution": 3,
                },
                "max_workers": 2,
            },
        )
        assert session_id.startswith("moo_")
        status = manager.get_status(session_id)
        assert status["status"] in ("pending", "running", "completed")

    def test_list_sessions(self):
        manager = SessionManager(max_sessions=5)
        manager.create_session(
            goal="test1",
            constraints={},
            config={
                "layers": "map_elites",
                "map_elites": {
                    "n_iterations": 1,
                    "batch_size": 4,
                    "initial_population": 4,
                    "resolution": 2,
                },
            },
        )
        sessions = manager.list_sessions()
        assert len(sessions) >= 1

    def test_max_sessions_enforced(self):
        manager = SessionManager(max_sessions=2)
        cfg = {
            "layers": "map_elites",
            "map_elites": {
                "n_iterations": 100,
                "batch_size": 4,
                "initial_population": 4,
                "resolution": 2,
            },
        }
        manager.create_session(goal="test1", constraints={}, config=cfg)
        manager.create_session(goal="test2", constraints={}, config=cfg)

        with pytest.raises(RuntimeError, match="Maximum sessions"):
            manager.create_session(goal="test3", constraints={}, config=cfg)

    def test_get_nonexistent_session(self):
        manager = SessionManager()
        status = manager.get_status("nonexistent")
        assert "error" in status

    def test_session_lifecycle(self):
        """Full lifecycle: create -> running -> completed."""
        manager = SessionManager(max_sessions=5)
        session_id = manager.create_session(
            goal="lifecycle test",
            constraints={"max_power_watts": 10.0},
            config={
                "layers": "map_elites",
                "map_elites": {
                    "n_iterations": 2,
                    "batch_size": 4,
                    "initial_population": 8,
                    "resolution": 3,
                },
                "max_workers": 2,
            },
        )

        # Wait for completion
        timeout = 30
        while timeout > 0:
            status = manager.get_status(session_id)
            if status["status"] == "completed":
                break
            time.sleep(0.5)
            timeout -= 0.5

        result = manager.get_result(session_id)
        assert result is not None
        assert "pareto_front" in result


class TestMCPServer:
    def test_tool_definitions(self):
        from embodied_ai_architect.mcp.server import get_mcp_tool_definitions

        tools = get_mcp_tool_definitions()
        assert len(tools) == 5
        names = {t["name"] for t in tools}
        assert "start_exploration" in names
        assert "get_pareto_front" in names
        assert "get_sensitivity" in names
        assert "explain_tradeoff" in names
        assert "get_exploration_status" in names

    def test_execute_start_exploration(self):
        import json
        from embodied_ai_architect.mcp.server import execute_mcp_tool

        result = json.loads(
            execute_mcp_tool(
                "start_exploration",
                {
                    "goal": "test SoC",
                    "constraints": {"max_power_watts": 10.0},
                    "config": {
                        "layers": "map_elites",
                        "map_elites": {
                            "n_iterations": 2,
                            "batch_size": 4,
                            "initial_population": 8,
                            "resolution": 3,
                        },
                        "max_workers": 2,
                    },
                },
            )
        )
        assert "session_id" in result
        assert result["status"] == "running"
