"""Session management for MCP optimization sessions.

Tracks active optimization runs in background threads, provides
queryable status and results.

Usage:
    from embodied_ai_architect.mcp.session import SessionManager

    manager = SessionManager()
    session_id = manager.create_session(goal, constraints, config)
    status = manager.get_status(session_id)
    result = manager.get_result(session_id)
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class SessionStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class OptimizationSession:
    """A single optimization session running in a background thread."""

    def __init__(
        self,
        session_id: str,
        goal: str,
        constraints: dict[str, Any],
        config: dict[str, Any] | None = None,
    ):
        self.session_id = session_id
        self.goal = goal
        self.constraints = constraints
        self.config_dict = config or {}
        self.status = SessionStatus.PENDING
        self.result: dict[str, Any] | None = None
        self.error: str | None = None
        self.progress: dict[str, Any] = {"layer": "", "iteration": 0, "metric": 0.0}
        self.created_at = time.time()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start optimization in a background thread."""
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.status = SessionStatus.RUNNING
        self._thread.start()

    def _run(self) -> None:
        """Execute the optimization pipeline."""
        try:
            from embodied_ai_architect.graphs.moo.design_space import create_soc_design_space
            from embodied_ai_architect.graphs.moo.evaluator import DesignEvaluator
            from embodied_ai_architect.graphs.moo.engine import (
                OptimizationConfig,
                OptimizationEngine,
            )

            ds = create_soc_design_space(self.constraints)
            evaluator = DesignEvaluator(
                design_space=ds,
                base_state={"constraints": self.constraints},
                constraint_bounds=ds.constraint_bounds,
            )

            config = (
                OptimizationConfig(**self.config_dict) if self.config_dict else OptimizationConfig()
            )

            engine = OptimizationEngine(ds, evaluator, config)

            def callback(layer: str, iteration: int, evals: int, metric: float) -> None:
                self.progress = {
                    "layer": layer,
                    "iteration": iteration,
                    "evaluations": evals,
                    "metric": metric,
                }

            result = engine.run(callback=callback)
            engine.shutdown()

            self.result = result.model_dump()
            self.status = SessionStatus.COMPLETED

        except Exception as e:
            logger.error("Optimization session %s failed: %s", self.session_id, e)
            self.error = str(e)
            self.status = SessionStatus.FAILED

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()


class SessionManager:
    """Manages active optimization sessions."""

    def __init__(self, max_sessions: int = 10, ttl_seconds: int = 3600):
        self._sessions: dict[str, OptimizationSession] = {}
        self._lock = threading.Lock()
        self.max_sessions = max_sessions
        self.ttl_seconds = ttl_seconds

    def create_session(
        self,
        goal: str,
        constraints: dict[str, Any],
        config: dict[str, Any] | None = None,
    ) -> str:
        """Create and start a new optimization session.

        Returns session_id.
        """
        self._cleanup_expired()

        with self._lock:
            if len(self._sessions) >= self.max_sessions:
                raise RuntimeError(
                    f"Maximum sessions ({self.max_sessions}) reached. "
                    "Wait for existing sessions to complete."
                )

            session_id = f"moo_{uuid.uuid4().hex[:8]}"
            session = OptimizationSession(session_id, goal, constraints, config)
            self._sessions[session_id] = session

        session.start()
        return session_id

    def get_session(self, session_id: str) -> OptimizationSession | None:
        return self._sessions.get(session_id)

    def get_status(self, session_id: str) -> dict[str, Any]:
        session = self._sessions.get(session_id)
        if session is None:
            return {"error": f"Session {session_id} not found"}
        return {
            "session_id": session.session_id,
            "status": session.status.value,
            "progress": session.progress,
            "error": session.error,
        }

    def get_result(self, session_id: str) -> dict[str, Any] | None:
        session = self._sessions.get(session_id)
        if session is None:
            return None
        return session.result

    def list_sessions(self) -> list[dict[str, Any]]:
        return [
            {
                "session_id": s.session_id,
                "goal": s.goal,
                "status": s.status.value,
                "created_at": s.created_at,
            }
            for s in self._sessions.values()
        ]

    def _cleanup_expired(self) -> None:
        """Remove expired sessions."""
        now = time.time()
        with self._lock:
            expired = [
                sid
                for sid, s in self._sessions.items()
                if (
                    now - s.created_at > self.ttl_seconds
                    and s.status in (SessionStatus.COMPLETED, SessionStatus.FAILED)
                )
            ]
            for sid in expired:
                del self._sessions[sid]
