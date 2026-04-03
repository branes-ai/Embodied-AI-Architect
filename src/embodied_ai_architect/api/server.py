"""FastAPI server for design session data access.

Thin read-only layer over SessionStore. All endpoints are GET — the API
does not modify sessions. Design execution happens through the CLI or
SoCDesignRunner.

Usage:
    from embodied_ai_architect.api.server import create_app

    app = create_app()
    # Run with: uvicorn embodied_ai_architect.api.server:app
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, AsyncGenerator

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse

from embodied_ai_architect.api.schemas import (
    ConstraintSlackness,
    HealthResponse,
    OperatorBreakdown,
    ParetoResponse,
    SessionSummary,
    TaskGraphNode,
    TaskGraphResponse,
    TrajectoryEntry,
    WorkloadResponse,
)
from embodied_ai_architect.graphs.session_store import SessionStore

logger = logging.getLogger(__name__)


def _get_store(request: Request) -> SessionStore:
    """Get the SessionStore from app state (no global mutable)."""
    return request.app.state.session_store


def create_app(
    cors_origins: list[str] | None = None,
    session_dir: str | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        cors_origins: Allowed CORS origins. Defaults to localhost dev ports.
        session_dir: Override session storage directory.

    Returns:
        Configured FastAPI app.
    """
    app = FastAPI(
        title="Branes Architect API",
        description="Design session data for the Branes Embodied AI Architect platform",
        version="0.1.0",
    )

    # CORS
    if cors_origins is None:
        cors_origins = [
            "http://localhost:3000",
            "http://localhost:5173",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:5173",
        ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    # Session store on app.state — no global mutable
    if session_dir:
        app.state.session_store = SessionStore(session_dir=session_dir)
    else:
        app.state.session_store = SessionStore()

    # Register routes
    app.include_router(_build_router())

    return app


def _build_router():
    """Build the API router with all endpoints."""
    from fastapi import APIRouter

    router = APIRouter(prefix="/api")

    @router.get("/health", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        """Server health check."""
        store = _get_store(request)
        return HealthResponse(
            session_dir=str(store.session_dir),
            session_count=len(store.list_sessions()),
        )

    @router.get("/sessions", response_model=list[SessionSummary])
    async def list_sessions(request: Request) -> list[SessionSummary]:
        """List all saved design sessions."""
        store = _get_store(request)
        return [SessionSummary(**s) for s in store.list_sessions()]

    @router.get("/sessions/{session_id}")
    async def get_session(session_id: str, request: Request) -> dict[str, Any]:
        """Get full session state (untyped — full SoCDesignState)."""
        return _load_or_404(session_id, request)

    @router.get("/sessions/{session_id}/pareto", response_model=ParetoResponse)
    async def get_pareto(session_id: str, request: Request) -> ParetoResponse:
        """Get Pareto frontier data for visualization."""
        state = _load_or_404(session_id, request)
        pareto_points = state.get("pareto_points", [])
        pareto_results = state.get("pareto_results", {})
        opt_snap = state.get("optimization_review_snapshot", {})

        return ParetoResponse(
            points=pareto_points,
            front=pareto_results.get("front", []),
            knee_point_index=pareto_results.get("knee_point_index"),
            pareto_front_size=opt_snap.get("pareto_front_size", len(pareto_points)),
            hypervolume=opt_snap.get("hypervolume"),
        )

    @router.get("/sessions/{session_id}/slackness", response_model=list[ConstraintSlackness])
    async def get_slackness(session_id: str, request: Request) -> list[ConstraintSlackness]:
        """Get constraint slackness analysis."""
        state = _load_or_404(session_id, request)

        # Try optimization review snapshot first (pre-computed)
        opt_snap = state.get("optimization_review_snapshot", {})
        if opt_snap.get("constraint_slackness"):
            return [ConstraintSlackness(**cs) for cs in opt_snap["constraint_slackness"]]

        # Fall back to computing from state
        try:
            from embodied_ai_architect.graphs.optimization_review import (
                compute_constraint_slackness,
            )

            slackness = compute_constraint_slackness(state)
            return [ConstraintSlackness(**cs.model_dump()) for cs in slackness]
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("Failed to compute slackness for session %s: %s", session_id, exc)
            return []

    @router.get("/sessions/{session_id}/trajectory", response_model=list[TrajectoryEntry])
    async def get_trajectory(session_id: str, request: Request) -> list[TrajectoryEntry]:
        """Get optimization trajectory (PPA history across iterations)."""
        state = _load_or_404(session_id, request)
        history = state.get("optimization_history", [])
        return [TrajectoryEntry(**entry) for entry in history]

    @router.get("/sessions/{session_id}/taskgraph", response_model=TaskGraphResponse)
    async def get_taskgraph(session_id: str, request: Request) -> TaskGraphResponse:
        """Get task graph structure for DAG visualization."""
        state = _load_or_404(session_id, request)
        task_graph = state.get("task_graph", {"nodes": {}})
        nodes = task_graph.get("nodes", {})

        # Compute execution order and parallel groups
        try:
            from embodied_ai_architect.graphs.task_graph import TaskGraph
            from embodied_ai_architect.graphs.review import compute_parallel_groups

            graph = TaskGraph.from_dict(task_graph)
            execution_order = graph.execution_order()
            parallel_groups = compute_parallel_groups(graph)
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("Failed to compute task graph layout for %s: %s", session_id, exc)
            execution_order = list(nodes.keys())
            parallel_groups = []

        return TaskGraphResponse(
            nodes=[
                TaskGraphNode(
                    id=tid,
                    name=node.get("name", ""),
                    agent=node.get("agent", ""),
                    status=node.get("status", "pending"),
                    dependencies=node.get("dependencies", []),
                )
                for tid, node in nodes.items()
            ],
            execution_order=execution_order,
            parallel_groups=parallel_groups,
        )

    @router.get("/sessions/{session_id}/workload", response_model=WorkloadResponse)
    async def get_workload(session_id: str, request: Request) -> WorkloadResponse:
        """Get per-operator workload breakdown."""
        state = _load_or_404(session_id, request)
        wp = state.get("workload_profile", {})

        operators = []
        for w in wp.get("workloads", []):
            operators.append(
                OperatorBreakdown(
                    name=w.get("name", "unknown"),
                    model_class=w.get("model_class", ""),
                    gflops=w.get("estimated_gflops"),
                    memory_mb=w.get("estimated_memory_mb"),
                    latency_ms=w.get("latency_ms"),
                    mapped_to=w.get("mapped_to"),
                    bound=w.get("bound"),
                    scheduling=w.get("scheduling", ""),
                )
            )

        return WorkloadResponse(
            operators=operators,
            total_gflops=wp.get("total_estimated_gflops"),
            total_memory_mb=wp.get("total_estimated_memory_mb"),
            dominant_op=wp.get("dominant_op", ""),
            source=wp.get("source", ""),
        )

    @router.get("/sessions/{session_id}/stream")
    async def stream_session(session_id: str, request: Request) -> StreamingResponse:
        """Stream live session updates via Server-Sent Events.

        Watches the session JSON file for modifications and sends partial
        state updates as SSE events. Useful for monitoring active optimization
        runs in the frontend.

        Event types:
        - state_update: session state changed (iteration, ppa_metrics, status)
        - complete: session reached terminal state
        - error: session file disappeared or became unreadable
        """
        _validate_session_id(session_id)
        store = _get_store(request)

        # Verify session exists
        state = store.load(session_id)
        if state is None:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

        return StreamingResponse(
            _session_event_generator(store, session_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return router


async def _session_event_generator(
    store: SessionStore,
    session_id: str,
    poll_interval: float = 1.0,
    timeout: float = 600.0,
) -> AsyncGenerator[str, None]:
    """Generate SSE events by polling the session file for changes.

    Args:
        store: SessionStore to read from.
        session_id: Session to watch.
        poll_interval: Seconds between polls.
        timeout: Maximum stream duration in seconds.
    """
    session_path = store.session_dir / f"{session_id}.json"
    last_mtime: float = 0.0
    elapsed: float = 0.0

    # Send initial state
    state = store.load(session_id)
    if state:
        last_mtime = session_path.stat().st_mtime if session_path.exists() else 0.0
        yield _format_sse("state_update", _extract_update(state))

        # If already terminal, send complete and stop — no polling needed
        status = state.get("status", "")
        if status in ("complete", "failed"):
            yield _format_sse(
                "complete", {"status": status, "iteration": state.get("iteration", 0)}
            )
            return

    while elapsed < timeout:
        await asyncio.sleep(poll_interval)
        elapsed += poll_interval

        # Check if file was modified
        if not session_path.exists():
            yield _format_sse("error", {"message": "Session file not found"})
            return

        current_mtime = session_path.stat().st_mtime
        if current_mtime <= last_mtime:
            # Send keepalive comment to prevent connection timeout
            yield ": keepalive\n\n"
            continue

        last_mtime = current_mtime

        # File changed — read new state with retry for transient errors
        # (e.g., JSONDecodeError from reading mid-write, though atomic
        # saves should prevent this)
        state = None
        for attempt in range(3):
            try:
                state = store.load(session_id)
                break
            except (json.JSONDecodeError, OSError) as exc:
                if attempt < 2:
                    logger.debug(
                        "Transient read error for %s (attempt %d): %s",
                        session_id,
                        attempt + 1,
                        exc,
                    )
                    await asyncio.sleep(0.1)
                else:
                    logger.warning(
                        "Failed to read session %s after 3 attempts: %s",
                        session_id,
                        exc,
                    )
                    yield _format_sse("error", {"message": f"Read failed: {exc}"})
                    return
        if state is None:
            yield _format_sse("error", {"message": "Failed to read session"})
            return

        current_iteration = state.get("iteration", 0)
        status = state.get("status", "")

        yield _format_sse("state_update", _extract_update(state))

        # Check for terminal state
        if status in ("complete", "failed"):
            yield _format_sse("complete", {"status": status, "iteration": current_iteration})
            return

    # Timeout reached
    yield _format_sse("timeout", {"message": f"Stream timed out after {timeout}s"})


def _extract_update(state: dict[str, Any]) -> dict[str, Any]:
    """Extract the fields relevant for a streaming update."""
    ppa = state.get("ppa_metrics", {})
    return {
        "session_id": state.get("session_id", ""),
        "status": state.get("status", ""),
        "iteration": state.get("iteration", 0),
        "max_iterations": state.get("max_iterations", 20),
        "ppa_metrics": {
            "power_watts": ppa.get("power_watts"),
            "latency_ms": ppa.get("latency_ms"),
            "area_mm2": ppa.get("area_mm2"),
            "cost_usd": ppa.get("cost_usd"),
        },
        "verdicts": ppa.get("verdicts", {}),
    }


def _format_sse(event: str, data: dict[str, Any]) -> str:
    """Format a Server-Sent Event string."""
    payload = json.dumps(data, default=str)
    return f"event: {event}\ndata: {payload}\n\n"


# Session ID must match soc_<alphanumeric> — prevents path traversal
_SESSION_ID_PATTERN = re.compile(r"^soc_[a-zA-Z0-9_-]+$")


def _validate_session_id(session_id: str) -> None:
    """Validate session_id format to prevent path traversal."""
    if not _SESSION_ID_PATTERN.match(session_id):
        raise HTTPException(status_code=400, detail="Invalid session ID format")


def _load_or_404(session_id: str, request: Request) -> dict[str, Any]:
    """Validate session_id, load session, or raise 404."""
    _validate_session_id(session_id)
    store = _get_store(request)
    state = store.load(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return state


# Module-level app for `uvicorn embodied_ai_architect.api.server:app`
app = create_app()
